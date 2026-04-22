from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tests.benchmark.helpers import build_tiny_bundle

from orbital_shepherd_contracts import compile_scenario_bundle
from orbital_shepherd_core import canonical_json_dumps, sha256_fingerprint
from orbital_shepherd_training import (
    DEFAULT_EXPERT_PLANNER_IDS,
    OFFLINE_DATASET_SCHEMA_VERSION,
    OfflineDatasetBuildManifest,
    OfflineDatasetManifest,
    OfflineTransitionRecord,
    build_offline_datasets,
)
from orbital_shepherd_training.models import (
    ScenarioSplitEntry,
    ScenarioSplitRegistry,
    TrainingPackEntry,
    TrainingPackManifest,
)

PHASE2_BENCHMARK_ID = "osbench-phase2-foundation-v1"
GENERATED_AT = datetime(2026, 4, 13, 0, 0, tzinfo=UTC)


def test_offline_dataset_schema_validity_and_feature_action_alignment(tmp_path: Path) -> None:
    training_pack, split_registry = _fixture_training_pack(tmp_path)

    build_manifest = build_offline_datasets(
        training_pack=training_pack,
        split_registry=split_registry,
        output_root=tmp_path / "datasets",
        manifest_root=tmp_path / "manifests",
        planner_ids=DEFAULT_EXPERT_PLANNER_IDS,
        splits=("train",),
        top_k=4,
    )

    dataset_manifest = OfflineDatasetManifest.model_validate(
        _read_json(build_manifest.dataset_manifests[0])
    )
    transition = OfflineTransitionRecord.model_validate(
        _read_jsonl(Path(dataset_manifest.artifacts[1].path))[0]
    )
    selected_slot = transition.projected_slot_mapping[transition.selected_slot]

    assert dataset_manifest.dataset_schema_version == OFFLINE_DATASET_SCHEMA_VERSION
    assert len(transition.canonical_actions) == transition.legal_action_count
    assert len(transition.global_features) == len(dataset_manifest.feature_schema.global_features)
    assert len(transition.candidate_features) == dataset_manifest.top_k
    assert len(transition.training_action_mask) == dataset_manifest.top_k + 1
    assert transition.training_action_mask[transition.selected_slot] == 1
    assert transition.selected_action_id == selected_slot["action_id"]
    assert transition.selected_runtime_action_index == selected_slot["runtime_action_index"]
    assert (
        transition.canonical_actions[transition.selected_runtime_action_index]["action_id"]
        == transition.selected_action_id
    )


def test_offline_dataset_split_integrity(tmp_path: Path) -> None:
    training_pack, split_registry = _fixture_training_pack(tmp_path)

    build_manifest = build_offline_datasets(
        training_pack=training_pack,
        split_registry=split_registry,
        output_root=tmp_path / "datasets",
        manifest_root=tmp_path / "manifests",
        planner_ids=("urgency_greedy",),
        splits=("train", "val", "test"),
        top_k=4,
    )

    manifests = [
        OfflineDatasetManifest.model_validate(_read_json(path))
        for path in build_manifest.dataset_manifests
    ]
    bundle_ids_by_split = {
        manifest.split: set(manifest.source_bundle_ids) for manifest in manifests
    }

    assert bundle_ids_by_split["train"].isdisjoint(bundle_ids_by_split["val"])
    assert bundle_ids_by_split["train"].isdisjoint(bundle_ids_by_split["test"])
    assert bundle_ids_by_split["val"].isdisjoint(bundle_ids_by_split["test"])
    assert bundle_ids_by_split["train"] == {
        entry.bundle_id for entry in split_registry.entries if entry.split == "train"
    }
    assert bundle_ids_by_split["val"] == {
        entry.bundle_id for entry in split_registry.entries if entry.split == "val"
    }
    assert bundle_ids_by_split["test"] == {
        entry.bundle_id for entry in split_registry.entries if entry.split == "test"
    }


def test_offline_dataset_build_is_deterministic(tmp_path: Path) -> None:
    training_pack, split_registry = _fixture_training_pack(tmp_path)
    output_root = tmp_path / "datasets"
    manifest_root = tmp_path / "manifests"

    first_manifest = build_offline_datasets(
        training_pack=training_pack,
        split_registry=split_registry,
        output_root=output_root,
        manifest_root=manifest_root,
        planner_ids=("urgency_greedy",),
        splits=("train",),
        top_k=4,
        build_id="offbuild:phase2:test-deterministic",
    )
    first_manifest_path = manifest_root / "offbuild--phase2--test-deterministic.json"
    first_manifest_bytes = first_manifest_path.read_bytes()
    first_dataset = OfflineDatasetManifest.model_validate(
        _read_json(first_manifest.dataset_manifests[0])
    )
    first_steps_bytes = Path(first_dataset.artifacts[1].path).read_bytes()

    second_manifest = build_offline_datasets(
        training_pack=training_pack,
        split_registry=split_registry,
        output_root=output_root,
        manifest_root=manifest_root,
        planner_ids=("urgency_greedy",),
        splits=("train",),
        top_k=4,
        build_id="offbuild:phase2:test-deterministic",
    )
    second_dataset = OfflineDatasetManifest.model_validate(
        _read_json(second_manifest.dataset_manifests[0])
    )

    assert first_manifest_bytes == first_manifest_path.read_bytes()
    assert first_steps_bytes == Path(second_dataset.artifacts[1].path).read_bytes()
    assert first_manifest.artifact_fingerprint == second_manifest.artifact_fingerprint
    assert first_dataset.artifact_fingerprint == second_dataset.artifact_fingerprint


def test_offline_dataset_build_manifest_schema(tmp_path: Path) -> None:
    training_pack, split_registry = _fixture_training_pack(tmp_path)

    build_manifest = build_offline_datasets(
        training_pack=training_pack,
        split_registry=split_registry,
        output_root=tmp_path / "datasets",
        manifest_root=tmp_path / "manifests",
        planner_ids=("urgency_greedy",),
        splits=("train",),
        top_k=4,
    )
    manifest_path = (
        tmp_path / "manifests" / f"{str(build_manifest.build_id).replace(':', '--')}.json"
    )
    validated = OfflineDatasetBuildManifest.model_validate(_read_json(manifest_path))

    assert validated.top_k == 4
    assert validated.planner_ids == ["urgency_greedy"]
    assert validated.dataset_manifests


def _fixture_training_pack(tmp_path: Path) -> tuple[TrainingPackManifest, ScenarioSplitRegistry]:
    bundles = [
        ("train", "cloud_trap", 501),
        ("val", "burst_outbreak", 502),
        ("test", "sparse_frontier", 503),
    ]
    training_entries: list[TrainingPackEntry] = []
    split_entries: list[ScenarioSplitEntry] = []
    for split_name, scenario_family, simulation_seed in bundles:
        bundle = _tiny_bundle_variant(
            scenario_family=scenario_family,
            simulation_seed=simulation_seed,
        )
        scenario_path = tmp_path / "scenario-pack" / f"{bundle.bundle_id}.json"
        scenario_path.parent.mkdir(parents=True, exist_ok=True)
        scenario_path.write_text(
            canonical_json_dumps(bundle.model_dump(mode="json")) + "\n",
            encoding="utf-8",
        )
        recipe_id = f"recipe:phase2:{scenario_family}:{simulation_seed}"
        training_entries.append(
            TrainingPackEntry(
                recipe_id=recipe_id,
                manifest_id=bundle.manifest_id,
                bundle_id=bundle.bundle_id,
                scenario_family=scenario_family,
                split=split_name,
                scenario_path=str(scenario_path),
                difficulty_tier=1,
            )
        )
        split_entries.append(
            ScenarioSplitEntry(
                recipe_id=recipe_id,
                manifest_id=bundle.manifest_id,
                bundle_id=bundle.bundle_id,
                scenario_family=scenario_family,
                split=split_name,
                simulation_seed=simulation_seed,
                difficulty_tier=1,
                curriculum_stage="curriculum:phase2:test",
                ood_axes=[],
            )
        )

    training_manifest = TrainingPackManifest(
        training_pack_id="trainpack:phase2:test-fixtures",
        benchmark_id=PHASE2_BENCHMARK_ID,
        split_registry_id="splitreg:phase2:test-fixtures",
        generated_at_utc=GENERATED_AT,
        output_dir=str(tmp_path / "scenario-pack"),
        bundle_count=len(training_entries),
        entries=training_entries,
        artifact_fingerprint="sha256:"
        + sha256_fingerprint([entry.model_dump(mode="json") for entry in training_entries]),
    )
    split_registry = ScenarioSplitRegistry(
        registry_id="splitreg:phase2:test-fixtures",
        benchmark_id=PHASE2_BENCHMARK_ID,
        seed_namespace="phase2-test-fixtures",
        entries=split_entries,
    )
    return training_manifest, split_registry


def _tiny_bundle_variant(*, scenario_family: str, simulation_seed: int):
    base_bundle = build_tiny_bundle()
    manifest_id = f"sm:{PHASE2_BENCHMARK_ID}:{scenario_family}:seed-{simulation_seed}"
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": manifest_id,
        "benchmark_id": PHASE2_BENCHMARK_ID,
        "scenario_family": scenario_family,
        "simulation_seed": simulation_seed,
        "decision_interval_seconds": base_bundle.decision_interval_seconds,
        "time_window": base_bundle.time_window.model_dump(mode="json"),
        "satellites": [item.model_dump(mode="json") for item in base_bundle.satellites],
        "ground_stations": [item.model_dump(mode="json") for item in base_bundle.ground_stations],
        "target_cells": [item.model_dump(mode="json") for item in base_bundle.target_cells],
        "incidents": [item.model_dump(mode="json") for item in base_bundle.incidents],
        "config": base_bundle.config.model_dump(mode="json"),
    }
    return compile_scenario_bundle(
        manifest,
        compiled_at=GENERATED_AT,
        observation_opportunities=[
            item.model_dump(mode="json") for item in base_bundle.observation_opportunities
        ],
        downlink_windows=[item.model_dump(mode="json") for item in base_bundle.downlink_windows],
    )


def _read_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
