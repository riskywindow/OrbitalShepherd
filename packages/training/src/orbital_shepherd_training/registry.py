from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orbital_shepherd_contracts import bundle_id_from_manifest_id
from orbital_shepherd_core import canonical_json_dumps, sha256_fingerprint, stable_id
from orbital_shepherd_scenario_engine import (
    ScenarioEngineConfig,
    ScenarioRecipe,
    build_scenario_pack,
    builtin_phase2_recipes,
    validate_scenario_pack,
)
from orbital_shepherd_training.config_io import (
    load_phase2_config,
    load_yaml_document,
    write_yaml_document,
)
from orbital_shepherd_training.models import (
    BehaviorCloningConfig,
    CurriculumConfig,
    EvaluationConfig,
    ModelArchitectureConfig,
    PpoTrainingConfig,
    RewardShapingConfig,
    ScenarioSplitEntry,
    ScenarioSplitRegistry,
    TrainingPackEntry,
    TrainingPackManifest,
)
from orbital_shepherd_training.paths import (
    PHASE2_BENCHMARK_ID,
    phase2_scenario_pack_dir,
    phase2_split_registry_path,
    phase2_training_pack_manifest_path,
    training_config_root,
)

PHASE2_SPLIT_REGISTRY_ID = "splitreg:osbench-phase2-foundation-v1:v1"


def phase2_artifact_layout() -> dict[str, str]:
    return {
        "dataset_root": "data/training/datasets",
        "checkpoint_root": "data/training/checkpoints",
        "report_root": "data/training/reports",
        "manifest_root": "data/training/manifests",
        "scenario_pack_root": "data/training/scenario_packs",
    }


def phase2_split_registry(
    recipes: Sequence[ScenarioRecipe] | None = None,
) -> ScenarioSplitRegistry:
    selected_recipes = tuple(recipes or builtin_phase2_recipes(PHASE2_BENCHMARK_ID))
    entries = [_registry_entry_for_recipe(recipe) for recipe in selected_recipes]
    registry = ScenarioSplitRegistry(
        registry_id=PHASE2_SPLIT_REGISTRY_ID,
        benchmark_id=PHASE2_BENCHMARK_ID,
        seed_namespace="phase2-family-seed-contract-v1",
        entries=entries,
    )
    validate_phase2_split_registry(registry, recipes=selected_recipes)
    return registry


def build_phase2_split_registry(output_path: Path | None = None) -> ScenarioSplitRegistry:
    registry = phase2_split_registry()
    destination = output_path or phase2_split_registry_path()
    write_yaml_document(destination, registry.model_dump(mode="json"))
    return registry


def validate_phase2_split_registry(
    registry: ScenarioSplitRegistry | Mapping[str, Any],
    *,
    recipes: Sequence[ScenarioRecipe] | None = None,
) -> ScenarioSplitRegistry:
    registry_model = (
        registry
        if isinstance(registry, ScenarioSplitRegistry)
        else ScenarioSplitRegistry.model_validate(registry)
    )
    expected_recipes = tuple(recipes or builtin_phase2_recipes(PHASE2_BENCHMARK_ID))
    expected_bundle_ids = {
        bundle_id_from_manifest_id(
            stable_id(
                "sm",
                recipe.benchmark_id,
                recipe.family,
                f"seed-{recipe.seed}",
            )
        )
        for recipe in expected_recipes
    }
    entry_bundle_ids = {entry.bundle_id for entry in registry_model.entries}
    if entry_bundle_ids != expected_bundle_ids:
        raise ValueError("split registry entries do not match the built-in Phase 2 recipe set")

    bundle_ids_by_split: dict[str, set[str]] = {
        "train": set(),
        "val": set(),
        "test": set(),
        "ood": set(),
    }
    for entry in registry_model.entries:
        bucket = bundle_ids_by_split[entry.split]
        if entry.bundle_id in bucket:
            raise ValueError(f"bundle {entry.bundle_id} appears twice in split {entry.split}")
        bucket.add(entry.bundle_id)
        if entry.scenario_family in {"station_outage", "constellation_degradation"}:
            if entry.split != "ood":
                raise ValueError("OOD families must remain in the OOD split")
        elif entry.split == "ood":
            raise ValueError("in-distribution families must not appear in the OOD split")

    overlap = set().union(
        bundle_ids_by_split["train"] & bundle_ids_by_split["val"],
        bundle_ids_by_split["train"] & bundle_ids_by_split["test"],
        bundle_ids_by_split["val"] & bundle_ids_by_split["test"],
    )
    if overlap:
        raise ValueError(f"split registry has overlapping bundle ids: {sorted(overlap)}")
    return registry_model


def validate_phase2_configs(config_root: Path | None = None) -> dict[str, Any]:
    root = config_root or training_config_root()
    config_paths = {
        "reward": root / "reward" / "phase2_reward.yaml",
        "curriculum": root / "curriculum" / "phase2_curriculum.yaml",
        "model": root / "model" / "phase2_policy.yaml",
        "bc": root / "bc" / "phase2_bc.yaml",
        "ppo": root / "ppo" / "phase2_ppo.yaml",
        "evaluation": root / "evaluation" / "phase2_eval.yaml",
        "splits": root / "curriculum" / "phase2_splits.yaml",
    }
    validated = {name: load_phase2_config(path) for name, path in config_paths.items()}
    validate_phase2_split_registry(validated["splits"])
    _validate_cross_config_refs(validated)
    return validated


def build_phase2_training_pack(
    *,
    output_dir: Path | None = None,
    manifest_output: Path | None = None,
    split_registry_output: Path | None = None,
    recipes: Sequence[ScenarioRecipe] | None = None,
) -> TrainingPackManifest:
    selected_recipes = tuple(recipes or builtin_phase2_recipes(PHASE2_BENCHMARK_ID))
    registry = phase2_split_registry(selected_recipes)
    if split_registry_output is not None:
        write_yaml_document(split_registry_output, registry.model_dump(mode="json"))
    destination = output_dir or phase2_scenario_pack_dir()
    records = build_scenario_pack(
        engine_config=ScenarioEngineConfig(
            benchmark_id=PHASE2_BENCHMARK_ID,
            scenario_dir=destination,
        ),
        output_dir=destination,
        recipes=selected_recipes,
    )
    record_path_by_bundle = {record.bundle_id: record.path for record in records}
    entries = [
        TrainingPackEntry(
            recipe_id=entry.recipe_id,
            manifest_id=entry.manifest_id,
            bundle_id=entry.bundle_id,
            scenario_family=entry.scenario_family,
            split=entry.split,
            scenario_path=str(record_path_by_bundle[entry.bundle_id]),
            difficulty_tier=entry.difficulty_tier,
        )
        for entry in registry.entries
        if entry.bundle_id in record_path_by_bundle
    ]
    fingerprint = _artifact_fingerprint(
        {
            "benchmark_id": PHASE2_BENCHMARK_ID,
            "entries": [entry.model_dump(mode="json") for entry in entries],
        }
    )
    manifest = TrainingPackManifest(
        training_pack_id="trainpack:osbench-phase2-foundation-v1:bundle-pack",
        benchmark_id=PHASE2_BENCHMARK_ID,
        split_registry_id=registry.registry_id,
        generated_at_utc=datetime(2026, 4, 12, 0, 0, tzinfo=UTC),
        output_dir=str(destination),
        bundle_count=len(entries),
        entries=entries,
        artifact_fingerprint=fingerprint,
    )
    target = manifest_output or phase2_training_pack_manifest_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        canonical_json_dumps(manifest.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_phase2_training_pack(
    *,
    input_dir: Path | None = None,
    manifest_path: Path | None = None,
    recipes: Sequence[ScenarioRecipe] | None = None,
) -> TrainingPackManifest:
    source_dir = input_dir or phase2_scenario_pack_dir()
    expected_recipes = tuple(recipes or builtin_phase2_recipes(PHASE2_BENCHMARK_ID))
    validate_scenario_pack(
        engine_config=ScenarioEngineConfig(
            benchmark_id=PHASE2_BENCHMARK_ID,
            scenario_dir=source_dir,
        ),
        input_dir=source_dir,
        recipes=expected_recipes,
    )
    target = manifest_path or phase2_training_pack_manifest_path()
    document = load_yaml_document(target) if target.suffix == ".yaml" else None
    if document is None:
        import json

        document = json.loads(target.read_text(encoding="utf-8"))
    manifest = TrainingPackManifest.model_validate(document)
    if manifest.bundle_count != len(manifest.entries):
        raise ValueError("training pack manifest bundle_count does not match entries length")
    expected_fingerprint = _artifact_fingerprint(
        {
            "benchmark_id": manifest.benchmark_id,
            "entries": [entry.model_dump(mode="json") for entry in manifest.entries],
        }
    )
    if manifest.artifact_fingerprint != expected_fingerprint:
        raise ValueError("training pack manifest fingerprint mismatch")
    return manifest


def _registry_entry_for_recipe(recipe: ScenarioRecipe) -> ScenarioSplitEntry:
    split, difficulty_tier, curriculum_stage, ood_axes = _phase2_split_metadata(recipe)
    manifest_id = stable_id("sm", recipe.benchmark_id, recipe.family, f"seed-{recipe.seed}")
    return ScenarioSplitEntry(
        recipe_id=recipe.recipe_id,
        manifest_id=manifest_id,
        bundle_id=bundle_id_from_manifest_id(manifest_id),
        scenario_family=recipe.family,
        split=split,
        simulation_seed=recipe.seed,
        difficulty_tier=difficulty_tier,
        curriculum_stage=curriculum_stage,
        ood_axes=ood_axes,
    )


def _phase2_split_metadata(
    recipe: ScenarioRecipe,
) -> tuple[str, int, str, list[str]]:
    if recipe.family == "sparse_frontier":
        ordinal = recipe.seed - 401
        return _id_family_assignment(
            ordinal=ordinal,
            curriculum_stage="curriculum:phase2:stage1-sparse-bootstrap",
            difficulty_tier=1,
        )
    if recipe.family in {"burst_outbreak", "cloud_trap"}:
        base_seed = 411 if recipe.family == "burst_outbreak" else 421
        ordinal = recipe.seed - base_seed
        return _id_family_assignment(
            ordinal=ordinal,
            curriculum_stage="curriculum:phase2:stage2-incident-pressure",
            difficulty_tier=2,
        )
    if recipe.family == "downlink_crunch":
        ordinal = recipe.seed - 431
        return _id_family_assignment(
            ordinal=ordinal,
            curriculum_stage="curriculum:phase2:stage3-downlink-pressure",
            difficulty_tier=3,
        )
    if recipe.family == "station_outage":
        return ("ood", 4, "curriculum:phase2:stage4-generalization-gate", ["ground_station_outage"])
    return ("ood", 4, "curriculum:phase2:stage4-generalization-gate", ["constellation_degradation"])


def _id_family_assignment(
    *,
    ordinal: int,
    curriculum_stage: str,
    difficulty_tier: int,
) -> tuple[str, int, str, list[str]]:
    if ordinal in {0, 1}:
        return ("train", difficulty_tier, curriculum_stage, [])
    if ordinal == 2:
        return ("val", difficulty_tier, curriculum_stage, [])
    return ("test", difficulty_tier, curriculum_stage, [])


def _validate_cross_config_refs(validated: Mapping[str, Any]) -> None:
    reward = _require_type(validated["reward"], RewardShapingConfig)
    curriculum = _require_type(validated["curriculum"], CurriculumConfig)
    model = _require_type(validated["model"], ModelArchitectureConfig)
    bc = _require_type(validated["bc"], BehaviorCloningConfig)
    ppo = _require_type(validated["ppo"], PpoTrainingConfig)
    evaluation = _require_type(validated["evaluation"], EvaluationConfig)
    splits = _require_type(validated["splits"], ScenarioSplitRegistry)

    if (
        curriculum.benchmark_id != reward.benchmark_id
        or curriculum.benchmark_id != splits.benchmark_id
    ):
        raise ValueError("benchmark ids must match across reward, curriculum, and split registry")
    for config in (bc, ppo):
        if config.reward_id != reward.reward_id:
            raise ValueError("training configs must reference the shared reward config")
        if config.curriculum_id != curriculum.curriculum_id:
            raise ValueError("training configs must reference the shared curriculum config")
        if config.model_id != model.model_id:
            raise ValueError("training configs must reference the shared model architecture")
    if evaluation.benchmark_id != reward.benchmark_id:
        raise ValueError("evaluation config must target the shared benchmark id")


def _require_type(value: Any, expected_type: type[Any]) -> Any:
    if not isinstance(value, expected_type):
        raise TypeError(f"expected {expected_type.__name__}, got {type(value).__name__}")
    return value


def _artifact_fingerprint(document: Mapping[str, Any]) -> str:
    return f"sha256:{sha256_fingerprint(canonical_json_dumps(document))}"
