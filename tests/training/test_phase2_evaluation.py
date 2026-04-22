from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tests.benchmark.helpers import build_tiny_bundle

from orbital_shepherd_benchmark.metrics import DistributionSummary, EpisodeMetrics
from orbital_shepherd_core import canonical_json_dumps, sha256_fingerprint
from orbital_shepherd_training import (
    EvaluationConfig,
    EvaluationRunManifest,
    ModelArchitectureConfig,
    PolicyCheckpointManifest,
    ScenarioSplitRegistry,
    TrainingPackManifest,
    export_phase2_policy_checkpoint,
    run_phase2_evaluation,
)
from orbital_shepherd_training.evaluation import _pairwise_comparisons, _select_scenario_entries
from orbital_shepherd_training.models import (
    ArtifactLayoutConfig,
    EvaluationReportManifest,
    ScenarioSplitEntry,
    TrainingPackEntry,
    WandbConfig,
)


def test_phase2_evaluation_toy_run_writes_reports_and_manifests(tmp_path: Path) -> None:
    training_pack_path, split_registry_path, bundle_ids = _write_toy_training_contracts(tmp_path)
    checkpoint_manifest_path = _write_toy_policy_checkpoint(tmp_path, source_bundle_ids=bundle_ids)
    artifact_layout = ArtifactLayoutConfig(
        dataset_root=str(tmp_path / "datasets"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        report_root=str(tmp_path / "reports"),
        manifest_root=str(tmp_path / "manifests"),
        scenario_pack_root=str(tmp_path / "scenario-pack"),
    )

    summary = run_phase2_evaluation(
        evaluation_config=EvaluationConfig(
            evaluation_id="eval:test-phase2-toy-v1",
            benchmark_id="osbench-phase2-foundation-v1",
            split_order=["val", "test", "ood"],
            planner_agnostic_metrics=[
                "time_to_first_useful_observation_seconds",
                "useful_observation_value_captured",
                "cloud_waste_rate",
                "downlink_latency_seconds",
                "missed_urgent_incident_rate",
                "opportunity_utilization_efficiency",
                "mission_utility",
            ],
            compare_against_baselines=["random_valid_action", "urgency_greedy"],
            report_formats=["json", "markdown", "csv"],
            artifact_layout=artifact_layout,
            wandb=WandbConfig(
                enabled=False,
                mode="disabled",
                project="orbital-shepherd-phase2",
                entity=None,
                group="pytest",
                tags=["test"],
            ),
            bootstrap_replicates=32,
            confidence_level=0.95,
            notable_episode_count=2,
            primary_ranking_metric="mission_utility",
            emit_reward_audit=True,
        ),
        training_pack=training_pack_path,
        split_registry=split_registry_path,
        planner_ids=("random_valid_action", "urgency_greedy"),
        checkpoint_manifests=(checkpoint_manifest_path,),
        policy_labels=("toy_policy",),
        splits=("val", "test", "ood"),
        limit_bundles_per_split=1,
    )

    assert summary.summary_path.exists()
    assert summary.markdown_path is not None and summary.markdown_path.exists()
    assert summary.episode_metrics_path is not None and summary.episode_metrics_path.exists()
    assert (
        summary.pairwise_comparisons_path is not None and summary.pairwise_comparisons_path.exists()
    )
    assert summary.notable_episodes_path is not None and summary.notable_episodes_path.exists()
    assert summary.run_manifest_path.exists()
    assert summary.report_manifest_paths

    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert payload["selected_splits"] == ["val", "test", "ood"]
    assert len(payload["episodes"]) == 9
    assert "toy_policy" in payload["notable_episodes"]["policy_vs_best_baseline"]
    assert (summary.report_dir / "dvc_metrics.json").exists()
    assert (summary.report_dir / "dvc_plots" / "planner_metric_means.csv").exists()

    run_manifest = EvaluationRunManifest.model_validate(
        json.loads(summary.run_manifest_path.read_text(encoding="utf-8"))
    )
    assert run_manifest.run_id == summary.run_id
    assert len(run_manifest.planners) == 3
    assert run_manifest.selected_splits == ["val", "test", "ood"]

    validated_report = EvaluationReportManifest.model_validate(
        json.loads(summary.report_manifest_paths[0].read_text(encoding="utf-8"))
    )
    assert validated_report.evaluation_config_id == "eval:test-phase2-toy-v1"


def test_phase2_evaluation_split_selection_preserves_requested_splits(tmp_path: Path) -> None:
    training_pack_path, split_registry_path, _ = _write_toy_training_contracts(tmp_path)
    training_pack = TrainingPackManifest.model_validate(
        json.loads(training_pack_path.read_text(encoding="utf-8"))
    )
    split_registry = ScenarioSplitRegistry.model_validate(
        json.loads(split_registry_path.read_text(encoding="utf-8"))
    )

    selected = _select_scenario_entries(
        training_pack=training_pack,
        split_registry=split_registry,
        splits=("val", "ood"),
        limit_bundles_per_split=1,
        bundle_ids=(),
    )

    assert [entry.split for entry in selected] == ["val", "ood"]
    assert {entry.bundle_id for entry in selected} == {
        "sb:test-phase2-eval:burst_outbreak:seed-101",
        "sb:test-phase2-eval:station_outage:seed-103",
    }


def test_phase2_pairwise_comparison_counts_wins_losses_and_ties() -> None:
    episodes = [
        _episode_result("left", "sb:test:one", 5.0, 2.0),
        _episode_result("left", "sb:test:two", 1.0, 3.0),
        _episode_result("left", "sb:test:three", 4.0, 4.0),
        _episode_result("right", "sb:test:one", 2.0, 1.0),
        _episode_result("right", "sb:test:two", 3.0, 3.0),
        _episode_result("right", "sb:test:three", 4.0, 5.0),
    ]

    comparisons = _pairwise_comparisons(
        episodes=episodes,
        planner_order=["left", "right"],
        metric_names=["mission_utility", "cloud_waste_rate"],
        bootstrap_replicates=16,
        confidence_level=0.95,
        primary_metric="mission_utility",
    )

    assert len(comparisons) == 1
    comparison = comparisons[0]
    mission = comparison["metrics"]["mission_utility"]
    cloud = comparison["metrics"]["cloud_waste_rate"]
    assert mission["wins_left"] == 1
    assert mission["wins_right"] == 1
    assert mission["ties"] == 1
    assert mission["difference_summary"]["mean"] == round((3.0 + -2.0 + 0.0) / 3.0, 6)
    assert cloud["wins_left"] == 1
    assert cloud["wins_right"] == 1
    assert cloud["ties"] == 1


def _write_toy_training_contracts(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    bundles = [
        ("val", "burst_outbreak", 101),
        ("test", "cloud_trap", 102),
        ("ood", "station_outage", 103),
    ]
    entries: list[TrainingPackEntry] = []
    split_entries: list[ScenarioSplitEntry] = []
    bundle_ids: list[str] = []
    for split_name, family, seed in bundles:
        bundle = _bundle_with_family(family=family, seed=seed)
        bundle_path = tmp_path / f"{family}-{seed}.json"
        bundle_path.write_text(
            canonical_json_dumps(bundle.model_dump(mode="json")) + "\n",
            encoding="utf-8",
        )
        entries.append(
            TrainingPackEntry(
                recipe_id=f"recipe:test-phase2-eval:{family}:{seed}",
                manifest_id=bundle.manifest_id,
                bundle_id=bundle.bundle_id,
                scenario_family=family,
                split=split_name,
                scenario_path=str(bundle_path),
                difficulty_tier=1 if split_name != "ood" else 4,
            )
        )
        split_entries.append(
            ScenarioSplitEntry(
                recipe_id=f"recipe:test-phase2-eval:{family}:{seed}",
                manifest_id=bundle.manifest_id,
                bundle_id=bundle.bundle_id,
                scenario_family=family,
                split=split_name,
                simulation_seed=seed,
                difficulty_tier=1 if split_name != "ood" else 4,
                curriculum_stage=f"curriculum:test-phase2-eval:{split_name}",
                ood_axes=["ground_station_outage"] if split_name == "ood" else [],
            )
        )
        bundle_ids.append(bundle.bundle_id)

    training_pack = TrainingPackManifest(
        training_pack_id="trainpack:test-phase2-eval:v1",
        benchmark_id="osbench-phase2-foundation-v1",
        split_registry_id="splitreg:test-phase2-eval:v1",
        generated_at_utc=datetime(2026, 4, 13, 0, 0, tzinfo=UTC),
        output_dir=str(tmp_path),
        bundle_count=len(entries),
        entries=entries,
        artifact_fingerprint="sha256:" + sha256_fingerprint("trainpack:test-phase2-eval:v1"),
    )
    split_registry = ScenarioSplitRegistry(
        registry_id="splitreg:test-phase2-eval:v1",
        benchmark_id="osbench-phase2-foundation-v1",
        seed_namespace="phase2-test-eval-contract",
        entries=split_entries,
    )
    training_pack_path = tmp_path / "phase2-training-pack-manifest.json"
    split_registry_path = tmp_path / "phase2-splits.yaml"
    training_pack_path.write_text(
        canonical_json_dumps(training_pack.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    split_registry_path.write_text(
        canonical_json_dumps(split_registry.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    return training_pack_path, split_registry_path, bundle_ids


def _write_toy_policy_checkpoint(tmp_path: Path, *, source_bundle_ids: list[str]) -> Path:
    architecture = ModelArchitectureConfig(
        model_id="model:test-phase2-eval:v1",
        observation_encoder_type="tabular_transformer",
        action_encoder_type="masked_action_mlp",
        hidden_dim=32,
        encoder_layers=2,
        attention_heads=4,
        dropout=0.0,
        recurrent_memory_steps=1,
        shared_policy_value_backbone=True,
    )
    from orbital_shepherd_training import build_phase2_policy_model

    model = build_phase2_policy_model(architecture=architecture, top_k=64)
    exported = export_phase2_policy_checkpoint(
        checkpoint_root=tmp_path / "checkpoints",
        manifest_root=tmp_path / "checkpoint-manifests",
        checkpoint_name="checkpoint_000001",
        algorithm="behavior_cloning",
        run_id="bc:test-phase2-eval:v1",
        benchmark_id="osbench-phase2-foundation-v1",
        reward_id="reward:test-phase2-eval:v1",
        architecture=architecture,
        top_k=64,
        source_dataset_ids=["offds:test-phase2-eval:v1"],
        source_training_pack_id="trainpack:test-phase2-eval:v1",
        source_bundle_ids=source_bundle_ids,
        global_step=1,
        metrics={"toy_metric": 1.0},
        metadata={"source_planner_ids": ["urgency_greedy"]},
        policy_state_dict=model.state_dict(),
        created_at=datetime(2026, 4, 13, 0, 0, tzinfo=UTC),
    )
    PolicyCheckpointManifest.model_validate(
        json.loads(exported.manifest_path.read_text(encoding="utf-8"))
    )
    return exported.manifest_path


def _bundle_with_family(*, family: str, seed: int):
    bundle = build_tiny_bundle()
    document = bundle.model_dump(mode="json")
    document["benchmark_id"] = "osbench-phase2-foundation-v1"
    document["scenario_family"] = family
    document["manifest_id"] = f"sm:test-phase2-eval:{family}:seed-{seed}"
    document["bundle_id"] = f"sb:test-phase2-eval:{family}:seed-{seed}"
    document["simulation_seed"] = seed
    document["compilation"]["source_manifest_id"] = document["manifest_id"]
    document["bundle_fingerprint"] = sha256_fingerprint(document)
    from orbital_shepherd_contracts import ScenarioBundle

    return ScenarioBundle.model_validate(document)


def _episode_result(
    planner_key: str,
    bundle_id: str,
    mission_utility: float,
    cloud_waste_rate: float,
):
    from orbital_shepherd_training.evaluation import (
        EvaluationEpisodeArtifacts,
        EvaluationEpisodeResult,
    )

    return EvaluationEpisodeResult(
        planner_key=planner_key,
        planner_kind="builtin",
        display_name=planner_key,
        evaluated_artifact_id=f"planner:{planner_key}",
        planner_version="test-v1",
        bundle_id=bundle_id,
        manifest_id=bundle_id.replace("sb:", "sm:"),
        split="val",
        scenario_family="burst_outbreak",
        scenario_path=Path("/tmp/fake.json"),
        ood_axes=(),
        difficulty_tier=1,
        episode_id=f"ep:{bundle_id}",
        episode_seed=1,
        episode_fingerprint="sha256:" + "1" * 64,
        replay_fingerprint="sha256:" + "2" * 64,
        action_count=3,
        metrics=EpisodeMetrics(
            time_to_first_useful_observation_seconds=DistributionSummary(
                count=1,
                mean=1.0,
                median=1.0,
                p90=1.0,
                values=(1.0,),
            ),
            useful_observation_value_captured=1.0,
            cloud_waste_rate=cloud_waste_rate,
            downlink_latency_seconds=DistributionSummary(
                count=1,
                mean=1.0,
                median=1.0,
                p90=1.0,
                values=(1.0,),
            ),
            missed_urgent_incident_rate=0.0,
            opportunity_utilization_efficiency=1.0,
            mission_utility=mission_utility,
            observation_commit_count=1,
            useful_packet_count=1,
            urgent_incident_count=1,
            missed_urgent_incident_count=0,
        ),
        reward_audit={"total_reward_sum": mission_utility},
        bundle_profile={"cloud_pressure_index": 0.5, "downlink_pressure_ratio": 1.0},
        artifacts=EvaluationEpisodeArtifacts(
            replay_path=Path("/tmp/replay.ndjson"),
            summary_path=Path("/tmp/summary.json"),
        ),
    )
