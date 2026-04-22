from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from _bootstrap import REPO_ROOT, install_repo_sources

DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "demo" / "phase2-verification.json"


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    install_repo_sources()

    from fastapi.testclient import TestClient

    from orbital_shepherd_api.app import create_app
    from orbital_shepherd_api.settings import ApiSettings
    from orbital_shepherd_core import canonical_json_dumps
    from orbital_shepherd_training.bc_training import train_behavior_cloning
    from orbital_shepherd_training.config_io import load_phase2_config
    from orbital_shepherd_training.evaluation import run_phase2_evaluation
    from orbital_shepherd_training.models import (
        ArtifactLayoutConfig,
        BehaviorCloningConfig,
        EvaluationConfig,
        PolicyInitializationConfig,
        PpoTrainingConfig,
    )
    from orbital_shepherd_training.offline_dataset import build_offline_datasets
    from orbital_shepherd_training.registry import (
        build_phase2_training_pack,
        validate_phase2_training_pack,
    )
    from orbital_shepherd_training.rllib_training import train_ppo_with_rllib

    temp_root = Path(tempfile.mkdtemp(prefix="phase2-verify-"))
    training_root = temp_root / "training"
    data_root = temp_root / "data"
    scenario_pack_root = training_root / "scenario_packs"
    manifest_root = training_root / "manifests"
    checkpoint_root = training_root / "checkpoints"
    report_root = training_root / "reports"
    dataset_root = training_root / "datasets"
    split_registry_path = training_root / "phase2-splits.yaml"
    training_pack_manifest_path = manifest_root / "phase2-training-pack-manifest.json"
    phase2_pack_dir = scenario_pack_root / "osbench-phase2-foundation-v1"

    artifact_layout = ArtifactLayoutConfig(
        dataset_root=str(dataset_root),
        checkpoint_root=str(checkpoint_root),
        report_root=str(report_root),
        manifest_root=str(manifest_root),
        scenario_pack_root=str(scenario_pack_root),
    )

    training_pack = build_phase2_training_pack(
        output_dir=phase2_pack_dir,
        manifest_output=training_pack_manifest_path,
        split_registry_output=split_registry_path,
    )
    validate_phase2_training_pack(
        input_dir=phase2_pack_dir,
        manifest_path=training_pack_manifest_path,
    )

    dataset_manifest = build_offline_datasets(
        training_pack=training_pack_manifest_path,
        split_registry=split_registry_path,
        output_root=dataset_root / training_pack.benchmark_id,
        manifest_root=manifest_root,
        planner_ids=("urgency_greedy",),
        splits=("train", "val"),
        top_k=64,
        limit_bundles_per_split=1,
        build_id="offbuild:phase2:verify-smoke",
    )

    bc_config = _phase2_bc_smoke_config(
        artifact_layout=artifact_layout,
        training_pack_path=training_pack_manifest_path,
        split_registry_path=split_registry_path,
    )
    bc_summary = train_behavior_cloning(
        bc_config=bc_config,
        model_config=REPO_ROOT / "training" / "configs" / "model" / "phase2_policy.yaml",
    )
    if bc_summary.best_checkpoint_manifest_path is None:
        raise RuntimeError("BC smoke run did not produce a best checkpoint manifest")

    ppo_config = _phase2_ppo_smoke_config(
        artifact_layout=artifact_layout,
        training_pack_path=training_pack_manifest_path,
        split_registry_path=split_registry_path,
        checkpoint_manifest_path=bc_summary.best_checkpoint_manifest_path,
    )
    ppo_summary = train_ppo_with_rllib(
        ppo_config=ppo_config,
        model_config=REPO_ROOT / "training" / "configs" / "model" / "phase2_policy.yaml",
    )
    if not ppo_summary.checkpoint_manifest_paths:
        raise RuntimeError("PPO smoke run did not produce checkpoint manifests")

    evaluation_summary = run_phase2_evaluation(
        evaluation_config=_phase2_eval_config(artifact_layout=artifact_layout),
        training_pack=training_pack_manifest_path,
        split_registry=split_registry_path,
        checkpoint_manifests=(ppo_summary.checkpoint_manifest_paths[-1],),
        policy_labels=("phase2_verify_policy",),
        splits=("val", "test", "ood"),
        limit_bundles_per_split=1,
    )

    api_root = data_root / "api"
    app = create_app(
        ApiSettings(
            scenario_dir=data_root / "scenarios",
            training_scenario_pack_dir=scenario_pack_root,
            episode_dir=api_root / "episodes",
            baseline_run_dir=api_root / "baseline-runs",
            model_run_dir=api_root / "model-runs",
            demo_defaults_path=data_root / "demo" / "phase1-defaults.json",
            phase2_demo_defaults_path=data_root / "demo" / "phase2-defaults.json",
            training_manifest_dir=manifest_root,
            training_checkpoint_dir=checkpoint_root,
            training_report_dir=report_root,
        )
    )

    selected_bundle_id = next(
        entry.bundle_id for entry in training_pack.entries if entry.split == "val"
    )
    selected_model_manifest = json.loads(
        ppo_summary.checkpoint_manifest_paths[-1].read_text(encoding="utf-8")
    )

    with TestClient(app) as client:
        scenarios_response = client.get("/v1/scenarios")
        models_response = client.get("/v1/models")
        reports_response = client.get("/v1/reports")
        model_run_response = client.post(
            f"/v1/models/{selected_model_manifest['checkpoint_id']}/run",
            json={
                "bundle_id": selected_bundle_id,
                "simulation_seed": _simulation_seed_for_bundle(training_pack, selected_bundle_id),
                "include_inference_traces": True,
            },
        )
        episode_id = str(model_run_response.json()["episode_id"])
        episode_response = client.get(f"/v1/episodes/{episode_id}")
        traces_response = client.get(f"/v1/episodes/{episode_id}/inference-traces")
        report_detail_response = client.get(f"/v1/reports/{evaluation_summary.report_dir.name}")

    verification_payload = {
        "training_pack": {
            "bundle_count": training_pack.bundle_count,
            "artifact_fingerprint": training_pack.artifact_fingerprint,
            "manifest_path": str(training_pack_manifest_path),
            "split_registry_path": str(split_registry_path),
        },
        "offline_dataset": {
            "build_id": dataset_manifest.build_id,
            "artifact_fingerprint": dataset_manifest.artifact_fingerprint,
            "dataset_manifests": dataset_manifest.dataset_manifests,
        },
        "behavior_cloning": {
            "run_id": bc_summary.run_id,
            "best_checkpoint_manifest_path": str(bc_summary.best_checkpoint_manifest_path),
            "final_metrics": bc_summary.final_metrics,
        },
        "ppo": {
            "run_id": ppo_summary.run_id,
            "checkpoint_manifest_paths": [str(path) for path in ppo_summary.checkpoint_manifest_paths],
            "final_result": ppo_summary.final_result,
        },
        "evaluation": {
            "run_id": evaluation_summary.run_id,
            "summary_path": str(evaluation_summary.summary_path),
            "report_manifest_paths": [str(path) for path in evaluation_summary.report_manifest_paths],
            "episode_count": evaluation_summary.episode_count,
            "planner_keys": list(evaluation_summary.planner_keys),
        },
        "api": {
            "scenario_count": len(scenarios_response.json()),
            "model_count": len(models_response.json()),
            "report_count": len(reports_response.json()),
            "episode_id": episode_id,
            "planner_kind": episode_response.json()["planner_kind"],
            "trace_step_count": traces_response.json()["step_count"],
            "report_title": report_detail_response.json()["title"],
        },
    }
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(
        canonical_json_dumps(verification_payload) + "\n",
        encoding="utf-8",
    )
    print(DEFAULT_OUTPUT_PATH)
    print(json.dumps(verification_payload["api"], sort_keys=True))
    return 0


def _phase2_bc_smoke_config(
    *,
    artifact_layout: "ArtifactLayoutConfig",
    training_pack_path: Path,
    split_registry_path: Path,
) -> "BehaviorCloningConfig":
    from orbital_shepherd_training.config_io import load_phase2_config
    from orbital_shepherd_training.models import BehaviorCloningConfig

    loaded = load_phase2_config(REPO_ROOT / "training" / "configs" / "bc" / "phase2_bc_smoke.yaml")
    if not isinstance(loaded, BehaviorCloningConfig):
        raise TypeError("phase2_bc_smoke.yaml did not load as BehaviorCloningConfig")
    return loaded.model_copy(
        update={
            "run_id": "bc:phase2-verify-smoke-v1",
            "artifact_layout": artifact_layout,
            "training_pack_path": str(training_pack_path),
            "split_registry_path": str(split_registry_path),
            "device": "cpu",
        }
    )


def _phase2_ppo_smoke_config(
    *,
    artifact_layout: "ArtifactLayoutConfig",
    training_pack_path: Path,
    split_registry_path: Path,
    checkpoint_manifest_path: Path,
) -> "PpoTrainingConfig":
    from orbital_shepherd_training.config_io import load_phase2_config
    from orbital_shepherd_training.models import PolicyInitializationConfig, PpoTrainingConfig

    loaded = load_phase2_config(
        REPO_ROOT / "training" / "configs" / "ppo" / "phase2_ppo_bc_warmstart_smoke.yaml"
    )
    if not isinstance(loaded, PpoTrainingConfig):
        raise TypeError("phase2_ppo_bc_warmstart_smoke.yaml did not load as PpoTrainingConfig")
    return loaded.model_copy(
        update={
            "run_id": "ppo:phase2-verify-online-v1",
            "artifact_layout": artifact_layout,
            "training_pack_path": str(training_pack_path),
            "split_registry_path": str(split_registry_path),
            "initialization": PolicyInitializationConfig(
                mode="checkpoint",
                checkpoint_manifest_path=str(checkpoint_manifest_path),
                source_run_id=None,
                selection="latest",
            ),
        }
    )


def _phase2_eval_config(*, artifact_layout: "ArtifactLayoutConfig") -> "EvaluationConfig":
    from orbital_shepherd_training.config_io import load_phase2_config
    from orbital_shepherd_training.models import EvaluationConfig

    loaded = load_phase2_config(REPO_ROOT / "training" / "configs" / "evaluation" / "phase2_eval.yaml")
    if not isinstance(loaded, EvaluationConfig):
        raise TypeError("phase2_eval.yaml did not load as EvaluationConfig")
    return loaded.model_copy(
        update={
            "evaluation_id": "eval:phase2-verify-smoke-v1",
            "artifact_layout": artifact_layout,
            "bootstrap_replicates": 32,
            "notable_episode_count": 2,
        }
    )


def _simulation_seed_for_bundle(training_pack: "TrainingPackManifest", bundle_id: str) -> int:
    for entry in training_pack.entries:
        if entry.bundle_id == bundle_id:
            marker = str(entry.manifest_id).split("seed-")[-1]
            return int(marker)
    raise KeyError(f"unknown bundle_id {bundle_id!r}")


if __name__ == "__main__":
    raise SystemExit(main())
