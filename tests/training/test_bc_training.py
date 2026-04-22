from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from tests.benchmark.helpers import build_tiny_bundle

from orbital_shepherd_contracts import ScenarioBundle
from orbital_shepherd_core import canonical_json_dumps, sha256_fingerprint
from orbital_shepherd_training import (
    BehaviorCloningConfig,
    ModelArchitectureConfig,
    PolicyCheckpointManifest,
    PolicyInitializationConfig,
    PpoTrainingConfig,
    ScenarioSplitRegistry,
    TrainingPackManifest,
    WandbConfig,
    build_phase2_rl_module,
    load_policy_checkpoint_manifest,
    load_raw_policy_state_dict,
    resolve_rllib_module_path,
    train_behavior_cloning,
    train_ppo_with_rllib,
)
from orbital_shepherd_training.models import (
    ArtifactLayoutConfig,
    ScenarioSplitEntry,
    TrainingPackEntry,
)


def test_behavior_cloning_overfits_tiny_slice(tmp_path: Path) -> None:
    training_pack_path, split_registry_path = _write_tiny_training_contracts(tmp_path)
    artifact_layout = _artifact_layout(tmp_path)
    architecture = _small_architecture()
    summary = train_behavior_cloning(
        bc_config=BehaviorCloningConfig(
            run_id="bc:test-overfit-v1",
            benchmark_id="osbench-phase2-foundation-v1",
            model_id=architecture.model_id,
            reward_id="reward:phase2-auditable-v1",
            curriculum_id="curriculum:phase2-foundation-v1",
            artifact_layout=artifact_layout,
            wandb=_wandb_config(),
            seed=7,
            batch_size=8,
            epochs=24,
            learning_rate=3.0e-3,
            weight_decay=0.0,
            label_smoothing=0.0,
            training_pack_path=str(training_pack_path),
            split_registry_path=str(split_registry_path),
            source_planner_ids=["urgency_greedy"],
            train_split="train",
            validation_split="train",
            top_k=4,
            limit_bundles_per_split=1,
            max_train_transitions=8,
            max_validation_transitions=8,
            class_weighting="none",
            class_weight_clip=None,
            checkpoint_frequency=6,
            checkpoint_at_end=True,
            require_parquet=False,
            device="cpu",
            early_stopping={
                "enabled": False,
                "metric": "val_loss",
                "mode": "min",
                "patience": 2,
                "min_delta": 0.0,
            },
        ),
        model_config=architecture,
    )

    assert summary.metrics_path.exists()
    assert summary.best_checkpoint_manifest_path is not None
    assert float(summary.final_metrics["train_accuracy"] or 0.0) >= 0.95
    assert float(summary.final_metrics["train_loss"] or 1.0) <= 0.25


def test_behavior_cloning_checkpoint_exports_shared_policy_for_rllib(tmp_path: Path) -> None:
    training_pack_path, split_registry_path = _write_tiny_training_contracts(tmp_path)
    artifact_layout = _artifact_layout(tmp_path)
    architecture = _small_architecture()
    summary = train_behavior_cloning(
        bc_config=_bc_config(
            artifact_layout=artifact_layout,
            training_pack_path=training_pack_path,
            split_registry_path=split_registry_path,
            run_id="bc:test-export-v1",
            epochs=6,
        ),
        model_config=architecture,
    )

    assert summary.best_checkpoint_manifest_path is not None
    manifest = load_policy_checkpoint_manifest(summary.best_checkpoint_manifest_path)
    raw_state = load_raw_policy_state_dict(manifest)
    restored_module = build_phase2_rl_module(architecture=architecture, top_k=4)
    restored_module.restore_from_path(resolve_rllib_module_path(manifest))

    assert manifest.source_dataset_ids
    assert Path(manifest.checkpoint_path).exists()
    assert Path(str(manifest.metadata["raw_policy_state_path"])).exists()
    assert Path(str(manifest.metadata["rllib_module_path"])).exists()
    for key, value in restored_module.policy_model.state_dict().items():
        assert key in raw_state
        assert torch.allclose(value.cpu(), raw_state[key].cpu())


def test_ppo_warm_start_from_bc_writes_consistent_manifests(tmp_path: Path) -> None:
    training_pack_path, split_registry_path = _write_tiny_training_contracts(tmp_path)
    artifact_layout = _artifact_layout(tmp_path)
    architecture = _small_architecture()
    bc_summary = train_behavior_cloning(
        bc_config=_bc_config(
            artifact_layout=artifact_layout,
            training_pack_path=training_pack_path,
            split_registry_path=split_registry_path,
            run_id="bc:test-warmstart-v1",
            epochs=4,
        ),
        model_config=architecture,
    )
    assert bc_summary.best_checkpoint_manifest_path is not None
    bc_manifest = load_policy_checkpoint_manifest(bc_summary.best_checkpoint_manifest_path)

    ppo_summary = train_ppo_with_rllib(
        ppo_config=PpoTrainingConfig(
            run_id="ppo:test-bc-warmstart-v1",
            benchmark_id="osbench-phase2-foundation-v1",
            model_id=architecture.model_id,
            reward_id="reward:phase2-auditable-v1",
            curriculum_id="curriculum:phase2-foundation-v1",
            artifact_layout=artifact_layout,
            wandb=_wandb_config(),
            seed=11,
            total_timesteps=64,
            rollout_steps=32,
            minibatch_size=16,
            update_epochs=1,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            learning_rate=1.0e-4,
            max_grad_norm=0.8,
            target_kl=0.03,
            framework="torch",
            trainer_backend="rllib",
            training_pack_path=str(training_pack_path),
            split_registry_path=str(split_registry_path),
            train_split="train",
            evaluation_split="train",
            top_k=4,
            scenario_limit=1,
            num_env_runners=0,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1.0,
            num_gpus=0.0,
            evaluation_interval=1,
            evaluation_duration=1,
            checkpoint_frequency=1,
            checkpoint_at_end=True,
            enable_rl_module_api=True,
            initialization=PolicyInitializationConfig(
                mode="checkpoint",
                checkpoint_manifest_path=str(bc_summary.best_checkpoint_manifest_path),
                source_run_id=None,
                selection="latest",
            ),
            local_mode=True,
            log_level="INFO",
        ),
        model_config=architecture,
    )

    run_manifest = json.loads(ppo_summary.run_manifest_path.read_text(encoding="utf-8"))
    checkpoint_manifest = PolicyCheckpointManifest.model_validate(
        json.loads(ppo_summary.checkpoint_manifest_paths[-1].read_text(encoding="utf-8"))
    )

    assert run_manifest["warm_start_checkpoint_id"] == bc_manifest.checkpoint_id
    assert checkpoint_manifest.metadata["warm_start_checkpoint_id"] == bc_manifest.checkpoint_id
    assert checkpoint_manifest.source_dataset_ids == bc_manifest.source_dataset_ids
    assert Path(checkpoint_manifest.checkpoint_path).exists()


def _bc_config(
    *,
    artifact_layout: ArtifactLayoutConfig,
    training_pack_path: Path,
    split_registry_path: Path,
    run_id: str,
    epochs: int,
) -> BehaviorCloningConfig:
    return BehaviorCloningConfig(
        run_id=run_id,
        benchmark_id="osbench-phase2-foundation-v1",
        model_id="model:phase2-policy-transformer-v1",
        reward_id="reward:phase2-auditable-v1",
        curriculum_id="curriculum:phase2-foundation-v1",
        artifact_layout=artifact_layout,
        wandb=_wandb_config(),
        seed=5,
        batch_size=16,
        epochs=epochs,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        label_smoothing=0.0,
        training_pack_path=str(training_pack_path),
        split_registry_path=str(split_registry_path),
        source_planner_ids=["urgency_greedy"],
        train_split="train",
        validation_split="train",
        top_k=4,
        limit_bundles_per_split=1,
        max_train_transitions=16,
        max_validation_transitions=16,
        class_weighting="none",
        class_weight_clip=None,
        checkpoint_frequency=2,
        checkpoint_at_end=True,
        require_parquet=False,
        device="cpu",
        early_stopping={
            "enabled": False,
            "metric": "val_loss",
            "mode": "min",
            "patience": 2,
            "min_delta": 0.0,
        },
    )


def _artifact_layout(tmp_path: Path) -> ArtifactLayoutConfig:
    return ArtifactLayoutConfig(
        dataset_root=str(tmp_path / "datasets"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        report_root=str(tmp_path / "reports"),
        manifest_root=str(tmp_path / "manifests"),
        scenario_pack_root=str(tmp_path / "scenario-pack"),
    )


def _small_architecture() -> ModelArchitectureConfig:
    return ModelArchitectureConfig(
        model_id="model:phase2-policy-transformer-v1",
        observation_encoder_type="tabular_transformer",
        action_encoder_type="masked_action_mlp",
        hidden_dim=32,
        encoder_layers=2,
        attention_heads=4,
        dropout=0.0,
        recurrent_memory_steps=1,
        shared_policy_value_backbone=True,
    )


def _wandb_config() -> WandbConfig:
    return WandbConfig(
        enabled=False,
        mode="disabled",
        project="orbital-shepherd-phase2",
        entity=None,
        group="pytest",
        tags=["test"],
    )


def _write_tiny_training_contracts(tmp_path: Path) -> tuple[Path, Path]:
    bundle_path = tmp_path / "bundle-train.json"
    bundle = _phase2_tiny_bundle()
    bundle_path.write_text(
        canonical_json_dumps(bundle.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    training_pack = TrainingPackManifest(
        training_pack_id="trainpack:test-phase2-bc:v1",
        benchmark_id="osbench-phase2-foundation-v1",
        split_registry_id="splitreg:test-phase2-bc:v1",
        generated_at_utc=datetime(2026, 4, 13, 0, 0, tzinfo=UTC),
        output_dir=str(tmp_path),
        bundle_count=1,
        entries=[
            TrainingPackEntry(
                recipe_id="recipe:test-phase2-bc:cloud-trap",
                manifest_id=bundle.manifest_id,
                bundle_id=bundle.bundle_id,
                scenario_family="cloud_trap",
                split="train",
                scenario_path=str(bundle_path),
                difficulty_tier=1,
            )
        ],
        artifact_fingerprint="sha256:" + sha256_fingerprint("trainpack:test-phase2-bc:v1"),
    )
    training_pack_path = tmp_path / "phase2-training-pack-manifest.json"
    training_pack_path.write_text(
        canonical_json_dumps(training_pack.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    split_registry = ScenarioSplitRegistry(
        registry_id="splitreg:test-phase2-bc:v1",
        benchmark_id="osbench-phase2-foundation-v1",
        seed_namespace="phase2-test-bc-contract",
        entries=[
            ScenarioSplitEntry(
                recipe_id="recipe:test-phase2-bc:cloud-trap",
                manifest_id=bundle.manifest_id,
                bundle_id=bundle.bundle_id,
                scenario_family="cloud_trap",
                split="train",
                simulation_seed=bundle.simulation_seed,
                difficulty_tier=1,
                curriculum_stage="curriculum:test-phase2-bc:stage1",
            )
        ],
    )
    split_registry_path = tmp_path / "phase2-splits.yaml"
    split_registry_path.write_text(
        canonical_json_dumps(split_registry.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    return training_pack_path, split_registry_path


def _phase2_tiny_bundle() -> ScenarioBundle:
    bundle = build_tiny_bundle()
    document = bundle.model_dump(mode="json")
    document["benchmark_id"] = "osbench-phase2-foundation-v1"
    document["manifest_id"] = "sm:test-phase2-bc:cloud-trap:seed-17"
    document["bundle_id"] = "sb:test-phase2-bc:cloud-trap:seed-17"
    document["simulation_seed"] = 17
    document["compilation"]["source_manifest_id"] = document["manifest_id"]
    document["bundle_fingerprint"] = sha256_fingerprint(document)
    return ScenarioBundle.model_validate(document)
