from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from ray.rllib.core.columns import Columns
from tests.benchmark.helpers import build_tiny_bundle

from orbital_shepherd_contracts import ScenarioBundle
from orbital_shepherd_core import canonical_json_dumps, sha256_fingerprint
from orbital_shepherd_training import (
    ModelArchitectureConfig,
    OrbitalMaskedActionTorchRLModule,
    PpoTrainingConfig,
    RllibOrbitalTrainingEnv,
    TrainingPackManifest,
    train_ppo_with_rllib,
)
from orbital_shepherd_training.models import (
    ArtifactLayoutConfig,
    ScenarioSplitEntry,
    ScenarioSplitRegistry,
    TrainingPackEntry,
    WandbConfig,
)


def test_rllib_env_and_module_share_the_same_observation_contract(tmp_path: Path) -> None:
    training_pack_path, split_registry_path = _write_tiny_training_contracts(tmp_path)
    env = RllibOrbitalTrainingEnv(
        {
            "bundle_paths": [str(training_pack_path.parent / "bundle-train.json")],
            "top_k": 4,
            "base_seed": 17,
            "planner_id": "planner:test-rllib",
        }
    )

    observation, _ = env.reset(seed=17)
    module = OrbitalMaskedActionTorchRLModule(
        observation_space=env.observation_space,
        action_space=env.action_space,
        inference_only=False,
        model_config={
            "hidden_dim": 32,
            "encoder_layers": 2,
            "attention_heads": 4,
            "dropout": 0.0,
        },
        catalog_class=None,
    )
    outputs = module._forward_inference(
        {
            Columns.OBS: {
                "global_features": torch.tensor(observation["global_features"]).unsqueeze(0),
                "candidate_features": torch.tensor(observation["candidate_features"]).unsqueeze(0),
                "action_mask": torch.tensor(observation["action_mask"]).unsqueeze(0),
            }
        }
    )

    assert outputs[Columns.ACTION_DIST_INPUTS].shape == (1, env.action_space.n)
    assert outputs[Columns.ACTION_DIST_INPUTS][0, 2].item() < -1.0e8
    env.close()
    assert training_pack_path.exists()
    assert split_registry_path.exists()


def test_short_rllib_ppo_smoke_run_writes_checkpoint_manifests(tmp_path: Path) -> None:
    training_pack_path, split_registry_path = _write_tiny_training_contracts(tmp_path)
    artifact_layout = ArtifactLayoutConfig(
        dataset_root=str(tmp_path / "datasets"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        report_root=str(tmp_path / "reports"),
        manifest_root=str(tmp_path / "manifests"),
        scenario_pack_root=str(tmp_path / "scenario-pack"),
    )
    summary = train_ppo_with_rllib(
        ppo_config=PpoTrainingConfig(
            run_id="ppo:test-rllib-smoke-v1",
            benchmark_id="osbench-phase2-foundation-v1",
            model_id="model:phase2-policy-transformer-v1",
            reward_id="reward:phase2-auditable-v1",
            curriculum_id="curriculum:phase2-foundation-v1",
            artifact_layout=artifact_layout,
            wandb=WandbConfig(
                enabled=False,
                mode="disabled",
                project="orbital-shepherd-phase2",
                entity=None,
                group="pytest",
                tags=["test"],
            ),
            seed=20260413,
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
            local_mode=True,
            log_level="INFO",
        ),
        model_config=ModelArchitectureConfig(
            model_id="model:phase2-policy-transformer-v1",
            observation_encoder_type="tabular_transformer",
            action_encoder_type="masked_action_mlp",
            hidden_dim=32,
            encoder_layers=2,
            attention_heads=4,
            dropout=0.0,
            recurrent_memory_steps=1,
            shared_policy_value_backbone=True,
        ),
    )

    assert summary.metrics_path.exists()
    assert summary.run_manifest_path.exists()
    assert summary.checkpoint_manifest_paths
    checkpoint_manifest = json.loads(
        summary.checkpoint_manifest_paths[-1].read_text(encoding="utf-8")
    )
    assert checkpoint_manifest["trainer_backend"] == "rllib"
    assert checkpoint_manifest["framework"] == "torch"
    assert checkpoint_manifest["source_bundle_ids"]
    assert Path(checkpoint_manifest["checkpoint_path"]).exists()
    assert checkpoint_manifest["metadata"]["enable_rl_module_api"] is True


def _write_tiny_training_contracts(tmp_path: Path) -> tuple[Path, Path]:
    bundle_path = tmp_path / "bundle-train.json"
    bundle = _phase2_tiny_bundle()
    bundle_path.write_text(
        canonical_json_dumps(bundle.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    training_pack = TrainingPackManifest(
        training_pack_id="trainpack:test-phase2-rllib:v1",
        benchmark_id="osbench-phase2-foundation-v1",
        split_registry_id="splitreg:test-phase2-rllib:v1",
        generated_at_utc=datetime(2026, 4, 13, 0, 0, tzinfo=UTC),
        output_dir=str(tmp_path),
        bundle_count=1,
        entries=[
            TrainingPackEntry(
                recipe_id="recipe:test-phase2-rllib:cloud-trap",
                manifest_id=bundle.manifest_id,
                bundle_id=bundle.bundle_id,
                scenario_family="cloud_trap",
                split="train",
                scenario_path=str(bundle_path),
                difficulty_tier=1,
            )
        ],
        artifact_fingerprint="sha256:" + sha256_fingerprint("trainpack:test-phase2-rllib:v1"),
    )
    training_pack_path = tmp_path / "phase2-training-pack-manifest.json"
    training_pack_path.write_text(
        canonical_json_dumps(training_pack.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    split_registry = ScenarioSplitRegistry(
        registry_id="splitreg:test-phase2-rllib:v1",
        benchmark_id="osbench-phase2-foundation-v1",
        seed_namespace="phase2-test-seed-contract",
        entries=[
            ScenarioSplitEntry(
                recipe_id="recipe:test-phase2-rllib:cloud-trap",
                manifest_id=bundle.manifest_id,
                bundle_id=bundle.bundle_id,
                scenario_family="cloud_trap",
                split="train",
                simulation_seed=bundle.simulation_seed,
                difficulty_tier=1,
                curriculum_stage="curriculum:test-phase2-rllib:stage1",
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
    document["manifest_id"] = "sm:test-phase2-rllib:cloud-trap:seed-17"
    document["bundle_id"] = "sb:test-phase2-rllib:cloud-trap:seed-17"
    document["simulation_seed"] = 17
    document["compilation"]["source_manifest_id"] = document["manifest_id"]
    document["bundle_fingerprint"] = sha256_fingerprint(document)
    return ScenarioBundle.model_validate(document)
