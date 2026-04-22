from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from orbital_shepherd_training import (
    ModelArchitectureConfig,
    build_phase2_policy_model,
    export_phase2_policy_checkpoint,
)


def export_toy_policy_checkpoint(
    root: Path,
    *,
    run_id: str = "bc:test-trained-policy:v1",
    checkpoint_name: str = "checkpoint_000001",
    top_k: int = 4,
) -> Path:
    architecture = ModelArchitectureConfig(
        model_id="model:test-trained-policy:v1",
        observation_encoder_type="tabular_transformer",
        action_encoder_type="masked_action_mlp",
        hidden_dim=32,
        encoder_layers=2,
        attention_heads=4,
        dropout=0.0,
        recurrent_memory_steps=1,
        shared_policy_value_backbone=True,
    )
    model = build_phase2_policy_model(architecture=architecture, top_k=top_k)
    exported = export_phase2_policy_checkpoint(
        checkpoint_root=root / "checkpoints" / run_id,
        manifest_root=root / "manifests" / run_id,
        checkpoint_name=checkpoint_name,
        algorithm="behavior_cloning",
        run_id=run_id,
        benchmark_id="osbench-phase2-foundation-v1",
        reward_id="reward:test-trained-policy:v1",
        architecture=architecture,
        top_k=top_k,
        source_dataset_ids=["offds:test-trained-policy:v1"],
        source_training_pack_id="trainpack:test-trained-policy:v1",
        source_bundle_ids=["sb:test-trained-policy:v1"],
        global_step=1,
        metrics={"toy_metric": 1.0},
        metadata={"source_planner_ids": ["urgency_greedy"]},
        policy_state_dict=model.state_dict(),
        created_at=datetime(2026, 4, 14, 0, 0, tzinfo=UTC),
    )
    return exported.manifest_path
