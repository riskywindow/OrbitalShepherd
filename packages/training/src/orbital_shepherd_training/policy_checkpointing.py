from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from orbital_shepherd_core import (
    canonical_json_dumps,
    format_utc_timestamp,
    sha256_fingerprint,
    stable_id,
)
from orbital_shepherd_policy_models import OrbitalPhase2PolicyModel, Phase2PolicyModelConfig
from orbital_shepherd_training.models import (
    ModelArchitectureConfig,
    PolicyCheckpointManifest,
    PolicyInitializationConfig,
)
from orbital_shepherd_training.rllib_module import OrbitalMaskedActionTorchRLModule
from orbital_shepherd_training.training_env import (
    CANDIDATE_FEATURE_SPECS,
    GLOBAL_FEATURE_SPECS,
)


@dataclass(frozen=True, slots=True)
class ExportedPolicyCheckpoint:
    checkpoint_path: Path
    manifest_path: Path
    manifest: PolicyCheckpointManifest
    raw_policy_state_path: Path
    rllib_module_path: Path


def build_phase2_policy_model(
    *,
    architecture: ModelArchitectureConfig,
    top_k: int,
) -> OrbitalPhase2PolicyModel:
    return OrbitalPhase2PolicyModel(
        Phase2PolicyModelConfig(
            global_feature_dim=len(GLOBAL_FEATURE_SPECS),
            candidate_feature_dim=len(CANDIDATE_FEATURE_SPECS),
            action_dim=top_k + 1,
            top_k=top_k,
            hidden_dim=architecture.hidden_dim,
            encoder_layers=architecture.encoder_layers,
            attention_heads=architecture.attention_heads,
            dropout=architecture.dropout,
        )
    )


def export_phase2_policy_checkpoint(
    *,
    checkpoint_root: Path,
    manifest_root: Path,
    checkpoint_name: str,
    algorithm: str,
    run_id: str,
    benchmark_id: str,
    reward_id: str,
    architecture: ModelArchitectureConfig,
    top_k: int,
    source_dataset_ids: list[str],
    source_training_pack_id: str | None,
    source_bundle_ids: list[str],
    global_step: int,
    metrics: Mapping[str, float],
    metadata: Mapping[str, str | int | float | bool | list[str] | None],
    policy_state_dict: Mapping[str, torch.Tensor],
    created_at: datetime | None = None,
    trainer_backend: str = "torch",
    framework: str = "torch",
) -> ExportedPolicyCheckpoint:
    resolved_created_at = (created_at or datetime.now(UTC)).astimezone(UTC)
    checkpoint_path = checkpoint_root / checkpoint_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    raw_policy_state_path = checkpoint_path / "policy_model_state.pt"
    detached_state = {
        key: value.detach().cpu()
        for key, value in policy_state_dict.items()
    }
    torch.save(detached_state, raw_policy_state_path)

    module = build_phase2_rl_module(architecture=architecture, top_k=top_k)
    module.policy_model.load_state_dict(detached_state)
    rllib_module_path = checkpoint_path / "rllib_module" / "default_policy"
    module.save_to_path(rllib_module_path)

    manifest_payload: dict[str, Any] = {
        "checkpoint_id": stable_id("ckpt", run_id, checkpoint_name.replace("_", "-")),
        "algorithm": algorithm,
        "created_at_utc": format_utc_timestamp(resolved_created_at),
        "benchmark_id": benchmark_id,
        "run_id": run_id,
        "model_id": architecture.model_id,
        "reward_id": reward_id,
        "trainer_backend": trainer_backend,
        "framework": framework,
        "source_dataset_ids": source_dataset_ids,
        "source_training_pack_id": source_training_pack_id,
        "source_bundle_ids": source_bundle_ids,
        "checkpoint_path": str(checkpoint_path),
        "global_step": global_step,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "metadata": {
            **dict(metadata),
            "raw_policy_state_path": str(raw_policy_state_path),
            "rllib_module_path": str(rllib_module_path),
            "top_k": top_k,
            "hidden_dim": architecture.hidden_dim,
            "encoder_layers": architecture.encoder_layers,
            "attention_heads": architecture.attention_heads,
            "dropout": architecture.dropout,
        },
    }
    manifest_payload["artifact_fingerprint"] = "sha256:" + sha256_fingerprint(
        {
            key: value
            for key, value in manifest_payload.items()
            if key != "artifact_fingerprint"
        }
    )
    manifest = PolicyCheckpointManifest.model_validate(manifest_payload)
    checkpoint_manifest_path = checkpoint_path / "manifest.json"
    checkpoint_manifest_path.write_text(
        canonical_json_dumps(manifest.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    manifest_index_path = manifest_root / f"{checkpoint_name}.json"
    manifest_index_path.write_text(
        canonical_json_dumps(manifest.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    return ExportedPolicyCheckpoint(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_index_path,
        manifest=manifest,
        raw_policy_state_path=raw_policy_state_path,
        rllib_module_path=rllib_module_path,
    )


def load_policy_checkpoint_manifest(source: str | Path) -> PolicyCheckpointManifest:
    path = Path(source)
    if path.is_dir():
        path = path / "manifest.json"
    return PolicyCheckpointManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def resolve_policy_initialization_manifest(
    *,
    initialization: PolicyInitializationConfig,
    manifest_root: Path,
) -> PolicyCheckpointManifest | None:
    if initialization.mode == "scratch":
        return None
    if initialization.checkpoint_manifest_path is not None:
        return load_policy_checkpoint_manifest(initialization.checkpoint_manifest_path)
    assert initialization.source_run_id is not None
    checkpoint_manifests: list[PolicyCheckpointManifest] = []
    for path in sorted(manifest_root.glob("*/checkpoint_*.json")):
        manifest = load_policy_checkpoint_manifest(path)
        if manifest.run_id == initialization.source_run_id:
            checkpoint_manifests.append(manifest)
    if not checkpoint_manifests:
        raise FileNotFoundError(
            "no checkpoint manifests found for "
            f"run_id={initialization.source_run_id} in {manifest_root}"
        )
    return max(
        checkpoint_manifests,
        key=lambda item: (item.global_step, item.created_at_utc, item.checkpoint_id),
    )


def resolve_rllib_module_path(manifest: PolicyCheckpointManifest) -> Path:
    rllib_module_path = manifest.metadata.get("rllib_module_path")
    if not isinstance(rllib_module_path, str) or not rllib_module_path:
        raise ValueError(
            f"checkpoint {manifest.checkpoint_id} does not expose an RLModule adapter path"
        )
    path = Path(rllib_module_path)
    if not path.exists():
        raise FileNotFoundError(f"RLModule adapter path does not exist: {path}")
    return path


def load_raw_policy_state_dict(manifest: PolicyCheckpointManifest) -> dict[str, torch.Tensor]:
    state_path = manifest.metadata.get("raw_policy_state_path")
    if not isinstance(state_path, str) or not state_path:
        raise ValueError(
            f"checkpoint {manifest.checkpoint_id} does not expose a raw policy state path"
        )
    loaded = torch.load(Path(state_path), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"expected a dict state dict at {state_path}")
    return {
        str(key): value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        for key, value in loaded.items()
    }


def validate_policy_checkpoint_compatibility(
    *,
    manifest: PolicyCheckpointManifest,
    architecture: ModelArchitectureConfig,
    top_k: int,
) -> None:
    if manifest.model_id != architecture.model_id:
        raise ValueError(
            f"checkpoint model_id {manifest.model_id} does not match {architecture.model_id}"
        )
    expected_metadata = {
        "top_k": top_k,
        "hidden_dim": architecture.hidden_dim,
        "encoder_layers": architecture.encoder_layers,
        "attention_heads": architecture.attention_heads,
        "dropout": architecture.dropout,
    }
    for key, expected_value in expected_metadata.items():
        actual_value = manifest.metadata.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"checkpoint metadata {key}={actual_value!r} does not match {expected_value!r}"
            )


def build_phase2_rl_module(
    *,
    architecture: ModelArchitectureConfig,
    top_k: int,
) -> OrbitalMaskedActionTorchRLModule:
    observation_space = gym.spaces.Dict(
        {
            "global_features": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(GLOBAL_FEATURE_SPECS),),
                dtype=np.float32,
            ),
            "candidate_features": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(top_k, len(CANDIDATE_FEATURE_SPECS)),
                dtype=np.float32,
            ),
            "action_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(top_k + 1,),
                dtype=np.int8,
            ),
        }
    )
    return OrbitalMaskedActionTorchRLModule(
        observation_space=observation_space,
        action_space=gym.spaces.Discrete(top_k + 1),
        inference_only=False,
        model_config={
            "hidden_dim": architecture.hidden_dim,
            "encoder_layers": architecture.encoder_layers,
            "attention_heads": architecture.attention_heads,
            "dropout": architecture.dropout,
        },
        catalog_class=None,
    )
