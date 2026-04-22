from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch

from orbital_shepherd_contracts.paths import repo_root
from orbital_shepherd_policy_models.phase2_policy import OrbitalPhase2PolicyModel, Phase2PolicyModelConfig
from orbital_shepherd_policy_models.projection import CANDIDATE_FEATURE_SPECS, GLOBAL_FEATURE_SPECS

PolicyBackend = Literal["raw_state_dict", "rllib_module"]


@dataclass(frozen=True, slots=True)
class PolicyArchitectureSpec:
    top_k: int
    hidden_dim: int | None = None
    encoder_layers: int | None = None
    attention_heads: int | None = None
    dropout: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "hidden_dim": self.hidden_dim,
            "encoder_layers": self.encoder_layers,
            "attention_heads": self.attention_heads,
            "dropout": self.dropout,
        }


@dataclass(frozen=True, slots=True)
class PolicyCheckpointManifestRecord:
    checkpoint_id: str
    model_id: str
    run_id: str
    algorithm: str
    benchmark_id: str
    reward_id: str
    trainer_backend: str
    framework: str
    created_at_utc: datetime
    checkpoint_path: Path
    manifest_path: Path
    artifact_fingerprint: str
    global_step: int
    source_dataset_ids: tuple[str, ...]
    source_bundle_ids: tuple[str, ...]
    source_training_pack_id: str | None
    metadata: dict[str, Any]
    metrics: dict[str, float]

    @classmethod
    def load(cls, source: str | Path) -> PolicyCheckpointManifestRecord:
        path = Path(source)
        if path.is_dir():
            path = path / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            checkpoint_id=str(payload["checkpoint_id"]),
            model_id=str(payload["model_id"]),
            run_id=str(payload["run_id"]),
            algorithm=str(payload["algorithm"]),
            benchmark_id=str(payload["benchmark_id"]),
            reward_id=str(payload["reward_id"]),
            trainer_backend=str(payload["trainer_backend"]),
            framework=str(payload["framework"]),
            created_at_utc=_coerce_datetime(payload["created_at_utc"]),
            checkpoint_path=Path(str(payload["checkpoint_path"])).resolve(),
            manifest_path=path.resolve(),
            artifact_fingerprint=str(payload["artifact_fingerprint"]),
            global_step=int(payload["global_step"]),
            source_dataset_ids=tuple(str(item) for item in payload.get("source_dataset_ids", [])),
            source_bundle_ids=tuple(str(item) for item in payload.get("source_bundle_ids", [])),
            source_training_pack_id=(
                str(payload["source_training_pack_id"])
                if payload.get("source_training_pack_id") is not None
                else None
            ),
            metadata=dict(payload.get("metadata", {})),
            metrics={str(key): float(value) for key, value in dict(payload.get("metrics", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class RegisteredTrainedPolicy:
    model_key: str
    planner_id: str
    checkpoint_id: str
    checkpoint_manifest_path: Path
    checkpoint_path: Path
    checkpoint_fingerprint: str
    model_id: str
    run_id: str
    algorithm: str
    benchmark_id: str
    reward_id: str
    trainer_backend: str
    framework: str
    created_at_utc: datetime
    architecture: PolicyArchitectureSpec
    raw_policy_state_path: Path | None
    rllib_module_path: Path | None
    metadata: dict[str, Any]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "planner_id": self.planner_id,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_manifest_path": str(self.checkpoint_manifest_path),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "model_id": self.model_id,
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "benchmark_id": self.benchmark_id,
            "reward_id": self.reward_id,
            "trainer_backend": self.trainer_backend,
            "framework": self.framework,
            "created_at_utc": self.created_at_utc.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "architecture": self.architecture.to_dict(),
            "raw_policy_state_path": str(self.raw_policy_state_path) if self.raw_policy_state_path else None,
            "rllib_module_path": str(self.rllib_module_path) if self.rllib_module_path else None,
            "metadata": dict(sorted(self.metadata.items())),
            "metrics": dict(sorted(self.metrics.items())),
        }


class LoadedTrainedPolicy:
    def __init__(
        self,
        *,
        record: RegisteredTrainedPolicy,
        policy_model: OrbitalPhase2PolicyModel,
        backend: PolicyBackend,
    ) -> None:
        self.record = record
        self.policy_model = policy_model.to(torch.device("cpu"))
        self.policy_model.eval()
        self.backend = backend

    def inference_config(self) -> dict[str, Any]:
        config = getattr(self.policy_model, "config", None)
        if isinstance(config, Phase2PolicyModelConfig):
            model_config = {
                "top_k": config.top_k,
                "hidden_dim": config.hidden_dim,
                "encoder_layers": config.encoder_layers,
                "attention_heads": config.attention_heads,
                "dropout": config.dropout,
            }
        else:
            model_config = self.record.architecture.to_dict()
        return {
            "backend": self.backend,
            "device": "cpu",
            "model_config": model_config,
        }


class PolicyModelRegistry:
    def __init__(
        self,
        *,
        manifest_roots: Sequence[Path] | None = None,
        checkpoint_roots: Sequence[Path] | None = None,
    ) -> None:
        repo = repo_root()
        self._manifest_roots = tuple(
            Path(path).resolve()
            for path in (manifest_roots or (repo / "data" / "training" / "manifests",))
        )
        self._checkpoint_roots = tuple(
            Path(path).resolve()
            for path in (checkpoint_roots or (repo / "data" / "training" / "checkpoints",))
        )
        self._entries: dict[str, RegisteredTrainedPolicy] = {}
        self.refresh()

    def refresh(self) -> None:
        discovered: dict[str, RegisteredTrainedPolicy] = {}
        for manifest_path in self._discover_manifest_paths():
            manifest = PolicyCheckpointManifestRecord.load(manifest_path)
            entry = self._entry_from_manifest(manifest)
            discovered[entry.model_key] = entry
        self._entries = dict(sorted(discovered.items()))

    def register_checkpoint(
        self,
        *,
        manifest_path: str | Path,
        model_key: str,
    ) -> RegisteredTrainedPolicy:
        manifest = PolicyCheckpointManifestRecord.load(manifest_path)
        entry = self._entry_from_manifest(manifest, model_key=model_key)
        self._entries[entry.model_key] = entry
        return entry

    def list_policies(self) -> list[RegisteredTrainedPolicy]:
        return [self._entries[key] for key in sorted(self._entries)]

    def get_policy(self, model_key: str) -> RegisteredTrainedPolicy:
        try:
            return self._entries[model_key]
        except KeyError as exc:
            known = ", ".join(sorted(self._entries))
            raise KeyError(
                f"unknown trained policy {model_key!r}; expected one of: {known}"
            ) from exc

    def load_policy(self, model_key: str) -> LoadedTrainedPolicy:
        return _load_policy_from_entry(self.get_policy(model_key))

    def _discover_manifest_paths(self) -> list[Path]:
        manifest_paths: dict[str, Path] = {}
        for root in self._manifest_roots:
            if not root.exists():
                continue
            for path in root.rglob("checkpoint_*.json"):
                manifest_paths[str(path.resolve())] = path.resolve()
        for root in self._checkpoint_roots:
            if not root.exists():
                continue
            for path in root.rglob("manifest.json"):
                manifest_paths[str(path.resolve())] = path.resolve()
        return [manifest_paths[key] for key in sorted(manifest_paths)]

    def _entry_from_manifest(
        self,
        manifest: PolicyCheckpointManifestRecord,
        *,
        model_key: str | None = None,
    ) -> RegisteredTrainedPolicy:
        raw_policy_state_path = _resolve_optional_path(manifest.metadata.get("raw_policy_state_path"))
        rllib_module_path = _resolve_rllib_module_path(
            metadata=manifest.metadata,
            checkpoint_path=manifest.checkpoint_path,
        )
        architecture = PolicyArchitectureSpec(
            top_k=_coerce_optional_int(manifest.metadata.get("top_k")) or _infer_top_k_from_checkpoint_path(
                manifest.checkpoint_path
            ),
            hidden_dim=_coerce_optional_int(manifest.metadata.get("hidden_dim")),
            encoder_layers=_coerce_optional_int(manifest.metadata.get("encoder_layers")),
            attention_heads=_coerce_optional_int(manifest.metadata.get("attention_heads")),
            dropout=_coerce_optional_float(manifest.metadata.get("dropout")),
        )
        resolved_key = model_key or manifest.checkpoint_id
        return RegisteredTrainedPolicy(
            model_key=resolved_key,
            planner_id=f"trained_policy:{resolved_key}",
            checkpoint_id=manifest.checkpoint_id,
            checkpoint_manifest_path=manifest.manifest_path,
            checkpoint_path=manifest.checkpoint_path,
            checkpoint_fingerprint=manifest.artifact_fingerprint,
            model_id=manifest.model_id,
            run_id=manifest.run_id,
            algorithm=manifest.algorithm,
            benchmark_id=manifest.benchmark_id,
            reward_id=manifest.reward_id,
            trainer_backend=manifest.trainer_backend,
            framework=manifest.framework,
            created_at_utc=manifest.created_at_utc,
            architecture=architecture,
            raw_policy_state_path=raw_policy_state_path,
            rllib_module_path=rllib_module_path,
            metadata=dict(manifest.metadata),
            metrics=dict(manifest.metrics),
        )


@lru_cache(maxsize=1)
def default_policy_model_registry() -> PolicyModelRegistry:
    return PolicyModelRegistry()


def _load_policy_from_entry(entry: RegisteredTrainedPolicy) -> LoadedTrainedPolicy:
    if entry.raw_policy_state_path is not None:
        return LoadedTrainedPolicy(
            record=entry,
            policy_model=_load_raw_state_policy(entry),
            backend="raw_state_dict",
        )
    if entry.rllib_module_path is not None:
        return LoadedTrainedPolicy(
            record=entry,
            policy_model=_load_rllib_module_policy(entry),
            backend="rllib_module",
        )
    raise FileNotFoundError(
        f"checkpoint {entry.checkpoint_id} does not expose a CPU-loadable local policy artifact"
    )


def _load_raw_state_policy(entry: RegisteredTrainedPolicy) -> OrbitalPhase2PolicyModel:
    if entry.raw_policy_state_path is None:
        raise FileNotFoundError(f"checkpoint {entry.checkpoint_id} is missing raw policy state")
    architecture = entry.architecture
    if (
        architecture.hidden_dim is None
        or architecture.encoder_layers is None
        or architecture.attention_heads is None
        or architecture.dropout is None
    ):
        raise ValueError(
            f"checkpoint {entry.checkpoint_id} is missing architecture metadata for raw-state loading"
        )
    model = OrbitalPhase2PolicyModel(
        Phase2PolicyModelConfig(
            global_feature_dim=len(GLOBAL_FEATURE_SPECS),
            candidate_feature_dim=len(CANDIDATE_FEATURE_SPECS),
            action_dim=architecture.top_k + 1,
            top_k=architecture.top_k,
            hidden_dim=architecture.hidden_dim,
            encoder_layers=architecture.encoder_layers,
            attention_heads=architecture.attention_heads,
            dropout=architecture.dropout,
        )
    )
    loaded = torch.load(entry.raw_policy_state_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"expected a dict state dict at {entry.raw_policy_state_path}")
    model.load_state_dict(
        {
            str(key): value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            for key, value in loaded.items()
        }
    )
    return model


def _load_rllib_module_policy(entry: RegisteredTrainedPolicy) -> OrbitalPhase2PolicyModel:
    if entry.rllib_module_path is None:
        raise FileNotFoundError(f"checkpoint {entry.checkpoint_id} is missing an RLModule path")
    from orbital_shepherd_policy_models.rl_module import OrbitalMaskedActionTorchRLModule

    module = OrbitalMaskedActionTorchRLModule.from_checkpoint(str(entry.rllib_module_path))
    module.eval()
    module.policy_model.eval()
    return module.policy_model


def _resolve_rllib_module_path(
    *,
    metadata: Mapping[str, Any],
    checkpoint_path: Path,
) -> Path | None:
    metadata_path = _resolve_optional_path(metadata.get("rllib_module_path"))
    if metadata_path is not None:
        return metadata_path
    candidates = (
        checkpoint_path / "rllib_module" / "default_policy",
        checkpoint_path / "learner_group" / "learner" / "rl_module" / "default_policy",
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _resolve_optional_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        return None
    return path


def _infer_top_k_from_checkpoint_path(checkpoint_path: Path) -> int:
    rllib_path = _resolve_rllib_module_path(metadata={}, checkpoint_path=checkpoint_path)
    if rllib_path is None:
        raise ValueError(f"unable to infer top_k from checkpoint {checkpoint_path}")
    from orbital_shepherd_policy_models.rl_module import OrbitalMaskedActionTorchRLModule

    module = OrbitalMaskedActionTorchRLModule.from_checkpoint(str(rllib_path))
    candidate_shape = getattr(module.observation_space["candidate_features"], "shape", None)
    if not candidate_shape:
        raise ValueError(f"unable to infer top_k from checkpoint {checkpoint_path}")
    return int(candidate_shape[0])


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
