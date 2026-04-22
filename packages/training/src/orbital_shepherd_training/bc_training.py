from __future__ import annotations

import json
import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from orbital_shepherd_core import (
    canonical_json_dumps,
    format_utc_timestamp,
    stable_id,
)
from orbital_shepherd_training.config_io import load_phase2_config
from orbital_shepherd_training.models import (
    BehaviorCloningConfig,
    ModelArchitectureConfig,
    OfflineDatasetManifest,
)
from orbital_shepherd_training.offline_dataset import (
    DEFAULT_EXPERT_PLANNER_IDS,
    build_offline_datasets,
)
from orbital_shepherd_training.policy_checkpointing import (
    build_phase2_policy_model,
    export_phase2_policy_checkpoint,
)
from orbital_shepherd_training.tracking import maybe_init_wandb

try:  # pragma: no cover - exercised when numpy is installed.
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - covered by local tests.
    np = None


@dataclass(frozen=True, slots=True)
class BehaviorCloningTrainingSummary:
    run_id: str
    run_dir: Path
    metrics_path: Path
    summary_report_path: Path
    run_manifest_path: Path
    dataset_build_manifest_path: Path
    train_dataset_manifest_path: Path
    validation_dataset_manifest_path: Path
    checkpoint_manifest_paths: tuple[Path, ...]
    best_checkpoint_manifest_path: Path | None
    final_metrics: dict[str, float | int | str | bool | None]


def train_behavior_cloning(
    *,
    bc_config: BehaviorCloningConfig | Path,
    model_config: ModelArchitectureConfig | Path,
) -> BehaviorCloningTrainingSummary:
    bc = _load_bc_config(bc_config)
    architecture = _load_model_config(model_config)
    selected_planners = tuple(bc.source_planner_ids or DEFAULT_EXPERT_PLANNER_IDS)
    selected_splits = tuple(dict.fromkeys((bc.train_split, bc.validation_split)))

    dataset_output_root = Path(bc.artifact_layout.dataset_root) / bc.benchmark_id
    manifest_root = Path(bc.artifact_layout.manifest_root)
    dataset_build_manifest = build_offline_datasets(
        training_pack=Path(bc.training_pack_path),
        split_registry=Path(bc.split_registry_path),
        output_root=dataset_output_root,
        manifest_root=manifest_root,
        planner_ids=selected_planners,
        splits=selected_splits,
        top_k=bc.top_k,
        algorithm="behavior_cloning",
        reward_id=bc.reward_id,
        limit_bundles_per_split=bc.limit_bundles_per_split,
        build_id=bc.dataset_build_id,
        require_parquet=bc.require_parquet,
    )
    dataset_build_manifest_path = manifest_root / (
        f"{str(dataset_build_manifest.build_id).replace(':', '--')}.json"
    )
    dataset_manifests = {
        manifest.split: manifest
        for manifest in (
            OfflineDatasetManifest.model_validate(
                json.loads(Path(path).read_text(encoding="utf-8"))
            )
            for path in dataset_build_manifest.dataset_manifests
        )
    }
    train_manifest = dataset_manifests[bc.train_split]
    validation_manifest = dataset_manifests[bc.validation_split]
    source_dataset_ids = list(
        dict.fromkeys([train_manifest.dataset_id, validation_manifest.dataset_id])
    )
    train_arrays = _load_training_arrays(
        train_manifest,
        max_records=bc.max_train_transitions,
    )
    validation_arrays = _load_training_arrays(
        validation_manifest,
        max_records=bc.max_validation_transitions,
    )
    if train_arrays.tensors[0].shape[0] == 0:
        raise ValueError("behavior cloning requires at least one training transition")
    if train_manifest.top_k != bc.top_k or validation_manifest.top_k != bc.top_k:
        raise ValueError("dataset top_k does not match the BC config top_k")

    started_at = datetime.now(UTC)
    run_label = _safe_filename(
        stable_id("trainrun", bc.run_id, format_utc_timestamp(started_at).lower())
    )
    checkpoint_run_dir = (Path(bc.artifact_layout.checkpoint_root) / run_label).resolve()
    report_run_dir = (Path(bc.artifact_layout.report_root) / run_label).resolve()
    manifest_run_dir = (manifest_root / run_label).resolve()
    checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
    report_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_run_dir / "metrics.jsonl"
    summary_report_path = report_run_dir / "summary.json"
    run_manifest_path = manifest_run_dir / "run_manifest.json"

    _set_random_seeds(bc.seed)
    device = _resolve_device(bc.device)
    model = build_phase2_policy_model(architecture=architecture, top_k=bc.top_k).to(device)
    class_weights = _compute_class_weights(
        labels=train_arrays.labels,
        action_dim=bc.top_k + 1,
        strategy=bc.class_weighting,
        clip=bc.class_weight_clip,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=bc.learning_rate,
        weight_decay=bc.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=bc.label_smoothing,
    )
    train_loader = DataLoader(
        train_arrays.dataset,
        batch_size=min(bc.batch_size, len(train_arrays.dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(bc.seed),
    )
    validation_loader = DataLoader(
        validation_arrays.dataset,
        batch_size=min(bc.batch_size, len(validation_arrays.dataset)),
        shuffle=False,
    )

    run_manifest_path.write_text(
        canonical_json_dumps(
            {
                "run_id": bc.run_id,
                "benchmark_id": bc.benchmark_id,
                "algorithm": "behavior_cloning",
                "trainer_backend": "torch",
                "framework": "torch",
                "started_at_utc": format_utc_timestamp(started_at),
                "seed": bc.seed,
                "training_pack_path": bc.training_pack_path,
                "split_registry_path": bc.split_registry_path,
                "train_split": bc.train_split,
                "validation_split": bc.validation_split,
                "source_planner_ids": list(selected_planners),
                "device": device.type,
                "dataset_build_manifest_path": str(dataset_build_manifest_path),
                "train_dataset_manifest_path": str(
                    Path(train_manifest.dataset_path) / "manifest.json"
                ),
                "validation_dataset_manifest_path": str(
                    Path(validation_manifest.dataset_path) / "manifest.json"
                ),
                "bc_config": bc.model_dump(mode="json"),
                "model_config": architecture.model_dump(mode="json"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    wandb_run = maybe_init_wandb(
        bc.wandb,
        run_name=bc.run_id,
        run_config={
            "algorithm": "behavior_cloning",
            "benchmark_id": bc.benchmark_id,
            "model_id": architecture.model_id,
            "reward_id": bc.reward_id,
            "train_split": bc.train_split,
            "validation_split": bc.validation_split,
            "source_planner_ids": list(selected_planners),
            "top_k": bc.top_k,
        },
    )
    checkpoint_manifest_paths: list[Path] = []
    best_checkpoint_manifest_path: Path | None = None
    best_metric = float("inf") if bc.early_stopping.mode == "min" else float("-inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    epochs_since_improvement = 0
    global_step = 0
    final_metrics: dict[str, float | int | str | bool | None] = {}
    stopped_early = False
    last_row: dict[str, float | int] | None = None

    try:
        for epoch in range(1, bc.epochs + 1):
            train_metrics = _run_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=True,
            )
            global_step += int(train_metrics["examples"])
            validation_metrics = _run_epoch(
                model=model,
                data_loader=validation_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                train=False,
            )
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_examples": int(train_metrics["examples"]),
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["accuracy"],
                "val_examples": int(validation_metrics["examples"]),
                "learning_rate": bc.learning_rate,
            }
            last_row = row
            _append_metrics_row(metrics_path, row)
            wandb_run.log(row, step=epoch)

            current_metric = float(row[str(bc.early_stopping.metric)])
            if _is_improvement(
                current=current_metric,
                best=best_metric,
                mode=bc.early_stopping.mode,
                min_delta=bc.early_stopping.min_delta,
            ):
                best_metric = current_metric
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epoch % bc.checkpoint_frequency == 0:
                exported = export_phase2_policy_checkpoint(
                    checkpoint_root=checkpoint_run_dir,
                    manifest_root=manifest_run_dir,
                    checkpoint_name=f"checkpoint_{epoch:06d}",
                    algorithm="behavior_cloning",
                    run_id=bc.run_id,
                    benchmark_id=bc.benchmark_id,
                    reward_id=bc.reward_id,
                    architecture=architecture,
                    top_k=bc.top_k,
                    source_dataset_ids=source_dataset_ids,
                    source_training_pack_id=train_manifest.source_training_pack_id,
                    source_bundle_ids=sorted(
                        {
                            *train_manifest.source_bundle_ids,
                            *validation_manifest.source_bundle_ids,
                        }
                    ),
                    global_step=global_step,
                    metrics={
                        "train_loss": float(row["train_loss"]),
                        "train_accuracy": float(row["train_accuracy"]),
                        "val_loss": float(row["val_loss"]),
                        "val_accuracy": float(row["val_accuracy"]),
                    },
                    metadata={
                        "checkpoint_epoch": epoch,
                        "best_epoch_so_far": best_epoch,
                        "best_metric_so_far": best_metric,
                        "run_manifest_path": str(run_manifest_path),
                        "train_split": bc.train_split,
                        "validation_split": bc.validation_split,
                        "source_planner_ids": list(selected_planners),
                        "class_weighting": bc.class_weighting,
                        "device": device.type,
                    },
                    policy_state_dict=model.state_dict(),
                    created_at=datetime.now(UTC),
                )
                checkpoint_manifest_paths.append(exported.manifest_path)
                if epoch == best_epoch:
                    best_checkpoint_manifest_path = exported.manifest_path

            if bc.early_stopping.enabled and epochs_since_improvement >= bc.early_stopping.patience:
                stopped_early = True
                break

        model.load_state_dict(best_state)
        if bc.checkpoint_at_end:
            final_epoch = best_epoch if best_epoch else min(bc.epochs, 1)
            exported = export_phase2_policy_checkpoint(
                checkpoint_root=checkpoint_run_dir,
                manifest_root=manifest_run_dir,
                checkpoint_name=f"checkpoint_{len(checkpoint_manifest_paths) + 1:06d}",
                algorithm="behavior_cloning",
                run_id=bc.run_id,
                benchmark_id=bc.benchmark_id,
                reward_id=bc.reward_id,
                architecture=architecture,
                top_k=bc.top_k,
                source_dataset_ids=source_dataset_ids,
                source_training_pack_id=train_manifest.source_training_pack_id,
                source_bundle_ids=sorted(
                    {
                        *train_manifest.source_bundle_ids,
                        *validation_manifest.source_bundle_ids,
                    }
                ),
                global_step=global_step,
                metrics={
                    "best_validation_metric": float(best_metric),
                    "best_epoch": float(best_epoch),
                },
                metadata={
                    "checkpoint_epoch": final_epoch,
                    "best_epoch_so_far": best_epoch,
                    "best_metric_so_far": best_metric,
                    "run_manifest_path": str(run_manifest_path),
                    "train_split": bc.train_split,
                    "validation_split": bc.validation_split,
                    "source_planner_ids": list(selected_planners),
                    "class_weighting": bc.class_weighting,
                    "device": device.type,
                    "is_best_checkpoint": True,
                },
                policy_state_dict=model.state_dict(),
                created_at=datetime.now(UTC),
            )
            checkpoint_manifest_paths.append(exported.manifest_path)
            best_checkpoint_manifest_path = exported.manifest_path
    finally:
        wandb_run.finish()

    final_metrics = {
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric if best_epoch else None,
        "stopped_early": stopped_early,
        "global_step": global_step,
        "train_examples": len(train_arrays.dataset),
        "validation_examples": len(validation_arrays.dataset),
        "device": device.type,
        "train_loss": last_row["train_loss"] if last_row is not None else None,
        "train_accuracy": last_row["train_accuracy"] if last_row is not None else None,
        "val_loss": last_row["val_loss"] if last_row is not None else None,
        "val_accuracy": last_row["val_accuracy"] if last_row is not None else None,
    }
    summary_report_path.write_text(
        canonical_json_dumps(final_metrics) + "\n",
        encoding="utf-8",
    )
    return BehaviorCloningTrainingSummary(
        run_id=bc.run_id,
        run_dir=checkpoint_run_dir,
        metrics_path=metrics_path,
        summary_report_path=summary_report_path,
        run_manifest_path=run_manifest_path,
        dataset_build_manifest_path=dataset_build_manifest_path,
        train_dataset_manifest_path=Path(train_manifest.dataset_path) / "manifest.json",
        validation_dataset_manifest_path=Path(validation_manifest.dataset_path) / "manifest.json",
        checkpoint_manifest_paths=tuple(checkpoint_manifest_paths),
        best_checkpoint_manifest_path=best_checkpoint_manifest_path,
        final_metrics=final_metrics,
    )


@dataclass(frozen=True, slots=True)
class LoadedTrainingArrays:
    dataset: TensorDataset
    labels: torch.Tensor
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _load_training_arrays(
    manifest: OfflineDatasetManifest,
    *,
    max_records: int | None,
) -> LoadedTrainingArrays:
    arrays_path = next(
        (
            Path(artifact.path)
            for artifact in manifest.artifacts
            if artifact.artifact_role == "training_arrays" and artifact.format == "npz"
        ),
        None,
    )
    if arrays_path is None or np is None:
        tensors = _load_training_arrays_from_steps(manifest, max_records=max_records)
        return LoadedTrainingArrays(
            dataset=TensorDataset(*tensors),
            labels=tensors[-1],
            tensors=tensors,
        )

    with np.load(arrays_path) as loaded:
        global_features = torch.tensor(loaded["global_features"], dtype=torch.float32)
        candidate_features = torch.tensor(loaded["candidate_features"], dtype=torch.float32)
        action_mask = torch.tensor(loaded["action_mask"], dtype=torch.float32)
        labels = torch.tensor(loaded["selected_slot"], dtype=torch.long)
    if max_records is not None:
        global_features = global_features[:max_records]
        candidate_features = candidate_features[:max_records]
        action_mask = action_mask[:max_records]
        labels = labels[:max_records]
    tensors = (global_features, candidate_features, action_mask, labels)
    return LoadedTrainingArrays(
        dataset=TensorDataset(*tensors),
        labels=labels,
        tensors=tensors,
    )


def _load_training_arrays_from_steps(
    manifest: OfflineDatasetManifest,
    *,
    max_records: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    steps_path = next(
        Path(artifact.path)
        for artifact in manifest.artifacts
        if artifact.artifact_role == "canonical_steps" and artifact.format == "jsonl"
    )
    rows = []
    with steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_records is not None and len(rows) >= max_records:
                break
    return (
        torch.tensor([row["global_features"] for row in rows], dtype=torch.float32),
        torch.tensor([row["candidate_features"] for row in rows], dtype=torch.float32),
        torch.tensor([row["training_action_mask"] for row in rows], dtype=torch.float32),
        torch.tensor([row["selected_slot"] for row in rows], dtype=torch.long),
    )


def _run_epoch(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for global_features, candidate_features, action_mask, labels in data_loader:
        global_features = global_features.to(device)
        candidate_features = candidate_features.to(device)
        action_mask = action_mask.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            outputs = model(
                global_features=global_features,
                candidate_features=candidate_features,
                action_mask=action_mask,
            )
            loss = criterion(outputs.masked_logits, labels)
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()
        predictions = torch.argmax(outputs.masked_logits, dim=-1)
        batch_size = labels.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_examples += batch_size
    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "examples": 0.0}
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": float(total_examples),
    }


def _compute_class_weights(
    *,
    labels: torch.Tensor,
    action_dim: int,
    strategy: str,
    clip: float | None,
    device: torch.device,
) -> torch.Tensor | None:
    if strategy == "none":
        return None
    counts = torch.bincount(labels, minlength=action_dim).float()
    counts = torch.clamp(counts, min=1.0)
    if strategy == "balanced":
        weights = counts.sum() / (len(counts) * counts)
    elif strategy == "inverse_frequency":
        weights = 1.0 / counts
        weights = weights / torch.mean(weights)
    else:  # pragma: no cover - schema validation should prevent this.
        raise ValueError(f"unsupported class weighting strategy: {strategy}")
    if clip is not None:
        weights = torch.clamp(weights, max=clip)
    weights = weights / torch.mean(weights)
    return weights.to(device)


def _is_improvement(
    *,
    current: float,
    best: float,
    mode: str,
    min_delta: float,
) -> bool:
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("BC config requested CUDA but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _append_metrics_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_dumps(dict(row)))
        handle.write("\n")


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_bc_config(source: BehaviorCloningConfig | Path) -> BehaviorCloningConfig:
    if isinstance(source, BehaviorCloningConfig):
        return source
    model = load_phase2_config(source)
    if not isinstance(model, BehaviorCloningConfig):
        raise TypeError(f"{source} is not a behavior cloning config")
    return model


def _load_model_config(source: ModelArchitectureConfig | Path) -> ModelArchitectureConfig:
    if isinstance(source, ModelArchitectureConfig):
        return source
    model = load_phase2_config(source)
    if not isinstance(model, ModelArchitectureConfig):
        raise TypeError(f"{source} is not a model architecture config")
    return model


def _safe_filename(value: str) -> str:
    return value.replace(":", "--").replace("/", "-")
