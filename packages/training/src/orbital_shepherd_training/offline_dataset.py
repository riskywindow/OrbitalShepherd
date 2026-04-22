from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from orbital_shepherd_benchmark.planners import PlannerEpisodeContext, build_planner
from orbital_shepherd_contracts import ReplayEvent, ScenarioBundle, load_json
from orbital_shepherd_core import (
    canonical_json_dumps,
    sha256_fingerprint,
    stable_id,
    stable_token,
)
from orbital_shepherd_env_runtime import EnvRuntimeConfig, replay_events_to_ndjson
from orbital_shepherd_training.config_io import load_yaml_document
from orbital_shepherd_training.models import (
    OfflineDatasetActionSemantics,
    OfflineDatasetArtifact,
    OfflineDatasetBuildManifest,
    OfflineDatasetFeatureDescriptor,
    OfflineDatasetFeatureSchema,
    OfflineDatasetManifest,
    OfflineDatasetPlannerSource,
    ScenarioSplitRegistry,
    TrainingModel,
    TrainingPackManifest,
)
from orbital_shepherd_training.paths import training_config_root
from orbital_shepherd_training.registry import validate_phase2_configs
from orbital_shepherd_training.training_env import (
    CANDIDATE_FEATURE_SPECS,
    GLOBAL_FEATURE_SPECS,
    OrbitalTrainingEnv,
)

try:  # pragma: no cover - exercised where numpy is installed.
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - covered in local tests.
    np = None

try:  # pragma: no cover - exercised when pyarrow is installed.
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - covered in local tests.
    pa = None
    pq = None


OFFLINE_DATASET_SCHEMA_VERSION = "phase2-offline-dataset-v1"
DEFAULT_EXPERT_PLANNER_IDS: tuple[str, ...] = (
    "urgency_greedy",
    "value_density_greedy",
    "ortools_receding_horizon",
)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC")
    return value.astimezone(UTC)


class OfflineEpisodeRecord(TrainingModel):
    dataset_schema_version: str
    dataset_id: str
    split: str
    episode_id: str
    bundle_id: str
    manifest_id: str
    scenario_family: str
    simulation_seed: int
    planner_id: str
    planner_version: str
    planner_description: str
    planner_seed: int
    replay_path: str
    replay_fingerprint: str
    episode_fingerprint: str
    action_count: int = Field(ge=0)
    total_reward: float
    terminated: bool
    truncated: bool


class OfflineTransitionRecord(TrainingModel):
    dataset_schema_version: str
    dataset_id: str
    split: str
    transition_id: str
    episode_id: str
    episode_index: int = Field(ge=0)
    step_index: int = Field(ge=0)
    next_step_index: int | None = Field(default=None, ge=0)
    next_transition_id: str | None = None
    bundle_id: str
    manifest_id: str
    scenario_family: str
    simulation_seed: int
    planner_id: str
    planner_version: str
    planner_description: str
    sim_tick: int = Field(ge=0)
    sim_time_utc: datetime
    observation: dict[str, Any]
    canonical_action_mask_id: str
    legal_action_count: int = Field(ge=1)
    canonical_actions: list[dict[str, Any]] = Field(min_length=1)
    global_features: list[float]
    candidate_features: list[list[float]]
    training_action_mask: list[int]
    projected_candidate_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    truncated_candidate_count: int = Field(ge=0)
    projected_slot_mapping: list[dict[str, Any]]
    planner_trace: dict[str, Any]
    selected_slot: int = Field(ge=0)
    selected_action_id: str
    selected_action_type: str
    selected_action_ref: str
    selected_runtime_action_index: int = Field(ge=0)
    reward: float
    reward_components: dict[str, float] = Field(default_factory=dict)
    terminated: bool
    truncated: bool

    _validate_sim_time_utc = field_validator("sim_time_utc")(_ensure_utc)


@dataclass(frozen=True, slots=True)
class SplitBuildResult:
    manifest: OfflineDatasetManifest
    manifest_path: Path


def build_offline_datasets(
    *,
    training_pack: TrainingPackManifest | Path,
    split_registry: ScenarioSplitRegistry | Path,
    output_root: Path,
    manifest_root: Path,
    planner_ids: Sequence[str] = DEFAULT_EXPERT_PLANNER_IDS,
    splits: Sequence[str] = ("train", "val", "test"),
    top_k: int = 64,
    algorithm: str = "behavior_cloning",
    reward_id: str | None = None,
    limit_bundles_per_split: int | None = None,
    build_id: str | None = None,
    require_parquet: bool = False,
) -> OfflineDatasetBuildManifest:
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")
    training_pack_model = _load_training_pack(training_pack)
    split_registry_model = _load_split_registry(split_registry)
    selected_planners = tuple(dict.fromkeys(planner_ids))
    selected_splits = tuple(dict.fromkeys(str(split) for split in splits))
    if not selected_planners:
        raise ValueError("at least one planner id is required")
    if not selected_splits:
        raise ValueError("at least one split is required")
    _validate_training_pack_against_split_registry(
        training_pack=training_pack_model,
        split_registry=split_registry_model,
    )
    resolved_reward_id = reward_id or _default_reward_id()
    build_spec = {
        "training_pack_id": training_pack_model.training_pack_id,
        "split_registry_id": split_registry_model.registry_id,
        "planner_ids": list(selected_planners),
        "splits": list(selected_splits),
        "top_k": top_k,
        "algorithm": algorithm,
        "limit_bundles_per_split": limit_bundles_per_split,
        "require_parquet": require_parquet,
    }
    build_fingerprint = sha256_fingerprint(build_spec)
    resolved_build_id = build_id or stable_id(
        "offbuild",
        training_pack_model.training_pack_id,
        stable_token(build_fingerprint, length=12),
    )
    build_output_dir = output_root / _safe_filename(resolved_build_id)
    build_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    dataset_manifest_paths: list[str] = []
    split_results: list[SplitBuildResult] = []
    for split_name in selected_splits:
        split_result = _build_split_dataset(
            training_pack=training_pack_model,
            split_registry=split_registry_model,
            output_dir=build_output_dir / split_name,
            split_name=split_name,
            planner_ids=selected_planners,
            top_k=top_k,
            algorithm=algorithm,
            reward_id=resolved_reward_id,
            limit_bundles=limit_bundles_per_split,
            require_parquet=require_parquet,
        )
        split_results.append(split_result)
        dataset_manifest_paths.append(str(split_result.manifest_path))

    created_at_utc = training_pack_model.generated_at_utc
    build_manifest = OfflineDatasetBuildManifest(
        build_id=resolved_build_id,
        benchmark_id=training_pack_model.benchmark_id,
        source_training_pack_id=training_pack_model.training_pack_id,
        split_registry_id=split_registry_model.registry_id,
        created_at_utc=created_at_utc,
        output_dir=str(build_output_dir),
        planner_ids=list(selected_planners),
        splits=list(selected_splits),
        top_k=top_k,
        dataset_manifests=dataset_manifest_paths,
        artifact_fingerprint="sha256:"
        + sha256_fingerprint(
            {
                "build_id": resolved_build_id,
                "dataset_manifests": dataset_manifest_paths,
                "planner_ids": list(selected_planners),
                "splits": list(selected_splits),
                "top_k": top_k,
            }
        ),
    )
    build_manifest_path = manifest_root / f"{_safe_filename(resolved_build_id)}.json"
    build_manifest_path.write_text(
        canonical_json_dumps(build_manifest.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    return build_manifest


def inspect_offline_dataset(source: Path, *, preview_steps: int = 3) -> str:
    if source.is_dir():
        candidate = source / "manifest.json"
        if candidate.exists():
            source = candidate
    document = _load_json_or_yaml(source)
    config_type = str(document.get("config_type", ""))
    if config_type == "offline_dataset_build_manifest":
        manifest = OfflineDatasetBuildManifest.model_validate(document)
        return _inspect_build_manifest(manifest)
    if config_type == "offline_dataset_manifest":
        manifest = OfflineDatasetManifest.model_validate(document)
        return _inspect_dataset_manifest(manifest, preview_steps=preview_steps)
    raise ValueError(f"unrecognized dataset document: {source}")


def _build_split_dataset(
    *,
    training_pack: TrainingPackManifest,
    split_registry: ScenarioSplitRegistry,
    output_dir: Path,
    split_name: str,
    planner_ids: Sequence[str],
    top_k: int,
    algorithm: str,
    reward_id: str,
    limit_bundles: int | None,
    require_parquet: bool,
) -> SplitBuildResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = output_dir / "canonical"
    adapter_dir = output_dir / "adapters"
    replay_dir = output_dir / "replays"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)

    split_entry_by_bundle = {
        entry.bundle_id: entry for entry in split_registry.entries if entry.split == split_name
    }
    pack_entries = [
        entry
        for entry in sorted(training_pack.entries, key=lambda item: item.bundle_id)
        if entry.bundle_id in split_entry_by_bundle
    ]
    if limit_bundles is not None:
        pack_entries = pack_entries[:limit_bundles]
    if not pack_entries:
        raise ValueError(f"no training pack entries matched split {split_name!r}")

    dataset_spec = {
        "training_pack_id": training_pack.training_pack_id,
        "split_registry_id": split_registry.registry_id,
        "split": split_name,
        "planner_ids": list(planner_ids),
        "bundle_ids": [entry.bundle_id for entry in pack_entries],
        "top_k": top_k,
        "algorithm": algorithm,
    }
    dataset_id = stable_id(
        "offds",
        training_pack.benchmark_id,
        split_name,
        stable_token(dataset_spec, length=12),
    )

    episodes: list[OfflineEpisodeRecord] = []
    transitions: list[OfflineTransitionRecord] = []
    rllib_rows: list[dict[str, Any]] = []
    planner_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "records": 0})
    max_legal_action_count = 0
    max_projected_candidate_count = 0

    for planner_id in sorted(planner_ids):
        planner_metadata = build_planner(planner_id).metadata
        for episode_index, pack_entry in enumerate(pack_entries):
            planner = build_planner(planner_id)
            bundle = ScenarioBundle.model_validate(load_json(Path(pack_entry.scenario_path)))
            training_env = OrbitalTrainingEnv(
                bundle,
                config=EnvRuntimeConfig(
                    planner_id=f"planner:{planner_metadata.planner_id}",
                    decision_interval_seconds=bundle.decision_interval_seconds,
                ),
                top_k=top_k,
            )
            raw_observation, raw_info = training_env.runtime_env.reset(
                seed=bundle.simulation_seed,
                planner_id=f"planner:{planner_metadata.planner_id}",
            )
            projected_observation, projected_info = training_env._project_observation(
                raw_observation,
                raw_info,
            )
            planner_seed = _planner_seed(bundle=bundle, planner_id=planner_metadata.planner_id)
            planner.start_episode(
                context=PlannerEpisodeContext(
                    bundle=bundle,
                    episode_id=str(raw_observation["episode_id"]),
                    episode_seed=int(raw_observation["episode_seed"]),
                    planner_seed=planner_seed,
                ),
                initial_observation=raw_observation,
            )

            step_index = 0
            total_reward = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                decision = planner.select_action(raw_observation)
                selected_slot, selected_mapping = _selected_slot_mapping(
                    slot_mapping=projected_info["slot_mapping"],
                    action_id=decision.action.action_id,
                )
                if selected_mapping is None:
                    raise ValueError(
                        "planner action is outside the projected training slots; "
                        f"increase top_k for {planner_metadata.planner_id} on {bundle.bundle_id}"
                    )
                selected_runtime_action_index = int(
                    selected_mapping.get("runtime_action_index")
                    if selected_mapping.get("runtime_action_index") is not None
                    else 0
                )
                transition_id = stable_id(
                    "transition",
                    raw_observation["episode_id"],
                    f"step-{step_index:05d}",
                )
                next_transition_id = stable_id(
                    "transition",
                    raw_observation["episode_id"],
                    f"step-{step_index + 1:05d}",
                )
                next_raw_observation, reward, terminated, truncated, next_raw_info = (
                    training_env.runtime_env.step(
                        decision.action,
                        planner_trace=decision.to_trace_payload(),
                    )
                )
                next_projected_observation, next_projected_info = training_env._project_observation(
                    next_raw_observation,
                    next_raw_info,
                )
                total_reward = round(total_reward + reward, 6)
                record = OfflineTransitionRecord(
                    dataset_schema_version=OFFLINE_DATASET_SCHEMA_VERSION,
                    dataset_id=dataset_id,
                    split=split_name,
                    transition_id=transition_id,
                    episode_id=str(raw_observation["episode_id"]),
                    episode_index=episode_index,
                    step_index=step_index,
                    next_step_index=None if (terminated or truncated) else step_index + 1,
                    next_transition_id=None if (terminated or truncated) else next_transition_id,
                    bundle_id=bundle.bundle_id,
                    manifest_id=bundle.manifest_id,
                    scenario_family=bundle.scenario_family,
                    simulation_seed=bundle.simulation_seed,
                    planner_id=planner_metadata.planner_id,
                    planner_version=planner_metadata.version,
                    planner_description=planner_metadata.description,
                    sim_tick=int(raw_observation["sim_tick"]),
                    sim_time_utc=raw_observation["sim_time_utc"],
                    observation=_normalize_jsonable(raw_observation),
                    canonical_action_mask_id=str(raw_observation["action_mask"]["mask_id"]),
                    legal_action_count=int(raw_observation["action_mask"]["legal_action_count"]),
                    canonical_actions=_normalize_jsonable(raw_observation["action_mask"]["actions"]),
                    global_features=_float_list(projected_observation["global_features"]),
                    candidate_features=_float_matrix(projected_observation["candidate_features"]),
                    training_action_mask=_int_list(projected_observation["action_mask"]),
                    projected_candidate_count=int(projected_info["projected_candidate_count"]),
                    candidate_count=int(projected_info["candidate_count"]),
                    truncated_candidate_count=int(projected_info["truncated_candidate_count"]),
                    projected_slot_mapping=_normalize_jsonable(projected_info["slot_mapping"]),
                    planner_trace=_normalize_jsonable(decision.to_trace_payload()),
                    selected_slot=selected_slot,
                    selected_action_id=decision.action.action_id,
                    selected_action_type=decision.action.action_type,
                    selected_action_ref=decision.action.ref,
                    selected_runtime_action_index=selected_runtime_action_index,
                    reward=reward,
                    reward_components=_normalize_reward_components(
                        next_raw_info.get("reward_components", {})
                    ),
                    terminated=terminated,
                    truncated=truncated,
                )
                transitions.append(record)
                rllib_rows.append(
                    {
                        "episode_id": str(raw_observation["episode_id"]),
                        "t": step_index,
                        "obs": {
                            "global_features": record.global_features,
                            "candidate_features": record.candidate_features,
                            "action_mask": record.training_action_mask,
                        },
                        "next_obs": {
                            "global_features": _float_list(
                                next_projected_observation["global_features"]
                            ),
                            "candidate_features": _float_matrix(
                                next_projected_observation["candidate_features"]
                            ),
                            "action_mask": _int_list(next_projected_observation["action_mask"]),
                        },
                        "actions": selected_slot,
                        "rewards": reward,
                        "terminateds": terminated,
                        "truncateds": truncated,
                        "infos": {
                            "planner_id": planner_metadata.planner_id,
                            "selected_action_id": decision.action.action_id,
                            "selected_action_type": decision.action.action_type,
                            "selected_action_ref": decision.action.ref,
                            "selected_runtime_action_index": selected_runtime_action_index,
                            "slot_mapping": _normalize_jsonable(projected_info["slot_mapping"]),
                            "reward_components": _normalize_reward_components(
                                next_raw_info.get("reward_components", {})
                            ),
                        },
                    }
                )
                max_legal_action_count = max(max_legal_action_count, record.legal_action_count)
                max_projected_candidate_count = max(
                    max_projected_candidate_count,
                    record.projected_candidate_count,
                )
                planner_counts[planner_metadata.planner_id]["records"] += 1
                raw_observation = next_raw_observation
                raw_info = next_raw_info
                projected_observation = next_projected_observation
                projected_info = next_projected_info
                step_index += 1

            replay_path = replay_dir / _safe_filename(planner_metadata.planner_id) / (
                f"{_safe_filename(bundle.bundle_id)}.ndjson"
            )
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            replay_ndjson = replay_events_to_ndjson(training_env.runtime_env.replay_events)
            replay_path.write_text(f"{replay_ndjson}\n", encoding="utf-8")
            replay_fingerprint = f"sha256:{sha256_fingerprint(replay_ndjson)}"
            episodes.append(
                OfflineEpisodeRecord(
                    dataset_schema_version=OFFLINE_DATASET_SCHEMA_VERSION,
                    dataset_id=dataset_id,
                    split=split_name,
                    episode_id=raw_observation["episode_id"],
                    bundle_id=bundle.bundle_id,
                    manifest_id=bundle.manifest_id,
                    scenario_family=bundle.scenario_family,
                    simulation_seed=bundle.simulation_seed,
                    planner_id=planner_metadata.planner_id,
                    planner_version=planner_metadata.version,
                    planner_description=planner_metadata.description,
                    planner_seed=planner_seed,
                    replay_path=str(replay_path),
                    replay_fingerprint=replay_fingerprint,
                    episode_fingerprint=_episode_fingerprint(training_env.runtime_env.replay_events),
                    action_count=step_index,
                    total_reward=total_reward,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
            planner_counts[planner_metadata.planner_id]["episodes"] += 1
            training_env.close()

    episodes_path = canonical_dir / "episodes.jsonl"
    steps_path = canonical_dir / "steps.jsonl"
    _write_jsonl(episodes_path, [episode.model_dump(mode="json") for episode in episodes])
    _write_jsonl(steps_path, [record.model_dump(mode="json") for record in transitions])

    artifacts = [
        _artifact_record("canonical_episodes", "jsonl", episodes_path, len(episodes)),
        _artifact_record("canonical_steps", "jsonl", steps_path, len(transitions)),
    ]

    parquet_path = canonical_dir / "steps.parquet"
    if pa is not None and pq is not None:
        table = pa.Table.from_pylist([record.model_dump(mode="json") for record in transitions])
        pq.write_table(table, parquet_path)
        artifacts.append(
            _artifact_record("canonical_steps", "parquet", parquet_path, len(transitions))
        )
    elif require_parquet:
        raise RuntimeError("pyarrow is required for parquet output but is not installed")

    rllib_path = adapter_dir / "rllib_transitions.jsonl"
    _write_jsonl(rllib_path, rllib_rows)
    artifacts.append(_artifact_record("rllib_transitions", "jsonl", rllib_path, len(rllib_rows)))

    if np is not None:
        arrays_path = adapter_dir / "training_arrays.npz"
        global_features = np.asarray(
            [record.global_features for record in transitions],
            dtype="float32",
        )
        candidate_features = np.asarray(
            [record.candidate_features for record in transitions],
            dtype="float32",
        )
        action_mask = np.asarray(
            [record.training_action_mask for record in transitions],
            dtype="int8",
        )
        selected_slot = np.asarray(
            [record.selected_slot for record in transitions],
            dtype="int64",
        )
        reward = np.asarray([record.reward for record in transitions], dtype="float32")
        terminated_array = np.asarray(
            [record.terminated for record in transitions],
            dtype="bool",
        )
        truncated_array = np.asarray(
            [record.truncated for record in transitions],
            dtype="bool",
        )
        np.savez_compressed(
            arrays_path,
            global_features=global_features,
            candidate_features=candidate_features,
            action_mask=action_mask,
            selected_slot=selected_slot,
            reward=reward,
            terminated=terminated_array,
            truncated=truncated_array,
        )
        artifacts.append(_artifact_record("training_arrays", "npz", arrays_path, len(transitions)))

    feature_schema = OfflineDatasetFeatureSchema(
        top_k=top_k,
        global_features=[
            OfflineDatasetFeatureDescriptor(
                name=item.name,
                lower_bound=item.lower_bound,
                upper_bound=item.upper_bound,
                description=item.description,
            )
            for item in GLOBAL_FEATURE_SPECS
        ],
        candidate_features=[
            OfflineDatasetFeatureDescriptor(
                name=item.name,
                lower_bound=item.lower_bound,
                upper_bound=item.upper_bound,
                description=item.description,
            )
            for item in CANDIDATE_FEATURE_SPECS
        ],
        action_mask_shape=[top_k + 1],
        action_mask_dtype="int8",
        slot_mapping_fields=[
            "slot_index",
            "action_id",
            "action_type",
            "action_ref",
            "source",
            "projected_rank",
            "runtime_action_index",
        ],
    )
    action_semantics = OfflineDatasetActionSemantics(
        noop_slot_index=0,
        projected_slots_start_index=1,
        projected_slots_end_index=top_k,
        canonical_action_mask_field="canonical_actions",
        projected_action_mask_field="training_action_mask",
        selected_slot_field="selected_slot",
        selected_runtime_action_index_field="selected_runtime_action_index",
        selected_action_id_field="selected_action_id",
    )
    planner_sources = [
        OfflineDatasetPlannerSource(
            planner_id=planner_id,
            planner_version=build_planner(planner_id).metadata.version,
            description=build_planner(planner_id).metadata.description,
            episode_count=planner_counts[planner_id]["episodes"],
            record_count=planner_counts[planner_id]["records"],
        )
        for planner_id in sorted(planner_ids)
    ]
    manifest = OfflineDatasetManifest(
        dataset_id=dataset_id,
        benchmark_id=training_pack.benchmark_id,
        source_training_pack_id=training_pack.training_pack_id,
        split_registry_id=split_registry.registry_id,
        split=split_name,
        algorithm=algorithm,
        created_at_utc=training_pack.generated_at_utc,
        dataset_schema_version=OFFLINE_DATASET_SCHEMA_VERSION,
        dataset_path=str(output_dir),
        record_count=len(transitions),
        episode_count=len(episodes),
        top_k=top_k,
        max_legal_action_count=max_legal_action_count,
        max_projected_candidate_count=max_projected_candidate_count,
        source_bundle_ids=[entry.bundle_id for entry in pack_entries],
        source_manifest_ids=[entry.manifest_id for entry in pack_entries],
        source_planners=planner_sources,
        scenario_families=sorted({entry.scenario_family for entry in pack_entries}),
        reward_id=reward_id,
        feature_schema=feature_schema,
        action_semantics=action_semantics,
        artifacts=artifacts,
        artifact_fingerprint="sha256:"
        + sha256_fingerprint(
            {
                "dataset_id": dataset_id,
                "record_count": len(transitions),
                "episode_count": len(episodes),
                "source_bundle_ids": [entry.bundle_id for entry in pack_entries],
                "planners": [item.model_dump(mode="json") for item in planner_sources],
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
                "top_k": top_k,
                "split": split_name,
            }
        ),
    )
    card_path = output_dir / "dataset_card.md"
    card_path.write_text(_dataset_card(markdown_manifest=manifest), encoding="utf-8")
    artifacts.append(_artifact_record("dataset_card", "markdown", card_path, None))
    manifest = manifest.model_copy(
        update={
            "artifacts": artifacts,
            "artifact_fingerprint": "sha256:"
            + sha256_fingerprint(
                {
                    "dataset_id": dataset_id,
                    "record_count": len(transitions),
                    "episode_count": len(episodes),
                    "source_bundle_ids": [entry.bundle_id for entry in pack_entries],
                    "planners": [item.model_dump(mode="json") for item in planner_sources],
                    "artifacts": [item.model_dump(mode="json") for item in artifacts],
                    "top_k": top_k,
                    "split": split_name,
                }
            ),
        }
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        canonical_json_dumps(manifest.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    return SplitBuildResult(manifest=manifest, manifest_path=manifest_path)


def _artifact_record(
    artifact_role: str,
    format_name: str,
    path: Path,
    record_count: int | None,
) -> OfflineDatasetArtifact:
    return OfflineDatasetArtifact(
        artifact_role=artifact_role,
        format=format_name,
        path=str(path),
        record_count=record_count,
        artifact_fingerprint=f"sha256:{sha256_fingerprint(path.read_bytes())}",
    )


def _dataset_card(*, markdown_manifest: OfflineDatasetManifest) -> str:
    planners = ", ".join(
        f"{item.planner_id} ({item.planner_version})" for item in markdown_manifest.source_planners
    )
    artifacts = "\n".join(
        f"- `{item.artifact_role}` `{item.format}` `{item.path}`"
        for item in markdown_manifest.artifacts
    )
    feature_names = ", ".join(
        item.name for item in markdown_manifest.feature_schema.global_features
    )
    candidate_feature_names = ", ".join(
        item.name for item in markdown_manifest.feature_schema.candidate_features
    )
    return (
        f"# Dataset Card: {markdown_manifest.dataset_id}\n\n"
        f"## Summary\n\n"
        f"- Split: `{markdown_manifest.split}`\n"
        f"- Benchmark: `{markdown_manifest.benchmark_id}`\n"
        f"- Planners: {planners}\n"
        f"- Episodes: {markdown_manifest.episode_count}\n"
        f"- Transitions: {markdown_manifest.record_count}\n"
        f"- Source families: {', '.join(markdown_manifest.scenario_families)}\n\n"
        f"## Source Planners\n\n"
        "This dataset is compiled from actual rollouts of deterministic expert planners "
        "against the Phase 1 runtime. Each row keeps the planner identity and the "
        "canonical selected action.\n\n"
        f"## Split Definition\n\n"
        f"The dataset uses the committed split registry `{markdown_manifest.split_registry_id}`. "
        f"Bundles in this dataset appear only in the `{markdown_manifest.split}` split.\n\n"
        f"## Feature Schema\n\n"
        f"- Global features: {feature_names}\n"
        f"- Candidate features: {candidate_feature_names}\n"
        f"- Top-K projected slots: `{markdown_manifest.top_k}` plus slot `0` for `noop`\n\n"
        f"## Action Semantics\n\n"
        f"- `selected_slot` points to the projected training slot.\n"
        f"- `selected_runtime_action_index` maps back to `canonical_actions`.\n"
        f"- `selected_action_id` is the stable canonical action identifier.\n\n"
        f"## Artifacts\n\n"
        f"{artifacts}\n\n"
        f"## Known Limitations\n\n"
        f"- Rewards are environment-emitted and replay-auditable, but not benchmark metrics.\n"
        "- The projected action view is limited to `top_k`; builds fail if an expert choice "
        "falls outside that view.\n"
        "- RLlib export is an adapter artifact. The canonical audit record remains the step "
        "dataset.\n"
    )


def _inspect_build_manifest(manifest: OfflineDatasetBuildManifest) -> str:
    lines = [
        f"build_id\t{manifest.build_id}",
        f"benchmark_id\t{manifest.benchmark_id}",
        f"source_training_pack_id\t{manifest.source_training_pack_id}",
        f"splits\t{','.join(manifest.splits)}",
        f"planner_ids\t{','.join(manifest.planner_ids)}",
        f"top_k\t{manifest.top_k}",
        f"output_dir\t{manifest.output_dir}",
        f"artifact_fingerprint\t{manifest.artifact_fingerprint}",
    ]
    for dataset_manifest in manifest.dataset_manifests:
        lines.append(f"dataset_manifest\t{dataset_manifest}")
    return "\n".join(lines) + "\n"


def _inspect_dataset_manifest(manifest: OfflineDatasetManifest, *, preview_steps: int) -> str:
    lines = [
        f"dataset_id\t{manifest.dataset_id}",
        f"split\t{manifest.split}",
        f"record_count\t{manifest.record_count}",
        f"episode_count\t{manifest.episode_count}",
        f"benchmark_id\t{manifest.benchmark_id}",
        f"source_training_pack_id\t{manifest.source_training_pack_id}",
        f"source_planners\t{','.join(item.planner_id for item in manifest.source_planners)}",
        f"scenario_families\t{','.join(manifest.scenario_families)}",
        f"top_k\t{manifest.top_k}",
        f"max_legal_action_count\t{manifest.max_legal_action_count}",
        f"max_projected_candidate_count\t{manifest.max_projected_candidate_count}",
        f"artifact_fingerprint\t{manifest.artifact_fingerprint}",
    ]
    for artifact in manifest.artifacts:
        lines.append(
            "artifact\t"
            f"{artifact.artifact_role}:{artifact.format}:{artifact.record_count}:{artifact.path}"
        )
    steps_path = next(
        (
            Path(artifact.path)
            for artifact in manifest.artifacts
            if artifact.path.endswith("steps.jsonl")
        ),
        None,
    )
    if steps_path is not None and steps_path.exists() and preview_steps > 0:
        for row in _read_jsonl(steps_path)[:preview_steps]:
            lines.append(
                "preview\t"
                f"step={row['step_index']} slot={row['selected_slot']} "
                f"action={row['selected_action_type']}:{row['selected_action_ref']} "
                f"reward={row['reward']:.6f}"
            )
    return "\n".join(lines) + "\n"


def _load_training_pack(source: TrainingPackManifest | Path) -> TrainingPackManifest:
    if isinstance(source, TrainingPackManifest):
        return source
    return TrainingPackManifest.model_validate(_load_json_or_yaml(source))


def _load_split_registry(source: ScenarioSplitRegistry | Path) -> ScenarioSplitRegistry:
    if isinstance(source, ScenarioSplitRegistry):
        return source
    return ScenarioSplitRegistry.model_validate(_load_json_or_yaml(source))


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        return dict(load_yaml_document(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json_dumps(row))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _validate_training_pack_against_split_registry(
    *,
    training_pack: TrainingPackManifest,
    split_registry: ScenarioSplitRegistry,
) -> None:
    split_by_bundle = {entry.bundle_id: entry.split for entry in split_registry.entries}
    seen_bundle_ids: set[str] = set()
    for entry in training_pack.entries:
        if entry.bundle_id in seen_bundle_ids:
            raise ValueError(f"duplicate bundle_id in training pack: {entry.bundle_id}")
        seen_bundle_ids.add(entry.bundle_id)
        expected_split = split_by_bundle.get(entry.bundle_id)
        if expected_split is None:
            raise ValueError(f"bundle missing from split registry: {entry.bundle_id}")
        if entry.split != expected_split:
            raise ValueError(
                f"training pack split mismatch for {entry.bundle_id}: "
                f"{entry.split!r} != {expected_split!r}"
            )


def _selected_slot_mapping(
    *,
    slot_mapping: Sequence[Mapping[str, Any]],
    action_id: str,
) -> tuple[int, Mapping[str, Any] | None]:
    for item in slot_mapping:
        if str(item.get("action_id")) == action_id:
            return int(item["slot_index"]), item
    return -1, None


def _planner_seed(*, bundle: ScenarioBundle, planner_id: str) -> int:
    token = stable_token(f"{planner_id}:{bundle.bundle_id}:{bundle.simulation_seed}", length=16)
    return int(token, 16)


def _episode_fingerprint(events: Sequence[ReplayEvent]) -> str:
    for event in events:
        if event.event_type == "episode_started":
            return str(event.payload["episode_fingerprint"])
    raise ValueError("episode_started event missing from replay")


def _normalize_jsonable(value: Any) -> Any:
    return json.loads(canonical_json_dumps(value))


def _normalize_reward_components(value: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): round(float(item), 6) for key, item in sorted(value.items())}


def _float_list(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        return [round(float(item), 6) for item in values.tolist()]
    return [round(float(item), 6) for item in values]


def _float_matrix(values: Any) -> list[list[float]]:
    if hasattr(values, "tolist"):
        rows = values.tolist()
    else:
        rows = values
    return [[round(float(item), 6) for item in row] for row in rows]


def _int_list(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        return [int(item) for item in values.tolist()]
    return [int(item) for item in values]


def _default_reward_id() -> str:
    validated = validate_phase2_configs(training_config_root())
    return str(validated["reward"].reward_id)


def _safe_filename(value: str) -> str:
    return value.replace(":", "--").replace("/", "--")
