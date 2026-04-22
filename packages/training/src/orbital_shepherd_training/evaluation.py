from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from math import ceil
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Literal

import torch

from orbital_shepherd_benchmark.metrics import EpisodeMetrics, compute_episode_metrics
from orbital_shepherd_benchmark.planners import (
    PlannerDecision,
    PlannerEpisodeContext,
    build_planner,
)
from orbital_shepherd_contracts import ReplayEvent, ScenarioBundle, load_json
from orbital_shepherd_core import (
    canonical_json_dumps,
    format_utc_timestamp,
    sha256_fingerprint,
    stable_id,
    stable_token,
)
from orbital_shepherd_env_runtime import (
    EnvRuntimeConfig,
    OrbitalAction,
    OrbitalEnv,
    replay_events_to_ndjson,
)
from orbital_shepherd_policy_models import PolicyModelRegistry
from orbital_shepherd_training.config_io import load_phase2_config, load_yaml_document
from orbital_shepherd_training.models import (
    EvaluationConfig,
    EvaluationReportManifest,
    EvaluationRunManifest,
    EvaluationRunPlanner,
    PolicyCheckpointManifest,
    ScenarioSplitEntry,
    ScenarioSplitRegistry,
    TrainingPackEntry,
    TrainingPackManifest,
)
from orbital_shepherd_training.rllib_module import OrbitalMaskedActionTorchRLModule
from orbital_shepherd_training.tracking import maybe_init_wandb
from orbital_shepherd_training.training_env import OrbitalTrainingEnv

PlannerKind = Literal["builtin", "policy_checkpoint"]
MetricDirection = Literal["higher", "lower"]

METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "time_to_first_useful_observation_seconds": "lower",
    "useful_observation_value_captured": "higher",
    "cloud_waste_rate": "lower",
    "downlink_latency_seconds": "lower",
    "missed_urgent_incident_rate": "lower",
    "opportunity_utilization_efficiency": "higher",
    "mission_utility": "higher",
}
PAIRWISE_TIE_EPSILON = 1.0e-9


@dataclass(frozen=True, slots=True)
class EvaluatedPlannerSpec:
    planner_key: str
    planner_kind: PlannerKind
    display_name: str
    evaluated_artifact_id: str
    version: str
    description: str
    source: str
    checkpoint_manifest_path: Path | None = None
    checkpoint_manifest: PolicyCheckpointManifest | None = None
    module_path: Path | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class EvaluationScenarioEntry:
    training_entry: TrainingPackEntry
    split_entry: ScenarioSplitEntry

    @property
    def bundle_id(self) -> str:
        return self.training_entry.bundle_id

    @property
    def scenario_path(self) -> Path:
        return Path(self.training_entry.scenario_path)

    @property
    def split(self) -> str:
        return self.split_entry.split


@dataclass(frozen=True, slots=True)
class EvaluationEpisodeArtifacts:
    replay_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class EvaluationEpisodeResult:
    planner_key: str
    planner_kind: PlannerKind
    display_name: str
    evaluated_artifact_id: str
    planner_version: str
    bundle_id: str
    manifest_id: str
    split: str
    scenario_family: str
    scenario_path: Path
    ood_axes: tuple[str, ...]
    difficulty_tier: int
    episode_id: str
    episode_seed: int
    episode_fingerprint: str
    replay_fingerprint: str
    action_count: int
    metrics: EpisodeMetrics
    reward_audit: dict[str, float]
    bundle_profile: dict[str, float | int | str | list[str]]
    artifacts: EvaluationEpisodeArtifacts

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_key": self.planner_key,
            "planner_kind": self.planner_kind,
            "display_name": self.display_name,
            "evaluated_artifact_id": self.evaluated_artifact_id,
            "planner_version": self.planner_version,
            "bundle_id": self.bundle_id,
            "manifest_id": self.manifest_id,
            "split": self.split,
            "scenario_family": self.scenario_family,
            "scenario_path": str(self.scenario_path),
            "ood_axes": list(self.ood_axes),
            "difficulty_tier": self.difficulty_tier,
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "episode_fingerprint": self.episode_fingerprint,
            "replay_fingerprint": self.replay_fingerprint,
            "action_count": self.action_count,
            "metrics": self.metrics.to_dict(),
            "reward_audit": dict(sorted(self.reward_audit.items())),
            "bundle_profile": _json_compatible(self.bundle_profile),
            "replay_path": str(self.artifacts.replay_path),
            "summary_path": str(self.artifacts.summary_path),
        }


@dataclass(frozen=True, slots=True)
class Phase2EvaluationSummary:
    run_id: str
    report_dir: Path
    manifest_dir: Path
    summary_path: Path
    markdown_path: Path | None
    episode_metrics_path: Path | None
    pairwise_comparisons_path: Path | None
    episode_outcomes_path: Path | None
    notable_episodes_path: Path | None
    run_manifest_path: Path
    report_manifest_paths: tuple[Path, ...]
    episode_count: int
    planner_keys: tuple[str, ...]
    selected_splits: tuple[str, ...]
    selected_bundle_ids: tuple[str, ...]


class _BuiltinPlannerRunner:
    def __init__(self, spec: EvaluatedPlannerSpec) -> None:
        self.spec = spec
        self._planner = build_planner(spec.source)

    def start_episode(self, *, bundle: ScenarioBundle, observation: Mapping[str, Any]) -> None:
        self._planner.start_episode(
            context=PlannerEpisodeContext(
                bundle=bundle,
                episode_id=str(observation["episode_id"]),
                episode_seed=int(observation["episode_seed"]),
                planner_seed=_planner_seed(bundle=bundle, planner_key=self.spec.planner_key),
            ),
            initial_observation=observation,
        )

    def select_action(self, observation: Mapping[str, Any]) -> tuple[OrbitalAction, dict[str, Any]]:
        decision = self._planner.select_action(observation)
        return decision.action, _decision_trace_payload(decision)


class _PolicyModulePlannerRunner:
    def __init__(self, spec: EvaluatedPlannerSpec) -> None:
        self.spec = spec
        if spec.checkpoint_manifest_path is None:
            raise ValueError(
                f"policy planner {spec.planner_key} is missing a checkpoint manifest path"
            )
        registry = PolicyModelRegistry(manifest_roots=(), checkpoint_roots=())
        registry.register_checkpoint(
            manifest_path=spec.checkpoint_manifest_path,
            model_key=spec.planner_key,
        )
        loaded_policy = registry.load_policy(spec.planner_key)
        self._policy_model = loaded_policy.policy_model
        self._policy_model.eval()

    def start_episode(self) -> None:
        return None

    def select_action(
        self,
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        with torch.no_grad():
            outputs = self._policy_model.forward_observation(
                {
                    "global_features": torch.tensor(
                        observation["global_features"], dtype=torch.float32
                    ).unsqueeze(0),
                    "candidate_features": torch.tensor(
                        observation["candidate_features"], dtype=torch.float32
                    ).unsqueeze(0),
                    "action_mask": torch.tensor(
                        observation["action_mask"], dtype=torch.float32
                    ).unsqueeze(0),
                }
            )
        masked_logits = outputs.masked_logits.squeeze(0).cpu()
        probabilities = torch.softmax(masked_logits, dim=0)
        selected_slot = int(torch.argmax(masked_logits).item())
        slot_mapping = list(info.get("slot_mapping", []))
        top_count = min(5, len(slot_mapping))
        top_slots = torch.topk(masked_logits, k=top_count).indices.tolist()
        return selected_slot, {
            "policy_trace": {
                "selected_slot": selected_slot,
                "selected_action_id": info.get("slot_to_action_id", [None])[selected_slot],
                "selected_slot_mapping": (
                    slot_mapping[selected_slot] if selected_slot < len(slot_mapping) else None
                ),
                "selected_logit": round(float(masked_logits[selected_slot].item()), 6),
                "selected_probability": round(float(probabilities[selected_slot].item()), 6),
                "value_estimate": round(float(outputs.values.squeeze(0).item()), 6),
                "top_slots": [
                    {
                        "slot_index": int(slot_index),
                        "action_id": (
                            info.get("slot_to_action_id", [None])[slot_index]
                            if slot_index < len(info.get("slot_to_action_id", []))
                            else None
                        ),
                        "slot_mapping": (
                            slot_mapping[slot_index] if slot_index < len(slot_mapping) else None
                        ),
                        "logit": round(float(masked_logits[slot_index].item()), 6),
                        "probability": round(float(probabilities[slot_index].item()), 6),
                    }
                    for slot_index in top_slots
                ],
            }
        }


def run_phase2_evaluation(
    *,
    evaluation_config: EvaluationConfig | Path,
    training_pack: TrainingPackManifest | Path,
    split_registry: ScenarioSplitRegistry | Path,
    planner_ids: Sequence[str] | None = None,
    checkpoint_manifests: Sequence[str | Path] = (),
    policy_labels: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    limit_bundles_per_split: int | None = None,
    bundle_ids: Sequence[str] = (),
    run_id: str | None = None,
) -> Phase2EvaluationSummary:
    evaluation = _load_evaluation_config(evaluation_config)
    training_pack_manifest = _load_training_pack_manifest(training_pack)
    split_registry_model = _load_split_registry(split_registry)
    selected_splits = tuple(splits or evaluation.split_order)
    scenario_entries = _select_scenario_entries(
        training_pack=training_pack_manifest,
        split_registry=split_registry_model,
        splits=selected_splits,
        limit_bundles_per_split=limit_bundles_per_split,
        bundle_ids=bundle_ids,
    )
    planner_specs = _resolve_planner_specs(
        planner_ids=planner_ids,
        checkpoint_manifests=checkpoint_manifests,
        policy_labels=policy_labels,
    )
    if not planner_specs:
        raise ValueError("at least one baseline planner or policy checkpoint is required")

    started_at = datetime.now(UTC)
    run_label = _safe_filename(
        run_id
        or stable_id("evalrun", evaluation.evaluation_id, format_utc_timestamp(started_at).lower())
    )
    report_dir = (Path(evaluation.artifact_layout.report_root) / run_label).resolve()
    manifest_dir = (Path(evaluation.artifact_layout.manifest_root) / run_label).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    bundle_cache: dict[Path, ScenarioBundle] = {}
    runner_cache: dict[str, _BuiltinPlannerRunner | _PolicyModulePlannerRunner] = {}
    episode_results: list[EvaluationEpisodeResult] = []
    for spec in planner_specs:
        runner_cache[spec.planner_key] = _build_runner(spec)
        for entry in scenario_entries:
            bundle = bundle_cache.setdefault(entry.scenario_path, _load_bundle(entry.scenario_path))
            episode_results.append(
                _run_episode(
                    spec=spec,
                    runner=runner_cache[spec.planner_key],
                    scenario=entry,
                    bundle=bundle,
                    report_dir=report_dir,
                )
            )

    summary_payload = _build_run_summary(
        evaluation=evaluation,
        training_pack=training_pack_manifest,
        split_registry=split_registry_model,
        planner_specs=planner_specs,
        scenario_entries=scenario_entries,
        episode_results=episode_results,
    )
    summary_payload["run_id"] = run_label
    summary_path = report_dir / "summary.json"
    summary_path.write_text(_pretty_json(summary_payload), encoding="utf-8")

    markdown_path: Path | None = None
    if "markdown" in evaluation.report_formats:
        markdown_path = report_dir / "summary.md"
        markdown_path.write_text(_build_markdown_report(summary_payload), encoding="utf-8")

    csv_outputs = _write_csv_outputs(report_dir=report_dir, summary_payload=summary_payload)
    report_manifest_paths = _write_report_manifests(
        manifest_dir=manifest_dir,
        evaluation=evaluation,
        summary_payload=summary_payload,
        summary_path=summary_path,
    )
    run_manifest_path = _write_run_manifest(
        manifest_dir=manifest_dir,
        evaluation=evaluation,
        training_pack=training_pack_manifest,
        split_registry=split_registry_model,
        planner_specs=planner_specs,
        summary_payload=summary_payload,
        report_dir=report_dir,
        summary_path=summary_path,
        markdown_path=markdown_path,
        report_manifest_paths=report_manifest_paths,
        csv_outputs=csv_outputs,
        run_id=run_label,
        generated_at=started_at,
    )
    _write_dvc_outputs(report_dir=report_dir, summary_payload=summary_payload)
    _log_to_wandb(
        evaluation=evaluation,
        run_name=run_label,
        summary_payload=summary_payload,
        report_dir=report_dir,
    )
    return Phase2EvaluationSummary(
        run_id=run_label,
        report_dir=report_dir,
        manifest_dir=manifest_dir,
        summary_path=summary_path,
        markdown_path=markdown_path,
        episode_metrics_path=csv_outputs.get("episode_metrics"),
        pairwise_comparisons_path=csv_outputs.get("pairwise_comparisons"),
        episode_outcomes_path=csv_outputs.get("episode_outcomes"),
        notable_episodes_path=csv_outputs.get("notable_episodes"),
        run_manifest_path=run_manifest_path,
        report_manifest_paths=tuple(report_manifest_paths),
        episode_count=len(episode_results),
        planner_keys=tuple(spec.planner_key for spec in planner_specs),
        selected_splits=selected_splits,
        selected_bundle_ids=tuple(entry.bundle_id for entry in scenario_entries),
    )


def _load_evaluation_config(value: EvaluationConfig | Path) -> EvaluationConfig:
    if isinstance(value, EvaluationConfig):
        return value
    loaded = load_phase2_config(Path(value))
    if not isinstance(loaded, EvaluationConfig):
        raise TypeError(f"expected EvaluationConfig at {value}")
    return loaded


def _load_training_pack_manifest(value: TrainingPackManifest | Path) -> TrainingPackManifest:
    if isinstance(value, TrainingPackManifest):
        return value
    return TrainingPackManifest.model_validate(json.loads(Path(value).read_text(encoding="utf-8")))


def _load_split_registry(value: ScenarioSplitRegistry | Path) -> ScenarioSplitRegistry:
    if isinstance(value, ScenarioSplitRegistry):
        return value
    return ScenarioSplitRegistry.model_validate(load_yaml_document(Path(value)))


def _resolve_planner_specs(
    *,
    planner_ids: Sequence[str] | None,
    checkpoint_manifests: Sequence[str | Path],
    policy_labels: Sequence[str] | None,
) -> list[EvaluatedPlannerSpec]:
    specs: list[EvaluatedPlannerSpec] = []
    for planner_id in tuple(planner_ids or ()):
        planner = build_planner(planner_id)
        specs.append(
            EvaluatedPlannerSpec(
                planner_key=planner.metadata.planner_id,
                planner_kind="builtin",
                display_name=planner.metadata.planner_id,
                evaluated_artifact_id=f"planner:{planner.metadata.planner_id}",
                version=planner.metadata.version,
                description=planner.metadata.description,
                source=planner.metadata.planner_id,
            )
        )
    labels = list(policy_labels or ())
    if labels and len(labels) != len(checkpoint_manifests):
        raise ValueError("policy_labels must be omitted or match checkpoint_manifests length")
    for index, manifest_source in enumerate(checkpoint_manifests):
        manifest_path = Path(manifest_source)
        manifest = PolicyCheckpointManifest.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        label = labels[index] if labels else f"{manifest.run_id}@{manifest.global_step}"
        specs.append(
            EvaluatedPlannerSpec(
                planner_key=_safe_filename(label),
                planner_kind="policy_checkpoint",
                display_name=label,
                evaluated_artifact_id=manifest.checkpoint_id,
                version=f"{manifest.algorithm}-step-{manifest.global_step}",
                description=(
                    "Trained RL policy restored from "
                    f"{manifest.checkpoint_id} ({manifest.algorithm})."
                ),
                source=str(manifest_path),
                checkpoint_manifest_path=manifest_path.resolve(),
                checkpoint_manifest=manifest,
                module_path=_resolve_policy_module_path(manifest),
                top_k=_coerce_optional_int(manifest.metadata.get("top_k")),
            )
        )
    return specs


def _select_scenario_entries(
    *,
    training_pack: TrainingPackManifest,
    split_registry: ScenarioSplitRegistry,
    splits: Sequence[str],
    limit_bundles_per_split: int | None,
    bundle_ids: Sequence[str],
) -> list[EvaluationScenarioEntry]:
    registry_by_bundle = {entry.bundle_id: entry for entry in split_registry.entries}
    requested_bundle_ids = set(bundle_ids)
    selected: list[EvaluationScenarioEntry] = []
    for split in splits:
        count_for_split = 0
        for training_entry in training_pack.entries:
            if training_entry.split != split:
                continue
            if requested_bundle_ids and training_entry.bundle_id not in requested_bundle_ids:
                continue
            split_entry = registry_by_bundle.get(training_entry.bundle_id)
            if split_entry is None:
                raise ValueError(
                    f"training pack bundle {training_entry.bundle_id} missing from split registry"
                )
            if split_entry.split != training_entry.split:
                raise ValueError(
                    "training pack and split registry disagree for "
                    f"{training_entry.bundle_id}: {training_entry.split} vs {split_entry.split}"
                )
            selected.append(
                EvaluationScenarioEntry(
                    training_entry=training_entry,
                    split_entry=split_entry,
                )
            )
            count_for_split += 1
            if limit_bundles_per_split is not None and count_for_split >= limit_bundles_per_split:
                break
    if not selected:
        raise ValueError("no scenario bundles matched the requested splits and filters")
    return selected


def _build_runner(spec: EvaluatedPlannerSpec) -> _BuiltinPlannerRunner | _PolicyModulePlannerRunner:
    if spec.planner_kind == "builtin":
        return _BuiltinPlannerRunner(spec)
    return _PolicyModulePlannerRunner(spec)


def _run_episode(
    *,
    spec: EvaluatedPlannerSpec,
    runner: _BuiltinPlannerRunner | _PolicyModulePlannerRunner,
    scenario: EvaluationScenarioEntry,
    bundle: ScenarioBundle,
    report_dir: Path,
) -> EvaluationEpisodeResult:
    if spec.planner_kind == "builtin":
        return _run_builtin_episode(
            spec=spec,
            runner=runner,
            scenario=scenario,
            bundle=bundle,
            report_dir=report_dir,
        )
    return _run_policy_episode(
        spec=spec,
        runner=runner,
        scenario=scenario,
        bundle=bundle,
        report_dir=report_dir,
    )


def _run_builtin_episode(
    *,
    spec: EvaluatedPlannerSpec,
    runner: _BuiltinPlannerRunner | _PolicyModulePlannerRunner,
    scenario: EvaluationScenarioEntry,
    bundle: ScenarioBundle,
    report_dir: Path,
) -> EvaluationEpisodeResult:
    assert isinstance(runner, _BuiltinPlannerRunner)
    env = OrbitalEnv(
        bundle,
        config=EnvRuntimeConfig(
            planner_id=f"planner:{spec.planner_key}",
            replay_dir=report_dir / "replays",
            decision_interval_seconds=bundle.decision_interval_seconds,
        ),
    )
    try:
        observation, _ = env.reset(
            seed=bundle.simulation_seed,
            planner_id=f"planner:{spec.planner_key}",
        )
        runner.start_episode(bundle=bundle, observation=observation)
        action_count = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, planner_trace = runner.select_action(observation)
            observation, _, terminated, truncated, _ = env.step(
                action,
                planner_trace=planner_trace,
            )
            action_count += 1
        replay_events = env.replay_events
    finally:
        env.close()
    return _persist_episode_result(
        spec=spec,
        scenario=scenario,
        bundle=bundle,
        replay_events=replay_events,
        action_count=action_count,
        report_dir=report_dir,
        episode_id=str(observation["episode_id"]),
        episode_seed=int(observation["episode_seed"]),
    )


def _run_policy_episode(
    *,
    spec: EvaluatedPlannerSpec,
    runner: _BuiltinPlannerRunner | _PolicyModulePlannerRunner,
    scenario: EvaluationScenarioEntry,
    bundle: ScenarioBundle,
    report_dir: Path,
) -> EvaluationEpisodeResult:
    assert isinstance(runner, _PolicyModulePlannerRunner)
    env = OrbitalTrainingEnv(
        bundle,
        config=EnvRuntimeConfig(
            planner_id=f"planner:{spec.planner_key}",
            replay_dir=report_dir / "replays",
            decision_interval_seconds=bundle.decision_interval_seconds,
        ),
        top_k=spec.top_k or 64,
    )
    try:
        observation, info = env.reset(
            seed=bundle.simulation_seed,
            options={"planner_id": f"planner:{spec.planner_key}"},
        )
        runner.start_episode()
        action_count = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            slot_index, planner_trace = runner.select_action(observation, info)
            observation, _, terminated, truncated, info = env.step(
                slot_index,
                planner_trace=planner_trace,
            )
            action_count += 1
        replay_events = env.runtime_env.replay_events
    finally:
        env.close()
    episode_id = str(info.get("episode_id", bundle.bundle_id))
    return _persist_episode_result(
        spec=spec,
        scenario=scenario,
        bundle=bundle,
        replay_events=replay_events,
        action_count=action_count,
        report_dir=report_dir,
        episode_id=episode_id,
        episode_seed=bundle.simulation_seed,
    )


def _persist_episode_result(
    *,
    spec: EvaluatedPlannerSpec,
    scenario: EvaluationScenarioEntry,
    bundle: ScenarioBundle,
    replay_events: Sequence[ReplayEvent],
    action_count: int,
    report_dir: Path,
    episode_id: str,
    episode_seed: int,
) -> EvaluationEpisodeResult:
    replay_path = (
        report_dir
        / "replays"
        / _safe_filename(scenario.split)
        / _safe_filename(spec.planner_key)
        / f"{_safe_filename(bundle.bundle_id)}.ndjson"
    )
    summary_path = (
        report_dir
        / "episodes"
        / _safe_filename(scenario.split)
        / _safe_filename(spec.planner_key)
        / f"{_safe_filename(bundle.bundle_id)}.json"
    )
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    replay_ndjson = replay_events_to_ndjson(replay_events)
    replay_path.write_text(f"{replay_ndjson}\n", encoding="utf-8")
    metrics = compute_episode_metrics(bundle=bundle, replay_events=replay_events)
    result = EvaluationEpisodeResult(
        planner_key=spec.planner_key,
        planner_kind=spec.planner_kind,
        display_name=spec.display_name,
        evaluated_artifact_id=spec.evaluated_artifact_id,
        planner_version=spec.version,
        bundle_id=bundle.bundle_id,
        manifest_id=bundle.manifest_id,
        split=scenario.split,
        scenario_family=bundle.scenario_family,
        scenario_path=scenario.scenario_path,
        ood_axes=tuple(scenario.split_entry.ood_axes),
        difficulty_tier=scenario.split_entry.difficulty_tier,
        episode_id=episode_id,
        episode_seed=episode_seed,
        episode_fingerprint=_episode_fingerprint(replay_events),
        replay_fingerprint=f"sha256:{sha256_fingerprint(replay_ndjson)}",
        action_count=action_count,
        metrics=metrics,
        reward_audit=_reward_audit_from_replay(replay_events),
        bundle_profile=_bundle_profile(bundle=bundle, split_entry=scenario.split_entry),
        artifacts=EvaluationEpisodeArtifacts(
            replay_path=replay_path.resolve(),
            summary_path=summary_path.resolve(),
        ),
    )
    summary_path.write_text(_pretty_json(result.to_dict()), encoding="utf-8")
    return result


def _build_run_summary(
    *,
    evaluation: EvaluationConfig,
    training_pack: TrainingPackManifest,
    split_registry: ScenarioSplitRegistry,
    planner_specs: Sequence[EvaluatedPlannerSpec],
    scenario_entries: Sequence[EvaluationScenarioEntry],
    episode_results: Sequence[EvaluationEpisodeResult],
) -> dict[str, Any]:
    episodes_by_split: dict[str, list[EvaluationEpisodeResult]] = defaultdict(list)
    for episode in episode_results:
        episodes_by_split[episode.split].append(episode)

    split_summaries: dict[str, Any] = {}
    for split in evaluation.split_order:
        split_episodes = episodes_by_split.get(split, [])
        if not split_episodes:
            continue
        split_summaries[split] = {
            "planner_summaries": _summaries_by_planner(
                episodes=split_episodes,
                metric_names=evaluation.planner_agnostic_metrics,
                bootstrap_replicates=evaluation.bootstrap_replicates,
                confidence_level=evaluation.confidence_level,
            ),
            "family_breakdowns": _family_breakdowns(
                episodes=split_episodes,
                metric_names=evaluation.planner_agnostic_metrics,
                bootstrap_replicates=evaluation.bootstrap_replicates,
                confidence_level=evaluation.confidence_level,
            ),
            "pairwise_comparisons": _pairwise_comparisons(
                episodes=split_episodes,
                planner_order=[spec.planner_key for spec in planner_specs],
                metric_names=evaluation.planner_agnostic_metrics,
                bootstrap_replicates=evaluation.bootstrap_replicates,
                confidence_level=evaluation.confidence_level,
                primary_metric=evaluation.primary_ranking_metric,
            ),
        }

    notable_episodes = _select_notable_episodes(
        episode_results=episode_results,
        planner_specs=planner_specs,
        primary_metric=evaluation.primary_ranking_metric,
        count=evaluation.notable_episode_count,
    )
    summary_payload = {
        "run_id": "",
        "evaluation_id": evaluation.evaluation_id,
        "benchmark_id": evaluation.benchmark_id,
        "training_pack_id": training_pack.training_pack_id,
        "split_registry_id": split_registry.registry_id,
        "selected_splits": sorted(
            {entry.split for entry in scenario_entries},
            key=evaluation.split_order.index,
        ),
        "selected_bundle_ids": [entry.bundle_id for entry in scenario_entries],
        "planners": [
            {
                "planner_key": spec.planner_key,
                "planner_kind": spec.planner_kind,
                "display_name": spec.display_name,
                "evaluated_artifact_id": spec.evaluated_artifact_id,
                "version": spec.version,
                "description": spec.description,
                "source": spec.source,
                "checkpoint_manifest_path": (
                    str(spec.checkpoint_manifest_path) if spec.checkpoint_manifest_path else None
                ),
            }
            for spec in planner_specs
        ],
        "config": evaluation.model_dump(mode="json"),
        "split_summaries": split_summaries,
        "notable_episodes": notable_episodes,
        "episodes": [episode.to_dict() for episode in episode_results],
    }
    return summary_payload


def _summaries_by_planner(
    *,
    episodes: Sequence[EvaluationEpisodeResult],
    metric_names: Sequence[str],
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, Any]:
    grouped: dict[str, list[EvaluationEpisodeResult]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.planner_key].append(episode)
    result: dict[str, Any] = {}
    for planner_key, planner_episodes in sorted(grouped.items()):
        first = planner_episodes[0]
        reward_summary = _reward_audit_summary(planner_episodes)
        result[planner_key] = {
            "display_name": first.display_name,
            "planner_kind": first.planner_kind,
            "evaluated_artifact_id": first.evaluated_artifact_id,
            "episode_count": len(planner_episodes),
            "scenario_families": sorted({episode.scenario_family for episode in planner_episodes}),
            "metrics": {
                metric_name: _metric_summary(
                    [_metric_value(episode.metrics, metric_name) for episode in planner_episodes],
                    bootstrap_replicates=bootstrap_replicates,
                    confidence_level=confidence_level,
                    seed_material=f"{planner_key}:{metric_name}:{len(planner_episodes)}",
                )
                for metric_name in metric_names
            },
            "reward_audit_summary": reward_summary,
        }
    return result


def _family_breakdowns(
    *,
    episodes: Sequence[EvaluationEpisodeResult],
    metric_names: Sequence[str],
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[EvaluationEpisodeResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode in episodes:
        grouped[episode.planner_key][episode.scenario_family].append(episode)
    output: dict[str, Any] = {}
    for planner_key in sorted(grouped):
        output[planner_key] = {}
        for family in sorted(grouped[planner_key]):
            family_episodes = grouped[planner_key][family]
            output[planner_key][family] = {
                "episode_count": len(family_episodes),
                "metrics": {
                    metric_name: _metric_summary(
                        [
                            _metric_value(episode.metrics, metric_name)
                            for episode in family_episodes
                        ],
                        bootstrap_replicates=bootstrap_replicates,
                        confidence_level=confidence_level,
                        seed_material=f"{planner_key}:{family}:{metric_name}",
                    )
                    for metric_name in metric_names
                },
            }
    return output


def _pairwise_comparisons(
    *,
    episodes: Sequence[EvaluationEpisodeResult],
    planner_order: Sequence[str],
    metric_names: Sequence[str],
    bootstrap_replicates: int,
    confidence_level: float,
    primary_metric: str,
) -> list[dict[str, Any]]:
    by_planner_bundle: dict[str, dict[str, EvaluationEpisodeResult]] = defaultdict(dict)
    for episode in episodes:
        by_planner_bundle[episode.planner_key][episode.bundle_id] = episode
    pairs: list[dict[str, Any]] = []
    for left, right in combinations(planner_order, 2):
        if left not in by_planner_bundle or right not in by_planner_bundle:
            continue
        matched_bundle_ids = sorted(set(by_planner_bundle[left]) & set(by_planner_bundle[right]))
        if not matched_bundle_ids:
            continue
        metrics: dict[str, Any] = {}
        for metric_name in metric_names:
            diffs: list[float] = []
            outcomes: list[dict[str, Any]] = []
            for bundle_id in matched_bundle_ids:
                left_episode = by_planner_bundle[left][bundle_id]
                right_episode = by_planner_bundle[right][bundle_id]
                left_value = _metric_value(left_episode.metrics, metric_name)
                right_value = _metric_value(right_episode.metrics, metric_name)
                if left_value is None or right_value is None:
                    continue
                diff = round(left_value - right_value, 6)
                diffs.append(diff)
                outcomes.append(
                    {
                        "bundle_id": bundle_id,
                        "left_value": round(left_value, 6),
                        "right_value": round(right_value, 6),
                        "difference": diff,
                        "outcome": _pairwise_outcome(
                            left_value=left_value,
                            right_value=right_value,
                            direction=METRIC_DIRECTIONS[metric_name],
                        ),
                    }
                )
            metrics[metric_name] = {
                "sample_count": len(outcomes),
                "difference_summary": _metric_summary(
                    diffs,
                    bootstrap_replicates=bootstrap_replicates,
                    confidence_level=confidence_level,
                    seed_material=f"{left}:{right}:{metric_name}",
                ),
                "wins_left": sum(1 for item in outcomes if item["outcome"] == "left"),
                "wins_right": sum(1 for item in outcomes if item["outcome"] == "right"),
                "ties": sum(1 for item in outcomes if item["outcome"] == "tie"),
                "episode_outcomes": outcomes if metric_name == primary_metric else [],
            }
        pairs.append(
            {
                "left_planner": left,
                "right_planner": right,
                "matched_episode_count": len(matched_bundle_ids),
                "primary_metric": primary_metric,
                "metrics": metrics,
            }
        )
    return pairs


def _select_notable_episodes(
    *,
    episode_results: Sequence[EvaluationEpisodeResult],
    planner_specs: Sequence[EvaluatedPlannerSpec],
    primary_metric: str,
    count: int,
) -> dict[str, Any]:
    bundles = _bundle_views(episode_results)
    policy_planners = [spec for spec in planner_specs if spec.planner_kind == "policy_checkpoint"]
    baseline_keys = [spec.planner_key for spec in planner_specs if spec.planner_kind == "builtin"]
    notable: dict[str, Any] = {
        "policy_vs_best_baseline": {},
        "cloud_heavy_episodes": [],
        "downlink_heavy_episodes": [],
        "outage_degradation_episodes": [],
    }
    for spec in policy_planners:
        candidate_rows: list[dict[str, Any]] = []
        for bundle in bundles:
            policy_episode = bundle["episodes_by_planner"].get(spec.planner_key)
            if policy_episode is None:
                continue
            baseline_pool = [
                bundle["episodes_by_planner"][planner_key]
                for planner_key in baseline_keys
                if planner_key in bundle["episodes_by_planner"]
            ]
            if not baseline_pool:
                continue
            best_baseline = max(
                baseline_pool,
                key=lambda episode: _metric_value_or_inf(
                    episode.metrics,
                    primary_metric,
                    direction=METRIC_DIRECTIONS[primary_metric],
                ),
            )
            policy_value = _metric_value(policy_episode.metrics, primary_metric)
            baseline_value = _metric_value(best_baseline.metrics, primary_metric)
            if policy_value is None or baseline_value is None:
                continue
            candidate_rows.append(
                {
                    **_bundle_slice_payload(bundle, primary_metric=primary_metric),
                    "policy_planner": spec.planner_key,
                    "policy_value": round(policy_value, 6),
                    "best_baseline_planner": best_baseline.planner_key,
                    "best_baseline_value": round(baseline_value, 6),
                    "difference_vs_best_baseline": round(policy_value - baseline_value, 6),
                }
            )
        ordered = sorted(
            candidate_rows,
            key=lambda item: item["difference_vs_best_baseline"],
            reverse=True,
        )
        wins = [item for item in ordered if item["difference_vs_best_baseline"] > 0]
        losses = [
            item
            for item in sorted(candidate_rows, key=lambda item: item["difference_vs_best_baseline"])
            if item["difference_vs_best_baseline"] < 0
        ]
        notable["policy_vs_best_baseline"][spec.planner_key] = {
            "biggest_rl_wins": wins[:count],
            "biggest_rl_losses": losses[:count],
        }

    cloud_ranked = sorted(
        bundles,
        key=lambda item: float(item["bundle_profile"]["cloud_pressure_index"]),
        reverse=True,
    )
    notable["cloud_heavy_episodes"] = [
        _bundle_slice_payload(item, primary_metric=primary_metric) for item in cloud_ranked[:count]
    ]

    downlink_ranked = sorted(
        bundles,
        key=lambda item: float(item["bundle_profile"]["downlink_pressure_ratio"]),
        reverse=True,
    )
    notable["downlink_heavy_episodes"] = [
        _bundle_slice_payload(item, primary_metric=primary_metric)
        for item in downlink_ranked[:count]
    ]

    outage_candidates = [
        item
        for item in bundles
        if item["scenario_family"] in {"station_outage", "constellation_degradation"}
        or item["ood_axes"]
        or float(item["bundle_profile"]["station_outage_index"]) > 0
        or float(item["bundle_profile"]["downlink_outage_risk_mean"]) > 0
    ]
    outage_ranked = sorted(
        outage_candidates,
        key=lambda item: (
            float(item["bundle_profile"]["station_outage_index"]),
            float(item["bundle_profile"]["downlink_outage_risk_mean"]),
        ),
        reverse=True,
    )
    notable["outage_degradation_episodes"] = [
        _bundle_slice_payload(item, primary_metric=primary_metric) for item in outage_ranked[:count]
    ]
    return notable


def _bundle_views(episode_results: Sequence[EvaluationEpisodeResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[EvaluationEpisodeResult]] = defaultdict(list)
    for episode in episode_results:
        grouped[(episode.split, episode.bundle_id)].append(episode)
    bundles: list[dict[str, Any]] = []
    for (_, bundle_id), rows in sorted(grouped.items()):
        first = rows[0]
        bundles.append(
            {
                "bundle_id": bundle_id,
                "split": first.split,
                "manifest_id": first.manifest_id,
                "scenario_family": first.scenario_family,
                "ood_axes": list(first.ood_axes),
                "bundle_profile": dict(first.bundle_profile),
                "episodes_by_planner": {row.planner_key: row for row in rows},
            }
        )
    return bundles


def _bundle_slice_payload(bundle_view: Mapping[str, Any], *, primary_metric: str) -> dict[str, Any]:
    planner_rows = sorted(
        bundle_view["episodes_by_planner"].values(),
        key=lambda episode: _metric_value_or_inf(
            episode.metrics,
            primary_metric,
            direction=METRIC_DIRECTIONS[primary_metric],
        ),
        reverse=True,
    )
    return {
        "bundle_id": bundle_view["bundle_id"],
        "split": bundle_view["split"],
        "manifest_id": bundle_view["manifest_id"],
        "scenario_family": bundle_view["scenario_family"],
        "ood_axes": list(bundle_view["ood_axes"]),
        "bundle_profile": _json_compatible(bundle_view["bundle_profile"]),
        "planner_rankings": [
            {
                "planner_key": episode.planner_key,
                "display_name": episode.display_name,
                "primary_metric_value": _metric_value(episode.metrics, primary_metric),
                "mission_utility": episode.metrics.mission_utility,
                "replay_path": str(episode.artifacts.replay_path),
                "summary_path": str(episode.artifacts.summary_path),
            }
            for episode in planner_rows
        ],
    }


def _metric_summary(
    values: Sequence[float | None],
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    seed_material: str,
) -> dict[str, Any]:
    observed = [float(value) for value in values if value is not None]
    if not observed:
        return {
            "sample_count": 0,
            "missing_count": len(values),
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    ci_lower, ci_upper = _bootstrap_ci(
        observed,
        replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        seed_material=seed_material,
    )
    return {
        "sample_count": len(observed),
        "missing_count": len(values) - len(observed),
        "mean": round(float(mean(observed)), 6),
        "median": round(float(median(observed)), 6),
        "stdev": round(float(pstdev(observed)), 6) if len(observed) > 1 else 0.0,
        "min": round(min(observed), 6),
        "max": round(max(observed), 6),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _bootstrap_ci(
    values: Sequence[float],
    *,
    replicates: int,
    confidence_level: float,
    seed_material: str,
) -> tuple[float, float]:
    if len(values) == 1:
        only = round(float(values[0]), 6)
        return only, only
    rng = random.Random(stable_token(seed_material, length=16))
    samples: list[float] = []
    size = len(values)
    for _ in range(replicates):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        samples.append(float(mean(sample)))
    samples.sort()
    lower_index = max(0, ceil(((1.0 - confidence_level) / 2.0) * len(samples)) - 1)
    upper_index = min(
        len(samples) - 1,
        ceil((1.0 - ((1.0 - confidence_level) / 2.0)) * len(samples)) - 1,
    )
    return round(samples[lower_index], 6), round(samples[upper_index], 6)


def _reward_audit_from_replay(replay_events: Sequence[ReplayEvent]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    reward_event_count = 0
    total_reward = 0.0
    for event in replay_events:
        if event.event_type != "reward_assessed":
            continue
        reward_event_count += 1
        total_reward += float(event.payload.get("total_reward", 0.0))
        for component, value in dict(event.payload.get("components", {})).items():
            totals[str(component)] += float(value)
    reward_summary = {
        "reward_event_count": float(reward_event_count),
        "total_reward_sum": round(total_reward, 6),
    }
    for component, value in sorted(totals.items()):
        reward_summary[component] = round(value, 6)
    return reward_summary


def _reward_audit_summary(episodes: Sequence[EvaluationEpisodeResult]) -> dict[str, float]:
    keys = sorted({key for episode in episodes for key in episode.reward_audit})
    summary: dict[str, float] = {}
    for key in keys:
        values = [episode.reward_audit.get(key, 0.0) for episode in episodes]
        summary[f"{key}_mean"] = round(float(mean(values)), 6)
    return summary


def _bundle_profile(
    *,
    bundle: ScenarioBundle,
    split_entry: ScenarioSplitEntry,
) -> dict[str, float | int | str | list[str]]:
    total_observation_volume = sum(
        float(opportunity.estimated_data_volume_mb)
        for opportunity in bundle.observation_opportunities
    )
    total_downlink_capacity = max(
        sum(float(window.max_volume_mb) for window in bundle.downlink_windows),
        1.0,
    )
    cloud_values = [
        float(opportunity.predicted_cloud_obstruction_prob)
        for opportunity in bundle.observation_opportunities
    ]
    outage_values = [float(window.outage_risk) for window in bundle.downlink_windows]
    offline_station_fraction = sum(
        1 for station in bundle.ground_stations if station.capabilities.availability == "offline"
    ) / max(len(bundle.ground_stations), 1)
    return {
        "difficulty_tier": split_entry.difficulty_tier,
        "cloud_pressure_index": round(_mean_or_zero(cloud_values), 6),
        "downlink_pressure_ratio": round(total_observation_volume / total_downlink_capacity, 6),
        "downlink_outage_risk_mean": round(_mean_or_zero(outage_values), 6),
        "station_outage_index": round(float(offline_station_fraction), 6),
        "incident_count": len(bundle.incidents),
        "observation_opportunity_count": len(bundle.observation_opportunities),
        "downlink_window_count": len(bundle.downlink_windows),
        "ood_axes": list(split_entry.ood_axes),
        "scenario_family": bundle.scenario_family,
    }


def _write_report_manifests(
    *,
    manifest_dir: Path,
    evaluation: EvaluationConfig,
    summary_payload: Mapping[str, Any],
    summary_path: Path,
) -> list[Path]:
    report_paths: list[Path] = []
    for split, split_summary in summary_payload["split_summaries"].items():
        planner_summaries = split_summary["planner_summaries"]
        for planner_key, planner_summary in planner_summaries.items():
            metrics_payload: dict[str, float] = {}
            for metric_name, metric_summary in planner_summary["metrics"].items():
                for suffix in ("mean", "ci_lower", "ci_upper"):
                    value = metric_summary.get(suffix)
                    if value is not None:
                        metrics_payload[f"{metric_name}.{suffix}"] = float(value)
            manifest = EvaluationReportManifest(
                report_id=stable_id("report", evaluation.evaluation_id, split, planner_key),
                benchmark_id=evaluation.benchmark_id,
                evaluated_artifact_id=planner_summary["evaluated_artifact_id"],
                evaluation_config_id=evaluation.evaluation_id,
                split=split,
                generated_at_utc=datetime.now(UTC),
                summary_path=str(summary_path),
                benchmark_metrics=metrics_payload,
                reward_audit_summary=planner_summary["reward_audit_summary"],
                artifact_fingerprint="sha256:"
                + sha256_fingerprint(
                    {
                        "planner_key": planner_key,
                        "split": split,
                        "metrics": metrics_payload,
                        "reward_audit_summary": planner_summary["reward_audit_summary"],
                    }
                ),
            )
            report_path = (
                manifest_dir / f"{_safe_filename(split)}--{_safe_filename(planner_key)}.json"
            )
            report_path.write_text(
                canonical_json_dumps(manifest.model_dump(mode="json")) + "\n",
                encoding="utf-8",
            )
            report_paths.append(report_path)
    return report_paths


def _write_run_manifest(
    *,
    manifest_dir: Path,
    evaluation: EvaluationConfig,
    training_pack: TrainingPackManifest,
    split_registry: ScenarioSplitRegistry,
    planner_specs: Sequence[EvaluatedPlannerSpec],
    summary_payload: Mapping[str, Any],
    report_dir: Path,
    summary_path: Path,
    markdown_path: Path | None,
    report_manifest_paths: Sequence[Path],
    csv_outputs: Mapping[str, Path],
    run_id: str,
    generated_at: datetime,
) -> Path:
    manifest_payload = {
        "run_id": run_id,
        "benchmark_id": evaluation.benchmark_id,
        "evaluation_config_id": evaluation.evaluation_id,
        "training_pack_id": training_pack.training_pack_id,
        "split_registry_id": split_registry.registry_id,
        "generated_at_utc": format_utc_timestamp(generated_at),
        "report_dir": str(report_dir),
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path) if markdown_path else None,
        "episode_metrics_path": (
            str(csv_outputs["episode_metrics"]) if "episode_metrics" in csv_outputs else None
        ),
        "pairwise_comparisons_path": (
            str(csv_outputs["pairwise_comparisons"])
            if "pairwise_comparisons" in csv_outputs
            else None
        ),
        "notable_episodes_path": (
            str(csv_outputs["notable_episodes"]) if "notable_episodes" in csv_outputs else None
        ),
        "selected_splits": list(summary_payload["selected_splits"]),
        "selected_bundle_ids": list(summary_payload["selected_bundle_ids"]),
        "planners": [
            {
                "planner_key": spec.planner_key,
                "planner_kind": spec.planner_kind,
                "display_name": spec.display_name,
                "evaluated_artifact_id": spec.evaluated_artifact_id,
                "description": spec.description,
                "source": spec.source,
                "checkpoint_manifest_path": (
                    str(spec.checkpoint_manifest_path) if spec.checkpoint_manifest_path else None
                ),
            }
            for spec in planner_specs
        ],
        "report_manifests": [str(path) for path in report_manifest_paths],
    }
    artifact_fingerprint = "sha256:" + sha256_fingerprint(manifest_payload)
    manifest = EvaluationRunManifest(
        run_id=run_id,
        benchmark_id=evaluation.benchmark_id,
        evaluation_config_id=evaluation.evaluation_id,
        training_pack_id=training_pack.training_pack_id,
        split_registry_id=split_registry.registry_id,
        generated_at_utc=generated_at,
        report_dir=str(report_dir),
        summary_path=str(summary_path),
        markdown_path=str(markdown_path) if markdown_path else None,
        episode_metrics_path=(
            str(csv_outputs["episode_metrics"]) if "episode_metrics" in csv_outputs else None
        ),
        pairwise_comparisons_path=(
            str(csv_outputs["pairwise_comparisons"])
            if "pairwise_comparisons" in csv_outputs
            else None
        ),
        notable_episodes_path=(
            str(csv_outputs["notable_episodes"]) if "notable_episodes" in csv_outputs else None
        ),
        selected_splits=list(summary_payload["selected_splits"]),
        selected_bundle_ids=list(summary_payload["selected_bundle_ids"]),
        planners=[
            EvaluationRunPlanner(
                planner_key=spec.planner_key,
                planner_kind=spec.planner_kind,
                display_name=spec.display_name,
                evaluated_artifact_id=spec.evaluated_artifact_id,
                description=spec.description,
                source=spec.source,
                checkpoint_manifest_path=(
                    str(spec.checkpoint_manifest_path) if spec.checkpoint_manifest_path else None
                ),
            )
            for spec in planner_specs
        ],
        report_manifests=[str(path) for path in report_manifest_paths],
        artifact_fingerprint=artifact_fingerprint,
    )
    path = manifest_dir / "run_manifest.json"
    path.write_text(canonical_json_dumps(manifest.model_dump(mode="json")) + "\n", encoding="utf-8")
    return path


def _write_csv_outputs(
    *,
    report_dir: Path,
    summary_payload: Mapping[str, Any],
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    episode_metrics_path = report_dir / "episode_metrics.csv"
    with episode_metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "planner_key",
                "planner_kind",
                "bundle_id",
                "scenario_family",
                "difficulty_tier",
                "mission_utility",
                "useful_observation_value_captured",
                "cloud_waste_rate",
                "missed_urgent_incident_rate",
                "opportunity_utilization_efficiency",
                "time_to_first_useful_observation_seconds_mean",
                "downlink_latency_seconds_mean",
                "replay_path",
                "summary_path",
            ],
        )
        writer.writeheader()
        for episode in summary_payload["episodes"]:
            writer.writerow(
                {
                    "split": episode["split"],
                    "planner_key": episode["planner_key"],
                    "planner_kind": episode["planner_kind"],
                    "bundle_id": episode["bundle_id"],
                    "scenario_family": episode["scenario_family"],
                    "difficulty_tier": episode["difficulty_tier"],
                    "mission_utility": episode["metrics"]["mission_utility"],
                    "useful_observation_value_captured": episode["metrics"][
                        "useful_observation_value_captured"
                    ],
                    "cloud_waste_rate": episode["metrics"]["cloud_waste_rate"],
                    "missed_urgent_incident_rate": episode["metrics"][
                        "missed_urgent_incident_rate"
                    ],
                    "opportunity_utilization_efficiency": episode["metrics"][
                        "opportunity_utilization_efficiency"
                    ],
                    "time_to_first_useful_observation_seconds_mean": episode["metrics"][
                        "time_to_first_useful_observation_seconds"
                    ]["mean"],
                    "downlink_latency_seconds_mean": episode["metrics"]["downlink_latency_seconds"][
                        "mean"
                    ],
                    "replay_path": episode["replay_path"],
                    "summary_path": episode["summary_path"],
                }
            )
    outputs["episode_metrics"] = episode_metrics_path

    pairwise_path = report_dir / "pairwise_comparisons.csv"
    outcomes_path = report_dir / "episode_outcomes.csv"
    with (
        pairwise_path.open("w", encoding="utf-8", newline="") as pairwise_handle,
        outcomes_path.open("w", encoding="utf-8", newline="") as outcomes_handle,
    ):
        pairwise_writer = csv.DictWriter(
            pairwise_handle,
            fieldnames=[
                "split",
                "left_planner",
                "right_planner",
                "metric",
                "matched_episode_count",
                "sample_count",
                "mean_difference",
                "ci_lower",
                "ci_upper",
                "wins_left",
                "wins_right",
                "ties",
            ],
        )
        outcomes_writer = csv.DictWriter(
            outcomes_handle,
            fieldnames=[
                "split",
                "left_planner",
                "right_planner",
                "metric",
                "bundle_id",
                "left_value",
                "right_value",
                "difference",
                "outcome",
            ],
        )
        pairwise_writer.writeheader()
        outcomes_writer.writeheader()
        for split, split_summary in summary_payload["split_summaries"].items():
            for pair in split_summary["pairwise_comparisons"]:
                for metric_name, metric_summary in pair["metrics"].items():
                    diff_summary = metric_summary["difference_summary"]
                    pairwise_writer.writerow(
                        {
                            "split": split,
                            "left_planner": pair["left_planner"],
                            "right_planner": pair["right_planner"],
                            "metric": metric_name,
                            "matched_episode_count": pair["matched_episode_count"],
                            "sample_count": metric_summary["sample_count"],
                            "mean_difference": diff_summary["mean"],
                            "ci_lower": diff_summary["ci_lower"],
                            "ci_upper": diff_summary["ci_upper"],
                            "wins_left": metric_summary["wins_left"],
                            "wins_right": metric_summary["wins_right"],
                            "ties": metric_summary["ties"],
                        }
                    )
                    for row in metric_summary["episode_outcomes"]:
                        outcomes_writer.writerow(
                            {
                                "split": split,
                                "left_planner": pair["left_planner"],
                                "right_planner": pair["right_planner"],
                                "metric": metric_name,
                                **row,
                            }
                        )
    outputs["pairwise_comparisons"] = pairwise_path
    outputs["episode_outcomes"] = outcomes_path

    notable_path = report_dir / "notable_episodes.csv"
    with notable_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "planner_key",
                "bundle_id",
                "split",
                "scenario_family",
                "primary_metric_value",
                "difference_vs_best_baseline",
            ],
        )
        writer.writeheader()
        notable = summary_payload["notable_episodes"]
        for planner_key, rows in notable["policy_vs_best_baseline"].items():
            for category in ("biggest_rl_wins", "biggest_rl_losses"):
                for row in rows[category]:
                    writer.writerow(
                        {
                            "category": category,
                            "planner_key": planner_key,
                            "bundle_id": row["bundle_id"],
                            "split": row["split"],
                            "scenario_family": row["scenario_family"],
                            "primary_metric_value": row["policy_value"],
                            "difference_vs_best_baseline": row["difference_vs_best_baseline"],
                        }
                    )
        for category in (
            "cloud_heavy_episodes",
            "downlink_heavy_episodes",
            "outage_degradation_episodes",
        ):
            for row in notable.get(category, []):
                top_value = (
                    row["planner_rankings"][0]["primary_metric_value"]
                    if row["planner_rankings"]
                    else None
                )
                writer.writerow(
                    {
                        "category": category,
                        "planner_key": "",
                        "bundle_id": row["bundle_id"],
                        "split": row["split"],
                        "scenario_family": row["scenario_family"],
                        "primary_metric_value": top_value,
                        "difference_vs_best_baseline": "",
                    }
                )
        outputs["notable_episodes"] = notable_path
    return outputs


def _write_dvc_outputs(*, report_dir: Path, summary_payload: Mapping[str, Any]) -> None:
    dvc_metrics_path = report_dir / "dvc_metrics.json"
    planner_metric_means: dict[str, Any] = {}
    for split, split_summary in summary_payload["split_summaries"].items():
        planner_metric_means[split] = {}
        for planner_key, planner_summary in split_summary["planner_summaries"].items():
            planner_metric_means[split][planner_key] = {
                metric_name: metric_summary["mean"]
                for metric_name, metric_summary in planner_summary["metrics"].items()
            }
    dvc_metrics_path.write_text(_pretty_json(planner_metric_means), encoding="utf-8")

    dvc_plot_dir = report_dir / "dvc_plots"
    dvc_plot_dir.mkdir(parents=True, exist_ok=True)
    means_path = dvc_plot_dir / "planner_metric_means.csv"
    with means_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "planner_key", "metric", "mean", "ci_lower", "ci_upper"],
        )
        writer.writeheader()
        for split, split_summary in summary_payload["split_summaries"].items():
            for planner_key, planner_summary in split_summary["planner_summaries"].items():
                for metric_name, metric_summary in planner_summary["metrics"].items():
                    writer.writerow(
                        {
                            "split": split,
                            "planner_key": planner_key,
                            "metric": metric_name,
                            "mean": metric_summary["mean"],
                            "ci_lower": metric_summary["ci_lower"],
                            "ci_upper": metric_summary["ci_upper"],
                        }
                    )
    pairwise_path = dvc_plot_dir / "pairwise_primary_metric.csv"
    with pairwise_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "left_planner",
                "right_planner",
                "metric",
                "mean_difference",
                "ci_lower",
                "ci_upper",
            ],
        )
        writer.writeheader()
        for split, split_summary in summary_payload["split_summaries"].items():
            for pair in split_summary["pairwise_comparisons"]:
                metric = pair["primary_metric"]
                summary = pair["metrics"][metric]["difference_summary"]
                writer.writerow(
                    {
                        "split": split,
                        "left_planner": pair["left_planner"],
                        "right_planner": pair["right_planner"],
                        "metric": metric,
                        "mean_difference": summary["mean"],
                        "ci_lower": summary["ci_lower"],
                        "ci_upper": summary["ci_upper"],
                    }
                )


def _log_to_wandb(
    *,
    evaluation: EvaluationConfig,
    run_name: str,
    summary_payload: Mapping[str, Any],
    report_dir: Path,
) -> None:
    wandb_run = maybe_init_wandb(
        evaluation.wandb,
        run_name=run_name,
        run_config={
            "evaluation_id": evaluation.evaluation_id,
            "benchmark_id": evaluation.benchmark_id,
            "report_dir": str(report_dir),
        },
    )
    try:
        for split, split_summary in summary_payload["split_summaries"].items():
            for planner_key, planner_summary in split_summary["planner_summaries"].items():
                row = {
                    "split": split,
                    "planner_key": planner_key,
                }
                for metric_name, metric_summary in planner_summary["metrics"].items():
                    row[f"{metric_name}_mean"] = metric_summary["mean"]
                    row[f"{metric_name}_ci_lower"] = metric_summary["ci_lower"]
                    row[f"{metric_name}_ci_upper"] = metric_summary["ci_upper"]
                wandb_run.log(row)
    finally:
        wandb_run.finish()


def _build_markdown_report(summary_payload: Mapping[str, Any]) -> str:
    lines = [
        f"# Phase 2 Evaluation: {summary_payload['evaluation_id']}",
        "",
        f"- Benchmark: `{summary_payload['benchmark_id']}`",
        f"- Training pack: `{summary_payload['training_pack_id']}`",
        f"- Split registry: `{summary_payload['split_registry_id']}`",
        f"- Splits: {', '.join(summary_payload['selected_splits'])}",
        f"- Bundles: {len(summary_payload['selected_bundle_ids'])}",
        "",
    ]
    for split, split_summary in summary_payload["split_summaries"].items():
        lines.extend(
            [
                f"## Split `{split}`",
                "",
                (
                    "| Planner | Mission Utility | UOVC | Cloud Waste | Missed Urgent | "
                    "OUE | TTFUO Mean (s) | Downlink Mean (s) |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for planner_key, planner_summary in split_summary["planner_summaries"].items():
            metrics = planner_summary["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        planner_key,
                        _md_float(metrics["mission_utility"]["mean"]),
                        _md_float(metrics["useful_observation_value_captured"]["mean"]),
                        _md_float(metrics["cloud_waste_rate"]["mean"]),
                        _md_float(metrics["missed_urgent_incident_rate"]["mean"]),
                        _md_float(metrics["opportunity_utilization_efficiency"]["mean"]),
                        _md_float(metrics["time_to_first_useful_observation_seconds"]["mean"]),
                        _md_float(metrics["downlink_latency_seconds"]["mean"]),
                    ]
                )
                + " |"
            )
        lines.extend(["", "### Paired Comparisons", ""])
        for pair in split_summary["pairwise_comparisons"]:
            metric = pair["primary_metric"]
            primary_summary = pair["metrics"][metric]
            diff_summary = primary_summary["difference_summary"]
            lines.append(
                f"- `{pair['left_planner']}` vs `{pair['right_planner']}` on `{metric}`: "
                f"mean diff {_md_float(diff_summary['mean'])} "
                f"[{_md_float(diff_summary['ci_lower'])}, {_md_float(diff_summary['ci_upper'])}], "
                "W/L/T "
                f"{primary_summary['wins_left']}/"
                f"{primary_summary['wins_right']}/"
                f"{primary_summary['ties']}"
            )
        lines.append("")
    notable = summary_payload["notable_episodes"]
    lines.extend(["## Notable Episodes", ""])
    for planner_key, rows in notable["policy_vs_best_baseline"].items():
        lines.append(f"### `{planner_key}`")
        for category in ("biggest_rl_wins", "biggest_rl_losses"):
            lines.append(f"- {category}:")
            for row in rows[category]:
                lines.append(
                    f"  - `{row['bundle_id']}` ({row['split']}, {row['scenario_family']}): "
                    f"delta={_md_float(row['difference_vs_best_baseline'])} "
                    f"vs `{row['best_baseline_planner']}`"
                )
        lines.append("")
    for category in (
        "cloud_heavy_episodes",
        "downlink_heavy_episodes",
        "outage_degradation_episodes",
    ):
        lines.append(f"### {category.replace('_', ' ').title()}")
        for row in notable[category]:
            winner = row["planner_rankings"][0]["planner_key"] if row["planner_rankings"] else "-"
            lines.append(
                f"- `{row['bundle_id']}` ({row['split']}, {row['scenario_family']}): "
                f"winner=`{winner}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _resolve_policy_module_path(manifest: PolicyCheckpointManifest) -> Path:
    candidates: list[Path] = []
    metadata_path = manifest.metadata.get("rllib_module_path")
    if isinstance(metadata_path, str) and metadata_path:
        candidates.append(Path(metadata_path))
    checkpoint_path = Path(manifest.checkpoint_path)
    candidates.extend(
        [
            checkpoint_path / "rllib_module" / "default_policy",
            checkpoint_path / "learner_group" / "learner" / "rl_module" / "default_policy",
            checkpoint_path / "default_policy",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "unable to resolve RLModule checkpoint for "
        f"{manifest.checkpoint_id} under {manifest.checkpoint_path}"
    )


def _metric_value(metrics: EpisodeMetrics, metric_name: str) -> float | None:
    if metric_name == "time_to_first_useful_observation_seconds":
        return metrics.time_to_first_useful_observation_seconds.mean
    if metric_name == "downlink_latency_seconds":
        return metrics.downlink_latency_seconds.mean
    return getattr(metrics, metric_name)


def _metric_value_or_inf(
    metrics: EpisodeMetrics,
    metric_name: str,
    *,
    direction: MetricDirection,
) -> float:
    value = _metric_value(metrics, metric_name)
    if value is None:
        return float("-inf") if direction == "higher" else float("inf")
    return float(value) if direction == "higher" else -float(value)


def _pairwise_outcome(
    *,
    left_value: float,
    right_value: float,
    direction: MetricDirection,
) -> str:
    delta = left_value - right_value
    if abs(delta) <= PAIRWISE_TIE_EPSILON:
        return "tie"
    if direction == "higher":
        return "left" if delta > 0 else "right"
    return "left" if delta < 0 else "right"


def _decision_trace_payload(decision: PlannerDecision) -> dict[str, Any]:
    return decision.to_trace_payload()


def _planner_seed(*, bundle: ScenarioBundle, planner_key: str) -> int:
    token = stable_token(
        f"{planner_key}:{bundle.bundle_id}:{bundle.simulation_seed}",
        length=16,
    )
    return int(token, 16)


def _episode_fingerprint(replay_events: Sequence[ReplayEvent]) -> str:
    for event in replay_events:
        if event.event_type == "episode_started":
            return str(event.payload["episode_fingerprint"])
    raise ValueError("episode_started event missing from replay")


def _load_bundle(path: Path) -> ScenarioBundle:
    return ScenarioBundle.model_validate(load_json(path))


def _safe_filename(value: str) -> str:
    return value.replace(":", "--").replace("/", "--").replace("@", "--")


def _pretty_json(value: Any) -> str:
    return json.dumps(_json_compatible(value), indent=2, sort_keys=True) + "\n"


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_compatible(item) for item in value]
    return value


def _md_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
