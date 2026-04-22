from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbital_shepherd_benchmark.config import BenchmarkConfig
from orbital_shepherd_benchmark.metrics import (
    EpisodeMetrics,
    aggregate_episode_metrics,
    compute_episode_metrics,
)
from orbital_shepherd_benchmark.planners import (
    PlannerEpisodeContext,
    build_planner,
    planner_runtime_metadata,
    planner_descriptions,
)
from orbital_shepherd_contracts import ReplayEvent, ScenarioBundle, load_json
from orbital_shepherd_core import (
    canonical_json_dumps,
    sha256_fingerprint,
    stable_id,
    stable_token,
)
from orbital_shepherd_env_runtime import EnvRuntimeConfig, OrbitalEnv, replay_events_to_ndjson


@dataclass(frozen=True, slots=True)
class EpisodeArtifactPaths:
    replay_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    planner_id: str
    planner_version: str
    bundle_id: str
    scenario_family: str
    episode_id: str
    episode_seed: int
    episode_fingerprint: str
    replay_fingerprint: str
    action_count: int
    metrics: EpisodeMetrics
    artifacts: EpisodeArtifactPaths
    planner_kind: str = "builtin"
    planner_artifact_id: str | None = None
    planner_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_id": self.planner_id,
            "planner_version": self.planner_version,
            "planner_kind": self.planner_kind,
            "planner_artifact_id": self.planner_artifact_id,
            "planner_metadata": self.planner_metadata,
            "bundle_id": self.bundle_id,
            "scenario_family": self.scenario_family,
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "episode_fingerprint": self.episode_fingerprint,
            "replay_fingerprint": self.replay_fingerprint,
            "action_count": self.action_count,
            "replay_path": str(self.artifacts.replay_path),
            "summary_path": str(self.artifacts.summary_path),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    run_id: str
    run_fingerprint: str
    planners: tuple[str, ...]
    bundle_ids: tuple[str, ...]
    report_dir: Path
    replay_dir: Path
    episode_results: tuple[EpisodeResult, ...]
    planner_summaries: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_fingerprint": self.run_fingerprint,
            "planners": list(self.planners),
            "bundle_ids": list(self.bundle_ids),
            "report_dir": str(self.report_dir),
            "replay_dir": str(self.replay_dir),
            "planner_summaries": self.planner_summaries,
            "episodes": [episode.to_dict() for episode in self.episode_results],
        }


def run_benchmark(
    *,
    config: BenchmarkConfig | None = None,
    planner_ids: Sequence[str] | None = None,
    scenario_paths: Sequence[Path] | None = None,
    scenario_families: Sequence[str] = (),
    bundle_ids: Sequence[str] = (),
    limit: int | None = None,
    run_id: str | None = None,
) -> BenchmarkRunResult:
    benchmark_config = config or BenchmarkConfig()
    selected_planner_ids = tuple(planner_ids or benchmark_config.planner_ids)
    bundles = load_scenario_bundles(
        scenario_dir=benchmark_config.scenario_dir,
        scenario_paths=scenario_paths,
        scenario_families=scenario_families,
        bundle_ids=bundle_ids,
        limit=limit,
    )
    selected_bundle_ids = tuple(bundle.bundle_id for bundle in bundles)
    run_fingerprint = sha256_fingerprint(
        canonical_json_dumps(
            {
                "benchmark_id": benchmark_config.benchmark_id,
                "planner_ids": list(selected_planner_ids),
                "bundle_ids": list(selected_bundle_ids),
                "urgent_incident_threshold": benchmark_config.urgent_incident_threshold,
            }
        )
    )
    resolved_run_id = run_id or stable_id(
        "benchrun",
        benchmark_config.benchmark_id,
        stable_token(run_fingerprint, length=10),
    )
    report_dir = benchmark_config.report_dir / resolved_run_id
    replay_dir = benchmark_config.replay_dir / resolved_run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)

    episode_results: list[EpisodeResult] = []
    for planner_id in selected_planner_ids:
        for bundle in bundles:
            episode_result = _run_episode(
                bundle=bundle,
                planner_id=planner_id,
                benchmark_config=benchmark_config,
                report_dir=report_dir,
                replay_dir=replay_dir,
            )
            episode_results.append(episode_result)

    planner_summaries = _build_planner_summaries(episode_results)
    result = BenchmarkRunResult(
        run_id=resolved_run_id,
        run_fingerprint=f"sha256:{run_fingerprint}",
        planners=selected_planner_ids,
        bundle_ids=selected_bundle_ids,
        report_dir=report_dir,
        replay_dir=replay_dir,
        episode_results=tuple(episode_results),
        planner_summaries=planner_summaries,
    )
    _write_run_reports(result, benchmark_config=benchmark_config)
    return result


def load_scenario_bundles(
    *,
    scenario_dir: Path,
    scenario_paths: Sequence[Path] | None = None,
    scenario_families: Sequence[str] = (),
    bundle_ids: Sequence[str] = (),
    limit: int | None = None,
) -> list[ScenarioBundle]:
    paths = (
        list(scenario_paths) if scenario_paths is not None else sorted(scenario_dir.glob("*.json"))
    )
    family_filter = set(scenario_families)
    bundle_filter = set(bundle_ids)
    bundles: list[ScenarioBundle] = []
    for path in sorted(paths):
        bundle = ScenarioBundle.model_validate(load_json(path))
        if family_filter and bundle.scenario_family not in family_filter:
            continue
        if bundle_filter and bundle.bundle_id not in bundle_filter:
            continue
        bundles.append(bundle)
        if limit is not None and len(bundles) >= limit:
            break
    if not bundles:
        raise ValueError("no scenario bundles matched the requested filters")
    return bundles


def _run_episode(
    *,
    bundle: ScenarioBundle,
    planner_id: str,
    benchmark_config: BenchmarkConfig,
    report_dir: Path,
    replay_dir: Path,
) -> EpisodeResult:
    planner = build_planner(planner_id)
    runtime_metadata = planner_runtime_metadata(planner)
    planner_seed = _planner_seed(bundle=bundle, planner_id=planner_id)
    env = OrbitalEnv(
        bundle,
        config=EnvRuntimeConfig(
            planner_id=f"planner:{planner.metadata.planner_id}",
            replay_dir=replay_dir,
            decision_interval_seconds=bundle.decision_interval_seconds,
            planner_metadata=runtime_metadata,
        ),
    )
    observation, _ = env.reset(
        seed=bundle.simulation_seed,
        planner_id=f"planner:{planner.metadata.planner_id}",
    )
    planner.start_episode(
        context=PlannerEpisodeContext(
            bundle=bundle,
            episode_id=str(observation["episode_id"]),
            episode_seed=int(observation["episode_seed"]),
            planner_seed=planner_seed,
        ),
        initial_observation=observation,
    )

    action_count = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        decision = planner.select_action(observation)
        observation, _, terminated, truncated, _ = env.step(
            decision.action,
            planner_trace=decision.to_trace_payload(),
        )
        action_count += 1

    replay_events = env.replay_events
    replay_path = (
        replay_dir
        / _safe_filename(planner.metadata.planner_id)
        / (f"{_safe_filename(bundle.bundle_id)}.ndjson")
    )
    summary_path = (
        report_dir
        / "episodes"
        / _safe_filename(planner.metadata.planner_id)
        / (f"{_safe_filename(bundle.bundle_id)}.json")
    )
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    replay_ndjson = replay_events_to_ndjson(replay_events)
    replay_path.write_text(f"{replay_ndjson}\n", encoding="utf-8")
    replay_fingerprint = f"sha256:{sha256_fingerprint(replay_ndjson)}"
    metrics = compute_episode_metrics(
        bundle=bundle,
        replay_events=replay_events,
        urgent_incident_threshold=benchmark_config.urgent_incident_threshold,
    )
    episode_fingerprint = _episode_fingerprint(replay_events)
    result = EpisodeResult(
        planner_id=planner.metadata.planner_id,
        planner_version=planner.metadata.version,
        planner_kind=str(runtime_metadata.get("planner_kind", "builtin")),
        planner_artifact_id=(
            str(runtime_metadata.get("checkpoint_id"))
            if runtime_metadata.get("checkpoint_id") is not None
            else f"planner:{planner.metadata.planner_id}"
        ),
        planner_metadata=runtime_metadata,
        bundle_id=bundle.bundle_id,
        scenario_family=bundle.scenario_family,
        episode_id=str(observation["episode_id"]),
        episode_seed=int(observation["episode_seed"]),
        episode_fingerprint=episode_fingerprint,
        replay_fingerprint=replay_fingerprint,
        action_count=action_count,
        metrics=metrics,
        artifacts=EpisodeArtifactPaths(replay_path=replay_path, summary_path=summary_path),
    )
    summary_path.write_text(_pretty_json(result.to_dict()), encoding="utf-8")
    return result


def _build_planner_summaries(episode_results: Sequence[EpisodeResult]) -> dict[str, dict[str, Any]]:
    metrics_by_planner: dict[str, list[EpisodeMetrics]] = defaultdict(list)
    families_by_planner: dict[str, set[str]] = defaultdict(set)
    for episode in episode_results:
        metrics_by_planner[episode.planner_id].append(episode.metrics)
        families_by_planner[episode.planner_id].add(episode.scenario_family)
    summaries: dict[str, dict[str, Any]] = {}
    for planner_id in sorted(metrics_by_planner):
        summary = aggregate_episode_metrics(metrics_by_planner[planner_id])
        summary["scenario_families"] = sorted(families_by_planner[planner_id])
        summary["description"] = planner_descriptions().get(planner_id, "")
        summaries[planner_id] = summary
    return summaries


def _write_run_reports(
    result: BenchmarkRunResult,
    *,
    benchmark_config: BenchmarkConfig,
) -> None:
    summary_path = result.report_dir / "summary.json"
    summary_path.write_text(_pretty_json(result.to_dict()), encoding="utf-8")
    if benchmark_config.write_csv:
        (result.report_dir / "planner_summary.csv").write_text(
            _planner_summary_csv(result.planner_summaries),
            encoding="utf-8",
        )
    if benchmark_config.write_markdown:
        (result.report_dir / "planner_summary.md").write_text(
            _planner_summary_markdown(result.planner_summaries),
            encoding="utf-8",
        )


def _planner_summary_csv(planner_summaries: dict[str, dict[str, Any]]) -> str:
    header = (
        "planner_id,episode_count,mission_utility_mean,useful_observation_value_captured_mean,"
        "cloud_waste_rate_mean,missed_urgent_incident_rate_mean,"
        "opportunity_utilization_efficiency_mean,ttfuo_mean_seconds,downlink_latency_mean_seconds"
    )
    rows = [header]
    for planner_id in sorted(planner_summaries):
        summary = planner_summaries[planner_id]
        rows.append(
            ",".join(
                [
                    planner_id,
                    str(summary["episode_count"]),
                    _csv_float(summary["mission_utility_mean"]),
                    _csv_float(summary["useful_observation_value_captured_mean"]),
                    _csv_float(summary["cloud_waste_rate_mean"]),
                    _csv_float(summary["missed_urgent_incident_rate_mean"]),
                    _csv_float(summary["opportunity_utilization_efficiency_mean"]),
                    _csv_float(summary["time_to_first_useful_observation_seconds"]["mean"]),
                    _csv_float(summary["downlink_latency_seconds"]["mean"]),
                ]
            )
        )
    return "\n".join(rows) + "\n"


def _planner_summary_markdown(planner_summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        (
            "| Planner | Episodes | Mission Utility | UOVC | Cloud Waste | Missed Urgent | "
            "OUE | TTFUO Mean (s) | Downlink Mean (s) |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for planner_id in sorted(planner_summaries):
        summary = planner_summaries[planner_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    planner_id,
                    str(summary["episode_count"]),
                    _md_float(summary["mission_utility_mean"]),
                    _md_float(summary["useful_observation_value_captured_mean"]),
                    _md_float(summary["cloud_waste_rate_mean"]),
                    _md_float(summary["missed_urgent_incident_rate_mean"]),
                    _md_float(summary["opportunity_utilization_efficiency_mean"]),
                    _md_float(summary["time_to_first_useful_observation_seconds"]["mean"]),
                    _md_float(summary["downlink_latency_seconds"]["mean"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _episode_fingerprint(replay_events: Sequence[ReplayEvent]) -> str:
    for event in replay_events:
        if event.event_type == "episode_started":
            return str(event.payload["episode_fingerprint"])
    raise ValueError("episode_started event missing from replay")


def _planner_seed(*, bundle: ScenarioBundle, planner_id: str) -> int:
    token = stable_token(f"{planner_id}:{bundle.bundle_id}:{bundle.simulation_seed}", length=16)
    return int(token, 16)


def _safe_filename(value: str) -> str:
    return value.replace(":", "--").replace("/", "--")


def _csv_float(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _md_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"
