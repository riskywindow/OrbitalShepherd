from __future__ import annotations

import json
from pathlib import Path

from orbital_shepherd_benchmark import (
    BenchmarkConfig,
    BenchmarkRunResult,
    compute_episode_metrics,
    run_benchmark,
)
from orbital_shepherd_env_runtime import EnvRuntimeConfig, OrbitalEnv
from tests.benchmark.helpers import build_tiny_bundle


def test_phase1_metrics_are_correct_on_toy_episode() -> None:
    bundle = build_tiny_bundle()
    env = OrbitalEnv(bundle, config=EnvRuntimeConfig(planner_id="planner:test-scripted"))
    env.reset(seed=bundle.simulation_seed, planner_id="planner:test-scripted")
    env.step(1)
    env.step(1)
    env.step(None)

    metrics = compute_episode_metrics(bundle=bundle, replay_events=env.replay_events)
    observation_event = next(
        event for event in env.replay_events if event.event_type == "observation_executed"
    )
    realized_quality = float(observation_event.payload["realized_quality"])
    expected_uovc = round(0.8 * 0.9 * (1.0 / (1.0 + ((1.0 / 60.0) / 6.0))) * realized_quality, 6)

    assert metrics.time_to_first_useful_observation_seconds.values == (120.0,)
    assert metrics.time_to_first_useful_observation_seconds.mean == 120.0
    assert metrics.downlink_latency_seconds.values == (60.0,)
    assert metrics.downlink_latency_seconds.mean == 60.0
    assert metrics.useful_observation_value_captured == expected_uovc
    assert metrics.cloud_waste_rate == 0.0
    assert metrics.missed_urgent_incident_rate == 0.0
    assert metrics.opportunity_utilization_efficiency == expected_uovc
    assert metrics.mission_utility == 2.082109


def test_benchmark_runner_is_deterministic_for_fixed_config(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    bundle = build_tiny_bundle()
    scenario_path = scenario_dir / "tiny-bundle.json"
    scenario_path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")

    first_result = run_benchmark(
        config=BenchmarkConfig(
            scenario_dir=scenario_dir,
            replay_dir=tmp_path / "replays-first",
            report_dir=tmp_path / "reports-first",
            planner_ids=(
                "random_valid_action",
                "urgency_greedy",
                "value_density_greedy",
                "ortools_receding_horizon",
            ),
        ),
        run_id="deterministic-phase1",
    )
    second_result = run_benchmark(
        config=BenchmarkConfig(
            scenario_dir=scenario_dir,
            replay_dir=tmp_path / "replays-second",
            report_dir=tmp_path / "reports-second",
            planner_ids=(
                "random_valid_action",
                "urgency_greedy",
                "value_density_greedy",
                "ortools_receding_horizon",
            ),
        ),
        run_id="deterministic-phase1",
    )

    assert _normalized_run(first_result) == _normalized_run(second_result)
    assert (
        first_result.report_dir / "planner_summary.md"
    ).read_text(encoding="utf-8") == (
        second_result.report_dir / "planner_summary.md"
    ).read_text(encoding="utf-8")
    assert (
        first_result.report_dir / "planner_summary.csv"
    ).read_text(encoding="utf-8") == (
        second_result.report_dir / "planner_summary.csv"
    ).read_text(encoding="utf-8")

    for first_episode, second_episode in zip(
        first_result.episode_results, second_result.episode_results, strict=True
    ):
        assert first_episode.artifacts.replay_path.read_text(
            encoding="utf-8"
        ) == second_episode.artifacts.replay_path.read_text(encoding="utf-8")


def _normalized_run(result: BenchmarkRunResult) -> dict[str, object]:
    episode_results = result.episode_results
    planner_summaries = result.planner_summaries
    return {
        "run_id": result.run_id,
        "run_fingerprint": result.run_fingerprint,
        "planners": list(result.planners),
        "bundle_ids": list(result.bundle_ids),
        "planner_summaries": planner_summaries,
        "episodes": [
            {
                "planner_id": episode.planner_id,
                "bundle_id": episode.bundle_id,
                "scenario_family": episode.scenario_family,
                "episode_id": episode.episode_id,
                "episode_seed": episode.episode_seed,
                "episode_fingerprint": episode.episode_fingerprint,
                "replay_fingerprint": episode.replay_fingerprint,
                "action_count": episode.action_count,
                "metrics": episode.metrics.to_dict(),
            }
            for episode in episode_results
        ],
    }
