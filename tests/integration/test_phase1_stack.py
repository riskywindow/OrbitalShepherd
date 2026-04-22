from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from orbital_shepherd_api.app import create_app
from orbital_shepherd_api.settings import ApiSettings
from orbital_shepherd_benchmark import BenchmarkConfig, BenchmarkRunResult, run_benchmark
from orbital_shepherd_scenario_engine import (
    ScenarioEngineConfig,
    build_scenario_pack,
    validate_scenario_pack,
)


def test_phase1_pack_is_demoable_and_deterministic(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    api_root = tmp_path / "api"
    scenario_config = ScenarioEngineConfig(scenario_dir=scenario_dir)

    built = build_scenario_pack(engine_config=scenario_config, output_dir=scenario_dir)
    validated = validate_scenario_pack(engine_config=scenario_config, input_dir=scenario_dir)

    assert len(built) == 10
    assert [record.bundle_id for record in built] == [record.bundle_id for record in validated]

    first_result = run_benchmark(
        config=BenchmarkConfig(
            benchmark_id=scenario_config.benchmark_id,
            scenario_dir=scenario_dir,
            report_dir=tmp_path / "reports-first",
            replay_dir=tmp_path / "replays-first",
        ),
        run_id="phase1-integration",
        scenario_families=("cloud_trap",),
        limit=2,
    )
    second_result = run_benchmark(
        config=BenchmarkConfig(
            benchmark_id=scenario_config.benchmark_id,
            scenario_dir=scenario_dir,
            report_dir=tmp_path / "reports-second",
            replay_dir=tmp_path / "replays-second",
        ),
        run_id="phase1-integration",
        scenario_families=("cloud_trap",),
        limit=2,
    )

    assert _normalized_run(first_result) == _normalized_run(second_result)
    _assert_same_text(
        first_result.report_dir / "planner_summary.md",
        second_result.report_dir / "planner_summary.md",
    )
    _assert_same_text(
        first_result.report_dir / "planner_summary.csv",
        second_result.report_dir / "planner_summary.csv",
    )
    assert _normalized_summary(first_result.report_dir / "summary.json") == _normalized_summary(
        second_result.report_dir / "summary.json"
    )
    for first_episode, second_episode in zip(
        first_result.episode_results,
        second_result.episode_results,
        strict=True,
    ):
        _assert_same_text(first_episode.artifacts.replay_path, second_episode.artifacts.replay_path)

    app = create_app(
        ApiSettings(
            scenario_dir=scenario_dir,
            episode_dir=api_root / "episodes",
            baseline_run_dir=api_root / "baseline-runs",
            demo_defaults_path=api_root / "demo" / "phase1-defaults.json",
        )
    )
    with TestClient(app) as client:
        scenario_response = client.get("/v1/scenarios")
        assert scenario_response.status_code == 200
        scenarios = scenario_response.json()
        assert len(scenarios) == 10

        demo_defaults_response = client.get("/v1/demo/defaults")
        assert demo_defaults_response.status_code == 200
        assert demo_defaults_response.json()["baseline_id"] == "urgency_greedy"

        selected_bundle = next(
            scenario for scenario in scenarios if scenario["scenario_family"] == "cloud_trap"
        )
        baseline_response = client.post(
            "/v1/baselines/urgency_greedy/run",
            json={
                "bundle_id": selected_bundle["bundle_id"],
                "simulation_seed": selected_bundle["simulation_seed"],
            },
        )
        assert baseline_response.status_code == 202
        baseline_payload = baseline_response.json()
        episode_id = str(baseline_payload["episode_id"])

        episode_response = client.get(f"/v1/episodes/{episode_id}")
        metrics_response = client.get(f"/v1/episodes/{episode_id}/metrics")
        events_response = client.get(f"/v1/episodes/{episode_id}/events")
        preview_response = client.get(f"/v1/scenarios/{selected_bundle['bundle_id']}/preview")

    assert episode_response.status_code == 200
    assert episode_response.json()["terminated"] is True
    assert metrics_response.status_code == 200
    assert metrics_response.json()["mission_utility"] != 0
    assert events_response.status_code == 200
    assert any(
        json.loads(line)["event_type"] == "episode_ended"
        for line in events_response.text.splitlines()
        if line.strip()
    )
    assert preview_response.status_code == 200
    assert preview_response.json()["counts"]["observation_opportunities"] > 0


def _normalized_run(result: BenchmarkRunResult) -> dict[str, object]:
    return {
        "run_id": result.run_id,
        "run_fingerprint": result.run_fingerprint,
        "planners": list(result.planners),
        "bundle_ids": list(result.bundle_ids),
        "planner_summaries": result.planner_summaries,
        "episodes": [
            {
                "planner_id": episode.planner_id,
                "bundle_id": episode.bundle_id,
                "episode_fingerprint": episode.episode_fingerprint,
                "replay_fingerprint": episode.replay_fingerprint,
                "metrics": episode.metrics.to_dict(),
            }
            for episode in result.episode_results
        ],
    }


def _assert_same_text(left: Path, right: Path) -> None:
    assert left.read_text(encoding="utf-8") == right.read_text(encoding="utf-8")


def _normalized_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["report_dir"] = ""
    payload["replay_dir"] = ""
    for episode in payload["episodes"]:
        episode["replay_path"] = ""
        episode["summary_path"] = ""
    return payload
