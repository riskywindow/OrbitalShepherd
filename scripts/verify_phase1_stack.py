from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from _bootstrap import REPO_ROOT, install_repo_sources

DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "demo" / "phase1-verification.json"


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    install_repo_sources()
    from fastapi.testclient import TestClient

    from orbital_shepherd_api.app import create_app
    from orbital_shepherd_api.settings import ApiSettings
    from orbital_shepherd_benchmark import BenchmarkConfig, run_benchmark
    from orbital_shepherd_core import canonical_json_dumps
    from orbital_shepherd_scenario_engine import (
        ScenarioEngineConfig,
        build_scenario_pack,
        validate_scenario_pack,
    )

    temp_root = Path(tempfile.mkdtemp(prefix="phase1-verify-"))
    scenario_dir = temp_root / "scenarios"
    api_root = temp_root / "api"
    first_report_dir = temp_root / "reports-first"
    first_replay_dir = temp_root / "replays-first"
    second_report_dir = temp_root / "reports-second"
    second_replay_dir = temp_root / "replays-second"

    scenario_config = ScenarioEngineConfig(scenario_dir=scenario_dir)
    built = build_scenario_pack(engine_config=scenario_config, output_dir=scenario_dir)
    validated = validate_scenario_pack(engine_config=scenario_config, input_dir=scenario_dir)

    first_result = run_benchmark(
        config=BenchmarkConfig(
            benchmark_id=scenario_config.benchmark_id,
            scenario_dir=scenario_dir,
            report_dir=first_report_dir,
            replay_dir=first_replay_dir,
        ),
        run_id="phase1-verify",
        scenario_families=("cloud_trap",),
        limit=2,
    )
    second_result = run_benchmark(
        config=BenchmarkConfig(
            benchmark_id=scenario_config.benchmark_id,
            scenario_dir=scenario_dir,
            report_dir=second_report_dir,
            replay_dir=second_replay_dir,
        ),
        run_id="phase1-verify",
        scenario_families=("cloud_trap",),
        limit=2,
    )

    if _normalized_summary(first_report_dir / "phase1-verify" / "summary.json") != _normalized_summary(
        second_report_dir / "phase1-verify" / "summary.json"
    ):
        raise AssertionError("benchmark summary payload drift detected")
    _assert_identical_bytes(
        first_report_dir / "phase1-verify" / "planner_summary.md",
        second_report_dir / "phase1-verify" / "planner_summary.md",
    )
    _assert_identical_bytes(
        first_report_dir / "phase1-verify" / "planner_summary.csv",
        second_report_dir / "phase1-verify" / "planner_summary.csv",
    )
    for first_episode, second_episode in zip(
        first_result.episode_results,
        second_result.episode_results,
        strict=True,
    ):
        _assert_identical_bytes(
            first_episode.artifacts.replay_path,
            second_episode.artifacts.replay_path,
        )

    app = create_app(
        ApiSettings(
            scenario_dir=scenario_dir,
            episode_dir=api_root / "episodes",
            baseline_run_dir=api_root / "baseline-runs",
            demo_defaults_path=api_root / "demo" / "phase1-defaults.json",
        )
    )
    with TestClient(app) as client:
        scenarios_response = client.get("/v1/scenarios")
        demo_response = client.get("/v1/demo/defaults")
        scenarios_payload = scenarios_response.json()
        bundle_id = str(scenarios_payload[0]["bundle_id"])
        seed = int(scenarios_payload[0]["simulation_seed"])
        baseline_response = client.post(
            "/v1/baselines/urgency_greedy/run",
            json={"bundle_id": bundle_id, "simulation_seed": seed},
        )
        baseline_payload = baseline_response.json()
        episode_id = str(baseline_payload["episode_id"])
        episode_response = client.get(f"/v1/episodes/{episode_id}")
        events_response = client.get(f"/v1/episodes/{episode_id}/events")
        metrics_response = client.get(f"/v1/episodes/{episode_id}/metrics")

    verification_payload = {
        "built_bundle_count": len(built),
        "validated_bundle_count": len(validated),
        "bundle_ids": [record.bundle_id for record in built],
        "deterministic_run_fingerprint": first_result.run_fingerprint,
        "deterministic_episode_fingerprints": [
            episode.episode_fingerprint for episode in first_result.episode_results
        ],
        "api": {
            "scenario_count": len(scenarios_payload),
            "demo_defaults": demo_response.json(),
            "episode_id": episode_id,
            "episode_status": episode_response.json()["terminated"],
            "replay_event_count": len([line for line in events_response.text.splitlines() if line]),
            "mission_utility": metrics_response.json()["mission_utility"],
        },
    }
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(
        canonical_json_dumps(verification_payload) + "\n",
        encoding="utf-8",
    )
    print(DEFAULT_OUTPUT_PATH)
    print(json.dumps(verification_payload["api"], sort_keys=True))
    return 0


def _assert_identical_bytes(left: Path, right: Path) -> None:
    if left.read_text(encoding="utf-8") != right.read_text(encoding="utf-8"):
        raise AssertionError(f"artifact drift detected between {left} and {right}")


def _normalized_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["report_dir"] = ""
    payload["replay_dir"] = ""
    for episode in payload["episodes"]:
        episode["replay_path"] = ""
        episode["summary_path"] = ""
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
