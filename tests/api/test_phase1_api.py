from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from tests.benchmark.helpers import build_tiny_bundle

from orbital_shepherd_api.app import create_app
from orbital_shepherd_api.settings import ApiSettings
from orbital_shepherd_contracts import ScenarioBundle


def test_health_endpoint(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_scenario_registration_and_retrieval(tmp_path: Path) -> None:
    bundle = build_tiny_bundle()

    with _client(tmp_path) as client:
        register_response = client.post(
            "/v1/scenarios",
            json=bundle.model_dump(mode="json", exclude_none=True),
        )
        list_response = client.get("/v1/scenarios")
        get_response = client.get(f"/v1/scenarios/{bundle.bundle_id}")
        preview_response = client.get(f"/v1/scenarios/{bundle.bundle_id}/preview")

    assert register_response.status_code == 201
    assert register_response.json() == {"bundle_id": bundle.bundle_id}
    assert list_response.status_code == 200
    assert [item["bundle_id"] for item in list_response.json()] == [bundle.bundle_id]
    assert get_response.status_code == 200
    assert get_response.json()["bundle_id"] == bundle.bundle_id
    assert preview_response.status_code == 200
    assert preview_response.json()["counts"]["observation_opportunities"] == 1


def test_scenario_trajectory_czml_endpoint(tmp_path: Path) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "scenarios"
        / "sb--osbench-phase1-pack-v1--cloud_trap--seed-301.json"
    )
    bundle = ScenarioBundle.model_validate(json.loads(fixture_path.read_text(encoding="utf-8")))

    with _client(tmp_path) as client:
        _register_bundle(client, bundle)
        response = client.get(
            f"/v1/scenarios/{bundle.bundle_id}/trajectory-czml",
            params={"step_seconds": 300},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["id"] == "document"
    assert payload[0]["clock"]["interval"].startswith("2026-04-17T00:00:00Z/")
    satellite_packets = [item for item in payload[1:] if item["id"].startswith("sat:")]
    assert len(satellite_packets) == len(bundle.satellites)
    assert satellite_packets[0]["position"]["referenceFrame"] == "FIXED"
    assert len(satellite_packets[0]["position"]["cartographicDegrees"]) > 4


def test_episode_lifecycle_and_replay_metrics(tmp_path: Path) -> None:
    bundle = build_tiny_bundle()

    with _client(tmp_path) as client:
        _register_bundle(client, bundle)
        start_response = client.post(
            "/v1/episodes",
            json={
                "bundle_id": bundle.bundle_id,
                "planner_id": "planner:manual-ui",
                "simulation_seed": bundle.simulation_seed,
            },
        )
        episode_id = start_response.json()["episode_id"]

        detail_response = client.get(f"/v1/episodes/{episode_id}")
        first_action = _non_noop_action(detail_response.json()["action_mask"]["actions"])
        first_step = client.post(
            f"/v1/episodes/{episode_id}/step",
            json={"action": first_action},
        )

        second_detail = client.get(f"/v1/episodes/{episode_id}")
        second_action = _non_noop_action(second_detail.json()["action_mask"]["actions"])
        second_step = client.post(
            f"/v1/episodes/{episode_id}/step",
            json={"action": second_action},
        )

        final_step = client.post(
            f"/v1/episodes/{episode_id}/step",
            json={"action": {"action_type": "noop", "action_ref": "noop"}},
        )
        events_response = client.get(f"/v1/episodes/{episode_id}/events")
        metrics_response = client.get(f"/v1/episodes/{episode_id}/metrics")

    assert start_response.status_code == 201
    assert detail_response.status_code == 200
    assert first_step.status_code == 200
    assert first_step.json()["terminated"] is False
    assert first_step.json()["observation"]["incidents"][0]["status"] == "observed"
    assert second_step.status_code == 200
    assert second_step.json()["observation"]["incidents"][0]["status"] == "downlinked"
    assert final_step.status_code == 200
    assert final_step.json()["terminated"] is True

    assert events_response.status_code == 200
    assert events_response.headers["content-type"].startswith("application/x-ndjson")
    events = [
        json.loads(line)
        for line in events_response.text.splitlines()
        if line.strip()
    ]
    assert events[0]["episode_id"] == episode_id
    assert events[-1]["event_type"] == "episode_ended"

    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["useful_packet_count"] == 1
    assert metrics["mission_utility"] > 0


def test_baseline_run_orchestration(tmp_path: Path) -> None:
    bundle = build_tiny_bundle()

    with _client(tmp_path) as client:
        _register_bundle(client, bundle)
        run_response = client.post(
            "/v1/baselines/urgency_greedy/run",
            json={
                "bundle_id": bundle.bundle_id,
                "simulation_seed": bundle.simulation_seed,
            },
        )
        job_id = run_response.json()["job_id"]
        episode_id = run_response.json()["episode_id"]
        job_response = client.get(f"/v1/baseline-runs/{job_id}")
        episode_response = client.get(f"/v1/episodes/{episode_id}")
        metrics_response = client.get(f"/v1/episodes/{episode_id}/metrics")

    assert run_response.status_code == 202
    assert run_response.json()["status"] == "completed"
    assert episode_id is not None
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "completed"
    assert job_response.json()["episode_id"] == episode_id
    assert episode_response.status_code == 200
    assert episode_response.json()["terminated"] is True
    assert metrics_response.status_code == 200
    assert metrics_response.json()["mission_utility"] > 0


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        ApiSettings(
            scenario_dir=tmp_path / "scenarios",
            episode_dir=tmp_path / "episodes",
            baseline_run_dir=tmp_path / "baseline-runs",
        )
    )
    return TestClient(app)


def _register_bundle(client: TestClient, bundle: ScenarioBundle) -> None:
    response = client.post(
        "/v1/scenarios",
        json=bundle.model_dump(mode="json", exclude_none=True),
    )
    assert response.status_code == 201


def _non_noop_action(actions: list[dict[str, object]]) -> dict[str, object]:
    for action in actions:
        if action["action_type"] != "noop":
            return {
                "action_type": str(action["action_type"]),
                "action_ref": str(action["action_ref"]),
            }
    raise AssertionError("expected at least one non-noop action")
