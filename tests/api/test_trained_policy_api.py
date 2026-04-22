from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from tests.benchmark.helpers import build_tiny_bundle
from tests.trained_policy_helpers import export_toy_policy_checkpoint

from orbital_shepherd_api.app import create_app
from orbital_shepherd_api.settings import ApiSettings
from orbital_shepherd_contracts import ScenarioBundle
from orbital_shepherd_policy_models import PolicyCheckpointManifestRecord


def test_model_endpoints_execute_trained_checkpoint_and_expose_inference_traces(
    tmp_path: Path,
) -> None:
    bundle = build_tiny_bundle()
    manifest_path = export_toy_policy_checkpoint(tmp_path, run_id="bc:test-api-model-run:v1")
    manifest = PolicyCheckpointManifestRecord.load(manifest_path)

    with _client(tmp_path) as client:
        _register_bundle(client, bundle)
        models_response = client.get("/v1/models")
        detail_response = client.get(f"/v1/models/{manifest.checkpoint_id}")
        run_response = client.post(
            f"/v1/models/{manifest.checkpoint_id}/run",
            json={
                "bundle_id": bundle.bundle_id,
                "simulation_seed": bundle.simulation_seed,
                "include_inference_traces": True,
            },
        )

        job_id = run_response.json()["job_id"]
        episode_id = run_response.json()["episode_id"]
        model_run_response = client.get(f"/v1/model-runs/{job_id}")
        episode_response = client.get(f"/v1/episodes/{episode_id}")
        metrics_response = client.get(f"/v1/episodes/{episode_id}/metrics")
        traces_response = client.get(f"/v1/episodes/{episode_id}/inference-traces")

    assert models_response.status_code == 200
    assert [item["model_key"] for item in models_response.json()] == [manifest.checkpoint_id]
    assert detail_response.status_code == 200
    assert detail_response.json()["checkpoint_id"] == manifest.checkpoint_id
    assert run_response.status_code == 202
    assert run_response.json()["status"] == "completed"
    assert episode_id is not None
    assert model_run_response.status_code == 200
    assert model_run_response.json()["status"] == "completed"
    assert episode_response.status_code == 200
    assert episode_response.json()["planner_kind"] == "trained_policy"
    assert episode_response.json()["planner_metadata"]["checkpoint_id"] == manifest.checkpoint_id
    assert metrics_response.status_code == 200
    assert "mission_utility" in metrics_response.json()
    assert traces_response.status_code == 200
    traces = traces_response.json()
    assert traces["step_count"] >= 1
    assert traces["traces"][0]["planner_trace"]["planner_kind"] == "trained_policy"
    assert "selected_slot" in traces["traces"][0]["planner_trace"]
    assert "legal_action_count" in traces["traces"][0]["planner_trace"]


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        ApiSettings(
            scenario_dir=tmp_path / "scenarios",
            episode_dir=tmp_path / "episodes",
            baseline_run_dir=tmp_path / "baseline-runs",
            model_run_dir=tmp_path / "model-runs",
            training_manifest_dir=tmp_path / "manifests",
            training_checkpoint_dir=tmp_path / "checkpoints",
        )
    )
    return TestClient(app)


def _register_bundle(client: TestClient, bundle: ScenarioBundle) -> None:
    response = client.post(
        "/v1/scenarios",
        json=bundle.model_dump(mode="json", exclude_none=True),
    )
    assert response.status_code == 201
