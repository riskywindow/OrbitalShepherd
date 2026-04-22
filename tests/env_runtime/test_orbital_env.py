from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from orbital_shepherd_contracts import compile_scenario_bundle, validate_canonical_replay_events
from orbital_shepherd_contracts.models import (
    GroundStation,
    Incident,
    Satellite,
    ScenarioBundle,
    ScenarioConfig,
    ScenarioManifest,
    TargetCell,
    TimeWindow,
)
from orbital_shepherd_env_runtime import (
    CANONICAL_ENV_EVENT_NAMES,
    LEGACY_ENV_EVENT_ALIASES,
    EnvRuntimeConfig,
    InMemoryReplaySink,
    NdjsonReplayWriter,
    OrbitalEnv,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "env_runtime"


def test_orbital_env_reset_and_step_flow() -> None:
    bundle = _build_tiny_bundle()
    sink = InMemoryReplaySink()
    env = OrbitalEnv(
        bundle,
        config=EnvRuntimeConfig(
            planner_id="planner:test-greedy",
            env_id="env:test-phase1",
            reward_actor_id="reward:test-phase1",
            metrics_actor_id="metrics:test-phase1",
            packetizer_actor_id="packetizer:test-phase1",
        ),
        replay_sinks=[sink],
    )

    observation, info = env.reset(seed=17)

    assert observation["sim_tick"] == 0
    assert observation["incidents"][0]["status"] == "unseen"
    assert [action["action_type"] for action in info["action_mask"]["actions"]] == [
        "noop",
        "schedule_observation",
    ]
    assert [event.event_type for event in sink.events[:4]] == [
        "scenario_bundle_loaded",
        "episode_started",
        "candidate_set_materialized",
        "action_mask_emitted",
    ]

    observation, reward, terminated, truncated, info = env.step(1)

    assert reward > 0
    assert not terminated
    assert not truncated
    assert observation["incidents"][0]["status"] == "observed"
    assert len(observation["onboard_queue"]) == 1
    assert [action["action_type"] for action in info["action_mask"]["actions"]] == [
        "noop",
        "schedule_downlink",
    ]

    observation, reward, terminated, truncated, info = env.step(1)

    assert reward > 0
    assert not terminated
    assert not truncated
    assert observation["incidents"][0]["status"] == "downlinked"
    assert observation["onboard_queue"] == []
    assert [action["action_type"] for action in info["action_mask"]["actions"]] == ["noop"]

    observation, reward, terminated, truncated, info = env.step(None)

    assert reward <= 0
    assert terminated
    assert not truncated
    assert observation["terminated"] is True
    assert observation["mission_utility"] > 0
    assert info["events"][-1]["event_type"] == "episode_ended"


def test_orbital_env_replay_is_deterministic_and_validated() -> None:
    first_bundle = _build_tiny_bundle()
    second_bundle = _build_tiny_bundle()

    first_replay = _run_episode(first_bundle, seed=17)
    second_replay = _run_episode(second_bundle, seed=17)

    assert first_replay == second_replay
    validate_canonical_replay_events(first_replay)


def test_orbital_env_writes_golden_replay_sequence(tmp_path: Path) -> None:
    bundle = _build_tiny_bundle()
    output_path = tmp_path / "episode.ndjson"
    writer = NdjsonReplayWriter(output_path)
    env = OrbitalEnv(
        bundle,
        config=EnvRuntimeConfig(
            planner_id="planner:test-greedy",
            env_id="env:test-phase1",
            reward_actor_id="reward:test-phase1",
            metrics_actor_id="metrics:test-phase1",
            packetizer_actor_id="packetizer:test-phase1",
        ),
        replay_sinks=[writer],
    )

    env.reset(seed=17)
    env.step(1)
    env.step(1)
    env.step(None)

    golden_path = FIXTURES_DIR / "phase1_golden_replay.ndjson"
    assert output_path.read_text(encoding="utf-8") == golden_path.read_text(encoding="utf-8")


def test_legacy_event_names_map_to_canonical_runtime_events() -> None:
    assert CANONICAL_ENV_EVENT_NAMES["scenario_loaded"] == "scenario_bundle_loaded"
    assert CANONICAL_ENV_EVENT_NAMES["opportunities_materialized"] == "candidate_set_materialized"
    assert LEGACY_ENV_EVENT_ALIASES["observation_committed"] == "observation_executed"
    assert LEGACY_ENV_EVENT_ALIASES["downlink_committed"] == "downlink_executed"


def _run_episode(bundle: ScenarioBundle, *, seed: int) -> list[dict[str, object]]:
    env = OrbitalEnv(
        bundle,
        config=EnvRuntimeConfig(
            planner_id="planner:test-greedy",
            env_id="env:test-phase1",
            reward_actor_id="reward:test-phase1",
            metrics_actor_id="metrics:test-phase1",
            packetizer_actor_id="packetizer:test-phase1",
        ),
    )
    env.reset(seed=seed)
    env.step(1)
    env.step(1)
    env.step(None)
    return [event.model_dump(mode="json", exclude_none=True) for event in env.replay_events]


def _build_tiny_bundle() -> ScenarioBundle:
    start_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    manifest = ScenarioManifest(
        manifest_id="sm:phase1:tiny-runtime:seed-17",
        benchmark_id="osbench-phase1",
        scenario_family="cloud_trap",
        simulation_seed=17,
        decision_interval_seconds=60,
        time_window=TimeWindow(
            start_time_utc=start_time,
            end_time_utc=start_time + timedelta(minutes=3),
        ),
        satellites=[
            Satellite.model_validate(
                {
                    "schema_version": "1.0.0",
                    "satellite_id": "sat:test-01",
                    "name": "Test Sat 01",
                    "norad_catalog_id": 620001,
                    "sensor": {
                        "sensor_id": "sensor:test-optical-01",
                        "swath_km": 100.0,
                        "quality_nominal": 0.95,
                        "max_off_nadir_deg": 30.0,
                        "estimated_data_volume_mb": 100.0,
                    },
                    "downlink": {
                        "buffer_capacity_mb": 500.0,
                        "nominal_downlink_rate_mbps": 300.0,
                    },
                    "constraints": {
                        "max_retargets_per_orbit": 2,
                        "availability": "nominal",
                    },
                }
            )
        ],
        ground_stations=[
            GroundStation.model_validate(
                {
                    "schema_version": "1.0.0",
                    "station_id": "gs:test-01",
                    "name": "Test Ground 01",
                    "location": {"lat": 64.8, "lon": -147.7, "alt_m": 136.0},
                    "capabilities": {
                        "max_concurrent_contacts": 1,
                        "downlink_rate_mbps": 500.0,
                        "availability": "nominal",
                    },
                }
            )
        ],
        target_cells=[
            TargetCell.model_validate(
                {
                    "schema_version": "1.0.0",
                    "target_cell_id": "tc:test-001",
                    "h3_cell": "8928308280fffff",
                    "centroid": {"lat": 34.2523, "lon": -118.5134},
                    "region_name": "Southern California",
                    "static_value": 0.8,
                    "priority_class": "critical",
                }
            )
        ],
        incidents=[
            Incident.model_validate(
                {
                    "schema_version": "1.0.0",
                    "incident_id": "inc:test-001",
                    "incident_type": "wildfire",
                    "target_cell_id": "tc:test-001",
                    "ignition_time_utc": start_time,
                    "urgency_score": 0.9,
                    "confidence": 0.85,
                    "state": "active",
                    "estimated_area_ha": 42.0,
                }
            )
        ],
        config=ScenarioConfig.model_validate(
            {
                "horizon_hours": 1,
                "notes": "Tiny deterministic env runtime scenario.",
                "weather_model": "fixture:test-v1",
                "opportunity_generation": {
                    "quality_threshold": 0.6,
                    "cloud_block_threshold": 0.5,
                },
            }
        ),
    )
    return compile_scenario_bundle(
        manifest,
        compiled_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        observation_opportunities=[
            {
                "schema_version": "1.0.0",
                "opportunity_id": "opp:test-001",
                "satellite_id": "sat:test-01",
                "target_cell_id": "tc:test-001",
                "start_time_utc": start_time,
                "end_time_utc": start_time + timedelta(minutes=1),
                "predicted_quality_mean": 0.95,
                "predicted_cloud_obstruction_prob": 0.05,
                "estimated_data_volume_mb": 100.0,
                "slew_cost": 0.1,
                "incident_ids": ["inc:test-001"],
            }
        ],
        downlink_windows=[
            {
                "schema_version": "1.0.0",
                "window_id": "dw:test-001",
                "satellite_id": "sat:test-01",
                "station_id": "gs:test-01",
                "start_time_utc": start_time + timedelta(minutes=1),
                "end_time_utc": start_time + timedelta(minutes=2),
                "max_volume_mb": 120.0,
                "expected_rate_mbps": 100.0,
                "outage_risk": 0.0,
            }
        ],
    )
