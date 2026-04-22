from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from tests.benchmark.helpers import build_tiny_bundle

from orbital_shepherd_contracts import ScenarioBundle, compile_scenario_bundle
from orbital_shepherd_contracts.models import (
    GroundStation,
    Satellite,
    ScenarioConfig,
    ScenarioManifest,
    TargetCell,
    TimeWindow,
)
from orbital_shepherd_env_runtime import EnvRuntimeConfig, OrbitalEnv
from orbital_shepherd_training import (
    CANDIDATE_FEATURE_SPECS,
    GLOBAL_FEATURE_SPECS,
    FlattenedOrbitalTrainingEnv,
    OrbitalTrainingEnv,
)


def test_orbital_training_env_reset_shapes_and_mask() -> None:
    env = OrbitalTrainingEnv(build_tiny_bundle(), top_k=4)

    observation, info = env.reset(seed=17)

    assert set(observation) == {"global_features", "candidate_features", "action_mask"}
    assert len(observation["global_features"]) == len(GLOBAL_FEATURE_SPECS)
    assert len(observation["candidate_features"]) == 4
    assert len(observation["candidate_features"][0]) == len(CANDIDATE_FEATURE_SPECS)
    assert tuple(observation["action_mask"]) == (1, 1, 0, 0, 0)
    assert env.observation_space["candidate_features"].shape == (
        4,
        len(CANDIDATE_FEATURE_SPECS),
    )
    assert info["projected_candidate_count"] == 1
    assert info["candidate_count"] == 1
    assert info["truncated_candidate_count"] == 0
    assert info["slot_mapping"][1]["action_type"] == "schedule_observation"
    assert info["slot_mapping"][1]["action_ref"] == "opp:test-001"
    assert env.normalization_metadata()["top_k"] == 4


def test_orbital_training_env_rejects_padded_slots() -> None:
    env = OrbitalTrainingEnv(build_tiny_bundle(), top_k=4)
    env.reset(seed=17)

    with pytest.raises(ValueError, match="illegal training slot selected"):
        env.step(4)


def test_orbital_training_env_projection_is_deterministic_with_stable_ties() -> None:
    bundle = _build_projection_bundle(candidate_count=5)
    first_env = OrbitalTrainingEnv(bundle, top_k=3)
    second_env = OrbitalTrainingEnv(bundle, top_k=3)

    first_observation, first_info = first_env.reset(seed=31)
    second_observation, second_info = second_env.reset(seed=31)

    assert tuple(first_observation["action_mask"]) == tuple(second_observation["action_mask"])
    assert [slot["action_ref"] for slot in first_info["slot_mapping"][1:4]] == [
        "opp:proj-000",
        "opp:proj-001",
        "opp:proj-002",
    ]
    assert first_info["slot_mapping"] == second_info["slot_mapping"]
    assert first_info["truncated_candidate_count"] == 2

    first_observation, _, _, _, first_info = first_env.step(0)
    second_observation, _, _, _, second_info = second_env.step(0)

    assert tuple(first_observation["action_mask"]) == tuple(second_observation["action_mask"])
    assert first_info["slot_mapping"] == second_info["slot_mapping"]


def test_training_slot_maps_to_equivalent_runtime_action() -> None:
    bundle = build_tiny_bundle()
    training_env = OrbitalTrainingEnv(bundle, top_k=4)
    runtime_env = OrbitalEnv(
        bundle,
        config=EnvRuntimeConfig(
            planner_id="planner:test-runtime",
            env_id="env:test-phase1",
            reward_actor_id="reward:test-phase1",
            metrics_actor_id="metrics:test-phase1",
            packetizer_actor_id="packetizer:test-phase1",
        ),
    )

    training_env.reset(seed=17)
    runtime_env.reset(seed=17)
    mapped_action = training_env.runtime_action_for_slot(1)

    assert mapped_action is not None
    assert training_env.decode_action_slot(1)["action_id"] == mapped_action.action_id

    _, reward_train, terminated_train, truncated_train, info_train = training_env.step(1)
    runtime_observation, reward_runtime, terminated_runtime, truncated_runtime, _ = (
        runtime_env.step(mapped_action)
    )

    assert reward_train == reward_runtime
    assert terminated_train is terminated_runtime
    assert truncated_train is truncated_runtime
    assert info_train["selected_action_id"] == mapped_action.action_id
    assert training_env.runtime_env.state.to_observation(
        bundle=training_env.bundle,
        action_mask=training_env.runtime_env.action_mask,
    ) == runtime_observation


def test_flattened_training_wrapper_preserves_mask_and_fixed_size_features() -> None:
    env = FlattenedOrbitalTrainingEnv(OrbitalTrainingEnv(build_tiny_bundle(), top_k=4))

    observation, _ = env.reset(seed=17)

    assert set(observation) == {"flat_features", "action_mask"}
    assert len(observation["flat_features"]) == len(GLOBAL_FEATURE_SPECS) + (
        4 * len(CANDIDATE_FEATURE_SPECS)
    )
    assert tuple(observation["action_mask"]) == (1, 1, 0, 0, 0)
    assert len(env.flat_feature_names) == len(observation["flat_features"])


def _build_projection_bundle(*, candidate_count: int) -> ScenarioBundle:
    start_time = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    manifest = ScenarioManifest(
        manifest_id="sm:phase2:projection-ties:seed-31",
        benchmark_id="osbench-phase2",
        scenario_family="projection_ties",
        simulation_seed=31,
        decision_interval_seconds=60,
        time_window=TimeWindow(
            start_time_utc=start_time,
            end_time_utc=start_time + timedelta(minutes=4),
        ),
        satellites=[
            Satellite.model_validate(
                {
                    "schema_version": "1.0.0",
                    "satellite_id": "sat:proj-01",
                    "name": "Projection Sat 01",
                    "norad_catalog_id": 620901,
                    "sensor": {
                        "sensor_id": "sensor:proj-optical-01",
                        "swath_km": 100.0,
                        "quality_nominal": 0.95,
                        "max_off_nadir_deg": 30.0,
                        "estimated_data_volume_mb": 50.0,
                    },
                    "downlink": {
                        "buffer_capacity_mb": 1000.0,
                        "nominal_downlink_rate_mbps": 300.0,
                    },
                    "constraints": {
                        "max_retargets_per_orbit": 4,
                        "availability": "nominal",
                    },
                }
            )
        ],
        ground_stations=[
            GroundStation.model_validate(
                {
                    "schema_version": "1.0.0",
                    "station_id": "gs:proj-01",
                    "name": "Projection Ground 01",
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
                    "target_cell_id": "tc:proj-001",
                    "h3_cell": "8928308280fffff",
                    "centroid": {"lat": 34.2523, "lon": -118.5134},
                    "region_name": "Projection Range",
                    "static_value": 0.7,
                    "priority_class": "critical",
                }
            )
        ],
        incidents=[],
        config=ScenarioConfig.model_validate(
            {
                "horizon_hours": 1,
                "notes": "Projection tie-break scenario.",
                "weather_model": "fixture:test-v1",
                "opportunity_generation": {
                    "quality_threshold": 0.6,
                    "cloud_block_threshold": 0.5,
                },
            }
        ),
    )
    observation_opportunities = []
    for index in range(candidate_count):
        observation_opportunities.append(
            {
                "schema_version": "1.0.0",
                "opportunity_id": f"opp:proj-{index:03d}",
                "satellite_id": "sat:proj-01",
                "target_cell_id": "tc:proj-001",
                "start_time_utc": start_time,
                "end_time_utc": start_time + timedelta(minutes=2),
                "predicted_quality_mean": 0.9,
                "predicted_cloud_obstruction_prob": 0.2,
                "estimated_data_volume_mb": 50.0,
                "slew_cost": 0.1,
                "incident_ids": [],
            }
        )
    return compile_scenario_bundle(
        manifest,
        compiled_at=start_time,
        observation_opportunities=observation_opportunities,
        downlink_windows=[],
    )
