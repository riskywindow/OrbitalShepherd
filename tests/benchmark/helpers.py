from __future__ import annotations

from datetime import UTC, datetime, timedelta

from orbital_shepherd_contracts import ScenarioBundle, compile_scenario_bundle
from orbital_shepherd_contracts.models import (
    GroundStation,
    Incident,
    Satellite,
    ScenarioConfig,
    ScenarioManifest,
    TargetCell,
    TimeWindow,
)


def build_tiny_bundle() -> ScenarioBundle:
    start_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    manifest = ScenarioManifest(
        manifest_id="sm:phase1:tiny-benchmark:seed-17",
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
                "notes": "Tiny deterministic benchmark scenario.",
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


def build_value_density_choice_bundle() -> ScenarioBundle:
    start_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    manifest = ScenarioManifest(
        manifest_id="sm:phase2:value-density-choice:seed-19",
        benchmark_id="osbench-phase2",
        scenario_family="planner_toy",
        simulation_seed=19,
        decision_interval_seconds=60,
        time_window=TimeWindow(
            start_time_utc=start_time,
            end_time_utc=start_time + timedelta(minutes=3),
        ),
        satellites=[
            Satellite.model_validate(
                {
                    "schema_version": "1.0.0",
                    "satellite_id": "sat:choice-01",
                    "name": "Choice Sat 01",
                    "norad_catalog_id": 620011,
                    "sensor": {
                        "sensor_id": "sensor:choice-optical-01",
                        "swath_km": 100.0,
                        "quality_nominal": 0.95,
                        "max_off_nadir_deg": 30.0,
                        "estimated_data_volume_mb": 80.0,
                    },
                    "downlink": {
                        "buffer_capacity_mb": 400.0,
                        "nominal_downlink_rate_mbps": 250.0,
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
                    "station_id": "gs:choice-01",
                    "name": "Choice Ground 01",
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
                    "target_cell_id": "tc:choice-high",
                    "h3_cell": "8928308280fffff",
                    "centroid": {"lat": 34.2523, "lon": -118.5134},
                    "region_name": "High Value Region",
                    "static_value": 1.0,
                    "priority_class": "critical",
                }
            ),
            TargetCell.model_validate(
                {
                    "schema_version": "1.0.0",
                    "target_cell_id": "tc:choice-low",
                    "h3_cell": "8928308281fffff",
                    "centroid": {"lat": 34.4523, "lon": -118.3134},
                    "region_name": "Lower Value Region",
                    "static_value": 0.55,
                    "priority_class": "medium",
                }
            ),
        ],
        incidents=[
            Incident.model_validate(
                {
                    "schema_version": "1.0.0",
                    "incident_id": "inc:choice-high",
                    "incident_type": "wildfire",
                    "target_cell_id": "tc:choice-high",
                    "ignition_time_utc": start_time,
                    "urgency_score": 0.95,
                    "confidence": 0.9,
                    "state": "active",
                    "estimated_area_ha": 64.0,
                }
            ),
            Incident.model_validate(
                {
                    "schema_version": "1.0.0",
                    "incident_id": "inc:choice-low",
                    "incident_type": "wildfire",
                    "target_cell_id": "tc:choice-low",
                    "ignition_time_utc": start_time,
                    "urgency_score": 0.55,
                    "confidence": 0.9,
                    "state": "active",
                    "estimated_area_ha": 21.0,
                }
            ),
        ],
        config=ScenarioConfig.model_validate(
            {
                "horizon_hours": 1,
                "notes": "Two simultaneous observations where one is obviously better.",
                "weather_model": "fixture:planner-toy-v1",
                "opportunity_generation": {
                    "quality_threshold": 0.6,
                    "cloud_block_threshold": 0.5,
                },
            }
        ),
    )
    return compile_scenario_bundle(
        manifest,
        compiled_at=start_time,
        observation_opportunities=[
            {
                "schema_version": "1.0.0",
                "opportunity_id": "opp:choice-high",
                "satellite_id": "sat:choice-01",
                "target_cell_id": "tc:choice-high",
                "start_time_utc": start_time,
                "end_time_utc": start_time + timedelta(minutes=1),
                "predicted_quality_mean": 0.94,
                "predicted_cloud_obstruction_prob": 0.04,
                "estimated_data_volume_mb": 70.0,
                "slew_cost": 0.05,
                "incident_ids": ["inc:choice-high"],
            },
            {
                "schema_version": "1.0.0",
                "opportunity_id": "opp:choice-low",
                "satellite_id": "sat:choice-01",
                "target_cell_id": "tc:choice-low",
                "start_time_utc": start_time,
                "end_time_utc": start_time + timedelta(minutes=1),
                "predicted_quality_mean": 0.8,
                "predicted_cloud_obstruction_prob": 0.35,
                "estimated_data_volume_mb": 70.0,
                "slew_cost": 0.12,
                "incident_ids": ["inc:choice-low"],
            },
        ],
        downlink_windows=[
            {
                "schema_version": "1.0.0",
                "window_id": "dw:choice-01",
                "satellite_id": "sat:choice-01",
                "station_id": "gs:choice-01",
                "start_time_utc": start_time + timedelta(minutes=1),
                "end_time_utc": start_time + timedelta(minutes=2),
                "max_volume_mb": 120.0,
                "expected_rate_mbps": 100.0,
                "outage_risk": 0.0,
            }
        ],
    )


def build_lookahead_tradeoff_bundle() -> ScenarioBundle:
    start_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    manifest = ScenarioManifest(
        manifest_id="sm:phase2:lookahead-tradeoff:seed-23",
        benchmark_id="osbench-phase2",
        scenario_family="planner_toy",
        simulation_seed=23,
        decision_interval_seconds=60,
        time_window=TimeWindow(
            start_time_utc=start_time,
            end_time_utc=start_time + timedelta(minutes=4),
        ),
        satellites=[
            Satellite.model_validate(
                {
                    "schema_version": "1.0.0",
                    "satellite_id": "sat:lookahead-a",
                    "name": "Lookahead Sat A",
                    "norad_catalog_id": 620021,
                    "sensor": {
                        "sensor_id": "sensor:lookahead-a",
                        "swath_km": 100.0,
                        "quality_nominal": 0.95,
                        "max_off_nadir_deg": 30.0,
                        "estimated_data_volume_mb": 80.0,
                    },
                    "downlink": {
                        "buffer_capacity_mb": 400.0,
                        "nominal_downlink_rate_mbps": 250.0,
                    },
                    "constraints": {
                        "max_retargets_per_orbit": 2,
                        "availability": "nominal",
                    },
                }
            ),
            Satellite.model_validate(
                {
                    "schema_version": "1.0.0",
                    "satellite_id": "sat:lookahead-b",
                    "name": "Lookahead Sat B",
                    "norad_catalog_id": 620022,
                    "sensor": {
                        "sensor_id": "sensor:lookahead-b",
                        "swath_km": 100.0,
                        "quality_nominal": 0.95,
                        "max_off_nadir_deg": 30.0,
                        "estimated_data_volume_mb": 80.0,
                    },
                    "downlink": {
                        "buffer_capacity_mb": 400.0,
                        "nominal_downlink_rate_mbps": 250.0,
                    },
                    "constraints": {
                        "max_retargets_per_orbit": 2,
                        "availability": "nominal",
                    },
                }
            ),
        ],
        ground_stations=[
            GroundStation.model_validate(
                {
                    "schema_version": "1.0.0",
                    "station_id": "gs:lookahead-01",
                    "name": "Lookahead Ground 01",
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
                    "target_cell_id": "tc:lookahead-a",
                    "h3_cell": "8928308282fffff",
                    "centroid": {"lat": 35.0, "lon": -118.0},
                    "region_name": "Undeliverable High Value",
                    "static_value": 1.0,
                    "priority_class": "critical",
                }
            ),
            TargetCell.model_validate(
                {
                    "schema_version": "1.0.0",
                    "target_cell_id": "tc:lookahead-b",
                    "h3_cell": "8928308283fffff",
                    "centroid": {"lat": 35.2, "lon": -118.2},
                    "region_name": "Deliverable Slightly Lower Value",
                    "static_value": 0.85,
                    "priority_class": "high",
                }
            ),
        ],
        incidents=[
            Incident.model_validate(
                {
                    "schema_version": "1.0.0",
                    "incident_id": "inc:lookahead-a",
                    "incident_type": "wildfire",
                    "target_cell_id": "tc:lookahead-a",
                    "ignition_time_utc": start_time,
                    "urgency_score": 0.98,
                    "confidence": 0.9,
                    "state": "active",
                    "estimated_area_ha": 77.0,
                }
            ),
            Incident.model_validate(
                {
                    "schema_version": "1.0.0",
                    "incident_id": "inc:lookahead-b",
                    "incident_type": "wildfire",
                    "target_cell_id": "tc:lookahead-b",
                    "ignition_time_utc": start_time,
                    "urgency_score": 0.88,
                    "confidence": 0.9,
                    "state": "active",
                    "estimated_area_ha": 61.0,
                }
            ),
        ],
        config=ScenarioConfig.model_validate(
            {
                "horizon_hours": 1,
                "notes": "Lookahead should favor the only observation with a reachable downlink.",
                "weather_model": "fixture:planner-toy-v1",
                "opportunity_generation": {
                    "quality_threshold": 0.6,
                    "cloud_block_threshold": 0.5,
                },
            }
        ),
    )
    return compile_scenario_bundle(
        manifest,
        compiled_at=start_time,
        observation_opportunities=[
            {
                "schema_version": "1.0.0",
                "opportunity_id": "opp:lookahead-a",
                "satellite_id": "sat:lookahead-a",
                "target_cell_id": "tc:lookahead-a",
                "start_time_utc": start_time,
                "end_time_utc": start_time + timedelta(minutes=1),
                "predicted_quality_mean": 0.95,
                "predicted_cloud_obstruction_prob": 0.03,
                "estimated_data_volume_mb": 80.0,
                "slew_cost": 0.03,
                "incident_ids": ["inc:lookahead-a"],
            },
            {
                "schema_version": "1.0.0",
                "opportunity_id": "opp:lookahead-b",
                "satellite_id": "sat:lookahead-b",
                "target_cell_id": "tc:lookahead-b",
                "start_time_utc": start_time,
                "end_time_utc": start_time + timedelta(minutes=1),
                "predicted_quality_mean": 0.9,
                "predicted_cloud_obstruction_prob": 0.04,
                "estimated_data_volume_mb": 80.0,
                "slew_cost": 0.04,
                "incident_ids": ["inc:lookahead-b"],
            },
        ],
        downlink_windows=[
            {
                "schema_version": "1.0.0",
                "window_id": "dw:lookahead-b",
                "satellite_id": "sat:lookahead-b",
                "station_id": "gs:lookahead-01",
                "start_time_utc": start_time + timedelta(minutes=1),
                "end_time_utc": start_time + timedelta(minutes=2),
                "max_volume_mb": 120.0,
                "expected_rate_mbps": 100.0,
                "outage_risk": 0.0,
            }
        ],
    )
