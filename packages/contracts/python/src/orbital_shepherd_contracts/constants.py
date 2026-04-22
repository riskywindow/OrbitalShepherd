from __future__ import annotations

from typing import Final

PHASE0_SCHEMA_VERSION: Final = "0.1.0"
PHASE1_SCHEMA_VERSION: Final = "1.0.0"
PHASE1_COMPILER_VERSION: Final = "orbital-shepherd-contracts/1.0.0"

LEGACY_REPLAY_EVENT_ALIASES: Final[dict[str, str]] = {
    "scenario_loaded": "scenario_bundle_loaded",
    "opportunities_materialized": "candidate_set_materialized",
    "observation_committed": "observation_executed",
    "downlink_committed": "downlink_executed",
}

CANONICAL_REPLAY_EVENT_TYPES: Final[tuple[str, ...]] = (
    "scenario_bundle_loaded",
    "episode_started",
    "candidate_set_materialized",
    "action_mask_emitted",
    "action_selected",
    "observation_executed",
    "downlink_executed",
    "incident_packet_emitted",
    "reward_assessed",
    "episode_ended",
)

CANONICAL_TACTICAL_REPLAY_EVENT_TYPES: Final[tuple[str, ...]] = (
    "tactical_activation_created",
    "tactical_scenario_bundle_loaded",
    "tactical_episode_started",
    "tactical_candidate_set_materialized",
    "tactical_action_selected",
    "dispatch_unit_assigned",
    "route_plan_committed",
    "unit_position_updated",
    "facility_status_updated",
    "incident_state_updated",
    "tactical_metrics_assessed",
    "tactical_episode_ended",
)

LEGACY_REPLAY_PAYLOAD_FIELD_ALIASES: Final[dict[str, dict[str, str]]] = {
    "candidate_set_materialized": {
        "observation_count": "observation_opportunity_count",
        "downlink_count": "downlink_window_count",
    },
}

SCHEMA_FILE_NAMES: Final[tuple[str, ...]] = (
    "common.schema.json",
    "satellite.schema.json",
    "ground_station.schema.json",
    "target_cell.schema.json",
    "incident.schema.json",
    "observation_opportunity.schema.json",
    "downlink_window.schema.json",
    "incident_packet.schema.json",
    "scenario_manifest.schema.json",
    "scenario_bundle.schema.json",
    "replay_event.schema.json",
    "spatial_ingest_manifest.schema.json",
    "facility.schema.json",
    "dispatch_unit.schema.json",
    "route_plan.schema.json",
    "region_manifest.schema.json",
    "region_bundle.schema.json",
    "tactical_activation.schema.json",
    "tactical_scenario_manifest.schema.json",
    "tactical_scenario_bundle.schema.json",
    "tactical_replay_event.schema.json",
    "tactical_metrics_summary.schema.json",
)
