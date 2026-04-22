# Phase 3 Architecture Overview

## Purpose

Phase 3 adds the tactical foundation beneath the existing orbital detection stack.
It does not replace orbital artifacts.
It extends them with a typed downstream layer for regional dispatch and routing.

## Design Rules

- Keep `IncidentPacket` as the official orbital-to-tactical handoff.
- Keep tactical planners interchangeable.
- Keep manifests, bundle fingerprints, and replay first-class.
- Keep tactical replay separate from orbital replay.
- Do not introduce local tactical RL training yet.

## Main Flow

1. Orbital execution runs as it does in Phases 1 and 2.
2. Orbital replay emits `incident_packet_emitted`.
3. The emitted `IncidentPacket` is wrapped by `TacticalActivation`.
4. `RegionManifest` plus `SpatialIngestManifest` compile into `RegionBundle`.
5. `TacticalScenarioManifest` combines the activation, region bundle reference, tactical units, facilities, and config.
6. `TacticalScenarioBundle` compiles deterministic route candidates and runtime-ready tactical state.
7. Tactical execution emits canonical `TacticalReplayEvent` records.
8. Tactical replay is reduced into `TacticalMetricsSummary`.

## Package Surfaces

- `packages/region_builder`: deterministic region manifest and bundle compilation boundary
- `packages/geo_artifacts`: shared helpers for spatial ingest and geospatial artifact handling
- `packages/routing_engine`: planner-agnostic route candidate and route-plan surface
- `packages/tactical_scenario_engine`: tactical manifest and bundle construction boundary
- `packages/ground_env`: tactical execution runtime surface
- `packages/tactical_baselines`: heuristic and OR tactical baseline surface
- `packages/tactical_metrics`: replay-derived tactical metric computation surface
- `packages/escalation_bridge`: typed `IncidentPacket` to `TacticalActivation` bridge surface

## Replay Strategy

Orbital replay remains the source of truth for orbital decisions.
Tactical replay mirrors the same philosophy with a separate envelope:

- append-only
- monotonic `event_index`
- monotonic non-decreasing `sim_tick`
- typed payloads per canonical event name

The tactical replay family starts with these event names:

- `tactical_activation_created`
- `tactical_scenario_bundle_loaded`
- `tactical_episode_started`
- `tactical_candidate_set_materialized`
- `tactical_action_selected`
- `dispatch_unit_assigned`
- `route_plan_committed`
- `unit_position_updated`
- `facility_status_updated`
- `incident_state_updated`
- `tactical_metrics_assessed`
- `tactical_episode_ended`

## Compatibility

Compatibility rules for Phase 3:

- additive only within the v1 contract family
- no breaking change to `IncidentPacket`
- no rewrite of orbital `ScenarioBundle` or `ReplayEvent`
- explicit tactical contracts instead of overloading orbital ones
