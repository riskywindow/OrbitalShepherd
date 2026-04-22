# Data Contracts and Canonical Schemas

## 1. Contract philosophy

Orbital Shepherd uses **explicit schema versioning** and **append-only replay** so that every artifact can be validated and every benchmark run can be reproduced later.

## 2. ID conventions

All IDs are lower-case, namespaced strings.

Examples:
- `sat:worldview-01`
- `gs:alaska-fairbanks`
- `tc:8928308280fffff`
- `inc:2026-08-14-ca-001`
- `opp:sat-worldview-01:tc-8928308280fffff:000124`
- `ep:osbench-v01:cloud-trap:seed-42`

## 3. Canonical artifacts

### 3.1 ScenarioBundle
Top-level immutable artifact for an episode.
Contains:
- time horizon and seed
- satellites
- ground stations
- target cells
- incidents
- scenario configuration
- optional precomputed opportunity descriptors

### 3.2 ReplayEvent
Append-only event envelope used to reconstruct the episode and power UI replay.

### 3.3 IncidentPacket
Forward-compatible handoff object for later local tactical planning.

## 4. Event envelope

Every replay event contains:
- `schema_version`
- `event_id`
- `episode_id`
- `event_index`
- `sim_tick`
- `sim_time_utc`
- `event_type`
- `actor_type`
- `actor_id`
- `payload`

## 5. Canonical event taxonomy

### scenario_loaded
The environment has accepted and validated the scenario bundle.

### episode_started
The control loop has started.

### opportunities_materialized
The candidate action set for the current control interval has been computed.

### action_mask_emitted
The legal action mask exposed to a planner has been recorded.

### action_selected
A planner selected an action.

### observation_committed
A scheduled observation was executed and yielded a realized usefulness estimate.

### downlink_committed
Queued data was delivered to a ground station.

### incident_packet_emitted
A downstream incident handoff packet was emitted.

### reward_assessed
The environment scored the current transition.

### episode_ended
The episode terminated or truncated.

## 6. ScenarioBundle layout

```text
ScenarioBundle
  schema_version
  bundle_id
  benchmark_id
  scenario_family
  simulation_seed
  time_window
  decision_interval_seconds
  satellites[]
  ground_stations[]
  target_cells[]
  incidents[]
  config
```

## 7. Schema files in this package

- `schemas/satellite.schema.json`
- `schemas/ground_station.schema.json`
- `schemas/target_cell.schema.json`
- `schemas/incident.schema.json`
- `schemas/observation_opportunity.schema.json`
- `schemas/downlink_window.schema.json`
- `schemas/incident_packet.schema.json`
- `schemas/replay_event.schema.json`
- `schemas/scenario_bundle.schema.json`

## 8. Versioning rules

1. Every artifact includes `schema_version`.
2. Backward-compatible additions may add optional fields.
3. Breaking changes require a major schema version bump.
4. A replay consumer must reject unknown major versions.

## 9. Validation rules that matter

- timestamps must be RFC 3339 UTC strings
- `end_time_utc` must be >= `start_time_utc`
- probabilities are constrained to `[0,1]`
- H3 cell identifiers are strings, not integers
- event ordering is monotonic in `event_index`
- `sim_tick` is monotonically non-decreasing within an episode

## 10. Why this is intentionally strict

A lot of ambitious projects get soft around interfaces: half the semantics live in docs, half in code, and none in the artifacts.

Orbital Shepherd does the opposite. The goal is to make Phase 1 implementation feel like integrating against a real platform contract rather than improvising a personal side project.
