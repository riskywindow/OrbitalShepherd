# Architecture - Phase 0 System Design

## 1. Architectural stance

Orbital Shepherd is designed as an **event-sourced digital twin**.

That means:
- scenario inputs are immutable,
- decision events are append-only,
- derived state is reconstructible,
- and every benchmark result is replayable.

This is a deliberate prestige move: it pushes the project away from notebook culture and toward real systems engineering.

## 2. Top-level component model

The Phase 0 architecture consists of nine logical components:

1. **Ephemeris Service**
   - fetches GP/OMM snapshots,
   - normalizes identifiers,
   - produces propagation-ready orbital state assets.

2. **Incident Service**
   - ingests wildfire detections,
   - normalizes them into H3 target-cell demand.

3. **Weather Service**
   - ingests cloud-cover and other weather fields,
   - emits time-indexed environmental constraints.

4. **Scenario Engine**
   - compiles orbital, incident, and weather assets into a deterministic `ScenarioBundle`.

5. **Opportunity Builder**
   - generates `ObservationOpportunity` and `DownlinkWindow` objects from assets + time horizon.

6. **Environment Runtime**
   - exposes the control problem as a deterministic single-agent environment.

7. **Baseline Solver Service**
   - hosts non-learning planners under the same action contract.

8. **Replay Store and Metrics Engine**
   - persists event streams,
   - computes benchmark metrics,
   - supports counterfactual comparisons.

9. **Planner API + Mission UI**
   - serves scenario metadata, step-by-step decisions, and replay streams to the globe interface.

## 3. Data flow

### 3.1 Ingest path
`ephemeris` + `incidents` + `weather` -> `scenario-engine`

### 3.2 Compilation path
`scenario-engine` -> `ScenarioBundle`

### 3.3 Precomputation path
`ScenarioBundle` -> `opportunity-builder` -> candidate opportunities/windows

### 3.4 Control path
`ScenarioBundle` + candidates -> `env-runtime`

### 3.5 Decision path
`env-runtime` <-> `baseline-solver | future-policy-adapter`

### 3.6 Observability path
`env-runtime` -> `replay-store` -> `metrics-engine` -> `planner-api` -> `globe-ui`

## 4. Determinism contract

Every episode must be reproducible from:
- `scenario_bundle_id`
- `schema_version`
- `simulation_seed`
- environment configuration
- ordered decision events

### 4.1 Forbidden sources of nondeterminism
- wall-clock timestamps inside transition logic
- mutable upstream data fetched during an episode
- hidden randomness outside the seeded RNG contract
- dependence on insertion order of non-ordered collections

### 4.2 Episode fingerprint

Each episode must expose a stable fingerprint:

```text
episode_fingerprint = hash(
  scenario_bundle_bytes,
  env_config_bytes,
  simulation_seed,
  actor_implementation_version
)
```

## 5. Time and geometry model

### 5.1 Time
- simulation horizon: canonical default 24 hours
- decision interval: 60 seconds
- all timestamps: RFC 3339 UTC
- event ordering: `sim_tick` primary, `event_index` secondary

### 5.2 Geometry
- external coordinates: WGS84 latitude / longitude / altitude
- target aggregation: H3 cell IDs
- propagation computations: internal Earth-fixed reference frame
- local tactical handoff: deferred; represented only as `IncidentPacket`

## 6. Storage strategy

### 6.1 Canonical artifact storage
Use immutable object storage semantics even in local development:
- `ScenarioBundle` stored as versioned JSON or Parquet-backed bundle
- replay stored as NDJSON append log with optional Parquet export

### 6.2 Query/index storage
A relational store may index metadata for fast lookup:
- scenario registry
- episode registry
- metric summaries
- replay pointers

### 6.3 Why event logs matter
The replay log is not just debugging scaffolding. It is the **source of truth for evaluation**, allowing side-by-side comparisons between:
- RL policy,
- greedy planner,
- OR baseline,
- and future hierarchical controllers.

## 7. Service boundaries

### 7.1 Ephemeris Service boundary
Input:
- GP/OMM source snapshots

Output:
- canonical orbit asset descriptors
- propagation query interface

### 7.2 Scenario Engine boundary
Input:
- orbit assets
- target-cell demand
- weather fields
- scenario family config

Output:
- validated `ScenarioBundle`

### 7.3 Environment Runtime boundary
Input:
- `ScenarioBundle`
- decision implementation (baseline or learned policy)

Output:
- replay event stream
- metric summary

### 7.4 Planner API boundary
Input:
- scenario registry
- replay store
- metric summaries

Output:
- scenario list
- episode start/step endpoints
- replay retrieval
- benchmark reports

## 8. Event taxonomy

Phase 0 reserves the following core event types:
- `scenario_loaded`
- `episode_started`
- `opportunities_materialized`
- `action_mask_emitted`
- `action_selected`
- `observation_committed`
- `downlink_committed`
- `incident_packet_emitted`
- `reward_assessed`
- `episode_ended`

These are intentionally high-signal events. Low-level debug telemetry may exist, but the canonical replay contract should stay compact enough to diff and inspect.

## 9. System diagram

See `diagrams/orbital_shepherd_phase0_architecture.svg`.

The diagram illustrates four architectural principles:
1. upstream data is compiled into immutable scenario bundles,
2. opportunities are precomputed and fed into a deterministic environment,
3. planners all act through the same action contract,
4. every result is routed through the replay/metrics layer.

## 10. Why this architecture is the right amount of ambitious

This architecture is aggressive enough to look serious because it bakes in:
- benchmarkability,
- reproducibility,
- planner interchangeability,
- and UI-ready replay.

But it stays sane because it does **not** yet require:
- production streaming,
- exact spacecraft subsystems,
- tactical street-level integration,
- or distributed training infrastructure.

## 11. Phase 1 implementation guidance

The fastest path into implementation is:
1. build the scenario engine and validator,
2. build the ephemeris adapter and sample orbit asset compiler,
3. build the opportunity builder,
4. stub the env runtime with a no-op agent,
5. render the resulting state and replay in the globe UI.

That path preserves the architecture while getting something visual early.
