# Phase 1 Environment Runtime

## Purpose

The Phase 1 environment runtime turns a compiled `ScenarioBundle` into a deterministic single-agent control loop with replay-first step semantics. It is intentionally simple enough to inspect by hand and strict enough to serve as the simulator boundary for future RL work.

The implementation lives in [`packages/env_runtime/src/orbital_shepherd_env_runtime`](/Users/rishivinodkumar/OrbitalShepherd/packages/env_runtime/src/orbital_shepherd_env_runtime).

## Runtime Boundary

`OrbitalEnv` consumes:

- a canonical `ScenarioBundle`
- precomputed `observation_opportunities`
- precomputed `downlink_windows`

The environment does not fetch live data, does not use wall-clock time in transition logic, and does not regenerate opportunities during an episode.

## Step Contract

The runtime exposes a Gymnasium-style shape without taking a hard dependency on Gymnasium:

- `reset(seed=...) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`

`action` can be supplied as:

- an integer index into the current action mask
- an `OrbitalAction`
- a mapping with `action_type` and `action_ref`
- `None`, which resolves to `noop`

## Runtime State Model

The inspectable state tracked by `OrbitalState` is:

- `sim_tick`
- `sim_time_utc`
- `episode_id`
- `episode_seed`
- `pending_observation_ids`
- `pending_downlink_ids`
- `completed_observation_ids`
- `completed_downlink_ids`
- `onboard_queue`
- `incident_records`
- `incident_packets`
- cumulative `mission_utility`

### Incident lifecycle

Each incident is tracked as one of:

- `unseen`
- `observed`
- `downlinked`
- `missed`

`observed` means a usable observation has been committed to the onboard queue but not yet delivered. `missed` is assigned when no future deterministic path remains to produce or deliver a usable observation.

### Onboard queue

The queue stores committed observation products with:

- source opportunity ID
- satellite ID
- observation time
- realized quality
- cloud fraction
- usable flag
- incident references
- data volume

This is the state that downlink actions consume.

### Ground-station availability

Phase 1 uses static station capability availability from the bundle plus the currently materialized downlink windows. The observation returned by the environment includes station capability summaries and the current legal downlink actions.

## Action Model

The legal action set is emitted as an `OrbitalActionMask` each decision tick. Phase 1 supports:

- `noop`
- `schedule_observation`
- `schedule_downlink`

### Observation legality

An observation action is legal when:

- the opportunity has materialized
- it has not already been scheduled or completed
- its window has not ended
- the satellite still has deterministic buffer capacity

Observation opportunities tied to incidents materialize as soon as the incident is known and remain legal until the opportunity window ends.

### Downlink legality

A downlink action is legal when:

- the contact window is active
- the window has not already been scheduled or completed
- the associated satellite has payload in the onboard queue
- the station is not offline

## Transition Semantics

Each `step` applies an action for the current control interval and then advances the simulator by exactly one deterministic decision interval.

During the interval, the runtime:

1. records the selected action
2. commits any scheduled observations whose window ends in the interval
3. commits any scheduled downlinks whose contact ends in the interval
4. emits incident packets for usable downlinked observations
5. assesses per-step reward components
6. materializes the next candidate set and next action mask, unless the episode has ended

## Reward Decomposition

Phase 1 keeps reward shaping explicit even though the benchmark baselines are still simple. The runtime currently emits per-step components including:

- `observation_value`
- `downlink_value`
- `cloud_penalty`
- `latency_penalty`
- `buffer_pressure_penalty`
- `missed_incident_penalty`

`reward_assessed` events always carry the total plus the component dictionary so later metrics or RL wrappers can reuse the same attribution.

## Replay Events

The runtime emits canonical Phase 1 replay events and accepts legacy aliases at the emitter boundary.

Canonical event order is:

1. `scenario_bundle_loaded`
2. `episode_started`
3. `candidate_set_materialized`
4. `action_mask_emitted`
5. `action_selected`
6. `observation_executed`
7. `downlink_executed`
8. `incident_packet_emitted`
9. `reward_assessed`
10. `episode_ended`

Legacy names such as `scenario_loaded`, `opportunities_materialized`, `observation_committed`, and `downlink_committed` are normalized before validation or writing.

`action_selected.payload` is also the canonical home for optional planner-side diagnostics. Baselines may
attach `planner_trace` with considered candidates, score components, formulas, and solver metadata while
keeping the action contract itself unchanged.

## Replay Writers

The runtime supports deterministic sink hooks:

- `InMemoryReplaySink`
- `NdjsonReplayWriter`

`NdjsonReplayWriter` writes canonical compact JSON lines using the shared deterministic serializer, so identical episode inputs produce identical replay bytes.

## Determinism Notes

Phase 1 determinism depends on:

- immutable bundle inputs
- fixed decision interval
- seed-derived observation realization
- stable action ordering
- canonical replay event IDs and payload serialization

Repeated runs with the same bundle, env config, planner behavior, and seed produce identical replay streams.
