# Phase 2 Training Environment

The Phase 2 training interface wraps the existing deterministic Phase 1 runtime instead of changing it.

Implementation lives in [`packages/training/src/orbital_shepherd_training/training_env.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/training/src/orbital_shepherd_training/training_env.py).

## Goals

- preserve `OrbitalEnv` transition semantics and replay behavior
- expose a fixed-size action interface for RL training
- keep features explicit, auditable, and limited to current-state information
- stay Gymnasium-compatible when `gymnasium` is installed, while remaining locally runnable with the built-in shim

## Main API

`OrbitalTrainingEnv` wraps `OrbitalEnv` and presents:

- `reset(*, seed=None, options=None) -> (observation, info)`
- `step(slot: int) -> (observation, reward, terminated, truncated, info)`
- `action_space = Discrete(K + 1)`
- `observation_space = Dict(...)`

Default `K` is `64`.

Slot semantics:

- slot `0`: always `noop`
- slots `1..K`: projected legal non-noop actions
- padded slots beyond the projected candidate count are masked out and rejected if selected

The wrapper always executes the underlying canonical `OrbitalAction`, not a projected index into the Phase 1 mask. That keeps projection logic separate from runtime semantics.

## Observation Schema

The structured observation dict contains:

- `global_features`: shape `[G]`
- `candidate_features`: shape `[K, F]`
- `action_mask`: shape `[K + 1]`

Current defaults:

- `G = 17`
- `F = 25`

`global_features` currently include:

1. `progress_ratio`
2. `remaining_ratio`
3. `mission_utility`
4. `pending_observation_ratio`
5. `pending_downlink_ratio`
6. `completed_observation_ratio`
7. `completed_downlink_ratio`
8. `queue_entry_ratio`
9. `queue_volume_ratio`
10. `open_incident_ratio`
11. `observed_incident_ratio`
12. `downlinked_incident_ratio`
13. `missed_incident_ratio`
14. `open_incident_urgency_mean`
15. `open_incident_urgency_max`
16. `nominal_station_fraction`
17. `offline_station_fraction`

`candidate_features` are explicit and type-aware. Common features cover:

- candidate type flags
- current runtime `score_hint`
- deterministic rank position
- time-to-start / time-to-end / window span
- satellite queue pressure and pending observation pressure

Observation-only features cover:

- predicted quality
- predicted cloud obstruction probability
- estimated data volume ratio
- slew cost
- target static value
- linked open-incident count and urgency summaries

Downlink-only features cover:

- expected rate ratio
- outage risk
- max volume ratio
- station nominal flag
- queued entry pressure
- queued usable volume and urgency summaries

Type-irrelevant feature fields are zero-filled.

## Top-K Projection Policy

The variable-size Phase 1 legal action set is projected deterministically each tick:

1. start from the current legal non-noop actions emitted by `OrbitalEnv`
2. rank them by:
   - runtime action-type order from `EnvRuntimeConfig.action_order`
   - descending `score_hint`
   - earlier candidate end time
   - earlier candidate start time
   - canonical action ref
   - canonical action id
3. take the first `K`
4. place them into slots `1..K`
5. zero-mask any remaining padded slots

This guarantees stable projection for identical bundle + seed + tick.

## Masking And Debug Metadata

The training mask is explicit:

- `action_mask[0]` is `1` for `noop`
- projected legal slots are `1`
- padded / invalid slots are `0`

If a masked slot is selected, `OrbitalTrainingEnv.step()` raises `ValueError` instead of silently converting it into a runtime action.

`info` exposes projection metadata for debugging and offline export:

- `projected_candidate_count`
- `candidate_count`
- `truncated_candidate_count`
- `slot_to_action_id`
- `candidate_ids`
- `candidate_types`
- `slot_mapping`

Each `slot_mapping` entry includes:

- `slot_index`
- `action_id`
- `action_type`
- `action_ref`
- `source` (`noop`, `projected`, or `padding`)
- `projected_rank`
- `runtime_action_index`

This makes it possible to map any chosen slot back to the canonical runtime action and replay stream.

## Helpers

Normalization metadata is available via:

- `OrbitalTrainingEnv.normalization_metadata()`

Flattened ablation interface:

- `flatten_training_observation(observation)`
- `FlattenedOrbitalTrainingEnv`

The flattened wrapper keeps `action_mask` separate and concatenates:

1. `global_features`
2. flattened `candidate_features`

## Terminal Observations

The wrapper collapses terminal observations to:

- `noop` in slot `0`
- all projected candidate slots masked to `0`

This prevents post-terminal action sampling from seeing stale legal actions from the underlying runtime.

## Test Coverage

Training-env tests cover:

- observation shapes
- action-mask legality
- deterministic slot mapping under stable ties
- equivalence between projected slot selection and underlying runtime action execution
