# Phase 2 Strong Baselines

Phase 2 adds two serious non-learning planners on top of the same runtime action contract already used by
the Phase 1 baselines:

- `value_density_greedy`
- `ortools_receding_horizon`

Both planners consume only:

- the immutable `ScenarioBundle`
- the current observation payload
- the current legal action mask

Neither planner reads hidden realized cloud outcomes, hidden future action masks, or any simulator-only
state.

## `value_density_greedy`

`value_density_greedy` is a deterministic one-step heuristic with configurable weights.

Observation actions use:

```text
score_obs =
  w_expected_value * expected_mission_value
  + w_value_density * value_density
  + w_freshness * freshness_bonus
  + w_quality * predicted_quality
  - w_cloud_risk * cloud_risk
  - w_retarget * retarget_cost
  - w_downlink_consequence * downlink_consequence
```

Where:

- `expected_mission_value` is the linked target value and incident urgency after freshness decay,
  multiplied by predicted quality and predicted cloud clearance
- `value_density = expected_mission_value / estimated_data_volume_mb`
- `freshness_bonus` decays with incident age using only visible incident timing
- `downlink_consequence` penalizes observations that push a satellite deeper into buffer pressure or
  farther from the next visible downlink opportunity

Downlink actions use:

```text
score_downlink =
  w_expected_value * queued_delivery_value
  + w_value_density * queued_value_density
  + w_freshness * latency_pressure
  + w_downlink_consequence * buffer_relief
  - w_cloud_risk * outage_risk
```

Where:

- `queued_delivery_value` comes from currently onboard usable observations that the current window can
  actually deliver under FIFO volume limits
- `queued_value_density` normalizes that value by delivered volume
- `latency_pressure` increases with observation age
- `buffer_relief` rewards clearing constrained satellites

The weights live in `ValueDensityScoringConfig` inside
[planners.py](/Users/rishivinodkumar/OrbitalShepherd/packages/benchmark/src/orbital_shepherd_benchmark/planners.py).

## `ortools_receding_horizon`

`ortools_receding_horizon` keeps the same immediate scoring terms, then adds a small deterministic
lookahead.

Approximation:

1. Enumerate the current legal actions from the canonical action mask.
2. For each current legal action, build a deterministic projected state one decision step later using
   expected values only.
3. Enumerate future observation and downlink candidates over a short finite horizon.
4. Solve a small OR-Tools CP-SAT model with:
   - at most one action per future decision tick
   - simple satellite buffer-capacity constraints
   - observation-to-downlink pairing variables so future observations only earn full value when a future
     downlink can carry them
5. Score the current action as:

```text
current_branch_score =
  immediate_value_density_score
  + future_discount * ORTools_future_objective
```

This is not a full simulator clone. It is intentionally a small, debuggable approximation that captures
the main strategic question Phase 2 needs: whether a current action creates a reachable path to future
delivery and buffer relief.

## Trace Data

Both planners write trace metadata into replay `action_selected.payload.planner_trace`.

Included fields:

- `selected_candidate`
- `considered_candidates`
- score components for the selected and considered actions
- scoring formulas and weight config for `value_density_greedy`
- solver status, objective, bound, projected state summary, and selected future actions for
  `ortools_receding_horizon`

This keeps replay inspection, benchmark analysis, and future policy comparisons on the same artifact path.
