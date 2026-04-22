# Phase 1 Benchmark Runner

The Phase 1 benchmark stack lives in `packages/benchmark` and is intentionally planner-agnostic:

- planners receive only the current observation payload and emitted legal action mask
- the environment owns action legality and replay generation
- the metrics engine scores only from the immutable `ScenarioBundle` plus replay events

## Built-in planners

### `random_valid_action`

Uniformly samples one legal action from the current action mask. The random stream is seeded deterministically from:

- planner id
- scenario bundle id
- episode seed

This keeps the baseline reproducible under fixed configs.

### `urgency_greedy`

Ranks legal actions with an explicit myopic heuristic.

Observation actions use:

```text
score_obs =
  expected_incident_value * predicted_usefulness
  - cloud_risk_penalty
  - slew_penalty
  - age_penalty
```

Where:

- `expected_incident_value` comes from current linked incident urgency plus target static value
- `predicted_usefulness = predicted_quality_mean * (1 - predicted_cloud_obstruction_prob)`
- cloud and slew terms are direct penalties from the candidate opportunity payload
- age penalty is small and only uses currently visible incident timing

Downlink actions use:

```text
score_downlink =
  queued_usable_value * latency_pressure * link_reliability
  + buffer_pressure_bonus
```

Where:

- `queued_usable_value` comes from currently onboard usable observations for that satellite
- `latency_pressure` increases with time already spent waiting onboard
- `link_reliability = 1 - 0.55 * outage_risk`
- `buffer_pressure_bonus` helps clear congested satellites

The heuristic does not inspect future realized cloud outcomes or any hidden environment state.

### `value_density_greedy`

Deterministic Phase 2 heuristic with an explicit configurable score.

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

Downlink actions use:

```text
score_downlink =
  w_expected_value * queued_delivery_value
  + w_value_density * queued_value_density
  + w_freshness * latency_pressure
  + w_downlink_consequence * buffer_relief
  - w_cloud_risk * outage_risk
```

All terms come from the same visible bundle metadata, current observation payload, and current legal
action mask used by every baseline and future learned policies.

### `ortools_receding_horizon`

Deterministic OR-Tools baseline that keeps the same immediate value-density scoring terms, then adds a
small lookahead over future candidate actions:

- branch on each current legal action
- project one deterministic expected step forward without using hidden realized outcomes
- build a small OR-Tools model over the next few decision ticks
- optimize future observations, downlinks, and observation-to-downlink pairings under one-action-per-tick
  and simple buffer-capacity constraints

The approximation intentionally stays small and inspectable rather than trying to emulate the full
runtime exactly.

## Outputs

Running the benchmark writes:

- replay NDJSON per episode under `data/replays/<run-id>/...`
- episode JSON summaries under `data/benchmarks/<run-id>/episodes/...`
- run summary JSON under `data/benchmarks/<run-id>/summary.json`
- optional planner rollups:
  - `planner_summary.md`
  - `planner_summary.csv`

Each episode summary records:

- planner id and version
- bundle id and scenario family
- episode fingerprint
- replay fingerprint
- Phase 1 metrics

Replay `action_selected` events now also carry `payload.planner_trace`, including:

- selected candidate and score components
- considered current legal candidates
- scoring formulas and config for `value_density_greedy`
- solver status, objective, and selected future actions for `ortools_receding_horizon`

## Metrics

The implemented Phase 1 metric set is:

- time to first useful observation
- useful observation value captured
- cloud waste rate
- downlink latency
- missed urgent incident rate
- opportunity utilization efficiency
- mission utility

Definitions used by the local runner:

- useful observations are observations that eventually emit an `incident_packet_emitted`
- TTFUO is measured from incident ignition to the first useful downlinked packet for that incident
- useful observation value captured is:

```text
target_static_value * incident_urgency * freshness_decay * realized_quality
```

- `freshness_decay = 1 / (1 + delay_hours / 6)`
- cloud waste rate is the fraction of committed observations marked unusable
- downlink latency is measured from observation end time to packet downlink time
- urgent incidents are incidents with `urgency_score >= 0.7`
- opportunity utilization efficiency is useful observation value captured per committed observation
- mission utility is the final environment mission utility reported in the replay

## CLI

List planners:

```bash
python scripts/run_phase1_benchmark.py list-planners
```

Run all built-in planners over the whole Phase 1 pack:

```bash
python scripts/run_phase1_benchmark.py run --run-id phase1-pack-v1
```

Run a small deterministic subset:

```bash
python scripts/run_phase1_benchmark.py run \
  --run-id phase1-smoke \
  --family cloud_trap \
  --limit 2
```

Or through `make`:

```bash
make benchmark-run BENCH_ARGS="run --run-id phase1-smoke --family cloud_trap --limit 2"
```

Compare planners from the generated rollups:

- inspect `summary.json` for full per-episode details
- inspect `planner_summary.md` for a quick side-by-side view
- inspect `planner_summary.csv` for spreadsheet or notebook analysis
- inspect replay `action_selected.payload.planner_trace` for per-decision rationale
