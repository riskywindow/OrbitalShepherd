# Orbital Shepherd Benchmark Specification (OS-Bench v0.1)

## 1. Benchmark purpose

OS-Bench measures how well a planner allocates scarce observation and downlink opportunities under orbital constraints, weather uncertainty, and time-sensitive incident demand.

The benchmark is designed so that a strong result means something to a skeptical engineer.

## 2. Formal problem statement

Given:
- a finite horizon,
- a constellation of observation assets,
- a set of ground stations,
- a time-varying set of target-cell incident values,
- uncertain or partially observed observation quality,
- and finite onboard/downlink resources,

select actions over time to maximize mission utility.

## 3. Episode defaults

- horizon: 24 hours
- control interval: 60 seconds
- default target resolution: H3 cells at mission-tuned resolution
- default planner mode: single top-level orbital controller

## 4. Scenario families

The benchmark must include at least the following scenario families.

### 4.1 Sparse Frontier
Few incidents, low contention, large geographic dispersion.
Purpose: checks whether a planner wastes effort when the world is quiet.

### 4.2 Burst Outbreak
Many incidents ignite in a short temporal burst.
Purpose: stresses triage and retargeting.

### 4.3 Cloud Trap
High-value targets have early observation windows with poor forecasted usability, followed by later cleaner passes.
Purpose: punishes short-sighted planners.

### 4.4 Downlink Crunch
Observation opportunities are plentiful but station capacity is scarce or delayed.
Purpose: tests whether planners understand that a useful observation is not mission value until it can be delivered.

### 4.5 Station Outage
A major ground station becomes unavailable mid-episode.
Purpose: forces replanning under infrastructure disruption.

### 4.6 Constellation Degradation
One or more assets have reduced availability or stricter retargeting limits.
Purpose: checks policy robustness to degraded assets.

## 5. Split contract

- **Train:** scenario families with randomized seeds and parameter sweeps.
- **Validation:** held-out seeds from seen families.
- **Test:** held-out seeds plus at least one withheld composition pattern per family.
- **Secret test option:** future extension; optional private pack to prevent overtuning.

No result claim is valid unless train/validation/test are separated by scenario IDs and seeds.

## 6. Required baselines

### 6.1 Random valid-action
Chooses uniformly from legal actions.
Purpose: sanity floor.

### 6.2 Urgency-greedy
Chooses the legal action with highest immediate incident urgency.
Purpose: strong naive operator baseline.

### 6.3 Value-density greedy
Chooses the action with best local ratio of expected value to opportunity cost.
Purpose: stronger myopic baseline.

### 6.4 OR-Tools receding-horizon planner
Solves a finite-horizon approximation over a rolling lookahead window.
Purpose: serious structured baseline.

## 7. Core metrics

### 7.1 Time to first useful observation (TTFUO)
For each incident, measure the elapsed time between incident ignition and the first successfully downlinked usable observation.

Report:
- mean
- median
- p90

### 7.2 Useful observation value captured (UOVC)
Sum over all downlinked observations:

```text
UOVC = sum(value * freshness_decay * usable_quality)
```

### 7.3 Cloud waste rate (CWR)
Fraction of committed observations whose realized usefulness falls below a configured threshold due to cloud or quality loss.

### 7.4 Downlink latency (DLL)
Elapsed time between observation commit and successful downlink.

### 7.5 Missed urgent incident rate (MUIR)
Fraction of urgent incidents that never receive a usable observation within their service window.

### 7.6 Opportunity utilization efficiency (OUE)
Useful mission value per observation opportunity consumed.

### 7.7 Mission utility (MU)
Primary scalar objective for ranking planners.

## 8. Secondary diagnostics

- retarget count per orbit
- buffer occupancy over time
- station congestion profile
- candidate-set entropy
- action-mask pressure
- reward component attribution

These are not leaderboard metrics but are required for debugging and serious comparison.

## 9. Reporting contract

Any benchmark run must emit:
- planner name and version
- scenario pack ID
- seed list
- episode fingerprints
- core metrics
- secondary diagnostics
- replay artifact pointer

## 10. Anti-overfitting rules

1. Planners may not inspect future ground-truth values outside the allowed lookahead contract.
2. Heuristics and OR baselines must operate on the same observation/downlink candidate set available to RL.
3. Metric scripts must be planner-agnostic.
4. Reward shaping may differ internally for RL, but leaderboard ranking uses benchmark metrics only.

## 11. Acceptance criteria for Phase 0 benchmark readiness

Phase 0 benchmark design is complete when:
- scenario families are named and parameterized,
- metrics are mathematically defined,
- baselines are specified,
- artifact schemas exist,
- and a sample scenario + replay validate against contract.

## 12. Phase 1 benchmark target

By the end of Phase 1, the benchmark should support:
- at least 10 deterministic episodes across 3 scenario families,
- random + urgency-greedy baselines,
- replay export,
- and a visible metric report in the UI.

## 13. Long-range benchmark ambition

The long-term prestige version of OS-Bench is not just "my environment." It is a reusable benchmark where:
- OR methods,
- RL methods,
- imitation learning,
- and hierarchical control systems

can all compete on the same replay-first contract.
