# RFC-0001: Orbital Shepherd Mission Contract

- **Status:** Accepted
- **Version:** 0.1.0
- **Phase:** 0
- **Owner:** Orbital Shepherd core design
- **Purpose:** Freeze the problem definition, system boundaries, success criteria, invariants, and non-goals before implementation.

## 1. One-line definition

Orbital Shepherd is an event-sourced, replayable Earth-observation mission-control system that learns how to allocate satellite observations and downlink opportunities to maximize the value of time-critical incident intelligence.

## 2. Elevator pitch

Orbital Shepherd is not a toy RL environment. It is a **digital twin of orbital sensing operations** with explicit uncertainty, downlink scarcity, mission priorities, and replay-driven evaluation. The Phase 0 system contract is designed so that future RL results are benchmarkable, explainable, and defensible to hard-core engineers.

## 3. Mission statement

The system shall optimize the following objective over a finite horizon:

> Maximize expected mission utility from **useful** incident observations, subject to orbital visibility, retargeting limits, onboard buffer limits, downlink scarcity, and weather uncertainty.

Mission utility is defined at a high level as:

```text
U = sum_i(value_i * freshness_i * usable_quality_i)
    - lambda_cloud * cloudy_or_unusable_observations
    - lambda_delay * downlink_latency
    - lambda_miss  * missed_urgent_incidents
    - lambda_waste * wasted_visibility_windows
```

This objective is intentionally mission-facing rather than algorithm-facing. It can be optimized by heuristics, OR methods, or RL without changing the benchmark contract.

## 4. Why this project is technically impressive

1. **It combines aerospace, geospatial systems, optimization, and RL.**
2. **It is event-sourced and replayable by design,** which makes results auditable rather than hand-wavy.
3. **It forces rigorous baselining** against structured optimization instead of weak toy baselines.
4. **It is hierarchical by contract.** Phase 0 reserves a street-level incident handoff without prematurely forcing joint training.
5. **It treats simulation as infrastructure,** not just a notebook helper.

## 5. Primary users

### 5.1 Mission-ops engineer
Needs to inspect opportunities, compare planners, and replay decisions.

### 5.2 RL/ML engineer
Needs a deterministic environment, clear reward decomposition, and benchmark splits.

### 5.3 Systems evaluator / interviewer
Needs to see that the project has real interfaces, non-goals, and measurable success criteria.

## 6. Scope

### 6.1 In scope for Phase 0
- Wildfire detection and response as the flagship mission vertical.
- Global orbital tasking and downlink scheduling.
- Deterministic scenario bundles.
- Immutable event logs and replay support.
- Benchmark families, metrics, and baseline definitions.
- Canonical schemas for all primary entities.
- A reserved `IncidentPacket` contract for later tactical handoff.

### 6.2 Explicitly out of scope for Phase 0
- Raw satellite imagery ingestion or image processing.
- Computer vision fire detection.
- Exact RF link-budget modeling.
- Precise spacecraft attitude dynamics.
- Multi-sensor fusion.
- End-to-end hierarchical training.
- Real-time stream processing and production SLAs.

## 7. Locked design decisions

### 7.1 Mission vertical
The first mission vertical is **wildfire detection**. This is locked to preserve focus and produce a coherent benchmark.

### 7.2 Primary control problem
The first control problem is **orbital tasking + downlink scheduling**. Tactical street-level response is represented as a future downstream consumer through an `IncidentPacket`, but does not influence the initial control loop directly.

### 7.3 Time model
- Canonical time zone: **UTC**.
- Episode clock: **fixed control timestep of 60 seconds**.
- Fine-grained geometry may be precomputed at sub-minute resolution, but decisions are emitted at control ticks.
- Wall-clock time is never used inside simulation logic.

### 7.4 Geometry model
- Earth reference: **WGS84 geodetic** for external coordinates.
- Derived frame for propagation / geometry computations: **ECEF-compatible internal representation**.
- Target aggregation primitive: **H3 cell IDs**.

### 7.5 Replay model
All episode state must be derivable from:
1. the immutable scenario bundle,
2. the deterministic seed,
3. and the ordered action/event stream.

### 7.6 Baseline rule
No RL result may be claimed until the following baselines exist and are benchmarked on identical scenario packs:
- random valid-action baseline,
- urgency-greedy baseline,
- value-density-greedy baseline,
- OR-Tools receding-horizon baseline.

## 8. Core entities

The following entities are canonical and required in Phase 0:
- `Satellite`
- `GroundStation`
- `TargetCell`
- `Incident`
- `ObservationOpportunity`
- `DownlinkWindow`
- `IncidentPacket`
- `ReplayEvent`
- `ScenarioBundle`

## 9. System invariants

1. **Determinism:** identical bundle + config + seed => identical replay.
2. **Immutability:** replay events are append-only.
3. **Explainability:** every action event can be paired with candidate context and reward attribution.
4. **Physical validity:** action masks must prevent impossible actions.
5. **Benchmark integrity:** train/val/test scenario families are isolated by contract.
6. **Versionability:** every schema and artifact includes `schema_version`.

## 10. Mission success criteria

Phase 0 is considered complete when the following are true:

### 10.1 Contract completeness
- The problem statement is frozen.
- Success metrics are defined mathematically.
- Non-goals are explicit.

### 10.2 Artifact completeness
- All primary entities have machine-readable schemas.
- A scenario bundle format exists and includes a sample artifact.
- A replay event format exists and includes a sample artifact.
- A baseline and benchmark spec exists.
- A system diagram exists.

### 10.3 Engineering readiness
- Another engineer can implement Phase 1 services from these docs without clarifying core concepts.
- A future environment implementation can validate inputs against the provided schemas.

## 11. Risks and mitigations

### Risk: the simulator becomes too toy-like
**Mitigation:** observation utility explicitly depends on geometry, delay, cloud obstruction, and downlink latency.

### Risk: RL appears strong only because baselines are weak
**Mitigation:** OR-Tools receding-horizon baseline is mandatory and benchmark packs are fixed.

### Risk: replay/debugging becomes impossible
**Mitigation:** event sourcing, deterministic bundles, reward decomposition, and candidate-set logging are required.

### Risk: scope explosion into a full aerospace program
**Mitigation:** Phase 0 freezes non-goals and defers imagery, CV, and exact attitude/RF modeling.

## 12. Exit checklist

- [x] Mission vertical frozen
- [x] Control problem frozen
- [x] Time/geometry model frozen
- [x] Benchmark contract defined
- [x] Event taxonomy defined
- [x] Schemas authored
- [x] Example artifacts authored
- [x] Early API contract authored

## 13. Statement of technical ambition

The bar for Orbital Shepherd is not "a model that learns something." The bar is:

> a benchmarked, replayable, simulation-first mission-control system that can eventually host RL, optimization, and hybrid planners under the same contract.

That is the standard this Phase 0 package is designed to enforce.
