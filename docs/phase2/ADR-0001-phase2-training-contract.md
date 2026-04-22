# ADR-0001: Phase 2 Training Contract

## Status

Accepted on 2026-04-12.

## Context

Phase 1 established three invariants that Phase 2 must not break:

1. the simulator boundary is an immutable `ScenarioBundle`
2. benchmark metrics remain planner-agnostic and replay-derived
3. reward shaping exists, but it is explicit and decomposed rather than hidden inside benchmark scoring

Phase 2 adds a training foundation, not a new product surface. The current repo does not yet need policy serving, distributed orchestration, or a tactical street layer. It needs deterministic pack generation, split discipline, typed manifests, and config contracts that let BC and PPO reuse the Phase 1 simulator and metric boundaries.

## Decision

Phase 2 remains:

- orbital-only
- single-agent
- replay-first
- benchmark-metric preserving

Phase 2 introduces:

- deterministic Phase 2 scenario recipe families
- a train / val / test / OOD split registry contract
- typed config files for reward shaping, curriculum, model architecture, BC, PPO, and evaluation
- typed manifests for scenario packs, offline datasets, checkpoints, and evaluation reports
- optional W&B hooks that degrade cleanly to disabled local execution
- DVC pipeline scaffolding that works locally with no configured remote

## In Scope

- BC and PPO configuration contracts
- deterministic scenario-pack build and validation commands
- auditable training reward configuration
- curriculum stages over fixed scenario families and split assignments
- manifest contracts for datasets, checkpoints, and reports
- stable local directories for datasets, checkpoints, reports, manifests, and scenario packs

## Out of Scope

- tactical street-level map or handoff execution
- multi-agent planning or decentralized coordination
- live remote training infrastructure
- policy serving APIs
- replacing planner-agnostic benchmark metrics with training reward
- committing large generated scenario packs or datasets into git

## Benchmark Metrics Versus Training Reward

Benchmark metrics and training reward are related but not identical.

Benchmark metrics remain:

- replay-derived
- planner-agnostic
- suitable for baseline comparison and reporting

Training reward remains:

- environment-emitted
- decomposed by explicit components
- configurable through auditable term weights

Phase 2 therefore adopts this rule:

1. benchmark metrics are never used as the direct reward signal
2. reward shaping may reference environment reward components such as observation value, downlink value, latency penalty, cloud penalty, buffer pressure penalty, and missed incident penalty
3. evaluation must report benchmark metrics and reward audit summaries side by side

This keeps RL tuning possible without contaminating benchmark comparability.

## Scenario Split Contract

The deterministic split registry is the source of truth for Phase 2 split assignment.

Contract:

- `train`
  - in-distribution training bundles only
  - used for BC fitting and PPO environment rollouts
- `val`
  - in-distribution holdout bundles
  - used for model selection and early stopping
- `test`
  - in-distribution unseen bundles
  - used for final held-out reporting
- `ood`
  - out-of-distribution robustness bundles
  - never used for BC fitting, PPO updates, or reward tuning

Phase 2 family allocation:

- in-distribution families:
  - `sparse_frontier`
  - `burst_outbreak`
  - `cloud_trap`
  - `downlink_crunch`
- OOD-only families:
  - `station_outage`
  - `constellation_degradation`

Seed discipline:

- seed-to-split assignment is deterministic and committed in the split registry
- a bundle may appear in exactly one split
- OOD families may appear only in the `ood` split

## Artifact Expectations

### Scenario Packs

Scenario packs are generated on demand into `data/training/scenario_packs/<benchmark-id>/`.

Expected properties:

- deterministic bundle bytes for the same recipe set
- small committed recipe and split configs
- generated pack manifest written separately from bundle files

### Offline Datasets

Offline datasets are represented by `OfflineDatasetManifest`.

Expected properties:

- references the source training pack id
- names the split and algorithm intent
- records record count and bundle provenance
- stores a stable artifact fingerprint

### Policy Checkpoints

Checkpoints are represented by `PolicyCheckpointManifest`.

Expected properties:

- references the model architecture id
- references source dataset ids
- records the reward config id and global step
- stores a stable artifact fingerprint

### Evaluation Reports

Reports are represented by `EvaluationReportManifest`.

Expected properties:

- reference the evaluated artifact id and evaluation config id
- report benchmark metrics separately from reward audit summaries
- store a stable artifact fingerprint

## Consequences

Positive:

- Phase 2 can start BC + PPO work without bypassing the Phase 1 simulator boundary
- reward tuning becomes explicit and reviewable
- split discipline is executable rather than hand-waved in docs
- large generated artifacts stay out of git

Tradeoffs:

- Phase 2 adds more config and manifest surface area
- DVC and W&B are scaffolding only until real training jobs exist
- OOD evaluation is intentionally stricter than what Phase 2 training can optimize against directly
