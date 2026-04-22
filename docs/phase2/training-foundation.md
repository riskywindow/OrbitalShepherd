# Phase 2 Training Foundation

## Purpose

Phase 2 adds the deterministic training foundation for Orbital Shepherd without changing the core simulator contract. The environment remains orbital-only and single-agent. Scenario bundles, replay events, and benchmark metrics still come from the existing Phase 1 architecture; Phase 2 adds the build, split, config, and manifest layers needed for BC + PPO work.

## What Was Added

- Phase 2 scenario family registry in the scenario engine
- deterministic train / val / test / OOD split registry
- typed config models and committed config files under `training/configs`
- typed manifests for training packs, offline datasets, checkpoints, and evaluation reports
- optional W&B initialization hooks that fall back to a local no-op run
- DVC pipeline scaffolding in `dvc.yaml`
- stable local artifact roots under `data/training`

## Scenario Families

Phase 2 supports six deterministic family groups:

- `sparse_frontier`
  - low contention bootstrap family
- `burst_outbreak`
  - dense incident pressure family
- `cloud_trap`
  - observation-timing pressure family
- `downlink_crunch`
  - communication bottleneck family with reduced station throughput and tighter buffers
- `station_outage`
  - OOD robustness family with offline or degraded ground stations
- `constellation_degradation`
  - OOD robustness family with reduced orbital asset capacity

## Split Contract

The split registry lives at `training/configs/curriculum/phase2_splits.yaml`.

Interpretation:

- `train`
  - only in-distribution bundles
- `val`
  - in-distribution holdout bundles for model selection
- `test`
  - in-distribution held-out bundles for final reporting
- `ood`
  - robustness-only bundles

Phase 2 currently reserves `station_outage` and `constellation_degradation` for `ood` only.

## Curriculum Stages

Committed curriculum config: `training/configs/curriculum/phase2_curriculum.yaml`

Stages:

1. `Sparse Bootstrap`
   - BC only
   - `sparse_frontier`
   - difficulty tier `1`
2. `Incident Pressure`
   - BC then PPO
   - `sparse_frontier`, `burst_outbreak`, `cloud_trap`
   - difficulty tiers `1-2`
3. `Downlink Pressure`
   - PPO
   - adds `downlink_crunch`
   - difficulty tiers `2-3`
4. `Generalization Gate`
   - PPO on in-distribution train split only
   - evaluation on `val`, `test`, and `ood`
   - keeps outage and constellation degradation families out of training updates

## Config Layer

Committed configs:

- `training/configs/reward/phase2_reward.yaml`
- `training/configs/curriculum/phase2_curriculum.yaml`
- `training/configs/curriculum/phase2_splits.yaml`
- `training/configs/model/phase2_policy.yaml`
- `training/configs/bc/phase2_bc.yaml`
- `training/configs/ppo/phase2_ppo.yaml`
- `training/configs/evaluation/phase2_eval.yaml`

These files use JSON-compatible YAML so they validate locally without requiring a hard YAML dependency.

## Reward Versus Benchmark Metrics

This distinction is intentional and enforced by config:

- benchmark metrics are replay-derived and planner-agnostic
- training reward is a weighted composition of environment reward components
- evaluation reports must preserve both views

The reward config keeps the benchmark metric policy fixed to `audit_only`. That means:

- no metric such as mission utility, cloud waste rate, or downlink latency is redefined as the direct reward target
- reward shaping stays decomposed and attributable to existing runtime reward components

## Typed Manifests

Phase 2 defines these Pydantic-backed manifests:

- `TrainingPackManifest`
- `OfflineDatasetManifest`
- `PolicyCheckpointManifest`
- `EvaluationReportManifest`

These are defined in `packages/training/src/orbital_shepherd_training/models.py`.

## Artifact Layout

Stable local roots:

- `data/training/datasets`
- `data/training/checkpoints`
- `data/training/reports`
- `data/training/manifests`
- `data/training/scenario_packs`

Generated Phase 2 scenario bundles default to:

- `data/training/scenario_packs/osbench-phase2-foundation-v1`

Generated Phase 2 pack manifest defaults to:

- `data/training/manifests/phase2-training-pack-manifest.json`

## W&B Hooks

The training package exposes `maybe_init_wandb(...)`.

Behavior:

- if W&B is disabled in config, a no-op local run object is returned
- if W&B is enabled but the `wandb` package is unavailable, the code also falls back to the no-op run
- local validation and tests therefore do not require W&B

## DVC Pipeline

`dvc.yaml` now captures the reproducible local Phase 2 smoke graph:

- `phase2_training_pack`
- `phase2_training_pack_validate`
- `phase2_offline_dataset`
- `phase2_bc`
- `phase2_ppo`
- `phase2_evaluation`
- `phase2_demo_defaults`

The training stages still create timestamped run directories for full artifacts, but they also materialize
stable alias manifests:

- `data/training/manifests/phase2-bc-latest.json`
- `data/training/manifests/phase2-ppo-latest.json`

This keeps DVC and local demo scripts anchored to stable paths while preserving per-run manifests and full
checkpoint directories.

## Commands

Validate configs:

```bash
python scripts/phase2_training.py validate-configs
```

Build the deterministic split registry:

```bash
python scripts/phase2_training.py build-split-registry
```

Build the deterministic Phase 2 training pack and manifest:

```bash
python scripts/phase2_training.py build-training-pack
```

Validate the generated pack and manifest:

```bash
python scripts/phase2_training.py validate-training-pack
```

Or through `make`:

```bash
make phase2-config-validate
make phase2-split-build
make phase2-pack-build
make phase2-pack-validate
make phase2-smoke
make phase2-train
make phase2-eval
make phase2-demo
```

Recommended local flow:

1. `make phase2-smoke`
2. `make phase2-train`
3. `make phase2-eval`
4. `make phase2-demo-prepare`
5. `make phase2-demo`

## Developer Notes

- generated scenario packs should not be committed wholesale
- the split registry is the source of truth for split assignment, not directory layout
- OOD families are for robustness reporting only in Phase 2
- the tactical street layer is still out of scope
- PPO uses RLlib where local Ray runtime startup is available; restricted environments fall back to a
  single-process local PPO loop while preserving the same checkpoint and API contracts
