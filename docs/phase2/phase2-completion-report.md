# Phase 2 Completion Report

## Status

Phase 2 is now a reproducible orbital-only RL system instead of a loose collection of experiments.
The repository has a coherent local path for:

- deterministic Phase 2 scenario-pack generation
- deterministic split assignment
- offline expert dataset construction from real planner rollouts
- behavior cloning pretraining
- PPO finetuning from scratch or BC warm-start
- held-out evaluation against strong non-learning baselines
- checkpoint discovery through the planner API
- trained-policy replay in the same UI surface used for Phase 1

The stack preserves the Phase 1 product contract:

- canonical scenario bundles remain the source of truth
- replay events remain the trust boundary
- trained policies are consumed through the same API/UI workflow as baselines
- Phase 2 stays orbital-only and single-agent

## What Was Implemented

### Core training pipeline

- deterministic Phase 2 family registry and split registry in the scenario engine and training package
- typed config layer under `training/configs`
- typed manifests for training packs, offline datasets, checkpoints, and evaluation reports
- offline dataset builder that rolls out real planners and writes canonical episodes, steps, and training arrays
- shared `OrbitalPhase2PolicyModel` used by BC, PPO, evaluation, API serving, and UI inference traces
- BC checkpoint export with both raw-policy and RLModule adapters
- PPO checkpoint export with the same manifest contract

### Reproducibility and ergonomics

- stable fingerprints for training packs and offline datasets
- checkpoint manifests with source dataset ids, source bundle ids, and architecture metadata
- evaluation report manifests and stable report directories
- stable alias manifests for the latest BC and PPO checkpoints:
  - `data/training/manifests/phase2-bc-latest.json`
  - `data/training/manifests/phase2-ppo-latest.json`
- `dvc.yaml` pipeline covering pack build, dataset build, BC, PPO, evaluation, and demo-default generation
- `Makefile` targets for:
  - `make phase2-smoke`
  - `make phase2-train`
  - `make phase2-eval`
  - `make phase2-demo`
- `scripts/verify_phase2_stack.py` for the smallest practical end-to-end validation path
- `scripts/phase2_demo.py` for evaluation-backed demo preparation and serving

### Product-surface integration

- trained checkpoints are discoverable through `/v1/models`
- trained-policy runs flow through `/v1/models/{model_key}/run`
- evaluation reports are exposed through `/v1/reports`
- the UI can run trained policies, inspect inference traces, and compare API-backed or report-backed replays
- `data/demo/phase2-defaults.json` is now the default demo surface when present

## Final Artifact Graph

```text
training/configs/*
        |
        v
build-training-pack
        |
        +--> training/configs/curriculum/phase2_splits.yaml
        +--> data/training/manifests/phase2-training-pack-manifest.json
        +--> data/training/scenario_packs/osbench-phase2-foundation-v1/*
        |
        v
build-offline-dataset
        |
        +--> data/training/datasets/osbench-phase2-foundation-v1/offbuild--*
        +--> data/training/manifests/offbuild--*.json
        |
        v
train-bc
        |
        +--> data/training/checkpoints/trainrun--bc-*/checkpoint_*/
        +--> data/training/manifests/trainrun--bc-*/checkpoint_*.json
        +--> data/training/manifests/phase2-bc-latest.json
        |
        v
train-ppo
        |
        +--> data/training/checkpoints/trainrun--ppo-*/checkpoint_*/
        +--> data/training/manifests/trainrun--ppo-*/checkpoint_*.json
        +--> data/training/manifests/phase2-ppo-latest.json
        |
        v
evaluate
        |
        +--> data/training/reports/evalrun--*/summary.json
        +--> data/training/reports/evalrun--*/summary.md
        +--> data/training/reports/evalrun--*/episode_metrics.csv
        +--> data/training/manifests/evalrun--*/*.json
        |
        v
phase2_demo.py prepare
        |
        +--> data/demo/phase2-defaults.json
        +--> API-visible trained-policy replay episode
```

## Scenario Families And Split Strategy

Phase 2 supports six deterministic orbital families:

- `sparse_frontier`
- `burst_outbreak`
- `cloud_trap`
- `downlink_crunch`
- `station_outage`
- `constellation_degradation`

Split policy:

- `train`: in-distribution bundles only
- `val`: in-distribution holdout used for model selection
- `test`: in-distribution held-out reporting set
- `ood`: robustness-only evaluation set

OOD policy is strict:

- `station_outage` and `constellation_degradation` remain out of training updates
- OOD performance is reported, not optimized directly

The split registry is committed and deterministic. Directory layout is not the split source of truth.

## Baselines Implemented

Phase 2 baselines all operate on the same visible action contract as learned policies:

- `random_valid_action`
- `urgency_greedy`
- `value_density_greedy`
- `ortools_receding_horizon`

`value_density_greedy` and `ortools_receding_horizon` are the strong non-learning baselines for Phase 2.
They are not simulator cheats; they consume only the scenario bundle, current observation, and legal action mask.

## Model Architecture And Training Flows

### Shared model

Committed architecture: `model:phase2-policy-transformer-v1`

- global tabular encoder
- candidate encoder shared across action slots
- transformer/set encoder over `[global token | candidate tokens]`
- explicit masked action logits over projected slots
- shared value head from the same policy context
- `top_k=64` projected action slots by default

### Offline behavior cloning

BC uses offline expert traces built from deterministic planner rollouts.

Default full BC experts:

- `urgency_greedy`
- `value_density_greedy`
- `ortools_receding_horizon`

Smoke BC is intentionally smaller and uses `urgency_greedy` only so the local flow stays fast.

### PPO finetuning

PPO supports:

- scratch initialization
- checkpoint warm-start from BC

Execution modes:

- RLlib path when Ray can start normally
- deterministic local single-process PPO fallback when the environment blocks Ray socket startup

Both paths emit the same manifest shape so checkpoints remain loadable by the API and UI.

## Evaluation Methodology

Evaluation is replay-first and held-out. It does not treat shaped reward as the final metric surface.

Reported metrics:

- `mission_utility`
- `useful_observation_value_captured`
- `cloud_waste_rate`
- `missed_urgent_incident_rate`
- `opportunity_utilization_efficiency`
- `time_to_first_useful_observation_seconds`
- `downlink_latency_seconds`

Artifacts:

- per-episode JSON summaries
- NDJSON replays
- CSV summaries
- markdown summary report
- report manifests
- notable-episode extraction

The current checked-in smoke report compares one bundle each from `val`, `test`, and `ood`.

## Honest Results And Known Limitations

### What is solid

- The engineering loop is coherent end to end.
- Checkpoints are real artifacts, not notebook-only weights.
- The planner API can execute trained checkpoints and expose inference traces.
- The UI can replay trained-policy episodes and compare them against baselines or report episodes.
- Deterministic manifests and fingerprints now exist for the major artifact classes.

### What is not solved yet

- PPO quality is not mature. The current smoke policy can beat some baselines on shaped `mission_utility` while
  collapsing `useful_observation_value_captured` and `opportunity_utilization_efficiency` to zero on the smoke
  report. That is a real reward-alignment warning, not a success signal.
- The smoke evaluation is intentionally tiny. It validates plumbing and artifact compatibility, not statistical
  generalization.
- The strong heuristic baselines are still competitive and often tie each other exactly on the current held-out
  slice, which means the learned policy has not yet demonstrated durable superiority.
- The local PPO fallback preserves the product contract but is not a replacement for larger-scale RLlib training on
  an unrestricted host.
- Some library warnings remain:
  - PyTorch transformer nested-tensor warning
  - RLlib deprecation warnings around module construction
  - `pyarrow` host-probe warnings on restricted macOS sandboxes

## Final Validation Path

Smallest practical end-to-end validation:

```bash
make phase2-smoke
```

This writes `data/demo/phase2-verification.json` and confirms:

- scenario pack build
- offline dataset build
- BC checkpoint export
- PPO checkpoint export
- evaluation report generation
- trained-policy discovery through `/v1/models`
- API replay and inference-trace retrieval for a trained checkpoint

Persistent local demo flow:

```bash
make phase2-train
make phase2-eval
make phase2-demo-prepare
make phase2-demo
```

## Direct Next Steps For Phase 3

1. Fix reward alignment before scaling PPO. The current smoke policy can optimize away the wrong behavior.
2. Expand held-out evaluation beyond one bundle per split and add repeated-seed comparison runs.
3. Add checkpoint-selection policy beyond “latest,” including explicit promoted-model manifests.
4. Add richer report views for failure clustering by family, outage mode, and queue/downlink regime.
5. Move from local-only orchestration to reproducible multi-run experiment management without breaking the replay
   boundary.
6. Add stronger rollout/debug tooling around policy collapse, action-mask pressure, and reward-component drift.
