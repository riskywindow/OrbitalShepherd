# Phase 2 BC And Finetune

Phase 2 now supports a two-stage learning path on top of the shared `OrbitalPhase2PolicyModel`:

1. behavior cloning from offline expert traces
2. PPO fine-tuning in the online environment

The same core policy weights are reused in both stages. BC exports a raw Torch policy checkpoint plus an explicit RLlib single-agent `RLModule` adapter, and PPO warm-start consumes that adapter through RLlib's supported `load_state_path` hook.

Implementation lives in:

- [`packages/training/src/orbital_shepherd_training/bc_training.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/training/src/orbital_shepherd_training/bc_training.py)
- [`packages/training/src/orbital_shepherd_training/policy_checkpointing.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/training/src/orbital_shepherd_training/policy_checkpointing.py)
- [`packages/training/src/orbital_shepherd_training/rllib_training.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/training/src/orbital_shepherd_training/rllib_training.py)

## Workflow

Build the deterministic Phase 2 scenario pack first:

```bash
python3 scripts/phase2_training.py build-training-pack
```

Run BC only:

```bash
python3 scripts/phase2_training.py train-bc \
  --config training/configs/bc/phase2_bc.yaml
```

Run PPO from scratch:

```bash
python3 scripts/phase2_training.py train-ppo \
  --config training/configs/ppo/phase2_ppo.yaml
```

Run PPO from BC-pretrained weights:

```bash
python3 scripts/phase2_training.py train-ppo \
  --config training/configs/ppo/phase2_ppo_bc_warmstart.yaml
```

Local smoke variants are also committed:

- [`training/configs/bc/phase2_bc_smoke.yaml`](/Users/rishivinodkumar/OrbitalShepherd/training/configs/bc/phase2_bc_smoke.yaml)
- [`training/configs/ppo/phase2_ppo_smoke.yaml`](/Users/rishivinodkumar/OrbitalShepherd/training/configs/ppo/phase2_ppo_smoke.yaml)
- [`training/configs/ppo/phase2_ppo_bc_warmstart_smoke.yaml`](/Users/rishivinodkumar/OrbitalShepherd/training/configs/ppo/phase2_ppo_bc_warmstart_smoke.yaml)

## BC Config Surface

`BehaviorCloningConfig` now controls:

- offline source planners through `source_planner_ids`
- train / validation split selection through `train_split` and `validation_split`
- dataset shaping through `top_k`, `limit_bundles_per_split`, `max_train_transitions`, and `max_validation_transitions`
- loss balancing through `class_weighting` and `class_weight_clip`
- optimization budget through `batch_size`, `epochs`, `learning_rate`, and `weight_decay`
- early stopping through `early_stopping.enabled`, `metric`, `patience`, and `min_delta`

BC training builds the required offline datasets directly from:

- `training_pack_path`
- `split_registry_path`
- `source_planner_ids`

That keeps the run reproducible without requiring a separate handwritten dataset manifest.

## Checkpoint Flow

BC checkpoints write:

- `policy_model_state.pt`
  - raw shared `OrbitalPhase2PolicyModel` weights
- `rllib_module/default_policy/...`
  - RLlib-compatible single-agent module checkpoint exported by the adapter layer
- `manifest.json`
  - typed `PolicyCheckpointManifest`

The indexed checkpoint manifest under `data/training/manifests/<run-dir>/checkpoint_*.json` records:

- source dataset ids
- source training pack id
- source bundle ids
- architecture compatibility metadata such as `top_k`, `hidden_dim`, `encoder_layers`, and `attention_heads`
- adapter paths for the raw policy weights and RLlib module checkpoint

PPO warm-start resolution:

1. load `PolicyInitializationConfig`
2. resolve a BC checkpoint manifest either directly or by `source_run_id`
3. verify architecture compatibility against the active PPO config
4. pass the exported `rllib_module_path` into `RLModuleSpec(load_state_path=...)`
5. let RLlib restore the module state before training begins

This keeps PPO initialization explicit and avoids patching weights into an already-running algorithm with ad hoc state surgery.

## Artifacts

BC writes:

- metrics JSONL: `data/training/reports/<run-dir>/metrics.jsonl`
- summary JSON: `data/training/reports/<run-dir>/summary.json`
- run manifest: `data/training/manifests/<run-dir>/run_manifest.json`
- checkpoint manifests: `data/training/manifests/<run-dir>/checkpoint_*.json`
- offline dataset build manifest referenced from the run manifest

PPO writes:

- metrics JSONL
- run manifest
- checkpoint manifests

Warm-start PPO checkpoint manifests also preserve BC provenance through:

- `source_dataset_ids`
- `metadata.initialization_mode`
- `metadata.warm_start_checkpoint_id`

## Comparison

Committed comparison configs:

- BC only: `training/configs/bc/phase2_bc.yaml`
- PPO scratch: `training/configs/ppo/phase2_ppo.yaml`
- BC -> PPO: `training/configs/ppo/phase2_ppo_bc_warmstart.yaml`

There is also a simple artifact summarizer:

```bash
python3 scripts/phase2_compare_learning_paths.py \
  --bc-run-id bc:phase2-bootstrap-v1 \
  --ppo-scratch-run-id ppo:phase2-online-v1 \
  --ppo-warmstart-run-id ppo:phase2-online-from-bc-v1
```

It finds the latest run manifests for those run ids and prints a JSON summary of:

- latest checkpoint ids
- latest checkpoint steps
- initialization provenance
- final report summary where present
- last metrics row

## Notes

- W&B credentials are optional. BC uses `maybe_init_wandb(...)`, which falls back to a local no-op run when disabled or unavailable.
- BC and PPO stay aligned through the shared model config and explicit checkpoint compatibility checks.
- For smoke runs, keep `top_k` and the model architecture identical across BC and PPO. Warm-start compatibility is intentionally strict.
