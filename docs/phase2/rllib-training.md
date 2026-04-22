# Phase 2 RLlib Training

Phase 2 online RL now uses an RLlib-first PyTorch policy stack, with a single-process PPO fallback
for restricted environments where Ray cannot open local ports.

Implementation spans:

- [`packages/policy_models/src/orbital_shepherd_policy_models/phase2_policy.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/policy_models/src/orbital_shepherd_policy_models/phase2_policy.py)
- [`packages/training/src/orbital_shepherd_training/rllib_env.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/training/src/orbital_shepherd_training/rllib_env.py)
- [`packages/training/src/orbital_shepherd_training/rllib_module.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/training/src/orbital_shepherd_training/rllib_module.py)
- [`packages/training/src/orbital_shepherd_training/rllib_training.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/training/src/orbital_shepherd_training/rllib_training.py)
- [`training/configs/ppo/phase2_ppo.yaml`](/Users/rishivinodkumar/OrbitalShepherd/training/configs/ppo/phase2_ppo.yaml)
- [`training/configs/ppo/phase2_ppo_smoke.yaml`](/Users/rishivinodkumar/OrbitalShepherd/training/configs/ppo/phase2_ppo_smoke.yaml)

## Versions

The scaffold was exercised locally with:

- `ray[rllib]==2.54.1`
- `torch==2.11.0`
- `gymnasium==1.2.2`

## Integration Choice

This stack uses RLlib's current builder-style API in the installed version:

- `PPOConfig().api_stack(...).environment(...).env_runners(...).training(...).evaluation(...).rl_module(...)`
- `RLModuleSpec`
- a custom `TorchRLModule` + `ValueFunctionAPI`

The implementation intentionally does **not** use the deprecated `PPOTorchRLModule` alias. The policy is wired through a custom RLModule so the same PyTorch model can be reused later by behavior cloning code without depending on RLlib internals.

## Environment Registration

`register_orbital_rllib_env()` registers `orbital_shepherd_phase2_training_env_v1` with RLlib.

`RllibOrbitalTrainingEnv`:

- loads one or more compiled scenario bundles from the training-pack manifest
- rotates bundles deterministically by worker/env index and episode count
- wraps the existing `OrbitalTrainingEnv`
- preserves the explicit observation contract:
  - `global_features`
  - `candidate_features`
  - `action_mask`
- adds bundle metadata into `info`
- tracks lightweight episode metrics for RLlib callbacks

The runtime action semantics remain canonical. RLlib only sees projected training slots.

## Policy Architecture

`OrbitalPhase2PolicyModel` is the shared PyTorch model used by PPO now and intended for BC later.

Structure:

1. `global_features -> global encoder`
2. `candidate_features -> shared candidate encoder`
3. `[global token | candidate tokens] -> transformer/set encoder`
4. pooled candidate context + global context -> shared policy context
5. policy heads:
   - per-candidate logits over projected slots
   - separate noop logit
6. explicit `action_mask` application in policy logic
7. value head from the shared policy context

Design properties:

- candidate encoder is shared across all slots
- masking happens inside the policy, not as a wrapper-side side effect
- the value path shares the same contextual representation as the policy path
- outputs are reusable outside RLlib

## Checkpoints And Metadata

Each run writes:

- checkpoints under `artifact_layout.checkpoint_root/<run-id>/checkpoint_*`
- per-checkpoint manifests both:
  - inside the checkpoint directory as `manifest.json`
  - under `artifact_layout.manifest_root/<run-id>/checkpoint_*.json`
- a run-level manifest as `run_manifest.json`
- iteration metrics as JSONL under `artifact_layout.report_root/<run-id>/metrics.jsonl`

Checkpoint manifests record:

- algorithm and framework
- benchmark, run, model, and reward ids
- source training-pack id
- selected bundle ids
- global step
- flattened numeric training metrics
- reproducibility metadata such as seed, `top_k`, rollout/minibatch settings, the run manifest path,
  and the execution backend
- raw-policy and RLModule adapter paths when available, so trained checkpoints stay loadable through
  the API without notebook-only restore code

## Local Smoke Run

Build the deterministic Phase 2 training pack first:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase2_training.py build-training-pack
```

Run the checked-in PPO smoke config:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase2_training.py train-ppo \
  --config training/configs/ppo/phase2_ppo_smoke.yaml
```

The smoke config keeps execution local-friendly:

- `num_env_runners: 0`
- `num_envs_per_env_runner: 1`
- `num_gpus: 0`
- `scenario_limit: 2`
- short rollout and timestep budgets

If Ray can bind local sockets, the training path uses RLlib directly. If the host environment blocks
that startup path, `train_ppo_with_rllib()` falls back to a deterministic local PPO loop that preserves
the same checkpoint manifest contract and trained-policy API surface.

## Tests

Training coverage includes:

- [`tests/training/test_policy_models.py`](/Users/rishivinodkumar/OrbitalShepherd/tests/training/test_policy_models.py)
- [`tests/training/test_rllib_training.py`](/Users/rishivinodkumar/OrbitalShepherd/tests/training/test_rllib_training.py)

These validate:

- model forward shape and value-path behavior
- explicit action masking
- env/module compatibility
- short end-to-end PPO smoke training

## Version-Specific Notes

Observed with `ray==2.54.1`:

- `AlgorithmConfig.build()` is deprecated, so the scaffold uses `build_algo()`.
- RLlib still emits some internal deprecation warnings around its default logger stack.
- `local_mode=True` is also warned as experimental by Ray itself; the scaffold keeps it configurable rather than relying on it as the default path.

These are framework warnings, not blockers for the current Phase 2 scaffold.
