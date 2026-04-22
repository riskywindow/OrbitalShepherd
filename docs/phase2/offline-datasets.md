# Phase 2 Offline Datasets

Phase 2 offline datasets are built from actual planner rollouts against the deterministic orbital runtime.

Implementation lives in [`packages/training/src/orbital_shepherd_training/offline_dataset.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/training/src/orbital_shepherd_training/offline_dataset.py).

## Goals

- keep the dataset replay-auditable
- preserve canonical runtime actions and the projected training-slot mapping
- build train / val / test datasets from the committed split registry with no bundle leakage
- emit artifacts that are easy to inspect from the terminal and easy to consume in later BC code

## CLI

Build datasets:

```bash
python3 scripts/phase2_training.py build-offline-dataset \
  --training-pack-manifest data/training/manifests/phase2-training-pack-manifest.json \
  --split-registry training/configs/curriculum/phase2_splits.yaml \
  --split train \
  --planner urgency_greedy \
  --planner value_density_greedy \
  --planner ortools_receding_horizon
```

Inspect a dataset or build manifest:

```bash
python3 scripts/phase2_training.py inspect-offline-dataset \
  data/training/datasets/osbench-phase2-foundation-v1/<build-id>/train/manifest.json
```

## Canonical Schema

The canonical per-step schema is `OfflineTransitionRecord`.

Each transition row captures:

- episode metadata: `episode_id`, `bundle_id`, `manifest_id`, `scenario_family`, `simulation_seed`
- source planner identity: `planner_id`, `planner_version`, `planner_description`
- current simulator state snapshot: `observation`
- canonical action mask: `canonical_action_mask_id`, `canonical_actions`, `legal_action_count`
- projected training view: `global_features`, `candidate_features`, `training_action_mask`, `projected_slot_mapping`
- expert action choice: `selected_slot`, `selected_action_id`, `selected_action_type`, `selected_action_ref`, `selected_runtime_action_index`
- reward audit: `reward`, `reward_components`
- transition linkage: `step_index`, `next_step_index`, `transition_id`, `next_transition_id`
- termination flags: `terminated`, `truncated`

The per-episode schema is `OfflineEpisodeRecord`.

Each episode row captures:

- replay path and replay fingerprint
- episode fingerprint from the runtime replay stream
- planner seed, action count, total reward, and terminal flags

## Storage Layout

Each split dataset is written under:

`data/training/datasets/osbench-phase2-foundation-v1/<build-id>/<split>/`

Artifacts:

- `canonical/episodes.jsonl`: canonical episode table
- `canonical/steps.jsonl`: canonical step table
- `canonical/steps.parquet`: optional parquet mirror when `pyarrow` is installed
- `adapters/rllib_transitions.jsonl`: RLlib-style transition adapter
- `adapters/training_arrays.npz`: dense arrays for immediate BC consumption when `numpy` is available
- `replays/<planner>/<bundle>.ndjson`: replay trace for every compiled episode
- `dataset_card.md`: lightweight dataset card emitted alongside the manifest
- `manifest.json`: typed `OfflineDatasetManifest`

The top-level build writes a typed `OfflineDatasetBuildManifest` into `data/training/manifests/`.

## Split Discipline

Split assignment comes from the committed Phase 2 split registry.

Rules:

- every source bundle belongs to exactly one split
- train / val / test datasets are built only from bundles assigned to that split
- source bundle ids are written into each dataset manifest
- tests assert pairwise disjoint bundle sets across train / val / test

## Action Semantics

The runtime remains canonical.

- `canonical_actions` is the real legal action set emitted by `OrbitalEnv`
- `projected_slot_mapping` is the deterministic `OrbitalTrainingEnv` top-k projection
- `selected_runtime_action_index` maps the chosen training slot back to the canonical action list
- builds fail if an expert action falls outside the projected top-k view

This preserves auditability and prevents silent label corruption.

## Source Planners

Supported expert planners:

- `urgency_greedy`
- `value_density_greedy`
- `ortools_receding_horizon`

These are run directly against the environment. The dataset never invents planner labels from static heuristics or future information.

## Known Limitations

- Reward is the environment-emitted training reward, not the benchmark metric suite.
- The canonical audit store is JSONL. Parquet is emitted when the local environment has `pyarrow`.
- The RLlib artifact is an adapter export, not the source of truth.
- The dataset is single-agent and orbital-only for Phase 2.
