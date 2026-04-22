# Phase 2 Evaluation

Phase 2 evaluation compares held-out benchmark performance, not training reward.

The evaluation runner stays planner-agnostic at the reporting layer:

- built-in planners run against the raw simulator interface
- trained policies run through the Phase 2 projected-observation interface
- both paths emit the same replay artifacts and the same replay-derived benchmark metrics

## What Gets Compared

Supported planners:

- `random_valid_action`
- `urgency_greedy`
- `value_density_greedy`
- `ortools_receding_horizon`
- one or more trained policy checkpoints

Supported splits:

- `val`
- `test`
- `ood`

The source of truth for split assignment remains:

- training pack manifest: `data/training/manifests/phase2-training-pack-manifest.json`
- split registry: `training/configs/curriculum/phase2_splits.yaml`

## Metrics

Reports use benchmark metrics derived from replay artifacts:

- time to first useful observation
- useful observation value captured
- cloud waste rate
- downlink latency
- missed urgent incident rate
- opportunity utilization efficiency
- mission utility

Reward is kept as a separate audit trail. It is never used to rank planners.

## Statistical Reporting

For each split and planner, the evaluator emits:

- metric means
- medians
- min and max
- population standard deviation
- bootstrap confidence intervals
- missing-sample counts for metrics such as downlink latency when an episode has no packets

For each matched planner pair on the same bundle set, the evaluator emits:

- mean paired difference per metric
- bootstrap confidence interval on the paired difference
- win/loss/tie counts by episode
- per-episode outcome rows in CSV for direct inspection

## Notable Episode Slices

The evaluator surfaces:

- biggest RL wins versus the best baseline on the primary metric
- biggest RL losses versus the best baseline on the primary metric
- cloud-heavy episodes
- downlink-heavy episodes
- outage/degradation episodes

These slices point back to the underlying replay and episode-summary files for each planner.

## Artifacts

Each evaluation run writes under:

- reports: `data/training/reports/<eval-run-id>/`
- manifests: `data/training/manifests/<eval-run-id>/`

Key report artifacts:

- `summary.json`
- `summary.md`
- `episode_metrics.csv`
- `pairwise_comparisons.csv`
- `episode_outcomes.csv`
- `notable_episodes.csv`
- `dvc_metrics.json`
- `dvc_plots/planner_metric_means.csv`
- `dvc_plots/pairwise_primary_metric.csv`
- `episodes/<split>/<planner>/...json`
- `replays/<split>/<planner>/...ndjson`

Key manifest artifacts:

- `run_manifest.json`
- `<split>--<planner>.json` per `EvaluationReportManifest`

`summary.json` includes absolute replay and episode-summary paths so any aggregate can be traced back to the underlying episode artifacts without opening notebooks.

## Running It

Example: compare all baselines plus a trained PPO checkpoint on one bundle from each held-out split.

```bash
python scripts/phase2_training.py evaluate \
  --config training/configs/evaluation/phase2_eval.yaml \
  --training-pack-manifest data/training/manifests/phase2-training-pack-manifest.json \
  --split-registry training/configs/curriculum/phase2_splits.yaml \
  --checkpoint-manifest data/training/manifests/trainrun--ppo-phase2-online-from-bc-smoke-v1--2026-04-13t23-01-03.244114z/checkpoint_000002.json \
  --policy-label ppo_from_bc_smoke \
  --split val \
  --split test \
  --split ood \
  --limit-bundles-per-split 1
```

You can also pin exact bundles:

```bash
python scripts/phase2_training.py evaluate \
  --checkpoint-manifest <checkpoint-manifest> \
  --bundle-id sb:osbench-phase2-foundation-v1:sparse_frontier:seed-403 \
  --bundle-id sb:osbench-phase2-foundation-v1:cloud_trap:seed-424
```

## Reading Results

Recommended order:

1. Open `summary.md` for the split-by-split comparison.
2. Open `pairwise_comparisons.csv` for explicit paired deltas and CIs.
3. Open `notable_episodes.csv` to jump to the biggest wins, losses, and stress cases.
4. Follow the replay paths in `summary.json` or `episode_metrics.csv` when a metric needs explanation.

This keeps the evaluation story inspectable from files alone, with no notebook dependency.
