# Phase 1 Completion Report

## What was implemented

Phase 1 now behaves like a coherent local platform slice rather than a set of disconnected prototypes.

Implemented end-to-end:

- deterministic ephemeris fixture ingestion and compiled orbit asset bundle loading
- deterministic scenario-pack compilation under `data/scenarios`
- environment-side opportunity and downlink candidate materialization
- baseline planner execution through a shared action-mask contract
- canonical NDJSON replay emission
- replay-derived metric computation
- benchmark runner summaries under `data/benchmarks`
- FastAPI scenario, episode, baseline, replay, metrics, preview, and CZML endpoints
- React + Cesium globe replay UI
- demo defaults and deep-link support for loading a prepared replay directly in the UI
- scripted stack verification and integration coverage on the actual Phase 1 pack

## Scenario families included

The Phase 1 pack `osbench-phase1-pack-v1` includes three deterministic wildfire scenario families:

- `sparse_frontier`
  Quiet demand with sparse urgent incidents. Useful for checking planner discipline when little is happening.
- `burst_outbreak`
  Dense incident bursts with competing urgent targets. Useful for myopic triage pressure.
- `cloud_trap`
  Cloud-obstructed incidents and downlink timing pressure. Useful for exposing waste from poor observation timing.

The current built-in pack contains 10 bundled scenarios across those families.

## Available baselines

Implemented and exposed through both the benchmark runner and API:

- `random_valid_action`
  Deterministic seeded random legal-action sampler.
- `urgency_greedy`
  Myopic heuristic using incident urgency, predicted observation usefulness, cloud risk, slew cost, queue value, and downlink pressure.
- `value_density_greedy`
  Stronger deterministic heuristic with explicit value-density, freshness, quality, cloud-risk, retarget, and downlink-consequence terms.
- `ortools_receding_horizon`
  Small deterministic OR-Tools receding-horizon optimizer over the same observation and action contract.

Intentionally not included yet:

- RL policy adapters or training loops

## Determinism status

What is fixed and checked:

- identical scenario recipe inputs rebuild to identical bundle bytes and bundle fingerprints
- identical bundle + planner + seed produce the same episode fingerprint
- repeated benchmark runs with the same run id and config produce byte-stable replay artifacts and stable `summary.json`, `planner_summary.md`, and `planner_summary.csv`
- the runtime replay event sequence is already golden-tested on the toy environment and integration-tested on the real Phase 1 pack

What is intentionally not claimed as byte-stable:

- API episode metadata records and baseline job records use wall-clock timestamps for operator visibility
- prepared demo defaults are local generated artifacts, not immutable benchmark artifacts

## Known limitations

- No RL training, policy serving, or offline dataset generation pipeline exists yet.
- Opportunity generation remains Phase 1 coarse-grained and deterministic; it is not yet extracted into a richer standalone planner-candidate service.
- Storage is file-backed and local-only.
- The globe UI is replay-first and operationally useful, but still minimal:
  no tactical map, no multi-panel comparison mode, no annotation workflow, no remote imagery dependency.
- Full-pack benchmark performance is still weak, especially on urgent-incident service rate. That is a truthful baseline, not a presentation artifact.

## Developer onboarding notes

The cleanest entrypoints are now:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --all-groups
COREPACK_HOME=/tmp/corepack pnpm install
.venv/bin/python scripts/phase1_demo.py prepare
.venv/bin/python scripts/phase1_demo.py serve
```

Useful supporting commands:

```bash
.venv/bin/python scripts/run_phase1_api.py --host 127.0.0.1 --port 8000
.venv/bin/python scripts/verify_phase1_stack.py
.venv/bin/python scripts/run_phase1_benchmark.py run --run-id phase1-pack-v1
```

## Direct next steps for Phase 2

1. Freeze the Phase 1 pack and benchmark outputs as the non-learning baseline corpus for RL evaluation.
2. Add replay-to-training-dataset extraction so policy training consumes the same action and observation contract used by baselines.
3. Introduce a policy adapter that can act through the existing runtime loop without special-case environment logic.
4. Add train/eval split discipline for scenario seeds and benchmark reporting so RL claims remain comparable to the fixed baselines.
5. Keep the new strong non-learning baselines frozen and reproducible before claiming RL progress beyond them.
6. Expand the UI for side-by-side planner comparison once multiple serious planners exist.
