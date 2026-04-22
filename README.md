# Orbital Shepherd

Orbital Shepherd is a replay-first orbital mission-control platform for wildfire detection. The repository now contains:

- a deterministic Phase 1 scenario, benchmark, API, and UI stack
- a reproducible Phase 2 orbital-only RL workflow covering scenario packs, offline datasets, BC, PPO, evaluation, checkpoint registration, API execution, and UI replay
- a Phase 3 tactical contract foundation that adds typed regional artifacts, an explicit orbital-to-tactical bridge, and canonical tactical replay without introducing local tactical RL yet

Phase 1 established the product surface:

- deterministic scenario-pack compilation
- opportunity materialization inside the runtime loop
- baseline planners and benchmark execution
- canonical NDJSON replay artifacts
- benchmark metrics and stable planner summaries
- FastAPI orchestration for scenarios, episodes, baselines, and replay retrieval
- a Cesium globe UI that can open directly onto a prepared demo replay

Phase 2 hardens the learning system around the same contracts:

- deterministic Phase 2 scenario families and split registry
- offline expert dataset builds from real planner rollouts
- behavior cloning checkpoints with raw-policy and RLModule adapters
- PPO training with BC warm-start support and a local single-process fallback when Ray cannot bind local ports
- held-out evaluation reports with baseline comparisons and report manifests
- trained-policy discovery through the planner API and replay/browser support in the UI

## Phase 1 Quickstart

Install dependencies:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --all-groups
COREPACK_HOME=/tmp/corepack pnpm install
```

Prepare the full Phase 1 artifacts:

```bash
.venv/bin/python scripts/phase1_demo.py prepare
```

Start the local demo stack:

```bash
.venv/bin/python scripts/phase1_demo.py serve
```

Or through `make`:

```bash
make quickstart
```

The demo flow:

1. builds and validates the Phase 1 scenario pack under `data/scenarios`
2. runs the benchmark pack into `data/benchmarks/phase1-pack-v1`
3. creates a default API-visible replay episode for the UI
4. writes demo defaults to `data/demo/phase1-defaults.json`
5. starts the API and web app and prints the exact demo URL

## Validation

Run the stack verification script:

```bash
.venv/bin/python scripts/verify_phase1_stack.py
```

This verifies:

- scenario-pack build and deterministic rebuild validation
- benchmark summary and replay byte stability on repeated runs
- API scenario, baseline, replay, and metrics flow on the real Phase 1 pack

The verification summary is written to `data/demo/phase1-verification.json`.

## Phase 2 Quickstart

Build the reproducible local RL smoke flow:

```bash
make phase2-smoke
```

This runs the smallest practical end-to-end validation path:

1. builds the deterministic Phase 2 training pack
2. builds an offline expert dataset slice
3. trains BC on the offline slice
4. trains PPO from the BC checkpoint
5. evaluates the trained checkpoint against the baseline set
6. loads the checkpoint through the API service and fetches replay/inference traces

The verification summary is written to `data/demo/phase2-verification.json`.

For persistent local artifacts instead of an isolated smoke workspace:

```bash
make phase2-train
make phase2-eval
make phase2-demo-prepare
```

`make phase2-train` writes stable manifest aliases:

- `data/training/manifests/phase2-bc-latest.json`
- `data/training/manifests/phase2-ppo-latest.json`

`make phase2-eval` writes the held-out report under:

- `data/training/reports/evalrun--phase2-heldout-smoke-v1`

`make phase2-demo-prepare` writes:

- `data/demo/phase2-defaults.json`

To launch the local demo stack after artifacts exist:

```bash
make phase2-demo
```

## Key Commands

```bash
make scenario-pack-build
make scenario-pack-validate
make benchmark-run BENCH_ARGS="run --run-id phase1-pack-v1"
make api-dev
make web-build
make phase1-prepare
make phase1-verify
make phase2-smoke
make phase2-train
make phase2-eval
make phase2-demo
make demo
```

## Repository Layout

```text
apps/
  api/                     FastAPI orchestration and replay-serving layer
  web/                     React + Vite + Cesium mission-control UI
packages/
  benchmark/               Baselines, metrics, and benchmark runner
  contracts/               Canonical schema and validation adapters
  core/                    Deterministic JSON, fingerprints, timestamps, IDs
  env_runtime/             Deterministic single-agent orbital runtime
  escalation_bridge/       Typed IncidentPacket handoff into tactical activation artifacts
  ephemeris/               Deterministic orbit asset ingestion and sampling
  geo_artifacts/           Spatial artifact helpers for tactical maps and ingest provenance
  ground_env/              Tactical ground-response environment scaffolding
  opportunity_builder/     Reserved package surface for future extracted logic
  region_builder/          Deterministic region manifest and bundle compiler scaffolding
  routing_engine/          Planner-agnostic tactical route planning surface
  scenario_engine/         Phase 1 scenario family compiler and pack builder
  tactical_baselines/      Tactical heuristic and OR baseline scaffolding
  tactical_metrics/        Replay-derived tactical metric surface
  tactical_scenario_engine/ Tactical scenario manifest and bundle scaffolding
data/
  api/                     API episode and baseline-run records
  benchmarks/              Benchmark summaries and per-episode reports
  demo/                    Generated demo defaults and verification summaries
  fixtures/                Local deterministic fixture inputs
  replays/                 Benchmark replay artifacts
  scenarios/               Compiled Phase 1 scenario bundles
docs/
  phase1/                  Service, UI, runtime, and completion docs
  phase2/                  Training, evaluation, UI, and completion docs
  phase3/                  Tactical architecture, artifact graph, ADR, and glossary
tests/
  api/
  benchmark/
  env_runtime/
  integration/
  scenario_engine/
```

## Scope

Phase 1 included:

- wildfire-only scenario families
- deterministic orbital replay loop
- `random_valid_action`, `urgency_greedy`, `value_density_greedy`, and `ortools_receding_horizon` baselines
- file-backed API and UI stack

Phase 2 adds:

- orbital-only RL scenario families and split discipline
- offline dataset generation from baseline planners
- behavior cloning and PPO training flows
- checkpoint manifests and trained-policy API/UI replay support
- held-out evaluation reports and demo defaults

Phase 3 now formalizes:

- deterministic regional ingest and region bundle artifacts
- explicit `IncidentPacket` to `TacticalActivation` handoff contracts
- planner-agnostic tactical scenario, routing, replay, and metrics contracts
- package scaffolding for later tactical engines, baselines, and bridge logic

Still not included:

- tactical street-level map or imagery workflows
- remote storage, message buses, or distributed orchestration
- non-orbital RL domains or notebook-only training paths
- hierarchical RL or local tactical policy training

## Recommended Reading

1. `01_mission_rfc.md`
2. `02_architecture.md`
3. `03_benchmark_spec.md`
4. `04_data_contracts.md`
5. `docs/phase1/scenario-engine.md`
6. `docs/phase1/env-runtime.md`
7. `docs/phase1/benchmark-runner.md`
8. `docs/phase1/api.md`
9. `docs/phase1/ui.md`
10. `docs/phase1/phase1-completion-report.md`
11. `docs/phase2/training-foundation.md`
12. `docs/phase2/offline-datasets.md`
13. `docs/phase2/bc-and-finetune.md`
14. `docs/phase2/evaluation.md`
15. `docs/phase2/ui-trained-policy.md`
16. `docs/phase2/phase2-completion-report.md`
17. `docs/phase3/ADR-0001-phase3-tactical-scope.md`
18. `docs/phase3/architecture-overview.md`
19. `docs/phase3/artifact-graph.md`
20. `docs/phase3/glossary.md`
# OrbitalShepherd
