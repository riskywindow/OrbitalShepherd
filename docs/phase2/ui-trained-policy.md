# Phase 2 Trained-Policy UI

The Cesium mission-control UI now treats trained policies as first-class planner inputs instead of a side path.

## What Changed

### Planner runner

- The left control rail now supports two execution modes:
  - `Baseline`
  - `Trained Policy`
- Baselines still come from `/v1/baselines`.
- Trained checkpoints now come from `/v1/models`.
- Running a trained checkpoint calls `/v1/models/{model_key}/run` and loads the resulting replay into either the primary or compare lane.

### Replay lanes and compare mode

- The center stack now has a `Compare Deck`.
- Two lanes are supported:
  - `Primary`
  - `Compare`
- Each lane keeps:
  - its own replay source
  - its own scrub position
  - per-lane metrics
  - per-lane event slices
  - per-lane inference trace context
- The Cesium globe remains a single hero surface and follows the focused lane.
- Compare mode can mix:
  - API-produced trained-policy episodes
  - API-produced baseline episodes
  - report-backed replays loaded from local evaluation artifacts

### Inference telemetry

The right rail now exposes planner-trace details in a technical format:

- selected training slot
- canonical selected action ID
- top candidate slots with probabilities and logits
- action entropy
- value estimate
- legal-action count
- mask pressure

Telemetry is sourced from `/v1/episodes/{episode_id}/inference-traces` for API runs and `/v1/report-episodes/{report_episode_id}/inference-traces` for report-backed runs.

### Report browser

- The left rail now includes an evaluation report browser backed by local files.
- New API routes:
  - `/v1/reports`
  - `/v1/reports/{report_id}`
  - `/v1/report-episodes/{report_episode_id}`
  - `/v1/report-episodes/{report_episode_id}/events`
  - `/v1/report-episodes/{report_episode_id}/inference-traces`
- The browser can:
  - list evaluation reports
  - browse report episodes for the selected scenario
  - jump directly into notable episodes
  - load report episodes into the primary or compare lane

## Backend Support

The API now also:

- searches both `data/scenarios/` and `data/training/scenario_packs/` for scenario bundles
- exposes evaluation reports from `data/training/reports/`
- prefers `data/demo/phase2-defaults.json` over the Phase 1 defaults file when a trained-policy demo is prepared
- synthesizes replayable report-backed episode details so the frontend can use a single replay workflow for API and file-backed episodes

## Demo Path

The default Phase 2 local demo path is:

1. `make phase2-train`
2. `make phase2-eval`
3. `make phase2-demo-prepare`
4. `make phase2-demo`

`phase2-demo-prepare` writes `data/demo/phase2-defaults.json` after running the selected checkpoint
through the API service, so the UI can open directly onto a trained-policy replay without manual timestamp
lookup.

## Smoke Coverage

`apps/web/src/App.test.tsx` now covers:

- launching a trained-policy replay and surfacing inference telemetry
- loading report-backed policy and baseline replays into compare lanes

## Manual Verification

1. Start the API and the web app.
2. In `Run Control`, switch from `Baseline` to `Trained Policy`.
3. Select a checkpoint and run it into the `Primary` lane.
4. Confirm:
   - the globe stays active
   - the replay loads
   - `Inference Telemetry` shows slot/action/candidate/value fields
5. Enable `Compare Mode`.
6. Load a baseline into the `Compare` lane, either from:
   - `Saved API Replay`
   - `Report Browser`
7. Confirm:
   - both lanes show independent metrics and event stacks
   - the top bar shows a compare delta when both lanes reference the same bundle
   - focusing a lane moves the globe to that replay
8. In `Report Browser`, select `evalrun--phase2-heldout-smoke-v1` and load a notable episode into each lane.
9. Confirm report-backed replays behave like normal replays.

## Verification Performed

- `apps/web`: `./node_modules/.bin/vitest run`
- `apps/web`: `./node_modules/.bin/tsc -b && ./node_modules/.bin/vite build`

## Rough Edges

- Older report artifacts may not contain every newer inference field. The UI renders what is available and shows `n/a` where the artifact predates the richer trace schema.
- The globe only renders one lane at a time by design. Compare mode keeps the hero surface coherent by making the focused lane the globe source.
- The report browser currently targets evaluation summaries with replay artifacts. Training run summaries without episode/replay lists are intentionally not surfaced as replay sources.
