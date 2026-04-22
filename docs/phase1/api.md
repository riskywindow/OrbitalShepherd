# Phase 1 API

The Phase 1 API lives in `apps/api` and is implemented with FastAPI. It keeps the Phase 0
planner paths and field names intact while adding a small set of read endpoints for the UI.

## Contract alignment

Implemented Phase 0 endpoints:

- `GET /v1/health`
- `POST /v1/scenarios`
- `GET /v1/scenarios/{bundleId}`
- `POST /v1/episodes`
- `POST /v1/episodes/{episodeId}/step`
- `GET /v1/episodes/{episodeId}/events`
- `GET /v1/episodes/{episodeId}/metrics`
- `POST /v1/baselines/{baselineId}/run`

Scenario registration accepts:

- canonical Phase 1 `ScenarioBundle` payloads
- legacy Phase 0 bundle-shaped payloads, which are normalized into canonical Phase 1 bundles

All stored scenarios are persisted as canonical Phase 1 bundles and validated through the
contracts package.

Replay event streams are exposed as `application/x-ndjson`.

## UI-oriented extensions

Non-breaking additive endpoints:

- `GET /v1/scenarios`
- `GET /v1/scenarios/{bundleId}/preview`
- `GET /v1/scenarios/{bundleId}/trajectory-czml`
- `GET /v1/episodes`
- `GET /v1/episodes/{episodeId}`
- `GET /v1/baselines`
- `GET /v1/baseline-runs/{jobId}`

Non-breaking response additions:

- `POST /v1/episodes` returns `runtime_episode_id` in addition to `episode_id`
- `POST /v1/episodes/{episodeId}/step` returns `reward`, `mission_utility`,
  `observation`, `action_mask`, `events`, and `runtime_episode_id`
- `POST /v1/baselines/{baselineId}/run` returns `status` and `episode_id` in addition to
  `job_id`

These additions are intended to reduce the number of follow-up requests needed by the Cesium UI.
The trajectory endpoint returns CZML packets generated from the deterministic local
ephemeris bundle so the web app can render time-dynamic orbital paths without carrying
its own propagation stack.

## Storage model

Phase 1 keeps orchestration local and file-backed:

- scenarios: `data/scenarios/*.json`
- API episode records: `data/api/episodes/*.json`
- API replay streams: `data/api/episodes/*.ndjson`
- baseline run records: `data/api/baseline_runs/*.json`

Episode stepping is intentionally simple. The service reconstructs runtime state from the
stored action history on each step request, then rewrites and persists the canonical replay
stream and metrics snapshot.

Baseline runs reuse the existing planner, runtime, and benchmark metric modules and persist the
completed episode so the UI can fetch replay and metrics through the same episode endpoints.

## Validation and errors

- scenario inputs and stored outputs are validated with `validate_canonical_bundle`
- replay outputs are validated with `validate_canonical_replay_events`
- metrics and API request/response payloads use Pydantic models
- explicit JSON errors are returned for not found, contract errors, invalid actions, and closed
  episodes

## Running

Install dependencies and start the API:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --all-groups
.venv/bin/python scripts/run_phase1_api.py --host 127.0.0.1 --port 8000
```

The wrapper script bootstraps the repo source roots, so a new engineer does not have to
manually export `PYTHONPATH`.

## Demo defaults

`GET /v1/demo/defaults` exposes the current local demo entrypoint for the web app.

If `data/demo/phase1-defaults.json` exists, the API returns:

- the prepared scenario bundle id
- baseline id
- prepared episode id
- benchmark summary path
- replay path
- a ready-to-use UI query string

If the file does not exist, the endpoint falls back to the first available scenario and the
`urgency_greedy` baseline without an episode.

FastAPI-generated docs are available at:

- `/docs`
- `/openapi.json`

## Tests

API coverage lives in `tests/api/test_phase1_api.py` and covers:

- health
- scenario registration and retrieval
- episode lifecycle
- replay and metrics retrieval
- baseline run orchestration
