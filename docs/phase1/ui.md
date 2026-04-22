# Phase 1 Globe UI

## Purpose

The Phase 1 web app provides a mission-control replay surface for Orbital Shepherd.
It replaces notebook-only inspection with a Cesium-based globe, scenario selection,
baseline execution, replay scrubbing, event review, and metrics summary panels.

The app lives in [`apps/web`](/Users/rishivinodkumar/OrbitalShepherd/apps/web) and talks to
the FastAPI service in [`apps/api`](/Users/rishivinodkumar/OrbitalShepherd/apps/api).

## Implemented flow

1. Start the API and web app locally.
2. Choose a scenario bundle from the selector.
3. Choose a baseline planner from the selector.
4. Start a baseline run.
5. Load the resulting replay episode.
6. Scrub or play the timeline.
7. Inspect:
   - globe overlays for available targets, selected opportunities, downlinks, and degraded or missed outcomes
   - replay event panel for major decisions and outcomes
   - metrics summary for utility, packet count, waste, latency, and missed-incident rates

## Local run

Install Python and web dependencies:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --all-groups
COREPACK_HOME=/tmp/corepack pnpm install
```

Start the API:

```bash
.venv/bin/python scripts/run_phase1_api.py --host 127.0.0.1 --port 8000
```

Start the web app:

```bash
COREPACK_HOME=/tmp/corepack pnpm --dir apps/web dev --host 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000`.

If the Phase 1 demo has been prepared with `scripts/phase1_demo.py prepare`, the UI will:

- fetch `GET /v1/demo/defaults`
- preselect the default scenario and baseline
- auto-load the prepared replay episode when available

The UI also accepts deep-link query params:

- `?scenario=<bundle_id>`
- `&baseline=<baseline_id>`
- `&episode=<episode_id>`

Optional environment override:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Build and test

Smoke test:

```bash
COREPACK_HOME=/tmp/corepack pnpm --dir apps/web test
```

Production build:

```bash
COREPACK_HOME=/tmp/corepack pnpm --dir apps/web build
```

## Screenshot checklist

Capture these three states for docs or demo notes:

1. Scenario loaded, before baseline run:
   - selectors visible
   - globe populated with satellites, ground stations, and target cells
2. Replay mid-run:
   - scrubber at an observation or downlink event
   - event panel showing the corresponding action and execution entries
3. Replay complete:
   - metrics panel visible
   - final mission utility and packet counts visible

On macOS, one simple path is:

1. Run the API and web app.
2. Open `http://127.0.0.1:3000`.
3. Use the built-in screenshot tool with `Shift` + `Command` + `4`.

## UI notes

- Satellites are rendered through Cesium `CzmlDataSource`.
- The API exposes `GET /v1/scenarios/{bundle_id}/trajectory-czml`, backed by deterministic local ephemeris sampling.
- Ground stations and target overlays are rendered as Cesium entities on top of the trajectory layer.
- Replay state is derived from canonical NDJSON replay events, not from notebook-side transforms.

## Current rough edges

- The globe uses a dark, local-safe ellipsoid presentation instead of remote imagery, to avoid network dependence.
- The replay panel intentionally filters out noisy per-tick events such as every action mask emission.
- Scrubbing is tied to replay ticks rather than a freeform continuous time cursor.
- Demo defaults are generated locally and are intentionally file-backed rather than multi-user or remotely shared.
