# Orbital Shepherd Phase 1 — Detailed Plan and Codex Prompt Pack

## What Phase 1 should achieve

Phase 1 is not “train the RL agent.”

Phase 1 is the full platform slice that makes RL worth adding later:
- a deterministic orbital mission-control backend,
- canonical validated artifacts,
- a scenario compiler,
- opportunity generation,
- a step-driven environment runtime,
- replay logs and metrics,
- two non-learning baselines,
- and a Cesium-powered globe UI that can replay decisions.

The end state of Phase 1 should satisfy the Phase 0 benchmark target:
- at least 10 deterministic episodes,
- at least 3 scenario families,
- random and urgency-greedy baselines,
- replay export,
- and a visible metric report in the UI.

## The important Phase 0 cleanup before coding

The Phase 0 package is strong, but it has a few contract inconsistencies that should be normalized deliberately instead of hand-waved in code:

1. `ScenarioBundle` is referenced in the RFC, architecture doc, and OpenAPI, but the provided example/schema is a `ScenarioManifest`.
2. The replay event taxonomy in the RFC differs from the provided replay-event schema enum.
3. The data-contracts doc lists more schemas than are actually present in the package.

### Resolution strategy

Use this normalization in Phase 1:
- `ScenarioManifest`: authoring / recipe artifact.
- `ScenarioBundle`: compiled immutable episode artifact consumed by the simulator and API.
- `ReplayEvent`: canonical Phase 1 event envelope, with a compatibility alias layer for any older event names from the Phase 0 examples.

Add the missing schemas in Phase 1 and preserve backward compatibility for the provided examples where practical.

## Recommended Phase 1 architecture

### Stack
- **Backend:** Python 3.11+, FastAPI, Pydantic v2, pytest, ruff, mypy.
- **Frontend:** React + TypeScript + Vite + CesiumJS.
- **Contracts:** JSON Schema 2020-12 as source of truth, with generated/validated Python and TypeScript models.
- **Geospatial:** H3.
- **Storage:** immutable JSON bundles + NDJSON replay logs + lightweight metadata registry.
- **Propagation:** a `PropagationBackend` abstraction; Phase 1 may use a lightweight local backend and/or fixtures, but must preserve a clean upgrade path to an Orekit sidecar.

### Design stance
- replay-first,
- deterministic by default,
- fixture-backed tests,
- live data adapters allowed but never required for tests,
- candidate-set and reward attribution visible in logs.

## Definition of done for Phase 1

A correct Phase 1 implementation should let you do the following:

1. Build or register a scenario pack with at least 10 deterministic scenarios across `Sparse Frontier`, `Burst Outbreak`, and `Cloud Trap`.
2. Start an episode from a scenario bundle through the API.
3. Step the episode with `random` and `urgency_greedy` planners.
4. Emit validated replay events to NDJSON.
5. Compute metric summaries for each episode.
6. Load the episode into a Cesium globe UI and scrub/play through time.
7. View scenario metadata, replay events, and metrics without opening notebooks.

## Proposed Phase 1 repository shape

```text
orbital-shepherd/
  apps/
    api/
    web/
  packages/
    contracts/
    core/
    scenario_engine/
    opportunity_builder/
    env_runtime/
    benchmark/
  data/
    fixtures/
    scenarios/
    replays/
  docs/
    phase0/
    phase1/
  scripts/
  tests/
```

## Run order for the Codex sessions

1. Foundation scaffold
2. Contract audit, schema completion, and validators
3. Ephemeris adapter and propagation abstraction
4. Scenario engine and benchmark fixture packs
5. Opportunity builder
6. Environment runtime and canonical replay events
7. Baselines and benchmark runner
8. Planner API and replay/metrics service
9. Cesium mission-control UI
10. End-to-end integration, determinism hardening, and demo polish

---

# Copy-paste Codex prompts

Each prompt below is written to be executable in a fresh Codex session. They assume the repository already contains prior merged work, but they do **not** assume chat history. Run them in order.

---

## Prompt 1 — Foundation scaffold

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Start by inspecting the repo and any existing Phase 0 artifacts. If there are Phase 0 docs under paths like `orbital_shepherd_phase0/`, `docs/phase0/`, or the repo root, treat them as the source of truth. If they are missing, use this locked contract summary:
- Orbital Shepherd is an event-sourced, replayable orbital mission-control system for wildfire detection.
- UTC only, 60-second control interval, WGS84 external coordinates, H3 target cells.
- Phase 1 goal: deterministic scenarios, opportunity generation, env runtime, random + urgency-greedy baselines, replay logs, metrics, and a Cesium globe UI.
- No imagery/CV, no street-level tactical layer, no real-time streaming.

Your task is to create the Phase 1 engineering foundation.

Objectives:
1. Establish a clean repo structure for:
   - `apps/api`
   - `apps/web`
   - `packages/contracts`
   - `packages/core`
   - `packages/scenario_engine`
   - `packages/opportunity_builder`
   - `packages/env_runtime`
   - `packages/benchmark`
   - `data/fixtures`, `data/scenarios`, `data/replays`
   - `docs/phase1`
2. Set up Python tooling for backend/packages:
   - `pyproject.toml`
   - ruff
   - mypy
   - pytest
3. Set up frontend tooling for `apps/web`:
   - React + TypeScript + Vite
   - eslint + prettier
4. Add top-level dev ergonomics:
   - `Makefile` or `justfile`
   - `.editorconfig`
   - `.gitignore`
   - `.env.example`
   - developer README with install/run/test instructions
5. Add shared deterministic utilities in `packages/core`:
   - canonical JSON serialization helper
   - SHA-256 fingerprint helper
   - UTC timestamp parsing/formatting helper
   - seeded RNG helper
   - stable ID helper
6. Add a lightweight `docker-compose.yml` only if it meaningfully improves local dev; otherwise keep the stack simple.

Constraints:
- Do not implement mission logic yet.
- Keep the structure modular so later prompts can add code cleanly.
- Prefer simple local file-backed storage in Phase 1.
- If the repo already contains some of this, improve and normalize rather than rewriting from scratch.

Deliverables:
- repo scaffold
- tooling configs
- shared deterministic utility module
- docs/phase1/engineering-foundation.md
- passing lint/type/test smoke checks

Acceptance criteria:
- `pytest` runs without import errors
- Python lint/type checks pass on the new foundation code
- frontend installs and builds successfully
- README explains how to start the API and web app later

At the end:
- run the relevant checks
- summarize changed files
- list any assumptions made
```

---

## Prompt 2 — Contract audit, schema completion, and validators

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Start by inspecting the repo and the Phase 0 package/docs. There are known contract inconsistencies to resolve carefully:
- docs/API refer to `ScenarioBundle`, but the provided example/schema is `ScenarioManifest`
- the replay event taxonomy in the docs differs from the example replay-event schema
- several schemas referenced in the docs are missing from the package

Your task is to normalize the contracts and make them executable.

Objectives:
1. Audit the current Phase 0 contracts and write `docs/phase1/ADR-0001-contract-normalization.md` that:
   - identifies the mismatches
   - explains the chosen normalization
   - preserves backward compatibility where practical
2. Define canonical Phase 1 artifact types:
   - `ScenarioManifest` = authoring recipe
   - `ScenarioBundle` = compiled immutable simulator artifact
   - `ReplayEvent` = canonical append-only event envelope
3. Complete the missing JSON Schemas under `packages/contracts/schemas/`:
   - satellite
   - ground_station
   - target_cell
   - incident
   - downlink_window
   - scenario_bundle
   - keep or update scenario_manifest, replay_event, observation_opportunity, incident_packet
4. Use JSON Schema 2020-12 consistently.
5. Add Python validation/models in `packages/contracts/python/` using Pydantic v2.
6. Add TypeScript types/utilities in `packages/contracts/typescript/` or a sensible equivalent.
7. Add a validation CLI/script that validates all example artifacts and fixtures.
8. Add compatibility adapters:
   - old replay event names -> canonical Phase 1 event names
   - scenario manifest -> scenario bundle compilation boundary

Constraints:
- Do not silently delete old artifacts.
- Preserve the spirit of the Phase 0 package.
- Favor explicit aliases and adapters over hidden magic.
- Keep the schemas readable and well-documented.

Deliverables:
- completed schemas
- validation utilities
- generated/hand-authored Python and TS contract types
- ADR documenting the normalization
- tests that validate the provided examples and any new examples

Acceptance criteria:
- all example JSON artifacts validate
- schemas are versioned and internally consistent
- canonical `ScenarioBundle` exists and is ready for the API/env runtime
- the normalization ADR is clear enough that another engineer would not be confused by the old package

At the end:
- run tests/validation
- summarize the canonical artifacts and event names
```

---

## Prompt 3 — Ephemeris adapter and propagation abstraction

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state and the normalized contracts. Build the orbital ingest layer for Phase 1.

The design intent is:
- CelesTrak OMM/GP snapshots are the canonical external ephemeris input.
- The simulator must remain deterministic.
- Propagation must be behind an abstraction so Phase 1 can use a practical local backend/fixtures while preserving a clean path to a higher-fidelity Orekit sidecar later.

Your task is to implement the ephemeris service/library slice.

Objectives:
1. Create a `PropagationBackend` interface in Python with methods for:
   - loading a normalized orbital asset snapshot
   - sampling satellite state over a time window
   - computing coarse visibility/contact facts needed later by the opportunity builder
2. Implement a practical Phase 1 backend for local development and tests.
   - It can be fixture-backed and/or use a lightweight local propagation implementation.
   - It must be deterministic and easy to run in CI.
3. Create a `CelesTrakClient` or equivalent adapter that:
   - fetches or loads OMM-style snapshots
   - normalizes satellite identifiers
   - writes immutable raw snapshots under `data/fixtures/ephemeris/raw/` or similar
   - compiles normalized orbit assets under a stable schema
4. Add a CLI/script such as:
   - `fetch-ephemeris`
   - `compile-orbit-assets`
5. Define the orbit asset format consumed by the scenario engine.
6. Add golden fixtures for at least a few satellites suitable for demo/test use.
7. Add tests for:
   - parsing/normalization
   - deterministic output from the propagation backend
   - stable asset fingerprints

Constraints:
- Keep tests independent of live network calls.
- Live fetch support is allowed, but tests must run from fixtures.
- Do not overbuild exact spacecraft physics in Phase 1.
- Document how an Orekit-backed implementation would plug in later.

Deliverables:
- ephemeris package/module
- CelesTrak adapter
- normalized orbit asset schema/model
- fixture snapshots and compiled assets
- CLI commands
- docs/phase1/ephemeris-service.md

Acceptance criteria:
- can produce a normalized orbit asset bundle from fixtures
- repeated runs yield identical compiled assets and fingerprints
- tests cover the parsing and deterministic sampling path

At the end:
- run the relevant tests
- summarize the chosen backend and the future Orekit integration seam
```

---

## Prompt 4 — Scenario engine and benchmark fixture packs

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the repo and use the normalized contracts plus ephemeris layer. Your task is to implement the deterministic scenario engine for Phase 1.

Phase 1 only needs orbital wildfire scenarios. The required scenario families are:
- Sparse Frontier
- Burst Outbreak
- Cloud Trap

Your task is to create a scenario compiler that turns authoring recipes into compiled immutable scenario bundles.

Objectives:
1. Implement `ScenarioManifest -> ScenarioBundle` compilation.
2. Add deterministic generators/adapters for:
   - wildfire incidents
   - weather / cloud risk
   - target-cell demand aggregated to H3
3. Support three scenario modes:
   - fixture-backed historical-style replay
   - synthetic
   - optional live adapter boundary (not required for tests)
4. Define scenario family parameterizations for:
   - Sparse Frontier
   - Burst Outbreak
   - Cloud Trap
5. Generate at least 10 deterministic scenarios across those 3 families.
   - Example split: 4 sparse, 3 burst, 3 cloud-trap
6. Persist compiled scenario bundles under `data/scenarios/` with stable IDs and fingerprints.
7. Add a CLI/script such as:
   - `build-scenario-pack`
   - `validate-scenario-pack`
8. Document the scenario configuration knobs clearly.

Important modeling guidance:
- Do not simulate raw imagery.
- Model observation usefulness as a function of geometry, cloud risk, urgency, and delay.
- Keep the world state deterministic and replayable.
- The scenario bundle should be directly consumable by the opportunity builder and env runtime.

Deliverables:
- scenario engine module/package
- scenario manifest/bundle compiler
- benchmark fixture pack with at least 10 scenarios
- CLI commands
- docs/phase1/scenario-engine.md

Acceptance criteria:
- all compiled bundles validate against schema
- the same seed produces the same scenario bytes/fingerprint
- the scenario pack clearly contains the 3 required families
- a developer can build the pack locally without live network calls

At the end:
- run validation/tests
- print the generated scenario IDs and families
```

---

## Prompt 5 — Opportunity builder

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the repo and use the compiled scenario bundles plus ephemeris layer. Your task is to implement the opportunity builder.

This module must take a `ScenarioBundle` and generate the candidate action substrate for the orbital controller:
- observation opportunities
- downlink windows
- per-tick candidate sets or equivalent precomputations

Objectives:
1. Implement a package/module that generates `ObservationOpportunity` records from:
   - satellite state / visibility
   - target-cell incident demand
   - cloud risk / predicted usability
   - retarget/slew proxy cost
2. Implement a `DownlinkWindow` model and generator from:
   - satellite state
   - ground-station availability and status
   - coarse contact/queue facts
3. Build a per-tick opportunity index for 60-second control intervals.
4. Define a simple but defensible predicted-quality model.
5. Compute useful metadata needed by later planners:
   - expected value
   - predicted quality
   - cloud risk
   - retarget cost
   - downlink options
   - action mask reason when unavailable
6. Persist generated artifacts in a deterministic way.
7. Add tests for:
   - window generation
   - quality/value calculations
   - deterministic per-tick candidate indexing

Constraints:
- Keep the physics simplified but structured.
- Do not add hidden randomness.
- Keep the generated artifacts inspectable and schema-validated.
- Prefer explicit formulas and well-named helper functions over cleverness.

Deliverables:
- opportunity builder module
- `DownlinkWindow` contract implementation
- deterministic per-tick candidate index
- docs/phase1/opportunity-builder.md
- tests and example outputs

Acceptance criteria:
- a compiled scenario bundle can be transformed into validated opportunities/windows
- repeated runs produce identical outputs
- there is enough information for a baseline planner to act without additional hidden state

At the end:
- run tests/validation
- summarize the opportunity scoring model and storage format
```

---

## Prompt 6 — Environment runtime and canonical replay events

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state. Your task is to implement the deterministic single-agent orbital environment runtime for Phase 1.

This is not RL training yet. The goal is a serious simulator boundary with replayable step semantics.

Objectives:
1. Implement an `OrbitalEnv` or equivalent runtime with a Gymnasium-compatible shape where practical:
   - `reset(seed=...)`
   - `step(action)`
   - `observation`
   - `reward`
   - `terminated/truncated`
   - `info`
2. The runtime must consume:
   - compiled `ScenarioBundle`
   - precomputed opportunities/downlink windows
3. Model key runtime state:
   - current tick/time
   - onboard observation queue/buffer
   - ground-station availability
   - incident status (unseen / observed / downlinked / missed)
4. Implement action masking for legal actions.
5. Emit canonical replay events at the right points, including:
   - scenario_loaded
   - episode_started
   - opportunities_materialized
   - action_mask_emitted
   - action_selected
   - observation_committed
   - downlink_committed
   - reward_assessed
   - episode_ended
6. Include compatibility mapping if older sample event names still exist.
7. Add reward decomposition fields even if Phase 1 only uses simple planner baselines.
8. Add determinism tests and golden replay tests.

Constraints:
- No wall-clock use in transition logic.
- Same scenario + config + seed must produce identical replay.
- Keep the state representation inspectable.
- The runtime should be easy to wrap with RL later.

Deliverables:
- env runtime module
- canonical replay event emitter
- deterministic replay writer hooks
- docs/phase1/env-runtime.md
- tests including at least one golden replay sequence

Acceptance criteria:
- env reset/step works end to end on a compiled scenario
- replay events are emitted in canonical order
- repeated runs with the same seed match exactly

At the end:
- run tests
- summarize the runtime state model and action model
```

---

## Prompt 7 — Baselines and benchmark runner

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state. Your task is to implement the Phase 1 planners and benchmark runner.

Phase 1 requires two non-learning baselines:
- random valid-action
- urgency-greedy

Your task is to make them serious, comparable, and benchmarkable.

Objectives:
1. Implement planner interfaces so baselines can act through the same action contract as future RL policies.
2. Implement `random_valid_action` baseline.
3. Implement `urgency_greedy` baseline.
   - It should rank legal actions using immediate incident urgency, adjusted by predicted observation usefulness and/or cloud risk.
   - Keep the heuristic explicit and documented.
4. Implement a benchmark runner that:
   - runs planners over scenario packs
   - writes episode replays
   - computes metric summaries
   - records episode fingerprints
5. Implement the Phase 1 metric set at minimum:
   - time to first useful observation
   - useful observation value captured
   - cloud waste rate
   - downlink latency
   - missed urgent incident rate
   - opportunity utilization efficiency
   - mission utility
6. Add a simple report output:
   - JSON summaries
   - optionally a Markdown table or CSV export
7. Add tests covering:
   - planner legality
   - metric correctness on controlled toy episodes
   - benchmark determinism

Constraints:
- Benchmark code must be planner-agnostic.
- Do not leak privileged future information to one planner but not another.
- Keep the runner local and reproducible.

Deliverables:
- planner interface
- random and urgency-greedy baselines
- benchmark runner CLI
- metrics engine
- docs/phase1/benchmark-runner.md

Acceptance criteria:
- both baselines can run across the Phase 1 scenario pack
- replay + metrics are produced for each episode
- benchmark runs are deterministic under fixed seeds/configs

At the end:
- run the benchmark on at least a small subset
- summarize where outputs are written and how to compare planners
```

---

## Prompt 8 — Planner API and replay/metrics service

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state and the Phase 0 OpenAPI contract. Your task is to implement the Phase 1 API layer.

Use FastAPI and keep the service aligned with the Phase 0 planner API while allowing non-breaking additions that help the UI.

Objectives:
1. Implement API endpoints aligned with the Phase 0 contract:
   - GET `/v1/health`
   - POST `/v1/scenarios`
   - GET `/v1/scenarios/{bundleId}`
   - POST `/v1/episodes`
   - POST `/v1/episodes/{episodeId}/step`
   - GET `/v1/episodes/{episodeId}/events`
   - GET `/v1/episodes/{episodeId}/metrics`
   - POST `/v1/baselines/{baselineId}/run`
2. Back the API with the existing local file-based registries and benchmark/runtime modules.
3. Stream replay events as NDJSON where appropriate.
4. Add useful non-breaking endpoints for the UI if needed, for example:
   - list scenarios
   - list episodes
   - get scenario preview metadata
   - get Cesium/CZML-ready episode view model
5. Validate all inputs/outputs with the contract layer.
6. Add tests for:
   - health endpoint
   - scenario registration/retrieval
   - episode lifecycle
   - events/metrics retrieval
   - baseline run orchestration
7. Keep error handling clean and explicit.

Constraints:
- Do not drift from the Phase 0 path/field names without documenting it.
- Favor simple local orchestration in Phase 1 over premature queues/workers.
- Keep the API ready for the Cesium UI.

Deliverables:
- working FastAPI app in `apps/api`
- OpenAPI docs generated by FastAPI
- tested endpoints
- docs/phase1/api.md

Acceptance criteria:
- API can register a scenario, start an episode, run a baseline, and expose replay + metrics
- tests pass locally
- the API is ready for the UI to consume

At the end:
- run API tests
- summarize any non-breaking extensions you added beyond the Phase 0 contract
```

---

## Prompt 9 — Cesium mission-control UI

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state. Your task is to implement the Phase 1 globe UI.

This UI should feel like a mission-control application, not a toy dashboard.

Objectives:
1. Build a React + TypeScript + Vite UI in `apps/web` that talks to the API.
2. Use CesiumJS as the hero surface.
3. Prefer a time-dynamic data path that makes Cesium do the heavy lifting cleanly.
   - If practical, use CZML/CzmlDataSource for satellite trajectories and time availability.
4. Implement the core UI surfaces:
   - scenario selector
   - episode runner / planner selector
   - globe view with satellites and ground stations
   - incident / target-cell overlays
   - timeline / time scrubber
   - replay event panel
   - metrics summary panel
5. Make the UI capable of:
   - loading a scenario
   - starting a baseline run
   - replaying the resulting episode
   - scrubbing through events/time
6. Render clearly differentiated states for:
   - available targets
   - selected observation opportunities
   - downlink events
   - missed incidents or degraded utility
7. Add at least one automated smoke test if feasible, otherwise add a solid manual test script.
8. Write `docs/phase1/ui.md` with screenshots or instructions for generating them.

Constraints:
- Keep the UI clean and dark, mission-control style.
- Avoid overbuilding polished design systems; prioritize clarity and technical legibility.
- Do not implement the later street-level tactical map yet.
- Favor a robust local demo path over clever animation tricks.

Deliverables:
- Cesium-based web app
- API integration
- scenario/episode/replay views
- metrics panel
- docs for running the UI

Acceptance criteria:
- a user can launch the app and see a replayable orbital scenario on a globe
- metrics and replay data are visible without using notebooks
- the UI clearly demonstrates the technical shape of the system

At the end:
- build the web app
- summarize the user flows implemented and any known rough edges
```

---

## Prompt 10 — End-to-end integration, determinism hardening, and demo polish

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the full repo state. Your task is to harden and integrate the completed Phase 1 stack into a clean demoable system.

Objectives:
1. Verify the full Phase 1 path end to end:
   - compile/load scenario pack
   - build opportunities
   - run baseline planners
   - emit replay logs
   - compute metrics
   - serve through API
   - view through globe UI
2. Add determinism checks:
   - same scenario/config/seed => same episode fingerprint
   - same run => byte-stable replay artifacts where practical
3. Add a `demo` or `quickstart` command/script that:
   - prepares fixtures/scenarios
   - starts the API
   - starts the web app
   - points to a default scenario/run for demonstration
4. Add an integration test or scripted verification path.
5. Add a benchmark summary artifact for the Phase 1 scenario pack.
6. Write `docs/phase1/phase1-completion-report.md` that includes:
   - what was implemented
   - scenario families included
   - available baselines
   - known limitations
   - direct next steps for Phase 2 (RL training)
7. Tighten rough edges:
   - naming consistency
   - config defaults
   - file layout
   - developer onboarding notes

Constraints:
- Keep Phase 1 scoped: no tactical map, no RL training, no OR-Tools planner yet unless trivial.
- Do not hide determinism issues; fix or document them.
- Optimize for a strong engineering demo.

Deliverables:
- end-to-end working local stack
- quickstart/demo script
- integration verification
- Phase 1 completion report

Acceptance criteria:
- a new engineer can get a scenario running locally with clear steps
- at least one full replay is visible in the UI
- the benchmark runner outputs stable summaries
- the repo now feels like a serious platform slice, not disconnected prototypes

At the end:
- run the end-to-end validation path
- summarize the final Phase 1 state and any remaining debt
```

---

## Notes on execution strategy

- Keep everything deterministic unless a prompt explicitly asks for optional live data.
- Prefer fixture-backed tests and local reproducibility over live integrations.
- Resist the urge to start RL training in Phase 1.
- The most important prestige signal in this phase is **systems quality**: schemas, replay, determinism, baselines, and a serious globe UI.
