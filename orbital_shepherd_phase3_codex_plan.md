# Orbital Shepherd Phase 3 — Detailed Plan and Codex Prompt Pack

## What Phase 3 should achieve

Phase 3 is the **tactical ground-response digital twin** and **globe-to-street handoff** phase.

By the end of Phase 3, Orbital Shepherd should support:

- deterministic **regional tactical maps** compiled from real OpenStreetMap road networks,
- a **routing backend** that understands closures, dynamic edge penalties, service areas, and route alternatives,
- a **tactical scenario engine** that can instantiate local episodes from either synthetic seeds or real orbital `IncidentPacket` handoffs,
- a **ground response runtime** with replayable decisions and metrics,
- strong **non-learning tactical baselines** for dispatch, staging, and rerouting,
- a **MapLibre + deck.gl** tactical workspace that makes local plans legible,
- and a clean **orbital-to-tactical bridge** that turns an escalated observation into a local response episode.

This phase is where Orbital Shepherd stops being “an RL satellite lab with a globe” and becomes a true **multi-scale geospatial system**.

## What Phase 3 should not do

Phase 3 is deliberately **not** the end-to-end hierarchical RL phase.

Do **not** do the following yet:
- local tactical RL training as the primary milestone,
- joint orbital + ground policy optimization,
- live production telemetry ingestion from field systems,
- full suppression physics or fire spread simulation,
- or real emergency-operations integrations.

The prestige move in Phase 3 is not “more RL.” It is building a **serious downstream tactical layer** that Phase 4 can later learn over.

## Why this phase matters technically

Phase 2 made the orbital layer planner- and policy-ready. Phase 3 gives that orbital layer a **credible downstream consumer**. Once an orbital observation is escalated, the system now has to answer:

- Which region pack should be activated?
- Which roads are traversable right now?
- Which units should move first?
- Which assets or facilities matter most?
- How does uncertainty change dispatch?
- What downstream value did the orbital decision actually unlock?

That is what makes future hierarchical RL meaningful rather than cosmetic.

## Recommended Phase 3 stack

- **Region extraction / graph compilation:** OSMnx
- **Global-local bridge / indexing:** H3
- **Spatial source of truth:** PostGIS
- **Routing / network analysis:** pgRouting
- **Portable geospatial artifacts:** GeoParquet
- **Tile / static layer delivery:** ST_AsMVT and/or PMTiles
- **Frontend tactical map:** MapLibre GL JS + deck.gl
- **Optimization baselines:** OR-Tools where it materially improves dispatch or staging
- **API / orchestration:** existing FastAPI stack from Phases 1–2
- **Replay / artifact discipline:** continue using manifest-driven immutable bundles and NDJSON replay logs

## Design stance for Phase 3

### 1. Region-bundle-first
Just as earlier phases introduced `ScenarioBundle`, Phase 3 should introduce compiled, immutable **region bundles** so local tactical episodes do not depend on live OSM queries or hand-built GIS notebooks.

### 2. Immutable base graph, dynamic overlays
Do **not** destructively mutate the road graph. Keep the base network immutable and apply closures, smoke/risk penalties, and temporary restrictions as **overlay layers** that alter effective edge costs.

### 3. Planner-agnostic tactical layer
The tactical layer must not assume RL. It should support simple dispatchers, coverage-aware heuristics, and optionally OR-Tools-driven assignment. That makes the local layer valuable even before Phase 4.

### 4. Globe-to-street handoff through explicit contracts
Do not let the tactical side reach directly into orbital replay internals. Use:
- `IncidentPacket` (existing cross-phase contract),
- `TacticalActivation` (new normalized bridge object),
- and `TacticalScenarioBundle` (new immutable runtime artifact).

### 5. Static and dynamic map data should travel differently
Static road/facility/asset layers want a tile-friendly format. Dynamic units, incidents, and routes want direct API payloads. Keep those paths separate.

### 6. Determinism matters more than realism at first
Every tactical episode must be reproducible from manifests, bundles, and seeds. It is better to have a controlled, inspectable local layer than an “almost real” one that is impossible to benchmark.

## Core Phase 3 artifact graph

Recommended canonical artifact flow:

```text
RegionManifest
  -> RegionBundle
  -> SpatialIngestManifest
  -> TacticalActivation
  -> TacticalScenarioBundle
  -> TacticalEpisode
  -> TacticalReplay.ndjson
  -> TacticalMetricsSummary
```

### Canonical new artifact types

- `RegionManifest`
  - authoring recipe for a local tactical region
  - source area, network extraction options, facility seeds, asset overlays, H3 resolution, and region metadata

- `RegionBundle`
  - compiled immutable tactical-region artifact
  - canonical nodes, edges, travel-time model defaults, facilities, assets, H3 cover, and fingerprints

- `SpatialIngestManifest`
  - describes how a `RegionBundle` has been materialized into PostGIS/pgRouting

- `TacticalActivation`
  - normalization boundary between orbital `IncidentPacket` and tactical scenario creation
  - includes region resolution, AOI, escalation metadata, confidence, urgency, and initiating source references

- `TacticalScenarioManifest`
  - authoring recipe for a tactical episode family

- `TacticalScenarioBundle`
  - deterministic compiled tactical episode consumed by the runtime

- `TacticalReplayEvent`
  - append-only local event envelope using the same replay philosophy as earlier phases

- `TacticalMetricsSummary`
  - final episode metrics, route risk, ETA stats, verification delays, and asset-coverage outcomes

## Tactical state model

The ground layer does not need full suppression physics yet. It needs a strong **dispatch and routing state model**.

### Incident state
Recommended incident lifecycle:

```text
reported -> pending_verification -> verified -> response_planned -> units_en_route -> on_scene -> stabilized -> resolved
```

### Unit state
Recommended unit lifecycle:

```text
idle -> mobilizing -> en_route -> staged -> on_scene -> returning -> unavailable
```

### Example unit types
- engine / crew
- command vehicle
- drone scout
- medical or support unit (optional, later)

### Route state
A route should carry:
- route id
- unit id
- ordered path edges
- travel time estimate
- risk score
- version / replanning count
- overlay assumptions used when it was planned

## Tactical action model

Phase 3 does not need the full action masking and RL interface from Phase 2. It does need a **clear dispatch action space** the runtime and baselines can share.

Recommended high-level actions:
- dispatch a unit to an incident
- dispatch a drone scout to verify or refine incident extent
- reroute an active unit
- stage a unit to a waypoint or facility
- hold / no-op
- optionally reserve a unit to preserve future coverage

These can later become the local policy action space in Phase 4.

## Tactical event taxonomy

Recommended event families:

- `tactical_activation_received`
- `region_bundle_loaded`
- `tactical_scenario_loaded`
- `tactical_episode_started`
- `units_materialized`
- `dispatch_candidates_materialized`
- `dispatch_decision_selected`
- `route_planned`
- `route_replanned`
- `unit_departed`
- `unit_arrived_staging`
- `drone_launched`
- `verification_completed`
- `closure_applied`
- `overlay_state_updated`
- `incident_state_changed`
- `on_scene_reached`
- `tactical_episode_completed`
- `tactical_metrics_finalized`

Every event should carry source references back to the orbital handoff when applicable:
- `source_orbital_episode_id`
- `source_orbital_event_id`
- `incident_packet_id`

## Tactical metrics

The local layer needs benchmark metrics that are legible to both engineers and future RL work.

Recommended metrics:
- `activation_to_first_dispatch_seconds`
- `activation_to_first_arrival_seconds`
- `activation_to_verification_seconds`
- `mean_route_risk`
- `max_route_risk`
- `route_replan_count`
- `unserved_incident_count`
- `asset_coverage_gap`
- `reserve_capacity_minutes_lost`
- `dispatch_cost_score`
- `orbital_to_ground_handoff_seconds`
- `planner_runtime_ms`

Keep these separate from any future local training reward.

## Scenario families for Phase 3

Recommended minimum tactical scenario families:

### 1. Foothill Access
Sparse graph, long approach times, limited ingress/egress.

### 2. Urban Interface
Dense road graph, multiple facilities, high-value asset zones, tighter dispatch tradeoffs.

### 3. Closure Cascade
One or more key road closures invalidate naive shortest-path plans.

### 4. Depot Saturation
Multiple simultaneous incidents strain available crews and vehicles.

### 5. Smoke Corridor
Nominally short routes become undesirable because risk overlays dominate travel cost.

### 6. Drone Scout Gap
Fast drone verification improves later crew dispatch, so a purely nearest-unit strategy can lose.

## Recommended region-pack strategy

Ship at least three region packs:

- `fixture_micro_region`
  - tiny checked-in region for CI and local smoke tests
  - no external network access required

- `pilot_wui_region`
  - real-ish wildland-urban interface region compiled from OSM recipe
  - used for demos and benchmark screenshots

- `pilot_dense_interface_region`
  - denser street network with more facilities and asset layers
  - used to stress routing, service areas, and compare mode

The repo should be able to **rebuild** region bundles deterministically from manifests, but Phase 3 should also check in enough small artifacts to make CI/local validation fast.

## Definition of done for Phase 3

Phase 3 is complete when all of the following are true:

1. `RegionManifest -> RegionBundle` compilation exists and is deterministic.
2. At least one region can be ingested into PostGIS/pgRouting and queried for:
   - shortest path
   - ETA matrix
   - service area / isochrone-like results
   - dynamic closure/risk-aware route cost
3. There is a `TacticalActivation` bridge from orbital `IncidentPacket` to tactical scenario creation.
4. There is a deterministic `TacticalScenarioBundle` compiler with at least 24 scenarios across the 6 tactical families.
5. There is a tactical runtime that emits canonical local replay events.
6. There are at least 3 tactical baselines:
   - nearest ETA
   - risk-aware dispatch
   - coverage-preserving / multi-incident
7. The web app can open a tactical map, scrub a tactical replay, and show routes, units, facilities, closures, and hazards.
8. The Cesium globe can open the tactical workspace from an escalated incident.
9. A developer can run one end-to-end flow:
   - load orbital replay or incident packet
   - activate tactical region
   - run baseline planner
   - inspect replay on tactical map
   - export metrics
10. The docs clearly tee up Phase 4: local RL and hierarchical integration.

## Recommended Phase 3 repository additions

```text
orbital-shepherd/
  apps/
    api/
    web/
  packages/
    contracts/
    core/
    region_builder/
    geo_artifacts/
    routing_engine/
    tactical_scenario_engine/
    ground_env/
    tactical_baselines/
    tactical_metrics/
    escalation_bridge/
  data/
    regions/
      manifests/
      bundles/
      fixtures/
    tactical_scenarios/
    tactical_replays/
    tiles/
  infra/
    docker/
    sql/
  docs/
    phase3/
  scripts/
```

## Suggested API surface for Phase 3

This is a recommended shape, not a locked requirement.

### Tactical region endpoints
- `POST /api/tactical/regions/compile`
- `GET /api/tactical/regions`
- `GET /api/tactical/regions/{region_id}`
- `POST /api/tactical/regions/{region_id}/ingest`

### Routing endpoints
- `POST /api/tactical/route`
- `POST /api/tactical/service-area`
- `POST /api/tactical/eta-matrix`
- `POST /api/tactical/overlays/preview`

### Tactical scenario / episode endpoints
- `POST /api/tactical/activations`
- `POST /api/tactical/scenarios/compile`
- `POST /api/tactical/episodes/start`
- `POST /api/tactical/episodes/{episode_id}/step`
- `GET /api/tactical/episodes/{episode_id}/replay`
- `GET /api/tactical/episodes/{episode_id}/metrics`

### Integration endpoints
- `POST /api/integrations/orbital-escalation`
- `GET /api/integrations/orbital/{orbital_episode_id}/linked-tactical`

## Suggested map delivery strategy

Recommended split:

- **Static regional layers**
  - roads
  - facilities
  - asset zones
  - H3 overlays
  - optionally region boundaries and AOI polygons

  Deliver as:
  - MVT endpoints from PostGIS, or
  - PMTiles artifacts generated from region bundles

- **Dynamic tactical layers**
  - incidents
  - units
  - current routes
  - closures
  - route alternatives
  - service areas

  Deliver as:
  - API payloads / replay payloads
  - rendered in deck.gl overlays

This gives you a map that feels industrial without forcing all dynamic data through a tile pipeline.

## Run order for the Codex sessions

1. Phase 3 tactical foundation, contracts, and ADR
2. Region builder and geospatial artifact pipeline
3. PostGIS / pgRouting ingest and routing engine
4. Tactical scenario engine and activation compiler
5. Ground runtime and tactical replay events
6. Tactical baselines and benchmark runner
7. Planner API and orbital-to-tactical bridge
8. Tactical workspace UI
9. Globe-to-street integration and compare mode
10. End-to-end hardening, tile optimization, and completion report

---

# Copy-paste Codex prompts

Each prompt below is written to be executable in a fresh Codex session. They assume the repository already contains prior merged work, but they do **not** assume chat history. Run them in order.

---

## Prompt 1 — Phase 3 tactical foundation, contracts, and ADR

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Start by inspecting the repo and any existing docs under paths like:
- docs/phase0/
- docs/phase1/
- docs/phase2/
- phase completion reports
- current contracts, replay schemas, API docs, and UI routes

Treat the current repo state as the source of truth. Improve and extend it rather than rewriting it.

Your task is to create the Phase 3 tactical foundation.

Objectives:
1. Write `docs/phase3/ADR-0001-phase3-tactical-scope.md` that defines:
   - what Phase 3 is for
   - what remains out of scope
   - the boundary between orbital and tactical layers
   - why Phase 3 is planner-agnostic and not yet hierarchical RL
2. Extend the contracts layer with the new Phase 3 artifact types:
   - `RegionManifest`
   - `RegionBundle`
   - `SpatialIngestManifest`
   - `TacticalActivation`
   - `TacticalScenarioManifest`
   - `TacticalScenarioBundle`
   - `DispatchUnit`
   - `Facility`
   - `RoutePlan`
   - `TacticalReplayEvent`
   - `TacticalMetricsSummary`
3. Preserve compatibility with the existing `IncidentPacket` and replay/event philosophy from earlier phases.
4. Define a canonical tactical event taxonomy and document it clearly.
5. Add package scaffolding for:
   - `packages/region_builder`
   - `packages/geo_artifacts`
   - `packages/routing_engine`
   - `packages/tactical_scenario_engine`
   - `packages/ground_env`
   - `packages/tactical_baselines`
   - `packages/tactical_metrics`
   - `packages/escalation_bridge`
6. Add docs under `docs/phase3/` for:
   - architecture overview
   - artifact graph
   - glossary of tactical objects
7. Add validation/tests for the new contracts and example artifacts.

Important design guidance:
- Keep the tactical layer planner-agnostic.
- Do not add local RL training yet.
- Do not destroy or bypass existing orbital contracts.
- Make the orbital-to-tactical bridge explicit and typed.
- Preserve determinism, manifests, fingerprints, and replayability as first-class concerns.

Deliverables:
- Phase 3 ADR
- extended schemas and typed models
- tactical package scaffolding
- docs/phase3 architecture notes
- tests validating new artifact examples

Acceptance criteria:
- new contracts validate cleanly
- docs clearly explain the new artifact graph
- `IncidentPacket` remains the official cross-phase handoff object
- Phase 3 now has a formal implementation contract rather than an informal idea

At the end:
- run validation/tests
- summarize the canonical Phase 3 artifact types and event names
- list any assumptions you had to make
```

---

## Prompt 2 — Region builder and geospatial artifact pipeline

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- contracts and docs from Phases 0–3
- any existing scenario/bundle builders
- replay conventions
- geospatial dependencies already present in the repo

Your task is to build the Phase 3 region compiler.

Objectives:
1. Implement `packages/region_builder` so a `RegionManifest` can be compiled into a deterministic `RegionBundle`.
2. Use OSM-derived road networks where practical, but do not make the build depend entirely on live network access.
   - If the repo already contains fixture extracts, use them.
   - If not, support both:
     - a checked-in small fixture path for CI/smoke tests
     - an optional OSMnx-backed compile path for larger pilot regions
3. Compile at least the following into `RegionBundle`:
   - canonical directed road graph (nodes + edges)
   - travel time defaults / speed assumptions
   - facilities / depots / staging points
   - asset zones or points of interest
   - H3 cover metadata
   - bundle fingerprint / provenance metadata
4. Add an export layer in `packages/geo_artifacts` that can materialize region data as:
   - GeoParquet (preferred canonical portable artifact)
   - GeoJSON for debugging or small fixtures
   - optional tile-friendly intermediate outputs if useful
5. Create at least three region manifests:
   - `fixture_micro_region`
   - `pilot_wui_region`
   - `pilot_dense_interface_region`
6. Generate at least one tiny checked-in compiled fixture so local tests do not rely on live downloads.
7. Write tests ensuring:
   - identical manifest + seed -> identical bundle fingerprint
   - edge/node counts are stable for fixture builds
   - H3 cover generation is deterministic
8. Document the region-build pipeline in `docs/phase3/region-builder.md`.

Important design guidance:
- The compiled region bundle must be immutable and portable.
- Separate authoring recipes from compiled artifacts.
- Prefer small manifest metadata plus generated artifacts over hand-edited GIS blobs.
- Add OSM attribution notes where appropriate.
- Keep the small fixture path easy to run on a laptop.

Deliverables:
- region compiler
- portable geospatial artifact exports
- region manifests
- one checked-in fixture bundle
- tests and docs

Acceptance criteria:
- a developer can build the fixture region locally without internet
- a developer can optionally compile a larger pilot region from a manifest
- compiled bundles are deterministic and fingerprinted
- the repo now has a serious local geospatial artifact pipeline

At the end:
- run the fixture compile path and tests
- summarize the shape of `RegionBundle`
- list any external-data assumptions you had to make
```

---

## Prompt 3 — PostGIS / pgRouting ingest and routing engine

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the repo and the new Phase 3 contracts, especially `RegionBundle`, `SpatialIngestManifest`, and any existing API / infra patterns.

Your task is to build the tactical routing backend.

Objectives:
1. Provision a local spatial database workflow for Phase 3.
   - Prefer a Docker Compose service using a PostGIS + pgRouting image or another maintainable setup already present in the repo.
   - Keep the developer experience clear and documented.
2. Add SQL migrations / bootstrap scripts for:
   - regions
   - nodes
   - edges
   - facilities
   - asset layers
   - overlay tables for closures / risk multipliers / temporary restrictions
3. Implement `packages/routing_engine` with at least:
   - region ingest from `RegionBundle`
   - shortest-path routing
   - ETA matrix for multiple origin/destination pairs
   - service-area / reachable-within-time query
   - dynamic overlay-aware effective edge cost computation
4. Do not destructively mutate the base graph.
   - Base graph stays immutable.
   - Overlays alter effective cost or blocked status at query time.
5. Add a pure-Python or fixture-backed test mode where practical so unit tests do not all require Docker, but keep the real PostGIS/pgRouting path as the primary integration target.
6. Expose a small integration API or service layer the later prompts can call.
7. Write `docs/phase3/routing-engine.md` covering:
   - schema layout
   - overlay strategy
   - performance notes
   - limitations

Important design guidance:
- Use the database for authoritative route/network analysis, not just as a cache.
- Make overlay logic explicit and testable.
- Prefer a few clean query types over an explosion of ad hoc route helpers.
- Keep ingestion idempotent and fingerprint-aware.
- Tag integration tests clearly if Docker is required.

Deliverables:
- local PostGIS/pgRouting dev setup
- SQL/schema/bootstrap scripts
- routing engine package
- ingest flow from `RegionBundle`
- tests and docs

Acceptance criteria:
- a fixture region can be ingested
- shortest path, ETA matrix, and service-area queries all work
- closures and risk overlays change route outputs without mutating the base graph
- a new engineer can follow the docs and get the spatial backend running

At the end:
- run the smallest practical ingest + query validation path
- summarize the main query surfaces and overlay model
- list any infra assumptions you had to make
```

---

## Prompt 4 — Tactical scenario engine and activation compiler

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- `IncidentPacket`
- any existing scenario-engine patterns from Phases 1–2
- the new region and routing contracts from Phase 3

Your task is to build the tactical scenario engine.

Objectives:
1. Implement `packages/tactical_scenario_engine`.
2. Introduce `TacticalActivation` as the normalized bridge between:
   - orbital `IncidentPacket`
   - region resolution
   - tactical scenario creation
3. Build deterministic compilation from:
   - `TacticalScenarioManifest` -> `TacticalScenarioBundle`
   - and optionally `IncidentPacket` -> `TacticalActivation` -> `TacticalScenarioBundle`
4. Support at least these tactical scenario families:
   - Foothill Access
   - Urban Interface
   - Closure Cascade
   - Depot Saturation
   - Smoke Corridor
   - Drone Scout Gap
5. Tactical scenario bundles should include:
   - selected region id / bundle fingerprint
   - incident geometry and AOI
   - unit roster and depot assignments
   - optional drones or scout assets
   - overlay events (closures, risk zones, temporary penalties)
   - timing / severity / urgency metadata
   - deterministic seed and provenance
6. Create at least 24 deterministic tactical scenarios across the families above.
7. Add CLIs/scripts to:
   - build tactical scenario packs
   - compile from incident packet + region selection
   - inspect a compiled scenario
8. Add tests ensuring:
   - identical activation + seed -> identical bundle fingerprint
   - region resolution is stable
   - family-specific overlays are actually represented in the bundle
9. Write `docs/phase3/tactical-scenario-engine.md`.

Important design guidance:
- Tactical scenarios must be replayable and fingerprinted like earlier phases.
- Do not embed live DB state directly into the bundle; keep it declarative.
- Make the orbital-to-tactical bridge explicit in the artifact metadata.
- Prefer a few clear family templates over endless one-off scenario scripts.

Deliverables:
- tactical scenario engine
- tactical scenario manifests/bundles
- activation compiler from `IncidentPacket`
- deterministic scenario pack
- tests and docs

Acceptance criteria:
- a developer can compile a tactical scenario bundle from a seed
- a developer can compile one from a real or fixture `IncidentPacket`
- scenario families are differentiated and legible
- the tactical layer now has a proper deterministic benchmark pack

At the end:
- run the scenario-pack build and validation path
- summarize the structure of `TacticalActivation` and `TacticalScenarioBundle`
- list any family-design assumptions you had to make
```

---

## Prompt 5 — Ground runtime and tactical replay events

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- Phase 1/2 env runtime and replay architecture
- Phase 3 tactical contracts and scenario bundles
- routing engine query surfaces

Your task is to implement the local tactical runtime.

Objectives:
1. Implement `packages/ground_env` as a deterministic, step-driven tactical runtime.
2. The runtime should consume `TacticalScenarioBundle` and emit canonical `TacticalReplayEvent`s.
3. Support at least the following action types:
   - dispatch a unit to an incident
   - dispatch a drone scout
   - reroute an active unit
   - stage a unit to a waypoint/facility
   - hold / no-op
4. Model and update at least:
   - incident state
   - unit state
   - active route state
   - overlay state impacts relevant to routing
5. Reuse the existing replay philosophy from earlier phases:
   - append-only NDJSON replay
   - deterministic event ordering
   - event indices and fingerprints where appropriate
6. Emit and document a canonical tactical event taxonomy, including events like:
   - tactical_activation_received
   - tactical_scenario_loaded
   - tactical_episode_started
   - dispatch_candidates_materialized
   - dispatch_decision_selected
   - route_planned
   - unit_departed
   - drone_launched
   - verification_completed
   - route_replanned
   - on_scene_reached
   - tactical_episode_completed
   - tactical_metrics_finalized
7. Add runtime metrics hooks so the benchmark layer can compute:
   - time to first dispatch
   - time to arrival
   - route risk
   - replan counts
   - unserved incidents
8. Add tests ensuring deterministic replay for fixture episodes.
9. Write `docs/phase3/ground-runtime.md`.

Important design guidance:
- Keep the runtime inspectable and event-sourced.
- Do not add local RL abstractions yet beyond what is naturally shared with planners.
- Avoid hidden state transitions.
- Every planner decision should be reconstructable from the replay log.

Deliverables:
- ground runtime package
- tactical replay event implementation
- metrics hooks
- tests and docs

Acceptance criteria:
- a tactical scenario can be run to completion deterministically
- replay logs are stable across repeated runs with the same seed
- planner decisions and route transitions are inspectable in the replay
- the local layer now behaves like a real simulator/runtime, not just an API wrapper

At the end:
- run a fixture tactical episode and validate the replay output
- summarize the main runtime states and actions
- list any simplifications you made
```

---

## Prompt 6 — Tactical baselines and benchmark runner

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- the tactical scenario pack
- the ground runtime
- the routing engine
- the benchmark/reporting patterns from earlier phases

Your task is to build the tactical planning baselines and benchmark harness.

Objectives:
1. Implement at least these tactical baselines:
   - `nearest_eta_dispatch`
   - `risk_aware_dispatch`
   - `coverage_preserving_dispatch`
2. If the repo already has OR-Tools or a good optimization layer available, add an optional stronger baseline for:
   - multi-incident assignment
   - staging/repositioning
   - or dispatch under simple capacity/time-window constraints
3. Build a tactical benchmark runner that evaluates baselines across the 24-scenario pack.
4. Produce benchmark artifacts that include:
   - per-family metrics
   - paired planner comparisons
   - episode fingerprints
   - notable failure/success cases
5. Add reports/manifests for benchmark results.
6. Keep the tactical metrics planner-agnostic and separate from any future training reward.
7. Write `docs/phase3/tactical-benchmarks.md`.

Important design guidance:
- Do not fake sophistication with arbitrary score formulas.
- The baselines should be simple, strong, and interpretable.
- Benchmark artifacts must be replayable and tied to scenario fingerprints.
- This phase should make future local RL look necessary only where the baselines actually fail.

Deliverables:
- tactical baseline planners
- benchmark runner
- result manifests/reports
- tests and docs

Acceptance criteria:
- all baselines run end-to-end on the tactical scenario pack
- per-family benchmark summaries are produced
- benchmark outputs can be linked back to scenario/replay artifacts
- the repo now has a credible local tactical benchmark layer

At the end:
- run the smallest practical multi-scenario benchmark
- summarize the implemented baselines and their intended behaviors
- list any scenarios where the baselines obviously struggle
```

---

## Prompt 7 — Planner API and orbital-to-tactical bridge

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- the planner API from earlier phases
- the new tactical packages
- `IncidentPacket`
- replay and metrics endpoint patterns
- any existing policy/baseline registration layer

Your task is to extend the backend so the tactical layer is a first-class part of the platform.

Objectives:
1. Extend the API with tactical endpoints for:
   - region listing / inspection
   - tactical activation creation
   - tactical scenario compilation
   - tactical episode start / step / replay / metrics
   - route / service-area / ETA matrix queries
2. Implement an orbital-to-tactical bridge so an incoming `IncidentPacket` can:
   - resolve to a region
   - create a `TacticalActivation`
   - compile or select a `TacticalScenarioBundle`
   - start a tactical episode
3. Preserve source traceability:
   - orbital episode id
   - orbital replay event id
   - incident packet id
   - tactical episode id
4. Add planner registration / selection support for tactical baselines.
5. Keep the API backward compatible with existing orbital workflows.
6. Add or update OpenAPI docs and example payloads.
7. Write `docs/phase3/tactical-api.md`.

Important design guidance:
- The tactical layer should feel native to the platform, not bolted on.
- Use explicit typed payloads; avoid magical ad hoc dictionaries.
- Keep route and replay endpoints composable enough for the UI.
- Do not mix tactical episode stepping with raw database mutation endpoints.

Deliverables:
- API extensions
- orbital-to-tactical bridge implementation
- tactical planner selection support
- OpenAPI updates
- tests and docs

Acceptance criteria:
- a fixture `IncidentPacket` can create a tactical activation and start a tactical episode
- replay and metrics are accessible through the API
- route/service-area queries are exposed cleanly
- the tactical layer is now reachable through the same product surface as the orbital layer

At the end:
- run the smallest practical API smoke path for a tactical activation
- summarize the new endpoint groups
- list any compatibility decisions you made
```

---

## Prompt 8 — Tactical workspace UI

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- the existing web app
- the Cesium mission-control UI
- API clients and state-management patterns
- tactical endpoints added in Phase 3

Your task is to build the tactical workspace in the web app.

Objectives:
1. Add a tactical map workspace using MapLibre GL JS and deck.gl.
2. The tactical workspace should render:
   - base road network / regional layers
   - facilities / depots / staging points
   - incidents and AOI overlays
   - response units
   - active routes
   - closures / risk overlays
   - optional service-area and route-alternative layers
3. Add interaction patterns for:
   - selecting units, incidents, or routes
   - toggling layers
   - scrubbing through a tactical replay
   - inspecting event details and metrics
4. Integrate the tactical API clients so the workspace can:
   - load a tactical episode
   - fetch replay events
   - fetch metrics
   - optionally preview routes or service areas
5. Keep the UI consistent with the existing dark mission-control aesthetic.
6. Add a tactical episode side panel showing:
   - activation metadata
   - planner used
   - selected action / current route
   - ETA and risk summaries
   - linked orbital context if available
7. Write `docs/phase3/tactical-ui.md`.

Important design guidance:
- Static regional layers and dynamic overlays can travel differently.
- Do not force everything through one rendering path.
- Keep the replay legible; do not build a pretty map with no debugging value.
- The UI should make planner decisions inspectable, not just scenic.

Deliverables:
- tactical workspace UI
- API clients/hooks
- replay controls and inspection panels
- docs

Acceptance criteria:
- a user can open a tactical episode and understand what units moved where and why
- routes, incidents, and facilities are visually distinct and inspectable
- the workspace builds cleanly and matches the platform aesthetic
- the tactical layer now has a serious front-end debugger/demo surface

At the end:
- run the web build and any relevant UI tests
- summarize the main tactical UI surfaces
- list any rendering/data-flow tradeoffs you made
```

---

## Prompt 9 — Globe-to-street integration and compare mode

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- the Cesium globe UI
- the tactical workspace
- the API bridge between orbital and tactical layers
- existing compare/replay patterns from earlier phases

Your task is to connect the globe and tactical layers into one coherent product flow.

Objectives:
1. Add a globe-to-street user flow so an escalated incident on the Cesium globe can open the tactical workspace with the correct context.
2. Support navigation from:
   - an orbital replay event
   - an `IncidentPacket`
   - or a linked tactical episode id
3. Surface linked-context metadata in the UI:
   - source orbital episode
   - source orbital event
   - incident packet id
   - selected region pack
   - tactical planner
4. Add compare mode for tactical episodes so a user can inspect:
   - baseline A vs baseline B
   - or an orbital-triggered tactical run vs a manually started one
5. Reuse the replay/event philosophy:
   - compare timelines
   - key metric deltas
   - route deltas
   - dispatch timing deltas
6. Add at least one polished end-to-end demo path:
   - load orbital replay
   - choose escalated incident
   - open tactical workspace
   - compare two tactical planners
   - export or inspect metrics
7. Write `docs/phase3/globe-to-street.md`.

Important design guidance:
- The handoff should feel like one product, not two separate apps.
- Avoid clever but opaque implicit linking.
- Make deep links and shareable URLs where practical.
- Keep the compare mode useful for debugging, not just marketing screenshots.

Deliverables:
- integrated globe-to-street flow
- linked-context UI
- tactical compare mode
- docs

Acceptance criteria:
- a user can move from the globe to the tactical map without manual data wrangling
- linked context is preserved and visible
- compare mode surfaces meaningful differences between planners
- Orbital Shepherd now convincingly feels like a multi-scale operations system

At the end:
- run the smallest practical end-to-end UI smoke path
- summarize the user flow and compare surfaces
- list any navigation/state assumptions you made
```

---

## Prompt 10 — End-to-end hardening, tile optimization, and Phase 3 completion report

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially all Phase 3 packages, docs, API routes, UI routes, and build/test scripts.

Your task is to harden Phase 3 into a reproducible, demoable platform increment.

Objectives:
1. Add or finalize ergonomic commands such as:
   - `make phase3-region-build`
   - `make phase3-routing-smoke`
   - `make phase3-benchmark`
   - `make phase3-demo`
2. Tighten determinism and artifact traceability:
   - stable region bundle fingerprints
   - tactical scenario fingerprints
   - replay manifests
   - metrics report manifests
3. Optimize static map delivery where useful.
   - If the current implementation already serves vector tiles from PostGIS, improve and document it.
   - Otherwise consider adding a PMTiles or similar static artifact path for regional layers.
   - Do not overcomplicate local development.
4. Add smoke/demo paths that validate:
   - fixture region build
   - region ingest
   - tactical scenario compile
   - tactical baseline run
   - tactical replay load in the UI
   - globe-to-street handoff
5. Write `docs/phase3/phase3-completion-report.md` that includes:
   - what was implemented
   - artifact graph
   - region packs and scenario families
   - routing strategy
   - tactical baselines
   - UI capabilities
   - known limitations
   - direct next steps for Phase 4
6. Clean up rough edges:
   - naming consistency
   - onboarding docs
   - config defaults
   - compose/dev ergonomics
   - sample payloads and screenshots where practical

Important design guidance:
- Keep Phase 3 focused on the tactical digital twin and handoff.
- Do not smuggle in local RL or hierarchical training here.
- Optimize for a strong engineering demo and clean platform story.
- Be honest about limitations and unresolved technical debt.

Deliverables:
- end-to-end reproducible Phase 3 workflow
- improved dev/demo commands
- optional tile-delivery optimization if justified
- completion report
- final docs polish

Acceptance criteria:
- a new engineer can run the smallest practical Phase 3 demo path locally
- region build, tactical activation, routing, replay, and UI all work together
- the completion report makes the Phase 4 jump obvious
- the repo now feels like a serious geospatial operations platform, not a collection of map demos

At the end:
- run the smallest practical end-to-end validation path
- summarize the final Phase 3 state and remaining debt
```

---

## Notes on execution strategy

- Keep the **fixture path** sacred. A tiny deterministic region and a tiny tactical scenario pack make the whole phase testable.
- Preserve the same **replay-first discipline** used in the orbital stack.
- The tactical layer should be useful even if Phase 4 RL never happens.
- The prestige signal in Phase 3 is:
  - typed contracts,
  - deterministic geospatial artifacts,
  - real routing over a real road graph,
  - a clean orbital-to-local bridge,
  - and a tactical map that explains decisions rather than merely rendering them.

## Phase 4 tee-up

If Phase 3 lands cleanly, Phase 4 should become:

- local tactical RL or policy learning over the ground runtime,
- downstream-value modeling for orbital decisions,
- and finally hierarchical orbital-to-ground optimization.

That only works if Phase 3 is built as a **serious tactical substrate**, not as a map-shaped side quest.
