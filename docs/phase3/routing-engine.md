# Phase 3 Routing Engine

`packages/routing_engine` is the tactical routing backend for Phase 3.
It treats PostGIS + pgRouting as the authoritative route-analysis layer and keeps the in-memory backend limited to fixture-backed tests and fast local unit coverage.

## Local Workflow

Start the spatial database:

```bash
docker compose -f infra/compose/phase3-routing.compose.yaml up -d
```

The local Docker daemon must already be running before this step.

Bootstrap extensions and schema objects:

```bash
python3 scripts/phase3_routing.py bootstrap-db
```

Ingest the checked-in fixture bundle:

```bash
python3 scripts/phase3_routing.py ingest-bundle \
  data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json
```

Run the smallest end-to-end validation path:

```bash
python3 scripts/phase3_routing.py smoke
```

Run the fast fixture-backed unit suite without Docker:

```bash
.venv/bin/pytest tests/routing_engine/test_routing_engine.py -q
```

Run the Docker-gated integration test once the database is up:

```bash
ORBITAL_SHEPHERD_ENABLE_DOCKER_TESTS=1 \
  .venv/bin/pytest tests/integration/test_phase3_routing_postgres.py -q
```

The default DSN is:

```text
postgresql://orbital:orbital@127.0.0.1:55432/orbital_shepherd_phase3
```

Override it with `ORBITAL_SHEPHERD_ROUTING_DSN` or `--dsn`.

## Query Surfaces

`RoutingEngineService` exposes a small planner-facing surface:

- `ingest_region_bundle(...)`
  - idempotent on `region_bundle_id` + `bundle_fingerprint`
  - rejects a conflicting fingerprint under the same bundle id
- `register_overlay(...)`
  - persists immutable overlay batches
  - idempotent on `overlay_id` + overlay fingerprint
- `shortest_path(...)`
  - accepts facility ids, node ids, or raw points
  - snaps off-network endpoints to the nearest graph node
- `eta_matrix(...)`
  - many-to-many ETA matrix on the same endpoint model
- `service_area(...)`
  - returns nodes and facilities reachable within a travel-time budget

For ad hoc CLI use, the same query types are available through `scripts/phase3_routing.py`.

## Schema Layout

The authoritative schema lives in the `routing` namespace.

Base immutable graph tables:

- `routing.regions`
  - one row per ingested `RegionBundle`
  - stores bundle fingerprint, travel defaults, H3 metadata, and compilation payload
- `routing.spatial_ingests`
  - normalized `SpatialIngestManifest` records
- `routing.nodes`
  - numeric `vertex_id` assignments for pgRouting
- `routing.edges`
  - immutable directed edges derived from `RegionBundle.road_edges`
  - stores base travel time, geometry, and source ingest references
- `routing.facilities`
  - facility points plus precomputed nearest-node linkage and approach time
- `routing.asset_layers`
  - derived layer headers grouped by `(asset_kind, geometry_type)`
- `routing.asset_features`
  - point/polygon tactical assets with geometry and JSON payloads

Overlay tables:

- `routing.overlay_sets`
  - immutable overlay headers with kind, active window, metadata, and fingerprint
- `routing.overlay_edge_closures`
  - route-blocking edge closures
- `routing.overlay_edge_risk_multipliers`
  - multiplicative cost inflation per edge
- `routing.overlay_edge_temporary_restrictions`
  - speed caps, additive delay, and extra cost multipliers

## Overlay Strategy

The base graph is never mutated after ingest.
All dynamic conditions are modeled as overlay rows that are selected at query time.

The SQL function `routing.effective_edges(region_bundle_id, overlay_ids, effective_at_utc)` materializes the runtime edge view:

1. Select active overlay batches by id and time window.
2. Fold closure rows into a boolean blocked flag.
3. Fold risk rows into a multiplicative edge cost factor.
4. Fold temporary restrictions into:
   - minimum active speed cap
   - additive delay
   - multiplicative restriction factor
5. Recompute `effective_cost_seconds` without altering the stored base edge cost.

pgRouting queries use `routing.effective_edges_pgr_sql(...)`, which projects that runtime view into the `id/source/target/cost/reverse_cost` shape expected by `pgr_dijkstra`, `pgr_dijkstraCost`, and `pgr_drivingDistance`.

The in-memory backend mirrors the same overlay fold so unit tests can assert route changes without Docker.

## Performance Notes

- `routing.nodes.geom`, `routing.edges.geom`, and the bounding tables use GIST indexes for nearest-node lookup and map-side inspection.
- The graph is stored as directed edges with preassigned integer vertex ids, so pgRouting does not need a mutable topology build step during normal queries.
- Facilities store nearest-node linkage on ingest, avoiding repeated snap work for common planner endpoints.
- ETA matrices use `pgr_dijkstraCost` over deduplicated origin/destination vertex arrays instead of repeated one-off shortest-path calls.
- Service-area queries use `pgr_drivingDistance`, which keeps the database responsible for reachable-within-budget expansion.

The fixture region is intentionally tiny, so the current implementation optimizes for clarity and testability over bulk-ingest throughput.
If Phase 3 grows into larger regions, the next scaling steps are bulk `COPY` ingest, partitioning by `region_bundle_id`, and denormalized overlay caches for hot routing windows.

## Limitations

- Routing currently supports `travel_mode="road"` only.
- Overlay effects are edge-local; there is no turn-penalty or intersection-state overlay model yet.
- Facility snapping uses nearest-node distance, not driveway geometry or parcel access rules.
- Asset layers are stored for tactical context and later joins, but they do not yet affect route cost automatically.
- The primary integration path assumes a locally running PostGIS + pgRouting container and a `psycopg` client install.
