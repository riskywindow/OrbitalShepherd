# Phase 3 Region Builder

`packages/region_builder` compiles a typed `RegionManifest` into an immutable, portable `RegionBundle`.

The pipeline follows the same discipline as the orbital scenario compiler:

- manifests are small authoring recipes
- compiled bundles are deterministic and fingerprinted
- portable exports are materialized as separate derived artifacts
- checked-in fixture inputs keep local tests offline

## What Compiles

The region compiler materializes all of the runtime state the tactical layer needs:

- canonical directed road graph
  - `road_nodes[]`
  - `road_edges[]`
- travel-time defaults
  - default speed
  - per-highway speed overrides
  - intersection penalty
- facilities
  - stations, helibases, staging areas, command posts, and related tactical nodes
- asset features
  - point POIs and polygon asset zones
- H3 cover metadata
  - deterministic sorted cell ids
  - resolution and generation strategy
- provenance
  - `SpatialIngestManifest`
  - manifest hash
  - compiler version
  - bundle fingerprint

## Source Selection

Road-network compilation supports ordered fallbacks through `road_network.sources[]`.

Supported source kinds:

- `fixture_geojson`
  - checked-in or locally staged GeoJSON linework
  - preferred for CI and smoke tests
- `osmnx`
  - optional larger-region path for pilot manifests
  - only used when `osmnx` is installed and the earlier local sources are unavailable

The compiler always picks the lowest `fallback_priority` source that is usable. This keeps the fixture path offline while still allowing larger local pilot builds.

## H3 Strategy

`h3_cover` supports two strategies:

- `explicit`
  - used by the checked-in fixture manifest
  - no extra dependency required
- `bounds`
  - uses `h3.geo_to_cells(...)` when the optional `h3` package is installed

This keeps local smoke tests dependency-light without blocking a real H3-backed pilot compile path.

## Portable Outputs

`packages/geo_artifacts` exports bundle layers into:

- GeoParquet
  - canonical portable layer export
  - stored with WKB geometry and GeoParquet metadata
- GeoJSON
  - debugging and inspection format for small regions

Current exported layers:

- `road_nodes`
- `road_edges`
- `facilities`
- `asset_features`

The compiled `RegionBundle` itself remains the canonical immutable tactical artifact.

## Checked-In Fixture

Offline fixture inputs live under:

- `data/fixtures/region_builder/raw/fixture_micro_region_roads.geojson`
- `data/regions/manifests/fixture_micro_region.json`

Checked-in compiled outputs live under:

- `data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json`
- `data/fixtures/region_builder/exports/fixture_micro_region/`

The fixture compile path requires no live downloads.

## Pilot Manifests

Additional manifests live under `data/regions/manifests/`:

- `fixture_micro_region.json`
- `pilot_wui_region.json`
- `pilot_dense_interface_region.json`

The pilot manifests are set up to:

1. prefer a checked-in or locally staged GeoJSON extract if one exists
2. otherwise use the optional OSMnx path

## Commands

Compile a manifest into a bundle:

```bash
python3 scripts/build_region_bundle.py compile-manifest \
  data/regions/manifests/fixture_micro_region.json \
  data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json \
  --export-dir data/fixtures/region_builder/exports/fixture_micro_region
```

Run the focused tests:

```bash
pytest tests/region_builder/test_region_builder.py \
  tests/contracts/test_phase3_tactical_contracts.py \
  tests/contracts/test_contract_normalization.py -q
```

## Determinism Rules

- nodes and edges are sorted by stable ids before serialization
- H3 cells are sorted and deduplicated before fingerprinting
- bundle fingerprints exclude only `bundle_fingerprint` itself
- compilation timestamps come from `RegionBuilderConfig`
- the fixture bundle is rebuilt from the committed manifest and road extract

## Attribution

Road data derived from OpenStreetMap should retain attribution. The compiler appends an OSM attribution note to provenance whenever it uses either the checked-in OSM-style fixture path or the OSMnx-backed path.
