# Phase 1 Ephemeris Service

## Purpose

Phase 1 treats CelesTrak GP/OMM snapshots as the canonical external ephemeris input. The ingest path converts those raw snapshots into a deterministic `OrbitAssetBundle` that the scenario engine and later opportunity-building logic can consume without depending on live network access or heavyweight propagation services.

The implementation lives in [`packages/ephemeris/src/orbital_shepherd_ephemeris`](/Users/rishivinodkumar/OrbitalShepherd/packages/ephemeris/src/orbital_shepherd_ephemeris).

## Components

- `CelesTrakClient`
  - fetches a live `gp.php?GROUP=...&FORMAT=json` snapshot, or wraps a local fixture JSON array
  - persists immutable raw snapshots under [`data/fixtures/ephemeris/raw`](/Users/rishivinodkumar/OrbitalShepherd/data/fixtures/ephemeris/raw)
  - normalizes identifiers and mean-element fields into an `OrbitAssetBundle`
- `PropagationBackend`
  - abstract seam for loading a normalized orbit asset snapshot, sampling state, and computing coarse visibility/contact facts
- `DeterministicKeplerPropagationBackend`
  - Phase 1 backend
  - uses a lightweight local Keplerian propagator with WGS84 coordinate transforms
  - deterministic, dependency-light, and CI-safe

## Orbit Asset Format

The normalized artifact is the `OrbitAssetBundle` Pydantic model in [`models.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/ephemeris/src/orbital_shepherd_ephemeris/models.py).

Top-level fields:

- `schema_version`
- `bundle_id`
- `source`
  - provider, source mode, catalog group, snapshot ID, fetched timestamp, raw snapshot fingerprint, raw snapshot path
- `compilation`
  - deterministic `compiled_at_utc` and compiler version
- `assets`
  - one `OrbitAsset` per NORAD object, sorted by NORAD catalog ID
- `bundle_fingerprint`

Each `OrbitAsset` contains:

- `satellite_id`
- `norad_catalog_id`
- `name`
- optional external identity metadata such as `international_designator`
- `orbit`
  - epoch, mean motion, inclination, RAAN, eccentricity, argument of perigee, mean anomaly, and related coarse mean-element fields
- `source`
  - snapshot provenance and raw-record fingerprint
- `asset_fingerprint`

The compiled golden bundle is stored at [`data/fixtures/ephemeris/compiled/eph--demo-phase1--raw-celestrak-demo-phase1-2026-04-01t00-00-00z.json`](/Users/rishivinodkumar/OrbitalShepherd/data/fixtures/ephemeris/compiled/eph--demo-phase1--raw-celestrak-demo-phase1-2026-04-01t00-00-00z.json).

## CLI

Repo-local scripts are provided so the commands work without installing the project as a package:

```bash
python scripts/fetch_ephemeris.py \
  --group demo-phase1 \
  --input tests/fixtures/ephemeris/celestrak_demo_records.json \
  --output-dir data/fixtures/ephemeris/raw

python scripts/compile_orbit_assets.py \
  data/fixtures/ephemeris/raw/raw:celestrak:demo-phase1:2026-04-01t00-00-00z--f7955187d221.json \
  --output-dir data/fixtures/ephemeris/compiled
```

For fixture-backed fetches, if `--fetched-at` is omitted the script derives it from the first record `EPOCH`, so repeated runs stay deterministic.

## Phase 1 Backend Choice

The selected backend is intentionally coarse:

- it propagates classical mean elements locally with no external binary/runtime dependency
- it computes satellite samples in ECI and ECEF coordinates
- it derives coarse contact windows from ground-point elevation and off-nadir checks

This is sufficient for Phase 1 scenario compilation and opportunity-seeding work, while keeping CI fast and deterministic.

## Orekit Integration Seam

An Orekit-backed implementation should plug in by implementing `PropagationBackend` with the same three responsibilities:

1. load an `OrbitAssetBundle`
2. sample satellite states for a time window
3. compute coarse visibility/contact facts against a set of ground targets

That lets Phase 1 keep the stable ingest and bundle format while swapping only the propagation engine. The `OrbitAssetBundle` is deliberately backend-agnostic: the CelesTrak adapter emits normalized mean elements plus immutable provenance, and a future Orekit sidecar can interpret those same inputs at higher fidelity without changing scenario-engine-facing artifacts.
