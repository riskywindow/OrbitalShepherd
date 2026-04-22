# Phase 1 Scenario Engine

## Purpose

The Phase 1 scenario engine compiles deterministic wildfire authoring recipes into immutable `ScenarioBundle` artifacts under [`data/scenarios`](/Users/rishivinodkumar/OrbitalShepherd/data/scenarios). It sits on top of the normalized Phase 1 contracts and the deterministic ephemeris layer, so a developer can rebuild the full benchmark fixture pack locally with no live network calls.

The implementation lives in [`packages/scenario_engine/src/orbital_shepherd_scenario_engine`](/Users/rishivinodkumar/OrbitalShepherd/packages/scenario_engine/src/orbital_shepherd_scenario_engine).

## Compilation Pipeline

The compiler runs in two deterministic stages:

1. `ScenarioRecipe -> ScenarioManifest`
   - target-demand points are aggregated to fixed H3 cells
   - wildfire incidents are materialized from fixture-backed or synthetic generators
   - scenario config captures the adapter modes and family parameters needed for replayable rebuilds
2. `ScenarioManifest -> ScenarioBundle`
   - the ephemeris bundle drives target visibility and downlink contact generation
   - cloud risk is resolved from fixture-backed or synthetic weather adapters
   - observation opportunities are emitted with `predicted_quality_mean` treated as a usefulness proxy rather than raw imagery quality

The bundle writer uses canonical JSON plus a fixed compilation timestamp so the same recipe seed produces identical scenario bytes and the same `bundle_fingerprint`.

## Scenario Modes

Phase 1 supports three adapter modes:

- `fixture`
  - historical-style replay using local CSV fixtures for demand points, ignition templates, and cloud profiles
- `synthetic`
  - deterministic seeded generation with no external dependencies
- `live`
  - explicit boundary only; the interfaces exist, but Phase 1 does not implement live data fetches and tests do not require them

Each manifest stores adapter mode metadata in `config.adapter_modes`.

## Wildfire Modeling

Phase 1 intentionally does not simulate raw imagery.

Observation usefulness is modeled as a deterministic function of:

- geometry
  - peak elevation and minimum off-nadir from the ephemeris visibility window
- cloud risk
  - fixture or synthetic cloud obstruction probability at the opportunity midpoint
- urgency
  - maximum urgency among active incidents in the target cell
- delay
  - freshness decay from ignition time to candidate observation time

The compiler folds those terms into `predicted_quality_mean`, which Phase 1 uses as a precomputed usefulness score for downstream opportunity selection.

## Family Parameterization

Canonical families:

- `sparse_frontier` = Sparse Frontier
  - few incidents
  - broad geographic dispersion
  - lower contention
- `burst_outbreak` = Burst Outbreak
  - many ignitions in a short window
  - high urgency pressure
  - variable cloud and smoke conditions
- `cloud_trap` = Cloud Trap
  - early high-value windows have heavy cloud risk
  - later cleaner passes arrive with freshness loss

The built-in fixture pack contains 10 scenarios:

- 4 `sparse_frontier`
- 3 `burst_outbreak`
- 3 `cloud_trap`

## Configuration Knobs

Primary knobs live in [`catalog.py`](/Users/rishivinodkumar/OrbitalShepherd/packages/scenario_engine/src/orbital_shepherd_scenario_engine/catalog.py) recipe definitions and in manifest `config` extras.

Important knobs:

- `scenario_mode`
  - dominant authoring mode for the recipe (`fixture`, `synthetic`, or `hybrid`)
- `adapter_modes`
  - per-source mode split for demand, incidents, and weather
- `family_parameters`
  - family-specific deterministic controls such as burst window length, trap relief hour, demand scale, and urgency bias
- `weather_profile_by_target_cell`
  - fixture weather profile mapping used during manifest-to-bundle compilation
- `opportunity_generation.quality_threshold`
  - minimum usefulness threshold for non-incident monitoring opportunities
- `opportunity_generation.cloud_block_threshold`
  - family-specific cloud severity reference retained in the manifest for downstream consumers

## Fixture Inputs

The local no-network fixture tables live in [`data/fixtures/scenario_engine`](/Users/rishivinodkumar/OrbitalShepherd/data/fixtures/scenario_engine):

- [`target_demand_points.csv`](/Users/rishivinodkumar/OrbitalShepherd/data/fixtures/scenario_engine/target_demand_points.csv)
- [`incident_replay_rows.csv`](/Users/rishivinodkumar/OrbitalShepherd/data/fixtures/scenario_engine/incident_replay_rows.csv)
- [`cloud_profiles.csv`](/Users/rishivinodkumar/OrbitalShepherd/data/fixtures/scenario_engine/cloud_profiles.csv)

These are not contract artifacts; they are local deterministic authoring fixtures consumed only by the scenario engine.

## CLI

Build the pack:

```bash
python scripts/build_scenario_pack.py
```

Validate schema, fingerprint, and deterministic rebuild equivalence:

```bash
python scripts/validate_scenario_pack.py
```

Compile one manifest through the Phase 1 engine:

```bash
python -m orbital_shepherd_scenario_engine compile-manifest input.json output.json
```

## Output Contract

Each compiled scenario bundle includes:

- stable `bundle_id`
- stable `bundle_fingerprint`
- canonical manifest provenance
- deterministic `observation_opportunities`
- deterministic `downlink_windows`

The resulting bundles are ready to be consumed directly by the future opportunity builder and environment runtime.
