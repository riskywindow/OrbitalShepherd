# Phase 3 Tactical Scenario Engine

`packages/tactical_scenario_engine` compiles the Phase 3 tactical benchmark layer.
It keeps the orbital-to-tactical bridge explicit, deterministic, and replayable:

1. `IncidentPacket`
2. `TacticalActivation`
3. `TacticalScenarioManifest`
4. `TacticalScenarioBundle`

The engine does not read live routing or fleet state from a database.
It builds declarative scenario artifacts against immutable `RegionBundle` inputs and uses the in-memory routing backend only to synthesize deterministic route plans.

## Main Surfaces

- `compile_incident_packet_to_activation(...)`
  - resolves a region bundle
  - normalizes incident geometry, severity, and bridge provenance
- `compile_activation_to_manifest(...)`
  - expands a family template into facilities, units, depot assignments, scouts, and overlay events
- `compile_manifest_to_bundle(...)`
  - compiles overlay-aware route plans
  - fingerprints the immutable runtime bundle
- `build_scenario_pack(...)`
  - materializes the 24-scenario deterministic tactical benchmark pack
- `validate_scenario_pack(...)`
  - rebuilds the pack and compares canonical JSON plus fingerprints

## TacticalActivation

`TacticalActivation` is the normalized handoff artifact between orbital detection and tactical compilation.
It contains:

- the original `IncidentPacket`
- `region_selection`
  - `region_id`
  - `region_manifest_id`
  - `region_bundle_id`
  - `region_bundle_fingerprint`
  - deterministic resolution strategy and matched H3 cell when available
- `incident_context`
  - normalized incident geometry
  - severity class and score
  - urgency, confidence, downstream value
  - activation delay from packet downlink to tactical activation
- `bridge_provenance`
  - explicit `incident_packet_to_tactical_activation` bridge metadata
  - source observation/downlink timestamps
  - bridge compiler version
- `activation_fingerprint`

This makes the orbital-to-tactical bridge inspectable without requiring replay context or a live service.

## TacticalScenarioBundle

`TacticalScenarioBundle` is the immutable tactical runtime artifact consumed by planners or a later ground environment.
Each bundle contains:

- tactical manifest and activation references
- `region_selection` with selected region id and region bundle fingerprint
- `incident_context` with geometry, severity, urgency, and timing metadata
- `dispatch_units`
- `depot_assignments`
- scenario facilities including temporary command/drop/air nodes
- optional `scout_assets`
- `overlay_events`
  - closures
  - risk zones
  - temporary penalties
- compiled `route_plans`
- deterministic `simulation_seed`
- bundle fingerprint and compilation provenance

## Supported Families

The built-in family set is:

- `foothill_access`
- `urban_interface`
- `closure_cascade`
- `depot_saturation`
- `smoke_corridor`
- `drone_scout_gap`

The checked-in benchmark pack includes 24 bundles total, four deterministic seeds per family.

## Commands

Build the pack:

```bash
python3 scripts/build_tactical_scenario_pack.py
```

Validate the pack:

```bash
python3 scripts/validate_tactical_scenario_pack.py
```

Compile an activation from a fixture packet:

```bash
python3 scripts/phase3_tactical_scenarios.py compile-activation \
  data/fixtures/tactical_scenario_engine/incident_packets/foothill_access_seed4101.json \
  /tmp/foothill_activation.json \
  --region-bundle data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json
```

Compile a bundle directly from a packet plus region selection:

```bash
python3 scripts/phase3_tactical_scenarios.py compile-incident-packet \
  data/fixtures/tactical_scenario_engine/incident_packets/foothill_access_seed4101.json \
  /tmp/foothill_bundle.json \
  --scenario-family foothill_access \
  --seed 4101 \
  --region-bundle data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json
```

Compile from a saved manifest:

```bash
python3 scripts/phase3_tactical_scenarios.py compile-manifest \
  packages/contracts/examples/phase3_tactical_scenario_manifest.json \
  /tmp/phase3_example_bundle.json \
  --region-bundle data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json
```

Inspect a compiled bundle:

```bash
python3 scripts/phase3_tactical_scenarios.py inspect-bundle \
  data/tactical_scenarios/phase3-pack-v1/tsb--inc-osbench-phase3-tactical-v1-foothill_access-seed-4101--foothill_access--seed-4101.json
```

Equivalent make targets:

- `make phase3-tactical-pack-build`
- `make phase3-tactical-pack-validate`

## Pack Layout

Default outputs live under:

- compiled benchmark bundles: `data/tactical_scenarios/phase3-pack-v1/`
- fixture packet examples: `data/fixtures/tactical_scenario_engine/incident_packets/`

The default region input is the checked-in fixture bundle:

- `data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json`

## Determinism Rules

- region resolution is stable for a fixed packet and catalog
- incident geometry fallback is deterministic when packets omit geometry
- family templates expand deterministically from `scenario_family` and `simulation_seed`
- route synthesis uses the in-memory routing backend against immutable region data
- fingerprints exclude only the fingerprint field being computed
- pack validation rebuilds every bundle and compares canonical serialized JSON
