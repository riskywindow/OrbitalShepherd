# ADR-0001: Contract Normalization

## Status

Accepted on 2026-04-09.

## Context

Phase 0 froze the problem framing successfully, but the contract package itself drifted in three places that block executable validation:

1. `ScenarioBundle` was used as both the authoring input and the immutable runtime artifact.
   The provided sample scenario does not include compilation provenance or compiled opportunity/window data, so it behaves more like an authoring manifest than a fully compiled simulator bundle.
2. The replay contract was only partially normalized.
   The docs describe a compact canonical taxonomy, while the JSON Schema only models a subset of payload shapes and leaves some event kinds under-specified.
3. `packages/contracts` did not actually contain the schemas referenced by the docs.
   The schemas lived at the repo root, and the Python package only exposed path helpers.

The result was understandable for a human reader but not executable as a package boundary for the API, environment runtime, or validation tooling.

## Decision

Phase 1 introduces an explicit three-artifact model:

- `ScenarioManifest`: the authoring recipe.
- `ScenarioBundle`: the compiled immutable simulator artifact.
- `ReplayEvent`: the canonical append-only event envelope.

Canonical Phase 1 contracts now live under [packages/contracts/schemas](/Users/rishivinodkumar/OrbitalShepherd/packages/contracts/schemas) and use JSON Schema 2020-12 with `schema_version = "1.0.0"`.

### Normalized artifact boundary

`ScenarioManifest` carries the scenario inputs an author or generator is expected to control directly:

- mission horizon
- decision interval
- satellites
- ground stations
- target cells
- incidents
- scenario config

`ScenarioBundle` is what the API and environment runtime consume. It preserves the Phase 0 structure, but adds the missing compilation boundary:

- `manifest_id`
- `compilation.source_manifest_id`
- `compilation.source_manifest_schema_version`
- `compilation.source_manifest_sha256`
- `compilation.compiled_at_utc`
- `compilation.compiler_version`
- compiled `observation_opportunities`
- compiled `downlink_windows`

This keeps the spirit of the Phase 0 `ScenarioBundle` while making the runtime artifact unambiguously immutable and derived.

### Replay event normalization

The Phase 1 replay taxonomy keeps the Phase 0 envelope shape but tightens the event names and payloads.

Canonical Phase 1 event names:

- `scenario_bundle_loaded`
- `episode_started`
- `candidate_set_materialized`
- `action_mask_emitted`
- `action_selected`
- `observation_executed`
- `downlink_executed`
- `incident_packet_emitted`
- `reward_assessed`
- `episode_ended`

Compatibility aliases are explicit, not implicit:

| Legacy Phase 0 name | Canonical Phase 1 name |
| --- | --- |
| `scenario_loaded` | `scenario_bundle_loaded` |
| `opportunities_materialized` | `candidate_set_materialized` |
| `observation_committed` | `observation_executed` |
| `downlink_committed` | `downlink_executed` |

Unchanged names remain valid after normalization:

- `episode_started`
- `action_mask_emitted`
- `action_selected`
- `incident_packet_emitted`
- `reward_assessed`
- `episode_ended`

### Backward compatibility policy

Backward compatibility is preserved where it is practical and explicit:

1. The repo-root Phase 0 schemas and examples remain in place as legacy artifacts.
2. The validation tooling continues to validate the provided Phase 0 examples.
3. A manifest-to-bundle compiler adapter defines the new runtime boundary.
4. Replay events are normalized through a public alias map before canonical validation.

This avoids hidden dual semantics inside a single schema.

## Mismatches Identified

### 1. Bundle versus manifest semantics

Observed mismatch:

- [04_data_contracts.md](/Users/rishivinodkumar/OrbitalShepherd/04_data_contracts.md) and [05_planner_api.openapi.yaml](/Users/rishivinodkumar/OrbitalShepherd/05_planner_api.openapi.yaml) speak in terms of `ScenarioBundle`.
- [examples/sample_scenario.json](/Users/rishivinodkumar/OrbitalShepherd/examples/sample_scenario.json) has no compilation metadata and no compiled candidate windows.

Normalization:

- treat the Phase 0 sample scenario as a legacy manifest-shaped artifact
- compile it into a canonical `ScenarioBundle` explicitly

### 2. Replay taxonomy versus replay schema coverage

Observed mismatch:

- [04_data_contracts.md](/Users/rishivinodkumar/OrbitalShepherd/04_data_contracts.md) defines the taxonomy at the doc level
- [schemas/replay_event.schema.json](/Users/rishivinodkumar/OrbitalShepherd/schemas/replay_event.schema.json) only partially constrains payloads and leaves some event kinds undocumented in executable form

Normalization:

- canonicalize the event names for the runtime contract
- define payload requirements for every canonical event kind
- provide an adapter for legacy names and legacy payload field names

### 3. Missing package-local schemas

Observed mismatch:

- the docs reference a schema package
- [packages/contracts/src/orbital_shepherd_contracts](/Users/rishivinodkumar/OrbitalShepherd/packages/contracts/src/orbital_shepherd_contracts) only exposed path helpers

Normalization:

- add package-local schemas, examples, Python validators/models, TypeScript types/utilities, and a validation CLI

## Consequences

Positive:

- the contracts are now executable instead of only descriptive
- the API/runtime boundary is unambiguous
- another engineer can read one canonical package and one ADR without reverse-engineering Phase 0 intent

Tradeoffs:

- there are now legacy and canonical artifacts in the repository
- adapters add code, but that complexity is preferable to silently mutating old files or overloading one schema with two meanings

## Implementation Notes

- JSON Schema version: 2020-12 everywhere in the canonical package
- Canonical schema version: `1.0.0`
- Legacy Phase 0 schema version: `0.1.0`
- Validation strategy: JSON Schema for artifact compatibility, Pydantic v2 for Python models, and explicit adapters for legacy inputs
