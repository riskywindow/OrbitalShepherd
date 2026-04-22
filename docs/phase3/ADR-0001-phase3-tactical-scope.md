# ADR-0001: Phase 3 Tactical Scope

## Status

Accepted on 2026-04-19.

## Context

Phase 1 made the orbital simulator and replay boundary executable.
Phase 2 added orbital-only training and evaluation on top of that boundary.
The remaining gap is the handoff from orbital wildfire detection into local dispatch and route synthesis.

That gap was only implicit.
The repo had an `IncidentPacket` contract and a stated intent to hand incidents to a later tactical layer, but it did not yet define:

- what tactical artifacts exist
- what the orbital layer owns versus what the tactical layer owns
- how manifests, fingerprints, and replay work once the problem leaves orbit
- how to add tactical planning without prematurely locking the repo into one planner or one RL strategy

## Decision

Phase 3 is the tactical foundation layer.

It introduces typed contracts and package scaffolding for:

- deterministic spatial ingest provenance
- deterministic regional compilation
- explicit `IncidentPacket` to `TacticalActivation` handoff
- tactical scenario manifests and compiled bundles
- tactical route-plan objects
- canonical tactical replay events
- replay-derived tactical metrics summaries

Phase 3 remains planner-agnostic.
The tactical layer is defined as a contract surface that can be consumed by heuristics, OR solvers, imitation pipelines, or later RL agents without changing the artifact model.

## What Phase 3 Is For

Phase 3 exists to formalize the tactical execution substrate.

In scope:

- `RegionManifest` and `RegionBundle` as the deterministic geographic runtime boundary
- `SpatialIngestManifest` as explicit provenance for roads, facilities, hazards, terrain, and weather layers
- `TacticalActivation` as the typed bridge from orbital detection into tactical planning
- `TacticalScenarioManifest` and `TacticalScenarioBundle` as the tactical authoring/runtime pair
- `DispatchUnit`, `Facility`, and `RoutePlan` as canonical tactical objects
- `TacticalReplayEvent` and `TacticalMetricsSummary` as append-only replay and replay-derived reporting artifacts
- package scaffolding for region building, geo artifacts, routing, tactical scenarios, ground execution, tactical baselines, tactical metrics, and escalation bridging

## Out of Scope

Still out of scope in Phase 3:

- local tactical RL training
- hierarchical RL linking orbital and tactical policies
- live GIS backends, fleet telemetry ingestion, or message buses
- replacing the existing orbital API/UI with a tactical UI surface
- changing `IncidentPacket` from the official cross-phase handoff object
- collapsing orbital and tactical replay into one overloaded schema

## Orbital Versus Tactical Boundary

The orbital layer owns:

- detection from satellites through downlink
- orbital `ScenarioManifest` and `ScenarioBundle`
- orbital `ReplayEvent`
- benchmark metrics for orbital planners
- emission of `IncidentPacket`

The tactical layer owns:

- regional spatial context and its compilation artifacts
- conversion of an `IncidentPacket` into a `TacticalActivation`
- dispatch-unit, facility, and route-plan state
- tactical `TacticalScenarioManifest` and `TacticalScenarioBundle`
- tactical `TacticalReplayEvent`
- replay-derived tactical metrics

The bridge is explicit:

1. orbital replay emits `incident_packet_emitted`
2. the emitted `IncidentPacket` remains the official handoff payload
3. Phase 3 wraps that packet in `TacticalActivation`
4. tactical planning starts from `TacticalActivation` plus a compiled `RegionBundle`

This keeps the orbital product surface intact and typed while making the tactical step auditable.

## Why Phase 3 Is Planner-Agnostic

The repo does not yet know whether tactical planning will settle on:

- heuristics
- OR / routing solvers
- imitation learning
- RL
- some hybrid stack

Locking contracts to one planner family now would contaminate the platform boundary with premature implementation detail.

Phase 3 therefore standardizes:

- the inputs a planner may consume
- the append-only replay it must emit
- the summary metrics used to compare outputs

It does not standardize:

- the internal planner algorithm
- an action-value interface for RL
- a policy checkpoint format for tactical execution

## Why Phase 3 Is Not Hierarchical RL Yet

Hierarchical RL would introduce assumptions the repo cannot defend yet:

- a stable tactical environment interface
- settled tactical action semantics
- replay fields sufficient for credit assignment
- reward definitions that do not distort planner-agnostic benchmarking

Phase 3 intentionally stops earlier.
It defines the tactical environment contract, artifact graph, and replay taxonomy first.
That makes later HRL work possible without requiring another round of contract rewrites.

## Consequences

Positive:

- Phase 3 turns a vague tactical idea into an implementation contract
- `IncidentPacket` remains stable and official
- tactical work can proceed without bypassing deterministic manifests, fingerprints, or replay
- later tactical planners can compete on shared artifacts

Tradeoffs:

- there are now parallel orbital and tactical replay families
- more artifacts exist before full tactical execution logic exists
- some package surfaces are scaffolding until Phase 4 implementation work lands
