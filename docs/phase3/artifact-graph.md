# Phase 3 Artifact Graph

## Canonical Graph

```text
SpatialIngestManifest
        |
        v
   RegionManifest ---> RegionBundle
                           ^
                           |
IncidentPacket ---> TacticalActivation
                           |
                           v
                 TacticalScenarioManifest
                           |
                           v
                 TacticalScenarioBundle
                           |
                           v
                    TacticalReplayEvent*
                           |
                           v
                 TacticalMetricsSummary
```

## Notes

- `IncidentPacket` remains the official cross-phase handoff object.
- `TacticalActivation` is the explicit typed bridge, not a replacement for `IncidentPacket`.
- `RegionBundle` is immutable and compiled, just like orbital `ScenarioBundle`.
- `TacticalScenarioBundle` is the tactical runtime artifact a planner or environment consumes.
- `TacticalMetricsSummary` is replay-derived and planner-agnostic.

## Object Reuse

- `Facility` can appear in `RegionBundle`, `TacticalScenarioManifest`, and `TacticalScenarioBundle`.
- `DispatchUnit` appears in tactical scenario artifacts.
- `RoutePlan` is compiled into `TacticalScenarioBundle` and referenced by tactical replay.

## Why The Graph Is Split This Way

- spatial ingest provenance should exist independently from tactical episodes
- region compilation should be reusable across many incidents
- incident activation should preserve the orbital packet exactly
- tactical execution should consume compiled bundles, not ad hoc mutable inputs
