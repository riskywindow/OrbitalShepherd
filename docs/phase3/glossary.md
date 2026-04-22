# Phase 3 Glossary

## RegionManifest

Authoring artifact describing a tactical region before compilation.

## SpatialIngestManifest

Immutable provenance record for a spatial source layer used to build a region.

## RegionBundle

Compiled immutable regional artifact containing the facilities and traversable graph metadata needed by tactical execution.

## TacticalActivation

Typed bridge object that wraps the canonical `IncidentPacket` with tactical-region context and activation intent.

## TacticalScenarioManifest

Authoring artifact for a tactical episode.
It declares the activation reference, tactical units, facilities, objectives, and config.

## TacticalScenarioBundle

Compiled immutable tactical runtime artifact.
It is the tactical analogue of orbital `ScenarioBundle`.

## DispatchUnit

Canonical tactical unit object representing an engine, crew, dozer, helicopter, tanker, or command asset.

## Facility

Canonical fixed tactical node such as a station, helibase, drop point, hospital, staging area, or command post.

## RoutePlan

Canonical route artifact for moving a specific dispatch unit between tactical nodes.

## TacticalReplayEvent

Append-only tactical replay envelope with typed payloads and a fixed canonical event taxonomy.

## TacticalMetricsSummary

Replay-derived tactical benchmark summary used to compare tactical planners without tying the contract to a learning algorithm.
