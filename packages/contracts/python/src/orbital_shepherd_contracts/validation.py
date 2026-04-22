from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]
from referencing import Registry, Resource

from orbital_shepherd_contracts.adapters import (
    compile_scenario_bundle,
    normalize_replay_events,
    to_canonical_manifest,
)
from orbital_shepherd_contracts.constants import SCHEMA_FILE_NAMES
from orbital_shepherd_contracts.models import (
    DispatchUnit,
    Facility,
    RegionBundle,
    RegionManifest,
    ReplayEvent,
    RoutePlan,
    ScenarioBundle,
    ScenarioManifest,
    SpatialIngestManifest,
    TacticalActivation,
    TacticalMetricsSummary,
    TacticalReplayEvent,
    TacticalScenarioBundle,
    TacticalScenarioManifest,
)
from orbital_shepherd_contracts.paths import (
    data_contract_fixture_roots,
    phase0_examples_dir,
    phase0_schemas_dir,
    phase1_examples_dir,
    phase1_schemas_dir,
)
from orbital_shepherd_ephemeris.models import CelesTrakRawSnapshot, OrbitAssetBundle


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@dataclass(frozen=True)
class ValidatedArtifact:
    path: Path
    artifact_type: str
    details: str


class SchemaCatalog:
    def __init__(self, schema_dir: Path) -> None:
        self.schema_dir = schema_dir
        self._schemas_by_name = {
            path.name: load_json(path)
            for path in sorted(schema_dir.glob("*.json"))
            if path.name in SCHEMA_FILE_NAMES or schema_dir == phase0_schemas_dir()
        }
        resources = [
            (schema["$id"], Resource.from_contents(schema))
            for schema in self._schemas_by_name.values()
        ]
        self._registry: Any = Registry().with_resources(resources)

    def validate(self, schema_name: str, instance: Any) -> None:
        schema = self._schemas_by_name[schema_name]
        validator = Draft202012Validator(
            schema,
            registry=self._registry,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        )
        validator.validate(instance)


_PHASE0_CATALOG = SchemaCatalog(phase0_schemas_dir())
_PHASE1_CATALOG = SchemaCatalog(phase1_schemas_dir())


def validate_phase0_instance(schema_name: str, instance: Any) -> None:
    _PHASE0_CATALOG.validate(schema_name, instance)


def validate_phase1_instance(schema_name: str, instance: Any) -> None:
    _PHASE1_CATALOG.validate(schema_name, instance)


def validate_canonical_manifest(instance: Mapping[str, Any]) -> ScenarioManifest:
    validate_phase1_instance("scenario_manifest.schema.json", instance)
    return ScenarioManifest.model_validate(instance)


def validate_canonical_bundle(instance: Mapping[str, Any]) -> ScenarioBundle:
    validate_phase1_instance("scenario_bundle.schema.json", instance)
    return ScenarioBundle.model_validate(instance)


def validate_spatial_ingest_manifest(instance: Mapping[str, Any]) -> SpatialIngestManifest:
    validate_phase1_instance("spatial_ingest_manifest.schema.json", instance)
    return SpatialIngestManifest.model_validate(instance)


def validate_facility(instance: Mapping[str, Any]) -> Facility:
    validate_phase1_instance("facility.schema.json", instance)
    return Facility.model_validate(instance)


def validate_dispatch_unit(instance: Mapping[str, Any]) -> DispatchUnit:
    validate_phase1_instance("dispatch_unit.schema.json", instance)
    return DispatchUnit.model_validate(instance)


def validate_route_plan(instance: Mapping[str, Any]) -> RoutePlan:
    validate_phase1_instance("route_plan.schema.json", instance)
    return RoutePlan.model_validate(instance)


def validate_region_manifest(instance: Mapping[str, Any]) -> RegionManifest:
    validate_phase1_instance("region_manifest.schema.json", instance)
    return RegionManifest.model_validate(instance)


def validate_region_bundle(instance: Mapping[str, Any]) -> RegionBundle:
    validate_phase1_instance("region_bundle.schema.json", instance)
    return RegionBundle.model_validate(instance)


def validate_tactical_activation(instance: Mapping[str, Any]) -> TacticalActivation:
    validate_phase1_instance("tactical_activation.schema.json", instance)
    return TacticalActivation.model_validate(instance)


def validate_tactical_scenario_manifest(
    instance: Mapping[str, Any],
) -> TacticalScenarioManifest:
    validate_phase1_instance("tactical_scenario_manifest.schema.json", instance)
    return TacticalScenarioManifest.model_validate(instance)


def validate_tactical_scenario_bundle(instance: Mapping[str, Any]) -> TacticalScenarioBundle:
    validate_phase1_instance("tactical_scenario_bundle.schema.json", instance)
    return TacticalScenarioBundle.model_validate(instance)


def validate_canonical_replay_events(events: Iterable[Mapping[str, Any]]) -> list[ReplayEvent]:
    normalized_events = []
    previous_event_index = -1
    previous_sim_tick = -1
    for event in events:
        validate_phase1_instance("replay_event.schema.json", event)
        normalized = ReplayEvent.model_validate(event)
        if normalized.event_index <= previous_event_index:
            raise ValueError("event_index must be strictly increasing within a replay")
        if normalized.sim_tick < previous_sim_tick:
            raise ValueError("sim_tick must be monotonically non-decreasing within a replay")
        previous_event_index = normalized.event_index
        previous_sim_tick = normalized.sim_tick
        normalized_events.append(normalized)
    return normalized_events


def validate_tactical_replay_events(
    events: Iterable[Mapping[str, Any]],
) -> list[TacticalReplayEvent]:
    normalized_events = []
    previous_event_index = -1
    previous_sim_tick = -1
    for event in events:
        validate_phase1_instance("tactical_replay_event.schema.json", event)
        normalized = TacticalReplayEvent.model_validate(event)
        if normalized.event_index <= previous_event_index:
            raise ValueError("event_index must be strictly increasing within a replay")
        if normalized.sim_tick < previous_sim_tick:
            raise ValueError("sim_tick must be monotonically non-decreasing within a replay")
        previous_event_index = normalized.event_index
        previous_sim_tick = normalized.sim_tick
        normalized_events.append(normalized)
    return normalized_events


def validate_tactical_metrics_summary(instance: Mapping[str, Any]) -> TacticalMetricsSummary:
    validate_phase1_instance("tactical_metrics_summary.schema.json", instance)
    return TacticalMetricsSummary.model_validate(instance)


def validate_all_contract_examples(
    *,
    include_data_fixtures: bool = False,
) -> list[ValidatedArtifact]:
    results: list[ValidatedArtifact] = []
    results.extend(_validate_phase0_examples())
    results.extend(_validate_contract_package_examples())
    if include_data_fixtures:
        results.extend(_validate_data_fixture_artifacts())
    return results


def _validate_phase0_examples() -> list[ValidatedArtifact]:
    results: list[ValidatedArtifact] = []
    sample_scenario_path = phase0_examples_dir() / "sample_scenario.json"
    sample_scenario = load_json(sample_scenario_path)
    validate_phase0_instance("scenario_bundle.schema.json", sample_scenario)
    canonical_manifest = to_canonical_manifest(sample_scenario)
    validate_canonical_manifest(canonical_manifest.model_dump(mode="json"))
    canonical_bundle = compile_scenario_bundle(sample_scenario)
    validate_canonical_bundle(canonical_bundle.model_dump(mode="json"))
    results.append(
        ValidatedArtifact(
            path=sample_scenario_path,
            artifact_type="legacy-scenario-manifest-like",
            details=f"legacy schema ok, canonical manifest {canonical_manifest.manifest_id}, "
            f"canonical bundle {canonical_bundle.bundle_id}",
        )
    )

    sample_replay_path = phase0_examples_dir() / "sample_replay.ndjson"
    sample_replay = load_ndjson(sample_replay_path)
    for event in sample_replay:
        validate_phase0_instance("replay_event.schema.json", event)
    normalized_replay = normalize_replay_events(sample_replay)
    validate_canonical_replay_events([event.model_dump(mode="json") for event in normalized_replay])
    results.append(
        ValidatedArtifact(
            path=sample_replay_path,
            artifact_type="legacy-replay",
            details=f"{len(normalized_replay)} events normalized and validated",
        )
    )
    return results


def _validate_contract_package_examples() -> list[ValidatedArtifact]:
    results: list[ValidatedArtifact] = []
    manifest_path = phase1_examples_dir() / "phase1_scenario_manifest.json"
    manifest = load_json(manifest_path)
    manifest_model = validate_canonical_manifest(manifest)
    results.append(
        ValidatedArtifact(
            path=manifest_path,
            artifact_type="scenario_manifest",
            details=f"manifest_id={manifest_model.manifest_id}",
        )
    )

    bundle_path = phase1_examples_dir() / "phase1_scenario_bundle.json"
    bundle = load_json(bundle_path)
    bundle_model = validate_canonical_bundle(bundle)
    results.append(
        ValidatedArtifact(
            path=bundle_path,
            artifact_type="scenario_bundle",
            details=f"bundle_id={bundle_model.bundle_id}",
        )
    )

    replay_path = phase1_examples_dir() / "phase1_replay.ndjson"
    replay = load_ndjson(replay_path)
    validate_canonical_replay_events(replay)
    results.append(
        ValidatedArtifact(
            path=replay_path,
            artifact_type="replay",
            details=f"{len(replay)} canonical replay events",
        )
    )

    spatial_ingest_path = phase1_examples_dir() / "phase3_spatial_ingest_manifest.json"
    spatial_ingest = load_json(spatial_ingest_path)
    spatial_ingest_model = validate_spatial_ingest_manifest(spatial_ingest)
    results.append(
        ValidatedArtifact(
            path=spatial_ingest_path,
            artifact_type="spatial_ingest_manifest",
            details=f"spatial_ingest_id={spatial_ingest_model.spatial_ingest_id}",
        )
    )

    region_manifest_path = phase1_examples_dir() / "phase3_region_manifest.json"
    region_manifest = load_json(region_manifest_path)
    region_manifest_model = validate_region_manifest(region_manifest)
    results.append(
        ValidatedArtifact(
            path=region_manifest_path,
            artifact_type="region_manifest",
            details=f"region_manifest_id={region_manifest_model.region_manifest_id}",
        )
    )

    region_bundle_path = phase1_examples_dir() / "phase3_region_bundle.json"
    region_bundle = load_json(region_bundle_path)
    region_bundle_model = validate_region_bundle(region_bundle)
    results.append(
        ValidatedArtifact(
            path=region_bundle_path,
            artifact_type="region_bundle",
            details=f"region_bundle_id={region_bundle_model.region_bundle_id}",
        )
    )

    tactical_activation_path = phase1_examples_dir() / "phase3_tactical_activation.json"
    tactical_activation = load_json(tactical_activation_path)
    tactical_activation_model = validate_tactical_activation(tactical_activation)
    results.append(
        ValidatedArtifact(
            path=tactical_activation_path,
            artifact_type="tactical_activation",
            details=f"activation_id={tactical_activation_model.activation_id}",
        )
    )

    tactical_manifest_path = phase1_examples_dir() / "phase3_tactical_scenario_manifest.json"
    tactical_manifest = load_json(tactical_manifest_path)
    tactical_manifest_model = validate_tactical_scenario_manifest(tactical_manifest)
    results.append(
        ValidatedArtifact(
            path=tactical_manifest_path,
            artifact_type="tactical_scenario_manifest",
            details=f"tactical_manifest_id={tactical_manifest_model.tactical_manifest_id}",
        )
    )

    tactical_bundle_path = phase1_examples_dir() / "phase3_tactical_scenario_bundle.json"
    tactical_bundle = load_json(tactical_bundle_path)
    tactical_bundle_model = validate_tactical_scenario_bundle(tactical_bundle)
    results.append(
        ValidatedArtifact(
            path=tactical_bundle_path,
            artifact_type="tactical_scenario_bundle",
            details=f"tactical_bundle_id={tactical_bundle_model.tactical_bundle_id}",
        )
    )

    tactical_replay_path = phase1_examples_dir() / "phase3_tactical_replay.ndjson"
    tactical_replay = load_ndjson(tactical_replay_path)
    validate_tactical_replay_events(tactical_replay)
    results.append(
        ValidatedArtifact(
            path=tactical_replay_path,
            artifact_type="tactical_replay",
            details=f"{len(tactical_replay)} canonical tactical replay events",
        )
    )

    tactical_metrics_path = phase1_examples_dir() / "phase3_tactical_metrics_summary.json"
    tactical_metrics = load_json(tactical_metrics_path)
    tactical_metrics_model = validate_tactical_metrics_summary(tactical_metrics)
    results.append(
        ValidatedArtifact(
            path=tactical_metrics_path,
            artifact_type="tactical_metrics_summary",
            details=f"summary_id={tactical_metrics_model.summary_id}",
        )
    )
    return results


def _validate_data_fixture_artifacts() -> list[ValidatedArtifact]:
    results: list[ValidatedArtifact] = []
    for root in data_contract_fixture_roots():
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".json", ".ndjson"} or not path.is_file():
                continue
            if path.suffix == ".ndjson":
                events = load_ndjson(path)
                validate_canonical_replay_events(events)
                results.append(
                    ValidatedArtifact(
                        path=path,
                        artifact_type="replay",
                        details=f"{len(events)} canonical replay events",
                    )
                )
                continue

            document = load_json(path)
            artifact_type = _detect_json_artifact_type(document)
            if artifact_type == "scenario_manifest":
                validate_canonical_manifest(document)
            elif artifact_type == "scenario_bundle":
                validate_canonical_bundle(document)
            elif artifact_type == "region_manifest":
                validate_region_manifest(document)
            elif artifact_type == "region_bundle":
                validate_region_bundle(document)
            elif artifact_type == "spatial_ingest_manifest":
                validate_spatial_ingest_manifest(document)
            elif artifact_type == "tactical_activation":
                validate_tactical_activation(document)
            elif artifact_type == "tactical_scenario_manifest":
                validate_tactical_scenario_manifest(document)
            elif artifact_type == "tactical_scenario_bundle":
                validate_tactical_scenario_bundle(document)
            elif artifact_type == "tactical_metrics_summary":
                validate_tactical_metrics_summary(document)
            elif artifact_type == "orbit_asset_bundle":
                OrbitAssetBundle.model_validate(document)
            elif artifact_type == "celestrak_raw_snapshot":
                CelesTrakRawSnapshot.model_validate(document)
            else:
                raise ValidationError(f"unrecognized JSON contract artifact: {path}")
            results.append(ValidatedArtifact(path=path, artifact_type=artifact_type, details="ok"))
    return results


def _detect_json_artifact_type(document: Mapping[str, Any]) -> str:
    if "raw_snapshot_sha256" in document and "records" in document:
        return "celestrak_raw_snapshot"
    if "bundle_fingerprint" in document and "assets" in document:
        return "orbit_asset_bundle"
    if "region_bundle_id" in document:
        return "region_bundle"
    if "region_manifest_id" in document:
        return "region_manifest"
    if "spatial_ingest_id" in document:
        return "spatial_ingest_manifest"
    if "summary_id" in document and "tactical_bundle_id" in document:
        return "tactical_metrics_summary"
    if "tactical_bundle_id" in document:
        return "tactical_scenario_bundle"
    if "tactical_manifest_id" in document:
        return "tactical_scenario_manifest"
    if "activation_id" in document and "incident_packet" in document:
        return "tactical_activation"
    if "compilation" in document and "bundle_id" in document:
        return "scenario_bundle"
    if "manifest_id" in document:
        return "scenario_manifest"
    raise ValidationError("artifact type could not be determined")
