from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import cos, radians, sqrt
from pathlib import Path
from typing import Any

from orbital_shepherd_contracts import (
    DispatchUnit,
    Facility,
    IncidentPacket,
    RegionBundle,
    RoutePlan,
    ScoutAsset,
    TacticalActivation,
    TacticalBridgeProvenance,
    TacticalDepotAssignment,
    TacticalIncidentContext,
    TacticalIncidentGeometry,
    TacticalOverlayEdgeEffect,
    TacticalOverlayEvent,
    TacticalRegionSelection,
    TacticalScenarioBundle,
    TacticalScenarioManifest,
    load_json,
)
from orbital_shepherd_contracts.models import (
    BundleCompilation,
    RegionResolutionStrategy,
    TacticalScenarioConfig,
    TacticalSeverityClass,
    TimeWindow,
    Wgs84GroundPoint,
    Wgs84Point,
)
from orbital_shepherd_core import (
    canonical_json_dumps,
    seeded_rng,
    sha256_fingerprint,
    stable_id,
)
from orbital_shepherd_region_builder.compiler import region_bundle_fingerprint
from orbital_shepherd_routing_engine import (
    ClosureEdgeEffect,
    ClosureOverlaySpec,
    RiskMultiplierEdgeEffect,
    RiskMultiplierOverlaySpec,
    RoutingEndpoint,
    RoutingEngineService,
    ShortestPathRequest,
    TemporaryRestrictionEdgeEffect,
    TemporaryRestrictionOverlaySpec,
)
from orbital_shepherd_routing_engine.geometry import haversine_m
from orbital_shepherd_routing_engine.models import OverlaySelection, OverlayWindow
from orbital_shepherd_tactical_scenario_engine.catalog import (
    FAMILY_SPECS,
    TacticalFamilySpec,
    TacticalScenarioRecipe,
    UnitTemplate,
    builtin_phase3_recipes,
)
from orbital_shepherd_tactical_scenario_engine.config import TacticalScenarioEngineConfig


@dataclass(frozen=True, slots=True)
class RegionResolutionRecord:
    bundle: RegionBundle
    selection: TacticalRegionSelection
    source_path: Path | None


@dataclass(frozen=True, slots=True)
class TacticalBundleRecord:
    tactical_bundle_id: str
    scenario_family: str
    bundle_fingerprint: str
    path: Path


def tactical_activation_fingerprint(document: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(document))
    normalized.pop("activation_fingerprint", None)
    return sha256_fingerprint(canonical_json_dumps(normalized))


def tactical_bundle_id_from_manifest_id(tactical_manifest_id: str) -> str:
    if tactical_manifest_id.startswith("tsm:"):
        return f"tsb:{tactical_manifest_id[4:]}"
    return f"tsb:{tactical_manifest_id}"


def tactical_bundle_fingerprint(document: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(document))
    normalized.pop("bundle_fingerprint", None)
    return sha256_fingerprint(canonical_json_dumps(normalized))


def resolve_region_for_incident_packet(
    packet: IncidentPacket | Mapping[str, Any],
    *,
    region_bundle: RegionBundle | Mapping[str, Any] | Path | str | None = None,
    catalog_sources: Sequence[RegionBundle | Mapping[str, Any] | Path | str] | None = None,
    config: TacticalScenarioEngineConfig | None = None,
) -> RegionResolutionRecord:
    packet_model = _coerce_packet(packet)
    if region_bundle is not None:
        bundle_model = _load_region_bundle(region_bundle)
        matched_h3 = _matched_h3_cell(packet_model, bundle_model)
        selection = TacticalRegionSelection(
            region_id=bundle_model.region_id,
            region_manifest_id=bundle_model.region_manifest_id,
            region_bundle_id=bundle_model.region_bundle_id,
            region_bundle_fingerprint=bundle_model.bundle_fingerprint,
            resolution_strategy="explicit_bundle",
            matched_h3_cell=matched_h3,
            candidate_region_bundle_ids=[bundle_model.region_bundle_id],
        )
        source_path = Path(region_bundle) if isinstance(region_bundle, (Path, str)) else None
        return RegionResolutionRecord(
            bundle=bundle_model,
            selection=selection,
            source_path=source_path,
        )

    engine_config = config or TacticalScenarioEngineConfig()
    sources = tuple(catalog_sources or engine_config.region_bundle_catalog)
    bundle_records = _load_region_catalog(sources)
    if not bundle_records:
        raise LookupError("no region bundles available for tactical region resolution")

    matching_records = [
        record
        for record in bundle_records
        if _matched_h3_cell(packet_model, record.bundle) is not None
    ]
    if matching_records:
        selected = sorted(
            matching_records,
            key=lambda record: (
                len(record.bundle.h3_cover.cell_ids),
                record.bundle.region_bundle_id,
            ),
        )[0]
        strategy: RegionResolutionStrategy = "h3_cover"
        candidate_ids = [record.bundle.region_bundle_id for record in matching_records]
        matched_h3 = _matched_h3_cell(packet_model, selected.bundle)
    elif len(bundle_records) == 1:
        selected = bundle_records[0]
        strategy = "fallback_single_bundle"
        candidate_ids = [selected.bundle.region_bundle_id]
        matched_h3 = None
    else:
        raise LookupError(
            f"unable to resolve region bundle for incident packet {packet_model.packet_id!r}"
        )

    selection = TacticalRegionSelection(
        region_id=selected.bundle.region_id,
        region_manifest_id=selected.bundle.region_manifest_id,
        region_bundle_id=selected.bundle.region_bundle_id,
        region_bundle_fingerprint=selected.bundle.bundle_fingerprint,
        resolution_strategy=strategy,
        matched_h3_cell=matched_h3,
        candidate_region_bundle_ids=candidate_ids,
    )
    return RegionResolutionRecord(
        bundle=selected.bundle,
        selection=selection,
        source_path=selected.source_path,
    )


def compile_incident_packet_to_activation(
    packet: IncidentPacket | Mapping[str, Any],
    *,
    region_bundle: RegionBundle | Mapping[str, Any] | Path | str | None = None,
    catalog_sources: Sequence[RegionBundle | Mapping[str, Any] | Path | str] | None = None,
    config: TacticalScenarioEngineConfig | None = None,
    activation_time_utc: datetime | None = None,
    requested_capabilities: Sequence[str] | None = None,
    activation_reason: str | None = None,
    incident_geometry_override: TacticalIncidentGeometry | Mapping[str, Any] | None = None,
    provenance_notes: Sequence[str] | None = None,
) -> TacticalActivation:
    engine_config = config or TacticalScenarioEngineConfig()
    packet_model = _coerce_packet(packet)
    resolution = resolve_region_for_incident_packet(
        packet_model,
        region_bundle=region_bundle,
        catalog_sources=catalog_sources,
        config=engine_config,
    )
    activation_time = activation_time_utc or (
        packet_model.downlink_time_utc + timedelta(seconds=120)
    )
    if activation_time.tzinfo is None or activation_time.utcoffset() != timedelta(0):
        raise ValueError("activation_time_utc must be a UTC-aware datetime")
    geometry = _coerce_incident_geometry(incident_geometry_override) or _infer_incident_geometry(
        packet_model,
        resolution.bundle,
    )
    severity_score = _severity_score(packet_model)
    incident_context = TacticalIncidentContext(
        incident_id=packet_model.incident_id,
        target_cell_id=packet_model.target_cell_id,
        geometry=geometry,
        severity_class=_severity_class(severity_score),
        severity_score=severity_score,
        urgency_score=packet_model.urgency_score,
        confidence=packet_model.confidence,
        downstream_value_estimate=packet_model.downstream_value_estimate,
        recommended_action=packet_model.recommended_action,
        activation_delay_seconds=int(
            (activation_time - packet_model.downlink_time_utc).total_seconds()
        ),
        summary=packet_model.summary,
    )
    bridge_provenance = TacticalBridgeProvenance(
        source_packet_id=packet_model.packet_id,
        source_observation_time_utc=packet_model.observation_time_utc,
        source_downlink_time_utc=packet_model.downlink_time_utc,
        bridge_version=engine_config.bridge_version,
        notes=[
            f"region_resolution:{resolution.selection.resolution_strategy}",
            f"region_bundle:{resolution.bundle.region_bundle_id}",
        ],
    )
    activation_document = {
        "schema_version": packet_model.schema_version,
        "activation_id": stable_id("act", packet_model.incident_id, "tactical"),
        "incident_packet_id": packet_model.packet_id,
        "region_bundle_id": resolution.bundle.region_bundle_id,
        "activation_time_utc": activation_time,
        "activation_reason": activation_reason or _default_activation_reason(packet_model),
        "requested_capabilities": list(
            requested_capabilities or _default_requested_capabilities(packet_model)
        ),
        "region_selection": resolution.selection.model_dump(mode="json", exclude_none=True),
        "incident_context": incident_context.model_dump(mode="json", exclude_none=True),
        "bridge_provenance": bridge_provenance.model_dump(mode="json", exclude_none=True),
        "provenance_notes": list(provenance_notes or ()),
        "incident_packet": packet_model.model_dump(mode="json", exclude_none=True),
    }
    activation_document["activation_fingerprint"] = tactical_activation_fingerprint(
        activation_document
    )
    return TacticalActivation.model_validate(activation_document)


def compile_activation_to_manifest(
    activation: TacticalActivation | Mapping[str, Any],
    *,
    scenario_family: str,
    simulation_seed: int,
    region_bundle: RegionBundle | Mapping[str, Any] | Path | str | None = None,
    config: TacticalScenarioEngineConfig | None = None,
    family_parameters: Mapping[str, Any] | None = None,
) -> TacticalScenarioManifest:
    engine_config = config or TacticalScenarioEngineConfig()
    activation_model = _coerce_activation(activation)
    spec = FAMILY_SPECS[scenario_family]
    bundle_model = _load_region_bundle(
        region_bundle or _bundle_source_for_activation(activation_model, engine_config)
    )
    scenario_start = activation_model.activation_time_utc + timedelta(minutes=1)
    time_window = TimeWindow(
        start_time_utc=scenario_start,
        end_time_utc=scenario_start + timedelta(minutes=spec.planning_horizon_minutes),
    )
    facilities = _build_scenario_facilities(
        activation_model,
        bundle_model,
        spec=spec,
        simulation_seed=simulation_seed,
    )
    dispatch_units = _build_dispatch_units(
        spec=spec,
        facilities=facilities,
        scenario_family=scenario_family,
        simulation_seed=simulation_seed,
    )
    depot_assignments = _build_depot_assignments(
        spec=spec,
        facilities=facilities,
        dispatch_units=dispatch_units,
    )
    scout_assets = _build_scout_assets(
        spec=spec,
        facilities=facilities,
        scenario_family=scenario_family,
        simulation_seed=simulation_seed,
    )
    overlay_events = _build_overlay_events(
        activation_model,
        bundle_model,
        facilities=facilities,
        scenario_family=scenario_family,
        spec=spec,
        time_window=time_window,
        simulation_seed=simulation_seed,
    )
    manifest_document = {
        "schema_version": activation_model.schema_version,
        "tactical_manifest_id": stable_id(
            "tsm",
            activation_model.incident_context.incident_id,
            scenario_family,
            f"seed-{simulation_seed}",
        ),
        "activation_id": activation_model.activation_id,
        "activation_fingerprint": activation_model.activation_fingerprint,
        "incident_packet_id": activation_model.incident_packet_id,
        "region_bundle_id": activation_model.region_bundle_id,
        "scenario_family": scenario_family,
        "simulation_seed": simulation_seed,
        "decision_interval_seconds": spec.decision_interval_seconds,
        "time_window": time_window.model_dump(mode="json"),
        "incident_packet": activation_model.incident_packet.model_dump(
            mode="json", exclude_none=True
        ),
        "region_selection": activation_model.region_selection.model_dump(
            mode="json", exclude_none=True
        ),
        "incident_context": activation_model.incident_context.model_dump(
            mode="json", exclude_none=True
        ),
        "bridge_provenance": activation_model.bridge_provenance.model_dump(
            mode="json", exclude_none=True
        ),
        "dispatch_units": [
            unit.model_dump(mode="json", exclude_none=True) for unit in dispatch_units
        ],
        "facilities": [
            facility.model_dump(mode="json", exclude_none=True) for facility in facilities
        ],
        "depot_assignments": [
            assignment.model_dump(mode="json", exclude_none=True)
            for assignment in depot_assignments
        ],
        "scout_assets": [
            scout.model_dump(mode="json", exclude_none=True) for scout in scout_assets
        ],
        "overlay_events": [
            overlay.model_dump(mode="json", exclude_none=True) for overlay in overlay_events
        ],
        "operational_objectives": list(spec.operational_objectives),
        "config": TacticalScenarioConfig.model_validate(
            {
                "planning_horizon_minutes": spec.planning_horizon_minutes,
                "max_active_routes": spec.max_active_routes,
                "reroute_on_blockage": spec.reroute_on_blockage,
                "notes": spec.description,
                "benchmark_id": engine_config.benchmark_id,
                "family_display_name": spec.display_name,
                "family_parameters": dict(family_parameters or {}),
                "overlay_focus": spec.overlay_focus,
                "region_bundle_fingerprint": (
                    activation_model.region_selection.region_bundle_fingerprint
                ),
            }
        ).model_dump(mode="json", exclude_none=True),
    }
    return TacticalScenarioManifest.model_validate(manifest_document)


def compile_manifest_to_bundle(
    manifest: TacticalScenarioManifest | Mapping[str, Any],
    *,
    region_bundle: RegionBundle | Mapping[str, Any] | Path | str | None = None,
    config: TacticalScenarioEngineConfig | None = None,
) -> TacticalScenarioBundle:
    engine_config = config or TacticalScenarioEngineConfig()
    manifest_model = _coerce_manifest(manifest)
    bundle_model = _load_region_bundle(
        region_bundle or _bundle_source_for_manifest(manifest_model, engine_config)
    )
    augmented_bundle = _augment_region_bundle(bundle_model, manifest_model.facilities)
    service = RoutingEngineService.in_memory()
    service.ingest_region_bundle(augmented_bundle)
    overlay_ids = _register_manifest_overlays(
        service,
        region_bundle_id=manifest_model.region_bundle_id,
        overlay_events=manifest_model.overlay_events,
    )
    route_plans, updated_units = _compile_route_plans(
        service,
        manifest_model=manifest_model,
        overlay_ids=overlay_ids,
    )
    manifest_document = manifest_model.model_dump(mode="json", exclude_none=True)
    bundle_document = {
        **manifest_document,
        "tactical_bundle_id": tactical_bundle_id_from_manifest_id(
            manifest_model.tactical_manifest_id
        ),
        "dispatch_units": [
            unit.model_dump(mode="json", exclude_none=True) for unit in updated_units
        ],
        "route_plans": [
            route_plan.model_dump(mode="json", exclude_none=True) for route_plan in route_plans
        ],
        "compilation": BundleCompilation(
            source_manifest_id=manifest_model.tactical_manifest_id,
            source_manifest_schema_version=manifest_model.schema_version,
            source_manifest_sha256=sha256_fingerprint(canonical_json_dumps(manifest_document)),
            compiled_at_utc=engine_config.compiled_at_utc,
            compiler_version=engine_config.compiler_version,
        ).model_dump(mode="json"),
    }
    bundle_document["bundle_fingerprint"] = tactical_bundle_fingerprint(bundle_document)
    return TacticalScenarioBundle.model_validate(bundle_document)


def compile_incident_packet_to_bundle(
    packet: IncidentPacket | Mapping[str, Any],
    *,
    scenario_family: str,
    simulation_seed: int,
    region_bundle: RegionBundle | Mapping[str, Any] | Path | str | None = None,
    catalog_sources: Sequence[RegionBundle | Mapping[str, Any] | Path | str] | None = None,
    config: TacticalScenarioEngineConfig | None = None,
    activation_time_utc: datetime | None = None,
    requested_capabilities: Sequence[str] | None = None,
    activation_reason: str | None = None,
    incident_geometry_override: TacticalIncidentGeometry | Mapping[str, Any] | None = None,
    family_parameters: Mapping[str, Any] | None = None,
) -> TacticalScenarioBundle:
    activation = compile_incident_packet_to_activation(
        packet,
        region_bundle=region_bundle,
        catalog_sources=catalog_sources,
        config=config,
        activation_time_utc=activation_time_utc,
        requested_capabilities=requested_capabilities,
        activation_reason=activation_reason,
        incident_geometry_override=incident_geometry_override,
    )
    manifest = compile_activation_to_manifest(
        activation,
        scenario_family=scenario_family,
        simulation_seed=simulation_seed,
        region_bundle=region_bundle,
        config=config,
        family_parameters=family_parameters,
    )
    return compile_manifest_to_bundle(
        manifest,
        region_bundle=region_bundle,
        config=config,
    )


def compile_recipe_to_bundle(
    recipe: TacticalScenarioRecipe,
    *,
    config: TacticalScenarioEngineConfig | None = None,
) -> TacticalScenarioBundle:
    engine_config = config or TacticalScenarioEngineConfig()
    bundle_model = _load_region_bundle(recipe.region_bundle_path)
    incident_geometry = _incident_geometry_from_recipe(recipe, bundle_model)
    activation = compile_incident_packet_to_activation(
        recipe.packet,
        region_bundle=bundle_model,
        config=engine_config,
        activation_time_utc=recipe.packet.downlink_time_utc
        + timedelta(seconds=recipe.activation_delay_seconds),
        requested_capabilities=FAMILY_SPECS[recipe.family_id].requested_capabilities,
        activation_reason=(
            f"{FAMILY_SPECS[recipe.family_id].display_name} activation synthesized from "
            "deterministic Phase 3 packet replay."
        ),
        incident_geometry_override=incident_geometry,
        provenance_notes=[f"recipe:{recipe.recipe_id}"],
    )
    manifest = compile_activation_to_manifest(
        activation,
        scenario_family=recipe.family_id,
        simulation_seed=recipe.seed,
        region_bundle=bundle_model,
        config=engine_config,
        family_parameters={
            "recipe_id": recipe.recipe_id,
            **recipe.family_parameters,
        },
    )
    return compile_manifest_to_bundle(
        manifest,
        region_bundle=bundle_model,
        config=engine_config,
    )


def build_scenario_pack(
    *,
    config: TacticalScenarioEngineConfig | None = None,
    output_dir: Path | None = None,
    recipes: Sequence[TacticalScenarioRecipe] | None = None,
) -> list[TacticalBundleRecord]:
    engine_config = config or TacticalScenarioEngineConfig()
    destination = output_dir or engine_config.scenario_dir
    destination.mkdir(parents=True, exist_ok=True)
    recipe_set = tuple(
        recipes
        or builtin_phase3_recipes(
            engine_config.benchmark_id,
            region_bundle_path=engine_config.default_region_bundle_path,
        )
    )
    records: list[TacticalBundleRecord] = []
    for recipe in recipe_set:
        bundle = compile_recipe_to_bundle(recipe, config=engine_config)
        output_path = destination / f"{bundle.tactical_bundle_id.replace(':', '--')}.json"
        output_path.write_text(
            canonical_json_dumps(bundle.model_dump(mode="json", exclude_none=True)) + "\n",
            encoding="utf-8",
        )
        records.append(
            TacticalBundleRecord(
                tactical_bundle_id=bundle.tactical_bundle_id,
                scenario_family=bundle.scenario_family,
                bundle_fingerprint=bundle.bundle_fingerprint,
                path=output_path,
            )
        )
    return sorted(records, key=lambda item: item.tactical_bundle_id)


def validate_scenario_pack(
    *,
    config: TacticalScenarioEngineConfig | None = None,
    input_dir: Path | None = None,
    recipes: Sequence[TacticalScenarioRecipe] | None = None,
) -> list[TacticalBundleRecord]:
    engine_config = config or TacticalScenarioEngineConfig()
    source_dir = input_dir or engine_config.scenario_dir
    recipe_set = tuple(
        recipes
        or builtin_phase3_recipes(
            engine_config.benchmark_id,
            region_bundle_path=engine_config.default_region_bundle_path,
        )
    )
    expected_bundles = {
        bundle.tactical_bundle_id: bundle
        for bundle in (
            compile_recipe_to_bundle(recipe, config=engine_config) for recipe in recipe_set
        )
    }
    records: list[TacticalBundleRecord] = []
    for path in sorted(source_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        bundle = TacticalScenarioBundle.model_validate(document)
        if bundle.bundle_fingerprint != tactical_bundle_fingerprint(
            bundle.model_dump(mode="json", exclude_none=True)
        ):
            raise ValueError(f"bundle fingerprint mismatch: {bundle.tactical_bundle_id}")
        expected = expected_bundles.get(bundle.tactical_bundle_id)
        if expected is None:
            raise ValueError(
                f"unexpected tactical scenario bundle in pack: {bundle.tactical_bundle_id}"
            )
        actual_canonical = canonical_json_dumps(
            bundle.model_dump(mode="json", exclude_none=True)
        )
        expected_canonical = canonical_json_dumps(
            expected.model_dump(mode="json", exclude_none=True)
        )
        if actual_canonical != expected_canonical:
            raise ValueError(
                f"compiled tactical bundle drift for {bundle.tactical_bundle_id}: "
                "pack contents do not match deterministic rebuild"
            )
        records.append(
            TacticalBundleRecord(
                tactical_bundle_id=bundle.tactical_bundle_id,
                scenario_family=bundle.scenario_family,
                bundle_fingerprint=bundle.bundle_fingerprint,
                path=path,
            )
        )
    if len(records) != len(expected_bundles):
        raise ValueError(
            f"expected {len(expected_bundles)} tactical bundles but found {len(records)}"
        )
    return records


def inspect_bundle(
    bundle: TacticalScenarioBundle | Mapping[str, Any] | Path | str,
) -> dict[str, Any]:
    bundle_model = (
        bundle
        if isinstance(bundle, TacticalScenarioBundle)
        else TacticalScenarioBundle.model_validate(
            load_json(Path(bundle)) if isinstance(bundle, (Path, str)) else bundle
        )
    )
    unit_counts = Counter(unit.unit_type for unit in bundle_model.dispatch_units)
    route_status_counts = Counter(route.status for route in bundle_model.route_plans)
    overlay_counts = Counter(overlay.overlay_kind for overlay in bundle_model.overlay_events)
    config_extras = bundle_model.config.model_extra or {}
    return {
        "tactical_bundle_id": bundle_model.tactical_bundle_id,
        "bundle_fingerprint": bundle_model.bundle_fingerprint,
        "scenario_family": bundle_model.scenario_family,
        "family_display_name": str(config_extras.get("family_display_name", "")),
        "region_bundle_id": bundle_model.region_bundle_id,
        "region_bundle_fingerprint": bundle_model.region_selection.region_bundle_fingerprint,
        "incident_id": bundle_model.incident_context.incident_id,
        "incident_geometry": {
            "centroid": bundle_model.incident_context.geometry.centroid.model_dump(mode="json"),
            "estimated_area_ha": bundle_model.incident_context.geometry.estimated_area_ha,
        },
        "severity_class": bundle_model.incident_context.severity_class,
        "severity_score": bundle_model.incident_context.severity_score,
        "urgency_score": bundle_model.incident_context.urgency_score,
        "unit_type_counts": dict(sorted(unit_counts.items())),
        "depot_assignments": [
            assignment.model_dump(mode="json", exclude_none=True)
            for assignment in bundle_model.depot_assignments
        ],
        "scout_assets": [
            scout.model_dump(mode="json", exclude_none=True) for scout in bundle_model.scout_assets
        ],
        "overlay_counts": dict(sorted(overlay_counts.items())),
        "route_status_counts": dict(sorted(route_status_counts.items())),
        "route_plans": [
            {
                "route_plan_id": route.route_plan_id,
                "unit_id": route.unit_id,
                "origin_facility_id": route.origin_facility_id,
                "destination_facility_id": route.destination_facility_id,
                "travel_mode": route.travel_mode,
                "status": route.status,
                "distance_km": route.distance_km,
                "estimated_duration_seconds": route.estimated_duration_seconds,
                "risk_score": route.risk_score,
            }
            for route in bundle_model.route_plans
        ],
        "operational_objectives": list(bundle_model.operational_objectives),
        "compilation": bundle_model.compilation.model_dump(mode="json"),
    }


def _coerce_packet(packet: IncidentPacket | Mapping[str, Any]) -> IncidentPacket:
    return packet if isinstance(packet, IncidentPacket) else IncidentPacket.model_validate(packet)


def _coerce_activation(
    activation: TacticalActivation | Mapping[str, Any],
) -> TacticalActivation:
    return (
        activation
        if isinstance(activation, TacticalActivation)
        else TacticalActivation.model_validate(activation)
    )


def _coerce_manifest(
    manifest: TacticalScenarioManifest | Mapping[str, Any],
) -> TacticalScenarioManifest:
    return (
        manifest
        if isinstance(manifest, TacticalScenarioManifest)
        else TacticalScenarioManifest.model_validate(manifest)
    )


def _coerce_incident_geometry(
    value: TacticalIncidentGeometry | Mapping[str, Any] | None,
) -> TacticalIncidentGeometry | None:
    if value is None:
        return None
    return (
        value
        if isinstance(value, TacticalIncidentGeometry)
        else TacticalIncidentGeometry.model_validate(value)
    )


@dataclass(frozen=True, slots=True)
class _CatalogBundleRecord:
    bundle: RegionBundle
    source_path: Path | None


def _load_region_catalog(
    sources: Sequence[RegionBundle | Mapping[str, Any] | Path | str],
) -> list[_CatalogBundleRecord]:
    records: list[_CatalogBundleRecord] = []
    for source in sources:
        if isinstance(source, RegionBundle):
            records.append(_CatalogBundleRecord(bundle=source, source_path=None))
        elif isinstance(source, Mapping):
            records.append(
                _CatalogBundleRecord(
                    bundle=RegionBundle.model_validate(source),
                    source_path=None,
                )
            )
        else:
            path = Path(source)
            if path.is_dir():
                for bundle_path in sorted(path.glob("*.json")):
                    records.append(
                        _CatalogBundleRecord(
                            bundle=RegionBundle.model_validate(load_json(bundle_path)),
                            source_path=bundle_path,
                        )
                    )
            elif path.exists():
                records.append(
                    _CatalogBundleRecord(
                        bundle=RegionBundle.model_validate(load_json(path)),
                        source_path=path,
                    )
                )
    return records


def _load_region_bundle(
    source: RegionBundle | Mapping[str, Any] | Path | str,
) -> RegionBundle:
    if isinstance(source, RegionBundle):
        return source
    if isinstance(source, Mapping):
        return RegionBundle.model_validate(source)
    return RegionBundle.model_validate(load_json(Path(source)))


def _matched_h3_cell(packet: IncidentPacket, bundle: RegionBundle) -> str | None:
    hints = _target_cell_hints(packet.target_cell_id)
    for hint in hints:
        if hint in bundle.h3_cover.cell_ids:
            return hint
    return None


def _target_cell_hints(target_cell_id: str) -> tuple[str, ...]:
    pieces = [piece for piece in target_cell_id.split(":") if piece]
    hints = [target_cell_id]
    hints.extend(pieces)
    if pieces:
        hints.append(pieces[-1])
    seen: set[str] = set()
    ordered: list[str] = []
    for hint in hints:
        if hint not in seen:
            ordered.append(hint)
            seen.add(hint)
    return tuple(ordered)


def _infer_incident_geometry(
    packet: IncidentPacket,
    bundle: RegionBundle,
) -> TacticalIncidentGeometry:
    sorted_edges = sorted(bundle.road_edges, key=lambda item: item.edge_id)
    if not sorted_edges:
        center = _bounds_center(bundle)
        return TacticalIncidentGeometry(centroid=center, estimated_area_ha=18.0)
    edge_index = (
        int(sha256_fingerprint(f"{packet.packet_id}:{packet.target_cell_id}")[:16], 16)
        % len(sorted_edges)
    )
    return _incident_geometry_for_edge(
        sorted_edges[edge_index],
        area_ha=round(
            18.0
            + packet.urgency_score * 48.0
            + packet.confidence * 16.0
            + (packet.downstream_value_estimate or 0.0) * 24.0,
            3,
        ),
    )


def _incident_geometry_from_recipe(
    recipe: TacticalScenarioRecipe,
    bundle: RegionBundle,
) -> TacticalIncidentGeometry:
    sorted_edges = sorted(bundle.road_edges, key=lambda item: item.edge_id)
    if not sorted_edges:
        return TacticalIncidentGeometry(
            centroid=_bounds_center(bundle),
            estimated_area_ha=FAMILY_SPECS[recipe.family_id].base_area_ha,
        )
    edge = sorted_edges[recipe.incident_anchor_index % len(sorted_edges)]
    variant_index = int(recipe.family_parameters.get("variant_index", 0))
    area_ha = FAMILY_SPECS[recipe.family_id].base_area_ha + variant_index * 6.5
    return _incident_geometry_for_edge(edge, area_ha=round(area_ha, 3))


def _incident_geometry_for_edge(edge: Any, *, area_ha: float) -> TacticalIncidentGeometry:
    start = edge.geometry[0]
    end = edge.geometry[-1]
    centroid = Wgs84Point(
        lat=round((start.lat + end.lat) / 2.0, 6),
        lon=round((start.lon + end.lon) / 2.0, 6),
    )
    lat_radius = max(0.00045, min(0.0022, sqrt(max(area_ha, 1.0)) / 5000.0))
    lon_radius = lat_radius / max(cos(radians(centroid.lat)), 0.35)
    ring = [
        Wgs84Point(lat=round(centroid.lat + lat_radius, 6), lon=round(centroid.lon, 6)),
        Wgs84Point(lat=round(centroid.lat, 6), lon=round(centroid.lon + lon_radius, 6)),
        Wgs84Point(lat=round(centroid.lat - lat_radius, 6), lon=round(centroid.lon, 6)),
        Wgs84Point(lat=round(centroid.lat, 6), lon=round(centroid.lon - lon_radius, 6)),
        Wgs84Point(lat=round(centroid.lat + lat_radius, 6), lon=round(centroid.lon, 6)),
    ]
    return TacticalIncidentGeometry(
        centroid=centroid,
        estimated_area_ha=area_ha,
        perimeter_ring=ring,
    )


def _default_requested_capabilities(packet: IncidentPacket) -> tuple[str, ...]:
    if packet.recommended_action == "dispatch_ground":
        return ("initial_attack", "road_dispatch", "command")
    if packet.recommended_action == "dispatch_recon":
        return ("air_recon", "manual_recon")
    return ("monitor",)


def _default_activation_reason(packet: IncidentPacket) -> str:
    return (
        f"IncidentPacket {packet.packet_id} escalated into tactical planning after "
        f"{packet.recommended_action} guidance."
    )


def _severity_score(packet: IncidentPacket) -> float:
    value = packet.downstream_value_estimate or packet.urgency_score
    return round(
        min(1.0, 0.55 * packet.urgency_score + 0.25 * packet.confidence + 0.20 * value),
        6,
    )


def _severity_class(severity_score: float) -> TacticalSeverityClass:
    if severity_score >= 0.86:
        return "extreme"
    if severity_score >= 0.76:
        return "very_high"
    if severity_score >= 0.62:
        return "high"
    return "moderate"


def _bundle_source_for_activation(
    activation: TacticalActivation,
    config: TacticalScenarioEngineConfig,
) -> Path:
    for candidate in config.region_bundle_catalog:
        path = Path(candidate)
        if path.exists():
            bundle = RegionBundle.model_validate(load_json(path))
            if bundle.region_bundle_id == activation.region_bundle_id:
                return path
    if config.default_region_bundle_path.exists():
        return config.default_region_bundle_path
    raise LookupError(f"unable to locate region bundle {activation.region_bundle_id!r}")


def _bundle_source_for_manifest(
    manifest: TacticalScenarioManifest,
    config: TacticalScenarioEngineConfig,
) -> Path:
    for candidate in config.region_bundle_catalog:
        path = Path(candidate)
        if path.exists():
            bundle = RegionBundle.model_validate(load_json(path))
            if bundle.region_bundle_id == manifest.region_bundle_id:
                return path
    if config.default_region_bundle_path.exists():
        return config.default_region_bundle_path
    raise LookupError(f"unable to locate region bundle {manifest.region_bundle_id!r}")


def _bounds_center(bundle: RegionBundle) -> Wgs84Point:
    return Wgs84Point(
        lat=round((bundle.bounds.min_lat + bundle.bounds.max_lat) / 2.0, 6),
        lon=round((bundle.bounds.min_lon + bundle.bounds.max_lon) / 2.0, 6),
    )


def _build_scenario_facilities(
    activation: TacticalActivation,
    bundle: RegionBundle,
    *,
    spec: TacticalFamilySpec,
    simulation_seed: int,
) -> list[Facility]:
    existing = {facility.facility_id: facility for facility in bundle.facilities}
    type_index: dict[str, Facility] = {}
    for facility in sorted(bundle.facilities, key=lambda item: item.facility_id):
        type_index.setdefault(facility.facility_type, facility)
    incident = activation.incident_context.geometry.centroid
    drop_point = existing.get(
        stable_id("fac", activation.incident_context.incident_id, "drop-point")
    ) or Facility(
        facility_id=stable_id("fac", activation.incident_context.incident_id, "drop-point"),
        facility_name="Incident Drop Point",
        facility_type="drop_point",
        location=Wgs84GroundPoint(lat=incident.lat, lon=incident.lon, alt_m=670.0),
        availability="nominal",
        capacity_units=max(2, spec.max_active_routes),
        supported_unit_types=["engine", "crew", "dozer", "helicopter", "air_tanker", "command"],
    )
    command_post = existing.get(
        stable_id("fac", activation.incident_context.incident_id, "command-post")
    ) or Facility(
        facility_id=stable_id("fac", activation.incident_context.incident_id, "command-post"),
        facility_name="Incident Command Post",
        facility_type="command_post",
        location=_ground_point_with_alt(
            _point_near(bundle, anchor="west", seed_key=f"command:{simulation_seed}"),
            alt_m=640.0,
        ),
        availability="nominal",
        capacity_units=4,
        supported_unit_types=["command", "engine", "crew", "helicopter"],
    )
    helibase = type_index.get("helibase") or Facility(
        facility_id=stable_id("fac", activation.incident_context.incident_id, "helibase"),
        facility_name="Tactical Helibase",
        facility_type="helibase",
        location=_ground_point_with_alt(
            _point_near(bundle, anchor="north_east", seed_key=f"helibase:{simulation_seed}"),
            alt_m=710.0,
        ),
        availability="nominal",
        capacity_units=3,
        supported_unit_types=["helicopter", "air_tanker", "command"],
    )
    station = type_index.get("station") or Facility(
        facility_id=stable_id("fac", activation.incident_context.incident_id, "station"),
        facility_name="Forward Station",
        facility_type="station",
        location=_ground_point_with_alt(
            _point_near(bundle, anchor="west", seed_key=f"station:{simulation_seed}"),
            alt_m=615.0,
        ),
        availability="nominal",
        capacity_units=6,
        supported_unit_types=["engine", "crew", "command"],
    )
    staging = type_index.get("staging_area") or Facility(
        facility_id=stable_id("fac", activation.incident_context.incident_id, "staging"),
        facility_name="Mutual Aid Staging",
        facility_type="staging_area",
        location=_ground_point_with_alt(
            _point_near(bundle, anchor="east", seed_key=f"staging:{simulation_seed}"),
            alt_m=630.0,
        ),
        availability="nominal",
        capacity_units=8,
        supported_unit_types=["engine", "crew", "dozer", "command"],
    )
    facilities = {
        facility.facility_id: facility
        for facility in [*bundle.facilities, station, staging, helibase, command_post, drop_point]
    }
    return sorted(facilities.values(), key=lambda item: item.facility_id)


def _point_near(bundle: RegionBundle, *, anchor: str, seed_key: str) -> Wgs84Point:
    points = [node.location for node in bundle.road_nodes]
    if not points:
        return _bounds_center(bundle)
    rng = seeded_rng(seed_key)
    if anchor == "west":
        base = min(points, key=lambda point: (point.lon, point.lat))
    elif anchor == "east":
        base = max(points, key=lambda point: (point.lon, point.lat))
    elif anchor == "north_east":
        base = max(points, key=lambda point: (point.lat + point.lon / 1000.0, point.lon))
    else:
        base = points[rng.randrange(len(points))]
    lat_jitter = rng.uniform(-0.00035, 0.00035)
    lon_jitter = rng.uniform(-0.00035, 0.00035)
    return Wgs84Point(lat=round(base.lat + lat_jitter, 6), lon=round(base.lon + lon_jitter, 6))


def _ground_point_with_alt(point: Wgs84Point, *, alt_m: float) -> Wgs84GroundPoint:
    return Wgs84GroundPoint(lat=point.lat, lon=point.lon, alt_m=alt_m)


def _build_dispatch_units(
    *,
    spec: TacticalFamilySpec,
    facilities: Sequence[Facility],
    scenario_family: str,
    simulation_seed: int,
) -> list[DispatchUnit]:
    facility_by_label = _facility_label_index(facilities)
    units: list[DispatchUnit] = []
    for template in spec.unit_templates:
        home = facility_by_label[template.home_label]
        unit_id = stable_id("unit", scenario_family, template.role, f"seed-{simulation_seed}")
        callsign = _callsign(spec, template)
        units.append(
            DispatchUnit(
                unit_id=unit_id,
                callsign=callsign,
                unit_type=template.unit_type,  # type: ignore[arg-type]
                status="available",
                home_facility_id=home.facility_id,
                current_facility_id=home.facility_id,
                travel_mode=template.travel_mode,  # type: ignore[arg-type]
                personnel_count=template.personnel_count,
                equipment_capacity=template.equipment_capacity,
                location=home.location,
            )
        )
    return units


def _callsign(spec: TacticalFamilySpec, template: UnitTemplate) -> str:
    family_code = "".join(word[0].upper() for word in spec.display_name.split())
    role_name = template.role.replace("_", " ").title()
    return f"{family_code} {role_name}"


def _facility_label_index(facilities: Sequence[Facility]) -> dict[str, Facility]:
    typed: dict[str, Facility] = {}
    for facility in facilities:
        if facility.facility_type == "station":
            typed.setdefault("station", facility)
        elif facility.facility_type == "staging_area":
            typed.setdefault("staging", facility)
        elif facility.facility_type == "helibase":
            typed.setdefault("helibase", facility)
        elif facility.facility_type == "command_post":
            typed.setdefault("command", facility)
        elif facility.facility_type == "drop_point":
            typed.setdefault("drop_point", facility)
    return typed


def _build_depot_assignments(
    *,
    spec: TacticalFamilySpec,
    facilities: Sequence[Facility],
    dispatch_units: Sequence[DispatchUnit],
) -> list[TacticalDepotAssignment]:
    unit_by_role = {
        template.role: unit
        for template, unit in zip(spec.unit_templates, dispatch_units, strict=True)
    }
    facility_by_label = _facility_label_index(facilities)
    assignments: list[TacticalDepotAssignment] = []
    for queue_order, template in enumerate(spec.unit_templates):
        assignments.append(
            TacticalDepotAssignment(
                unit_id=unit_by_role[template.role].unit_id,
                depot_facility_id=facility_by_label[template.home_label].facility_id,
                assignment_role=template.role,
                queue_order=queue_order,
            )
        )
    return assignments


def _build_scout_assets(
    *,
    spec: TacticalFamilySpec,
    facilities: Sequence[Facility],
    scenario_family: str,
    simulation_seed: int,
) -> list[ScoutAsset]:
    facility_by_label = _facility_label_index(facilities)
    scouts: list[ScoutAsset] = []
    for index, template in enumerate(spec.scout_templates, 1):
        home = facility_by_label[template.home_label]
        scouts.append(
            ScoutAsset(
                scout_asset_id=stable_id(
                    "scout",
                    scenario_family,
                    template.asset_type,
                    f"{simulation_seed}-{index}",
                ),
                asset_name=f"{spec.display_name} Scout {index:02d}",
                asset_type=template.asset_type,  # type: ignore[arg-type]
                status=template.status,  # type: ignore[arg-type]
                home_facility_id=home.facility_id,
                location=home.location,
                endurance_minutes=template.endurance_minutes,
                sensor_focus=list(template.sensor_focus),
            )
        )
    return scouts


def _build_overlay_events(
    activation: TacticalActivation,
    bundle: RegionBundle,
    *,
    facilities: Sequence[Facility],
    scenario_family: str,
    spec: TacticalFamilySpec,
    time_window: TimeWindow,
    simulation_seed: int,
) -> list[TacticalOverlayEvent]:
    augmented_bundle = _augment_region_bundle(bundle, facilities)
    service = RoutingEngineService.in_memory()
    service.ingest_region_bundle(augmented_bundle)
    facility_by_label = _facility_label_index(facilities)
    base_route = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=bundle.region_bundle_id,
            origin=RoutingEndpoint(facility_id=facility_by_label["station"].facility_id),
            destination=RoutingEndpoint(facility_id=facility_by_label["drop_point"].facility_id),
        )
    )
    primary_edges = list(base_route.edge_ids)
    secondary_edges = [
        edge.edge_id
        for edge in sorted(bundle.road_edges, key=lambda item: item.edge_id)
        if edge.edge_id not in primary_edges
    ]
    affected_assets = [
        asset.asset_id
        for asset in sorted(bundle.asset_features, key=lambda item: item.asset_id)
        if asset.priority_class in {"high", "critical"}
    ]
    incident_ring = activation.incident_context.geometry.perimeter_ring or [
        activation.incident_context.geometry.centroid,
        activation.incident_context.geometry.centroid,
        activation.incident_context.geometry.centroid,
        activation.incident_context.geometry.centroid,
    ]
    overlay_events: list[TacticalOverlayEvent] = []

    def closure_event(name: str, edge_ids: Sequence[str], severity: float) -> TacticalOverlayEvent:
        return TacticalOverlayEvent(
            overlay_event_id=stable_id("ovl", scenario_family, name, simulation_seed),
            overlay_kind="closure",
            title=name.replace("-", " ").title(),
            summary=f"{spec.display_name} closure set blocks the primary ingress corridor.",
            severity_score=severity,
            window=time_window,
            edge_effects=[
                TacticalOverlayEdgeEffect(edge_id=edge_id, closed=True, reason="hard_closure")
                for edge_id in edge_ids
            ],
            zone_ring=list(incident_ring),
            affected_asset_ids=affected_assets[:1],
        )

    def risk_event(
        name: str,
        edge_ids: Sequence[str],
        multiplier: float,
        severity: float,
    ) -> TacticalOverlayEvent:
        return TacticalOverlayEvent(
            overlay_event_id=stable_id("ovl", scenario_family, name, simulation_seed),
            overlay_kind="risk_zone",
            title=name.replace("-", " ").title(),
            summary=f"{spec.display_name} risk zone inflates corridor travel cost.",
            severity_score=severity,
            window=time_window,
            edge_effects=[
                TacticalOverlayEdgeEffect(
                    edge_id=edge_id,
                    cost_multiplier=multiplier,
                    reason="risk_inflation",
                )
                for edge_id in edge_ids
            ],
            zone_ring=list(incident_ring),
            affected_asset_ids=affected_assets,
        )

    def penalty_event(
        name: str,
        edge_ids: Sequence[str],
        *,
        speed_cap_kph: float,
        delay_seconds: float,
        severity: float,
    ) -> TacticalOverlayEvent:
        return TacticalOverlayEvent(
            overlay_event_id=stable_id("ovl", scenario_family, name, simulation_seed),
            overlay_kind="temporary_penalty",
            title=name.replace("-", " ").title(),
            summary=(
                f"{spec.display_name} temporary penalty degrades travel speed "
                "and staging flow."
            ),
            severity_score=severity,
            window=time_window,
            edge_effects=[
                TacticalOverlayEdgeEffect(
                    edge_id=edge_id,
                    speed_cap_kph=speed_cap_kph,
                    delay_seconds=delay_seconds,
                    reason="temporary_penalty",
                )
                for edge_id in edge_ids
            ],
            zone_ring=list(incident_ring),
            affected_asset_ids=affected_assets[:2],
        )

    if scenario_family == "foothill_access":
        target_edges = primary_edges[-2:] or primary_edges or secondary_edges[:1]
        overlay_events.append(
            penalty_event(
                "foothill-access-grade",
                target_edges,
                speed_cap_kph=14.0,
                delay_seconds=35.0,
                severity=0.64,
            )
        )
    elif scenario_family == "urban_interface":
        target_edges = primary_edges[:2] or secondary_edges[:2]
        overlay_events.append(risk_event("urban-interface-ember-band", target_edges, 1.35, 0.72))
        if secondary_edges:
            overlay_events.append(
                penalty_event(
                    "urban-interface-intersection-delay",
                    secondary_edges[:1],
                    speed_cap_kph=18.0,
                    delay_seconds=40.0,
                    severity=0.48,
                )
            )
    elif scenario_family == "closure_cascade":
        target_edges = primary_edges[1:3] or primary_edges[:1] or secondary_edges[:1]
        overlay_events.append(closure_event("closure-cascade-primary", target_edges, 0.81))
        fallback_edges = secondary_edges[:2] or primary_edges[:1]
        overlay_events.append(risk_event("closure-cascade-fallback", fallback_edges, 1.45, 0.66))
    elif scenario_family == "depot_saturation":
        target_edges = primary_edges[:1] or secondary_edges[:1]
        overlay_events.append(
            penalty_event(
                "depot-saturation-launch-queue",
                target_edges,
                speed_cap_kph=12.0,
                delay_seconds=65.0,
                severity=0.7,
            )
        )
    elif scenario_family == "smoke_corridor":
        target_edges = primary_edges or secondary_edges[:2]
        overlay_events.append(risk_event("smoke-corridor-band", target_edges, 1.42, 0.76))
        if target_edges:
            overlay_events.append(
                penalty_event(
                    "smoke-corridor-visibility",
                    target_edges[-1:],
                    speed_cap_kph=16.0,
                    delay_seconds=28.0,
                    severity=0.57,
                )
            )
    elif scenario_family == "drone_scout_gap":
        target_edges = primary_edges[-1:] or secondary_edges[:1]
        overlay_events.append(risk_event("drone-scout-gap-blind-sector", target_edges, 1.28, 0.61))
    else:
        raise KeyError(f"unsupported tactical scenario family: {scenario_family}")
    return overlay_events


def _augment_region_bundle(bundle: RegionBundle, facilities: Sequence[Facility]) -> RegionBundle:
    combined = bundle.model_dump(mode="json", exclude_none=True)
    facility_map = {
        existing.facility_id: existing for existing in [*bundle.facilities, *facilities]
    }
    combined["facilities"] = [
        facility_map[facility_id].model_dump(mode="json", exclude_none=True)
        for facility_id in sorted(facility_map)
    ]
    combined["bundle_fingerprint"] = region_bundle_fingerprint(combined)
    return RegionBundle.model_validate(combined)


def _register_manifest_overlays(
    service: RoutingEngineService,
    *,
    region_bundle_id: str,
    overlay_events: Sequence[TacticalOverlayEvent],
) -> list[str]:
    overlay_ids: list[str] = []
    for overlay_event in overlay_events:
        overlay_spec = _routing_overlay_from_event(region_bundle_id, overlay_event)
        if overlay_spec is None:
            continue
        service.register_overlay(overlay_spec)
        overlay_ids.append(overlay_event.overlay_event_id)
    return overlay_ids


def _routing_overlay_from_event(
    region_bundle_id: str,
    overlay_event: TacticalOverlayEvent,
) -> ClosureOverlaySpec | RiskMultiplierOverlaySpec | TemporaryRestrictionOverlaySpec | None:
    window = _overlay_window(overlay_event.window)
    if not overlay_event.edge_effects:
        return None
    if overlay_event.overlay_kind == "closure":
        return ClosureOverlaySpec(
            overlay_id=overlay_event.overlay_event_id,
            overlay_name=overlay_event.title,
            region_bundle_id=region_bundle_id,
            notes=overlay_event.summary,
            window=window,
            metadata={"severity_score": overlay_event.severity_score},
            edges=[
                ClosureEdgeEffect(edge_id=edge_effect.edge_id, reason=edge_effect.reason)
                for edge_effect in overlay_event.edge_effects
                if edge_effect.closed
            ],
        )
    if overlay_event.overlay_kind == "risk_zone":
        return RiskMultiplierOverlaySpec(
            overlay_id=overlay_event.overlay_event_id,
            overlay_name=overlay_event.title,
            region_bundle_id=region_bundle_id,
            notes=overlay_event.summary,
            window=window,
            metadata={"severity_score": overlay_event.severity_score},
            edges=[
                RiskMultiplierEdgeEffect(
                    edge_id=edge_effect.edge_id,
                    cost_multiplier=edge_effect.cost_multiplier or 1.0,
                    reason=edge_effect.reason,
                )
                for edge_effect in overlay_event.edge_effects
                if edge_effect.cost_multiplier is not None
            ],
        )
    return TemporaryRestrictionOverlaySpec(
        overlay_id=overlay_event.overlay_event_id,
        overlay_name=overlay_event.title,
        region_bundle_id=region_bundle_id,
        notes=overlay_event.summary,
        window=window,
        metadata={"severity_score": overlay_event.severity_score},
        edges=[
            TemporaryRestrictionEdgeEffect(
                edge_id=edge_effect.edge_id,
                speed_cap_kph=edge_effect.speed_cap_kph,
                delay_seconds=edge_effect.delay_seconds,
                reason=edge_effect.reason,
            )
            for edge_effect in overlay_event.edge_effects
            if edge_effect.speed_cap_kph is not None or edge_effect.delay_seconds > 0
        ],
    )


def _compile_route_plans(
    service: RoutingEngineService,
    *,
    manifest_model: TacticalScenarioManifest,
    overlay_ids: Sequence[str],
) -> tuple[list[RoutePlan], list[DispatchUnit]]:
    facility_by_id = {facility.facility_id: facility for facility in manifest_model.facilities}
    assignment_by_unit = {
        assignment.unit_id: assignment for assignment in manifest_model.depot_assignments
    }
    sorted_units = sorted(
        manifest_model.dispatch_units,
        key=lambda unit: (
            assignment_by_unit[unit.unit_id].queue_order,
            unit.unit_id,
        ),
    )
    active_units = sorted_units[: manifest_model.config.max_active_routes]
    route_plans: list[RoutePlan] = []
    assigned_unit_ids: set[str] = set()
    for unit in active_units:
        origin = facility_by_id[assignment_by_unit[unit.unit_id].depot_facility_id]
        destination = _destination_facility(unit, manifest_model.facilities)
        if unit.travel_mode == "road":
            overlay_selection = OverlaySelection(
                overlay_ids=list(overlay_ids),
                effective_at_utc=manifest_model.time_window.start_time_utc,
            )
            overlay_request = ShortestPathRequest(
                region_bundle_id=manifest_model.region_bundle_id,
                origin=RoutingEndpoint(facility_id=origin.facility_id),
                destination=RoutingEndpoint(facility_id=destination.facility_id),
                overlay_selection=overlay_selection,
            )
            routed = service.shortest_path(overlay_request)
            if routed.path_found:
                route_plans.append(
                    _road_route_plan(
                        manifest_model=manifest_model,
                        unit=unit,
                        origin=origin,
                        destination=destination,
                        total_distance_m=routed.total_distance_m or 0.0,
                        total_cost_seconds=routed.total_cost_seconds or 0.0,
                        edge_ids=routed.edge_ids,
                        waypoints=routed.waypoints,
                        status="planned",
                    )
                )
                assigned_unit_ids.add(unit.unit_id)
                continue
            base = service.shortest_path(
                ShortestPathRequest(
                    region_bundle_id=manifest_model.region_bundle_id,
                    origin=RoutingEndpoint(facility_id=origin.facility_id),
                    destination=RoutingEndpoint(facility_id=destination.facility_id),
                )
            )
            if base.path_found:
                route_plans.append(
                    _road_route_plan(
                        manifest_model=manifest_model,
                        unit=unit,
                        origin=origin,
                        destination=destination,
                        total_distance_m=base.total_distance_m or 0.0,
                        total_cost_seconds=base.total_cost_seconds or 0.0,
                        edge_ids=base.edge_ids,
                        waypoints=base.waypoints,
                        status="aborted",
                    )
                )
            continue
        route_plans.append(
            _air_route_plan(
                manifest_model=manifest_model,
                unit=unit,
                origin=origin,
                destination=destination,
            )
        )
        assigned_unit_ids.add(unit.unit_id)
    updated_units = [
        DispatchUnit.model_validate(
            {
                **unit.model_dump(mode="json", exclude_none=True),
                "status": "assigned" if unit.unit_id in assigned_unit_ids else unit.status,
            }
        )
        for unit in manifest_model.dispatch_units
    ]
    return route_plans, updated_units


def _destination_facility(unit: DispatchUnit, facilities: Sequence[Facility]) -> Facility:
    facility_by_type: dict[str, Facility] = {}
    for facility in sorted(facilities, key=lambda item: item.facility_id):
        facility_by_type.setdefault(facility.facility_type, facility)
    if unit.unit_type == "command":
        return facility_by_type.get("command_post") or facility_by_type["staging_area"]
    if unit.unit_type in {"helicopter", "air_tanker"}:
        return facility_by_type.get("drop_point") or facility_by_type["helibase"]
    if unit.unit_type == "dozer":
        return facility_by_type.get("staging_area") or facility_by_type["drop_point"]
    return facility_by_type.get("drop_point") or facility_by_type["staging_area"]


def _road_route_plan(
    *,
    manifest_model: TacticalScenarioManifest,
    unit: DispatchUnit,
    origin: Facility,
    destination: Facility,
    total_distance_m: float,
    total_cost_seconds: float,
    edge_ids: Sequence[str],
    waypoints: Sequence[Wgs84Point],
    status: str,
) -> RoutePlan:
    if total_distance_m <= 0:
        total_distance_m = haversine_m(
            start_lat=origin.location.lat,
            start_lon=origin.location.lon,
            end_lat=destination.location.lat,
            end_lon=destination.location.lon,
        )
    if total_distance_m <= 0:
        total_distance_m = 1.0
    effective_waypoints = (
        list(waypoints)
        if len(waypoints) >= 2
        else [origin.location, destination.location]
    )
    risk_score = _route_risk_score(
        edge_ids=edge_ids,
        overlay_events=manifest_model.overlay_events,
        incident_context=manifest_model.incident_context,
        status=status,
    )
    return RoutePlan(
        route_plan_id=stable_id("route", unit.unit_id, manifest_model.incident_context.incident_id),
        unit_id=unit.unit_id,
        origin_facility_id=origin.facility_id,
        destination_facility_id=destination.facility_id,
        travel_mode="road",
        status=status,  # type: ignore[arg-type]
        distance_km=round(total_distance_m / 1000.0, 3),
        estimated_duration_seconds=max(1, int(round(total_cost_seconds or 1.0))),
        risk_score=risk_score,
        waypoints=_ground_waypoints(
            effective_waypoints,
            origin_alt=origin.location.alt_m,
            destination_alt=destination.location.alt_m,
        ),
        road_segment_ids=list(edge_ids),
    )


def _air_route_plan(
    *,
    manifest_model: TacticalScenarioManifest,
    unit: DispatchUnit,
    origin: Facility,
    destination: Facility,
) -> RoutePlan:
    incident = manifest_model.incident_context.geometry.centroid
    midpoint = Wgs84GroundPoint(
        lat=round((origin.location.lat + incident.lat) / 2.0, 6),
        lon=round((origin.location.lon + incident.lon) / 2.0, 6),
        alt_m=round(max(origin.location.alt_m, destination.location.alt_m) + 420.0, 1),
    )
    distance_m = haversine_m(
        start_lat=origin.location.lat,
        start_lon=origin.location.lon,
        end_lat=destination.location.lat,
        end_lon=destination.location.lon,
    )
    speed_kph = 165.0 if unit.unit_type == "helicopter" else 260.0
    duration_seconds = distance_m / (speed_kph * 1000.0 / 3600.0) + 180.0
    return RoutePlan(
        route_plan_id=stable_id("route", unit.unit_id, manifest_model.incident_context.incident_id),
        unit_id=unit.unit_id,
        origin_facility_id=origin.facility_id,
        destination_facility_id=destination.facility_id,
        travel_mode="air",
        status="planned",
        distance_km=round(distance_m / 1000.0, 3),
        estimated_duration_seconds=max(1, int(round(duration_seconds))),
        risk_score=min(
            1.0,
            round(
                0.18
                + manifest_model.incident_context.severity_score * 0.42
                + 0.05 * len(manifest_model.overlay_events),
                6,
            ),
        ),
        waypoints=[origin.location, midpoint, destination.location],
    )


def _route_risk_score(
    *,
    edge_ids: Sequence[str],
    overlay_events: Sequence[TacticalOverlayEvent],
    incident_context: TacticalIncidentContext,
    status: str,
) -> float:
    max_overlay = 0.0
    edge_id_set = set(edge_ids)
    for overlay in overlay_events:
        if any(effect.edge_id in edge_id_set for effect in overlay.edge_effects):
            max_overlay = max(max_overlay, overlay.severity_score)
    aborted_penalty = 0.2 if status == "aborted" else 0.0
    return min(
        1.0,
        round(0.35 * incident_context.severity_score + 0.45 * max_overlay + aborted_penalty, 6),
    )


def _ground_waypoints(
    waypoints: Sequence[Wgs84Point],
    *,
    origin_alt: float,
    destination_alt: float,
) -> list[Wgs84GroundPoint]:
    if not waypoints:
        raise ValueError("route plans require at least one waypoint")
    ground_points: list[Wgs84GroundPoint] = []
    span = max(len(waypoints) - 1, 1)
    for index, waypoint in enumerate(waypoints):
        if index == 0:
            alt_m = origin_alt
        elif index == len(waypoints) - 1:
            alt_m = destination_alt
        else:
            alt_m = origin_alt + (destination_alt - origin_alt) * (index / span)
        ground_points.append(
            Wgs84GroundPoint(lat=waypoint.lat, lon=waypoint.lon, alt_m=round(alt_m, 1))
        )
    return ground_points


def _overlay_window(window: TimeWindow | None) -> OverlayWindow | None:
    if window is None:
        return None
    return OverlayWindow(
        start_time_utc=window.start_time_utc,
        end_time_utc=window.end_time_utc,
    )
