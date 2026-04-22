from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from orbital_shepherd_contracts import (
    ScenarioBundle,
    ScenarioManifest,
    compile_scenario_bundle,
    scenario_bundle_fingerprint,
    validate_canonical_bundle,
)
from orbital_shepherd_contracts.models import (
    DownlinkWindow,
    GroundStation,
    ObservationOpportunity,
    Satellite,
    ScenarioConfig,
    TimeWindow,
)
from orbital_shepherd_core import canonical_json_dumps, format_utc_timestamp, stable_id
from orbital_shepherd_ephemeris import DeterministicKeplerPropagationBackend
from orbital_shepherd_ephemeris.models import (
    GroundLocation,
    OrbitAssetBundle,
    VisibilityTarget,
    VisibilityWindow,
)
from orbital_shepherd_scenario_engine.adapters import (
    build_incident_adapter,
    build_target_demand_adapter,
    build_weather_adapter,
    target_cell_ids_by_h3,
)
from orbital_shepherd_scenario_engine.catalog import (
    FAMILY_SPECS,
    GROUND_STATION_PROFILES,
    SATELLITE_PROFILES,
    ScenarioRecipe,
    builtin_phase1_recipes,
)
from orbital_shepherd_scenario_engine.config import ScenarioEngineConfig


@dataclass(frozen=True, slots=True)
class ScenarioBundleRecord:
    bundle_id: str
    scenario_family: str
    bundle_fingerprint: str
    path: Path


def compile_recipe_to_manifest(
    recipe: ScenarioRecipe,
    *,
    engine_config: ScenarioEngineConfig | None = None,
    orbit_asset_bundle: OrbitAssetBundle | Mapping[str, Any] | Path | str | None = None,
) -> ScenarioManifest:
    config = engine_config or ScenarioEngineConfig(benchmark_id=recipe.benchmark_id)
    propagation_backend = DeterministicKeplerPropagationBackend()
    orbit_bundle = propagation_backend.load_orbit_asset_snapshot(
        orbit_asset_bundle or config.orbit_asset_bundle_path
    )
    demand_adapter = build_target_demand_adapter(recipe.demand_mode, config.fixture_dir)
    incident_adapter = build_incident_adapter(recipe.incident_mode, config.fixture_dir)
    target_cells = demand_adapter.build_target_cells(recipe)
    incidents = incident_adapter.build_incidents(recipe, target_cells)
    target_id_map = target_cell_ids_by_h3(target_cells)
    family_spec = FAMILY_SPECS[recipe.family]
    weather_profiles = {
        target_id_map[h3_cell]: profile_id
        for h3_cell, profile_id in recipe.fixture_weather_profiles
        if h3_cell in target_id_map
    }
    manifest = ScenarioManifest(
        manifest_id=stable_id("sm", recipe.benchmark_id, recipe.family, f"seed-{recipe.seed}"),
        benchmark_id=recipe.benchmark_id,
        scenario_family=recipe.family,
        simulation_seed=recipe.seed,
        decision_interval_seconds=recipe.decision_interval_seconds,
        time_window=TimeWindow(
            start_time_utc=recipe.start_time_utc,
            end_time_utc=recipe.start_time_utc + timedelta(hours=recipe.horizon_hours),
        ),
        satellites=_manifest_satellites(orbit_bundle, recipe=recipe),
        ground_stations=_manifest_ground_stations(recipe=recipe),
        target_cells=target_cells,
        incidents=incidents,
        config=ScenarioConfig.model_validate(
            {
                "horizon_hours": recipe.horizon_hours,
                "notes": recipe.notes,
                "weather_model": f"{recipe.weather_mode}:phase1-cloud-risk-v1",
                "opportunity_generation": {
                    "quality_threshold": family_spec.quality_threshold,
                    "cloud_block_threshold": family_spec.cloud_block_threshold,
                },
                "scenario_mode": _dominant_mode(recipe),
                "adapter_modes": {
                    "demand": recipe.demand_mode,
                    "incidents": recipe.incident_mode,
                    "weather": recipe.weather_mode,
                },
                "family_display_name": family_spec.display_name,
                "family_parameters": {
                    **recipe.family_parameters,
                    "cloud_style": family_spec.cloud_style,
                    "demand_scale": family_spec.demand_scale,
                    "urgency_bias": family_spec.urgency_bias,
                    "burst_window_minutes": recipe.burst_window_minutes,
                },
                "weather_profile_by_target_cell": weather_profiles,
                "orbit_asset_bundle_id": orbit_bundle.bundle_id,
                "target_index": "H3",
            }
        ),
    )
    return manifest


def compile_manifest_to_bundle(
    manifest: ScenarioManifest | Mapping[str, Any],
    *,
    engine_config: ScenarioEngineConfig | None = None,
    orbit_asset_bundle: OrbitAssetBundle | Mapping[str, Any] | Path | str | None = None,
    propagation_backend: DeterministicKeplerPropagationBackend | None = None,
) -> ScenarioBundle:
    config = engine_config or ScenarioEngineConfig()
    backend = propagation_backend or DeterministicKeplerPropagationBackend()
    manifest_model = (
        manifest
        if isinstance(manifest, ScenarioManifest)
        else ScenarioManifest.model_validate(manifest)
    )
    orbit_bundle = backend.load_orbit_asset_snapshot(
        orbit_asset_bundle or config.orbit_asset_bundle_path
    )
    config_extras = _scenario_config_extras(manifest_model)
    family_parameters = dict(_mapping(config_extras.get("family_parameters", {})))
    weather_adapter = build_weather_adapter(
        mode=str(_mapping(config_extras.get("adapter_modes", {})).get("weather", "synthetic")),
        fixture_dir=config.fixture_dir,
        family=manifest_model.scenario_family,
        seed=manifest_model.simulation_seed,
        start_time_utc=manifest_model.time_window.start_time_utc,
        weather_profile_by_target_cell=_mapping(
            config_extras.get("weather_profile_by_target_cell", {})
        ),
        cloud_style=str(
            family_parameters.get(
                "cloud_style", FAMILY_SPECS[manifest_model.scenario_family].cloud_style
            )
        ),
        trap_relief_hour=int(family_parameters.get("trap_window_hours", 12)),
    )
    observation_opportunities = _build_observation_opportunities(
        manifest_model,
        orbit_bundle=orbit_bundle,
        weather_adapter=weather_adapter,
        backend=backend,
    )
    downlink_windows = _build_downlink_windows(
        manifest_model,
        orbit_bundle=orbit_bundle,
        backend=backend,
    )
    return compile_scenario_bundle(
        manifest_model,
        compiled_at=config.compiled_at_utc,
        compiler_version=config.compiler_version,
        observation_opportunities=observation_opportunities,
        downlink_windows=downlink_windows,
    )


def build_scenario_pack(
    *,
    engine_config: ScenarioEngineConfig | None = None,
    output_dir: Path | None = None,
    recipes: Sequence[ScenarioRecipe] | None = None,
) -> list[ScenarioBundleRecord]:
    config = engine_config or ScenarioEngineConfig()
    destination = output_dir or config.scenario_dir
    destination.mkdir(parents=True, exist_ok=True)
    backend = DeterministicKeplerPropagationBackend()
    orbit_bundle = backend.load_orbit_asset_snapshot(config.orbit_asset_bundle_path)
    records: list[ScenarioBundleRecord] = []
    recipe_set = tuple(recipes or builtin_phase1_recipes(config.benchmark_id))
    for recipe in recipe_set:
        manifest = compile_recipe_to_manifest(
            recipe,
            engine_config=config,
            orbit_asset_bundle=orbit_bundle,
        )
        bundle = compile_manifest_to_bundle(
            manifest,
            engine_config=config,
            orbit_asset_bundle=orbit_bundle,
            propagation_backend=backend,
        )
        output_path = destination / f"{bundle.bundle_id.replace(':', '--')}.json"
        output_path.write_text(
            canonical_json_dumps(bundle.model_dump(mode="json", exclude_none=True)) + "\n",
            encoding="utf-8",
        )
        records.append(
            ScenarioBundleRecord(
                bundle_id=bundle.bundle_id,
                scenario_family=bundle.scenario_family,
                bundle_fingerprint=bundle.bundle_fingerprint,
                path=output_path,
            )
        )
    return sorted(records, key=lambda item: item.bundle_id)


def validate_scenario_pack(
    *,
    engine_config: ScenarioEngineConfig | None = None,
    input_dir: Path | None = None,
    recipes: Sequence[ScenarioRecipe] | None = None,
) -> list[ScenarioBundleRecord]:
    config = engine_config or ScenarioEngineConfig()
    source_dir = input_dir or config.scenario_dir
    backend = DeterministicKeplerPropagationBackend()
    orbit_bundle = backend.load_orbit_asset_snapshot(config.orbit_asset_bundle_path)
    expected_bundles: dict[str, ScenarioBundle] = {}
    recipe_set = tuple(recipes or builtin_phase1_recipes(config.benchmark_id))
    for recipe in recipe_set:
        manifest = compile_recipe_to_manifest(
            recipe,
            engine_config=config,
            orbit_asset_bundle=orbit_bundle,
        )
        bundle = compile_manifest_to_bundle(
            manifest,
            engine_config=config,
            orbit_asset_bundle=orbit_bundle,
            propagation_backend=backend,
        )
        expected_bundles[bundle.bundle_id] = bundle

    records: list[ScenarioBundleRecord] = []
    for path in sorted(source_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        bundle = validate_canonical_bundle(document)
        if bundle.bundle_fingerprint != scenario_bundle_fingerprint(
            bundle.model_dump(mode="json", exclude_none=True)
        ):
            raise ValueError(f"bundle fingerprint mismatch: {bundle.bundle_id}")
        expected = expected_bundles.get(bundle.bundle_id)
        if expected is None:
            raise ValueError(f"unexpected scenario bundle in pack: {bundle.bundle_id}")
        actual_canonical = canonical_json_dumps(
            bundle.model_dump(mode="json", exclude_none=True)
        )
        expected_canonical = canonical_json_dumps(
            expected.model_dump(mode="json", exclude_none=True)
        )
        if actual_canonical != expected_canonical:
            raise ValueError(f"bundle bytes drifted for {bundle.bundle_id}")
        records.append(
            ScenarioBundleRecord(
                bundle_id=bundle.bundle_id,
                scenario_family=bundle.scenario_family,
                bundle_fingerprint=bundle.bundle_fingerprint,
                path=path,
            )
        )

    if len(records) != len(expected_bundles):
        raise ValueError(
            "expected "
            f"{len(expected_bundles)} scenario bundles, found "
            f"{len(records)} in {source_dir}"
        )
    families = {record.scenario_family for record in records}
    required_families = {recipe.family for recipe in recipe_set}
    if families != required_families:
        raise ValueError(
            "scenario pack families mismatch: expected "
            f"{sorted(required_families)}, got {sorted(families)}"
        )
    return sorted(records, key=lambda item: item.bundle_id)


def _manifest_satellites(
    orbit_bundle: OrbitAssetBundle,
    *,
    recipe: ScenarioRecipe,
) -> list[Satellite]:
    family_parameters = dict(recipe.family_parameters)
    excluded_satellite_ids = set(
        _string_sequence(family_parameters.get("excluded_satellite_ids"))
    )
    availability_overrides = _string_mapping(
        family_parameters.get("satellite_availability_overrides")
    )
    quality_scale_by_id = _float_mapping(family_parameters.get("satellite_quality_scale_by_id"))
    downlink_rate_scale_by_id = _float_mapping(
        family_parameters.get("satellite_downlink_rate_scale_by_id")
    )
    buffer_scale_by_id = _float_mapping(family_parameters.get("satellite_buffer_scale_by_id"))
    satellites: list[Satellite] = []
    for asset in sorted(orbit_bundle.assets, key=lambda item: item.norad_catalog_id):
        if asset.satellite_id in excluded_satellite_ids:
            continue
        profile = SATELLITE_PROFILES.get(asset.norad_catalog_id)
        if profile is None:
            continue
        quality_scale = quality_scale_by_id.get(asset.satellite_id, 1.0)
        rate_scale = downlink_rate_scale_by_id.get(asset.satellite_id, 1.0)
        buffer_scale = buffer_scale_by_id.get(asset.satellite_id, 1.0)
        availability = availability_overrides.get(
            asset.satellite_id, str(profile["constraints"]["availability"])
        )
        satellites.append(
            Satellite(
                satellite_id=asset.satellite_id,
                name=str(profile["name"]),
                norad_catalog_id=asset.norad_catalog_id,
                sensor={
                    **dict(profile["sensor"]),
                    "quality_nominal": round(
                        _clamp(
                            float(profile["sensor"]["quality_nominal"]) * quality_scale,
                            0.2,
                            0.99,
                        ),
                        3,
                    ),
                },
                downlink={
                    **dict(profile["downlink"]),
                    "buffer_capacity_mb": round(
                        max(1.0, float(profile["downlink"]["buffer_capacity_mb"]) * buffer_scale),
                        1,
                    ),
                    "nominal_downlink_rate_mbps": round(
                        max(
                            1.0,
                            float(profile["downlink"]["nominal_downlink_rate_mbps"])
                            * rate_scale,
                        ),
                        1,
                    ),
                },
                constraints={
                    **dict(profile["constraints"]),
                    "availability": availability,
                },
            )
        )
    if not satellites:
        raise ValueError("no scenario satellites could be resolved from the orbit asset bundle")
    return satellites


def _manifest_ground_stations(*, recipe: ScenarioRecipe) -> list[GroundStation]:
    family_parameters = dict(recipe.family_parameters)
    availability_overrides = _string_mapping(
        family_parameters.get("station_availability_overrides")
    )
    rate_scale_by_id = _float_mapping(
        family_parameters.get("station_downlink_rate_scale_by_id")
    )
    stations: list[GroundStation] = []
    for profile in GROUND_STATION_PROFILES:
        station_id = str(profile["station_id"])
        rate_scale = rate_scale_by_id.get(station_id, 1.0)
        stations.append(
            GroundStation.model_validate(
                {
                    **profile,
                    "capabilities": {
                        **dict(profile["capabilities"]),
                        "availability": availability_overrides.get(
                            station_id, str(profile["capabilities"]["availability"])
                        ),
                        "downlink_rate_mbps": round(
                            max(
                                1.0,
                                float(profile["capabilities"]["downlink_rate_mbps"])
                                * rate_scale,
                            ),
                            1,
                        ),
                    },
                }
            )
        )
    return stations


def _build_observation_opportunities(
    manifest: ScenarioManifest,
    *,
    orbit_bundle: OrbitAssetBundle,
    weather_adapter: Any,
    backend: DeterministicKeplerPropagationBackend,
) -> list[ObservationOpportunity]:
    target_visibility_targets = [
        VisibilityTarget(
            target_id=target.target_cell_id,
            target_kind="target_cell",
            location=GroundLocation(lat=target.centroid.lat, lon=target.centroid.lon, alt_m=0.0),
            max_off_nadir_deg=60.0,
        )
        for target in manifest.target_cells
    ]
    target_windows = backend.compute_visibility(
        orbit_bundle,
        targets=target_visibility_targets,
        start_time_utc=manifest.time_window.start_time_utc,
        end_time_utc=manifest.time_window.end_time_utc,
        step_seconds=manifest.decision_interval_seconds,
    )
    satellites_by_id = {satellite.satellite_id: satellite for satellite in manifest.satellites}
    target_by_id = {target.target_cell_id: target for target in manifest.target_cells}
    incidents_by_target: dict[str, list[Any]] = defaultdict(list)
    for incident in manifest.incidents:
        incidents_by_target[incident.target_cell_id].append(incident)
    threshold = float(manifest.config.opportunity_generation.quality_threshold)
    opportunities: list[ObservationOpportunity] = []
    for window in target_windows:
        if window.satellite_id not in satellites_by_id:
            continue
        satellite = satellites_by_id[window.satellite_id]
        max_off_nadir = satellite.sensor.max_off_nadir_deg or 90.0
        if (
            window.minimum_off_nadir_deg is not None
            and window.minimum_off_nadir_deg > max_off_nadir
        ):
            continue
        target = target_by_id[window.target_id]
        midpoint = window.start_time_utc + ((window.end_time_utc - window.start_time_utc) / 2)
        active_incidents = [
            incident
            for incident in incidents_by_target.get(target.target_cell_id, [])
            if incident.ignition_time_utc <= midpoint
        ]
        cloud_prob = weather_adapter.cloud_risk(target, midpoint)
        geometry_score = _geometry_score(window, satellite)
        urgency_score = max(
            [incident.urgency_score for incident in active_incidents],
            default=target.static_value,
        )
        freshness = _freshness_score(
            active_incidents, midpoint, manifest.time_window.start_time_utc
        )
        usefulness = _clamp(
            (0.42 * geometry_score)
            + (0.28 * (1.0 - cloud_prob))
            + (0.2 * urgency_score)
            + (0.1 * freshness),
            0.01,
            1.0,
        )
        predicted_quality = _clamp(
            usefulness * satellite.sensor.quality_nominal,
            0.03,
            0.99,
        )
        if not active_incidents and predicted_quality < threshold:
            continue
        estimated_volume_mb = satellite.sensor.estimated_data_volume_mb or max(
            120.0, satellite.sensor.swath_km * 12.0
        )
        opportunities.append(
            ObservationOpportunity(
                opportunity_id=stable_id(
                    "opp",
                    window.satellite_id,
                    window.target_id,
                    format_utc_timestamp(window.start_time_utc),
                ),
                satellite_id=window.satellite_id,
                target_cell_id=target.target_cell_id,
                start_time_utc=window.start_time_utc,
                end_time_utc=window.end_time_utc,
                predicted_quality_mean=round(predicted_quality, 3),
                predicted_cloud_obstruction_prob=round(cloud_prob, 3),
                estimated_data_volume_mb=round(estimated_volume_mb, 1),
                slew_cost=round(_slew_cost(window, max_off_nadir), 3),
                incident_ids=[incident.incident_id for incident in active_incidents] or None,
            )
        )
    return sorted(
        opportunities,
        key=lambda item: (
            item.start_time_utc,
            item.satellite_id,
            item.target_cell_id,
            item.opportunity_id,
        ),
    )


def _build_downlink_windows(
    manifest: ScenarioManifest,
    *,
    orbit_bundle: OrbitAssetBundle,
    backend: DeterministicKeplerPropagationBackend,
) -> list[DownlinkWindow]:
    ground_targets = [
        VisibilityTarget(
            target_id=station.station_id,
            target_kind="ground_station",
            location=GroundLocation(
                lat=station.location.lat,
                lon=station.location.lon,
                alt_m=station.location.alt_m,
            ),
            min_elevation_deg=10.0,
        )
        for station in manifest.ground_stations
    ]
    contact_windows = backend.compute_visibility(
        orbit_bundle,
        targets=ground_targets,
        start_time_utc=manifest.time_window.start_time_utc,
        end_time_utc=manifest.time_window.end_time_utc,
        step_seconds=manifest.decision_interval_seconds,
    )
    satellites_by_id = {satellite.satellite_id: satellite for satellite in manifest.satellites}
    stations_by_id = {station.station_id: station for station in manifest.ground_stations}
    downlink_windows: list[DownlinkWindow] = []
    for window in contact_windows:
        if window.satellite_id not in satellites_by_id:
            continue
        if window.target_id not in stations_by_id:
            continue
        station = stations_by_id[window.target_id]
        satellite = satellites_by_id[window.satellite_id]
        duration_seconds = max(60.0, (window.end_time_utc - window.start_time_utc).total_seconds())
        expected_rate = min(
            satellite.downlink.nominal_downlink_rate_mbps,
            station.capabilities.downlink_rate_mbps,
        )
        max_volume = min(
            satellite.downlink.buffer_capacity_mb,
            (duration_seconds * expected_rate / 8.0) * 0.78,
        )
        downlink_windows.append(
            DownlinkWindow(
                window_id=stable_id(
                    "dw",
                    window.satellite_id,
                    window.target_id,
                    format_utc_timestamp(window.start_time_utc),
                ),
                satellite_id=window.satellite_id,
                station_id=window.target_id,
                start_time_utc=window.start_time_utc,
                end_time_utc=window.end_time_utc,
                max_volume_mb=round(max_volume, 1),
                expected_rate_mbps=round(expected_rate, 1),
                outage_risk=round(_outage_risk(window, station), 3),
            )
        )
    return sorted(
        downlink_windows,
        key=lambda item: (item.start_time_utc, item.satellite_id, item.station_id, item.window_id),
    )


def _scenario_config_extras(manifest: ScenarioManifest) -> dict[str, Any]:
    document = manifest.config.model_dump(mode="python")
    for key in ("horizon_hours", "notes", "weather_model", "opportunity_generation"):
        document.pop(key, None)
    return document


def _dominant_mode(recipe: ScenarioRecipe) -> str:
    modes = {recipe.demand_mode, recipe.incident_mode, recipe.weather_mode}
    if len(modes) == 1:
        return next(iter(modes))
    return "hybrid"


def _geometry_score(window: VisibilityWindow, satellite: Satellite) -> float:
    elevation_term = min(window.peak_elevation_deg / 90.0, 1.0)
    max_off_nadir = satellite.sensor.max_off_nadir_deg or 90.0
    off_nadir = window.minimum_off_nadir_deg or 0.0
    off_nadir_term = max(0.0, 1.0 - (off_nadir / max_off_nadir))
    return _clamp((0.55 * elevation_term) + (0.45 * off_nadir_term), 0.05, 1.0)


def _freshness_score(
    active_incidents: Sequence[Any], midpoint: datetime, scenario_start: datetime
) -> float:
    if not active_incidents:
        elapsed_hours = (midpoint - scenario_start).total_seconds() / 3600.0
        return _clamp(1.0 - (elapsed_hours / 24.0), 0.3, 0.75)
    earliest = min(incident.ignition_time_utc for incident in active_incidents)
    delay_hours = (midpoint - earliest).total_seconds() / 3600.0
    return _clamp(1.0 / (1.0 + (delay_hours / 6.0)), 0.2, 1.0)


def _slew_cost(window: VisibilityWindow, max_off_nadir: float) -> float:
    off_nadir = window.minimum_off_nadir_deg or 0.0
    duration_seconds = max(60.0, (window.end_time_utc - window.start_time_utc).total_seconds())
    return _clamp(
        0.08 + (off_nadir / max_off_nadir) * 0.42 + (240.0 / duration_seconds) * 0.05, 0.05, 1.0
    )


def _outage_risk(window: VisibilityWindow, station: GroundStation) -> float:
    availability_penalty = {"nominal": 0.02, "degraded": 0.09, "offline": 0.25}[
        station.capabilities.availability
    ]
    duration_seconds = max(60.0, (window.end_time_utc - window.start_time_utc).total_seconds())
    short_contact_penalty = max(0.0, 0.08 - min(duration_seconds / 1800.0, 0.08))
    return _clamp(availability_penalty + short_contact_penalty, 0.01, 0.35)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            result[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return result


def _string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value)
