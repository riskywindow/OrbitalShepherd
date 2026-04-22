from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbital_shepherd_contracts import (
    H3Cover,
    RegionBundle,
    RegionManifest,
    RoadEdge,
    RoadNode,
    SpatialIngestManifest,
    TravelTimeDefaults,
    load_json,
    repo_root,
)
from orbital_shepherd_contracts.models import BundleCompilation, Wgs84Point
from orbital_shepherd_core import (
    canonical_json_dumps,
    sha256_fingerprint,
    stable_id,
)
from orbital_shepherd_geo_artifacts import export_region_bundle
from orbital_shepherd_region_builder.config import RegionBuilderConfig


@dataclass(frozen=True, slots=True)
class RegionBuildRecord:
    bundle_id: str
    bundle_fingerprint: str
    bundle_path: Path


def region_bundle_id_from_manifest_id(region_manifest_id: str) -> str:
    if region_manifest_id.startswith("rm:"):
        return f"rb:{region_manifest_id[3:]}"
    return f"rb:{region_manifest_id}"


def region_bundle_fingerprint(document: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(document))
    normalized.pop("bundle_fingerprint", None)
    return sha256_fingerprint(canonical_json_dumps(normalized))


def compile_manifest_to_bundle(
    manifest: RegionManifest | Mapping[str, Any],
    *,
    config: RegionBuilderConfig | None = None,
) -> RegionBundle:
    builder_config = config or RegionBuilderConfig()
    manifest_model = (
        manifest
        if isinstance(manifest, RegionManifest)
        else RegionManifest.model_validate(manifest)
    )
    manifest_document = manifest_model.model_dump(mode="json", exclude_none=True)

    source, source_payload = _load_best_road_source(manifest_model)
    road_nodes, road_edges, source_fingerprint = _compile_road_graph(
        manifest_model,
        source=source,
        source_payload=source_payload,
    )
    facilities = sorted(manifest_model.facilities, key=lambda item: item.facility_id)
    asset_features = sorted(manifest_model.asset_features, key=lambda item: item.asset_id)
    h3_cover = _compile_h3_cover(manifest_model)

    spatial_ingests = [
        SpatialIngestManifest(
            spatial_ingest_id=source.ingest_id,
            region_id=manifest_model.region_id,
            source_name=source.source_name,
            layer_type="roads",
            source_uri=source.source_uri,
            source_fingerprint=source_fingerprint,
            transform_fingerprint=sha256_fingerprint(
                canonical_json_dumps(
                    {
                        "travel_time_defaults": (
                            manifest_model.road_network.travel_time_defaults.model_dump(
                                mode="json"
                            )
                        ),
                        "road_node_ids": [node.node_id for node in road_nodes],
                        "road_edge_ids": [edge.edge_id for edge in road_edges],
                    }
                )
            ),
            ingest_time_utc=builder_config.compiled_at_utc,
            coverage_bounds=manifest_model.bounds,
            feature_count=len(source_payload["features"]),
            notes=_spatial_ingest_notes(source, manifest_model),
        )
    ]

    bundle_document = {
        "schema_version": manifest_model.schema_version,
        "region_bundle_id": region_bundle_id_from_manifest_id(manifest_model.region_manifest_id),
        "region_manifest_id": manifest_model.region_manifest_id,
        "region_id": manifest_model.region_id,
        "region_name": manifest_model.region_name,
        "bounds": manifest_model.bounds.model_dump(mode="json"),
        "spatial_ingests": [
            spatial_ingest.model_dump(mode="json", exclude_none=True)
            for spatial_ingest in spatial_ingests
        ],
        "travel_time_defaults": (
            manifest_model.road_network.travel_time_defaults.model_dump(mode="json")
        ),
        "road_nodes": [node.model_dump(mode="json") for node in road_nodes],
        "road_edges": [edge.model_dump(mode="json", exclude_none=True) for edge in road_edges],
        "facilities": [
            facility.model_dump(mode="json", exclude_none=True)
            for facility in facilities
        ],
        "asset_features": [
            asset.model_dump(mode="json", exclude_none=True) for asset in asset_features
        ],
        "h3_cover": h3_cover.model_dump(mode="json"),
        "provenance_notes": _provenance_notes(manifest_model, source),
        "traversable_node_count": len(road_nodes),
        "traversable_edge_count": len(road_edges),
        "compilation": BundleCompilation(
            source_manifest_id=manifest_model.region_manifest_id,
            source_manifest_schema_version=manifest_model.schema_version,
            source_manifest_sha256=sha256_fingerprint(canonical_json_dumps(manifest_document)),
            compiled_at_utc=builder_config.compiled_at_utc,
            compiler_version=builder_config.compiler_version,
        ).model_dump(mode="json"),
    }
    bundle_document["bundle_fingerprint"] = region_bundle_fingerprint(bundle_document)
    return RegionBundle.model_validate(bundle_document)


def compile_manifest_path(
    manifest_path: Path,
    *,
    output_path: Path | None = None,
    export_dir: Path | None = None,
    config: RegionBuilderConfig | None = None,
) -> RegionBuildRecord:
    builder_config = config or RegionBuilderConfig()
    manifest = load_json(Path(manifest_path))
    bundle = compile_manifest_to_bundle(manifest, config=builder_config)
    destination = output_path or (
        builder_config.bundle_root / f"{bundle.region_bundle_id.replace(':', '--')}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        canonical_json_dumps(bundle.model_dump(mode="json", exclude_none=True)) + "\n",
        encoding="utf-8",
    )
    if export_dir is not None:
        export_region_bundle(bundle, output_dir=export_dir)
    return RegionBuildRecord(
        bundle_id=bundle.region_bundle_id,
        bundle_fingerprint=bundle.bundle_fingerprint,
        bundle_path=destination,
    )


def _compile_road_graph(
    manifest: RegionManifest,
    *,
    source: Any,
    source_payload: Mapping[str, Any],
) -> tuple[list[RoadNode], list[RoadEdge], str]:
    if source.source_kind == "fixture_geojson":
        return _compile_fixture_geojson_graph(
            manifest,
            source=source,
            source_payload=source_payload,
        )
    return _compile_osmnx_graph(manifest, source=source)


def _compile_fixture_geojson_graph(
    manifest: RegionManifest,
    *,
    source: Any,
    source_payload: Mapping[str, Any],
) -> tuple[list[RoadNode], list[RoadEdge], str]:
    features = source_payload["features"]
    node_map: dict[tuple[float, float], RoadNode] = {}
    road_edges: list[RoadEdge] = []
    travel_defaults = manifest.road_network.travel_time_defaults
    normalized_source = {
        "type": source_payload.get("type", "FeatureCollection"),
        "features": features,
    }
    source_fingerprint = sha256_fingerprint(canonical_json_dumps(normalized_source))

    for feature_index, feature in enumerate(features):
        geometry = feature["geometry"]
        geometry_type = geometry["type"]
        if geometry_type != "LineString":
            raise ValueError(f"unsupported road geometry type: {geometry_type}")
        coords = list(geometry["coordinates"])
        properties = dict(feature.get("properties", {}))
        direction, oneway = _edge_direction(properties.get("oneway"))
        if direction == "reverse":
            coords = list(reversed(coords))
        speed_kph = _edge_speed_kph(properties, travel_defaults)
        road_class = str(properties.get("highway", "unclassified"))
        source_edge_ref = str(properties.get("osm_way_id", feature_index))

        for segment_index, (start, end) in enumerate(zip(coords, coords[1:], strict=False)):
            start_node = _road_node(node_map, lon=float(start[0]), lat=float(start[1]))
            end_node = _road_node(node_map, lon=float(end[0]), lat=float(end[1]))
            geometry_points = [
                Wgs84Point(lat=float(start[1]), lon=float(start[0])),
                Wgs84Point(lat=float(end[1]), lon=float(end[0])),
            ]
            distance_m = _haversine_m(
                start_lat=float(start[1]),
                start_lon=float(start[0]),
                end_lat=float(end[1]),
                end_lon=float(end[0]),
            )
            if distance_m <= 0:
                continue
            road_edges.append(
                _road_edge(
                    ingest_id=source.ingest_id,
                    source_edge_ref=source_edge_ref,
                    segment_index=segment_index,
                    direction="forward",
                    source_node=start_node,
                    target_node=end_node,
                    road_class=road_class,
                    speed_kph=speed_kph,
                    distance_m=distance_m,
                    oneway=oneway,
                    geometry=geometry_points,
                    road_name=str(properties["name"]) if "name" in properties else None,
                    travel_defaults=travel_defaults,
                )
            )
            if not oneway:
                road_edges.append(
                    _road_edge(
                        ingest_id=source.ingest_id,
                        source_edge_ref=source_edge_ref,
                        segment_index=segment_index,
                        direction="reverse",
                        source_node=end_node,
                        target_node=start_node,
                        road_class=road_class,
                        speed_kph=speed_kph,
                        distance_m=distance_m,
                        oneway=oneway,
                        geometry=list(reversed(geometry_points)),
                        road_name=str(properties["name"]) if "name" in properties else None,
                        travel_defaults=travel_defaults,
                    )
                )

    road_nodes = sorted(node_map.values(), key=lambda item: item.node_id)
    road_edges = sorted(road_edges, key=lambda item: item.edge_id)
    return road_nodes, road_edges, source_fingerprint


def _compile_osmnx_graph(
    manifest: RegionManifest,
    *,
    source: Any,
) -> tuple[list[RoadNode], list[RoadEdge], str]:
    try:
        import osmnx as ox  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "manifest requests an osmnx-backed region compile, but osmnx is not installed"
        ) from exc

    bbox = (
        manifest.bounds.min_lon,
        manifest.bounds.min_lat,
        manifest.bounds.max_lon,
        manifest.bounds.max_lat,
    )
    graph = ox.graph.graph_from_bbox(
        bbox,
        network_type=source.network_type or "drive",
        simplify=False,
        retain_all=False,
        truncate_by_edge=False,
        custom_filter=source.custom_filter,
    )

    node_map: dict[str, RoadNode] = {}
    for node_key, data in graph.nodes(data=True):
        lat = float(data["y"])
        lon = float(data["x"])
        road_node = RoadNode(
            node_id=stable_id("rn", f"{lat:.6f}", f"{lon:.6f}"),
            location=Wgs84Point(lat=lat, lon=lon),
        )
        node_map[str(node_key)] = road_node

    travel_defaults = manifest.road_network.travel_time_defaults
    road_edges: list[RoadEdge] = []
    raw_edges: list[dict[str, Any]] = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        road_class_value = data.get("highway", "unclassified")
        if isinstance(road_class_value, list):
            road_class = str(sorted(str(item) for item in road_class_value)[0])
        else:
            road_class = str(road_class_value)
        speed_kph = _edge_speed_kph(data, travel_defaults)
        geometry = data.get("geometry")
        if geometry is None:
            geometry_points = [node_map[str(u)].location, node_map[str(v)].location]
        else:
            geometry_points = [
                Wgs84Point(lat=float(lat), lon=float(lon)) for lon, lat in geometry.coords
            ]
        distance_m = float(data.get("length") or _polyline_length_m(geometry_points))
        road_edges.append(
            _road_edge(
                ingest_id=source.ingest_id,
                source_edge_ref=f"{u}:{v}:{key}",
                segment_index=0,
                direction="forward",
                source_node=node_map[str(u)],
                target_node=node_map[str(v)],
                road_class=road_class,
                speed_kph=speed_kph,
                distance_m=distance_m,
                oneway=bool(data.get("oneway", True)),
                geometry=geometry_points,
                road_name=str(data["name"]) if "name" in data else None,
                travel_defaults=travel_defaults,
            )
        )
        raw_edges.append(
            {
                "u": str(u),
                "v": str(v),
                "key": str(key),
                "highway": road_class,
                "length": distance_m,
            }
        )

    road_nodes = sorted(node_map.values(), key=lambda item: item.node_id)
    road_edges = sorted(road_edges, key=lambda item: item.edge_id)
    source_fingerprint = sha256_fingerprint(
        canonical_json_dumps(
            {
                "nodes": [
                    {"node_id": node.node_id, "lat": node.location.lat, "lon": node.location.lon}
                    for node in road_nodes
                ],
                "edges": sorted(raw_edges, key=lambda item: (item["u"], item["v"], item["key"])),
            }
        )
    )
    return road_nodes, road_edges, source_fingerprint


def _compile_h3_cover(manifest: RegionManifest) -> H3Cover:
    strategy = manifest.h3_cover.strategy
    if strategy == "explicit":
        cell_ids = sorted(set(manifest.h3_cover.explicit_cell_ids))
        return H3Cover(
            resolution=manifest.h3_cover.resolution,
            generation_strategy=strategy,
            cell_ids=cell_ids,
            cell_count=len(cell_ids),
        )

    try:
        import h3  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "manifest requests bounds-derived h3 cover generation, but h3 is not installed"
        ) from exc

    bbox_geo = {
        "type": "Polygon",
        "coordinates": [
            [
                (manifest.bounds.min_lon, manifest.bounds.min_lat),
                (manifest.bounds.max_lon, manifest.bounds.min_lat),
                (manifest.bounds.max_lon, manifest.bounds.max_lat),
                (manifest.bounds.min_lon, manifest.bounds.max_lat),
                (manifest.bounds.min_lon, manifest.bounds.min_lat),
            ]
        ],
    }
    cell_ids = sorted(h3.geo_to_cells(bbox_geo, res=manifest.h3_cover.resolution))
    return H3Cover(
        resolution=manifest.h3_cover.resolution,
        generation_strategy=strategy,
        cell_ids=cell_ids,
        cell_count=len(cell_ids),
    )


def _load_best_road_source(manifest: RegionManifest) -> tuple[Any, Mapping[str, Any]]:
    errors: list[str] = []
    for source in sorted(
        manifest.road_network.sources,
        key=lambda item: (item.fallback_priority, item.ingest_id),
    ):
        if source.ingest_id not in manifest.spatial_ingest_ids:
            errors.append(f"{source.ingest_id} missing from spatial_ingest_ids")
            continue
        if source.source_kind == "fixture_geojson":
            path = _resolve_source_path(source.source_uri)
            if path.exists():
                document = load_json(path)
                if document.get("type") != "FeatureCollection":
                    raise ValueError(f"expected FeatureCollection road source at {path}")
                return source, document
            errors.append(f"{source.source_name}: fixture missing at {path}")
            continue
        if source.source_kind == "osmnx":
            try:
                import osmnx  # type: ignore[import-not-found]  # noqa: F401
            except ModuleNotFoundError:
                errors.append(f"{source.source_name}: osmnx not installed")
                continue
            return source, {"type": "FeatureCollection", "features": []}
        errors.append(f"{source.source_name}: unsupported source kind {source.source_kind}")
    raise RuntimeError("no usable road source for region manifest: " + "; ".join(errors))


def _resolve_source_path(source_uri: str) -> Path:
    if source_uri.startswith("file://"):
        return Path(source_uri.removeprefix("file://"))
    path = Path(source_uri)
    if path.is_absolute():
        return path
    return repo_root() / path


def _road_node(
    node_map: dict[tuple[float, float], RoadNode],
    *,
    lon: float,
    lat: float,
) -> RoadNode:
    key = (round(lat, 6), round(lon, 6))
    if key not in node_map:
        node_map[key] = RoadNode(
            node_id=stable_id("rn", f"{key[0]:.6f}", f"{key[1]:.6f}"),
            location=Wgs84Point(lat=lat, lon=lon),
        )
    return node_map[key]


def _road_edge(
    *,
    ingest_id: str,
    source_edge_ref: str,
    segment_index: int,
    direction: str,
    source_node: RoadNode,
    target_node: RoadNode,
    road_class: str,
    speed_kph: float,
    distance_m: float,
    oneway: bool,
    geometry: list[Wgs84Point],
    road_name: str | None,
    travel_defaults: TravelTimeDefaults,
) -> RoadEdge:
    travel_time_seconds = (
        distance_m / (speed_kph * 1000 / 3600)
    ) + travel_defaults.intersection_penalty_seconds
    return RoadEdge(
        edge_id=stable_id("re", ingest_id, source_edge_ref, f"seg-{segment_index}", direction),
        source_node_id=source_node.node_id,
        target_node_id=target_node.node_id,
        source_ingest_id=ingest_id,
        road_class=road_class,
        distance_m=round(distance_m, 3),
        speed_kph=speed_kph,
        travel_time_seconds=round(travel_time_seconds, 3),
        oneway=oneway,
        geometry=geometry,
        road_name=road_name,
        source_edge_ref=source_edge_ref,
    )


def _edge_speed_kph(properties: Mapping[str, Any], travel_defaults: TravelTimeDefaults) -> float:
    if "maxspeed_kph" in properties:
        return float(properties["maxspeed_kph"])
    if "maxspeed" in properties:
        parsed = _parse_speed_value(properties["maxspeed"])
        if parsed is not None:
            return parsed
    road_class_value = properties.get("highway", "unclassified")
    if isinstance(road_class_value, list):
        road_class = str(sorted(str(item) for item in road_class_value)[0])
    else:
        road_class = str(road_class_value)
    if road_class in travel_defaults.speed_kph_by_highway:
        return float(travel_defaults.speed_kph_by_highway[road_class])
    return float(travel_defaults.default_speed_kph)


def _parse_speed_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return None
    digits = "".join(char for char in text if char.isdigit() or char == ".")
    if not digits:
        return None
    speed = float(digits)
    if "mph" in text:
        return round(speed * 1.60934, 3)
    return speed


def _edge_direction(value: Any) -> tuple[str, bool]:
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1"}:
        return "forward", True
    if normalized == "-1":
        return "reverse", True
    return "forward", False


def _haversine_m(*, start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(start_lat)
    phi2 = math.radians(end_lat)
    delta_phi = math.radians(end_lat - start_lat)
    delta_lambda = math.radians(end_lon - start_lon)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _polyline_length_m(points: list[Wgs84Point]) -> float:
    return sum(
        _haversine_m(
            start_lat=start.lat,
            start_lon=start.lon,
            end_lat=end.lat,
            end_lon=end.lon,
        )
        for start, end in zip(points, points[1:], strict=False)
    )


def _spatial_ingest_notes(source: Any, manifest: RegionManifest) -> str:
    base = (
        "Deterministic directed road graph normalized into canonical segment IDs and "
        "travel-time defaults."
    )
    if source.source_kind == "osmnx":
        return base + " OSMnx retrieval path requested for this manifest."
    if manifest.notes:
        return f"{base} {manifest.notes}"
    return base


def _provenance_notes(manifest: RegionManifest, source: Any) -> list[str]:
    notes = list(manifest.provenance_notes)
    if source.source_kind in {"fixture_geojson", "osmnx"}:
        notes.append(
            "Road network contains OpenStreetMap-derived content. "
            "Attribution: OpenStreetMap contributors."
        )
    return sorted(set(notes))
