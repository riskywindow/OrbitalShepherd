from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from orbital_shepherd_contracts import RegionBundle
from orbital_shepherd_core import canonical_json_dumps

GeoJsonGeometry = dict[str, Any]
LayerName = Literal["road_nodes", "road_edges", "facilities", "asset_features"]


def export_region_bundle(
    bundle: RegionBundle | Mapping[str, Any],
    *,
    output_dir: Path,
    formats: Iterable[str] = ("geoparquet", "geojson"),
) -> dict[str, Path]:
    bundle_model = (
        bundle
        if isinstance(bundle, RegionBundle)
        else RegionBundle.model_validate(bundle)
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    normalized_formats = {item.lower() for item in formats}
    outputs: dict[str, Path] = {}
    layer_records = _bundle_layers(bundle_model)

    if "geojson" in normalized_formats:
        geojson_dir = destination / "geojson"
        geojson_dir.mkdir(parents=True, exist_ok=True)
        for layer_name, records in layer_records.items():
            path = geojson_dir / f"{layer_name}.geojson"
            _write_geojson(records, path)
            outputs[f"geojson:{layer_name}"] = path

    if "geoparquet" in normalized_formats:
        geoparquet_dir = destination / "geoparquet"
        geoparquet_dir.mkdir(parents=True, exist_ok=True)
        for layer_name, records in layer_records.items():
            path = geoparquet_dir / f"{layer_name}.parquet"
            _write_geoparquet(records, path, layer_name=layer_name)
            outputs[f"geoparquet:{layer_name}"] = path

    metadata_path = destination / "bundle_metadata.txt"
    metadata_path.write_text(
        canonical_json_dumps(
            {
                "region_bundle_id": bundle_model.region_bundle_id,
                "bundle_fingerprint": bundle_model.bundle_fingerprint,
                "h3_cover": bundle_model.h3_cover.model_dump(mode="json"),
                "traversable_node_count": bundle_model.traversable_node_count,
                "traversable_edge_count": bundle_model.traversable_edge_count,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["metadata"] = metadata_path
    return outputs


def _bundle_layers(bundle: RegionBundle) -> dict[LayerName, list[dict[str, Any]]]:
    road_nodes = [
        {
            "properties": {"node_id": node.node_id},
            "geometry": _point_geometry(node.location.lon, node.location.lat),
        }
        for node in bundle.road_nodes
    ]
    road_edges = [
        {
            "properties": {
                "edge_id": edge.edge_id,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "source_ingest_id": edge.source_ingest_id,
                "road_class": edge.road_class,
                "distance_m": edge.distance_m,
                "speed_kph": edge.speed_kph,
                "travel_time_seconds": edge.travel_time_seconds,
                "oneway": edge.oneway,
                "road_name": edge.road_name,
                "source_edge_ref": edge.source_edge_ref,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [(point.lon, point.lat) for point in edge.geometry],
            },
        }
        for edge in bundle.road_edges
    ]
    facilities = [
        {
            "properties": {
                "facility_id": facility.facility_id,
                "facility_name": facility.facility_name,
                "facility_type": facility.facility_type,
                "availability": facility.availability,
                "capacity_units": facility.capacity_units,
                "supported_unit_types": list(facility.supported_unit_types),
            },
            "geometry": _point_geometry(facility.location.lon, facility.location.lat),
        }
        for facility in bundle.facilities
    ]
    asset_features = [
        {
            "properties": {
                "asset_id": asset.asset_id,
                "asset_name": asset.asset_name,
                "asset_kind": asset.asset_kind,
                "geometry_type": asset.geometry_type,
                "priority_class": asset.priority_class,
                "tags": asset.tags or {},
            },
            "geometry": _asset_geometry(asset.model_dump(mode="python")),
        }
        for asset in bundle.asset_features
    ]
    return {
        "road_nodes": road_nodes,
        "road_edges": road_edges,
        "facilities": facilities,
        "asset_features": asset_features,
    }


def _write_geojson(records: list[dict[str, Any]], path: Path) -> None:
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": record["properties"],
                "geometry": record["geometry"],
            }
            for record in records
        ],
    }
    path.write_text(canonical_json_dumps(feature_collection) + "\n", encoding="utf-8")


def _write_geoparquet(records: list[dict[str, Any]], path: Path, *, layer_name: str) -> None:
    if not records:
        raise ValueError(f"cannot export empty layer {layer_name!r}")
    columns: dict[str, list[Any]] = {"geometry": []}
    geometry_types: set[str] = set()
    for record in records:
        properties = record["properties"]
        for key in sorted(properties):
            columns.setdefault(key, []).append(properties[key])
        for key in set(columns) - {"geometry"} - set(properties):
            columns[key].append(None)
        geometry = record["geometry"]
        geometry_types.add(str(geometry["type"]))
        columns["geometry"].append(_geometry_to_wkb(geometry))

    table = pa.table({key: _column_to_array(value) for key, value in columns.items()})
    geo_metadata = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": sorted(geometry_types),
            }
        },
    }
    metadata = dict(table.schema.metadata or {})
    metadata[b"geo"] = json.dumps(geo_metadata, sort_keys=True).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


def _column_to_array(values: list[Any]) -> pa.Array:
    if values and isinstance(values[0], (bytes, bytearray)):
        return pa.array(values, type=pa.binary())
    return pa.array(values)


def _point_geometry(lon: float, lat: float) -> GeoJsonGeometry:
    return {"type": "Point", "coordinates": (lon, lat)}


def _asset_geometry(asset: Mapping[str, Any]) -> GeoJsonGeometry:
    geometry_type = str(asset["geometry_type"])
    if geometry_type == "point":
        point = asset["point"]
        if point is None:
            raise ValueError("point asset missing geometry")
        return _point_geometry(float(point["lon"]), float(point["lat"]))
    ring = asset["ring"]
    if ring is None:
        raise ValueError("polygon asset missing ring")
    return {
        "type": "Polygon",
        "coordinates": [[(float(point["lon"]), float(point["lat"])) for point in ring]],
    }


def _geometry_to_wkb(geometry: Mapping[str, Any]) -> bytes:
    geometry_type = str(geometry["type"])
    if geometry_type == "Point":
        lon, lat = geometry["coordinates"]
        return _wkb_point(float(lon), float(lat))
    if geometry_type == "LineString":
        return _wkb_linestring([(float(lon), float(lat)) for lon, lat in geometry["coordinates"]])
    if geometry_type == "Polygon":
        rings = geometry["coordinates"]
        return _wkb_polygon(
            [[(float(lon), float(lat)) for lon, lat in ring] for ring in rings]
        )
    raise ValueError(f"unsupported geometry type for WKB export: {geometry_type}")


def _wkb_point(lon: float, lat: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, lon, lat)


def _wkb_linestring(coords: list[tuple[float, float]]) -> bytes:
    payload = [struct.pack("<BI", 1, 2), struct.pack("<I", len(coords))]
    payload.extend(struct.pack("<dd", lon, lat) for lon, lat in coords)
    return b"".join(payload)


def _wkb_polygon(rings: list[list[tuple[float, float]]]) -> bytes:
    payload = [struct.pack("<BI", 1, 3), struct.pack("<I", len(rings))]
    for ring in rings:
        payload.append(struct.pack("<I", len(ring)))
        payload.extend(struct.pack("<dd", lon, lat) for lon, lat in ring)
    return b"".join(payload)
