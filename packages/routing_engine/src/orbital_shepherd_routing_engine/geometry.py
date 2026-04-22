from __future__ import annotations

import json
import math
from collections.abc import Iterable

from orbital_shepherd_contracts.models import Wgs84BoundingBox, Wgs84Point


def haversine_m(*, start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> float:
    radius_m = 6_371_000.0
    start_lat_rad = math.radians(start_lat)
    end_lat_rad = math.radians(end_lat)
    delta_lat = math.radians(end_lat - start_lat)
    delta_lon = math.radians(end_lon - start_lon)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(start_lat_rad) * math.cos(end_lat_rad) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


def approach_time_seconds(distance_m: float, *, speed_kph: float) -> float:
    if distance_m <= 0:
        return 0.0
    meters_per_second = speed_kph * 1000.0 / 3600.0
    return distance_m / meters_per_second


def point_to_wkt(point: Wgs84Point) -> str:
    return f"POINT({point.lon:.12f} {point.lat:.12f})"


def linestring_to_wkt(points: Iterable[Wgs84Point]) -> str:
    coords = ", ".join(f"{point.lon:.12f} {point.lat:.12f}" for point in points)
    return f"LINESTRING({coords})"


def polygon_to_wkt(points: Iterable[Wgs84Point]) -> str:
    coords = ", ".join(f"{point.lon:.12f} {point.lat:.12f}" for point in points)
    return f"POLYGON(({coords}))"


def bounds_to_polygon_wkt(bounds: Wgs84BoundingBox) -> str:
    ring = [
        Wgs84Point(lat=bounds.min_lat, lon=bounds.min_lon),
        Wgs84Point(lat=bounds.min_lat, lon=bounds.max_lon),
        Wgs84Point(lat=bounds.max_lat, lon=bounds.max_lon),
        Wgs84Point(lat=bounds.max_lat, lon=bounds.min_lon),
        Wgs84Point(lat=bounds.min_lat, lon=bounds.min_lon),
    ]
    return polygon_to_wkt(ring)


def polyline_from_geojson(geometry_json: str) -> list[Wgs84Point]:
    geometry = json.loads(geometry_json)
    coordinates = geometry["coordinates"]
    return [Wgs84Point(lat=float(lat), lon=float(lon)) for lon, lat in coordinates]


def merge_edge_geometries(edge_geometries: Iterable[list[Wgs84Point]]) -> list[Wgs84Point]:
    merged: list[Wgs84Point] = []
    for geometry in edge_geometries:
        for point in geometry:
            if not merged or merged[-1] != point:
                merged.append(point)
    return merged


def dedupe_points(points: Iterable[Wgs84Point]) -> list[Wgs84Point]:
    deduped: list[Wgs84Point] = []
    for point in points:
        if not deduped or deduped[-1] != point:
            deduped.append(point)
    return deduped
