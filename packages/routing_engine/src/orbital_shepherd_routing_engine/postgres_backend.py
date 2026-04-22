from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from orbital_shepherd_contracts import RegionAsset, RegionBundle
from orbital_shepherd_contracts.models import Wgs84Point
from orbital_shepherd_core import sha256_fingerprint, stable_id
from orbital_shepherd_routing_engine.config import RoutingEngineConfig
from orbital_shepherd_routing_engine.geometry import (
    approach_time_seconds,
    bounds_to_polygon_wkt,
    dedupe_points,
    haversine_m,
    linestring_to_wkt,
    point_to_wkt,
    polygon_to_wkt,
    polyline_from_geojson,
)
from orbital_shepherd_routing_engine.models import (
    EtaMatrixEntry,
    EtaMatrixRequest,
    EtaMatrixResult,
    MatrixWaypoint,
    OverlayRegistrationResult,
    OverlaySelection,
    OverlaySpecType,
    PathEdgeSummary,
    ReachableFacility,
    ReachableNode,
    RegionIngestResult,
    ResolvedEndpoint,
    RoutingEndpoint,
    ServiceAreaRequest,
    ServiceAreaResult,
    ShortestPathRequest,
    ShortestPathResult,
    coerce_overlay_spec,
    overlay_fingerprint,
)


def _require_psycopg() -> Any:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in real integration path
        raise RuntimeError(
            "psycopg is required for the Postgres routing backend. "
            "Install project dependencies first."
        ) from exc
    return psycopg, dict_row


class PostgresRoutingBackend:
    def __init__(self, config: RoutingEngineConfig | None = None) -> None:
        self.config = config or RoutingEngineConfig()

    def apply_migrations(self) -> list[str]:
        psycopg, _dict_row = _require_psycopg()
        migration_paths = sorted(self.config.sql_migrations_dir.glob("*.sql"))
        applied: list[str] = []
        with psycopg.connect(self.config.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public.orbital_shepherd_schema_migrations (
                        component text NOT NULL,
                        filename text NOT NULL,
                        sha256 text NOT NULL,
                        applied_at_utc timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (component, filename)
                    )
                    """
                )
                for path in migration_paths:
                    checksum = sha256_fingerprint(path.read_text(encoding="utf-8"))
                    cursor.execute(
                        """
                        SELECT sha256
                        FROM public.orbital_shepherd_schema_migrations
                        WHERE component = %s AND filename = %s
                        """,
                        ("routing_engine", path.name),
                    )
                    row = cursor.fetchone()
                    if row is not None:
                        existing_checksum = row[0]
                        if existing_checksum != checksum:
                            raise RuntimeError(
                                f"migration {path.name!r} checksum changed after being applied"
                            )
                        continue
                    cursor.execute(path.read_text(encoding="utf-8"))
                    cursor.execute(
                        """
                        INSERT INTO public.orbital_shepherd_schema_migrations (
                            component,
                            filename,
                            sha256
                        )
                        VALUES (%s, %s, %s)
                        """,
                        ("routing_engine", path.name, checksum),
                    )
                    applied.append(path.name)
            connection.commit()
        return applied

    def ingest_region_bundle(
        self,
        bundle: RegionBundle | Mapping[str, object],
    ) -> RegionIngestResult:
        psycopg, _dict_row = _require_psycopg()
        bundle_model = (
            bundle if isinstance(bundle, RegionBundle) else RegionBundle.model_validate(bundle)
        )
        with psycopg.connect(self.config.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT bundle_fingerprint
                    FROM routing.regions
                    WHERE region_bundle_id = %s
                    """,
                    (bundle_model.region_bundle_id,),
                )
                row = cursor.fetchone()
                if row is not None:
                    if row[0] != bundle_model.bundle_fingerprint:
                        raise ValueError(
                            f"region_bundle_id {bundle_model.region_bundle_id!r} already exists "
                            "a different bundle_fingerprint"
                        )
                    return RegionIngestResult(
                        region_bundle_id=bundle_model.region_bundle_id,
                        bundle_fingerprint=bundle_model.bundle_fingerprint,
                        inserted=False,
                        traversable_node_count=bundle_model.traversable_node_count,
                        traversable_edge_count=bundle_model.traversable_edge_count,
                    )

                cursor.execute(
                    """
                    INSERT INTO routing.regions (
                        region_bundle_id,
                        region_manifest_id,
                        region_id,
                        region_name,
                        bundle_fingerprint,
                        bounds,
                        h3_resolution,
                        h3_cell_ids,
                        travel_time_defaults,
                        default_speed_kph,
                        intersection_penalty_seconds,
                        traversable_node_count,
                        traversable_edge_count,
                        compilation,
                        provenance_notes
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        ST_GeomFromText(%s, 4326),
                        %s,
                        %s,
                        %s::jsonb,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb,
                        %s::jsonb
                    )
                    """,
                    (
                        bundle_model.region_bundle_id,
                        bundle_model.region_manifest_id,
                        bundle_model.region_id,
                        bundle_model.region_name,
                        bundle_model.bundle_fingerprint,
                        bounds_to_polygon_wkt(bundle_model.bounds),
                        bundle_model.h3_cover.resolution,
                        bundle_model.h3_cover.cell_ids,
                        json.dumps(
                            bundle_model.travel_time_defaults.model_dump(mode="json"),
                            sort_keys=True,
                        ),
                        bundle_model.travel_time_defaults.default_speed_kph,
                        bundle_model.travel_time_defaults.intersection_penalty_seconds,
                        bundle_model.traversable_node_count,
                        bundle_model.traversable_edge_count,
                        json.dumps(
                            bundle_model.compilation.model_dump(mode="json"),
                            sort_keys=True,
                        ),
                        json.dumps(bundle_model.provenance_notes, sort_keys=True),
                    ),
                )

                spatial_rows = [
                    (
                        bundle_model.region_bundle_id,
                        ingest.spatial_ingest_id,
                        ingest.region_id,
                        ingest.source_name,
                        ingest.layer_type,
                        ingest.source_uri,
                        ingest.source_fingerprint,
                        ingest.transform_fingerprint,
                        ingest.ingest_time_utc,
                        bounds_to_polygon_wkt(ingest.coverage_bounds),
                        ingest.feature_count,
                        ingest.notes,
                        json.dumps(
                            ingest.model_dump(mode="json", exclude_none=True),
                            sort_keys=True,
                        ),
                    )
                    for ingest in bundle_model.spatial_ingests
                ]
                cursor.executemany(
                    """
                    INSERT INTO routing.spatial_ingests (
                        region_bundle_id,
                        spatial_ingest_id,
                        region_id,
                        source_name,
                        layer_type,
                        source_uri,
                        source_fingerprint,
                        transform_fingerprint,
                        ingest_time_utc,
                        coverage_bounds,
                        feature_count,
                        notes,
                        manifest
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        ST_GeomFromText(%s, 4326),
                        %s,
                        %s,
                        %s::jsonb
                    )
                    """,
                    spatial_rows,
                )

                node_vertex_map: dict[str, int] = {}
                node_rows: list[tuple[object, ...]] = []
                for vertex_id, node in enumerate(
                    sorted(bundle_model.road_nodes, key=lambda item: item.node_id),
                    1,
                ):
                    node_vertex_map[node.node_id] = vertex_id
                    node_rows.append(
                        (
                            bundle_model.region_bundle_id,
                            vertex_id,
                            node.node_id,
                            point_to_wkt(node.location),
                        )
                    )
                cursor.executemany(
                    """
                    INSERT INTO routing.nodes (
                        region_bundle_id,
                        vertex_id,
                        node_id,
                        geom
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        ST_GeomFromText(%s, 4326)
                    )
                    """,
                    node_rows,
                )

                edge_rows: list[tuple[object, ...]] = []
                for edge_index, edge in enumerate(
                    sorted(bundle_model.road_edges, key=lambda item: item.edge_id),
                    1,
                ):
                    edge_rows.append(
                        (
                            bundle_model.region_bundle_id,
                            edge_index,
                            edge.edge_id,
                            edge.source_node_id,
                            edge.target_node_id,
                            node_vertex_map[edge.source_node_id],
                            node_vertex_map[edge.target_node_id],
                            edge.source_ingest_id,
                            edge.road_class,
                            edge.distance_m,
                            edge.speed_kph,
                            edge.travel_time_seconds,
                            edge.oneway,
                            edge.road_name,
                            edge.source_edge_ref,
                            linestring_to_wkt(edge.geometry),
                        )
                    )
                cursor.executemany(
                    """
                    INSERT INTO routing.edges (
                        region_bundle_id,
                        edge_index,
                        edge_id,
                        source_node_id,
                        target_node_id,
                        source_vertex,
                        target_vertex,
                        source_ingest_id,
                        road_class,
                        distance_m,
                        speed_kph,
                        base_travel_time_seconds,
                        oneway,
                        road_name,
                        source_edge_ref,
                        geom
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        ST_GeomFromText(%s, 4326)
                    )
                    """,
                    edge_rows,
                )

                facility_rows: list[tuple[object, ...]] = []
                for facility in bundle_model.facilities:
                    nearest_node = min(
                        bundle_model.road_nodes,
                        key=lambda candidate: haversine_m(
                            start_lat=facility.location.lat,
                            start_lon=facility.location.lon,
                            end_lat=candidate.location.lat,
                            end_lon=candidate.location.lon,
                        ),
                    )
                    distance_m = haversine_m(
                        start_lat=facility.location.lat,
                        start_lon=facility.location.lon,
                        end_lat=nearest_node.location.lat,
                        end_lon=nearest_node.location.lon,
                    )
                    facility_rows.append(
                        (
                            bundle_model.region_bundle_id,
                            facility.facility_id,
                            facility.facility_name,
                            facility.facility_type,
                            facility.availability,
                            facility.capacity_units,
                            facility.supported_unit_types,
                            point_to_wkt(
                                Wgs84Point(lat=facility.location.lat, lon=facility.location.lon)
                            ),
                            nearest_node.node_id,
                            node_vertex_map[nearest_node.node_id],
                            distance_m,
                            approach_time_seconds(
                                distance_m,
                                speed_kph=bundle_model.travel_time_defaults.default_speed_kph,
                            ),
                            json.dumps(
                                facility.model_dump(mode="json", exclude_none=True),
                                sort_keys=True,
                            ),
                        )
                    )
                cursor.executemany(
                    """
                    INSERT INTO routing.facilities (
                        region_bundle_id,
                        facility_id,
                        facility_name,
                        facility_type,
                        availability,
                        capacity_units,
                        supported_unit_types,
                        geom,
                        nearest_node_id,
                        nearest_vertex_id,
                        nearest_distance_m,
                        approach_time_seconds,
                        payload
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        ST_GeomFromText(%s, 4326),
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb
                    )
                    """,
                    facility_rows,
                )

                layer_rows, feature_rows = _asset_layer_rows(bundle_model)
                cursor.executemany(
                    """
                    INSERT INTO routing.asset_layers (
                        region_bundle_id,
                        asset_layer_id,
                        layer_name,
                        asset_kind,
                        geometry_type,
                        feature_count
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    layer_rows,
                )
                cursor.executemany(
                    """
                    INSERT INTO routing.asset_features (
                        region_bundle_id,
                        asset_id,
                        asset_layer_id,
                        asset_name,
                        asset_kind,
                        geometry_type,
                        priority_class,
                        tags,
                        geom,
                        payload
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb,
                        ST_GeomFromText(%s, 4326),
                        %s::jsonb
                    )
                    """,
                    feature_rows,
                )
            connection.commit()
        return RegionIngestResult(
            region_bundle_id=bundle_model.region_bundle_id,
            bundle_fingerprint=bundle_model.bundle_fingerprint,
            inserted=True,
            traversable_node_count=bundle_model.traversable_node_count,
            traversable_edge_count=bundle_model.traversable_edge_count,
        )

    def register_overlay(
        self,
        overlay: OverlaySpecType | Mapping[str, object],
    ) -> OverlayRegistrationResult:
        psycopg, _dict_row = _require_psycopg()
        overlay_model = coerce_overlay_spec(overlay)
        fingerprint = overlay_fingerprint(overlay_model)
        with psycopg.connect(self.config.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT overlay_fingerprint
                    FROM routing.overlay_sets
                    WHERE region_bundle_id = %s AND overlay_id = %s
                    """,
                    (overlay_model.region_bundle_id, overlay_model.overlay_id),
                )
                row = cursor.fetchone()
                if row is not None:
                    if row[0] != fingerprint:
                        raise ValueError(
                            f"overlay_id {overlay_model.overlay_id!r} already exists "
                            "with a different payload"
                        )
                    return OverlayRegistrationResult(
                        region_bundle_id=overlay_model.region_bundle_id,
                        overlay_id=overlay_model.overlay_id,
                        overlay_kind=overlay_model.overlay_kind,
                        inserted=False,
                        affected_edge_count=len(overlay_model.edges),
                    )

                cursor.execute(
                    """
                    INSERT INTO routing.overlay_sets (
                        region_bundle_id,
                        overlay_id,
                        overlay_kind,
                        overlay_name,
                        overlay_fingerprint,
                        starts_at_utc,
                        ends_at_utc,
                        notes,
                        metadata
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb
                    )
                    """,
                    (
                        overlay_model.region_bundle_id,
                        overlay_model.overlay_id,
                        overlay_model.overlay_kind,
                        overlay_model.overlay_name,
                        fingerprint,
                        overlay_model.window.start_time_utc if overlay_model.window else None,
                        overlay_model.window.end_time_utc if overlay_model.window else None,
                        overlay_model.notes,
                        json.dumps(overlay_model.metadata, sort_keys=True),
                    ),
                )
                if overlay_model.overlay_kind == "closure":
                    rows = [
                        (
                            overlay_model.region_bundle_id,
                            overlay_model.overlay_id,
                            effect.edge_id,
                            effect.closed,
                            effect.reason,
                        )
                        for effect in overlay_model.edges
                    ]
                    cursor.executemany(
                        """
                        INSERT INTO routing.overlay_edge_closures (
                            region_bundle_id,
                            overlay_id,
                            edge_id,
                            is_closed,
                            reason
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        rows,
                    )
                elif overlay_model.overlay_kind == "risk_multiplier":
                    rows = [
                        (
                            overlay_model.region_bundle_id,
                            overlay_model.overlay_id,
                            effect.edge_id,
                            effect.cost_multiplier,
                            effect.reason,
                        )
                        for effect in overlay_model.edges
                    ]
                    cursor.executemany(
                        """
                        INSERT INTO routing.overlay_edge_risk_multipliers (
                            region_bundle_id,
                            overlay_id,
                            edge_id,
                            cost_multiplier,
                            reason
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        rows,
                    )
                else:
                    rows = [
                        (
                            overlay_model.region_bundle_id,
                            overlay_model.overlay_id,
                            effect.edge_id,
                            effect.speed_cap_kph,
                            effect.delay_seconds,
                            effect.cost_multiplier,
                            effect.reason,
                        )
                        for effect in overlay_model.edges
                    ]
                    cursor.executemany(
                        """
                        INSERT INTO routing.overlay_edge_temporary_restrictions (
                            region_bundle_id,
                            overlay_id,
                            edge_id,
                            speed_cap_kph,
                            delay_seconds,
                            cost_multiplier,
                            reason
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        rows,
                    )
            connection.commit()
        return OverlayRegistrationResult(
            region_bundle_id=overlay_model.region_bundle_id,
            overlay_id=overlay_model.overlay_id,
            overlay_kind=overlay_model.overlay_kind,
            inserted=True,
            affected_edge_count=len(overlay_model.edges),
        )

    def shortest_path(self, request: ShortestPathRequest) -> ShortestPathResult:
        psycopg, dict_row = _require_psycopg()
        effective_at = request.overlay_selection.effective_at_utc or datetime.now(UTC)
        with psycopg.connect(self.config.dsn, row_factory=dict_row) as connection:
            origin = self._resolve_endpoint(connection, request.region_bundle_id, request.origin)
            destination = self._resolve_endpoint(
                connection,
                request.region_bundle_id,
                request.destination,
            )
            base_network_cost = self._shortest_network_cost(
                connection,
                request.region_bundle_id,
                origin.vertex_id,
                destination.vertex_id,
                OverlaySelection(),
                effective_at,
            )
            if origin.vertex_id == destination.vertex_id:
                total_cost = origin.snapped_travel_seconds + destination.snapped_travel_seconds
                return ShortestPathResult(
                    region_bundle_id=request.region_bundle_id,
                    origin=origin,
                    destination=destination,
                    overlay_ids=list(request.overlay_selection.overlay_ids),
                    path_found=True,
                    total_cost_seconds=round(total_cost, 6),
                    base_network_cost_seconds=0.0,
                    total_distance_m=round(
                        origin.snapped_distance_m + destination.snapped_distance_m,
                        6,
                    ),
                    node_ids=[origin.node_id],
                    waypoints=dedupe_points([origin.point, destination.point]),
                )

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    WITH route AS (
                        SELECT *
                        FROM pgr_dijkstra(
                            routing.effective_edges_pgr_sql(%s, %s, %s),
                            %s,
                            %s,
                            true
                        )
                    )
                    SELECT
                        route.path_seq,
                        route.node,
                        route.edge,
                        route.cost,
                        route.agg_cost,
                        nodes.node_id,
                        edges.edge_id,
                        edges.source_node_id,
                        edges.target_node_id,
                        edges.distance_m,
                        edges.base_travel_time_seconds,
                        ST_AsGeoJSON(edges.geom) AS geometry_json
                    FROM route
                    LEFT JOIN routing.nodes AS nodes
                        ON nodes.region_bundle_id = %s
                       AND nodes.vertex_id = route.node
                    LEFT JOIN routing.edges AS edges
                        ON edges.region_bundle_id = %s
                       AND edges.edge_index = route.edge
                    ORDER BY route.path_seq
                    """,
                    (
                        request.region_bundle_id,
                        request.overlay_selection.overlay_ids,
                        effective_at,
                        origin.vertex_id,
                        destination.vertex_id,
                        request.region_bundle_id,
                        request.region_bundle_id,
                    ),
                )
                rows = cursor.fetchall()

        if not rows:
            return ShortestPathResult(
                region_bundle_id=request.region_bundle_id,
                origin=origin,
                destination=destination,
                overlay_ids=list(request.overlay_selection.overlay_ids),
                path_found=False,
                base_network_cost_seconds=base_network_cost,
            )

        edge_rows = [row for row in rows if row["edge_id"] is not None]
        total_distance_m = (
            origin.snapped_distance_m
            + destination.snapped_distance_m
            + sum(float(row["distance_m"]) for row in edge_rows)
        )
        effective_network_cost = float(rows[-1]["agg_cost"])
        return ShortestPathResult(
            region_bundle_id=request.region_bundle_id,
            origin=origin,
            destination=destination,
            overlay_ids=list(request.overlay_selection.overlay_ids),
            path_found=True,
            total_cost_seconds=round(
                origin.snapped_travel_seconds
                + effective_network_cost
                + destination.snapped_travel_seconds,
                6,
            ),
            base_network_cost_seconds=base_network_cost,
            total_distance_m=round(total_distance_m, 6),
            node_ids=[row["node_id"] for row in rows if row["node_id"] is not None],
            edge_ids=[row["edge_id"] for row in edge_rows],
            edges=[
                PathEdgeSummary(
                    edge_id=row["edge_id"],
                    source_node_id=row["source_node_id"],
                    target_node_id=row["target_node_id"],
                    distance_m=float(row["distance_m"]),
                    base_travel_time_seconds=float(row["base_travel_time_seconds"]),
                    effective_travel_time_seconds=float(row["cost"]),
                )
                for row in edge_rows
            ],
            waypoints=dedupe_points(
                [
                    origin.point,
                    *[
                        point
                        for row in edge_rows
                        for point in polyline_from_geojson(row["geometry_json"])
                    ],
                    destination.point,
                ]
            ),
        )

    def eta_matrix(self, request: EtaMatrixRequest) -> EtaMatrixResult:
        psycopg, dict_row = _require_psycopg()
        effective_at = request.overlay_selection.effective_at_utc or datetime.now(UTC)
        with psycopg.connect(self.config.dsn, row_factory=dict_row) as connection:
            origins = [
                self._resolve_matrix_waypoint(
                    connection,
                    request.region_bundle_id,
                    item,
                )
                for item in request.origins
            ]
            destinations = [
                self._resolve_matrix_waypoint(connection, request.region_bundle_id, item)
                for item in request.destinations
            ]
            effective_matrix = self._cost_matrix(
                connection,
                request.region_bundle_id,
                [item.endpoint.vertex_id for item in origins],
                [item.endpoint.vertex_id for item in destinations],
                request.overlay_selection,
                effective_at,
            )
            base_matrix = self._cost_matrix(
                connection,
                request.region_bundle_id,
                [item.endpoint.vertex_id for item in origins],
                [item.endpoint.vertex_id for item in destinations],
                OverlaySelection(),
                effective_at,
            )
        entries: list[EtaMatrixEntry] = []
        for origin in origins:
            for destination in destinations:
                effective_network = _cost_lookup(
                    effective_matrix,
                    origin.endpoint.vertex_id,
                    destination.endpoint.vertex_id,
                )
                base_network = _cost_lookup(
                    base_matrix,
                    origin.endpoint.vertex_id,
                    destination.endpoint.vertex_id,
                )
                if effective_network is None:
                    entries.append(
                        EtaMatrixEntry(
                            origin_id=origin.waypoint_id,
                            destination_id=destination.waypoint_id,
                            path_found=False,
                            base_network_cost_seconds=base_network,
                        )
                    )
                    continue
                entries.append(
                    EtaMatrixEntry(
                        origin_id=origin.waypoint_id,
                        destination_id=destination.waypoint_id,
                        path_found=True,
                        travel_seconds=round(
                            origin.endpoint.snapped_travel_seconds
                            + effective_network
                            + destination.endpoint.snapped_travel_seconds,
                            6,
                        ),
                        base_network_cost_seconds=base_network,
                    )
                )
        return EtaMatrixResult(
            region_bundle_id=request.region_bundle_id,
            overlay_ids=list(request.overlay_selection.overlay_ids),
            entries=entries,
        )

    def service_area(self, request: ServiceAreaRequest) -> ServiceAreaResult:
        psycopg, dict_row = _require_psycopg()
        effective_at = request.overlay_selection.effective_at_utc or datetime.now(UTC)
        with psycopg.connect(self.config.dsn, row_factory=dict_row) as connection:
            origin = self._resolve_endpoint(connection, request.region_bundle_id, request.origin)
            remaining_network_budget = request.max_travel_seconds - origin.snapped_travel_seconds
            reachable_nodes: list[ReachableNode] = []
            reachable_facilities: list[ReachableFacility] = []
            if remaining_network_budget >= 0:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT
                            dd.node,
                            dd.agg_cost,
                            nodes.node_id,
                            ST_Y(nodes.geom) AS lat,
                            ST_X(nodes.geom) AS lon
                        FROM pgr_drivingDistance(
                            routing.effective_edges_pgr_sql(%s, %s, %s),
                            %s,
                            %s,
                            true
                        ) AS dd
                        JOIN routing.nodes AS nodes
                          ON nodes.region_bundle_id = %s
                         AND nodes.vertex_id = dd.node
                        ORDER BY dd.agg_cost, nodes.node_id
                        """,
                        (
                            request.region_bundle_id,
                            request.overlay_selection.overlay_ids,
                            effective_at,
                            origin.vertex_id,
                            remaining_network_budget,
                            request.region_bundle_id,
                        ),
                    )
                    node_rows = cursor.fetchall()
                reachable_nodes = [
                    ReachableNode(
                        node_id=row["node_id"],
                        point=Wgs84Point(lat=float(row["lat"]), lon=float(row["lon"])),
                        travel_seconds=round(
                            origin.snapped_travel_seconds + float(row["agg_cost"]),
                            6,
                        ),
                    )
                    for row in node_rows
                ]
                node_costs = {int(row["node"]): float(row["agg_cost"]) for row in node_rows}
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT
                            facility_id,
                            facility_name,
                            nearest_vertex_id,
                            approach_time_seconds,
                            ST_Y(geom) AS lat,
                            ST_X(geom) AS lon
                        FROM routing.facilities
                        WHERE region_bundle_id = %s
                        ORDER BY facility_id
                        """,
                        (request.region_bundle_id,),
                    )
                    facility_rows = cursor.fetchall()
                for row in facility_rows:
                    if request.origin.facility_id == row["facility_id"]:
                        reachable_facilities.append(
                            ReachableFacility(
                                facility_id=row["facility_id"],
                                facility_name=row["facility_name"],
                                point=Wgs84Point(lat=float(row["lat"]), lon=float(row["lon"])),
                                travel_seconds=0.0,
                            )
                        )
                        continue
                    network_cost = _cost_lookup(
                        node_costs,
                        origin.vertex_id,
                        int(row["nearest_vertex_id"]),
                    )
                    if network_cost is None:
                        continue
                    total_cost = (
                        origin.snapped_travel_seconds
                        + network_cost
                        + float(row["approach_time_seconds"])
                    )
                    if total_cost <= request.max_travel_seconds:
                        reachable_facilities.append(
                            ReachableFacility(
                                facility_id=row["facility_id"],
                                facility_name=row["facility_name"],
                                point=Wgs84Point(lat=float(row["lat"]), lon=float(row["lon"])),
                                travel_seconds=round(total_cost, 6),
                            )
                        )
        return ServiceAreaResult(
            region_bundle_id=request.region_bundle_id,
            origin=origin,
            overlay_ids=list(request.overlay_selection.overlay_ids),
            max_travel_seconds=request.max_travel_seconds,
            reachable_nodes=reachable_nodes,
            reachable_facilities=sorted(
                reachable_facilities,
                key=lambda item: (item.travel_seconds, item.facility_id),
            ),
        )

    def _resolve_matrix_waypoint(
        self,
        connection: Any,
        region_bundle_id: str,
        waypoint: MatrixWaypoint,
    ) -> _ResolvedMatrixWaypoint:
        return _ResolvedMatrixWaypoint(
            waypoint_id=waypoint.waypoint_id,
            endpoint=self._resolve_endpoint(connection, region_bundle_id, waypoint.endpoint),
        )

    def _resolve_endpoint(
        self,
        connection: Any,
        region_bundle_id: str,
        endpoint: RoutingEndpoint,
    ) -> ResolvedEndpoint:
        default_speed_kph = self._default_speed_kph(connection, region_bundle_id)
        with connection.cursor() as cursor:
            if endpoint.facility_id is not None:
                cursor.execute(
                    """
                    SELECT
                        facility_id,
                        nearest_node_id,
                        nearest_vertex_id,
                        nearest_distance_m,
                        approach_time_seconds,
                        ST_Y(geom) AS lat,
                        ST_X(geom) AS lon
                    FROM routing.facilities
                    WHERE region_bundle_id = %s AND facility_id = %s
                    """,
                    (region_bundle_id, endpoint.facility_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise LookupError(f"unknown facility_id {endpoint.facility_id!r}")
                return ResolvedEndpoint(
                    endpoint=endpoint,
                    facility_id=row["facility_id"],
                    point=Wgs84Point(lat=float(row["lat"]), lon=float(row["lon"])),
                    node_id=row["nearest_node_id"],
                    vertex_id=int(row["nearest_vertex_id"]),
                    snapped_distance_m=float(row["nearest_distance_m"]),
                    snapped_travel_seconds=float(row["approach_time_seconds"]),
                )
            if endpoint.node_id is not None:
                cursor.execute(
                    """
                    SELECT
                        node_id,
                        vertex_id,
                        ST_Y(geom) AS lat,
                        ST_X(geom) AS lon
                    FROM routing.nodes
                    WHERE region_bundle_id = %s AND node_id = %s
                    """,
                    (region_bundle_id, endpoint.node_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise LookupError(f"unknown node_id {endpoint.node_id!r}")
                return ResolvedEndpoint(
                    endpoint=endpoint,
                    point=Wgs84Point(lat=float(row["lat"]), lon=float(row["lon"])),
                    node_id=row["node_id"],
                    vertex_id=int(row["vertex_id"]),
                    snapped_distance_m=0.0,
                    snapped_travel_seconds=0.0,
                )
            assert endpoint.point is not None
            cursor.execute(
                """
                SELECT
                    node_id,
                    vertex_id,
                    ST_Y(geom) AS lat,
                    ST_X(geom) AS lon,
                    ST_Distance(
                        geom::geography,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                    ) AS distance_m
                FROM routing.nodes
                WHERE region_bundle_id = %s
                ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                LIMIT 1
                """,
                (
                    endpoint.point.lon,
                    endpoint.point.lat,
                    region_bundle_id,
                    endpoint.point.lon,
                    endpoint.point.lat,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                raise LookupError(f"unknown region_bundle_id {region_bundle_id!r}")
            distance_m = float(row["distance_m"])
            return ResolvedEndpoint(
                endpoint=endpoint,
                point=endpoint.point,
                node_id=row["node_id"],
                vertex_id=int(row["vertex_id"]),
                snapped_distance_m=distance_m,
                snapped_travel_seconds=approach_time_seconds(
                    distance_m,
                    speed_kph=default_speed_kph,
                ),
            )

    def _default_speed_kph(self, connection: Any, region_bundle_id: str) -> float:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT default_speed_kph
                FROM routing.regions
                WHERE region_bundle_id = %s
                """,
                (region_bundle_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise LookupError(f"unknown region_bundle_id {region_bundle_id!r}")
            return float(row["default_speed_kph"])

    def _shortest_network_cost(
        self,
        connection: Any,
        region_bundle_id: str,
        origin_vertex_id: int,
        destination_vertex_id: int,
        selection: OverlaySelection,
        effective_at: datetime,
    ) -> float | None:
        if origin_vertex_id == destination_vertex_id:
            return 0.0
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT start_vid, end_vid, agg_cost
                FROM pgr_dijkstraCost(
                    routing.effective_edges_pgr_sql(%s, %s, %s),
                    %s,
                    %s,
                    true
                )
                """,
                (
                    region_bundle_id,
                    selection.overlay_ids,
                    effective_at,
                    origin_vertex_id,
                    destination_vertex_id,
                ),
            )
            row = cursor.fetchone()
        return None if row is None else float(row["agg_cost"])

    def _cost_matrix(
        self,
        connection: Any,
        region_bundle_id: str,
        origin_vertex_ids: Sequence[int],
        destination_vertex_ids: Sequence[int],
        selection: OverlaySelection,
        effective_at: datetime,
    ) -> dict[tuple[int, int], float]:
        origins = sorted(set(origin_vertex_ids))
        destinations = sorted(set(destination_vertex_ids))
        if not origins or not destinations:
            return {}
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT start_vid, end_vid, agg_cost
                FROM pgr_dijkstraCost(
                    routing.effective_edges_pgr_sql(%s, %s, %s),
                    %s::bigint[],
                    %s::bigint[],
                    true
                )
                """,
                (
                    region_bundle_id,
                    selection.overlay_ids,
                    effective_at,
                    origins,
                    destinations,
                ),
            )
            rows = cursor.fetchall()
        return {
            (int(row["start_vid"]), int(row["end_vid"])): float(row["agg_cost"]) for row in rows
        }


class _ResolvedMatrixWaypoint:
    def __init__(self, *, waypoint_id: str, endpoint: ResolvedEndpoint) -> None:
        self.waypoint_id = waypoint_id
        self.endpoint = endpoint


def _asset_layer_rows(
    bundle: RegionBundle,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    layer_ids: dict[tuple[str, str], str] = {}
    feature_counts: dict[str, int] = {}
    layer_rows: list[tuple[object, ...]] = []
    feature_rows: list[tuple[object, ...]] = []
    for asset in sorted(bundle.asset_features, key=lambda item: item.asset_id):
        key = (asset.asset_kind, asset.geometry_type)
        if key not in layer_ids:
            layer_id = stable_id(
                "al",
                bundle.region_bundle_id,
                asset.asset_kind,
                asset.geometry_type,
            )
            layer_ids[key] = layer_id
            feature_counts[layer_id] = 0
        layer_id = layer_ids[key]
        feature_counts[layer_id] += 1
        feature_rows.append(
            (
                bundle.region_bundle_id,
                asset.asset_id,
                layer_id,
                asset.asset_name,
                asset.asset_kind,
                asset.geometry_type,
                asset.priority_class,
                json.dumps(asset.tags or {}, sort_keys=True),
                _asset_geometry_wkt(asset),
                json.dumps(asset.model_dump(mode="json", exclude_none=True), sort_keys=True),
            )
        )
    for (asset_kind, geometry_type), layer_id in sorted(
        layer_ids.items(),
        key=lambda item: item[1],
    ):
        layer_rows.append(
            (
                bundle.region_bundle_id,
                layer_id,
                f"{asset_kind}:{geometry_type}",
                asset_kind,
                geometry_type,
                feature_counts[layer_id],
            )
        )
    return layer_rows, feature_rows


def _asset_geometry_wkt(asset: RegionAsset) -> str:
    if asset.point is not None:
        return point_to_wkt(asset.point)
    assert asset.ring is not None
    return polygon_to_wkt(asset.ring)


def _cost_lookup(
    costs: Mapping[tuple[int, int], float] | Mapping[int, float],
    origin: int,
    destination: int,
) -> float | None:
    if origin == destination:
        return 0.0
    if isinstance(next(iter(costs.keys()), None), tuple):
        return cast(Mapping[tuple[int, int], float], costs).get((origin, destination))
    return cast(Mapping[int, float], costs).get(destination)
