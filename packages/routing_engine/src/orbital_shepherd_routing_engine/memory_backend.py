from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import inf

from orbital_shepherd_contracts import Facility, RegionBundle, RoadEdge, RoadNode
from orbital_shepherd_contracts.models import Wgs84Point
from orbital_shepherd_routing_engine.geometry import (
    approach_time_seconds,
    dedupe_points,
    haversine_m,
    merge_edge_geometries,
)
from orbital_shepherd_routing_engine.models import (
    ClosureOverlaySpec,
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
    RiskMultiplierOverlaySpec,
    RoutingEndpoint,
    ServiceAreaRequest,
    ServiceAreaResult,
    ShortestPathRequest,
    ShortestPathResult,
    TemporaryRestrictionOverlaySpec,
    coerce_overlay_spec,
    overlay_fingerprint,
)


@dataclass(frozen=True, slots=True)
class _NodeRecord:
    vertex_id: int
    node: RoadNode


@dataclass(frozen=True, slots=True)
class _EdgeRecord:
    edge_index: int
    edge: RoadEdge


@dataclass(frozen=True, slots=True)
class _FacilityRecord:
    facility: Facility
    nearest_node_id: str
    nearest_vertex_id: int
    nearest_distance_m: float
    approach_seconds: float


@dataclass(slots=True)
class _RegionGraph:
    bundle: RegionBundle
    node_by_id: dict[str, _NodeRecord]
    node_by_vertex: dict[int, _NodeRecord]
    edge_by_id: dict[str, _EdgeRecord]
    adjacency: dict[str, list[_EdgeRecord]]
    facility_by_id: dict[str, _FacilityRecord]


@dataclass(frozen=True, slots=True)
class _CostDetails:
    blocked: bool
    effective_cost_seconds: float | None


class InMemoryRoutingBackend:
    def __init__(self) -> None:
        self._regions: dict[str, _RegionGraph] = {}
        self._overlays: dict[str, dict[str, OverlaySpecType]] = {}

    def apply_migrations(self) -> list[str]:
        return []

    def ingest_region_bundle(
        self,
        bundle: RegionBundle | Mapping[str, object],
    ) -> RegionIngestResult:
        bundle_model = (
            bundle if isinstance(bundle, RegionBundle) else RegionBundle.model_validate(bundle)
        )
        existing = self._regions.get(bundle_model.region_bundle_id)
        if existing is not None:
            if existing.bundle.bundle_fingerprint != bundle_model.bundle_fingerprint:
                raise ValueError(
                    f"region_bundle_id {bundle_model.region_bundle_id!r} already exists with a "
                    "different bundle_fingerprint"
                )
            return RegionIngestResult(
                region_bundle_id=bundle_model.region_bundle_id,
                bundle_fingerprint=bundle_model.bundle_fingerprint,
                inserted=False,
                traversable_node_count=bundle_model.traversable_node_count,
                traversable_edge_count=bundle_model.traversable_edge_count,
            )

        node_by_id: dict[str, _NodeRecord] = {}
        node_by_vertex: dict[int, _NodeRecord] = {}
        for vertex_id, node in enumerate(
            sorted(bundle_model.road_nodes, key=lambda item: item.node_id), 1
        ):
            record = _NodeRecord(vertex_id=vertex_id, node=node)
            node_by_id[node.node_id] = record
            node_by_vertex[vertex_id] = record

        edge_by_id: dict[str, _EdgeRecord] = {}
        adjacency: dict[str, list[_EdgeRecord]] = {node_id: [] for node_id in node_by_id}
        for edge_index, edge in enumerate(
            sorted(bundle_model.road_edges, key=lambda item: item.edge_id), 1
        ):
            record = _EdgeRecord(edge_index=edge_index, edge=edge)
            edge_by_id[edge.edge_id] = record
            adjacency.setdefault(edge.source_node_id, []).append(record)

        facility_by_id: dict[str, _FacilityRecord] = {}
        for facility in bundle_model.facilities:
            nearest_node = min(
                node_by_id.values(),
                key=lambda candidate: haversine_m(
                    start_lat=facility.location.lat,
                    start_lon=facility.location.lon,
                    end_lat=candidate.node.location.lat,
                    end_lon=candidate.node.location.lon,
                ),
            )
            distance_m = haversine_m(
                start_lat=facility.location.lat,
                start_lon=facility.location.lon,
                end_lat=nearest_node.node.location.lat,
                end_lon=nearest_node.node.location.lon,
            )
            facility_by_id[facility.facility_id] = _FacilityRecord(
                facility=facility,
                nearest_node_id=nearest_node.node.node_id,
                nearest_vertex_id=nearest_node.vertex_id,
                nearest_distance_m=distance_m,
                approach_seconds=approach_time_seconds(
                    distance_m,
                    speed_kph=bundle_model.travel_time_defaults.default_speed_kph,
                ),
            )

        self._regions[bundle_model.region_bundle_id] = _RegionGraph(
            bundle=bundle_model,
            node_by_id=node_by_id,
            node_by_vertex=node_by_vertex,
            edge_by_id=edge_by_id,
            adjacency=adjacency,
            facility_by_id=facility_by_id,
        )
        self._overlays[bundle_model.region_bundle_id] = {}
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
        overlay_model = coerce_overlay_spec(overlay)
        region_overlays = self._overlays.get(overlay_model.region_bundle_id)
        if region_overlays is None:
            raise LookupError(f"unknown region_bundle_id {overlay_model.region_bundle_id!r}")
        existing = region_overlays.get(overlay_model.overlay_id)
        if existing is not None:
            if overlay_fingerprint(existing) != overlay_fingerprint(overlay_model):
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
        region_overlays[overlay_model.overlay_id] = overlay_model
        return OverlayRegistrationResult(
            region_bundle_id=overlay_model.region_bundle_id,
            overlay_id=overlay_model.overlay_id,
            overlay_kind=overlay_model.overlay_kind,
            inserted=True,
            affected_edge_count=len(overlay_model.edges),
        )

    def shortest_path(self, request: ShortestPathRequest) -> ShortestPathResult:
        region = self._region(request.region_bundle_id)
        effective_at = _effective_at(request.overlay_selection)
        origin = self._resolve_endpoint(region, request.origin)
        destination = self._resolve_endpoint(region, request.destination)
        base_network_cost = self._shortest_cost(
            region,
            origin.node_id,
            destination.node_id,
            OverlaySelection(),
            effective_at=effective_at,
        )

        if origin.node_id == destination.node_id:
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

        distances, predecessors = self._dijkstra(
            region,
            origin.node_id,
            request.overlay_selection,
            effective_at=effective_at,
        )
        if destination.node_id not in distances:
            return ShortestPathResult(
                region_bundle_id=request.region_bundle_id,
                origin=origin,
                destination=destination,
                overlay_ids=list(request.overlay_selection.overlay_ids),
                path_found=False,
                base_network_cost_seconds=base_network_cost,
            )

        edge_records: list[tuple[_EdgeRecord, float]] = []
        node_ids = [destination.node_id]
        current = destination.node_id
        while current != origin.node_id:
            previous_node_id, edge_record, edge_cost = predecessors[current]
            edge_records.append((edge_record, edge_cost))
            current = previous_node_id
            node_ids.append(current)
        edge_records.reverse()
        node_ids.reverse()

        edge_summaries = [
            PathEdgeSummary(
                edge_id=edge_record.edge.edge_id,
                source_node_id=edge_record.edge.source_node_id,
                target_node_id=edge_record.edge.target_node_id,
                distance_m=edge_record.edge.distance_m,
                base_travel_time_seconds=edge_record.edge.travel_time_seconds,
                effective_travel_time_seconds=edge_cost,
            )
            for edge_record, edge_cost in edge_records
        ]
        network_waypoints = merge_edge_geometries(
            edge_record.edge.geometry for edge_record, _edge_cost in edge_records
        )
        total_cost = (
            origin.snapped_travel_seconds
            + distances[destination.node_id]
            + destination.snapped_travel_seconds
        )
        total_distance = (
            origin.snapped_distance_m
            + destination.snapped_distance_m
            + sum(edge_record.edge.distance_m for edge_record, _edge_cost in edge_records)
        )
        return ShortestPathResult(
            region_bundle_id=request.region_bundle_id,
            origin=origin,
            destination=destination,
            overlay_ids=list(request.overlay_selection.overlay_ids),
            path_found=True,
            total_cost_seconds=round(total_cost, 6),
            base_network_cost_seconds=base_network_cost,
            total_distance_m=round(total_distance, 6),
            node_ids=node_ids,
            edge_ids=[edge_record.edge.edge_id for edge_record, _edge_cost in edge_records],
            edges=edge_summaries,
            waypoints=dedupe_points([origin.point, *network_waypoints, destination.point]),
        )

    def eta_matrix(self, request: EtaMatrixRequest) -> EtaMatrixResult:
        region = self._region(request.region_bundle_id)
        effective_at = _effective_at(request.overlay_selection)
        origins = [self._resolve_matrix_waypoint(region, waypoint) for waypoint in request.origins]
        destinations = [
            self._resolve_matrix_waypoint(region, waypoint) for waypoint in request.destinations
        ]

        effective_cache = {
            waypoint.waypoint_id: self._dijkstra(
                region,
                waypoint.endpoint.node_id,
                request.overlay_selection,
                effective_at=effective_at,
            )[0]
            for waypoint in origins
        }
        base_cache = {
            waypoint.waypoint_id: self._dijkstra(
                region,
                waypoint.endpoint.node_id,
                OverlaySelection(),
                effective_at=effective_at,
            )[0]
            for waypoint in origins
        }

        entries: list[EtaMatrixEntry] = []
        for origin in origins:
            for destination in destinations:
                effective_network = _network_cost_lookup(
                    effective_cache[origin.waypoint_id],
                    origin.endpoint.node_id,
                    destination.endpoint.node_id,
                )
                base_network = _network_cost_lookup(
                    base_cache[origin.waypoint_id],
                    origin.endpoint.node_id,
                    destination.endpoint.node_id,
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
        region = self._region(request.region_bundle_id)
        effective_at = _effective_at(request.overlay_selection)
        origin = self._resolve_endpoint(region, request.origin)
        remaining_network_budget = request.max_travel_seconds - origin.snapped_travel_seconds
        reachable_nodes: list[ReachableNode] = []
        reachable_facilities: list[ReachableFacility] = []
        if remaining_network_budget >= 0:
            distances, _predecessors = self._dijkstra(
                region,
                origin.node_id,
                request.overlay_selection,
                effective_at=effective_at,
                max_cost=remaining_network_budget,
            )
            for node_id, network_cost in sorted(
                distances.items(), key=lambda item: (item[1], item[0])
            ):
                record = region.node_by_id[node_id]
                reachable_nodes.append(
                    ReachableNode(
                        node_id=node_id,
                        point=record.node.location,
                        travel_seconds=round(origin.snapped_travel_seconds + network_cost, 6),
                    )
                )
            for facility_id, facility_record in sorted(region.facility_by_id.items()):
                if request.origin.facility_id == facility_id:
                    reachable_facilities.append(
                        ReachableFacility(
                            facility_id=facility_id,
                            facility_name=facility_record.facility.facility_name,
                            point=Wgs84Point(
                                lat=facility_record.facility.location.lat,
                                lon=facility_record.facility.location.lon,
                            ),
                            travel_seconds=0.0,
                        )
                    )
                    continue
                network_cost = _network_cost_lookup(
                    distances,
                    origin.node_id,
                    facility_record.nearest_node_id,
                )
                if network_cost is None:
                    continue
                total_cost = (
                    origin.snapped_travel_seconds + network_cost + facility_record.approach_seconds
                )
                if total_cost <= request.max_travel_seconds:
                    reachable_facilities.append(
                        ReachableFacility(
                            facility_id=facility_id,
                            facility_name=facility_record.facility.facility_name,
                            point=Wgs84Point(
                                lat=facility_record.facility.location.lat,
                                lon=facility_record.facility.location.lon,
                            ),
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

    def _region(self, region_bundle_id: str) -> _RegionGraph:
        region = self._regions.get(region_bundle_id)
        if region is None:
            raise LookupError(f"unknown region_bundle_id {region_bundle_id!r}")
        return region

    def _resolve_matrix_waypoint(
        self, region: _RegionGraph, waypoint: MatrixWaypoint
    ) -> _ResolvedMatrixWaypoint:
        return _ResolvedMatrixWaypoint(
            waypoint_id=waypoint.waypoint_id,
            endpoint=self._resolve_endpoint(region, waypoint.endpoint),
        )

    def _resolve_endpoint(
        self, region: _RegionGraph, endpoint: RoutingEndpoint
    ) -> ResolvedEndpoint:
        if endpoint.facility_id is not None:
            facility = region.facility_by_id.get(endpoint.facility_id)
            if facility is None:
                raise LookupError(f"unknown facility_id {endpoint.facility_id!r}")
            return ResolvedEndpoint(
                endpoint=endpoint,
                facility_id=endpoint.facility_id,
                point=Wgs84Point(
                    lat=facility.facility.location.lat,
                    lon=facility.facility.location.lon,
                ),
                node_id=facility.nearest_node_id,
                vertex_id=facility.nearest_vertex_id,
                snapped_distance_m=round(facility.nearest_distance_m, 6),
                snapped_travel_seconds=round(facility.approach_seconds, 6),
            )
        if endpoint.node_id is not None:
            node = region.node_by_id.get(endpoint.node_id)
            if node is None:
                raise LookupError(f"unknown node_id {endpoint.node_id!r}")
            return ResolvedEndpoint(
                endpoint=endpoint,
                point=node.node.location,
                node_id=node.node.node_id,
                vertex_id=node.vertex_id,
                snapped_distance_m=0.0,
                snapped_travel_seconds=0.0,
            )
        assert endpoint.point is not None
        nearest = min(
            region.node_by_id.values(),
            key=lambda candidate: haversine_m(
                start_lat=endpoint.point.lat,
                start_lon=endpoint.point.lon,
                end_lat=candidate.node.location.lat,
                end_lon=candidate.node.location.lon,
            ),
        )
        distance_m = haversine_m(
            start_lat=endpoint.point.lat,
            start_lon=endpoint.point.lon,
            end_lat=nearest.node.location.lat,
            end_lon=nearest.node.location.lon,
        )
        return ResolvedEndpoint(
            endpoint=endpoint,
            point=endpoint.point,
            node_id=nearest.node.node_id,
            vertex_id=nearest.vertex_id,
            snapped_distance_m=round(distance_m, 6),
            snapped_travel_seconds=round(
                approach_time_seconds(
                    distance_m,
                    speed_kph=region.bundle.travel_time_defaults.default_speed_kph,
                ),
                6,
            ),
        )

    def _dijkstra(
        self,
        region: _RegionGraph,
        start_node_id: str,
        selection: OverlaySelection,
        *,
        effective_at: datetime,
        max_cost: float | None = None,
    ) -> tuple[dict[str, float], dict[str, tuple[str, _EdgeRecord, float]]]:
        active_overlays = self._active_overlays(
            region.bundle.region_bundle_id,
            selection,
            effective_at,
        )
        distances: dict[str, float] = {start_node_id: 0.0}
        predecessors: dict[str, tuple[str, _EdgeRecord, float]] = {}
        queue: list[tuple[float, str]] = [(0.0, start_node_id)]
        while queue:
            current_cost, current_node_id = heapq.heappop(queue)
            if current_cost > distances[current_node_id]:
                continue
            if max_cost is not None and current_cost > max_cost:
                continue
            for edge_record in region.adjacency.get(current_node_id, []):
                details = self._edge_cost_details(region, edge_record.edge, active_overlays)
                if details.blocked or details.effective_cost_seconds is None:
                    continue
                new_cost = current_cost + details.effective_cost_seconds
                if max_cost is not None and new_cost > max_cost:
                    continue
                target_node_id = edge_record.edge.target_node_id
                if new_cost + 1e-9 < distances.get(target_node_id, inf):
                    distances[target_node_id] = new_cost
                    predecessors[target_node_id] = (
                        current_node_id,
                        edge_record,
                        details.effective_cost_seconds,
                    )
                    heapq.heappush(queue, (new_cost, target_node_id))
        return distances, predecessors

    def _shortest_cost(
        self,
        region: _RegionGraph,
        origin_node_id: str,
        destination_node_id: str,
        selection: OverlaySelection,
        *,
        effective_at: datetime,
    ) -> float | None:
        if origin_node_id == destination_node_id:
            return 0.0
        distances, _predecessors = self._dijkstra(
            region,
            origin_node_id,
            selection,
            effective_at=effective_at,
        )
        return distances.get(destination_node_id)

    def _active_overlays(
        self,
        region_bundle_id: str,
        selection: OverlaySelection,
        effective_at: datetime,
    ) -> list[OverlaySpecType]:
        if not selection.overlay_ids:
            return []
        region_overlays = self._overlays.get(region_bundle_id, {})
        active: list[OverlaySpecType] = []
        for overlay_id in selection.overlay_ids:
            overlay = region_overlays.get(overlay_id)
            if overlay is None:
                raise LookupError(f"unknown overlay_id {overlay_id!r} for {region_bundle_id!r}")
            if overlay.window is not None and not overlay.window.contains(effective_at):
                continue
            active.append(overlay)
        return active

    def _edge_cost_details(
        self,
        region: _RegionGraph,
        edge: RoadEdge,
        overlays: list[OverlaySpecType],
    ) -> _CostDetails:
        blocked = False
        risk_multiplier = 1.0
        restriction_multiplier = 1.0
        delay_seconds = 0.0
        min_speed_cap_kph: float | None = None

        for overlay in overlays:
            if isinstance(overlay, ClosureOverlaySpec):
                for effect in overlay.edges:
                    if effect.edge_id == edge.edge_id and effect.closed:
                        blocked = True
            elif isinstance(overlay, RiskMultiplierOverlaySpec):
                for effect in overlay.edges:
                    if effect.edge_id == edge.edge_id:
                        risk_multiplier *= effect.cost_multiplier
            elif isinstance(overlay, TemporaryRestrictionOverlaySpec):
                for effect in overlay.edges:
                    if effect.edge_id != edge.edge_id:
                        continue
                    restriction_multiplier *= effect.cost_multiplier
                    delay_seconds += effect.delay_seconds
                    if effect.speed_cap_kph is not None:
                        min_speed_cap_kph = (
                            effect.speed_cap_kph
                            if min_speed_cap_kph is None
                            else min(min_speed_cap_kph, effect.speed_cap_kph)
                        )

        if blocked:
            return _CostDetails(blocked=True, effective_cost_seconds=None)

        movement_cost = edge.travel_time_seconds
        if min_speed_cap_kph is not None:
            speed_kph = min(edge.speed_kph, min_speed_cap_kph)
            movement_cost = (
                edge.distance_m / (speed_kph * 1000.0 / 3600.0)
                + region.bundle.travel_time_defaults.intersection_penalty_seconds
            )
        effective_cost = (movement_cost + delay_seconds) * restriction_multiplier * risk_multiplier
        return _CostDetails(blocked=False, effective_cost_seconds=round(effective_cost, 6))


@dataclass(frozen=True, slots=True)
class _ResolvedMatrixWaypoint:
    waypoint_id: str
    endpoint: ResolvedEndpoint


def _effective_at(selection: OverlaySelection) -> datetime:
    return selection.effective_at_utc or datetime.now(UTC)


def _network_cost_lookup(
    distances: Mapping[str, float],
    origin_node_id: str,
    destination_node_id: str,
) -> float | None:
    if origin_node_id == destination_node_id:
        return 0.0
    return distances.get(destination_node_id)
