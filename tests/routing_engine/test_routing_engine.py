from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from orbital_shepherd_contracts import load_json, validate_region_bundle
from orbital_shepherd_routing_engine import (
    ClosureEdgeEffect,
    ClosureOverlaySpec,
    EtaMatrixRequest,
    MatrixWaypoint,
    RiskMultiplierEdgeEffect,
    RiskMultiplierOverlaySpec,
    RoutingEndpoint,
    RoutingEngineService,
    ServiceAreaRequest,
    ShortestPathRequest,
    TemporaryRestrictionEdgeEffect,
    TemporaryRestrictionOverlaySpec,
)

FIXTURE_BUNDLE_PATH = Path("data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json")
FIXTURE_REGION_BUNDLE_ID = "rb:fixture-micro-region:v1"
FIXTURE_STATION_ID = "fac:fixture-station-01"
FIXTURE_STAGING_ID = "fac:fixture-staging-01"
FIXTURE_SOUTH_NODE_ID = "rn:38.698500:120.796800"
FIXTURE_FAR_EAST_NODE_ID = "rn:38.704200:120.789100"
FIXTURE_LOOKOUT_EDGE_ID = "re:si-fixture-micro-region-roads-v1:1002:seg-1:forward"
FIXTURE_STATION_ACCESS_REVERSE_EDGE_ID = "re:si-fixture-micro-region-roads-v1:1004:seg-0:reverse"
FIXTURE_PINE_RIDGE_EDGE_ID = "re:si-fixture-micro-region-roads-v1:1001:seg-0:forward"
FIXTURE_SPUR_ENTRY_EDGE_ID = "re:si-fixture-micro-region-roads-v1:1002:seg-0:forward"


def _build_service() -> RoutingEngineService:
    service = RoutingEngineService.in_memory()
    bundle = validate_region_bundle(load_json(FIXTURE_BUNDLE_PATH))
    service.ingest_region_bundle(bundle)
    return service


def _overlay_time() -> datetime:
    return datetime(2026, 4, 22, 12, 0, tzinfo=UTC)


def test_fixture_ingest_is_idempotent() -> None:
    service = RoutingEngineService.in_memory()
    bundle = validate_region_bundle(load_json(FIXTURE_BUNDLE_PATH))

    first = service.ingest_region_bundle(bundle)
    second = service.ingest_region_bundle(bundle)

    assert first.inserted is True
    assert second.inserted is False
    assert first.traversable_node_count == 8
    assert first.traversable_edge_count == 12


def test_shortest_path_uses_fixture_graph() -> None:
    service = _build_service()

    result = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origin=RoutingEndpoint(facility_id=FIXTURE_STATION_ID),
            destination=RoutingEndpoint(facility_id=FIXTURE_STAGING_ID),
        )
    )

    assert result.path_found is True
    assert result.edge_ids == [
        FIXTURE_STATION_ACCESS_REVERSE_EDGE_ID,
        FIXTURE_PINE_RIDGE_EDGE_ID,
        FIXTURE_LOOKOUT_EDGE_ID,
    ]
    assert result.total_cost_seconds == pytest.approx(119.859, rel=1e-6)
    assert result.base_network_cost_seconds == pytest.approx(119.859, rel=1e-6)


def test_eta_matrix_and_service_area_cover_fixture_queries() -> None:
    service = _build_service()

    matrix = service.eta_matrix(
        EtaMatrixRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origins=[
                MatrixWaypoint(
                    waypoint_id="station",
                    endpoint=RoutingEndpoint(facility_id=FIXTURE_STATION_ID),
                ),
                MatrixWaypoint(
                    waypoint_id="south_spur",
                    endpoint=RoutingEndpoint(node_id=FIXTURE_SOUTH_NODE_ID),
                ),
            ],
            destinations=[
                MatrixWaypoint(
                    waypoint_id="staging",
                    endpoint=RoutingEndpoint(facility_id=FIXTURE_STAGING_ID),
                ),
                MatrixWaypoint(
                    waypoint_id="far_east",
                    endpoint=RoutingEndpoint(node_id=FIXTURE_FAR_EAST_NODE_ID),
                ),
            ],
        )
    )

    by_pair = {(entry.origin_id, entry.destination_id): entry for entry in matrix.entries}
    assert by_pair[("station", "staging")].travel_seconds == pytest.approx(119.859, rel=1e-6)
    assert by_pair[("station", "far_east")].travel_seconds == pytest.approx(192.924, rel=1e-6)
    assert by_pair[("south_spur", "staging")].travel_seconds == pytest.approx(52.423, rel=1e-6)
    assert by_pair[("south_spur", "far_east")].travel_seconds == pytest.approx(125.488, rel=1e-6)

    service_area = service.service_area(
        ServiceAreaRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origin=RoutingEndpoint(facility_id=FIXTURE_STATION_ID),
            max_travel_seconds=125.0,
        )
    )

    assert [node.node_id for node in service_area.reachable_nodes] == [
        "rn:38.702100:120.801300",
        "rn:38.700200:120.800500",
        "rn:38.700200:120.796800",
        "rn:38.703400:120.795800",
    ]
    assert [facility.facility_id for facility in service_area.reachable_facilities] == [
        FIXTURE_STATION_ID,
        FIXTURE_STAGING_ID,
    ]


def test_risk_and_closure_overlays_change_route_outputs_without_mutating_base_graph() -> None:
    service = _build_service()

    service.register_overlay(
        RiskMultiplierOverlaySpec(
            overlay_id="ovl:fixture-risk:v1",
            overlay_name="Fixture Risk Inflation",
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            edges=[
                RiskMultiplierEdgeEffect(
                    edge_id=FIXTURE_STATION_ACCESS_REVERSE_EDGE_ID,
                    cost_multiplier=1.5,
                ),
                RiskMultiplierEdgeEffect(edge_id=FIXTURE_PINE_RIDGE_EDGE_ID, cost_multiplier=1.2),
            ],
        )
    )
    service.register_overlay(
        ClosureOverlaySpec(
            overlay_id="ovl:fixture-closure:v1",
            overlay_name="Fixture Closure",
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            edges=[ClosureEdgeEffect(edge_id=FIXTURE_LOOKOUT_EDGE_ID)],
        )
    )

    risk_result = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origin=RoutingEndpoint(facility_id=FIXTURE_STATION_ID),
            destination=RoutingEndpoint(facility_id=FIXTURE_STAGING_ID),
            overlay_selection={
                "overlay_ids": ["ovl:fixture-risk:v1"],
                "effective_at_utc": _overlay_time(),
            },
        )
    )
    closure_result = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origin=RoutingEndpoint(facility_id=FIXTURE_STATION_ID),
            destination=RoutingEndpoint(facility_id=FIXTURE_STAGING_ID),
            overlay_selection={
                "overlay_ids": ["ovl:fixture-closure:v1"],
                "effective_at_utc": _overlay_time(),
            },
        )
    )
    base_result = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origin=RoutingEndpoint(facility_id=FIXTURE_STATION_ID),
            destination=RoutingEndpoint(facility_id=FIXTURE_STAGING_ID),
        )
    )

    assert risk_result.path_found is True
    assert risk_result.edge_ids == base_result.edge_ids
    assert risk_result.total_cost_seconds == pytest.approx(150.3795, rel=1e-6)
    assert risk_result.total_cost_seconds > base_result.total_cost_seconds
    assert closure_result.path_found is False
    assert base_result.path_found is True


def test_temporary_restrictions_change_eta_without_docker() -> None:
    service = _build_service()

    first = service.register_overlay(
        TemporaryRestrictionOverlaySpec(
            overlay_id="ovl:fixture-restriction:v1",
            overlay_name="Fixture Temporary Restriction",
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            edges=[
                TemporaryRestrictionEdgeEffect(
                    edge_id=FIXTURE_SPUR_ENTRY_EDGE_ID,
                    speed_cap_kph=10.0,
                    delay_seconds=20.0,
                )
            ],
        )
    )
    second = service.register_overlay(
        TemporaryRestrictionOverlaySpec(
            overlay_id="ovl:fixture-restriction:v1",
            overlay_name="Fixture Temporary Restriction",
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            edges=[
                TemporaryRestrictionEdgeEffect(
                    edge_id=FIXTURE_SPUR_ENTRY_EDGE_ID,
                    speed_cap_kph=10.0,
                    delay_seconds=20.0,
                )
            ],
        )
    )

    matrix = service.eta_matrix(
        EtaMatrixRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origins=[
                MatrixWaypoint(
                    waypoint_id="south_spur",
                    endpoint=RoutingEndpoint(node_id=FIXTURE_SOUTH_NODE_ID),
                )
            ],
            destinations=[
                MatrixWaypoint(
                    waypoint_id="staging",
                    endpoint=RoutingEndpoint(facility_id=FIXTURE_STAGING_ID),
                )
            ],
            overlay_selection={
                "overlay_ids": ["ovl:fixture-restriction:v1"],
                "effective_at_utc": _overlay_time(),
            },
        )
    )

    assert first.inserted is True
    assert second.inserted is False
    assert matrix.entries[0].travel_seconds == pytest.approx(125.35116, rel=1e-6)
    assert matrix.entries[0].base_network_cost_seconds == pytest.approx(52.423, rel=1e-6)
