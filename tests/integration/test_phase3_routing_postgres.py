from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orbital_shepherd_contracts import load_json, validate_region_bundle
from orbital_shepherd_routing_engine import (
    ClosureEdgeEffect,
    ClosureOverlaySpec,
    RiskMultiplierEdgeEffect,
    RiskMultiplierOverlaySpec,
    RoutingEndpoint,
    RoutingEngineConfig,
    RoutingEngineService,
    ServiceAreaRequest,
    ShortestPathRequest,
)

pytestmark = [
    pytest.mark.docker_integration,
    pytest.mark.skipif(
        os.getenv("ORBITAL_SHEPHERD_ENABLE_DOCKER_TESTS") != "1",
        reason="set ORBITAL_SHEPHERD_ENABLE_DOCKER_TESTS=1 to run Docker-backed routing tests",
    ),
]

FIXTURE_BUNDLE_PATH = Path("data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json")
FIXTURE_REGION_BUNDLE_ID = "rb:fixture-micro-region:v1"
FIXTURE_STATION_ID = "fac:fixture-station-01"
FIXTURE_STAGING_ID = "fac:fixture-staging-01"
FIXTURE_LOOKOUT_EDGE_ID = "re:si-fixture-micro-region-roads-v1:1002:seg-1:forward"
FIXTURE_STATION_ACCESS_REVERSE_EDGE_ID = "re:si-fixture-micro-region-roads-v1:1004:seg-0:reverse"


def test_postgres_backend_ingests_fixture_and_applies_overlays() -> None:
    pytest.importorskip("psycopg")
    bundle = validate_region_bundle(load_json(FIXTURE_BUNDLE_PATH))
    service = RoutingEngineService.postgres(RoutingEngineConfig())

    service.bootstrap_database()
    ingest = service.ingest_region_bundle(bundle)
    risk = service.register_overlay(
        RiskMultiplierOverlaySpec(
            overlay_id="ovl:pg-fixture-risk:v1",
            overlay_name="Postgres Fixture Risk",
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            edges=[
                RiskMultiplierEdgeEffect(
                    edge_id=FIXTURE_STATION_ACCESS_REVERSE_EDGE_ID,
                    cost_multiplier=1.5,
                )
            ],
        )
    )
    closure = service.register_overlay(
        ClosureOverlaySpec(
            overlay_id="ovl:pg-fixture-closure:v1",
            overlay_name="Postgres Fixture Closure",
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            edges=[ClosureEdgeEffect(edge_id=FIXTURE_LOOKOUT_EDGE_ID)],
        )
    )

    base = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origin=RoutingEndpoint(facility_id=FIXTURE_STATION_ID),
            destination=RoutingEndpoint(facility_id=FIXTURE_STAGING_ID),
        )
    )
    risk_result = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origin=RoutingEndpoint(facility_id=FIXTURE_STATION_ID),
            destination=RoutingEndpoint(facility_id=FIXTURE_STAGING_ID),
            overlay_selection={
                "overlay_ids": [risk.overlay_id],
                "effective_at_utc": datetime(2026, 4, 22, 12, 0, tzinfo=UTC),
            },
        )
    )
    closed = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origin=RoutingEndpoint(facility_id=FIXTURE_STATION_ID),
            destination=RoutingEndpoint(facility_id=FIXTURE_STAGING_ID),
            overlay_selection={
                "overlay_ids": [closure.overlay_id],
                "effective_at_utc": datetime(2026, 4, 22, 12, 0, tzinfo=UTC),
            },
        )
    )
    service_area = service.service_area(
        ServiceAreaRequest(
            region_bundle_id=FIXTURE_REGION_BUNDLE_ID,
            origin=RoutingEndpoint(facility_id=FIXTURE_STATION_ID),
            max_travel_seconds=125.0,
        )
    )

    assert ingest.region_bundle_id == FIXTURE_REGION_BUNDLE_ID
    assert base.path_found is True
    assert base.total_cost_seconds == pytest.approx(119.859, rel=1e-6)
    assert risk_result.path_found is True
    assert risk_result.total_cost_seconds > base.total_cost_seconds
    assert closed.path_found is False
    assert [facility.facility_id for facility in service_area.reachable_facilities] == [
        FIXTURE_STATION_ID,
        FIXTURE_STAGING_ID,
    ]
