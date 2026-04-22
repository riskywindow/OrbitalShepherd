from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from orbital_shepherd_contracts.models import Wgs84Point
from orbital_shepherd_core import canonical_json_dumps
from orbital_shepherd_routing_engine.config import RoutingEngineConfig
from orbital_shepherd_routing_engine.models import (
    ClosureEdgeEffect,
    ClosureOverlaySpec,
    EtaMatrixRequest,
    MatrixWaypoint,
    RiskMultiplierEdgeEffect,
    RiskMultiplierOverlaySpec,
    RoutingEndpoint,
    ServiceAreaRequest,
    ShortestPathRequest,
    TemporaryRestrictionEdgeEffect,
    TemporaryRestrictionOverlaySpec,
)
from orbital_shepherd_routing_engine.service import RoutingEngineService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 routing engine helper CLI.")
    parser.add_argument("--dsn", default=RoutingEngineConfig().dsn)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap-db", help="Apply routing SQL migrations.")
    bootstrap.set_defaults(handler=_handle_bootstrap_db)

    ingest = subparsers.add_parser("ingest-bundle", help="Ingest a RegionBundle into Postgres.")
    ingest.add_argument("bundle_path", type=Path)
    ingest.set_defaults(handler=_handle_ingest_bundle)

    shortest = subparsers.add_parser("shortest-path", help="Run a shortest-path query.")
    shortest.add_argument("--region-bundle-id", required=True)
    shortest.add_argument("--origin", required=True)
    shortest.add_argument("--destination", required=True)
    shortest.add_argument("--overlay-id", action="append", default=[])
    shortest.add_argument("--effective-at-utc")
    shortest.set_defaults(handler=_handle_shortest_path)

    matrix = subparsers.add_parser("eta-matrix", help="Run an ETA matrix query.")
    matrix.add_argument("--region-bundle-id", required=True)
    matrix.add_argument("--origin", action="append", required=True)
    matrix.add_argument("--destination", action="append", required=True)
    matrix.add_argument("--overlay-id", action="append", default=[])
    matrix.add_argument("--effective-at-utc")
    matrix.set_defaults(handler=_handle_eta_matrix)

    service_area = subparsers.add_parser(
        "service-area",
        help="Run a reachable-within-time query.",
    )
    service_area.add_argument("--region-bundle-id", required=True)
    service_area.add_argument("--origin", required=True)
    service_area.add_argument("--max-travel-seconds", type=float, required=True)
    service_area.add_argument("--overlay-id", action="append", default=[])
    service_area.add_argument("--effective-at-utc")
    service_area.set_defaults(handler=_handle_service_area)

    smoke = subparsers.add_parser(
        "smoke",
        help="Bootstrap, ingest the fixture region, and run a minimal validation path.",
    )
    smoke.add_argument(
        "--bundle-path",
        type=Path,
        default=RoutingEngineConfig().fixture_bundle_path,
    )
    smoke.set_defaults(handler=_handle_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = RoutingEngineConfig(dsn=args.dsn)
    service = RoutingEngineService.postgres(config)
    payload = args.handler(service, args)
    print(canonical_json_dumps(payload.model_dump(mode="json", exclude_none=True)))
    return 0


def _handle_bootstrap_db(service: RoutingEngineService, _args: argparse.Namespace):
    applied = service.bootstrap_database()
    return _JsonEnvelope(status="ok", payload={"applied_migrations": applied})


def _handle_ingest_bundle(service: RoutingEngineService, args: argparse.Namespace):
    service.bootstrap_database()
    result = service.ingest_region_bundle_path(args.bundle_path)
    return _JsonEnvelope(status="ok", payload=result.model_dump(mode="json"))


def _handle_shortest_path(service: RoutingEngineService, args: argparse.Namespace):
    result = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=args.region_bundle_id,
            origin=_parse_endpoint(args.origin),
            destination=_parse_endpoint(args.destination),
            overlay_selection=_overlay_selection(args.overlay_id, args.effective_at_utc),
        )
    )
    return _JsonEnvelope(status="ok", payload=result.model_dump(mode="json", exclude_none=True))


def _handle_eta_matrix(service: RoutingEngineService, args: argparse.Namespace):
    result = service.eta_matrix(
        EtaMatrixRequest(
            region_bundle_id=args.region_bundle_id,
            origins=[_parse_matrix_waypoint(item) for item in args.origin],
            destinations=[_parse_matrix_waypoint(item) for item in args.destination],
            overlay_selection=_overlay_selection(args.overlay_id, args.effective_at_utc),
        )
    )
    return _JsonEnvelope(status="ok", payload=result.model_dump(mode="json", exclude_none=True))


def _handle_service_area(service: RoutingEngineService, args: argparse.Namespace):
    result = service.service_area(
        ServiceAreaRequest(
            region_bundle_id=args.region_bundle_id,
            origin=_parse_endpoint(args.origin),
            max_travel_seconds=args.max_travel_seconds,
            overlay_selection=_overlay_selection(args.overlay_id, args.effective_at_utc),
        )
    )
    return _JsonEnvelope(status="ok", payload=result.model_dump(mode="json", exclude_none=True))


def _handle_smoke(service: RoutingEngineService, args: argparse.Namespace):
    service.bootstrap_database()
    ingest_result = service.ingest_region_bundle_path(args.bundle_path)
    region_bundle_id = ingest_result.region_bundle_id
    effective_at = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)

    risk_overlay = RiskMultiplierOverlaySpec(
        overlay_id="ovl:fixture-risk-spike:v1",
        overlay_name="Fixture Risk Spike",
        region_bundle_id=region_bundle_id,
        edges=[
            RiskMultiplierEdgeEffect(
                edge_id="re:si-fixture-micro-region-roads-v1:1004:seg-0:reverse",
                cost_multiplier=1.5,
                reason="Staging traffic congestion near station access.",
            ),
            RiskMultiplierEdgeEffect(
                edge_id="re:si-fixture-micro-region-roads-v1:1001:seg-0:forward",
                cost_multiplier=1.2,
                reason="Reduced visibility on Pine Ridge Road.",
            ),
        ],
    )
    restriction_overlay = TemporaryRestrictionOverlaySpec(
        overlay_id="ovl:fixture-slowdown:v1",
        overlay_name="Fixture Slowdown",
        region_bundle_id=region_bundle_id,
        edges=[
            TemporaryRestrictionEdgeEffect(
                edge_id="re:si-fixture-micro-region-roads-v1:1002:seg-0:forward",
                speed_cap_kph=10.0,
                delay_seconds=20.0,
                reason="Temporary metering at the Lookout Spur entrance.",
            )
        ],
    )
    closure_overlay = ClosureOverlaySpec(
        overlay_id="ovl:fixture-closure:v1",
        overlay_name="Fixture Closure",
        region_bundle_id=region_bundle_id,
        edges=[
            ClosureEdgeEffect(
                edge_id="re:si-fixture-micro-region-roads-v1:1002:seg-1:forward",
                reason="Active closure into the staging area.",
            )
        ],
    )
    service.register_overlay(risk_overlay)
    service.register_overlay(restriction_overlay)
    service.register_overlay(closure_overlay)

    shortest_base = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=region_bundle_id,
            origin=RoutingEndpoint(facility_id="fac:fixture-station-01"),
            destination=RoutingEndpoint(facility_id="fac:fixture-staging-01"),
        )
    )
    shortest_risk = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=region_bundle_id,
            origin=RoutingEndpoint(facility_id="fac:fixture-station-01"),
            destination=RoutingEndpoint(facility_id="fac:fixture-staging-01"),
            overlay_selection=_overlay_selection(
                [risk_overlay.overlay_id],
                effective_at.isoformat(),
            ),
        )
    )
    shortest_closed = service.shortest_path(
        ShortestPathRequest(
            region_bundle_id=region_bundle_id,
            origin=RoutingEndpoint(facility_id="fac:fixture-station-01"),
            destination=RoutingEndpoint(facility_id="fac:fixture-staging-01"),
            overlay_selection=_overlay_selection(
                [closure_overlay.overlay_id],
                effective_at.isoformat(),
            ),
        )
    )
    matrix = service.eta_matrix(
        EtaMatrixRequest(
            region_bundle_id=region_bundle_id,
            origins=[
                MatrixWaypoint(
                    waypoint_id="station",
                    endpoint=RoutingEndpoint(facility_id="fac:fixture-station-01"),
                ),
                MatrixWaypoint(
                    waypoint_id="south_spur",
                    endpoint=RoutingEndpoint(node_id="rn:38.698500:120.796800"),
                ),
            ],
            destinations=[
                MatrixWaypoint(
                    waypoint_id="staging",
                    endpoint=RoutingEndpoint(facility_id="fac:fixture-staging-01"),
                ),
                MatrixWaypoint(
                    waypoint_id="far_east",
                    endpoint=RoutingEndpoint(node_id="rn:38.704200:120.789100"),
                ),
            ],
            overlay_selection=_overlay_selection(
                [risk_overlay.overlay_id, restriction_overlay.overlay_id],
                effective_at.isoformat(),
            ),
        )
    )
    service_area = service.service_area(
        ServiceAreaRequest(
            region_bundle_id=region_bundle_id,
            origin=RoutingEndpoint(facility_id="fac:fixture-station-01"),
            max_travel_seconds=125.0,
            overlay_selection=_overlay_selection([], effective_at.isoformat()),
        )
    )
    return _JsonEnvelope(
        status="ok",
        payload={
            "ingest": ingest_result.model_dump(mode="json"),
            "shortest_path_base": shortest_base.model_dump(mode="json", exclude_none=True),
            "shortest_path_risk": shortest_risk.model_dump(mode="json", exclude_none=True),
            "shortest_path_closed": shortest_closed.model_dump(mode="json", exclude_none=True),
            "eta_matrix": matrix.model_dump(mode="json", exclude_none=True),
            "service_area": service_area.model_dump(mode="json", exclude_none=True),
        },
    )


def _parse_endpoint(value: str) -> RoutingEndpoint:
    if value.startswith("facility:"):
        return RoutingEndpoint(facility_id=value.removeprefix("facility:"))
    if value.startswith("node:"):
        return RoutingEndpoint(node_id=value.removeprefix("node:"))
    if value.startswith("point:"):
        point_text = value.removeprefix("point:")
        lat_text, lon_text = point_text.split(",", maxsplit=1)
        return RoutingEndpoint(point=Wgs84Point(lat=float(lat_text), lon=float(lon_text)))
    raise ValueError("endpoint must use facility:<id>, node:<id>, or point:<lat>,<lon>")


def _parse_matrix_waypoint(value: str) -> MatrixWaypoint:
    waypoint_id, endpoint_text = value.split("=", maxsplit=1)
    return MatrixWaypoint(waypoint_id=waypoint_id, endpoint=_parse_endpoint(endpoint_text))


def _overlay_selection(overlay_ids: list[str], effective_at_utc: str | None):
    from orbital_shepherd_routing_engine.models import OverlaySelection

    return OverlaySelection(
        overlay_ids=overlay_ids,
        effective_at_utc=(
            datetime.fromisoformat(effective_at_utc.replace("Z", "+00:00"))
            if effective_at_utc
            else None
        ),
    )


class _JsonEnvelope:
    def __init__(self, *, status: str, payload: dict[str, object]) -> None:
        self.status = status
        self.payload = payload

    def model_dump(self, *, mode: str = "json", exclude_none: bool = True) -> dict[str, object]:
        return {"status": self.status, "payload": self.payload}
