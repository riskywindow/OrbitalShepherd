"""Phase 3 tactical routing engine."""

from orbital_shepherd_routing_engine.config import RoutingEngineConfig
from orbital_shepherd_routing_engine.memory_backend import InMemoryRoutingBackend
from orbital_shepherd_routing_engine.models import (
    ClosureEdgeEffect,
    ClosureOverlaySpec,
    EtaMatrixRequest,
    EtaMatrixResult,
    MatrixWaypoint,
    OverlayRegistrationResult,
    OverlaySelection,
    PathEdgeSummary,
    ReachableFacility,
    ReachableNode,
    RegionIngestResult,
    ResolvedEndpoint,
    RiskMultiplierEdgeEffect,
    RiskMultiplierOverlaySpec,
    RoutingEndpoint,
    ServiceAreaRequest,
    ServiceAreaResult,
    ShortestPathRequest,
    ShortestPathResult,
    TemporaryRestrictionEdgeEffect,
    TemporaryRestrictionOverlaySpec,
)
from orbital_shepherd_routing_engine.postgres_backend import PostgresRoutingBackend
from orbital_shepherd_routing_engine.service import RoutingEngineService, load_region_bundle

PACKAGE_NAME = "routing_engine"
PACKAGE_PURPOSE = "Provide overlay-aware tactical route analysis on immutable RegionBundle graphs."

__all__ = [
    "ClosureEdgeEffect",
    "ClosureOverlaySpec",
    "EtaMatrixRequest",
    "EtaMatrixResult",
    "InMemoryRoutingBackend",
    "MatrixWaypoint",
    "OverlayRegistrationResult",
    "OverlaySelection",
    "PACKAGE_NAME",
    "PACKAGE_PURPOSE",
    "PathEdgeSummary",
    "PostgresRoutingBackend",
    "ReachableFacility",
    "ReachableNode",
    "RegionIngestResult",
    "ResolvedEndpoint",
    "RiskMultiplierEdgeEffect",
    "RiskMultiplierOverlaySpec",
    "RoutingEndpoint",
    "RoutingEngineConfig",
    "RoutingEngineService",
    "ServiceAreaRequest",
    "ServiceAreaResult",
    "ShortestPathRequest",
    "ShortestPathResult",
    "TemporaryRestrictionEdgeEffect",
    "TemporaryRestrictionOverlaySpec",
    "load_region_bundle",
]
