from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from orbital_shepherd_contracts.models import Wgs84Point
from orbital_shepherd_core import sha256_fingerprint


class RoutingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC")
    return value.astimezone(UTC)


class OverlayWindow(RoutingModel):
    start_time_utc: datetime | None = None
    end_time_utc: datetime | None = None

    _validate_start = field_validator("start_time_utc")(_ensure_utc)
    _validate_end = field_validator("end_time_utc")(_ensure_utc)

    @model_validator(mode="after")
    def _validate_order(self) -> OverlayWindow:
        if self.start_time_utc and self.end_time_utc and self.end_time_utc < self.start_time_utc:
            raise ValueError("end_time_utc must be greater than or equal to start_time_utc")
        return self

    def contains(self, value: datetime) -> bool:
        normalized = _ensure_utc(value)
        assert normalized is not None
        if self.start_time_utc and normalized < self.start_time_utc:
            return False
        if self.end_time_utc and normalized > self.end_time_utc:
            return False
        return True


class RoutingEndpoint(RoutingModel):
    facility_id: str | None = None
    node_id: str | None = None
    point: Wgs84Point | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> RoutingEndpoint:
        count = sum(
            candidate is not None for candidate in (self.facility_id, self.node_id, self.point)
        )
        if count != 1:
            raise ValueError("exactly one of facility_id, node_id, or point must be provided")
        return self


class OverlaySelection(RoutingModel):
    overlay_ids: list[str] = Field(default_factory=list)
    effective_at_utc: datetime | None = None

    _validate_effective_at = field_validator("effective_at_utc")(_ensure_utc)


class MatrixWaypoint(RoutingModel):
    waypoint_id: str
    endpoint: RoutingEndpoint


class ClosureEdgeEffect(RoutingModel):
    edge_id: str
    closed: bool = True
    reason: str | None = None


class RiskMultiplierEdgeEffect(RoutingModel):
    edge_id: str
    cost_multiplier: float = Field(gt=0)
    reason: str | None = None


class TemporaryRestrictionEdgeEffect(RoutingModel):
    edge_id: str
    speed_cap_kph: float | None = Field(default=None, gt=0)
    delay_seconds: float = Field(default=0.0, ge=0)
    cost_multiplier: float = Field(default=1.0, gt=0)
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_effect(self) -> TemporaryRestrictionEdgeEffect:
        if self.speed_cap_kph is None and self.delay_seconds == 0 and self.cost_multiplier == 1.0:
            raise ValueError(
                "temporary restriction requires a speed cap, delay_seconds, or cost_multiplier"
            )
        return self


class OverlaySpec(RoutingModel):
    overlay_id: str
    overlay_name: str
    region_bundle_id: str
    window: OverlayWindow | None = None
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClosureOverlaySpec(OverlaySpec):
    overlay_kind: Literal["closure"] = "closure"
    edges: list[ClosureEdgeEffect] = Field(min_length=1)


class RiskMultiplierOverlaySpec(OverlaySpec):
    overlay_kind: Literal["risk_multiplier"] = "risk_multiplier"
    edges: list[RiskMultiplierEdgeEffect] = Field(min_length=1)


class TemporaryRestrictionOverlaySpec(OverlaySpec):
    overlay_kind: Literal["temporary_restriction"] = "temporary_restriction"
    edges: list[TemporaryRestrictionEdgeEffect] = Field(min_length=1)


type OverlaySpecType = (
    ClosureOverlaySpec | RiskMultiplierOverlaySpec | TemporaryRestrictionOverlaySpec
)


class ResolvedEndpoint(RoutingModel):
    endpoint: RoutingEndpoint
    point: Wgs84Point
    node_id: str
    vertex_id: int
    snapped_distance_m: float = Field(ge=0)
    snapped_travel_seconds: float = Field(ge=0)
    facility_id: str | None = None


class ShortestPathRequest(RoutingModel):
    region_bundle_id: str
    origin: RoutingEndpoint
    destination: RoutingEndpoint
    overlay_selection: OverlaySelection = Field(default_factory=OverlaySelection)
    travel_mode: Literal["road"] = "road"


class PathEdgeSummary(RoutingModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    distance_m: float = Field(gt=0)
    base_travel_time_seconds: float = Field(gt=0)
    effective_travel_time_seconds: float = Field(ge=0)


class ShortestPathResult(RoutingModel):
    region_bundle_id: str
    origin: ResolvedEndpoint
    destination: ResolvedEndpoint
    overlay_ids: list[str] = Field(default_factory=list)
    path_found: bool
    total_cost_seconds: float | None = None
    base_network_cost_seconds: float | None = None
    total_distance_m: float | None = None
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    edges: list[PathEdgeSummary] = Field(default_factory=list)
    waypoints: list[Wgs84Point] = Field(default_factory=list)


class EtaMatrixRequest(RoutingModel):
    region_bundle_id: str
    origins: list[MatrixWaypoint] = Field(min_length=1)
    destinations: list[MatrixWaypoint] = Field(min_length=1)
    overlay_selection: OverlaySelection = Field(default_factory=OverlaySelection)
    travel_mode: Literal["road"] = "road"


class EtaMatrixEntry(RoutingModel):
    origin_id: str
    destination_id: str
    path_found: bool
    travel_seconds: float | None = None
    base_network_cost_seconds: float | None = None


class EtaMatrixResult(RoutingModel):
    region_bundle_id: str
    overlay_ids: list[str] = Field(default_factory=list)
    entries: list[EtaMatrixEntry] = Field(default_factory=list)


class ServiceAreaRequest(RoutingModel):
    region_bundle_id: str
    origin: RoutingEndpoint
    max_travel_seconds: float = Field(gt=0)
    overlay_selection: OverlaySelection = Field(default_factory=OverlaySelection)
    travel_mode: Literal["road"] = "road"


class ReachableNode(RoutingModel):
    node_id: str
    point: Wgs84Point
    travel_seconds: float = Field(ge=0)


class ReachableFacility(RoutingModel):
    facility_id: str
    facility_name: str
    point: Wgs84Point
    travel_seconds: float = Field(ge=0)


class ServiceAreaResult(RoutingModel):
    region_bundle_id: str
    origin: ResolvedEndpoint
    overlay_ids: list[str] = Field(default_factory=list)
    max_travel_seconds: float = Field(gt=0)
    reachable_nodes: list[ReachableNode] = Field(default_factory=list)
    reachable_facilities: list[ReachableFacility] = Field(default_factory=list)


class RegionIngestResult(RoutingModel):
    region_bundle_id: str
    bundle_fingerprint: str
    inserted: bool
    traversable_node_count: int = Field(ge=0)
    traversable_edge_count: int = Field(ge=0)


class OverlayRegistrationResult(RoutingModel):
    region_bundle_id: str
    overlay_id: str
    overlay_kind: str
    inserted: bool
    affected_edge_count: int = Field(ge=0)


def coerce_overlay_spec(value: OverlaySpecType | Mapping[str, Any]) -> OverlaySpecType:
    if isinstance(
        value,
        (ClosureOverlaySpec, RiskMultiplierOverlaySpec, TemporaryRestrictionOverlaySpec),
    ):
        return value
    overlay_kind = value.get("overlay_kind")
    if overlay_kind == "closure":
        return ClosureOverlaySpec.model_validate(value)
    if overlay_kind == "risk_multiplier":
        return RiskMultiplierOverlaySpec.model_validate(value)
    if overlay_kind == "temporary_restriction":
        return TemporaryRestrictionOverlaySpec.model_validate(value)
    raise ValueError("overlay_kind must be one of closure, risk_multiplier, temporary_restriction")


def overlay_fingerprint(overlay: OverlaySpecType) -> str:
    return sha256_fingerprint(overlay.model_dump(mode="json", exclude_none=True))
