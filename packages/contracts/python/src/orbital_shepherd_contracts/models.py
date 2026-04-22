from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from orbital_shepherd_contracts.constants import (
    CANONICAL_REPLAY_EVENT_TYPES,
    CANONICAL_TACTICAL_REPLAY_EVENT_TYPES,
    PHASE1_SCHEMA_VERSION,
)

NamespacedId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._:-]{2,127}$")]
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
SchemaVersion = Literal["1.0.0"]
Availability = Literal["nominal", "degraded", "offline"]
PriorityClass = Literal["low", "medium", "high", "critical"]
IncidentState = Literal["candidate", "active", "contained", "archived"]
IncidentType = Literal["wildfire"]
RecommendedAction = Literal["monitor", "dispatch_recon", "dispatch_ground"]
ActorType = Literal["system", "planner", "satellite", "ground_station", "metric_engine"]
SpatialLayerType = Literal["roads", "facilities", "hazard_perimeter", "terrain", "weather"]
RoadSourceKind = Literal["fixture_geojson", "osmnx"]
DispatchUnitType = Literal["engine", "crew", "dozer", "air_tanker", "helicopter", "command"]
DispatchUnitStatus = Literal[
    "available",
    "assigned",
    "enroute",
    "on_scene",
    "restaging",
    "out_of_service",
]
FacilityType = Literal[
    "station",
    "helibase",
    "drop_point",
    "hospital",
    "staging_area",
    "command_post",
]
TravelMode = Literal["road", "air"]
RegionGeometryType = Literal["point", "polygon"]
H3CoverStrategy = Literal["explicit", "bounds"]
RoutePlanStatus = Literal["planned", "active", "completed", "aborted"]
TacticalIncidentState = Literal["reported", "mobilizing", "engaged", "contained", "demobilized"]
TacticalSeverityClass = Literal["moderate", "high", "very_high", "extreme"]
RegionResolutionStrategy = Literal["explicit_bundle", "h3_cover", "fallback_single_bundle"]
ScoutAssetType = Literal["drone", "scout_team", "lookout"]
ScoutAssetStatus = Literal["available", "deployed", "offline", "maintenance"]
OverlayEventKind = Literal["closure", "risk_zone", "temporary_penalty"]
TacticalActorType = Literal[
    "system",
    "planner",
    "dispatch_unit",
    "facility",
    "metric_engine",
    "bridge",
]
ReplayEventType = Literal[
    "scenario_bundle_loaded",
    "episode_started",
    "candidate_set_materialized",
    "action_mask_emitted",
    "action_selected",
    "observation_executed",
    "downlink_executed",
    "incident_packet_emitted",
    "reward_assessed",
    "episode_ended",
]
TacticalReplayEventType = Literal[
    "tactical_activation_created",
    "tactical_scenario_bundle_loaded",
    "tactical_episode_started",
    "tactical_candidate_set_materialized",
    "tactical_action_selected",
    "dispatch_unit_assigned",
    "route_plan_committed",
    "unit_position_updated",
    "facility_status_updated",
    "incident_state_updated",
    "tactical_metrics_assessed",
    "tactical_episode_ended",
]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Phase1ContractModel(ContractModel):
    schema_version: SchemaVersion = PHASE1_SCHEMA_VERSION


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC")
    return value.astimezone(UTC)


class Wgs84Point(ContractModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Wgs84GroundPoint(Wgs84Point):
    alt_m: float


class Wgs84BoundingBox(ContractModel):
    min_lat: float = Field(ge=-90, le=90)
    min_lon: float = Field(ge=-180, le=180)
    max_lat: float = Field(ge=-90, le=90)
    max_lon: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def _validate_order(self) -> Wgs84BoundingBox:
        if self.max_lat < self.min_lat:
            raise ValueError("max_lat must be greater than or equal to min_lat")
        if self.max_lon < self.min_lon:
            raise ValueError("max_lon must be greater than or equal to min_lon")
        return self


class TimeWindow(ContractModel):
    start_time_utc: datetime
    end_time_utc: datetime

    _validate_start = field_validator("start_time_utc")(_ensure_utc)
    _validate_end = field_validator("end_time_utc")(_ensure_utc)

    @model_validator(mode="after")
    def _validate_order(self) -> TimeWindow:
        if self.end_time_utc < self.start_time_utc:
            raise ValueError("end_time_utc must be greater than or equal to start_time_utc")
        return self


class OpportunityGenerationConfig(ContractModel):
    quality_threshold: float = Field(ge=0, le=1)
    cloud_block_threshold: float = Field(ge=0, le=1)


class ScenarioConfig(ContractModel):
    model_config = ConfigDict(extra="allow")

    horizon_hours: int = Field(ge=1)
    notes: str
    weather_model: NonEmptyString
    opportunity_generation: OpportunityGenerationConfig


class TacticalScenarioConfig(ContractModel):
    model_config = ConfigDict(extra="allow")

    planning_horizon_minutes: int = Field(ge=1)
    max_active_routes: int = Field(ge=1)
    reroute_on_blockage: bool
    notes: str


class SatelliteSensor(ContractModel):
    sensor_id: NamespacedId
    swath_km: float = Field(gt=0)
    quality_nominal: float = Field(ge=0, le=1)
    max_off_nadir_deg: float | None = Field(default=None, ge=0, le=90)
    estimated_data_volume_mb: float | None = Field(default=None, gt=0)


class SatelliteDownlink(ContractModel):
    buffer_capacity_mb: float = Field(gt=0)
    nominal_downlink_rate_mbps: float = Field(gt=0)


class SatelliteConstraints(ContractModel):
    max_retargets_per_orbit: int = Field(ge=0)
    availability: Availability | None = None


class Satellite(Phase1ContractModel):
    satellite_id: NamespacedId
    name: NonEmptyString
    norad_catalog_id: int = Field(ge=1)
    sensor: SatelliteSensor
    downlink: SatelliteDownlink
    constraints: SatelliteConstraints


class GroundStationCapabilities(ContractModel):
    max_concurrent_contacts: int = Field(ge=1)
    downlink_rate_mbps: float = Field(gt=0)
    availability: Availability


class GroundStation(Phase1ContractModel):
    station_id: NamespacedId
    name: NonEmptyString
    location: Wgs84GroundPoint
    capabilities: GroundStationCapabilities


class TargetCell(Phase1ContractModel):
    target_cell_id: NamespacedId
    h3_cell: NonEmptyString
    centroid: Wgs84Point
    region_name: str | None = None
    static_value: float = Field(ge=0)
    priority_class: PriorityClass | None = None


class Incident(Phase1ContractModel):
    incident_id: NamespacedId
    incident_type: IncidentType
    target_cell_id: NamespacedId
    ignition_time_utc: datetime
    urgency_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    state: IncidentState
    estimated_area_ha: float | None = Field(default=None, ge=0)

    _validate_ignition = field_validator("ignition_time_utc")(_ensure_utc)


class ObservationOpportunity(Phase1ContractModel):
    opportunity_id: NamespacedId
    satellite_id: NamespacedId
    target_cell_id: NamespacedId
    start_time_utc: datetime
    end_time_utc: datetime
    predicted_quality_mean: float = Field(ge=0, le=1)
    predicted_cloud_obstruction_prob: float = Field(ge=0, le=1)
    estimated_data_volume_mb: float = Field(gt=0)
    slew_cost: float = Field(ge=0)
    incident_ids: list[NamespacedId] | None = None

    _validate_start = field_validator("start_time_utc")(_ensure_utc)
    _validate_end = field_validator("end_time_utc")(_ensure_utc)

    @model_validator(mode="after")
    def _validate_order(self) -> ObservationOpportunity:
        if self.end_time_utc < self.start_time_utc:
            raise ValueError("end_time_utc must be greater than or equal to start_time_utc")
        return self


class DownlinkWindow(Phase1ContractModel):
    window_id: NamespacedId
    satellite_id: NamespacedId
    station_id: NamespacedId
    start_time_utc: datetime
    end_time_utc: datetime
    max_volume_mb: float = Field(ge=0)
    expected_rate_mbps: float = Field(gt=0)
    outage_risk: float = Field(ge=0, le=1)

    _validate_start = field_validator("start_time_utc")(_ensure_utc)
    _validate_end = field_validator("end_time_utc")(_ensure_utc)

    @model_validator(mode="after")
    def _validate_order(self) -> DownlinkWindow:
        if self.end_time_utc < self.start_time_utc:
            raise ValueError("end_time_utc must be greater than or equal to start_time_utc")
        return self


class IncidentPacket(Phase1ContractModel):
    packet_id: NamespacedId
    incident_id: NamespacedId
    target_cell_id: NamespacedId
    observation_time_utc: datetime
    downlink_time_utc: datetime
    confidence: float = Field(ge=0, le=1)
    urgency_score: float = Field(ge=0, le=1)
    recommended_action: RecommendedAction
    observation_opportunity_id: NamespacedId | None = None
    downstream_value_estimate: float | None = Field(default=None, ge=0)
    summary: str | None = None

    _validate_observation = field_validator("observation_time_utc")(_ensure_utc)
    _validate_downlink = field_validator("downlink_time_utc")(_ensure_utc)

    @model_validator(mode="after")
    def _validate_order(self) -> IncidentPacket:
        if self.downlink_time_utc < self.observation_time_utc:
            raise ValueError(
                "downlink_time_utc must be greater than or equal to observation_time_utc"
            )
        return self


class ScenarioManifest(Phase1ContractModel):
    manifest_id: NamespacedId
    benchmark_id: NonEmptyString
    scenario_family: NonEmptyString
    simulation_seed: int
    decision_interval_seconds: int = Field(ge=1)
    time_window: TimeWindow
    satellites: list[Satellite]
    ground_stations: list[GroundStation]
    target_cells: list[TargetCell]
    incidents: list[Incident]
    config: ScenarioConfig


class BundleCompilation(ContractModel):
    source_manifest_id: NamespacedId
    source_manifest_schema_version: NonEmptyString
    source_manifest_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    compiled_at_utc: datetime
    compiler_version: NonEmptyString

    _validate_compiled_at = field_validator("compiled_at_utc")(_ensure_utc)


class SpatialIngestManifest(Phase1ContractModel):
    spatial_ingest_id: NamespacedId
    region_id: NamespacedId
    source_name: NonEmptyString
    layer_type: SpatialLayerType
    source_uri: NonEmptyString
    source_fingerprint: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    transform_fingerprint: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    ingest_time_utc: datetime
    coverage_bounds: Wgs84BoundingBox
    feature_count: int = Field(ge=0)
    notes: str | None = None

    _validate_ingest_time = field_validator("ingest_time_utc")(_ensure_utc)


class Facility(Phase1ContractModel):
    facility_id: NamespacedId
    facility_name: NonEmptyString
    facility_type: FacilityType
    location: Wgs84GroundPoint
    availability: Availability
    capacity_units: int = Field(ge=0)
    supported_unit_types: list[DispatchUnitType] = Field(min_length=1)


class DispatchUnit(Phase1ContractModel):
    unit_id: NamespacedId
    callsign: NonEmptyString
    unit_type: DispatchUnitType
    status: DispatchUnitStatus
    home_facility_id: NamespacedId
    current_facility_id: NamespacedId | None = None
    travel_mode: TravelMode
    personnel_count: int = Field(ge=1)
    equipment_capacity: float = Field(ge=0)
    location: Wgs84GroundPoint


class RoutePlan(Phase1ContractModel):
    route_plan_id: NamespacedId
    unit_id: NamespacedId
    origin_facility_id: NamespacedId
    destination_facility_id: NamespacedId
    travel_mode: TravelMode
    status: RoutePlanStatus
    distance_km: float = Field(gt=0)
    estimated_duration_seconds: int = Field(ge=1)
    risk_score: float = Field(ge=0, le=1)
    waypoints: list[Wgs84GroundPoint] = Field(min_length=2)
    road_segment_ids: list[NonEmptyString] | None = None


class RoadNetworkSource(ContractModel):
    ingest_id: NamespacedId
    source_kind: RoadSourceKind
    source_name: NonEmptyString
    source_uri: NonEmptyString
    fallback_priority: int = Field(default=0, ge=0)
    network_type: NonEmptyString | None = None
    custom_filter: NonEmptyString | None = None
    notes: str | None = None


class TravelTimeDefaults(ContractModel):
    default_speed_kph: float = Field(gt=0)
    speed_kph_by_highway: dict[NonEmptyString, float] = Field(default_factory=dict)
    intersection_penalty_seconds: float = Field(default=0, ge=0)


class RegionAsset(Phase1ContractModel):
    asset_id: NamespacedId
    asset_name: NonEmptyString
    asset_kind: NonEmptyString
    geometry_type: RegionGeometryType
    point: Wgs84Point | None = None
    ring: list[Wgs84Point] | None = None
    priority_class: PriorityClass | None = None
    tags: dict[NonEmptyString, NonEmptyString] | None = None

    @model_validator(mode="after")
    def _validate_geometry(self) -> RegionAsset:
        if self.geometry_type == "point":
            if self.point is None or self.ring is not None:
                raise ValueError("point assets require point geometry and must not define ring")
            return self
        if self.ring is None or len(self.ring) < 4 or self.point is not None:
            raise ValueError("polygon assets require a closed ring with at least four points")
        if self.ring[0] != self.ring[-1]:
            raise ValueError("polygon asset rings must be closed")
        return self


class RoadNode(ContractModel):
    node_id: NamespacedId
    location: Wgs84Point


class RoadEdge(ContractModel):
    edge_id: NamespacedId
    source_node_id: NamespacedId
    target_node_id: NamespacedId
    source_ingest_id: NamespacedId
    road_class: NonEmptyString
    distance_m: float = Field(gt=0)
    speed_kph: float = Field(gt=0)
    travel_time_seconds: float = Field(gt=0)
    oneway: bool
    geometry: list[Wgs84Point] = Field(min_length=2)
    road_name: str | None = None
    source_edge_ref: str | None = None


class H3CoverConfig(ContractModel):
    resolution: int = Field(ge=0, le=15)
    strategy: H3CoverStrategy
    explicit_cell_ids: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_cells(self) -> H3CoverConfig:
        if self.strategy == "explicit" and not self.explicit_cell_ids:
            raise ValueError("explicit h3 cover strategy requires explicit_cell_ids")
        return self


class H3Cover(ContractModel):
    resolution: int = Field(ge=0, le=15)
    generation_strategy: H3CoverStrategy
    cell_ids: list[NonEmptyString] = Field(min_length=1)
    cell_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_count(self) -> H3Cover:
        if self.cell_count != len(self.cell_ids):
            raise ValueError("cell_count must match len(cell_ids)")
        return self


class RegionRoadNetworkConfig(ContractModel):
    sources: list[RoadNetworkSource] = Field(min_length=1)
    travel_time_defaults: TravelTimeDefaults


class RegionManifest(Phase1ContractModel):
    region_manifest_id: NamespacedId
    region_id: NamespacedId
    region_name: NonEmptyString
    bounds: Wgs84BoundingBox
    default_cell_size_m: float = Field(gt=0)
    spatial_ingest_ids: list[NamespacedId] = Field(min_length=1)
    build_seed: int = 0
    road_network: RegionRoadNetworkConfig
    facilities: list[Facility] = Field(default_factory=list)
    asset_features: list[RegionAsset] = Field(default_factory=list)
    h3_cover: H3CoverConfig
    provenance_notes: list[NonEmptyString] = Field(default_factory=list)
    notes: str | None = None


class RegionBundle(Phase1ContractModel):
    region_bundle_id: NamespacedId
    region_manifest_id: NamespacedId
    region_id: NamespacedId
    region_name: NonEmptyString
    bundle_fingerprint: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    bounds: Wgs84BoundingBox
    spatial_ingests: list[SpatialIngestManifest] = Field(min_length=1)
    travel_time_defaults: TravelTimeDefaults
    road_nodes: list[RoadNode] = Field(min_length=1)
    road_edges: list[RoadEdge] = Field(min_length=1)
    facilities: list[Facility]
    asset_features: list[RegionAsset]
    h3_cover: H3Cover
    provenance_notes: list[NonEmptyString] = Field(default_factory=list)
    traversable_node_count: int = Field(ge=0)
    traversable_edge_count: int = Field(ge=0)
    compilation: BundleCompilation


class TacticalIncidentGeometry(ContractModel):
    centroid: Wgs84Point
    estimated_area_ha: float = Field(ge=0)
    perimeter_ring: list[Wgs84Point] | None = None

    @model_validator(mode="after")
    def _validate_geometry(self) -> TacticalIncidentGeometry:
        if self.perimeter_ring is None:
            return self
        if len(self.perimeter_ring) < 4:
            raise ValueError("perimeter_ring must contain at least four points")
        if self.perimeter_ring[0] != self.perimeter_ring[-1]:
            raise ValueError("perimeter_ring must be closed")
        return self


class TacticalIncidentContext(ContractModel):
    incident_id: NamespacedId
    target_cell_id: NamespacedId
    geometry: TacticalIncidentGeometry
    severity_class: TacticalSeverityClass
    severity_score: float = Field(ge=0, le=1)
    urgency_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    downstream_value_estimate: float | None = Field(default=None, ge=0)
    recommended_action: RecommendedAction
    activation_delay_seconds: int = Field(ge=0)
    summary: str | None = None


class TacticalRegionSelection(ContractModel):
    region_id: NamespacedId
    region_manifest_id: NamespacedId
    region_bundle_id: NamespacedId
    region_bundle_fingerprint: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    resolution_strategy: RegionResolutionStrategy
    matched_h3_cell: NonEmptyString | None = None
    candidate_region_bundle_ids: list[NamespacedId] = Field(default_factory=list)


class TacticalBridgeProvenance(ContractModel):
    bridge_kind: Literal["incident_packet_to_tactical_activation"] = (
        "incident_packet_to_tactical_activation"
    )
    source_packet_id: NamespacedId
    source_observation_time_utc: datetime
    source_downlink_time_utc: datetime
    bridge_version: NonEmptyString
    notes: list[NonEmptyString] = Field(default_factory=list)

    _validate_observation = field_validator("source_observation_time_utc")(_ensure_utc)
    _validate_downlink = field_validator("source_downlink_time_utc")(_ensure_utc)

    @model_validator(mode="after")
    def _validate_order(self) -> TacticalBridgeProvenance:
        if self.source_downlink_time_utc < self.source_observation_time_utc:
            raise ValueError(
                "source_downlink_time_utc must be greater than or equal to "
                "source_observation_time_utc"
            )
        return self


class TacticalDepotAssignment(ContractModel):
    unit_id: NamespacedId
    depot_facility_id: NamespacedId
    assignment_role: NonEmptyString
    queue_order: int = Field(default=0, ge=0)


class ScoutAsset(Phase1ContractModel):
    scout_asset_id: NamespacedId
    asset_name: NonEmptyString
    asset_type: ScoutAssetType
    status: ScoutAssetStatus
    home_facility_id: NamespacedId | None = None
    location: Wgs84GroundPoint
    endurance_minutes: int = Field(ge=1)
    sensor_focus: list[NonEmptyString] = Field(default_factory=list)


class TacticalOverlayEdgeEffect(ContractModel):
    edge_id: NamespacedId
    closed: bool = False
    cost_multiplier: float | None = Field(default=None, gt=0)
    speed_cap_kph: float | None = Field(default=None, gt=0)
    delay_seconds: float = Field(default=0, ge=0)
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_effect(self) -> TacticalOverlayEdgeEffect:
        if (
            not self.closed
            and self.cost_multiplier is None
            and self.speed_cap_kph is None
            and self.delay_seconds == 0
        ):
            raise ValueError(
                "overlay edge effects require a closure, cost multiplier, speed cap, or delay"
            )
        return self


class TacticalOverlayEvent(ContractModel):
    overlay_event_id: NamespacedId
    overlay_kind: OverlayEventKind
    title: NonEmptyString
    summary: str | None = None
    severity_score: float = Field(ge=0, le=1)
    window: TimeWindow | None = None
    edge_effects: list[TacticalOverlayEdgeEffect] = Field(default_factory=list)
    zone_ring: list[Wgs84Point] | None = None
    affected_asset_ids: list[NamespacedId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_event(self) -> TacticalOverlayEvent:
        if self.zone_ring is not None:
            if len(self.zone_ring) < 4:
                raise ValueError("zone_ring must contain at least four points")
            if self.zone_ring[0] != self.zone_ring[-1]:
                raise ValueError("zone_ring must be closed")
        if not self.edge_effects and self.zone_ring is None and not self.affected_asset_ids:
            raise ValueError(
                "overlay events require edge_effects, a zone_ring, or affected_asset_ids"
            )
        return self


class TacticalActivation(Phase1ContractModel):
    activation_id: NamespacedId
    incident_packet_id: NamespacedId
    region_bundle_id: NamespacedId
    activation_time_utc: datetime
    activation_reason: NonEmptyString
    requested_capabilities: list[NonEmptyString] = Field(min_length=1)
    activation_fingerprint: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    region_selection: TacticalRegionSelection
    incident_context: TacticalIncidentContext
    bridge_provenance: TacticalBridgeProvenance
    provenance_notes: list[NonEmptyString] = Field(default_factory=list)
    incident_packet: IncidentPacket

    _validate_activation_time = field_validator("activation_time_utc")(_ensure_utc)

    @model_validator(mode="after")
    def _validate_packet_reference(self) -> TacticalActivation:
        if self.incident_packet_id != self.incident_packet.packet_id:
            raise ValueError("incident_packet_id must match incident_packet.packet_id")
        if self.region_bundle_id != self.region_selection.region_bundle_id:
            raise ValueError("region_bundle_id must match region_selection.region_bundle_id")
        if self.incident_packet.incident_id != self.incident_context.incident_id:
            raise ValueError("incident_context.incident_id must match incident_packet.incident_id")
        if self.incident_packet.target_cell_id != self.incident_context.target_cell_id:
            raise ValueError(
                "incident_context.target_cell_id must match incident_packet.target_cell_id"
            )
        if self.incident_packet.recommended_action != self.incident_context.recommended_action:
            raise ValueError(
                "incident_context.recommended_action must match "
                "incident_packet.recommended_action"
            )
        if self.bridge_provenance.source_packet_id != self.incident_packet.packet_id:
            raise ValueError(
                "bridge_provenance.source_packet_id must match incident_packet.packet_id"
            )
        if self.activation_time_utc < self.incident_packet.downlink_time_utc:
            raise ValueError(
                "activation_time_utc must be greater than or equal to downlink_time_utc"
            )
        expected_delay = int(
            (self.activation_time_utc - self.incident_packet.downlink_time_utc).total_seconds()
        )
        if expected_delay != self.incident_context.activation_delay_seconds:
            raise ValueError(
                "incident_context.activation_delay_seconds must match activation_time_utc "
                "minus incident_packet.downlink_time_utc"
            )
        return self


class TacticalScenarioManifest(Phase1ContractModel):
    tactical_manifest_id: NamespacedId
    activation_id: NamespacedId
    activation_fingerprint: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    incident_packet_id: NamespacedId
    region_bundle_id: NamespacedId
    scenario_family: NonEmptyString
    simulation_seed: int
    decision_interval_seconds: int = Field(ge=1)
    time_window: TimeWindow
    incident_packet: IncidentPacket
    region_selection: TacticalRegionSelection
    incident_context: TacticalIncidentContext
    bridge_provenance: TacticalBridgeProvenance
    dispatch_units: list[DispatchUnit] = Field(min_length=1)
    facilities: list[Facility] = Field(min_length=1)
    depot_assignments: list[TacticalDepotAssignment] = Field(min_length=1)
    scout_assets: list[ScoutAsset] = Field(default_factory=list)
    overlay_events: list[TacticalOverlayEvent] = Field(default_factory=list)
    operational_objectives: list[NonEmptyString] = Field(min_length=1)
    config: TacticalScenarioConfig

    @model_validator(mode="after")
    def _validate_packet_reference(self) -> TacticalScenarioManifest:
        if self.incident_packet_id != self.incident_packet.packet_id:
            raise ValueError("incident_packet_id must match incident_packet.packet_id")
        if self.region_bundle_id != self.region_selection.region_bundle_id:
            raise ValueError("region_bundle_id must match region_selection.region_bundle_id")
        if self.incident_packet.incident_id != self.incident_context.incident_id:
            raise ValueError("incident_context.incident_id must match incident_packet.incident_id")
        if self.incident_packet.target_cell_id != self.incident_context.target_cell_id:
            raise ValueError(
                "incident_context.target_cell_id must match incident_packet.target_cell_id"
            )
        if self.bridge_provenance.source_packet_id != self.incident_packet.packet_id:
            raise ValueError(
                "bridge_provenance.source_packet_id must match incident_packet.packet_id"
            )
        facility_ids = {facility.facility_id for facility in self.facilities}
        unit_ids = {unit.unit_id for unit in self.dispatch_units}
        for unit in self.dispatch_units:
            if unit.home_facility_id not in facility_ids:
                raise ValueError(
                    f"dispatch unit {unit.unit_id} references unknown home facility "
                    f"{unit.home_facility_id}"
                )
            if (
                unit.current_facility_id is not None
                and unit.current_facility_id not in facility_ids
            ):
                raise ValueError(
                    f"dispatch unit {unit.unit_id} references unknown current facility "
                    f"{unit.current_facility_id}"
                )
        for assignment in self.depot_assignments:
            if assignment.unit_id not in unit_ids:
                raise ValueError(
                    f"depot assignment references unknown dispatch unit {assignment.unit_id}"
                )
            if assignment.depot_facility_id not in facility_ids:
                raise ValueError(
                    "depot assignment references unknown facility "
                    f"{assignment.depot_facility_id}"
                )
        for scout_asset in self.scout_assets:
            if (
                scout_asset.home_facility_id is not None
                and scout_asset.home_facility_id not in facility_ids
            ):
                raise ValueError(
                    f"scout asset {scout_asset.scout_asset_id} references unknown home facility "
                    f"{scout_asset.home_facility_id}"
                )
        return self


class TacticalScenarioBundle(Phase1ContractModel):
    tactical_bundle_id: NamespacedId
    bundle_fingerprint: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    tactical_manifest_id: NamespacedId
    activation_id: NamespacedId
    activation_fingerprint: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    incident_packet_id: NamespacedId
    region_bundle_id: NamespacedId
    scenario_family: NonEmptyString
    simulation_seed: int
    decision_interval_seconds: int = Field(ge=1)
    time_window: TimeWindow
    incident_packet: IncidentPacket
    region_selection: TacticalRegionSelection
    incident_context: TacticalIncidentContext
    bridge_provenance: TacticalBridgeProvenance
    dispatch_units: list[DispatchUnit] = Field(min_length=1)
    facilities: list[Facility] = Field(min_length=1)
    depot_assignments: list[TacticalDepotAssignment] = Field(min_length=1)
    scout_assets: list[ScoutAsset] = Field(default_factory=list)
    overlay_events: list[TacticalOverlayEvent] = Field(default_factory=list)
    operational_objectives: list[NonEmptyString] = Field(min_length=1)
    route_plans: list[RoutePlan]
    config: TacticalScenarioConfig
    compilation: BundleCompilation

    @model_validator(mode="after")
    def _validate_packet_reference(self) -> TacticalScenarioBundle:
        if self.incident_packet_id != self.incident_packet.packet_id:
            raise ValueError("incident_packet_id must match incident_packet.packet_id")
        if self.region_bundle_id != self.region_selection.region_bundle_id:
            raise ValueError("region_bundle_id must match region_selection.region_bundle_id")
        if self.incident_packet.incident_id != self.incident_context.incident_id:
            raise ValueError("incident_context.incident_id must match incident_packet.incident_id")
        if self.incident_packet.target_cell_id != self.incident_context.target_cell_id:
            raise ValueError(
                "incident_context.target_cell_id must match incident_packet.target_cell_id"
            )
        if self.bridge_provenance.source_packet_id != self.incident_packet.packet_id:
            raise ValueError(
                "bridge_provenance.source_packet_id must match incident_packet.packet_id"
            )
        facility_ids = {facility.facility_id for facility in self.facilities}
        unit_ids = {unit.unit_id for unit in self.dispatch_units}
        for assignment in self.depot_assignments:
            if assignment.unit_id not in unit_ids:
                raise ValueError(
                    f"depot assignment references unknown dispatch unit {assignment.unit_id}"
                )
            if assignment.depot_facility_id not in facility_ids:
                raise ValueError(
                    "depot assignment references unknown facility "
                    f"{assignment.depot_facility_id}"
                )
        for route_plan in self.route_plans:
            if route_plan.unit_id not in unit_ids:
                raise ValueError(
                    f"route plan {route_plan.route_plan_id} references unknown unit "
                    f"{route_plan.unit_id}"
                )
            if route_plan.origin_facility_id not in facility_ids:
                raise ValueError(
                    f"route plan {route_plan.route_plan_id} references unknown origin facility "
                    f"{route_plan.origin_facility_id}"
                )
            if route_plan.destination_facility_id not in facility_ids:
                raise ValueError(
                    "route plan "
                    f"{route_plan.route_plan_id} references unknown destination facility "
                    f"{route_plan.destination_facility_id}"
                )
        return self


class TacticalMetricsSummary(Phase1ContractModel):
    summary_id: NamespacedId
    tactical_bundle_id: NamespacedId
    episode_id: NamespacedId
    incident_packet_id: NamespacedId
    summary_fingerprint: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    dispatched_unit_count: int = Field(ge=0)
    arrived_unit_count: int = Field(ge=0)
    route_completion_ratio: float = Field(ge=0, le=1)
    containment_progress: float = Field(ge=0, le=1)
    mean_dispatch_latency_seconds: float = Field(ge=0)
    facility_utilization_peak: float = Field(ge=0, le=1)
    operational_score: float
    metric_components: dict[str, float] | None = None


class ScenarioBundle(Phase1ContractModel):
    bundle_id: NamespacedId
    bundle_fingerprint: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    manifest_id: NamespacedId
    benchmark_id: NonEmptyString
    scenario_family: NonEmptyString
    simulation_seed: int
    decision_interval_seconds: int = Field(ge=1)
    time_window: TimeWindow
    satellites: list[Satellite]
    ground_stations: list[GroundStation]
    target_cells: list[TargetCell]
    incidents: list[Incident]
    config: ScenarioConfig
    compilation: BundleCompilation
    observation_opportunities: list[ObservationOpportunity]
    downlink_windows: list[DownlinkWindow]


class ReplayPayloadBase(ContractModel):
    model_config = ConfigDict(extra="allow")


class ScenarioBundleLoadedPayload(ReplayPayloadBase):
    bundle_id: NamespacedId
    scenario_family: NonEmptyString
    manifest_id: NamespacedId | None = None


class EpisodeStartedPayload(ReplayPayloadBase):
    planner_id: NonEmptyString
    episode_fingerprint: NonEmptyString


class CandidateSetMaterializedPayload(ReplayPayloadBase):
    observation_opportunity_count: int = Field(ge=0)
    downlink_window_count: int = Field(ge=0)
    top_target_cell_id: NamespacedId | None = None


class ActionMaskEmittedPayload(ReplayPayloadBase):
    legal_action_count: int = Field(ge=0)
    mask_id: NamespacedId | None = None


class ActionSelectedPayload(ReplayPayloadBase):
    action_type: NonEmptyString
    action_ref: NonEmptyString
    score: float | None = None


class ObservationExecutedPayload(ReplayPayloadBase):
    opportunity_id: NamespacedId
    realized_quality: float = Field(ge=0, le=1)
    usable: bool
    cloud_fraction: float | None = Field(default=None, ge=0, le=1)


class DownlinkExecutedPayload(ReplayPayloadBase):
    window_id: NamespacedId
    delivered_volume_mb: float = Field(ge=0)


class IncidentPacketEmittedPayload(ReplayPayloadBase):
    packet_id: NamespacedId
    incident_id: NamespacedId


class RewardAssessedPayload(ReplayPayloadBase):
    total_reward: float
    components: dict[str, float] | None = None


class EpisodeEndedPayload(ReplayPayloadBase):
    terminated: bool
    truncated: bool
    mission_utility: float


_PAYLOAD_MODELS: dict[str, type[ContractModel]] = {
    "scenario_bundle_loaded": ScenarioBundleLoadedPayload,
    "episode_started": EpisodeStartedPayload,
    "candidate_set_materialized": CandidateSetMaterializedPayload,
    "action_mask_emitted": ActionMaskEmittedPayload,
    "action_selected": ActionSelectedPayload,
    "observation_executed": ObservationExecutedPayload,
    "downlink_executed": DownlinkExecutedPayload,
    "incident_packet_emitted": IncidentPacketEmittedPayload,
    "reward_assessed": RewardAssessedPayload,
    "episode_ended": EpisodeEndedPayload,
}


class ReplayEvent(Phase1ContractModel):
    event_id: NamespacedId
    episode_id: NamespacedId
    event_index: int = Field(ge=0)
    sim_tick: int = Field(ge=0)
    sim_time_utc: datetime
    event_type: ReplayEventType
    actor_type: ActorType
    actor_id: NonEmptyString
    payload: dict[str, Any]

    _validate_sim_time = field_validator("sim_time_utc")(_ensure_utc)

    @model_validator(mode="after")
    def _validate_payload(self) -> ReplayEvent:
        if self.event_type not in CANONICAL_REPLAY_EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {self.event_type}")
        payload_model = _PAYLOAD_MODELS[self.event_type]
        validated = payload_model.model_validate(self.payload)
        self.payload = validated.model_dump(mode="python", exclude_none=True)
        return self


class TacticalActivationCreatedPayload(ReplayPayloadBase):
    activation_id: NamespacedId
    incident_packet_id: NamespacedId
    incident_id: NamespacedId
    region_bundle_id: NamespacedId


class TacticalScenarioBundleLoadedPayload(ReplayPayloadBase):
    tactical_bundle_id: NamespacedId
    tactical_manifest_id: NamespacedId
    activation_id: NamespacedId
    scenario_family: NonEmptyString


class TacticalEpisodeStartedPayload(ReplayPayloadBase):
    planner_id: NonEmptyString
    episode_fingerprint: NonEmptyString


class TacticalCandidateSetMaterializedPayload(ReplayPayloadBase):
    candidate_route_plan_count: int = Field(ge=0)
    available_unit_count: int = Field(ge=0)
    blocked_unit_count: int = Field(ge=0)


class TacticalActionSelectedPayload(ReplayPayloadBase):
    action_type: NonEmptyString
    action_ref: NonEmptyString
    unit_id: NamespacedId | None = None
    score: float | None = None


class DispatchUnitAssignedPayload(ReplayPayloadBase):
    unit_id: NamespacedId
    facility_id: NamespacedId
    incident_id: NamespacedId
    route_plan_id: NamespacedId | None = None


class RoutePlanCommittedPayload(ReplayPayloadBase):
    route_plan_id: NamespacedId
    unit_id: NamespacedId
    destination_facility_id: NamespacedId
    estimated_duration_seconds: int = Field(ge=1)


class UnitPositionUpdatedPayload(ReplayPayloadBase):
    unit_id: NamespacedId
    status: DispatchUnitStatus
    location: Wgs84GroundPoint


class FacilityStatusUpdatedPayload(ReplayPayloadBase):
    facility_id: NamespacedId
    availability: Availability
    queued_unit_count: int = Field(ge=0)


class IncidentStateUpdatedPayload(ReplayPayloadBase):
    incident_id: NamespacedId
    state: TacticalIncidentState
    containment_progress: float = Field(ge=0, le=1)


class TacticalMetricsAssessedPayload(ReplayPayloadBase):
    operational_score: float
    components: dict[str, float] | None = None


class TacticalEpisodeEndedPayload(ReplayPayloadBase):
    terminated: bool
    truncated: bool
    containment_progress: float = Field(ge=0, le=1)
    dispatched_unit_count: int = Field(ge=0)


_TACTICAL_PAYLOAD_MODELS: dict[str, type[ContractModel]] = {
    "tactical_activation_created": TacticalActivationCreatedPayload,
    "tactical_scenario_bundle_loaded": TacticalScenarioBundleLoadedPayload,
    "tactical_episode_started": TacticalEpisodeStartedPayload,
    "tactical_candidate_set_materialized": TacticalCandidateSetMaterializedPayload,
    "tactical_action_selected": TacticalActionSelectedPayload,
    "dispatch_unit_assigned": DispatchUnitAssignedPayload,
    "route_plan_committed": RoutePlanCommittedPayload,
    "unit_position_updated": UnitPositionUpdatedPayload,
    "facility_status_updated": FacilityStatusUpdatedPayload,
    "incident_state_updated": IncidentStateUpdatedPayload,
    "tactical_metrics_assessed": TacticalMetricsAssessedPayload,
    "tactical_episode_ended": TacticalEpisodeEndedPayload,
}


class TacticalReplayEvent(Phase1ContractModel):
    event_id: NamespacedId
    episode_id: NamespacedId
    event_index: int = Field(ge=0)
    sim_tick: int = Field(ge=0)
    sim_time_utc: datetime
    event_type: TacticalReplayEventType
    actor_type: TacticalActorType
    actor_id: NonEmptyString
    payload: dict[str, Any]

    _validate_sim_time = field_validator("sim_time_utc")(_ensure_utc)

    @model_validator(mode="after")
    def _validate_payload(self) -> TacticalReplayEvent:
        if self.event_type not in CANONICAL_TACTICAL_REPLAY_EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {self.event_type}")
        payload_model = _TACTICAL_PAYLOAD_MODELS[self.event_type]
        validated = payload_model.model_validate(self.payload)
        self.payload = validated.model_dump(mode="python", exclude_none=True)
        return self
