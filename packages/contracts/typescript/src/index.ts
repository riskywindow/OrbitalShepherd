export const PHASE0_SCHEMA_VERSION = "0.1.0" as const;
export const PHASE1_SCHEMA_VERSION = "1.0.0" as const;
export const PHASE1_COMPILER_VERSION = "orbital-shepherd-contracts/1.0.0" as const;

export const LEGACY_REPLAY_EVENT_ALIASES = {
  scenario_loaded: "scenario_bundle_loaded",
  opportunities_materialized: "candidate_set_materialized",
  observation_committed: "observation_executed",
  downlink_committed: "downlink_executed",
} as const;

export type CanonicalReplayEventType =
  | "scenario_bundle_loaded"
  | "episode_started"
  | "candidate_set_materialized"
  | "action_mask_emitted"
  | "action_selected"
  | "observation_executed"
  | "downlink_executed"
  | "incident_packet_emitted"
  | "reward_assessed"
  | "episode_ended";

export type LegacyReplayEventType = keyof typeof LEGACY_REPLAY_EVENT_ALIASES;
export type ReplayEventType = CanonicalReplayEventType | LegacyReplayEventType;
export type ActorType = "system" | "planner" | "satellite" | "ground_station" | "metric_engine";
export type Availability = "nominal" | "degraded" | "offline";
export type PriorityClass = "low" | "medium" | "high" | "critical";
export type IncidentState = "candidate" | "active" | "contained" | "archived";
export type RecommendedAction = "monitor" | "dispatch_recon" | "dispatch_ground";
export type SpatialLayerType = "roads" | "facilities" | "hazard_perimeter" | "terrain" | "weather";
export type DispatchUnitType =
  | "engine"
  | "crew"
  | "dozer"
  | "air_tanker"
  | "helicopter"
  | "command";
export type DispatchUnitStatus =
  | "available"
  | "assigned"
  | "enroute"
  | "on_scene"
  | "restaging"
  | "out_of_service";
export type FacilityType =
  | "station"
  | "helibase"
  | "drop_point"
  | "hospital"
  | "staging_area"
  | "command_post";
export type TravelMode = "road" | "air";
export type RoutePlanStatus = "planned" | "active" | "completed" | "aborted";
export type TacticalIncidentState =
  | "reported"
  | "mobilizing"
  | "engaged"
  | "contained"
  | "demobilized";
export type TacticalActorType =
  | "system"
  | "planner"
  | "dispatch_unit"
  | "facility"
  | "metric_engine"
  | "bridge";
export type CanonicalTacticalReplayEventType =
  | "tactical_activation_created"
  | "tactical_scenario_bundle_loaded"
  | "tactical_episode_started"
  | "tactical_candidate_set_materialized"
  | "tactical_action_selected"
  | "dispatch_unit_assigned"
  | "route_plan_committed"
  | "unit_position_updated"
  | "facility_status_updated"
  | "incident_state_updated"
  | "tactical_metrics_assessed"
  | "tactical_episode_ended";

export interface Wgs84Point {
  lat: number;
  lon: number;
}

export interface Wgs84GroundPoint extends Wgs84Point {
  alt_m: number;
}

export interface Wgs84BoundingBox {
  min_lat: number;
  min_lon: number;
  max_lat: number;
  max_lon: number;
}

export interface TimeWindow {
  start_time_utc: string;
  end_time_utc: string;
}

export interface OpportunityGenerationConfig {
  quality_threshold: number;
  cloud_block_threshold: number;
}

export interface ScenarioConfig {
  horizon_hours: number;
  notes: string;
  weather_model: string;
  opportunity_generation: OpportunityGenerationConfig;
  [key: string]: unknown;
}

export interface TacticalScenarioConfig {
  planning_horizon_minutes: number;
  max_active_routes: number;
  reroute_on_blockage: boolean;
  notes: string;
  [key: string]: unknown;
}

export interface SatelliteSensor {
  sensor_id: string;
  swath_km: number;
  quality_nominal: number;
  max_off_nadir_deg?: number;
  estimated_data_volume_mb?: number;
}

export interface SatelliteDownlink {
  buffer_capacity_mb: number;
  nominal_downlink_rate_mbps: number;
}

export interface SatelliteConstraints {
  max_retargets_per_orbit: number;
  availability?: Availability;
}

export interface Satellite {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  satellite_id: string;
  name: string;
  norad_catalog_id: number;
  sensor: SatelliteSensor;
  downlink: SatelliteDownlink;
  constraints: SatelliteConstraints;
}

export interface GroundStationCapabilities {
  max_concurrent_contacts: number;
  downlink_rate_mbps: number;
  availability: Availability;
}

export interface GroundStation {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  station_id: string;
  name: string;
  location: Wgs84GroundPoint;
  capabilities: GroundStationCapabilities;
}

export interface TargetCell {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  target_cell_id: string;
  h3_cell: string;
  centroid: Wgs84Point;
  region_name?: string;
  static_value: number;
  priority_class?: PriorityClass;
}

export interface Incident {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  incident_id: string;
  incident_type: "wildfire";
  target_cell_id: string;
  ignition_time_utc: string;
  urgency_score: number;
  confidence: number;
  state: IncidentState;
  estimated_area_ha?: number;
}

export interface ObservationOpportunity {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  opportunity_id: string;
  satellite_id: string;
  target_cell_id: string;
  start_time_utc: string;
  end_time_utc: string;
  predicted_quality_mean: number;
  predicted_cloud_obstruction_prob: number;
  estimated_data_volume_mb: number;
  slew_cost: number;
  incident_ids?: string[];
}

export interface DownlinkWindow {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  window_id: string;
  satellite_id: string;
  station_id: string;
  start_time_utc: string;
  end_time_utc: string;
  max_volume_mb: number;
  expected_rate_mbps: number;
  outage_risk: number;
}

export interface IncidentPacket {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  packet_id: string;
  incident_id: string;
  target_cell_id: string;
  observation_time_utc: string;
  downlink_time_utc: string;
  confidence: number;
  urgency_score: number;
  recommended_action: RecommendedAction;
  observation_opportunity_id?: string;
  downstream_value_estimate?: number;
  summary?: string;
}

export interface ScenarioManifest {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  manifest_id: string;
  benchmark_id: string;
  scenario_family: string;
  simulation_seed: number;
  decision_interval_seconds: number;
  time_window: TimeWindow;
  satellites: Satellite[];
  ground_stations: GroundStation[];
  target_cells: TargetCell[];
  incidents: Incident[];
  config: ScenarioConfig;
}

export interface BundleCompilation {
  source_manifest_id: string;
  source_manifest_schema_version: string;
  source_manifest_sha256: string;
  compiled_at_utc: string;
  compiler_version: string;
}

export interface SpatialIngestManifest {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  spatial_ingest_id: string;
  region_id: string;
  source_name: string;
  layer_type: SpatialLayerType;
  source_uri: string;
  source_fingerprint: string;
  transform_fingerprint: string;
  ingest_time_utc: string;
  coverage_bounds: Wgs84BoundingBox;
  feature_count: number;
  notes?: string;
}

export interface Facility {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  facility_id: string;
  facility_name: string;
  facility_type: FacilityType;
  location: Wgs84GroundPoint;
  availability: Availability;
  capacity_units: number;
  supported_unit_types: DispatchUnitType[];
}

export interface DispatchUnit {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  unit_id: string;
  callsign: string;
  unit_type: DispatchUnitType;
  status: DispatchUnitStatus;
  home_facility_id: string;
  current_facility_id?: string;
  travel_mode: TravelMode;
  personnel_count: number;
  equipment_capacity: number;
  location: Wgs84GroundPoint;
}

export interface RoutePlan {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  route_plan_id: string;
  unit_id: string;
  origin_facility_id: string;
  destination_facility_id: string;
  travel_mode: TravelMode;
  status: RoutePlanStatus;
  distance_km: number;
  estimated_duration_seconds: number;
  risk_score: number;
  waypoints: Wgs84GroundPoint[];
  road_segment_ids?: string[];
}

export type RoadSourceKind = "fixture_geojson" | "osmnx";
export type RegionGeometryType = "point" | "polygon";
export type H3CoverStrategy = "explicit" | "bounds";

export interface RoadNetworkSource {
  ingest_id: string;
  source_kind: RoadSourceKind;
  source_name: string;
  source_uri: string;
  fallback_priority?: number;
  network_type?: string;
  custom_filter?: string;
  notes?: string;
}

export interface TravelTimeDefaults {
  default_speed_kph: number;
  speed_kph_by_highway: Record<string, number>;
  intersection_penalty_seconds: number;
}

export interface RegionAsset {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  asset_id: string;
  asset_name: string;
  asset_kind: string;
  geometry_type: RegionGeometryType;
  point?: Wgs84Point;
  ring?: Wgs84Point[];
  priority_class?: PriorityClass;
  tags?: Record<string, string>;
}

export interface RoadNode {
  node_id: string;
  location: Wgs84Point;
}

export interface RoadEdge {
  edge_id: string;
  source_node_id: string;
  target_node_id: string;
  source_ingest_id: string;
  road_class: string;
  distance_m: number;
  speed_kph: number;
  travel_time_seconds: number;
  oneway: boolean;
  geometry: Wgs84Point[];
  road_name?: string;
  source_edge_ref?: string;
}

export interface H3CoverConfig {
  resolution: number;
  strategy: H3CoverStrategy;
  explicit_cell_ids?: string[];
}

export interface H3Cover {
  resolution: number;
  generation_strategy: H3CoverStrategy;
  cell_ids: string[];
  cell_count: number;
}

export interface RegionRoadNetworkConfig {
  sources: RoadNetworkSource[];
  travel_time_defaults: TravelTimeDefaults;
}

export interface RegionManifest {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  region_manifest_id: string;
  region_id: string;
  region_name: string;
  bounds: Wgs84BoundingBox;
  default_cell_size_m: number;
  spatial_ingest_ids: string[];
  build_seed?: number;
  road_network: RegionRoadNetworkConfig;
  facilities?: Facility[];
  asset_features?: RegionAsset[];
  h3_cover: H3CoverConfig;
  provenance_notes?: string[];
  notes?: string;
}

export interface RegionBundle {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  region_bundle_id: string;
  region_manifest_id: string;
  region_id: string;
  region_name: string;
  bundle_fingerprint: string;
  bounds: Wgs84BoundingBox;
  spatial_ingests: SpatialIngestManifest[];
  travel_time_defaults: TravelTimeDefaults;
  road_nodes: RoadNode[];
  road_edges: RoadEdge[];
  facilities: Facility[];
  asset_features: RegionAsset[];
  h3_cover: H3Cover;
  provenance_notes?: string[];
  traversable_node_count: number;
  traversable_edge_count: number;
  compilation: BundleCompilation;
}

export interface TacticalActivation {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  activation_id: string;
  incident_packet_id: string;
  region_bundle_id: string;
  activation_time_utc: string;
  activation_reason: string;
  requested_capabilities: string[];
  activation_fingerprint: string;
  incident_packet: IncidentPacket;
}

export interface TacticalScenarioManifest {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  tactical_manifest_id: string;
  activation_id: string;
  incident_packet_id: string;
  region_bundle_id: string;
  scenario_family: string;
  simulation_seed: number;
  decision_interval_seconds: number;
  time_window: TimeWindow;
  incident_packet: IncidentPacket;
  dispatch_units: DispatchUnit[];
  facilities: Facility[];
  operational_objectives: string[];
  config: TacticalScenarioConfig;
}

export interface TacticalScenarioBundle {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  tactical_bundle_id: string;
  bundle_fingerprint: string;
  tactical_manifest_id: string;
  activation_id: string;
  incident_packet_id: string;
  region_bundle_id: string;
  scenario_family: string;
  simulation_seed: number;
  decision_interval_seconds: number;
  time_window: TimeWindow;
  incident_packet: IncidentPacket;
  dispatch_units: DispatchUnit[];
  facilities: Facility[];
  route_plans: RoutePlan[];
  config: TacticalScenarioConfig;
  compilation: BundleCompilation;
}

export interface ScenarioBundle {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  bundle_id: string;
  bundle_fingerprint: string;
  manifest_id: string;
  benchmark_id: string;
  scenario_family: string;
  simulation_seed: number;
  decision_interval_seconds: number;
  time_window: TimeWindow;
  satellites: Satellite[];
  ground_stations: GroundStation[];
  target_cells: TargetCell[];
  incidents: Incident[];
  config: ScenarioConfig;
  compilation: BundleCompilation;
  observation_opportunities: ObservationOpportunity[];
  downlink_windows: DownlinkWindow[];
}

export interface ReplayEvent<TPayload extends Record<string, unknown> = Record<string, unknown>> {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  event_id: string;
  episode_id: string;
  event_index: number;
  sim_tick: number;
  sim_time_utc: string;
  event_type: CanonicalReplayEventType;
  actor_type: ActorType;
  actor_id: string;
  payload: TPayload;
}

export interface TacticalReplayEvent<TPayload extends Record<string, unknown> = Record<string, unknown>> {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  event_id: string;
  episode_id: string;
  event_index: number;
  sim_tick: number;
  sim_time_utc: string;
  event_type: CanonicalTacticalReplayEventType;
  actor_type: TacticalActorType;
  actor_id: string;
  payload: TPayload;
}

export interface TacticalMetricsSummary {
  schema_version: typeof PHASE1_SCHEMA_VERSION;
  summary_id: string;
  tactical_bundle_id: string;
  episode_id: string;
  incident_packet_id: string;
  summary_fingerprint: string;
  dispatched_unit_count: number;
  arrived_unit_count: number;
  route_completion_ratio: number;
  containment_progress: number;
  mean_dispatch_latency_seconds: number;
  facility_utilization_peak: number;
  operational_score: number;
  metric_components?: Record<string, number>;
}

export function manifestIdFromBundleId(bundleId: string): string {
  return bundleId.startsWith("sb:") ? `sm:${bundleId.slice(3)}` : `sm:${bundleId}`;
}

export function bundleIdFromManifestId(manifestId: string): string {
  return manifestId.startsWith("sm:") ? `sb:${manifestId.slice(3)}` : `sb:${manifestId}`;
}

export function toCanonicalManifest(
  manifestLike: Omit<ScenarioManifest, "schema_version" | "manifest_id"> &
    Partial<Pick<ScenarioManifest, "schema_version" | "manifest_id">> & {
      bundle_id?: string;
    },
): ScenarioManifest {
  const manifestId = manifestLike.manifest_id ?? manifestIdFromBundleId(manifestLike.bundle_id ?? "legacy");
  return {
    ...manifestLike,
    schema_version: PHASE1_SCHEMA_VERSION,
    manifest_id: manifestId,
  };
}

export function compileScenarioBundle(
  manifestLike: ScenarioManifest,
  options?: {
    bundle_id?: string;
    bundle_fingerprint?: string;
    compiled_at_utc?: string;
    compiler_version?: string;
    source_manifest_sha256?: string;
    observation_opportunities?: ObservationOpportunity[];
    downlink_windows?: DownlinkWindow[];
  },
): ScenarioBundle {
  return {
    ...manifestLike,
    schema_version: PHASE1_SCHEMA_VERSION,
    bundle_id: options?.bundle_id ?? bundleIdFromManifestId(manifestLike.manifest_id),
    bundle_fingerprint: options?.bundle_fingerprint ?? "sha256-pending",
    compilation: {
      source_manifest_id: manifestLike.manifest_id,
      source_manifest_schema_version: manifestLike.schema_version,
      source_manifest_sha256: options?.source_manifest_sha256 ?? "sha256-pending",
      compiled_at_utc: options?.compiled_at_utc ?? new Date().toISOString(),
      compiler_version: options?.compiler_version ?? PHASE1_COMPILER_VERSION,
    },
    observation_opportunities: options?.observation_opportunities ?? [],
    downlink_windows: options?.downlink_windows ?? [],
  };
}

export function normalizeReplayEvent<TPayload extends Record<string, unknown>>(
  event: Omit<ReplayEvent<TPayload>, "schema_version" | "event_type" | "payload"> & {
    schema_version?: string;
    event_type: ReplayEventType;
    payload: TPayload & {
      observation_count?: number;
      downlink_count?: number;
      observation_opportunity_count?: number;
      downlink_window_count?: number;
      bundle_id?: string;
      manifest_id?: string;
    };
  },
): ReplayEvent<Record<string, unknown>> {
  const canonicalEventType =
    LEGACY_REPLAY_EVENT_ALIASES[event.event_type as LegacyReplayEventType] ?? event.event_type;
  const payload: Record<string, unknown> = { ...event.payload };

  if (
    canonicalEventType === "candidate_set_materialized" &&
    payload.observation_count !== undefined &&
    payload.observation_opportunity_count === undefined
  ) {
    payload.observation_opportunity_count = payload.observation_count;
    delete payload.observation_count;
  }
  if (
    canonicalEventType === "candidate_set_materialized" &&
    payload.downlink_count !== undefined &&
    payload.downlink_window_count === undefined
  ) {
    payload.downlink_window_count = payload.downlink_count;
    delete payload.downlink_count;
  }
  if (
    canonicalEventType === "scenario_bundle_loaded" &&
    typeof payload.bundle_id === "string" &&
    payload.manifest_id === undefined
  ) {
    payload.manifest_id = manifestIdFromBundleId(payload.bundle_id);
  }

  return {
    ...event,
    schema_version: PHASE1_SCHEMA_VERSION,
    event_type: canonicalEventType,
    payload,
  };
}
