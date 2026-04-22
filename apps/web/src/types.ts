export interface Wgs84Point {
  lat: number;
  lon: number;
}

export interface Wgs84GroundPoint extends Wgs84Point {
  alt_m: number;
}

export interface TimeWindow {
  start_time_utc: string;
  end_time_utc: string;
}

export interface ScenarioSummary {
  bundle_id: string;
  benchmark_id: string;
  scenario_family: string;
  simulation_seed: number;
  decision_interval_seconds: number;
  start_time_utc: string;
  end_time_utc: string;
  satellite_count: number;
  ground_station_count: number;
  target_cell_count: number;
  incident_count: number;
  observation_opportunity_count: number;
  downlink_window_count: number;
}

export interface ScenarioPreview {
  bundle_id: string;
  benchmark_id: string;
  scenario_family: string;
  simulation_seed: number;
  time_window: TimeWindow;
  counts: Record<string, number>;
  target_cells: Array<{
    target_cell_id: string;
    h3_cell: string;
    region_name?: string;
    static_value: number;
    priority_class?: string;
    centroid: Wgs84Point;
  }>;
  incidents: Array<{
    incident_id: string;
    target_cell_id: string;
    urgency_score: number;
    confidence: number;
    state: string;
    ignition_time_utc: string;
  }>;
}

export interface Satellite {
  satellite_id: string;
  name: string;
  norad_catalog_id: number;
  sensor: {
    sensor_id: string;
    swath_km: number;
    quality_nominal: number;
    max_off_nadir_deg?: number;
    estimated_data_volume_mb?: number;
  };
  downlink: {
    buffer_capacity_mb: number;
    nominal_downlink_rate_mbps: number;
  };
  constraints: {
    max_retargets_per_orbit: number;
    availability?: string;
  };
}

export interface GroundStation {
  station_id: string;
  name: string;
  location: Wgs84GroundPoint;
  capabilities: {
    max_concurrent_contacts: number;
    downlink_rate_mbps: number;
    availability: string;
  };
}

export interface TargetCell {
  target_cell_id: string;
  h3_cell: string;
  centroid: Wgs84Point;
  region_name?: string;
  static_value: number;
  priority_class?: string;
}

export interface Incident {
  incident_id: string;
  incident_type: string;
  target_cell_id: string;
  ignition_time_utc: string;
  urgency_score: number;
  confidence: number;
  state: string;
  estimated_area_ha?: number;
}

export interface ObservationOpportunity {
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
  window_id: string;
  satellite_id: string;
  station_id: string;
  start_time_utc: string;
  end_time_utc: string;
  max_volume_mb: number;
  expected_rate_mbps: number;
  outage_risk: number;
}

export interface ScenarioBundle {
  bundle_id: string;
  benchmark_id: string;
  scenario_family: string;
  simulation_seed: number;
  decision_interval_seconds: number;
  time_window: TimeWindow;
  satellites: Satellite[];
  ground_stations: GroundStation[];
  target_cells: TargetCell[];
  incidents: Incident[];
  observation_opportunities: ObservationOpportunity[];
  downlink_windows: DownlinkWindow[];
  config: Record<string, unknown> & {
    family_display_name?: string;
    notes?: string;
    weather_model?: string;
    orbit_asset_bundle_id?: string;
  };
}

export interface BaselineDescriptor {
  baseline_id: string;
  version: string;
  description: string;
}

export interface ModelDescriptor {
  model_key: string;
  planner_id: string;
  checkpoint_id: string;
  checkpoint_fingerprint: string;
  model_id: string;
  algorithm: string;
  created_at_utc: string;
  description: string;
}

export interface DemoDefaults {
  bundle_id: string | null;
  baseline_id: string | null;
  episode_id: string | null;
  benchmark_run_id?: string | null;
  benchmark_summary_path?: string | null;
  replay_path?: string | null;
  ui_query: string;
}

export interface DistributionSummary {
  count: number;
  mean: number | null;
  median: number | null;
  p90: number | null;
  values: number[];
}

export interface EpisodeMetrics {
  time_to_first_useful_observation_seconds: DistributionSummary;
  useful_observation_value_captured: number;
  cloud_waste_rate: number;
  downlink_latency_seconds: DistributionSummary;
  missed_urgent_incident_rate: number;
  opportunity_utilization_efficiency: number;
  mission_utility: number;
  observation_commit_count: number;
  useful_packet_count: number;
  urgent_incident_count: number;
  missed_urgent_incident_count: number;
}

export interface ActionMaskAction {
  action_id: string;
  action_type: string;
  action_ref: string;
  score_hint?: number;
  satellite_id?: string;
  target_cell_id?: string;
}

export interface ActionMask {
  mask_id: string;
  legal_action_count: number;
  actions: ActionMaskAction[];
}

export interface EpisodeSummary {
  episode_id: string;
  runtime_episode_id: string;
  bundle_id: string;
  planner_id: string;
  display_name?: string | null;
  planner_kind?: string | null;
  planner_metadata?: Record<string, unknown> | null;
  episode_source?: "api" | "report";
  report_id?: string | null;
  report_episode_id?: string | null;
  report_split?: string | null;
  simulation_seed: number;
  created_at_utc: string;
  updated_at_utc: string;
  sim_tick: number;
  terminated: boolean;
  truncated: boolean;
  mission_utility: number;
  replay_event_count: number;
}

export interface EpisodeDetail extends EpisodeSummary {
  action_history_length: number;
  latest_observation: {
    sim_tick: number;
    sim_time_utc: string;
    horizon_tick: number;
    mission_utility: number;
    incidents: Array<{
      incident_id: string;
      target_cell_id: string;
      urgency_score: number;
      status: string;
      observed_time_utc?: string | null;
      downlinked_time_utc?: string | null;
      missed_time_utc?: string | null;
      last_opportunity_id?: string | null;
      last_packet_id?: string | null;
    }>;
    action_mask: ActionMask;
  };
  action_mask: ActionMask;
  metrics: EpisodeMetrics;
}

export interface BaselineRunResponse {
  job_id: string;
  status: "accepted" | "completed";
  episode_id: string | null;
}

export interface BaselineRunDetail {
  job_id: string;
  baseline_id: string;
  bundle_id: string;
  simulation_seed: number;
  status: "accepted" | "completed" | "failed";
  created_at_utc: string;
  completed_at_utc?: string | null;
  episode_id?: string | null;
  runtime_episode_id?: string | null;
  replay_path?: string | null;
  metrics?: EpisodeMetrics | null;
  error_message?: string | null;
}

export interface ModelRunResponse {
  job_id: string;
  status: "accepted" | "completed";
  episode_id: string | null;
}

export interface ModelRunDetail {
  job_id: string;
  model_key: string;
  bundle_id: string;
  simulation_seed: number;
  status: "accepted" | "completed" | "failed";
  created_at_utc: string;
  completed_at_utc?: string | null;
  episode_id?: string | null;
  runtime_episode_id?: string | null;
  replay_path?: string | null;
  metrics?: EpisodeMetrics | null;
  error_message?: string | null;
}

export interface ReportSummary {
  report_id: string;
  report_kind: "evaluation";
  title: string;
  summary_path: string;
  benchmark_id?: string | null;
  episode_count: number;
  planner_count: number;
  trained_policy_count: number;
  baseline_count: number;
  splits: string[];
  notable_episode_count: number;
}

export interface ReportNotableEpisode {
  category: string;
  planner_key?: string | null;
  bundle_id: string;
  split?: string | null;
  scenario_family: string;
  primary_metric_value?: number | null;
  difference_vs_best_baseline?: number | null;
}

export interface ReportEpisodeSummary {
  report_id: string;
  report_episode_id: string;
  episode_id: string;
  bundle_id: string;
  split?: string | null;
  scenario_family: string;
  planner_key: string;
  planner_kind?: string | null;
  planner_version?: string | null;
  display_name?: string | null;
  simulation_seed: number;
  action_count: number;
  metrics: EpisodeMetrics;
  replay_path: string;
  summary_path: string;
  scenario_path: string;
  reward_audit?: Record<string, number> | null;
  bundle_profile?: Record<string, number | string | string[]> | null;
}

export interface ReportDetail extends ReportSummary {
  episodes: ReportEpisodeSummary[];
  notable_episodes: ReportNotableEpisode[];
}

export interface SlotMapping {
  action_id?: string | null;
  action_ref?: string | null;
  action_type?: string | null;
  projected_rank?: number | null;
  runtime_action_index?: number | null;
  slot_index?: number | null;
  source?: string | null;
  satellite_id?: string | null;
  target_cell_id?: string | null;
}

export interface PolicyTraceSlot {
  slot_index: number;
  action_id?: string | null;
  logit?: number | null;
  probability?: number | null;
  slot_mapping?: SlotMapping | null;
}

export interface PolicyTrace {
  selected_action_id?: string | null;
  selected_logit?: number | null;
  selected_probability?: number | null;
  selected_slot?: number | null;
  selected_slot_mapping?: SlotMapping | null;
  action_entropy?: number | null;
  value_estimate?: number | null;
  legal_action_count?: number | null;
  mask_pressure?: number | null;
  top_slots?: PolicyTraceSlot[] | null;
}

export interface EpisodeInferenceTraceStep {
  event_index: number;
  sim_tick: number;
  sim_time_utc: string;
  action_id?: string | null;
  action_type: string;
  action_ref: string;
  planner_trace: Record<string, unknown> & {
    policy_trace?: PolicyTrace;
  };
}

export interface EpisodeInferenceTraceResponse {
  episode_id: string;
  planner_id: string;
  planner_kind?: string | null;
  step_count: number;
  traces: EpisodeInferenceTraceStep[];
}

export interface ReplayEvent<TPayload extends Record<string, unknown> = Record<string, unknown>> {
  event_id: string;
  episode_id: string;
  event_index: number;
  sim_tick: number;
  sim_time_utc: string;
  event_type: string;
  actor_type: string;
  actor_id: string;
  payload: TPayload;
}

export interface ApiErrorPayload {
  error: string;
  message: string;
  details?: Record<string, unknown> | null;
}

export type CzmlPacket = Record<string, unknown>;
