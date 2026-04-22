from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orbital_shepherd_contracts import ScenarioBundle


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: Literal["ok"]


class DemoDefaultsResponse(ApiModel):
    bundle_id: str | None = None
    baseline_id: str | None = None
    episode_id: str | None = None
    benchmark_run_id: str | None = None
    benchmark_summary_path: str | None = None
    replay_path: str | None = None
    ui_query: str = ""


class ErrorResponse(ApiModel):
    error: str
    message: str
    details: dict[str, Any] | None = None


class ScenarioRegistrationResponse(ApiModel):
    bundle_id: str


class ScenarioSummary(ApiModel):
    bundle_id: str
    benchmark_id: str
    scenario_family: str
    simulation_seed: int
    decision_interval_seconds: int
    start_time_utc: datetime
    end_time_utc: datetime
    satellite_count: int
    ground_station_count: int
    target_cell_count: int
    incident_count: int
    observation_opportunity_count: int
    downlink_window_count: int


class ScenarioPreview(ApiModel):
    bundle_id: str
    benchmark_id: str
    scenario_family: str
    simulation_seed: int
    time_window: dict[str, Any]
    counts: dict[str, int]
    target_cells: list[dict[str, Any]]
    incidents: list[dict[str, Any]]


class StartEpisodeRequest(ApiModel):
    bundle_id: str
    planner_id: str
    simulation_seed: int


class StartEpisodeResponse(ApiModel):
    episode_id: str
    runtime_episode_id: str


class StepEpisodeRequest(ApiModel):
    action: dict[str, Any]


class StepEpisodeResponse(ApiModel):
    episode_id: str
    runtime_episode_id: str
    sim_tick: int
    terminated: bool
    truncated: bool
    reward: float
    mission_utility: float
    observation: dict[str, Any]
    action_mask: dict[str, Any]
    events: list[dict[str, Any]]


class DistributionSummaryResponse(ApiModel):
    count: int
    mean: float | None
    median: float | None
    p90: float | None
    values: list[float]


class EpisodeMetricsResponse(ApiModel):
    time_to_first_useful_observation_seconds: DistributionSummaryResponse
    useful_observation_value_captured: float
    cloud_waste_rate: float
    downlink_latency_seconds: DistributionSummaryResponse
    missed_urgent_incident_rate: float
    opportunity_utilization_efficiency: float
    mission_utility: float
    observation_commit_count: int
    useful_packet_count: int
    urgent_incident_count: int
    missed_urgent_incident_count: int


class EpisodeSummary(ApiModel):
    episode_id: str
    runtime_episode_id: str
    bundle_id: str
    planner_id: str
    display_name: str | None = None
    planner_kind: str | None = None
    planner_metadata: dict[str, Any] | None = None
    episode_source: Literal["api", "report"] = "api"
    report_id: str | None = None
    report_episode_id: str | None = None
    report_split: str | None = None
    simulation_seed: int
    created_at_utc: datetime
    updated_at_utc: datetime
    sim_tick: int
    terminated: bool
    truncated: bool
    mission_utility: float
    replay_event_count: int


class EpisodeDetail(EpisodeSummary):
    action_history_length: int
    latest_observation: dict[str, Any]
    action_mask: dict[str, Any]
    metrics: EpisodeMetricsResponse


class EpisodeRecord(EpisodeSummary):
    action_history: list[Any] = Field(default_factory=list)
    replay_path: str
    latest_observation: dict[str, Any]
    latest_action_mask: dict[str, Any]
    metrics: EpisodeMetricsResponse


class BaselineDescriptor(ApiModel):
    baseline_id: str
    version: str
    description: str


class BaselineRunRequest(ApiModel):
    bundle_id: str
    simulation_seed: int


class BaselineRunResponse(ApiModel):
    job_id: str
    status: Literal["accepted", "completed"]
    episode_id: str | None = None


class BaselineRunDetail(ApiModel):
    job_id: str
    baseline_id: str
    bundle_id: str
    simulation_seed: int
    status: Literal["accepted", "completed", "failed"]
    created_at_utc: datetime
    completed_at_utc: datetime | None = None
    episode_id: str | None = None
    runtime_episode_id: str | None = None
    replay_path: str | None = None
    metrics: EpisodeMetricsResponse | None = None
    error_message: str | None = None


class ModelDescriptor(ApiModel):
    model_key: str
    planner_id: str
    checkpoint_id: str
    checkpoint_fingerprint: str
    model_id: str
    algorithm: str
    created_at_utc: datetime
    description: str


class ModelDetail(ModelDescriptor):
    benchmark_id: str
    reward_id: str
    run_id: str
    trainer_backend: str
    framework: str
    checkpoint_manifest_path: str
    checkpoint_path: str
    architecture: dict[str, Any]
    metadata: dict[str, Any]
    metrics: dict[str, float]


class ModelRunRequest(ApiModel):
    bundle_id: str
    simulation_seed: int
    include_inference_traces: bool = True


class ModelRunResponse(ApiModel):
    job_id: str
    status: Literal["accepted", "completed"]
    episode_id: str | None = None


class ModelRunDetail(ApiModel):
    job_id: str
    model_key: str
    bundle_id: str
    simulation_seed: int
    status: Literal["accepted", "completed", "failed"]
    created_at_utc: datetime
    completed_at_utc: datetime | None = None
    episode_id: str | None = None
    runtime_episode_id: str | None = None
    replay_path: str | None = None
    metrics: EpisodeMetricsResponse | None = None
    error_message: str | None = None


class ReportSummary(ApiModel):
    report_id: str
    report_kind: Literal["evaluation"]
    title: str
    summary_path: str
    benchmark_id: str | None = None
    episode_count: int
    planner_count: int
    trained_policy_count: int
    baseline_count: int
    splits: list[str]
    notable_episode_count: int


class ReportNotableEpisode(ApiModel):
    category: str
    planner_key: str | None = None
    bundle_id: str
    split: str | None = None
    scenario_family: str
    primary_metric_value: float | None = None
    difference_vs_best_baseline: float | None = None


class ReportEpisodeSummary(ApiModel):
    report_id: str
    report_episode_id: str
    episode_id: str
    bundle_id: str
    split: str | None = None
    scenario_family: str
    planner_key: str
    planner_kind: str | None = None
    planner_version: str | None = None
    display_name: str | None = None
    simulation_seed: int
    action_count: int
    metrics: EpisodeMetricsResponse
    replay_path: str
    summary_path: str
    scenario_path: str
    reward_audit: dict[str, Any] | None = None
    bundle_profile: dict[str, Any] | None = None


class ReportDetail(ReportSummary):
    episodes: list[ReportEpisodeSummary]
    notable_episodes: list[ReportNotableEpisode]


class EpisodeInferenceTraceStep(ApiModel):
    event_index: int
    sim_tick: int
    sim_time_utc: datetime
    action_id: str | None = None
    action_type: str
    action_ref: str
    planner_trace: dict[str, Any]


class EpisodeInferenceTraceResponse(ApiModel):
    episode_id: str
    planner_id: str
    planner_kind: str | None = None
    step_count: int
    traces: list[EpisodeInferenceTraceStep]


def scenario_summary_from_bundle(bundle: ScenarioBundle) -> ScenarioSummary:
    return ScenarioSummary(
        bundle_id=bundle.bundle_id,
        benchmark_id=bundle.benchmark_id,
        scenario_family=bundle.scenario_family,
        simulation_seed=bundle.simulation_seed,
        decision_interval_seconds=bundle.decision_interval_seconds,
        start_time_utc=bundle.time_window.start_time_utc,
        end_time_utc=bundle.time_window.end_time_utc,
        satellite_count=len(bundle.satellites),
        ground_station_count=len(bundle.ground_stations),
        target_cell_count=len(bundle.target_cells),
        incident_count=len(bundle.incidents),
        observation_opportunity_count=len(bundle.observation_opportunities),
        downlink_window_count=len(bundle.downlink_windows),
    )
