import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import App from "./App";
import * as api from "./api";
import type {
  BaselineDescriptor,
  DemoDefaults,
  EpisodeDetail,
  EpisodeInferenceTraceResponse,
  EpisodeSummary,
  ModelDescriptor,
  ReplayEvent,
  ReportDetail,
  ReportSummary,
  ScenarioBundle,
  ScenarioPreview,
  ScenarioSummary,
} from "./types";

vi.mock("./GlobeView", () => ({
  GlobeView: () => <div data-testid="mock-globe">mock globe</div>,
}));

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    getBaselineRun: vi.fn(),
    getDemoDefaults: vi.fn(),
    getEpisode: vi.fn(),
    getEpisodeEvents: vi.fn(),
    getEpisodeInferenceTraces: vi.fn(),
    getModelRun: vi.fn(),
    getReport: vi.fn(),
    getReportEpisode: vi.fn(),
    getReportEpisodeEvents: vi.fn(),
    getReportEpisodeInferenceTraces: vi.fn(),
    getScenario: vi.fn(),
    getScenarioPreview: vi.fn(),
    getScenarioTrajectoryCzml: vi.fn(),
    listBaselines: vi.fn(),
    listEpisodes: vi.fn(),
    listModels: vi.fn(),
    listReports: vi.fn(),
    listScenarios: vi.fn(),
    runBaseline: vi.fn(),
    runModel: vi.fn(),
  };
});

const scenarioSummary: ScenarioSummary = {
  bundle_id: "sb:osbench-phase2-foundation-v1:sparse_frontier:seed-403",
  benchmark_id: "osbench-phase2-foundation-v1",
  scenario_family: "sparse_frontier",
  simulation_seed: 403,
  decision_interval_seconds: 60,
  start_time_utc: "2026-04-17T00:00:00Z",
  end_time_utc: "2026-04-17T03:00:00Z",
  satellite_count: 1,
  ground_station_count: 1,
  target_cell_count: 1,
  incident_count: 1,
  observation_opportunity_count: 1,
  downlink_window_count: 1,
};

const scenarioBundle: ScenarioBundle = {
  bundle_id: scenarioSummary.bundle_id,
  benchmark_id: scenarioSummary.benchmark_id,
  scenario_family: scenarioSummary.scenario_family,
  simulation_seed: scenarioSummary.simulation_seed,
  decision_interval_seconds: 60,
  time_window: {
    start_time_utc: "2026-04-17T00:00:00Z",
    end_time_utc: "2026-04-17T03:00:00Z",
  },
  satellites: [
    {
      satellite_id: "sat:test-001",
      name: "Test Sentinel",
      norad_catalog_id: 12345,
      sensor: {
        sensor_id: "sensor:test-001",
        swath_km: 120,
        quality_nominal: 0.88,
      },
      downlink: {
        buffer_capacity_mb: 10000,
        nominal_downlink_rate_mbps: 420,
      },
      constraints: {
        max_retargets_per_orbit: 2,
        availability: "nominal",
      },
    },
  ],
  ground_stations: [
    {
      station_id: "gs:test-001",
      name: "Test Ground",
      location: {
        lat: 64.8378,
        lon: -147.7164,
        alt_m: 136,
      },
      capabilities: {
        max_concurrent_contacts: 2,
        downlink_rate_mbps: 700,
        availability: "nominal",
      },
    },
  ],
  target_cells: [
    {
      target_cell_id: "tc:test-001",
      h3_cell: "8928341aec3ffff",
      centroid: {
        lat: 39.5547,
        lon: -121.7349,
      },
      region_name: "Northern California",
      static_value: 0.77,
      priority_class: "high",
    },
  ],
  incidents: [
    {
      incident_id: "inc:test-001",
      incident_type: "wildfire",
      target_cell_id: "tc:test-001",
      ignition_time_utc: "2026-04-17T00:30:00Z",
      urgency_score: 0.95,
      confidence: 0.88,
      state: "active",
      estimated_area_ha: 90,
    },
  ],
  observation_opportunities: [
    {
      opportunity_id: "opp:test-001",
      satellite_id: "sat:test-001",
      target_cell_id: "tc:test-001",
      start_time_utc: "2026-04-17T01:00:00Z",
      end_time_utc: "2026-04-17T01:09:00Z",
      predicted_quality_mean: 0.68,
      predicted_cloud_obstruction_prob: 0.18,
      estimated_data_volume_mb: 920,
      slew_cost: 0.22,
      incident_ids: ["inc:test-001"],
    },
  ],
  downlink_windows: [
    {
      window_id: "dw:test-001",
      satellite_id: "sat:test-001",
      station_id: "gs:test-001",
      start_time_utc: "2026-04-17T01:30:00Z",
      end_time_utc: "2026-04-17T01:38:00Z",
      max_volume_mb: 2000,
      expected_rate_mbps: 420,
      outage_risk: 0.02,
    },
  ],
  config: {
    family_display_name: "Sparse Frontier",
    notes: "Fixture-backed scenario.",
    orbit_asset_bundle_id: "eph:test-001",
  },
};

const scenarioPreview: ScenarioPreview = {
  bundle_id: scenarioBundle.bundle_id,
  benchmark_id: scenarioBundle.benchmark_id,
  scenario_family: scenarioBundle.scenario_family,
  simulation_seed: scenarioBundle.simulation_seed,
  time_window: scenarioBundle.time_window,
  counts: {
    satellites: 1,
    ground_stations: 1,
    target_cells: 1,
    incidents: 1,
    observation_opportunities: 1,
    downlink_windows: 1,
  },
  target_cells: [
    {
      target_cell_id: "tc:test-001",
      h3_cell: "8928341aec3ffff",
      region_name: "Northern California",
      static_value: 0.77,
      priority_class: "high",
      centroid: {
        lat: 39.5547,
        lon: -121.7349,
      },
    },
  ],
  incidents: [
    {
      incident_id: "inc:test-001",
      target_cell_id: "tc:test-001",
      urgency_score: 0.95,
      confidence: 0.88,
      state: "active",
      ignition_time_utc: "2026-04-17T00:30:00Z",
    },
  ],
};

const baseline: BaselineDescriptor = {
  baseline_id: "urgency_greedy",
  version: "1.0.0",
  description: "Greedy urgency baseline.",
};

const model: ModelDescriptor = {
  model_key: "model:ppo-phase2-smoke",
  planner_id: "ppo_from_bc_smoke",
  checkpoint_id: "checkpoint_000002",
  checkpoint_fingerprint: "sha256:test",
  model_id: "ppo:test",
  algorithm: "PPO",
  created_at_utc: "2026-04-17T00:00:00Z",
  description: "Fixture PPO checkpoint.",
};

const apiEpisodeSummary: EpisodeSummary = {
  episode_id: "ep:api:ppo",
  runtime_episode_id: "runtime:ppo",
  bundle_id: scenarioBundle.bundle_id,
  planner_id: model.planner_id,
  display_name: model.planner_id,
  planner_kind: "trained_policy",
  episode_source: "api",
  simulation_seed: scenarioBundle.simulation_seed,
  created_at_utc: "2026-04-17T00:00:00Z",
  updated_at_utc: "2026-04-17T01:39:00Z",
  sim_tick: 99,
  terminated: true,
  truncated: false,
  mission_utility: -2.979,
  replay_event_count: 3,
};

function makeEpisodeDetail(
  summary: EpisodeSummary,
  overrides?: Partial<EpisodeDetail>,
): EpisodeDetail {
  return {
    ...summary,
    action_history_length: 2,
    latest_observation: {
      sim_tick: 99,
      sim_time_utc: "2026-04-17T01:39:00Z",
      horizon_tick: 180,
      mission_utility: summary.mission_utility,
      incidents: [],
      action_mask: {
        mask_id: "mask:test-001",
        legal_action_count: 5,
        actions: [],
      },
    },
    action_mask: {
      mask_id: "mask:test-001",
      legal_action_count: 5,
      actions: [],
    },
    metrics: {
      time_to_first_useful_observation_seconds: {
        count: 0,
        mean: null,
        median: null,
        p90: null,
        values: [],
      },
      useful_observation_value_captured: 0,
      cloud_waste_rate: 0,
      downlink_latency_seconds: {
        count: 0,
        mean: null,
        median: null,
        p90: null,
        values: [],
      },
      missed_urgent_incident_rate: 1,
      opportunity_utilization_efficiency: 0,
      mission_utility: summary.mission_utility,
      observation_commit_count: 0,
      useful_packet_count: 0,
      urgent_incident_count: 1,
      missed_urgent_incident_count: 1,
    },
    ...overrides,
  };
}

function makeInferenceTrace(
  episodeId: string,
  plannerId: string,
): EpisodeInferenceTraceResponse {
  return {
    episode_id: episodeId,
    planner_id: plannerId,
    planner_kind: plannerId.includes("ppo") ? "trained_policy" : "builtin",
    step_count: 1,
    traces: [
      {
        event_index: 1,
        sim_tick: 0,
        sim_time_utc: "2026-04-17T00:00:00Z",
        action_id: "act:noop:noop",
        action_type: "noop",
        action_ref: "noop",
        planner_trace: {
          policy_trace: {
            selected_action_id: "act:noop:noop",
            selected_logit: 2.0254,
            selected_probability: 1,
            selected_slot: 0,
            action_entropy: 0.0312,
            value_estimate: -0.1377,
            legal_action_count: 5,
            mask_pressure: 0.9231,
            top_slots: [
              {
                slot_index: 0,
                action_id: "act:noop:noop",
                logit: 2.0254,
                probability: 1,
                slot_mapping: {
                  action_id: "act:noop:noop",
                  action_ref: "noop",
                  action_type: "noop",
                  slot_index: 0,
                },
              },
            ],
          },
        },
      },
    ],
  };
}

const replayEvents: ReplayEvent[] = [
  {
    event_id: "evt:start",
    episode_id: "ep:test",
    event_index: 0,
    sim_tick: 0,
    sim_time_utc: "2026-04-17T00:00:00Z",
    event_type: "episode_started",
    actor_type: "planner",
    actor_id: "planner:test",
    payload: {
      planner_id: "planner:test",
      episode_seed: 403,
    },
  },
  {
    event_id: "evt:select",
    episode_id: "ep:test",
    event_index: 1,
    sim_tick: 0,
    sim_time_utc: "2026-04-17T00:00:00Z",
    event_type: "action_selected",
    actor_type: "planner",
    actor_id: "planner:test",
    payload: {
      action_id: "act:noop:noop",
      action_type: "noop",
      action_ref: "noop",
      planner_trace: {
        policy_trace: {
          selected_action_id: "act:noop:noop",
          selected_slot: 0,
          action_entropy: 0.0312,
          legal_action_count: 5,
          mask_pressure: 0.9231,
          value_estimate: -0.1377,
          top_slots: [
            {
              slot_index: 0,
              action_id: "act:noop:noop",
              probability: 1,
              logit: 2.0254,
              slot_mapping: {
                action_id: "act:noop:noop",
                action_ref: "noop",
                action_type: "noop",
                slot_index: 0,
              },
            },
          ],
        },
      },
    },
  },
  {
    event_id: "evt:end",
    episode_id: "ep:test",
    event_index: 2,
    sim_tick: 99,
    sim_time_utc: "2026-04-17T01:39:00Z",
    event_type: "episode_ended",
    actor_type: "env",
    actor_id: "env:test",
    payload: {
      mission_utility: -2.979,
      terminated: true,
      truncated: false,
    },
  },
];

const reportSummary: ReportSummary = {
  report_id: "evalrun--phase2-heldout-smoke-v1",
  report_kind: "evaluation",
  title: "eval:phase2-foundation-v1",
  summary_path: "/tmp/summary.json",
  benchmark_id: scenarioSummary.benchmark_id,
  episode_count: 2,
  planner_count: 2,
  trained_policy_count: 1,
  baseline_count: 1,
  splits: ["val"],
  notable_episode_count: 2,
};

const reportDetail: ReportDetail = {
  ...reportSummary,
  episodes: [
    {
      report_id: reportSummary.report_id,
      report_episode_id: "rep:policy",
      episode_id: "ep:report:policy",
      bundle_id: scenarioBundle.bundle_id,
      split: "val",
      scenario_family: scenarioBundle.scenario_family,
      planner_key: "ppo_from_bc_smoke",
      planner_kind: "policy_checkpoint",
      planner_version: "ppo-step-128",
      display_name: "ppo_from_bc_smoke",
      simulation_seed: scenarioBundle.simulation_seed,
      action_count: 1440,
      metrics: makeEpisodeDetail(apiEpisodeSummary).metrics,
      replay_path: "/tmp/policy.ndjson",
      summary_path: "/tmp/policy.json",
      scenario_path: "/tmp/scenario.json",
      reward_audit: {
        total_reward_sum: -2.979,
      },
      bundle_profile: {
        difficulty_tier: 1,
        scenario_family: scenarioBundle.scenario_family,
      },
    },
    {
      report_id: reportSummary.report_id,
      report_episode_id: "rep:baseline",
      episode_id: "ep:report:baseline",
      bundle_id: scenarioBundle.bundle_id,
      split: "val",
      scenario_family: scenarioBundle.scenario_family,
      planner_key: "urgency_greedy",
      planner_kind: "builtin",
      planner_version: "phase1-v1",
      display_name: "urgency_greedy",
      simulation_seed: scenarioBundle.simulation_seed,
      action_count: 1440,
      metrics: {
        ...makeEpisodeDetail(apiEpisodeSummary).metrics,
        mission_utility: -3.206,
      },
      replay_path: "/tmp/baseline.ndjson",
      summary_path: "/tmp/baseline.json",
      scenario_path: "/tmp/scenario.json",
      reward_audit: {
        total_reward_sum: -3.206,
      },
      bundle_profile: {
        difficulty_tier: 1,
        scenario_family: scenarioBundle.scenario_family,
      },
    },
  ],
  notable_episodes: [
    {
      category: "biggest_rl_wins",
      planner_key: "ppo_from_bc_smoke",
      bundle_id: scenarioBundle.bundle_id,
      split: "val",
      scenario_family: scenarioBundle.scenario_family,
      primary_metric_value: -2.979,
      difference_vs_best_baseline: 0.227,
    },
  ],
};

const demoDefaults: DemoDefaults = {
  bundle_id: scenarioBundle.bundle_id,
  baseline_id: baseline.baseline_id,
  episode_id: null,
  ui_query: "",
};

beforeEach(() => {
  vi.clearAllMocks();

  vi.mocked(api.listScenarios).mockResolvedValue([scenarioSummary]);
  vi.mocked(api.listBaselines).mockResolvedValue([baseline]);
  vi.mocked(api.listModels).mockResolvedValue([model]);
  vi.mocked(api.listEpisodes).mockResolvedValue([apiEpisodeSummary]);
  vi.mocked(api.listReports).mockResolvedValue([reportSummary]);
  vi.mocked(api.getDemoDefaults).mockResolvedValue(demoDefaults);
  vi.mocked(api.getScenario).mockResolvedValue(scenarioBundle);
  vi.mocked(api.getScenarioPreview).mockResolvedValue(scenarioPreview);
  vi.mocked(api.getScenarioTrajectoryCzml).mockResolvedValue([]);
  vi.mocked(api.getReport).mockResolvedValue(reportDetail);
  vi.mocked(api.getEpisode).mockResolvedValue(makeEpisodeDetail(apiEpisodeSummary));
  vi.mocked(api.getEpisodeEvents).mockResolvedValue(replayEvents);
  vi.mocked(api.getEpisodeInferenceTraces).mockResolvedValue(
    makeInferenceTrace(apiEpisodeSummary.episode_id, apiEpisodeSummary.planner_id),
  );
  vi.mocked(api.getReportEpisode).mockImplementation(async (reportEpisodeId) => {
    const entry = reportDetail.episodes.find(
      (episode) => episode.report_episode_id === reportEpisodeId,
    );
    if (!entry) {
      throw new Error("unknown report episode");
    }
    const summary: EpisodeSummary = {
      episode_id: reportEpisodeId,
      runtime_episode_id: entry.episode_id,
      bundle_id: entry.bundle_id,
      planner_id: entry.planner_key,
      display_name: entry.display_name,
      planner_kind: entry.planner_kind,
      episode_source: "report",
      report_id: reportDetail.report_id,
      report_episode_id: entry.report_episode_id,
      report_split: entry.split,
      simulation_seed: entry.simulation_seed,
      created_at_utc: "2026-04-17T00:00:00Z",
      updated_at_utc: "2026-04-17T00:00:00Z",
      sim_tick: 99,
      terminated: true,
      truncated: false,
      mission_utility: entry.metrics.mission_utility,
      replay_event_count: replayEvents.length,
    };
    return makeEpisodeDetail(summary, {
      metrics: entry.metrics,
    });
  });
  vi.mocked(api.getReportEpisodeEvents).mockResolvedValue(replayEvents);
  vi.mocked(api.getReportEpisodeInferenceTraces).mockImplementation(async (reportEpisodeId) =>
    makeInferenceTrace(reportEpisodeId, reportEpisodeId === "rep:policy" ? "ppo_from_bc_smoke" : "urgency_greedy"),
  );
  vi.mocked(api.runModel).mockResolvedValue({
    job_id: "job:model",
    status: "completed",
    episode_id: apiEpisodeSummary.episode_id,
  });
  vi.mocked(api.getModelRun).mockResolvedValue({
    job_id: "job:model",
    model_key: model.model_key,
    bundle_id: scenarioBundle.bundle_id,
    simulation_seed: scenarioBundle.simulation_seed,
    status: "completed",
    episode_id: apiEpisodeSummary.episode_id,
    created_at_utc: "2026-04-17T00:00:00Z",
  });
  vi.mocked(api.runBaseline).mockResolvedValue({
    job_id: "job:baseline",
    status: "completed",
    episode_id: apiEpisodeSummary.episode_id,
  });
  vi.mocked(api.getBaselineRun).mockResolvedValue({
    job_id: "job:baseline",
    baseline_id: baseline.baseline_id,
    bundle_id: scenarioBundle.bundle_id,
    simulation_seed: scenarioBundle.simulation_seed,
    status: "completed",
    episode_id: apiEpisodeSummary.episode_id,
    created_at_utc: "2026-04-17T00:00:00Z",
  });
});

test("runs a trained policy and surfaces inference telemetry", async () => {
  render(<App />);

  await screen.findByText("Phase 2 Mission Control");

  fireEvent.click(screen.getByRole("button", { name: "Trained Policy" }));
  fireEvent.click(screen.getByRole("button", { name: /Launch trained-policy replay/i }));

  await waitFor(() => {
    expect(api.runModel).toHaveBeenCalledWith(
      model.model_key,
      scenarioBundle.bundle_id,
      scenarioBundle.simulation_seed,
    );
  });
  await waitFor(() => {
    expect(api.getEpisode).toHaveBeenCalledWith(apiEpisodeSummary.episode_id);
  });

  await screen.findByText("Canonical Action");
  expect(screen.getAllByText("Selected Slot").length).toBeGreaterThan(0);
  expect(screen.getByText("Mask Pressure")).toBeInTheDocument();
});

test("loads report-backed policy and baseline replays into compare lanes", async () => {
  render(<App />);

  await screen.findByRole("button", { name: "Load report replay into primary" });

  fireEvent.click(screen.getByRole("button", { name: "Load report replay into primary" }));

  await waitFor(() => {
    expect(screen.getAllByText("ppo_from_bc_smoke").length).toBeGreaterThan(0);
  });

  fireEvent.click(screen.getByRole("button", { name: "Compare Mode" }));
  fireEvent.change(screen.getByLabelText("Load Target"), {
    target: { value: "compare" },
  });
  fireEvent.change(screen.getByLabelText("Report Replay"), {
    target: { value: "rep:baseline" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Load report replay into compare" }));

  await waitFor(() => {
    expect(screen.getByText("Compare Delta")).toBeInTheDocument();
    expect(screen.getAllByText("urgency_greedy").length).toBeGreaterThan(0);
  });
});
