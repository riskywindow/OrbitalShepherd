import { startTransition, useDeferredValue, useEffect, useState } from "react";

import {
  ApiError,
  getBaselineRun,
  getDemoDefaults,
  getEpisode,
  getEpisodeEvents,
  getEpisodeInferenceTraces,
  getModelRun,
  getReport,
  getReportEpisode,
  getReportEpisodeEvents,
  getReportEpisodeInferenceTraces,
  getScenario,
  getScenarioPreview,
  getScenarioTrajectoryCzml,
  listBaselines,
  listEpisodes,
  listModels,
  listReports,
  listScenarios,
  runBaseline,
  runModel,
} from "./api";
import { GlobeView } from "./GlobeView";
import { buildReplayModel } from "./replay";
import type {
  BaselineDescriptor,
  CzmlPacket,
  DemoDefaults,
  EpisodeDetail,
  EpisodeInferenceTraceResponse,
  EpisodeInferenceTraceStep,
  EpisodeMetrics,
  EpisodeSummary,
  ModelDescriptor,
  PolicyTrace,
  ReplayEvent,
  ReportDetail,
  ReportEpisodeSummary,
  ReportNotableEpisode,
  ReportSummary,
  ScenarioBundle,
  ScenarioPreview,
  ScenarioSummary,
} from "./types";

const PLAYBACK_INTERVAL_MS = 350;

type LaneId = "primary" | "compare";
type RunnerMode = "baseline" | "trained_policy";
type ReplaySourceRef =
  | {
      kind: "episode";
      id: string;
    }
  | {
      kind: "report_episode";
      id: string;
    };

interface ReplayLaneState {
  id: LaneId;
  label: string;
  source: ReplaySourceRef | null;
  detail: EpisodeDetail | null;
  metrics: EpisodeMetrics | null;
  events: ReplayEvent[];
  inference: EpisodeInferenceTraceResponse | null;
  bundle: ScenarioBundle | null;
  czml: CzmlPacket[] | null;
  replayTickIndex: number;
  replayModel: ReturnType<typeof buildReplayModel> | null;
  isLoading: boolean;
  errorMessage: string;
}

function createLaneState(id: LaneId, label: string): ReplayLaneState {
  return {
    id,
    label,
    source: null,
    detail: null,
    metrics: null,
    events: [],
    inference: null,
    bundle: null,
    czml: null,
    replayTickIndex: 0,
    replayModel: null,
    isLoading: false,
    errorMessage: "",
  };
}

function App() {
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [baselines, setBaselines] = useState<BaselineDescriptor[]>([]);
  const [models, setModels] = useState<ModelDescriptor[]>([]);
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState("");
  const [selectedBaselineId, setSelectedBaselineId] = useState("");
  const [selectedModelKey, setSelectedModelKey] = useState("");
  const [runnerMode, setRunnerMode] = useState<RunnerMode>("baseline");
  const [selectedSavedEpisodeId, setSelectedSavedEpisodeId] = useState("");
  const [selectedReportId, setSelectedReportId] = useState("");
  const [selectedReportEpisodeId, setSelectedReportEpisodeId] = useState("");
  const [loadTargetLane, setLoadTargetLane] = useState<LaneId>("primary");
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [focusedLaneId, setFocusedLaneId] = useState<LaneId>("primary");
  const [playingLaneId, setPlayingLaneId] = useState<LaneId | null>(null);
  const [scenarioBundle, setScenarioBundle] = useState<ScenarioBundle | null>(null);
  const [scenarioPreview, setScenarioPreview] = useState<ScenarioPreview | null>(null);
  const [scenarioCzml, setScenarioCzml] = useState<CzmlPacket[] | null>(null);
  const [selectedReport, setSelectedReport] = useState<ReportDetail | null>(null);
  const [demoDefaults, setDemoDefaults] = useState<DemoDefaults | null>(null);
  const [lanes, setLanes] = useState<Record<LaneId, ReplayLaneState>>({
    primary: createLaneState("primary", "Primary"),
    compare: createLaneState("compare", "Compare"),
  });
  const [loadingMessage, setLoadingMessage] = useState("Bootstrapping mission-control");
  const [errorMessage, setErrorMessage] = useState("");
  const [isBootstrapping, setIsBootstrapping] = useState(true);
  const [isScenarioLoading, setIsScenarioLoading] = useState(false);
  const [isReportLoading, setIsReportLoading] = useState(false);
  const [isRunningPlanner, setIsRunningPlanner] = useState(false);

  const focusedLane = lanes[focusedLaneId];
  const deferredFocusedTickIndex = useDeferredValue(focusedLane.replayTickIndex);
  const focusedReplayTick =
    focusedLane.replayModel?.ticks[deferredFocusedTickIndex] ??
    focusedLane.replayModel?.ticks[focusedLane.replayModel.ticks.length - 1] ??
    null;
  const activeHeroBundle = focusedLane.bundle ?? scenarioBundle;
  const activeHeroCzml = focusedLane.czml ?? scenarioCzml;
  const activeHeroTick = focusedLane.replayModel ? focusedReplayTick : null;
  const playingLane = playingLaneId ? lanes[playingLaneId] : null;
  const scenarioEpisodes = episodes
    .filter((episode) => episode.bundle_id === selectedScenarioId)
    .sort(
      (left, right) =>
        Date.parse(right.updated_at_utc) - Date.parse(left.updated_at_utc),
    );
  const reportEpisodes = selectedReport
    ? [...selectedReport.episodes]
        .filter((episode) => !selectedScenarioId || episode.bundle_id === selectedScenarioId)
        .sort((left, right) => right.metrics.mission_utility - left.metrics.mission_utility)
    : [];
  const reportNotableEpisodes = selectedReport
    ? selectedReport.notable_episodes.filter(
        (episode) => !selectedScenarioId || episode.bundle_id === selectedScenarioId,
      )
    : [];
  const missionUtilityDelta =
    compareEnabled &&
    lanes.primary.metrics &&
    lanes.compare.metrics &&
    lanes.primary.detail?.bundle_id === lanes.compare.detail?.bundle_id
      ? lanes.compare.metrics.mission_utility - lanes.primary.metrics.mission_utility
      : null;

  async function refreshCatalog(): Promise<void> {
    const [scenarioItems, baselineItems, modelItems, episodeItems, reportItems] =
      await Promise.all([
        listScenarios(),
        listBaselines(),
        listModels(),
        listEpisodes(),
        listReports().catch(() => []),
      ]);
    setScenarios(scenarioItems);
    setBaselines(baselineItems);
    setModels(modelItems);
    setReports(reportItems);
    setEpisodes(
      [...episodeItems].sort(
        (left, right) =>
          Date.parse(right.updated_at_utc) - Date.parse(left.updated_at_utc),
      ),
    );
    if (!selectedScenarioId && scenarioItems[0]) {
      setSelectedScenarioId(scenarioItems[0].bundle_id);
    }
    if (!selectedBaselineId && baselineItems[0]) {
      setSelectedBaselineId(baselineItems[0].baseline_id);
    }
    if (!selectedModelKey && modelItems[0]) {
      setSelectedModelKey(modelItems[0].model_key);
    }
    if (!selectedReportId && reportItems[0]) {
      setSelectedReportId(reportItems[0].report_id);
    }
  }

  function updateLane(
    laneId: LaneId,
    nextValue: ReplayLaneState | ((current: ReplayLaneState) => ReplayLaneState),
  ): void {
    setLanes((current) => ({
      ...current,
      [laneId]:
        typeof nextValue === "function"
          ? (nextValue as (lane: ReplayLaneState) => ReplayLaneState)(current[laneId])
          : nextValue,
    }));
  }

  function clearLane(laneId: LaneId): void {
    if (playingLaneId === laneId) {
      setPlayingLaneId(null);
    }
    updateLane(laneId, createLaneState(laneId, lanes[laneId].label));
  }

  function setLaneTickIndex(laneId: LaneId, nextIndex: number): void {
    if (playingLaneId === laneId) {
      setPlayingLaneId(null);
    }
    updateLane(laneId, (current) => {
      const maxIndex = Math.max((current.replayModel?.ticks.length ?? 1) - 1, 0);
      return {
        ...current,
        replayTickIndex: Math.min(Math.max(nextIndex, 0), maxIndex),
      };
    });
  }

  async function loadReplayLane(laneId: LaneId, source: ReplaySourceRef): Promise<void> {
    updateLane(laneId, (current) => ({
      ...current,
      isLoading: true,
      errorMessage: "",
      source,
    }));
    setLoadingMessage(`Loading ${source.kind === "episode" ? "saved replay" : "report replay"}`);
    setErrorMessage("");
    try {
      const [detail, events, inference] =
        source.kind === "episode"
          ? await Promise.all([
              getEpisode(source.id),
              getEpisodeEvents(source.id),
              getEpisodeInferenceTraces(source.id).catch(() => null),
            ])
          : await Promise.all([
              getReportEpisode(source.id),
              getReportEpisodeEvents(source.id),
              getReportEpisodeInferenceTraces(source.id).catch(() => null),
            ]);
      const [bundle, czml] = await Promise.all([
        getScenario(detail.bundle_id),
        getScenarioTrajectoryCzml(detail.bundle_id, 120).catch(() => null),
      ]);
      startTransition(() => {
        updateLane(laneId, (current) => ({
          ...current,
          source,
          detail,
          metrics: detail.metrics,
          events,
          inference,
          bundle,
          czml,
          replayTickIndex: 0,
          replayModel: buildReplayModel(bundle, events),
          isLoading: false,
          errorMessage: "",
        }));
      });
      setFocusedLaneId(laneId);
    } catch (error) {
      updateLane(laneId, (current) => ({
        ...current,
        detail: null,
        metrics: null,
        events: [],
        inference: null,
        bundle: null,
        czml: null,
        replayTickIndex: 0,
        replayModel: null,
        isLoading: false,
        errorMessage: errorToMessage(error),
      }));
      setErrorMessage(errorToMessage(error));
    }
  }

  async function handleRunPlanner(): Promise<void> {
    if (!selectedScenarioId) {
      return;
    }
    setIsRunningPlanner(true);
    setLoadingMessage(
      runnerMode === "baseline"
        ? "Running baseline planner"
        : "Running trained policy checkpoint",
    );
    setErrorMessage("");
    setPlayingLaneId(null);
    try {
      let episodeId: string | null = null;
      if (runnerMode === "baseline") {
        const run = await runBaseline(
          selectedBaselineId,
          selectedScenarioId,
          scenarioBundle?.simulation_seed ??
            scenarios.find((scenario) => scenario.bundle_id === selectedScenarioId)
              ?.simulation_seed ??
            0,
        );
        episodeId = run.episode_id;
        if (!episodeId) {
          for (let attempt = 0; attempt < 30; attempt += 1) {
            await delay(1000);
            const detail = await getBaselineRun(run.job_id);
            if (detail.status === "failed") {
              throw new Error(detail.error_message ?? "Baseline execution failed");
            }
            if (detail.episode_id) {
              episodeId = detail.episode_id;
              break;
            }
          }
        }
      } else {
        const run = await runModel(
          selectedModelKey,
          selectedScenarioId,
          scenarioBundle?.simulation_seed ??
            scenarios.find((scenario) => scenario.bundle_id === selectedScenarioId)
              ?.simulation_seed ??
            0,
        );
        episodeId = run.episode_id;
        if (!episodeId) {
          for (let attempt = 0; attempt < 30; attempt += 1) {
            await delay(1000);
            const detail = await getModelRun(run.job_id);
            if (detail.status === "failed") {
              throw new Error(detail.error_message ?? "Trained policy execution failed");
            }
            if (detail.episode_id) {
              episodeId = detail.episode_id;
              break;
            }
          }
        }
      }
      if (!episodeId) {
        throw new Error("Planner execution did not produce a replayable episode");
      }
      await refreshCatalog();
      setSelectedSavedEpisodeId(episodeId);
      await loadReplayLane(loadTargetLane, {
        kind: "episode",
        id: episodeId,
      });
    } catch (error) {
      setErrorMessage(errorToMessage(error));
    } finally {
      setIsRunningPlanner(false);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function bootstrap(): Promise<void> {
      try {
        const routeDefaults = readRouteDefaults();
        const [
          scenarioItems,
          baselineItems,
          modelItems,
          episodeItems,
          reportItems,
          remoteDemoDefaults,
        ] = await Promise.all([
          listScenarios(),
          listBaselines(),
          listModels(),
          listEpisodes(),
          listReports().catch(() => []),
          getDemoDefaults().catch(() => null),
        ]);
        if (cancelled) {
          return;
        }
        setScenarios(scenarioItems);
        setBaselines(baselineItems);
        setModels(modelItems);
        setReports(reportItems);
        setDemoDefaults(remoteDemoDefaults);
        setEpisodes(
          [...episodeItems].sort(
            (left, right) =>
              Date.parse(right.updated_at_utc) - Date.parse(left.updated_at_utc),
          ),
        );
        setSelectedScenarioId(
          routeDefaults.scenarioId ||
            remoteDemoDefaults?.bundle_id ||
            scenarioItems[0]?.bundle_id ||
            "",
        );
        setSelectedBaselineId(
          routeDefaults.baselineId ||
            remoteDemoDefaults?.baseline_id ||
            baselineItems[0]?.baseline_id ||
            "",
        );
        setSelectedModelKey(modelItems[0]?.model_key ?? "");
        setSelectedSavedEpisodeId(routeDefaults.episodeId || remoteDemoDefaults?.episode_id || "");
        setSelectedReportId(reportItems[0]?.report_id ?? "");

        const initialEpisodeId = routeDefaults.episodeId || remoteDemoDefaults?.episode_id;
        if (initialEpisodeId) {
          void loadReplayLane("primary", {
            kind: "episode",
            id: initialEpisodeId,
          });
        }
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(errorToMessage(error));
        }
      } finally {
        if (!cancelled) {
          setIsBootstrapping(false);
        }
      }
    }

    void bootstrap();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedScenarioId) {
      return;
    }
    let cancelled = false;

    async function loadScenarioWorkspace(): Promise<void> {
      setIsScenarioLoading(true);
      setLoadingMessage("Loading scenario geometry");
      try {
        const [bundle, preview, czml] = await Promise.all([
          getScenario(selectedScenarioId),
          getScenarioPreview(selectedScenarioId),
          getScenarioTrajectoryCzml(selectedScenarioId, 120).catch(() => null),
        ]);
        if (cancelled) {
          return;
        }
        setScenarioBundle(bundle);
        setScenarioPreview(preview);
        setScenarioCzml(czml);
      } catch (error) {
        if (!cancelled) {
          setScenarioBundle(null);
          setScenarioPreview(null);
          setScenarioCzml(null);
          setErrorMessage(errorToMessage(error));
        }
      } finally {
        if (!cancelled) {
          setIsScenarioLoading(false);
        }
      }
    }

    void loadScenarioWorkspace();

    return () => {
      cancelled = true;
    };
  }, [selectedScenarioId]);

  useEffect(() => {
    if (!selectedReportId) {
      setSelectedReport(null);
      return;
    }
    let cancelled = false;

    async function loadReport(): Promise<void> {
      setIsReportLoading(true);
      setLoadingMessage("Loading evaluation report");
      try {
        const detail = await getReport(selectedReportId);
        if (cancelled) {
          return;
        }
        setSelectedReport(detail);
        if (!selectedReportEpisodeId && detail.episodes[0]) {
          setSelectedReportEpisodeId(detail.episodes[0].report_episode_id);
        }
      } catch (error) {
        if (!cancelled) {
          setSelectedReport(null);
          setErrorMessage(errorToMessage(error));
        }
      } finally {
        if (!cancelled) {
          setIsReportLoading(false);
        }
      }
    }

    void loadReport();

    return () => {
      cancelled = true;
    };
  }, [selectedReportEpisodeId, selectedReportId]);

  useEffect(() => {
    if (compareEnabled) {
      return;
    }
    setLoadTargetLane("primary");
    if (focusedLaneId === "compare") {
      setFocusedLaneId("primary");
    }
    if (playingLaneId === "compare") {
      setPlayingLaneId(null);
    }
  }, [compareEnabled, focusedLaneId, playingLaneId]);

  useEffect(() => {
    if (!playingLaneId || !playingLane?.replayModel) {
      return;
    }
    const interval = window.setInterval(() => {
      let shouldStop = false;
      setLanes((current) => {
        const lane = current[playingLaneId];
        if (!lane.replayModel) {
          shouldStop = true;
          return current;
        }
        const nextIndex = lane.replayTickIndex + 1;
        if (nextIndex >= lane.replayModel.ticks.length) {
          shouldStop = true;
          return {
            ...current,
            [playingLaneId]: {
              ...lane,
              replayTickIndex: lane.replayModel.ticks.length - 1,
            },
          };
        }
        return {
          ...current,
          [playingLaneId]: {
            ...lane,
            replayTickIndex: nextIndex,
          },
        };
      });
      if (shouldStop) {
        window.clearInterval(interval);
        setPlayingLaneId(null);
      }
    }, PLAYBACK_INTERVAL_MS);
    return () => {
      window.clearInterval(interval);
    };
  }, [playingLane?.replayModel, playingLaneId]);

  useEffect(() => {
    writeRouteDefaults({
      scenarioId: selectedScenarioId,
      baselineId: selectedBaselineId,
      episodeId: lanes.primary.detail?.episode_source === "api" ? lanes.primary.detail.episode_id : "",
    });
  }, [lanes.primary.detail, selectedBaselineId, selectedScenarioId]);

  const availableTargetCount = countTargetsByState(activeHeroTick?.targetStates, "available");
  const selectedTargetCount = countTargetsByState(activeHeroTick?.targetStates, "selected");
  const downlinkedTargetCount = countTargetsByState(activeHeroTick?.targetStates, "downlinked");
  const degradedTargetCount =
    countTargetsByState(activeHeroTick?.targetStates, "degraded") +
    countTargetsByState(activeHeroTick?.targetStates, "missed");
  const focusedTrace = inferenceTraceForLane(focusedLane);
  const focusedPolicyTrace = normalizePolicyTrace(focusedTrace);

  return (
    <main className="app-shell">
      <header className="topbar panel">
        <div>
          <p className="eyebrow">Orbital Shepherd</p>
          <h1>Phase 2 Mission Control</h1>
          <p className="subtitle">
            Baselines, trained policies, replay comparison, and inference telemetry
            stay on one operational surface with the globe as the primary frame.
          </p>
        </div>
        <div className="topbar-metrics">
          <MetricChip
            label="Focused Lane"
            value={`${focusedLane.label} · ${lanePlannerLabel(focusedLane.detail)}`}
            tone={focusedLane.detail?.planner_kind?.includes("policy") ? "accent" : "neutral"}
          />
          <MetricChip
            label="Scenario"
            value={activeHeroBundle?.scenario_family ?? "idle"}
            tone="cool"
          />
          <MetricChip
            label="Mission Utility"
            value={
              focusedLane.metrics
                ? focusedLane.metrics.mission_utility.toFixed(3)
                : "n/a"
            }
            tone={
              focusedLane.metrics && focusedLane.metrics.mission_utility < 0
                ? "danger"
                : "neutral"
            }
          />
          <MetricChip
            label="Compare Delta"
            value={missionUtilityDelta === null ? "n/a" : signedFixed(missionUtilityDelta, 3)}
            tone={
              missionUtilityDelta === null
                ? "neutral"
                : missionUtilityDelta >= 0
                  ? "accent"
                  : "danger"
            }
          />
        </div>
      </header>

      {errorMessage ? (
        <section className="status-banner status-banner-error">{errorMessage}</section>
      ) : null}
      {isBootstrapping || isScenarioLoading || isReportLoading || isRunningPlanner ? (
        <section className="status-banner">{loadingMessage}</section>
      ) : null}

      <section className="workspace-grid">
        <aside className="sidebar">
          <section className="panel stack-panel">
            <div className="panel-heading">
              <h2>Run Control</h2>
              <p>
                Launch a built-in planner or trained checkpoint, then route the replay
                into the active compare lane.
              </p>
            </div>

            <label className="field">
              <span>Scenario Bundle</span>
              <select
                value={selectedScenarioId}
                onChange={(event) => setSelectedScenarioId(event.target.value)}
              >
                {scenarios.map((scenario) => (
                  <option key={scenario.bundle_id} value={scenario.bundle_id}>
                    {scenario.bundle_id}
                  </option>
                ))}
              </select>
            </label>

            <div className="segmented-control">
              <button
                className={runnerMode === "baseline" ? "segment is-active" : "segment"}
                onClick={() => setRunnerMode("baseline")}
                type="button"
              >
                Baseline
              </button>
              <button
                className={runnerMode === "trained_policy" ? "segment is-active" : "segment"}
                onClick={() => setRunnerMode("trained_policy")}
                type="button"
              >
                Trained Policy
              </button>
            </div>

            {runnerMode === "baseline" ? (
              <label className="field">
                <span>Planner</span>
                <select
                  value={selectedBaselineId}
                  onChange={(event) => setSelectedBaselineId(event.target.value)}
                >
                  {baselines.map((baseline) => (
                    <option key={baseline.baseline_id} value={baseline.baseline_id}>
                      {baseline.baseline_id}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <label className="field">
                <span>Checkpoint</span>
                <select
                  value={selectedModelKey}
                  onChange={(event) => setSelectedModelKey(event.target.value)}
                >
                  {models.map((model) => (
                    <option key={model.model_key} value={model.model_key}>
                      {model.planner_id} · {model.algorithm}
                    </option>
                  ))}
                </select>
              </label>
            )}

            <div className="inline-field-row">
              <label className="field compact-field">
                <span>Load Target</span>
                <select
                  value={loadTargetLane}
                  onChange={(event) => setLoadTargetLane(event.target.value as LaneId)}
                >
                  <option value="primary">Primary</option>
                  {compareEnabled ? <option value="compare">Compare</option> : null}
                </select>
              </label>
              <label className="field compact-field">
                <span>Compare Mode</span>
                <button
                  className={compareEnabled ? "secondary-action is-active" : "secondary-action"}
                  onClick={() => setCompareEnabled((value) => !value)}
                  type="button"
                >
                  {compareEnabled ? "Enabled" : "Disabled"}
                </button>
              </label>
            </div>

            <button
              className="primary-action"
              disabled={
                !selectedScenarioId ||
                isRunningPlanner ||
                (runnerMode === "baseline" ? !selectedBaselineId : !selectedModelKey)
              }
              onClick={() => {
                void handleRunPlanner();
              }}
              type="button"
            >
              {isRunningPlanner
                ? runnerMode === "baseline"
                  ? "Running baseline..."
                  : "Running policy..."
                : runnerMode === "baseline"
                  ? "Launch baseline replay"
                  : "Launch trained-policy replay"}
            </button>

            <label className="field">
              <span>Saved API Replay</span>
              <select
                value={selectedSavedEpisodeId}
                onChange={(event) => setSelectedSavedEpisodeId(event.target.value)}
              >
                <option value="">Select replay</option>
                {scenarioEpisodes.map((episode) => (
                  <option key={episode.episode_id} value={episode.episode_id}>
                    {lanePlannerLabel(episode)} · {formatCompactTime(episode.updated_at_utc)}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="secondary-action"
              disabled={!selectedSavedEpisodeId}
              onClick={() => {
                void loadReplayLane(loadTargetLane, {
                  kind: "episode",
                  id: selectedSavedEpisodeId,
                });
              }}
              type="button"
            >
              Load saved replay into {laneLabel(loadTargetLane).toLowerCase()}
            </button>

            {demoDefaults?.episode_id ? (
              <p className="body-copy">
                Demo default: {demoDefaults.bundle_id} · {demoDefaults.baseline_id}
              </p>
            ) : null}
          </section>

          <section className="panel stack-panel">
            <div className="panel-heading">
              <h2>Scenario Brief</h2>
              <p>
                {scenarioBundle?.config.family_display_name ??
                  scenarioBundle?.scenario_family ??
                  "No scenario selected"}
              </p>
            </div>
            <dl className="definition-grid">
              <Definition label="Window" value={formatWindow(scenarioBundle?.time_window)} />
              <Definition label="Targets" value={String(scenarioPreview?.counts.target_cells ?? 0)} />
              <Definition label="Incidents" value={String(scenarioPreview?.counts.incidents ?? 0)} />
              <Definition
                label="Opportunities"
                value={String(scenarioPreview?.counts.observation_opportunities ?? 0)}
              />
              <Definition
                label="Downlinks"
                value={String(scenarioPreview?.counts.downlink_windows ?? 0)}
              />
              <Definition label="Orbit Assets" value={String(scenarioPreview?.counts.satellites ?? 0)} />
            </dl>
            <p className="body-copy">
              {String(
                scenarioBundle?.config.notes ??
                  "Pick a scenario, then run or load a planner replay into either compare lane.",
              )}
            </p>
          </section>

          <section className="panel stack-panel">
            <div className="panel-heading">
              <h2>Report Browser</h2>
              <p>
                Evaluation reports stay browseable from local artifacts alone, with direct
                jumps into notable trained-policy and baseline episodes.
              </p>
            </div>

            <label className="field">
              <span>Evaluation Report</span>
              <select
                value={selectedReportId}
                onChange={(event) => setSelectedReportId(event.target.value)}
              >
                <option value="">Select report</option>
                {reports.map((report) => (
                  <option key={report.report_id} value={report.report_id}>
                    {report.report_id}
                  </option>
                ))}
              </select>
            </label>

            {selectedReport ? (
              <>
                <div className="report-summary-strip">
                  <MetricChip label="Episodes" value={String(selectedReport.episode_count)} />
                  <MetricChip
                    label="Policies"
                    value={String(selectedReport.trained_policy_count)}
                    tone="accent"
                  />
                  <MetricChip
                    label="Baselines"
                    value={String(selectedReport.baseline_count)}
                    tone="cool"
                  />
                </div>
                <label className="field">
                  <span>Report Replay</span>
                  <select
                    value={selectedReportEpisodeId}
                    onChange={(event) => setSelectedReportEpisodeId(event.target.value)}
                  >
                    <option value="">Select report replay</option>
                    {reportEpisodes.map((episode) => (
                      <option
                        key={episode.report_episode_id}
                        value={episode.report_episode_id}
                      >
                        {episode.display_name} · {episode.split} ·{" "}
                        {episode.metrics.mission_utility.toFixed(3)}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="secondary-action"
                  disabled={!selectedReportEpisodeId}
                  onClick={() => {
                    void loadReplayLane(loadTargetLane, {
                      kind: "report_episode",
                      id: selectedReportEpisodeId,
                    });
                  }}
                  type="button"
                >
                  Load report replay into {laneLabel(loadTargetLane).toLowerCase()}
                </button>
                <div className="notable-list">
                  {reportNotableEpisodes.slice(0, 6).map((episode) => {
                    const matchingReportEpisode = findReportEpisodeForNotable(
                      selectedReport.episodes,
                      episode,
                    );
                    return (
                      <article className="notable-card" key={notableEpisodeKey(episode)}>
                        <div className="event-header">
                          <span className="event-type tone-accent">
                            {episode.category.replaceAll("_", " ")}
                          </span>
                          <span className="event-time">
                            {episode.primary_metric_value === null ||
                            episode.primary_metric_value === undefined
                              ? "n/a"
                              : episode.primary_metric_value.toFixed(3)}
                          </span>
                        </div>
                        <p className="event-body">
                          {episode.bundle_id}
                          {episode.planner_key ? ` · ${episode.planner_key}` : ""}
                        </p>
                        <div className="mini-actions">
                          <button
                            className="timeline-action"
                            disabled={!matchingReportEpisode}
                            onClick={() => {
                              if (!matchingReportEpisode) {
                                return;
                              }
                              void loadReplayLane("primary", {
                                kind: "report_episode",
                                id: matchingReportEpisode.report_episode_id,
                              });
                            }}
                            type="button"
                          >
                            Load Primary
                          </button>
                          <button
                            className="timeline-action"
                            disabled={!matchingReportEpisode || !compareEnabled}
                            onClick={() => {
                              if (!matchingReportEpisode) {
                                return;
                              }
                              void loadReplayLane("compare", {
                                kind: "report_episode",
                                id: matchingReportEpisode.report_episode_id,
                              });
                            }}
                            type="button"
                          >
                            Load Compare
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </div>
              </>
            ) : (
              <p className="empty-copy">No evaluation report selected.</p>
            )}
          </section>
        </aside>

        <section className="main-column">
          <section className="panel globe-panel">
            <div className="globe-header">
              <div>
                <h2>Orbital Replay Surface</h2>
                <p>
                  The globe follows the focused lane while preserving the dark mission-control
                  posture of scenario geometry, active targets, and delivery outcomes.
                </p>
              </div>
              <div className="state-strip">
                <MetricChip label="Available" value={String(availableTargetCount)} tone="warm" />
                <MetricChip label="Selected" value={String(selectedTargetCount)} tone="accent" />
                <MetricChip label="Downlinked" value={String(downlinkedTargetCount)} tone="cool" />
                <MetricChip label="Degraded" value={String(degradedTargetCount)} tone="danger" />
              </div>
            </div>
            <div className="globe-frame">
              <GlobeView
                bundle={activeHeroBundle}
                czml={activeHeroCzml}
                snapshot={activeHeroTick}
              />
              <div className="legend">
                <LegendSwatch tone="warm" label="Available targets" />
                <LegendSwatch tone="accent" label="Selected opportunity" />
                <LegendSwatch tone="cool" label="Downlink delivered" />
                <LegendSwatch tone="danger" label="Missed / degraded" />
              </div>
            </div>
          </section>

          <section className="panel timeline-panel">
            <div className="panel-heading">
              <h2>Focused Timeline</h2>
              <p>
                Playback follows the focused lane. Side-by-side compare cards below keep
                their own scrub state for metric and event inspection.
              </p>
            </div>
            {focusedLane.replayModel ? (
              <>
                <div className="timeline-toolbar">
                  <button
                    className="timeline-action"
                    onClick={() =>
                      setPlayingLaneId((current) =>
                        current === focusedLaneId ? null : focusedLaneId,
                      )
                    }
                    type="button"
                  >
                    {playingLaneId === focusedLaneId ? "Pause" : "Play"}
                  </button>
                  <button
                    className="timeline-action"
                    disabled={focusedLane.replayTickIndex === 0}
                    onClick={() =>
                      setLaneTickIndex(focusedLaneId, focusedLane.replayTickIndex - 1)
                    }
                    type="button"
                  >
                    Step back
                  </button>
                  <button
                    className="timeline-action"
                    disabled={
                      focusedLane.replayTickIndex >=
                      focusedLane.replayModel.ticks.length - 1
                    }
                    onClick={() =>
                      setLaneTickIndex(focusedLaneId, focusedLane.replayTickIndex + 1)
                    }
                    type="button"
                  >
                    Step forward
                  </button>
                  <span className="timeline-readout">
                    {activeHeroTick?.simTimeUtc
                      ? `${focusedLane.label} · ${formatReplayTime(activeHeroTick.simTimeUtc)}`
                      : "No replay loaded"}
                  </span>
                </div>
                <input
                  className="scrubber"
                  max={Math.max(focusedLane.replayModel.ticks.length - 1, 0)}
                  min={0}
                  onChange={(event) =>
                    setLaneTickIndex(focusedLaneId, Number(event.target.value))
                  }
                  type="range"
                  value={focusedLane.replayTickIndex}
                />
                <div className="timeline-scale">
                  <span>{formatReplayTime(focusedLane.replayModel.ticks[0]?.simTimeUtc)}</span>
                  <span>
                    {formatReplayTime(
                      focusedLane.replayModel.ticks[focusedLane.replayModel.ticks.length - 1]
                        ?.simTimeUtc,
                    )}
                  </span>
                </div>
              </>
            ) : (
              <p className="empty-copy">
                Run a baseline or trained checkpoint, or load a report replay, to unlock
                focused playback.
              </p>
            )}
          </section>

          <section className="panel compare-panel">
            <div className="panel-heading">
              <h2>Compare Deck</h2>
              <p>
                Primary and compare lanes keep independent event navigation, while metrics stay
                aligned for quick baseline-versus-policy inspection.
              </p>
            </div>
            <div className={compareEnabled ? "compare-grid" : "compare-grid compare-grid-single"}>
              <ReplayLaneCard
                compareEnabled
                isFocused={focusedLaneId === "primary"}
                lane={lanes.primary}
                onClear={() => clearLane("primary")}
                onFocus={() => setFocusedLaneId("primary")}
                onScrub={(nextIndex) => setLaneTickIndex("primary", nextIndex)}
                onStep={(delta) => setLaneTickIndex("primary", lanes.primary.replayTickIndex + delta)}
              />
              {compareEnabled ? (
                <ReplayLaneCard
                  compareEnabled
                  isFocused={focusedLaneId === "compare"}
                  lane={lanes.compare}
                  onClear={() => clearLane("compare")}
                  onFocus={() => setFocusedLaneId("compare")}
                  onScrub={(nextIndex) => setLaneTickIndex("compare", nextIndex)}
                  onStep={(delta) =>
                    setLaneTickIndex("compare", lanes.compare.replayTickIndex + delta)
                  }
                />
              ) : null}
            </div>
          </section>
        </section>

        <aside className="sidebar">
          <section className="panel stack-panel">
            <div className="panel-heading">
              <h2>Replay Event Panel</h2>
              <p>
                The focused lane’s operational events are trimmed to decisions and outcomes that
                explain the current replay position.
              </p>
            </div>
            {focusedLane.replayModel ? (
              <div className="event-list">
                {visibleImportantEvents(focusedLane).map((event) => (
                  <article className="event-card" key={event.event_id}>
                    <div className="event-header">
                      <span className={`event-type tone-${eventTone(event.event_type)}`}>
                        {event.event_type.replaceAll("_", " ")}
                      </span>
                      <span className="event-time">{formatReplayTime(event.sim_time_utc)}</span>
                    </div>
                    <p className="event-body">{describeEvent(event)}</p>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-copy">No focused replay loaded.</p>
            )}
          </section>

          <section className="panel stack-panel">
            <div className="panel-heading">
              <h2>Inference Telemetry</h2>
              <p>
                Selected slot, canonical action identity, candidate probabilities, entropy,
                value estimate, and mask pressure are exposed directly from planner traces.
              </p>
            </div>
            {focusedPolicyTrace ? (
              <>
                <div className="telemetry-strip">
                  <MetricChip
                    label="Selected Slot"
                    value={nullableNumber(focusedPolicyTrace.selected_slot)}
                    tone="accent"
                  />
                  <MetricChip
                    label="Canonical Action"
                    value={
                      focusedPolicyTrace.selected_action_id ??
                      focusedPolicyTrace.selected_slot_mapping?.action_id ??
                      "n/a"
                    }
                    tone="cool"
                  />
                </div>
                <dl className="definition-grid">
                  <Definition
                    label="Action Entropy"
                    value={nullableFixed(focusedPolicyTrace.action_entropy, 4)}
                  />
                  <Definition
                    label="Value Estimate"
                    value={nullableFixed(focusedPolicyTrace.value_estimate, 4)}
                  />
                  <Definition
                    label="Legal Slots"
                    value={
                      nullableNumber(
                        focusedPolicyTrace.legal_action_count ??
                          focusedLane.detail?.action_mask.legal_action_count,
                      )
                    }
                  />
                  <Definition
                    label="Mask Pressure"
                    value={nullableFixed(focusedPolicyTrace.mask_pressure, 4)}
                  />
                </dl>
                <div className="candidate-table">
                  <div className="candidate-table-head">
                    <span>Slot</span>
                    <span>Action</span>
                    <span>Probability</span>
                    <span>Logit</span>
                  </div>
                  {(focusedPolicyTrace.top_slots ?? []).map((slot) => (
                    <div className="candidate-row" key={`${slot.slot_index}-${slot.action_id ?? "slot"}`}>
                      <span>{slot.slot_index}</span>
                      <span>
                        {slot.action_id ??
                          slot.slot_mapping?.action_id ??
                          slot.slot_mapping?.action_ref ??
                          "padding"}
                      </span>
                      <span>{nullableFixed(slot.probability, 4)}</span>
                      <span>{nullableFixed(slot.logit, 4)}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="empty-copy">
                No inference trace is available for the focused replay at the current tick.
              </p>
            )}
          </section>

          <section className="panel stack-panel">
            <div className="panel-heading">
              <h2>Focused Episode State</h2>
              <p>
                Live bundle, planner identity, and replay health for the current focused lane.
              </p>
            </div>
            {focusedLane.detail ? (
              <>
                <dl className="definition-grid">
                  <Definition label="Planner" value={lanePlannerLabel(focusedLane.detail)} />
                  <Definition
                    label="Kind"
                    value={focusedLane.detail.planner_kind ?? "unknown"}
                  />
                  <Definition
                    label="Replay Source"
                    value={focusedLane.detail.episode_source ?? "api"}
                  />
                  <Definition label="Bundle" value={focusedLane.detail.bundle_id} />
                  <Definition
                    label="Replay Events"
                    value={String(focusedLane.events.length)}
                  />
                  <Definition
                    label="Tick"
                    value={String(activeHeroTick?.simTick ?? focusedLane.detail.sim_tick)}
                  />
                </dl>
                <p className="body-copy">
                  {focusedLane.detail.report_id
                    ? `Report: ${focusedLane.detail.report_id} ${
                        focusedLane.detail.report_split
                          ? `· ${focusedLane.detail.report_split}`
                          : ""
                      }`
                    : `Updated ${formatCompactTime(focusedLane.detail.updated_at_utc)}`}
                </p>
              </>
            ) : (
              <p className="empty-copy">Focused lane state appears after loading a replay.</p>
            )}
          </section>
        </aside>
      </section>
    </main>
  );
}

function ReplayLaneCard({
  compareEnabled,
  isFocused,
  lane,
  onClear,
  onFocus,
  onScrub,
  onStep,
}: {
  compareEnabled: boolean;
  isFocused: boolean;
  lane: ReplayLaneState;
  onClear: () => void;
  onFocus: () => void;
  onScrub: (nextIndex: number) => void;
  onStep: (delta: number) => void;
}) {
  const activeTick =
    lane.replayModel?.ticks[lane.replayTickIndex] ??
    lane.replayModel?.ticks[lane.replayModel.ticks.length - 1] ??
    null;
  const trace = normalizePolicyTrace(inferenceTraceForLane(lane));

  return (
    <article className={isFocused ? "lane-card is-focused" : "lane-card"}>
      <div className="lane-card-header">
        <div>
          <p className="eyebrow">{lane.label}</p>
          <h3>{lanePlannerLabel(lane.detail)}</h3>
        </div>
        <div className="mini-actions">
          <button className="timeline-action" onClick={onFocus} type="button">
            {isFocused ? "Focused" : "Focus Globe"}
          </button>
          <button
            className="timeline-action"
            disabled={!lane.detail}
            onClick={onClear}
            type="button"
          >
            Clear
          </button>
        </div>
      </div>

      {lane.errorMessage ? <p className="lane-error">{lane.errorMessage}</p> : null}
      {!lane.detail ? (
        <p className="empty-copy">
          {compareEnabled && lane.id === "compare"
            ? "Load a second replay to compare against the primary lane."
            : "Load a replay into this lane to inspect metrics and events."}
        </p>
      ) : (
        <>
          <div className="lane-metadata">
            <MetricChip
              label="Source"
              value={lane.detail.episode_source ?? "api"}
              tone={lane.detail.episode_source === "report" ? "cool" : "neutral"}
            />
            <MetricChip
              label="Utility"
              value={lane.metrics ? lane.metrics.mission_utility.toFixed(3) : "n/a"}
              tone={
                lane.metrics && lane.metrics.mission_utility < 0 ? "danger" : "accent"
              }
            />
          </div>
          <div className="mini-timeline">
            <div className="mini-actions">
              <button
                className="timeline-action"
                disabled={lane.replayTickIndex === 0}
                onClick={() => onStep(-1)}
                type="button"
              >
                Prev
              </button>
              <button
                className="timeline-action"
                disabled={
                  !lane.replayModel ||
                  lane.replayTickIndex >= lane.replayModel.ticks.length - 1
                }
                onClick={() => onStep(1)}
                type="button"
              >
                Next
              </button>
              <span className="timeline-readout">
                {activeTick ? formatReplayTime(activeTick.simTimeUtc) : "No time"}
              </span>
            </div>
            {lane.replayModel ? (
              <input
                className="scrubber"
                max={Math.max(lane.replayModel.ticks.length - 1, 0)}
                min={0}
                onChange={(event) => onScrub(Number(event.target.value))}
                type="range"
                value={lane.replayTickIndex}
              />
            ) : null}
          </div>
          <div className="metric-grid compact-grid">
            <MetricCard
              label="Useful Packets"
              value={String(lane.metrics?.useful_packet_count ?? 0)}
            />
            <MetricCard
              label="Cloud Waste"
              value={lane.metrics ? formatPercent(lane.metrics.cloud_waste_rate) : "n/a"}
            />
            <MetricCard
              label="Missed Urgent"
              value={
                lane.metrics
                  ? formatPercent(lane.metrics.missed_urgent_incident_rate)
                  : "n/a"
              }
              tone={
                lane.metrics && lane.metrics.missed_urgent_incident_rate > 0
                  ? "danger"
                  : "neutral"
              }
            />
            <MetricCard
              label="Downlink p90"
              value={
                lane.metrics
                  ? formatDuration(lane.metrics.downlink_latency_seconds.p90)
                  : "n/a"
              }
            />
          </div>
          <div className="lane-events">
            {visibleImportantEvents(lane, 4).map((event) => (
              <article className="event-card compact-event" key={event.event_id}>
                <div className="event-header">
                  <span className={`event-type tone-${eventTone(event.event_type)}`}>
                    {event.event_type.replaceAll("_", " ")}
                  </span>
                  <span className="event-time">{formatReplayTime(event.sim_time_utc)}</span>
                </div>
                <p className="event-body">{describeEvent(event)}</p>
              </article>
            ))}
          </div>
          {trace ? (
            <div className="lane-trace-summary">
              <Definition
                label="Selected Slot"
                value={nullableNumber(trace.selected_slot)}
              />
              <Definition
                label="Entropy"
                value={nullableFixed(trace.action_entropy, 4)}
              />
              <Definition
                label="Legal Slots"
                value={nullableNumber(trace.legal_action_count)}
              />
              <Definition
                label="Value"
                value={nullableFixed(trace.value_estimate, 4)}
              />
            </div>
          ) : null}
        </>
      )}
    </article>
  );
}

function Definition({ label, value }: { label: string; value: string }) {
  return (
    <div className="definition-item">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function MetricChip({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "warm" | "accent" | "cool" | "danger";
}) {
  return (
    <div className={`metric-chip tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricCard({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "accent" | "danger";
}) {
  return (
    <article className={`metric-card tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function LegendSwatch({
  tone,
  label,
}: {
  tone: "warm" | "accent" | "cool" | "danger";
  label: string;
}) {
  return (
    <div className="legend-item">
      <span className={`legend-dot tone-${tone}`} />
      <span>{label}</span>
    </div>
  );
}

function countTargetsByState(
  targetStates: Record<string, string> | null | undefined,
  state: string,
): number {
  return Object.values(targetStates ?? {}).filter((value) => value === state).length;
}

function readRouteDefaults(): {
  scenarioId: string;
  baselineId: string;
  episodeId: string;
} {
  const params = new URLSearchParams(window.location.search);
  return {
    scenarioId: params.get("scenario") ?? "",
    baselineId: params.get("baseline") ?? "",
    episodeId: params.get("episode") ?? "",
  };
}

function writeRouteDefaults({
  scenarioId,
  baselineId,
  episodeId,
}: {
  scenarioId: string;
  baselineId: string;
  episodeId: string;
}): void {
  const params = new URLSearchParams(window.location.search);
  setQueryParam(params, "scenario", scenarioId);
  setQueryParam(params, "baseline", baselineId);
  setQueryParam(params, "episode", episodeId);
  const nextSearch = params.toString();
  const nextUrl = nextSearch
    ? `${window.location.pathname}?${nextSearch}`
    : window.location.pathname;
  window.history.replaceState(null, "", nextUrl);
}

function setQueryParam(params: URLSearchParams, key: string, value: string): void {
  if (value) {
    params.set(key, value);
    return;
  }
  params.delete(key);
}

function visibleImportantEvents(
  lane: ReplayLaneState,
  limit = 10,
): ReplayEvent[] {
  const activeEventIndex =
    lane.replayModel?.ticks[lane.replayTickIndex]?.currentEvents[
      lane.replayModel.ticks[lane.replayTickIndex].currentEvents.length - 1
    ]?.event_index ?? -1;
  return (
    lane.replayModel?.importantEvents
      .filter((event) => event.event_index <= activeEventIndex)
      .slice(-limit)
      .reverse() ?? []
  );
}

function inferenceTraceForLane(
  lane: ReplayLaneState,
): EpisodeInferenceTraceStep | null {
  if (!lane.inference?.traces.length || !lane.replayModel) {
    return null;
  }
  const activeEventIndex =
    lane.replayModel.ticks[lane.replayTickIndex]?.currentEvents[
      lane.replayModel.ticks[lane.replayTickIndex].currentEvents.length - 1
    ]?.event_index ?? Number.NEGATIVE_INFINITY;
  const candidates = lane.inference.traces.filter(
    (trace) => trace.event_index <= activeEventIndex,
  );
  return candidates[candidates.length - 1] ?? null;
}

function normalizePolicyTrace(trace: EpisodeInferenceTraceStep | null): PolicyTrace | null {
  if (!trace) {
    return null;
  }
  const raw = (trace.planner_trace.policy_trace ??
    (trace.planner_trace as PolicyTrace)) as PolicyTrace;
  return raw ?? null;
}

function lanePlannerLabel(detail: Pick<EpisodeSummary, "display_name" | "planner_id"> | null | undefined): string {
  return detail?.display_name || detail?.planner_id || "No replay";
}

function laneLabel(laneId: LaneId): string {
  return laneId === "primary" ? "Primary" : "Compare";
}

function findReportEpisodeForNotable(
  reportEpisodes: ReportEpisodeSummary[],
  notableEpisode: ReportNotableEpisode,
): ReportEpisodeSummary | undefined {
  return reportEpisodes.find(
    (episode) =>
      episode.bundle_id === notableEpisode.bundle_id &&
      (notableEpisode.split ? episode.split === notableEpisode.split : true) &&
      (notableEpisode.planner_key ? episode.planner_key === notableEpisode.planner_key : true),
  );
}

function notableEpisodeKey(episode: ReportNotableEpisode): string {
  return [
    episode.category,
    episode.planner_key ?? "all",
    episode.bundle_id,
    episode.split ?? "all",
  ].join(":");
}

function formatWindow(window: ScenarioBundle["time_window"] | null | undefined): string {
  if (!window) {
    return "n/a";
  }
  return `${window.start_time_utc.slice(0, 16)}Z -> ${window.end_time_utc.slice(11, 16)}Z`;
}

function formatReplayTime(value: string | null | undefined): string {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  return date.toLocaleString(undefined, {
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
    timeZone: "UTC",
  });
}

function formatCompactTime(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
  });
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDuration(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return "n/a";
  }
  if (value < 60) {
    return `${value.toFixed(0)}s`;
  }
  return `${(value / 60).toFixed(1)}m`;
}

function describeEvent(event: ReplayEvent): string {
  if (event.event_type === "action_selected") {
    return `${String(event.payload.action_type)} -> ${String(event.payload.action_ref)}`;
  }
  if (event.event_type === "observation_executed") {
    const usable = event.payload.usable === false ? "degraded" : "usable";
    return `${String(event.payload.opportunity_id)} on ${String(event.payload.target_cell_id)} (${usable})`;
  }
  if (event.event_type === "downlink_executed") {
    return `${String(event.payload.window_id)} delivered ${Number(event.payload.delivered_volume_mb ?? 0).toFixed(0)} MB`;
  }
  if (event.event_type === "incident_packet_emitted") {
    return `${String(event.payload.incident_id)} packetized from ${String(event.payload.observation_opportunity_id)}`;
  }
  if (event.event_type === "episode_started") {
    return `planner ${String(event.payload.planner_id)} seed ${String(event.payload.episode_seed)}`;
  }
  if (event.event_type === "episode_ended") {
    return `utility ${Number(event.payload.mission_utility ?? 0).toFixed(3)}`;
  }
  return JSON.stringify(event.payload);
}

function eventTone(eventType: string): "warm" | "accent" | "cool" | "danger" | "neutral" {
  if (eventType === "observation_executed") {
    return "warm";
  }
  if (eventType === "downlink_executed" || eventType === "incident_packet_emitted") {
    return "cool";
  }
  if (eventType === "episode_ended") {
    return "danger";
  }
  if (eventType === "action_selected") {
    return "accent";
  }
  return "neutral";
}

function nullableFixed(value: number | null | undefined, digits: number): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function nullableNumber(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? String(value) : "n/a";
}

function signedFixed(value: number, digits: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function errorToMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.payload?.message ?? error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected error";
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

export default App;
