import type {
  ApiErrorPayload,
  BaselineDescriptor,
  BaselineRunDetail,
  BaselineRunResponse,
  CzmlPacket,
  DemoDefaults,
  EpisodeDetail,
  EpisodeInferenceTraceResponse,
  EpisodeMetrics,
  EpisodeSummary,
  ModelDescriptor,
  ModelRunDetail,
  ModelRunResponse,
  ReplayEvent,
  ReportDetail,
  ReportSummary,
  ScenarioBundle,
  ScenarioPreview,
  ScenarioSummary,
} from "./types";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export class ApiError extends Error {
  payload?: ApiErrorPayload;
  status: number;

  constructor(message: string, status: number, payload?: ApiErrorPayload) {
    super(message);
    this.name = "ApiError";
    this.payload = payload;
    this.status = status;
  }
}

function apiBaseUrl(): string {
  return (import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL).replace(/\/$/, "");
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl()}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    let payload: ApiErrorPayload | undefined;
    try {
      payload = (await response.json()) as ApiErrorPayload;
    } catch {
      payload = undefined;
    }
    throw new ApiError(
      payload?.message ?? `Request failed with status ${response.status}`,
      response.status,
      payload,
    );
  }
  return (await response.json()) as T;
}

async function requestText(path: string, init?: RequestInit): Promise<string> {
  const response = await fetch(`${apiBaseUrl()}${path}`, init);
  if (!response.ok) {
    let payload: ApiErrorPayload | undefined;
    try {
      payload = (await response.json()) as ApiErrorPayload;
    } catch {
      payload = undefined;
    }
    throw new ApiError(
      payload?.message ?? `Request failed with status ${response.status}`,
      response.status,
      payload,
    );
  }
  return response.text();
}

function parseNdjson<T>(payload: string): T[] {
  return payload
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

export async function listScenarios(): Promise<ScenarioSummary[]> {
  return requestJson<ScenarioSummary[]>("/v1/scenarios");
}

export async function getDemoDefaults(): Promise<DemoDefaults> {
  return requestJson<DemoDefaults>("/v1/demo/defaults");
}

export async function getScenario(bundleId: string): Promise<ScenarioBundle> {
  return requestJson<ScenarioBundle>(`/v1/scenarios/${encodeURIComponent(bundleId)}`);
}

export async function getScenarioPreview(bundleId: string): Promise<ScenarioPreview> {
  return requestJson<ScenarioPreview>(
    `/v1/scenarios/${encodeURIComponent(bundleId)}/preview`,
  );
}

export async function getScenarioTrajectoryCzml(
  bundleId: string,
  stepSeconds?: number,
): Promise<CzmlPacket[]> {
  const search = stepSeconds ? `?step_seconds=${stepSeconds}` : "";
  return requestJson<CzmlPacket[]>(
    `/v1/scenarios/${encodeURIComponent(bundleId)}/trajectory-czml${search}`,
  );
}

export async function listBaselines(): Promise<BaselineDescriptor[]> {
  return requestJson<BaselineDescriptor[]>("/v1/baselines");
}

export async function listModels(): Promise<ModelDescriptor[]> {
  return requestJson<ModelDescriptor[]>("/v1/models");
}

export async function listEpisodes(): Promise<EpisodeSummary[]> {
  return requestJson<EpisodeSummary[]>("/v1/episodes");
}

export async function getEpisode(episodeId: string): Promise<EpisodeDetail> {
  return requestJson<EpisodeDetail>(`/v1/episodes/${encodeURIComponent(episodeId)}`);
}

export async function getEpisodeMetrics(episodeId: string): Promise<EpisodeMetrics> {
  return requestJson<EpisodeMetrics>(
    `/v1/episodes/${encodeURIComponent(episodeId)}/metrics`,
  );
}

export async function getEpisodeEvents(episodeId: string): Promise<ReplayEvent[]> {
  const ndjson = await requestText(`/v1/episodes/${encodeURIComponent(episodeId)}/events`);
  return parseNdjson<ReplayEvent>(ndjson);
}

export async function getEpisodeInferenceTraces(
  episodeId: string,
): Promise<EpisodeInferenceTraceResponse> {
  return requestJson<EpisodeInferenceTraceResponse>(
    `/v1/episodes/${encodeURIComponent(episodeId)}/inference-traces`,
  );
}

export async function listReports(): Promise<ReportSummary[]> {
  return requestJson<ReportSummary[]>("/v1/reports");
}

export async function getReport(reportId: string): Promise<ReportDetail> {
  return requestJson<ReportDetail>(`/v1/reports/${encodeURIComponent(reportId)}`);
}

export async function getReportEpisode(reportEpisodeId: string): Promise<EpisodeDetail> {
  return requestJson<EpisodeDetail>(
    `/v1/report-episodes/${encodeURIComponent(reportEpisodeId)}`,
  );
}

export async function getReportEpisodeEvents(
  reportEpisodeId: string,
): Promise<ReplayEvent[]> {
  const ndjson = await requestText(
    `/v1/report-episodes/${encodeURIComponent(reportEpisodeId)}/events`,
  );
  return parseNdjson<ReplayEvent>(ndjson);
}

export async function getReportEpisodeInferenceTraces(
  reportEpisodeId: string,
): Promise<EpisodeInferenceTraceResponse> {
  return requestJson<EpisodeInferenceTraceResponse>(
    `/v1/report-episodes/${encodeURIComponent(reportEpisodeId)}/inference-traces`,
  );
}

export async function runBaseline(
  baselineId: string,
  bundleId: string,
  simulationSeed: number,
): Promise<BaselineRunResponse> {
  return requestJson<BaselineRunResponse>(
    `/v1/baselines/${encodeURIComponent(baselineId)}/run`,
    {
      method: "POST",
      body: JSON.stringify({
        bundle_id: bundleId,
        simulation_seed: simulationSeed,
      }),
    },
  );
}

export async function getBaselineRun(jobId: string): Promise<BaselineRunDetail> {
  return requestJson<BaselineRunDetail>(
    `/v1/baseline-runs/${encodeURIComponent(jobId)}`,
  );
}

export async function runModel(
  modelKey: string,
  bundleId: string,
  simulationSeed: number,
  includeInferenceTraces = true,
): Promise<ModelRunResponse> {
  return requestJson<ModelRunResponse>(
    `/v1/models/${encodeURIComponent(modelKey)}/run`,
    {
      method: "POST",
      body: JSON.stringify({
        bundle_id: bundleId,
        simulation_seed: simulationSeed,
        include_inference_traces: includeInferenceTraces,
      }),
    },
  );
}

export async function getModelRun(jobId: string): Promise<ModelRunDetail> {
  return requestJson<ModelRunDetail>(`/v1/model-runs/${encodeURIComponent(jobId)}`);
}
