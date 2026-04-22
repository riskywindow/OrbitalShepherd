from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import Body, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from orbital_shepherd_api.models import (
    BaselineDescriptor,
    BaselineRunDetail,
    BaselineRunRequest,
    BaselineRunResponse,
    DemoDefaultsResponse,
    EpisodeDetail,
    EpisodeInferenceTraceResponse,
    EpisodeMetricsResponse,
    EpisodeSummary,
    ErrorResponse,
    HealthResponse,
    ModelDetail,
    ModelDescriptor,
    ModelRunDetail,
    ModelRunRequest,
    ModelRunResponse,
    ReportDetail,
    ReportSummary,
    ScenarioPreview,
    ScenarioRegistrationResponse,
    ScenarioSummary,
    StartEpisodeRequest,
    StartEpisodeResponse,
    StepEpisodeRequest,
    StepEpisodeResponse,
)
from orbital_shepherd_api.service import ApiError, Phase1ApiService
from orbital_shepherd_api.settings import ApiSettings
from orbital_shepherd_contracts import ScenarioBundle


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    service = Phase1ApiService(settings=settings)
    app = FastAPI(
        title="Orbital Shepherd Planner API",
        version="0.1.0",
        summary="Phase 1 FastAPI service aligned with the Phase 0 planner API contract.",
    )
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        payload = ErrorResponse(error=exc.error, message=exc.message, details=exc.details)
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        payload = ErrorResponse(
            error="invalid_request",
            message="request validation failed",
            details={"errors": exc.errors()},
        )
        return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))

    @app.exception_handler(ValidationError)
    async def handle_model_validation(_: Request, exc: ValidationError) -> JSONResponse:
        payload = ErrorResponse(
            error="validation_error",
            message="response or contract validation failed",
            details={"errors": exc.errors()},
        )
        return JSONResponse(status_code=500, content=payload.model_dump(mode="json"))

    @app.get("/v1/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/v1/demo/defaults", response_model=DemoDefaultsResponse)
    async def get_demo_defaults() -> DemoDefaultsResponse:
        return service.get_demo_defaults()

    @app.post(
        "/v1/scenarios",
        response_model=ScenarioRegistrationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def register_scenario(
        payload: Annotated[dict[str, Any], Body(...)],
    ) -> ScenarioRegistrationResponse:
        bundle = service.register_scenario(payload)
        return ScenarioRegistrationResponse(bundle_id=bundle.bundle_id)

    @app.get("/v1/scenarios", response_model=list[ScenarioSummary])
    async def list_scenarios() -> list[ScenarioSummary]:
        return service.list_scenarios()

    @app.get("/v1/scenarios/{bundle_id}", response_model=ScenarioBundle)
    async def get_scenario(bundle_id: str) -> ScenarioBundle:
        return service.get_scenario(bundle_id)

    @app.get("/v1/scenarios/{bundle_id}/preview", response_model=ScenarioPreview)
    async def get_scenario_preview(bundle_id: str) -> ScenarioPreview:
        return service.get_scenario_preview(bundle_id)

    @app.get("/v1/scenarios/{bundle_id}/trajectory-czml")
    async def get_scenario_trajectory_czml(
        bundle_id: str,
        step_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        return service.get_scenario_trajectory_czml(
            bundle_id,
            step_seconds=step_seconds,
        )

    @app.post(
        "/v1/episodes",
        response_model=StartEpisodeResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def start_episode(request: StartEpisodeRequest) -> StartEpisodeResponse:
        record = service.start_episode(
            bundle_id=request.bundle_id,
            planner_id=request.planner_id,
            simulation_seed=request.simulation_seed,
        )
        return StartEpisodeResponse(
            episode_id=record.episode_id,
            runtime_episode_id=record.runtime_episode_id,
        )

    @app.get("/v1/episodes", response_model=list[EpisodeSummary])
    async def list_episodes() -> list[EpisodeSummary]:
        return service.list_episodes()

    @app.get("/v1/episodes/{episode_id}", response_model=EpisodeDetail)
    async def get_episode(episode_id: str) -> EpisodeDetail:
        return service.get_episode(episode_id)

    @app.post("/v1/episodes/{episode_id}/step", response_model=StepEpisodeResponse)
    async def step_episode(episode_id: str, request: StepEpisodeRequest) -> StepEpisodeResponse:
        result = service.step_episode(episode_id, request.action)
        return StepEpisodeResponse(
            episode_id=result.record.episode_id,
            runtime_episode_id=result.record.runtime_episode_id,
            sim_tick=result.record.sim_tick,
            terminated=result.record.terminated,
            truncated=result.record.truncated,
            reward=result.snapshot.reward,
            mission_utility=result.record.mission_utility,
            observation=result.record.latest_observation,
            action_mask=result.record.latest_action_mask,
            events=result.snapshot.latest_events,
        )

    @app.get("/v1/episodes/{episode_id}/events")
    async def get_episode_events(episode_id: str) -> StreamingResponse:
        ndjson_payload = service.episode_events_ndjson(episode_id)
        return StreamingResponse(
            _iter_ndjson(ndjson_payload),
            media_type="application/x-ndjson",
        )

    @app.get("/v1/episodes/{episode_id}/metrics", response_model=EpisodeMetricsResponse)
    async def get_episode_metrics(episode_id: str) -> EpisodeMetricsResponse:
        return service.get_episode_metrics(episode_id)

    @app.get(
        "/v1/episodes/{episode_id}/inference-traces",
        response_model=EpisodeInferenceTraceResponse,
    )
    async def get_episode_inference_traces(episode_id: str) -> EpisodeInferenceTraceResponse:
        return service.get_episode_inference_traces(episode_id)

    @app.get("/v1/reports", response_model=list[ReportSummary])
    async def list_reports() -> list[ReportSummary]:
        return service.list_reports()

    @app.get("/v1/reports/{report_id}", response_model=ReportDetail)
    async def get_report(report_id: str) -> ReportDetail:
        return service.get_report(report_id)

    @app.get("/v1/report-episodes/{report_episode_id}", response_model=EpisodeDetail)
    async def get_report_episode(report_episode_id: str) -> EpisodeDetail:
        return service.get_report_episode(report_episode_id)

    @app.get("/v1/report-episodes/{report_episode_id}/events")
    async def get_report_episode_events(report_episode_id: str) -> StreamingResponse:
        ndjson_payload = service.report_episode_events_ndjson(report_episode_id)
        return StreamingResponse(
            _iter_ndjson(ndjson_payload),
            media_type="application/x-ndjson",
        )

    @app.get(
        "/v1/report-episodes/{report_episode_id}/inference-traces",
        response_model=EpisodeInferenceTraceResponse,
    )
    async def get_report_episode_inference_traces(
        report_episode_id: str,
    ) -> EpisodeInferenceTraceResponse:
        return service.get_report_episode_inference_traces(report_episode_id)

    @app.get("/v1/baselines", response_model=list[BaselineDescriptor])
    async def list_baselines() -> list[BaselineDescriptor]:
        return service.list_baselines()

    @app.get("/v1/models", response_model=list[ModelDescriptor])
    async def list_models() -> list[ModelDescriptor]:
        return service.list_models()

    @app.get("/v1/models/{model_key}", response_model=ModelDetail)
    async def get_model(model_key: str) -> ModelDetail:
        return service.get_model(model_key)

    @app.post(
        "/v1/models/{model_key}/run",
        response_model=ModelRunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def run_model(model_key: str, request: ModelRunRequest) -> ModelRunResponse:
        result = service.run_model(model_key, request)
        return ModelRunResponse(
            job_id=result.job_id,
            status=result.status,
            episode_id=result.episode_id,
        )

    @app.post(
        "/v1/baselines/{baseline_id}/run",
        response_model=BaselineRunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def run_baseline(
        baseline_id: str,
        request: BaselineRunRequest,
    ) -> BaselineRunResponse:
        result = service.run_baseline(baseline_id, request)
        return BaselineRunResponse(
            job_id=result.job_id,
            status=result.status,
            episode_id=result.episode_id,
        )

    @app.get("/v1/baseline-runs/{job_id}", response_model=BaselineRunDetail)
    async def get_baseline_run(job_id: str) -> BaselineRunDetail:
        return service.get_baseline_run(job_id)

    @app.get("/v1/model-runs/{job_id}", response_model=ModelRunDetail)
    async def get_model_run(job_id: str) -> ModelRunDetail:
        return service.get_model_run(job_id)

    return app


def _iter_ndjson(payload: str) -> Iterator[bytes]:
    for line in payload.splitlines(keepends=True):
        yield line.encode("utf-8")
