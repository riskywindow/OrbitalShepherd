from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from orbital_shepherd_api.models import (
    BaselineDescriptor,
    BaselineRunDetail,
    BaselineRunRequest,
    DemoDefaultsResponse,
    EpisodeDetail,
    EpisodeInferenceTraceResponse,
    EpisodeInferenceTraceStep,
    EpisodeMetricsResponse,
    EpisodeRecord,
    EpisodeSummary,
    ModelDescriptor,
    ModelDetail,
    ModelRunDetail,
    ModelRunRequest,
    ReportDetail,
    ReportEpisodeSummary,
    ReportNotableEpisode,
    ReportSummary,
    ScenarioPreview,
    ScenarioSummary,
    scenario_summary_from_bundle,
)
from orbital_shepherd_api.settings import ApiSettings
from orbital_shepherd_benchmark import (
    PlannerEpisodeContext,
    build_planner,
    compute_episode_metrics,
    planner_registry,
)
from orbital_shepherd_benchmark.planners import planner_descriptions, planner_runtime_metadata
from orbital_shepherd_contracts import (
    ReplayEvent,
    ScenarioBundle,
    compile_scenario_bundle,
    validate_canonical_bundle,
    validate_canonical_replay_events,
    validate_phase0_instance,
)
from orbital_shepherd_core import (
    canonical_json_dumps,
    format_utc_timestamp,
    stable_id,
    stable_token,
)
from orbital_shepherd_env_runtime import EnvRuntimeConfig, OrbitalEnv, replay_events_to_ndjson
from orbital_shepherd_ephemeris import DeterministicKeplerPropagationBackend
from orbital_shepherd_ephemeris.models import OrbitAssetBundle, SatelliteStateSample
from orbital_shepherd_policy_models import PolicyModelRegistry


class ApiError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        error: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message
        self.details = details


@dataclass(frozen=True, slots=True)
class EpisodeRuntimeSnapshot:
    runtime_episode_id: str
    observation: dict[str, Any]
    action_mask: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    mission_utility: float
    replay_events: list[ReplayEvent]
    latest_events: list[dict[str, Any]]
    metrics: EpisodeMetricsResponse


@dataclass(frozen=True, slots=True)
class EpisodeStepResult:
    record: EpisodeRecord
    snapshot: EpisodeRuntimeSnapshot


class Phase1ApiService:
    def __init__(self, settings: ApiSettings | None = None) -> None:
        self.settings = settings or ApiSettings()
        self.settings.ensure_directories()
        self.model_registry = PolicyModelRegistry(
            manifest_roots=(self.settings.training_manifest_dir,),
            checkpoint_roots=(self.settings.training_checkpoint_dir,),
        )

    def list_scenarios(self) -> list[ScenarioSummary]:
        bundles_by_id: dict[str, ScenarioBundle] = {}
        for path in self._scenario_paths():
            bundle = self._load_bundle_from_path(path)
            bundles_by_id.setdefault(bundle.bundle_id, bundle)
        return [
            scenario_summary_from_bundle(bundle)
            for bundle in sorted(bundles_by_id.values(), key=lambda item: item.bundle_id)
        ]

    def get_demo_defaults(self) -> DemoDefaultsResponse:
        registry = planner_registry()
        default_baseline_id = (
            "urgency_greedy" if "urgency_greedy" in registry else next(iter(registry))
        )
        defaults_path = self._preferred_demo_defaults_path()
        if defaults_path is not None:
            defaults = DemoDefaultsResponse.model_validate(
                self._read_json(defaults_path)
            )
            cleaned = defaults.model_copy(
                update={
                    "episode_id": (
                        defaults.episode_id
                        if defaults.episode_id and self._episode_path(defaults.episode_id).exists()
                        else None
                    ),
                    "replay_path": (
                        defaults.replay_path
                        if defaults.replay_path and Path(defaults.replay_path).exists()
                        else None
                    ),
                    "benchmark_summary_path": (
                        defaults.benchmark_summary_path
                        if defaults.benchmark_summary_path
                        and Path(defaults.benchmark_summary_path).exists()
                        else None
                    ),
                }
            )
            return cleaned.model_copy(
                update={
                    "baseline_id": cleaned.baseline_id or default_baseline_id,
                    "ui_query": self._demo_ui_query(
                        bundle_id=cleaned.bundle_id,
                        baseline_id=cleaned.baseline_id or default_baseline_id,
                        episode_id=cleaned.episode_id,
                    ),
                }
            )

        scenarios = self.list_scenarios()
        bundle_id = scenarios[0].bundle_id if scenarios else None
        return DemoDefaultsResponse(
            bundle_id=bundle_id,
            baseline_id=default_baseline_id,
            ui_query=self._demo_ui_query(
                bundle_id=bundle_id,
                baseline_id=default_baseline_id,
                episode_id=None,
            ),
        )

    def _preferred_demo_defaults_path(self) -> Path | None:
        candidates = (
            self.settings.phase2_demo_defaults_path,
            self.settings.demo_defaults_path,
        )
        for path in candidates:
            if path.exists():
                return path
        return None

    def register_scenario(self, payload: dict[str, Any]) -> ScenarioBundle:
        bundle = self._coerce_bundle(payload)
        path = self._scenario_path(bundle.bundle_id)
        if path.exists():
            existing = self._load_bundle_from_path(path)
            if existing.bundle_fingerprint != bundle.bundle_fingerprint:
                raise ApiError(
                    status_code=409,
                    error="scenario_conflict",
                    message=(
                        f"bundle_id {bundle.bundle_id!r} is already registered with a different "
                        "bundle_fingerprint"
                    ),
                    details={
                        "existing_bundle_fingerprint": existing.bundle_fingerprint,
                        "incoming_bundle_fingerprint": bundle.bundle_fingerprint,
                    },
                )
        self._write_json(path, bundle.model_dump(mode="json", exclude_none=True))
        return bundle

    def get_scenario(self, bundle_id: str) -> ScenarioBundle:
        path = self._scenario_path(bundle_id)
        if not path.exists():
            raise ApiError(
                status_code=404,
                error="scenario_not_found",
                message=f"unknown bundle_id {bundle_id!r}",
            )
        return self._load_bundle_from_path(path)

    def get_scenario_preview(self, bundle_id: str) -> ScenarioPreview:
        bundle = self.get_scenario(bundle_id)
        return ScenarioPreview(
            bundle_id=bundle.bundle_id,
            benchmark_id=bundle.benchmark_id,
            scenario_family=bundle.scenario_family,
            simulation_seed=bundle.simulation_seed,
            time_window=bundle.time_window.model_dump(mode="json"),
            counts={
                "satellites": len(bundle.satellites),
                "ground_stations": len(bundle.ground_stations),
                "target_cells": len(bundle.target_cells),
                "incidents": len(bundle.incidents),
                "observation_opportunities": len(bundle.observation_opportunities),
                "downlink_windows": len(bundle.downlink_windows),
            },
            target_cells=[
                {
                    "target_cell_id": target.target_cell_id,
                    "h3_cell": target.h3_cell,
                    "region_name": target.region_name,
                    "static_value": target.static_value,
                    "priority_class": target.priority_class,
                    "centroid": target.centroid.model_dump(mode="json"),
                }
                for target in bundle.target_cells
            ],
            incidents=[
                {
                    "incident_id": incident.incident_id,
                    "target_cell_id": incident.target_cell_id,
                    "urgency_score": incident.urgency_score,
                    "confidence": incident.confidence,
                    "state": incident.state,
                    "ignition_time_utc": format_utc_timestamp(incident.ignition_time_utc),
                }
                for incident in bundle.incidents
            ],
        )

    def get_scenario_trajectory_czml(
        self,
        bundle_id: str,
        *,
        step_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        bundle = self.get_scenario(bundle_id)
        backend = DeterministicKeplerPropagationBackend()
        orbit_bundle_id = str(
            bundle.config.model_dump(mode="json").get("orbit_asset_bundle_id", "")
        ).strip()
        if not orbit_bundle_id:
            raise ApiError(
                status_code=404,
                error="orbit_bundle_not_configured",
                message=f"scenario {bundle_id!r} does not declare an orbit asset bundle",
            )
        orbit_bundle = backend.load_orbit_asset_snapshot(
            self._orbit_asset_bundle_path(orbit_bundle_id)
        )
        sample_step_seconds = step_seconds or bundle.decision_interval_seconds
        if sample_step_seconds <= 0:
            raise ApiError(
                status_code=400,
                error="invalid_step_seconds",
                message="step_seconds must be a positive integer",
            )
        samples = backend.sample_satellite_states(
            orbit_bundle,
            start_time_utc=bundle.time_window.start_time_utc,
            end_time_utc=bundle.time_window.end_time_utc,
            step_seconds=sample_step_seconds,
            satellite_ids=[satellite.satellite_id for satellite in bundle.satellites],
        )
        return self._build_satellite_czml(
            bundle=bundle,
            orbit_bundle=orbit_bundle,
            samples=samples,
            step_seconds=sample_step_seconds,
        )

    def list_episodes(self) -> list[EpisodeSummary]:
        return [
            self._episode_summary(self._load_episode_record(path))
            for path in sorted(self.settings.episode_dir.glob("*.json"))
        ]

    def get_episode(self, episode_id: str) -> EpisodeDetail:
        record = self._load_episode_record(self._episode_path(episode_id))
        return EpisodeDetail(
            **self._episode_summary(record).model_dump(),
            action_history_length=len(record.action_history),
            latest_observation=record.latest_observation,
            action_mask=record.latest_action_mask,
            metrics=record.metrics,
        )

    def start_episode(
        self,
        *,
        bundle_id: str,
        planner_id: str,
        simulation_seed: int,
    ) -> EpisodeRecord:
        bundle = self.get_scenario(bundle_id)
        created_at = datetime.now(UTC)
        episode_id = stable_id(
            "episode",
            bundle.bundle_id,
            planner_id,
            f"seed-{simulation_seed}",
            suffix=format_utc_timestamp(created_at),
        )
        snapshot = self._replay_episode(
            bundle=bundle,
            planner_id=planner_id,
            simulation_seed=simulation_seed,
            episode_id=episode_id,
            action_history=[],
        )
        record = EpisodeRecord(
            episode_id=episode_id,
            runtime_episode_id=snapshot.runtime_episode_id,
            bundle_id=bundle.bundle_id,
            planner_id=planner_id,
            planner_kind="interactive",
            planner_metadata=None,
            simulation_seed=simulation_seed,
            created_at_utc=created_at,
            updated_at_utc=created_at,
            sim_tick=int(snapshot.observation["sim_tick"]),
            terminated=snapshot.terminated,
            truncated=snapshot.truncated,
            mission_utility=snapshot.mission_utility,
            replay_event_count=len(snapshot.replay_events),
            action_history=[],
            replay_path=str(self._episode_replay_path(episode_id)),
            latest_observation=snapshot.observation,
            latest_action_mask=snapshot.action_mask,
            metrics=snapshot.metrics,
        )
        self._persist_episode_record(record, snapshot.replay_events)
        return record

    def step_episode(self, episode_id: str, action: dict[str, Any]) -> EpisodeStepResult:
        record = self._load_episode_record(self._episode_path(episode_id))
        if record.terminated or record.truncated:
            raise ApiError(
                status_code=409,
                error="episode_closed",
                message=f"episode {episode_id!r} can no longer accept step requests",
            )
        normalized_action = self._normalize_action(action)
        action_history = [*record.action_history, normalized_action]
        bundle = self.get_scenario(record.bundle_id)
        snapshot = self._replay_episode(
            bundle=bundle,
            planner_id=record.planner_id,
            simulation_seed=record.simulation_seed,
            episode_id=record.episode_id,
            action_history=action_history,
        )
        updated = record.model_copy(
            update={
                "runtime_episode_id": snapshot.runtime_episode_id,
                "updated_at_utc": datetime.now(UTC),
                "sim_tick": int(snapshot.observation["sim_tick"]),
                "terminated": snapshot.terminated,
                "truncated": snapshot.truncated,
                "mission_utility": snapshot.mission_utility,
                "replay_event_count": len(snapshot.replay_events),
                "action_history": action_history,
                "latest_observation": snapshot.observation,
                "latest_action_mask": snapshot.action_mask,
                "metrics": snapshot.metrics,
            }
        )
        self._persist_episode_record(updated, snapshot.replay_events)
        return EpisodeStepResult(record=updated, snapshot=snapshot)

    def episode_events_ndjson(self, episode_id: str) -> str:
        record = self._load_episode_record(self._episode_path(episode_id))
        replay_path = Path(record.replay_path)
        if not replay_path.exists():
            raise ApiError(
                status_code=404,
                error="replay_not_found",
                message=f"episode replay for {episode_id!r} is missing",
            )
        return replay_path.read_text(encoding="utf-8")

    def get_episode_metrics(self, episode_id: str) -> EpisodeMetricsResponse:
        return self._load_episode_record(self._episode_path(episode_id)).metrics

    def list_reports(self) -> list[ReportSummary]:
        reports: list[ReportSummary] = []
        for summary_path in sorted(self.settings.training_report_dir.glob("*/summary.json")):
            report_document = self._read_json(summary_path)
            if not isinstance(report_document.get("episodes"), list):
                continue
            reports.append(self._report_summary(summary_path, report_document))
        return reports

    def get_report(self, report_id: str) -> ReportDetail:
        summary_path = self._report_summary_path(report_id)
        report_document = self._read_json(summary_path)
        if not isinstance(report_document.get("episodes"), list):
            raise ApiError(
                status_code=404,
                error="report_not_found",
                message=f"report {report_id!r} is not an evaluation report",
            )
        notable_path = summary_path.with_name("notable_episodes.csv")
        notable_episodes = self._read_notable_episodes(notable_path)
        episodes = [
            self._report_episode_summary(
                report_id=report_id,
                report_summary_path=summary_path,
                episode_document=episode_document,
            )
            for episode_document in report_document["episodes"]
            if isinstance(episode_document, dict)
        ]
        return ReportDetail(
            **self._report_summary(summary_path, report_document).model_dump(),
            episodes=episodes,
            notable_episodes=notable_episodes,
        )

    def get_report_episode(self, report_episode_id: str) -> EpisodeDetail:
        report_id, report_summary_path, episode = self._find_report_episode(report_episode_id)
        replay_events = self._read_replay_events(Path(episode.replay_path))
        latest_event = replay_events[-1] if replay_events else None
        ended_payload = next(
            (
                event.payload
                for event in reversed(replay_events)
                if event.event_type == "episode_ended"
            ),
            {},
        )
        last_trace = next(
            (
                dict(event.payload.get("planner_trace", {}))
                for event in reversed(replay_events)
                if event.event_type == "action_selected" and "planner_trace" in event.payload
            ),
            {},
        )
        action_mask = self._action_mask_from_trace(
            report_episode_id=report_episode_id,
            planner_trace=last_trace,
        )
        report_updated_at = datetime.fromtimestamp(
            report_summary_path.stat().st_mtime,
            tz=UTC,
        )
        return EpisodeDetail(
            episode_id=report_episode_id,
            runtime_episode_id=episode.episode_id,
            bundle_id=episode.bundle_id,
            planner_id=episode.planner_key,
            display_name=episode.display_name,
            planner_kind=episode.planner_kind,
            planner_metadata={
                "report_id": report_id,
                "planner_version": episode.planner_version,
                "scenario_family": episode.scenario_family,
            },
            episode_source="report",
            report_id=report_id,
            report_episode_id=report_episode_id,
            report_split=episode.split,
            simulation_seed=episode.simulation_seed,
            created_at_utc=report_updated_at,
            updated_at_utc=report_updated_at,
            sim_tick=latest_event.sim_tick if latest_event is not None else 0,
            terminated=bool(ended_payload.get("terminated", True)),
            truncated=bool(ended_payload.get("truncated", False)),
            mission_utility=episode.metrics.mission_utility,
            replay_event_count=len(replay_events),
            action_history_length=episode.action_count,
            latest_observation={
                "sim_tick": latest_event.sim_tick if latest_event is not None else 0,
                "sim_time_utc": (
                    latest_event.sim_time_utc if latest_event is not None else report_updated_at
                ),
                "horizon_tick": latest_event.sim_tick if latest_event is not None else 0,
                "mission_utility": episode.metrics.mission_utility,
                "incidents": [],
                "action_mask": action_mask,
            },
            action_mask=action_mask,
            metrics=episode.metrics,
        )

    def report_episode_events_ndjson(self, report_episode_id: str) -> str:
        _, _, episode = self._find_report_episode(report_episode_id)
        replay_path = Path(episode.replay_path)
        if not replay_path.exists():
            raise ApiError(
                status_code=404,
                error="report_replay_not_found",
                message=f"report replay for {report_episode_id!r} is missing",
            )
        return replay_path.read_text(encoding="utf-8")

    def get_report_episode_inference_traces(
        self, report_episode_id: str
    ) -> EpisodeInferenceTraceResponse:
        report_id, _, episode = self._find_report_episode(report_episode_id)
        events = self._read_replay_events(Path(episode.replay_path))
        return self._inference_traces_from_events(
            episode_id=report_episode_id,
            planner_id=episode.planner_key,
            planner_kind=episode.planner_kind,
            replay_events=events,
            planner_metadata={"report_id": report_id},
        )

    def list_baselines(self) -> list[BaselineDescriptor]:
        baselines: list[BaselineDescriptor] = []
        descriptions = planner_descriptions()
        for baseline_id in sorted(planner_registry()):
            planner = build_planner(baseline_id)
            baselines.append(
                BaselineDescriptor(
                    baseline_id=baseline_id,
                    version=planner.metadata.version,
                    description=descriptions[baseline_id],
                )
            )
        return baselines

    def list_models(self) -> list[ModelDescriptor]:
        self.model_registry.refresh()
        return [self._model_descriptor(entry) for entry in self.model_registry.list_policies()]

    def get_model(self, model_key: str) -> ModelDetail:
        self.model_registry.refresh()
        try:
            entry = self.model_registry.get_policy(model_key)
        except KeyError as exc:
            raise ApiError(
                status_code=404,
                error="model_not_found",
                message=str(exc),
            ) from exc
        return ModelDetail(
            **self._model_descriptor(entry).model_dump(),
            benchmark_id=entry.benchmark_id,
            reward_id=entry.reward_id,
            run_id=entry.run_id,
            trainer_backend=entry.trainer_backend,
            framework=entry.framework,
            checkpoint_manifest_path=str(entry.checkpoint_manifest_path),
            checkpoint_path=str(entry.checkpoint_path),
            architecture=entry.architecture.to_dict(),
            metadata=entry.metadata,
            metrics=entry.metrics,
        )

    def run_model(self, model_key: str, request: ModelRunRequest) -> ModelRunDetail:
        self.model_registry.refresh()
        try:
            entry = self.model_registry.get_policy(model_key)
        except KeyError as exc:
            raise ApiError(
                status_code=404,
                error="model_not_found",
                message=str(exc),
            ) from exc
        bundle = self.get_scenario(request.bundle_id)
        created_at = datetime.now(UTC)
        job_id = stable_id(
            "job",
            "model",
            model_key,
            request.bundle_id,
            f"seed-{request.simulation_seed}",
            suffix=format_utc_timestamp(created_at),
        )
        episode_id = stable_id(
            "episode",
            request.bundle_id,
            "trained-policy",
            stable_token(entry.checkpoint_id, length=12),
            f"seed-{request.simulation_seed}",
            suffix=job_id,
        )
        record = ModelRunDetail(
            job_id=job_id,
            model_key=model_key,
            bundle_id=request.bundle_id,
            simulation_seed=request.simulation_seed,
            status="accepted",
            created_at_utc=created_at,
        )
        self._write_json(self._model_run_path(job_id), record.model_dump(mode="json"))

        try:
            runtime_record = self._run_trained_policy_episode(
                bundle=bundle,
                planner_id=entry.planner_id,
                simulation_seed=request.simulation_seed,
                episode_id=episode_id,
            )
        except Exception as exc:
            failed_record = record.model_copy(
                update={
                    "status": "failed",
                    "completed_at_utc": datetime.now(UTC),
                    "error_message": str(exc),
                }
            )
            self._write_json(self._model_run_path(job_id), failed_record.model_dump(mode="json"))
            raise

        completed = record.model_copy(
            update={
                "status": "completed",
                "completed_at_utc": datetime.now(UTC),
                "episode_id": runtime_record.episode_id,
                "runtime_episode_id": runtime_record.runtime_episode_id,
                "replay_path": runtime_record.replay_path,
                "metrics": runtime_record.metrics,
            }
        )
        self._write_json(self._model_run_path(job_id), completed.model_dump(mode="json"))
        return completed

    def get_model_run(self, job_id: str) -> ModelRunDetail:
        path = self._model_run_path(job_id)
        if not path.exists():
            raise ApiError(
                status_code=404,
                error="model_run_not_found",
                message=f"unknown model job_id {job_id!r}",
            )
        return ModelRunDetail.model_validate(self._read_json(path))

    def get_episode_inference_traces(self, episode_id: str) -> EpisodeInferenceTraceResponse:
        record = self._load_episode_record(self._episode_path(episode_id))
        replay_path = Path(record.replay_path)
        if not replay_path.exists():
            raise ApiError(
                status_code=404,
                error="replay_not_found",
                message=f"episode replay for {episode_id!r} is missing",
            )
        events = self._read_replay_events(replay_path)
        return self._inference_traces_from_events(
            episode_id=episode_id,
            planner_id=record.planner_id,
            planner_kind=record.planner_kind,
            replay_events=events,
            planner_metadata=record.planner_metadata,
        )

    def run_baseline(self, baseline_id: str, request: BaselineRunRequest) -> BaselineRunDetail:
        try:
            planner = build_planner(baseline_id)
        except ValueError as exc:
            raise ApiError(
                status_code=404,
                error="baseline_not_found",
                message=str(exc),
            ) from exc

        bundle = self.get_scenario(request.bundle_id)
        created_at = datetime.now(UTC)
        job_id = stable_id(
            "job",
            "baseline",
            baseline_id,
            request.bundle_id,
            f"seed-{request.simulation_seed}",
            suffix=format_utc_timestamp(created_at),
        )
        episode_id = stable_id(
            "episode",
            request.bundle_id,
            baseline_id,
            f"seed-{request.simulation_seed}",
            suffix=job_id,
        )
        record = BaselineRunDetail(
            job_id=job_id,
            baseline_id=baseline_id,
            bundle_id=request.bundle_id,
            simulation_seed=request.simulation_seed,
            status="accepted",
            created_at_utc=created_at,
        )
        self._write_json(self._baseline_run_path(job_id), record.model_dump(mode="json"))

        try:
            runtime_record = self._run_baseline_episode(
                bundle=bundle,
                planner_id=planner.metadata.planner_id,
                simulation_seed=request.simulation_seed,
                episode_id=episode_id,
            )
        except Exception as exc:
            failed_record = record.model_copy(
                update={
                    "status": "failed",
                    "completed_at_utc": datetime.now(UTC),
                    "error_message": str(exc),
                }
            )
            self._write_json(self._baseline_run_path(job_id), failed_record.model_dump(mode="json"))
            raise

        completed = record.model_copy(
            update={
                "status": "completed",
                "completed_at_utc": datetime.now(UTC),
                "episode_id": runtime_record.episode_id,
                "runtime_episode_id": runtime_record.runtime_episode_id,
                "replay_path": runtime_record.replay_path,
                "metrics": runtime_record.metrics,
            }
        )
        self._write_json(self._baseline_run_path(job_id), completed.model_dump(mode="json"))
        return completed

    def get_baseline_run(self, job_id: str) -> BaselineRunDetail:
        path = self._baseline_run_path(job_id)
        if not path.exists():
            raise ApiError(
                status_code=404,
                error="baseline_run_not_found",
                message=f"unknown baseline job_id {job_id!r}",
            )
        return BaselineRunDetail.model_validate(self._read_json(path))

    def _run_baseline_episode(
        self,
        *,
        bundle: ScenarioBundle,
        planner_id: str,
        simulation_seed: int,
        episode_id: str,
    ) -> EpisodeRecord:
        created_at = datetime.now(UTC)
        actor_planner_id = f"planner:{planner_id}"
        planner = build_planner(planner_id)
        planner_metadata = planner_runtime_metadata(planner)
        env = OrbitalEnv(
            bundle,
            config=EnvRuntimeConfig(
                planner_id=actor_planner_id,
                decision_interval_seconds=bundle.decision_interval_seconds,
                planner_metadata=planner_metadata,
            ),
        )
        observation, _ = env.reset(seed=simulation_seed, planner_id=actor_planner_id)
        runtime_episode_id = str(observation["episode_id"])
        planner.start_episode(
            context=PlannerEpisodeContext(
                bundle=bundle,
                episode_id=runtime_episode_id,
                episode_seed=simulation_seed,
                planner_seed=self._planner_seed(
                    bundle_id=bundle.bundle_id,
                    planner_id=planner_id,
                    simulation_seed=simulation_seed,
                ),
            ),
            initial_observation=observation,
        )

        action_history: list[Any] = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            decision = planner.select_action(observation)
            action_payload = decision.action.to_payload()
            action_history.append(
                {
                    "action_type": action_payload["action_type"],
                    "action_ref": action_payload["action_ref"],
                }
            )
            observation, _, terminated, truncated, _ = env.step(
                decision.action,
                planner_trace=decision.to_trace_payload(),
            )

        public_events = self._public_replay_events(
            replay_events=env.replay_events,
            episode_id=episode_id,
        )
        metrics = self._metrics_response(bundle=bundle, replay_events=public_events)
        public_observation = self._public_observation(observation, episode_id=episode_id)
        public_action_mask = dict(public_observation["action_mask"])
        self._write_replay(
            self._episode_replay_path(episode_id),
            public_events,
        )
        record = EpisodeRecord(
            episode_id=episode_id,
            runtime_episode_id=runtime_episode_id,
            bundle_id=bundle.bundle_id,
            planner_id=planner_id,
            planner_kind=str(planner_metadata.get("planner_kind", "builtin")),
            planner_metadata=planner_metadata,
            simulation_seed=simulation_seed,
            created_at_utc=created_at,
            updated_at_utc=datetime.now(UTC),
            sim_tick=int(public_observation["sim_tick"]),
            terminated=terminated,
            truncated=truncated,
            mission_utility=float(public_observation["mission_utility"]),
            replay_event_count=len(public_events),
            action_history=action_history,
            replay_path=str(self._episode_replay_path(episode_id)),
            latest_observation=public_observation,
            latest_action_mask=public_action_mask,
            metrics=metrics,
        )
        self._persist_episode_record(record, public_events)
        return record

    def _run_trained_policy_episode(
        self,
        *,
        bundle: ScenarioBundle,
        planner_id: str,
        simulation_seed: int,
        episode_id: str,
    ) -> EpisodeRecord:
        planner = build_planner(planner_id, policy_registry=self.model_registry)
        created_at = datetime.now(UTC)
        actor_planner_id = f"planner:{planner.metadata.planner_id}"
        env = OrbitalEnv(
            bundle,
            config=EnvRuntimeConfig(
                planner_id=actor_planner_id,
                decision_interval_seconds=bundle.decision_interval_seconds,
                planner_metadata=planner_runtime_metadata(planner),
            ),
        )
        observation, _ = env.reset(seed=simulation_seed, planner_id=actor_planner_id)
        runtime_episode_id = str(observation["episode_id"])
        planner.start_episode(
            context=PlannerEpisodeContext(
                bundle=bundle,
                episode_id=runtime_episode_id,
                episode_seed=simulation_seed,
                planner_seed=self._planner_seed(
                    bundle_id=bundle.bundle_id,
                    planner_id=planner_id,
                    simulation_seed=simulation_seed,
                ),
            ),
            initial_observation=observation,
        )

        action_history: list[Any] = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            decision = planner.select_action(observation)
            action_payload = decision.action.to_payload()
            action_history.append(
                {
                    "action_type": action_payload["action_type"],
                    "action_ref": action_payload["action_ref"],
                }
            )
            observation, _, terminated, truncated, _ = env.step(
                decision.action,
                planner_trace=decision.to_trace_payload(),
            )

        public_events = self._public_replay_events(
            replay_events=env.replay_events,
            episode_id=episode_id,
        )
        metrics = self._metrics_response(bundle=bundle, replay_events=public_events)
        public_observation = self._public_observation(observation, episode_id=episode_id)
        public_action_mask = dict(public_observation["action_mask"])
        self._write_replay(self._episode_replay_path(episode_id), public_events)
        planner_metadata = planner_runtime_metadata(planner)
        record = EpisodeRecord(
            episode_id=episode_id,
            runtime_episode_id=runtime_episode_id,
            bundle_id=bundle.bundle_id,
            planner_id=planner_id,
            planner_kind=str(planner_metadata.get("planner_kind", "trained_policy")),
            planner_metadata=planner_metadata,
            simulation_seed=simulation_seed,
            created_at_utc=created_at,
            updated_at_utc=datetime.now(UTC),
            sim_tick=int(public_observation["sim_tick"]),
            terminated=terminated,
            truncated=truncated,
            mission_utility=float(public_observation["mission_utility"]),
            replay_event_count=len(public_events),
            action_history=action_history,
            replay_path=str(self._episode_replay_path(episode_id)),
            latest_observation=public_observation,
            latest_action_mask=public_action_mask,
            metrics=metrics,
        )
        self._persist_episode_record(record, public_events)
        return record

    def _replay_episode(
        self,
        *,
        bundle: ScenarioBundle,
        planner_id: str,
        simulation_seed: int,
        episode_id: str,
        action_history: list[Any],
    ) -> EpisodeRuntimeSnapshot:
        env = OrbitalEnv(
            bundle,
            config=EnvRuntimeConfig(
                planner_id=planner_id,
                decision_interval_seconds=bundle.decision_interval_seconds,
            ),
        )
        observation, info = env.reset(seed=simulation_seed, planner_id=planner_id)
        reward = 0.0
        terminated = False
        truncated = False
        for saved_action in action_history:
            observation, reward, terminated, truncated, info = env.step(saved_action)
        runtime_episode_id = str(observation["episode_id"])
        public_events = self._public_replay_events(
            replay_events=env.replay_events,
            episode_id=episode_id,
        )
        metrics = self._metrics_response(bundle=bundle, replay_events=public_events)
        public_observation = self._public_observation(observation, episode_id=episode_id)
        public_action_mask = dict(public_observation["action_mask"])
        latest_events = (
            [
                event.model_dump(mode="json", exclude_none=True)
                for event in public_events[-len(info["events"]) :]
            ]
            if info["events"]
            else []
        )
        return EpisodeRuntimeSnapshot(
            runtime_episode_id=runtime_episode_id,
            observation=public_observation,
            action_mask=public_action_mask,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            mission_utility=float(public_observation["mission_utility"]),
            replay_events=public_events,
            latest_events=latest_events,
            metrics=metrics,
        )

    def _persist_episode_record(
        self,
        record: EpisodeRecord,
        replay_events: list[ReplayEvent],
    ) -> None:
        self._write_json(self._episode_path(record.episode_id), record.model_dump(mode="json"))
        self._write_replay(self._episode_replay_path(record.episode_id), replay_events)

    def _public_observation(
        self,
        observation: dict[str, Any],
        *,
        episode_id: str,
    ) -> dict[str, Any]:
        public_observation = {
            **observation,
            "episode_id": episode_id,
        }
        public_observation["action_mask"] = dict(observation["action_mask"])
        return public_observation

    def _public_replay_events(
        self,
        *,
        replay_events: list[ReplayEvent],
        episode_id: str,
    ) -> list[ReplayEvent]:
        event_documents = []
        for event in replay_events:
            event_document = event.model_dump(mode="json", exclude_none=True)
            event_document["episode_id"] = episode_id
            event_documents.append(event_document)
        return validate_canonical_replay_events(event_documents)

    def _metrics_response(
        self,
        *,
        bundle: ScenarioBundle,
        replay_events: list[ReplayEvent],
    ) -> EpisodeMetricsResponse:
        metrics = compute_episode_metrics(bundle=bundle, replay_events=replay_events).to_dict()
        return EpisodeMetricsResponse.model_validate(metrics)

    def _episode_summary(self, record: EpisodeRecord) -> EpisodeSummary:
        return EpisodeSummary(
            episode_id=record.episode_id,
            runtime_episode_id=record.runtime_episode_id,
            bundle_id=record.bundle_id,
            planner_id=record.planner_id,
            display_name=record.display_name or record.planner_id,
            planner_kind=record.planner_kind,
            planner_metadata=record.planner_metadata,
            episode_source=record.episode_source,
            report_id=record.report_id,
            report_episode_id=record.report_episode_id,
            report_split=record.report_split,
            simulation_seed=record.simulation_seed,
            created_at_utc=record.created_at_utc,
            updated_at_utc=record.updated_at_utc,
            sim_tick=record.sim_tick,
            terminated=record.terminated,
            truncated=record.truncated,
            mission_utility=record.mission_utility,
            replay_event_count=record.replay_event_count,
        )

    def _model_descriptor(self, entry: Any) -> ModelDescriptor:
        return ModelDescriptor(
            model_key=str(entry.model_key),
            planner_id=str(entry.planner_id),
            checkpoint_id=str(entry.checkpoint_id),
            checkpoint_fingerprint=str(entry.checkpoint_fingerprint),
            model_id=str(entry.model_id),
            algorithm=str(entry.algorithm),
            created_at_utc=entry.created_at_utc,
            description=f"Trained {entry.algorithm} policy restored from {entry.checkpoint_id}.",
        )

    def _report_summary(
        self,
        summary_path: Path,
        report_document: dict[str, Any],
    ) -> ReportSummary:
        episodes = [
            episode
            for episode in report_document.get("episodes", [])
            if isinstance(episode, dict)
        ]
        planner_kinds = [str(episode.get("planner_kind", "")) for episode in episodes]
        unique_planners = {str(episode.get("planner_key", "")) for episode in episodes}
        notable_path = summary_path.with_name("notable_episodes.csv")
        notable_count = 0
        if notable_path.exists():
            notable_count = max(
                len(notable_path.read_text(encoding="utf-8").splitlines()) - 1,
                0,
            )
        return ReportSummary(
            report_id=summary_path.parent.name,
            report_kind="evaluation",
            title=str(
                report_document.get("config", {}).get(
                    "evaluation_id",
                    summary_path.parent.name,
                )
            ),
            summary_path=str(summary_path),
            benchmark_id=(
                str(report_document.get("benchmark_id"))
                if report_document.get("benchmark_id") is not None
                else None
            ),
            episode_count=len(episodes),
            planner_count=len(unique_planners),
            trained_policy_count=sum(
                1 for planner_kind in planner_kinds if "policy" in planner_kind
            ),
            baseline_count=sum(
                1 for planner_kind in planner_kinds if "policy" not in planner_kind
            ),
            splits=sorted(
                {
                    str(episode.get("split"))
                    for episode in episodes
                    if episode.get("split") not in (None, "")
                }
            ),
            notable_episode_count=notable_count,
        )

    def _read_notable_episodes(self, notable_path: Path) -> list[ReportNotableEpisode]:
        if not notable_path.exists():
            return []
        with notable_path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            return [
                ReportNotableEpisode(
                    category=str(row.get("category", "")),
                    planner_key=(
                        str(row["planner_key"]) if row.get("planner_key") else None
                    ),
                    bundle_id=str(row.get("bundle_id", "")),
                    split=str(row["split"]) if row.get("split") else None,
                    scenario_family=str(row.get("scenario_family", "")),
                    primary_metric_value=(
                        float(row["primary_metric_value"])
                        if row.get("primary_metric_value")
                        else None
                    ),
                    difference_vs_best_baseline=(
                        float(row["difference_vs_best_baseline"])
                        if row.get("difference_vs_best_baseline")
                        else None
                    ),
                )
                for row in rows
            ]

    def _report_episode_summary(
        self,
        *,
        report_id: str,
        report_summary_path: Path,
        episode_document: dict[str, Any],
    ) -> ReportEpisodeSummary:
        bundle_id = str(episode_document["bundle_id"])
        planner_key = str(episode_document["planner_key"])
        split = str(episode_document.get("split", "")) or None
        report_episode_id = stable_id(
            "report-episode",
            report_id,
            split or "unknown",
            planner_key,
            bundle_id,
        )
        return ReportEpisodeSummary(
            report_id=report_id,
            report_episode_id=report_episode_id,
            episode_id=str(episode_document["episode_id"]),
            bundle_id=bundle_id,
            split=split,
            scenario_family=str(episode_document.get("scenario_family", "")),
            planner_key=planner_key,
            planner_kind=(
                str(episode_document["planner_kind"])
                if episode_document.get("planner_kind") is not None
                else None
            ),
            planner_version=(
                str(episode_document["planner_version"])
                if episode_document.get("planner_version") is not None
                else None
            ),
            display_name=(
                str(episode_document["display_name"])
                if episode_document.get("display_name") is not None
                else planner_key
            ),
            simulation_seed=int(episode_document.get("episode_seed", 0)),
            action_count=int(episode_document.get("action_count", 0)),
            metrics=EpisodeMetricsResponse.model_validate(episode_document["metrics"]),
            replay_path=str(episode_document["replay_path"]),
            summary_path=str(episode_document.get("summary_path", report_summary_path)),
            scenario_path=str(episode_document["scenario_path"]),
            reward_audit=(
                dict(episode_document["reward_audit"])
                if isinstance(episode_document.get("reward_audit"), dict)
                else None
            ),
            bundle_profile=(
                dict(episode_document["bundle_profile"])
                if isinstance(episode_document.get("bundle_profile"), dict)
                else None
            ),
        )

    def _find_report_episode(
        self, report_episode_id: str
    ) -> tuple[str, Path, ReportEpisodeSummary]:
        for summary_path in sorted(self.settings.training_report_dir.glob("*/summary.json")):
            report_document = self._read_json(summary_path)
            if not isinstance(report_document.get("episodes"), list):
                continue
            report_id = summary_path.parent.name
            for episode_document in report_document["episodes"]:
                if not isinstance(episode_document, dict):
                    continue
                episode = self._report_episode_summary(
                    report_id=report_id,
                    report_summary_path=summary_path,
                    episode_document=episode_document,
                )
                if episode.report_episode_id == report_episode_id:
                    return report_id, summary_path, episode
        raise ApiError(
            status_code=404,
            error="report_episode_not_found",
            message=f"unknown report_episode_id {report_episode_id!r}",
        )

    def _report_summary_path(self, report_id: str) -> Path:
        summary_path = self.settings.training_report_dir / report_id / "summary.json"
        if not summary_path.exists():
            raise ApiError(
                status_code=404,
                error="report_not_found",
                message=f"unknown report_id {report_id!r}",
            )
        return summary_path

    def _read_replay_events(self, replay_path: Path) -> list[ReplayEvent]:
        if not replay_path.exists():
            raise ApiError(
                status_code=404,
                error="replay_not_found",
                message=f"replay file {str(replay_path)!r} is missing",
            )
        return validate_canonical_replay_events(
            [
                json.loads(line)
                for line in replay_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )

    def _inference_traces_from_events(
        self,
        *,
        episode_id: str,
        planner_id: str,
        planner_kind: str | None,
        replay_events: list[ReplayEvent],
        planner_metadata: dict[str, Any] | None,
    ) -> EpisodeInferenceTraceResponse:
        del planner_metadata
        traces = [
            EpisodeInferenceTraceStep(
                event_index=event.event_index,
                sim_tick=event.sim_tick,
                sim_time_utc=event.sim_time_utc,
                action_id=event.payload.get("action_id"),
                action_type=str(event.payload["action_type"]),
                action_ref=str(event.payload["action_ref"]),
                planner_trace=dict(event.payload["planner_trace"]),
            )
            for event in replay_events
            if event.event_type == "action_selected" and "planner_trace" in event.payload
        ]
        return EpisodeInferenceTraceResponse(
            episode_id=episode_id,
            planner_id=planner_id,
            planner_kind=planner_kind,
            step_count=len(traces),
            traces=traces,
        )

    def _action_mask_from_trace(
        self,
        *,
        report_episode_id: str,
        planner_trace: dict[str, Any],
    ) -> dict[str, Any]:
        policy_trace = planner_trace.get("policy_trace", planner_trace)
        top_slots = policy_trace.get("top_slots", [])
        actions = []
        for slot in top_slots:
            if not isinstance(slot, dict):
                continue
            slot_mapping = (
                slot.get("slot_mapping")
                if isinstance(slot.get("slot_mapping"), dict)
                else {}
            )
            action_id = slot.get("action_id") or slot_mapping.get("action_id")
            action_type = slot_mapping.get("action_type")
            action_ref = slot_mapping.get("action_ref")
            if action_type is None or action_ref is None:
                continue
            actions.append(
                {
                    "action_id": str(
                        action_id or f"{report_episode_id}:{slot.get('slot_index', 'slot')}"
                    ),
                    "action_type": str(action_type),
                    "action_ref": str(action_ref),
                    "score_hint": float(slot.get("probability", 0.0)),
                }
            )
        return {
            "mask_id": f"mask:{report_episode_id}",
            "legal_action_count": int(policy_trace.get("legal_action_count", len(actions))),
            "actions": actions,
        }

    def _normalize_action(self, action: dict[str, Any]) -> Any:
        if "action_id" in action and "action_type" not in action:
            return str(action["action_id"])
        if "action_type" not in action:
            raise ApiError(
                status_code=400,
                error="invalid_action",
                message="action payload must include either action_type or action_id",
            )
        normalized = dict(action)
        if "action_ref" not in normalized and "ref" in normalized:
            normalized["action_ref"] = normalized["ref"]
        if normalized["action_type"] == "noop" and "action_ref" not in normalized:
            normalized["action_ref"] = "noop"
        if "action_ref" not in normalized:
            raise ApiError(
                status_code=400,
                error="invalid_action",
                message="action payload must include action_ref for non-noop actions",
            )
        return normalized

    def _coerce_bundle(self, payload: dict[str, Any]) -> ScenarioBundle:
        try:
            return validate_canonical_bundle(payload)
        except Exception as canonical_error:
            try:
                validate_phase0_instance("scenario_bundle.schema.json", payload)
                return compile_scenario_bundle(payload)
            except Exception as phase0_error:
                raise ApiError(
                    status_code=400,
                    error="invalid_scenario_bundle",
                    message="scenario payload does not match the canonical or Phase 0 contract",
                    details={
                        "canonical_error": str(canonical_error),
                        "phase0_error": str(phase0_error),
                    },
                ) from phase0_error

    def _planner_seed(self, *, bundle_id: str, planner_id: str, simulation_seed: int) -> int:
        token = stable_token(f"{planner_id}:{bundle_id}:{simulation_seed}", length=16)
        return int(token, 16)

    def _scenario_paths(self) -> list[Path]:
        scenario_paths = sorted(self.settings.scenario_dir.glob("*.json"))
        training_paths = sorted(self.settings.training_scenario_pack_dir.rglob("*.json"))
        return [*scenario_paths, *training_paths]

    def _scenario_path(self, bundle_id: str) -> Path:
        direct_path = self.settings.scenario_dir / self._artifact_name(bundle_id, suffix=".json")
        if direct_path.exists():
            return direct_path
        for path in self._scenario_paths():
            document = self._read_json(path)
            if document.get("bundle_id") == bundle_id:
                return path
        return direct_path

    def _episode_path(self, episode_id: str) -> Path:
        return self.settings.episode_dir / self._artifact_name(episode_id, suffix=".json")

    def _episode_replay_path(self, episode_id: str) -> Path:
        return self.settings.episode_dir / self._artifact_name(episode_id, suffix=".ndjson")

    def _baseline_run_path(self, job_id: str) -> Path:
        return self.settings.baseline_run_dir / self._artifact_name(job_id, suffix=".json")

    def _model_run_path(self, job_id: str) -> Path:
        return self.settings.model_run_dir / self._artifact_name(job_id, suffix=".json")

    def _orbit_asset_bundle_path(self, orbit_bundle_id: str) -> Path:
        expected_path = self.settings.orbit_asset_dir / self._artifact_name(
            orbit_bundle_id,
            suffix=".json",
        )
        if expected_path.exists():
            return expected_path
        for path in sorted(self.settings.orbit_asset_dir.glob("*.json")):
            document = self._read_json(path)
            if document.get("bundle_id") == orbit_bundle_id:
                return path
        raise ApiError(
            status_code=404,
            error="orbit_bundle_not_found",
            message=f"unknown orbit asset bundle_id {orbit_bundle_id!r}",
        )

    def _load_bundle_from_path(self, path: Path) -> ScenarioBundle:
        return validate_canonical_bundle(self._read_json(path))

    def _load_episode_record(self, path: Path) -> EpisodeRecord:
        if not path.exists():
            raise ApiError(
                status_code=404,
                error="episode_not_found",
                message=f"unknown episode_id {path.stem.replace('--', ':')!r}",
            )
        return EpisodeRecord.model_validate(self._read_json(path))

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json_dumps(payload) + "\n", encoding="utf-8")

    def _write_replay(self, path: Path, replay_events: list[ReplayEvent]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(replay_events_to_ndjson(replay_events) + "\n", encoding="utf-8")

    def _demo_ui_query(
        self,
        *,
        bundle_id: str | None,
        baseline_id: str | None,
        episode_id: str | None,
    ) -> str:
        params = {
            key: value
            for key, value in {
                "scenario": bundle_id,
                "baseline": baseline_id,
                "episode": episode_id,
            }.items()
            if value
        }
        if not params:
            return ""
        return f"?{urlencode(params)}"

    def _build_satellite_czml(
        self,
        *,
        bundle: ScenarioBundle,
        orbit_bundle: OrbitAssetBundle,
        samples: list[SatelliteStateSample],
        step_seconds: int,
    ) -> list[dict[str, Any]]:
        interval = (
            f"{format_utc_timestamp(bundle.time_window.start_time_utc)}/"
            f"{format_utc_timestamp(bundle.time_window.end_time_utc)}"
        )
        sample_groups: dict[str, list[SatelliteStateSample]] = {}
        for sample in samples:
            sample_groups.setdefault(sample.satellite_id, []).append(sample)
        orbit_assets = {asset.satellite_id: asset for asset in orbit_bundle.assets}
        packets: list[dict[str, Any]] = [
            {
                "id": "document",
                "name": bundle.bundle_id,
                "version": "1.0",
                "clock": {
                    "interval": interval,
                    "currentTime": format_utc_timestamp(bundle.time_window.start_time_utc),
                    "multiplier": max(step_seconds, 60),
                    "range": "CLAMPED",
                    "step": "SYSTEM_CLOCK_MULTIPLIER",
                },
            }
        ]
        for satellite in bundle.satellites:
            satellite_samples = sorted(
                sample_groups.get(satellite.satellite_id, []),
                key=lambda item: item.timestamp_utc,
            )
            if not satellite_samples:
                continue
            epoch = satellite_samples[0].timestamp_utc
            cartographic_degrees: list[float] = []
            for sample in satellite_samples:
                seconds_from_epoch = (sample.timestamp_utc - epoch).total_seconds()
                cartographic_degrees.extend(
                    [
                        float(seconds_from_epoch),
                        round(sample.longitude_deg, 6),
                        round(sample.latitude_deg, 6),
                        round(sample.altitude_m, 3),
                    ]
                )
            orbit = orbit_assets.get(satellite.satellite_id)
            packets.append(
                {
                    "id": satellite.satellite_id,
                    "name": satellite.name,
                    "availability": interval,
                    "description": (
                        f"{satellite.name} | NORAD {satellite.norad_catalog_id}"
                        if orbit is None
                        else (
                            f"{satellite.name} | NORAD {satellite.norad_catalog_id} | "
                            f"epoch {format_utc_timestamp(orbit.orbit.epoch_utc)}"
                        )
                    ),
                    "label": {
                        "text": satellite.name,
                        "font": "12pt monospace",
                        "fillColor": {"rgba": [170, 214, 255, 255]},
                        "outlineColor": {"rgba": [6, 12, 22, 255]},
                        "outlineWidth": 2,
                        "showBackground": True,
                        "backgroundColor": {"rgba": [6, 12, 22, 180]},
                        "horizontalOrigin": "LEFT",
                        "pixelOffset": {"cartesian2": [14, -16]},
                    },
                    "point": {
                        "pixelSize": 10,
                        "color": {"rgba": [93, 190, 255, 255]},
                        "outlineColor": {"rgba": [233, 249, 255, 255]},
                        "outlineWidth": 2,
                    },
                    "path": {
                        "material": {
                            "polylineOutline": {
                                "color": {"rgba": [58, 145, 255, 210]},
                                "outlineColor": {"rgba": [6, 12, 22, 140]},
                                "outlineWidth": 1,
                            }
                        },
                        "width": 2,
                        "leadTime": 0,
                        "trailTime": step_seconds * 30,
                        "resolution": step_seconds,
                        "show": True,
                    },
                    "position": {
                        "epoch": format_utc_timestamp(epoch),
                        "interpolationAlgorithm": "LAGRANGE",
                        "interpolationDegree": 1,
                        "referenceFrame": "FIXED",
                        "cartographicDegrees": cartographic_degrees,
                    },
                    "properties": {
                        "satellite_id": satellite.satellite_id,
                        "sensor_id": satellite.sensor.sensor_id,
                    },
                }
            )
        return packets

    def _artifact_name(self, value: str, *, suffix: str) -> str:
        return value.replace(":", "--").replace("/", "--") + suffix
