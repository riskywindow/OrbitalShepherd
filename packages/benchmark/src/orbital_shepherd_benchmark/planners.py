from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import Any, Protocol

import torch

from orbital_shepherd_contracts import DownlinkWindow, ObservationOpportunity, ScenarioBundle
from orbital_shepherd_core import seeded_rng, stable_id
from orbital_shepherd_env_runtime import OrbitalAction
from orbital_shepherd_policy_models import (
    PolicyModelRegistry,
    TrainingObservationProjector,
    default_policy_model_registry,
)

PlannerFactory = Callable[[], "Planner"]
TRAINED_POLICY_PLANNER_PREFIX = "trained_policy:"

VALUE_DENSITY_OBSERVATION_FORMULA = (
    "score = w_expected_value * expected_mission_value"
    " + w_value_density * value_density"
    " + w_freshness * freshness_bonus"
    " + w_quality * predicted_quality"
    " - w_cloud_risk * cloud_risk"
    " - w_retarget * retarget_cost"
    " - w_downlink_consequence * downlink_consequence"
)
VALUE_DENSITY_DOWNLINK_FORMULA = (
    "score = w_expected_value * queued_delivery_value"
    " + w_value_density * queued_value_density"
    " + w_freshness * latency_pressure"
    " + w_downlink_consequence * buffer_relief"
    " - w_cloud_risk * outage_risk"
)


@dataclass(frozen=True, slots=True)
class PlannerMetadata:
    planner_id: str
    version: str
    description: str


@dataclass(frozen=True, slots=True)
class PlannerEpisodeContext:
    bundle: ScenarioBundle
    episode_id: str
    episode_seed: int
    planner_seed: int


@dataclass(frozen=True, slots=True)
class PlannerCandidate:
    action: OrbitalAction
    score: float
    rationale: str
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "action": self.action.to_payload(),
            "score": round(self.score, 6),
            "rationale": self.rationale,
            "diagnostics": {
                key: round(value, 6) for key, value in sorted(self.diagnostics.items())
            },
        }
        if self.metadata:
            payload["metadata"] = _json_compatible(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    action: OrbitalAction
    score: float
    rationale: str
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    considered_candidates: tuple[PlannerCandidate, ...] = ()
    trace_details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "action": self.action.to_payload(),
            "score": round(self.score, 6),
            "rationale": self.rationale,
            "diagnostics": {
                key: round(value, 6) for key, value in sorted(self.diagnostics.items())
            },
        }
        if self.considered_candidates:
            payload["considered_candidates"] = [
                candidate.to_dict() for candidate in self.considered_candidates
            ]
        if self.trace_details:
            payload["trace_details"] = _json_compatible(self.trace_details)
        return payload

    def to_trace_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "selected_candidate": PlannerCandidate(
                action=self.action,
                score=self.score,
                rationale=self.rationale,
                diagnostics=self.diagnostics,
            ).to_dict()
        }
        if self.considered_candidates:
            payload["considered_candidates"] = [
                candidate.to_dict() for candidate in self.considered_candidates
            ]
        if self.trace_details:
            payload.update(_json_compatible(self.trace_details))
        return payload


class Planner(Protocol):
    metadata: PlannerMetadata

    def start_episode(
        self,
        *,
        context: PlannerEpisodeContext,
        initial_observation: Mapping[str, Any],
    ) -> None: ...

    def select_action(self, observation: Mapping[str, Any]) -> PlannerDecision: ...


@dataclass(frozen=True, slots=True)
class ValueDensityScoringConfig:
    observation_expected_value_weight: float = 1.0
    observation_value_density_weight: float = 0.35
    observation_freshness_weight: float = 0.2
    observation_quality_weight: float = 0.15
    observation_cloud_risk_weight: float = 0.5
    observation_retarget_weight: float = 0.18
    observation_downlink_consequence_weight: float = 0.35
    downlink_expected_value_weight: float = 1.0
    downlink_value_density_weight: float = 0.25
    downlink_freshness_weight: float = 0.35
    downlink_buffer_relief_weight: float = 0.3
    downlink_outage_risk_weight: float = 0.55
    freshness_half_life_hours: float = 6.0
    comfortable_buffer_fill_ratio: float = 0.65
    future_downlink_gap_hours: float = 4.0
    no_future_downlink_penalty: float = 1.0


@dataclass(frozen=True, slots=True)
class OrtoolsLookaheadConfig:
    horizon_steps: int = 4
    future_discount: float = 0.9
    max_future_observations: int = 8
    max_future_downlinks: int = 6
    max_solver_seconds: float = 1.0
    solver_random_seed: int = 0


@dataclass(frozen=True, slots=True)
class ProjectedQueueEntry:
    queue_id: str
    satellite_id: str
    target_cell_id: str
    incident_ids: tuple[str, ...]
    data_volume_mb: float
    expected_value: float
    available_time_utc: datetime
    usable: bool
    source_ref: str


@dataclass(frozen=True, slots=True)
class ProjectedLookaheadState:
    current_time_utc: datetime
    buffer_usage_by_satellite: Mapping[str, float]
    queue_entries: tuple[ProjectedQueueEntry, ...]
    completed_observation_ids: frozenset[str]
    completed_downlink_ids: frozenset[str]
    incident_status_by_id: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FutureObservationCandidate:
    opportunity: ObservationOpportunity
    decision_tick: int
    completion_step: int
    base_score: float
    expected_delivery_value: float


@dataclass(frozen=True, slots=True)
class FutureDownlinkCandidate:
    window: DownlinkWindow
    decision_tick: int
    delivery_step: int
    base_score: float
    reserved_queue_volume_mb: float
    residual_capacity_mb: float


def planner_registry() -> dict[str, PlannerFactory]:
    return {
        "random_valid_action": RandomValidActionPlanner,
        "urgency_greedy": UrgencyGreedyPlanner,
        "value_density_greedy": ValueDensityGreedyPlanner,
        "ortools_receding_horizon": OrtoolsRecedingHorizonPlanner,
    }


def build_planner(
    planner_id: str,
    *,
    policy_registry: PolicyModelRegistry | None = None,
) -> Planner:
    if is_trained_policy_planner_id(planner_id):
        return TrainedPolicyPlanner(
            model_key=planner_model_key(planner_id),
            registry=policy_registry,
        )
    try:
        return planner_registry()[planner_id]()
    except KeyError as exc:
        known = ", ".join(sorted(planner_descriptions()))
        raise ValueError(f"unknown planner_id {planner_id!r}; expected one of: {known}") from exc


def planner_descriptions() -> dict[str, str]:
    descriptions = {
        planner_id: factory().metadata.description
        for planner_id, factory in planner_registry().items()
    }
    registry = default_policy_model_registry()
    for entry in registry.list_policies():
        descriptions[entry.planner_id] = (
            f"Trained {entry.algorithm} policy restored from {entry.checkpoint_id}."
        )
    return descriptions


def is_trained_policy_planner_id(planner_id: str) -> bool:
    return planner_id.startswith(TRAINED_POLICY_PLANNER_PREFIX)


def planner_model_key(planner_id: str) -> str:
    if not is_trained_policy_planner_id(planner_id):
        raise ValueError(f"{planner_id!r} is not a trained-policy planner id")
    return planner_id[len(TRAINED_POLICY_PLANNER_PREFIX) :]


def planner_runtime_metadata(planner: Planner) -> dict[str, Any]:
    metadata = {
        "planner_kind": "builtin",
        "planner_id": planner.metadata.planner_id,
        "planner_version": planner.metadata.version,
        "planner_description": planner.metadata.description,
    }
    runtime_metadata = getattr(planner, "runtime_metadata", None)
    if callable(runtime_metadata):
        metadata.update(_json_compatible(runtime_metadata()))
    return metadata


def legal_actions_from_observation(observation: Mapping[str, Any]) -> tuple[OrbitalAction, ...]:
    action_mask = observation["action_mask"]
    payloads = action_mask["actions"]
    actions: list[OrbitalAction] = []
    for payload in payloads:
        actions.append(
            OrbitalAction(
                action_type=str(payload["action_type"]),
                ref=str(payload.get("action_ref", payload.get("ref", "noop"))),
                score_hint=_optional_float(payload.get("score_hint")),
                satellite_id=_optional_str(payload.get("satellite_id")),
                target_cell_id=_optional_str(payload.get("target_cell_id")),
            )
        )
    return tuple(actions)


class TrainedPolicyPlanner:
    def __init__(
        self,
        *,
        model_key: str,
        registry: PolicyModelRegistry | None = None,
    ) -> None:
        self._model_key = model_key
        self._registry = registry or default_policy_model_registry()
        self._entry = self._registry.get_policy(model_key)
        self._loaded_policy = self._registry.load_policy(model_key)
        self._projector: TrainingObservationProjector | None = None
        self.metadata = PlannerMetadata(
            planner_id=self._entry.planner_id,
            version=self._entry.checkpoint_id,
            description=(
                f"Local CPU trained-policy adapter for {self._entry.checkpoint_id} "
                f"({self._entry.algorithm})."
            ),
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "planner_kind": "trained_policy",
            "planner_id": self.metadata.planner_id,
            "planner_version": self.metadata.version,
            "planner_description": self.metadata.description,
            "model_key": self._entry.model_key,
            "model_id": self._entry.model_id,
            "checkpoint_id": self._entry.checkpoint_id,
            "checkpoint_fingerprint": self._entry.checkpoint_fingerprint,
            "checkpoint_manifest_path": str(self._entry.checkpoint_manifest_path),
            "checkpoint_path": str(self._entry.checkpoint_path),
            "inference_config": self._loaded_policy.inference_config(),
        }

    def start_episode(
        self,
        *,
        context: PlannerEpisodeContext,
        initial_observation: Mapping[str, Any],
    ) -> None:
        self._projector = TrainingObservationProjector(
            context.bundle,
            top_k=self._entry.architecture.top_k,
        )
        self._project_for_policy(initial_observation)

    def select_action(self, observation: Mapping[str, Any]) -> PlannerDecision:
        if self._projector is None:
            raise RuntimeError("start_episode() must be called before select_action()")
        training_observation, info = self._project_for_policy(observation)
        with torch.no_grad():
            outputs = self._loaded_policy.policy_model.forward_observation(
                {
                    "global_features": torch.tensor(
                        training_observation["global_features"],
                        dtype=torch.float32,
                    ).unsqueeze(0),
                    "candidate_features": torch.tensor(
                        training_observation["candidate_features"],
                        dtype=torch.float32,
                    ).unsqueeze(0),
                    "action_mask": torch.tensor(
                        training_observation["action_mask"],
                        dtype=torch.float32,
                    ).unsqueeze(0),
                }
            )
        masked_logits = outputs.masked_logits.squeeze(0).cpu()
        probabilities = torch.softmax(masked_logits, dim=0)
        selected_slot = int(torch.argmax(masked_logits).item())
        selected_mapping = self._projector.decode_action_slot(selected_slot)
        selected_action = self._projector.runtime_action_for_slot(selected_slot)
        if selected_action is None:
            selected_action = OrbitalAction(action_type="noop", ref="noop", score_hint=0.0)
        slot_mapping = list(info.get("slot_mapping", []))
        legal_action_count = int(sum(int(value) for value in training_observation["action_mask"]))
        total_slot_count = len(training_observation["action_mask"])
        entropy = float(
            -(probabilities * torch.log(torch.clamp(probabilities, min=1.0e-12))).sum().item()
        )
        top_count = min(5, legal_action_count)
        top_slots = torch.topk(masked_logits, k=top_count).indices.tolist()
        considered_candidates = [
            PlannerCandidate(
                action=self._projector.runtime_action_for_slot(int(slot_index))
                or OrbitalAction(action_type="noop", ref="noop", score_hint=0.0),
                score=float(probabilities[slot_index].item()),
                rationale=f"policy top-{rank + 1} slot",
                diagnostics={
                    "slot_index": float(slot_index),
                    "logit": float(masked_logits[slot_index].item()),
                    "probability": float(probabilities[slot_index].item()),
                },
                metadata={
                    "slot_mapping": (
                        slot_mapping[slot_index] if slot_index < len(slot_mapping) else None
                    )
                },
            )
            for rank, slot_index in enumerate(top_slots)
        ]
        return PlannerDecision(
            action=selected_action,
            score=round(float(probabilities[selected_slot].item()), 6),
            rationale="masked argmax over trained-policy action slots",
            diagnostics={
                "selected_slot": float(selected_slot),
                "selected_logit": float(masked_logits[selected_slot].item()),
                "selected_probability": float(probabilities[selected_slot].item()),
                "legal_action_count": float(legal_action_count),
                "mask_pressure": float(1.0 - (legal_action_count / max(total_slot_count, 1))),
            },
            considered_candidates=tuple(considered_candidates),
            trace_details={
                "planner_kind": "trained_policy",
                "model": self.runtime_metadata(),
                "inference_backend": self._loaded_policy.backend,
                "selected_slot": selected_slot,
                "selected_slot_mapping": selected_mapping,
                "selected_action_id": info.get("slot_to_action_id", [None])[selected_slot],
                "selected_logit": round(float(masked_logits[selected_slot].item()), 6),
                "selected_probability": round(float(probabilities[selected_slot].item()), 6),
                "action_entropy": round(entropy, 6),
                "value_estimate": round(float(outputs.values.squeeze(0).item()), 6),
                "legal_action_count": legal_action_count,
                "mask_pressure": round(1.0 - (legal_action_count / max(total_slot_count, 1)), 6),
                "top_slots": [
                    {
                        "slot_index": int(slot_index),
                        "action_id": (
                            info.get("slot_to_action_id", [None])[slot_index]
                            if slot_index < len(info.get("slot_to_action_id", []))
                            else None
                        ),
                        "slot_mapping": (
                            slot_mapping[slot_index] if slot_index < len(slot_mapping) else None
                        ),
                        "logit": round(float(masked_logits[slot_index].item()), 6),
                        "probability": round(float(probabilities[slot_index].item()), 6),
                    }
                    for slot_index in top_slots
                ],
            },
        )

    def _project_for_policy(
        self,
        observation: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._projector is None:
            raise RuntimeError("start_episode() must be called before projection")
        return self._projector.project(
            raw_observation=observation,
            runtime_actions=legal_actions_from_observation(observation),
            raw_info={},
        )


class RandomValidActionPlanner:
    metadata = PlannerMetadata(
        planner_id="random_valid_action",
        version="phase1-v1",
        description="Uniformly sample one legal action from the emitted action mask.",
    )

    def __init__(self) -> None:
        self._rng = seeded_rng(0)

    def start_episode(
        self,
        *,
        context: PlannerEpisodeContext,
        initial_observation: Mapping[str, Any],
    ) -> None:
        del initial_observation
        self._rng = seeded_rng(context.planner_seed)

    def select_action(self, observation: Mapping[str, Any]) -> PlannerDecision:
        legal_actions = legal_actions_from_observation(observation)
        selected = self._rng.choice(legal_actions)
        return PlannerDecision(
            action=selected,
            score=1.0 / max(len(legal_actions), 1),
            rationale="uniform sample over current legal actions",
            diagnostics={"legal_action_count": float(len(legal_actions))},
            trace_details={"sampling_policy": "uniform_legal_actions"},
        )


class UrgencyGreedyPlanner:
    """Explicit myopic heuristic for Phase 1.

    The score is intentionally simple and planner-visible only:

    Observation score:
      expected_incident_value * predicted_usefulness
      + freshness_bonus
      - cloud_risk_penalty
      - slew_penalty

    Downlink score:
      queued_usable_value * latency_pressure * link_reliability
      + buffer_pressure_bonus
    """

    metadata = PlannerMetadata(
        planner_id="urgency_greedy",
        version="phase1-v1",
        description=(
            "Rank legal actions by urgent incident value, predicted observation usefulness, "
            "cloud risk, and current downlink pressure."
        ),
    )

    def __init__(self) -> None:
        self._bundle: ScenarioBundle | None = None
        self._opportunity_by_id: dict[str, ObservationOpportunity] = {}
        self._target_value_by_id: dict[str, float] = {}
        self._buffer_capacity_by_satellite: dict[str, float] = {}

    def start_episode(
        self,
        *,
        context: PlannerEpisodeContext,
        initial_observation: Mapping[str, Any],
    ) -> None:
        del initial_observation
        self._bundle = context.bundle
        self._opportunity_by_id = {
            opportunity.opportunity_id: opportunity
            for opportunity in context.bundle.observation_opportunities
        }
        self._target_value_by_id = {
            target.target_cell_id: target.static_value for target in context.bundle.target_cells
        }
        self._buffer_capacity_by_satellite = {
            satellite.satellite_id: satellite.downlink.buffer_capacity_mb
            for satellite in context.bundle.satellites
        }

    def select_action(self, observation: Mapping[str, Any]) -> PlannerDecision:
        if self._bundle is None:
            raise RuntimeError("start_episode() must be called before select_action()")
        incident_by_id = {
            str(incident["incident_id"]): incident for incident in observation.get("incidents", [])
        }
        candidates: list[PlannerCandidate] = []
        for action in legal_actions_from_observation(observation):
            score, rationale, diagnostics = self._score_action(
                action=action,
                observation=observation,
                incident_by_id=incident_by_id,
            )
            candidates.append(
                PlannerCandidate(
                    action=action,
                    score=score,
                    rationale=rationale,
                    diagnostics=diagnostics,
                )
            )
        if not candidates:
            raise RuntimeError("action mask must contain at least one legal action")
        considered = tuple(sorted(candidates, key=_candidate_sort_key, reverse=True))
        selected = considered[0]
        return PlannerDecision(
            action=selected.action,
            score=selected.score,
            rationale=selected.rationale,
            diagnostics=selected.diagnostics,
            considered_candidates=considered,
            trace_details={"formula_family": "phase1_urgency_greedy"},
        )

    def _score_action(
        self,
        *,
        action: OrbitalAction,
        observation: Mapping[str, Any],
        incident_by_id: Mapping[str, Mapping[str, Any]],
    ) -> tuple[float, str, dict[str, float]]:
        if action.action_type == "noop":
            return 0.0, "wait for a better legal action", {}
        if action.action_type == "schedule_observation":
            opportunity = self._opportunity_by_id[action.ref]
            return self._score_observation(
                opportunity=opportunity,
                observation=observation,
                incident_by_id=incident_by_id,
            )
        if action.action_type == "schedule_downlink":
            return self._score_downlink(
                satellite_id=action.satellite_id or "",
                outage_risk=self._downlink_outage_risk(action.ref),
                observation=observation,
                incident_by_id=incident_by_id,
            )
        return -1.0, "unsupported action type", {}

    def _score_observation(
        self,
        *,
        opportunity: ObservationOpportunity,
        observation: Mapping[str, Any],
        incident_by_id: Mapping[str, Mapping[str, Any]],
    ) -> tuple[float, str, dict[str, float]]:
        predicted_usefulness = opportunity.predicted_quality_mean * (
            1.0 - opportunity.predicted_cloud_obstruction_prob
        )
        linked_incidents = [
            incident_by_id[incident_id]
            for incident_id in opportunity.incident_ids or ()
            if incident_id in incident_by_id
            and str(incident_by_id[incident_id]["status"]) not in {"downlinked", "missed"}
        ]
        target_static_value = self._target_value_by_id.get(opportunity.target_cell_id, 0.0)
        urgency_mass = sum(float(item["urgency_score"]) for item in linked_incidents)
        urgency_peak = max((float(item["urgency_score"]) for item in linked_incidents), default=0.0)
        age_penalty = self._observation_age_penalty(
            sim_time_utc=_coerce_datetime(observation["sim_time_utc"]),
            linked_incidents=linked_incidents,
        )
        expected_incident_value = (
            (1.3 * urgency_peak) + (0.55 * urgency_mass) + (0.25 * target_static_value)
        )
        score = (
            (expected_incident_value * predicted_usefulness)
            - (0.45 * opportunity.predicted_cloud_obstruction_prob)
            - (0.18 * opportunity.slew_cost)
            - age_penalty
        )
        return (
            round(score, 6),
            "urgent predicted observation value minus cloud and slew risk",
            {
                "predicted_usefulness": predicted_usefulness,
                "urgency_peak": urgency_peak,
                "urgency_mass": urgency_mass,
                "target_static_value": target_static_value,
                "cloud_risk": opportunity.predicted_cloud_obstruction_prob,
                "slew_cost": opportunity.slew_cost,
                "age_penalty": age_penalty,
            },
        )

    def _score_downlink(
        self,
        *,
        satellite_id: str,
        outage_risk: float,
        observation: Mapping[str, Any],
        incident_by_id: Mapping[str, Mapping[str, Any]],
    ) -> tuple[float, str, dict[str, float]]:
        sim_time_utc = _coerce_datetime(observation["sim_time_utc"])
        queued_entries = [
            entry
            for entry in observation.get("onboard_queue", [])
            if str(entry["satellite_id"]) == satellite_id
        ]
        usable_entries = [entry for entry in queued_entries if bool(entry["usable"])]
        queued_value = 0.0
        for entry in usable_entries:
            incident_ids = [
                incident_id
                for incident_id in entry.get("incident_ids", [])
                if incident_id in incident_by_id
            ]
            urgency_peak = max(
                (
                    float(incident_by_id[incident_id]["urgency_score"])
                    for incident_id in incident_ids
                ),
                default=0.0,
            )
            target_static_value = self._target_value_by_id.get(str(entry["target_cell_id"]), 0.0)
            age_minutes = max(
                0.0,
                (sim_time_utc - _coerce_datetime(entry["observation_time_utc"])).total_seconds()
                / 60.0,
            )
            latency_pressure = 1.0 + min(age_minutes / 30.0, 2.0)
            queued_value += (
                urgency_peak
                * target_static_value
                * float(entry["realized_quality"])
                * latency_pressure
            )
        buffer_usage = float(
            observation.get("buffer_usage_by_satellite", {}).get(satellite_id, 0.0)
        )
        buffer_capacity = self._buffer_capacity_by_satellite.get(satellite_id, 1.0)
        buffer_fill_ratio = (
            0.0 if buffer_capacity <= 0 else min(buffer_usage / buffer_capacity, 1.0)
        )
        score = (queued_value * (1.0 - (0.55 * outage_risk))) + (0.25 * buffer_fill_ratio)
        if not usable_entries:
            score = 0.08 * buffer_fill_ratio * (1.0 - outage_risk)
        return (
            round(score, 6),
            "downlink queued usable value with latency pressure and outage adjustment",
            {
                "queued_value": queued_value,
                "outage_risk": outage_risk,
                "buffer_fill_ratio": buffer_fill_ratio,
                "usable_entry_count": float(len(usable_entries)),
            },
        )

    def _observation_age_penalty(
        self,
        *,
        sim_time_utc: datetime,
        linked_incidents: Sequence[Mapping[str, Any]],
    ) -> float:
        if not linked_incidents:
            return 0.0
        oldest_age_hours = max(
            0.0,
            max(
                (
                    (sim_time_utc - _coerce_datetime(incident["observed_time_utc"])).total_seconds()
                    / 3600.0
                    if incident.get("observed_time_utc") is not None
                    else (
                        sim_time_utc - self._incident_ignition_time(str(incident["incident_id"]))
                    ).total_seconds()
                    / 3600.0
                )
                for incident in linked_incidents
            ),
        )
        return min(oldest_age_hours / 12.0, 0.25)

    def _incident_ignition_time(self, incident_id: str) -> datetime:
        if self._bundle is None:
            raise RuntimeError("bundle missing")
        for incident in self._bundle.incidents:
            if incident.incident_id == incident_id:
                return incident.ignition_time_utc
        raise KeyError(incident_id)

    def _downlink_outage_risk(self, window_id: str) -> float:
        if self._bundle is None:
            raise RuntimeError("bundle missing")
        for window in self._bundle.downlink_windows:
            if window.window_id == window_id:
                return window.outage_risk
        raise KeyError(window_id)


class ValueDensityHeuristic:
    def __init__(self, config: ValueDensityScoringConfig | None = None) -> None:
        self.config = config or ValueDensityScoringConfig()
        self._bundle: ScenarioBundle | None = None
        self._opportunity_by_id: dict[str, ObservationOpportunity] = {}
        self._downlink_by_id: dict[str, DownlinkWindow] = {}
        self._target_value_by_id: dict[str, float] = {}
        self._buffer_capacity_by_satellite: dict[str, float] = {}
        self._incident_ignition_by_id: dict[str, datetime] = {}

    def start_episode(self, bundle: ScenarioBundle) -> None:
        self._bundle = bundle
        self._opportunity_by_id = {
            opportunity.opportunity_id: opportunity
            for opportunity in bundle.observation_opportunities
        }
        self._downlink_by_id = {window.window_id: window for window in bundle.downlink_windows}
        self._target_value_by_id = {
            target.target_cell_id: target.static_value for target in bundle.target_cells
        }
        self._buffer_capacity_by_satellite = {
            satellite.satellite_id: satellite.downlink.buffer_capacity_mb
            for satellite in bundle.satellites
        }
        self._incident_ignition_by_id = {
            incident.incident_id: incident.ignition_time_utc for incident in bundle.incidents
        }

    def score_action(
        self,
        action: OrbitalAction,
        observation: Mapping[str, Any],
    ) -> PlannerCandidate:
        if self._bundle is None:
            raise RuntimeError("start_episode() must be called before score_action()")
        if action.action_type == "noop":
            return PlannerCandidate(
                action=action,
                score=0.0,
                rationale="preserve the current state and wait",
                diagnostics={"idle": 1.0},
            )
        if action.action_type == "schedule_observation":
            opportunity = self._opportunity_by_id[action.ref]
            return self._score_observation(
                action=action,
                observation=observation,
                opportunity=opportunity,
            )
        if action.action_type == "schedule_downlink":
            window = self._downlink_by_id[action.ref]
            return self._score_downlink(action=action, observation=observation, window=window)
        return PlannerCandidate(
            action=action,
            score=-1.0,
            rationale="unsupported action type",
            diagnostics={},
        )

    def _score_observation(
        self,
        *,
        action: OrbitalAction,
        observation: Mapping[str, Any],
        opportunity: ObservationOpportunity,
    ) -> PlannerCandidate:
        sim_time_utc = _coerce_datetime(observation["sim_time_utc"])
        linked_incidents = [
            incident
            for incident in observation.get("incidents", [])
            if str(incident["incident_id"]) in set(opportunity.incident_ids or ())
            and str(incident["status"]) not in {"downlinked", "missed"}
        ]
        incident_value = 0.0
        freshness_values: list[float] = []
        urgency_peak = 0.0
        target_static_value = self._target_value_by_id.get(opportunity.target_cell_id, 0.0)
        for incident in linked_incidents:
            incident_id = str(incident["incident_id"])
            urgency = float(incident["urgency_score"])
            freshness = self._incident_freshness(
                sim_time_utc=sim_time_utc,
                incident_id=incident_id,
                observed_time_utc=incident.get("observed_time_utc"),
            )
            incident_value += target_static_value * urgency * freshness
            freshness_values.append(freshness)
            urgency_peak = max(urgency_peak, urgency)
        freshness_bonus = sum(freshness_values) / len(freshness_values) if freshness_values else 0.0
        predicted_quality = opportunity.predicted_quality_mean
        cloud_risk = opportunity.predicted_cloud_obstruction_prob
        expected_mission_value = incident_value * predicted_quality * (1.0 - cloud_risk)
        value_density = expected_mission_value / max(opportunity.estimated_data_volume_mb, 1.0)
        downlink_consequence = self._observation_downlink_consequence(
            satellite_id=opportunity.satellite_id,
            current_time_utc=sim_time_utc,
            current_buffer_usage=float(
                observation.get("buffer_usage_by_satellite", {}).get(opportunity.satellite_id, 0.0)
            ),
            added_volume_mb=opportunity.estimated_data_volume_mb,
            completed_downlink_ids=frozenset(
                str(item) for item in observation.get("completed_downlink_ids", [])
            ),
        )
        score = (
            self.config.observation_expected_value_weight * expected_mission_value
            + self.config.observation_value_density_weight * value_density
            + self.config.observation_freshness_weight * freshness_bonus
            + self.config.observation_quality_weight * predicted_quality
            - self.config.observation_cloud_risk_weight * cloud_risk
            - self.config.observation_retarget_weight * opportunity.slew_cost
            - self.config.observation_downlink_consequence_weight * downlink_consequence
        )
        diagnostics = {
            "expected_mission_value": expected_mission_value,
            "value_density": value_density,
            "freshness_bonus": freshness_bonus,
            "urgency_peak": urgency_peak,
            "predicted_quality": predicted_quality,
            "cloud_risk": cloud_risk,
            "retarget_cost": opportunity.slew_cost,
            "downlink_consequence": downlink_consequence,
            "data_volume_mb": opportunity.estimated_data_volume_mb,
        }
        return PlannerCandidate(
            action=action,
            score=round(score, 6),
            rationale=(
                "value-density observation score with explicit quality, risk, and queue effects"
            ),
            diagnostics=diagnostics,
            metadata={"formula": VALUE_DENSITY_OBSERVATION_FORMULA},
        )

    def _score_downlink(
        self,
        *,
        action: OrbitalAction,
        observation: Mapping[str, Any],
        window: DownlinkWindow,
    ) -> PlannerCandidate:
        queued_entries = [
            entry
            for entry in observation.get("onboard_queue", [])
            if str(entry["satellite_id"]) == window.satellite_id
        ]
        deliverable_volume_mb = self._window_capacity_mb(window)
        deliverable_entries = _fifo_deliverable_entries(
            queued_entries,
            max_volume_mb=deliverable_volume_mb,
        )
        queued_delivery_value = sum(
            self._queued_entry_value(entry=entry, at_time_utc=window.end_time_utc)
            for entry in deliverable_entries
            if bool(entry["usable"])
        )
        delivered_volume = sum(float(entry["data_volume_mb"]) for entry in deliverable_entries)
        queued_value_density = queued_delivery_value / max(delivered_volume, 1.0)
        latency_pressure = self._downlink_latency_pressure(
            entries=deliverable_entries,
            at_time_utc=window.end_time_utc,
        )
        buffer_capacity = self._buffer_capacity_by_satellite.get(window.satellite_id, 1.0)
        buffer_relief = (
            0.0 if buffer_capacity <= 0 else min(delivered_volume / buffer_capacity, 1.0)
        )
        score = (
            self.config.downlink_expected_value_weight * queued_delivery_value
            + self.config.downlink_value_density_weight * queued_value_density
            + self.config.downlink_freshness_weight * latency_pressure
            + self.config.downlink_buffer_relief_weight * buffer_relief
            - self.config.downlink_outage_risk_weight * window.outage_risk
        )
        diagnostics = {
            "queued_delivery_value": queued_delivery_value,
            "queued_value_density": queued_value_density,
            "latency_pressure": latency_pressure,
            "buffer_relief": buffer_relief,
            "outage_risk": window.outage_risk,
            "deliverable_volume_mb": delivered_volume,
        }
        return PlannerCandidate(
            action=action,
            score=round(score, 6),
            rationale=(
                "downlink queued value and relieve buffer pressure with explicit outage penalty"
            ),
            diagnostics=diagnostics,
            metadata={"formula": VALUE_DENSITY_DOWNLINK_FORMULA},
        )

    def projected_observation_value(
        self,
        *,
        opportunity: ObservationOpportunity,
        at_time_utc: datetime,
    ) -> float:
        if self._bundle is None:
            raise RuntimeError("bundle missing")
        target_static_value = self._target_value_by_id.get(opportunity.target_cell_id, 0.0)
        linked_incident_value = 0.0
        for incident_id in opportunity.incident_ids or ():
            if incident_id not in self._incident_ignition_by_id:
                continue
            freshness = self._incident_freshness(
                sim_time_utc=at_time_utc,
                incident_id=incident_id,
                observed_time_utc=None,
            )
            incident = next(
                item for item in self._bundle.incidents if item.incident_id == incident_id
            )
            linked_incident_value += target_static_value * incident.urgency_score * freshness
        return round(
            linked_incident_value
            * opportunity.predicted_quality_mean
            * (1.0 - opportunity.predicted_cloud_obstruction_prob),
            6,
        )

    def projected_downlink_queue_value(
        self,
        *,
        entries: Sequence[ProjectedQueueEntry],
        window: DownlinkWindow,
    ) -> float:
        deliverable_entries = _fifo_deliverable_entries_projected(
            entries=entries,
            max_volume_mb=self._window_capacity_mb(window),
        )
        return round(
            sum(
                self._projected_queue_entry_value(entry=entry, at_time_utc=window.end_time_utc)
                for entry in deliverable_entries
                if entry.usable
            ),
            6,
        )

    def _projected_queue_entry_value(
        self,
        *,
        entry: ProjectedQueueEntry,
        at_time_utc: datetime,
    ) -> float:
        age_hours = max(0.0, (at_time_utc - entry.available_time_utc).total_seconds() / 3600.0)
        freshness = 1.0 / (1.0 + (age_hours / max(self.config.freshness_half_life_hours, 0.1)))
        return round(entry.expected_value * freshness, 6)

    def _queued_entry_value(
        self,
        *,
        entry: Mapping[str, Any],
        at_time_utc: datetime,
    ) -> float:
        target_static_value = self._target_value_by_id.get(str(entry["target_cell_id"]), 0.0)
        incident_urgency = 0.0
        for incident_id in entry.get("incident_ids", []):
            if incident_id not in self._incident_ignition_by_id:
                continue
            if self._bundle is None:
                raise RuntimeError("bundle missing")
            incident = next(
                item for item in self._bundle.incidents if item.incident_id == incident_id
            )
            incident_urgency = max(incident_urgency, incident.urgency_score)
        age_hours = max(
            0.0,
            (at_time_utc - _coerce_datetime(entry["observation_time_utc"])).total_seconds()
            / 3600.0,
        )
        freshness = 1.0 / (1.0 + (age_hours / max(self.config.freshness_half_life_hours, 0.1)))
        return round(
            target_static_value * incident_urgency * freshness * float(entry["realized_quality"]),
            6,
        )

    def _downlink_latency_pressure(
        self,
        *,
        entries: Sequence[Mapping[str, Any]],
        at_time_utc: datetime,
    ) -> float:
        if not entries:
            return 0.0
        ages = [
            max(
                0.0,
                (at_time_utc - _coerce_datetime(entry["observation_time_utc"])).total_seconds()
                / 3600.0,
            )
            for entry in entries
        ]
        return min(sum(ages) / len(ages) / 2.0, 1.5)

    def _observation_downlink_consequence(
        self,
        *,
        satellite_id: str,
        current_time_utc: datetime,
        current_buffer_usage: float,
        added_volume_mb: float,
        completed_downlink_ids: frozenset[str],
    ) -> float:
        buffer_capacity = self._buffer_capacity_by_satellite.get(satellite_id, 1.0)
        projected_fill = 0.0
        if buffer_capacity > 0:
            projected_fill = min((current_buffer_usage + added_volume_mb) / buffer_capacity, 1.5)
        buffer_pressure = max(projected_fill - self.config.comfortable_buffer_fill_ratio, 0.0)
        next_window = next(
            (
                window
                for window in sorted(
                    self._downlink_by_id.values(),
                    key=lambda item: (item.start_time_utc, item.end_time_utc, item.window_id),
                )
                if window.satellite_id == satellite_id
                and window.window_id not in completed_downlink_ids
                and window.end_time_utc > current_time_utc
            ),
            None,
        )
        if next_window is None:
            gap_penalty = self.config.no_future_downlink_penalty
        else:
            gap_hours = max(
                0.0,
                (next_window.end_time_utc - current_time_utc).total_seconds() / 3600.0,
            )
            gap_penalty = min(
                gap_hours / max(self.config.future_downlink_gap_hours, 0.1),
                self.config.no_future_downlink_penalty,
            )
        return round((0.65 * buffer_pressure) + (0.35 * gap_penalty), 6)

    def _incident_freshness(
        self,
        *,
        sim_time_utc: datetime,
        incident_id: str,
        observed_time_utc: object | None,
    ) -> float:
        baseline_time = (
            _coerce_datetime(observed_time_utc)
            if observed_time_utc is not None
            else self._incident_ignition_by_id.get(incident_id, sim_time_utc)
        )
        age_hours = max(0.0, (sim_time_utc - baseline_time).total_seconds() / 3600.0)
        return round(
            1.0 / (1.0 + (age_hours / max(self.config.freshness_half_life_hours, 0.1))),
            6,
        )

    def _window_capacity_mb(self, window: DownlinkWindow) -> float:
        duration_seconds = (window.end_time_utc - window.start_time_utc).total_seconds()
        return min(window.max_volume_mb, (window.expected_rate_mbps * duration_seconds) / 8.0)


class ValueDensityGreedyPlanner:
    metadata = PlannerMetadata(
        planner_id="value_density_greedy",
        version="phase2-v1",
        description=(
            "Deterministic Phase 2 heuristic that ranks legal actions with a configurable "
            "value-density score over expected value, freshness, quality, risk, retarget cost, "
            "and downlink consequences."
        ),
    )

    def __init__(self, config: ValueDensityScoringConfig | None = None) -> None:
        self._scorer = ValueDensityHeuristic(config=config)

    def start_episode(
        self,
        *,
        context: PlannerEpisodeContext,
        initial_observation: Mapping[str, Any],
    ) -> None:
        del initial_observation
        self._scorer.start_episode(context.bundle)

    def select_action(self, observation: Mapping[str, Any]) -> PlannerDecision:
        candidates = [
            self._scorer.score_action(action, observation)
            for action in legal_actions_from_observation(observation)
        ]
        if not candidates:
            raise RuntimeError("action mask must contain at least one legal action")
        considered = tuple(sorted(candidates, key=_candidate_sort_key, reverse=True))
        selected = considered[0]
        return PlannerDecision(
            action=selected.action,
            score=selected.score,
            rationale=selected.rationale,
            diagnostics=selected.diagnostics,
            considered_candidates=considered,
            trace_details={
                "scoring_family": "value_density_greedy",
                "observation_formula": VALUE_DENSITY_OBSERVATION_FORMULA,
                "downlink_formula": VALUE_DENSITY_DOWNLINK_FORMULA,
                "scoring_config": asdict(self._scorer.config),
            },
        )


class OrtoolsRecedingHorizonPlanner:
    metadata = PlannerMetadata(
        planner_id="ortools_receding_horizon",
        version="phase2-v1",
        description=(
            "Deterministic OR-Tools receding-horizon planner that scores each current legal action "
            "using the same visible heuristic terms plus a small finite lookahead optimization."
        ),
    )

    def __init__(
        self,
        *,
        scoring_config: ValueDensityScoringConfig | None = None,
        lookahead_config: OrtoolsLookaheadConfig | None = None,
    ) -> None:
        self._scorer = ValueDensityHeuristic(config=scoring_config)
        self._lookahead = lookahead_config or OrtoolsLookaheadConfig()
        self._bundle: ScenarioBundle | None = None
        self._step_interval = timedelta(minutes=1)

    def start_episode(
        self,
        *,
        context: PlannerEpisodeContext,
        initial_observation: Mapping[str, Any],
    ) -> None:
        del initial_observation
        self._bundle = context.bundle
        self._step_interval = timedelta(seconds=context.bundle.decision_interval_seconds)
        self._scorer.start_episode(context.bundle)

    def select_action(self, observation: Mapping[str, Any]) -> PlannerDecision:
        if self._bundle is None:
            raise RuntimeError("start_episode() must be called before select_action()")
        current_candidates = [
            self._scorer.score_action(action, observation)
            for action in legal_actions_from_observation(observation)
        ]
        if not current_candidates:
            raise RuntimeError("action mask must contain at least one legal action")

        considered: list[PlannerCandidate] = []
        best_candidate: PlannerCandidate | None = None
        best_trace: dict[str, Any] | None = None
        for current_candidate in current_candidates:
            projected_state = self._project_after_current_action(
                observation=observation,
                candidate=current_candidate,
            )
            solver_trace = self._solve_future_plan(projected_state=projected_state)
            total_score = round(
                current_candidate.score
                + (self._lookahead.future_discount * solver_trace["objective_value"]),
                6,
            )
            candidate = PlannerCandidate(
                action=current_candidate.action,
                score=total_score,
                rationale=(
                    "current heuristic score plus discounted OR-Tools receding-horizon objective"
                ),
                diagnostics={
                    **current_candidate.diagnostics,
                    "immediate_score": current_candidate.score,
                    "lookahead_objective": float(solver_trace["objective_value"]),
                    "discounted_lookahead": round(
                        self._lookahead.future_discount * float(solver_trace["objective_value"]),
                        6,
                    ),
                },
                metadata={
                    "solver_status": solver_trace["status"],
                    "selected_future_actions": solver_trace["selected_actions"],
                    "projected_time_utc": projected_state.current_time_utc.isoformat().replace(
                        "+00:00", "Z"
                    ),
                },
            )
            considered.append(candidate)
            if best_candidate is None or _candidate_sort_key(candidate) > _candidate_sort_key(
                best_candidate
            ):
                best_candidate = candidate
                best_trace = {
                    "solver": solver_trace,
                    "projected_state": {
                        "projected_time_utc": projected_state.current_time_utc,
                        "buffer_usage_by_satellite": dict(
                            projected_state.buffer_usage_by_satellite
                        ),
                        "queue_entries": [
                            {
                                "queue_id": entry.queue_id,
                                "satellite_id": entry.satellite_id,
                                "source_ref": entry.source_ref,
                                "data_volume_mb": entry.data_volume_mb,
                                "expected_value": entry.expected_value,
                                "usable": entry.usable,
                            }
                            for entry in projected_state.queue_entries
                        ],
                    },
                }
        if best_candidate is None or best_trace is None:
            raise RuntimeError("lookahead planner could not evaluate any legal action")
        ordered = tuple(sorted(considered, key=_candidate_sort_key, reverse=True))
        return PlannerDecision(
            action=best_candidate.action,
            score=best_candidate.score,
            rationale=best_candidate.rationale,
            diagnostics=best_candidate.diagnostics,
            considered_candidates=ordered,
            trace_details={
                "scoring_family": "ortools_receding_horizon",
                "observation_formula": VALUE_DENSITY_OBSERVATION_FORMULA,
                "downlink_formula": VALUE_DENSITY_DOWNLINK_FORMULA,
                "scoring_config": asdict(self._scorer.config),
                "lookahead_config": asdict(self._lookahead),
                **best_trace,
            },
        )

    def _project_after_current_action(
        self,
        *,
        observation: Mapping[str, Any],
        candidate: PlannerCandidate,
    ) -> ProjectedLookaheadState:
        if self._bundle is None:
            raise RuntimeError("bundle missing")
        current_time_utc = _coerce_datetime(observation["sim_time_utc"])
        next_time_utc = current_time_utc + self._step_interval
        queue_entries = [
            ProjectedQueueEntry(
                queue_id=str(entry["queue_id"]),
                satellite_id=str(entry["satellite_id"]),
                target_cell_id=str(entry["target_cell_id"]),
                incident_ids=tuple(str(item) for item in entry.get("incident_ids", [])),
                data_volume_mb=float(entry["data_volume_mb"]),
                expected_value=self._scorer._queued_entry_value(
                    entry=entry,
                    at_time_utc=current_time_utc,
                ),
                available_time_utc=_coerce_datetime(entry["observation_time_utc"]),
                usable=bool(entry["usable"]),
                source_ref=str(entry["opportunity_id"]),
            )
            for entry in observation.get("onboard_queue", [])
        ]
        buffer_usage_by_satellite = {
            str(key): float(value)
            for key, value in observation.get("buffer_usage_by_satellite", {}).items()
        }
        completed_observation_ids = frozenset(
            str(item) for item in observation.get("completed_observation_ids", [])
        )
        completed_downlink_ids = frozenset(
            str(item) for item in observation.get("completed_downlink_ids", [])
        )
        incident_status_by_id = {
            str(incident["incident_id"]): str(incident["status"])
            for incident in observation.get("incidents", [])
        }
        action = candidate.action
        if action.action_type == "schedule_observation":
            opportunity = next(
                item
                for item in self._bundle.observation_opportunities
                if item.opportunity_id == action.ref
            )
            expected_value = self._scorer.projected_observation_value(
                opportunity=opportunity,
                at_time_utc=opportunity.end_time_utc,
            )
            queue_entries.append(
                ProjectedQueueEntry(
                    queue_id=stable_id("projq", opportunity.opportunity_id),
                    satellite_id=opportunity.satellite_id,
                    target_cell_id=opportunity.target_cell_id,
                    incident_ids=tuple(opportunity.incident_ids or ()),
                    data_volume_mb=opportunity.estimated_data_volume_mb,
                    expected_value=expected_value,
                    available_time_utc=opportunity.end_time_utc,
                    usable=expected_value > 0.0,
                    source_ref=opportunity.opportunity_id,
                )
            )
            buffer_usage_by_satellite[opportunity.satellite_id] = (
                buffer_usage_by_satellite.get(opportunity.satellite_id, 0.0)
                + opportunity.estimated_data_volume_mb
            )
            completed_observation_ids = frozenset(
                {*completed_observation_ids, opportunity.opportunity_id}
            )
        elif action.action_type == "schedule_downlink":
            window = next(
                item for item in self._bundle.downlink_windows if item.window_id == action.ref
            )
            delivered = _fifo_deliverable_entries_projected(
                entries=[
                    entry for entry in queue_entries if entry.satellite_id == window.satellite_id
                ],
                max_volume_mb=self._scorer._window_capacity_mb(window),
            )
            delivered_ids = {entry.queue_id for entry in delivered}
            queue_entries = [
                entry for entry in queue_entries if entry.queue_id not in delivered_ids
            ]
            buffer_usage_by_satellite[window.satellite_id] = max(
                0.0,
                buffer_usage_by_satellite.get(window.satellite_id, 0.0)
                - sum(entry.data_volume_mb for entry in delivered),
            )
            completed_downlink_ids = frozenset({*completed_downlink_ids, window.window_id})
        return ProjectedLookaheadState(
            current_time_utc=next_time_utc,
            buffer_usage_by_satellite=buffer_usage_by_satellite,
            queue_entries=tuple(
                sorted(
                    queue_entries,
                    key=lambda item: (item.available_time_utc, item.queue_id),
                )
            ),
            completed_observation_ids=completed_observation_ids,
            completed_downlink_ids=completed_downlink_ids,
            incident_status_by_id=incident_status_by_id,
        )

    def _solve_future_plan(
        self,
        *,
        projected_state: ProjectedLookaheadState,
    ) -> dict[str, Any]:
        try:
            from ortools.sat.python import cp_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ortools is required for the ortools_receding_horizon planner"
            ) from exc
        if self._bundle is None:
            raise RuntimeError("bundle missing")

        future_observations = self._future_observation_candidates(projected_state=projected_state)
        future_downlinks = self._future_downlink_candidates(projected_state=projected_state)
        model = cp_model.CpModel()
        obs_vars = {
            candidate.opportunity.opportunity_id: model.NewBoolVar(
                f"obs_{candidate.opportunity.opportunity_id.replace(':', '_')}"
            )
            for candidate in future_observations
        }
        downlink_vars = {
            candidate.window.window_id: model.NewBoolVar(
                f"dw_{candidate.window.window_id.replace(':', '_')}"
            )
            for candidate in future_downlinks
        }

        horizon_slots = range(max(self._lookahead.horizon_steps - 1, 0))
        for slot in horizon_slots:
            slot_vars = [
                obs_vars[candidate.opportunity.opportunity_id]
                for candidate in future_observations
                if candidate.decision_tick == slot
            ] + [
                downlink_vars[candidate.window.window_id]
                for candidate in future_downlinks
                if candidate.decision_tick == slot
            ]
            if slot_vars:
                model.Add(sum(slot_vars) <= 1)

        pair_vars: dict[tuple[str, str], Any] = {}
        objective_terms: list[Any] = []
        scale = 1000

        for candidate in future_observations:
            objective_terms.append(
                int(round(candidate.base_score * scale))
                * obs_vars[candidate.opportunity.opportunity_id]
            )
        for candidate in future_downlinks:
            objective_terms.append(
                int(round(candidate.base_score * scale)) * downlink_vars[candidate.window.window_id]
            )

        for obs_candidate in future_observations:
            for downlink_candidate in future_downlinks:
                if obs_candidate.opportunity.satellite_id != downlink_candidate.window.satellite_id:
                    continue
                if obs_candidate.completion_step > downlink_candidate.delivery_step:
                    continue
                pair_key = (
                    obs_candidate.opportunity.opportunity_id,
                    downlink_candidate.window.window_id,
                )
                pair_var = model.NewBoolVar(
                    "pair_"
                    f"{obs_candidate.opportunity.opportunity_id.replace(':', '_')}_"
                    f"{downlink_candidate.window.window_id.replace(':', '_')}"
                )
                pair_vars[pair_key] = pair_var
                model.Add(pair_var <= obs_vars[obs_candidate.opportunity.opportunity_id])
                model.Add(pair_var <= downlink_vars[downlink_candidate.window.window_id])
                model.Add(
                    pair_var
                    >= obs_vars[obs_candidate.opportunity.opportunity_id]
                    + downlink_vars[downlink_candidate.window.window_id]
                    - 1
                )
                delivered_value = obs_candidate.expected_delivery_value * (
                    1.0 - downlink_candidate.window.outage_risk
                )
                objective_terms.append(int(round(delivered_value * scale)) * pair_var)

        for obs_candidate in future_observations:
            candidate_pairs = [
                pair_vars[
                    (obs_candidate.opportunity.opportunity_id, downlink_candidate.window.window_id)
                ]
                for downlink_candidate in future_downlinks
                if (obs_candidate.opportunity.opportunity_id, downlink_candidate.window.window_id)
                in pair_vars
            ]
            if candidate_pairs:
                model.Add(
                    sum(candidate_pairs) <= obs_vars[obs_candidate.opportunity.opportunity_id]
                )

        for downlink_candidate in future_downlinks:
            pair_load_terms = []
            for obs_candidate in future_observations:
                pair_key = (
                    obs_candidate.opportunity.opportunity_id,
                    downlink_candidate.window.window_id,
                )
                if pair_key not in pair_vars:
                    continue
                pair_load_terms.append(
                    int(round(obs_candidate.opportunity.estimated_data_volume_mb * scale))
                    * pair_vars[pair_key]
                )
            if pair_load_terms:
                model.Add(
                    sum(pair_load_terms)
                    <= int(round(downlink_candidate.residual_capacity_mb * scale))
                    * downlink_vars[downlink_candidate.window.window_id]
                )

        for satellite_id, capacity in self._scorer._buffer_capacity_by_satellite.items():
            for step in range(1, self._lookahead.horizon_steps):
                additions = [
                    int(round(candidate.opportunity.estimated_data_volume_mb * scale))
                    * obs_vars[candidate.opportunity.opportunity_id]
                    for candidate in future_observations
                    if candidate.opportunity.satellite_id == satellite_id
                    and candidate.completion_step <= step
                ]
                removals = [
                    int(round(obs_candidate.opportunity.estimated_data_volume_mb * scale))
                    * pair_vars[
                        (
                            obs_candidate.opportunity.opportunity_id,
                            downlink_candidate.window.window_id,
                        )
                    ]
                    for obs_candidate in future_observations
                    for downlink_candidate in future_downlinks
                    if obs_candidate.opportunity.satellite_id == satellite_id
                    and (
                        obs_candidate.opportunity.opportunity_id,
                        downlink_candidate.window.window_id,
                    )
                    in pair_vars
                    and downlink_candidate.delivery_step <= step
                ]
                projected_usage = int(
                    round(projected_state.buffer_usage_by_satellite.get(satellite_id, 0.0) * scale)
                )
                model.Add(
                    projected_usage + sum(additions) - sum(removals) <= int(round(capacity * scale))
                )

        model.Maximize(sum(objective_terms))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._lookahead.max_solver_seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = self._lookahead.solver_random_seed
        status = solver.Solve(model)
        selected_actions = []
        for candidate in future_observations:
            if solver.Value(obs_vars[candidate.opportunity.opportunity_id]) == 1:
                selected_actions.append(
                    {
                        "decision_tick": candidate.decision_tick,
                        "action_type": "schedule_observation",
                        "action_ref": candidate.opportunity.opportunity_id,
                        "base_score": round(candidate.base_score, 6),
                    }
                )
        for candidate in future_downlinks:
            if solver.Value(downlink_vars[candidate.window.window_id]) == 1:
                selected_actions.append(
                    {
                        "decision_tick": candidate.decision_tick,
                        "action_type": "schedule_downlink",
                        "action_ref": candidate.window.window_id,
                        "base_score": round(candidate.base_score, 6),
                    }
                )
        selected_actions.sort(
            key=lambda item: (item["decision_tick"], item["action_type"], item["action_ref"])
        )
        return {
            "status": solver.StatusName(status),
            "objective_value": round(solver.ObjectiveValue() / scale, 6),
            "best_objective_bound": round(solver.BestObjectiveBound() / scale, 6),
            "future_observation_count": len(future_observations),
            "future_downlink_count": len(future_downlinks),
            "selected_actions": selected_actions,
        }

    def _future_observation_candidates(
        self,
        *,
        projected_state: ProjectedLookaheadState,
    ) -> list[FutureObservationCandidate]:
        if self._bundle is None:
            raise RuntimeError("bundle missing")
        candidates: list[FutureObservationCandidate] = []
        for opportunity in self._bundle.observation_opportunities:
            if opportunity.opportunity_id in projected_state.completed_observation_ids:
                continue
            if projected_state.incident_status_by_id and all(
                projected_state.incident_status_by_id.get(incident_id) in {"downlinked", "missed"}
                for incident_id in opportunity.incident_ids or ()
            ):
                continue
            decision_tick = _decision_tick_for_window(
                current_time_utc=projected_state.current_time_utc,
                first_legal_time_utc=self._observation_materialization_time(opportunity),
                end_time_utc=opportunity.end_time_utc,
                step_interval=self._step_interval,
            )
            if decision_tick is None or decision_tick >= max(self._lookahead.horizon_steps - 1, 0):
                continue
            completion_step = _step_index_for_time(
                current_time_utc=projected_state.current_time_utc,
                event_time_utc=opportunity.end_time_utc,
                step_interval=self._step_interval,
            )
            if completion_step >= self._lookahead.horizon_steps:
                continue
            expected_delivery_value = self._scorer.projected_observation_value(
                opportunity=opportunity,
                at_time_utc=opportunity.end_time_utc,
            )
            buffer_usage = float(
                projected_state.buffer_usage_by_satellite.get(opportunity.satellite_id, 0.0)
            )
            consequence = self._scorer._observation_downlink_consequence(
                satellite_id=opportunity.satellite_id,
                current_time_utc=projected_state.current_time_utc
                + (decision_tick * self._step_interval),
                current_buffer_usage=buffer_usage,
                added_volume_mb=opportunity.estimated_data_volume_mb,
                completed_downlink_ids=projected_state.completed_downlink_ids,
            )
            value_density = expected_delivery_value / max(opportunity.estimated_data_volume_mb, 1.0)
            base_score = (
                (0.4 * expected_delivery_value)
                + (0.2 * value_density)
                + (0.1 * opportunity.predicted_quality_mean)
                - (0.25 * opportunity.predicted_cloud_obstruction_prob)
                - (0.15 * opportunity.slew_cost)
                - (0.2 * consequence)
            )
            candidates.append(
                FutureObservationCandidate(
                    opportunity=opportunity,
                    decision_tick=decision_tick,
                    completion_step=completion_step,
                    base_score=round(base_score, 6),
                    expected_delivery_value=expected_delivery_value,
                )
            )
        candidates.sort(
            key=lambda item: (
                -item.base_score,
                item.decision_tick,
                item.opportunity.end_time_utc,
                item.opportunity.opportunity_id,
            )
        )
        return candidates[: self._lookahead.max_future_observations]

    def _future_downlink_candidates(
        self,
        *,
        projected_state: ProjectedLookaheadState,
    ) -> list[FutureDownlinkCandidate]:
        if self._bundle is None:
            raise RuntimeError("bundle missing")
        queue_entries_by_satellite: dict[str, list[ProjectedQueueEntry]] = {}
        for entry in projected_state.queue_entries:
            queue_entries_by_satellite.setdefault(entry.satellite_id, []).append(entry)
        earliest_window_by_satellite: dict[str, str] = {}
        for window in sorted(
            self._bundle.downlink_windows,
            key=lambda item: (item.start_time_utc, item.end_time_utc, item.window_id),
        ):
            if window.window_id in projected_state.completed_downlink_ids:
                continue
            if (
                window.satellite_id not in earliest_window_by_satellite
                and window.end_time_utc > projected_state.current_time_utc
            ):
                earliest_window_by_satellite[window.satellite_id] = window.window_id
        candidates: list[FutureDownlinkCandidate] = []
        for window in self._bundle.downlink_windows:
            if window.window_id in projected_state.completed_downlink_ids:
                continue
            decision_tick = _decision_tick_for_window(
                current_time_utc=projected_state.current_time_utc,
                first_legal_time_utc=window.start_time_utc,
                end_time_utc=window.end_time_utc,
                step_interval=self._step_interval,
            )
            if decision_tick is None or decision_tick >= max(self._lookahead.horizon_steps - 1, 0):
                continue
            delivery_step = _step_index_for_time(
                current_time_utc=projected_state.current_time_utc,
                event_time_utc=window.end_time_utc,
                step_interval=self._step_interval,
            )
            if delivery_step >= self._lookahead.horizon_steps:
                continue
            queue_entries = queue_entries_by_satellite.get(window.satellite_id, [])
            queued_value = self._scorer.projected_downlink_queue_value(
                entries=queue_entries, window=window
            )
            deliverable_volume = self._scorer._window_capacity_mb(window)
            buffer_capacity = self._scorer._buffer_capacity_by_satellite.get(
                window.satellite_id, 1.0
            )
            reserved_queue_volume = 0.0
            if earliest_window_by_satellite.get(window.satellite_id) == window.window_id:
                reserved_queue_volume = min(
                    sum(entry.data_volume_mb for entry in queue_entries),
                    deliverable_volume,
                )
            residual_capacity = max(deliverable_volume - reserved_queue_volume, 0.0)
            buffer_relief = (
                0.0 if buffer_capacity <= 0 else min(deliverable_volume / buffer_capacity, 1.0)
            )
            base_score = queued_value + (0.2 * buffer_relief) - (0.35 * window.outage_risk)
            candidates.append(
                FutureDownlinkCandidate(
                    window=window,
                    decision_tick=decision_tick,
                    delivery_step=delivery_step,
                    base_score=round(base_score, 6),
                    reserved_queue_volume_mb=reserved_queue_volume,
                    residual_capacity_mb=residual_capacity,
                )
            )
        candidates.sort(
            key=lambda item: (
                -item.base_score,
                item.decision_tick,
                item.window.end_time_utc,
                item.window.window_id,
            )
        )
        return candidates[: self._lookahead.max_future_downlinks]

    def _observation_materialization_time(self, opportunity: ObservationOpportunity) -> datetime:
        if self._bundle is None:
            raise RuntimeError("bundle missing")
        ignition_times = [
            incident.ignition_time_utc
            for incident in self._bundle.incidents
            if incident.incident_id in (opportunity.incident_ids or [])
        ]
        if not ignition_times:
            return opportunity.start_time_utc
        return min(opportunity.start_time_utc, min(ignition_times))


def _candidate_sort_key(candidate: PlannerCandidate) -> tuple[float, int, str]:
    action_priority = {
        "schedule_downlink": 2,
        "schedule_observation": 1,
        "noop": 0,
    }.get(candidate.action.action_type, -1)
    return (candidate.score, action_priority, _action_key(candidate.action))


def _decision_sort_key(decision: PlannerDecision) -> tuple[float, int, str]:
    return _candidate_sort_key(
        PlannerCandidate(
            action=decision.action,
            score=decision.score,
            rationale=decision.rationale,
            diagnostics=decision.diagnostics,
        )
    )


def _action_key(action: OrbitalAction) -> str:
    return f"{action.action_type}:{action.ref}"


def _decision_tick_for_window(
    *,
    current_time_utc: datetime,
    first_legal_time_utc: datetime,
    end_time_utc: datetime,
    step_interval: timedelta,
) -> int | None:
    if end_time_utc <= current_time_utc:
        return None
    step_seconds = max(step_interval.total_seconds(), 1.0)
    delta_seconds = max((first_legal_time_utc - current_time_utc).total_seconds(), 0.0)
    tick = int(ceil(delta_seconds / step_seconds))
    decision_time = current_time_utc + (tick * step_interval)
    if decision_time >= end_time_utc:
        return None
    return tick


def _step_index_for_time(
    *,
    current_time_utc: datetime,
    event_time_utc: datetime,
    step_interval: timedelta,
) -> int:
    if event_time_utc <= current_time_utc:
        return 0
    step_seconds = max(step_interval.total_seconds(), 1.0)
    return int(ceil((event_time_utc - current_time_utc).total_seconds() / step_seconds))


def _fifo_deliverable_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    max_volume_mb: float,
) -> list[Mapping[str, Any]]:
    delivered: list[Mapping[str, Any]] = []
    delivered_volume = 0.0
    for entry in sorted(
        entries,
        key=lambda item: (_coerce_datetime(item["observation_time_utc"]), str(item["queue_id"])),
    ):
        entry_volume = float(entry["data_volume_mb"])
        if delivered_volume + entry_volume > max_volume_mb + 1e-9:
            continue
        delivered.append(entry)
        delivered_volume += entry_volume
    return delivered


def _fifo_deliverable_entries_projected(
    *,
    entries: Sequence[ProjectedQueueEntry],
    max_volume_mb: float,
) -> list[ProjectedQueueEntry]:
    delivered: list[ProjectedQueueEntry] = []
    delivered_volume = 0.0
    for entry in sorted(entries, key=lambda item: (item.available_time_utc, item.queue_id)):
        if delivered_volume + entry.data_volume_mb > max_volume_mb + 1e-9:
            continue
        delivered.append(entry)
        delivered_volume += entry.data_volume_mb
    return delivered


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_compatible(asdict(value))
    if isinstance(value, float):
        return round(value, 6)
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
