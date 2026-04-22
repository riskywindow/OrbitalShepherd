from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from orbital_shepherd_contracts import DownlinkWindow, ObservationOpportunity, ScenarioBundle
from orbital_shepherd_env_runtime import OrbitalAction

try:  # pragma: no cover - exercised when numpy is installed.
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - covered by local tests.
    np = None


@dataclass(frozen=True, slots=True)
class FeatureDescriptor:
    name: str
    lower_bound: float | None
    upper_bound: float | None
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "description": self.description,
        }


GLOBAL_FEATURE_SPECS: tuple[FeatureDescriptor, ...] = (
    FeatureDescriptor("progress_ratio", 0.0, 1.0, "Current sim tick divided by horizon tick."),
    FeatureDescriptor("remaining_ratio", 0.0, 1.0, "Fraction of episode horizon still remaining."),
    FeatureDescriptor("mission_utility", None, None, "Cumulative reward emitted by the runtime."),
    FeatureDescriptor(
        "pending_observation_ratio",
        0.0,
        1.0,
        "Pending observation actions divided by total observation opportunities.",
    ),
    FeatureDescriptor(
        "pending_downlink_ratio",
        0.0,
        1.0,
        "Pending downlink actions divided by total downlink windows.",
    ),
    FeatureDescriptor(
        "completed_observation_ratio",
        0.0,
        1.0,
        "Completed observation actions divided by total observation opportunities.",
    ),
    FeatureDescriptor(
        "completed_downlink_ratio",
        0.0,
        1.0,
        "Completed downlink actions divided by total downlink windows.",
    ),
    FeatureDescriptor(
        "queue_entry_ratio",
        0.0,
        1.0,
        "Queued onboard observations divided by total observation opportunities.",
    ),
    FeatureDescriptor(
        "queue_volume_ratio",
        0.0,
        1.0,
        "Total queued data volume divided by total constellation buffer capacity.",
    ),
    FeatureDescriptor(
        "open_incident_ratio",
        0.0,
        1.0,
        "Incidents not yet downlinked or missed divided by total incidents.",
    ),
    FeatureDescriptor(
        "observed_incident_ratio",
        0.0,
        1.0,
        "Observed-but-not-downlinked incidents divided by total incidents.",
    ),
    FeatureDescriptor(
        "downlinked_incident_ratio",
        0.0,
        1.0,
        "Downlinked incidents divided by total incidents.",
    ),
    FeatureDescriptor(
        "missed_incident_ratio",
        0.0,
        1.0,
        "Missed incidents divided by total incidents.",
    ),
    FeatureDescriptor(
        "open_incident_urgency_mean",
        0.0,
        1.0,
        "Mean urgency of incidents still actionable at the current tick.",
    ),
    FeatureDescriptor(
        "open_incident_urgency_max",
        0.0,
        1.0,
        "Maximum urgency of incidents still actionable at the current tick.",
    ),
    FeatureDescriptor(
        "nominal_station_fraction",
        0.0,
        1.0,
        "Fraction of ground stations currently marked nominal.",
    ),
    FeatureDescriptor(
        "offline_station_fraction",
        0.0,
        1.0,
        "Fraction of ground stations currently marked offline.",
    ),
)


CANDIDATE_FEATURE_SPECS: tuple[FeatureDescriptor, ...] = (
    FeatureDescriptor("is_observation", 0.0, 1.0, "1 when the candidate schedules an observation."),
    FeatureDescriptor("is_downlink", 0.0, 1.0, "1 when the candidate schedules a downlink."),
    FeatureDescriptor("score_hint", 0.0, None, "Current runtime score hint for the candidate."),
    FeatureDescriptor(
        "rank_fraction",
        0.0,
        1.0,
        "Candidate rank index normalized over the current legal non-noop set.",
    ),
    FeatureDescriptor(
        "time_to_start_ratio",
        0.0,
        1.0,
        "Seconds until candidate start, normalized by episode duration.",
    ),
    FeatureDescriptor(
        "time_to_end_ratio",
        0.0,
        1.0,
        "Seconds until candidate end, normalized by episode duration.",
    ),
    FeatureDescriptor(
        "window_span_ratio",
        0.0,
        1.0,
        "Candidate window duration normalized by episode duration.",
    ),
    FeatureDescriptor(
        "satellite_buffer_fill_ratio",
        0.0,
        1.0,
        "Current onboard queued volume for the candidate satellite divided by its buffer capacity.",
    ),
    FeatureDescriptor(
        "satellite_pending_observation_ratio",
        0.0,
        1.0,
        "Pending observation volume for the candidate satellite divided by its buffer capacity.",
    ),
    FeatureDescriptor(
        "observation_predicted_quality_mean",
        0.0,
        1.0,
        "Predicted quality for observation candidates, else 0.",
    ),
    FeatureDescriptor(
        "observation_predicted_cloud_obstruction_prob",
        0.0,
        1.0,
        "Predicted cloud risk for observation candidates, else 0.",
    ),
    FeatureDescriptor(
        "observation_estimated_data_volume_ratio",
        0.0,
        1.0,
        "Observation estimated data volume divided by the satellite buffer capacity, else 0.",
    ),
    FeatureDescriptor("observation_slew_cost", 0.0, None, "Observation slew cost, else 0."),
    FeatureDescriptor(
        "observation_target_static_value",
        0.0,
        None,
        "Target-cell static value for observation candidates, else 0.",
    ),
    FeatureDescriptor(
        "observation_linked_incident_count_ratio",
        0.0,
        1.0,
        "Open incident links for the observation divided by total incidents, else 0.",
    ),
    FeatureDescriptor(
        "observation_linked_incident_urgency_mean",
        0.0,
        1.0,
        "Mean urgency over open incident links for the observation, else 0.",
    ),
    FeatureDescriptor(
        "observation_linked_incident_urgency_max",
        0.0,
        1.0,
        "Maximum urgency over open incident links for the observation, else 0.",
    ),
    FeatureDescriptor(
        "downlink_expected_rate_ratio",
        0.0,
        None,
        "Expected downlink rate divided by the satellite nominal downlink rate, else 0.",
    ),
    FeatureDescriptor("downlink_outage_risk", 0.0, 1.0, "Current downlink outage risk, else 0."),
    FeatureDescriptor(
        "downlink_max_volume_ratio",
        0.0,
        None,
        "Window max volume divided by the satellite buffer capacity, else 0.",
    ),
    FeatureDescriptor(
        "downlink_station_nominal",
        0.0,
        1.0,
        "1 when the downlink station is currently nominal, else 0.",
    ),
    FeatureDescriptor(
        "downlink_queued_entry_ratio",
        0.0,
        1.0,
        "Queued entries for the satellite divided by total observation opportunities, else 0.",
    ),
    FeatureDescriptor(
        "downlink_queued_usable_volume_ratio",
        0.0,
        1.0,
        "Usable queued volume for the satellite divided by its buffer capacity, else 0.",
    ),
    FeatureDescriptor(
        "downlink_queued_usable_urgency_mean",
        0.0,
        1.0,
        "Mean urgency of usable queued incidents for the satellite, else 0.",
    ),
    FeatureDescriptor(
        "downlink_queued_usable_urgency_max",
        0.0,
        1.0,
        "Maximum urgency of usable queued incidents for the satellite, else 0.",
    ),
)


@dataclass(frozen=True, slots=True)
class ProjectedActionSlot:
    slot_index: int
    action_id: str | None
    action_type: str
    action_ref: str | None
    source: str
    projected_rank: int | None
    runtime_action_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_ref": self.action_ref,
            "source": self.source,
            "projected_rank": self.projected_rank,
            "runtime_action_index": self.runtime_action_index,
        }


@dataclass(frozen=True, slots=True)
class CandidateProjection:
    projected_actions: tuple[OrbitalAction, ...]
    slot_mappings: tuple[ProjectedActionSlot, ...]
    projected_candidate_count: int
    candidate_count: int
    truncated_candidate_count: int


class TrainingObservationProjector:
    def __init__(
        self,
        bundle: ScenarioBundle | Mapping[str, Any],
        *,
        top_k: int = 64,
        action_order: Sequence[str] = ("noop", "schedule_observation", "schedule_downlink"),
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be >= 1")
        self.bundle = (
            bundle if isinstance(bundle, ScenarioBundle) else ScenarioBundle.model_validate(bundle)
        )
        self.top_k = top_k
        self._observation_by_id = {
            opportunity.opportunity_id: opportunity
            for opportunity in self.bundle.observation_opportunities
        }
        self._downlink_by_id = {window.window_id: window for window in self.bundle.downlink_windows}
        self._target_value_by_id = {
            target.target_cell_id: target.static_value for target in self.bundle.target_cells
        }
        self._satellite_buffer_capacity_by_id = {
            satellite.satellite_id: satellite.downlink.buffer_capacity_mb
            for satellite in self.bundle.satellites
        }
        self._satellite_nominal_downlink_rate_by_id = {
            satellite.satellite_id: satellite.downlink.nominal_downlink_rate_mbps
            for satellite in self.bundle.satellites
        }
        self._station_availability_by_id = {
            station.station_id: station.capabilities.availability
            for station in self.bundle.ground_stations
        }
        self._action_type_priority = {
            action_type: index for index, action_type in enumerate(action_order) if action_type != "noop"
        }
        self._episode_duration_seconds = max(
            (
                self.bundle.time_window.end_time_utc - self.bundle.time_window.start_time_utc
            ).total_seconds(),
            float(self.bundle.decision_interval_seconds),
        )
        self._latest_projection = self._empty_projection()

    @property
    def global_feature_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in GLOBAL_FEATURE_SPECS)

    @property
    def candidate_feature_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in CANDIDATE_FEATURE_SPECS)

    @property
    def latest_projection(self) -> CandidateProjection:
        return self._latest_projection

    @property
    def slot_mapping(self) -> list[dict[str, Any]]:
        return [slot.to_dict() for slot in self._latest_projection.slot_mappings]

    def normalization_metadata(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "global_features": [item.to_dict() for item in GLOBAL_FEATURE_SPECS],
            "candidate_features": [item.to_dict() for item in CANDIDATE_FEATURE_SPECS],
            "action_mask": {
                "shape": [self.top_k + 1],
                "slot_0": "noop",
                "dtype": "int8",
                "description": "1 for legal slots, 0 for padded or invalid slots.",
            },
        }

    def decode_action_slot(self, slot_index: int) -> dict[str, Any]:
        try:
            return self._latest_projection.slot_mappings[slot_index].to_dict()
        except IndexError as exc:
            raise ValueError(f"invalid training slot: {slot_index}") from exc

    def runtime_action_for_slot(self, slot_index: int) -> OrbitalAction | None:
        if slot_index == 0:
            return OrbitalAction(action_type="noop", ref="noop", score_hint=0.0)
        candidate_index = slot_index - 1
        if candidate_index < 0 or candidate_index >= len(self._latest_projection.projected_actions):
            return None
        return self._latest_projection.projected_actions[candidate_index]

    def project(
        self,
        *,
        raw_observation: Mapping[str, Any],
        runtime_actions: Sequence[OrbitalAction],
        raw_info: Mapping[str, Any],
        selected_slot: int | None = None,
        selected_mapping: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        terminal = bool(raw_observation.get("terminated") or raw_observation.get("truncated"))
        projection = self._build_projection(runtime_actions=runtime_actions, terminal=terminal)
        self._latest_projection = projection
        training_observation = {
            "global_features": self._extract_global_features(raw_observation),
            "candidate_features": self._extract_candidate_features(raw_observation, projection),
            "action_mask": self._build_training_action_mask(projection),
        }
        training_info = dict(raw_info)
        training_info.update(
            {
                "projected_candidate_count": projection.projected_candidate_count,
                "candidate_count": projection.candidate_count,
                "truncated_candidate_count": projection.truncated_candidate_count,
                "slot_to_action_id": [item.action_id for item in projection.slot_mappings],
                "candidate_ids": [item.action_id for item in projection.slot_mappings[1:]],
                "candidate_types": [item.action_type for item in projection.slot_mappings[1:]],
                "slot_mapping": [item.to_dict() for item in projection.slot_mappings],
            }
        )
        if selected_slot is not None and selected_mapping is not None:
            training_info["selected_slot"] = selected_slot
            training_info["selected_action_id"] = selected_mapping.get("action_id")
            training_info["selected_action_type"] = selected_mapping.get("action_type")
            training_info["selected_action_ref"] = selected_mapping.get("action_ref")
        return training_observation, training_info

    def _build_projection(
        self,
        *,
        runtime_actions: Sequence[OrbitalAction],
        terminal: bool,
    ) -> CandidateProjection:
        noop_slot = ProjectedActionSlot(
            slot_index=0,
            action_id=OrbitalAction(action_type="noop", ref="noop", score_hint=0.0).action_id,
            action_type="noop",
            action_ref="noop",
            source="noop",
            projected_rank=None,
            runtime_action_index=0,
        )
        if terminal:
            return self._empty_projection()
        legal_actions = tuple(runtime_actions[1:])
        ranked_actions = tuple(sorted(legal_actions, key=self._candidate_rank_key))
        projected_actions = ranked_actions[: self.top_k]
        runtime_index_by_action_id = {
            action.action_id: index for index, action in enumerate(runtime_actions)
        }
        slot_mappings = [noop_slot]
        for rank_index, action in enumerate(projected_actions):
            slot_mappings.append(
                ProjectedActionSlot(
                    slot_index=rank_index + 1,
                    action_id=action.action_id,
                    action_type=action.action_type,
                    action_ref=action.ref,
                    source="projected",
                    projected_rank=rank_index,
                    runtime_action_index=runtime_index_by_action_id.get(action.action_id),
                )
            )
        for slot_index in range(len(projected_actions) + 1, self.top_k + 1):
            slot_mappings.append(
                ProjectedActionSlot(
                    slot_index=slot_index,
                    action_id=None,
                    action_type="padding",
                    action_ref=None,
                    source="padding",
                    projected_rank=None,
                    runtime_action_index=None,
                )
            )
        return CandidateProjection(
            projected_actions=projected_actions,
            slot_mappings=tuple(slot_mappings),
            projected_candidate_count=len(projected_actions),
            candidate_count=len(ranked_actions),
            truncated_candidate_count=max(len(ranked_actions) - len(projected_actions), 0),
        )

    def _candidate_rank_key(self, action: OrbitalAction) -> tuple[Any, ...]:
        if action.action_type == "schedule_observation":
            candidate = self._observation_by_id[action.ref]
            return (
                self._action_type_priority.get(action.action_type, 999),
                -float(action.score_hint or 0.0),
                candidate.end_time_utc,
                candidate.start_time_utc,
                action.ref,
                action.action_id,
            )
        if action.action_type == "schedule_downlink":
            window = self._downlink_by_id[action.ref]
            return (
                self._action_type_priority.get(action.action_type, 999),
                -float(action.score_hint or 0.0),
                window.end_time_utc,
                window.start_time_utc,
                action.ref,
                action.action_id,
            )
        return (
            self._action_type_priority.get(action.action_type, 999),
            -float(action.score_hint or 0.0),
            action.ref,
            action.action_id,
        )

    def _extract_global_features(self, raw_observation: Mapping[str, Any]) -> Any:
        total_observations = max(len(self.bundle.observation_opportunities), 1)
        total_downlinks = max(len(self.bundle.downlink_windows), 1)
        total_incidents = max(len(self.bundle.incidents), 1)
        total_buffer_capacity = max(sum(self._satellite_buffer_capacity_by_id.values()), 1.0)
        horizon_tick = max(int(raw_observation["horizon_tick"]), 1)
        sim_tick = int(raw_observation["sim_tick"])
        queue_volume = sum(
            float(entry["data_volume_mb"]) for entry in raw_observation.get("onboard_queue", [])
        )
        incidents = raw_observation.get("incidents", [])
        open_incidents = [
            incident for incident in incidents if str(incident["status"]) not in {"downlinked", "missed"}
        ]
        observed_incidents = [
            incident for incident in incidents if str(incident["status"]) == "observed"
        ]
        downlinked_incidents = [
            incident for incident in incidents if str(incident["status"]) == "downlinked"
        ]
        missed_incidents = [
            incident for incident in incidents if str(incident["status"]) == "missed"
        ]
        station_states = raw_observation.get("ground_station_availability", {})
        nominal_station_count = sum(
            1 for station in station_states.values() if str(station["availability"]) == "nominal"
        )
        offline_station_count = sum(
            1 for station in station_states.values() if str(station["availability"]) == "offline"
        )
        total_stations = max(len(station_states), 1)
        open_urgencies = [float(incident["urgency_score"]) for incident in open_incidents]
        values = (
            _bounded_ratio(sim_tick, horizon_tick),
            _bounded_ratio(max(horizon_tick - sim_tick, 0), horizon_tick),
            round(float(raw_observation["mission_utility"]), 6),
            _bounded_ratio(len(raw_observation.get("pending_observation_ids", [])), total_observations),
            _bounded_ratio(len(raw_observation.get("pending_downlink_ids", [])), total_downlinks),
            _bounded_ratio(len(raw_observation.get("completed_observation_ids", [])), total_observations),
            _bounded_ratio(len(raw_observation.get("completed_downlink_ids", [])), total_downlinks),
            _bounded_ratio(len(raw_observation.get("onboard_queue", [])), total_observations),
            _bounded_ratio(queue_volume, total_buffer_capacity),
            _bounded_ratio(len(open_incidents), total_incidents),
            _bounded_ratio(len(observed_incidents), total_incidents),
            _bounded_ratio(len(downlinked_incidents), total_incidents),
            _bounded_ratio(len(missed_incidents), total_incidents),
            _bounded_mean(open_urgencies),
            max((round(value, 6) for value in open_urgencies), default=0.0),
            _bounded_ratio(nominal_station_count, total_stations),
            _bounded_ratio(offline_station_count, total_stations),
        )
        return _float_vector(values)

    def _extract_candidate_features(
        self,
        raw_observation: Mapping[str, Any],
        projection: CandidateProjection,
    ) -> Any:
        sim_time = _coerce_datetime(raw_observation["sim_time_utc"])
        pending_volume_by_satellite = self._pending_volume_by_satellite(raw_observation)
        queue_entries_by_satellite: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for entry in raw_observation.get("onboard_queue", []):
            queue_entries_by_satellite[str(entry["satellite_id"])].append(entry)
        incident_by_id = {
            str(incident["incident_id"]): incident for incident in raw_observation.get("incidents", [])
        }
        rows = []
        candidate_count = max(projection.candidate_count, 1)
        for rank_index, action in enumerate(projection.projected_actions):
            satellite_id = action.satellite_id or ""
            buffer_capacity = max(self._satellite_buffer_capacity_by_id.get(satellite_id, 0.0), 1.0)
            queue_entries = queue_entries_by_satellite.get(satellite_id, [])
            queued_volume = sum(float(entry["data_volume_mb"]) for entry in queue_entries)
            pending_volume = pending_volume_by_satellite.get(satellite_id, 0.0)
            common = [
                1.0 if action.action_type == "schedule_observation" else 0.0,
                1.0 if action.action_type == "schedule_downlink" else 0.0,
                round(float(action.score_hint or 0.0), 6),
                _bounded_ratio(rank_index, max(candidate_count - 1, 1)),
            ]
            if action.action_type == "schedule_observation":
                rows.append(
                    common
                    + self._observation_feature_suffix(
                        action=action,
                        sim_time=sim_time,
                        queue_volume=queued_volume,
                        pending_volume=pending_volume,
                        buffer_capacity=buffer_capacity,
                        incident_by_id=incident_by_id,
                    )
                )
                continue
            rows.append(
                common
                + self._downlink_feature_suffix(
                    action=action,
                    sim_time=sim_time,
                    queue_entries=queue_entries,
                    pending_volume=pending_volume,
                    buffer_capacity=buffer_capacity,
                    incident_by_id=incident_by_id,
                )
            )
        padding_row = [0.0] * len(CANDIDATE_FEATURE_SPECS)
        while len(rows) < self.top_k:
            rows.append(list(padding_row))
        return _float_matrix(rows)

    def _observation_feature_suffix(
        self,
        *,
        action: OrbitalAction,
        sim_time: datetime,
        queue_volume: float,
        pending_volume: float,
        buffer_capacity: float,
        incident_by_id: Mapping[str, Mapping[str, Any]],
    ) -> list[float]:
        opportunity = self._observation_by_id[action.ref]
        open_linked_incidents = [
            incident_by_id[incident_id]
            for incident_id in opportunity.incident_ids or ()
            if incident_id in incident_by_id
            and str(incident_by_id[incident_id]["status"]) not in {"downlinked", "missed"}
        ]
        urgencies = [float(incident["urgency_score"]) for incident in open_linked_incidents]
        total_incidents = max(len(self.bundle.incidents), 1)
        return [
            _time_ratio(opportunity.start_time_utc, sim_time, self._episode_duration_seconds),
            _time_ratio(opportunity.end_time_utc, sim_time, self._episode_duration_seconds),
            _duration_ratio(
                opportunity.start_time_utc,
                opportunity.end_time_utc,
                self._episode_duration_seconds,
            ),
            _bounded_ratio(queue_volume, buffer_capacity),
            _bounded_ratio(pending_volume, buffer_capacity),
            round(opportunity.predicted_quality_mean, 6),
            round(opportunity.predicted_cloud_obstruction_prob, 6),
            _bounded_ratio(opportunity.estimated_data_volume_mb, buffer_capacity),
            round(opportunity.slew_cost, 6),
            round(float(self._target_value_by_id.get(opportunity.target_cell_id, 0.0)), 6),
            _bounded_ratio(len(open_linked_incidents), total_incidents),
            _bounded_mean(urgencies),
            max((round(value, 6) for value in urgencies), default=0.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def _downlink_feature_suffix(
        self,
        *,
        action: OrbitalAction,
        sim_time: datetime,
        queue_entries: Sequence[Mapping[str, Any]],
        pending_volume: float,
        buffer_capacity: float,
        incident_by_id: Mapping[str, Mapping[str, Any]],
    ) -> list[float]:
        window = self._downlink_by_id[action.ref]
        satellite_id = action.satellite_id or ""
        usable_entries = [entry for entry in queue_entries if bool(entry["usable"])]
        usable_volume = sum(float(entry["data_volume_mb"]) for entry in usable_entries)
        usable_urgencies = [
            float(incident_by_id[incident_id]["urgency_score"])
            for entry in usable_entries
            for incident_id in entry.get("incident_ids", [])
            if incident_id in incident_by_id
            and str(incident_by_id[incident_id]["status"]) not in {"downlinked", "missed"}
        ]
        station_is_nominal = (
            self._station_availability_by_id.get(window.station_id, "offline") == "nominal"
        )
        total_observations = max(len(self.bundle.observation_opportunities), 1)
        nominal_rate = max(self._satellite_nominal_downlink_rate_by_id.get(satellite_id, 0.0), 1.0)
        queued_volume = sum(float(entry["data_volume_mb"]) for entry in queue_entries)
        return [
            _time_ratio(window.start_time_utc, sim_time, self._episode_duration_seconds),
            _time_ratio(window.end_time_utc, sim_time, self._episode_duration_seconds),
            _duration_ratio(
                window.start_time_utc,
                window.end_time_utc,
                self._episode_duration_seconds,
            ),
            _bounded_ratio(queued_volume, buffer_capacity),
            _bounded_ratio(pending_volume, buffer_capacity),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            round(window.expected_rate_mbps / nominal_rate, 6),
            round(window.outage_risk, 6),
            round(window.max_volume_mb / buffer_capacity, 6),
            1.0 if station_is_nominal else 0.0,
            _bounded_ratio(len(queue_entries), total_observations),
            _bounded_ratio(usable_volume, buffer_capacity),
            _bounded_mean(usable_urgencies),
            max((round(value, 6) for value in usable_urgencies), default=0.0),
        ]

    def _build_training_action_mask(self, projection: CandidateProjection) -> Any:
        mask = [1]
        mask.extend([1] * projection.projected_candidate_count)
        while len(mask) < self.top_k + 1:
            mask.append(0)
        return _int_vector(mask)

    def _pending_volume_by_satellite(self, raw_observation: Mapping[str, Any]) -> dict[str, float]:
        volumes: dict[str, float] = defaultdict(float)
        for opportunity_id in raw_observation.get("pending_observation_ids", []):
            opportunity = self._observation_by_id.get(str(opportunity_id))
            if opportunity is None:
                continue
            volumes[opportunity.satellite_id] += opportunity.estimated_data_volume_mb
        return {key: round(value, 6) for key, value in volumes.items()}

    def _empty_projection(self) -> CandidateProjection:
        slot_mappings = [ProjectedActionSlot(0, None, "noop", "noop", "noop", None, 0)]
        for slot_index in range(1, self.top_k + 1):
            slot_mappings.append(
                ProjectedActionSlot(slot_index, None, "padding", None, "padding", None, None)
            )
        return CandidateProjection(
            projected_actions=(),
            slot_mappings=tuple(slot_mappings),
            projected_candidate_count=0,
            candidate_count=0,
            truncated_candidate_count=0,
        )


def flatten_training_observation(observation: Mapping[str, Any]) -> Any:
    values: list[float] = []
    values.extend(_flatten_values(observation.get("global_features", ())))
    values.extend(_flatten_values(observation.get("candidate_features", ())))
    return _float_vector(values)


def _bounded_ratio(numerator: float | int, denominator: float | int) -> float:
    denom = max(float(denominator), 1.0)
    return round(min(max(float(numerator) / denom, 0.0), 1.0), 6)


def _bounded_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _time_ratio(target_time: datetime, sim_time: datetime, episode_duration_seconds: float) -> float:
    delta = max((target_time - sim_time).total_seconds(), 0.0)
    return _bounded_ratio(delta, episode_duration_seconds)


def _duration_ratio(
    start_time: datetime,
    end_time: datetime,
    episode_duration_seconds: float,
) -> float:
    span = max((end_time - start_time).total_seconds(), 0.0)
    return _bounded_ratio(span, episode_duration_seconds)


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _flatten_values(value: Any) -> list[float]:
    if value is None:
        return []
    if np is not None and hasattr(value, "tolist"):  # pragma: no cover - numpy path.
        return _flatten_values(value.tolist())
    if isinstance(value, (list, tuple)):
        items: list[float] = []
        for item in value:
            items.extend(_flatten_values(item))
        return items
    return [round(float(value), 6)]


def _float_vector(values: Sequence[float]) -> Any:
    rounded = tuple(round(float(value), 6) for value in values)
    if np is None:
        return rounded
    return np.asarray(rounded, dtype=np.float32)


def _float_matrix(rows: Sequence[Sequence[float]]) -> Any:
    rounded = tuple(tuple(round(float(value), 6) for value in row) for row in rows)
    if np is None:
        return rounded
    return np.asarray(rounded, dtype=np.float32)


def _int_vector(values: Sequence[int]) -> Any:
    casted = tuple(int(value) for value in values)
    if np is None:
        return casted
    return np.asarray(casted, dtype=np.int8)
