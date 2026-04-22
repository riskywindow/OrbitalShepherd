from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from math import ceil
from typing import Any, Literal

from orbital_shepherd_contracts import (
    DownlinkWindow,
    IncidentPacket,
    ObservationOpportunity,
    ReplayEvent,
    ScenarioBundle,
)
from orbital_shepherd_core import canonical_json_dumps, seeded_rng, sha256_fingerprint, stable_id
from orbital_shepherd_env_runtime.config import EnvRuntimeConfig
from orbital_shepherd_env_runtime.replay import ReplayEventEmitter, ReplaySink

ActionType = Literal["noop", "schedule_observation", "schedule_downlink"]
IncidentRuntimeStatus = Literal["unseen", "observed", "downlinked", "missed"]


@dataclass(frozen=True, slots=True)
class OrbitalAction:
    action_type: ActionType
    ref: str
    score_hint: float | None = None
    satellite_id: str | None = None
    target_cell_id: str | None = None

    @property
    def action_id(self) -> str:
        return stable_id("act", self.action_type, self.ref)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_ref": self.ref,
        }
        if self.score_hint is not None:
            payload["score_hint"] = self.score_hint
        if self.satellite_id is not None:
            payload["satellite_id"] = self.satellite_id
        if self.target_cell_id is not None:
            payload["target_cell_id"] = self.target_cell_id
        return payload


@dataclass(frozen=True, slots=True)
class OrbitalActionMask:
    mask_id: str
    actions: tuple[OrbitalAction, ...]

    def index_of(self, action_id: str) -> int:
        for index, action in enumerate(self.actions):
            if action.action_id == action_id:
                return index
        raise KeyError(action_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_id": self.mask_id,
            "legal_action_count": len(self.actions),
            "actions": [action.to_payload() for action in self.actions],
        }


@dataclass(slots=True)
class IncidentRuntimeRecord:
    incident_id: str
    target_cell_id: str
    urgency_score: float
    status: IncidentRuntimeStatus = "unseen"
    observed_time_utc: datetime | None = None
    downlinked_time_utc: datetime | None = None
    missed_time_utc: datetime | None = None
    last_opportunity_id: str | None = None
    last_packet_id: str | None = None


@dataclass(slots=True)
class QueueEntry:
    queue_id: str
    satellite_id: str
    opportunity_id: str
    observation_time_utc: datetime
    target_cell_id: str
    data_volume_mb: float
    realized_quality: float
    cloud_fraction: float
    usable: bool
    incident_ids: tuple[str, ...]


@dataclass(slots=True)
class OrbitalState:
    episode_id: str
    episode_seed: int
    sim_tick: int
    sim_time_utc: datetime
    horizon_tick: int
    terminated: bool = False
    truncated: bool = False
    mission_utility: float = 0.0
    pending_observation_ids: set[str] = field(default_factory=set)
    pending_downlink_ids: set[str] = field(default_factory=set)
    completed_observation_ids: set[str] = field(default_factory=set)
    completed_downlink_ids: set[str] = field(default_factory=set)
    onboard_queue: list[QueueEntry] = field(default_factory=list)
    incident_records: dict[str, IncidentRuntimeRecord] = field(default_factory=dict)
    incident_packets: list[IncidentPacket] = field(default_factory=list)

    def to_observation(
        self,
        *,
        bundle: ScenarioBundle,
        action_mask: OrbitalActionMask,
    ) -> dict[str, Any]:
        buffer_usage_by_satellite = {
            satellite.satellite_id: round(
                sum(
                    entry.data_volume_mb
                    for entry in self.onboard_queue
                    if entry.satellite_id == satellite.satellite_id
                ),
                6,
            )
            for satellite in sorted(bundle.satellites, key=lambda item: item.satellite_id)
        }
        ground_station_availability = {}
        for station in sorted(bundle.ground_stations, key=lambda item: item.station_id):
            ground_station_availability[station.station_id] = {
                "availability": station.capabilities.availability,
                "max_concurrent_contacts": station.capabilities.max_concurrent_contacts,
            }
        incidents = []
        for incident_id in sorted(self.incident_records):
            record = self.incident_records[incident_id]
            incidents.append(
                {
                    "incident_id": record.incident_id,
                    "target_cell_id": record.target_cell_id,
                    "urgency_score": record.urgency_score,
                    "status": record.status,
                    "observed_time_utc": record.observed_time_utc,
                    "downlinked_time_utc": record.downlinked_time_utc,
                    "missed_time_utc": record.missed_time_utc,
                    "last_opportunity_id": record.last_opportunity_id,
                    "last_packet_id": record.last_packet_id,
                }
            )
        onboard_queue = [
            {
                "queue_id": entry.queue_id,
                "satellite_id": entry.satellite_id,
                "opportunity_id": entry.opportunity_id,
                "observation_time_utc": entry.observation_time_utc,
                "target_cell_id": entry.target_cell_id,
                "data_volume_mb": entry.data_volume_mb,
                "realized_quality": entry.realized_quality,
                "cloud_fraction": entry.cloud_fraction,
                "usable": entry.usable,
                "incident_ids": list(entry.incident_ids),
            }
            for entry in self.onboard_queue
        ]
        return {
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "sim_tick": self.sim_tick,
            "sim_time_utc": self.sim_time_utc,
            "horizon_tick": self.horizon_tick,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "mission_utility": round(self.mission_utility, 6),
            "pending_observation_ids": sorted(self.pending_observation_ids),
            "pending_downlink_ids": sorted(self.pending_downlink_ids),
            "completed_observation_ids": sorted(self.completed_observation_ids),
            "completed_downlink_ids": sorted(self.completed_downlink_ids),
            "buffer_usage_by_satellite": buffer_usage_by_satellite,
            "ground_station_availability": ground_station_availability,
            "incidents": incidents,
            "onboard_queue": onboard_queue,
            "action_mask": action_mask.to_dict(),
        }


class OrbitalEnv:
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        bundle: ScenarioBundle | Mapping[str, Any],
        *,
        config: EnvRuntimeConfig | None = None,
        replay_sinks: Sequence[ReplaySink] = (),
        observation_opportunities: Sequence[ObservationOpportunity] | None = None,
        downlink_windows: Sequence[DownlinkWindow] | None = None,
    ) -> None:
        self.bundle = (
            bundle if isinstance(bundle, ScenarioBundle) else ScenarioBundle.model_validate(bundle)
        )
        self.config = config or EnvRuntimeConfig(
            decision_interval_seconds=self.bundle.decision_interval_seconds
        )
        if self.config.decision_interval_seconds != self.bundle.decision_interval_seconds:
            raise ValueError(
                "env decision interval must match the scenario bundle decision interval"
            )
        self._observation_opportunities = sorted(
            observation_opportunities or self.bundle.observation_opportunities,
            key=lambda item: (item.start_time_utc, item.end_time_utc, item.opportunity_id),
        )
        self._downlink_windows = sorted(
            downlink_windows or self.bundle.downlink_windows,
            key=lambda item: (item.start_time_utc, item.end_time_utc, item.window_id),
        )
        self._incident_by_id = {
            incident.incident_id: incident for incident in self.bundle.incidents
        }
        self._observation_by_id = {
            opportunity.opportunity_id: opportunity
            for opportunity in self._observation_opportunities
        }
        self._downlink_by_id = {window.window_id: window for window in self._downlink_windows}
        self._sinks = list(replay_sinks)
        self._step_interval = timedelta(seconds=self.config.decision_interval_seconds)
        total_seconds = (
            self.bundle.time_window.end_time_utc - self.bundle.time_window.start_time_utc
        ).total_seconds()
        self._horizon_steps = ceil(total_seconds / self.config.decision_interval_seconds)
        self._state: OrbitalState | None = None
        self._action_mask: OrbitalActionMask | None = None
        self._replay_events: list[ReplayEvent] = []
        self._replay_emitter: ReplayEventEmitter | None = None
        self._packet_sequence_by_incident: dict[str, int] = defaultdict(int)
        self._planner_id = self.config.planner_id

    @property
    def state(self) -> OrbitalState:
        if self._state is None:
            raise RuntimeError("reset() must be called before accessing state")
        return self._state

    @property
    def action_mask(self) -> OrbitalActionMask:
        if self._action_mask is None:
            raise RuntimeError("reset() must be called before accessing the action mask")
        return self._action_mask

    @property
    def replay_events(self) -> list[ReplayEvent]:
        return list(self._replay_events)

    def reset(
        self,
        *,
        seed: int | None = None,
        planner_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        episode_seed = self.bundle.simulation_seed if seed is None else seed
        episode_id = stable_id(
            "ep",
            self.bundle.benchmark_id,
            self.bundle.scenario_family,
            f"seed-{episode_seed}",
        )
        incident_records = {
            incident.incident_id: IncidentRuntimeRecord(
                incident_id=incident.incident_id,
                target_cell_id=incident.target_cell_id,
                urgency_score=incident.urgency_score,
            )
            for incident in self.bundle.incidents
        }
        self._state = OrbitalState(
            episode_id=episode_id,
            episode_seed=episode_seed,
            sim_tick=0,
            sim_time_utc=self.bundle.time_window.start_time_utc,
            horizon_tick=max(self._horizon_steps - 1, 0),
            incident_records=incident_records,
        )
        self._packet_sequence_by_incident = defaultdict(int)
        self._replay_events = []
        self._replay_emitter = ReplayEventEmitter(
            episode_id=episode_id,
            default_actor_id=self.config.env_id,
            sinks=[_ReplayListSink(self._replay_events), *self._sinks],
        )
        active_planner_id = planner_id or self.config.planner_id
        self._planner_id = active_planner_id
        self._emit_reset_events(planner_id=active_planner_id)
        self._action_mask = self._build_action_mask()
        self._emit_candidate_and_mask_events()
        observation = self.state.to_observation(bundle=self.bundle, action_mask=self.action_mask)
        return observation, self._build_info(events_since=0, reward=0.0, reward_components={})

    def step(
        self,
        action: int | str | OrbitalAction | Mapping[str, Any] | None,
        *,
        planner_trace: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        state = self.state
        if state.terminated or state.truncated:
            raise RuntimeError("cannot step a terminated environment; call reset() first")
        step_start_event_index = len(self._replay_events)
        selected_action = self._resolve_action(action)
        self._emit(
            event_type="action_selected",
            sim_tick=state.sim_tick,
            sim_time_utc=state.sim_time_utc,
            actor_type="planner",
            actor_id=self._planner_id,
            payload={
                "action_type": selected_action.action_type,
                "action_ref": selected_action.ref,
                "score": selected_action.score_hint,
                "action_id": selected_action.action_id,
                **({"planner_trace": dict(planner_trace)} if planner_trace is not None else {}),
            },
        )
        self._apply_selected_action(selected_action)
        previous_time = state.sim_time_utc
        next_time = previous_time + self._step_interval
        interval_end = min(next_time, self.bundle.time_window.end_time_utc)
        reward_components = self._process_interval(
            end_time=interval_end,
        )
        state.sim_tick += 1
        state.sim_time_utc = interval_end
        self._refresh_incident_statuses(interval_end)
        reward_components["buffer_pressure_penalty"] = round(
            reward_components.get("buffer_pressure_penalty", 0.0) + self._buffer_pressure_penalty(),
            6,
        )
        reward = round(sum(reward_components.values()), 6)
        state.mission_utility = round(state.mission_utility + reward, 6)
        should_terminate = interval_end >= self.bundle.time_window.end_time_utc
        should_truncate = (
            self.config.max_steps is not None and state.sim_tick >= self.config.max_steps
        )
        if self.config.include_zero_reward_events or reward != 0.0:
            self._emit(
                event_type="reward_assessed",
                sim_tick=state.sim_tick,
                sim_time_utc=interval_end,
                actor_type="metric_engine",
                actor_id=self.config.reward_actor_id,
                payload={
                    "total_reward": reward,
                    "components": {
                        key: round(value, 6) for key, value in reward_components.items()
                    },
                },
            )
        state.terminated = should_terminate and not should_truncate
        state.truncated = should_truncate
        if state.terminated or state.truncated:
            self._finalize_incident_statuses(end_time=interval_end)
            self._emit(
                event_type="episode_ended",
                sim_tick=state.sim_tick,
                sim_time_utc=interval_end,
                actor_type="metric_engine",
                actor_id=self.config.metrics_actor_id,
                payload={
                    "terminated": state.terminated,
                    "truncated": state.truncated,
                    "mission_utility": state.mission_utility,
                    "undelivered_queue_entries": len(state.onboard_queue),
                },
            )
        else:
            self._action_mask = self._build_action_mask()
            self._emit_candidate_and_mask_events()
        observation = self.state.to_observation(bundle=self.bundle, action_mask=self.action_mask)
        info = self._build_info(
            events_since=step_start_event_index,
            reward=reward,
            reward_components=reward_components,
        )
        return observation, reward, state.terminated, state.truncated, info

    def close(self) -> None:
        return None

    def _emit_reset_events(self, *, planner_id: str) -> None:
        episode_fingerprint = sha256_fingerprint(
            canonical_json_dumps(
                {
                    "bundle": self.bundle.model_dump(mode="json", exclude_none=True),
                    "config": {
                        key: value
                        for key, value in asdict(self.config).items()
                        if key != "replay_dir"
                    },
                    "episode_seed": self.state.episode_seed,
                    "planner_id": planner_id,
                }
            )
        )
        self._emit(
            event_type="scenario_loaded",
            sim_tick=0,
            sim_time_utc=self.state.sim_time_utc,
            actor_type="system",
            actor_id=self.config.env_id,
            payload={
                "bundle_id": self.bundle.bundle_id,
                "manifest_id": self.bundle.manifest_id,
                "scenario_family": self.bundle.scenario_family,
                "bundle_fingerprint": self.bundle.bundle_fingerprint,
            },
        )
        self._emit(
            event_type="episode_started",
            sim_tick=0,
            sim_time_utc=self.state.sim_time_utc,
            actor_type="planner",
            actor_id=planner_id,
            payload={
                "planner_id": planner_id,
                "episode_fingerprint": f"sha256:{episode_fingerprint}",
                "episode_seed": self.state.episode_seed,
                **(
                    {"planner_metadata": dict(self.config.planner_metadata)}
                    if self.config.planner_metadata
                    else {}
                ),
            },
        )

    def _emit_candidate_and_mask_events(self) -> None:
        materialized_observations = self._materialized_observation_candidates()
        materialized_downlinks = self._materialized_downlink_candidates()
        top_target_cell_id = None
        if materialized_observations:
            top_opportunity = max(
                materialized_observations,
                key=lambda item: self._observation_score_hint(item),
            )
            top_target_cell_id = top_opportunity.target_cell_id
        self._emit(
            event_type="opportunities_materialized",
            sim_tick=self.state.sim_tick,
            sim_time_utc=self.state.sim_time_utc,
            actor_type="system",
            actor_id="opportunity-builder:phase1",
            payload={
                "observation_opportunity_count": len(materialized_observations),
                "downlink_window_count": len(materialized_downlinks),
                "top_target_cell_id": top_target_cell_id,
                "materialized_observation_ids": [
                    item.opportunity_id for item in materialized_observations
                ],
                "materialized_downlink_ids": [item.window_id for item in materialized_downlinks],
            },
        )
        self._emit(
            event_type="action_mask_emitted",
            sim_tick=self.state.sim_tick,
            sim_time_utc=self.state.sim_time_utc,
            actor_type="system",
            actor_id=self.config.env_id,
            payload={
                "legal_action_count": len(self.action_mask.actions),
                "mask_id": self.action_mask.mask_id,
                "actions": [action.to_payload() for action in self.action_mask.actions],
            },
        )

    def _build_action_mask(self) -> OrbitalActionMask:
        actions: list[OrbitalAction] = [
            OrbitalAction(action_type="noop", ref="noop", score_hint=0.0)
        ]
        materialized_observations = self._materialized_observation_candidates()
        materialized_downlinks = self._materialized_downlink_candidates()
        ordered_groups = {
            "schedule_observation": [
                OrbitalAction(
                    action_type="schedule_observation",
                    ref=opportunity.opportunity_id,
                    score_hint=self._observation_score_hint(opportunity),
                    satellite_id=opportunity.satellite_id,
                    target_cell_id=opportunity.target_cell_id,
                )
                for opportunity in materialized_observations
                if self._has_buffer_capacity(opportunity)
            ],
            "schedule_downlink": [
                OrbitalAction(
                    action_type="schedule_downlink",
                    ref=window.window_id,
                    score_hint=self._downlink_score_hint(window),
                    satellite_id=window.satellite_id,
                )
                for window in materialized_downlinks
                if self._downlink_has_payload(window)
            ],
        }
        for action_type in self.config.action_order:
            if action_type == "noop":
                continue
            actions.extend(
                sorted(
                    ordered_groups.get(action_type, []),
                    key=lambda item: (-float(item.score_hint or 0.0), item.ref),
                )
            )
        return OrbitalActionMask(
            mask_id=stable_id("mask", self.state.episode_id, f"tick-{self.state.sim_tick:05d}"),
            actions=tuple(actions),
        )

    def _resolve_action(
        self,
        action: int | str | OrbitalAction | Mapping[str, Any] | None,
    ) -> OrbitalAction:
        if action is None:
            return self.action_mask.actions[0]
        if isinstance(action, int):
            try:
                return self.action_mask.actions[action]
            except IndexError as exc:
                raise ValueError(f"invalid action index: {action}") from exc
        if isinstance(action, OrbitalAction):
            candidate = action
        elif isinstance(action, str):
            if action == "noop":
                candidate = self.action_mask.actions[0]
            else:
                candidate = self.action_mask.actions[self.action_mask.index_of(action)]
        else:
            action_type = str(action["action_type"])
            ref = str(action.get("ref", action.get("action_ref", "noop")))
            candidate = OrbitalAction(action_type=action_type, ref=ref)
        for legal_action in self.action_mask.actions:
            if (
                legal_action.action_type == candidate.action_type
                and legal_action.ref == candidate.ref
            ):
                return legal_action
        raise ValueError(f"illegal action selected: {candidate.action_type}:{candidate.ref}")

    def _apply_selected_action(self, action: OrbitalAction) -> None:
        if action.action_type == "schedule_observation":
            self.state.pending_observation_ids.add(action.ref)
            return
        if action.action_type == "schedule_downlink":
            self.state.pending_downlink_ids.add(action.ref)

    def _process_interval(
        self,
        *,
        end_time: datetime,
    ) -> dict[str, float]:
        components: dict[str, float] = defaultdict(float)
        for opportunity in self._observation_opportunities:
            if opportunity.opportunity_id not in self.state.pending_observation_ids:
                continue
            if opportunity.end_time_utc > end_time:
                continue
            if opportunity.opportunity_id in self.state.completed_observation_ids:
                continue
            components = self._execute_observation(opportunity, components)
        for window in self._downlink_windows:
            if window.window_id not in self.state.pending_downlink_ids:
                continue
            if window.end_time_utc > end_time:
                continue
            if window.window_id in self.state.completed_downlink_ids:
                continue
            components = self._execute_downlink(window, components)
        self._refresh_incident_statuses(end_time, components)
        return dict(components)

    def _execute_observation(
        self,
        opportunity: ObservationOpportunity,
        components: dict[str, float],
    ) -> dict[str, float]:
        self.state.pending_observation_ids.discard(opportunity.opportunity_id)
        self.state.completed_observation_ids.add(opportunity.opportunity_id)
        cloud_fraction, realized_quality = self._realize_observation(opportunity)
        usable = (
            realized_quality >= self._quality_threshold and cloud_fraction <= self._cloud_threshold
        )
        queue_entry = QueueEntry(
            queue_id=stable_id("obsq", self.state.episode_id, opportunity.opportunity_id),
            satellite_id=opportunity.satellite_id,
            opportunity_id=opportunity.opportunity_id,
            observation_time_utc=opportunity.end_time_utc,
            target_cell_id=opportunity.target_cell_id,
            data_volume_mb=opportunity.estimated_data_volume_mb,
            realized_quality=realized_quality,
            cloud_fraction=cloud_fraction,
            usable=usable,
            incident_ids=tuple(opportunity.incident_ids or ()),
        )
        self.state.onboard_queue.append(queue_entry)
        if usable:
            components["observation_value"] += self._observation_value(
                opportunity,
                realized_quality,
            )
        else:
            components["cloud_penalty"] -= round(max(cloud_fraction, 0.05), 6)
        for incident_id in queue_entry.incident_ids:
            record = self.state.incident_records[incident_id]
            record.last_opportunity_id = opportunity.opportunity_id
            if usable and record.status == "unseen":
                record.status = "observed"
                record.observed_time_utc = opportunity.end_time_utc
        self._emit(
            event_type="observation_committed",
            sim_tick=self._sim_tick_for_time(opportunity.end_time_utc),
            sim_time_utc=opportunity.end_time_utc,
            actor_type="satellite",
            actor_id=opportunity.satellite_id,
            payload={
                "opportunity_id": opportunity.opportunity_id,
                "realized_quality": realized_quality,
                "usable": usable,
                "cloud_fraction": cloud_fraction,
                "target_cell_id": opportunity.target_cell_id,
                "incident_ids": list(opportunity.incident_ids or []),
            },
        )
        return components

    def _execute_downlink(
        self,
        window: DownlinkWindow,
        components: dict[str, float],
    ) -> dict[str, float]:
        self.state.pending_downlink_ids.discard(window.window_id)
        self.state.completed_downlink_ids.add(window.window_id)
        deliverable_entries = [
            entry for entry in self.state.onboard_queue if entry.satellite_id == window.satellite_id
        ]
        max_volume_mb = min(window.max_volume_mb, self._window_rate_capacity_mb(window))
        delivered_entries: list[QueueEntry] = []
        delivered_volume_mb = 0.0
        for entry in sorted(
            deliverable_entries,
            key=lambda item: (item.observation_time_utc, item.queue_id),
        ):
            if delivered_volume_mb + entry.data_volume_mb > max_volume_mb + 1e-9:
                continue
            delivered_entries.append(entry)
            delivered_volume_mb += entry.data_volume_mb
        if delivered_entries:
            delivered_ids = {entry.queue_id for entry in delivered_entries}
            self.state.onboard_queue = [
                entry for entry in self.state.onboard_queue if entry.queue_id not in delivered_ids
            ]
        self._emit(
            event_type="downlink_committed",
            sim_tick=self._sim_tick_for_time(window.end_time_utc),
            sim_time_utc=window.end_time_utc,
            actor_type="ground_station",
            actor_id=window.station_id,
            payload={
                "window_id": window.window_id,
                "delivered_volume_mb": round(delivered_volume_mb, 6),
                "delivered_queue_ids": [entry.queue_id for entry in delivered_entries],
            },
        )
        for entry in delivered_entries:
            if not entry.usable:
                continue
            for incident_id in entry.incident_ids:
                packet = self._build_incident_packet(
                    entry=entry,
                    incident_id=incident_id,
                    downlink_time=window.end_time_utc,
                )
                self.state.incident_packets.append(packet)
                record = self.state.incident_records[incident_id]
                record.status = "downlinked"
                record.downlinked_time_utc = window.end_time_utc
                record.last_packet_id = packet.packet_id
                latency_minutes = (
                    window.end_time_utc - entry.observation_time_utc
                ).total_seconds() / 60.0
                components["downlink_value"] += round(
                    self._observation_value(
                        self._observation_by_id[entry.opportunity_id],
                        entry.realized_quality,
                    )
                    * self.config.downlink_reward_scale,
                    6,
                )
                components["latency_penalty"] -= round(
                    latency_minutes * self.config.downlink_latency_penalty_per_minute,
                    6,
                )
                self._emit(
                    event_type="incident_packet_emitted",
                    sim_tick=self._sim_tick_for_time(window.end_time_utc),
                    sim_time_utc=window.end_time_utc,
                    actor_type="system",
                    actor_id=self.config.packetizer_actor_id,
                    payload={
                        "packet_id": packet.packet_id,
                        "incident_id": packet.incident_id,
                        "observation_opportunity_id": packet.observation_opportunity_id,
                        "target_cell_id": packet.target_cell_id,
                    },
                )
        return components

    def _refresh_incident_statuses(
        self,
        current_time: datetime,
        components: dict[str, float] | None = None,
    ) -> None:
        for record in self.state.incident_records.values():
            if record.status in {"downlinked", "missed"}:
                continue
            if record.observed_time_utc is not None:
                if self._has_future_downlink(record.incident_id, current_time):
                    continue
                record.status = "missed"
                record.missed_time_utc = current_time
                if components is not None:
                    components["missed_incident_penalty"] -= round(
                        record.urgency_score * self.config.missed_incident_penalty_scale,
                        6,
                    )
                continue
            if self._has_future_observation(record.incident_id, current_time):
                continue
            record.status = "missed"
            record.missed_time_utc = current_time
            if components is not None:
                components["missed_incident_penalty"] -= round(
                    record.urgency_score * self.config.missed_incident_penalty_scale,
                    6,
                )

    def _finalize_incident_statuses(self, *, end_time: datetime) -> None:
        self._refresh_incident_statuses(end_time)

    def _materialized_observation_candidates(self) -> list[ObservationOpportunity]:
        current_time = self.state.sim_time_utc
        candidates: list[ObservationOpportunity] = []
        for opportunity in self._observation_opportunities:
            if opportunity.opportunity_id in self.state.pending_observation_ids:
                continue
            if opportunity.opportunity_id in self.state.completed_observation_ids:
                continue
            if current_time >= opportunity.end_time_utc:
                continue
            materialization_time = opportunity.start_time_utc
            if opportunity.incident_ids:
                ignition_times = [
                    self._incident_by_id[incident_id].ignition_time_utc
                    for incident_id in opportunity.incident_ids
                    if incident_id in self._incident_by_id
                ]
                if ignition_times:
                    materialization_time = min(materialization_time, min(ignition_times))
            if current_time < materialization_time:
                continue
            candidates.append(opportunity)
        return candidates

    def _materialized_downlink_candidates(self) -> list[DownlinkWindow]:
        current_time = self.state.sim_time_utc
        candidates: list[DownlinkWindow] = []
        for window in self._downlink_windows:
            if window.window_id in self.state.pending_downlink_ids:
                continue
            if window.window_id in self.state.completed_downlink_ids:
                continue
            if current_time < window.start_time_utc or current_time >= window.end_time_utc:
                continue
            if not self._downlink_has_payload(window):
                continue
            candidates.append(window)
        return candidates

    def _downlink_has_payload(self, window: DownlinkWindow) -> bool:
        station = next(
            station
            for station in self.bundle.ground_stations
            if station.station_id == window.station_id
        )
        if station.capabilities.availability == "offline":
            return False
        return any(entry.satellite_id == window.satellite_id for entry in self.state.onboard_queue)

    def _has_buffer_capacity(self, opportunity: ObservationOpportunity) -> bool:
        satellite = next(
            satellite
            for satellite in self.bundle.satellites
            if satellite.satellite_id == opportunity.satellite_id
        )
        current_usage = sum(
            entry.data_volume_mb
            for entry in self.state.onboard_queue
            if entry.satellite_id == opportunity.satellite_id
        )
        pending_usage = sum(
            self._observation_by_id[opportunity_id].estimated_data_volume_mb
            for opportunity_id in self.state.pending_observation_ids
            if self._observation_by_id[opportunity_id].satellite_id == opportunity.satellite_id
        )
        return (
            current_usage + pending_usage + opportunity.estimated_data_volume_mb
            <= satellite.downlink.buffer_capacity_mb + 1e-9
        )

    def _buffer_pressure_penalty(self) -> float:
        penalty = 0.0
        for satellite in self.bundle.satellites:
            used = sum(
                entry.data_volume_mb
                for entry in self.state.onboard_queue
                if entry.satellite_id == satellite.satellite_id
            )
            if satellite.downlink.buffer_capacity_mb <= 0:
                continue
            penalty -= round(
                (used / satellite.downlink.buffer_capacity_mb)
                * self.config.buffer_pressure_penalty_scale,
                6,
            )
        return round(penalty, 6)

    def _observation_score_hint(self, opportunity: ObservationOpportunity) -> float:
        urgency = max(
            (
                self.state.incident_records[incident_id].urgency_score
                for incident_id in opportunity.incident_ids or ()
                if incident_id in self.state.incident_records
            ),
            default=0.25,
        )
        return round(
            urgency
            * opportunity.predicted_quality_mean
            * (1.0 - opportunity.predicted_cloud_obstruction_prob),
            6,
        )

    def _downlink_score_hint(self, window: DownlinkWindow) -> float:
        queued_value = sum(
            max(
                (
                    self.state.incident_records[incident_id].urgency_score
                    for incident_id in entry.incident_ids
                    if incident_id in self.state.incident_records
                ),
                default=0.0,
            )
            for entry in self.state.onboard_queue
            if entry.satellite_id == window.satellite_id and entry.usable
        )
        return round(queued_value, 6)

    def _observation_value(
        self,
        opportunity: ObservationOpportunity,
        realized_quality: float,
    ) -> float:
        static_value = next(
            target.static_value
            for target in self.bundle.target_cells
            if target.target_cell_id == opportunity.target_cell_id
        )
        urgency = max(
            (
                self.state.incident_records[incident_id].urgency_score
                for incident_id in opportunity.incident_ids or ()
                if incident_id in self.state.incident_records
            ),
            default=0.25,
        )
        return round(
            static_value * urgency * realized_quality * self.config.observation_reward_scale,
            6,
        )

    def _realize_observation(self, opportunity: ObservationOpportunity) -> tuple[float, float]:
        rng = seeded_rng(f"{self.state.episode_seed}:{opportunity.opportunity_id}")
        cloud_fraction = min(
            max(opportunity.predicted_cloud_obstruction_prob + rng.uniform(-0.08, 0.08), 0.0),
            1.0,
        )
        realized_quality = min(
            max(
                opportunity.predicted_quality_mean
                - 0.35 * cloud_fraction
                + rng.uniform(-0.05, 0.05),
                0.0,
            ),
            1.0,
        )
        return round(cloud_fraction, 6), round(realized_quality, 6)

    def _build_incident_packet(
        self,
        *,
        entry: QueueEntry,
        incident_id: str,
        downlink_time: datetime,
    ) -> IncidentPacket:
        self._packet_sequence_by_incident[incident_id] += 1
        incident = self._incident_by_id[incident_id]
        sequence = self._packet_sequence_by_incident[incident_id]
        packet_id = stable_id("pkt", incident_id, f"{sequence:04d}")
        return IncidentPacket(
            packet_id=packet_id,
            incident_id=incident_id,
            target_cell_id=incident.target_cell_id,
            observation_time_utc=entry.observation_time_utc,
            downlink_time_utc=downlink_time,
            confidence=incident.confidence,
            urgency_score=incident.urgency_score,
            recommended_action="dispatch_recon" if incident.urgency_score >= 0.7 else "monitor",
            observation_opportunity_id=entry.opportunity_id,
            downstream_value_estimate=self._observation_value(
                self._observation_by_id[entry.opportunity_id],
                entry.realized_quality,
            ),
            summary=f"Downlinked usable observation for {incident_id}",
        )

    def _build_info(
        self,
        *,
        events_since: int,
        reward: float,
        reward_components: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "episode_id": self.state.episode_id,
            "sim_tick": self.state.sim_tick,
            "sim_time_utc": self.state.sim_time_utc,
            "reward": reward,
            "reward_components": {key: round(value, 6) for key, value in reward_components.items()},
            "action_mask": self.action_mask.to_dict(),
            "events": [
                event.model_dump(mode="python", exclude_none=True)
                for event in self._replay_events[events_since:]
            ],
        }

    def _has_future_observation(self, incident_id: str, current_time: datetime) -> bool:
        for opportunity in self._observation_opportunities:
            if incident_id not in (opportunity.incident_ids or []):
                continue
            if opportunity.opportunity_id in self.state.completed_observation_ids:
                continue
            if opportunity.end_time_utc > current_time:
                return True
        return False

    def _has_future_downlink(self, incident_id: str, current_time: datetime) -> bool:
        queue_entries = [
            entry
            for entry in self.state.onboard_queue
            if incident_id in entry.incident_ids and entry.usable
        ]
        if not queue_entries:
            return False
        satellites = {entry.satellite_id for entry in queue_entries}
        return any(
            window.satellite_id in satellites
            and window.window_id not in self.state.completed_downlink_ids
            and window.end_time_utc > current_time
            for window in self._downlink_windows
        )

    def _window_rate_capacity_mb(self, window: DownlinkWindow) -> float:
        duration_seconds = (window.end_time_utc - window.start_time_utc).total_seconds()
        return round((window.expected_rate_mbps * duration_seconds) / 8.0, 6)

    def _sim_tick_for_time(self, value: datetime) -> int:
        delta_seconds = (value - self.bundle.time_window.start_time_utc).total_seconds()
        if delta_seconds <= 0:
            return 0
        return int(delta_seconds // self.config.decision_interval_seconds)

    @property
    def _quality_threshold(self) -> float:
        if self.config.quality_threshold is not None:
            return self.config.quality_threshold
        return self.bundle.config.opportunity_generation.quality_threshold

    @property
    def _cloud_threshold(self) -> float:
        if self.config.cloud_block_threshold is not None:
            return self.config.cloud_block_threshold
        return self.bundle.config.opportunity_generation.cloud_block_threshold

    def _emit(
        self,
        *,
        event_type: str,
        sim_tick: int,
        sim_time_utc: datetime,
        actor_type: str,
        actor_id: str,
        payload: dict[str, Any],
    ) -> ReplayEvent:
        if self._replay_emitter is None:
            raise RuntimeError("reset() must be called before emitting replay events")
        return self._replay_emitter.emit(
            event_type=event_type,
            sim_tick=sim_tick,
            sim_time_utc=sim_time_utc,
            actor_type=actor_type,
            actor_id=actor_id,
            payload=payload,
        )


class _ReplayListSink:
    def __init__(self, events: list[ReplayEvent]) -> None:
        self._events = events

    def handle_event(self, event: ReplayEvent) -> None:
        self._events.append(event)
