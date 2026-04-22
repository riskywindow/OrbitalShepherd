from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from statistics import mean, median
from typing import Any

from orbital_shepherd_contracts import ReplayEvent, ScenarioBundle


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    count: int
    mean: float | None
    median: float | None
    p90: float | None
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": _round_or_none(self.mean),
            "median": _round_or_none(self.median),
            "p90": _round_or_none(self.p90),
            "values": [round(value, 6) for value in self.values],
        }


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    time_to_first_useful_observation_seconds: DistributionSummary
    useful_observation_value_captured: float
    cloud_waste_rate: float
    downlink_latency_seconds: DistributionSummary
    missed_urgent_incident_rate: float
    opportunity_utilization_efficiency: float
    mission_utility: float
    observation_commit_count: int
    useful_packet_count: int
    urgent_incident_count: int
    missed_urgent_incident_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_to_first_useful_observation_seconds": (
                self.time_to_first_useful_observation_seconds.to_dict()
            ),
            "useful_observation_value_captured": round(self.useful_observation_value_captured, 6),
            "cloud_waste_rate": round(self.cloud_waste_rate, 6),
            "downlink_latency_seconds": self.downlink_latency_seconds.to_dict(),
            "missed_urgent_incident_rate": round(self.missed_urgent_incident_rate, 6),
            "opportunity_utilization_efficiency": round(
                self.opportunity_utilization_efficiency, 6
            ),
            "mission_utility": round(self.mission_utility, 6),
            "observation_commit_count": self.observation_commit_count,
            "useful_packet_count": self.useful_packet_count,
            "urgent_incident_count": self.urgent_incident_count,
            "missed_urgent_incident_count": self.missed_urgent_incident_count,
        }


def compute_episode_metrics(
    *,
    bundle: ScenarioBundle,
    replay_events: Sequence[ReplayEvent],
    urgent_incident_threshold: float = 0.7,
) -> EpisodeMetrics:
    observation_by_id = {
        opportunity.opportunity_id: opportunity for opportunity in bundle.observation_opportunities
    }
    incidents_by_id = {incident.incident_id: incident for incident in bundle.incidents}
    target_static_value_by_id = {
        target.target_cell_id: target.static_value for target in bundle.target_cells
    }

    observation_events = {
        str(event.payload["opportunity_id"]): event
        for event in replay_events
        if event.event_type == "observation_executed"
    }
    packet_events = [
        event for event in replay_events if event.event_type == "incident_packet_emitted"
    ]
    reward_events = [event for event in replay_events if event.event_type == "reward_assessed"]
    episode_end_event = next(
        (event for event in reversed(replay_events) if event.event_type == "episode_ended"),
        None,
    )

    first_useful_observation_seconds: list[float] = []
    downlink_latency_seconds: list[float] = []
    useful_observation_value_captured = 0.0
    first_packet_time_by_incident: dict[str, datetime] = {}
    for packet_event in packet_events:
        incident_id = str(packet_event.payload["incident_id"])
        opportunity_id = str(packet_event.payload["observation_opportunity_id"])
        incident = incidents_by_id[incident_id]
        opportunity = observation_by_id[opportunity_id]
        observation_event = observation_events[opportunity_id]
        downlink_time = _coerce_datetime(packet_event.sim_time_utc)
        observation_time = opportunity.end_time_utc
        downlink_latency_seconds.append(
            max(0.0, (downlink_time - observation_time).total_seconds())
        )
        realized_quality = float(observation_event.payload["realized_quality"])
        freshness_decay = _freshness_decay_hours(
            max(0.0, (observation_time - incident.ignition_time_utc).total_seconds() / 3600.0)
        )
        static_value = target_static_value_by_id.get(incident.target_cell_id, 0.0)
        useful_observation_value_captured += (
            static_value * incident.urgency_score * freshness_decay * realized_quality
        )
        first_packet_time_by_incident.setdefault(incident_id, downlink_time)

    for incident_id, first_packet_time in sorted(first_packet_time_by_incident.items()):
        ignition_time = incidents_by_id[incident_id].ignition_time_utc
        first_useful_observation_seconds.append(
            max(0.0, (first_packet_time - ignition_time).total_seconds())
        )

    urgent_incident_ids = {
        incident.incident_id
        for incident in bundle.incidents
        if incident.urgency_score >= urgent_incident_threshold
    }
    serviced_incident_ids = set(first_packet_time_by_incident)
    missed_urgent_incident_ids = urgent_incident_ids - serviced_incident_ids
    observation_commit_count = len(observation_events)
    usable_observation_count = sum(
        1 for event in observation_events.values() if bool(event.payload["usable"])
    )
    cloud_waste_rate = (
        0.0
        if observation_commit_count == 0
        else 1.0 - (usable_observation_count / observation_commit_count)
    )
    missed_urgent_incident_rate = (
        0.0
        if not urgent_incident_ids
        else len(missed_urgent_incident_ids) / len(urgent_incident_ids)
    )
    opportunity_utilization_efficiency = (
        0.0
        if observation_commit_count == 0
        else useful_observation_value_captured / observation_commit_count
    )
    if episode_end_event is not None:
        mission_utility = float(episode_end_event.payload["mission_utility"])
    else:
        mission_utility = sum(float(event.payload["total_reward"]) for event in reward_events)

    return EpisodeMetrics(
        time_to_first_useful_observation_seconds=summarize_distribution(
            first_useful_observation_seconds
        ),
        useful_observation_value_captured=round(useful_observation_value_captured, 6),
        cloud_waste_rate=round(cloud_waste_rate, 6),
        downlink_latency_seconds=summarize_distribution(downlink_latency_seconds),
        missed_urgent_incident_rate=round(missed_urgent_incident_rate, 6),
        opportunity_utilization_efficiency=round(opportunity_utilization_efficiency, 6),
        mission_utility=round(mission_utility, 6),
        observation_commit_count=observation_commit_count,
        useful_packet_count=len(packet_events),
        urgent_incident_count=len(urgent_incident_ids),
        missed_urgent_incident_count=len(missed_urgent_incident_ids),
    )


def summarize_distribution(values: Sequence[float]) -> DistributionSummary:
    sorted_values = tuple(sorted(float(value) for value in values))
    if not sorted_values:
        return DistributionSummary(count=0, mean=None, median=None, p90=None, values=())
    return DistributionSummary(
        count=len(sorted_values),
        mean=float(mean(sorted_values)),
        median=float(median(sorted_values)),
        p90=_percentile(sorted_values, 0.9),
        values=sorted_values,
    )


def aggregate_episode_metrics(episodes: Sequence[EpisodeMetrics]) -> dict[str, Any]:
    ttfuo_values = [
        value
        for episode in episodes
        for value in episode.time_to_first_useful_observation_seconds.values
    ]
    latency_values = [
        value for episode in episodes for value in episode.downlink_latency_seconds.values
    ]
    return {
        "episode_count": len(episodes),
        "time_to_first_useful_observation_seconds": summarize_distribution(ttfuo_values).to_dict(),
        "useful_observation_value_captured_mean": _mean_or_zero(
            [episode.useful_observation_value_captured for episode in episodes]
        ),
        "cloud_waste_rate_mean": _mean_or_zero(
            [episode.cloud_waste_rate for episode in episodes]
        ),
        "downlink_latency_seconds": summarize_distribution(latency_values).to_dict(),
        "missed_urgent_incident_rate_mean": _mean_or_zero(
            [episode.missed_urgent_incident_rate for episode in episodes]
        ),
        "opportunity_utilization_efficiency_mean": _mean_or_zero(
            [episode.opportunity_utilization_efficiency for episode in episodes]
        ),
        "mission_utility_mean": _mean_or_zero([episode.mission_utility for episode in episodes]),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return float(values[0])
    rank = max(1, ceil(len(values) * quantile)) - 1
    return float(values[min(rank, len(values) - 1)])


def _freshness_decay_hours(delay_hours: float) -> float:
    return 1.0 / (1.0 + (delay_hours / 6.0))


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(float(mean(values)), 6)
