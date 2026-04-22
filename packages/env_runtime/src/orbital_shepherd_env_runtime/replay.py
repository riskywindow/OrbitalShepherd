from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from orbital_shepherd_contracts import ReplayEvent, normalize_replay_event
from orbital_shepherd_contracts.constants import LEGACY_REPLAY_EVENT_ALIASES
from orbital_shepherd_core import canonical_json_dumps, stable_id, stable_token

CANONICAL_ENV_EVENT_NAMES: dict[str, str] = {
    "scenario_loaded": "scenario_bundle_loaded",
    "episode_started": "episode_started",
    "opportunities_materialized": "candidate_set_materialized",
    "action_mask_emitted": "action_mask_emitted",
    "action_selected": "action_selected",
    "observation_committed": "observation_executed",
    "downlink_committed": "downlink_executed",
    "reward_assessed": "reward_assessed",
    "episode_ended": "episode_ended",
    "incident_packet_emitted": "incident_packet_emitted",
}

LEGACY_ENV_EVENT_ALIASES: dict[str, str] = {
    **LEGACY_REPLAY_EVENT_ALIASES,
    **{
        "scenario_loaded": "scenario_bundle_loaded",
        "opportunities_materialized": "candidate_set_materialized",
        "observation_committed": "observation_executed",
        "downlink_committed": "downlink_executed",
    },
}


class ReplaySink(Protocol):
    def handle_event(self, event: ReplayEvent) -> None: ...


class ReplayEventEmitter:
    def __init__(
        self,
        *,
        episode_id: str,
        default_actor_id: str,
        sinks: Sequence[ReplaySink] = (),
    ) -> None:
        self.episode_id = episode_id
        self.default_actor_id = default_actor_id
        self._sinks = list(sinks)
        self._event_index = 0
        self._episode_token = stable_token({"episode_id": episode_id}, length=10)

    @property
    def event_index(self) -> int:
        return self._event_index

    def emit(
        self,
        *,
        event_type: str,
        sim_tick: int,
        sim_time_utc: object,
        actor_type: str,
        actor_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ReplayEvent:
        canonical_event_type = canonical_event_name(event_type)
        event = normalize_replay_event(
            {
                "event_id": stable_id(
                    "evt",
                    self._episode_token,
                    f"{self._event_index:06d}",
                ),
                "episode_id": self.episode_id,
                "event_index": self._event_index,
                "sim_tick": sim_tick,
                "sim_time_utc": sim_time_utc,
                "event_type": canonical_event_type,
                "actor_type": actor_type,
                "actor_id": actor_id or self.default_actor_id,
                "payload": _canonical_payload(payload or {}),
            }
        )
        self._event_index += 1
        for sink in self._sinks:
            sink.handle_event(event)
        return event


class InMemoryReplaySink:
    def __init__(self) -> None:
        self.events: list[ReplayEvent] = []

    def handle_event(self, event: ReplayEvent) -> None:
        self.events.append(event)


class NdjsonReplayWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def handle_event(self, event: ReplayEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json_dumps(event.model_dump(mode="json", exclude_none=True)))
            handle.write("\n")


def canonical_event_name(event_type: str) -> str:
    if event_type in CANONICAL_ENV_EVENT_NAMES:
        return CANONICAL_ENV_EVENT_NAMES[event_type]
    return LEGACY_ENV_EVENT_ALIASES.get(event_type, event_type)


def replay_events_to_ndjson(events: Iterable[ReplayEvent]) -> str:
    return "\n".join(
        canonical_json_dumps(event.model_dump(mode="json", exclude_none=True)) for event in events
    )


def _canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: _canonical_value(value) for key, value in payload.items()}


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _canonical_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical_value(item) for item in value]
    return value
