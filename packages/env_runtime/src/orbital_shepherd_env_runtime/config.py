from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EnvRuntimeConfig:
    replay_dir: Path = Path("data/replays")
    decision_interval_seconds: int = 60
    planner_id: str = "planner:manual"
    env_id: str = "env:orbital-phase1"
    metrics_actor_id: str = "metrics:phase1"
    reward_actor_id: str = "reward:phase1"
    packetizer_actor_id: str = "packetizer:phase1"
    planner_metadata: dict[str, Any] = field(default_factory=dict)
    timezone: str = "UTC"
    quality_threshold: float | None = None
    cloud_block_threshold: float | None = None
    observation_reward_scale: float = 1.0
    downlink_reward_scale: float = 2.0
    missed_incident_penalty_scale: float = 1.5
    downlink_latency_penalty_per_minute: float = 0.01
    buffer_pressure_penalty_scale: float = 0.05
    max_steps: int | None = None
    include_zero_reward_events: bool = True
    action_order: tuple[str, ...] = field(
        default=("noop", "schedule_observation", "schedule_downlink")
    )
