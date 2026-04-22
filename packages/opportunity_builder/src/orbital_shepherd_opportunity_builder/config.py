from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpportunityBuilderConfig:
    decision_interval_seconds: int = 60
    coordinate_system: str = "WGS84"
    target_index: str = "H3"
