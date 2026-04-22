from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from orbital_shepherd_contracts import repo_root


@dataclass(frozen=True, slots=True)
class TacticalScenarioEngineConfig:
    benchmark_id: str = "osbench-phase3-tactical-v1"
    compiler_version: str = "orbital-shepherd-tactical-scenario-engine/1.0.0"
    bridge_version: str = "orbital-shepherd-escalation-bridge/1.0.0"
    scenario_dir: Path = field(
        default_factory=lambda: repo_root() / "data" / "tactical_scenarios" / "phase3-pack-v1"
    )
    fixture_dir: Path = field(
        default_factory=lambda: repo_root() / "data" / "fixtures" / "tactical_scenario_engine"
    )
    default_region_bundle_path: Path = field(
        default_factory=lambda: (
            repo_root()
            / "data"
            / "fixtures"
            / "region_builder"
            / "compiled"
            / "fixture_micro_region_bundle.json"
        )
    )
    region_bundle_catalog: tuple[Path, ...] = field(
        default_factory=lambda: (
            repo_root()
            / "data"
            / "fixtures"
            / "region_builder"
            / "compiled"
            / "fixture_micro_region_bundle.json",
        )
    )
    decision_interval_seconds: int = 300
    compiled_at_utc: datetime = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
