from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from orbital_shepherd_contracts import repo_root


@dataclass(frozen=True, slots=True)
class ScenarioEngineConfig:
    benchmark_id: str = "osbench-phase1-pack-v1"
    compiler_version: str = "orbital-shepherd-scenario-engine/1.0.0"
    scenario_dir: Path = field(default_factory=lambda: repo_root() / "data" / "scenarios")
    fixture_dir: Path = field(
        default_factory=lambda: repo_root() / "data" / "fixtures" / "scenario_engine"
    )
    orbit_asset_bundle_path: Path = field(
        default_factory=lambda: (
            repo_root()
            / "data"
            / "fixtures"
            / "ephemeris"
            / "compiled"
            / "eph--demo-phase1--raw-celestrak-demo-phase1-2026-04-01t00-00-00z.json"
        )
    )
    decision_interval_seconds: int = 60
    horizon_hours: int = 24
    timezone: str = "UTC"
    compiled_at_utc: datetime = datetime(2026, 4, 9, 12, 0, tzinfo=UTC)
