from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    scenario_dir: Path = Path("data/scenarios")
    replay_dir: Path = Path("data/replays")
    report_dir: Path = Path("data/benchmarks")
    benchmark_id: str = "osbench-phase1-pack-v1"
    planner_ids: tuple[str, ...] = field(
        default=(
            "random_valid_action",
            "urgency_greedy",
            "value_density_greedy",
            "ortools_receding_horizon",
        )
    )
    urgent_incident_threshold: float = 0.7
    write_markdown: bool = True
    write_csv: bool = True
