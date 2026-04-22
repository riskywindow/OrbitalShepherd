from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from orbital_shepherd_contracts.paths import repo_root


@dataclass(frozen=True, slots=True)
class ApiSettings:
    repo_root: Path = field(default_factory=repo_root)
    scenario_dir: Path = field(default_factory=lambda: repo_root() / "data" / "scenarios")
    training_scenario_pack_dir: Path = field(
        default_factory=lambda: repo_root() / "data" / "training" / "scenario_packs"
    )
    episode_dir: Path = field(default_factory=lambda: repo_root() / "data" / "api" / "episodes")
    baseline_run_dir: Path = field(
        default_factory=lambda: repo_root() / "data" / "api" / "baseline_runs"
    )
    model_run_dir: Path = field(
        default_factory=lambda: repo_root() / "data" / "api" / "model_runs"
    )
    demo_defaults_path: Path = field(
        default_factory=lambda: repo_root() / "data" / "demo" / "phase1-defaults.json"
    )
    phase2_demo_defaults_path: Path = field(
        default_factory=lambda: repo_root() / "data" / "demo" / "phase2-defaults.json"
    )
    orbit_asset_dir: Path = field(
        default_factory=lambda: repo_root() / "data" / "fixtures" / "ephemeris" / "compiled"
    )
    training_manifest_dir: Path = field(
        default_factory=lambda: repo_root() / "data" / "training" / "manifests"
    )
    training_checkpoint_dir: Path = field(
        default_factory=lambda: repo_root() / "data" / "training" / "checkpoints"
    )
    training_report_dir: Path = field(
        default_factory=lambda: repo_root() / "data" / "training" / "reports"
    )

    def ensure_directories(self) -> None:
        self.scenario_dir.mkdir(parents=True, exist_ok=True)
        self.training_scenario_pack_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_run_dir.mkdir(parents=True, exist_ok=True)
        self.model_run_dir.mkdir(parents=True, exist_ok=True)
        self.demo_defaults_path.parent.mkdir(parents=True, exist_ok=True)
        self.phase2_demo_defaults_path.parent.mkdir(parents=True, exist_ok=True)
        self.training_report_dir.mkdir(parents=True, exist_ok=True)
