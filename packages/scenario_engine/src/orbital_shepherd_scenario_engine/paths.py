from __future__ import annotations

from pathlib import Path

from orbital_shepherd_contracts import repo_root


def scenario_engine_fixture_dir() -> Path:
    return repo_root() / "data" / "fixtures" / "scenario_engine"


def scenario_bundle_dir() -> Path:
    return repo_root() / "data" / "scenarios"
