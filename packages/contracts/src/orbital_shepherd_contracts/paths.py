from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def phase0_schemas_dir() -> Path:
    return repo_root() / "schemas"


def phase0_examples_dir() -> Path:
    return repo_root() / "examples"


def phase0_doc_paths() -> tuple[Path, ...]:
    root = repo_root()
    return (
        root / "01_mission_rfc.md",
        root / "02_architecture.md",
        root / "03_benchmark_spec.md",
        root / "04_data_contracts.md",
        root / "05_planner_api.openapi.yaml",
    )
