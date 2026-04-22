from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


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


def phase1_contracts_dir() -> Path:
    return repo_root() / "packages" / "contracts"


def phase1_schemas_dir() -> Path:
    return phase1_contracts_dir() / "schemas"


def phase1_examples_dir() -> Path:
    return phase1_contracts_dir() / "examples"


def phase1_python_src_dir() -> Path:
    return phase1_contracts_dir() / "python" / "src"


def data_contract_fixture_roots() -> tuple[Path, ...]:
    root = repo_root() / "data"
    return (root / "fixtures", root / "scenarios", root / "replays")
