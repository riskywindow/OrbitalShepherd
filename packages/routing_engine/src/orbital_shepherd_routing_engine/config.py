from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from orbital_shepherd_contracts import repo_root


def _default_dsn() -> str:
    return os.getenv(
        "ORBITAL_SHEPHERD_ROUTING_DSN",
        "postgresql://orbital:orbital@127.0.0.1:55432/orbital_shepherd_phase3",
    )


@dataclass(frozen=True, slots=True)
class RoutingEngineConfig:
    dsn: str = field(default_factory=_default_dsn)
    schema_name: str = "routing"
    compose_file: Path = field(
        default_factory=lambda: repo_root() / "infra" / "compose" / "phase3-routing.compose.yaml"
    )
    sql_migrations_dir: Path = field(
        default_factory=lambda: repo_root() / "packages" / "routing_engine" / "sql" / "migrations"
    )
    fixture_bundle_path: Path = field(
        default_factory=lambda: (
            repo_root()
            / "data"
            / "fixtures"
            / "region_builder"
            / "compiled"
            / "fixture_micro_region_bundle.json"
        )
    )
