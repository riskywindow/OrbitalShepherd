from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from orbital_shepherd_contracts import repo_root


@dataclass(frozen=True, slots=True)
class RegionBuilderConfig:
    compiler_version: str = "orbital-shepherd-region-builder/1.0.0"
    compiled_at_utc: datetime = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
    bundle_root: Path = field(default_factory=lambda: repo_root() / "data" / "regions" / "bundles")
    export_root: Path = field(default_factory=lambda: repo_root() / "data" / "regions" / "exports")
