from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPO_ROOT / "apps/api/src",
    REPO_ROOT / "packages/contracts/python/src",
    REPO_ROOT / "packages/core/src",
    REPO_ROOT / "packages/ephemeris/src",
    REPO_ROOT / "packages/scenario_engine/src",
    REPO_ROOT / "packages/opportunity_builder/src",
    REPO_ROOT / "packages/region_builder/src",
    REPO_ROOT / "packages/geo_artifacts/src",
    REPO_ROOT / "packages/routing_engine/src",
    REPO_ROOT / "packages/tactical_scenario_engine/src",
    REPO_ROOT / "packages/ground_env/src",
    REPO_ROOT / "packages/tactical_baselines/src",
    REPO_ROOT / "packages/tactical_metrics/src",
    REPO_ROOT / "packages/escalation_bridge/src",
    REPO_ROOT / "packages/env_runtime/src",
    REPO_ROOT / "packages/benchmark/src",
    REPO_ROOT / "packages/policy_models/src",
    REPO_ROOT / "packages/training/src",
)

for source_root in SOURCE_ROOTS:
    source_path = str(source_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
