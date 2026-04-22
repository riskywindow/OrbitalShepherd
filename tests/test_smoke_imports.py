from __future__ import annotations

from orbital_shepherd_api import build_status_document
from orbital_shepherd_benchmark import BenchmarkConfig
from orbital_shepherd_contracts import (
    PHASE1_SCHEMA_VERSION,
    phase0_doc_paths,
    phase0_examples_dir,
    phase0_schemas_dir,
    phase1_examples_dir,
    phase1_schemas_dir,
)
from orbital_shepherd_core import stable_id
from orbital_shepherd_env_runtime import EnvRuntimeConfig
from orbital_shepherd_ephemeris import EPHEMERIS_SCHEMA_VERSION
from orbital_shepherd_escalation_bridge import PACKAGE_PURPOSE as ESCALATION_BRIDGE_PURPOSE
from orbital_shepherd_geo_artifacts import PACKAGE_PURPOSE as GEO_ARTIFACTS_PURPOSE
from orbital_shepherd_ground_env import PACKAGE_PURPOSE as GROUND_ENV_PURPOSE
from orbital_shepherd_opportunity_builder import OpportunityBuilderConfig
from orbital_shepherd_region_builder import PACKAGE_PURPOSE as REGION_BUILDER_PURPOSE
from orbital_shepherd_routing_engine import PACKAGE_PURPOSE as ROUTING_ENGINE_PURPOSE
from orbital_shepherd_scenario_engine import ScenarioEngineConfig
from orbital_shepherd_tactical_baselines import PACKAGE_PURPOSE as TACTICAL_BASELINES_PURPOSE
from orbital_shepherd_tactical_metrics import PACKAGE_PURPOSE as TACTICAL_METRICS_PURPOSE
from orbital_shepherd_tactical_scenario_engine import (
    PACKAGE_PURPOSE as TACTICAL_SCENARIO_ENGINE_PURPOSE,
)
from orbital_shepherd_training import PHASE2_BENCHMARK_ID, phase2_artifact_layout


def test_workspace_packages_import_cleanly() -> None:
    status = build_status_document()

    assert status["status"] == "ready-for-phase1-implementation"
    assert stable_id("bundle", "cloud-trap", "seed-42") == "bundle:cloud-trap:seed-42"
    assert ScenarioEngineConfig().decision_interval_seconds == 60
    assert OpportunityBuilderConfig().target_index == "H3"
    assert EnvRuntimeConfig().timezone == "UTC"
    assert BenchmarkConfig().benchmark_id == "osbench-phase1-pack-v1"
    assert PHASE2_BENCHMARK_ID == "osbench-phase2-foundation-v1"
    assert "RegionBundle" in REGION_BUILDER_PURPOSE
    assert "spatial" in GEO_ARTIFACTS_PURPOSE
    assert "route" in ROUTING_ENGINE_PURPOSE
    assert "tactical scenario" in TACTICAL_SCENARIO_ENGINE_PURPOSE
    assert "ground response" in GROUND_ENV_PURPOSE
    assert "heuristic" in TACTICAL_BASELINES_PURPOSE
    assert "metrics" in TACTICAL_METRICS_PURPOSE
    assert "IncidentPacket" in ESCALATION_BRIDGE_PURPOSE
    assert phase2_artifact_layout()["dataset_root"] == "data/training/datasets"
    assert PHASE1_SCHEMA_VERSION == "1.0.0"
    assert EPHEMERIS_SCHEMA_VERSION == "1.0.0"
    assert phase0_schemas_dir().is_dir()
    assert phase0_examples_dir().is_dir()
    assert phase1_schemas_dir().is_dir()
    assert phase1_examples_dir().is_dir()
    assert all(path.exists() for path in phase0_doc_paths())
