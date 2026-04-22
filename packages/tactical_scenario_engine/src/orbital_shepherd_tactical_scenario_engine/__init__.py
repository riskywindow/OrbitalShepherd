"""Deterministic Phase 3 tactical scenario compiler and benchmark-pack builder."""

from orbital_shepherd_tactical_scenario_engine.catalog import (
    FAMILY_SPECS,
    TacticalFamilySpec,
    TacticalScenarioRecipe,
    builtin_phase3_recipes,
)
from orbital_shepherd_tactical_scenario_engine.cli import build_parser, main
from orbital_shepherd_tactical_scenario_engine.compiler import (
    TacticalBundleRecord,
    build_scenario_pack,
    compile_activation_to_manifest,
    compile_incident_packet_to_activation,
    compile_incident_packet_to_bundle,
    compile_manifest_to_bundle,
    compile_recipe_to_bundle,
    inspect_bundle,
    resolve_region_for_incident_packet,
    tactical_activation_fingerprint,
    tactical_bundle_fingerprint,
    tactical_bundle_id_from_manifest_id,
    validate_scenario_pack,
)
from orbital_shepherd_tactical_scenario_engine.config import TacticalScenarioEngineConfig

PACKAGE_NAME = "tactical_scenario_engine"
PACKAGE_PURPOSE = (
    "Build deterministic tactical scenario manifests and bundles for tactical scenario execution."
)

__all__ = [
    "FAMILY_SPECS",
    "PACKAGE_NAME",
    "PACKAGE_PURPOSE",
    "TacticalBundleRecord",
    "TacticalFamilySpec",
    "TacticalScenarioEngineConfig",
    "TacticalScenarioRecipe",
    "build_parser",
    "build_scenario_pack",
    "builtin_phase3_recipes",
    "compile_activation_to_manifest",
    "compile_incident_packet_to_activation",
    "compile_incident_packet_to_bundle",
    "compile_manifest_to_bundle",
    "compile_recipe_to_bundle",
    "inspect_bundle",
    "main",
    "resolve_region_for_incident_packet",
    "tactical_activation_fingerprint",
    "tactical_bundle_fingerprint",
    "tactical_bundle_id_from_manifest_id",
    "validate_scenario_pack",
]
