from __future__ import annotations

import json
from pathlib import Path

from orbital_shepherd_core import canonical_json_dumps
from orbital_shepherd_scenario_engine import (
    ScenarioEngineConfig,
    build_scenario_pack,
    builtin_phase2_recipes,
    compile_recipe_to_manifest,
    validate_scenario_pack,
)

PHASE2_BENCHMARK_ID = "osbench-phase2-foundation-v1"


def test_phase2_recipe_registry_contains_required_families() -> None:
    recipes = builtin_phase2_recipes(PHASE2_BENCHMARK_ID)

    assert len(recipes) == 20
    assert {recipe.family for recipe in recipes} == {
        "sparse_frontier",
        "burst_outbreak",
        "cloud_trap",
        "downlink_crunch",
        "station_outage",
        "constellation_degradation",
    }


def test_phase2_recipe_overrides_are_reflected_in_manifests() -> None:
    recipes = {recipe.family: recipe for recipe in builtin_phase2_recipes(PHASE2_BENCHMARK_ID)}

    crunch_manifest = compile_recipe_to_manifest(recipes["downlink_crunch"])
    outage_manifest = compile_recipe_to_manifest(recipes["station_outage"])
    degradation_manifest = compile_recipe_to_manifest(recipes["constellation_degradation"])

    assert any(
        station.capabilities.availability == "degraded"
        for station in crunch_manifest.ground_stations
    )
    assert any(
        station.capabilities.availability == "offline"
        for station in outage_manifest.ground_stations
    )
    assert len(degradation_manifest.satellites) < 3
    assert any(
        satellite.constraints.availability == "degraded"
        for satellite in degradation_manifest.satellites
    )


def test_small_phase2_pack_build_and_validation_are_deterministic(tmp_path: Path) -> None:
    recipes = builtin_phase2_recipes(PHASE2_BENCHMARK_ID)[:2]
    config = ScenarioEngineConfig(benchmark_id=PHASE2_BENCHMARK_ID, scenario_dir=tmp_path)

    first_build = build_scenario_pack(engine_config=config, output_dir=tmp_path, recipes=recipes)
    second_build = build_scenario_pack(engine_config=config, output_dir=tmp_path, recipes=recipes)
    validated = validate_scenario_pack(engine_config=config, input_dir=tmp_path, recipes=recipes)

    first_ids = [record.bundle_id for record in first_build]
    second_ids = [record.bundle_id for record in second_build]
    validated_ids = [record.bundle_id for record in validated]

    assert first_ids == second_ids
    assert first_ids == validated_ids

    for record in first_build:
        first_bytes = record.path.read_text(encoding="utf-8")
        second_bytes = next(
            item.path.read_text(encoding="utf-8")
            for item in second_build
            if item.bundle_id == record.bundle_id
        )
        assert canonical_json_dumps(json.loads(first_bytes)) == canonical_json_dumps(
            json.loads(second_bytes)
        )
