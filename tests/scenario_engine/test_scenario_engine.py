from __future__ import annotations

import json
from pathlib import Path

from orbital_shepherd_contracts import validate_canonical_bundle
from orbital_shepherd_core import canonical_json_dumps
from orbital_shepherd_scenario_engine import (
    ScenarioEngineConfig,
    build_scenario_pack,
    builtin_phase1_recipes,
    compile_manifest_to_bundle,
    compile_recipe_to_manifest,
    validate_scenario_pack,
)


def test_phase1_recipe_registry_contains_required_families() -> None:
    recipes = builtin_phase1_recipes(ScenarioEngineConfig().benchmark_id)

    assert len(recipes) == 10
    assert {recipe.family for recipe in recipes} == {
        "sparse_frontier",
        "burst_outbreak",
        "cloud_trap",
    }


def test_recipe_compilation_is_deterministic() -> None:
    recipe = builtin_phase1_recipes(ScenarioEngineConfig().benchmark_id)[0]

    first_manifest = compile_recipe_to_manifest(recipe)
    second_manifest = compile_recipe_to_manifest(recipe)
    first_bundle = compile_manifest_to_bundle(first_manifest)
    second_bundle = compile_manifest_to_bundle(second_manifest)

    assert canonical_json_dumps(first_manifest.model_dump(mode="json")) == canonical_json_dumps(
        second_manifest.model_dump(mode="json")
    )
    assert canonical_json_dumps(first_bundle.model_dump(mode="json")) == canonical_json_dumps(
        second_bundle.model_dump(mode="json")
    )
    assert first_bundle.bundle_fingerprint == second_bundle.bundle_fingerprint


def test_build_and_validate_pack(tmp_path: Path) -> None:
    config = ScenarioEngineConfig(scenario_dir=tmp_path)

    built = build_scenario_pack(engine_config=config, output_dir=tmp_path)
    validated = validate_scenario_pack(engine_config=config, input_dir=tmp_path)

    assert len(built) == 10
    assert [record.bundle_id for record in built] == [record.bundle_id for record in validated]

    for path in sorted(tmp_path.glob("*.json")):
        bundle = validate_canonical_bundle(json.loads(path.read_text(encoding="utf-8")))
        assert bundle.bundle_fingerprint
