from __future__ import annotations

from pathlib import Path

from orbital_shepherd_contracts import (
    IncidentPacket,
    RegionBundle,
    load_json,
    validate_region_bundle,
)
from orbital_shepherd_core import canonical_json_dumps
from orbital_shepherd_tactical_scenario_engine import (
    TacticalScenarioEngineConfig,
    TacticalScenarioRecipe,
    build_scenario_pack,
    builtin_phase3_recipes,
    compile_activation_to_manifest,
    compile_incident_packet_to_activation,
    compile_manifest_to_bundle,
    compile_recipe_to_bundle,
    resolve_region_for_incident_packet,
    validate_scenario_pack,
)

FIXTURE_REGION_BUNDLE_PATH = Path(
    "data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json"
)


def _config() -> TacticalScenarioEngineConfig:
    return TacticalScenarioEngineConfig(
        default_region_bundle_path=FIXTURE_REGION_BUNDLE_PATH,
        region_bundle_catalog=(FIXTURE_REGION_BUNDLE_PATH,),
    )


def _fixture_bundle() -> RegionBundle:
    return validate_region_bundle(load_json(FIXTURE_REGION_BUNDLE_PATH))


def _recipes() -> tuple[TacticalScenarioRecipe, ...]:
    config = _config()
    return builtin_phase3_recipes(
        config.benchmark_id,
        region_bundle_path=config.default_region_bundle_path,
    )


def test_identical_activation_and_seed_produce_identical_bundle_fingerprint() -> None:
    recipe = next(recipe for recipe in _recipes() if recipe.family_id == "foothill_access")
    config = _config()

    activation = compile_incident_packet_to_activation(
        recipe.packet,
        region_bundle=recipe.region_bundle_path,
        config=config,
        activation_time_utc=recipe.packet.downlink_time_utc,
    )
    manifest_one = compile_activation_to_manifest(
        activation,
        scenario_family=recipe.family_id,
        simulation_seed=recipe.seed,
        region_bundle=recipe.region_bundle_path,
        config=config,
        family_parameters={"recipe_id": recipe.recipe_id},
    )
    manifest_two = compile_activation_to_manifest(
        activation,
        scenario_family=recipe.family_id,
        simulation_seed=recipe.seed,
        region_bundle=recipe.region_bundle_path,
        config=config,
        family_parameters={"recipe_id": recipe.recipe_id},
    )
    bundle_one = compile_manifest_to_bundle(
        manifest_one,
        region_bundle=recipe.region_bundle_path,
        config=config,
    )
    bundle_two = compile_manifest_to_bundle(
        manifest_two,
        region_bundle=recipe.region_bundle_path,
        config=config,
    )

    assert bundle_one.bundle_fingerprint == bundle_two.bundle_fingerprint
    assert canonical_json_dumps(bundle_one.model_dump(mode="json", exclude_none=True)) == (
        canonical_json_dumps(bundle_two.model_dump(mode="json", exclude_none=True))
    )


def test_region_resolution_is_stable_across_catalog_order() -> None:
    bundle_document = _fixture_bundle().model_dump(mode="json", exclude_none=True)
    packet = IncidentPacket.model_validate(
        {
            "schema_version": "1.0.0",
            "packet_id": "pkt:test-region-resolution:0001",
            "incident_id": "inc:test-region-resolution",
            "target_cell_id": "tc:8928308280fffff",
            "observation_time_utc": "2026-08-14T03:34:00Z",
            "downlink_time_utc": "2026-08-14T03:46:00Z",
            "confidence": 0.88,
            "urgency_score": 0.91,
            "recommended_action": "dispatch_ground",
            "summary": "Deterministic resolution probe.",
        }
    )
    preferred = {
        **bundle_document,
        "region_bundle_id": "rb:aaa-resolution:v1",
        "region_id": "region:aaa-resolution",
        "region_manifest_id": "rm:aaa-resolution:v1",
    }
    secondary = {
        **bundle_document,
        "region_bundle_id": "rb:zzz-resolution:v1",
        "region_id": "region:zzz-resolution",
        "region_manifest_id": "rm:zzz-resolution:v1",
    }

    first = resolve_region_for_incident_packet(packet, catalog_sources=[secondary, preferred])
    second = resolve_region_for_incident_packet(packet, catalog_sources=[preferred, secondary])

    assert first.selection.region_bundle_id == "rb:aaa-resolution:v1"
    assert second.selection.region_bundle_id == "rb:aaa-resolution:v1"
    assert first.selection.region_bundle_id == second.selection.region_bundle_id


def test_family_specific_overlays_are_present_in_compiled_bundles() -> None:
    expectations = {
        "foothill_access": {"temporary_penalty"},
        "urban_interface": {"risk_zone", "temporary_penalty"},
        "closure_cascade": {"closure", "risk_zone"},
        "depot_saturation": {"temporary_penalty"},
        "smoke_corridor": {"risk_zone", "temporary_penalty"},
        "drone_scout_gap": {"risk_zone"},
    }

    for family_id, expected_overlay_kinds in expectations.items():
        recipe = next(recipe for recipe in _recipes() if recipe.family_id == family_id)
        bundle = compile_recipe_to_bundle(recipe, config=_config())

        overlay_kinds = {overlay.overlay_kind for overlay in bundle.overlay_events}
        assert overlay_kinds >= expected_overlay_kinds
        assert bundle.overlay_events


def test_build_and_validate_phase3_tactical_pack(tmp_path: Path) -> None:
    output_dir = tmp_path / "tactical-pack"
    config = TacticalScenarioEngineConfig(
        scenario_dir=output_dir,
        default_region_bundle_path=FIXTURE_REGION_BUNDLE_PATH,
        region_bundle_catalog=(FIXTURE_REGION_BUNDLE_PATH,),
    )

    built = build_scenario_pack(config=config, output_dir=output_dir)
    validated = validate_scenario_pack(config=config, input_dir=output_dir)

    assert len(built) == 24
    assert len(validated) == 24
    assert {record.scenario_family for record in built} == {
        "foothill_access",
        "urban_interface",
        "closure_cascade",
        "depot_saturation",
        "smoke_corridor",
        "drone_scout_gap",
    }
