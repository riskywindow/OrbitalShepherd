from __future__ import annotations

from pathlib import Path

from orbital_shepherd_scenario_engine import builtin_phase2_recipes
from orbital_shepherd_training import (
    NoopWandbRun,
    TrainingPackManifest,
    build_phase2_training_pack,
    maybe_init_wandb,
    phase2_split_registry,
    validate_phase2_configs,
    validate_phase2_training_pack,
)
from orbital_shepherd_training.models import WandbConfig

PHASE2_BENCHMARK_ID = "osbench-phase2-foundation-v1"


def test_phase2_configs_validate_and_cross_reference() -> None:
    validated = validate_phase2_configs()

    assert validated["reward"].reward_id == "reward:phase2-auditable-v1"
    assert validated["curriculum"].curriculum_id == "curriculum:phase2-foundation-v1"
    assert validated["ppo"].curriculum_id == validated["curriculum"].curriculum_id
    assert validated["evaluation"].benchmark_id == PHASE2_BENCHMARK_ID


def test_phase2_split_registry_contract() -> None:
    registry = phase2_split_registry()

    assert len(registry.entries) == 20
    assert {entry.split for entry in registry.entries} == {"train", "val", "test", "ood"}
    assert {
        entry.scenario_family
        for entry in registry.entries
        if entry.split == "ood"
    } == {"station_outage", "constellation_degradation"}


def test_phase2_training_pack_builds_and_validates_from_small_recipe_subset(tmp_path: Path) -> None:
    recipes = builtin_phase2_recipes(PHASE2_BENCHMARK_ID)[:2]
    output_dir = tmp_path / "scenario-pack"
    manifest_path = tmp_path / "phase2-training-pack-manifest.json"
    split_path = tmp_path / "phase2-splits.yaml"

    manifest = build_phase2_training_pack(
        output_dir=output_dir,
        manifest_output=manifest_path,
        split_registry_output=split_path,
        recipes=recipes,
    )
    validated = validate_phase2_training_pack(
        input_dir=output_dir,
        manifest_path=manifest_path,
        recipes=recipes,
    )

    assert isinstance(manifest, TrainingPackManifest)
    assert validated.bundle_count == 2
    assert split_path.exists()
    assert manifest_path.exists()


def test_wandb_disabled_mode_returns_noop() -> None:
    run = maybe_init_wandb(
        WandbConfig(
            enabled=False,
            mode="disabled",
            project="orbital-shepherd-phase2",
            entity=None,
            group="phase2-local",
            tags=["test"],
        ),
        run_name="phase2-test",
        run_config={"phase": 2},
    )

    assert isinstance(run, NoopWandbRun)
