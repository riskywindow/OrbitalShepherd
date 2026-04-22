from __future__ import annotations

from pathlib import Path

from orbital_shepherd_contracts import repo_root

PHASE2_BENCHMARK_ID = "osbench-phase2-foundation-v1"


def training_root() -> Path:
    return repo_root() / "training"


def training_config_root() -> Path:
    return training_root() / "configs"


def phase2_split_registry_path() -> Path:
    return training_config_root() / "curriculum" / "phase2_splits.yaml"


def phase2_curriculum_config_path() -> Path:
    return training_config_root() / "curriculum" / "phase2_curriculum.yaml"


def data_training_root() -> Path:
    return repo_root() / "data" / "training"


def training_manifest_root() -> Path:
    return data_training_root() / "manifests"


def phase2_dataset_root() -> Path:
    return data_training_root() / "datasets" / PHASE2_BENCHMARK_ID


def phase2_scenario_pack_dir() -> Path:
    return data_training_root() / "scenario_packs" / PHASE2_BENCHMARK_ID


def phase2_training_pack_manifest_path() -> Path:
    return training_manifest_root() / "phase2-training-pack-manifest.json"
