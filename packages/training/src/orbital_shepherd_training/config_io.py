from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orbital_shepherd_core import canonical_json_dumps
from orbital_shepherd_training.models import (
    BehaviorCloningConfig,
    CurriculumConfig,
    EvaluationConfig,
    ModelArchitectureConfig,
    PpoTrainingConfig,
    RewardShapingConfig,
    ScenarioSplitRegistry,
)

CONFIG_TYPE_TO_MODEL = {
    "reward_shaping": RewardShapingConfig,
    "curriculum": CurriculumConfig,
    "model_architecture": ModelArchitectureConfig,
    "behavior_cloning": BehaviorCloningConfig,
    "ppo_training": PpoTrainingConfig,
    "evaluation": EvaluationConfig,
    "scenario_split_registry": ScenarioSplitRegistry,
}


def load_yaml_document(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} is not valid JSON-compatible YAML and PyYAML is not installed"
            ) from exc
    document = yaml.safe_load(text)
    if document is None:
        raise ValueError(f"{path} is empty")
    return document


def write_yaml_document(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_dumps(document) + "\n", encoding="utf-8")


def load_phase2_config(path: Path) -> Any:
    document = load_yaml_document(path)
    config_type = str(document.get("config_type", ""))
    model = CONFIG_TYPE_TO_MODEL.get(config_type)
    if model is None:
        raise ValueError(f"{path} has unsupported config_type: {config_type or '<missing>'}")
    return model.model_validate(document)
