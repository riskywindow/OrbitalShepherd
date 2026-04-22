from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from ray.tune.registry import ENV_CREATOR, _global_registry, register_env

from orbital_shepherd_contracts import ScenarioBundle
from orbital_shepherd_env_runtime import EnvRuntimeConfig
from orbital_shepherd_training.training_env import OrbitalTrainingEnv

RLLIB_PHASE2_ENV_ID = "orbital_shepherd_phase2_training_env_v1"
_REGISTERED_ENV_IDS: set[str] = set()


@dataclass(frozen=True, slots=True)
class LoadedScenarioBundle:
    bundle: ScenarioBundle
    path: Path
    bundle_id: str
    manifest_id: str
    scenario_family: str
    simulation_seed: int


@dataclass(frozen=True, slots=True)
class OrbitalRllibEnvConfig:
    bundle_paths: tuple[Path, ...]
    top_k: int
    base_seed: int
    planner_id: str
    worker_index: int
    vector_index: int
    num_envs_per_env_runner: int


class RllibOrbitalTrainingEnv(gym.Env[dict[str, Any], int]):
    metadata = dict(OrbitalTrainingEnv.metadata)

    def __init__(self, env_config: Any) -> None:
        self._config = _coerce_env_config(env_config)
        self._bundle_records = tuple(_load_bundle(path) for path in self._config.bundle_paths)
        if not self._bundle_records:
            raise ValueError("RllibOrbitalTrainingEnv requires at least one scenario bundle")
        probe_env = self._build_training_env(self._bundle_records[0].bundle)
        self.action_space = probe_env.action_space
        self.observation_space = probe_env.observation_space
        probe_env.close()

        self._active_env: OrbitalTrainingEnv | None = None
        self._active_record: LoadedScenarioBundle | None = None
        self._episode_counter = 0
        self._latest_info: dict[str, Any] = {}
        self._episode_projected_candidate_total = 0.0
        self._episode_candidate_total = 0.0
        self._episode_truncated_candidate_total = 0.0
        self._episode_step_count = 0

    @property
    def latest_info(self) -> dict[str, Any]:
        return dict(self._latest_info)

    @property
    def current_bundle_metadata(self) -> dict[str, Any]:
        if self._active_record is None:
            return {}
        return {
            "bundle_id": self._active_record.bundle_id,
            "manifest_id": self._active_record.manifest_id,
            "scenario_family": self._active_record.scenario_family,
            "simulation_seed": self._active_record.simulation_seed,
            "scenario_path": str(self._active_record.path),
        }

    def episode_metric_values(self) -> dict[str, float]:
        steps = max(self._episode_step_count, 1)
        mission_utility = 0.0
        if self._active_env is not None:
            mission_utility = float(self._active_env.runtime_env.state.mission_utility)
        return {
            "projected_candidate_count_mean": self._episode_projected_candidate_total / steps,
            "candidate_count_mean": self._episode_candidate_total / steps,
            "truncated_candidate_count_mean": self._episode_truncated_candidate_total / steps,
            "mission_utility_final": mission_utility,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._close_active_env()
        self._active_record = self._select_record(options=options)
        self._active_env = self._build_training_env(self._active_record.bundle)
        resolved_seed = seed if seed is not None else self._episode_seed()
        self._episode_counter += 1
        self._episode_projected_candidate_total = 0.0
        self._episode_candidate_total = 0.0
        self._episode_truncated_candidate_total = 0.0
        self._episode_step_count = 0
        observation, info = self._active_env.reset(seed=resolved_seed, options=options)
        info = self._augment_info(info)
        self._latest_info = dict(info)
        return observation, info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._active_env is None:
            raise RuntimeError("reset() must be called before step()")
        observation, reward, terminated, truncated, info = self._active_env.step(int(action))
        info = self._augment_info(info)
        self._latest_info = dict(info)
        self._episode_projected_candidate_total += float(info.get("projected_candidate_count", 0))
        self._episode_candidate_total += float(info.get("candidate_count", 0))
        self._episode_truncated_candidate_total += float(info.get("truncated_candidate_count", 0))
        self._episode_step_count += 1
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._close_active_env()

    def _select_record(self, *, options: dict[str, Any] | None) -> LoadedScenarioBundle:
        if options is not None and "bundle_id" in options:
            bundle_id = str(options["bundle_id"])
            for record in self._bundle_records:
                if record.bundle_id == bundle_id:
                    return record
            raise ValueError(f"unknown bundle_id requested: {bundle_id}")
        base_offset = (
            (max(self._config.worker_index, 0) * max(self._config.num_envs_per_env_runner, 1))
            + max(self._config.vector_index, 0)
        )
        index = (base_offset + self._episode_counter) % len(self._bundle_records)
        return self._bundle_records[index]

    def _episode_seed(self) -> int:
        base_offset = (
            (max(self._config.worker_index, 0) * max(self._config.num_envs_per_env_runner, 1))
            + max(self._config.vector_index, 0)
        )
        return self._config.base_seed + (base_offset * 1000) + self._episode_counter

    def _build_training_env(self, bundle: ScenarioBundle) -> OrbitalTrainingEnv:
        return OrbitalTrainingEnv(
            bundle,
            config=EnvRuntimeConfig(
                decision_interval_seconds=bundle.decision_interval_seconds,
                planner_id=self._config.planner_id,
                env_id="env:orbital-phase2-rllib",
                metrics_actor_id="metrics:phase2-rllib",
                reward_actor_id="reward:phase2-rllib",
                packetizer_actor_id="packetizer:phase2-rllib",
            ),
            top_k=self._config.top_k,
        )

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        augmented = dict(info)
        augmented.update(self.current_bundle_metadata)
        return augmented

    def _close_active_env(self) -> None:
        if self._active_env is not None:
            self._active_env.close()
            self._active_env = None


def register_orbital_rllib_env(env_id: str = RLLIB_PHASE2_ENV_ID) -> str:
    if env_id not in _REGISTERED_ENV_IDS or not _global_registry.contains(
        ENV_CREATOR, env_id
    ):
        register_env(env_id, lambda env_config: RllibOrbitalTrainingEnv(env_config))
        _REGISTERED_ENV_IDS.add(env_id)
    return env_id


def make_env_config(
    *,
    bundle_paths: list[str],
    top_k: int,
    base_seed: int,
    planner_id: str,
) -> dict[str, Any]:
    return {
        "bundle_paths": list(bundle_paths),
        "top_k": top_k,
        "base_seed": base_seed,
        "planner_id": planner_id,
    }


def _coerce_env_config(value: Any) -> OrbitalRllibEnvConfig:
    config_mapping = dict(value)
    bundle_paths = tuple(Path(item) for item in config_mapping.get("bundle_paths", ()))
    if not bundle_paths:
        raise ValueError("env_config.bundle_paths must not be empty")
    return OrbitalRllibEnvConfig(
        bundle_paths=bundle_paths,
        top_k=int(config_mapping.get("top_k", 64)),
        base_seed=int(config_mapping.get("base_seed", 0)),
        planner_id=str(config_mapping.get("planner_id", "planner:rllib-ppo")),
        worker_index=int(getattr(value, "worker_index", config_mapping.get("worker_index", 0))),
        vector_index=int(getattr(value, "vector_index", config_mapping.get("vector_index", 0))),
        num_envs_per_env_runner=int(
            getattr(
                value,
                "num_envs_per_env_runner",
                config_mapping.get("num_envs_per_env_runner", 1),
            )
        ),
    )


def _load_bundle(path: Path) -> LoadedScenarioBundle:
    document = json.loads(path.read_text(encoding="utf-8"))
    bundle = ScenarioBundle.model_validate(document)
    return LoadedScenarioBundle(
        bundle=bundle,
        path=path,
        bundle_id=bundle.bundle_id,
        manifest_id=bundle.manifest_id,
        scenario_family=bundle.scenario_family,
        simulation_seed=bundle.simulation_seed,
    )
