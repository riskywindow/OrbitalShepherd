from __future__ import annotations

import json
import math
import random
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from torch.distributions import Categorical

from orbital_shepherd_core import (
    canonical_json_dumps,
    format_utc_timestamp,
    sha256_fingerprint,
    stable_id,
)
from orbital_shepherd_env_runtime import EnvRuntimeConfig
from orbital_shepherd_training.config_io import load_phase2_config
from orbital_shepherd_training.models import (
    ModelArchitectureConfig,
    PolicyCheckpointManifest,
    TrainingPackEntry,
    PpoTrainingConfig,
    ScenarioSplitRegistry,
    TrainingPackManifest,
)
from orbital_shepherd_training.policy_checkpointing import (
    build_phase2_policy_model,
    export_phase2_policy_checkpoint,
    load_raw_policy_state_dict,
    resolve_policy_initialization_manifest,
    resolve_rllib_module_path,
    validate_policy_checkpoint_compatibility,
)
from orbital_shepherd_training.rllib_callbacks import OrbitalTrainingMetricsCallback
from orbital_shepherd_training.rllib_env import (
    RLLIB_PHASE2_ENV_ID,
    RllibOrbitalTrainingEnv,
    make_env_config,
    register_orbital_rllib_env,
)
from orbital_shepherd_training.rllib_module import OrbitalMaskedActionTorchRLModule
from orbital_shepherd_training.training_env import OrbitalTrainingEnv

try:  # pragma: no cover - exercised where numpy is installed.
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - covered in local tests.
    np = None


@dataclass(frozen=True, slots=True)
class RllibPpoTrainingSummary:
    run_id: str
    run_dir: Path
    metrics_path: Path
    run_manifest_path: Path
    checkpoint_manifest_paths: tuple[Path, ...]
    selected_bundle_ids: tuple[str, ...]
    evaluation_bundle_ids: tuple[str, ...]
    final_result: dict[str, Any]


def train_ppo_with_rllib(
    *,
    ppo_config: PpoTrainingConfig | Path,
    model_config: ModelArchitectureConfig | Path,
) -> RllibPpoTrainingSummary:
    ppo = _load_ppo_config(ppo_config)
    architecture = _load_model_config(model_config)
    warm_start_manifest = resolve_policy_initialization_manifest(
        initialization=ppo.initialization,
        manifest_root=Path(ppo.artifact_layout.manifest_root),
    )
    warm_start_module_path: str | None = None
    if warm_start_manifest is not None:
        validate_policy_checkpoint_compatibility(
            manifest=warm_start_manifest,
            architecture=architecture,
            top_k=ppo.top_k,
        )
        warm_start_module_path = str(resolve_rllib_module_path(warm_start_manifest))
    training_pack = _load_training_pack(Path(ppo.training_pack_path))
    split_registry = _load_split_registry(Path(ppo.split_registry_path))
    train_entries = _select_entries(
        training_pack=training_pack,
        split_registry=split_registry,
        split=ppo.train_split,
        limit=ppo.scenario_limit,
    )
    evaluation_entries = _select_entries(
        training_pack=training_pack,
        split_registry=split_registry,
        split=ppo.evaluation_split,
        limit=ppo.scenario_limit,
    )
    if not train_entries:
        raise ValueError(f"no training bundles available for split={ppo.train_split}")
    if not evaluation_entries:
        raise ValueError(f"no evaluation bundles available for split={ppo.evaluation_split}")

    started_at = datetime.now(UTC)
    run_label = _safe_filename(
        stable_id("trainrun", ppo.run_id, format_utc_timestamp(started_at).lower())
    )
    checkpoint_run_dir = (Path(ppo.artifact_layout.checkpoint_root) / run_label).resolve()
    report_run_dir = (Path(ppo.artifact_layout.report_root) / run_label).resolve()
    manifest_run_dir = (Path(ppo.artifact_layout.manifest_root) / run_label).resolve()
    checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
    report_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_run_dir / "metrics.jsonl"
    run_manifest_path = manifest_run_dir / "run_manifest.json"

    run_manifest_payload = {
        "run_id": ppo.run_id,
        "benchmark_id": ppo.benchmark_id,
        "trainer_backend": ppo.trainer_backend,
        "framework": ppo.framework,
        "ray_version": ray.__version__,
        "started_at_utc": format_utc_timestamp(started_at),
        "training_pack_id": training_pack.training_pack_id,
        "split_registry_id": split_registry.registry_id,
        "train_split": ppo.train_split,
        "evaluation_split": ppo.evaluation_split,
        "seed": ppo.seed,
        "top_k": ppo.top_k,
        "selected_bundle_ids": [entry.bundle_id for entry in train_entries],
        "evaluation_bundle_ids": [entry.bundle_id for entry in evaluation_entries],
        "initialization": ppo.initialization.model_dump(mode="json"),
        "warm_start_checkpoint_id": (
            warm_start_manifest.checkpoint_id if warm_start_manifest else None
        ),
        "warm_start_checkpoint_path": (
            warm_start_manifest.checkpoint_path if warm_start_manifest else None
        ),
        "ppo_config": ppo.model_dump(mode="json"),
        "model_config": architecture.model_dump(mode="json"),
    }
    _write_run_manifest(run_manifest_path, run_manifest_payload)

    env_id = register_orbital_rllib_env(RLLIB_PHASE2_ENV_ID)
    train_env_config = make_env_config(
        bundle_paths=[entry.scenario_path for entry in train_entries],
        top_k=ppo.top_k,
        base_seed=ppo.seed,
        planner_id="planner:phase2-rllib-ppo",
    )
    evaluation_env_config = make_env_config(
        bundle_paths=[entry.scenario_path for entry in evaluation_entries],
        top_k=ppo.top_k,
        base_seed=ppo.seed + 100_000,
        planner_id="planner:phase2-rllib-eval",
    )
    probe_env = RllibOrbitalTrainingEnv(train_env_config)
    rl_module_spec = RLModuleSpec(
        module_class=OrbitalMaskedActionTorchRLModule,
        observation_space=probe_env.observation_space,
        action_space=probe_env.action_space,
        model_config={
            "hidden_dim": architecture.hidden_dim,
            "encoder_layers": architecture.encoder_layers,
            "attention_heads": architecture.attention_heads,
            "dropout": architecture.dropout,
        },
        load_state_path=warm_start_module_path,
    )
    probe_env.close()

    config = _build_algorithm_config(
        ppo=ppo,
        env_id=env_id,
        train_env_config=train_env_config,
        evaluation_env_config=evaluation_env_config,
        rl_module_spec=rl_module_spec,
    )
    checkpoint_manifest_paths: list[Path] = []
    final_result: dict[str, Any] = {}

    if not _can_bind_local_socket():
        checkpoint_manifest_paths, final_result = _train_ppo_locally(
            ppo=ppo,
            architecture=architecture,
            training_pack=training_pack,
            train_entries=train_entries,
            evaluation_entries=evaluation_entries,
            warm_start_manifest=warm_start_manifest,
            checkpoint_run_dir=checkpoint_run_dir,
            report_run_dir=report_run_dir,
            manifest_run_dir=manifest_run_dir,
            metrics_path=metrics_path,
        )
        run_manifest_payload["execution_backend"] = "local_ppo_fallback"
        run_manifest_payload["ray_runtime_enabled"] = False
        _write_run_manifest(run_manifest_path, run_manifest_payload)
        return RllibPpoTrainingSummary(
            run_id=ppo.run_id,
            run_dir=checkpoint_run_dir,
            metrics_path=metrics_path,
            run_manifest_path=run_manifest_path,
            checkpoint_manifest_paths=tuple(checkpoint_manifest_paths),
            selected_bundle_ids=tuple(entry.bundle_id for entry in train_entries),
            evaluation_bundle_ids=tuple(entry.bundle_id for entry in evaluation_entries),
            final_result=final_result,
        )

    _patch_ray_resource_probes()
    ray._private.ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    ray_temp_dir = Path("/tmp/orbital-ray") / sha256_fingerprint(
        {"run_id": ppo.run_id, "started_at_utc": format_utc_timestamp(started_at)}
    )[:12]
    ray_temp_dir.mkdir(parents=True, exist_ok=True)
    ray.init(
        address="local",
        ignore_reinit_error=True,
        include_dashboard=False,
        local_mode=ppo.local_mode,
        logging_level=ppo.log_level,
        _skip_env_hook=True,
        _node_ip_address="127.0.0.1",
        _temp_dir=str(ray_temp_dir),
    )
    try:
        algorithm = config.build_algo()
        try:
            iterations = ceil(ppo.total_timesteps / _expected_samples_per_iteration(ppo))
            for iteration in range(1, iterations + 1):
                final_result = algorithm.train()
                _append_metrics_row(metrics_path, final_result)
                if iteration % ppo.checkpoint_frequency == 0 or (
                    ppo.checkpoint_at_end and iteration == iterations
                ):
                    checkpoint_manifest_paths.append(
                        _write_checkpoint_manifest(
                            algorithm=algorithm,
                            checkpoint_root=checkpoint_run_dir,
                            manifest_root=manifest_run_dir,
                            iteration=iteration,
                            ppo=ppo,
                            architecture=architecture,
                            training_pack=training_pack,
                            selected_bundle_ids=[entry.bundle_id for entry in train_entries],
                            warm_start_manifest=warm_start_manifest,
                            result=final_result,
                        )
                    )
        finally:
            algorithm.stop()
    finally:
        if ray.is_initialized():
            ray.shutdown()

    run_manifest_payload["execution_backend"] = "ray_rllib"
    run_manifest_payload["ray_runtime_enabled"] = True
    _write_run_manifest(run_manifest_path, run_manifest_payload)

    return RllibPpoTrainingSummary(
        run_id=ppo.run_id,
        run_dir=checkpoint_run_dir,
        metrics_path=metrics_path,
        run_manifest_path=run_manifest_path,
        checkpoint_manifest_paths=tuple(checkpoint_manifest_paths),
        selected_bundle_ids=tuple(entry.bundle_id for entry in train_entries),
        evaluation_bundle_ids=tuple(entry.bundle_id for entry in evaluation_entries),
        final_result=final_result,
    )


def _build_algorithm_config(
    *,
    ppo: PpoTrainingConfig,
    env_id: str,
    train_env_config: dict[str, Any],
    evaluation_env_config: dict[str, Any],
    rl_module_spec: RLModuleSpec,
) -> PPOConfig:
    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=ppo.enable_rl_module_api,
            enable_env_runner_and_connector_v2=ppo.enable_rl_module_api,
        )
        .framework(ppo.framework)
        .environment(
            env=env_id,
            env_config=train_env_config,
            action_mask_key="action_mask",
        )
        .env_runners(
            num_env_runners=ppo.num_env_runners,
            create_local_env_runner=True,
            num_envs_per_env_runner=ppo.num_envs_per_env_runner,
            num_cpus_per_env_runner=int(ceil(ppo.num_cpus_per_env_runner)),
            rollout_fragment_length=ppo.rollout_steps,
            batch_mode="truncate_episodes",
        )
        .resources(num_gpus=ppo.num_gpus)
        .training(
            train_batch_size=_expected_samples_per_iteration(ppo),
            minibatch_size=ppo.minibatch_size,
            num_epochs=ppo.update_epochs,
            gamma=ppo.gamma,
            lambda_=ppo.gae_lambda,
            clip_param=ppo.clip_range,
            entropy_coeff=ppo.entropy_coef,
            vf_loss_coeff=ppo.value_loss_coef,
            grad_clip=ppo.max_grad_norm,
            kl_target=ppo.target_kl,
            lr=ppo.learning_rate,
        )
        .evaluation(
            evaluation_interval=ppo.evaluation_interval,
            evaluation_duration=ppo.evaluation_duration,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=0,
            evaluation_parallel_to_training=False,
            evaluation_config={
                "env_config": evaluation_env_config,
                "explore": False,
            },
        )
        .callbacks(callbacks_class=OrbitalTrainingMetricsCallback)
        .debugging(log_level=ppo.log_level, seed=ppo.seed)
        .rl_module(rl_module_spec=rl_module_spec)
    )


def _train_ppo_locally(
    *,
    ppo: PpoTrainingConfig,
    architecture: ModelArchitectureConfig,
    training_pack: TrainingPackManifest,
    train_entries: list[Any],
    evaluation_entries: list[Any],
    warm_start_manifest: PolicyCheckpointManifest | None,
    checkpoint_run_dir: Path,
    report_run_dir: Path,
    manifest_run_dir: Path,
    metrics_path: Path,
) -> tuple[list[Path], dict[str, Any]]:
    del report_run_dir
    _set_local_random_seeds(ppo.seed)
    device = torch.device("cpu")
    model = build_phase2_policy_model(architecture=architecture, top_k=ppo.top_k).to(device)
    if warm_start_manifest is not None:
        model.load_state_dict(_load_initial_policy_state_dict(warm_start_manifest))
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo.learning_rate)

    expected_samples = _expected_samples_per_iteration(ppo)
    iterations = ceil(ppo.total_timesteps / expected_samples)
    global_step = 0
    checkpoint_manifest_paths: list[Path] = []
    final_result: dict[str, Any] = {}

    train_entries_typed = [entry for entry in train_entries if isinstance(entry, TrainingPackEntry)]
    evaluation_entries_typed = [
        entry for entry in evaluation_entries if isinstance(entry, TrainingPackEntry)
    ]
    if not train_entries_typed or not evaluation_entries_typed:
        raise ValueError("local PPO requires typed training-pack entries")

    for iteration in range(1, iterations + 1):
        batch = _collect_local_rollouts(
            model=model,
            entries=train_entries_typed,
            ppo=ppo,
            target_steps=expected_samples,
            starting_seed=ppo.seed + (iteration * 10_000),
            device=device,
        )
        update_metrics = _run_local_ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            ppo=ppo,
            device=device,
        )
        evaluation_metrics = _evaluate_local_policy(
            model=model,
            entries=evaluation_entries_typed,
            ppo=ppo,
            device=device,
        )
        global_step += batch["timesteps_total"]
        final_result = {
            "iteration": iteration,
            "timesteps_total": global_step,
            "timesteps_this_iter": batch["timesteps_total"],
            "episodes_this_iter": batch["episodes_total"],
            "train/episode_reward_mean": batch["episode_reward_mean"],
            "train/episode_reward_max": batch["episode_reward_max"],
            "train/episode_len_mean": batch["episode_len_mean"],
            "train/mission_utility_mean": batch["mission_utility_mean"],
            "train/advantage_mean": batch["advantage_mean"],
            "train/return_mean": batch["return_mean"],
            "train/action_entropy_mean": batch["action_entropy_mean"],
            "train/padding_action_rate": batch["padding_action_rate"],
            "learn/policy_loss": update_metrics["policy_loss"],
            "learn/value_loss": update_metrics["value_loss"],
            "learn/entropy": update_metrics["entropy"],
            "learn/approx_kl": update_metrics["approx_kl"],
            "learn/clip_fraction": update_metrics["clip_fraction"],
            "evaluation/episode_reward_mean": evaluation_metrics["episode_reward_mean"],
            "evaluation/episode_len_mean": evaluation_metrics["episode_len_mean"],
            "evaluation/mission_utility_mean": evaluation_metrics["mission_utility_mean"],
            "runtime_backend": "local_ppo_fallback",
        }
        _append_metrics_row(metrics_path, final_result)
        if iteration % ppo.checkpoint_frequency == 0 or (
            ppo.checkpoint_at_end and iteration == iterations
        ):
            checkpoint_manifest_paths.append(
                _write_local_checkpoint_manifest(
                    checkpoint_root=checkpoint_run_dir,
                    manifest_root=manifest_run_dir,
                    iteration=iteration,
                    ppo=ppo,
                    architecture=architecture,
                    training_pack=training_pack,
                    selected_bundle_ids=[entry.bundle_id for entry in train_entries_typed],
                    warm_start_manifest=warm_start_manifest,
                    result=final_result,
                    policy_state_dict=model.state_dict(),
                )
            )
    return checkpoint_manifest_paths, final_result


def _collect_local_rollouts(
    *,
    model: Any,
    entries: list[TrainingPackEntry],
    ppo: PpoTrainingConfig,
    target_steps: int,
    starting_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    global_features: list[list[float]] = []
    candidate_features: list[list[list[float]]] = []
    action_masks: list[list[int]] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    returns: list[float] = []
    advantages: list[float] = []
    rewards_flat: list[float] = []
    entropies: list[float] = []
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_utilities: list[float] = []
    padding_actions = 0
    total_steps = 0
    episode_index = 0

    while total_steps < target_steps:
        entry = entries[episode_index % len(entries)]
        env = _build_local_training_env(entry, top_k=ppo.top_k, planner_id="planner:phase2-local-ppo")
        observation, _ = env.reset(
            seed=starting_seed + episode_index,
            options={"planner_id": "planner:phase2-local-ppo"},
        )
        trajectory_obs: list[dict[str, Any]] = []
        trajectory_actions: list[int] = []
        trajectory_log_probs: list[float] = []
        trajectory_values: list[float] = []
        trajectory_rewards: list[float] = []
        trajectory_entropies: list[float] = []
        terminated = False
        truncated = False
        next_observation = observation

        while not (terminated or truncated) and total_steps < target_steps:
            action, log_prob, value, entropy = _sample_local_action(
                model=model,
                observation=observation,
                device=device,
            )
            next_observation, reward, terminated, truncated, _ = env.step(action)
            trajectory_obs.append(observation)
            trajectory_actions.append(action)
            trajectory_log_probs.append(log_prob)
            trajectory_values.append(value)
            trajectory_rewards.append(float(reward))
            trajectory_entropies.append(entropy)
            total_steps += 1
            observation = next_observation

        bootstrap_value = 0.0
        if not (terminated or truncated):
            bootstrap_value = _value_for_observation(model=model, observation=next_observation, device=device)
        episode_advantages, episode_returns = _compute_gae(
            rewards=trajectory_rewards,
            values=trajectory_values,
            bootstrap_value=bootstrap_value,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
        )
        global_features.extend(
            [list(item["global_features"]) for item in trajectory_obs]
        )
        candidate_features.extend(
            [list(item["candidate_features"]) for item in trajectory_obs]
        )
        action_masks.extend([list(item["action_mask"]) for item in trajectory_obs])
        actions.extend(trajectory_actions)
        old_log_probs.extend(trajectory_log_probs)
        returns.extend(episode_returns)
        advantages.extend(episode_advantages)
        rewards_flat.extend(trajectory_rewards)
        entropies.extend(trajectory_entropies)
        episode_rewards.append(sum(trajectory_rewards))
        episode_lengths.append(len(trajectory_rewards))
        episode_utilities.append(float(env.runtime_env.state.mission_utility))
        env.close()
        episode_index += 1

    return {
        "global_features": torch.tensor(
            np.asarray(global_features, dtype=np.float32) if np is not None else global_features,
            dtype=torch.float32,
        ),
        "candidate_features": torch.tensor(
            (
                np.asarray(candidate_features, dtype=np.float32)
                if np is not None
                else candidate_features
            ),
            dtype=torch.float32,
        ),
        "action_masks": torch.tensor(
            np.asarray(action_masks, dtype=np.float32) if np is not None else action_masks,
            dtype=torch.float32,
        ),
        "actions": torch.tensor(actions, dtype=torch.int64),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
        "returns": torch.tensor(returns, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "timesteps_total": total_steps,
        "episodes_total": len(episode_rewards),
        "episode_reward_mean": _safe_mean(episode_rewards),
        "episode_reward_max": max(episode_rewards) if episode_rewards else 0.0,
        "episode_len_mean": _safe_mean(episode_lengths),
        "mission_utility_mean": _safe_mean(episode_utilities),
        "return_mean": _safe_mean(returns),
        "advantage_mean": _safe_mean(advantages),
        "action_entropy_mean": _safe_mean(entropies),
        "padding_action_rate": (padding_actions / max(total_steps, 1)),
    }


def _run_local_ppo_update(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    ppo: PpoTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    observations = {
        "global_features": batch["global_features"].to(device),
        "candidate_features": batch["candidate_features"].to(device),
        "action_mask": batch["action_masks"].to(device),
    }
    actions = batch["actions"].to(device)
    old_log_probs = batch["old_log_probs"].to(device)
    returns = batch["returns"].to(device)
    advantages = batch["advantages"].to(device)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

    sample_count = int(actions.shape[0])
    generator = torch.Generator().manual_seed(ppo.seed + sample_count)
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    stop_early = False

    for _ in range(ppo.update_epochs):
        permutation = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, ppo.minibatch_size):
            indices = permutation[start : start + ppo.minibatch_size]
            minibatch = {
                key: value.index_select(0, indices)
                for key, value in observations.items()
            }
            outputs = model.forward_observation(minibatch)
            distribution = Categorical(logits=outputs.masked_logits)
            new_log_probs = distribution.log_prob(actions.index_select(0, indices))
            entropy = distribution.entropy().mean()
            value_predictions = outputs.values
            ratio = torch.exp(new_log_probs - old_log_probs.index_select(0, indices))
            minibatch_advantages = advantages.index_select(0, indices)
            unclipped = ratio * minibatch_advantages
            clipped = torch.clamp(ratio, 1.0 - ppo.clip_range, 1.0 + ppo.clip_range) * minibatch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.mean(
                torch.square(value_predictions - returns.index_select(0, indices))
            )
            loss = policy_loss + (ppo.value_loss_coef * value_loss) - (ppo.entropy_coef * entropy)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo.max_grad_norm)
            optimizer.step()

            approx_kl = torch.mean(old_log_probs.index_select(0, indices) - new_log_probs).abs()
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > ppo.clip_range).float())
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            approx_kls.append(float(approx_kl.item()))
            clip_fractions.append(float(clip_fraction.item()))
            if float(approx_kl.item()) > (ppo.target_kl * 1.5):
                stop_early = True
                break
        if stop_early:
            break

    return {
        "policy_loss": _safe_mean(policy_losses),
        "value_loss": _safe_mean(value_losses),
        "entropy": _safe_mean(entropies),
        "approx_kl": _safe_mean(approx_kls),
        "clip_fraction": _safe_mean(clip_fractions),
    }


def _evaluate_local_policy(
    *,
    model: Any,
    entries: list[TrainingPackEntry],
    ppo: PpoTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    rewards: list[float] = []
    lengths: list[int] = []
    utilities: list[float] = []
    episode_count = max(ppo.evaluation_duration, 1)
    for episode_index in range(episode_count):
        entry = entries[episode_index % len(entries)]
        env = _build_local_training_env(
            entry,
            top_k=ppo.top_k,
            planner_id="planner:phase2-local-eval",
        )
        observation, _ = env.reset(
            seed=ppo.seed + 100_000 + episode_index,
            options={"planner_id": "planner:phase2-local-eval"},
        )
        reward_total = 0.0
        step_count = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = _greedy_local_action(model=model, observation=observation, device=device)
            observation, reward, terminated, truncated, _ = env.step(action)
            reward_total += float(reward)
            step_count += 1
        rewards.append(reward_total)
        lengths.append(step_count)
        utilities.append(float(env.runtime_env.state.mission_utility))
        env.close()
    return {
        "episode_reward_mean": _safe_mean(rewards),
        "episode_len_mean": _safe_mean(lengths),
        "mission_utility_mean": _safe_mean(utilities),
    }


def _sample_local_action(
    *,
    model: Any,
    observation: dict[str, Any],
    device: torch.device,
) -> tuple[int, float, float, float]:
    observation_batch = _observation_batch(observation, device=device)
    with torch.no_grad():
        outputs = model.forward_observation(observation_batch)
        distribution = Categorical(logits=outputs.masked_logits.squeeze(0))
        action = int(distribution.sample().item())
        log_prob = float(distribution.log_prob(torch.tensor(action, device=device)).item())
        value = float(outputs.values.squeeze(0).item())
        entropy = float(distribution.entropy().item())
    return action, log_prob, value, entropy


def _greedy_local_action(*, model: Any, observation: dict[str, Any], device: torch.device) -> int:
    with torch.no_grad():
        outputs = model.forward_observation(_observation_batch(observation, device=device))
    return int(torch.argmax(outputs.masked_logits.squeeze(0)).item())


def _value_for_observation(*, model: Any, observation: dict[str, Any], device: torch.device) -> float:
    with torch.no_grad():
        outputs = model.forward_observation(_observation_batch(observation, device=device))
    return float(outputs.values.squeeze(0).item())


def _observation_batch(observation: dict[str, Any], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "global_features": torch.tensor(
            observation["global_features"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        "candidate_features": torch.tensor(
            observation["candidate_features"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        "action_mask": torch.tensor(
            observation["action_mask"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
    }


def _compute_gae(
    *,
    rewards: list[float],
    values: list[float],
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    next_advantage = 0.0
    next_value = bootstrap_value
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + (gamma * next_value) - values[index]
        next_advantage = delta + (gamma * gae_lambda * next_advantage)
        advantages[index] = next_advantage
        returns[index] = advantages[index] + values[index]
        next_value = values[index]
    return advantages, returns


def _build_local_training_env(
    entry: TrainingPackEntry,
    *,
    top_k: int,
    planner_id: str,
) -> OrbitalTrainingEnv:
    bundle = _load_bundle_from_path(Path(entry.scenario_path))
    return OrbitalTrainingEnv(
        bundle,
        config=EnvRuntimeConfig(
            decision_interval_seconds=bundle.decision_interval_seconds,
            planner_id=planner_id,
            env_id="env:orbital-phase2-local-ppo",
            metrics_actor_id="metrics:phase2-local-ppo",
            reward_actor_id="reward:phase2-local-ppo",
            packetizer_actor_id="packetizer:phase2-local-ppo",
        ),
        top_k=top_k,
    )


def _load_bundle_from_path(path: Path) -> Any:
    from orbital_shepherd_contracts import ScenarioBundle

    return ScenarioBundle.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _write_local_checkpoint_manifest(
    *,
    checkpoint_root: Path,
    manifest_root: Path,
    iteration: int,
    ppo: PpoTrainingConfig,
    architecture: ModelArchitectureConfig,
    training_pack: TrainingPackManifest,
    selected_bundle_ids: list[str],
    warm_start_manifest: PolicyCheckpointManifest | None,
    result: dict[str, Any],
    policy_state_dict: dict[str, Any],
) -> Path:
    exported = export_phase2_policy_checkpoint(
        checkpoint_root=checkpoint_root,
        manifest_root=manifest_root,
        checkpoint_name=f"checkpoint_{iteration:06d}",
        algorithm="ppo",
        run_id=ppo.run_id,
        benchmark_id=ppo.benchmark_id,
        reward_id=ppo.reward_id,
        architecture=architecture,
        top_k=ppo.top_k,
        source_dataset_ids=(
            list(warm_start_manifest.source_dataset_ids) if warm_start_manifest else []
        ),
        source_training_pack_id=training_pack.training_pack_id,
        source_bundle_ids=selected_bundle_ids,
        global_step=int(result.get("timesteps_total", iteration)),
        metrics=_extract_checkpoint_metrics(result),
        metadata={
            "ray_version": ray.__version__,
            "enable_rl_module_api": ppo.enable_rl_module_api,
            "top_k": ppo.top_k,
            "seed": ppo.seed,
            "rollout_steps": ppo.rollout_steps,
            "minibatch_size": ppo.minibatch_size,
            "update_epochs": ppo.update_epochs,
            "run_manifest_path": str(manifest_root / "run_manifest.json"),
            "initialization_mode": ppo.initialization.mode,
            "warm_start_checkpoint_id": (
                warm_start_manifest.checkpoint_id if warm_start_manifest else None
            ),
            "warm_start_checkpoint_path": (
                warm_start_manifest.checkpoint_path if warm_start_manifest else None
            ),
            "execution_backend": "local_ppo_fallback",
        },
        policy_state_dict=policy_state_dict,
        created_at=datetime.now(UTC),
        trainer_backend=ppo.trainer_backend,
        framework=ppo.framework,
    )
    return exported.manifest_path


def _load_initial_policy_state_dict(
    manifest: PolicyCheckpointManifest,
) -> dict[str, torch.Tensor]:
    try:
        return load_raw_policy_state_dict(manifest)
    except (FileNotFoundError, TypeError, ValueError):
        module_path = resolve_rllib_module_path(manifest)
        module = OrbitalMaskedActionTorchRLModule.from_checkpoint(str(module_path))
        return {
            key: value.detach().cpu()
            for key, value in module.policy_model.state_dict().items()
        }


def _set_local_random_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)


def _safe_mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _can_bind_local_socket() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False


def _write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(canonical_json_dumps(payload) + "\n", encoding="utf-8")


def _append_metrics_row(path: Path, result: dict[str, Any]) -> None:
    row = _flatten_primitives(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_dumps(row))
        handle.write("\n")


def _write_checkpoint_manifest(
    *,
    algorithm: Any,
    checkpoint_root: Path,
    manifest_root: Path,
    iteration: int,
    ppo: PpoTrainingConfig,
    architecture: ModelArchitectureConfig,
    training_pack: TrainingPackManifest,
    selected_bundle_ids: list[str],
    warm_start_manifest: PolicyCheckpointManifest | None,
    result: dict[str, Any],
) -> Path:
    checkpoint_dir = checkpoint_root / f"checkpoint_{iteration:06d}"
    checkpoint_path = Path(algorithm.save_to_path(checkpoint_dir))
    metrics = _extract_checkpoint_metrics(result)
    created_at = datetime.now(UTC)
    manifest_payload = {
        "checkpoint_id": stable_id("ckpt", ppo.run_id, f"iter-{iteration:06d}"),
        "algorithm": "ppo",
        "created_at_utc": format_utc_timestamp(created_at),
        "benchmark_id": ppo.benchmark_id,
        "run_id": ppo.run_id,
        "model_id": architecture.model_id,
        "reward_id": ppo.reward_id,
        "trainer_backend": ppo.trainer_backend,
        "framework": ppo.framework,
        "source_dataset_ids": (
            list(warm_start_manifest.source_dataset_ids) if warm_start_manifest else []
        ),
        "source_training_pack_id": training_pack.training_pack_id,
        "source_bundle_ids": selected_bundle_ids,
        "checkpoint_path": str(checkpoint_path),
        "global_step": int(metrics.get("num_env_steps_sampled_lifetime", iteration)),
        "metrics": metrics,
        "metadata": {
            "ray_version": ray.__version__,
            "enable_rl_module_api": ppo.enable_rl_module_api,
            "top_k": ppo.top_k,
            "seed": ppo.seed,
            "rollout_steps": ppo.rollout_steps,
            "minibatch_size": ppo.minibatch_size,
            "update_epochs": ppo.update_epochs,
            "run_manifest_path": str(manifest_root / "run_manifest.json"),
            "initialization_mode": ppo.initialization.mode,
            "warm_start_checkpoint_id": (
                warm_start_manifest.checkpoint_id if warm_start_manifest else None
            ),
            "warm_start_checkpoint_path": (
                warm_start_manifest.checkpoint_path if warm_start_manifest else None
            ),
        },
    }
    manifest_payload["artifact_fingerprint"] = "sha256:" + sha256_fingerprint(
        {
            key: value
            for key, value in manifest_payload.items()
            if key != "artifact_fingerprint"
        }
    )
    manifest = PolicyCheckpointManifest.model_validate(manifest_payload)
    checkpoint_manifest_path = checkpoint_path / "manifest.json"
    checkpoint_manifest_path.write_text(
        canonical_json_dumps(manifest.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    manifest_index_path = manifest_root / f"{checkpoint_path.name}.json"
    manifest_index_path.write_text(
        canonical_json_dumps(manifest.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    return manifest_index_path


def _extract_checkpoint_metrics(result: dict[str, Any]) -> dict[str, float]:
    flattened = _flatten_primitives(result)
    metrics: dict[str, float] = {}
    for key, value in flattened.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            metrics[key] = float(value)
    return metrics


def _flatten_primitives(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, str | int | float | bool | None]:
    if isinstance(value, dict):
        flattened: dict[str, str | int | float | bool | None] = {}
        for key, item in value.items():
            if key in {"config", "hist_stats", "sampler_results", "episode_media"}:
                continue
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_primitives(item, prefix=next_prefix))
        return flattened
    if isinstance(value, list):
        return {}
    if hasattr(value, "item") and callable(value.item):
        return _flatten_primitives(value.item(), prefix=prefix)
    if isinstance(value, str | int | float | bool) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return {prefix: None}
        return {prefix: value}
    return {}


def _expected_samples_per_iteration(ppo: PpoTrainingConfig) -> int:
    active_env_runners = max(ppo.num_env_runners, 1)
    return int(ppo.rollout_steps * active_env_runners * ppo.num_envs_per_env_runner)


def _select_entries(
    *,
    training_pack: TrainingPackManifest,
    split_registry: ScenarioSplitRegistry,
    split: str,
    limit: int | None,
) -> list[Any]:
    split_by_bundle = {entry.bundle_id: entry.split for entry in split_registry.entries}
    selected = [
        entry
        for entry in sorted(training_pack.entries, key=lambda item: item.bundle_id)
        if split_by_bundle.get(entry.bundle_id) == split
    ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _load_training_pack(path: Path) -> TrainingPackManifest:
    return TrainingPackManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _load_split_registry(path: Path) -> ScenarioSplitRegistry:
    if path.suffix in {".yaml", ".yml"}:
        return ScenarioSplitRegistry.model_validate(load_phase2_config(path))
    return ScenarioSplitRegistry.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _load_ppo_config(source: PpoTrainingConfig | Path) -> PpoTrainingConfig:
    if isinstance(source, PpoTrainingConfig):
        return source
    model = load_phase2_config(source)
    if not isinstance(model, PpoTrainingConfig):
        raise TypeError(f"{source} is not a PPO training config")
    return model


def _load_model_config(source: ModelArchitectureConfig | Path) -> ModelArchitectureConfig:
    if isinstance(source, ModelArchitectureConfig):
        return source
    model = load_phase2_config(source)
    if not isinstance(model, ModelArchitectureConfig):
        raise TypeError(f"{source} is not a model architecture config")
    return model


def _safe_filename(value: str) -> str:
    return value.replace(":", "--").replace("/", "-")


def _patch_ray_resource_probes() -> None:
    common_utils = ray._common.utils
    private_utils = ray._private.utils

    if getattr(private_utils, "_orbital_shepherd_resource_probe_patch", False):
        return

    original_get_system_memory = common_utils.get_system_memory
    original_get_used_memory = private_utils.get_used_memory

    def _safe_get_system_memory(*args: Any, **kwargs: Any) -> int:
        try:
            return int(original_get_system_memory(*args, **kwargs))
        except PermissionError:
            return 8 * 1024**3

    def _safe_get_used_memory() -> int:
        try:
            return int(original_get_used_memory())
        except PermissionError:
            return 2 * 1024**3

    def _safe_estimate_available_memory() -> int:
        return max(_safe_get_system_memory() - _safe_get_used_memory(), 512 * 1024**2)

    common_utils.get_system_memory = _safe_get_system_memory
    private_utils.get_used_memory = _safe_get_used_memory
    private_utils.estimate_available_memory = _safe_estimate_available_memory
    private_utils._orbital_shepherd_resource_probe_patch = True
