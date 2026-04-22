from __future__ import annotations

from pathlib import Path

from tests.benchmark.helpers import build_tiny_bundle
from tests.trained_policy_helpers import export_toy_policy_checkpoint

from orbital_shepherd_benchmark import PlannerEpisodeContext, build_planner
from orbital_shepherd_env_runtime import EnvRuntimeConfig, OrbitalEnv
from orbital_shepherd_policy_models import PolicyModelRegistry, default_policy_model_registry


def test_model_registry_discovers_and_loads_checkpoint(tmp_path: Path) -> None:
    manifest_path = export_toy_policy_checkpoint(tmp_path)
    registry = PolicyModelRegistry(
        manifest_roots=(tmp_path / "manifests",),
        checkpoint_roots=(tmp_path / "checkpoints",),
    )

    entries = registry.list_policies()

    assert len(entries) == 1
    entry = entries[0]
    assert entry.checkpoint_manifest_path == manifest_path.resolve()
    assert entry.model_id == "model:test-trained-policy:v1"
    assert entry.architecture.top_k == 4

    loaded = registry.load_policy(entry.model_key)
    assert loaded.backend == "raw_state_dict"
    assert loaded.inference_config()["device"] == "cpu"
    assert loaded.inference_config()["model_config"]["top_k"] == 4


def test_trained_policy_planner_selects_legal_action_and_emits_trace(tmp_path: Path) -> None:
    manifest_path = export_toy_policy_checkpoint(tmp_path, run_id="bc:test-planner-adapter:v1")
    registry = default_policy_model_registry()
    entry = registry.register_checkpoint(
        manifest_path=manifest_path,
        model_key="test-trained-policy-planner",
    )
    bundle = build_tiny_bundle()
    env = OrbitalEnv(
        bundle,
        config=EnvRuntimeConfig(planner_id=f"planner:{entry.planner_id}"),
    )
    observation, _ = env.reset(
        seed=bundle.simulation_seed,
        planner_id=f"planner:{entry.planner_id}",
    )
    planner = build_planner(entry.planner_id)
    planner.start_episode(
        context=PlannerEpisodeContext(
            bundle=bundle,
            episode_id=str(observation["episode_id"]),
            episode_seed=int(observation["episode_seed"]),
            planner_seed=7,
        ),
        initial_observation=observation,
    )

    decision = planner.select_action(observation)

    assert decision.action.action_type in {"noop", "schedule_observation", "schedule_downlink"}
    trace = decision.to_trace_payload()
    assert trace["planner_kind"] == "trained_policy"
    assert trace["selected_slot"] >= 0
    assert "value_estimate" in trace
    assert "action_entropy" in trace
    assert "top_slots" in trace
