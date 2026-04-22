from __future__ import annotations

from orbital_shepherd_benchmark.planners import (
    PlannerEpisodeContext,
    build_planner,
    legal_actions_from_observation,
)
from orbital_shepherd_env_runtime import EnvRuntimeConfig, OrbitalEnv
from tests.benchmark.helpers import (
    build_lookahead_tradeoff_bundle,
    build_tiny_bundle,
    build_value_density_choice_bundle,
)


def test_built_in_planners_only_select_legal_actions() -> None:
    bundle = build_tiny_bundle()

    for planner_id in (
        "random_valid_action",
        "urgency_greedy",
        "value_density_greedy",
        "ortools_receding_horizon",
    ):
        planner = build_planner(planner_id)
        env = OrbitalEnv(
            bundle,
            config=EnvRuntimeConfig(planner_id=f"planner:{planner_id}"),
        )
        observation, _ = env.reset(seed=bundle.simulation_seed, planner_id=f"planner:{planner_id}")
        planner.start_episode(
            context=PlannerEpisodeContext(
                bundle=bundle,
                episode_id=str(observation["episode_id"]),
                episode_seed=int(observation["episode_seed"]),
                planner_seed=12345,
            ),
            initial_observation=observation,
        )
        terminated = False
        truncated = False
        while not (terminated or truncated):
            legal_actions = legal_actions_from_observation(observation)
            decision = planner.select_action(observation)
            assert any(
                action.action_type == decision.action.action_type
                and action.ref == decision.action.ref
                for action in legal_actions
            )
            observation, _, terminated, truncated, _ = env.step(
                decision.action,
                planner_trace=decision.to_trace_payload(),
            )


def test_urgency_greedy_prefers_useful_non_noop_actions_in_toy_episode() -> None:
    bundle = build_tiny_bundle()
    planner = build_planner("urgency_greedy")
    env = OrbitalEnv(bundle, config=EnvRuntimeConfig(planner_id="planner:urgency_greedy"))
    observation, _ = env.reset(seed=bundle.simulation_seed, planner_id="planner:urgency_greedy")
    planner.start_episode(
        context=PlannerEpisodeContext(
            bundle=bundle,
            episode_id=str(observation["episode_id"]),
            episode_seed=int(observation["episode_seed"]),
            planner_seed=12345,
        ),
        initial_observation=observation,
    )

    first_decision = planner.select_action(observation)
    assert first_decision.action.action_type == "schedule_observation"

    observation, _, _, _, _ = env.step(
        first_decision.action,
        planner_trace=first_decision.to_trace_payload(),
    )
    second_decision = planner.select_action(observation)
    assert second_decision.action.action_type == "schedule_downlink"


def test_value_density_greedy_prefers_obviously_better_observation() -> None:
    bundle = build_value_density_choice_bundle()
    planner = build_planner("value_density_greedy")
    env = OrbitalEnv(bundle, config=EnvRuntimeConfig(planner_id="planner:value_density_greedy"))
    observation, _ = env.reset(
        seed=bundle.simulation_seed,
        planner_id="planner:value_density_greedy",
    )
    planner.start_episode(
        context=PlannerEpisodeContext(
            bundle=bundle,
            episode_id=str(observation["episode_id"]),
            episode_seed=int(observation["episode_seed"]),
            planner_seed=12345,
        ),
        initial_observation=observation,
    )

    decision = planner.select_action(observation)

    assert decision.action.action_type == "schedule_observation"
    assert decision.action.ref == "opp:choice-high"
    assert decision.considered_candidates[0].diagnostics["expected_mission_value"] > (
        decision.considered_candidates[1].diagnostics["expected_mission_value"]
    )
    trace = decision.to_trace_payload()
    assert trace["selected_candidate"]["action"]["action_ref"] == "opp:choice-high"
    assert len(trace["considered_candidates"]) == 3
    assert "observation_formula" in trace


def test_ortools_receding_horizon_prefers_reachable_downlink_path() -> None:
    bundle = build_lookahead_tradeoff_bundle()
    planner = build_planner("ortools_receding_horizon")
    env = OrbitalEnv(
        bundle,
        config=EnvRuntimeConfig(planner_id="planner:ortools_receding_horizon"),
    )
    observation, _ = env.reset(
        seed=bundle.simulation_seed,
        planner_id="planner:ortools_receding_horizon",
    )
    planner.start_episode(
        context=PlannerEpisodeContext(
            bundle=bundle,
            episode_id=str(observation["episode_id"]),
            episode_seed=int(observation["episode_seed"]),
            planner_seed=12345,
        ),
        initial_observation=observation,
    )

    decision = planner.select_action(observation)

    assert decision.action.action_type == "schedule_observation"
    assert decision.action.ref == "opp:lookahead-b"
    trace = decision.to_trace_payload()
    assert trace["solver"]["status"] in {"OPTIMAL", "FEASIBLE"}
    assert trace["solver"]["objective_value"] >= 0.0
    assert any(
        item["action_ref"] == "dw:lookahead-b" for item in trace["solver"]["selected_actions"]
    )


def test_phase2_planners_emit_replay_trace_metadata() -> None:
    bundle = build_tiny_bundle()
    for planner_id in ("value_density_greedy", "ortools_receding_horizon"):
        planner = build_planner(planner_id)
        env = OrbitalEnv(bundle, config=EnvRuntimeConfig(planner_id=f"planner:{planner_id}"))
        observation, _ = env.reset(seed=bundle.simulation_seed, planner_id=f"planner:{planner_id}")
        planner.start_episode(
            context=PlannerEpisodeContext(
                bundle=bundle,
                episode_id=str(observation["episode_id"]),
                episode_seed=int(observation["episode_seed"]),
                planner_seed=12345,
            ),
            initial_observation=observation,
        )

        decision = planner.select_action(observation)
        env.step(decision.action, planner_trace=decision.to_trace_payload())

        action_event = next(
            event for event in env.replay_events if event.event_type == "action_selected"
        )
        planner_trace = action_event.payload["planner_trace"]
        assert (
            planner_trace["selected_candidate"]["action"]["action_type"]
            == "schedule_observation"
        )
        assert len(planner_trace["considered_candidates"]) >= 2
        if planner_id == "ortools_receding_horizon":
            assert planner_trace["solver"]["status"] in {"OPTIMAL", "FEASIBLE"}
