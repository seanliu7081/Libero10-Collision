"""Regression checks on actual frozen scenes; run with pytest --simulation.

The contact probes modify in-memory simulation state only. They do not evaluate
complete policy trajectories or establish end-to-end task solvability.
"""

import numpy as np
import pytest

from libero10_collision import load_episode
from libero10_collision.collision import AddedObstacleCollisionMonitor, SceneCollisionMonitor
from libero10_collision.runtime import HOLD, obs_hash

pytestmark = pytest.mark.simulation


def put(env, name, position):
    joint = env.env.objects_dict[name].joints[0]
    pose = np.array(env.sim.data.get_joint_qpos(joint)).copy()
    pose[:3] = position
    env.sim.data.set_joint_qpos(joint, pose)


def pair_present(result, first, second):
    return any(set(row["owners"]) == {first, second} for row in result["colliding_pairs"])


def test_original_targets_count_as_collision_and_onsets_are_cumulative():
    with load_episode("t00", 0, render_size=128) as (env, _, _):
        first, second = "alphabet_soup_1", "tomato_sauce_1"
        put(env, first, [.28, -.30, .75])
        put(env, second, [.295, -.30, .75])
        env.sim.forward()
        previous = env.sim.step
        with AddedObstacleCollisionMonitor(env) as legacy:
            assert legacy.result(raw_goal_success=False)["added_obstacle_collision_free"]
        assert env.sim.step == previous
        with SceneCollisionMonitor(env) as monitor:
            result = monitor.result(raw_goal_success=False)
            assert result["metric_version"] == "scene-contact-v1"
            assert not result["scene_collision_free"]
            assert result["added_obstacle_collision_free"]
            assert pair_present(result, first, second)
            initial_events = result["distinct_onset_events"]
            for _ in range(3):
                monitor.sample_endpoint()
            repeated = monitor.result(raw_goal_success=False)
            assert repeated["distinct_onset_events"] == initial_events
            repeated["colliding_pairs"].clear()
            assert monitor.snapshot()["colliding_pairs"]
            put(env, second, [.45, -.30, .75])
            monitor.sample_endpoint()
            assert not monitor.result(raw_goal_success=False)["scene_collision_free"]
            put(env, second, [.295, -.30, .75])
            monitor.sample_endpoint()
            assert monitor.result(raw_goal_success=False)["distinct_onset_events"] == initial_events + 1
        assert env.sim.step == previous
        with pytest.raises(RuntimeError):
            monitor.attach()


def test_original_target_distractor_collision_is_counted():
    with load_episode("t00", 0, render_size=128) as (env, _, _):
        first, second = "alphabet_soup_1", "cream_cheese_1"
        put(env, first, [.28, -.30, .75])
        put(env, second, [.285, -.30, .75])
        env.sim.forward()
        with SceneCollisionMonitor(env) as monitor:
            result = monitor.result(raw_goal_success=False)
        assert not result["scene_collision_free"]
        assert pair_present(result, first, second)


def test_monitor_preserves_dynamics_observations_and_solver_warmstart():
    trajectory = []
    with load_episode("t01", 0, render_size=128) as (env, _, _):
        for _ in range(4):
            observation, _, _, _ = env.step(HOLD)
            trajectory.append((np.asarray(env.sim.get_state().flatten()).copy(),
                               obs_hash(observation), env.sim.data.qacc_warmstart.copy()))
    with load_episode("t01", 0, render_size=128) as (env, _, _):
        previous = env.sim.step
        with SceneCollisionMonitor(env) as monitor:
            assert monitor.result(raw_goal_success=False)["scene_collision_free"]
            for expected_state, expected_obs, expected_warmstart in trajectory:
                observation, _, _, _ = env.step(HOLD)
                before = np.asarray(env.sim.get_state().flatten()).copy()
                warmstart = env.sim.data.qacc_warmstart.copy()
                monitor.sample_endpoint()
                assert np.array_equal(np.asarray(env.sim.get_state().flatten()), before)
                assert np.array_equal(env.sim.data.qacc_warmstart, warmstart)
                assert np.array_equal(before, expected_state)
                assert obs_hash(observation) == expected_obs
                assert np.array_equal(warmstart, expected_warmstart)
            result = monitor.result(raw_goal_success=False)
            assert result["physics_steps"] == 100
            assert result["inspections_by_phase"] == {"initial": 1, "physics": 100, "endpoint": 4}
            assert result["scene_collision_free"]
        assert env.sim.step == previous
        with pytest.raises(ValueError, match="deliberate"):
            with SceneCollisionMonitor(env):
                raise ValueError("deliberate context exception")
        assert env.sim.step == previous
