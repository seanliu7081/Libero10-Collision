"""Validate the released dataset through its public location resolver.

These tests also work against an installed wheel from outside the checkout.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from libero10_collision import runtime


def test_all_100_released_episodes_are_complete_and_hash_verified():
    root = runtime.get_data_root()
    registry = json.loads((root / "benchmark.json").read_text())
    assert registry["task_count"] == 10
    assert registry["episode_count"] == 100
    assert [task["task"] for task in registry["tasks"]] == [f"t{i:02d}" for i in range(10)]
    episode_paths = []
    for task in registry["tasks"]:
        assert len(task["episode_manifests"]) == 10
        for index, relative_path in enumerate(task["episode_manifests"]):
            path = root / relative_path
            metadata = json.loads(path.read_text())
            assert metadata["task"] == task["task"]
            assert metadata["episode_id"] == index
            assert metadata["language"] == task["language"]
            assert metadata["task_goal"] == task["task_goal"]
            assert metadata["checks_passed"] is True
            assert metadata["settle_steps"] == 10
            assert set(metadata["artifact_sha256"]) == {
                "initial_state.npy", "obstacle_asset.json", "task.bddl",
            }
            for name, expected in metadata["artifact_sha256"].items():
                assert hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == expected
            state = np.load(path.parent / "initial_state.npy", allow_pickle=False)
            assert state.ndim == 1 and state.size > 0 and np.isfinite(state).all()
            assert not state.dtype.hasobject
            episode_paths.append(path)
    assert len(set(episode_paths)) == 100
    assert len(list((root / "tasks").glob("t*/episodes/ep*/*"))) == 400


def test_task_obstacles_and_manipulation_layouts_are_distinct():
    root = runtime.get_data_root()
    for task_index in range(10):
        metadata = [json.loads(p.read_text()) for p in sorted(
            (root / "tasks" / f"t{task_index:02d}" / "episodes").glob("ep*/episode.json")
        )]
        assert len({item["obstacle"]["asset"] for item in metadata}) == 10
        obstacle_positions = {
            tuple(item["settled_object_poses"]["collision_obstacle_1"][:2])
            for item in metadata
        }
        assert len(obstacle_positions) == 10
        manipulation_layouts = {
            tuple(tuple(item["settled_object_poses"][name])
                  for name in item["manipulation_targets"])
            for item in metadata
        }
        assert len(manipulation_layouts) == 10
