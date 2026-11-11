"""Controlled contacts against the real compiled scene geometry.

Unlike physical integration probes, these tests inject a single contact record
to exercise boundary conditions for the scene-contact-v1 exceptions.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from libero10_collision import load_episode
from libero10_collision.collision import SceneCollisionMonitor

pytestmark = pytest.mark.simulation


def assert_classification(env, first, second, *, allowed, reason=None,
                          distance=-.0001, margin=0., normal=(0, 0, 1), point=None):
    model, data = env.sim.model, env.sim.data
    ids = [model.geom_name2id(g) if isinstance(g, str) else g for g in (first, second)]
    contact = SimpleNamespace(geom1=ids[0], geom2=ids[1], dist=distance,
                              includemargin=margin, frame=np.r_[normal, [0.] * 6],
                              pos=np.array(point if point is not None else [0., 0., 0.]))

    class DataProxy:
        ncon = 1

        def __getattr__(self, key):
            return getattr(data, key)

    proxy_data = DataProxy()
    proxy_data.contact = [contact]
    monitor = SceneCollisionMonitor(env)
    monitor.sim = SimpleNamespace(model=model, data=proxy_data, step=lambda: None)
    with monitor:
        result = monitor.result(raw_goal_success=False)
    assert result["scene_collision_free"] is allowed
    if reason is not None:
        assert result["allowed_contacts_by_reason"].get(reason, {}).get("initial", 0) == 1
    if not allowed:
        assert result["initial_contact_samples"] == 1
        assert result["distinct_onset_events"] == 1


def top_point(env, geom):
    gid = env.sim.model.geom_name2id(geom) if isinstance(geom, str) else geom
    rotation = env.sim.data.geom_xmat[gid].reshape(3, 3)
    axis = int(np.argmax(abs(rotation[2])))
    sign = 1 if rotation[2, axis] > 0 else -1
    return env.sim.data.geom_xpos[gid] + rotation[:, axis] * env.sim.model.geom_size[gid, axis] * sign


def test_table_finger_and_contact_margin_boundaries():
    with load_episode("t00", 0, render_size=128) as (env, _, _):
        monitor = SceneCollisionMonitor(env)
        finger = next(iter(monitor._finger_geoms))
        arm = next(g for g, item in monitor.geometry.items()
                   if item["owner"] == "robot" and g not in monitor._finger_geoms)
        target = "alphabet_soup_1_g1"
        distractor = "cream_cheese_1_g1"
        obstacle = next(g for g, item in monitor.geometry.items()
                        if item["owner"] == "collision_obstacle_1")
        table = next(iter(monitor._table_geoms))
        top = top_point(env, table)
        assert_classification(env, table, target, allowed=True, reason="tabletop_support", point=top)
        assert_classification(env, target, table, allowed=True, reason="tabletop_support",
                              point=top, normal=(0, 0, -1))
        assert_classification(env, table, target, allowed=False, point=top, normal=(1, 0, 0))
        assert_classification(env, table, target, allowed=False, point=top - [0, 0, .1])
        assert_classification(env, table, target, allowed=False, point=top, distance=-.0011)
        assert_classification(env, finger, target, allowed=True, reason="target_finger_grasp")
        assert_classification(env, finger, target, allowed=False, distance=-.0011)
        assert_classification(env, arm, target, allowed=False)
        assert_classification(env, finger, distractor, allowed=False)
        assert_classification(env, finger, obstacle, allowed=False)
        assert_classification(env, target, distractor, allowed=True, distance=.001, margin=0.)
        assert_classification(env, target, distractor, allowed=False, distance=.001, margin=.001)
        for tolerance in [-.001, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                SceneCollisionMonitor(env, intended_contact_tolerance_m=tolerance)


def test_only_correct_plate_upper_surface_is_allowed():
    with load_episode("t04", 0, render_size=128) as (env, _, _):
        target, left, right = "porcelain_mug_1_g1", "plate_1_g1", "plate_2_g1"
        top = top_point(env, left)
        assert_classification(env, left, target, allowed=True, reason="goal_destination_support", point=top)
        assert_classification(env, right, target, allowed=False, point=top_point(env, right))
        assert_classification(env, left, target, allowed=False, point=top, normal=(1, 0, 0))
        assert_classification(env, left, target, allowed=False, point=top + [.5, 0, 0])


def test_caddy_support_respects_destination_compartment():
    with load_episode("t05", 0, render_size=128) as (env, _, _):
        target, floor = "black_book_1_g1", "desk_caddy_1_g1"
        site = env.sim.model.site_name2id("desk_caddy_1_back_contain_region")
        point = env.sim.data.site_xpos[site].copy()
        point[2] = top_point(env, floor)[2]
        assert_classification(env, floor, target, allowed=True, reason="goal_destination_support", point=point)
        assert_classification(env, floor, target, allowed=False, point=point + [.09, 0, 0])


@pytest.mark.parametrize("task,handle", [
    ("t02", "flat_stove_1_g5"),
    ("t03", "white_cabinet_1_g40"),
    ("t09", "microwave_1_g14"),
])
def test_fixture_exception_only_allows_fingers_on_declared_handles(task, handle):
    with load_episode(task, 0, render_size=128) as (env, _, _):
        monitor = SceneCollisionMonitor(env)
        finger = next(iter(monitor._finger_geoms))
        arm = next(g for g, item in monitor.geometry.items()
                   if item["owner"] == "robot" and g not in monitor._finger_geoms)
        handle_owner = monitor.geometry[env.sim.model.geom_name2id(handle)]["owner"]
        wrong_box = next(g for g, item in monitor.geometry.items()
                         if item["owner"] == handle_owner and g not in monitor._handle_geoms
                         and int(env.sim.model.geom_type[g]) == 6)
        assert_classification(env, finger, handle, allowed=True, reason="fixture_finger_interaction")
        assert_classification(env, finger, wrong_box, allowed=False)
        assert_classification(env, arm, handle, allowed=False)
