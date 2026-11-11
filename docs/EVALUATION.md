# Evaluate a policy

The benchmark contains 100 frozen evaluation episodes, grouped into ten tasks.
Load each episode with `load_episode` and create a new `SceneCollisionMonitor`
after the loader returns the policy-ready observation. Keep the monitor active
for every action, including release and post-placement settling. Report the
control-step budget, observation resolution, model version, action conversion,
and stopping rule with scores. The toolkit does not impose a universal rollout
horizon or provide a trained policy; choose a fixed budget for comparisons.

The robot uses a 7D robosuite `OSC_POSE` action at 20 Hz: three translation
controls, three rotation controls, and one gripper control. Use the environment's
`action_spec` limits. Controls are normalized controller inputs, not absolute
world positions. Negative gripper control opens; positive closes.

Provide the policy only the original language instruction, RGB images and robot
proprioception. Whitelist observation keys rather than forwarding the raw
observation dictionary: it also contains privileged object states. Likewise,
recipes, saved states, object poses and collision diagnostics are evaluator-only.
The native `agentview_image` and `robot0_eye_in_hand_image` arrays are returned
unchanged; choose and report your model's image orientation/preprocessing. The
CLI's review PNGs are vertically flipped for display.

A minimal integration looks like this (`policy` is supplied by your model):

```python
import numpy as np
from libero10_collision import load_episode, SceneCollisionMonitor

policy_keys = (
    "agentview_image", "robot0_eye_in_hand_image",
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos",
)

def evaluate_episode(policy, task, episode_id, horizon):
    with load_episode(task, episode_id) as (env, obs, manifest):
        with SceneCollisionMonitor(env) as collisions:
            for step in range(horizon):
                inputs = {key: obs[key].copy() for key in policy_keys}
                action = np.asarray(policy(inputs, manifest["language"]), dtype=float)
                low, high = env.action_spec
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("Policy must return a finite 7D action")
                if np.any(action < low) or np.any(action > high):
                    raise ValueError("Action exceeds controller limits")
                obs, reward, done, info = env.step(action)
                collisions.sample_endpoint()
                if done:
                    break
            result = collisions.result(raw_goal_success=bool(env.check_success()))
        result.update(task=task, episode_id=episode_id, control_steps=step + 1)
        return result
```

Use a positive `horizon` and reset the policy's internal state between episodes.
Evaluate all ten episodes for every task. Aggregate native task success,
collision freedom, and their conjunction over the 100 episodes, and include
per-task rates. A neutral smoke test is an installation check, not a policy
baseline or task demonstration.

The loader verifies each input hash, restores `initial_state.npy`, and executes
exactly ten neutral open-gripper settling controls. Do not call `env.reset()` or
add another initialization settling sequence after loading. BDDL bootstrap
regions construct the model; the saved state fixes the layout.

## Collision metric and validation scope

Use **`SceneCollisionMonitor`** for the current `scene-contact-v1` benchmark
metric. Report native task success and scene collision freedom separately:

| Result | Meaning |
|---|---|
| `raw_goal_success` | Original LIBERO task predicates are satisfied |
| `scene_collision_free` | No forbidden cross-object scene contact occurred during monitoring |
| `added_obstacle_collision_free` | No forbidden contact involving the added obstacle under these scene-contact rules |
| `colliding_pairs` / `onset_events` | Involved owners/geometries, contact counts, penetration and first-contact locations |
| `allowed_contacts_by_reason` | Counts of normal task contacts, separated from collisions |

For collision-free task success, require both `raw_goal_success` and
`scene_collision_free`. The monitor includes original objects, other targets,
distractors, destinations, fixtures, robot links and the environment. Two targets
touching each other count even if both are already inside the correct basket.
A target remains tracked after release. Any forbidden contact keeps the episode's
collision-free result false even after the objects separate.

The following narrow exceptions allow normal task execution:

- **Table support:** a movable object contacts the measured main tabletop's top
  face with an upward normal. Table sides/underside and robot-table contacts
  remain forbidden.
- **Grasp:** original finger/fingerpad collision geoms contact a goal-designated
  manipulation target. Palm/arm contacts and finger contacts with distractors or
  the added obstacle remain forbidden.
- **Fixture operation:** fingers contact the task's explicit drawer/microwave
  handle or stove-knob geoms. Other fixture surfaces remain forbidden.
- **Goal placement:** the correct target contacts the explicit upper support
  surface of its designated destination, within the goal region's horizontal
  footprint where applicable. Plates include their actual sloped support facets.
  Wrong destinations, container walls/dividers and other objects remain forbidden.

Finger–target permission describes a geometric contact category; it does not
infer policy intent or prove a stable grasp.

Allowed contacts must have no more than **1 mm simulated penetration** by default
(`intended_contact_tolerance_m=0.001`). Upward support uses a 35-degree normal
limit and a contact-point/top-face check. Active contact retains the convention
`contact.dist <= contact.includemargin`; there is no debounce that hides short
contacts. The hook inspects every physics substep, with optional endpoint samples.
This is a contact metric, not a force/impact severity test: a shallow contact on an
allowed support surface is allowed regardless of impact speed.

Same-owner contacts are excluded, including robot self-contact and internal
mechanics within one fixture. This metric does not certify general robot safety
or strict terminal placement. Normal task contacts are classified geometrically;
native goal success alone never exempts an entire target–fixture pair.

`AddedObstacleCollisionMonitor` remains available with its unchanged
`added-obstacle-contact-v1` behavior for historical comparisons. New and old
metrics must be labeled with their versions: the new obstacle subset applies the
stricter support/tolerance rules and is not identical to the legacy metric in all
cases. Existing evaluation loops must instantiate `SceneCollisionMonitor` to use
the expanded check.

The frozen suites previously passed scene, scaling, reachability and individual
physical grasp/fixture checks: 160 grasp/lifts and 30 fixture actions. Physical
probes started at initialized pregrasp/precontact; initial approaches were
checked separately. **Full placement, sequential task completion and policy
success remain untested.** Native LIBERO goal predicates are unchanged; they do
not constitute a stricter terminal-placement evaluator.

LIBERO-derived task definitions and source state data retain the upstream
license notice in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
