# Libero10-Collision

A frozen evaluation benchmark containing the ten official LIBERO-10 tasks
(`t00`–`t09`), with ten episodes per task (`ep000`–`ep009`): **100 episodes**.
Each task varies the added obstacle identity and position, and the manipulation
object layouts. See [benchmark.json](benchmark.json) for the task registry and
[GENERATION_PROTOCOL.md](GENERATION_PROTOCOL.md) for the benchmark definition.

This folder contains the runtime distribution only: episode inputs, compact
metadata, the loader, the scene collision metric, and documentation.

## Contents

```text
Libero10-Collision/
├── README.md
├── GENERATION_PROTOCOL.md
├── benchmark.json
├── runtime_provenance.json
├── THIRD_PARTY_NOTICES.md
├── libero10_collision/
│   ├── __init__.py
│   ├── runtime.py
│   ├── assets.py
│   └── collision.py
└── tasks/t00…t09/episodes/ep000…ep009/
    ├── episode.json
    ├── task.bddl
    ├── obstacle_asset.json
    └── initial_state.npy
```

`task.bddl`, `initial_state.npy`, and `obstacle_asset.json` define each frozen
scene. Their original bytes are retained. `episode.json` provides the instruction,
goals, IDs, seed, layout metadata, and SHA-256 hashes of those three inputs.
Do not mix files between episodes.

## Runtime

Use the existing interpreter `/workspace/venvs/libero-plus/bin/python` from
this directory. The matching LIBERO checkout and its native assets are required;
they are not duplicated in this package. The default checkout is
`/workspace/heading_flow_pub/third_party/LIBERO`. To use another matching checkout,
set `LIBERO_ROOT` before importing the package. Recorded dependency versions and
simulator settings are in [runtime_provenance.json](runtime_provenance.json).

The loader defaults to EGL rendering. Select an available GPU with
`MUJOCO_EGL_DEVICE_ID` if needed. It retains the stock Panda robot, OSC_POSE
controller, 20 Hz control frequency, and native side/wrist cameras.

```python
from libero10_collision import load_episode
from libero10_collision.collision import SceneCollisionMonitor

with load_episode("t00", 0) as (env, observation, episode):
    instruction = episode["language"]
    with SceneCollisionMonitor(env) as collisions:
        # Replace this neutral action with your policy's 7D OSC_POSE action.
        observation, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
        collisions.sample_endpoint()
        result = collisions.result(raw_goal_success=bool(env.check_success()))
    print(result)
```

The loader registers the declared obstacle variant, checks input hashes,
restores `initial_state.npy`, and executes exactly ten neutral open-gripper
settling controls. The returned observation is the policy-ready starting point.
Do not call `env.reset()` or add another settling sequence after loading.
The BDDL bootstrap regions construct the model; the saved state fixes the layout.

Use only RGB, robot proprioception, and the original instruction as policy inputs.
Raw LIBERO observations also expose object state. Episode metadata, object poses,
saved simulator states, and collision diagnostics are evaluator-only inputs.

## Metrics and scope

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
license notice in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
