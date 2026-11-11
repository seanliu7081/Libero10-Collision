# Libero10-Collision generation protocol

This document defines the frozen benchmark and its scene acceptance criteria.
For executable reconstruction from public source inputs, see the
[generation guide](docs/GENERATION.md); begin with the
[installation guide](docs/INSTALL.md) on a fresh machine. Historical check
outputs, demonstrations and review pages are not runtime dependencies.

## Task and episode identity

Use official zero-based LIBERO-10 task IDs with `task_order_index=0`. Each task
has ten episodes, indexed 0–9. The original task instruction and native goal
expressions are preserved. Source task names and source initial-state indices
are recorded in episode metadata.

| Task | Original objective |
|---|---|
| t00 | Alphabet soup and tomato sauce into the basket |
| t01 | Cream cheese and butter into the basket |
| t02 | Turn on the stove and put the moka pot on it |
| t03 | Black bowl into the bottom drawer, then close it |
| t04 | White mug onto the left plate; yellow-and-white mug onto the right plate |
| t05 | Book into the caddy's back compartment |
| t06 | White mug onto the plate; chocolate pudding to its right |
| t07 | Alphabet soup and cream cheese into the basket |
| t08 | Both moka pots onto the stove |
| t09 | Yellow-and-white mug into the microwave, then close it |

## Scene construction

For each task, select ten distinct obstacle source identities, ten different
obstacle positions, and ten layouts for every named manipulation object.
Preserve original target/fixture geometry, materials, physics, robot, controller,
native cameras, fixed destination frames, instruction and goals. Permitted
free-object layout changes are encoded in the frozen state.

Added obstacles use separate registered categories. Their visual and collision
geometry scale together; density is preserved and MuJoCo computes the resulting
mass and inertia. Selected rigid assets support source-local axis scaling.
`obstacle_asset.json` declares the exact variant; `assets.py` constructs it from
stock LIBERO assets without modifying those assets.

Obstacle size and placement must obstruct a relevant sampled payload route while
allowing a collision-free alternative and the required individual robot
interactions. Route checks use actual collision geometry and at most 2 mm sample
spacing. The same route must clear when only the added obstacle is disabled.
These authoring checks do not establish full robot transport or insertion.

Fixed fixtures retain the original sampling order or explicitly preserved source
pose. In particular, t03/t09 fixed-fixture positions are encoded in the episode
BDDL and must not be resampled from an unmodified source task. t06 retains the
original fixed right-of-plate region. t08 starts with the original stove on.

## Reset and observations

Construct the model from the episode BDDL and obstacle specification with the
recorded seed, restore its `initial_state.npy`, then execute exactly ten neutral
open-gripper controls at 20 Hz. States are specific to their paired model and
recorded simulator versions. Keep the original native camera array conventions.
The default observation resolution is 512 × 512.

Policies receive only RGB, robot proprioception and original language. Frozen
states, episode metadata, object poses, and evaluator contact data are not policy
inputs. The package provides a loader and collision metric, not a policy adapter.

## Authoring acceptance criteria

The original acceptance process checked exact reset observations, original scene
preservation, support, stability, native-camera framing, goal wiring, obstacle
relevance and asset scaling. Passive stability limits were 2 mm translation and
1 degree rotation. Native frame checks used an 8 px margin at 512 px; inherited
fixed-fixture clipping was disclosed instead of moving fixed destinations.

Every required object or fixture interaction was tested in every episode.
Connected initial robot approaches were sampled at no more than 0.02 rad joint
spacing. Individual physical grasps commanded a 9 cm lift and required at least
5 cm held for 1.25 seconds. Physical fixtures were actuated with real robot
contacts and checked after withdrawal. The suites contain 190 such physical
probes; demonstration reruns matched their baselines exactly.

The t06 fixed right-region site's original camera projection slightly misses
the framing margin, while movable-object bounds are framed. t09's unchanged
microwave hinge has a soft limit: measured closure overshoot reached about
0.003394 rad (0.194 degrees), followed by native Close throughout the released
hold. These source properties are preserved in the runtime scenes.

## Evaluation

The current metric is `scene-contact-v1`, provided by `SceneCollisionMonitor`.
Keep raw LIBERO goal success separate from `scene_collision_free`. For
collision-free task success require both. Every forbidden cross-object contact
counts, including target–target, target–distractor, robot–distractor, incorrect
destination contact and contacts involving the added obstacle. Track original
objects throughout the rollout, including after release.

Allow only explicit normal task contacts: shallow object support on the measured
main tabletop; original fingers grasping a goal target; original fingers
operating the goal's designated handle/knob; and the correct target resting on
its destination's explicit upper support geometry. Destination support uses the
current contact point, upward normal, actual geometry face, and the native goal
region's horizontal footprint where present. Plate support includes its sloped
facets. Container walls/dividers, wrong plates, table sides/undersides and
palm/arm contact receive no such exemption. Native success predicates alone do
not grant permission for contact.

The default intended-contact penetration tolerance is 1 mm. Upward support uses
a 35-degree normal limit and contact-point/top-face checks. The monitor observes
every physics substep using `distance <= includemargin`, without temporal
filtering. Record initial, physics and endpoint contact samples separately;
onsets are counted per owner pair. A forbidden contact remains an episode failure
after separation. Allowed contacts are counted separately by reason. Shallow
allowed support contacts have no speed/force threshold; the metric does not
measure impact severity.

Same-owner internal contacts are excluded, including robot self-contact and
fixture mechanics. The legacy `AddedObstacleCollisionMonitor` retains
`added-obstacle-contact-v1` for old results; report the metric version explicitly.
See [README.md](README.md) for the runtime interface and result fields.

Full target placement, dynamic execution of sampled initial approaches,
sequential task completion and policy success remain untested. The frozen
benchmark cases are unchanged by removing development and review artifacts.
