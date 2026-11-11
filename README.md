# Libero10-Collision

**100 frozen robot-manipulation episodes with scene collision evaluation**, built
from the ten official LIBERO-10 tasks. Each task has ten different obstacle
identities and positions, plus different manipulation-object layouts. Task IDs
follow official zero-based ordering: `t00`–`t09`, episodes `ep000`–`ep009`.

The toolkit includes the frozen test set, an installable Python package and CLI,
public-source environment setup, deterministic scene generation, and collision
monitoring. No files from the authors' Python environment are required.

- [Install from scratch](docs/INSTALL.md): system dependencies, isolated Python,
  pinned LIBERO source/assets, CPU or GPU rendering, and troubleshooting.
- [Generate from scratch](docs/GENERATION.md): reconstruct all 100 episodes from
  upstream states and recipes, or create and qualify a separate variant.
- [Evaluate your policy](docs/EVALUATION.md): action/observation interface,
  collision rules, reporting, and limitations.
- [Generation protocol](GENERATION_PROTOCOL.md): task definitions and acceptance
  criteria used to select the released scenes.

## Quick start

Reference platform: Linux x86_64, Python 3.10. On Ubuntu/Debian, first install
`git`, `python3.10-venv`, `python3.10-dev`, `build-essential`, `libgl1`,
`libegl1`, `libosmesa6` and
`libglib2.0-0`; see the installation guide for the full command.

```bash
git clone https://github.com/seanliu7081/Libero10-Collision.git
cd Libero10-Collision
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
libero10-collision setup

# CPU rendering; select EGL instead for a graphics-capable NVIDIA GPU.
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
libero10-collision list
libero10-collision validate
libero10-collision smoke --task t01 --episode 0 --images results/t01-ep000
```

`setup` downloads the pinned public LIBERO source and native assets into the user
cache. Runtime evaluation does not need PyTorch, model weights, demonstration
datasets, or a CUDA toolkit. For NVIDIA rendering, set `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl` before starting Python.

Check all 100 initial scenes in the simulator:

```bash
libero10-collision validate --simulate --output results/validation.json
```

This checks input integrity, settled object poses, initially false goals and
neutral-action scene contacts. RGB hashes are reported separately because GPU
drivers and CPU rendering can change pixels. These checks do not run a policy
or demonstrate full task completion.

## Reconstruct the benchmark

Generation starts from upstream LIBERO initial-state pools and
[generation/recipes.json](generation/recipes.json); it never reads the distributed
benchmark state arrays. CPU PyTorch is needed only to deserialize the upstream
state archives:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
libero10-collision generate --output generated/reconstruction
libero10-collision validate --data-root generated/reconstruction --simulate
```

With the reference stack, the 300 generated BDDL, obstacle and state files match
the released inputs byte for byte. Generated manifests record fresh checks and
provenance. Select a subset with `--tasks t00 t01 --episodes 0 1`, or follow the
[generation guide](docs/GENERATION.md) to edit recipes and validate a new variant.
Generation refuses an existing output directory.

## Python interface

```python
from libero10_collision import load_episode, SceneCollisionMonitor

with load_episode("t00", 0) as (env, observation, episode):
    with SceneCollisionMonitor(env) as collisions:
        # Neutral action for this example; replace with your policy's 7D action.
        observation, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
        collisions.sample_endpoint()
        result = collisions.result(raw_goal_success=bool(env.check_success()))
    print(result)
```

The loader returns the policy-ready start after the prescribed ten settling
controls; do not reset again. Policy inputs are RGB, robot proprioception and
original language only. Raw observations also contain privileged object states;
use the whitelist in the [policy integration guide](docs/EVALUATION.md).

Report **native task success**, **scene collision freedom**, and their
conjunction separately. `SceneCollisionMonitor` checks original objects, targets,
distractors, destinations, fixtures, robot links, the environment and the added
obstacle at every physics substep. Narrow geometric exceptions permit intended
table support, finger grasps, fixture operation and correct destination support.
Target–target contacts count, including after release. See the
[complete metric definition](docs/EVALUATION.md#collision-metric-and-validation-scope)
for tolerances and exclusions; the metric version is `scene-contact-v1`.

## Data and reproducibility

| Path | Purpose |
| --- | --- |
| `benchmark.json` | Task registry, metric settings and release metadata |
| `tasks/tNN/episodes/epNNN/` | Four files per episode: BDDL, obstacle spec, saved state and manifest |
| `generation/recipes.json` | Source-state selections and explicit scene construction recipes |
| `libero10_collision/` | Portable loader, collision monitor, setup, generator and CLI |
| `runtime_provenance.json` | Reference simulator settings and pinned public source revision |
| `requirements-lock-py310-linux.txt` | Tested dependency version constraints |
| `tests/` | Data, packaging, runtime and real-scene contact regression tests |

The package supports editable installs and wheels; wheels bundle all 100 cases
and the generation recipes. Commands also work outside the repository directory.
Use `--data-root PATH` or `load_episode(..., benchmark_root=PATH)` to select a
generated dataset. Set `LIBERO_ROOT` to select a custom pinned LIBERO checkout.

The released suites had authoring-time individual approach, grasp/lift and
fixture checks: **160 grasps/lifts and 30 fixture actions**. Physical probes
started at initialized pregrasp/precontact; approaches were checked separately.
The portable generator reruns scene construction and stability checks, not those
historical robot probes. **Full placement, sequential task completion and policy
success remain untested.** An edited recipe needs the additional acceptance
checks described in the generation protocol.

## License and upstream credit

Benchmark code is [MIT licensed](LICENSE). LIBERO-derived task definitions and
state data retain their upstream MIT notice in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The simulator and its assets come
from [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO); third-party
packages retain their own licenses. Please acknowledge LIBERO as well as this
benchmark when using the test set.
