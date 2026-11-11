# Generate the benchmark from scratch

The generator reconstructs the 100 released scenes from **public LIBERO task
files, public LIBERO initial-state pools, and explicit placement recipes**. It
never reads the released benchmark's `initial_state.npy` files. The recipes
record the selected object layouts, obstacle identities/scales/poses, source
state indices, and reference checksums.

This is a deterministic reconstruction of the published test set. Changing a
recipe creates a separate experimental variant and requires new acceptance
checks before it can be used as a benchmark release.

## 1. Install the environment

Follow [INSTALL.md](INSTALL.md) to create a fresh Python environment, install this
package and rendering dependencies, and fetch the pinned public LIBERO checkout.
Generation additionally needs PyTorch to deserialize the official upstream
initial-state archives; it does not run a neural network or require CUDA PyTorch.
For the documented Linux CPU installation, install CPU PyTorch before the extra:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[generate]'
libero10-collision setup
```

These commands run from the repository root with the new environment activated.
`setup` downloads the upstream code, object assets, BDDL tasks and initial states;
no LIBERO demonstration dataset or model checkpoint is required. A pre-existing
checkout can be selected with `LIBERO_ROOT`. The generator checks the task and
state-pool SHA-256 values before loading source states and rejects a different
LIBERO variant, task order or state collection.

## 2. Generate one case, then all 100

Choose an output directory that does not yet exist. Outputs are separate from
the released `tasks/` directory, which is never overwritten.

```bash
libero10-collision generate --output ./generated/one --tasks t01 --episodes 0
libero10-collision smoke --data-root ./generated/one --task t01 --episode 0 --images ./results/generated-one-views
libero10-collision generate --output ./generated/all
libero10-collision validate --data-root ./generated/all --simulate --output ./results/generated-validation.json
```

`generated-one-views` contains the native external/side and wrist camera images.
The complete command uses official zero-based task IDs `t00`–`t09`, with episode
IDs `0`–`9` for each task. To generate several tasks or episodes, use, for example,
`--tasks t00 t04 --episodes 0 5 9`. A selected subset remains a subset; it must not
be reported as the complete 100-case benchmark.

For EGL rendering, select a GPU before starting the process if needed:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

The setup guide also describes CPU rendering. Generation is sequential within a
process. Independent processes may generate disjoint subsets into different
output directories. Use the same pinned simulator stack for comparisons.

## 3. What the generator does

For each recipe, the generator performs the following steps:

1. Verify and load the original task BDDL and the official source initial-state
   pool. Select the recorded source episode, seed and task order.
2. Create the original scene and restore its source state. Preserve its robot,
   original objects, fixtures, language and native goals.
3. Add a distinct obstacle category and bootstrap initializer. Scale the
   obstacle's visual and collision geometry together while preserving density.
   Preserve initializer ordering where required. For t02, t03 and t09, constrain
   fixed fixture initialization to the original sampled root position with a
   2 nm wide region, because a new object can otherwise alter placement RNG.
4. Compile the augmented scene. Map the original state by **joint name**, rather
   than copying its flattened vector into a model with different state offsets.
5. Apply ordered XY placements and world-yaw offsets to the original free
   objects. Original free-object orientations come from the source state;
   obstacle orientation comes from its asset constructor, with the recorded
   optional world-Y rotation. Compute Z using compiled collision geometry so
   that each object starts 0.2 mm above the tabletop.
6. Write the newly constructed `initial_state.npy`. Restore it and execute the
   recorded ten neutral controls at 20 Hz. Measure the resulting observations
   and object poses, then continue another twenty neutral controls to check
   passive stability.
7. Check original language/goals, fixed fixture roots and articulation, finite
   state, initially false task success, collision bounds inside the table,
   passive translation below 2 mm and rotation below 1 degree, and scene contact
   validity throughout settling and holding. Compare the three generated input
   files against reference hashes when strict reconstruction is enabled.
8. Write new manifests and an aggregate report. A failed check raises an error;
   the partial output and report remain available for inspection. Correct the
   cause and use a new output path for the next run.

The generated input files are `task.bddl`, `obstacle_asset.json` and
`initial_state.npy`. With the reference environment and source checkout, all
three are expected to match the released inputs byte for byte. Generated
`episode.json` files deliberately include newly computed checks and generation
provenance; they are not byte copies of the released manifests.

At the default 512 × 512 resolution, `reference_observation_matches` also
compares the reset RGB/proprioception hash. Rendering can differ across graphics
backends or drivers, so this comparison is reported separately from the strict
input-file comparison. `--render-size` changes observation resolution, not scene
geometry; a non-default resolution cannot match the original observation hash.

## 4. Inspect the output

```text
generated/all/
  benchmark.json
  generation_report.json
  tasks/
    t00/episodes/ep000/
      task.bddl
      obstacle_asset.json
      initial_state.npy
      episode.json
    ...
```

`generation_report.json` records completed cases, exact input-file matches,
observation-hash matches, and completion/failure status. Each generated episode
manifest records its own checks, measured passive drift, actual input hashes
and the validation scope. Load a generated case with the same runtime API:

```python
from libero10_collision.runtime import load_episode

with load_episode("t01", 0, benchmark_root="generated/all") as (env, obs, manifest):
    # Start a fresh collision monitor here and run your policy.
    print(manifest["generation"]["reference_artifact_matches"])
```

See the main [README](../README.md) for the policy interface and scene collision
metric. Frozen states, recipes, object poses and contact information belong to
the evaluator and must not be sent to the policy.

## 5. Create and qualify a new variant

Copy the recipe file before editing it:

```bash
cp generation/recipes.json ./my-recipes.json
```

Each `tasks.tNN.episodes[]` entry contains:

| Field | Meaning |
|---|---|
| `manifest.source_episode_index` | State selected from the original task's official initial-state pool |
| `manifest.seed` | Scene construction seed |
| `placements[]` | Ordered free-object name, world XY in metres, and world yaw offset in degrees |
| `obstacle_asset` | Stock source asset, separate registered category, and uniform or local-axis scale |
| `obstacle.xy` | Obstacle world XY in metres |
| `obstacle.yaw_delta_deg` | World yaw applied after the constructor orientation |
| `obstacle.world_y_rotation_deg` | Optional upright orientation adjustment, applied before yaw |
| `reference` | Expected released-input and reset-observation hashes |

`tasks.tNN.bddl` defines the original workspace, bootstrap range, insertion
position, and preserved fixture roots. Keep it unchanged for ordinary layout
variants. Recipe `manifest` fields describe task identity and authoring intent;
if you change the layout or obstacle, update the corresponding descriptive
fields too. Actual settled poses, obstacle dimensions, artifact hashes and
validation results are recomputed during generation. Give each distinct asset
configuration a distinct category name, particularly if generating multiple
configurations in one process.

To allow deliberate differences from the released input hashes:

```bash
libero10-collision generate --recipes ./my-recipes.json --output ./generated/my-variant --allow-variants
libero10-collision validate --data-root ./generated/my-variant --simulate
libero10-collision smoke --data-root ./generated/my-variant --task t00 --episode 0 --images ./results/my-variant-views/t00-ep000
```

`--allow-variants` disables only equality with released input-file hashes. It
keeps the implemented scene checks enabled. It does not certify a variant as a
released benchmark. Retain its recipe/report, identify the variant explicitly,
and do not mix its scores with the frozen release.

Before accepting new cases, apply the full
[generation protocol](../GENERATION_PROTOCOL.md):

- Check ten distinct obstacle identities and positions, and ten distinct layouts
  for each manipulation target in every task. Preserve language, goals, original
  geometry/physics, cameras, robot and fixed destination frames.
- Inspect every episode with native external and wrist views. Check visual
  bounds, not only collision bounds, against the workspace and native camera
  framing. Disclose any clipping already present in original fixed fixtures.
- Verify goal predicates with positive and negative configurations, including
  the correct destination for each target and articulated conditions.
- Establish obstacle relevance using actual payload collision geometry along
  the intended route, a matching obstacle-disabled counterfactual, and a clear
  alternative route. A visually large obstacle alone is insufficient.
- Check every target's reachable approach and physical grasp/lift, and each
  required fixture interaction. Record whether approaches are sampled
  kinematically or executed dynamically. Validate full sequential task execution
  separately if claiming end-to-end solvability.
- Repeat resets, freeze artifact checksums, and review the full suite before
  publishing a new version.

The public reconstruction command automatically runs the checks listed in step
3. It does **not** rerun the historical route search, goal truth tables, camera
projection audits, 190 robot interaction probes, or demonstration videos. Those
historical checks informed selection of the released recipes. Basic generation
success alone does not establish obstacle relevance, robot reachability or full
task completion for an edited recipe. The original release also does not claim
full sequential task completion or a policy success rate.
