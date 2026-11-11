# Install from scratch

This guide starts with a fresh machine, the public repository, and a new Python
environment. No private Python environment, model weights, robot demonstration
dataset, or files from the authors' workstation are required.

The reference platform is **Linux x86_64 with CPython 3.10** (Ubuntu 22.04 or
newer). NVIDIA EGL and CPU Mesa rendering are supported. Other Python versions
and native macOS/Windows installations are not reference configurations; use a
Linux environment such as WSL2 for those hosts. The benchmark itself does not
require a CUDA toolkit or a GPU.

## 1. System packages

On Ubuntu/Debian, install Git, Python's virtual environment support, a compiler
for upstream packages that may build native extensions, and rendering libraries:

```bash
sudo apt-get update
sudo apt-get install -y git python3.10 python3.10-venv python3.10-dev build-essential \
  libgl1 libegl1 libosmesa6 libglib2.0-0
```

Use a distribution or environment manager that provides Python 3.10 if your OS
does not have the `python3.10` packages. Check `python3.10 --version` before
continuing. For NVIDIA rendering, the host must also have a working NVIDIA
driver with EGL support. GPU compute access alone does not guarantee EGL access
inside a container.

## 2. Install the benchmark into an isolated environment

```bash
git clone https://github.com/seanliu7081/Libero10-Collision.git
cd Libero10-Collision
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements-lock-py310-linux.txt -e '.[test]'
python -m pip check
```

The constraint file pins the tested Linux/Python 3.10 runtime and its transitive
dependencies. It is a package-version lock, not a wheel hash lock. `pyproject.toml`
also pins the core simulator dependencies, including NumPy 1.26.4, MuJoCo 3.2.6,
and robosuite 1.4.1. Do not install LIBERO's upstream training `requirements.txt`
over this environment: it targets different versions and adds training packages
that evaluation does not need.

Runtime loading and collision monitoring do **not** require PyTorch. Your own
policy can use a separate model environment or install its dependencies here if
they remain compatible with the reference constraints.

## 3. Download the pinned public LIBERO source and assets

```bash
libero10-collision setup
libero10-collision setup --verify-only
```

Setup fetches upstream [LIBERO commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01`](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01)
into `${XDG_CACHE_HOME:-~/.cache}/libero10-collision/libero`. The checkout contains
the BDDL tasks, initial-state archives, meshes, and textures. It needs roughly
650 MB of disk space. There are no compatibility patches: this public revision's
runtime code and assets match those used to author the frozen episodes.

The command is idempotent and verifies both the exact Git revision and tracked
file integrity. It refuses to overwrite an existing unrelated or modified
directory. It does not install packages into your active environment or write
to your normal `~/.libero` configuration.

For another installation location:

```bash
libero10-collision setup --libero-root /path/to/LIBERO
export LIBERO_ROOT=/path/to/LIBERO
libero10-collision setup --verify-only --libero-root "$LIBERO_ROOT"
```

Set `LIBERO_ROOT` in each new shell when using a custom location. Runtime source
resolution prefers that explicit path, then an installed LIBERO package, then
the default cache. A mismatched installed LIBERO can therefore override the
cache; select the pinned source explicitly in that case. Once dependencies and
the source/assets are installed, evaluation requires no network connection.

## 4. Select a rendering backend

For CPU-only execution, including machines without NVIDIA drivers:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

For a headless NVIDIA GPU with EGL:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

Set these before Python imports MuJoCo or PyOpenGL. CPU rendering is slower, but
does not change the action interface or collision definition. The default
backend is EGL. Choose OSMesa explicitly on CPU hosts.

## 5. Verify the installation

```bash
libero10-collision list
libero10-collision validate
libero10-collision smoke --task t01 --episode 0 --images results/smoke
libero10-collision validate --simulate --tasks t00 t01 t02 t03 t04 t05 t06 t07 t08 t09 --episodes 0
```

`validate` checks the frozen artifact hashes without starting the simulator.
`smoke` loads a scene, applies the specified initialization settling, checks its
state and collision monitor, and writes camera images when `--images` is given.
The final command exercises one episode from each of the ten tasks. To check
all 100 scenes, omit `--tasks` and `--episodes`:

```bash
libero10-collision validate --simulate --output results/validation.json
```

The released observation hash includes RGB pixels. Different graphics drivers,
rendering backends, and camera sizes can change that hash even when object
poses and physics agree. Use `--strict-rgb` only when deliberately testing exact
512-pixel reference-renderer reproduction. The regular simulation validation
compares settled free-object positions (10 micrometre tolerance) and rotations
(0.0001 radian tolerance, allowing equivalent quaternion signs), and checks
initial goals and neutral-action contacts. It does not treat a cross-driver pixel
difference alone as an invalid benchmark.

For package regression tests:

```bash
python -m pytest
python -m pytest --simulation
```

These checks establish environment loading and collision behavior. They do not
establish that a policy completes all task goals.

For repository CI, an optional [GitHub Actions template](github-actions.yml)
builds and tests the installed wheel on Python 3.10/3.11 and runs real scene
checks with CPU OSMesa. To enable it in your repository, copy it to
`.github/workflows/tests.yml` and push with credentials authorized to manage
Actions workflows. The template is not an active workflow in this release.

## 6. Generate scenes from the public source inputs

Generation additionally reads LIBERO's upstream `.pruned_init` archives using
PyTorch. A CPU wheel is sufficient, even when EGL is used for rendering:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
libero10-collision generate --output generated/Libero10-Collision
libero10-collision validate --data-root generated/Libero10-Collision --simulate
```

The checked-in recipes define layouts, obstacle assets and sizes, and fixed
fixture placements. The generator starts from the public upstream inputs;
it does not copy the released `initial_state.npy` files. See
[GENERATION.md](GENERATION.md) for the full construction and review procedure,
including how to create a distinct benchmark variant.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `libero10-collision: command not found` | Activate `.venv` and run the package installation command again. |
| Missing LIBERO source/assets | Run `libero10-collision setup`; check custom `LIBERO_ROOT` values. |
| `EGL_NOT_INITIALIZED`, EGL library errors, or no graphics-capable GPU | Use OSMesa, or make the host's EGL driver libraries available in the container. |
| Cannot load `OSMesa` | Install `libosmesa6` and set both backend variables before launching Python. |
| `DISPLAY` or GLFW errors on a headless host | Set `MUJOCO_GL=egl` or `osmesa`; restart Python after changing it. |
| NumPy, MuJoCo, or robosuite incompatibility | Recreate the isolated environment with the provided constraints. |
| Pixel hash differs on another machine | Compare state checks first; EGL/OSMesa/driver output need not be bit identical. |
| Git verification reports modified files | Preserve your changes and choose a fresh `--libero-root` destination. |

robosuite's optional private-macros warning and upstream Gym's maintenance
notice are expected with this reference stack. Creating a private macros file
or replacing Gym is not required to run the benchmark.
