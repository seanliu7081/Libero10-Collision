"""Portable, deterministic loading of frozen Libero10-Collision episodes."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import operator
import os
from pathlib import Path
import random
import re
import sys
import tempfile

import numpy as np

HOLD = [0., 0., 0., 0., 0., 0., -1.]
_CONFIG = None


def get_data_root(override=None):
    """Find benchmark data in an explicit directory, checkout, or installed wheel.

    LIBERO10_COLLISION_ROOT selects another generated dataset. An explicitly
    selected invalid directory raises rather than silently evaluating other data.
    """
    selected = override if override is not None else os.environ.get('LIBERO10_COLLISION_ROOT')
    if selected is not None:
        candidates = [Path(selected).expanduser().resolve()]
    else:
        package = Path(__file__).resolve().parent
        candidates = [package.parent, package / 'data']
    for root in candidates:
        if (root / 'benchmark.json').is_file() and (root / 'tasks').is_dir():
            return root
    raise FileNotFoundError(
        'Benchmark data not found. Set LIBERO10_COLLISION_ROOT to a directory '
        'containing benchmark.json and tasks/, or reinstall the complete package. '
        f'Searched: {", ".join(map(str, candidates))}')


def default_libero_root():
    """Default bootstrap checkout, independent of the current working directory."""
    cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))).expanduser()
    return (cache / 'libero10-collision' / 'libero').resolve()


def resolve_libero_root():
    """Resolve LIBERO without importing its interactive configuration module."""
    selected = os.environ.get('LIBERO_ROOT')
    if selected:
        candidates = [Path(selected).expanduser().resolve()]
    else:
        candidates = []
        spec = importlib.util.find_spec('libero')
        if spec is not None:
            if spec.submodule_search_locations:
                candidates.extend(Path(p).resolve().parent for p in spec.submodule_search_locations)
            elif spec.origin:
                candidates.append(Path(spec.origin).resolve().parent.parent)
        candidates.append(default_libero_root())
    for root in candidates:
        base = root / 'libero' / 'libero'
        if (base / '__init__.py').is_file() and (base / 'assets').is_dir() and (base / 'bddl_files').is_dir():
            return root
    raise FileNotFoundError(
        'LIBERO source/assets not found. Run `libero10-collision setup`, or set '
        'LIBERO_ROOT to the matching LIBERO checkout. '
        f'Searched: {", ".join(map(str, candidates))}')


# Compatibility for evaluator helpers; loading resolves overrides dynamically.
ROOT = get_data_root()


def setup():
    """Configure rendering and a private LIBERO config before backend imports."""
    global _CONFIG
    backend = os.environ.setdefault('MUJOCO_GL', 'egl')
    if backend in ('egl', 'osmesa'):
        platform = os.environ.setdefault('PYOPENGL_PLATFORM', backend)
        if platform != backend:
            raise ValueError(f'PYOPENGL_PLATFORM={platform} conflicts with MUJOCO_GL={backend}; '
                             'set both to the same backend in a fresh process.')
    root = resolve_libero_root()
    if _CONFIG is None:
        _CONFIG = tempfile.TemporaryDirectory(prefix='libero10-collision-')
    base = root / 'libero' / 'libero'
    # Runtime/generation do not need expert demonstration datasets. Use an
    # existing private directory so LIBERO never prompts or warns for that path.
    datasets = Path(_CONFIG.name) / 'datasets'
    datasets.mkdir(exist_ok=True)
    paths = {key: str(value) for key, value in dict(
        benchmark_root=base, bddl_files=base / 'bddl_files',
        init_states=base / 'init_files', assets=base / 'assets', datasets=datasets).items()}
    Path(_CONFIG.name, 'config.yaml').write_text(json.dumps(paths))
    os.environ['LIBERO_CONFIG_PATH'] = _CONFIG.name
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import libero.libero as package
    if not Path(package.__file__).resolve().is_relative_to(root):
        raise RuntimeError('LIBERO was already imported from a different checkout; use a fresh process.')
    package.libero_config_path = _CONFIG.name
    package.config_file = str(Path(_CONFIG.name, 'config.yaml'))
    return root


def make_env(bddl, seed=7, render_size=512):
    setup()
    from libero.libero.envs import OffScreenRenderEnv
    asset_config = Path(bddl).parent / 'obstacle_asset.json'
    if asset_config.exists():
        from .assets import register_scaled_asset
        register_scaled_asset(json.loads(asset_config.read_text()))
    random.seed(seed)
    np.random.seed(seed)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=render_size,
        camera_widths=render_size, control_freq=20, horizon=math.inf, ignore_done=True,
        controller='OSC_POSE', robots=['Panda'])
    try:
        env.seed(seed)
        env.reset()
    except BaseException:
        env.close()
        raise
    return env


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_directory(task, episode, benchmark_root=None):
    """Validate public IDs before touching simulator dependencies."""
    if not isinstance(task, str) or re.fullmatch(r't0[0-9]', task) is None:
        raise ValueError('Expected a task ID from t00 through t09')
    if isinstance(episode, (bool, np.bool_)):
        raise ValueError('Expected an integer episode ID from 0 through 9')
    try:
        number = operator.index(episode)
    except TypeError as error:
        raise ValueError('Expected an integer episode ID from 0 through 9') from error
    if number not in range(10):
        raise ValueError('Expected an integer episode ID from 0 through 9')
    return get_data_root(benchmark_root) / 'tasks' / task / 'episodes' / f'ep{number:03d}'


def read_episode(task, episode, benchmark_root=None, verify=True):
    """Read and optionally verify one episode without importing LIBERO/MuJoCo."""
    directory = episode_directory(task, episode, benchmark_root)
    manifest = json.loads((directory / 'episode.json').read_text())
    if manifest.get('task') != task or manifest.get('episode_id') != int(episode):
        raise ValueError('Episode manifest identity does not match its directory')
    if verify:
        if not manifest.get('checks_passed'):
            raise ValueError('Episode has not passed scene validation')
        required = {'task.bddl', 'initial_state.npy', 'obstacle_asset.json'}
        hashes = manifest.get('artifact_sha256', {})
        if not required.issubset(hashes):
            raise ValueError('Episode manifest is missing required artifact hashes')
        for name, digest in hashes.items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError(f'Artifact path escapes episode directory: {name}')
            if sha(path) != digest:
                raise ValueError(f'Artifact hash mismatch: {name}')
    return directory, manifest


@contextmanager
def load_episode(task='t01', episode=0, render_size=512, verify=True, *, benchmark_root=None):
    """Yield (env, raw_obs, manifest), closing the environment on exit.

    The policy-ready start is initial_state.npy followed by exactly ten HOLDs.
    Raw observations contain evaluator object state; policy adapters must select
    only RGB/proprioception and the original language instruction.
    """
    if isinstance(render_size, bool) or not isinstance(render_size, int) or render_size <= 0:
        raise ValueError('render_size must be a positive integer')
    directory, manifest = read_episode(task, episode, benchmark_root, verify)
    state = np.load(directory / 'initial_state.npy', allow_pickle=False)
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError('Saved state must be a finite one-dimensional array')
    env = make_env(directory / 'task.bddl', manifest['seed'], render_size)
    try:
        if state.shape != np.asarray(env.sim.get_state().flatten()).shape:
            raise ValueError('Saved state does not match simulator model; check the pinned dependency versions')
        obs = env.set_init_state(state)
        for _ in range(manifest['settle_steps']):
            obs, _, _, _ = env.step(HOLD)
        yield env, obs, manifest
    finally:
        env.close()


def obs_hash(obs):
    digest = hashlib.sha256()
    for key in ('agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos',
                'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_joint_pos'):
        value = np.ascontiguousarray(obs[key])
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()
