"""Deterministic loading of the frozen Libero10-Collision episodes."""


from __future__ import annotations


from contextlib import contextmanager


import hashlib


import json


import math


import os


from pathlib import Path


import random


import sys


import tempfile


import numpy as np


ROOT = Path(__file__).resolve().parents[1]


LIBERO_ROOT = Path(os.environ.get('LIBERO_ROOT', '/workspace/heading_flow_pub/third_party/LIBERO')).resolve()


HOLD = [0., 0., 0., 0., 0., 0., -1.]


_CONFIG = None


def setup():
    global _CONFIG
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    if _CONFIG is None:
        _CONFIG = tempfile.TemporaryDirectory(prefix='libero10-collision-')
    base = LIBERO_ROOT / 'libero/libero'
    paths = {k: str(v) for k, v in dict(benchmark_root=base, bddl_files=base/'bddl_files',
             init_states=base/'init_files', assets=base/'assets', datasets=LIBERO_ROOT/'libero/datasets').items()}
    Path(_CONFIG.name, 'config.yaml').write_text(json.dumps(paths))
    os.environ['LIBERO_CONFIG_PATH'] = _CONFIG.name
    if str(LIBERO_ROOT) not in sys.path:
        sys.path.insert(0, str(LIBERO_ROOT))
    import libero.libero as package
    if not Path(package.__file__).resolve().is_relative_to(LIBERO_ROOT):
        raise RuntimeError('LIBERO imported from wrong checkout; use a fresh process')
    package.libero_config_path = _CONFIG.name
    package.config_file = str(Path(_CONFIG.name, 'config.yaml'))


def make_env(bddl, seed=7, render_size=512):
    setup()
    from libero.libero.envs import OffScreenRenderEnv
    asset_config=Path(bddl).parent/'obstacle_asset.json'
    if asset_config.exists():
        from .assets import register_scaled_asset
        register_scaled_asset(json.loads(asset_config.read_text()))
    random.seed(seed)
    np.random.seed(seed)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=render_size,
        camera_widths=render_size, control_freq=20, horizon=math.inf, ignore_done=True,
        controller='OSC_POSE', robots=['Panda'])
    env.seed(seed)
    env.reset()
    return env


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def load_episode(task='t01', episode=0, render_size=512, verify=True):
    """Yield (env, raw_obs, manifest). Raw observations include evaluator state.

    The policy-ready start is initial_state.npy followed by exactly ten HOLDs.
    Episode metadata and raw object-state channels are evaluator-only.
    """
    directory = ROOT/'tasks'/task/'episodes'/f'ep{episode:03d}'
    manifest = json.loads((directory/'episode.json').read_text())
    if verify:
        if not manifest.get('checks_passed'):
            raise ValueError('Episode has not passed scene validation')
        for name, digest in manifest['artifact_sha256'].items():
            if sha(directory/name) != digest: raise ValueError(f'Artifact hash mismatch: {name}')
    env = make_env(directory/'task.bddl', manifest['seed'], render_size)
    try:
        state = np.load(directory/'initial_state.npy', allow_pickle=False)
        if state.shape != np.asarray(env.sim.get_state().flatten()).shape or not np.isfinite(state).all():
            raise ValueError('Saved state does not match simulator model')
        obs = env.set_init_state(state)
        for _ in range(manifest['settle_steps']): obs, _, _, _ = env.step(HOLD)
        yield env, obs, manifest
    finally:
        env.close()


def obs_hash(obs):
    h=hashlib.sha256()
    # Native camera array convention is retained for policy channels.
    for k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','robot0_joint_pos'):
        x=np.ascontiguousarray(obs[k]); h.update(k.encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()
