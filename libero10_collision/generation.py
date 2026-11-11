"""Reconstruct scenes from official LIBERO states and explicit placement recipes.

The released benchmark's initial_state.npy files are never generation inputs.
"""
from __future__ import annotations

from copy import deepcopy
import json
from importlib.metadata import version
import platform
from numbers import Integral
from pathlib import Path
import re

import numpy as np

from . import runtime

OBSTACLE = 'collision_obstacle_1'


def _write_json(path, value):
    def native(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                   allow_nan=False, default=native) + '\n')


def _owner_function(env):
    model = env.sim.model
    roots = {model.body_name2id(o.root_body): n for n, o in
             {**env.env.objects_dict, **env.env.fixtures_dict}.items()}
    def owner(geom):
        body = int(model.geom_bodyid[geom])
        while body:
            if body in roots:
                return roots[body]
            if (model.body_id2name(body) or '').startswith(('robot', 'gripper')):
                return 'robot'
            body = int(model.body_parentid[body])
        return 'world'
    return owner


def _bounds(env, name):
    """World AABB of active collision geometry, including compiled mesh vertices."""
    model, data = env.sim.model, env.sim.data
    owner = _owner_function(env)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for gid in range(model.ngeom):
        if owner(gid) != name or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        typ = int(model.geom_type[gid])
        rotation = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        if typ == 6:  # box
            extent = np.abs(rotation) @ size
        elif typ == 2:  # sphere
            extent = np.repeat(size[0], 3)
        elif typ in (3, 5):  # capsule, cylinder
            axis = rotation[:, 2]
            extent = size[0] * np.sqrt(np.maximum(0, 1 - axis * axis)) + size[1] * np.abs(axis)
            if typ == 3:
                extent = np.repeat(size[0], 3) + size[1] * np.abs(axis)
        elif typ == 7:  # mesh: MuJoCo already applied asset scales
            mesh = int(model.geom_dataid[gid])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start:start + count] @ rotation.T + data.geom_xpos[gid]
            lo, hi = np.minimum(lo, vertices.min(0)), np.maximum(hi, vertices.max(0))
            continue
        else:
            raise ValueError(f'Unsupported collision geometry type {typ} for {name}')
        lo = np.minimum(lo, data.geom_xpos[gid] - extent)
        hi = np.maximum(hi, data.geom_xpos[gid] + extent)
    if not np.isfinite(lo).all():
        raise ValueError(f'No collision geometry for {name}')
    return np.array([lo, hi])


def _table_bounds(env):
    model, data = env.sim.model, env.sim.data
    candidates = []
    for gid in range(model.ngeom):
        body = model.body_id2name(int(model.geom_bodyid[gid])) or ''
        if int(model.geom_type[gid]) != 6 or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        if 'table' not in body and model.geom_id2name(gid) != 'table_collision':
            continue
        extent = np.abs(data.geom_xmat[gid].reshape(3, 3)) @ model.geom_size[gid]
        if min(extent[:2]) > .25 and extent[2] < .1:
            candidates.append(np.array([data.geom_xpos[gid] - extent, data.geom_xpos[gid] + extent]))
    if not candidates:
        raise ValueError('No tabletop support geometry found')
    return max(candidates, key=lambda box: box[1, 2])


def _quat_mul(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def _place(env, name, xy, yaw=0, quat=None, gap=.0002):
    joint = env.env.objects_dict[name].joints[0]
    q = np.array(env.sim.data.get_joint_qpos(joint))
    q[:2], q[2] = xy, 1.
    if quat is not None:
        q[3:] = quat
    r = np.deg2rad(yaw) / 2
    q[3:] = _quat_mul([np.cos(r), 0, 0, np.sin(r)], q[3:])
    env.sim.data.set_joint_qpos(joint, q)
    env.sim.data.set_joint_qvel(joint, np.zeros(6))
    env.sim.forward()
    q[2] += _table_bounds(env)[1, 2] - _bounds(env, name)[0, 2] + gap
    env.sim.data.set_joint_qpos(joint, q)
    env.sim.forward()


def _fixtures(env):
    model, data = env.sim.model, env.sim.data
    return {name: {'position': data.body_xpos[model.body_name2id(obj.root_body)].copy(),
                   'quaternion': data.body_xquat[model.body_name2id(obj.root_body)].copy(),
                   'joints': {joint: np.asarray(data.get_joint_qpos(joint)).copy()
                              for joint in obj.joints}}
            for name, obj in env.env.fixtures_dict.items()}


def _fixtures_equal(before, after):
    if before.keys() != after.keys():
        return False
    for name in before:
        for key in ('position', 'quaternion'):
            if not np.allclose(before[name][key], after[name][key], rtol=0, atol=2e-9):
                return False
        if before[name]['joints'].keys() != after[name]['joints'].keys():
            return False
        for joint in before[name]['joints']:
            if not np.allclose(before[name]['joints'][joint], after[name]['joints'][joint], rtol=0, atol=2e-9):
                return False
    return True


def _augment_bddl(source, spec, asset, original):
    workspace = spec['workspace']
    text = source.replace('(:regions', '(:regions\n      (collision_bootstrap_region '
        f'(:target {workspace}) (:ranges (({spec["bootstrap_range"]}))) '
        '(:yaw_rotation ((0.0 0.0))))', 1)
    fixtures = _fixtures(original)
    for obj, region in spec['fixture_clamps']:
        sx, sy = fixtures[obj]['position'][:2]
        epsilon = 1e-9  # positive region width required by MuJoCo
        pattern = r'(\(' + re.escape(region) + r'.*?\(:ranges\s*\(\s*)\([^)]*\)'
        text, count = re.subn(pattern, lambda match: match.group(1) +
            f'({sx-epsilon:.17g} {sy-epsilon:.17g} {sx+epsilon:.17g} {sy+epsilon:.17g})',
            text, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f'Missing fixture initialization region: {region}')
    text = text.replace('(:objects', f'(:objects\n    {OBSTACLE} - {asset["category"]}', 1)
    fact = f'(On {OBSTACLE} {workspace}_collision_bootstrap_region)'
    insertion = spec['init_insertion']
    if insertion == 'prepend':
        text = text.replace('(:init', f'(:init\n    {fact}', 1)
    elif insertion == 'after':
        anchor = spec['init_after']
        if anchor not in text:
            raise ValueError(f'Missing initializer: {anchor}')
        text = text.replace(anchor, anchor + '\n    ' + fact, 1)
    elif insertion == 'append':
        start, goal = text.index('(:init'), text.index('(:goal')
        closing = text.rfind(')', start, goal)
        text = text[:closing] + f'    {fact}\n  ' + text[closing:]
    else:
        raise ValueError(f'Unknown initializer insertion mode: {insertion}')
    return text


def _poses(env):
    return {name: np.array(env.sim.data.get_joint_qpos(obj.joints[0]))
            for name, obj in env.env.objects_dict.items()}


def _one_episode(output, task_recipe, recipe, source, states, render_size, strict_reference, gap):
    from .collision import SceneCollisionMonitor
    manifest = deepcopy(recipe['manifest'])
    directory = output / 'tasks' / manifest['task'] / 'episodes' / f'ep{manifest["episode_id"]:03d}'
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / 'obstacle_asset.json', recipe['obstacle_asset'])
    original = runtime.make_env(source, manifest['seed'], render_size)
    env = None
    try:
        original.set_init_state(np.asarray(states[manifest['source_episode_index']]))
        fixed_source = _fixtures(original)
        bddl = _augment_bddl(source.read_text(), task_recipe['bddl'], recipe['obstacle_asset'], original)
        (directory / 'task.bddl').write_text(bddl)
        env = runtime.make_env(directory / 'task.bddl', manifest['seed'], render_size)
        for jid in range(original.sim.model.njnt):
            name = original.sim.model.joint_id2name(jid)
            env.sim.data.set_joint_qpos(name, original.sim.data.get_joint_qpos(name))
            env.sim.data.set_joint_qvel(name, original.sim.data.get_joint_qvel(name))
        env.sim.data.time = original.sim.data.time
        if env.sim.model.na:
            env.sim.data.act[:] = original.sim.data.act[:]
        env.sim.forward()
        fixed_mapped = _fixtures(env)
        for placement in recipe['placements']:
            _place(env, placement['object'], placement['xy'], placement['yaw_delta_deg'], gap=gap)
        obstacle = recipe['obstacle']
        quat = env.sim.data.get_joint_qpos(env.env.objects_dict[OBSTACLE].joints[0])[3:].copy()
        tilt = obstacle.get('world_y_rotation_deg', 0)
        if tilt:
            # Preserve the original arithmetic for the selected 90 degree cookie orientation.
            rotation = [np.sqrt(.5), 0, np.sqrt(.5), 0] if tilt == 90 else [
                np.cos(np.deg2rad(tilt) / 2), 0, np.sin(np.deg2rad(tilt) / 2), 0]
            quat = _quat_mul(rotation, quat)
        _place(env, OBSTACLE, obstacle['xy'], obstacle['yaw_delta_deg'], quat, gap=gap)
        initial = np.array(env.sim.get_state().flatten())
        np.save(directory / 'initial_state.npy', initial, allow_pickle=False)
        obs = env.set_init_state(initial)
        with SceneCollisionMonitor(env) as monitor:
            for _ in range(manifest['settle_steps']):
                obs, _, _, _ = env.step(runtime.HOLD)
            settled_poses = _poses(env)
            observation_hash = runtime.obs_hash(obs)
            initial_goal = bool(env.check_success())
            boxes = {name: _bounds(env, name) for name in env.env.objects_dict}
            for _ in range(20):
                env.step(runtime.HOLD)
            contact_result = monitor.result(raw_goal_success=bool(env.check_success()))
        final_poses = _poses(env)
        table = _table_bounds(env)
        drift = {name: {'translation_m': float(np.linalg.norm(final_poses[name][:3] - q[:3])),
                       'rotation_rad': float(2*np.arccos(np.clip(abs(np.dot(final_poses[name][3:], q[3:])), 0, 1)))}
                 for name, q in settled_poses.items()}
        actual_hashes = {name: runtime.sha(directory / name)
                         for name in ('task.bddl', 'initial_state.npy', 'obstacle_asset.json')}
        reference = recipe.get('reference', {})
        matches = {name: digest == reference.get('artifact_sha256', {}).get(name)
                   for name, digest in actual_hashes.items()}
        checks = {
            'finite_initial_state': bool(np.isfinite(initial).all()),
            'source_language_preserved': env.env.parsed_problem['language_instruction'] == original.env.parsed_problem['language_instruction'],
            'source_goals_preserved': env.env.parsed_problem['goal_state'] == original.env.parsed_problem['goal_state'] == manifest['task_goal'],
            'source_fixed_fixture_pose_and_articulation_preserved': _fixtures_equal(fixed_source, fixed_mapped),
            'initial_goal_false': not initial_goal,
            'no_scene_collision_during_settle_and_hold': contact_result['scene_collision_free'],
            'all_collision_bounds_inside_table': all(bool(np.all(box[0, :2] >= table[0, :2]+.005) and np.all(box[1, :2] <= table[1, :2]-.005)) for box in boxes.values()),
            'passive_translation_under_2mm': all(item['translation_m'] < .002 for item in drift.values()),
            'passive_rotation_under_1deg': all(item['rotation_rad'] < np.deg2rad(1) for item in drift.values()),
        }
        if strict_reference:
            checks['reference_input_artifacts_identical'] = all(matches.values())
        manifest['obstacle']['asset'] = recipe['obstacle_asset']['source_asset']
        manifest['obstacle']['name'] = OBSTACLE
        manifest['obstacle']['dimensions_m'] = boxes[OBSTACLE][1] - boxes[OBSTACLE][0]
        manifest.update(schema_version=5, distribution='generated',
                        artifact_sha256=actual_hashes, settled_object_poses=settled_poses,
                        initial_observation_sha256=observation_hash,
                        checks_passed=all(checks.values()), checks=checks,
                        generation={'method': 'upstream_state_plus_explicit_geometry_placements_v1',
                                    'reference_artifact_matches': matches,
                                    'reference_observation_matches': observation_hash == reference.get('initial_observation_sha256'),
                                    'observation_render_size': render_size,
                                    'passive_drift': drift,
                                    'validation_scope': 'Source semantics and fixed fixtures, input hash reconstruction, initial scene contacts, tabletop bounds, and passive stability. No policy rollout, robot reachability, or full task completion.'})
        _write_json(directory / 'episode.json', manifest)
        if not manifest['checks_passed']:
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f'{manifest["task"]}/ep{manifest["episode_id"]:03d}: generation checks failed: {failed}; see {directory / "episode.json"}')
        return {'task': manifest['task'], 'episode_id': manifest['episode_id'],
                'checks_passed': True, 'reference_artifact_matches': matches,
                'reference_observation_matches': manifest['generation']['reference_observation_matches']}
    finally:
        if env is not None:
            env.close()
        original.close()


def generate(output_root, tasks=None, episodes=None, *, recipes_path=None,
             render_size=512, strict_reference=True, progress=None):
    """Build a new directory without reading any shipped benchmark state files.

    ``tasks`` is a sequence of t00..t09; ``episodes`` is a sequence of 0..9.
    A nonempty/existing output path is rejected to protect released cases.
    For modified recipes, explicitly disable ``strict_reference``; generated
    variants still need the acceptance process in docs/GENERATION.md.
    """
    if isinstance(render_size, (bool, np.bool_)) or not isinstance(render_size, Integral) or render_size < 1:
        raise ValueError('render_size must be a positive integer')
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    root = runtime.get_data_root() if hasattr(runtime, 'get_data_root') else runtime.ROOT
    path = Path(recipes_path) if recipes_path is not None else root / 'generation' / 'recipes.json'
    recipes = json.loads(path.read_text())
    if recipes.get('schema_version') != 1:
        raise ValueError('Unsupported generation recipe schema')
    gap = float(recipes.get('placement_gap_m', .0002))
    if not np.isfinite(gap) or gap < 0:
        raise ValueError('placement_gap_m must be finite and nonnegative')
    selected_tasks = list(recipes['tasks']) if tasks is None else list(tasks)
    if not selected_tasks or len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError('Select at least one unique task')
    for task in selected_tasks:
        if task not in recipes['tasks']:
            raise ValueError(f'Unknown task {task}; choose from {list(recipes["tasks"])}')
    selected_episodes = list(range(10)) if episodes is None else list(episodes)
    if not selected_episodes or len(set(selected_episodes)) != len(selected_episodes) or any(isinstance(e, (bool, np.bool_)) or not isinstance(e, Integral) or e not in range(10) for e in selected_episodes):
        raise ValueError('Select unique integer episode IDs from 0 through 9')
    runtime.setup()
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Generation requires PyTorch. See docs/INSTALL.md; from a source checkout install CPU PyTorch, then pip install -e '.[generate]'.") from error
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()['libero_10'](task_order_index=0)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'benchmark': 'Libero10-Collision',
              'environment': {'python': platform.python_version(),
                              'packages': {name: version(name) for name in ('numpy', 'mujoco', 'robosuite', 'torch')}},
              'recipe_sha256': runtime.sha(path), 'strict_reference': strict_reference,
              'status': 'in_progress', 'episodes': []}
    try:
        for task_id in selected_tasks:
            task_recipe = recipes['tasks'][task_id]
            first = task_recipe['episodes'][0]['manifest']
            task = suite.get_task(first['official_task_id'])
            source = Path(suite.get_task_bddl_file_path(first['official_task_id']))
            source_states = Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file
            if runtime.sha(source) != first['source_bddl_sha256']:
                raise ValueError(f'Upstream BDDL checksum mismatch: {source}')
            if runtime.sha(source_states) != first['source_states_sha256']:
                raise ValueError(f'Upstream state pool checksum mismatch: {source_states}')
            # These are trusted, SHA-256 verified upstream archives, never arbitrary user pickle files.
            states = torch.load(source_states, map_location='cpu', weights_only=False)
            by_index = {recipe['episode_id']: recipe for recipe in task_recipe['episodes']}
            for episode in selected_episodes:
                result = _one_episode(output, task_recipe, by_index[episode], source, states,
                                      render_size, strict_reference, gap)
                report['episodes'].append(result)
                _write_json(output / 'generation_report.json', report)
                if progress is not None:
                    progress(result)
        source_index = json.loads((root / 'benchmark.json').read_text())
        index = {key: source_index[key] for key in ('name', 'version', 'metric', 'task_numbering', 'tasks')}
        index['validation_scope'] = ('Generated scenes checked for original semantics and fixed fixture preservation, '
                                     'initial contacts, tabletop bounds and passive stability. '
                                     'No new reachability, grasp/fixture probes, route relevance or complete task execution checks.')
        index['runtime_provenance'] = 'generation_report.json'
        index.update(distribution='generated', task_count=len(selected_tasks),
                     episode_count=len(report['episodes']), episodes_per_task=len(selected_episodes))
        index['tasks'] = [task for task in index['tasks'] if task['task'] in selected_tasks]
        for task in index['tasks']:
            task['episode_count'] = len(selected_episodes)
            task['episode_manifests'] = [f'tasks/{task["task"]}/episodes/ep{e:03d}/episode.json' for e in selected_episodes]
        _write_json(output / 'benchmark.json', index)
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        _write_json(output / 'generation_report.json', report)
    return report
