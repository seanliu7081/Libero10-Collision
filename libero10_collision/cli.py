"""Command-line setup, dataset inspection, generation and simulation smoke tools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .runtime import HOLD, get_data_root, load_episode, obs_hash, read_episode


def _write_result(result, output=None):
    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


def _selection(args):
    root = get_data_root(args.data_root)
    registry = json.loads((root / 'benchmark.json').read_text())
    available = {task['task']: task for task in registry['tasks']}
    tasks = args.tasks if args.tasks is not None else sorted(available)
    if len(set(tasks)) != len(tasks) or any(task not in available for task in tasks):
        raise ValueError('Select distinct task IDs present in the dataset registry')
    episodes = args.episodes
    if episodes is not None and (len(set(episodes)) != len(episodes) or any(i not in range(10) for i in episodes)):
        raise ValueError('Select distinct episode IDs from 0 through 9')
    selected = []
    for task in tasks:
        declared = available[task].get('episode_manifests', [])
        declared_ids = sorted(int(Path(name).parent.name.removeprefix('ep')) for name in declared)
        for episode in episodes if episodes is not None else declared_ids:
            if episode not in declared_ids:
                raise ValueError(f'{task} ep{episode:03d} is not declared in this dataset')
            selected.append((task, episode))
    if not selected:
        raise ValueError('No episodes selected')
    return root, selected


def _check_settled_poses(env, manifest, position_tolerance_m=1e-5, rotation_tolerance_rad=1e-4):
    """Compare the policy-ready object state independently of rendered pixels."""
    import numpy as np
    expected = manifest.get('settled_object_poses', {})
    objects = env.env.objects_dict
    if not expected or set(expected) != set(objects):
        raise ValueError('Settled object names differ from the episode manifest')
    max_position, max_rotation = 0., 0.
    for name, pose in expected.items():
        reference = np.asarray(pose, dtype=float)
        actual = np.asarray(env.sim.data.get_joint_qpos(objects[name].joints[0]), dtype=float)
        if reference.shape != (7,) or actual.shape != (7,) or not np.isfinite([reference, actual]).all():
            raise ValueError(f'Invalid settled pose for {name}')
        reference_norm, actual_norm = np.linalg.norm(reference[3:]), np.linalg.norm(actual[3:])
        if min(reference_norm, actual_norm) < 1e-12:
            raise ValueError(f'Invalid settled quaternion for {name}')
        translation = float(np.linalg.norm(actual[:3] - reference[:3]))
        # q and -q describe the same rotation.
        cosine = abs(float(np.dot(actual[3:], reference[3:]) / (reference_norm * actual_norm)))
        rotation = float(2 * np.arccos(np.clip(cosine, 0, 1)))
        max_position, max_rotation = max(max_position, translation), max(max_rotation, rotation)
        if translation > position_tolerance_m or rotation > rotation_tolerance_rad:
            raise ValueError(f'Settled pose mismatch for {name}: {translation:.6g} m, {rotation:.6g} rad')
    return dict(object_count=len(expected), max_translation_error_m=max_position,
                max_rotation_error_rad=max_rotation, position_tolerance_m=position_tolerance_m,
                rotation_tolerance_rad=rotation_tolerance_rad)


def _validate(args):
    import numpy as np
    from .collision import SceneCollisionMonitor
    root, selected = _selection(args)
    records = []
    for task, episode in selected:
        record = dict(task=task, episode_id=episode)
        try:
            directory, manifest = read_episode(task, episode, root)
            state = np.load(directory / 'initial_state.npy', allow_pickle=False)
            if state.ndim != 1 or not np.isfinite(state).all():
                raise ValueError('Invalid saved simulator state array')
            record['artifact_hashes_valid'] = True
            if args.simulate:
                with load_episode(task, episode, render_size=args.render_size, benchmark_root=root) as (env, obs, _):
                    record['settled_pose_check'] = _check_settled_poses(env, manifest)
                    match = obs_hash(obs) == manifest.get('initial_observation_sha256')
                    record['reference_observation_hash_matches'] = match
                    if args.strict_rgb and not match:
                        raise ValueError('Reference RGB/proprioception hash differs; this can depend on renderer/hardware')
                    if bool(env.check_success()):
                        raise ValueError('Task goal is already true at the saved start')
                    with SceneCollisionMonitor(env) as monitor:
                        for _ in range(args.steps):
                            env.step(HOLD)
                            monitor.sample_endpoint()
                        contact = monitor.result(raw_goal_success=bool(env.check_success()))
                    record['scene_collision_free'] = contact['scene_collision_free']
                    record['physics_steps'] = contact['physics_steps']
                    if not contact['scene_collision_free'] or contact['raw_goal_success']:
                        record['collision_result'] = contact
                        raise ValueError('Neutral scene check failed')
            record['passed'] = True
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            record.update(passed=False, error=str(error))
        records.append(record)
        if args.simulate:
            print(f'{task}/ep{episode:03d}: {"PASS" if record["passed"] else "FAIL"}', file=sys.stderr, flush=True)
    result = dict(passed=all(row['passed'] for row in records), episode_count=len(records),
                  mode='simulation_smoke' if args.simulate else 'artifact_integrity',
                  records=records, full_task_execution_tested=False)
    _write_result(result, args.output)
    return 0 if result['passed'] else 1


def _smoke(args):
    from .collision import SceneCollisionMonitor
    with load_episode(args.task, args.episode, render_size=args.render_size,
                      benchmark_root=args.data_root) as (env, obs, episode):
        pose_check = _check_settled_poses(env, episode)
        initial_shapes = {key: list(obs[key].shape) for key in ('agentview_image', 'robot0_eye_in_hand_image')}
        with SceneCollisionMonitor(env) as monitor:
            for _ in range(args.steps):
                obs, _, _, _ = env.step(HOLD)
                monitor.sample_endpoint()
            result = monitor.result(raw_goal_success=bool(env.check_success()))
        result.update(task=args.task, episode_id=args.episode, instruction=episode['language'],
                      observation_shapes=initial_shapes, settled_pose_check=pose_check,
                      neutral_control_steps=args.steps,
                      scope='Neutral-action environment smoke test; no policy or task-completion demonstration.')
        if args.images:
            from PIL import Image
            destination = Path(args.images).expanduser()
            destination.mkdir(parents=True, exist_ok=True)
            for camera, name in [('agentview', 'side.png'), ('robot0_eye_in_hand', 'wrist.png')]:
                frame = env.sim.render(camera_name=camera, width=args.render_size, height=args.render_size)
                Image.fromarray(frame[::-1]).save(destination / name)
            result['review_images_directory'] = str(destination.resolve())
    _write_result(result, args.output)
    return 0 if result['scene_collision_free'] else 1


def build_parser():
    parser = argparse.ArgumentParser(prog='libero10-collision', description=__doc__)
    parser.add_argument('--version', action='version', version='%(prog)s 1.2.0')
    sub = parser.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list', help='List the tasks in the bundled or selected dataset')
    listing.add_argument('--data-root')
    listing.add_argument('--json', action='store_true')
    setup = sub.add_parser('setup', help='Fetch the pinned public LIBERO source and assets')
    setup.add_argument('--libero-root', help='Checkout destination; default is the user cache')
    setup.add_argument('--verify-only', action='store_true')
    setup.add_argument('--output', help='Optional JSON setup report outside the benchmark')
    validate = sub.add_parser('validate', help='Check frozen files; optionally run neutral simulation checks')
    validate.add_argument('--data-root')
    validate.add_argument('--tasks', '--task', nargs='+')
    validate.add_argument('--episodes', nargs='+', type=int)
    validate.add_argument('--simulate', action='store_true')
    validate.add_argument('--strict-rgb', action='store_true', help='Require exact reference renderer hashes')
    validate.add_argument('--render-size', type=int, default=512)
    validate.add_argument('--steps', type=int, default=2)
    validate.add_argument('--output')
    smoke = sub.add_parser('smoke', help='Load one scene and execute neutral actions with collision monitoring')
    smoke.add_argument('--data-root')
    smoke.add_argument('--task', default='t00')
    smoke.add_argument('--episode', type=int, default=0)
    smoke.add_argument('--render-size', type=int, default=512)
    smoke.add_argument('--steps', type=int, default=2)
    smoke.add_argument('--images', help='Optional output directory for side/wrist review PNGs')
    smoke.add_argument('--output')
    generate = sub.add_parser('generate', help='Regenerate episodes from public LIBERO and layout recipes')
    generate.add_argument('--output', required=True, help='A new output directory, separate from the frozen dataset')
    generate.add_argument('--tasks', '--task', nargs='+')
    generate.add_argument('--episodes', nargs='+', type=int)
    generate.add_argument('--recipes', help='Custom generation recipe JSON; default is the bundled recipe set')
    generate.add_argument('--render-size', type=int, default=512)
    generate.add_argument('--allow-variants', action='store_true', help='Allow changed recipes; disables exact release-reference matching, not scene checks')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, 'steps') and args.steps < 1:
        parser.error('--steps must be positive')
    if hasattr(args, 'render_size') and args.render_size < 1:
        parser.error('--render-size must be positive')
    if getattr(args, 'strict_rgb', False) and not args.simulate:
        parser.error('--strict-rgb requires --simulate')
    try:
        if args.command == 'list':
            registry = json.loads((get_data_root(args.data_root) / 'benchmark.json').read_text())
            if args.json:
                _write_result(registry)
            else:
                for task in registry['tasks']:
                    print(f'{task["task"]}  {task["episode_count"]:2d} episodes  {task["language"]}')
            return 0
        if args.command == 'setup':
            from .bootstrap import default_libero_root, setup_libero, verify_libero_checkout
            root = Path(args.libero_root).expanduser() if args.libero_root else default_libero_root()
            if not args.verify_only:
                root = setup_libero(root)
            _write_result(verify_libero_checkout(root), args.output)
            return 0
        if args.command == 'validate':
            return _validate(args)
        if args.command == 'smoke':
            return _smoke(args)
        if args.command == 'generate':
            from .generation import generate
            result = generate(args.output, tasks=args.tasks, episodes=args.episodes,
                              recipes_path=args.recipes, render_size=args.render_size,
                              strict_reference=not args.allow_variants,
                              progress=lambda message: print(message, file=sys.stderr, flush=True))
            _write_result(result)
            return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'error: {error}\n')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
