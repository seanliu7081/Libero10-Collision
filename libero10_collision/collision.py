"""Evaluator-only collision metrics for Libero10-Collision.

SceneCollisionMonitor checks inter-object scene contacts with narrow task-contact
exceptions. AddedObstacleCollisionMonitor retains the original obstacle-only v1
metric. Both report native LIBERO goal success separately from collision freedom.
"""
from __future__ import annotations

from copy import deepcopy
import math

OBSTACLE_NAME = 'collision_obstacle_1'
METRIC_VERSION = 'added-obstacle-contact-v1'
SUPPORT_COS = math.cos(math.radians(35.0))


class AddedObstacleCollisionMonitor:
    """Context manager hooking every actual ``env.sim.step`` physics step.

    Attach after the benchmark's prescribed reset settling. Initial and optional
    endpoint samples are counted separately from physics samples. An onset is a
    forbidden *owner pair* absent in the previous inspection, then present; a
    changing contact manifold does not create additional onsets. These are
    observed contact onsets, not impact-force estimates.

    Example::

        with AddedObstacleCollisionMonitor(env) as monitor:
            obs, reward, done, info = env.step(action)
            monitor.sample_endpoint()  # optional endpoint refresh, warmstart kept
            result = monitor.result(raw_goal_success=bool(env.check_success()))

    Closing the context restores the exact previous sim.step callable. It does
    not reset the cumulative record; construct a new monitor for each episode.
    """

    def __init__(self, env, obstacle_name=OBSTACLE_NAME):
        self.env = env
        self.sim = env.sim
        self.core = getattr(env, 'env', env)
        self.obstacle_name = obstacle_name
        if obstacle_name not in self.core.objects_dict:
            raise ValueError(f'Added obstacle is missing: {obstacle_name}')
        self.geometry = self._geometry()
        if not any(g['owner'] == obstacle_name for g in self.geometry.values()):
            raise ValueError('Added obstacle has no active collision geometry')
        self.physics_steps = 0
        self.inspections = {'initial': 0, 'physics': 0, 'endpoint': 0}
        self.samples = {'initial': 0, 'physics': 0, 'endpoint': 0}
        self.table_support_samples = {'initial': 0, 'physics': 0, 'endpoint': 0}
        self._active_pairs = set()
        self._pairs = {}
        self.events = []
        self.max_penetration_m = 0.0
        self._previous_step = None
        self._hook = self._physics_step
        self._attached = False
        self._ever_attached = False

    def _geometry(self):
        model = self.sim.model
        roots = {int(model.body_name2id(obj.root_body)): name for name, obj in
                 {**self.core.objects_dict, **self.core.fixtures_dict}.items()}
        workspace = getattr(self.core, 'workspace_name', None)
        table_bodies = {workspace, f'{workspace}_col'} if workspace else set()
        result = {}
        for geom in range(int(model.ngeom)):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            geom_name = model.geom_id2name(geom) or f'geom_{geom}'
            body = int(model.geom_bodyid[geom])
            owner = None
            table = geom_name == 'table_collision'
            while body:
                body_name = model.body_id2name(body) or ''
                table = table or body_name in table_bodies
                if owner is None and body in roots:
                    owner = roots[body]
                if owner is None and body_name.startswith(('robot', 'gripper', 'mount')):
                    owner = 'robot'
                body = int(model.body_parentid[body])
            if table and owner is None:
                owner = workspace or 'table'
            result[geom] = {'name': geom_name, 'owner': owner or 'scene', 'table': table}
        return result

    def attach(self):
        if self._attached or self._ever_attached:
            raise RuntimeError('Use a fresh collision monitor for each episode')
        self._previous_step = self.sim.step
        self.sim.step = self._hook
        self._attached = self._ever_attached = True
        try:
            self._inspect('initial')
        except BaseException:
            self.detach()
            raise
        return self

    def detach(self):
        if self._attached:
            # Restoring the original callable also works when an exception leaves
            # env.step midway through a control action.
            self.sim.step = self._previous_step
            self._attached = False

    def __enter__(self):
        return self.attach()

    def __exit__(self, exc_type, exc, tb):
        self.detach()
        return False

    def _physics_step(self, *args, **kwargs):
        result = self._previous_step(*args, **kwargs)
        self.physics_steps += 1
        self._inspect('physics')
        return result

    def sample_endpoint(self, *, refresh=True):
        """Optional extra sample; never contributes to physics contact counts.

        MuJoCo forward updates the solver warmstart. Restore it even on errors so
        this diagnostic refresh does not seed the next solve differently.
        """
        if not self._attached:
            raise RuntimeError('Attach the monitor before endpoint sampling')
        if refresh:
            data = getattr(self.sim.data, '_data', self.sim.data)
            warmstart = data.qacc_warmstart.copy()
            try:
                self.sim.forward()
            finally:
                data.qacc_warmstart[:] = warmstart
        self._inspect('endpoint')

    def _inspect(self, phase):
        self.inspections[phase] += 1
        active = set()
        first_for_pair = {}
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            distance, margin = float(contact.dist), float(contact.includemargin)
            if not math.isfinite(distance) or not math.isfinite(margin):
                raise ValueError('Non-finite contact distance/margin')
            if distance > margin:
                continue
            first = self.geometry.get(int(contact.geom1))
            second = self.geometry.get(int(contact.geom2))
            if first is None or second is None:
                raise ValueError('Active contact has unmapped collision geometry')
            if first['owner'] != self.obstacle_name and second['owner'] != self.obstacle_name:
                continue
            if first['owner'] == second['owner']:
                continue
            obstacle_is_second = second['owner'] == self.obstacle_name
            other = first if obstacle_is_second else second
            # MuJoCo's normal points from geom1 towards geom2.
            upward = float(contact.frame[2]) * (1.0 if obstacle_is_second else -1.0)
            if other['table'] and upward >= SUPPORT_COS:
                self.table_support_samples[phase] += 1
                continue
            pair = tuple(sorted((self.obstacle_name, other['owner'])))
            active.add(pair)
            penetration = max(0.0, -distance)
            record = self._pairs.setdefault(pair, {
                'owners': list(pair), 'physics_contact_samples': 0,
                'initial_contact_samples': 0, 'endpoint_contact_samples': 0,
                'distinct_onset_events': 0, 'max_penetration_m': 0.0,
                'geom_pairs': set(),
            })
            record[f'{phase}_contact_samples'] += 1
            record['geom_pairs'].add(tuple(sorted((first['name'], second['name']))))
            record['max_penetration_m'] = max(record['max_penetration_m'], penetration)
            self.samples[phase] += 1
            self.max_penetration_m = max(self.max_penetration_m, penetration)
            first_for_pair.setdefault(pair, {
                'owners': list(pair), 'phase': phase, 'physics_step': self.physics_steps,
                'geom1': first['name'], 'geom2': second['name'],
                'distance_m': distance, 'includemargin_m': margin,
            })
        for pair in sorted(active - self._active_pairs):
            self._pairs[pair]['distinct_onset_events'] += 1
            self.events.append(first_for_pair[pair])
        self._active_pairs = active

    def snapshot(self):
        pairs = []
        for pair in sorted(self._pairs):
            row = deepcopy(self._pairs[pair])
            row['geom_pairs'] = [list(p) for p in sorted(row['geom_pairs'])]
            pairs.append(row)
        return {
            'metric_version': METRIC_VERSION,
            'metric_scope': 'added_obstacle_only',
            'obstacle_name': self.obstacle_name,
            'added_obstacle_collision_free': not bool(self.events),
            'physics_steps': self.physics_steps,
            'inspections_by_phase': dict(self.inspections),
            'physics_contact_samples': self.samples['physics'],
            'initial_contact_samples': self.samples['initial'],
            'endpoint_contact_samples': self.samples['endpoint'],
            'allowed_table_support_samples_by_phase': dict(self.table_support_samples),
            'distinct_onset_events': len(self.events),
            'colliding_pairs': pairs,
            'onset_events': deepcopy(self.events),
            'max_penetration_m': self.max_penetration_m,
        }

    def result(self, *, raw_goal_success):
        if type(raw_goal_success) is not bool:
            raise TypeError('raw_goal_success must be a bool from LIBERO check_success()')
        return {'raw_goal_success': raw_goal_success, **self.snapshot()}


SCENE_METRIC_VERSION = 'scene-contact-v1'


class SceneCollisionMonitor(AddedObstacleCollisionMonitor):
    """Monitor inter-object scene contacts with explicit task-contact exceptions.

    All cross-owner contacts are forbidden unless they are shallow tabletop
    support, intended finger grasps, declared fixture-handle interaction, or
    upward support on the correct destination surface within its goal footprint.
    Original objects remain tracked after release. Same-owner contacts, including
    robot self-contact and internal fixture mechanics, are outside this metric.
    Geometry rules cover the frozen LIBERO-10 scenes; unsupported destinations
    fail closed. No native success predicate grants a blanket contact exemption.
    """

    _DESTINATION_FLOORS = {
        'basket_1': ('basket_1_g1',),
        'desk_caddy_1': ('desk_caddy_1_g1',),
        'white_cabinet_1': ('white_cabinet_1_g38',),
        'microwave_1': ('microwave_1_g7',),
        'flat_stove_1': ('flat_stove_1_collision_burner',),
        'plate_1': tuple(f'plate_1_g{i}' for i in range(1, 11)),
        'plate_2': tuple(f'plate_2_g{i}' for i in range(1, 11)),
    }
    _FIXTURE_HANDLES = {
        'white_cabinet_1_bottom_region': tuple(f'white_cabinet_1_g{i}' for i in (40, 41, 42)),
        'microwave_1': tuple(f'microwave_1_g{i}' for i in (14, 15, 16)),
        'flat_stove_1': tuple(f'flat_stove_1_g{i}' for i in range(5, 23)),
    }

    def __init__(self, env, *, intended_contact_tolerance_m=0.001):
        if not math.isfinite(intended_contact_tolerance_m) or intended_contact_tolerance_m < 0:
            raise ValueError('Expected a finite nonnegative intended-contact penetration tolerance')
        self.intended_contact_tolerance_m = float(intended_contact_tolerance_m)
        super().__init__(env)
        self._allowed = {}
        self._ignored_same_owner = dict.fromkeys(self.inspections, 0)
        self._movables = set(self.core.objects_dict)
        self._fixtures = set(self.core.fixtures_dict)
        self._finger_geoms = set()
        for robot in self.core.robots:
            for key in ('left_finger', 'right_finger', 'left_fingerpad', 'right_fingerpad'):
                for name in robot.gripper.important_geoms.get(key, []):
                    self._finger_geoms.add(self._geom_id(name))
        if not self._finger_geoms:
            raise ValueError('Cannot identify the original finger collision geometries')
        self._table_geoms = self._find_tabletop()
        self._grasp_targets = set()
        self._handle_geoms = set()
        self._support_rules = {}
        self._prepare_goal_contacts()

    def _geom_id(self, name):
        gid = int(self.sim.model.geom_name2id(name))
        if gid not in self.geometry:
            raise ValueError(f'Expected active collision geometry: {name}')
        return gid

    def _find_tabletop(self):
        import numpy as np
        model, data = self.sim.model, self.sim.data
        candidates = []
        for gid, geometry in self.geometry.items():
            if not geometry['table'] or int(model.geom_type[gid]) != 6:
                continue
            extent = np.abs(data.geom_xmat[gid].reshape(3, 3)) @ model.geom_size[gid]
            if min(extent[:2]) > 0.25 and extent[2] < 0.1:
                candidates.append((float(data.geom_xpos[gid, 2] + extent[2]), gid))
        if not candidates:
            raise ValueError('Cannot identify the main tabletop collision box')
        return {max(candidates)[1]}

    def _prepare_goal_contacts(self):
        model = self.sim.model
        for atom in self.core.parsed_problem['goal_state']:
            predicate = atom[0].lower()
            if predicate in ('in', 'on'):
                target, destination = atom[1:3]
                if target not in self._movables or target == self.obstacle_name:
                    raise ValueError(f'Unsupported manipulation target: {target}')
                self._grasp_targets.add(target)
                site = None
                if destination in self._movables or destination in self._fixtures:
                    parent = destination
                else:
                    state = self.core.object_states_dict[destination]
                    parent = getattr(state, 'parent_name', None)
                    site = int(model.site_name2id(destination))
                    if parent is None:
                        # Fixed tabletop destination (t06 pudding): ordinary
                        # table support is allowed; goal success is separate.
                        if predicate != 'on':
                            raise ValueError(f'Unsupported parentless destination: {destination}')
                        continue
                if parent not in self._DESTINATION_FLOORS:
                    raise ValueError(f'No explicit support geometry for {destination}')
                floors = {self._geom_id(n) for n in self._DESTINATION_FLOORS[parent]}
                if any(self.geometry[g]['owner'] != parent for g in floors):
                    raise ValueError(f'Destination geometry owner mismatch: {destination}')
                if any(int(model.geom_type[g]) != 6 for g in floors):
                    raise ValueError(f'Expected original box support geometry: {destination}')
                self._support_rules.setdefault((target, parent), []).append(
                    dict(destination=destination, geoms=floors, site=site))
            elif predicate in ('open', 'close', 'turnon', 'turnoff'):
                destination = atom[1]
                if destination not in self._FIXTURE_HANDLES:
                    raise ValueError(f'No explicit fixture interaction geometry for {destination}')
                self._handle_geoms.update(self._geom_id(n) for n in self._FIXTURE_HANDLES[destination])
            else:
                raise ValueError(f'Unsupported task goal predicate: {predicate}')

    def _on_top_face(self, contact, support_gid, supported_gid):
        """Use the actual box's current top face, not table-body ancestry."""
        import numpy as np
        model, data = self.sim.model, self.sim.data
        normal_z = float(contact.frame[2]) * (1 if int(contact.geom2) == supported_gid else -1)
        if normal_z < SUPPORT_COS:
            return False
        rotation = data.geom_xmat[support_gid].reshape(3, 3)
        vertical = int(np.argmax(np.abs(rotation[2])))
        sign = 1 if rotation[2, vertical] >= 0 else -1
        if abs(float(rotation[2, vertical])) < SUPPORT_COS:
            return False
        local = rotation.T @ (np.asarray(contact.pos) - data.geom_xpos[support_gid])
        size = model.geom_size[support_gid]
        # A contact point can lie between the two surfaces, at up to the
        # penetration tolerance plus the simulator's positive contact margin.
        tolerance = self.intended_contact_tolerance_m + max(0., float(contact.includemargin))
        if abs(float(local[vertical] - sign * size[vertical])) > tolerance + 1e-9:
            return False
        return all(abs(float(local[i])) <= float(size[i]) + tolerance + 1e-9
                   for i in range(3) if i != vertical)

    def _inside_goal_footprint(self, contact, site):
        if site is None:
            return True  # Explicit plate floor geometry supplies its footprint.
        import numpy as np
        model, data = self.sim.model, self.sim.data
        rotation = data.site_xmat[site].reshape(3, 3)
        vertical = int(np.argmax(np.abs(rotation[2])))
        if abs(float(rotation[2, vertical])) < SUPPORT_COS:
            return False
        local = rotation.T @ (np.asarray(contact.pos) - data.site_xpos[site])
        return all(abs(float(local[i])) <= float(model.site_size[site, i]) + 1e-9
                   for i in range(3) if i != vertical)

    def _allowed_reason(self, contact, first_gid, second_gid):
        first, second = self.geometry[first_gid], self.geometry[second_gid]
        if float(contact.dist) < -self.intended_contact_tolerance_m:
            return None
        # A support exception applies to movable objects, never to robot links.
        for support_gid, supported_gid, supported in (
            (first_gid, second_gid, second), (second_gid, first_gid, first)
        ):
            if support_gid in self._table_geoms and supported['owner'] in self._movables:
                if self._on_top_face(contact, support_gid, supported_gid):
                    return 'tabletop_support'
        # The added obstacle gets no grasp, handle, or destination exception.
        if self.obstacle_name in (first['owner'], second['owner']):
            return None
        for finger_gid, other_gid, other in (
            (first_gid, second_gid, second), (second_gid, first_gid, first)
        ):
            if finger_gid in self._finger_geoms:
                if other['owner'] in self._grasp_targets:
                    return 'target_finger_grasp'
                if other_gid in self._handle_geoms:
                    return 'fixture_finger_interaction'
        for target_gid, support_gid, target, support in (
            (first_gid, second_gid, first, second), (second_gid, first_gid, second, first)
        ):
            for rule in self._support_rules.get((target['owner'], support['owner']), []):
                if (support_gid in rule['geoms']
                        and self._on_top_face(contact, support_gid, target_gid)
                        and self._inside_goal_footprint(contact, rule['site'])):
                    return 'goal_destination_support'
        return None

    def _collision_kind(self, first, second):
        def kind(geometry):
            owner = geometry['owner']
            if owner == 'robot': return 'robot'
            if owner in self._movables: return 'object'
            if owner in self._fixtures: return 'fixture'
            return 'world'
        return '_'.join(sorted((kind(first), kind(second))))

    def _inspect(self, phase):
        self.inspections[phase] += 1
        active, first_for_pair = set(), {}
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            distance, margin = float(contact.dist), float(contact.includemargin)
            if not math.isfinite(distance) or not math.isfinite(margin):
                raise ValueError('Non-finite contact distance/margin')
            if distance > margin:
                continue
            first_gid, second_gid = int(contact.geom1), int(contact.geom2)
            first, second = self.geometry.get(first_gid), self.geometry.get(second_gid)
            if first is None or second is None:
                raise ValueError('Active contact has unmapped collision geometry')
            if first['owner'] == second['owner']:
                self._ignored_same_owner[phase] += 1
                continue
            reason = self._allowed_reason(contact, first_gid, second_gid)
            if reason is not None:
                self._allowed.setdefault(reason, dict.fromkeys(self.inspections, 0))[phase] += 1
                if reason == 'tabletop_support':
                    self.table_support_samples[phase] += 1
                continue
            pair = tuple(sorted((first['owner'], second['owner'])))
            active.add(pair)
            penetration = max(0., -distance)
            record = self._pairs.setdefault(pair, dict(
                owners=list(pair), collision_kind=self._collision_kind(first, second),
                physics_contact_samples=0, initial_contact_samples=0,
                endpoint_contact_samples=0, distinct_onset_events=0,
                max_penetration_m=0., geom_pairs=set()))
            record[f'{phase}_contact_samples'] += 1
            record['geom_pairs'].add(tuple(sorted((first['name'], second['name']))))
            record['max_penetration_m'] = max(record['max_penetration_m'], penetration)
            self.samples[phase] += 1
            self.max_penetration_m = max(self.max_penetration_m, penetration)
            first_for_pair.setdefault(pair, dict(
                owners=list(pair), collision_kind=record['collision_kind'],
                phase=phase, physics_step=self.physics_steps,
                geom1=first['name'], geom2=second['name'],
                distance_m=distance, includemargin_m=margin,
                contact_position_m=list(map(float, contact.pos))))
        for pair in sorted(active - self._active_pairs):
            self._pairs[pair]['distinct_onset_events'] += 1
            self.events.append(first_for_pair[pair])
        self._active_pairs = active

    def snapshot(self):
        result = super().snapshot()
        obstacle_pairs = [r for r in result['colliding_pairs'] if self.obstacle_name in r['owners']]
        result.update(
            metric_version=SCENE_METRIC_VERSION,
            metric_scope='scene_inter_object',
            scene_collision_free=not bool(self.events),
            added_obstacle_collision_free=not bool(obstacle_pairs),
            added_obstacle_distinct_onset_events=sum(r['distinct_onset_events'] for r in obstacle_pairs),
            allowed_contacts_by_reason=deepcopy(self._allowed),
            ignored_same_owner_contact_samples_by_phase=dict(self._ignored_same_owner),
            intended_contact_tolerance_m=self.intended_contact_tolerance_m,
            scope_exclusions=['same-owner contacts, including robot self-contact and internal fixture mechanics'],
        )
        return result
