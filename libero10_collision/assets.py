"""Explicit obstacle variants; stock LIBERO assets stay unchanged.

Uniform scaling preserves the original implementation. Selected rigid assets
also support source-local axis scaling: textured meshes keep their materials,
while rotated collision boxes become their exact affine convex meshes. Density
is unchanged and MuJoCo computes mass and inertia from the scaled geometry.
Only the added obstacle uses a distinct registered category.
"""
from __future__ import annotations
import itertools
import math
import xml.etree.ElementTree as ET
import numpy as np

_REGISTERED = {}


def _numbers(element, key, factor, default=None):
    value=element.get(key, default)
    if value is not None:
        values=np.fromstring(value,sep=' ')
        if not len(values) or not np.isfinite(values).all(): raise ValueError(f'Invalid {key}: {value}')
        element.set(key,' '.join(format(float(x*factor),'.17g') for x in values))


def scale_object(obj, scale):
    """Scale both XML source tree and detached object subtree exactly once."""
    seen=set()
    for root in (obj.root,obj.get_obj()):
        for element in root.iter():
            if id(element) in seen: continue
            seen.add(id(element))
            tag=element.tag
            if tag in ('body','geom','site','joint','inertial','camera'):
                _numbers(element,'pos',scale)
            if tag in ('geom','site'):
                _numbers(element,'size',scale)
                _numbers(element,'fromto',scale)
            if tag=='mesh':
                _numbers(element,'scale',scale,default='1 1 1')
                if element.get('refpos') is not None:
                    raise ValueError('Mesh refpos scaling is not supported by these benchmark assets')
            if tag in ('geom','inertial'):
                _numbers(element,'mass',scale**3)
            if tag=='inertial':
                _numbers(element,'diaginertia',scale**5)
                _numbers(element,'fullinertia',scale**5)
    return obj



_ORIENTATION_KEYS = ('quat', 'axisangle', 'euler', 'xyaxes', 'zaxis')
_AFFINE_SOURCES = {'red_coffee_mug', 'alphabet_soup'}
_BOX_CORNERS = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
# Outward-oriented triangles for the corner ordering above. Faces make volume
# and inertia computation explicit, instead of depending on inferred topology.
_BOX_FACES = ((0,1,3),(0,3,2),(4,6,7),(4,7,5),
              (0,4,5),(0,5,1),(2,3,7),(2,7,6),
              (0,2,6),(0,6,4),(1,5,7),(1,7,3))


def _vector(element, key, length, default):
    raw=element.get(key, default)
    values=np.fromstring(raw, sep=' ')
    if values.shape!=(length,) or not np.isfinite(values).all():
        raise ValueError(f'Invalid {element.tag} {key}: {raw}')
    return values


def _sphere_radius(element):
    values=np.fromstring(element.get('size','0.005'),sep=' ')
    if len(values) not in (1,3) or not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError('Invalid sphere marker size')
    return values[0]


def _text(values):
    return ' '.join(format(float(x), '.17g') for x in np.asarray(values).reshape(-1))


def _rotation(element):
    if any(element.get(k) is not None for k in _ORIENTATION_KEYS[1:]):
        raise ValueError('Axis scaling supports quaternion orientation only')
    q=_vector(element, 'quat', 4, '1 0 0 0')
    norm=float(np.linalg.norm(q))
    if norm<=1e-12: raise ValueError('Zero quaternion in scaled asset')
    w,x,y,z=q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def scale_object_axes(obj, local_scale):
    """Apply diag(local_scale) exactly in a single rigid object's local frame.

    Restricted deliberately to the two reviewed assets: untransformed body
    hierarchy, box collision decomposition and identity-oriented visual meshes.
    Scaling a rotated box's half-sizes alone would be wrong; D R differs from
    R D. Instead its eight vertices become D R v, retaining a convex collision
    solid with the exact affine volume and no enclosing-box approximation.
    """
    scale=np.asarray(local_scale,dtype=float)
    if scale.shape!=(3,) or not np.isfinite(scale).all() or np.any((scale<.5)|(scale>3.5)):
        raise ValueError('Invalid source-local obstacle scale')
    elements=[];seen=set()
    for root in (obj.root,obj.get_obj()):
        for element in root.iter():
            if id(element) not in seen:
                seen.add(id(element));elements.append(element)
    # Fail before mutation if this asset contains transformations that would
    # require general affine body hierarchy or explicit-inertia handling.
    for element in elements:
        tag=element.tag
        if tag in ('inertial','camera','default') or element.get('class') is not None:
            raise ValueError(f'Unsupported axis-scaled asset element: {tag}')
        if tag=='body':
            if not np.allclose(_vector(element,'pos',3,'0 0 0'),0,atol=1e-14) or not np.allclose(_rotation(element),np.eye(3),atol=1e-14):
                raise ValueError('Axis scaling requires identity body transforms')
        if tag=='joint':
            if element.get('type')!='free' or not np.allclose(_vector(element,'pos',3,'0 0 0'),0,atol=1e-14):
                raise ValueError('Axis scaling requires a free joint at the body origin')
        if tag=='mesh':
            if element.get('refpos') is not None or element.get('refquat') is not None:
                raise ValueError('Axis scaling does not support mesh reference transforms')
            _vector(element,'scale',3,'1 1 1')
        if tag=='geom':
            if element.get('fromto') is not None:
                raise ValueError('Axis scaling does not support geom fromto')
            kind=element.get('type')
            if kind=='box':
                size=_vector(element,'size',3,'')
                if np.any(size<=0): raise ValueError('Invalid collision box size')
                _rotation(element)
            elif kind=='mesh':
                if not np.allclose(_rotation(element),np.eye(3),atol=1e-14):
                    raise ValueError('Axis scaling requires identity-oriented visual meshes')
                if element.get('contype','1')!='0' or element.get('conaffinity','1')!='0':
                    raise ValueError('Axis scaling only accepts box collision decomposition')
            else: raise ValueError(f'Unsupported axis-scaled geom type: {kind}')
            _vector(element,'pos',3,'0 0 0')
        if tag=='site':
            if element.get('type','sphere')!='sphere' or not np.allclose(_rotation(element),np.eye(3),atol=1e-14):
                raise ValueError('Axis scaling only supports unrotated sphere marker sites')
            _sphere_radius(element)
    volume_scale=float(np.prod(scale))
    meshes={}
    for element in elements:
        tag=element.tag
        if tag in ('body','geom','site','joint') and element.get('pos') is not None:
            element.set('pos',_text(_vector(element,'pos',3,'0 0 0')*scale))
        if tag=='mesh':
            element.set('scale',_text(_vector(element,'scale',3,'1 1 1')*scale))
        if tag=='site':
            radius=_sphere_radius(element)
            element.set('type','ellipsoid');element.set('size',_text(radius*scale))
        if tag=='geom':
            _numbers(element,'mass',volume_scale)
            if element.get('type')=='box':
                vertices=(_BOX_CORNERS*_vector(element,'size',3,''))@_rotation(element).T
                vertices*=scale
                key=vertices.tobytes()
                if key not in meshes:
                    name=f'{obj.naming_prefix}axis_scaled_box_{len(meshes):03d}'
                    mesh=ET.SubElement(obj.asset,'mesh',name=name,vertex=_text(vertices),
                                       face=' '.join(str(i) for face in _BOX_FACES for i in face))
                    meshes[key]=name
                element.set('type','mesh');element.set('mesh',meshes[key])
                element.attrib.pop('size',None)
                element.attrib.pop('quat',None)
    return obj


def register_scaled_asset(spec):
    from libero.libero.envs.base_object import OBJECTS_DICT
    source=spec['source_asset']; category=spec['category']
    if not category.startswith('collision_') or category==source:
        raise ValueError('Scaled obstacle must have its own collision_ category')
    if ('uniform_scale' in spec)==('local_scale' in spec):
        raise ValueError('Specify exactly one uniform_scale or local_scale')
    axis_scaled='local_scale' in spec
    if axis_scaled:
        if source not in _AFFINE_SOURCES:
            raise ValueError(f'Source-local scaling is not reviewed for {source}')
        scale=np.asarray(spec['local_scale'],dtype=float)
        if scale.shape!=(3,) or not np.isfinite(scale).all() or np.any((scale<.5)|(scale>3.5)):
            raise ValueError('Invalid source-local obstacle scale')
        signature=(source,'local',*map(float,scale))
    else:
        scale=float(spec['uniform_scale'])
        if not math.isfinite(scale) or not .5<=scale<=3.5:
            raise ValueError('Invalid obstacle scale')
        signature=(source,scale)
    if category in _REGISTERED:
        if _REGISTERED[category]!=signature: raise ValueError(f'Conflicting asset definition: {category}')
        return
    if category in OBJECTS_DICT: raise ValueError(f'Category already registered outside benchmark: {category}')
    base=OBJECTS_DICT[source]
    class ScaledObstacle(base):
        def __init__(self,name,**kwargs):
            super().__init__(name=name,**kwargs)
            if axis_scaled: scale_object_axes(self,scale)
            else: scale_object(self,scale)
            self.category_name=category
    OBJECTS_DICT[category]=ScaledObstacle
    _REGISTERED[category]=signature
