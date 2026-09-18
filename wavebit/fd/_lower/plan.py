"""``plan(scene) -> Plan``: everything static about a run, hashable, with its skeleton hash.

The plan is the scene with every derived quantity computed once: the cell counts and the
low corner, ``dt``, the six face rules with their pads and masks, the observers as index
tuples, the dtype, and the term list (empty in stage 0; the test hook adds to it). It holds
no array and no traced value -- those are ``Params`` -- which is what makes it hashable,
comparable, and safe to close over in a jitted body: a JAX tracer cannot be branched on,
and every branch the kernel takes is decided here.

**The skeleton** is a SHA-256 over the plan's static content: the types and values of every
field, recursively, with ``None``-versus-present encoded and floats written in hex so that
two processes agree on it. It never reads an array or an ``id()``. ``fd.run(scene,
plan=other)`` refuses a plan whose skeleton differs from ``plan(scene).skeleton``: a plan
prepared for one scene must not silently drive another.

**Dtype.** Real (``float64``) unless any face is Bloch; a Bloch phase that is concretely 0
was refused at ``Bloch(...)``, so the dtype rule is simply "any Bloch face means complex".
"""
import dataclasses
import hashlib
import json
import math

from ..boundaries import FACES, Bloch
from .._kernel.faces import lower_face
from .._kernel.grid import Grid
from .._kernel.observe import PointRecord

__all__ = ['Plan', 'plan', 'skeleton_of']


def _encode(v):
    """A JSON-able, type-tagged encoding of a static value; refuses anything traced."""
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return [type(v).__name__, [[f.name, _encode(getattr(v, f.name))]
                                   for f in dataclasses.fields(v)]]
    if isinstance(v, (tuple, list)):
        return ['tuple', [_encode(x) for x in v]]
    if isinstance(v, dict):
        return ['dict', [[str(k), _encode(x)] for k, x in v.items()]]
    if v is None:
        return ['none']
    if isinstance(v, bool):
        return ['bool', v]
    if isinstance(v, int):
        return ['int', v]
    if isinstance(v, float):
        return ['float', v.hex()]
    if isinstance(v, str):
        return ['str', v]
    raise TypeError(f'{type(v).__name__} has no place in a static plan, which holds Python '
                    f'bool, int, float, str, None, tuples and dataclasses of those; a traced '
                    f'value belongs in Params, and a numpy scalar is normalised at its door.')


def skeleton_of(**static):
    return hashlib.sha256(json.dumps(_encode(static), sort_keys=False).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Plan:
    """The static plan; see the module docstring. ``faces[name]`` gives that face's rule."""

    nx: int
    ny: int
    nz: int
    dx: float
    dt: float
    steps: int
    courant: float
    low: tuple
    face_rules: tuple            # ((name, FaceRule), ...) for the six faces
    pads: tuple                  # ((face name, component, Pad), ...) of the padded field
    masks: tuple
    terms: tuple
    observers: tuple
    dtype: str
    has_E0: bool
    has_H0: bool
    schedule: object = None      # checkpoint block size; None is a plain scan
    skeleton: str = ''

    @property
    def faces(self):
        return dict(self.face_rules)

    @property
    def grid(self):
        return Grid(n=(self.nx, self.ny, self.nz), dx=self.dx, low=self.low)

    @property
    def shape(self):
        return (3, self.nz, self.ny, self.nx)

    @property
    def cells(self):
        return self.nx * self.ny * self.nz

    def pad(self, face, c):
        """The ``Pad`` of component ``c`` of the field ``face`` pads."""
        for name, comp, pad in self.pads:
            if name == face and comp == c:
                return pad
        raise KeyError((face, c))

    def with_static(self, **changes):
        """A plan with ``changes`` and a recomputed skeleton (the test hook uses this)."""
        return _finish(dataclasses.replace(self, **changes))


def _finish(p):
    static = {f.name: getattr(p, f.name) for f in dataclasses.fields(p) if f.name != 'skeleton'}
    return dataclasses.replace(p, skeleton=skeleton_of(**static))


def plan(scene):
    """The static plan of ``scene``."""
    grid = Grid.from_domain(scene.domain.size, scene.domain.center, scene.resolution.dx)
    dt = scene.run.courant * grid.dx / math.sqrt(3.0)
    rules, pads, masks = [], [], []
    for name in FACES:
        rule, face_pads, face_masks = lower_face(name, scene.boundaries.face(name), grid)
        rules.append((name, rule))
        pads.extend((name, c, pad) for c, pad in enumerate(face_pads))
        masks.extend(face_masks)
    complex_run = any(isinstance(scene.boundaries.face(n), Bloch) for n in FACES)
    observers = []
    for m in scene.monitors:
        k, j, i = grid.nearest(m.field, m.index, m.at)
        observers.append(PointRecord(name=m.name, field=m.field, c=m.index, k=k, j=j, i=i))
    initial = scene.initial
    p = Plan(nx=grid.n[0], ny=grid.n[1], nz=grid.n[2], dx=grid.dx, dt=dt,
             steps=scene.run.steps, courant=scene.run.courant, low=grid.low,
             face_rules=tuple(rules), pads=tuple(pads), masks=tuple(masks), terms=(),
             observers=tuple(observers), dtype='complex128' if complex_run else 'float64',
             has_E0=initial is not None and initial.E is not None,
             has_H0=initial is not None and initial.H is not None)
    return _finish(p)
