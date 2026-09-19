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

**Dtype.** Real (``float64``) unless any face is Bloch or (stage 3) ``Axis1D``; a Bloch phase
that is concretely 0 was refused at ``Bloch(...)``, so the dtype rule is simply "any Bloch or
Axis1D face means complex" (an Axis1D face is complex even at ``k = 0``).

**Stage 1 adds four static things** and no new kind of thing: the CPML faces with their slab
geometry (``cpml``), the absorbing faces and their depths (``absorbing_faces``, split into
``pml_cells`` and ``absorber_cells``), the snapped structure spans (``spans``), and one
:class:`~wavebit.fd._kernel.poles.PoleTerm` per dispersive structure in ``terms``. All of it
is integers, floats and strings, so the skeleton still hashes the whole plan and still reads
no array. The materials' *numbers* are not here -- they may be traced, and they are leaves.

The stage-1 refusals run here, in ``wavebit/fd/_lower/checks.py``, because this is the first place that
sees the scene and the grid at once: a layer that does not fit its axis, a structure that
ends inside a PML, a mesh too coarse for its own material, a pole past the CFL bound. The
warnings are emitted here too and listed again by ``check``.

**Stage 3 adds the frequency-domain observers** (``wavebit/fd/_lower/monitors.py``): each
``Flux``, ``Field`` and ``Diffraction`` monitor becomes one or more ``DFTRegion`` observers in
``observers``, after the ``Record`` ones -- static regions and frequencies, no new plan field --
so a scene without them has the stage-2 plan and skeleton exactly. Two stage-3 doors run
here too, on the user's scene and never on a source's internal column: a ``Bloch`` phase on an
axis one cell long is refused naming ``Axis1D`` (``checks.check_one_cell_bloch``), and the
grazing-order rule (``grazing.enforce``) refuses an order a PML face cannot absorb to the
measured tolerance.

**Stage 2 adds the source terms** (``wavebit/fd/_lower/sources.py``) after the stage-1 ones,
in scene order, and a step count read from ``Run(periods=)``. A scene with no source adds
nothing to the plan, so its skeleton -- and every stage-0/1 golden -- is unchanged.
"""
import dataclasses
import hashlib
import json
import math

from ..boundaries import FACES, Absorbing, Axis1D, Bloch, PML
from ..geometry import cell_box, span_rows
from ..materials import Dispersive
from ..monitors import Record
from .._kernel.cpml import CpmlFace
from .._kernel.faces import lower_face
from .._kernel.grid import Grid
from .._kernel.masks import Region, Slab
from .._kernel.observe import PointRecord
from .._kernel.poles import PoleTerm
from . import checks, grazing
from .monitors import monitor_regions
from .sources import run_steps, source_terms

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
    if isinstance(v, complex):
        # A PlaneWave's drive weights carry their stagger phase (_lower/sources.py).
        return ['complex', [v.real.hex(), v.imag.hex()]]
    if isinstance(v, str):
        return ['str', v]
    raise TypeError(f'{type(v).__name__} has no place in a static plan, which holds Python '
                    f'bool, int, float, complex, str, None, tuples and dataclasses of those; a '
                    f'traced value belongs in Params, and a numpy scalar is normalised at its door.')


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
    cpml: tuple = ()             # (CpmlFace, ...) -- one per PML face
    absorbing_faces: tuple = ()  # ((face name, cells), ...) for PML and Absorber alike
    pml_cells: tuple = ()        # ((face name, cells), ...) for PML faces only
    absorber_cells: tuple = ()   # ((face name, cells), ...) for Absorber faces only
    pml_wavelengths: tuple = ()  # ((face name, wavelength), ...)
    spans: tuple = ()            # ((axis, asked lo, asked hi, snapped lo, snapped hi), ...)
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


def _absorbing(scene, grid):
    """The four absorbing tuples of the plan, and the CPML faces, from the scene's boundaries."""
    cpml, absorbing, pml, absorber, wavelengths = [], [], [], [], []
    for name in FACES:
        face = scene.boundaries.face(name)
        if not isinstance(face, Absorbing):
            continue
        axis = 'xyz'.index(name[0])
        absorbing.append((name, face.cells))
        if isinstance(face, PML):
            pml.append((name, face.cells))
            wavelengths.append((name, checks.face_wavelength(face, scene, name)))
            cpml.append(CpmlFace(name=name, axis=axis, side=name[1], cells=face.cells,
                                 region=Region.face(axis, name[1], face.cells, grid.n)))
        else:
            absorber.append((name, face.cells))
    return (tuple(cpml), tuple(absorbing), tuple(pml), tuple(absorber), tuple(wavelengths))


def _pole_terms(scene, grid):
    """One :class:`PoleTerm` per dispersive structure, over that structure's snapped cells.

    "That structure's cells" and not "the cells it still owns": the overlap rule lives in
    ``coefficients.owner_map`` and this does not read it, which is why a scene where a
    dispersive structure overlaps another is refused outright
    (``checks.check_overlaps``, which measures what the wrong answer looks like).
    """
    terms = []
    for index, structure in enumerate(getattr(scene, 'structures', ()) or ()):
        if not isinstance(structure.material, Dispersive):
            continue
        box = cell_box(structure.shape, grid)
        region = Region(z=tuple(box[2]), y=tuple(box[1]), x=tuple(box[0]))
        slabs = tuple(Slab(c=c, z=region.z, y=region.y, x=region.x) for c in range(3))
        terms.append(PoleTerm(name=f'poles[{index}]', region=region, slabs=slabs,
                              n_poles=len(structure.material.poles)))
    if getattr(scene, 'background', None) is not None \
            and isinstance(scene.background, Dispersive):
        raise NotImplementedError(
            'Scene(background=fd.Dispersive(...)): a dispersive background fills the whole '
            'box, including every absorbing layer, and a PML cannot be matched to it (see '
            'fd.Absorber). Stage 1 takes a dispersive material as a Structure; a dispersive '
            'background is not refused on principle, it simply has no gate behind it yet.')
    return tuple(terms)


def plan(scene, _column=False):
    """The static plan of ``scene``, with the stage-1 refusals run and its warnings emitted.

    ``_column`` is the stage-2 source columns' own door (``sources._column_plan``): a column is
    one cell across and Bloch by construction, which is not a user face, so the one-cell Bloch
    refusal and the grazing rule are the user's scene's alone.
    """
    grid = Grid.from_domain(scene.domain.size, scene.domain.center, scene.resolution.dx)
    dt = scene.run.courant * grid.dx / math.sqrt(3.0)
    steps = run_steps(scene, dt)
    if not _column:
        checks.check_one_cell_bloch(scene, grid)
    checks.check_faces(scene, grid, None)
    checks.check_overlaps(scene, grid)
    rules, pads, masks = [], [], []
    for name in FACES:
        rule, face_pads, face_masks = lower_face(name, scene.boundaries.face(name), grid)
        rules.append((name, rule))
        pads.extend((name, c, pad) for c, pad in enumerate(face_pads))
        masks.extend(face_masks)
    complex_run = any(isinstance(scene.boundaries.face(n), (Bloch, Axis1D)) for n in FACES)
    observers = []
    for m in scene.monitors:
        if not isinstance(m, Record):
            continue
        k, j, i = grid.nearest(m.field, m.index, m.at)
        observers.append(PointRecord(name=m.name, field=m.field, c=m.index, k=k, j=j, i=i))
    # Stage 3: every frequency-domain monitor as DFTRegion observers, after the records -- a
    # scene with none adds nothing, so its observers and its skeleton are stage 2's.
    observers.extend(monitor_regions(scene, grid, dt))
    initial = scene.initial
    cpml, absorbing, pml_cells, absorber_cells, wavelengths = _absorbing(scene, grid)
    spans = tuple(row for s in (getattr(scene, 'structures', ()) or ())
                  for row in span_rows(s.shape, grid))
    p = Plan(nx=grid.n[0], ny=grid.n[1], nz=grid.n[2], dx=grid.dx, dt=dt,
             steps=steps, courant=scene.run.courant, low=grid.low,
             face_rules=tuple(rules), pads=tuple(pads), masks=tuple(masks),
             terms=_pole_terms(scene, grid),
             observers=tuple(observers), dtype='complex128' if complex_run else 'float64',
             has_E0=initial is not None and initial.E is not None,
             has_H0=initial is not None and initial.H is not None,
             cpml=cpml, absorbing_faces=absorbing, pml_cells=pml_cells,
             absorber_cells=absorber_cells, pml_wavelengths=wavelengths, spans=spans)
    p = _finish(p)
    checks.check_absorbing_layers(scene, p)
    checks.check_stability(scene, p)
    sources = source_terms(scene, p)
    if sources:
        p = p.with_static(terms=tuple(p.terms) + sources)
    if not _column:
        grazing.enforce(scene, p)
    # Emitted through the caller's own filters, and deliberately not forced past them: a
    # caller who silenced CoarseMeshWarning silenced it, and the default "once per location"
    # rule is what keeps a sweep of a thousand plans from printing a thousand copies.
    checks.warn_all(scene, p)
    return p
