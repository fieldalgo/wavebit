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

Empty ``Plan.junctions`` records are omitted, including in a source's nested column
plan, so scenes without junctions retain their existing skeletons.

``Plan.shape_ranges`` records optional fixed support boxes for body sweeps.
Their exact declarations and structure indices are static; absent records are omitted
by the same rule. Extrusion orientation and prism vertex count are also static. The
current body's geometry is checked on every concrete planning call.

**One field may be hashed instead of written out.** A dataclass field declared
``metadata={'digest': True}`` encodes as the pair ``['digest', sha256]`` over its arrays
(:func:`_digest`) rather than as the value itself. Stage 4's
:class:`~wavebit.fd._kernel.stencil.StencilTerm` is what carries the mark: its integer gather
pattern and its diagonal are static -- assembled once on the host by ``fd.plan`` -- and large,
so their content has to move the skeleton while writing tens of megabytes into the hash input
to say so does not (50,911,380 bytes of active rows on the 56-cell research sphere). No
**other** field in the package declares ``metadata=``, so ``f.metadata`` is the empty mapping
on every one of them and a plan with no prepared operator never reaches the branch: every
frozen skeleton is the one it was, byte for byte.

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

**Stage 4's C3 adds one optional term and no plan field.** A structure that asked for
``fd.Structure(discretisation='stencil')`` has its constitutive term replaced, here, by a
:class:`~wavebit.fd._kernel.stencil.StencilTerm` holding the prepared operator
(``wavebit/fd/_lower/stencil.py::prepare``) -- eagerly, on the host, upstream of every trace.
The selector's value rides on that term and is **not** a ``Plan`` field: a field would encode
as ``['tuple', []]`` on every scene that does not use it and would have moved all 23 frozen
reference skeletons for no change of scene.

**Stage 2 adds the source terms** (``wavebit/fd/_lower/sources.py``) after the stage-1 ones,
in scene order, and a step count read from ``Run(periods=)``. A scene with no source adds
nothing to the plan, so its skeleton -- and every stage-0/1 golden -- is unchanged.
"""
import dataclasses
import hashlib
import json
import math

import numpy as np

from ..boundaries import FACES, Absorbing, Axis1D, Bloch, PML
from ..materials import Dispersive
from ..monitors import Record
from .._kernel.cpml import CpmlFace, StabilizedFace
from .._kernel.faces import lower_face
from .._kernel.grid import Grid
from .._kernel.masks import Region, Slab
from .._kernel.observe import ENERGY_STRIDE, InteriorEnergy, PointRecord
from .._kernel.poles import PoleTerm
from . import checks, grazing
from .cpml import DEFAULT_STABILIZE
from .monitors import monitor_regions
from .sources import run_steps, source_terms

__all__ = ['Plan', 'plan', 'skeleton_of']


def _digest(v):
    """``['digest', sha256]`` over a field's arrays: each one's dtype, shape and C-order bytes.

    A dataclass field declared ``metadata={'digest': True}`` is hashed here instead of being
    written out by :func:`_encode`. It is for a static field whose *content* has to move the
    skeleton but whose size makes writing it into the hash input absurd: a prepared material
    map's index and value arrays are tens of megabytes (measured 50,911,380 bytes of active
    rows on the 56-cell research sphere, ``docs/fd-symmetric-stencil.md``), and they are
    static -- assembled once on the host, never traced, never differentiated.

    The dtype and the shape go in beside the bytes, so that two arrays sharing a buffer but
    not a view are not read as one value, and so that ``int32`` indices and ``float64``
    values cannot collide. A digest is a function of the content alone: no ``id()``, no
    ``repr()``, no address (memory ``a-digest-you-did-not-validate``), which is what lets the
    freezers' two-freeze check still hold.

    **That claim is enforced here and not merely asserted.** ``np.ascontiguousarray`` takes
    *anything*, and what it makes of a list of Python objects -- or of one object -- is an
    ``object`` array, whose ``tobytes()`` writes the elements' **pointers**. That digest is an
    address: it moves between two processes, it moves between two runs in one process, and it
    is exactly what the docstring above promises does not happen. Only ``bool``, integer,
    unsigned, float and complex arrays are content, so every other kind is refused by name.
    That an object array's bytes are addresses is not assumed: the mutant of the commit that
    added this function -- which routed *every* field here -- made ``freeze_fd_stage0.py``
    exit 1 with "two freezes of the same code disagree on: metadata", one process, two
    freezes, because the object arrays it then hashed carried them.

    A ``str`` and a ``datetime64`` array meet the same door although both would in fact
    digest reproducibly. A field marked for a digest is an array of numbers -- a prepared
    material map's indices and values -- and admitting the kinds that merely happen to be
    safe means re-deciding that list at every addition.

    The mark belongs on a field of a dataclass the plan *holds* -- a term -- and not on a
    ``Plan`` field: :func:`_finish` hands :func:`skeleton_of` a plain dict, so a ``Plan``
    field is reached through the dict branch below, which never reads ``dataclasses.fields``.
    """
    h = hashlib.sha256()
    for a in (v if isinstance(v, (tuple, list)) else (v,)):
        a = np.ascontiguousarray(a)
        if a.dtype.kind not in 'biufc':
            raise TypeError(f"a field marked metadata={{'digest': True}} holds an array of "
                            f"dtype {a.dtype.str!r}, whose bytes are not its content: only "
                            f"bool, integer, float and complex arrays are digested here. An "
                            f"object array writes the addresses of its elements, so its digest "
                            f"would differ between two processes and between two runs of one.")
        h.update(f'{a.dtype.str}{a.shape}'.encode())
        h.update(a.tobytes())
    return ['digest', h.hexdigest()]


def _encode(v):
    """A JSON-able, type-tagged encoding of a static value; refuses anything traced."""
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return [type(v).__name__,
                [[f.name, _digest(getattr(v, f.name)) if f.metadata.get('digest')
                  else _encode(getattr(v, f.name))]
                 for f in dataclasses.fields(v)
                 if not (isinstance(v, Plan) and f.name in ('junctions', 'shape_ranges')
                         and not getattr(v, f.name))]]
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
    junctions: tuple = ()        # static flush-pair geometry and write masks
    shape_ranges: tuple = ()     # index, kind, support bounds, then any orientation/vertex count
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

    @property
    def constitutive(self):
        from .._kernel.constitutive import ConstitutiveTerm
        return tuple(t for t in self.terms if isinstance(t, ConstitutiveTerm))

    @property
    def energy(self):
        """The interior-energy observer: every cell but the PML faces', read every
        ``ENERGY_STRIDE`` steps, with the part of each dispersive structure's pole term that
        lies inside them (``wavebit/fd/_kernel/observe.py::InteriorEnergy``). An ``Absorber``'s
        cells are judged with the rest: a passive loss backed by a wall, whose energy is
        physical and returns part of what it holds.

        The cells a ``Symmetry`` plane halves come with it (:func:`_mirror_planes`), and the
        step count, which says which steps are read.

        Derived, not a field: it is a function of ``pml_cells``, ``face_rules``, ``steps`` and
        the cell counts, which the skeleton already holds, so it adds nothing to it -- and a
        field would have moved every frozen skeleton
        (``tests/references/fd_golden_stage*.npz``) for no change of scene. A source's column
        plan has one too and never steps it (the column is stepped by its term, not by
        ``step``).
        """
        cells = dict(self.pml_cells)
        n = {'x': self.nx, 'y': self.ny, 'z': self.nz}
        lo = tuple(cells.get(f'{a}-', 0) for a in 'zyx')
        hi = tuple(n[a] - cells.get(f'{a}+', 0) for a in 'zyx')
        poles = []
        for term in self.terms:
            if not isinstance(term, PoleTerm):
                continue
            spans = tuple((max(a, l), min(b, h))
                          for (a, b), l, h in zip((term.region.z, term.region.y, term.region.x),
                                                  lo, hi))
            if all(b > a for a, b in spans):
                offsets = tuple(a - r[0] for (a, _), r in
                                zip(spans, (term.region.z, term.region.y, term.region.x)))
                poles.append((term.name, term.n_poles, spans, offsets))
        return InteriorEnergy(lo=lo, hi=hi, stride=ENERGY_STRIDE, dt=self.dt,
                              ratio=self.dt / self.dx, steps=self.steps, poles=tuple(poles),
                              planes=_mirror_planes(self.face_rules, lo, hi),
                              constitutive=self.constitutive)

    def pad(self, face, c):
        """The ``Pad`` of component ``c`` of the field ``face`` pads."""
        for name, comp, pad in self.pads:
            if name == face and comp == c:
                return pad
        raise KeyError((face, c))

    def with_static(self, **changes):
        """A plan with ``changes`` and a recomputed skeleton (the test hook uses this)."""
        result = _finish(dataclasses.replace(self, **changes))
        if hasattr(self, '_shape_seeds'):
            object.__setattr__(result, '_shape_seeds', self._shape_seeds)
        return result


def _finish(p):
    static = {f.name: getattr(p, f.name) for f in dataclasses.fields(p) if f.name != 'skeleton'}
    # Omit absent additions, including in nested column plans in _encode above.
    # This preserves every existing skeleton for a scene that does not use them.
    for name in ('junctions', 'shape_ranges'):
        if not getattr(p, name):
            static.pop(name)
    return dataclasses.replace(p, skeleton=skeleton_of(**static))


def _mirror_planes(face_rules, lo, hi):
    """The judged cells a ``Symmetry`` plane halves: ``((coefficient, field, c, box), ...)``.

    A component that lies on a mirror plane is shared by the half box and its image, so the
    half box holds half of its energy: the leapfrog's identity (``InteriorEnergy``) sums over
    the full box, whose energy is twice the half box's off-plane cells plus the plane's once.
    Counted whole, the plane's term is left in -- measured: the half box's reading then swings
    with that term (``tests/test_fd_decay.py``). Which components lie on the plane is
    ``faces.py``'s table: on a low face's node plane those at a node along the axis (tangential
    E, normal H), on a high face's half plane those half a cell along it (normal E, tangential
    H); the ones of sign ``-1`` there are masked to zero and need nothing. A component on
    planes of two or three axes (an edge, a corner) weighs ``1/4`` or ``1/8``: the parts are
    every nonempty set ``S`` of its planes with coefficient ``(-1/2)^|S|``, which sums to
    ``(1/2)^m - 1`` over a cell on ``m`` of them (inclusion-exclusion).
    """
    from .._kernel.grid import OFFSETS
    on = {}
    for name, rule in face_rules:
        if rule.kind != 'symmetry':
            continue
        axis, side = 'xyz'.index(name[0]), name[1]
        zyx = 2 - axis
        for field in 'EH':
            for c in range(3):
                normal = c == axis
                sign = rule.parity * (-1 if normal == (field == 'E') else 1)
                offset = OFFSETS[field][c][axis]
                if sign < 0 or offset != (0 if side == '-' else 1):
                    continue
                layer = lo[zyx] if side == '-' else hi[zyx] - 1
                on.setdefault((field, c), []).append((zyx, layer))
    planes = []
    for (field, c), where in sorted(on.items()):
        for mask in range(1, 2 ** len(where)):
            box = list(zip(lo, hi))
            chosen = [w for i, w in enumerate(where) if mask >> i & 1]
            for zyx, layer in chosen:
                box[zyx] = (layer, layer + 1)
            planes.append(((-0.5) ** len(chosen), field, c, tuple(box)))
    return tuple(planes)


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
            rho = DEFAULT_STABILIZE if face.stabilize is None else face.stabilize
            fields = dict(name=name, axis=axis, side=name[1], cells=face.cells,
                          region=Region.face(axis, name[1], face.cells, grid.n))
            cpml.append(StabilizedFace(stabilize=float(rho), **fields) if rho > 0
                        else CpmlFace(**fields))
        else:
            absorber.append((name, face.cells))
    return (tuple(cpml), tuple(absorbing), tuple(pml), tuple(absorber), tuple(wavelengths))


def _pole_terms(scene, grid):
    """One :class:`PoleTerm` per dispersive structure, over that structure's snapped cells.

    "That structure's cells" and not "the cells it still owns": the overlap rule lives in
    ``coefficients.painters`` and this does not read it, which is why a scene where a
    dispersive structure overlaps another is refused outright
    (``checks.check_overlaps``, which measures what the wrong answer looks like).
    """
    terms = []
    for index, structure in enumerate(getattr(scene, 'structures', ()) or ()):
        if not isinstance(structure.material, Dispersive):
            continue
        # The support and not the filled cells: the region is where the pole's own state
        # array lives and where ``drive`` is applied, so it has to hold every cell the shape
        # reaches. What fraction of each of those cells the shape actually fills is the
        # coefficients' business (stage 4, C2), not the region's.
        box = structure.shape.support(grid)
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


def plan(scene, _column=False, _grazing=None, _reuse=None, _shape_trace=False):
    """The static plan of ``scene``, with the stage-1 refusals run and its warnings emitted.

    ``_column`` is the stage-2 source columns' own door (``sources._column_plan``): a column is
    one cell across and Bloch by construction, which is not a user face, so the one-cell Bloch
    refusal and the grazing rule are the user's scene's alone. ``_grazing`` is ``fd.run``'s
    own: a list the grazing rule's survey rows are appended to (as one list), so the run's
    diagnostics report the survey its refusal judged instead of surveying again. They are not
    a plan field: the survey reads the scene's materials, which the skeleton does not hold,
    and a plan field would have to stay out of the skeleton, the hash and the equality.

    ``_reuse`` is ``fd.run``'s too: the plan the caller passed, whose prepared extended-stencil
    operator :func:`~wavebit.fd._lower.stencil.prepare` takes as it is when it was prepared
    from the same arguments, instead of solving it again (every door still runs).
    """
    from .shape_ranges import traced, remember, name as traced_name
    if traced(scene):
        raise ValueError(f'fd.plan requires concrete {traced_name(scene)} geometry; build the plan '
                         'before differentiation and pass plan= to fd.run or fd.lower.')
    grid = Grid.from_domain(scene.domain.size, scene.domain.center, scene.resolution.dx)
    dt = scene.run.courant * grid.dx / math.sqrt(3.0)
    steps = run_steps(scene, dt)
    if not _column:
        checks.check_one_cell_bloch(scene, grid)
        checks.check_wavelength_resolution(scene, grid)
    checks.check_faces(scene, grid, None)
    checks.check_overlaps(scene, grid)
    junctions = checks.check_smoothed(scene, grid)
    checks.check_symmetry(scene, grid)
    checks.check_seams(scene, grid)
    checks.check_cells_across(scene, grid)
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
                  for row in s.shape.span_rows(grid))
    from .shape_ranges import records as shape_ranges
    p = Plan(nx=grid.n[0], ny=grid.n[1], nz=grid.n[2], dx=grid.dx, dt=dt,
             steps=steps, courant=scene.run.courant, low=grid.low,
             face_rules=tuple(rules), pads=tuple(pads), masks=tuple(masks),
             terms=_pole_terms(scene, grid),
             observers=tuple(observers), dtype='complex128' if complex_run else 'float64',
             has_E0=initial is not None and initial.E is not None,
             has_H0=initial is not None and initial.H is not None,
             cpml=cpml, absorbing_faces=absorbing, pml_cells=pml_cells,
             absorber_cells=absorber_cells, pml_wavelengths=wavelengths, spans=spans,
             junctions=junctions, shape_ranges=shape_ranges(scene))
    p = _finish(p)
    checks.check_absorbing_layers(scene, p)
    checks.check_stability(scene, p)
    from . import constitutive
    coupled = constitutive.terms(scene, p)
    if coupled:
        p = p.with_static(terms=p.terms + coupled)
    # Stage 4 C3: a structure that asked for ``discretisation='stencil'`` has its constitutive
    # term replaced by a prepared one, here, eagerly, on the host -- before ``validate`` (which
    # certifies the diagonal rule's cross coefficients and has nothing to say about a prepared
    # operator, which carries its own certificate) and long before any trace. It runs whether
    # or not there are coupled terms, because a *cell-snapped* structure that asked for the
    # stencil makes none and has to hear a refusal rather than be waved through. A scene where
    # nothing asks gets this plan object back unchanged and identical, which is what keeps
    # every stage-0..3 skeleton byte-identical.
    from . import stencil
    p = stencil.prepare(scene, p, reuse=_reuse)
    lifted = constitutive.validate(scene, p) if coupled else None
    sources = source_terms(scene, p)
    if sources:
        p = p.with_static(terms=tuple(p.terms) + sources)
    if not _column:
        rows = grazing.enforce(scene, p)
        if _grazing is not None and rows is not None:
            _grazing.append(rows)
    # Emitted through the caller's own filters, and deliberately not forced past them: a
    # caller who silenced CoarseMeshWarning silenced it, and the default "once per location"
    # rule is what keeps a sweep of a thousand plans from printing a thousand copies.
    if not _shape_trace:
        checks.warn_all(scene, p, lifted)
    if lifted:
        # Not a field: the skeleton and equality never see it, and a replaced plan drops it.
        # checks._lift_notes reads it for this scene only (identity), so fd.run's report after
        # the run does not lower the scene a third time.
        object.__setattr__(p, '_lifted', (scene, lifted))
    remember(scene, p)
    return p
