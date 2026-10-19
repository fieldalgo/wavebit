"""The update coefficients: ``ae``, ``be``, ``ah``, ``bh`` -- scalars in vacuum, arrays when not.

The step is written once against these four numbers (``src/wavebit/fdtd/_kernel/step.py``), so a material
changes them and not the code::

    ae = (1 - dt*rate_e/2) / (1 + dt*rate_e/2)      rate_e = sigma/eps + damping_e
    be = (dt/dx) / eps / (1 + dt*rate_e/2)
    ah = (1 - dt*rate_h/2) / (1 + dt*rate_h/2)      rate_h = damping_h  (Absorber only)
    bh = (dt/dx) / mu  / (1 + dt*rate_h/2)

``dt/dx`` rather than ``dt`` because the curl is the bare difference and carries no ``1/dx``
(``src/wavebit/fdtd/_kernel/curl.py``); a magnetic loss is not a material here and exists only inside an
``Absorber``, where it is the matched half of the graded damping.

**An array is built only where the value is not uniform**, and only along the axes it
actually varies on. Every stage-1 non-uniformity is axis-aligned -- a ``Slab`` varies along
one axis, an ``Absorber`` along one axis, and stage 3's cell-snapped ``Region`` along each axis
it does not span -- so the coefficients are built at shape
``(3, nz|1, ny|1, nx|1)`` with a full axis only where something varies along it, and
broadcasting does the rest. A vacuum run keeps the stage-0 Python scalars, which is what
makes its buffer count, its transient allocation and its golden all still hold; a scene whose
only structure is a z slab in a 128^3 box pays ``3 * 128 * 8`` bytes per coefficient and not
``3 * 128^3 * 8``. ``check`` prints the total beside the state's.

Material values may be **traced** -- a gradient with respect to a permittivity, a
conductivity or a pole strength is the point of stage 1's AD gate -- so the combination is
done with ``rt.xp.where`` over static boolean masks rather than with NumPy indexing, and the
plan's concrete checks (the CFL bound, the cells per wavelength) ask ``materials.concrete``
first and say in the report when the answer was ``None``.

Structures are applied **in the order given**, so a later one wins where two overlap.
:func:`painters` and :func:`_blend` are the one place that rule is written, and an overlap
that a *dispersive* structure is party to is refused at the door instead, because the pole
term is not routed through them (``src/wavebit/fdtd/_prepare/checks.py::check_overlaps``).

**Two discretisations meet in that loop, and only one of them averages.** A cell-snapped
shape (``Slab``, ``Region``) *replaces* the cell's material with its own -- one ``where`` over a
static boolean mask, which is what every stage-0..3 golden holds to the bit. A smoothed shape
(``Sphere``) *mixes*: it is handed the fraction of each component's own voxel it fills and how
the surface is oriented there (``src/wavebit/fdtd/_prepare/occupancy.py``) and combines its material
with whatever the cell already held, by :func:`mix`. "Whatever the cell already held" is the
running value, background or an earlier structure.

**How far painting order carries across that seam.** Exactly as far as "one voxel, one
surface". Where the arriving smoothed shape fills the voxel *whole* the mix returns its own
material and the order rule is untouched, so a ball on a cell-snapped substrate, or a core
inside a shell, composes to 2.0e-16 against the hand-written two-phase mix. Where **two**
smoothed surfaces pass through one voxel it does not compose at all -- the second mix reads
the first mixture as a medium and puts a second interface across it, leaving the background
with weight ``(1-f1)(1-f2)`` in a voxel that holds none of it, measured 73% high on two half
spaces of the same material and up to 122% on one ``fd.Sphere`` declared twice. That is a
refusal (``checks.check_smoothed``, door 5), except for a validated flush pair stored in
``Plan.junctions``. Such a pair captures its ground before either member is painted, then
replaces only its recorded write set with the separate split or staircase rule.

The two are not one branch of the other, and the reason is bits rather than taste. A
cell-snapped shape *is* the mixing rule at ``f = 1``, algebraically -- but ``mix`` returns
``1 / (w/eps_a + (1-w)/eps_a)``, and a double reciprocal is not the identity in binary64.
Measured over ``w`` on a 1001-point grid: 0 ulp at ``eps`` = 1, 2.25, 4 and 16, **2 ulp** at
11.68 and at 12 (worst at ``w = 0.193``), and 2 ulp worst over ``eps`` swept through [1, 16].
``be = (dt/dx)/eps`` would carry that straight into four frozen archives for no change of
physics. So the snapped path is written as the ``where`` it has always been, and :func:`mix`
is only ever reached by a shape that really has a fractional edge.

The ``eps`` these four are built from is ``material.eps_for_update(dt)`` and not
``material.eps``: a :class:`~wavebit.fdtd._legacy.Debye` pole's trapezoidal recurrence is implicit in
``E``, and the part of it that is simultaneous with the field being solved for is a constant
added to the permittivity. Every pole that is not Debye adds nothing and the two readings are
the same object, so a vacuum, Drude or Lorentz scene's coefficients are unchanged to the bit.
Nothing that reports *physics* reads ``eps_for_update`` -- the CFL bound, the cells per
wavelength and ``eps_at`` all read ``eps``.
"""
import numpy as np

from wavebit.fdtd._kernel.grid import OFFSETS
from wavebit.materials._time_domain import as_material
from wavebit.fdtd._prepare.cpml import damping_profile
from wavebit.fdtd._prepare.occupancy import occupancy
from wavebit.fdtd._public_names import public_description

__all__ = ['build', 'varying_axes', 'painters', 'mix', 'materials_of', 'params_bytes']

#: Position of each spatial axis (0 = x, 1 = y, 2 = z) in a ``(3, nz, ny, nx)`` array.
ARRAY_AXIS = {0: 3, 1: 2, 2: 1}


def materials_of(scene):
    """``(background, [(structure, material), ...])`` with every shorthand already coerced."""
    background = as_material(getattr(scene, 'background', None), 'Scene(background=)')
    return background, tuple(getattr(scene, 'structures', ()) or ())


def varying_axes(scene, plan):
    """The spatial axes (0 = x, 1 = y, 2 = z) along which a coefficient is not constant."""
    axes = set()
    _, structures = materials_of(scene)
    for s in structures:
        axes.update(s.shape.bounded_axes(plan.grid))
    for name, cells in plan.absorber_cells:
        axes.add('xyz'.index(name[0]))
    return axes


def _shape(plan, axes):
    return (3,
            plan.nz if 2 in axes else 1,
            plan.ny if 1 in axes else 1,
            plan.nx if 0 in axes else 1)


def mix(xp, held, value, fill, w):
    """The two-phase effective permittivity of one component's voxel -- the smoothing rule.

    ``held`` is what the cell already holds (``eps_b``: the background, or an earlier
    structure), ``value`` is the arriving shape's (``eps_a``), ``fill`` is the fraction of
    *this component's own* voxel the shape fills and ``w`` is the mean of ``n_c^2`` over the
    interface inside it, ``c`` being that component's direction::

        1 / eps_c = w (f/eps_a + (1-f)/eps_b) + (1 - w) / (f eps_a + (1-f) eps_b)

    and the same with ``mu`` at the ``H`` points. This is ``diag(eps~^-1)`` of the
    Farjadpour/Kottke effective tensor ``eps~^-1 = P <eps^-1> + (I - P) <eps>^-1`` with
    ``P = <n n^T>``: harmonic in the direction across the interface, arithmetic along it, and
    the two blended by how much of the component's own direction is normal.

    Farjadpour et al., Opt. Lett. **31**, 2972 (2006); Kottke, Farjadpour & Johnson, Phys.
    Rev. E **77**, 036611 (2008); Oskooi et al., Comput. Phys. Commun. **181**, 687 (2010).
    The tensor form is quoted from Meep's **documentation**; no Meep source is read at any
    point, and the Meep-derived FDTDX projection file this repository has already excluded
    (``docs/validation/fdtd/fdtd-layer-smoothing.md``:74, ``docs/validation/fdtd/fdtd-layer-gradients.md``:72) stays excluded.

    **What this rule is, and therefore what it is not.** *Two* phases, *one* interface, both
    *isotropic*. All three neighbouring cases are refused rather than approximated, each by
    name and each with the number the scene reads with its door held open
    (``checks.check_smoothed``): two smoothed surfaces in one voxel, 73% on two half spaces of
    the same material and up to 122% on one ``fd.Sphere`` declared twice (door 5); an
    anisotropic ``eps`` or ``mu``, for which the Kottke construction is the tau-transform and
    not a per-component blend (door 4); and a conductivity or a pole, which C8 owns (door 1).
    A smoothed shape arriving over a cell-snapped one is one of them only where a snapped face
    cuts the same voxel (door 6, C4: the snapped cells are all or nothing where the geometry
    splits them); a voxel clear of the snapped faces is a whole voxel of one medium to the
    painter, the mix has its two phases and is exact, measured at 2.0e-16. And the rule is
    gated where it has a closed form to meet: the Airy slab spelled as a smoothed ``fd.Box``
    (G5, ``tests/fdtd/test_fd_airy.py``) runs the profile it lowers to within 4.3e-7, needs no
    reference-plane correction against the continuum formula, and converges at second
    order.

    Three properties decide this form over the alternatives, and each is a gate:

    * **It contains the axis-aligned rule exactly.** ``w = 1`` is harmonic, ``w = 0``
      arithmetic, which is what ``src/wavebit/fdtd/_periodic/legacy.py``:375 ``component_property`` already
      does for a flat interface, so stage 4 does not compete with the legacy rule -- it
      generalises it. Both ends are pinned by
      ``tests/fdtd/test_fd_occupancy.py::test_the_rule_is_the_axis_aligned_rule_at_the_two_ends``.
    * **The inverse's diagonal, not the diagonal's inverse.** ``be = (dt/dx)/eps`` stores
      inverse permittivity. The tilted-interface study measures the retained diagonal
      directly; it does not justify interchanging inversion and taking a diagonal.
    * **It is continuous in the normal**, so it differentiates and has no branch to get
      wrong. The "harmonic along the component's own axis" spelling has to *pick* an axis and
      so is discontinuous at 45 degrees; ``src/wavebit/fdtd/_periodic/scene.py``:1657-1661 records the
      patch that cost -- a fixed x-then-y order broke x/y covariance at crossed corners.

    **Why the reciprocal is taken here rather than carried.** ``be`` is ``(dt/dx)/eps``, so a
    caller could keep ``1/eps_c`` and multiply. It does not, because ``build``'s four frozen
    archives pin ``(dt/dx)/eps`` for every cell a smoothed shape does not reach, and one
    expression per array is worth more than the extra rounding, which is 2 ulp at worst
    (2.2e-16 relative, swept over ``eps`` in [1, 16] and ``w`` in [0, 1]).

    **The truncation has a measured cost, not a universal convergence guarantee.** The
    corrected S1 sweep at contrast 12, 16/32/48 cells per layer period, reads diagonal
    fitted orders 0.999 / 1.012 / 0.864 at 14.036 / 26.565 / 45 degrees. Its self-adjoint
    full-tensor comparator is non-monotone at the last two tilts; the naive comparator is
    non-Hermitian and has complex modes that the earlier eigensolver silently projected
    onto the real axis. Neither establishes a general verdict on every possible full-tensor
    scheme. This function keeps the diagonal part. The constitutive term now adds adjoint
    cross coefficients from ``src/wavebit/fdtd/_prepare/constitutive.py``; the controlled sphere study measured
    order 2.023 with them against 1.714 without them. See ``docs/validation/fdtd/fd-mie-validation.md`` and
    ``docs/validation/fdtd/fd-subpixel-occupancy.md`` for the distinct instruments and their limits.
    """
    harmonic = fill / value + (1.0 - fill) / held
    arithmetic = fill * value + (1.0 - fill) * held
    return 1.0 / (w * harmonic + (1.0 - w) / arithmetic)


def _box_slices(box):
    """The ``[z, y, x]`` basic-indexing tuple of an x-first cell box."""
    return (slice(*box[2]), slice(*box[1]), slice(*box[0]))


def _paste(xp, out, box, value):
    """``out`` with its ``box`` sub-block replaced by ``value``, **out of place**.

    Three concatenations, innermost axis first. Written this way and not as an indexed
    assignment because ``out`` may hold tracers: ``a[i] = v`` is not available on JAX at all,
    and on Torch it mutates a tensor an earlier painter may still be holding. ``concat`` is
    the one Array-API spelling that is out of place, differentiable and the same three lines
    on all three backends, which is the rule this package is built on -- one body, and an
    input that reaches it must not give three different answers.
    """
    (i0, i1), (j0, j1), (k0, k1) = box
    band = out[..., k0:k1, j0:j1, :]
    row = xp.concat([band[..., :i0], value, band[..., i1:]], axis=-1)
    plane = out[..., k0:k1, :, :]
    plane = xp.concat([plane[..., :j0, :], row, plane[..., j1:, :]], axis=-2)
    return xp.concat([out[..., :k0, :, :], plane, out[..., k1:, :, :]], axis=-3)


class _Snapped:
    """A cell-snapped shape: its material **replaces** the cell's, on a static boolean mask.

    One ``xp.where``, exactly the line stage 1 wrote, so a scene of ``Slab``s and ``Region``s
    lowers to the same bits it always did. The mask is this shape's own filled cells and does
    not exclude the cells a later structure will claim: a later painter's ``where`` overwrites
    them, which is the same last-wins answer an owner index gave and is a select rather than
    an arithmetic, so the result is bit-identical rather than merely equal.
    """

    def __init__(self, mask):
        self.mask = mask

    def paint(self, rt, out, value, c, field, rule):
        return rt.xp.where(rt.array(self.mask), rt.array(value, 'float64'), out)


class _Smoothed:
    """A body: :func:`mix` over its support box, at each component's own Yee point.

    ``occ`` is one structure's entry from ``occupancy.occupancy`` -- the box and, per field
    and component, the ``(fill, w)`` pair. ``shape`` is the full ``(nz, ny, nx)`` the
    coefficient array has, which a smoothed shape always forces because it varies along every
    axis it reaches (``geometry.Sphere.bounded_axes``).

    **What it mixes into is the running value, and that is exact for one surface only.** At
    ``f = 1`` the mix returns this shape's own material, so an earlier painter's answer is
    replaced and the order rule is untouched; at ``f = 0`` it returns the earlier one. In
    between it is the two-phase rule between this material and *one* other, which is right
    while the voxel holds one surface -- an earlier cell-snapped shape puts none in it
    (measured 2.0e-16 for a ball on a snapped substrate) and an earlier smoothed shape that
    fills the voxel whole puts none in it either (a core inside a shell). Two smoothed
    surfaces in one voxel is the case it cannot do, and ``checks.check_smoothed`` refuses it
    before the scene reaches here rather than this class approximating it.

    **Only the support box is touched.** The rest of the array is carried through untouched
    rather than mixed at ``f = 0``, for the same reason the snapped painter is not written as
    ``f = 1``: ``mix`` at ``f = 0`` returns ``1/(1/eps_b)``, which is the same number to
    within a double reciprocal and not to within a bit.

    Inside the box, cells the body misses *are* mixed at ``f = 0`` and come back off the
    background by the double reciprocal plus whatever round-off the body's own fill carries.
    Measured, ``Sphere(radius=0.25)`` in ``eps = 1`` at 5 / 10 / 20 cells per radius: the
    worst departure of ``eps`` from the background's 1.0 anywhere outside the ball is
    3.33e-14 / 3.42e-14 / 1.42e-12, on **two** array entries of 41 472 / 192 000 / 1 119 744.
    Both are the same two voxels, and the cause is not the reciprocal: the exact clipped-ball
    volume returns a fill of -6.03e-14 / -6.16e-14 / -2.55e-12 in a voxel the sphere just
    misses -- accumulated round-off in the clip, which grows with the grid, as
    ``src/wavebit/fdtd/_shapes/_exact.py`` records. It is **not clamped**. A clamp would be a
    statement about the geometry where the truth is a round-off of order 1e-12, ten orders
    below the rule's own measured error, and it would flatten the fill's derivative at exactly
    the voxels the surface is passing through. C5's ``eps <= 0`` guard is where a fill that is
    negative *enough* to matter is caught (``checks._check_fill_contract``, which refuses a fill
    or a normal moment outside ``[0, 1]`` by more than ``checks._CLIP_ROUNDOFF``), and it fires on
    a bug rather than on this.

    ``rule`` is ``'mix'`` for a quantity this commit smooths (``eps``, ``mu``) and ``'keep'``
    for one it does not (``sigma``). ``'keep'`` leaves the cell's value alone, and that is
    only honest because ``checks.check_smoothed`` refuses a conductive or dispersive material
    anywhere a smoothed shape can reach -- so "not touched" and "zero" are the same number
    here. Measured with that door held open: a lossless ``Sphere(radius=0.25)`` in a
    ``Conductor(eps=1, sigma=0.2)`` background reads ``sigma = 0.2001`` on all 402 voxels the
    ball fills outright, where the truth is 0. That is the door's mutant, and it is what
    "silently wrong" means here -- the run completes and the number is the right shape.
    """

    def __init__(self, occ, shape):
        self.occ, self.shape = occ, shape

    def paint(self, rt, out, value, c, field, rule):
        if rule != 'mix':
            return out
        if field not in self.occ:
            raise KeyError(
                f'occupancy was not asked for the {field} points, so a smoothed shape cannot '
                f'mix a quantity that lives on them. coefficients.build decides that pair '
                f'(it skips the H points when nothing in the scene is magnetic, because bh '
                f'is then the scalar dt/dx); a caller that wants the mix must ask for the '
                f'field.')
        xp = rt.xp
        box = self.occ['box']
        fill, w = self.occ[field][c]
        full = out if getattr(out, 'shape', ()) == self.shape else xp.broadcast_to(out,
                                                                                  self.shape)
        held = full[_box_slices(box)]
        mixed = mix(xp, held, rt.array(value, 'float64'), fill, w)
        return _paste(xp, full, box, mixed)


def painters(scene, plan, axes, occ):
    """One painter per structure, in scene order: how it puts its material into a cell.

    ``occ`` maps a structure's index to its occupancy entry (``occupancy.occupancy``); a
    structure that is not in it is cell-snapped and gets the static mask its
    :meth:`~wavebit.fdtd._legacy_geometry.Shape.filled_cells` names. That is the **only** reader of
    ``filled_cells``, which is why a smoothed shape refuses the method rather than
    over-answering it: the caller that must change when a fractional edge arrives is this one
    and no other.

    A smoothed shape must have put **every** axis in its ``bounded_axes``, and that is refused
    here rather than assumed. A cell-snapped ``Region`` may leave out an axis it spans, because a
    box clipped to the domain really is uniform along it and an extent-1 coefficient axis is
    then exact. A body that curves is not, so the same saving would broadcast one plane of
    fills over the whole grid -- a plausible array of the right shape. The extruded bodies
    (C4) answer all three axes as well, though they *are* uniform along the axis they span:
    one painter rule serves every body, and the one-plane saving is not taken.
    """
    _, structures = materials_of(scene)
    shape = _shape(plan, axes)[1:]
    full = _shape(plan, (0, 1, 2))[1:]
    grid = plan.grid
    out = []
    for k, s in enumerate(structures):
        if k in occ:
            if shape != full:
                raise ValueError(
                    f'structures[{k}] ({public_description(s.shape)}) is smoothed but the coefficient '
                    f'arrays are {shape} rather than {full}: some shape in this scene left an '
                    f'axis out of bounded_axes(). A smoothed shape varies along every axis it '
                    f'reaches, so an axis of extent 1 would broadcast one plane of fill '
                    f'fractions over the whole grid. Return (0, 1, 2) from bounded_axes().')
            out.append(_Smoothed(occ[k], shape))
            continue
        box = s.shape.filled_cells(grid)
        mask = np.zeros(shape, dtype=bool)
        index = [slice(None)] * 3
        for a in axes:
            index[ARRAY_AXIS[a] - 1] = slice(*box[a])
        mask[tuple(index)] = True
        out.append(_Snapped(mask))
    return out


def _blend(rt, paint_list, background_value, structure_values, component, field, rule,
           junctions=(), scene=None, plan=None):
    """``background`` everywhere, then each structure in turn -- replacing it or mixed into it.

    The values may be traced, so this is a chain of ``xp.where`` and ``mix`` and never an
    assignment into an array. With no structure it returns the background scalar, which is
    what keeps a vacuum run on the stage-0 Python scalars.
    """
    out = rt.array(background_value[component], 'float64')
    grounds = {}
    for k, (painter, value) in enumerate(zip(paint_list, structure_values)):
        for record in junctions:
            if k == min(record.face, record.standing):
                grounds[record] = out
        out = painter.paint(rt, out, value[component], component, field, rule)
    if junctions:
        from wavebit.fdtd._prepare.junction import paint
        for record in junctions:
            out = paint(rt, record, scene, plan, out, grounds[record], component)
    return out


def _damping(plan, axes, field):
    """``(3, nz|1, ny|1, nx|1)`` of the ``Absorber`` rate at each component's own points.

    The profile ``6 u^3 / thickness`` (``cpml.damping_profile``) of the depth into the layer,
    sampled at ``index + offset`` where ``offset`` is the component's Yee half-cell offset
    along the absorbing axis -- the legacy engine's ``damping_profile((0, 0, .5))`` for E and
    ``(.5, .5, 0)`` for H, written for all three axes.
    """
    out = np.zeros(_shape(plan, axes))
    n = (plan.nx, plan.ny, plan.nz)
    for name, cells in plan.absorber_cells:
        a, side = 'xyz'.index(name[0]), name[1]
        thickness = cells * plan.dx
        for c in range(3):
            offset = 0.5 * OFFSETS[field][c][a]
            index = np.arange(n[a]) + offset
            distance = index if side == '-' else (n[a] - index)
            depth = np.maximum(cells - distance, 0.0) * plan.dx
            profile = damping_profile(depth, thickness)
            shape = [1, 1, 1]
            shape[ARRAY_AXIS[a] - 1] = n[a]
            out[c] = out[c] + profile.reshape(shape)
    return out


def build(scene, plan, rt):
    """``{'ae', 'be', 'ah', 'bh', 'minus_bh'}``; scalars when nothing makes them otherwise.

    ``minus_bh`` is ``-bh``, lowered once because the H update subtracts. Negating inside the
    step body cost one full-volume temporary *per step* as soon as an ``Absorber`` or a set of
    structures made ``bh`` an array -- measured with ``tracemalloc`` over three warm NumPy
    steps of a 40^3 ``Absorber`` box, 1.002 volumes per step against 0.138 with the negation
    hoisted, and it broke the preallocation contract ``src/wavebit/fdtd/_runtime.py`` states. The
    hoist is bit-identical (30 steps of a 24^3 box, final E and H equal under
    ``np.array_equal``); a sign argument on ``rt.update`` would save the leaf as well but
    reassociates the sign, so it is not.
    """
    background, structures = materials_of(scene)
    axes = varying_axes(scene, plan)
    dt, dx = plan.dt, plan.dx
    magnetic = background.is_magnetic or any(s.material.is_magnetic for s in structures)
    lossy = _has_loss(background, structures)
    absorbing = bool(plan.absorber_cells)

    if not axes and not lossy and not magnetic:
        # Uniform, lossless, non-magnetic: three numbers, or one when they agree.
        eps = list(background.eps_for_update(dt))
        if all(e is eps[0] or e == eps[0] for e in eps[1:]) and _is_one(eps[0]):
            # Vacuum, to the bit: the stage-0 scalars, so the stage-0 golden still holds.
            return _with_minus({'ae': rt.array(1.0, 'float64'),
                                    'be': rt.array(dt / dx, 'float64'),
                                    'ah': rt.array(1.0, 'float64'),
                                    'bh': rt.array(dt / dx, 'float64')})
        be = rt.xp.stack([rt.array(dt / dx, 'float64') / rt.array(e, 'float64') for e in eps])
        return _with_minus({'ae': rt.array(1.0, 'float64'),
                                'be': be.reshape((3, 1, 1, 1)),
                                'ah': rt.array(1.0, 'float64'),
                                'bh': rt.array(dt / dx, 'float64')})

    # ``mu`` is read at the H points and ``eps`` at the E points -- a smoothed shape's fill
    # differs between them by half a cell. The H pass is built only when something in the
    # scene is magnetic, because ``bh`` is the scalar ``dt/dx`` otherwise and the three
    # H-point evaluations would be discarded.
    from wavebit.fdtd._prepare.conductive_film import descriptor
    from wavebit.fdtd._prepare.conductive_film import material_profiles
    film_profiles = material_profiles(scene, plan, rt)
    film = descriptor(scene, plan.grid) if film_profiles is not None else None
    if film_profiles is None:
        from wavebit.fdtd._prepare.single_pole_film import descriptor
        from wavebit.fdtd._prepare.single_pole_film import material_profiles
        film_profiles = material_profiles(scene, plan, rt)
        film = descriptor(scene, plan.grid) if film_profiles is not None else None
    if film_profiles is None:
        from wavebit.fdtd._prepare.debye_film import descriptor
        from wavebit.fdtd._prepare.debye_film import material_profiles
        film_profiles = material_profiles(scene, plan, rt)
        film = descriptor(scene, plan.grid) if film_profiles is not None else None
    if film_profiles is None:
        from wavebit.fdtd._prepare.two_debye_film import descriptor
        from wavebit.fdtd._prepare.two_debye_film import material_profiles
        film_profiles = material_profiles(scene, plan, rt)
        film = descriptor(scene, plan.grid) if film_profiles is not None else None
    if film_profiles is None:
        from wavebit.fdtd._prepare.debye_sum_film import descriptor
        from wavebit.fdtd._prepare.debye_sum_film import material_profiles
        film_profiles = material_profiles(scene, plan, rt)
        film = descriptor(scene, plan.grid) if film_profiles is not None else None
    if film_profiles is None:
        from wavebit.fdtd._prepare.oscillator_sum_film import descriptor
        from wavebit.fdtd._prepare.oscillator_sum_film import material_profiles
        film_profiles = material_profiles(scene, plan, rt)
        film = descriptor(scene, plan.grid) if film_profiles is not None else None
    if film_profiles is None:
        from wavebit.fdtd._prepare.hybrid_film import descriptor
        from wavebit.fdtd._prepare.hybrid_film import material_profiles
        film_profiles = material_profiles(scene, plan, rt)
        film = descriptor(scene, plan.grid) if film_profiles is not None else None
    if film_profiles is None:
        from wavebit.fdtd._prepare.conductive_stack import descriptor
        from wavebit.fdtd._prepare.conductive_stack import material_profiles
        film_profiles = material_profiles(scene, plan, rt)
        film = descriptor(scene, plan.grid) if film_profiles is not None else None
    if film_profiles is None:
        from wavebit.fdtd._prepare.debye_stack import descriptor
        from wavebit.fdtd._prepare.debye_stack import material_profiles
        film_profiles = material_profiles(scene, plan, rt)
        film = descriptor(scene, plan.grid) if film_profiles is not None else None
    if film_profiles is None:
        from wavebit.fdtd._prepare.oscillator_stack import descriptor
        from wavebit.fdtd._prepare.oscillator_stack import material_profiles
        film_profiles = material_profiles(scene, plan, rt)
        film = descriptor(scene, plan.grid) if film_profiles is not None else None
    if film_profiles is not None:
        # The supported films vary on one axis and lower their exact local response
        # separately from the ordinary occupancy painter.
        axes = {film.axis}
        eps, sigma = film_profiles
        mu = rt.array(1.0, 'float64')
    else:
        occ = occupancy(scene, plan, rt, fields=('E', 'H') if magnetic else ('E',))
        if rt.name == 'numpy':
            # The body contract, before any coefficient is built from it -- and so before
            # constitutive.validate's certificate reads one (checks._check_fill_contract).
            from wavebit.fdtd._prepare.checks import _check_fill_contract
            _check_fill_contract(scene, occ, plan.grid)
        paint_list = painters(scene, plan, axes, occ) if structures else []
        eps = _stack(rt, paint_list, background.eps_for_update(dt),
                     [s.material.eps_for_update(dt) for s in structures], 'E', 'mix',
                     junctions=plan.junctions, scene=scene, plan=plan)
        mu = _stack(rt, paint_list, background.mu_value,
                    [s.material.mu_value for s in structures], 'H', 'mix' if magnetic else 'keep')
        sigma = _stack(rt, paint_list, background.sigma,
                       [s.material.sigma for s in structures], 'E', 'keep')

    # Each half is built only if something makes it non-trivial: ``ae`` and ``ah`` stay the
    # stage-0 scalar 1.0 unless there is a loss, and ``bh`` stays dt/dx unless a permeability
    # or an absorber moves it. The step multiplies by whatever it is given, and NumPy's
    # update skips a multiply by the scalar one, so this is an operation count and not only
    # a byte count.
    if lossy or absorbing:
        rate_e = sigma / eps
        if absorbing:
            rate_e = rate_e + rt.array(_damping(plan, axes, 'E'), 'float64')
        half_e = 0.5 * dt * rate_e
        ae, be = (1.0 - half_e) / (1.0 + half_e), (dt / dx) / eps / (1.0 + half_e)
    else:
        ae, be = rt.array(1.0, 'float64'), (dt / dx) / eps
    if absorbing:
        half_h = 0.5 * dt * rt.array(_damping(plan, axes, 'H'), 'float64')
        ah, bh = (1.0 - half_h) / (1.0 + half_h), (dt / dx) / mu / (1.0 + half_h)
    elif magnetic:
        ah, bh = rt.array(1.0, 'float64'), (dt / dx) / mu
    else:
        ah, bh = rt.array(1.0, 'float64'), rt.array(dt / dx, 'float64')
    return _with_minus({'ae': ae, 'be': be, 'ah': ah, 'bh': bh})


def _with_minus(out):
    """``out`` plus ``minus_bh``; see :func:`build` for why the negation is not in the step."""
    out['minus_bh'] = -out['bh']
    return out


def _stack(rt, paint_list, background_value, structure_values, field, rule,
           junctions=(), scene=None, plan=None):
    """A ``(3, ...)`` array of one material quantity: background, then each structure.

    ``field`` says which set of Yee points a smoothed structure is read at -- ``'E'`` for
    ``eps`` and ``sigma``, ``'H'`` for ``mu`` -- and ``rule`` whether a smoothed structure
    mixes into the cell or leaves that quantity alone. A cell-snapped structure ignores both:
    its cells are the same range for all six components, which is what made one owner index
    serve them in stage 1 and what keeps those scenes bit-identical here.
    """
    parts = [_blend(rt, paint_list, background_value, structure_values, c, field, rule,
                    junctions=junctions, scene=scene, plan=plan)
             for c in range(3)]
    parts = [rt.xp.broadcast_to(p, _broadcast_shape(parts)) for p in parts]
    return rt.xp.stack(parts, axis=0)


def _broadcast_shape(parts):
    shape = [1, 1, 1]
    for p in parts:
        s = tuple(getattr(p, 'shape', ()) or (1, 1, 1))
        s = (1,) * (3 - len(s)) + s
        shape = [max(a, b) for a, b in zip(shape, s)]
    return tuple(shape)


def _is_one(v):
    from wavebit.materials._time_domain import concrete
    c = concrete(v)
    return c is not None and c == 1.0


def _has_loss(background, structures):
    from wavebit.materials._time_domain import concrete
    for material in (background,) + tuple(s.material for s in structures):
        for s in material.sigma:
            c = concrete(s)
            if c is None or c != 0.0:
                return True
    return False


#: Params the report does not count: the initial field *becomes* the state at ``n = 0`` and is
#: counted there, the amplitude, the Bloch and Axis1D factors and the stability guard are
#: scalars.
NOT_COUNTED = ('E0', 'H0', 'amplitude', 'bloch', 'axis1d', 'stable')


def params_bytes(params):
    """The bytes the coefficient and profile leaves of ``params`` occupy, for the report.

    The state's bytes are two ``(3, nz, ny, nx)`` fields plus the psi, and are a number a
    reader can predict; the params' are not, because stage 1 made them depend on what the
    scene declares -- four coefficient arrays whose shape is whatever varies, the CPML
    profiles, the pole coefficients. Printing them beside each other is what turns "why did
    this scene need 400 MB" into an arithmetic a reader can check.
    """
    total = 0
    stack = [{k: v for k, v in params.items() if k not in NOT_COUNTED}]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (tuple, list)):
            stack.extend(v)
        elif v is not None:
            nbytes = getattr(v, 'nbytes', None)
            if nbytes is None:
                size = getattr(v, 'numel', None)
                itemsize = getattr(getattr(v, 'dtype', None), 'itemsize', 8)
                nbytes = (size() * itemsize) if size is not None else 0
            total += int(nbytes)
    return total
