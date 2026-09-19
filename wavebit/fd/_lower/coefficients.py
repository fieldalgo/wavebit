"""The update coefficients: ``ae``, ``be``, ``ah``, ``bh`` -- scalars in vacuum, arrays when not.

The step is written once against these four numbers (``wavebit/fd/_kernel/step.py``), so a material
changes them and not the code::

    ae = (1 - dt*rate_e/2) / (1 + dt*rate_e/2)      rate_e = sigma/eps + damping_e
    be = (dt/dx) / eps / (1 + dt*rate_e/2)
    ah = (1 - dt*rate_h/2) / (1 + dt*rate_h/2)      rate_h = damping_h  (Absorber only)
    bh = (dt/dx) / mu  / (1 + dt*rate_h/2)

``dt/dx`` rather than ``dt`` because the curl is the bare difference and carries no ``1/dx``
(``wavebit/fd/_kernel/curl.py``); a magnetic loss is not a material here and exists only inside an
``Absorber``, where it is the matched half of the graded damping.

**An array is built only where the value is not uniform**, and only along the axes it
actually varies on. Every stage-1 non-uniformity is axis-aligned -- a ``Slab`` varies along
one axis, an ``Absorber`` along one axis, and stage 3's cell-snapped ``Box`` along each axis
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

Structures are applied **in the order given**, so a later one wins where two overlap --
``owner_map`` is the one place that rule is written, and an overlap that a *dispersive*
structure is party to is refused at the door instead, because the pole term is not routed
through this map (``wavebit/fd/_lower/checks.py::check_overlaps``).

The ``eps`` these four are built from is ``material.eps_for_update(dt)`` and not
``material.eps``: a :class:`~wavebit.fd.Debye` pole's trapezoidal recurrence is implicit in
``E``, and the part of it that is simultaneous with the field being solved for is a constant
added to the permittivity. Every pole that is not Debye adds nothing and the two readings are
the same object, so a vacuum, Drude or Lorentz scene's coefficients are unchanged to the bit.
Nothing that reports *physics* reads ``eps_for_update`` -- the CFL bound, the cells per
wavelength and ``eps_at`` all read ``eps``.
"""
import numpy as np

from .._kernel.grid import OFFSETS
from ..geometry import bounded_axes, cell_box
from ..materials import as_material
from .cpml import damping_profile

__all__ = ['build', 'varying_axes', 'owner_map', 'materials_of', 'params_bytes']

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
        axes.update(bounded_axes(s.shape, plan.grid))
    for name, cells in plan.absorber_cells:
        axes.add('xyz'.index(name[0]))
    return axes


def _shape(plan, axes):
    return (3,
            plan.nz if 2 in axes else 1,
            plan.ny if 1 in axes else 1,
            plan.nx if 0 in axes else 1)


def owner_map(scene, plan, axes):
    """``(nz|1, ny|1, nx|1)`` of ints: ``-1`` for the background, else the structure's index.

    Cell-snapped, so the same cell range serves all six components -- see ``geometry.py`` for
    why that is one rule and not six, and why it is the inaccurate choice it says it is.
    """
    _, structures = materials_of(scene)
    shape = _shape(plan, axes)[1:]
    owner = np.full(shape, -1, dtype=np.int32)
    grid = plan.grid
    for k, s in enumerate(structures):
        box = cell_box(s.shape, grid)
        index = [slice(None)] * 3
        for a in axes:
            index[ARRAY_AXIS[a] - 1] = slice(*box[a])
        owner[tuple(index)] = k
    return owner


def _select(rt, owner, background_value, structure_values, component):
    """``background`` everywhere, overwritten by each structure's value on its cells.

    ``owner`` is the static map; the values may be traced, so this is a chain of
    ``xp.where`` and never an assignment into an array.
    """
    out = rt.array(background_value[component], 'float64')
    if owner is None:
        return out
    for k, value in enumerate(structure_values):
        mask = rt.array(owner == k)
        out = rt.xp.where(mask, rt.array(value[component], 'float64'), out)
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
    hoisted, and it broke the preallocation contract ``wavebit/fd/_runtime.py`` states. The
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

    owner = owner_map(scene, plan, axes) if structures else None
    eps = _stack(rt, owner, background.eps_for_update(dt),
                 [s.material.eps_for_update(dt) for s in structures])
    mu = _stack(rt, owner, background.mu_value, [s.material.mu_value for s in structures])
    sigma = _stack(rt, owner, background.sigma, [s.material.sigma for s in structures])

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


def _stack(rt, owner, background_value, structure_values):
    """A ``(3, ...)`` array of one material quantity: background, then each structure's cells."""
    parts = [_select(rt, owner, background_value, structure_values, c) for c in range(3)]
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
    from ..materials import concrete
    c = concrete(v)
    return c is not None and c == 1.0


def _has_loss(background, structures):
    from ..materials import concrete
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
