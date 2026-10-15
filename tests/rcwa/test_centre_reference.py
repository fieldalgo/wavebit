"""One lateral phase reference for every per-order quantity: the cell centre, ``x = y = 0``.

Every public coordinate is measured from the centre of the cell (2026-08-31). The kernels still
integrate from the corner -- their private convolution matrices use that frame --
and the corner is also where every per-order phase used to be referenced. Moving a plane wave's
reference from the corner to the centre ``o = (a1 + a2)/2`` multiplies order ``(m, n)`` by
``exp(1j * k_mn . o) = exp(1j * k_inc . o) * (-1)**(m + n)``, on any cell, because
``b_i . a_j = 2 pi delta_ij``. The scalar is a normalisation; the sign is what an order carries.

Since 2026-09-27 every public per-order quantity is referenced at the centre: the kernels are
handed the public weights with the odd orders negated, their amplitudes and scattering matrix
are signed back (``C L C w``, ``C S C``), and the field takes the scalar once. Each check below
has an oracle outside that arithmetic:

1. **the rule** -- ``conventions.centre_parity`` is that sign, by order label, on a rectangle, a
   60-degree cell and a shear, windows moved by an excitation included; ``corner_column`` is an
   exact sign bit and leaves an undeclared solve alone.
2. **an ordinary solve is the plane wave it names** -- the coherent-source path's ``PlaneWave``
   (its own first-order cascade and incident basis), on every path, a rotated cell included, and
   both sides, through both field doors and every boundary wave, with no factor anywhere; and a
   y-invariant scene's field does not depend on its fictitious y period.
3. **``excitation`` weights are ``OrderExcitation`` weights** -- the same dict, the same light;
   a lit set of one parity class is the light it always was.
4. **every amplitude is the centre Fourier coefficient of the field** -- the reconstructed field
   on each face, decomposed in public coordinates, is ``r_xyz`` / ``t_xyz``.
5. **first-order Born** -- a weak scatterer centred at the origin scatters every order in phase
   with the specular one: ``Ey(G)/Ey(0)`` real, positive and the closed form's.
6. **the scattering matrix speaks the amplitudes' variables** -- all four coupled s/p sub-blocks,
   the far blocks against a solve lit from the other side, and a cascade across an odd label
   shift.
7. **complex ``theta``** -- the incidence is unit at the origin, not at the corner.
8. **the two field doors agree** where only the scalar could be wrong.
9. **gradients** -- JAX against central differences, through ``theta``, the wavelength and the
   cell; one Torch backward.
10. **backends and jit** -- the three backends agree, and a result that crossed a ``jit``
    boundary with the solver as the argument reconstructs the same field.
"""

from wavebit._common.backend import ComplexIncidenceWarning as _legacy_wavebit_ComplexIncidenceWarning

from wavebit.rcwa._engine.sources import OrderExcitation as _legacy_wavebit_OrderExcitation

from wavebit.rcwa._engine.sources import PlaneWave as _legacy_wavebit_PlaneWave

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.lattice import meshgrid_xy as _legacy_wavebit_meshgrid_xy

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

from wavebit._common.conventions import centre_parity, corner_column, incident_column, lit_orders, order_window
from wavebit.rcwa._engine.solve_2d import order_grid_kxy

_HEX = np.column_stack(([1.6, 0.0], [0.8, 1.6 * np.sqrt(3) / 2]))
_SHEAR = np.array([[1.6, 0.5], [0.0, 1.7]])
_CELLS = {'rectangle': dict(period=(1.3, 1.7)),
          'hexagonal': dict(lattice=_HEX, factorization='nvf'),
          'shear': dict(lattice=_SHEAR, factorization='nvf')}
#: A 1D grating that is not mirror-symmetric about x = 0 or about the cell edge, so neither a
#: sign nor a half-period shift can hide in a symmetry.
_EPS_1D = np.array([[[4., 4., 1., 1., 2.25, 1.]]])
_RASTER_2D = np.random.default_rng(3).uniform(1.0, 2.2, (1, 6, 5)) ** 2


def _host(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return np.asarray(value)


def _rel(got, want):
    """Largest entry of ``got - want`` against the largest of ``want``."""
    got, want = _host(got), _host(want)
    assert got.shape == want.shape, (got.shape, want.shape)
    return float(np.abs(got - want).max() / np.abs(want).max())


def _rod(lattice, n=16, centre=(0.3, -0.2), radius=0.5, eps=5.76):
    """An off-centre rod on ``meshgrid_xy``'s parallelogram samples, periodic images included."""
    X, Y = (np.asarray(c) for c in _legacy_wavebit_meshgrid_xy((n, n), lattice=lattice))
    inside = np.zeros(X.shape, dtype=bool)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            shift = i * lattice[:, 0] + j * lattice[:, 1]
            inside |= ((X - centre[0] - shift[0]) ** 2
                       + (Y - centre[1] - shift[1]) ** 2 <= radius ** 2)
    cell = np.full((1, n, n), 1.0)
    cell[0][inside] = eps
    return cell


def _label_offset(solver):
    """``(centre_u - half_u) + (centre_v - half_v)``: the first retained label, both axes."""
    (cu, hu), (cv, hv) = solver.order_window
    return (cu - hu) + (cv - hv)


# ------------------------------------------------------------------------------ 1. the rule

@pytest.mark.parametrize('excitation', [None, {(1, 0): 1.0, (0, 1): 0.5j},
                                        {(2, 1): 1.0, (1, 1): 0.3}],
                         ids=['default', 'moved-by-one', 'moved-by-two-and-one'])
@pytest.mark.parametrize('cell', sorted(_CELLS))
def test_the_centre_sign_is_the_phase_the_centre_adds(cell, excitation):
    """``(-1)**(m + n) == exp(1j * G_mn . (a1 + a2)/2)``, read off the solver's own wavevectors.

    ``G_mn`` is each retained order's in-plane wavevector minus the specular order's, from
    ``order_kxy()`` -- the grid the solve is built on, not a second spelling of it -- and the
    centre from ``solver.lattice``. On the two skewed cells ``G . o`` is ``pi (m + n)`` only
    because ``b_i . a_j = 2 pi delta_ij`` holds for the *pair*; a rectangle-only formula for the
    centre would pass the first cell and fail these. The excitations move the window by an odd
    step on each axis, where parity by array index and parity by label part company.
    """
    solver = _legacy_wavebit_rcwa(backend='numpy', wavelength=0.9, theta=0.3, phi=0.4, top=1.0, bot=2.25,
                          structure=np.full((1, 8, 8), 2.25), thickness=[0.2],
                          truncation_order=(2, 2), excitation=excitation, **_CELLS[cell])
    window_u, window_v = solver.order_window
    gx, gy = order_grid_kxy(np, *(_host(k) for k in solver.order_kxy()))
    specular = (0 - window_v[0] + window_v[1], 0 - window_u[0] + window_u[1])
    k0 = 2 * np.pi / solver.wavelength
    G = k0 * np.stack([gx - gx[specular], gy - gy[specular]], axis=-1)
    o = _host(solver.lattice.columns).sum(axis=1) / 2
    phase = np.exp(1j * G @ o)
    sign = np.where(centre_parity(window_u, window_v), -1.0, 1.0)
    assert phase.shape == sign.shape
    assert np.abs(phase - sign).max() < 1e-13, np.abs(phase - sign).max()
    if excitation is not None:
        # The premise: this window's first label is odd on at least one axis, so an index
        # parity would disagree with the label parity here.
        assert (window_u[0] - window_u[1]) % 2 or (window_v[0] - window_v[1]) % 2


def test_the_parity_is_read_by_label():
    """Label ``(m, n)``, not the array position: a window moved by one flips every entry."""
    assert centre_parity((0, 1), (0, 0)).tolist() == [[True, False, True]]
    assert centre_parity((1, 1), (0, 0)).tolist() == [[False, True, False]]
    assert centre_parity((0, 1), (0, 1)).tolist() == [[False, True, False],
                                                      [True, False, True],
                                                      [False, True, False]]
    assert centre_parity((0, 1), (-1, 1)).tolist() == [[True, False, True],
                                                       [False, True, False],
                                                       [True, False, True]]
    assert centre_parity((0, 2), (0, 0)).dtype == bool


def test_the_corner_column_is_a_sign_bit():
    """Odd weights negated -- bit for bit, signed zeros and all -- even ones untouched.

    ``None`` stays ``None``, so an undeclared solve keeps the unit column it always had.
    """
    assert corner_column(None, (0, 3), (0, 0)) is None
    weights = {-1: 0.4 + 0.2j, 0: 1.0, 1: 0.6 - 0.8j, 2: complex(-0.0, 0.3), 3: -0.0}
    window = order_window(3, 0, weights)
    lit = lit_orders(weights, *window)
    corner = corner_column(lit, *window)
    odd = centre_parity(*window).reshape(-1)
    assert odd[[one.index for one in lit]].tolist() == [True, False, True, False, True]
    for public, kernel in zip(lit, corner):
        assert (kernel.index, kernel.order) == (public.index, public.order)
        want = np.asarray(-public.amplitude if odd[public.index] else public.amplitude)
        got = np.asarray(kernel.amplitude)
        assert (got.dtype, got.tobytes()) == (want.dtype, want.tobytes())
    n = centre_parity(*window).size
    public_column = incident_column(np, np.complex128, lit, n, None)
    kernel_column = incident_column(np, np.complex128, corner, n, None)
    # `==`, not bytes: a real weight takes its `+0j` after the negation, so its imaginary zero
    # is unsigned in the kernels' column and signed in the negated public one.
    assert np.array_equal(kernel_column, np.where(odd, -public_column, public_column))
    # The specular order is even, so the one-wave default a direct kx names is untouched.
    default = lit_orders({(0, 0): 1.0}, *order_window(3, 2, None))
    assert corner_column(default, *order_window(3, 2, None)) == default


# ------------------------------------------------------ 2. an ordinary solve is its plane wave

_PATHS = ('te', 'tm', 'conical', 'second_order', 'first_order', 'hex', 'shear', 'rotated')
#: The pure 1D paths return three components: TE (Ey, Hx, Hz), TM (Hy, Ex, Ez).
_SLOTS = {'te': [1, 3, 5], 'tm': [4, 0, 2]}
#: A 1.3 x 1.7 rectangle turned by 0.4 rad: a legal basis, and the only cell here whose ``a1``
#: has a ``y`` component. ``_HEX`` and ``_SHEAR`` both start ``a1 = (a, 0)``, so a centre spelled
#: without ``a1_y`` passes on every other cell in this file.
_ROTATED = (np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
            @ np.diag([1.3, 1.7]))


def _plane_scene(path):
    """``(geometry, incidence, points)`` per path: the scene, its angles, where to sample.

    Explicit Cartesian points on the axis-aligned rectangles; counts on the skewed and the
    rotated cells, where an explicit ``sample_x`` does not name a ``u`` at the solver door.
    """
    one_d = dict(period=(2.0, 2.0), wavelength=1.3, structure=_EPS_1D, thickness=[0.12],
                 truncation_order=(3, 0))
    line = dict(sample_x=[-0.37, 0.05, 0.61], sample_y=[0.0], sample_z=[0.02, 0.1])
    if path == 'te':
        return one_d, dict(theta=0.3), line
    if path == 'tm':
        return one_d, dict(theta=-0.3), line
    if path == 'conical':
        return one_d, dict(theta=0.3, phi=0.6, polarization=(0.6, 0.8j)), line
    coupled = dict(theta=0.25, phi=0.6, polarization=(0.6, 0.8j))
    if path in ('second_order', 'first_order'):
        return (dict(period=(1.3, 1.7), wavelength=1.1, structure=_RASTER_2D, thickness=[0.15],
                     truncation_order=(2, 1)), coupled,
                dict(sample_x=[-0.4, 0.2], sample_y=[-0.3, 0.5], sample_z=[0.02, 0.1]))
    lattice = {'hex': _HEX, 'shear': _SHEAR, 'rotated': _ROTATED}[path]
    return (dict(lattice=lattice, structure=_rod(lattice), thickness=[0.15], wavelength=1.3,
                 truncation_order=(2, 1), factorization='nvf', raster_normal_method='contour'),
            coupled, dict(sample_x=5, sample_y=4, sample_z=[0.02, 0.1]))


def _ordinary(path, geometry, incidence, side, **over):
    kw = dict(backend='numpy', top=1.0, bot=2.25, answers='field', incidence_side=side, **geometry)
    kw.update(incidence)
    if path in ('te', 'tm'):
        kw['tetm'] = path
    kw.update(over)
    solver = _legacy_wavebit_rcwa(**kw)
    if path == 'first_order':
        solver.formulation = 'first_order'
    return solver, solver.solve()


def _coherent(geometry, side, source, **over):
    kw = dict(backend='numpy', top=1.0, bot=2.25, answers='field', excitation={side: source},
              **geometry)
    kw.update(over)
    return _legacy_wavebit_rcwa(**kw).solve()


def _solver_door(path, solver, points):
    """The solver-level field in the result door's six-component layout, where it has one."""
    cell = _host(solver.reconstruct_field(**points))
    return cell if path in ('te', 'tm') else cell[0]


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path', _PATHS)
def test_an_ordinary_solve_is_the_plane_wave_it_names(path, side):
    """The ordinary solve's field **is** ``PlaneWave``'s, on both doors and every boundary wave.

    The coherent-source path is an independent construction: its own first-order cascade, its own
    incident basis, and phases referenced at ``x = y = 0`` from the start. Both solves are unit
    at the public origin since 2026-09-27, so there is no factor between them -- until then an
    ordinary incidence was unit at the corner, ``exp(1j k_inc . (a1 + a2)/2)`` away, which is
    O(1) here (every case is off normal incidence). Pure TE and TM (TM at ``theta < 0``), the
    conical path, 2D in both formulations, a 60-degree cell, a shear and a rotated rectangle --
    the one basis whose ``a1`` has a ``y`` component, which the centre ``(a1 + a2)/2`` must read
    -- lit from either side.

    Every case's first retained label is odd on some axis (asserted), so a parity read by
    array index rather than by label would show at the solver door, the one place the sign
    does not cancel.
    """
    geometry, incidence, points = _plane_scene(path)
    solver, ordinary = _ordinary(path, geometry, incidence, side)
    assert _label_offset(solver) % 2 == 1, solver.order_window
    wave = dict(incidence)
    if path in ('te', 'tm'):
        wave['polarization'] = (1.0, 0.0) if path == 'te' else (0.0, 1.0)
    coherent = _coherent(geometry, side, _legacy_wavebit_PlaneWave(**wave))

    want = _host(coherent.reconstruct_field(**points))
    assert _rel(ordinary.reconstruct_field(**points), want) < 1e-10, 'result door'
    solver_door = _solver_door(path, solver, points)
    want_door = want[..., _SLOTS[path]] if path in ('te', 'tm') else want
    assert _rel(solver_door, want_door) < 1e-10, 'solver door'

    face = dict(sample_x=points['sample_x'], sample_y=points['sample_y'])
    for which in ('top', 'bottom'):
        for part in ('incoming', 'outgoing', 'total'):
            got = _host(ordinary.boundary_fields(which, part=part, **face))
            ref = _host(coherent.boundary_fields(which, part=part, **face))
            if not np.abs(ref).max():
                assert not np.abs(got).max(), (which, part)
                continue
            assert _rel(got, ref) < 1e-10, (which, part)


def test_the_incident_wave_is_unit_at_the_public_origin():
    """``boundary_fields(part='incoming')`` at ``x = y = 0`` is the polarization, exactly.

    The contract in one number, and the corner beside it: half a period away the same wave is
    ``exp(-1j k_inc . o)`` of that, a unit-modulus phase that is not 1 at this angle.
    """
    geometry, incidence, _ = _plane_scene('conical')
    solver, result = _ordinary('conical', geometry, incidence, 'top')
    origin = _host(result.boundary_fields('top', [0.0], [0.0], part='incoming'))[0, 0, :3]
    phi = incidence['phi']
    a_s, a_p = np.array(incidence['polarization']) / np.linalg.norm(incidence['polarization'])
    kz = np.cos(incidence['theta'])
    s_hat = np.array([-np.sin(phi), np.cos(phi), 0.0])
    p_hat = np.array([-kz * np.cos(phi), -kz * np.sin(phi), -np.sin(incidence['theta'])])
    assert np.abs(origin - (a_s * s_hat + a_p * p_hat)).max() < 1e-13
    corner = -_host(solver.lattice.columns).sum(axis=1) / 2
    at_corner = _host(result.boundary_fields('top', [corner[0]], [corner[1]],
                                             part='incoming'))[0, 0, :3]
    k0 = 2 * np.pi / geometry['wavelength']
    k_inc = k0 * np.sin(incidence['theta']) * np.array([np.cos(phi), np.sin(phi)])
    assert np.abs(at_corner - origin * np.exp(1j * k_inc @ corner)).max() < 1e-13
    assert np.abs(at_corner - origin).max() > 0.5


@pytest.mark.parametrize('door', ['result', 'solver'])
def test_a_fictitious_period_does_not_move_the_field(door):
    """A y-invariant grating's field does not depend on the ``y`` period it was declared with.

    ``period=(Px, Py)`` on a 1D structure names a ``Py`` that nothing in the scene has; at
    ``phi != 0`` the incidence still has a ``ky``. While the incidence was unit at the corner,
    ``y = -Py/2``, two spellings of one grating gave fields 0.99 apart (relative); unit at
    ``y = 0`` they are one field.
    """
    cells = []
    for period_y in (1.6, 2.9):
        solver = _legacy_wavebit_rcwa(backend='numpy', wavelength=1.1, period=(1.6, period_y),
                              structure=np.array([[[4., 4., 1., 1., 2.25, 1., 1.]]]),
                              thickness=[0.2], truncation_order=(4, 0), top=1.0, bot=2.25,
                              theta=0.4, phi=0.8, psi=0.3, answers='field')
        result = solver.solve()
        points = dict(sample_x=[-0.3, 0.2], sample_y=[0.0], sample_z=[0.1])
        cells.append(_host(result.reconstruct_field(**points)) if door == 'result'
                     else _host(solver.reconstruct_field(**points))[0])
    assert _rel(cells[0], cells[1]) < 1e-12


# ------------------------------------------------ 3. excitation weights are OrderExcitation's

#: `tests/rcwa/test_excitation.py`'s scene: lambda/period = 0.3, orders -3..+3 propagate at 0.25.
_EXC = dict(backend='numpy', truncation_order=(6, 0), period=(2000.0, 2000.0), wavelength=600.0,
            thickness=[130.0, 220.0], top=1.0, bot=2.25)
_W2 = {0: 1.0, 1: 0.6 - 0.8j}


def _grating():
    eps = np.ones((2, 1, 64), dtype=complex)
    eps[1, 0, :26] = 6.25
    eps[0, 0, 20:] = 4.0
    return eps


@pytest.mark.parametrize('side, anchor, corner', [('top', 0.151782, 0.452653),
                                                  ('bottom', 0.241294, 0.604108)])
def test_excitation_weights_are_order_excitation_weights(side, anchor, corner):
    """One dict through ``excitation=`` and through ``OrderExcitation(orders=)``: one light.

    Efficiencies, both field doors and the boundary waves. The anchor is the coherent door's
    ``sum(R)``; ``corner`` is what the ordinary door answered with the same weights while they
    were referenced at the cell corner (measured on the tree before 2026-09-27) -- order 1
    negated against order 0, different light, not a global phase.
    """
    solver = _legacy_wavebit_rcwa(**_EXC, theta=0.25, phi=0.0, polarization=(1.0, 0.0),
                          incidence_side=side, excitation=_W2, answers='field')
    solver.structure = _grating()
    ordinary = solver.solve()
    coherent = _legacy_wavebit_rcwa(**_EXC, structure=_grating(), answers='field', excitation={
        side: _legacy_wavebit_OrderExcitation(theta=0.25, phi=0.0, polarization=(1.0, 0.0),
                                      orders=_W2)}).solve()
    lit, far = ((coherent.output_top, coherent.output_bottom) if side == 'top' else
                (coherent.output_bottom, coherent.output_top))
    total_r, total_t = float(np.sum(ordinary.R)), float(np.sum(ordinary.T))
    assert total_r == pytest.approx(float(lit), abs=1e-12)
    assert total_t == pytest.approx(float(far), abs=1e-12)
    assert total_r == pytest.approx(anchor, abs=1e-6)
    assert abs(total_r - corner) > 0.1

    points = dict(sample_x=[-700.0, 90.0, 610.0], sample_y=[0.0], sample_z=[40.0, 200.0])
    want = _host(coherent.reconstruct_field(**points))
    assert _rel(ordinary.reconstruct_field(**points), want) < 1e-10
    assert _rel(_host(solver.reconstruct_field(**points))[0], want) < 1e-10
    for which in ('top', 'bottom'):
        for part in ('incoming', 'outgoing'):
            got = ordinary.boundary_fields(which, [-700.0, 90.0, 610.0], part=part)
            ref = _host(coherent.boundary_fields(which, [-700.0, 90.0, 610.0], part=part))
            if not np.abs(ref).max():
                continue
            assert _rel(got, ref) < 1e-10, (which, part)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_2d_mixed_parity_excitation_is_the_order_excitation(side):
    """The same on a 2D cell, with lit orders odd along ``u`` and along ``v`` separately."""
    weights = {(0, 0): 1.0, (1, 0): 0.5j, (0, 1): 0.3 - 0.2j}
    geometry, incidence, points = _plane_scene('second_order')
    geometry = dict(geometry, wavelength=0.8, truncation_order=(1, 1))
    solver, ordinary = _ordinary('second_order', geometry, dict(incidence, theta=0.2, phi=0.5),
                                 side, excitation=weights)
    coherent = _coherent(geometry, side, _legacy_wavebit_OrderExcitation(
        theta=0.2, phi=0.5, polarization=incidence['polarization'], orders=weights))
    lit, far = ((coherent.output_top, coherent.output_bottom) if side == 'top' else
                (coherent.output_bottom, coherent.output_top))
    assert float(np.sum(ordinary.R)) == pytest.approx(float(lit), abs=1e-12)
    assert float(np.sum(ordinary.T)) == pytest.approx(float(far), abs=1e-12)
    want = _host(coherent.reconstruct_field(**points))
    assert _rel(ordinary.reconstruct_field(**points), want) < 1e-10
    assert _rel(_host(solver.reconstruct_field(**points))[0], want) < 1e-10


def test_a_lit_set_of_one_parity_is_the_light_it_always_was():
    """The move is a *relative* sign between lit orders, so one parity class does not see it.

    The premise that the change is not a global phase in disguise. Measured on the tree before
    2026-09-27 -- the same scene, the same weights: all even ``{0, 2}`` and all odd ``{-1, 1}``
    (the kernels are handed ``-w``, a global sign) came back byte-identical on the machine that
    measured them. Asserted to 1e-12 rather than by ``==``: the last bits of a solve belong to
    the LAPACK build -- JAX and Torch on that same machine already sit up to 2.8e-15 away -- and
    a relative sign moves ``sum(R)`` by O(0.1). The mixed set beside them is the one that moved,
    from 0.452653.
    """
    before = {((0, 1.0), (2, 0.5 - 0.3j)): 0.2520161772048634,
              ((-1, -0.2j), (1, 0.7)): 0.3403904810727651,
              ((0, 1.0), (1, 0.6 - 0.8j)): 0.4526529332822695}
    for key, old in before.items():
        solver = _legacy_wavebit_rcwa(**_EXC, theta=0.25, tetm='te', excitation=dict(key))
        solver.structure = _grating()
        total = float(np.sum(solver.solve().R))
        if len({m % 2 for m, _ in key}) == 1:
            assert total == pytest.approx(old, abs=1e-12), (key, total)
        else:
            assert total == pytest.approx(0.151782, abs=1e-6), (key, total)


# --------------------------------------- 4. every amplitude is a centre Fourier coefficient

def _decompose(result, solver, z, n_u, n_v):
    """Every retained order's ``(Ex, Ey)`` on the plane at height ``z``, in public coordinates.

    A discrete Fourier transform on a uniform grid over one rectangular cell, measured from its
    centre, against ``exp(-1j k_m . x)`` -- no half-cell anywhere. More samples than retained
    orders each way, so nothing aliases.
    """
    period = np.diag(_host(solver.lattice.columns))
    xs = (np.arange(n_u) / n_u - 0.5) * period[0]
    ys = (np.arange(n_v) / n_v - 0.5) * period[1] if n_v > 1 else np.array([0.0])
    cell = _host(result.reconstruct_field(sample_x=xs, sample_y=ys, sample_z=[z]))[0][..., :2]
    kx, ky = order_grid_kxy(np, *(_host(k) for k in solver.order_kxy()))
    k0 = 2 * np.pi / solver.wavelength
    X, Y = np.meshgrid(xs, ys)
    basis = np.exp(-1j * k0 * (np.multiply.outer(kx, X) + np.multiply.outer(ky, Y)))
    return np.einsum('abij,ijc->abc', basis, cell) / X.size, xs, ys


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('scene', ['1d', '2d'])
def test_every_amplitude_is_the_centre_fourier_coefficient_of_the_field(scene, side):
    """The reconstructed field on each face, decomposed about the centre, is ``r_xyz`` / ``t_xyz``.

    The **reconstructed** field, not ``boundary_fields``: those spectra are built from the
    amplitudes and would restate them. On the lit face the incoming wave is taken off first
    (built from the public weights, which the tests above pin to the coherent door), and the
    transverse part compared -- ``E_z`` is not continuous across the face. The far face is
    ``t_xyz`` outright, which is the only check that would see transmitted amplitudes left in
    the kernels' frame.

    Each window is moved by an excitation so that its first label, summed over both axes, is
    odd -- where a parity read by array index would differ from the label's -- and the 2D one
    carries rows of odd ``n``, where a parity that ignored ``n`` would show.
    """
    if scene == '1d':
        solver = _legacy_wavebit_rcwa(backend='numpy', wavelength=0.6, period=(2.0, 2.0), structure=_EPS_1D,
                              thickness=[0.12], truncation_order=(3, 0), top=1.0, bot=2.25,
                              theta=0.3, tetm='te', incidence_side=side,
                              excitation={1: 1.0, 2: 0.5j}, answers='field')
        grid = (32, 1)
    else:
        solver = _legacy_wavebit_rcwa(backend='numpy', wavelength=0.4, period=(1.3, 1.7), structure=_RASTER_2D,
                              thickness=[0.15], truncation_order=(2, 2), top=1.0, bot=2.25,
                              theta=0.2, phi=0.3, polarization=(0.6, 0.8j), incidence_side=side,
                              excitation={(0, 0): 1.0, (1, 0): 0.5j}, answers='field')
        grid = (16, 12)
    assert _label_offset(solver) % 2 == 1, solver.order_window
    result = solver.solve()
    top = float(np.sum(_host(solver.thickness)))
    lit_z, far_z = (top, 0.0) if side == 'top' else (0.0, top)
    far = 'bottom' if side == 'top' else 'top'

    coefficients, xs, ys = _decompose(result, solver, lit_z, *grid)
    incoming, _, _ = _decompose_boundary(result, solver, side, xs, ys)
    want = _host(result.r_xyz)[..., :2]
    assert _rel(coefficients - incoming, want) < 1e-9, 'lit face'
    coefficients, _, _ = _decompose(result, solver, far_z, *grid)
    assert _rel(coefficients, _host(result.t_xyz)[..., :2]) < 1e-9, f'far face ({far})'


def _decompose_boundary(result, solver, side, xs, ys):
    """The incoming boundary wave through the same transform as :func:`_decompose`."""
    cell = _host(result.boundary_fields(side, xs, ys, part='incoming'))[..., :2]
    kx, ky = order_grid_kxy(np, *(_host(k) for k in solver.order_kxy()))
    k0 = 2 * np.pi / solver.wavelength
    X, Y = np.meshgrid(xs, ys)
    basis = np.exp(-1j * k0 * (np.multiply.outer(kx, X) + np.multiply.outer(ky, Y)))
    return np.einsum('abij,ijc->abc', basis, cell) / X.size, xs, ys


# ---------------------------------------------------------------------- 5. first-order Born

_BORN_WL = 1.0
_BORN_D = _BORN_WL / 400
_BORN_DELTA = 0.02


def _born_ratios(solver, result, deps):
    """``Ey(G)/Ey(0)`` through ``r_xyz`` against the Born closed form, per propagating order.

    To first order a thin layer of ``delta eps`` in air reflects order ``G`` with
    ``E(G) ~ (i k0 d / 2 kz_G) (I - k_G k_G / k0^2) . E_inc * deps_G``, ``deps_G`` the Fourier
    coefficient of the perturbation about the **reference point**. For s incidence
    (``E_inc = y``) the ``y`` component of the projector is ``1 - ky_G^2``, so
    ``Ey(G)/Ey(0) = deps_G/deps_0 * kz_0/kz_G * (1 - ky_G^2)/(1 - ky_0^2)``. About the centre a
    centred feature has ``deps_G`` real and positive for small ``|G|``; about the corner it is
    ``(-1)**(m + n)`` times that. Read through ``r_xyz``, which carries no per-order s/p frame
    that could flip a sign on its own.
    """
    kx, ky = order_grid_kxy(np, *(_host(k) for k in solver.order_kxy()))
    (cu, hu), (cv, hv) = solver.order_window
    ey = _host(result.r_xyz)[..., 1]
    kz = np.sqrt(1 - kx ** 2 - ky ** 2 + 0j)
    zero = (hv - cv, hu - cu)
    out = []
    for m, n in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (2, 0)):
        i, j = n - cv + hv, m - cu + hu
        if not (0 <= i < ey.shape[0] and 0 <= j < ey.shape[1]) or kz[i, j].real <= 0.05:
            continue
        born = (deps(m, n) * kz[zero].real / kz[i, j].real
                * (1 - ky[i, j].real ** 2) / (1 - ky[zero].real ** 2))
        out.append(((m, n), ey[i, j] / ey[zero], born))
    return out


@pytest.mark.parametrize('theta', [0.0, 0.1])
def test_a_centred_stripe_scatters_in_phase_as_born_says(theta):
    """A weak, thin stripe centred at ``x = 0``: every order in phase with the specular one."""
    period = 1.6 * _BORN_WL
    X, _ = (np.asarray(c) for c in _legacy_wavebit_meshgrid_xy((1, 400), period=(period, period)))
    inside = np.abs(X) < 0.15 * period
    solver = _legacy_wavebit_rcwa(backend='numpy', period=(period, period), wavelength=_BORN_WL, theta=theta,
                          tetm='te', top=1.0, bot=1.0, thickness=[_BORN_D],
                          structure=np.where(inside, 1 + _BORN_DELTA, 1.0).reshape(1, 1, -1),
                          truncation_order=(4, 0))
    fill = float(np.mean(inside))
    ratios = _born_ratios(solver, solver.solve(), lambda m, n: np.sinc(m * fill))
    assert len(ratios) >= 2
    for order, got, born in ratios:
        assert abs(got - born) < 0.02 * abs(born), (order, got, born)


@pytest.mark.parametrize('cell', ['rectangle', 'hexagonal'])
def test_a_centred_disc_scatters_in_phase_as_born_says(cell):
    """The same with a disc on the shape path, on a rectangle and a 60-degree cell.

    ``(1, 1)`` has ``m + n`` even and did not move with the reference; ``(1, 0)`` and ``(0, 1)``
    did, on the skewed cell as on the rectangle -- ``G . (a1 + a2)/2 = pi (m + n)`` there too.
    """
    lattice = (np.array([[1.7, 0.0], [0.0, 1.5]]) if cell == 'rectangle' else
               np.array([[1.7, 0.85], [0.0, 1.7 * np.sqrt(3) / 2]]))
    radius = 0.35
    solver = _legacy_wavebit_rcwa(backend='numpy', lattice=lattice, wavelength=_BORN_WL, theta=0.0, phi=0.0,
                          polarization=(1.0, 0.0), top=1.0, bot=1.0, thickness=[_BORN_D],
                          structure=[{'background': 1.0, 'shapes': [
                              {'type': 'circle', 'cx': 0.0, 'cy': 0.0, 'lx': 2 * radius,
                               'material': 1 + _BORN_DELTA}]}],
                          truncation_order=(3, 3), factorization='laurent')
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    scipy_special = pytest.importorskip('scipy.special')

    def deps(m, n):
        x = np.linalg.norm(reciprocal @ np.array([m, n])) * radius
        return 2 * scipy_special.j1(x) / x

    ratios = _born_ratios(solver, solver.solve(), deps)
    assert {order for order, _, _ in ratios} >= {(1, 0), (0, 1)}
    for order, got, born in ratios:
        assert abs(got - born) < 0.02 * abs(born), (order, got, born)


# ------------------------------------- 6. the scattering matrix speaks the amplitudes' variables

def _column(solver, weights):
    (cu, hu), (cv, hv) = solver.order_window
    column = np.zeros((2 * hv + 1) * (2 * hu + 1), dtype=complex)
    for key, value in weights.items():
        m, n = key if isinstance(key, tuple) else (key, 0)
        column[(n - cv + hv) * (2 * hu + 1) + (m - cu + hu)] = value
    return column


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_all_four_coupled_blocks_read_the_centre_amplitudes(side):
    """``S11``'s four s/p sub-blocks on a mixed-parity column, with the p port factors.

    The p port variable is the p wave's magnetic field, ``-1j h a_p`` incident and ``1j h r_p``
    reflected, ``h = n/mu`` of the lit medium; so ``S11[:N,:N] w = r_s(s_inc)``,
    ``S11[N:,:N] w = 1j h r_p(s_inc)``, ``S11[:N,N:] (-1j h w) = r_s(p_inc)`` and
    ``S11[N:,N:] w = -r_p(p_inc)``. Every block, because a sign pattern applied to the s-s
    block alone, or doubled the wrong way on the ``[s; p]`` layout, passes a check on one.
    """
    weights = {(0, 0): 1.0, (1, 0): 0.4j, (0, -1): 0.3 - 0.2j, (1, 1): 0.25 + 0.1j}
    solver = _legacy_wavebit_rcwa(backend='numpy', wavelength=0.6, period=(1.3, 1.7), structure=_RASTER_2D,
                          thickness=[0.21], truncation_order=(2, 2), top=1.0, bot=2.1025,
                          theta=0.35, phi=0.7, polarization=(0.6, 0.8j), incidence_side=side,
                          excitation=weights, layer_algorithm='SMM')
    result = solver.solve()
    S11 = _host(solver.scattering.S11)
    n = S11.shape[0] // 2
    w = _column(solver, weights)
    h = 1.0 if side == 'top' else 1.45
    s, p = result.s_inc, result.p_inc
    ravel = lambda a: _host(a).ravel()
    assert _rel(S11[:n, :n] @ w, ravel(s.r_s)) < 1e-10
    assert _rel(S11[n:, :n] @ w, 1j * h * ravel(s.r_p)) < 1e-10
    assert _rel(S11[:n, n:] @ (-1j * h * w), ravel(p.r_s)) < 1e-10
    assert _rel(S11[n:, n:] @ w, -ravel(p.r_p)) < 1e-10


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_far_blocks_are_a_solve_lit_from_the_other_side(tetm):
    """Top-lit ``S22`` / ``S12`` against a bottom-lit solve with the same weights and ``kx`` grid.

    ``S11`` / ``S21`` are what a top-lit solve reads, so a conversion that reached those two
    blocks and not the far pair passes every check on them; a bottom-lit solve is the far pair's
    own reading. ``sin(theta_b) = n_top sin(theta) / n_bot`` keeps the grid. TE reads ``r_s`` /
    ``t_s``; TM the p port variables, ``-r_p`` and ``t_p (n/mu)_top / (n/mu)_bot``.
    """
    weights = {-1: 0.4 + 0.2j, 0: 1.0, 1: 0.6 - 0.8j}
    common = dict(backend='numpy', wavelength=0.6, period=(2.0, 2.0), structure=_EPS_1D,
                  thickness=[0.12], truncation_order=(3, 0), top=1.0, bot=2.25, tetm=tetm,
                  layer_algorithm='SMM', excitation=weights)
    top = _legacy_wavebit_rcwa(**common, theta=0.3)
    top.solve()
    bottom = _legacy_wavebit_rcwa(**common, theta=float(np.arcsin(np.sin(0.3) / 1.5)),
                          incidence_side='bottom')
    lit_below = bottom.solve()
    assert np.abs(_host(top.order_kxy()[0]) - _host(bottom.order_kxy()[0])).max() < 1e-12
    w = _column(top, weights)
    S = top.scattering
    if tetm == 'te':
        reflected, transmitted = _host(lit_below.r_s).ravel(), _host(lit_below.t_s).ravel()
    else:
        reflected = -_host(lit_below.r_p).ravel()
        transmitted = _host(lit_below.t_p).ravel() * (1.0 / 1.5)
    assert _rel(_host(S.S22) @ w, reflected) < 1e-10
    assert _rel(_host(S.S12) @ w, transmitted) < 1e-10


def test_a_cascade_across_an_odd_label_shift_is_the_joined_stack():
    """Two pieces whose order labels differ by one still join into the stack's own matrix.

    Piece B is lit on order ``-1`` at the angle that puts it on A's ``kx`` grid, so the same
    wave is order ``m`` in A and ``m - 1`` in B and its centre sign is opposite in the two:
    ``C_B = -C_A``. ``C S C`` is unchanged under ``C -> -C``, so the join needs nothing from
    ``Port`` -- measured here rather than argued, all four blocks.
    """
    layer_a = np.array([[[4., 4., 1., 1., 2.25, 1.]]])
    layer_b = np.array([[[1., 3., 3., 1., 1., 1.]]])
    period, wavelength, joint = 2.0, 0.6, 2.0
    theta_a = 0.2
    theta_b = float(np.arcsin((np.sin(theta_a) + wavelength / period) / np.sqrt(joint)))
    base = dict(backend='numpy', wavelength=wavelength, period=(period, period),
                truncation_order=(3, 0), tetm='te', layer_algorithm='SMM')
    a = _legacy_wavebit_rcwa(**base, top=1.0, bot=joint, structure=layer_a, thickness=[0.12],
                     theta=theta_a)
    a.solve()
    b = _legacy_wavebit_rcwa(**base, top=joint, bot=2.25, structure=layer_b, thickness=[0.09],
                     theta=theta_b, excitation={-1: 1.0})
    b.solve()
    assert (a.order_window[0][0] - b.order_window[0][0]) % 2 == 1
    joined = _legacy_wavebit_rcwa(**base, top=1.0, bot=2.25, structure=np.concatenate([layer_b, layer_a]),
                          thickness=[0.09, 0.12], theta=theta_a)
    joined.solve()
    cascaded = a.scattering.cascade(b.scattering)
    for got, want in zip(cascaded.blocks, joined.scattering.blocks):
        assert _rel(got, want) < 1e-10


# -------------------------------------------------------------------- 7. complex theta

def test_a_complex_angle_is_unit_at_the_origin_not_at_the_corner():
    """Evanescent incidence: unit at ``x = 0``, and ``exp(Im(kx) P/2)`` larger at the corner.

    A complex ``theta`` makes the incident wave non-uniform along the surface, so *where* it is
    unit sets the field's magnitude. Here the corner sees 7.7 times the origin's amplitude
    (``theta = 0.3 + 0.1j`` in a lossy superstrate). The origin is the choice because it is the
    one point every public coordinate names, and the factor that puts the field there is
    holomorphic in the angle. Amplitudes and efficiencies are ratios and do not see it; both
    field doors agree.
    """
    period, wavelength = 1.6, 0.6
    solver = _legacy_wavebit_rcwa(backend='numpy', period=(period, period), wavelength=wavelength,
                          structure=_EPS_1D, thickness=[0.12], truncation_order=(3, 0),
                          top=2.25 + 1j, bot=2.25 + 0.5j, theta=0.3 + 0.1j, tetm='te',
                          answers='field')
    with pytest.warns(_legacy_wavebit_ComplexIncidenceWarning):
        result = solver.solve()
    origin = _host(result.boundary_fields('top', [0.0], [0.0], part='incoming'))[0, 0, :3]
    assert np.abs(origin - np.array([0.0, 1.0, 0.0])).max() < 1e-12
    corner = _host(result.boundary_fields('top', [-period / 2], [0.0], part='incoming'))[0, 0, 1]
    k_x = 2 * np.pi / wavelength * np.sqrt(2.25 + 1j) * np.sin(0.3 + 0.1j)
    assert abs(corner) == pytest.approx(np.exp(k_x.imag * period / 2), rel=1e-12)
    assert abs(corner) > 5
    points = dict(sample_x=[-0.37, 0.05, 0.61], sample_y=[0.0], sample_z=[0.05])
    assert _rel(solver.reconstruct_field(**points),
                _host(result.reconstruct_field(**points))[..., _SLOTS['te']]) < 1e-10


# ------------------------------------------------------------- 8. the two field doors agree

@pytest.mark.parametrize('scene', ['hexagonal', 'rotated', '1d-odd-centre', '2d-odd-centre'])
def test_the_two_field_doors_agree_on_a_moved_window(scene):
    """``solver.reconstruct_field`` (the scalar) and ``result.reconstruct_field`` (a sign per order).

    The two doors convert the kernels' state differently -- one scalar at the incident ``k``,
    one sign per order label -- so they agree only if both are right. Windows moved by the
    excitation, which is where the incident ``k`` and the window centre's part company, and
    where a parity that ignored ``n`` shows: a skewed cell; a rotated one, whose ``a1`` has a
    ``y`` component; a 1D window whose centre is odd; a 2D one odd along both axes.
    """
    if scene in ('hexagonal', 'rotated'):
        geometry, incidence, points = _plane_scene('hex' if scene == 'hexagonal' else scene)
        geometry = dict(geometry, wavelength=0.9)       # orders (1, 0) and (0, 1) propagate
        solver = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=2.25, answers='field', **geometry,
                              **incidence, excitation={(1, 0): 1.0, (0, 1): 0.5j})
    elif scene == '1d-odd-centre':
        solver = _legacy_wavebit_rcwa(backend='numpy', wavelength=0.6, period=(2.0, 2.0), structure=_EPS_1D,
                              thickness=[0.12], truncation_order=(3, 0), top=1.0, bot=2.25,
                              theta=0.3, tetm='te', excitation={1: 1.0, 2: 0.5j},
                              answers='field')
        points = dict(sample_x=[-0.37, 0.05, 0.61], sample_y=[0.0], sample_z=[0.05])
    else:
        solver = _legacy_wavebit_rcwa(backend='numpy', wavelength=0.4, period=(1.3, 1.7), structure=_RASTER_2D,
                              thickness=[0.15], truncation_order=(2, 2), top=1.0, bot=2.25,
                              theta=0.2, phi=0.3, polarization=(0.6, 0.8j),
                              excitation={(1, 1): 1.0, (2, 1): 0.5j}, answers='field')
        points = dict(sample_x=[-0.4, 0.2], sample_y=[-0.3, 0.5], sample_z=[0.05])
    result = solver.solve()
    if scene not in ('hexagonal', 'rotated'):
        assert solver.order_window[0][0] % 2 == 1, solver.order_window
    if scene == '2d-odd-centre':
        assert solver.order_window[1][0] % 2 == 1, solver.order_window
    path = 'te' if scene == '1d-odd-centre' else 'second_order'
    want = _host(result.reconstruct_field(**points))
    want = want[..., _SLOTS['te']] if path == 'te' else want
    assert _rel(_solver_door(path, solver, points), want) < 1e-10


# --------------------------------------------------------------------------- 9. gradients

def _gradient_scene(backend, theta=0.3, wavelength=1.0):
    return _legacy_wavebit_rcwa(backend=backend, period=(1.7, 1.7), wavelength=wavelength, theta=theta,
                        tetm='te', top=1.0, bot=2.25, thickness=[0.2], structure=_EPS_1D,
                        truncation_order=(3, 0), answers='field')


def _sheared_scene(backend, lattice):
    return _legacy_wavebit_rcwa(backend=backend, lattice=lattice, wavelength=1.0, theta=0.2, phi=0.3,
                        polarization=(0.6, 0.8j), top=1.0, bot=2.25, thickness=[0.15],
                        structure=[{'background': 1.0, 'shapes': [
                            {'type': 'circle', 'cx': 0.1, 'cy': -0.05, 'lx': 0.6,
                             'material': 4.0}]}],
                        truncation_order=(2, 2), factorization='laurent', answers='field')


def _probes_1d(solver):
    """``r_xyz`` of order ``+1`` (odd), and ``E_y`` at a point through both doors."""
    result = solver.solve()
    (cu, hu), _ = solver.order_window
    point = dict(sample_x=[0.37], sample_y=[0.0], sample_z=[0.1])
    return (result.r_xyz[0, 1 - cu + hu, 1],
            result.reconstruct_field(**point)[0, 0, 0, 1],
            solver.reconstruct_field(**point)[0, 0, 0, 0])


def _probes_sheared(solver):
    """``r_xyz`` of order ``(1, 0)`` (odd), and ``E_x`` at a point through both doors."""
    result = solver.solve()
    (cu, hu), (cv, hv) = solver.order_window
    return (result.r_xyz[hv - cv, 1 - cu + hu, 1],
            result.reconstruct_field(sample_x=3, sample_y=3, sample_z=[0.1])[0, 1, 2, 0],
            solver.reconstruct_field(sample_x=3, sample_y=3, sample_z=[0.1])[0, 0, 1, 2, 0])


def _ad_against_fd(jax, probes, build, x0, step):
    """JAX reverse mode on the real and imaginary parts, against a NumPy central difference."""
    import jax.numpy as jnp

    def parts(x):
        values = probes(build('jax', x))
        return jnp.stack([f(v) for v in values for f in (jnp.real, jnp.imag)])

    ad = np.asarray(jax.jacrev(parts)(x0))
    plus = np.array([complex(_host(v)) for v in probes(build('numpy', x0 + step))])
    minus = np.array([complex(_host(v)) for v in probes(build('numpy', x0 - step))])
    fd = (plus - minus) / (2 * step)
    ad = ad[0::2] + 1j * ad[1::2]
    return ad, fd


@pytest.mark.parametrize('variable', ['theta', 'wavelength'])
def test_the_centre_reference_differentiates_through_the_angle(variable):
    """d/d``theta`` and d/d``wavelength`` of an odd order's ``r_xyz`` and of the field, both doors.

    The solver door carries ``exp(-1j k_inc . o)``, which depends on both; the sign on an order
    is a constant. JAX AD against a NumPy central difference, to 1e-6 relative.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    if variable == 'theta':
        build = lambda backend, x: _gradient_scene(backend, theta=x)
        x0 = 0.3
    else:
        build = lambda backend, x: _gradient_scene(backend, wavelength=x)
        x0 = 1.0
    ad, fd = _ad_against_fd(jax, _probes_1d, build, x0, 1e-6)
    for name, a, f in zip(('r_xyz', 'result door', 'solver door'), ad, fd):
        assert abs(f) > 1e-3, name
        assert abs(a - f) < 1e-6 * abs(f), (name, a, f)


@pytest.mark.parametrize('entry', [(0, 0), (0, 1)], ids=['a1x', 'a2x'])
def test_the_centre_reference_differentiates_through_the_cell(entry):
    """d/d``a1x`` and d/d the shear ``a2x`` on a skewed cell -- the centre moves with the basis.

    The shape path with Laurent's rule, where the structure is smooth in the basis and a point
    difference is meaningful (a raster's normal field would staircase it).
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    import jax.numpy as jnp
    unit = np.zeros((2, 2))
    unit[entry] = 1.0

    def build(backend, t):
        lattice = (jnp.asarray(_SHEAR) + t * jnp.asarray(unit) if backend == 'jax'
                   else _SHEAR + t * unit)
        return _sheared_scene(backend, lattice)

    ad, fd = _ad_against_fd(jax, _probes_sheared, build, 0.0, 1e-6)
    for name, a, f in zip(('r_xyz', 'result door', 'solver door'), ad, fd):
        assert abs(f) > 1e-3, name
        assert abs(a - f) < 1e-6 * abs(f), (name, a, f)


def test_torch_differentiates_the_solver_door_through_the_angle():
    """One Torch ``backward`` on ``theta`` through the solver door's scalar."""
    torch = pytest.importorskip('torch')
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    value = _probes_1d(_gradient_scene('torch', theta=theta))[2]
    grad_re, = torch.autograd.grad(value.real, theta, retain_graph=True)
    grad_im, = torch.autograd.grad(value.imag, theta)
    h = 1e-6
    fd = (complex(_host(_probes_1d(_gradient_scene('numpy', theta=0.3 + h))[2]))
          - complex(_host(_probes_1d(_gradient_scene('numpy', theta=0.3 - h))[2]))) / (2 * h)
    got = float(grad_re) + 1j * float(grad_im)
    assert abs(got - fd) < 1e-6 * abs(fd), (got, fd)


# ---------------------------------------------------------------------- 10. backends and jit

@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_backends_agree_on_odd_orders_and_both_doors(backend):
    """JAX and Torch against NumPy: an odd order's ``r_s`` and ``r_xyz``, and both field doors.

    The solver door by counts and at explicit points written as a list, which reached Torch in
    single precision until the sampling helpers converted at the solve's own.
    """
    pytest.importorskip(backend)
    if backend == 'jax':
        _legacy_wavebit_enable_x64()
    excitation = {0: 1.0, 1: 0.5 - 0.3j}
    out = []
    for b in ('numpy', backend):
        solver = _legacy_wavebit_rcwa(backend=b, period=(1.7, 1.7), wavelength=1.0, theta=0.3, tetm='te',
                              top=1.0, bot=2.25, thickness=[0.2], structure=_EPS_1D,
                              truncation_order=(3, 0), answers='field', excitation=excitation)
        result = solver.solve()
        points = dict(sample_x=[-0.3, 0.2], sample_y=[0.0], sample_z=[0.1])
        out.append((_host(result.r_s), _host(result.r_xyz),
                    _host(result.reconstruct_field(**points)),
                    _host(solver.reconstruct_field(sample_x=5, sample_y=1, sample_z=3)),
                    _host(solver.reconstruct_field(**points))))
    names = ('r_s', 'r_xyz', 'result door', 'solver door, counts', 'solver door, points')
    for name, want, got in zip(names, *out):
        assert _rel(got, want) < 1e-12, name


def test_a_result_that_crossed_jit_reconstructs_the_centre_field():
    """The jitted ``solve_field`` -- the solver is the argument -- against the eager one.

    A mixed-parity excitation, so the per-order sign the result's field adapter applies is not
    uniform. The window it is read from rides in the adapter's aux data; a result rebuilt
    without it would fail here. Negative control: the same scene with the odd weight negated
    must not give the same field, or the comparison would pass with the sign ignored.
    """
    pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    points = dict(sample_x=4, sample_y=1, sample_z=3)

    def scene(weights):
        solver = _legacy_wavebit_rcwa(backend='jax', period=(1.7, 1.7), wavelength=1.0, theta=0.3, phi=0.2,
                              psi=0.4, top=1.0, bot=2.25, thickness=[0.2], structure=_EPS_1D,
                              truncation_order=(3, 0), answers='field', excitation=weights)
        return solver

    solver = scene({0: 1.0, 1: 0.5 - 0.3j})
    jitted, jitted_field = solver.solve_field(**points, geometry=solver.freeze_geometry())
    eager, eager_field = scene({0: 1.0, 1: 0.5 - 0.3j}).solve_field_eager(**points)
    assert _rel(jitted_field, eager_field) < 1e-12
    public = dict(sample_x=[-0.3, 0.2], sample_y=[0.1], sample_z=[0.1])
    want = _host(eager.reconstruct_field(**public))
    assert _rel(jitted.reconstruct_field(**public), want) < 1e-12
    _, flipped = scene({0: 1.0, 1: -0.5 + 0.3j}).solve_field_eager(**points)
    assert _rel(flipped, eager_field) > 1e-2
