"""``r_xyz`` / ``t_xyz``: each outgoing order's E as a Cartesian vector, per unit incidence.

The amplitudes ``r_s`` / ``r_p`` are quoted in each order's own s/p frame, which on the
``ky == 0`` row is ``sign(kx) * y_hat`` and so turns over where an order's ``kx`` crosses zero.
``r_xyz = r_s * s_hat + r_p * p_hat_ref`` carries the same wave with no frame in it: continuous
there, and one meaning on every path. It is sliced out of the boundary spectra
``field_result.ordinary_spectra`` already builds, so the checks here have to come from outside
that arithmetic or they would be restating it:

1. **an independent spelling** of the formula -- the frame from ``arctan2`` and the order grid,
   ``kz`` read from ``boundary_kz`` (never respelled), on every path including a 60-degree cell;
2. **the reconstructed field** -- decomposed into orders at the lit and far faces, the incident
   wave written out and taken off, and what is left is ``r_xyz`` / ``t_xyz``; ``E_z`` by
   transversality, since the normal component is not continuous across the face;
3. **continuity** across an order's own ``kx = 0`` on the ``ky == 0`` row, where ``r_s`` steps,
   and one vector on every backend for an order sitting at an exact zero, whose frame rounding
   picks;
4. **the paths agree** -- pure 1D, conical at ``phi = 0`` and 2D on the same grating;
5. **power** -- ``|r_xyz|**2 Re(kz_m)/Re(kz_inc)`` is ``R`` on a lossless stack;

plus the plumbing a new result field needs: the ``answers`` rung, ``jit`` and gradients on
the traced backends, and ``order()`` on a widened, off-centre window.
"""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

from wavebit._common.conventions import boundary_kz
from wavebit.rcwa._engine.solve_2d import order_grid_kxy

# The pure-1D investigation's grating: absorbing, x-asymmetric, 13 orders straddling kx = 0.
_ROW_TOP = np.array([2.2 + 0.3j] * 5 + [1] * 3 + [3] * 2 + [1] * 6) ** 2
_ROW_BOT = np.array([3] * 7 + [1.5] * 3 + [1] * 6, dtype=complex) ** 2
_1D = dict(structure=np.stack([_ROW_BOT, _ROW_TOP])[:, None, :], thickness=[0.15, 0.22],
           period=[1.6], wavelength=0.6, top=1.0, bot=2.25, truncation_order=[6, 0])
_RASTER_2D = np.random.default_rng(5).uniform(1, 4, (1, 6, 6)) ** 2
_2D = dict(structure=_RASTER_2D, thickness=[0.2], period=[1.6, 1.8], wavelength=0.6,
           top=1.0, bot=2.25, truncation_order=[3, 2])
_HEX_P = 1.6
_HEX = np.column_stack(([_HEX_P, 0.0], [_HEX_P / 2, _HEX_P * np.sqrt(3) / 2]))


def _hex_rod(n=24, radius=0.45):
    """A rod at the centre of the 60-degree cell, on the parallelogram ``meshgrid_xy`` samples."""
    X, Y = (np.asarray(c) for c in _legacy_wavebit.meshgrid_xy((n, n), lattice=_HEX))
    a1, a2 = _HEX[:, 0], _HEX[:, 1]
    inside = np.zeros(X.shape, dtype=bool)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            shift = i * a1 + j * a2
            inside |= (X - shift[0]) ** 2 + (Y - shift[1]) ** 2 <= radius ** 2
    cell = np.full((1, n, n), 1.0)
    cell[0][inside] = 5.76
    return cell


def _scene(path, side='top', theta=0.3, answers='field', backend='numpy', **over):
    """One solve per path. Each coupled one is lit by a mixed pair, so s and p both carry."""
    common = dict(backend=backend, theta=theta, incidence_side=side, answers=answers)
    if path in ('te', 'tm'):
        kw = dict(_1D, tetm=path)
    elif path == 'conical':
        kw = dict(_1D, phi=0.4, psi=0.7)
    elif path in ('2d', 'first_order'):
        kw = dict(_2D, phi=0.4, psi=0.7)
    elif path == 'hex':
        kw = dict(structure=_hex_rod(), thickness=[0.25], lattice=_HEX, wavelength=0.9,
                  top=1.0, bot=2.25, truncation_order=[2, 2], phi=0.4, psi=0.7,
                  factorization='nvf')
    else:
        raise ValueError(path)
    kw.update(common)
    kw.update(over)
    solver = _legacy_wavebit.rcwa(**kw)
    if path == 'first_order':
        solver.formulation = 'first_order'
    return solver, solver.solve()


def _host(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).astype(np.complex128)


def _geometry(solver):
    """The order grid, both media and the lit-side sign -- read off the solver, nothing more.

    ``d = -1`` lit from above (the incident wave travels down), ``+1`` from below.
    """
    kx, ky = order_grid_kxy(np, *(_host(k) for k in solver.order_kxy()))
    top, bot = complex(_host(solver.n_top)), complex(_host(solver.n_bot))
    n_lit, n_far = (top, bot) if solver.incidence_side == 'top' else (bot, top)
    d = -1.0 if solver.incidence_side == 'top' else 1.0
    kz_lit = boundary_kz(np, n_lit ** 2, kx ** 2 + ky ** 2)
    kz_far = boundary_kz(np, n_far ** 2, kx ** 2 + ky ** 2)
    return kx, ky, n_lit, n_far, kz_lit, kz_far, d


def _frame(solver, kx, ky):
    """``(cos, sin)`` of each order's azimuth, spelled independently of the solver.

    ``arctan2`` of the order grid, with the origin -- where the in-plane wavevector has no
    direction -- taken from ``phi``, as the library's rule does. Not ``arctan2`` there: it
    reads ``(0, -0.0)`` as ``pi`` and would report a disagreement that is this function's.
    """
    phi = 0.0 if solver.phi is None else float(np.real(_host(solver.phi)))
    angle = np.arctan2(np.real(ky), np.real(kx))
    angle = np.where((kx == 0) & (ky == 0), phi, angle)
    return np.cos(angle), np.sin(angle)


PATHS = ['te', 'tm', 'conical', '2d', 'first_order', 'hex']


# ----------------------------------------------------------------- 1. an independent spelling

@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path', PATHS)
def test_the_vector_is_the_amplitudes_in_the_public_frame(path, side):
    """``r_xyz = r_s s_hat + r_p p_hat_ref`` and ``t_xyz = t_s s_hat + t_p p_hat_trn``.

    ``s_hat = (-sin, cos, 0)``; ``p_hat``'s in-plane part lies along ``k_hat_par`` with length
    ``cos(theta_m)`` in the medium the wave travels in, ``+`` for the reflected wave and ``-``
    for the transmitted one, and its z part is fixed by ``p_hat . k = 0``
    (``docs/reference/common/conventions.md``). The 60-degree cell is where an order axis is not x, so a u/v
    slip in the slicing shows there and nowhere else.

    Every incidence the solve answers is checked against its **own** amplitudes: the coupled
    scenes are lit by a mixed pair, so ``incident``, ``s_inc`` and ``p_inc`` are three
    different waves, and a vector handed to the wrong variant is caught here.
    """
    solver, result = _scene(path, side, answers='amplitude')
    kx, ky, n_lit, n_far, kz_lit, kz_far, d = _geometry(solver)
    c, s = _frame(solver, kx, ky)
    rho = kx * c + ky * s
    zero = np.zeros_like(kx)
    s_hat = np.stack([-s, c, zero], axis=-1)
    p_ref = np.stack([kz_lit * c / n_lit, kz_lit * s / n_lit, d * rho / n_lit], axis=-1)
    p_trn = np.stack([-kz_far * c / n_far, -kz_far * s / n_far, d * rho / n_far], axis=-1)
    subs = [sub for sub in ('incident', 's_inc', 'p_inc') if getattr(result, sub) is not None]
    if path not in ('te', 'tm'):
        assert subs == ['incident', 's_inc', 'p_inc'], subs
    wants = {}
    for sub in subs:
        res = getattr(result, sub)
        r_s, r_p, t_s, t_p = (_host(getattr(res, a)) for a in ('r_s', 'r_p', 't_s', 't_p'))
        r_want = r_s[..., None] * s_hat + r_p[..., None] * p_ref
        t_want = t_s[..., None] * s_hat + t_p[..., None] * p_trn
        r_got, t_got = _host(res.r_xyz), _host(res.t_xyz)
        assert r_got.shape == r_s.shape + (3,) and t_got.shape == t_s.shape + (3,)
        assert np.abs(r_want).max() > 1e-2 and np.abs(t_want).max() > 1e-2
        for name, got, want in (('r_xyz', r_got, r_want), ('t_xyz', t_got, t_want)):
            gap = np.abs(got - want).max() / np.abs(want).max()
            assert gap < 1e-13, f'{path}/{side}/{sub}: {name} off by {gap:.2e}'
        wants[sub] = r_want
    # The premise of the per-variant check: the three waves are different waves.
    for a, b in (('incident', 's_inc'), ('incident', 'p_inc'), ('s_inc', 'p_inc')):
        if a in wants and b in wants:
            assert np.abs(wants[a] - wants[b]).max() > 1e-2 * np.abs(wants[a]).max(), (a, b)


# ------------------------------------------------------------------ 2. the field says so

def _face_orders(result, solver, z, kx, ky):
    """Every retained order's ``(Ex, Ey)`` on the plane at height ``z``, and the fit residual.

    Points on a rectangular Cartesian grid, twice as many as orders each way, so the fit is
    overdetermined and a missing order shows as a residual. Both the request and the plane-wave
    basis are in public coordinates, measured from the centre of the cell -- where the
    amplitudes are referenced since 2026-09-27 -- so no half-cell appears anywhere. (Until then
    the basis was written from the corner, the Fourier origin, and each request carried
    ``(a1 + a2)/2`` back off.)
    """
    lattice = np.real(_host(solver.lattice.columns))
    span_x, span_y = lattice[0].sum(), max(lattice[1].sum(), 1.0)
    xs = np.linspace(-0.5 * span_x, 0.5 * span_x, 2 * kx.shape[1] + 3, endpoint=False)
    ys = np.linspace(-0.5 * span_y, 0.5 * span_y, 2 * kx.shape[0] + 3, endpoint=False)
    cell = _host(result.reconstruct_field(sample_x=xs, sample_y=ys, sample_z=np.array([z])))
    cell = cell.reshape(ys.size, xs.size, 6)
    k0 = 2 * np.pi / solver.wavelength
    X, Y = np.meshgrid(xs, ys)
    basis = np.exp(1j * k0 * (X.reshape(-1, 1) * kx.reshape(1, -1)
                              + Y.reshape(-1, 1) * ky.reshape(1, -1)))
    ex, ey = (np.linalg.lstsq(basis, cell[:, :, i].reshape(-1), rcond=None)[0].reshape(kx.shape)
              for i in (0, 1))
    return ex, ey, np.abs(basis @ ex.reshape(-1) - cell[:, :, 0].reshape(-1)).max()


def _incident_vector(solver, kx, ky, n_lit, kz_lit, d):
    """The incident wave's E on its own order, written out: ``a_s s_hat + a_p p_hat_inc``.

    ``p_hat_inc`` leans like a transmitted wave (in-plane ``-cos(theta)``), and the pair is the
    one each scene is lit with, in the incident order's own frame.
    """
    c, s = _frame(solver, kx, ky)
    i = tuple(n // 2 for n in kx.shape)
    rho = kx[i] * c[i] + ky[i] * s[i]
    s_hat = np.array([-s[i], c[i], 0.0])
    p_hat = np.array([-kz_lit[i] * c[i] / n_lit, -kz_lit[i] * s[i] / n_lit, d * rho / n_lit])
    if solver.tetm is not None:
        a_s, a_p = (1.0, 0.0) if solver.tetm == 'te' else (0.0, 1.0)
    else:
        psi = float(np.real(_host(solver.psi)))
        a_s, a_p = np.sin(psi), np.cos(psi)
    return i, a_s * s_hat + a_p * p_hat


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', 'hex'])
def test_the_vector_is_the_reconstructed_field(path, side):
    """Decompose the field at each face into orders; what is left is ``r_xyz`` / ``t_xyz``.

    Transverse E is continuous across an interface, so at the lit face order ``m``'s
    coefficient is the incident wave (its own order only) plus the reflected one, and at the far
    face it is the transmitted wave. The reconstruction runs through the kernels' per-layer
    state and shares nothing with ``ordinary_spectra``, which is what makes this the check
    that the reuse is right rather than a restatement of it. ``E_z`` is not continuous across
    the face, so it is held to transversality instead: ``k . E = 0`` for each outgoing wave.
    """
    solver, result = _scene(path, side)
    res = result.incident
    kx, ky, n_lit, n_far, kz_lit, kz_far, d = _geometry(solver)
    top = float(np.sum(np.real(_host(solver.thickness))))
    lit_z, far_z = (top, 0.0) if side == 'top' else (0.0, top)

    ex, ey, residual = _face_orders(result, solver, lit_z, kx, ky)
    assert residual < 1e-11, f'the lit face is not a sum of the retained orders ({residual:.1e})'
    centre, e_inc = _incident_vector(solver, kx, ky, n_lit, kz_lit, d)
    ex[centre] -= e_inc[0]
    ey[centre] -= e_inc[1]
    r = _host(res.r_xyz)
    scale = np.abs(r).max()
    assert np.abs(ex - r[..., 0]).max() < 1e-11 * scale, 'reflected Ex'
    assert np.abs(ey - r[..., 1]).max() < 1e-11 * scale, 'reflected Ey'

    ex, ey, residual = _face_orders(result, solver, far_z, kx, ky)
    assert residual < 1e-11
    t = _host(res.t_xyz)
    scale = np.abs(t).max()
    assert np.abs(ex - t[..., 0]).max() < 1e-11 * scale, 'transmitted Ex'
    assert np.abs(ey - t[..., 1]).max() < 1e-11 * scale, 'transmitted Ey'

    # The reflected wave travels against the incident one in z, the transmitted wave with it.
    k_ref = np.stack([kx, ky, -d * kz_lit], axis=-1)
    k_trn = np.stack([kx, ky, d * kz_far], axis=-1)
    assert np.abs(np.sum(k_ref * r, axis=-1)).max() < 1e-12 * np.abs(r).max() * np.abs(k_ref).max()
    assert np.abs(np.sum(k_trn * t, axis=-1)).max() < 1e-12 * np.abs(t).max() * np.abs(k_trn).max()
    assert np.abs(r[..., 2]).max() > 1e-3 or path in ('te',), 'Ez carries nothing to check'


# ----------------------------------------------------------------------- 3. continuity

@pytest.mark.parametrize('path', ['te', 'tm', 'conical_s', 'conical_p'])
def test_the_vector_is_continuous_where_the_amplitude_turns_over(path):
    """Across order ``-1``'s own ``kx = 0`` on the ``ky == 0`` row: the frame steps, E does not.

    ``kx(-1) = sin(theta) - lambda/P`` crosses zero at ``theta = asin(lambda/P)``. There ``r_s``
    (or ``r_p``) changes sign, because ``s_hat`` goes from ``-y`` to ``+y`` -- asserted, so the
    comparison has a premise. ``r_xyz`` must step as an equal step either side of the crossing
    does: the physical field does not know where the frame is singular. The crossing step
    measures 0.9999 of the control on all four, so the bound is 1.2.

    At ``lambda = 0.62``, not the grating's 0.6. There ``n_bot * P / lambda = 4``, so orders
    ``+3`` and ``-5`` graze the substrate at the same angle, and that branch point makes the
    crossing step 2.0x (TE) and 1.6x (TM) the larger control step -- a bound that passes it
    passes a real jump of the same size.
    """
    wavelength = 0.62
    theta_c = float(np.arcsin(wavelength / _1D['period'][0]))
    if path.startswith('conical'):
        over = dict(phi=0.0, psi=np.pi / 2 if path.endswith('s') else 0.0)
        kw = dict(_1D, wavelength=wavelength, **over)
    else:
        kw = dict(_1D, wavelength=wavelength, tetm=path)
    name = 'r_s' if path in ('te', 'conical_s') else 'r_p'
    steps = [_legacy_wavebit.rcwa(backend='numpy', theta=theta_c + dt, answers='amplitude', **kw).solve().incident
             for dt in (-3e-5, -1e-5, 1e-5, 3e-5)]
    at = (0, _1D['truncation_order'][0] - 1)
    a = [complex(_host(getattr(s, name))[at]) for s in steps]
    v = [_host(s.r_xyz)[at] for s in steps]
    assert abs(a[2] + a[1]) < 0.1 * abs(a[1]) and abs(a[1]) > 1e-2, f'{name} did not turn over: {a}'
    control = max(np.abs(v[1] - v[0]).max(), np.abs(v[3] - v[2]).max())
    crossing = np.abs(v[2] - v[1]).max()
    assert crossing < 1.2 * control, (
        f'r_xyz stepped {crossing:.2e} against a control of {control:.2e}')


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_an_order_at_exact_kx_zero_reads_one_vector_on_every_backend(tetm):
    """On the singular point itself rounding picks the frame; ``r_xyz`` does not care which.

    A direct ``kx = 2*pi/P`` puts order ``-1`` at ``kx = 0`` in exact arithmetic. Which side of
    zero it is computed on is a matter of rounding and differs by backend (NumPy and JAX give
    ``+0.0`` here and Torch ``-5.55e-17``, ``docs/reference/common/conventions.md``), and ``r_s`` / ``r_p``
    follow that side. So the amplitude is compared up to exactly the sign its own ``kx`` gives,
    and the vector without one. Nothing here depends on which side any backend lands on.
    """
    pytest.importorskip('jax')
    pytest.importorskip('torch')
    _legacy_wavebit.enable_x64()
    backends = ('numpy', 'jax', 'torch')
    name = 'r_s' if tetm == 'te' else 'r_p'
    kw = dict(_1D, kx=2 * np.pi / _1D['period'][0], tetm=tetm, answers='amplitude')
    at = (0, _1D['truncation_order'][0] - 1)
    seen = {}
    for backend in backends:
        solver = _legacy_wavebit.rcwa(backend=backend, **kw)
        res = solver.solve().incident
        kx = complex(_host(solver.order_kxy()[0]).ravel()[at[1]])
        assert abs(kx) < 1e-15, kx
        seen[backend] = (np.real(kx) < 0, complex(_host(getattr(res, name))[at]),
                         _host(res.r_xyz)[at])
    flip0, a0, v0 = seen['numpy']
    assert abs(a0) > 1e-2
    for backend, (flip, a, v) in seen.items():
        sign = -1.0 if flip != flip0 else 1.0
        assert abs(a - sign * a0) < 1e-7, (backend, a, a0)
        assert np.abs(v - v0).max() < 1e-7, (backend, v, v0)


# ---------------------------------------------------------------------- 4. the paths agree

@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_one_grating_three_paths_one_vector(tetm, side):
    """Pure 1D, conical at ``phi = 0`` and the 2D path on the same grating, per order.

    The 2D path gets the grating as two identical rows; with ``truncation_order[1] = 0`` it
    retains the same single row of orders. Three formulations, three field builders, one
    wave -- so one vector, to 1e-12.
    """
    kw = dict(_1D, backend='numpy', theta=0.3, incidence_side=side, answers='amplitude')
    pure = _legacy_wavebit.rcwa(tetm=tetm, **kw).solve().incident
    pol = (1, 0) if tetm == 'te' else (0, 1)
    conical = _legacy_wavebit.rcwa(phi=0.0, polarization=pol, **kw).solve().incident
    two_d = _legacy_wavebit.rcwa(phi=0.0, polarization=pol,
                         **dict(kw, structure=np.repeat(_1D['structure'], 2, axis=1),
                                period=[1.6, 1.6])).solve().incident
    for name in ('r_xyz', 't_xyz'):
        a, b, c = (_host(getattr(x, name)) for x in (pure, conical, two_d))
        assert a.shape == b.shape == c.shape, (a.shape, b.shape, c.shape)
        assert np.abs(a - b).max() < 1e-12, f'{name}: pure vs conical'
        assert np.abs(a - c).max() < 1e-12, f'{name}: pure vs 2D'


# --------------------------------------------------------------------------- 5. power

@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path', ['te', 'tm', '2d'])
def test_the_vector_carries_the_power(path, side):
    """``R_m = |r_xyz_m|**2 Re(kz_m) / Re(kz_inc)`` on a lossless stack, and ``T`` likewise.

    ``|p_hat|**2 = (|kz|**2 + rho**2)/n**2 = 1`` for a propagating order in a lossless medium,
    so the vector's norm is the wave's amplitude and its flux follows. Lossless and
    non-magnetic throughout, so the identity has no admittance in it to get wrong.
    """
    lossless = dict(structure=np.real(_1D['structure']) if path != '2d' else _RASTER_2D)
    solver, result = _scene(path, side, answers='amplitude', **lossless)
    res = result.incident
    kx, ky, n_lit, n_far, kz_lit, kz_far, d = _geometry(solver)
    centre = tuple(n // 2 for n in kx.shape)
    r, t = _host(res.r_xyz), _host(res.t_xyz)
    R, T = np.real(_host(res.R)), np.real(_host(res.T))
    up = np.abs(np.imag(kz_lit)) < 1e-12
    down = np.abs(np.imag(kz_far)) < 1e-12
    assert up.sum() >= 3 and down.sum() >= 3
    power_r = np.sum(np.abs(r) ** 2, axis=-1) * np.real(kz_lit) / np.real(kz_lit[centre])
    power_t = np.sum(np.abs(t) ** 2, axis=-1) * np.real(kz_far) / np.real(kz_lit[centre])
    assert np.abs(power_r - R)[up].max() < 1e-12
    assert np.abs(power_t - T)[down].max() < 1e-12
    assert abs(R.sum() + T.sum() - 1.0) < 1e-12, 'the stack was meant to be lossless'


# ----------------------------------------------------------------------- the plumbing

@pytest.mark.parametrize('path', ['te', 'conical', '2d'])
def test_the_answers_rung_keeps_it_exactly_where_it_keeps_r_s(path):
    """``None`` at ``'efficiency'``; bit-equal between ``'amplitude'`` and ``'field'``.

    The two upper rungs run the same three-column solve, so a vector that differed between
    them would be a vector computed from something other than the amplitudes.
    """
    _, low = _scene(path, answers='efficiency')
    assert low.incident.r_xyz is None and low.incident.t_xyz is None
    _, mid = _scene(path, answers='amplitude')
    _, high = _scene(path, answers='field')
    for sub in ('incident', 's_inc', 'p_inc'):
        a, b = getattr(mid, sub), getattr(high, sub)
        if a is None:
            assert b is None
            continue
        for name in ('r_xyz', 't_xyz'):
            x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
            assert x.tobytes() == y.tobytes(), (sub, name)
    assert mid.r_xyz is mid.incident.r_xyz


@pytest.mark.parametrize('path, excitation, over', [
    ('conical', {0: 1.0, 2: 0.5 - 0.3j}, dict(period=[2.0])),
    ('2d', {(0, 0): 1.0, (-1, 2): 0.5 - 0.3j}, {}),
])
def test_order_reads_the_vector_off_a_widened_off_centre_window(path, excitation, over):
    """``result.order(m, n).r_xyz`` is the length-3 vector at order ``(m, n)``, not a component.

    The window is widened and moved off zero by an off-centre ``excitation`` -- on the 2D scene
    along both axes -- so an index computed from the declared ``truncation_order``, a picker
    that took the trailing 3 for the order axis, or a row index with the wrong sign or without
    the centre lands on a plausible wrong number. Every cell of the window is read both ways,
    and the scalar picker has to land on the same cell as the vector one.
    """
    solver, result = _scene(path, answers='amplitude', excitation=excitation, **over)
    (cx, hx), (cy, hy) = solver.order_window
    declared = _1D['truncation_order'] if path == 'conical' else _2D['truncation_order']
    assert cx != 0 and hx > declared[0]
    if path == '2d':
        assert cy != 0 and hy > declared[1], solver.order_window
    raw, raw_t, raw_s = (_host(getattr(result, name)) for name in ('r_xyz', 't_xyz', 'r_s'))
    assert raw.shape == (2 * hy + 1, 2 * hx + 1, 3), raw.shape
    for n in range(cy - hy, cy + hy + 1):
        for m in range(cx - hx, cx + hx + 1):
            view = result.order(m, n)
            cell = (n - cy + hy, m - cx + hx)
            assert np.asarray(view.r_xyz).shape == (3,)
            assert np.array_equal(_host(view.r_xyz), raw[cell]), (m, n)
            assert np.array_equal(_host(view.t_xyz), raw_t[cell]), (m, n)
            assert _host(view.r_s) == raw_s[cell], (m, n)
    # The premise: neighbouring rows and columns hold different vectors, so a slip shows.
    assert np.abs(np.diff(raw, axis=1)).max() > 1e-3
    if path == '2d':
        assert np.abs(np.diff(raw, axis=0)).max() > 1e-3
    assert 'r_xyz=' in repr(result.order(cx, cy)) and 't_xyz=' in repr(result.order(cx, cy))


def test_jax_carries_the_vector_through_jit_and_grad():
    """Through a ``jit`` boundary the solver crosses, and differentiated against a quotient.

    ``ResultSub`` is a pytree whose children are positional, so a field missing from its list
    is dropped on the way out of ``jit`` and the next one slides into its slot -- which is why
    the jitted function returns the whole result, not an array read off it inside the trace
    (that would never flatten a ``ResultSub`` at all, and could not see the loss).
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    import jax.numpy as jnp

    solver, eager = _scene('conical', answers='amplitude', backend='jax')

    @jax.jit
    def solved(s):
        return s.solve()

    crossed = solved(solver)
    for sub in ('incident', 's_inc', 'p_inc'):
        for name in ('r_xyz', 't_xyz'):
            got, want = getattr(getattr(crossed, sub), name), getattr(getattr(eager, sub), name)
            assert got is not None, (sub, name)
            np.testing.assert_allclose(_host(got), _host(want), atol=1e-13, rtol=0,
                                       err_msg=f'{sub}.{name}')

    geometry = solver.freeze_geometry()

    def power(d0):
        res = solver.solve(thickness=jnp.stack([d0, jnp.asarray(0.22)]), geometry=geometry,
                           auto_detune=False)
        return jnp.abs(res.incident.r_xyz[0, 5, 0]) ** 2

    grad = float(jax.grad(power)(jnp.asarray(0.15)))
    h = 1e-6
    fd = (float(power(jnp.asarray(0.15 + h))) - float(power(jnp.asarray(0.15 - h)))) / (2 * h)
    assert abs(grad) > 1e-4
    assert grad == pytest.approx(fd, rel=3e-7, abs=1e-9)


def test_torch_differentiates_the_vector():
    torch = pytest.importorskip('torch')
    solver, _ = _scene('conical', answers='amplitude', backend='torch')

    def power(d0):
        res = solver.solve(thickness=torch.stack([d0, d0.new_tensor(0.22)]), auto_detune=False)
        return torch.abs(res.incident.r_xyz[0, 5, 0]) ** 2

    d0 = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    grad, = torch.autograd.grad(power(d0), d0)
    h = 1e-6
    fd = (float(power(d0.detach() + h)) - float(power(d0.detach() - h))) / (2 * h)
    assert abs(float(grad)) > 1e-4
    assert float(grad) == pytest.approx(fd, rel=3e-7, abs=1e-9)
