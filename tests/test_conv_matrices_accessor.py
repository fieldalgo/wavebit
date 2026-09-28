"""
`RCWAEngine.eps_conv_matrices` — the public way to see what a modeling path built.

`solve_conv` has always *taken* these five matrices; nothing returned them, so a caller who
wanted to compare two modeling paths, check a geometry, or build a path of their own had to reach
into `_core` and reassemble the quantity from the doubled reciprocal grid, reproducing its layout
and the order-index sign convention by hand. That fails silently when guessed — three separate
attempts produced three wrong numbers in one day, all of them from someone doing something
legitimate.

These tests pin the public centre frame, the exact round trip into the solver, and the
common matrix layout across modeling paths. The internal solve uses corner-frame operators.
"""
import numpy as np
import pytest

import wavebit

P = 1.0
BASE = dict(period=[P, P], thickness=[0.4], wavelength=0.8, theta=0.0, phi=0.0,
            top=1.0, bot=2.25, truncation_order=[4, 4])
CIRCLE = dict(type='circle', cx=0.0, cy=0.0, lx=0.6, material=9.0)   # eps 9 is the old n = 3


def disc_raster(N, r=0.30):
    j = (np.arange(N) + 0.5) / N * P
    X, Y = np.meshgrid(j, j, indexing='xy')
    return np.where(r - np.hypot(X - 0.5, Y - 0.5) > 0, 3.0, 1.0)[None, :, :]


def test_it_is_exactly_what_solve_conv_takes():
    """The round trip is the contract: feeding the result back reproduces solve()."""
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=[CIRCLE])], **BASE)
    direct = solver.solve().incident.T
    relayed = solver.solve_conv(*solver.eps_conv_matrices()).incident.T
    assert np.abs(np.asarray(direct) - np.asarray(relayed)).max() == 0.0


def test_every_modeling_path_answers_with_the_same_shaped_object():
    """The point of the accessor: one quantity, comparable across paths that share nothing."""
    n_orders = (2 * 4 + 1) ** 2
    paths = {
        'raster+nvf': wavebit.rcwa(structure=disc_raster(96), factorization='nvf', **BASE),
        'smooth': wavebit.rcwa(structure=[dict(background=1.0, shapes=[CIRCLE])], **BASE),
        'boundary': wavebit.rcwa(structure=[dict(background=1.0, shapes=[
            dict(type='rectangle', cx=-0.10, cy=-0.10, lx=0.34, ly=0.34, material=4.0),
            dict(type='rectangle', cx=0.06, cy=0.06, lx=0.34, ly=0.34, angle=0.3,
                 material=9.0)])], **BASE),
    }
    for name, solver in paths.items():
        epx, epy, epz_i, epxy, epyx = solver.eps_conv_matrices()
        for tag, m in (('epx', epx), ('epy', epy), ('epz_i', epz_i)):
            assert np.asarray(m).shape == (1, n_orders, n_orders), (name, tag, np.asarray(m).shape)
            assert np.isfinite(np.asarray(m)).all(), (name, tag)
        assert epxy is not None and epyx is not None, name       # all three are NVF paths


def test_a_frozen_geometry_is_accepted_the_same_way_solve_accepts_one():
    solver = wavebit.rcwa(structure=disc_raster(96), factorization='nvf', **BASE)
    eager = solver.eps_conv_matrices()
    replayed = solver.eps_conv_matrices(geometry=solver.freeze_geometry())
    for a, b in zip(eager, replayed):
        if a is None:
            assert b is None
        else:
            assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-14


def test_it_does_not_solve():
    """Cheaper than solve by the whole eigenproblem — otherwise there is no reason to have
    it separately, and a sweep over geometries would pay for a solve it discards."""
    import time
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=[CIRCLE])], **dict(BASE, truncation_order=[8, 8]))
    solver.eps_conv_matrices(), solver.solve()             # warm

    t = time.perf_counter()
    for _ in range(3):
        solver.eps_conv_matrices()
    build = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(3):
        solver.solve()
    whole = time.perf_counter() - t

    assert build < 0.7 * whole, (build, whole)


def test_a_misspelled_layer_key_is_refused_rather_than_ignored():
    """The failure this guards is silent: no shapes and nothing else read is a legal *uniform*
    layer, so a typo comes back as a slab of `background` that solves and conserves energy to
    1e-14.

    Found the hard way, on a layer whose one meaningful key went to a build that did not read it.
    """
    with pytest.raises(ValueError, match='not read by any path'):
        wavebit.rcwa(structure=[dict(background=1.0, shapez=[CIRCLE])], **BASE).solve()
    with pytest.raises(ValueError, match='not read by any path'):
        wavebit.rcwa(structure=[dict(background=1.0, shapes=[CIRCLE], combyne='add')],
                       **BASE).solve()


# Since 2026-09-28 the public matrices share the amplitudes' centre origin.
# Closed forms below use public shape centres directly; no kernel builder or
# production frame helper supplies the expected Fourier coefficients.
def _host(value):
    return np.asarray(value.detach().cpu() if hasattr(value, 'detach') else value)


def _labels(order):
    u, v = order
    return np.array([(m, n) for n in range(-v, v+1) for m in range(-u, u+1)])


def _rectangle_matrix(basis, order, shape, background=1., value=None):
    labels = _labels(order)
    delta = labels[:, None] - labels[None, :]
    reciprocal = 2*np.pi*np.linalg.inv(basis).T
    g = delta @ reciprocal.T
    material = shape['material'] if value is None else value
    area = np.linalg.det(basis)
    indicator = (shape['lx']*shape['ly']/area
                 * np.sinc(g[..., 0]*shape['lx']/(2*np.pi))
                 * np.sinc(g[..., 1]*shape['ly']/(2*np.pi))
                 * np.exp(-1j*(g[..., 0]*shape.get('cx', 0.)
                               + g[..., 1]*shape.get('cy', 0.))))
    return background*np.eye(len(labels)) + (material-background)*indicator


@pytest.mark.parametrize('rule', ['li', 'laurent', 'nvf'])
def test_an_off_centre_rectangle_publishes_its_closed_form(rule):
    order, basis = (2, 1), np.diag([1.6, 1.8])
    shape = dict(type='rectangle', cx=.13, cy=-.11, lx=.47, ly=.63, material=3.2)
    s = wavebit.rcwa(lattice=basis, thickness=[.2], truncation_order=order,
                    structure=[dict(background=1., shapes=[shape])], factorization=rule)
    conv = s.eps_conv_matrices()
    expected = _rectangle_matrix(basis, order, shape)
    np.testing.assert_allclose(np.linalg.inv(conv.epz_i[0]), expected, atol=3e-13, rtol=0)
    if rule == 'laurent':
        np.testing.assert_allclose(conv.epx[0], expected, atol=3e-14, rtol=0)


@pytest.mark.parametrize('basis', [np.diag([1.6, 1.8]), np.array([[1.6, .8], [0., 1.4]])])
def test_a_centred_disc_has_the_bessel_coefficients_on_any_cell(basis):
    from scipy.special import j1
    order, radius, epsilon = (1, 1), .27, 3.2
    s = wavebit.rcwa(lattice=basis, thickness=[.2], truncation_order=order,
        structure=[dict(background=1., shapes=[dict(type='circle', lx=2*radius, material=epsilon)])],
        factorization='laurent')
    modes = _labels(order)
    g = (modes[:, None]-modes[None, :]) @ (2*np.pi*np.linalg.inv(basis))
    x = np.linalg.norm(g, axis=-1)*radius
    form = np.ones_like(x)
    np.divide(2*j1(x), x, out=form, where=x != 0)
    expected = np.eye(len(modes)) + (epsilon-1)*np.pi*radius**2/np.linalg.det(basis)*form
    got = s.eps_conv_matrices().epx[0]
    np.testing.assert_allclose(got, expected, atol=3e-14, rtol=0)
    assert got[5, 4].real > .1 and got[7, 4].real > .1


def test_a_centred_raster_stripe_and_magnetic_shape_publish_positive_first_harmonics():
    raster = np.ones((1, 1, 100)); raster[..., 30:70] = 3.
    s = wavebit.rcwa(period=(1.6, 1.8), thickness=[.2], truncation_order=(2, 0),
                    structure=raster, tetm='te')
    np.testing.assert_allclose(s.eps_conv_matrices().epy[0, 3, 2],
                               2*.4*np.sinc(.4), atol=2e-14, rtol=0)
    shape = dict(type='rectangle', cx=.13, cy=-.11, lx=.47, ly=.63,
                 material=wavebit.Material(3.2, mu=1.7))
    magnetic = wavebit.rcwa(period=(1.6, 1.8), thickness=[.2], truncation_order=(2, 1),
        structure=[dict(background=wavebit.Material(1., mu=1.1), shapes=[shape])], factorization='laurent')
    expected = _rectangle_matrix(np.diag([1.6, 1.8]), (2, 1), shape, background=1.1, value=1.7)
    np.testing.assert_allclose(magnetic.mu_conv_matrices().epx[0], expected, atol=3e-14, rtol=0)


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('excitation', [None, {(0, 0): 1., (1, 0): .4j, (0, 1): .2}])
def test_a_hand_built_centre_stack_is_the_shape_it_draws(side, excitation):
    order, basis = (2, 1), np.diag([1.6, 1.8])
    shapes = [dict(type='rectangle', cx=.13, cy=-.11, lx=.47, ly=.63, material=3.2),
              dict(type='rectangle', cx=-.21, cy=.17, lx=.55, ly=.41, material=2.7)]
    kw = dict(lattice=basis, thickness=[.17, .12], wavelength=.8, theta=.17, phi=.3,
              truncation_order=order, top=1., bot=2.25, factorization='laurent',
              layer_algorithm='SMM', incidence_side=side, excitation=excitation)
    s = wavebit.rcwa(structure=[dict(background=1., shapes=[a]) for a in shapes], **kw)
    order = tuple(window[1] for window in s.order_window)
    expected = s.solve()
    expected_s = s.scattering
    eps = np.stack([_rectangle_matrix(basis, order, a) for a in shapes])
    conv = wavebit.ConvMatrices(eps, eps, np.linalg.inv(eps), None, None)
    for mixed in (False, True):
        # Mixing an accessor layer with a hand-built layer tests relative placement,
        # which a one-wave efficiency on a wholly translated stack cannot see.
        if mixed:
            built = s.eps_conv_matrices()
            conv = wavebit.ConvMatrices(*(None if a is None else
                np.concatenate([b[:1], a[1:]]) for a, b in zip(conv, built)))
        actual = s.solve_conv(*conv)
        for name in ('R', 'T', 'r_s', 'r_p', 't_s', 't_p', 'r_xyz', 't_xyz'):
            np.testing.assert_allclose(getattr(actual.incident, name), getattr(expected.incident, name),
                                       atol=3e-12, rtol=0)
        for name in ('S11', 'S12', 'S21', 'S22'):
            np.testing.assert_allclose(getattr(s.scattering, name), getattr(expected_s, name),
                                       atol=5e-12, rtol=0)


def _round_trip_scene(case, backend):
    kw = dict(backend=backend, period=(1.6, 1.8), wavelength=1.1, theta=.24,
              phi=.3, thickness=[.17], truncation_order=(1, 1), top=1., bot=2.25,
              answers='field', layer_algorithm='SMM')
    row = np.array([[[1., 4., 2., 1., 3., 1.]]])
    shape = dict(type='rectangle', cx=.13, cy=-.11, lx=.47, ly=.63, material=3.2)
    kw['structure'] = [dict(background=1., shapes=[shape])]
    if case in ('te', 'tm', 'conical'):
        kw.update(structure=row, truncation_order=(2, 0))
        if case != 'conical': kw.update(tetm=case, phi=None)
    elif case in ('li', 'laurent', 'nvf'):
        kw['factorization'] = case
    elif case == 'raster':
        kw.update(structure=np.random.default_rng(62).uniform(1., 3., (1, 5, 6)), factorization='nvf')
    elif case == 'hex':
        kw.pop('period');kw.update(lattice=[[1.6, .8], [0., 1.4]], factorization='laurent')
    elif case == 'bottom': kw['incidence_side'] = 'bottom'
    elif case == 'moved':
        kw.update(wavelength=.5, truncation_order=(2, 1), excitation={(1, 0): 1., (0, 1): .4j, (2, 1): .2})
    elif case == 'mu':
        shape['material'] = wavebit.Material(3.2, mu=1.7)
        kw.update(factorization='laurent', top=wavebit.Material(1., mu=1.1))
    elif case == 'tensor':
        from test_full_tensor_patterns import tensors
        shape['material'] = wavebit.Material(*tensors(.04))
        kw['factorization'] = 'li'
    return wavebit.rcwa(**kw)


def _snapshot(solver, result):
    values = []
    for sub in (result.incident, result.s_inc, result.p_inc):
        if sub is not None:
            values.extend(getattr(sub, name) for name in
                ('R', 'T', 'R_s', 'R_p', 'T_s', 'T_p', 'r_s', 'r_p', 't_s', 't_p', 'r_xyz', 't_xyz'))
    values.extend(getattr(solver.scattering, name) for name in ('S11', 'S12', 'S21', 'S22'))
    values += [solver.T1, result.reconstruct_field(3, 2, 2), solver.reconstruct_field(3, 2, 2)]
    return [(a.shape, a.dtype, a.tobytes()) for a in (_host(value) for value in values if value is not None)]


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('case', ['te', 'tm', 'conical', 'li', 'laurent', 'nvf', 'raster',
                                  'hex', 'bottom', 'moved', 'mu', 'tensor'])
def test_the_public_round_trip_preserves_every_answer_byte(case, backend):
    if backend: pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
    s = _round_trip_scene(case, backend)
    before = _snapshot(s, s.solve())
    conv, mu = s.eps_conv_matrices(), s.mu_conv_matrices()
    extra = {} if not s._is_magnetic() else dict(mu=mu, mu_top=s.top.mu, mu_bot=s.bot.mu,
                                                eps_top=s.top.epsilon, eps_bot=s.bot.epsilon)
    assert before == _snapshot(s, s.solve_conv(*conv, **extra))


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_the_origin_swap_is_exact_and_its_own_inverse(backend):
    from wavebit._core.conventions import swap_fourier_origin
    from _conv_frame import about_centre
    if backend: pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
    s = wavebit.rcwa(backend=backend)
    values = np.arange(81, dtype=float).reshape(9, 9).astype(complex)
    values[0, 1] = complex(-0., 0.);values[1, 0] = complex(0., -0.)
    values[2, 3] = complex(np.nan, -0.)
    a = s.xp.asarray(values, dtype=s.xp.complex128)
    for slots in (wavebit.ConvMatrices(a, a, a, None, None), (a, None), [a, None]):
        got = swap_fourier_origin(slots, (1, 1), (-1, 1))
        back = swap_fourier_origin(got, (1, 1), (-1, 1))
        assert type(got) is type(slots) and type(back) is type(slots)
        for old, converted, restored in zip(slots, got, back):
            if old is None: assert converted is None and restored is None
            else:
                assert _host(restored).tobytes() == _host(old).tobytes()
                assert _host(converted).tobytes() == _host(about_centre(old, (1, 1))).tobytes()
    assert swap_fourier_origin(None, (0, 1), (0, 1)) is None
    with pytest.raises(ValueError, match='retained orders'):
        swap_fourier_origin((a,), (0, 2), (0, 0))


def test_the_origin_swap_is_the_lattice_translation_for_general_operators():
    from wavebit._core.conventions import swap_fourier_origin
    basis = np.array([[1.6, .8], [0., 1.4]])
    modes = _labels((1, 1))
    g = modes @ (2*np.pi*np.linalg.inv(basis))
    phase = np.exp(1j*g @ (basis.sum(axis=1)/2))
    rng = np.random.default_rng(71)
    a = rng.normal(size=(9, 9)) + 1j*rng.normal(size=(9, 9)) + 10*np.eye(9)
    b = rng.normal(size=(9, 9)) + 1j*rng.normal(size=(9, 9))
    flip = lambda a: swap_fourier_origin((a,), (0, 1), (0, 1))[0]
    np.testing.assert_allclose(flip(a), phase[:, None]*a/phase[None, :], atol=1e-14, rtol=0)
    np.testing.assert_allclose(flip(a@b), flip(a)@flip(b), atol=1e-13, rtol=0)
    np.testing.assert_allclose(flip(np.linalg.inv(a)), np.linalg.inv(flip(a)), atol=1e-14, rtol=0)


def test_from_energy_preserves_the_supplied_centre_frame():
    """The energy closure is covariant; a second frame change here would be wrong."""
    from _conv_frame import about_centre
    rng = np.random.default_rng(92)
    q = np.eye(18) + .03*(rng.normal(size=(18, 18)) + 1j*rng.normal(size=(18, 18)))
    g = rng.normal(size=(18, 18)) + 1j*rng.normal(size=(18, 18))
    g = g.conj().T@g + np.eye(18)
    z = rng.normal(size=(9, 9));z = z.T@z + np.eye(9)
    inverse = np.linalg.inv(q)
    tensor = inverse.conj().T@g@inverse
    expected = (tensor[:9, :9], tensor[9:, 9:], np.linalg.inv(z), -tensor[:9, 9:], -tensor[9:, :9])
    actual = wavebit.ConvMatrices.from_energy(q, g, z)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=8e-14, rtol=0)
    translated = wavebit.ConvMatrices.from_energy(*(about_centre(a, (1, 1)) for a in (q, g, z)))
    for a, b in zip(translated, actual):
        np.testing.assert_allclose(a, about_centre(b, (1, 1)), atol=8e-14, rtol=0)


@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
def test_a_public_matrix_gradient_is_the_kernel_gradient_restated(backend):
    from _conv_frame import about_centre
    module = pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
    wavebit.enable_x64()
    s = _round_trip_scene('laurent', backend)
    public, corner = s.eps_conv_matrices(), s._eps_conv_matrices()
    def loss(v, rest, solve):
        return s.xp.sum(solve(v, rest.epy, rest.epz_i).R)
    if backend == 1:
        g_public = module.grad(lambda v: loss(v, public, s.solve_conv))(public.epx)
        g_corner = module.grad(lambda v: loss(v, corner, s._solve_conv))(corner.epx)
        eager = loss(public.epx, public, s.solve_conv)
        traced = module.jit(lambda v: loss(v, public, s.solve_conv))(public.epx)
        np.testing.assert_allclose(traced, eager, atol=1e-12, rtol=0)
        # Differentiate a raster value through the accessor, before any eigensolve.
        row = module.numpy.asarray([[[1., 3., 2., 1.]]])
        a = wavebit.rcwa(backend=1, period=(1.6, 1.8), thickness=[.2],
                         truncation_order=(1, 0), structure=row)
        jp = module.jacfwd(lambda v: a.eps_conv_matrices(structure=row.at[0, 0, 1].set(v)).epx)(3.)
        def kernel(v):
            with a._conditions(dict(structure=row.at[0, 0, 1].set(v))):
                return a._eps_conv_matrices().epx
        jk = module.jacfwd(kernel)(3.)
        assert _host(jp).tobytes() == _host(about_centre(jk, (1, 0))).tobytes()
    else:
        vp = public.epx.detach().clone().requires_grad_(True)
        vc = corner.epx.detach().clone().requires_grad_(True)
        loss(vp, public, s.solve_conv).backward()
        loss(vc, corner, s._solve_conv).backward()
        g_public, g_corner = vp.grad, vc.grad
    expected = about_centre(g_corner, (1, 1))
    np.testing.assert_array_equal(_host(g_public), _host(expected))
    assert _host(g_public).tobytes() == _host(expected).tobytes()
    if backend == 1: module.clear_caches()


@pytest.mark.parametrize('theta', [0., .1])
def test_born_links_the_published_harmonic_to_the_published_amplitude(theta):
    period, width, center = 1.6, .48, .13
    s = wavebit.rcwa(period=(period, period), wavelength=1., thickness=[1/400],
        truncation_order=(4, 0), theta=theta, phi=0., factorization='laurent',
        structure=[dict(background=1., shapes=[dict(type='rectangle',
            cx=center, lx=width, ly=period, material=1.02)])])
    c = s.eps_conv_matrices().epx[0] - np.eye(9)
    r = s.solve()
    ey = np.asarray(r.r_xyz)[0, :, 1]
    kx = np.asarray(s.order_kxy()[0]).reshape(-1)
    kz = np.sqrt(1-kx*kx+0j)
    for j in (3, 5):
        expected = c[j, 4]/c[4, 4]*kz[4]/kz[j]
        np.testing.assert_allclose(ey[j]/ey[4], expected, atol=.009*abs(expected), rtol=0)
