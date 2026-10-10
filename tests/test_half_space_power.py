"""Interface and propagation power from independent E/H boundary conditions.

The transverse wavevector is real. All powers refer to the finite boundary
with the structure, normalized by the incident wave alone at that boundary.
"""
import numpy as np
import pytest

import wavebit


ROUTES = [
    ('1d', 'ETM'), ('1d', 'SMM'), ('conical', 'ETM'), ('conical', 'SMM'),
    ('2d', 'ETM'), ('2d', 'SMM'), ('first_order', 'ETM'), ('first_order', 'SMM'),
]
ROUTE_IDS = [f'{path}_{algorithm.lower()}' for path, algorithm in ROUTES]
MEDIA = [
    ((1.5 + .3j)**2, 1., 1., 1.),
    (4. + .2j, 1.5 + .1j, 1., 1.),
    (1., 1., (1.5 + .3j)**2, 1.),
    (2.25, 1.3, 1., 1.),
]
MEDIA_IDS = ['absorbing_incident', 'magnetic_incident', 'absorbing_exit', 'lossless']


def _solver(route, medium, beta, pol, side='top', *, excitation=None,
            thickness=0., answers='amplitude'):
    path, algorithm = route
    ei, mi, et, mt = medium
    incoming = wavebit.Material(ei, mu=mi)
    outgoing = wavebit.Material(et, mu=mt)
    opts = dict(backend=0, top=incoming if side == 'top' else outgoing,
                bot=outgoing if side == 'top' else incoming,
                structure=np.full((1, 1 if path in ('1d', 'conical') else 2, 2), ei),
                mu=np.full((1, 1 if path in ('1d', 'conical') else 2, 2), mi),
                thickness=[thickness], wavelength=500., period=[1000., 1000.],
                theta=np.arcsin(beta / np.sqrt(complex(ei * mi))),
                truncation_order=0 if excitation is None else dict(x=1, y=0),
                incidence_side=side, excitation=excitation,
                layer_algorithm=algorithm, answers=answers)
    if path == '1d':
        opts['tetm'] = pol
    else:
        opts['phi'] = 0.
        opts['polarization'] = {'te': (1., 0.), 'tm': (0., 1.),
                                'mixed': (1., 1j)}[pol]
    solver = wavebit.rcwa(**opts)
    if path == 'first_order':
        solver.formulation = 'first_order'
    return solver


def _interface(medium, beta):
    ei, mi, et, mt = medium
    ni, nt = np.sqrt(complex(ei * mi)), np.sqrt(complex(et * mt))
    qi, qt = np.sqrt(complex(ei * mi - beta**2)), np.sqrt(complex(et * mt - beta**2))
    yi, yt = np.array([qi / mi, qi / ei]), np.array([qt / mt, qt / et])
    r = (yi - yt) / (yi + yt)
    t = 2 * yi / (yi + yt)
    # TE is an E amplitude, TM is an H amplitude in this boundary calculation.
    reflectance = abs(r)**2
    transmittance = abs(t)**2 * yt.real / yi.real
    t[1] *= (ni / mi) / (nt / mt)
    incident_weights = yi.real * np.array([1., abs(ni / mi)**2])
    return r, t, reflectance, transmittance, incident_weights


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize('medium', MEDIA, ids=MEDIA_IDS)
@pytest.mark.parametrize('beta', [0., .5], ids=['normal', 'oblique'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('pol', ['te', 'tm', 'mixed'])
def test_interface_power(route, medium, beta, side, pol):
    if route[0] == '1d' and pol == 'mixed':
        pytest.skip('pure 1D has one polarization')
    result = _solver(route, medium, beta, pol, side).solve()
    r, t, R, T, weights = _interface(medium, beta)
    a = np.array({'te': (1., 0.), 'tm': (0., 1.), 'mixed': (1., 1j)}[pol], complex)
    a /= np.linalg.norm(a)
    power = weights * abs(a)**2
    for name, expected in [('R', np.sum(power * R) / np.sum(power)),
                           ('T', np.sum(power * T) / np.sum(power))]:
        assert float(np.sum(getattr(result, name))) == pytest.approx(expected, abs=2e-12)
    # With zero finite thickness, all net entering power crosses the interface.
    assert float(result.power_entering) == pytest.approx(np.sum(power * T) / power.sum(), abs=2e-12)
    assert float(result.absorption) == pytest.approx(0., abs=2e-12)
    for j, suffix in enumerate(('s', 'p')):
        np.testing.assert_allclose(getattr(result.incident, 'r_' + suffix), a[j] * r[j], rtol=0, atol=2e-12)
        np.testing.assert_allclose(getattr(result.incident, 't_' + suffix), a[j] * t[j], rtol=0, atol=2e-12)
    if route[0] != '1d':
        for j, variant in enumerate((result.s_inc, result.p_inc)):
            assert float(np.sum(variant.R)) == pytest.approx(R[j], abs=2e-12)
            assert float(np.sum(variant.T)) == pytest.approx(T[j], abs=2e-12)
            assert float(variant.power_entering) == pytest.approx(T[j], abs=2e-12)
            assert float(variant.absorption) == pytest.approx(0., abs=2e-12)


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize('pol', ['te', 'tm'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('eps', [2.25, (1.5 + .3j)**2], ids=['lossless', 'lossy'])
def test_two_incident_orders_in_the_same_medium(route, pol, side, eps):
    beams = {(0, 0): 1., (1, 0): .5j}
    result = _solver(route, (eps, 1., eps, 1.), 0., pol, side,
                     excitation=beams, thickness=75.).solve()
    orders = result.orders_u
    q = np.sqrt(eps - (.5 * orders)**2 + 0j)
    a = np.array([beams.get((m, 0), 0.) for m in orders])
    weight = q.real if pol == 'te' else (q / eps).real
    incoming = abs(a)**2 * weight
    expected = incoming * np.exp(-4 * np.pi / 500. * q.imag * 75.) / incoming.sum()
    np.testing.assert_allclose(result.R, 0., rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.asarray(result.T).ravel(), expected, rtol=0, atol=2e-12)
    assert float(result.power_entering) == pytest.approx(1., abs=2e-12)
    assert float(result.absorption) == pytest.approx(1 - expected.sum(), abs=2e-12)
    if route[0] != '1d':
        variant = result.s_inc if pol == 'te' else result.p_inc
        for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T', 'power_entering', 'absorption'):
            np.testing.assert_allclose(getattr(variant, name), getattr(result.incident, name),
                                       rtol=0, atol=2e-12)


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize('answers', ['efficiency', 'amplitude', 'field'])
def test_power_agrees_across_answer_levels(route, answers):
    medium = MEDIA[0]
    expected = _interface(medium, .5)[3][1]
    result = _solver(route, medium, .5, 'tm', answers=answers).solve()
    assert float(np.sum(result.T)) == pytest.approx(expected, abs=2e-12)
    assert float(result.power_entering) == pytest.approx(expected, abs=2e-12)
    assert float(result.absorption) == pytest.approx(0., abs=2e-12)


def _slab_heat(ei, ef, et, beta, thickness):
    """Independent boundary solve and volume integral of dielectric heating.

    Use E_y as the TE primary and Z0*H_y as the TM primary. Integrate
    omega*epsilon0*Im(epsilon)*|E|^2/2 inside the finite layer, then divide
    by the incident-only flux. No solver power helper is used here.
    """
    k0 = 2 * np.pi / 500.
    qi, qf, qt = np.sqrt(np.array([ei, ef, et], complex) - beta**2)
    phase = np.exp(1j * k0 * qf * thickness)
    nodes, weights = np.polynomial.legendre.leggauss(40)
    z = (nodes + 1) * thickness / 2
    flux, heat = [], []
    for p in (False, True):
        yi, yf, yt = (qi / ei, qf / ef, qt / et) if p else (qi, qf, qt)
        matrix = np.array([[-1, 1, 1, 0], [yi, yf, -yf, 0],
                           [0, phase, 1 / phase, -1],
                           [0, yf * phase, -yf / phase, -yt]], complex)
        r, f, b, t = np.linalg.solve(matrix, [1., yi, 0., 0.])
        forward = f * np.exp(1j * k0 * qf * z)
        backward = b * np.exp(-1j * k0 * qf * z)
        if p:
            field_squared = (abs(yf * (forward - backward))**2
                             + abs(beta / ef * (forward + backward))**2)
        else:
            field_squared = abs(forward + backward)**2
        absorbed = k0 * ef.imag * thickness / 2 * np.dot(weights, field_squared) / yi.real
        transmitted = abs(t)**2 * yt.real / yi.real
        heat.append(absorbed)
        flux.append(transmitted + absorbed)
    incident_weights = np.array([qi.real, (qi / ei).real * abs(np.sqrt(ei))**2])
    return np.array(flux), np.array(heat), incident_weights


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('beta', [0., .5])
@pytest.mark.parametrize('pol', ['te', 'tm', 'mixed'])
@pytest.mark.parametrize('ei', [2.25, (1.5 + .3j)**2], ids=['lossless_incident', 'absorbing_incident'])
def test_absorption_matches_volume_heating(route, side, beta, pol, ei):
    if route[0] == '1d' and pol == 'mixed':
        pytest.skip('pure 1D has one polarization')
    ef, et, d = 4. + .8j, 1., 75.
    solver = _solver(route, (ei, 1., et, 1.), beta, pol, side, thickness=d)
    solver.structure = np.full(solver.structure.shape, ef)
    result = solver.solve()
    if np.imag(ei) == 0:
        assert float(result.power_entering) == pytest.approx(1 - result.R.sum(), abs=2e-12)
        assert float(result.absorption) > 0.  # A lossless incidence medium does not imply a lossless layer.
    entering, absorbed, weights = _slab_heat(ei, ef, et, beta, d)
    power = weights * np.array({'te': [1., 0.], 'tm': [0., 1.], 'mixed': [.5, .5]}[pol])
    for name, values in [('power_entering', entering), ('absorption', absorbed)]:
        assert float(getattr(result, name)) == pytest.approx(np.dot(values, power) / power.sum(), abs=2e-12)
        if route[0] != '1d':
            for j, variant in enumerate((result.s_inc, result.p_inc)):
                assert float(getattr(variant, name)) == pytest.approx(values[j], abs=2e-12)


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('pol', ['te', 'tm'])
def test_two_order_reflection_in_a_lossless_pattern(route, side, pol):
    """An absorbing incident port must not look like gain in a lossless grating."""
    solver = _solver(route, MEDIA[0], .3, pol, side,
                     excitation={(0, 0): 1., (1, 0): .5j}, thickness=80.)
    solver.mu = None
    if route[0] in ('1d', 'conical'):
        solver.structure = np.array([[[1., 1., 4., 4., 4., 1.]]])
    else:
        solver.structure = np.array([[[1., 4., 4.], [4., 1., 4.], [4., 4., 1.]]])
        solver.phi = .4
    solver.mu = np.ones_like(solver.structure)
    result = solver.solve()
    assert float(result.absorption) == pytest.approx(0., abs=2e-11)
    assert float(result.power_entering) == pytest.approx(float(result.T.sum()), abs=2e-11)


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
def test_complex_transverse_k_does_not_claim_cell_absorption(route):
    solver = _solver(route, MEDIA[0], 0., 'tm')
    solver.theta = .4 + .1j  # Explicit complex-angle illumination has complex k_parallel.
    result = solver.solve()
    assert np.isnan(result.power_entering)
    assert np.isnan(result.absorption)


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_total_absorption_when_a_film_is_split_into_three_layers(route, side):
    ei, ef = (1.5 + .3j)**2, 4. + .8j
    solver = _solver(route, (ei, 1., 1., 1.), .5, 'tm', side)
    shape = (3, *solver.structure.shape[1:])
    solver.mu = None
    solver.structure = np.full(shape, ef)
    solver.thickness = [15., 20., 40.]
    result = solver.solve()
    entering, absorbed, _ = _slab_heat(ei, ef, 1., .5, 75.)
    assert float(result.power_entering) == pytest.approx(entering[1], abs=2e-12)
    assert float(result.absorption) == pytest.approx(absorbed[1], abs=2e-12)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_public_absorption_result_and_gradient(backend, algorithm):
    """The new scalar survives a jitted Result return and carries material gradients."""
    pytest.importorskip(backend)
    if backend == 'jax':
        import jax
        import jax.numpy as xp
        wavebit.enable_x64()
    else:
        import torch as xp

    def solve(loss):
        n = 1.5 + 1j * loss
        solver = wavebit.rcwa(
            backend=1 if backend == 'jax' else 2,
            top=n**2, bot=1., theta=xp.arcsin(.5 / n), phi=0.,
            polarization=(.6, .8j), wavelength=500., period=[1000., 1000.],
            structure=xp.full((1, 2, 2), 4. + .8j, dtype=xp.complex128),
            thickness=[75.], truncation_order=0, layer_algorithm=algorithm)
        return solver.solve()

    def expected(loss):
        entering, absorbed, weights = _slab_heat((1.5 + 1j * loss)**2, 4. + .8j, 1., .5, 75.)
        power = weights * [.36, .64]
        return np.array([np.dot(entering, power), np.dot(absorbed, power)]) / power.sum()

    h = 1e-5
    want = expected(.3)
    derivative = (expected(.3 + h) - expected(.3 - h)) / (2 * h)
    if backend == 'jax':
        result = jax.jit(solve)(xp.asarray(.3))
        def quantities(x):
            r = solve(x)
            return xp.stack([r.power_entering, r.absorption])
        grad = jax.jit(jax.jacrev(quantities))(xp.asarray(.3))
        actual = np.array([result.power_entering, result.absorption])
    else:
        loss = xp.tensor(.3, dtype=xp.float64, requires_grad=True)
        result = solve(loss)
        values = (result.power_entering, result.absorption)
        grad = np.array([xp.autograd.grad(v, loss, retain_graph=True)[0].item() for v in values])
        actual = np.array([v.detach().item() for v in values])
    np.testing.assert_allclose(actual, want, rtol=0, atol=2e-12)
    np.testing.assert_allclose(grad, derivative, rtol=0, atol=2e-7)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('route', [('1d', 'ETM'), ('1d', 'SMM'), ('2d', 'ETM'),
                                  ('2d', 'SMM'), ('first_order', 'ETM')])
def test_absorbing_incident_power_gradient(backend, route):
    """Differentiate incident-medium absorption at fixed real transverse k."""
    pytest.importorskip(backend)
    if backend == 'jax':
        import jax
        import jax.numpy as xp
        wavebit.enable_x64()
    else:
        import torch as xp

    from wavebit._core.solve_1d import solve_1d
    from wavebit._core.solve_2d import solve_2d
    from wavebit._core.solve_2d_first_order import solve_2d_first_order
    from wavebit._core.scattering import smm_1d, smm_2d
    path, algorithm = route
    solve = {('1d', 'ETM'): solve_1d, ('1d', 'SMM'): smm_1d,
             ('2d', 'ETM'): solve_2d, ('2d', 'SMM'): smm_2d,
             ('first_order', 'ETM'): solve_2d_first_order}[route]

    def objective(loss):
        n = 1.5 + 1j * loss
        kx = xp.asarray([.5], dtype=xp.complex128)
        layers = xp.ones((1, 1, 1), dtype=xp.complex128)
        kwargs = dict(wavelength=500., kx=kx, n_top=n, n_bot=1., theta=0.,
                      thickness=[0.], epx_conv_all=layers, epy_conv_all=layers,
                      epz_conv_i_all=layers, eps_top=n**2, eps_bot=1.)
        if path == '1d':
            kwargs['tetm_index'] = 1
        else:
            kwargs.update(ky=xp.zeros((1,), dtype=xp.complex128),
                          incidence=(.6, .8j), phi=0.)
        return xp.sum(solve(**kwargs)[1])

    def expected(loss):
        _, _, _, T, w = _interface(((1.5 + 1j * loss)**2, 1., 1., 1.), .5)
        if path == '1d':
            return T[1]
        power = w * np.array([.36, .64])
        return np.sum(power * T) / np.sum(power)

    step = 1e-5
    derivative = (expected(.3 + step) - expected(.3 - step)) / (2 * step)
    if backend == 'jax':
        value, gradient = jax.jit(jax.value_and_grad(objective))(xp.asarray(.3))
    else:
        loss = xp.tensor(.3, dtype=xp.float64, requires_grad=True)
        value = objective(loss)
        value.backward()
        value, gradient = value.detach(), loss.grad
    assert float(value) == pytest.approx(expected(.3), abs=2e-12)
    assert float(gradient) == pytest.approx(derivative, abs=2e-7)
