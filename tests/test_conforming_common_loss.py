"""Exact common-loss scaling, passive stacks, constrained derivatives and refusals."""
import numpy as np
import pytest

import wavebit
from wavebit._core.parametric import assemble_eps_coeffs, conv_matrix_2d
from wavebit._core.conforming.polynomial_body import PolynomialBodyGapModel
from test_conforming_nvf import SETTINGS, _objective, solve
from test_conforming_layer_axes import _contact_reference
from test_conforming_separated_rectangles import PARAMETERS, scene
from test_nvf_energy_builder import _airy


def lossy_scene(p=PARAMETERS, loss=.23, xp=np, axis=0):
    structure, lattice = scene(p, xp, axis)
    for layer in structure:
        layer['background'] *= 1+1j*loss
        for shape in layer['shapes']:
            shape['material'] *= 1+1j*loss
    return structure, lattice


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('loss', [0., .17, 1.3])
def test_common_loss_scales_storage_matrices_and_preserves_passivity(axis, loss):
    builder = wavebit.ConformingNVF((2, 1), **SETTINGS)
    plan = builder.plan(*scene(axis=axis))
    storage = builder.conv_matrices(*scene(axis=axis), plan=plan)
    actual = builder.conv_matrices(*lossy_scene(loss=loss, axis=axis), plan=plan)
    factors = (1+1j*loss, 1+1j*loss, 1/(1+1j*loss), 1+1j*loss, 1+1j*loss)
    for got, want, factor in zip(actual, storage, factors):
        np.testing.assert_allclose(got, factor*want, atol=2e-12, rtol=0)
    transverse = np.block([[actual.epx[0], -actual.epxy[0]], [-actual.epyx[0], actual.epy[0]]])
    longitudinal = np.linalg.inv(actual.epz_i[0])
    for matrix in (transverse, longitudinal):
        dissipation = (matrix-matrix.conj().T)/(2j)
        assert np.linalg.eigvalsh(dissipation).min() > -2e-12
        if loss:
            assert np.linalg.eigvalsh(dissipation).min() > 0


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('order', [(2, 1), (1, 0), (0, 1)])
def test_lossy_material_matches_independent_analytic_integrals(axis, order):
    structure, lattice = lossy_scene(axis=axis)
    actual = wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(structure, lattice)
    layer = structure[0]
    eps, _ = assemble_eps_coeffs(np, layer['shapes'], layer['background'], *order, lattice, np.complex128)
    np.testing.assert_allclose(np.linalg.inv(actual.epz_i[0]), conv_matrix_2d(np, eps, *order),
                               atol=2e-12, rtol=0)


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_different_loss_ratios_across_layers_match_independent_contact_controls(backend):
    if backend == 1:
        pytest.importorskip('jax')
        wavebit.enable_x64()
    elif backend == 2:
        pytest.importorskip('torch')
    p = PARAMETERS.copy()
    p[1] = 0.
    horizontal, lattice = lossy_scene(p, loss=.13)
    vertical, _ = lossy_scene(p, loss=.41, axis=1)
    uniform = 2.3+.7j
    structure = horizontal+[dict(background=uniform)]+vertical
    order = (2, 1)
    actual = wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(structure, lattice)
    identity = np.eye(15, dtype=complex)
    layers = [_contact_reference(horizontal[0], lattice, order, 0),
              (uniform*identity, uniform*identity, identity/uniform, 0*identity, 0*identity),
              _contact_reference(vertical[0], lattice, order, 1)]
    expected = wavebit.ConvMatrices(*(np.stack([layer[i] for layer in layers]) for i in range(5)))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=4e-12, rtol=0)
    powers = solve(actual, lattice, order, scale=.65, backend=backend)
    np.testing.assert_allclose(powers, solve(expected, lattice, order, scale=.65), atol=2e-10, rtol=0)
    assert powers.min() >= -2e-12
    absorption = 1-powers.sum(axis=(1, 2, 3))
    assert np.all(absorption > 1e-3)


@pytest.mark.parametrize('epsilon', [3.+.2j, 1.7+1.2j])
@pytest.mark.parametrize('backend', [0, 1, 2])
def test_uniform_lossy_layer_matches_independent_airy_formula(epsilon, backend):
    if backend == 1:
        pytest.importorskip('jax')
        wavebit.enable_x64()
    elif backend == 2:
        pytest.importorskip('torch')
    lattice = np.diag([700., 430.])
    builder = wavebit.ConformingNVF((1, 0), **SETTINGS)
    conv = builder.conv_matrices([dict(background=wavebit.Material(epsilon=epsilon))], lattice)
    solver = wavebit.rcwa(backend=backend, lattice=lattice, wavelength=910., thickness=[243.],
                          theta=.23, phi=0., truncation_order=(1, 0), top=1., bot=1.44)
    result = solver.solve_conv(*(solver.xp.asarray(a, dtype=solver.xp.complex128) for a in conv))
    for name, polarization in (('p_inc', 'p'), ('s_inc', 's')):
        response = getattr(result, name)
        transmission, reflection = float(response.T.sum()), float(response.R.sum())
        assert transmission == pytest.approx(_airy(epsilon, 910., 243., .23, polarization), abs=2e-11)
        assert transmission > 0 and reflection >= 0 and transmission+reflection < 1


@pytest.mark.parametrize('axis', [0, 1])
def test_geometry_material_and_common_loss_derivatives_remain_live(axis):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*scene(axis=axis))
    parameters = np.r_[PARAMETERS, .23]
    def objective(p, xp):
        return _objective(builder.conv_matrices(*lossy_scene(p[:-1], p[-1], xp, axis), plan=plan, xp=xp), xp)
    value, gradient = jax.jit(jax.value_and_grad(lambda p: objective(p, jnp)))(jnp.asarray(parameters))
    assert value == pytest.approx(objective(parameters, np), abs=3e-10)
    steps = [1e-3]*4+[1e-4]*3+[1e-3]*2+[1e-5]+[1e-3]*2+[1e-5]
    for i, step in enumerate(steps):
        for scale in (1., .5):
            delta = np.eye(13)[i]*step*scale
            secant = (objective(parameters+delta, np)-objective(parameters-delta, np))/(2*step*scale)
            assert gradient[i] == pytest.approx(secant, abs=3e-7, rel=5e-6)
    storage = builder.conv_matrices(*scene(axis=axis), plan=plan)
    factors = (1j, 1j, -1j/(1+1j*parameters[-1])**2, 1j, 1j)
    exact = wavebit.ConvMatrices(*(a*factor for a, factor in zip(storage, factors)))
    assert gradient[-1] == pytest.approx(_objective(exact, np), abs=3e-10)
    jax.clear_caches()


@pytest.mark.parametrize('axis', [0, 1])
def test_loss_derivative_is_live_at_the_lossless_boundary(axis):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*scene(axis=axis))
    def objective(loss, xp):
        return _objective(builder.conv_matrices(*lossy_scene(loss=loss, xp=xp, axis=axis),
                                                plan=plan, xp=xp), xp)
    value, derivative = jax.jit(jax.value_and_grad(lambda loss: objective(loss, jnp)))(0.)
    storage = builder.conv_matrices(*scene(axis=axis), plan=plan)
    expected = wavebit.ConvMatrices(*(a*factor for a, factor in zip(storage, (1j, 1j, -1j, 1j, 1j))))
    assert derivative == pytest.approx(_objective(expected, np), abs=3e-10)
    baseline = objective(0., np)
    assert value == pytest.approx(baseline, abs=3e-10)
    for h in (1e-4, 1e-5):
        secant = 2*(objective(h/2, np)-baseline)/(h/2)-(objective(h, np)-baseline)/h
        assert derivative == pytest.approx(secant, abs=3e-6)
    jax.clear_caches()


@pytest.mark.parametrize('axis', [0, 1])
def test_lossy_internal_and_perpendicular_contact_derivatives(axis):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*lossy_scene(axis=axis))
    p = PARAMETERS.copy()
    p[[1, 3]] = 0.
    directions = np.eye(12)[[1, 3]]
    def objective(gaps, xp):
        live = xp.asarray(p)+xp.asarray(directions).T@gaps
        return _objective(builder.conv_matrices(*lossy_scene(live, .23, xp, axis), plan=plan, xp=xp), xp)
    value, gradient = jax.jit(jax.value_and_grad(lambda gaps: objective(gaps, jnp)))(jnp.zeros(2))
    baseline = objective(np.zeros(2), np)
    assert value == pytest.approx(baseline, abs=3e-10)
    for i, steps in enumerate(((1e-6, 1e-7), (1e-4, 1e-5))):
        for h in steps:
            direction = np.eye(2)[i]*h
            secant = 2*(objective(direction/2, np)-baseline)/(h/2)-(objective(direction, np)-baseline)/h
            assert gradient[i] == pytest.approx(secant, abs=3e-6)
    jax.clear_caches()


def test_unequal_loss_ratios_and_gain_are_refused_eagerly_and_poison_traced_replay():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(0, **SETTINGS)
    plan = builder.plan(*scene())
    def build(losses, xp):
        structure, lattice = scene(xp=xp)
        structure[0]['background'] *= 1+1j*losses[0]
        for shape, loss in zip(structure[0]['shapes'], losses[1:]):
            shape['material'] *= 1+1j*loss
        return builder.conv_matrices(structure, lattice, plan=plan, xp=xp)
    traced = jax.jit(lambda losses: build(losses, jnp))
    for losses in ([.2, .2, .2], [.2, .2+np.spacing(.2), .2]):
        assert all(np.isfinite(np.asarray(a)).all() for a in traced(jnp.asarray(losses)))
    for losses in ([.2, .21, .2], [.21, .2, .2], [.2, .2, .2+1e-9], [-.1, -.1, -.1], [np.nan, .2, .2]):
        with pytest.raises(ValueError):
            build(np.asarray(losses), np)
        assert all(np.isnan(np.asarray(a)).all() for a in traced(jnp.asarray(losses)))
    jax.clear_caches()


def test_material_wrappers_preserve_the_common_factor_and_inputs():
    structure, lattice = lossy_scene()
    builder = wavebit.ConformingNVF(0, **SETTINGS)
    expected = builder.conv_matrices(structure, lattice)
    layer = structure[0]
    layer['background'] = wavebit.Material(epsilon=layer['background'])
    layer['shapes'][0]['material'] = wavebit.Material(epsilon=layer['shapes'][0]['material'])
    wrapped = layer['shapes'][0]['material']
    actual = builder.conv_matrices(structure, lattice)
    assert layer['shapes'][0]['material'] is wrapped
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


def test_real_corrector_refuses_unfactored_loss_instead_of_discarding_it():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    cell = PolynomialBodyGapModel(0, flux_basis='local', **SETTINGS)
    with pytest.raises(ValueError, match='real materials'):
        cell.coefficients(17., 4.+.2j)
    build = jax.jit(lambda loss: cell.coefficients(17., 4.+1j*loss, xp=jnp)[0])
    np.testing.assert_allclose(np.asarray(build(0.)), cell.coefficients(17., 4.)[0], atol=2e-12, rtol=0)
    assert np.isnan(np.asarray(build(.2))).all()
    jax.clear_caches()
