"""Independent material loss, complex contact rules, passivity and live replay."""
import numpy as np
import pytest

import wavebit
from wavebit._core.conforming.material_grading import graded_fractions
from wavebit._core.conforming.polynomial_body import ComplexMaterialGapModel
from wavebit._core.parametric import assemble_eps_coeffs, conv_matrix_2d
from test_conforming_nvf import SETTINGS, _objective, solve
from test_conforming_layer_axes import _contact_reference
from test_conforming_separated_rectangles import PARAMETERS, scene


def complex_scene(p=PARAMETERS, losses=(0., .8, 2.1), xp=np, axis=0):
    structure, lattice = scene(p, xp, axis)
    layer = structure[0]
    layer['background'] += 1j*losses[0]
    for shape, loss in zip(layer['shapes'], losses[1:]):
        shape['material'] += 1j*loss
    return structure, lattice


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('losses', [(0., .8, 2.1), (.9, 0., .3), (3., 5., .01)])
@pytest.mark.parametrize('order', [(2, 1), (1, 0), (0, 1)])
def test_independent_loss_has_analytic_longitudinal_material_and_passive_transverse_form(axis, losses, order):
    structure, lattice = complex_scene(losses=losses, axis=axis)
    actual = wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(structure, lattice)
    layer = structure[0]
    eps, _ = assemble_eps_coeffs(np, layer['shapes'], layer['background'], *order, lattice, np.complex128)
    np.testing.assert_allclose(np.linalg.inv(actual.epz_i[0]), conv_matrix_2d(np, eps, *order),
                               atol=2e-12, rtol=0)
    tensor = np.block([[actual.epx[0], -actual.epxy[0]], [-actual.epyx[0], actual.epy[0]]])
    assert np.linalg.eigvalsh((tensor+tensor.conj().T)/2).min() > 0
    assert np.linalg.eigvalsh((tensor-tensor.conj().T)/(2j)).min() > -2e-12
    powers = solve(actual, lattice, order, scale=.65)
    assert powers.min() > -2e-12
    assert np.all(1-powers.sum(axis=(1, 2, 3)) > 1e-3)


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_independent_loss_mixed_contact_stack_matches_analytic_inverse_rules(backend):
    if backend == 1:
        pytest.importorskip('jax')
        wavebit.enable_x64()
    elif backend == 2:
        pytest.importorskip('torch')
    p = PARAMETERS.copy()
    p[1] = 0.
    horizontal, lattice = complex_scene(p)
    vertical, _ = complex_scene(p, (.9, 0., .3), axis=1)
    uniform = 2.3+.7j
    structure = horizontal+[dict(background=uniform)]+vertical
    order = (2, 1)
    actual = wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(structure, lattice)
    identity = np.eye(15, dtype=complex)
    reference = [_contact_reference(horizontal[0], lattice, order, 0),
                 (uniform*identity, uniform*identity, identity/uniform, 0*identity, 0*identity),
                 _contact_reference(vertical[0], lattice, order, 1)]
    expected = wavebit.ConvMatrices(*(np.stack([layer[i] for layer in reference]) for i in range(5)))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=4e-12, rtol=0)
    np.testing.assert_allclose(solve(actual, lattice, order, scale=.65, backend=backend),
                               solve(expected, lattice, order, scale=.65), atol=2e-10, rtol=0)


@pytest.mark.parametrize('axis', [0, 1])
def test_fifteen_independent_geometry_storage_and_loss_derivatives(axis):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*scene(axis=axis))
    parameters = np.r_[PARAMETERS, .3, .8, 2.1]
    def objective(p, xp):
        return _objective(builder.conv_matrices(*complex_scene(p[:12], p[12:], xp, axis), plan=plan, xp=xp), xp)
    value, gradient = jax.jit(jax.value_and_grad(lambda p: objective(p, jnp)))(jnp.asarray(parameters))
    assert value == pytest.approx(objective(parameters, np), abs=3e-10)
    steps = [1e-3]*4+[1e-4]*3+[1e-3]*2+[1e-5]+[1e-3]*2+[1e-5]*3
    for i, step in enumerate(steps):
        for scale in (1., .5):
            delta = np.eye(15)[i]*step*scale
            secant = (objective(parameters+delta, np)-objective(parameters-delta, np))/(2*step*scale)
            assert gradient[i] == pytest.approx(secant, abs=3e-7, rel=5e-6)
    jax.clear_caches()


@pytest.mark.parametrize('axis', [0, 1])
def test_contact_and_independent_lossless_boundary_derivatives(axis):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*scene(axis=axis))
    p = PARAMETERS.copy()
    p[[1, 3]] = 0.
    directions = np.eye(12)[[1, 3]]
    def objective(parameters, xp):
        live = xp.asarray(p)+xp.asarray(directions).T@parameters[:2]
        return _objective(builder.conv_matrices(*complex_scene(live, parameters[2:], xp, axis),
                                                plan=plan, xp=xp), xp)
    parameters = np.array([0., 0., 0., .8, 2.1])
    value, gradient = jax.jit(jax.value_and_grad(lambda v: objective(v, jnp)))(jnp.asarray(parameters))
    baseline = objective(parameters, np)
    assert value == pytest.approx(baseline, abs=3e-10)
    for i, steps in enumerate(((1e-6, 5e-7), (1e-4, 1e-5), (1e-4, 1e-5))):
        for h in steps:
            # At perpendicular contact the tiny-support enrichment has a
            # narrow curvature scale. Remove three truncation terms before
            # roundoff dominates, instead of weakening the derivative bound.
            degree = 4 if i == 0 else 2
            estimates = [(objective(parameters+np.eye(5)[i]*h/2**k, np)-baseline)/(h/2**k)
                         for k in range(degree)]
            for power in range(1, degree):
                estimates = [(2**power*estimates[k+1]-estimates[k])/(2**power-1)
                             for k in range(len(estimates)-1)]
            assert gradient[i] == pytest.approx(estimates[0], abs=3e-6)
    jax.clear_caches()


def test_complex_grading_merges_interfaces_and_resolves_phase_contrast():
    targets = np.linspace(0., 1., 31)
    a, b = 4.+2j, 9.+.3j
    merged, _ = graded_fractions(targets, [.43], [a, b])
    for middle in (1., 1.7+3j, 27.-2j):
        actual, residual = graded_fractions(targets, [.43, .43], [a, middle, b])
        np.testing.assert_allclose(actual, merged, atol=2e-14, rtol=0)
        assert residual < 2e-14
    phase_only, residual = graded_fractions(targets, [.43], [4., 4*np.exp(.7j)])
    assert np.max(np.abs(phase_only-targets)) > .01
    assert residual < 2e-14


@pytest.mark.parametrize('gap', [0., 17., 400.])
def test_complex_local_and_global_bases_span_the_same_conforming_field(gap):
    settings = dict(rows=4, body_degree=2, edge_fraction=1e-4)
    profile = dict(seams=[-73., 31.], values=[4.+.8j, 9.+2.1j, 6.25-.6j])
    global_model = ComplexMaterialGapModel(2, **settings)
    local_model = ComplexMaterialGapModel(2, flux_basis='local', **settings)
    np.testing.assert_allclose(local_model.coefficients(gap, profile)[0],
                               global_model.coefficients(gap, profile)[0], atol=3e-11, rtol=0)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('body_degree,gap_degree', [(1, 1), (1, 4), (2, 1), (3, 4), (4, 6)])
def test_complex_polynomial_variants_preserve_periodic_fields_and_contact_rules(axis, body_degree, gap_degree):
    settings = dict(SETTINGS, body_degree=body_degree, gap_degree=gap_degree)
    model = ComplexMaterialGapModel(1, flux_basis='local', **settings)
    coefficients, metadata = model.coefficients(29., dict(seams=[31.], values=[4.+.8j, 9.+2.1j]))
    np.testing.assert_allclose(coefficients[:4, 2, 2].reshape(2, 2), np.eye(2), atol=3e-12, rtol=0)
    assert metadata['cell_residual'] < 1e-8
    builder = wavebit.ConformingNVF((2, 1), **settings)
    p = PARAMETERS.copy()
    plan = builder.plan(*complex_scene(p, axis=axis))
    for gap in (29., 0.):
        p[1] = gap
        structure, lattice = complex_scene(p, axis=axis)
        actual = builder.conv_matrices(structure, lattice, plan=plan)
        assert all(np.isfinite(a).all() for a in actual)
        if gap == 0:
            reference = _contact_reference(structure[0], lattice, (2, 1), axis)
            for got, want in zip(actual, reference):
                np.testing.assert_allclose(got[0], want, atol=4e-12, rtol=0)


def test_collapsed_interval_complex_material_disappears_from_full_field_and_energy():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = ComplexMaterialGapModel(1, flux_basis='local', **SETTINGS)
    values = [4.+2j, 9.+.3j]
    reference = model.coefficients(17., dict(seams=[31.], values=values))[0]
    weights = np.random.default_rng(936).normal(size=reference.shape)
    def objective(parts):
        c, _ = model.coefficients(17., dict(seams=jnp.array([31., 31.]),
                                           values=jnp.stack([values[0], parts[0]+1j*parts[1], values[1]])), xp=jnp)
        return jnp.sum(jnp.asarray(weights)*(c.real+.37*c.imag))
    differentiate = jax.jit(jax.value_and_grad(objective))
    for middle in (1., 1.7+3j, 27.-2j):
        actual = model.coefficients(17., dict(seams=[31., 31.], values=[values[0], middle, values[1]]))[0]
        np.testing.assert_allclose(actual, reference, atol=4e-12, rtol=0)
        value, derivative = differentiate(jnp.array([np.real(middle), np.imag(middle)]))
        assert value == pytest.approx(np.sum(weights*(reference.real+.37*reference.imag)), abs=3e-10)
        np.testing.assert_allclose(derivative, 0., atol=3e-10, rtol=0)
    jax.clear_caches()
