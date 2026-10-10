"""Background intervals, representation invariance and live gap closure."""
import numpy as np
from _conv_frame import about_centre
import pytest

import wavebit
from wavebit._core.parametric import assemble_eps_coeffs, conv_matrix_2d
from wavebit._core.conforming.material_grading import graded_fractions, independent_graded_fractions
from wavebit._core.conforming.polynomial_body import PolynomialBodyGapModel
from test_conforming_nvf import SETTINGS, _objective, solve
from test_conforming_layer_axes import _contact_reference


PARAMETERS = np.array([350., 29., 31., 23., 4., 9., 1.7, 700., 430., .31, 17., -21.])


def scene(p=PARAMETERS, xp=np, axis=0, fill=False):
    span, gap, seam, internal, e1, e2, background, px, py, angle, cx, cy = p
    rotation = xp.stack([xp.stack([xp.cos(angle), -xp.sin(angle)]),
                         xp.stack([xp.sin(angle), xp.cos(angle)])])
    pieces = [(-span/2, seam-internal/2, e1), (seam+internal/2, span/2, e2)]
    if fill:
        pieces.append((seam-internal/2, seam+internal/2, background))
    shapes = []
    for lower, upper, epsilon in pieces:
        x, y = (cx+(lower+upper)/2, cy) if axis == 0 else (cx, cy+(lower+upper)/2)
        lx, ly = (upper-lower, py-gap) if axis == 0 else (px-gap, upper-lower)
        center = rotation@xp.stack([x, y])
        shapes.append(dict(lx=lx, ly=ly, cx=center[0], cy=center[1], angle=angle, material=epsilon))
    return [dict(background=background, shapes=shapes[::-1])], rotation*xp.stack([px, py])[None]


@pytest.mark.research
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('order', [(2, 1), (1, 0), (0, 1)])
def test_internal_background_matches_independent_analytic_material_integrals(axis, order):
    structure, lattice = scene(axis=axis)
    builder = wavebit.ConformingNVF(order, **SETTINGS)
    plan = builder.plan(structure, lattice)
    assert plan.axes == (axis,)
    actual = builder.conv_matrices(structure, lattice, plan=plan)
    layer = structure[0]
    eps, _ = assemble_eps_coeffs(np, layer['shapes'], layer['background'], *order, lattice, np.complex128)
    np.testing.assert_allclose(np.linalg.inv(actual.epz_i[0]), about_centre(conv_matrix_2d(np, eps, *order), order),
                               atol=2e-12, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('axis', [0, 1])
def test_explicit_background_rectangle_preserves_all_five_matrices(axis):
    builder = wavebit.ConformingNVF((2, 1), **SETTINGS)
    implicit = builder.conv_matrices(*scene(axis=axis))
    explicit = builder.conv_matrices(*scene(axis=axis, fill=True))
    for got, want in zip(implicit, explicit):
        np.testing.assert_allclose(got, want, atol=4e-12, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('axis', [0, 1])
def test_closing_equal_material_pieces_recovers_a_single_rectangle_with_the_same_plan(axis):
    p = PARAMETERS.copy()
    p[5] = p[4]
    builder = wavebit.ConformingNVF((2, 1), **SETTINGS)
    plan = builder.plan(*scene(p, axis=axis))
    p[3] = 0.
    structure, lattice = scene(p, axis=axis)
    actual = builder.conv_matrices(structure, lattice, plan=plan)
    shape = dict(structure[0]['shapes'][0])
    shape['lx' if axis == 0 else 'ly'] = p[0]
    rotation = lattice/np.linalg.norm(lattice, axis=0)
    shape['cx'], shape['cy'] = rotation@p[-2:]
    single = [dict(background=p[6], shapes=[shape])]
    original_width = shape['lx']
    if axis == 1:
        # A fresh single rectangle prefers axis 0 when both frames are valid.
        # Freeze axis 1 at full first-period extent, then replay the same shape.
        shape['lx'] = p[7]
    single_plan = builder.plan(single, lattice)
    shape['lx'] = original_width
    assert single_plan.axes == (axis,)
    expected = builder.conv_matrices(single, lattice, plan=single_plan)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=4e-12, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_separated_contact_stack_matches_independent_inverse_rules_on_all_backends(backend):
    if backend == 'jax':
        pytest.importorskip('jax')
        wavebit.enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')
    p = PARAMETERS.copy()
    p[1] = 0.
    horizontal, lattice = scene(p)
    vertical, _ = scene(p, axis=1)
    order = (2, 1)
    actual = wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(horizontal+vertical, lattice)
    layers = [_contact_reference(horizontal[0], lattice, order, 0),
              _contact_reference(vertical[0], lattice, order, 1)]
    expected = wavebit.ConvMatrices(*(np.stack([layer[i] for layer in layers]) for i in range(5)))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=4e-12, rtol=0)
    np.testing.assert_allclose(solve(actual, lattice, order, scale=.65, backend=backend),
                               solve(expected, lattice, order, scale=.65), atol=2e-10, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('axis', [0, 1])
def test_twelve_live_parameters_match_two_independent_secant_steps(axis):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*scene(axis=axis))
    def objective(p, xp):
        return _objective(builder.conv_matrices(*scene(p, xp, axis), plan=plan, xp=xp), xp)
    value, gradient = jax.jit(jax.value_and_grad(lambda p: objective(p, jnp)))(jnp.asarray(PARAMETERS))
    assert value == pytest.approx(objective(PARAMETERS, np), abs=3e-10)
    for i, step in enumerate([1e-3]*4+[1e-4]*3+[1e-3]*2+[1e-5]+[1e-3]*2):
        for scale in (1., .5):
            delta = np.eye(12)[i]*step*scale
            secant = (objective(PARAMETERS+delta, np)-objective(PARAMETERS-delta, np))/(2*step*scale)
            assert gradient[i] == pytest.approx(secant, abs=3e-7, rel=5e-6)
    jax.clear_caches()


@pytest.mark.research
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('equal', [False, True])
def test_gap_opening_derivative_is_live_at_internal_contact(axis, equal):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    p = PARAMETERS.copy()
    if equal:
        p[5] = p[4]
    builder = wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*scene(p, axis=axis))
    p[3] = 0.
    def objective(internal, xp):
        live = xp.asarray(p)+xp.asarray(np.eye(12)[3])*internal
        return _objective(builder.conv_matrices(*scene(live, xp, axis), plan=plan, xp=xp), xp)
    value, derivative = jax.jit(jax.value_and_grad(lambda d: objective(d, jnp)))(0.)
    baseline = objective(0., np)
    assert value == pytest.approx(baseline, abs=3e-10)
    for h in (1e-4, 1e-5):
        secant = 2*(objective(h/2, np)-baseline)/(h/2)-(objective(h, np)-baseline)/h
        assert derivative == pytest.approx(secant, abs=3e-6)
    assert abs(derivative) > 1e-3
    jax.clear_caches()


@pytest.mark.parametrize('axis', [0, 1])
def test_opening_and_closing_are_valid_replays_but_overlap_is_not(axis):
    builder = wavebit.ConformingNVF(0, **SETTINGS)
    p = PARAMETERS.copy()
    p[3] = 0.
    plan = builder.plan(*scene(p, axis=axis))
    for internal in (23., 1e-5, 0., -np.spacing(700.)):
        p[3] = internal
        assert all(np.isfinite(a).all() for a in builder.conv_matrices(*scene(p, axis=axis), plan=plan))
    p[3] = -1.
    with pytest.raises(wavebit.StaleGeometryError):
        builder.conv_matrices(*scene(p, axis=axis), plan=plan)
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    build = jax.jit(lambda p: builder.conv_matrices(*scene(p, jnp, axis), plan=plan, xp=jnp))
    assert all(np.isnan(np.asarray(a)).all() for a in build(jnp.asarray(p)))
    jax.clear_caches()


@pytest.mark.research
@pytest.mark.parametrize('values', [[4., 1., 4.], [4., 1., 9.], [4., 1.7, 9.]])
def test_coincident_grading_interfaces_merge_without_the_intermediate_material(values):
    targets = np.linspace(0., 1., 31)
    actual, residual = graded_fractions(targets, [.43, .43], values)
    expected, _ = graded_fractions(targets, [.43], [values[0], values[-1]])
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    assert residual < 2e-14
    # Squared independent jumps leave a fictitious resolution feature at contact.
    historical, _ = independent_graded_fractions(targets, [.43, .43], values)
    assert np.max(np.abs(historical-expected)) > .01


@pytest.mark.research
def test_collapsed_interval_material_has_zero_mesh_derivative():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    targets = jnp.linspace(0., 1., 31)
    derivative = jax.jit(jax.jacfwd(lambda value: graded_fractions(
        targets, jnp.array([.43, .43]), jnp.stack([4., value, 9.]), xp=jnp)[0]))(1.7)
    np.testing.assert_allclose(derivative, 0., atol=5e-14, rtol=0)
    jax.clear_caches()


@pytest.mark.research
def test_collapsed_interval_material_disappears_from_the_full_field_and_energy():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = PolynomialBodyGapModel(1, flux_basis='local', **SETTINGS)
    reference = model.coefficients(17., dict(seams=[31.], values=[4., 9.]))[0]
    weights = np.random.default_rng(936).normal(size=reference.shape)
    def objective(middle):
        c, _ = model.coefficients(17., dict(seams=jnp.array([31., 31.]),
                                           values=jnp.stack([4., middle, 9.])), xp=jnp)
        return jnp.sum(jnp.asarray(weights)*(c.real+.37*c.imag))
    differentiate = jax.jit(jax.value_and_grad(objective))
    for middle in (1., 1.7, 27.):
        actual = model.coefficients(17., dict(seams=[31., 31.], values=[4., middle, 9.]))[0]
        np.testing.assert_allclose(actual, reference, atol=4e-12, rtol=0)
        value, derivative = differentiate(middle)
        assert value == pytest.approx(np.sum(weights*(reference.real+.37*reference.imag)), abs=3e-10)
        assert derivative == pytest.approx(0., abs=3e-10)
    jax.clear_caches()
