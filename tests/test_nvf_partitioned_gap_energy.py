"""Joint partition/contact controls for the material-jump integration study."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from studies.nvf_harmonic_energy import powers
from studies.nvf_partitioned_gap_energy import PartitionedGapEnergyModel, interval_moments
from studies.nvf_thin_gap_energy import ThinGapEnergyModel
from studies.nvf_harmonic_coordinates import HarmonicCoordinateGapModel
from studies.nvf_continuous_flux import ContinuousFluxGapModel
from studies.nvf_interface_resolution import MaterialGradedGapModel
from studies.nvf_polynomial_body import PolynomialBodyGapModel
from studies.nvf_polynomial_flux import PolynomialFluxGapModel

pytestmark = pytest.mark.research

MODEL_CLASSES = [PartitionedGapEnergyModel, HarmonicCoordinateGapModel, ContinuousFluxGapModel,
                 MaterialGradedGapModel, PolynomialBodyGapModel, PolynomialFluxGapModel]


def test_interval_moments_against_independent_gauss_integration():
    nodes, lengths = np.array([-.35, .12, -.5]), np.array([.4, .003, .6])
    lo, hi = np.array([0., .25, .51]), np.array([1., .25, .71])
    modes = np.array([-21., -1., -1e-9, 0., 1e-9, 1., 21.])
    actual = interval_moments(nodes, lengths, lo, hi, modes)
    q, w = leggauss(80)
    r = lo[:, None]+(hi-lo)[:, None]*(q+1)/2
    x = nodes[:, None]+lengths[:, None]*r
    weights = lengths[:, None]*(hi-lo)[:, None]*w/2
    expected = np.stack([np.sum(weights[None]*r[None]**degree
                                *np.exp(-2j*np.pi*modes[:, None, None]*x[None]), axis=-1)
                         for degree in range(3)], axis=-1)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
    assert np.array_equal(actual[:, 1], np.zeros((len(modes), 3)))


def test_zero_width_interval_has_its_live_one_sided_derivative_under_jit():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    modes = np.array([-14, 0, 14])

    def moments(length):
        return interval_moments(jnp.array([.13]), jnp.array([length]),
                                 jnp.array([0.]), jnp.array([1.]), modes, jnp)[:, 0]
    value, derivative = jax.jit(lambda g: jax.jvp(moments, (g,), (jnp.ones_like(g),)))(0.)
    np.testing.assert_array_equal(value, np.zeros((3, 3)))
    expected = np.exp(-2j*np.pi*modes*.13)[:, None]/np.arange(1, 4)[None]
    np.testing.assert_allclose(derivative, expected, atol=2e-15, rtol=0)


@pytest.mark.parametrize('gap', [0., .3, 17., 400.])
@pytest.mark.parametrize('model_class', [PartitionedGapEnergyModel, HarmonicCoordinateGapModel])
def test_analytic_projection_agrees_with_conforming_quadrature(gap, model_class):
    model = model_class(3, rows=4, quadrature=24)
    expected, _ = ThinGapEnergyModel(3, rows=4, quadrature=24).coefficients(gap)
    actual, metadata = model.coefficients(gap)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    assert metadata['cell_residual'] < 1e-9


@pytest.mark.parametrize('gap', [0., 1e-5, 400.])
@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_redundant_seams_do_not_change_any_projected_coefficient(gap, model_class):
    model = model_class(3, rows=4)
    merged = model.coefficients(gap, 6.25)[0]
    split = model.coefficients(gap, dict(seams=np.array([-73., 0., 31.]), values=np.full(4, 6.25)))[0]
    np.testing.assert_array_equal(split, merged)
    # Insert redundant seams into two different materials; only genuine
    # contrasts should affect the model, with no value-based branching.
    two = model.coefficients(gap, dict(seams=np.array([-50.]), values=np.array([4., 9.])))[0]
    four = model.coefficients(gap, dict(seams=np.array([-100., -50., 31.]), values=np.array([4., 4., 9., 9.])))[0]
    np.testing.assert_array_equal(two, four)


@pytest.mark.parametrize('polarization', ['p', 's'])
@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_shape_partition_values_and_gradients_including_moving_seam_and_translation(polarization, model_class):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = model_class(3, rows=4)

    def shapes(p, split):
        width, height, epsilon, seam, cx, cy = p
        if split:
            return [dict(lx=seam+width/2, ly=height, epsilon=epsilon, cx=cx+(seam-width/2)/2, cy=cy),
                    dict(lx=width/2-seam, ly=height, epsilon=epsilon, cx=cx+(seam+width/2)/2, cy=cy)]
        return [dict(lx=width, ly=height, epsilon=epsilon, cx=cx, cy=cy)]

    functions = [jax.jit(jax.value_and_grad(lambda p, split=split: model.shape_transmission(
        shapes(p, split), jnp, polarization=polarization)['T'])) for split in (False, True)]
    for height, seam in ((300., 31.), (700., 0.)):
        p = jnp.array([600., height, 4., seam, 37., -29.])
        merged, split = [f(p) for f in functions]
        np.testing.assert_allclose(merged[0], split[0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(merged[1], split[1], atol=2e-10, rtol=0)
        assert abs(float(split[1][3])) < 1e-12
        np.testing.assert_allclose(split[1][-2:], 0., atol=1e-10, rtol=0)
        # Phase-only translations, including wrapping through cell boundaries,
        # must not change total powers.
        shifted = p.at[-2:].set(jnp.array([37.+700., -29.-1400.]))
        np.testing.assert_allclose(functions[1](shifted)[0], split[0], atol=1e-10, rtol=0)
    jax.clear_caches()


@pytest.mark.parametrize('polarization', ['p', 's'])
@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_partitioned_contact_still_passes_independent_physics_gate(polarization, model_class):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from contact_instruments import rectangular_li
    from wavebit._core.lattice import as_lattice
    jax.config.update('jax_enable_x64', True)
    model = model_class(7, rows=8, edge_fraction=1e-5)
    profile = dict(seams=jnp.array([31.]), values=jnp.array([12.25, 12.25]))
    value, derivative = jax.jit(jax.value_and_grad(
        lambda gap: model.transmission(gap, profile, 350., jnp, polarization=polarization)['T']))(0.)

    def independent(gap):
        slots = rectangular_li(np, [dict(type='rectangle', lx=350., ly=700.-gap, material=12.25)],
                                1., 7, 7, as_lattice((700., 700.)), np.complex128)
        return float(powers(7, slots, polarization=polarization)[0])
    closed = independent(0.)
    coarse, fine = [(independent(h)-closed)/h for h in (1e-3, 5e-4)]
    assert float(value) == pytest.approx(closed, abs=2e-10)
    assert float(derivative) == pytest.approx(2*fine-coarse, abs=1e-7, rel=0)
    jax.clear_caches()


@pytest.mark.parametrize('polarization', ['p', 's'])
@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_independent_material_values_remain_live_at_equality_and_seams_remain_live_at_contrast(polarization, model_class):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = model_class(2, rows=4, edge_fraction=1e-4)

    def profile(p, xp):
        return dict(values=xp.asarray(p[:2]), seams=xp.asarray(p[2:]))

    objective = jax.jit(jax.value_and_grad(lambda p: model.transmission(
        17., profile(p, jnp), 440., jnp, polarization=polarization)['T']))
    for p in (np.array([6.25, 6.25, 31.]), np.array([4., 9., 31.]), np.array([1., 1., 31.])):
        value, derivative = objective(p)
        result = model.transmission(17., profile(p, np), 440., polarization=polarization)
        assert float(value) == pytest.approx(result['T'], abs=2e-10)
        assert result['energy_error'] < 1e-11
        assert result['hermiticity'] < 1e-11
        assert np.min(abs(np.asarray(derivative[:2]))) > 1e-5
        for i, h in enumerate((1e-4, 1e-4, 1e-3)):
            delta = np.eye(3)[i]*h
            plus = model.transmission(17., profile(p+delta, np), 440., polarization=polarization)['T']
            minus = model.transmission(17., profile(p-delta, np), 440., polarization=polarization)['T']
            assert float(derivative[i]) == pytest.approx((plus-minus)/(2*h), rel=3e-6, abs=2e-9)
    jax.clear_caches()


@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_closed_mixed_bar_matches_independent_inverse_rule_when_interface_is_resolved(model_class):
    from wavebit._core.factorization import fourier2d_integrated

    order, width, eps = 3, 350., np.array([1., 4., 9., 1.])
    # The seam is at an existing mesh node. This checks a real material contrast,
    # not just an equal-material spelling of the single-bar control.
    x = np.array([.5-width/1400., .5, .5+width/1400., 1.])[:, None]
    y = np.array([[1.]])
    cell = eps[None].astype(complex)
    ex = fourier2d_integrated(np, cell, x, y, order, order, inverse_rule='u')
    ey = fourier2d_integrated(np, cell, x, y, order, order, inverse_rule='v')
    ez = np.linalg.inv(fourier2d_integrated(np, cell, x, y, order, order, inverse_rule=None))
    n, m = np.meshgrid(np.arange(-order, order+1), np.arange(-order, order+1), indexing='ij')
    phase = (-1.)**(n.ravel()+m.ravel())
    model = model_class(order, rows=4)
    actual, _ = model.matrices(0., dict(seams=np.array([0.]), values=np.array([4., 9.])), width)
    for a, b in zip(actual, (ex, ey, ez, np.zeros_like(ex), np.zeros_like(ex))):
        np.testing.assert_allclose(a, phase[:, None]*b*phase[None], atol=2e-10, rtol=0)


@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_invalid_profiles_and_non_tiling_shapes_are_rejected(model_class):
    model = model_class(1, rows=4)
    for profile in (dict(seams=[0., -30.], values=[4., 4., 4.]),
                    dict(seams=[0.], values=[4.]), dict(seams=[180.], values=[4., 4.]),
                    dict(seams=[0.], values=[4., -1.])):
        with pytest.raises(ValueError, match='profile'):
            model.coefficients(0., profile)
    for shapes in ([dict(lx=300., ly=300., epsilon=4., angle=.1)],
                   [dict(lx=300., ly=300., epsilon=4., cx=-151.), dict(lx=300., ly=300., epsilon=4., cx=151.)],
                   [dict(lx=300., ly=300., epsilon=4., cx=-150.), dict(lx=300., ly=290., epsilon=4., cx=150.)]):
        with pytest.raises(ValueError, match='tiling'):
            model.shape_coefficients(shapes)


def test_unfitted_material_error_decreases_with_fixed_mesh_refinement():
    profile = dict(seams=np.array([31.]), values=np.array([4., 9.]))
    exact = 1/(.5+(175+31)/700/4+(175-31)/700/9)
    errors = []
    for subdivisions in (1, 2, 4):
        model = PartitionedGapEnergyModel(0, rows=4, edge_fraction=1e-4, x_subdivisions=subdivisions)
        coefficients, _ = model.coefficients(0., profile)
        errors.append(coefficients[4, 0, 0].real-exact)
        # The refinement policy is independent of primitive partitioning.
        merged = model.coefficients(0., 4.)[0]
        split = model.coefficients(0., dict(seams=np.array([31.]), values=np.array([4., 4.])))[0]
        np.testing.assert_array_equal(merged, split)
    assert 0 < errors[2] < errors[1] < errors[0]
    assert errors[2] < .3*errors[0]


@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_invalid_traced_material_and_shape_updates_produce_nonfinite_data(model_class):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = model_class(1, rows=4, edge_fraction=1e-4)
    profile = jax.jit(lambda seam: model.coefficients(
        0., dict(seams=jnp.array([seam]), values=jnp.array([4., 9.])), xp=jnp)[0])
    assert np.isfinite(np.asarray(profile(31.))).all()
    assert not np.isfinite(np.asarray(profile(180.))).all()
    shapes = jax.jit(lambda displacement: model.shape_coefficients(
        [dict(lx=300., ly=300., epsilon=4., cx=-150.),
         dict(lx=300., ly=300., epsilon=4., cx=150.+displacement)], jnp)[0])
    assert np.isfinite(np.asarray(shapes(0.))).all()
    assert not np.isfinite(np.asarray(shapes(1e-4))).all()
    # The shared cell-solve refactor must retain the original study's live
    # domain guard as well.
    original = ThinGapEnergyModel(1, rows=4, edge_fraction=1e-4)
    result = jax.jit(lambda epsilon: original.coefficients(0., epsilon, xp=jnp)[0])(-1.)
    assert not np.isfinite(np.asarray(result)).all()
    jax.clear_caches()


@pytest.mark.parametrize('model_class', [
    pytest.param(PartitionedGapEnergyModel, marks=pytest.mark.xfail(
        strict=True, reason='fixed P1 space misses the moving real material interface field')),
    HarmonicCoordinateGapModel,
    ContinuousFluxGapModel,
    MaterialGradedGapModel,
    PolynomialBodyGapModel,
    PolynomialFluxGapModel,
])
def test_real_seam_motion_at_a_mesh_node_matches_independent_static_derivative(model_class):
    # Matching a model's own AD/FD is insufficient at an interface. This
    # independent laminate control records an additional adoption blocker.
    model = model_class(0, rows=4, edge_fraction=1e-4)
    energy = lambda seam: model.coefficients(
        0., dict(seams=np.array([seam]), values=np.array([4., 9.])))[0][4, 0, 0].real
    exact = 1/(.5+.25/4+.25/9)
    derivative = -exact**2*(1/4-1/9)/700
    closed = energy(0.)
    assert closed == pytest.approx(exact, abs=1e-11)
    h = 1e-4
    for secant in ((closed-energy(-h))/h, (energy(h)-closed)/h):
        assert secant == pytest.approx(derivative, rel=1e-3, abs=1e-8)
