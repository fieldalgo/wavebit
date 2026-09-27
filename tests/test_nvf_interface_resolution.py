"""Material-grading controls independent of primitive-count or optical fitting."""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.integrate import quad

from studies.nvf_cell_bounds import energy_bounds
from studies.nvf_continuous_flux import ContinuousFluxGapModel
from studies.nvf_gap_convergence import mean_energy
from studies.nvf_interface_resolution import MaterialGradedGapModel, graded_fractions

pytestmark = pytest.mark.research


def independent_roots(targets, seams, values, strength=8., radius=.01):
    seams, values = np.asarray(seams), np.asarray(values)
    jumps = np.diff(np.log(values))/2
    def cdf(x):
        # Integrate sampled density independently of the erf-product primitive.
        points = (seams[:, None]+radius*np.arange(-8, 9)[None]).ravel()
        points = np.unique(points[(points > 0) & (points < x)])
        bounds = np.r_[0., points, x]
        result = x
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            anchor = seams[np.argmin(abs(seams-(lo+hi)/2))]
            # Integrate in radius units. Physical-coordinate quadrature loses
            # digits in (x-seam)/radius at the narrowest supported radius.
            def density(u):
                return strength/np.sqrt(np.pi)*abs(np.sum(
                    jumps*np.exp(-.5*(u+(anchor-seams)/radius)**2)))**2
            result += quad(density, (lo-anchor)/radius, (hi-anchor)/radius,
                           epsabs=2e-14, epsrel=2e-14)[0]
        return result
    total = cdf(1.)
    return np.array([brentq(lambda x: cdf(x)-target*total, 0., 1., xtol=5e-15, rtol=5e-15) for target in targets])


@pytest.mark.parametrize('radius', [.01, .001, .0001])
@pytest.mark.parametrize('complex_material', [False, True])
def test_density_roots_match_independent_bracketed_solver_and_cancel_redundant_seams(radius, complex_material):
    targets = np.r_[0., np.geomspace(1e-5, .1, 5), .25, .5, .75, .9, .99, 1.]
    seams, values = np.array([.21, .63]), np.array([4., 9., 6.25])
    if complex_material:
        values = values+1j*np.array([.8, -.6, 2.1])
    actual, residual = graded_fractions(targets, seams, values, radius=radius)
    np.testing.assert_allclose(actual, independent_roots(targets, seams, values, radius=radius), atol=8e-15, rtol=0)
    density = 1+8/(np.sqrt(np.pi)*radius)*abs(np.sum(
        np.diff(np.log(values))[None]/2*np.exp(-.5*((actual[:, None]-seams)/radius)**2), axis=1))**2
    # A float64 root can have an irreducible CDF residual on a steep density.
    # C(1)>=1, so this bounds its half-ULP contribution without underestimating
    # the normalized residual. The independent root tolerance remains 8e-15;
    # the production CDF guard remains 2e-12.
    rounding_floor = np.max(np.spacing(actual)*density)/2
    assert residual < 2e-14+rounding_floor
    assert np.min(np.diff(actual)) > 0
    redundant, _ = graded_fractions(targets, [.1, .21, .5, .63, .8], np.repeat(values, 2), radius=radius)
    np.testing.assert_array_equal(actual, redundant)
    constant, _ = graded_fractions(targets, [.21, .63], [4., 4., 4.], radius=radius)
    np.testing.assert_array_equal(constant, targets)


@pytest.mark.parametrize('radius', [.01, .001, .0001])
def test_complex_mesh_jacobian_matches_independent_quadrature_root_secants(radius):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    targets = np.array([.1, .25, .5, .75, .9])
    p = np.array([.51, 4., 9., .8, -.6])
    actual = np.asarray(jax.jit(jax.jacfwd(lambda p: graded_fractions(
        targets, p[:1], p[1:3]+1j*p[3:], radius=radius, xp=jnp)[0]))(jnp.asarray(p)))
    for i, step in enumerate((1e-5, 1e-4, 1e-4, 1e-4, 1e-4)):
        delta = np.eye(5)[i]*step
        plus, minus = p+delta, p-delta
        reference = (independent_roots(targets, plus[:1], plus[1:3]+1j*plus[3:], radius=radius)
                     -independent_roots(targets, minus[:1], minus[1:3]+1j*minus[3:], radius=radius))/(2*step)
        np.testing.assert_allclose(actual[:, i], reference, atol=2e-8, rtol=1e-6)
    jax.clear_caches()


@pytest.mark.parametrize('radius', [.01, .001, .0001])
def test_live_mesh_jacobian_matches_independent_root_secants_including_zero_contrast(radius):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    targets = np.array([.1, .25, .5, .75, .9])
    function = jax.jit(jax.jacfwd(lambda p: graded_fractions(targets, p[:1], p[1:], radius=radius, xp=jnp)[0]))
    for p in (np.array([.51, 4., 9.]), np.array([.51, 6.25, 6.25])):
        actual = np.asarray(function(p))
        for i, step in enumerate((1e-5, 1e-4, 1e-4)):
            delta = np.eye(3)[i]*step
            plus = independent_roots(targets, (p+delta)[:1], (p+delta)[1:], radius=radius)
            minus = independent_roots(targets, (p-delta)[:1], (p-delta)[1:], radius=radius)
            np.testing.assert_allclose(actual[:, i], (plus-minus)/(2*step), atol=2e-8, rtol=1e-6)
        if p[1] == p[2]:
            np.testing.assert_array_equal(actual, 0.)
    jax.clear_caches()


@pytest.mark.parametrize('gap', [0., 17., 400.])
def test_multiple_real_jumps_remain_bit_identical_after_redundant_partition(gap):
    model = MaterialGradedGapModel(2, rows=4, edge_fraction=1e-4)
    original = dict(seams=[-73., 31.], values=[4., 9., 6.25])
    split = dict(seams=[-100., -73., 0., 31., 100.], values=[4., 4., 9., 9., 6.25, 6.25])
    np.testing.assert_array_equal(model.coefficients(gap, original)[0], model.coefficients(gap, split)[0])


def test_material_grading_reduces_static_uncertainty_at_the_same_dimension():
    bounds, dimensions = [], []
    for strength in (0., 8.):
        model = MaterialGradedGapModel(0, rows=8, x_subdivisions=2, y_subdivisions=2,
                                       edge_fraction=1e-4, grading_strength=strength)
        energies = []
        for values in ([4., 9.], [.25, 1/9]):
            c, metadata = model.coefficients(17., dict(seams=[31.], values=values))
            energies.append(mean_energy(c))
            dimensions.append(metadata['degrees_of_freedom'])
        bounds.append(energy_bounds(*energies))
    assert len(set(dimensions)) == 1
    assert bounds[1]['relative_gap'] < .4*bounds[0]['relative_gap']
    # The independent fitted FEM reference lies inside both computed enclosures.
    reference_lower = np.diag([1.670777582297659, 3.184544887602813])
    reference_upper = np.diag([1.670777804152657, 3.184547414999149])
    for bound in bounds:
        assert np.linalg.eigvalsh(bound['upper']-reference_upper).min() > 0
        assert np.linalg.eigvalsh(reference_lower-bound['lower']).min() > 0


def test_zero_strength_preserves_fixed_mesh_and_invalid_settings_are_rejected():
    profile = dict(seams=[31.], values=[4., 9.])
    fixed = ContinuousFluxGapModel(1, rows=4)
    control = MaterialGradedGapModel(1, rows=4, grading_strength=0.)
    np.testing.assert_array_equal(fixed.coefficients(17., profile)[0], control.coefficients(17., profile)[0])
    for settings in (dict(grading_strength=-1), dict(grading_strength=np.nan), dict(grading_radius=.001),
                     dict(grading_radius=np.inf)):
        with pytest.raises(ValueError, match='grading'):
            MaterialGradedGapModel(1, **settings)
