"""Independent controls for gap bubbles and separate convergence diagnostics."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from studies.nvf_cell_bounds import energy_bounds, projected_matrices
from studies.nvf_continuous_flux import (ContinuousFluxGapModel, polynomial_moments,
                                        project_flux_polynomial, project_flux_strip)
from studies.nvf_gap_convergence import load_projection, mean_energy, profile_li
from studies.nvf_harmonic_energy import PERIOD, THETA, THICKNESS, WAVELENGTH, polarized_powers, powers

pytestmark = pytest.mark.research


def test_batched_polarizations_match_independent_slab_and_preserve_column_order():
    slots = (np.array([[4.]]), np.array([[4.]]), np.array([[.25]]), np.zeros((1, 1)), np.zeros((1, 1)))
    polarizations = ('s', 'p', 's')
    t, r = polarized_powers(0, slots, polarizations=polarizations)
    for i, polarization in enumerate(polarizations):
        epsilon = np.array([1., 4., 2.25])
        kz = np.sqrt(epsilon-np.sin(THETA)**2)
        y0, y1, y2 = kz if polarization == 's' else epsilon/kz
        phase = np.exp(2j*np.pi/WAVELENGTH*THICKNESS*kz[1])
        amplitude = 4*y0*y1*phase/((y0+y1)*(y1+y2)+(y0-y1)*(y1-y2)*phase**2)
        assert t[i] == pytest.approx(y2/y0*abs(amplitude)**2, abs=2e-13)
    np.testing.assert_allclose(t+r, 1., atol=2e-13, rtol=0)
    for invalid in ((), 'p', ('p', 'invalid')):
        with pytest.raises(ValueError, match='polarizations'):
            polarized_powers(0, slots, polarizations=invalid)


def test_cartesian_profile_matches_independent_single_rectangle_and_cancels_redundant_seam():
    from contact_instruments import rectangular_li
    from wavebit._core.lattice import as_lattice
    for gap in (0., 17., 400.):
        actual = profile_li(2, gap, 350., [31.], [6.25, 6.25])
        expected = rectangular_li(np, [dict(type='rectangle', lx=350., ly=PERIOD-gap, material=6.25)],
                                  1., 2, 2, as_lattice((PERIOD, PERIOD)), np.complex128)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, atol=2e-13, rtol=0)
    with pytest.raises(ValueError, match='profile'):
        profile_li(2, 17., 350., [180.], [4., 9.])


def test_maximal_projection_can_feed_lower_outer_order_without_changing_the_model():
    profile = dict(seams=[31.], values=[4., 9.])
    high = ContinuousFluxGapModel(4, rows=4, gap_degree=4)
    low = ContinuousFluxGapModel(2, rows=4, gap_degree=4)
    coefficients, _ = high.coefficients(17., profile)
    actual = projected_matrices(coefficients, 2)
    expected, _ = low.matrices(17., profile)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=2e-12, rtol=0)


@pytest.mark.parametrize('degree', [1, 2, 4, 6])
@pytest.mark.parametrize('metric_y', [.36, 1., 3.24])
def test_gap_polynomial_projection_matches_independent_tensor_gauss_rule(degree, metric_y):
    q, qw = leggauss(64)
    x, wx = q/2, qw/2
    r, wr = (q+1)/2, qw/2
    gap = .2
    y, wy = .5-gap/2+gap*r, gap*wr
    f = np.column_stack([np.ones_like(x), np.cos(2*np.pi*x), np.sin(2*np.pi*x)])
    df = np.column_stack([np.zeros_like(x), -2*np.pi*np.sin(2*np.pi*x), 2*np.pi*np.cos(2*np.pi*x)])
    lift = 1+.2*np.cos(2*np.pi*x)
    modes = np.arange(-6, 7)
    wave = np.exp(-2j*np.pi*modes[:, None]*x[None])*wx[None]
    pack = dict(mass=np.einsum('mx,xi,xj->mij', wave, f, f),
                stiffness=np.einsum('mx,xi,xj->mij', wave, df, df),
                derivative=wave@df, load=wave@f, force=(wave*lift[None])@df,
                material=wave.sum(axis=1), lift=wave@lift, lift_squared=wave@lift**2)
    potential = np.random.default_rng(731).normal(size=(degree+1, 3, 2))*.02
    normal = np.arange(1, degree+1)[:, None, None]*potential[1:]/gap
    moments = polynomial_moments(np.array([.5-gap/2]), np.array([gap]), np.array([0.]),
                                  np.array([1.]), modes, degree=2*degree)[:, 0]
    actual_field, actual_gram, actual_material = project_flux_polynomial(
        potential, normal, moments, pack, metric_y=metric_y)
    v = sum(r[:, None, None]**i*potential[i] for i in range(degree+1))
    n = sum(r[:, None, None]**i*normal[i] for i in range(degree))
    bx, by = np.einsum('yic,xi->yxc', v, df), np.einsum('yic,xi->yxc', n, f)
    basis = np.stack([bx, by], axis=2)
    basis[:, :, 0, 0] += lift[None]
    basis[:, :, 1, 1] += 1
    weighted = basis*np.array([1., metric_y])[None, None, :, None]
    gram = np.einsum('yxic,yxid->yxcd', basis, weighted)
    ey = np.exp(-2j*np.pi*modes[:, None]*y[None])*wy[None]
    expected_field = np.einsum('ny,yxcd,mx->nmcd', ey, basis, wave)
    expected_gram = np.einsum('ny,yxcd,mx->nmcd', ey, gram, wave)
    np.testing.assert_allclose(actual_field, expected_field, atol=2e-13, rtol=0)
    np.testing.assert_allclose(actual_gram, expected_gram, atol=2e-12, rtol=0)
    if degree == 1:
        strip = project_flux_strip(potential[0][None], potential.sum(axis=0)[None],
                                    normal[0][None], moments[:, None], pack, metric_y=metric_y)
        for got, want in zip(strip, (expected_field, expected_gram, actual_material)):
            np.testing.assert_allclose(got, want, atol=2e-12, rtol=0)
    np.testing.assert_allclose(actual_material, ey.sum(axis=1)[:, None]*wave.sum(axis=1)[None], atol=2e-13, rtol=0)


def test_gap_enrichment_narrows_primal_complementary_bounds_without_moving_mesh():
    previous = None
    for degree in (1, 2, 4):
        model = ContinuousFluxGapModel(0, rows=4, edge_fraction=1e-4, gap_degree=degree)
        energies = [mean_energy(model.coefficients(17., dict(seams=[31.], values=values))[0])
                    for values in ([4., 9.], [.25, 1/9])]
        bounds = energy_bounds(*energies)
        if previous is not None:
            assert np.linalg.eigvalsh(previous['upper']-bounds['upper']).min() > -2e-11
            assert np.linalg.eigvalsh(bounds['lower']-previous['lower']).min() > -2e-11
            assert bounds['relative_gap'] < previous['relative_gap']
        previous = bounds


@pytest.mark.parametrize('polarization', ['p', 's'])
def test_enriched_gap_keeps_independent_contact_derivative(polarization):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = ContinuousFluxGapModel(7, rows=8, edge_fraction=1e-5, gap_degree=4)
    value, gradient = jax.jit(jax.value_and_grad(lambda gap: model.transmission(
        gap, 12.25, xp=jnp, polarization=polarization)['T']))(0.)
    independent = lambda gap: float(powers(7, profile_li(7, gap, 350., [], [12.25]), polarization=polarization)[0])
    closed = independent(0.)
    slopes = [(independent(h)-closed)/h for h in (1e-3, 5e-4)]
    assert float(value) == pytest.approx(closed, abs=2e-10)
    assert float(gradient) == pytest.approx(2*slopes[1]-slopes[0], abs=1e-7, rel=0)
    jax.clear_caches()


@pytest.mark.parametrize('polarization', ['p', 's'])
def test_enriched_gap_jit_parameters_at_real_node_and_material_equality(polarization):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = ContinuousFluxGapModel(2, rows=4, edge_fraction=1e-4, gap_degree=4)
    objective = jax.jit(jax.value_and_grad(lambda p: model.transmission(
        p[0], dict(seams=p[1:2], values=p[2:]), xp=jnp, polarization=polarization)['T']))
    def reference(p):
        return model.transmission(p[0], dict(seams=p[1:2], values=p[2:]), polarization=polarization)
    for p in (np.array([17., 0., 4., 9.]), np.array([17., 31., 6.25, 6.25])):
        value, gradient = objective(p)
        result = reference(p)
        assert float(value) == pytest.approx(result['T'], abs=2e-10)
        assert result['energy_error'] < 1e-11
        for i, step in enumerate((1e-3, 1e-4, 1e-4, 1e-4)):
            delta = np.eye(4)[i]*step
            secant = (reference(p+delta)['T']-reference(p-delta)['T'])/(2*step)
            assert float(gradient[i]) == pytest.approx(secant, rel=3e-6, abs=2e-9)
        if p[2] == p[3]:
            assert np.min(abs(np.asarray(gradient[2:]))) > 1e-5
            np.testing.assert_array_equal(model.coefficients(p[0], dict(seams=p[1:2], values=p[2:]))[0],
                                          model.coefficients(p[0], p[2])[0])
    assert not np.isfinite(np.asarray(objective(jnp.array([-1., 31., 4., 9.]))[0]))
    jax.clear_caches()


def test_gap_degree_domain_is_explicit():
    for degree in (0, 7, 2.5):
        with pytest.raises(ValueError, match='gap_degree'):
            ContinuousFluxGapModel(0, gap_degree=degree)
    for subdivisions in (0, 1.5):
        with pytest.raises(ValueError, match='y_subdivisions'):
            ContinuousFluxGapModel(0, y_subdivisions=subdivisions)


def test_nested_body_refinement_narrows_static_bounds_and_preserves_mean():
    previous = None
    for subdivisions in (1, 2):
        model = ContinuousFluxGapModel(0, rows=4, edge_fraction=1e-4, y_subdivisions=subdivisions)
        coefficients = [model.coefficients(17., dict(seams=[31.], values=values))[0]
                        for values in ([4., 9.], [.25, 1/9])]
        for c in coefficients:
            np.testing.assert_allclose(c[:4, 0, 0].reshape(2, 2), np.eye(2), atol=2e-12, rtol=0)
        bounds = energy_bounds(*map(mean_energy, coefficients))
        if previous is not None:
            assert np.linalg.eigvalsh(previous['upper']-bounds['upper']).min() > -2e-11
            assert np.linalg.eigvalsh(bounds['lower']-previous['lower']).min() > -2e-11
            assert bounds['relative_gap'] < previous['relative_gap']
        previous = bounds


def test_reference_projection_cannot_silently_compare_a_different_profile(tmp_path):
    import json
    path = tmp_path/'reference.npz'
    metadata = dict(fixture='partitioned_gap', gap=17., width=350., seams=[31.], values=[4., 9.])
    coefficients = np.zeros((8, 5, 5), complex)
    np.savez(path, coefficients=coefficients, metadata=json.dumps(metadata))
    actual, _ = load_projection(path, 17., 350., [31.], [4., 9.])
    np.testing.assert_array_equal(actual, coefficients)
    with pytest.raises(ValueError, match='profile mismatch'):
        load_projection(path, 17., 350., [32.], [4., 9.])
