"""Independent integration and physical controls for higher-order x flux."""
import math

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from studies.rcwa.nvf_cell_bounds.nvf_cell_bounds import energy_bounds
from studies.rcwa.nvf_gap_convergence.nvf_gap_convergence import mean_energy
from studies.rcwa.nvf_polynomial_body.nvf_polynomial_body import PolynomialBodyGapModel
from studies.rcwa.nvf_polynomial_flux.nvf_polynomial_flux import PolynomialFluxGapModel, polynomial_flux_integrals

pytestmark = pytest.mark.research


@pytest.mark.parametrize('degree', [2, 3])
def test_flux_packs_match_independent_nested_gauss_integration(degree):
    nodes, lengths = np.array([-.5, -.25, 0., .25]), np.full(4, .25)
    seams, values = np.array([-.11, .13]), np.array([4., 9., 6.25])
    modes, inside = np.arange(-7, 8), np.array([False, True, True, False])
    actual = polynomial_flux_integrals(nodes, lengths, seams, values, inside, modes, 2, degree=degree)
    q, qw = leggauss(48)
    cumulative = np.zeros(4*degree)
    xs, ws, es, potentials, derivatives = [], [], [], [], []

    def flux(element, x):
        r = (x-nodes[element])/lengths[element]
        f = np.zeros((*x.shape, 4*degree))
        f[..., element], f[..., (element+1) % 4] = 1-r, r
        for k in range(1, degree):
            f[..., 4+element*(degree-1)+k-1] = math.comb(degree, k)*r**k*(1-r)**(degree-k)
        return f

    # Integrate physical flux directly at Gauss points, including a second
    # Gauss rule from each piece's start to every evaluation point. No use of
    # production polynomial coefficients, Fourier moments or jump subtraction.
    cuts = np.unique(np.r_[nodes, .5, seams])
    for a, b in zip(cuts[:-1], cuts[1:]):
        middle = (a+b)/2
        element = np.searchsorted(nodes, middle, side='right')-1
        epsilon = values[np.searchsorted(seams, middle)] if inside[element] else 1.
        x = middle+(b-a)/2*q
        w = (b-a)/2*qw
        inner_x = a+(x-a)[:, None]*(q+1)[None]/2
        inner_w = (x-a)[:, None]*qw[None]/2
        potential = cumulative+np.einsum('qi,qij->qj', inner_w, flux(element, inner_x))/epsilon
        derivative = flux(element, x)/epsilon
        cumulative += w@derivative
        xs.extend(x)
        ws.extend(w)
        es.extend(np.full_like(x, epsilon))
        potentials.extend(potential)
        derivatives.extend(derivative)
    x, weights, epsilon, f, df = map(np.asarray, (xs, ws, es, potentials, derivatives))
    total = cumulative[:4].sum()
    lift = 1/epsilon/total
    keep = np.arange(4*degree) != 2
    f = np.column_stack([np.ones(len(x)), (f/cumulative-f[:, :4].sum(axis=1)[:, None]/total)[:, keep]])
    df = np.column_stack([np.zeros(len(x)), (df/cumulative-lift[:, None])[:, keep]])
    wave = np.exp(-2j*np.pi*modes[:, None]*x[None])*weights[None]
    for got, material in zip(actual, (epsilon, np.ones_like(epsilon))):
        w = wave*material[None]
        expected = dict(mass=np.einsum('mq,qi,qj->mij', w, f, f),
                        stiffness=np.einsum('mq,qi,qj->mij', w, df, df),
                        load=w@f, derivative=w@df, force=(w*lift[None])@df,
                        lift=w@lift, lift_squared=w@lift**2, material=w.sum(axis=1))
        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], atol=4e-12, rtol=2e-13)


def test_nested_flux_degrees_narrow_static_enclosure_and_keep_unit_mean():
    previous = None
    for degree in (1, 2, 3):
        model = PolynomialFluxGapModel(0, rows=4, flux_degree=degree)
        c = [model.coefficients(17., dict(seams=[31.], values=values))[0]
             for values in ([4., 9.], [.25, 1/9])]
        for coefficients in c:
            np.testing.assert_allclose(coefficients[:4, 0, 0].reshape(2, 2), np.eye(2), atol=3e-12, rtol=0)
        bounds = energy_bounds(*map(mean_energy, c))
        if previous is not None:
            assert np.linalg.eigvalsh(previous['upper']-bounds['upper']).min() > -2e-11
            assert np.linalg.eigvalsh(bounds['lower']-previous['lower']).min() > -2e-11
            assert bounds['relative_gap'] < previous['relative_gap']
        previous = bounds


def test_degree_one_is_bit_identical_and_invalid_degrees_are_rejected():
    old = PolynomialBodyGapModel(2, rows=4)
    control = PolynomialFluxGapModel(2, rows=4, flux_degree=1)
    for gap in (0., 17.):
        profile = dict(seams=[31.], values=[4., 9.])
        np.testing.assert_array_equal(old.coefficients(gap, profile)[0], control.coefficients(gap, profile)[0])
    for degree in (0, 4, 1.5):
        with pytest.raises(ValueError, match='flux_degree'):
            PolynomialFluxGapModel(1, flux_degree=degree)


def test_convergence_driver_keeps_explicit_body_degree_and_grading_settings():
    from studies.rcwa.nvf_gap_convergence.nvf_gap_convergence import convergence, power_record
    from studies.rcwa.nvf_cell_bounds.nvf_cell_bounds import projected_matrices
    records = list(convergence([1], [(1, 4)], [4], body_degree=1, flux_degree=2))
    candidate = records[-1]
    assert candidate['kind'] == 'gap_polynomial_flux'
    assert candidate['body_degree'] == 1
    model = PolynomialFluxGapModel(1, rows=4, body_degree=1, flux_degree=2, grading_strength=0., edge_fraction=1e-4)
    coefficients, metadata = model.coefficients(17., dict(seams=[31.], values=[4., 9.]))
    expected = power_record(1, projected_matrices(coefficients, 1))
    assert records[-2]['degrees_of_freedom'] == metadata['degrees_of_freedom']
    for key in ('p_T', 's_T'):
        assert candidate[key] == pytest.approx(expected[key], abs=2e-12, rel=0)


@pytest.mark.parametrize('degree', [2, 3])
def test_real_material_profile_is_bit_identical_after_redundant_partition(degree):
    model = PolynomialFluxGapModel(2, rows=4, flux_degree=degree)
    original = dict(seams=[-73., 31.], values=[4., 9., 6.25])
    split = dict(seams=[-100., -73., 0., 31., 100.], values=[4., 4., 9., 9., 6.25, 6.25])
    for gap in (0., 17.):
        np.testing.assert_array_equal(model.coefficients(gap, original)[0], model.coefficients(gap, split)[0])


@pytest.mark.parametrize('degree,polarization', [(2, 'p'), (2, 's'), (3, 'p'), (3, 's')])
def test_jit_gap_seam_material_and_width_derivatives_match_secants(degree, polarization):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = PolynomialFluxGapModel(2, rows=4, flux_degree=degree)
    def power(p, xp):
        return model.transmission(p[0], dict(seams=p[1:2], values=p[2:4]), p[4], xp,
                                  polarization=polarization)['T']
    p = np.array([17., 0., 4., 9., 350.])
    value, derivative = jax.jit(jax.value_and_grad(lambda p: power(p, jnp)))(jnp.asarray(p))
    assert float(value) == pytest.approx(power(p, np), abs=2e-10, rel=0)
    for i, step in enumerate((1e-3, 1e-4, 1e-4, 1e-4, 1e-3)):
        delta = np.eye(5)[i]*step
        secant = (power(p+delta, np)-power(p-delta, np))/(2*step)
        assert float(derivative[i]) == pytest.approx(secant, abs=2e-9, rel=3e-6)
    jax.clear_caches()
