"""Independent quadrature and live-node controls for the integrated flux space."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from studies.rcwa.nvf_continuous_flux.nvf_continuous_flux import ContinuousFluxGapModel, flux_integrals, polynomial_moments
from studies.rcwa.nvf_interface_resolution.nvf_interface_resolution import MaterialGradedGapModel
from studies.rcwa.nvf_polynomial_body.nvf_polynomial_body import PolynomialBodyGapModel
from studies.rcwa.nvf_polynomial_flux.nvf_polynomial_flux import PolynomialFluxGapModel

pytestmark = pytest.mark.research


def test_degree_four_fourier_moments_match_independent_quadrature():
    nodes, lengths = np.array([-.35, .12, -.5]), np.array([.4, .003, .6])
    lo, hi = np.array([0., .25, .51]), np.array([1., .25, .71])
    modes = np.array([-21., -1., -1e-9, 0., 1e-9, 1., 21.])
    actual = polynomial_moments(nodes, lengths, lo, hi, modes)
    q, w = leggauss(80)
    r = lo[:, None]+(hi-lo)[:, None]*(q+1)/2
    x = nodes[:, None]+lengths[:, None]*r
    weight = lengths[:, None]*(hi-lo)[:, None]*w/2
    expected = np.stack([np.sum(weight[None]*r[None]**degree
                                *np.exp(-2j*np.pi*modes[:, None, None]*x[None]), axis=-1)
                         for degree in range(5)], axis=-1)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
    np.testing.assert_array_equal(actual[:, 1], 0.)


def test_zero_width_degree_four_moments_retain_their_jit_derivative():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    modes = np.array([-14, 0, 14])

    def moments(length):
        return polynomial_moments(jnp.array([.13]), jnp.array([length]),
                                   jnp.array([0.]), jnp.array([1.]), modes, jnp)[:, 0]
    value, derivative = jax.jit(lambda g: jax.jvp(moments, (g,), (jnp.ones_like(g),)))(0.)
    np.testing.assert_array_equal(value, np.zeros((3, 5)))
    expected = np.exp(-2j*np.pi*modes*.13)[:, None]/np.arange(1, 6)[None]
    np.testing.assert_allclose(derivative, expected, atol=2e-15, rtol=0)
    jax.clear_caches()


@pytest.mark.parametrize('basis', ['global', 'local'])
@pytest.mark.parametrize('nodes', [[-.5, -.25, 0., .25], [-.5, -.29, .04, .33]])
@pytest.mark.parametrize('complex_material', [False, True])
@pytest.mark.parametrize('mixed', [False, True])
def test_integrated_flux_packs_match_independent_piecewise_gauss_integration(basis, nodes, complex_material, mixed):
    nodes = np.asarray(nodes)
    lengths = np.diff(np.r_[nodes, .5])
    seams, values = np.array([-.11, .13]), np.array([4., 9., 6.25])
    if complex_material:
        values = values+1j*np.array([.8, -.6, 2.1])
    modes, inside = np.arange(-7, 8), np.array([False, True, True, False])
    body, gap = flux_integrals(nodes, lengths, seams, values, inside, modes, 2,
                               basis=basis, sesquilinear=complex_material, mixed=mixed)
    q, qw = leggauss(48)
    cumulative = np.zeros(len(nodes), dtype=values.dtype)
    xs, ws, es, potentials, derivatives = [], [], [], [], []
    # Integrate physical flux hats piece by piece. This path uses no Fourier
    # moment recurrence, material-jump cancellation or polynomial-pack code.
    cuts = np.unique(np.r_[nodes, .5, seams])
    for a, b in zip(cuts[:-1], cuts[1:]):
        middle = (a+b)/2
        element = np.searchsorted(nodes, middle, side='right')-1
        epsilon = values[np.searchsorted(seams, middle)] if inside[element] else 1.
        x = middle+(b-a)/2*q
        left = np.eye(4)[element]
        change = np.eye(4)[(element+1) % 4]-left
        length = lengths[element]
        ra, rb, r = (a-nodes[element])/length, (b-nodes[element])/length, (x-nodes[element])/length
        potential = cumulative+length/epsilon*(np.outer(r-ra, left)+np.outer((r*r-ra*ra)/2, change))
        derivative = (left+np.outer(r, change))/epsilon
        cumulative += length/epsilon*((rb-ra)*left+(rb*rb-ra*ra)/2*change)
        xs.extend(x)
        ws.extend((b-a)/2*qw)
        es.extend(np.full(x.shape, epsilon))
        potentials.extend(potential)
        derivatives.extend(derivative)
    x, weights, epsilon, f, df = map(np.asarray, (xs, ws, es, potentials, derivatives))
    total = cumulative.sum()
    lift = 1/epsilon/total
    keep = [0, 1, 3]
    if basis == 'local':
        first = x < nodes[1]
        origin_fraction = np.sum(weights[first]*df[first, 0])/cumulative[0]
        f, df = f/cumulative, df/cumulative
        offsets = np.array([1-origin_fraction, 0., 0., origin_fraction])
        f, df = (np.roll(f, -1, axis=1)-f-offsets)[:, keep], (np.roll(df, -1, axis=1)-df)[:, keep]
    else:
        f = (f/cumulative-f.sum(axis=1)[:, None]/total)[:, keep]
        df = (df/cumulative-lift[:, None])[:, keep]
    f = np.column_stack([np.ones(len(x)), f])
    df = np.column_stack([np.zeros(len(x)), df])
    wave = np.exp(-2j*np.pi*modes[:, None]*x[None])*weights[None]
    for actual, material in ((body, epsilon), (gap, np.ones_like(epsilon))):
        weighted = wave*material[None]
        dual_f, dual_df = (f.conj(), df.conj()) if complex_material else (f, df)
        expected = dict(mass=np.einsum('mq,qi,qj->mij', weighted, dual_f, f),
                        stiffness=np.einsum('mq,qi,qj->mij', weighted, dual_df, df),
                        load=weighted@f, derivative=weighted@df,
                        force=(weighted*lift[None])@dual_df,
                        lift=weighted@lift, lift_squared=weighted@(lift*(lift.conj() if complex_material else lift)),
                        material=weighted.sum(axis=1))
        if complex_material:
            expected.update(dual_load=weighted@f.conj(), reverse_force=(weighted*lift.conj()[None])@df)
        if mixed:
            dual_lift = lift.conj() if complex_material else lift
            expected.update(mixed=np.einsum('mq,qi,qj->mij', weighted, dual_df, f),
                            reverse_mixed=np.einsum('mq,qi,qj->mij', weighted, dual_f, df),
                            potential_force=(weighted*lift[None])@dual_f,
                            reverse_potential_force=(weighted*dual_lift[None])@f,
                            dual_derivative=weighted@dual_df, dual_lift=weighted@dual_lift)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-12, rtol=1e-13)


@pytest.mark.parametrize('polarization', ['p', 's'])
@pytest.mark.parametrize('model_class', [ContinuousFluxGapModel, MaterialGradedGapModel,
                                        PolynomialBodyGapModel, PolynomialFluxGapModel])
def test_positive_gap_jit_seam_derivative_matches_both_sides_of_mesh_node(polarization, model_class):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = model_class(2, rows=4, edge_fraction=1e-4)
    objective = jax.jit(jax.value_and_grad(lambda seam: model.transmission(
        17., dict(seams=jnp.array([seam]), values=jnp.array([4., 9.])),
        xp=jnp, polarization=polarization)['T']))
    value, derivative = objective(0.)
    power = lambda s: model.transmission(17., dict(seams=[s], values=[4., 9.]), polarization=polarization)['T']
    center = power(0.)
    assert float(value) == pytest.approx(center, abs=2e-10, rel=0)
    h = 1e-4
    left, right = (center-power(-h))/h, (power(h)-center)/h
    np.testing.assert_allclose(float(derivative), [left, right], atol=1e-7, rtol=0)
    jax.clear_caches()
