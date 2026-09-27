"""Independent conforming-space integrals and physical gates for body bubbles."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

from studies.nvf_cell_bounds import energy_bounds
from studies.nvf_continuous_flux import polynomial_moments
from studies.nvf_gap_convergence import mean_energy
from studies.nvf_interface_resolution import MaterialGradedGapModel
from studies.nvf_polynomial_body import PolynomialBodyGapModel, body_space, project_body_polynomial

pytestmark = pytest.mark.research


@pytest.mark.parametrize('degree', [2, 3, 4])
def test_body_matrices_match_independent_gauss_rule_with_periodic_traces(degree):
    edges = np.array([0., .07, .5, .83, 1.])
    indices, _, actual = body_space(edges, degree)
    q, weights = leggauss(24)
    r, weights = (q+1)/2, weights/2
    count = 4*degree
    mass, stiffness = np.zeros((count, count)), np.zeros((count, count))
    load, cross = np.zeros(count), np.zeros(count)
    basis = np.column_stack([1-r, r]+[(eval_legendre(k+1, q)-eval_legendre(k-1, q))/(2*(2*k+1))
                                     for k in range(1, degree)])
    derivative = np.column_stack([-np.ones_like(r), np.ones_like(r)]
                                 +[eval_legendre(k, q) for k in range(1, degree)])
    for e, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        length = right-left
        f, df = np.zeros((len(r), count)), np.zeros((len(r), count))
        f[:, indices[e]], df[:, indices[e]] = basis, derivative/length
        w = weights*length
        mass += (f.T*w)@f
        stiffness += (df.T*w)@df
        load += w@f
        cross -= (w*(left+length*r-.5))@f
    for key, expected in dict(mass=mass, stiffness=stiffness, load=load, cross=cross).items():
        np.testing.assert_allclose(actual[key], expected, atol=5e-12, rtol=0)
    constant = np.r_[np.ones(4), np.zeros(count-4)]
    np.testing.assert_allclose(actual['stiffness']@constant, 0., atol=3e-12, rtol=0)
    np.testing.assert_allclose(actual['mass']@constant, actual['load'], atol=3e-14, rtol=0)
    np.testing.assert_array_equal(actual['boundary'], np.eye(count)[0])


@pytest.mark.parametrize('degree', [2, 3, 4])
@pytest.mark.parametrize('metric_y', [.36, 1., 3.24])
@pytest.mark.parametrize('complex_material', [False, True])
def test_polynomial_body_field_and_energy_match_independent_tensor_quadrature(degree, metric_y, complex_material):
    q, weights = leggauss(64)
    x, wx = q/2, weights/2
    r, wr = (q+1)/2, weights/2
    nodes, lengths = np.array([-.4, .1]), np.array([.2, .15])
    modes = np.arange(-4, 5)
    f = np.column_stack([np.ones_like(x), np.cos(2*np.pi*x), np.sin(2*np.pi*x)])
    df = np.column_stack([np.zeros_like(x), -2*np.pi*np.sin(2*np.pi*x), 2*np.pi*np.cos(2*np.pi*x)])
    lift = 1+.2*np.cos(2*np.pi*x)
    epsilon = 2+.5*np.sin(2*np.pi*x)
    if complex_material:
        f, df = f*(1+.3j), df*(1+.3j)
        lift = lift+.1j*np.sin(2*np.pi*x)
        epsilon = epsilon+1j*(.6+.3*np.sin(2*np.pi*x))
    wave = np.exp(-2j*np.pi*modes[:, None]*x[None])*wx[None]
    def pack(material):
        w = wave*material[None]
        dual_f, dual_df = (f.conj(), df.conj()) if complex_material else (f, df)
        result = dict(mass=np.einsum('mx,xi,xj->mij', w, dual_f, f),
                      stiffness=np.einsum('mx,xi,xj->mij', w, dual_df, df), derivative=w@df,
                      load=w@f, force=(w*lift[None])@dual_df, material=w.sum(axis=1),
                      lift=w@lift, lift_squared=w@(lift*(lift.conj() if complex_material else lift)))
        if complex_material:
            result.update(dual_load=w@f.conj(), reverse_force=(w*lift.conj()[None])@df)
        return result
    potential = np.random.default_rng(811).normal(size=(2, degree+1, 3, 2))*.02
    if complex_material:
        potential = potential+1j*np.random.default_rng(812).normal(size=potential.shape)*.02
    normal = potential[:, 1:]*np.arange(1, degree+1)[None, :, None, None]/lengths[:, None, None, None]
    moments = polynomial_moments(nodes, lengths, np.zeros(2), np.ones(2), modes, degree=2*degree)
    actual = project_body_polynomial(potential, normal, moments, pack(epsilon), pack(np.ones_like(x)),
                                     metric_y=metric_y, sesquilinear=complex_material)
    field = np.zeros((len(modes), len(modes), 2, 2), complex)
    gram = field.copy()
    material = np.zeros((len(modes), len(modes)), complex)
    for e in range(2):
        v = sum(r[:, None, None]**k*potential[e, k] for k in range(degree+1))
        n = sum(r[:, None, None]**k*normal[e, k] for k in range(degree))
        b = np.stack([np.einsum('yic,xi->yxc', v, df), np.einsum('yic,xi->yxc', n, f)], axis=2)
        b[:, :, 0, 0] += lift[None]
        b[:, :, 1, 1] += 1
        weighted = b*np.array([1., metric_y])[None, None, :, None]
        energy = epsilon[None, :, None, None]*np.einsum('yxic,yxid->yxcd', b.conj() if complex_material else b, weighted)
        ey = np.exp(-2j*np.pi*modes[:, None]*(nodes[e]+lengths[e]*r)[None])*(lengths[e]*wr)[None]
        field += np.einsum('ny,yxcd,mx->nmcd', ey, b, wave)
        gram += np.einsum('ny,yxcd,mx->nmcd', ey, energy, wave)
        material += ey.sum(axis=1)[:, None]*(wave@epsilon)[None]
    for got, want in zip(actual, (field, gram, material)):
        np.testing.assert_allclose(got, want, atol=3e-12, rtol=0)


def test_nested_body_degrees_narrow_static_enclosure_and_keep_the_mean():
    previous = None
    for degree in (1, 2, 3, 4):
        model = PolynomialBodyGapModel(0, rows=4, body_degree=degree)
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


def test_degree_one_is_the_original_model_and_invalid_degrees_are_rejected():
    old = MaterialGradedGapModel(2, rows=4)
    control = PolynomialBodyGapModel(2, rows=4, body_degree=1)
    for gap in (0., 17.):
        profile = dict(seams=[31.], values=[4., 9.])
        np.testing.assert_array_equal(old.coefficients(gap, profile)[0], control.coefficients(gap, profile)[0])
    for degree in (0, 5, 1.5):
        with pytest.raises(ValueError, match='body_degree'):
            PolynomialBodyGapModel(1, body_degree=degree)


@pytest.mark.parametrize('polarization', ['p', 's'])
def test_degree_four_jit_gap_seam_and_material_derivatives_match_secants(polarization):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = PolynomialBodyGapModel(2, rows=4, body_degree=4, gap_degree=6)
    def power(p, xp):
        return model.transmission(p[0], dict(seams=p[1:2], values=p[2:]), xp=xp, polarization=polarization)['T']
    p = np.array([17., 0., 4., 9.])
    value, derivative = jax.jit(jax.value_and_grad(lambda p: power(p, jnp)))(jnp.asarray(p))
    assert float(value) == pytest.approx(power(p, np), abs=2e-10, rel=0)
    for i, step in enumerate((1e-3, 1e-4, 1e-4, 1e-4)):
        delta = np.eye(4)[i]*step
        secant = (power(p+delta, np)-power(p-delta, np))/(2*step)
        assert float(derivative[i]) == pytest.approx(secant, abs=2e-9, rel=3e-6)
    jax.clear_caches()
