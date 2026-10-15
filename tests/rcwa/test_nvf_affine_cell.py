"""Affine cell controls: lamination, physical covariance and live derivatives."""

from wavebit.rcwa._engine.factorization import ConvMatrices as _legacy_wavebit_ConvMatrices

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from wavebit.rcwa._engine.conforming.affine_cell import AffineCellGapModel, project_metric_polynomial
from wavebit.rcwa._engine.conforming.polynomial_body import ComplexMaterialGapModel
from tests.rcwa.test_conforming_nvf import solve

pytestmark = pytest.mark.research


SETTINGS = dict(rows=2, body_degree=2, gap_degree=2, grade_y=False, flux_basis='local')
BASIS = np.array([[1., .45], [0., .7]])


def tensors(coefficients):
    c = coefficients
    return (np.stack([np.stack([c[0], c[1]], -1), np.stack([c[2], c[3]], -1)], -2),
            np.stack([np.stack([c[4], c[5]], -1), np.stack([c[8], c[6]], -1)], -2))


@pytest.mark.parametrize('degrees', [(1, 1), (1, 4), (2, 1), (2, 2), (4, 4), (4, 6)])
@pytest.mark.parametrize('gap', [0., 17.])
@pytest.mark.parametrize('flux_basis', ['local', 'global'])
def test_diagonal_metric_recovers_existing_complex_cell(degrees, gap, flux_basis):
    settings = dict(SETTINGS, body_degree=degrees[0], gap_degree=degrees[1], flux_basis=flux_basis)
    old, new = ComplexMaterialGapModel(1, **settings), AffineCellGapModel(1, **settings)
    epsilon = dict(seams=[31.], values=[4.+.8j, 9.+2.1j])
    expected, _ = old.coefficients(gap, epsilon, aspect=1.4)
    actual, metadata = new.coefficients(gap, epsilon, basis=np.diag([1., 1/1.4]))
    assert metadata['cell_residual'] < 1e-8
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)


@pytest.mark.parametrize('shear', [-.8, .45, 1.3])
@pytest.mark.parametrize('epsilon', [4., 4.+.8j])
@pytest.mark.parametrize('order', [2, 5])
def test_contact_is_exact_physical_lamination(shear, epsilon, order):
    model = AffineCellGapModel(order, **SETTINGS)
    basis = np.array([[1., shear], [0., .7]])
    coefficients, _ = model.coefficients(0., epsilon, basis=basis)
    field, gram = tensors(coefficients)
    normal = np.linalg.solve(basis.T, [1., 0.])
    normal /= np.linalg.norm(normal)
    nn, identity = np.outer(normal, normal), np.eye(2)
    m, n = np.meshgrid(model.modes, model.modes)
    delta = ((m == 0) & (n == 0)).astype(float)
    shape = .5*np.sinc(m*.5)*(n == 0)
    eps = delta+(epsilon-1)*shape
    rho = delta+(1/epsilon-1)*shape
    mean_rho = (1+1/epsilon)/2
    expected_field = delta[..., None, None]*identity+(rho/mean_rho-delta)[..., None, None]*nn
    expected_gram = eps[..., None, None]*(identity-nn)+(rho.conj()/abs(mean_rho)**2)[..., None, None]*nn
    np.testing.assert_allclose(field, expected_field, atol=2e-11, rtol=0)
    np.testing.assert_allclose(gram, expected_gram, atol=2e-11, rtol=0)
    np.testing.assert_allclose(coefficients[7], eps, atol=2e-14, rtol=0)


@pytest.mark.parametrize('flux_basis', ['local', 'global'])
def test_batched_positive_gap_recovers_separate_diagonal_metric_kernel(flux_basis):
    settings = dict(SETTINGS, flux_basis=flux_basis)
    profile = dict(seams=[-53., 31.], values=[4.+.8j, 9.+2.1j, 2.+.3j])
    expected, _ = ComplexMaterialGapModel(5, **settings).coefficients(17., profile, aspect=1.4)
    actual, metadata = AffineCellGapModel(5, **settings).coefficients(
        17., profile, basis=np.diag([1., 1/1.4]))
    assert metadata['cell_residual'] < 1e-8
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)


@pytest.mark.parametrize('gap', [0., 23.])
@pytest.mark.parametrize('grade_y', [False, True])
def test_physical_rotation_common_scale_and_passive_energy(gap, grade_y):
    model = AffineCellGapModel(1, **dict(SETTINGS, grade_y=grade_y))
    theta = .37
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    c, _ = model.coefficients(gap, 4.+.8j, basis=BASIS)
    rotated, _ = model.coefficients(gap, 4.+.8j, basis=730*rotation@BASIS)
    for actual, expected in zip(tensors(rotated), tensors(c)):
        np.testing.assert_allclose(actual, rotation@expected@rotation.T, atol=3e-11, rtol=0)
    np.testing.assert_allclose(rotated[7], c[7], atol=2e-14, rtol=0)
    np.testing.assert_allclose(tensors(c)[0][2, 2], np.eye(2), atol=3e-12, rtol=0)
    # The full Fourier Gram form, including both independent off-diagonals.
    _, gram = tensors(c)
    blocks = gram[model.dn, model.dm].transpose(2, 0, 3, 1).reshape(2*model.count, 2*model.count)
    assert np.linalg.eigvalsh((blocks+blocks.conj().T)/2).min() > 0
    assert np.linalg.eigvalsh((blocks-blocks.conj().T)/(2j)).min() > -1e-11


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('gap', [0., 23.])
@pytest.mark.parametrize('loss', [0., .8])
def test_affine_matrices_feed_the_engine_with_contact_inverse_rule(backend, gap, loss):
    if backend == 'jax':
        pytest.importorskip('jax')
        _legacy_wavebit_enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')
    model = AffineCellGapModel(1, **SETTINGS)
    lattice, epsilon = 700*BASIS, 4.+1j*loss
    actual = model.conv_matrices(gap, epsilon, basis=lattice)
    powers = solve(actual, lattice, (1, 1), backend=backend, kernel_frame=True)
    assert powers.min() > -2e-12
    if loss:
        assert np.all(1-powers.sum(axis=(1, 2, 3)) > 1e-3)
    else:
        np.testing.assert_allclose(powers.sum(axis=(1, 2, 3)), 1., atol=2e-10, rtol=0)
    if gap == 0:
        # The physical laminate normal follows a reciprocal vector, not a
        # direct lattice vector. Form the inverse rule without the cell solve.
        normal = np.linalg.solve(lattice.T, [1., 0.])
        normal /= np.linalg.norm(normal)
        nn = np.outer(normal, normal)
        m, n = np.meshgrid(model.modes, model.modes)
        shape = (.5*np.sinc(m*.5)*(n == 0))[model.dn, model.dm]
        identity = np.eye(model.count, dtype=complex)
        eps = identity+(epsilon-1)*shape
        harmonic = np.linalg.inv(identity+(1/epsilon-1)*shape)
        transverse = [[nn[i, j]*harmonic+((i == j)-nn[i, j])*eps for j in range(2)] for i in range(2)]
        expected = _legacy_wavebit_ConvMatrices(*(a[None] for a in (transverse[0][0], transverse[1][1],
                                                            np.linalg.inv(eps), -transverse[0][1], -transverse[1][0])))
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, atol=3e-11, rtol=0)
        np.testing.assert_allclose(powers, solve(expected, lattice, (1, 1), kernel_frame=True), atol=2e-10, rtol=0)
    else:
        theta = .37
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_lattice = rotation@lattice
        rotated = model.conv_matrices(gap, epsilon, basis=rotated_lattice)
        np.testing.assert_allclose(solve(rotated, rotated_lattice, (1, 1), angle=.31+theta, backend=backend, kernel_frame=True),
                                   powers, atol=2e-10, rtol=0)


@pytest.mark.parametrize('degree', [1, 2, 4])
def test_full_metric_projection_matches_independent_tensor_quadrature(degree):
    q, w = leggauss(48)
    x, wx = q/2, w/2
    y, wy = .13+.27*(q+1)/2, .27*w/2
    r = (y-.13)/.27
    f = np.column_stack([np.ones_like(x), np.cos(2*np.pi*x)*(1+.3j), np.sin(2*np.pi*x)+.1j])
    df = np.column_stack([np.zeros_like(x), -2*np.pi*np.sin(2*np.pi*x)*(1+.3j), 2*np.pi*np.cos(2*np.pi*x)])
    lift = 1+.17j*np.cos(2*np.pi*x)
    epsilon = 2.+.8*np.cos(2*np.pi*x)+1j*(.6+.2*np.sin(2*np.pi*x))
    modes = np.array([-3, 0, 2])
    wave = np.exp(-2j*np.pi*modes[:, None]*x[None])*wx[None]
    def pack(material):
        weighted = wave*material
        def product(a, b):
            return np.einsum('mx,xi,xj->mij', weighted, a.conj(), b)
        return dict(mass=product(f, f), stiffness=product(df, df), mixed=product(df, f), reverse_mixed=product(f, df),
                    load=weighted@f, dual_load=weighted@f.conj(), derivative=weighted@df, dual_derivative=weighted@df.conj(),
                    force=(weighted*lift)@df.conj(), reverse_force=(weighted*lift.conj())@df,
                    potential_force=(weighted*lift)@f.conj(), reverse_potential_force=(weighted*lift.conj())@f,
                    lift=weighted@lift, dual_lift=weighted@lift.conj(), lift_squared=weighted@abs(lift)**2,
                    material=weighted.sum(axis=1))
    rng = np.random.default_rng(834)
    p = rng.normal(size=(degree+1, 3, 2))+.2j*rng.normal(size=(degree+1, 3, 2))
    normal = rng.normal(size=(degree, 3, 2))+.2j*rng.normal(size=(degree, 3, 2))
    ywave = np.exp(-2j*np.pi*modes[:, None]*y[None])*wy
    moments = np.stack([ywave@r**k for k in range(2*degree+1)], axis=-1)[:, None]
    metric = np.array([[1.3, -.45], [-.45, 2.4]])
    actual_f, actual_g, actual_eps = project_metric_polynomial(
        p[None], normal[None], moments, pack(epsilon), pack(np.ones_like(x)), metric)
    fx = np.einsum('yp,pic,xi->yxc', r[:, None]**np.arange(degree+1), p, df)
    fy = np.einsum('yp,pic,xi->yxc', r[:, None]**np.arange(degree), normal, f)
    field = np.stack([fx, fy], axis=-2)
    field[:, :, 0, 0] += lift
    field[:, :, 0, 1] += metric[0, 1]/metric[0, 0]*(lift-1)
    field[:, :, 1, 1] += 1
    expected_f = np.einsum('ny,mx,yxcd->nmcd', ywave, wave, field)
    expected_g = np.einsum('ny,mx,x,yxic,ij,yxjd->nmcd', ywave, wave, epsilon, field.conj(), metric, field)
    np.testing.assert_allclose(actual_f, expected_f, atol=3e-13, rtol=0)
    np.testing.assert_allclose(actual_g, expected_g, atol=2e-12, rtol=0)
    np.testing.assert_allclose(actual_eps, ywave.sum(axis=1)[:, None]*(wave@epsilon)[None], atol=2e-14, rtol=0)


@pytest.mark.parametrize('basis', [np.eye(3), [[1., 1.], [0., 0.]], [[1., 0.], [0., -1.]],
                                   [[1., np.nan], [0., 1.]], [[1., .2j], [0., 1.]]])
def test_invalid_affine_basis_raises(basis):
    with pytest.raises(ValueError, match='basis'):
        AffineCellGapModel(0, **SETTINGS).coefficients(17., basis=basis)


@pytest.mark.parametrize('gap', [0., 23.])
@pytest.mark.parametrize('batch_size', [2, 16])
def test_live_shear_material_and_opening_derivatives(gap, batch_size, monkeypatch):
    monkeypatch.setattr('wavebit.rcwa._engine.conforming.affine_cell._PROJECTION_BATCH_SIZE', batch_size)
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = AffineCellGapModel(1, **SETTINGS)
    p = np.array([gap/700, .45, .7, .5, 4., .8])
    # At contact, validate material and affine geometry along the boundary.
    # The opening derivative has an independent contact energy oracle below.
    direction = np.array([.7 if gap else 0., -.2, .13, .1, .8, .3])
    weights = np.random.default_rng(14).normal(size=(9, 5, 5))
    def objective(parameters, xp):
        g, shear, height, width, real, imag = parameters
        basis = xp.stack([xp.stack([xp.ones_like(shear), shear]), xp.stack([xp.zeros_like(height), height])])
        coefficients, _ = model.coefficients(700*g, real+1j*imag, 700*width, xp, basis=basis)
        return xp.real(xp.sum(coefficients*xp.asarray(weights)))
    value, gradient = jax.jit(jax.value_and_grad(lambda p: objective(p, jnp)))(jnp.asarray(p))
    assert float(value) == pytest.approx(objective(p, np), abs=2e-10, rel=0)
    derivative = np.dot(gradient, direction)
    for h in [2e-6, 1e-6]:
        fd = (objective(p+h*direction, np)-objective(p-h*direction, np))/(2*h)
        assert float(derivative) == pytest.approx(fd, abs=3e-6, rel=0)
    invalid = p.copy()
    invalid[2] = 0
    assert np.isnan(jax.jit(lambda p: objective(p, jnp))(jnp.asarray(invalid)))
    jax.clear_caches()


@pytest.mark.parametrize('shear', [-.8, .45])
def test_contact_opening_energy_derivative_matches_independent_trace_minimization(shear):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = AffineCellGapModel(0, **SETTINGS)
    basis = np.array([[1., shear], [0., .7]])
    inverse = np.linalg.inv(basis)
    metric = inverse@inverse.T
    c = metric[0, 1]/metric[0, 0]
    x, y = np.array([[1., c], [0., 0.]]), np.array([[0., -c], [0., 1.]])
    _, edges, values, nodes, lengths, _ = model._profile_geometry(0., 4., 350., np)
    body, background = model._x_integrals(nodes, lengths, edges[1:-1], values, np)
    body, background = [{k: v[0].real for k, v in pack.items()} for pack in (body, background)]
    # At contact the exact body field solves the strong transmission equation.
    # The first energy variation is a trace minimization on the vanishing gap;
    # derivatives of the periodic body corrector cancel by stationarity.
    # Minimize the one-dimensional L2 trace form independently of the coupled
    # cell system, its g scaling, Fourier projection and automatic derivative.
    forcing = ((body['potential_force']-background['potential_force'])[:, None]*(metric@x)[1]
               +(body['dual_load']-background['dual_load'])[:, None]*(metric@y)[1])
    a = np.linalg.solve(metric[1, 1]*background['mass'], forcing)
    def baseline_energy(pack):
        return (pack['lift_squared']*x.T@metric@x+pack['dual_lift']*x.T@metric@y
                +pack['lift']*y.T@metric@x+pack['material']*y.T@metric@y)
    expected = basis@(baseline_energy(background)-baseline_energy(body)
                       -metric[1, 1]*a.T@background['mass']@a)@basis.T
    def mean_energy(g):
        coefficients, _ = model.coefficients(700*g, 4., xp=jnp, basis=jnp.asarray(basis))
        return jnp.stack([jnp.stack([coefficients[4, 0, 0], coefficients[5, 0, 0]]),
                          jnp.stack([coefficients[8, 0, 0], coefficients[6, 0, 0]])])
    _, derivative = jax.jit(lambda g: jax.jvp(mean_energy, (g,), (jnp.ones_like(g),)))(jnp.asarray(0.))
    np.testing.assert_allclose(derivative, expected, atol=3e-9, rtol=0)
    jax.clear_caches()
