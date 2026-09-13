"""Independent constitutive-form controls for the experimental passive NVF closure."""
import numpy as np
import pytest

from wavebit._core.passive_nvf import passive_nvf_assemble


def physical(slots):
    return np.block([[np.asarray(slots[0]), -np.asarray(slots[3])],
                     [-np.asarray(slots[4]), np.asarray(slots[1])]])


def inputs(seed=12):
    rng = np.random.default_rng(seed)
    unitary, _ = np.linalg.qr(rng.normal(size=(6, 6))+1j*rng.normal(size=(6, 6)))
    g = (unitary*np.array([0., .13, .31, .67, .91, 1.]))@unitary.conj().T
    matrices = []
    for _ in range(2):
        real = rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3))
        loss = rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3))
        matrices.append((real+real.conj().T)/2 + 1j*(loss@loss.conj().T+.03*np.eye(3)))
    return *matrices, g


def build(xp, direct, inverse, g):
    n = direct.shape[0]
    return passive_nvf_assemble(xp, direct, xp.linalg.inv(inverse),
                                g[:n, :n], g[:n, n:], g[n:, n:], g[n:, :n])


def test_passive_forms_and_an_independent_spectral_normalization():
    e, a, g = inputs()
    result = physical(build(np, e, a, g))
    q = np.eye(6)-g
    gram = q.conj().T@q+g.conj().T@g
    values, vectors = np.linalg.eigh(gram)
    # Spectral normalization is an independent oracle for the iterative kernel.
    z = (vectors/np.sqrt(values))@vectors.conj().T
    expected = z.conj().T@(q.conj().T@np.kron(np.eye(2), e)@q
                          +g.conj().T@np.kron(np.eye(2), a)@g)@z
    np.testing.assert_allclose(result, expected, rtol=0., atol=4e-13)
    loss = (result-result.conj().T)/(2j)
    assert np.linalg.eigvalsh(loss).min() > 0.
def test_a_passive_counterexample_to_the_anticommutator():
    # A concrete two-mode counterexample, with both input losses >= I.
    e, a = 1j*np.diag([1., 100.]), 1j*np.eye(2)
    c = np.full((2, 2), .5)
    zero = np.zeros_like(c)
    g = np.block([[c, zero], [zero, np.eye(2)-c]])
    result = physical(build(np, e, a, g))
    assert np.linalg.eigvalsh((result-result.conj().T)/(2j)).min() >= 1.-1e-13
    from wavebit._core.factorization import _nvf_assemble
    old = physical(_nvf_assemble(np, e, np.linalg.inv(a),
                                 c, zero, np.eye(2)-c))
    assert np.linalg.eigvalsh((old-old.conj().T)/(2j)).min() < -.01


def test_positive_storage_is_preserved_for_nonuniform_material_rules():
    e, a, g = inputs()
    e = e@e.conj().T+.1*np.eye(3)
    a = a@a.conj().T+.1*np.eye(3)
    result = physical(build(np, e, a, g))
    assert np.linalg.eigvalsh((result+result.conj().T)/2).min() > 0.
    np.testing.assert_allclose(result, result.conj().T, rtol=0., atol=4e-12)


@pytest.mark.parametrize('epsilon', [3., 4.+.3j, -10.+1j])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_uniform_material_and_positive_storage(epsilon, dtype):
    _, _, g = inputs()
    e = (epsilon*np.eye(3)).astype(dtype)
    result = physical(build(np, e, e, g.astype(dtype)))
    tolerance = 2e-5 if dtype == np.complex64 else 4e-13
    np.testing.assert_allclose(result, epsilon*np.eye(6), atol=tolerance, rtol=0.)
    if np.real(epsilon) > 0:
        assert np.linalg.eigvalsh((result+result.conj().T)/2).min() > 0.


@pytest.mark.parametrize('angle', [0., .31, np.pi/2])
def test_constant_normal_is_the_rotated_one_dimensional_li_rule(angle):
    e, a, _ = inputs()
    c, s = np.cos(angle), np.sin(angle)
    identity = np.eye(3)
    g = np.block([[c*c*identity, c*s*identity], [c*s*identity, s*s*identity]])
    result = build(np, e, a, g)
    expected = np.block([[c*c*a+s*s*e, c*s*(a-e)],
                         [c*s*(a-e), s*s*a+c*c*e]])
    np.testing.assert_allclose(physical(result), expected, rtol=0., atol=4e-13)
    np.testing.assert_allclose(result[2], np.linalg.inv(e), rtol=0., atol=1e-14)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('real_material', [False, True])
def test_real_and_complex_coefficient_inputs_promote_together(backend, real_material):
    e, a, g = inputs()
    if real_material:
        e, a = np.diag([2., 3., 4.]), np.diag([4., 5., 6.])
    else:
        g = np.diag([1., 1., 1., 0., 0., 0.])
    expected = build(np, e, a, g)
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as xp
        jax.config.update('jax_enable_x64', True)
    else:
        pytest.importorskip('torch')
        import array_api_compat.torch as xp
    actual = build(xp, xp.asarray(e), xp.asarray(a), xp.asarray(g))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got), want, rtol=0., atol=4e-12)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('projector', ['constant', 'compressed'])
def test_live_material_and_projector_gradients(backend, projector):
    e, a, g0 = inputs()
    if projector == 'constant':
        g0 = np.diag([1., 1., 1., 0., 0., 0.]).astype(complex)
    def loss(xp, t):
        e1 = xp.asarray(e)*(1.+.2*t)
        a1 = xp.asarray(a)*(1.-.13*t)
        c, s = xp.cos(t), xp.sin(t)
        eye = xp.eye(3, dtype=e1.dtype)
        rotation = xp.concat([xp.concat([c*eye, -s*eye], axis=1),
                              xp.concat([s*eye, c*eye], axis=1)], axis=0)
        g = rotation@xp.asarray(g0)@xp.matrix_transpose(rotation)
        result = build(xp, e1, a1, g)
        return xp.real(xp.sum(result[0])+.3*xp.sum(result[3]))
    t0 = .17
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as xp
        jax.config.update('jax_enable_x64', True)
        value, grad = jax.jit(jax.value_and_grad(lambda t: loss(xp, t)))(t0)
        actual, derivative = float(value), float(grad)
    else:
        torch = pytest.importorskip('torch')
        import array_api_compat.torch as xp
        t = torch.tensor(t0, dtype=torch.float64, requires_grad=True)
        value = loss(xp, t)
        value.backward()
        actual, derivative = float(value.detach()), float(t.grad)
    assert actual == pytest.approx(float(loss(np, t0)), abs=2e-12)
    for h in (1e-4, 1e-5):
        secant = (float(loss(np, t0+h))-float(loss(np, t0-h)))/(2*h)
        assert derivative == pytest.approx(secant, rel=1e-7, abs=2e-9)
