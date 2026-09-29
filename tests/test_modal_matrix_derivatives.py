"""Matrix identities and independent Frechet exponentials constrain modal tangents."""
import numpy as np
import pytest
from scipy.linalg import expm
from wavebit._core.backend import eig, array_namespace
from wavebit._core.modal_gradient import generator, exponential, inverse, _stop

RNG = np.random.default_rng(941)
BASIS = np.eye(3)+.2*(RNG.normal(size=(3, 3))+1j*RNG.normal(size=(3, 3)))
DIRECTION = RNG.normal(size=(3, 3))+1j*RNG.normal(size=(3, 3))
WEIGHT = RNG.normal(size=(3, 3))+1j*RNG.normal(size=(3, 3))


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
@pytest.mark.parametrize('gap', [0., 1e-15, 1e-12, 1e-7, .1])
@pytest.mark.parametrize('operation', ['sqrt_squared', 'squared', 'exponential', 'inverse'])
def test_completed_modes_have_the_matrix_functions_derivative(backend, gap, operation):
    A = BASIS @ np.diag([1+.2j, 1+.2j+gap, 2-.3j]) @ np.linalg.inv(BASIS)
    if operation == 'sqrt_squared':
        expected = DIRECTION
    elif operation == 'inverse':
        expected = -np.linalg.inv(A) @ DIRECTION @ np.linalg.inv(A)
    elif operation == 'squared':
        expected = A@DIRECTION+DIRECTION@A
    else:
        # The upper-right block is the Frechet derivative of exp(.31 A).
        zero = np.zeros_like(A)
        expected = expm(.31*np.block([[A, DIRECTION], [zero, A]]))[:3, 3:]
    expected = np.real(np.sum(expected*np.conj(WEIGHT)))
    def loss(t):
        xp = array_namespace(t)
        a = xp.asarray(A)+t*xp.asarray(DIRECTION)
        w, W = eig(_stop(a))
        root = operation == 'sqrt_squared'
        q = generator(xp, a, w, W, xp.sqrt(w) if root else w, square_root=root)
        fun = (exponential(xp, q, .31) if operation == 'exponential' else
               inverse(xp, q) if operation == 'inverse' else q@q)
        result = W@fun@xp.linalg.inv(W)
        return xp.real(xp.sum(result*xp.conj(xp.asarray(WEIGHT))))
    if backend in ('jax', 'jit'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        calculate = jax.grad(loss)
        if backend == 'jit':
            calculate = jax.jit(calculate)
        got = float(calculate(jnp.asarray(0.)))
    else:
        torch = pytest.importorskip('torch')
        t = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        loss(t).backward()
        got = float(t.grad)
    assert got == pytest.approx(expected, abs=2e-10, rel=2e-10)


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
def test_second_material_derivatives_are_refused_instead_of_partial(backend):
    def loss(t):
        xp = array_namespace(t)
        a = xp.asarray(np.diag([1., 1., 2.]).astype(complex))+t*xp.asarray(DIRECTION)
        w, W = eig(_stop(a))
        q = generator(xp, a, w, W, w, square_root=False)
        result = W @ exponential(xp, q, .31) @ xp.linalg.inv(W)
        return xp.real(xp.sum(result*xp.conj(xp.asarray(WEIGHT))))
    if backend in ('jax', 'jit'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        calculate = jax.grad(jax.grad(loss))
        if backend == 'jit':
            calculate = jax.jit(calculate)
        with pytest.raises(NotImplementedError, match='first order'):
            calculate(jnp.asarray(0.))
    else:
        torch = pytest.importorskip('torch')
        t = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        with pytest.raises(RuntimeError, match='first order'):
            grad, = torch.autograd.grad(loss(t), t, create_graph=True)
            torch.autograd.grad(grad, t)


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
@pytest.mark.parametrize('gap', [0., 1e-14, .1])
def test_directional_subspaces_match_the_complete_matrix_exponential(backend, gap):
    from wavebit._core.modal_gradient import split_generator
    rng = np.random.default_rng(9214)
    basis = np.eye(4)+.15*(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
    A = basis @ np.diag([1+.2j, 1+.2j+gap, -2-.1j, -2-.1j-gap]) @ np.linalg.inv(basis)
    direction = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    weight = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    expected_matrix = expm(.21*np.block([[A, direction], [np.zeros_like(A), A]]))[:4, 4:]
    expected = np.real(np.sum(expected_matrix*np.conj(weight)))
    def loss(t):
        xp = array_namespace(t)
        a = xp.asarray(A)+t*xp.asarray(direction)
        w, W = eig(_stop(a))
        q, v = split_generator(xp, a, w, W, xp.real(w) > 0)
        result = v @ exponential(xp, q, .21) @ xp.linalg.inv(v)
        return xp.real(xp.sum(result*xp.conj(xp.asarray(weight))))
    if backend in ('jax', 'jit'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        calculation = jax.grad(loss)
        if backend == 'jit':
            calculation = jax.jit(calculation)
        got = float(calculation(jnp.asarray(0.)))
    else:
        torch = pytest.importorskip('torch')
        t = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        loss(t).backward()
        got = float(t.grad)
    assert got == pytest.approx(expected, abs=2e-10, rel=2e-10)


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
@pytest.mark.parametrize('gap', [0., 1e-15])
def test_opposing_modal_collision_refuses_an_invented_finite_derivative(backend, gap):
    from wavebit._core.modal_gradient import split_generator
    def loss(t):
        xp = array_namespace(t)
        w = xp.asarray(np.array([1., 1.+gap, -2., -2.+gap], complex))
        v = xp.asarray(np.eye(4, dtype=complex))
        a = xp.diag(w)+t*xp.asarray(np.ones((4, 4), complex))
        q, basis = split_generator(xp, a, w, v, xp.asarray([True, False, True, False]))
        return xp.real(xp.sum(basis @ exponential(xp, q, .2)))
    if backend in ('jax', 'jit'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        if backend == 'jit':
            result = jax.jit(jax.value_and_grad(loss))(jnp.asarray(0.))
            assert not np.isfinite(result[0])
            assert not np.isfinite(result[1])
        else:
            with pytest.raises(ValueError, match='coincident forward and backward'):
                jax.grad(loss)(jnp.asarray(0.))
    else:
        torch = pytest.importorskip('torch')
        with pytest.raises(ValueError, match='coincident forward and backward'):
            loss(torch.tensor(0., dtype=torch.float64, requires_grad=True))


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
def test_a_real_operator_can_have_a_complex_eigenbasis(backend):
    A = np.array([[1., -2., .2], [2., 1., .3], [.1, -.2, 3.]])
    D = DIRECTION.real
    expected = expm(.31*np.block([[A, D], [np.zeros_like(A), A]]))[:3, 3:]
    expected = np.sum(expected*WEIGHT.real)
    def loss(t):
        xp = array_namespace(t)
        a = xp.asarray(A)+t*xp.asarray(D)
        w, W = eig(_stop(a))
        q = generator(xp, a, w, W, w, square_root=False)
        result = W @ exponential(xp, q, .31) @ xp.linalg.inv(W)
        return xp.real(xp.sum(result*xp.asarray(WEIGHT.real)))
    if backend in ('jax', 'jit'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        calculation = jax.grad(loss)
        if backend == 'jit':
            calculation = jax.jit(calculation)
        got = float(calculation(jnp.asarray(0.)))
    else:
        torch = pytest.importorskip('torch')
        t = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        loss(t).backward()
        got = float(t.grad)
    assert got == pytest.approx(expected, rel=2e-10, abs=2e-10)
