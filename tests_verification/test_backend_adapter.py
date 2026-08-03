"""Backend adapter (fieldwave._core.backend): eig dispatch + differentiability.

Verifies the single per-framework primitive: eigenvalues match NumPy, and gradients through
eig flow correctly on JAX and PyTorch (matching finite differences).
"""
import numpy as np
import pytest

from fieldwave._core.backend import eig, array_namespace


A_np = np.array([[2.0 + 0j, 1.0, 0.2],
                 [0.3, 3.0, 0.5],
                 [0.1, 0.2, 1.0]])


def _sorted_eigvals(w):
    w = np.asarray(w)
    return np.sort_complex(np.round(w, 8))


def test_eig_eigenvalues_numpy_reference():
    w, V = eig(A_np)
    ref = np.linalg.eigvals(A_np)
    assert np.allclose(_sorted_eigvals(w), _sorted_eigvals(ref))
    # reconstruction A V = V diag(w)
    assert np.allclose(A_np @ V, V @ np.diag(w))


def test_array_namespace_numpy():
    xp = array_namespace(A_np)
    assert 'numpy' in xp.__name__


def _grad_scalar_from_eig(xp, backend_scalar):
    # f(p) = sum |sqrt(A)|^2 with sqrt(A) = V diag(sqrt(w)) V^-1, A = A0 + p * P.
    # Gauge-invariant in the eigenvectors (as real RCWA outputs are), so both jax and
    # torch accept it -- torch.linalg.eig backward rejects phase-dependent losses.
    A0 = xp.asarray(A_np)
    P = xp.asarray(np.array([[0, 1.0, 0], [0.0, 0, 0.4], [1.0, 0, 0]], dtype=complex))
    A = A0 + backend_scalar * P
    w, V = eig(A)
    M = V @ (xp.diag(w ** 0.5) @ xp.linalg.inv(V))
    return xp.real(xp.sum(M * xp.conj(M)))


def _fd(p, h=1e-6):
    import numpy as _np
    return (float(_grad_scalar_from_eig(_np, p + h)) - float(_grad_scalar_from_eig(_np, p - h))) / (2 * h)


def test_eig_grad_jax_matches_fd():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    g = float(jax.grad(lambda p: _grad_scalar_from_eig(jnp, p))(jnp.asarray(2.0)))
    assert abs(g - _fd(2.0)) < 1e-4


def test_eig_grad_torch_matches_fd():
    torch = pytest.importorskip("torch")
    p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    _grad_scalar_from_eig(torch, p).backward()
    assert abs(p.grad.item() - _fd(2.0)) < 1e-4
