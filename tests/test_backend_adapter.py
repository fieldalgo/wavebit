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


# ---------------------------------------------------------------- Bessel J1

def _j1_reference(x):
    """SciPy's J1, which is the accurate one and is what NumPy already uses.

    Checked against ``mpmath`` at 40 digits over ``(0, 95]``: max absolute error 2.2e-16.
    That check is not run here because mpmath is not a dependency; this records the result
    so the constant below is not a mystery.
    """
    from scipy.special import j1
    return j1(np.asarray(x))


# The series/asymptotic hybrid the traced backends use is not exact. Its error is set by
# cancellation in the series just below the crossover, and 1e-11 is comfortably above the
# measured 1.2e-12 while still being four orders tighter than what torch.special.bessel_j1
# delivered (4.1e-7) before this path was shared.
J1_TOL = 1e-11

# Includes the crossover at 12.5 from both sides, the origin, and the far field where the
# asymptotic expansion is the only branch in play.
J1_PROBE = np.concatenate([
    np.array([0.0, 1e-8, 1e-3]),
    np.linspace(0.05, 12.4, 200),
    np.array([12.45, 12.499, 12.5, 12.501, 12.55]),
    np.linspace(12.6, 95.0, 400),
])


@pytest.mark.parametrize('lib', ['jax', 'torch'])
def test_j1_matches_scipy_on_every_backend(lib):
    """The traced backends must agree with SciPy, not merely with each other.

    Both used to be wrong in different ways and by different amounts: torch took
    ``torch.special.bessel_j1`` (4.1e-7 from truth) while jax took the hybrid with a
    crossover that picked the worse branch on both sides of itself (7.2e-11). They now
    share one implementation, so this test would pass vacuously if it compared them to
    each other -- SciPy is the reference for that reason.
    """
    xp = pytest.importorskip(lib)
    from fieldwave._core.backend import j1
    if lib == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
        x = xp.numpy.asarray(J1_PROBE, dtype=xp.numpy.float64)
    else:
        x = xp.as_tensor(J1_PROBE, dtype=xp.float64)
    got = np.asarray(j1(x), dtype=float)
    assert np.max(np.abs(got - _j1_reference(J1_PROBE))) < J1_TOL


@pytest.mark.parametrize('lib', ['jax', 'torch'])
def test_j1_derivative_is_right_including_at_zero(lib):
    """``dJ1/dx = J0 - J1/x``, and 1/2 at the origin.

    The origin is the case worth pinning. An ``abs(x)`` guard at the top of ``j1`` looks
    harmless on a domain of ``x >= 0``, and it silently zeroes this derivative, because
    ``d|x|/dx`` is 0 at 0 in both frameworks. The series branch is odd by construction and
    gives 1/2 without needing the guard.
    """
    pytest.importorskip(lib)
    from scipy.special import j0, j1 as sj1
    from fieldwave._core.backend import j1
    probes = [0.0, 0.7, 5.0, 12.4, 12.6, 40.0]
    exact = [0.5 if p == 0 else float(j0(p) - sj1(p) / p) for p in probes]

    if lib == 'jax':
        import jax, jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        got = [float(jax.grad(j1)(jnp.asarray(p, dtype=jnp.float64))) for p in probes]
    else:
        import torch
        got = []
        for p in probes:
            t = torch.tensor(p, dtype=torch.float64, requires_grad=True)
            j1(t).backward()
            got.append(float(t.grad))

    assert np.allclose(got, exact, atol=1e-11), list(zip(probes, got, exact))
