"""Backend adapter — the only code that dispatches on the array framework.

The single-source core (migration Phase 3+) calls these; the current per-backend solvers do
not yet. Keeping the surface tiny is the point: everything divergent between NumPy / JAX /
PyTorch lives here, so it can be tested in one place.

``eig`` is the one primitive that genuinely differs. JAX >= 0.11 supports gradients through
non-symmetric complex ``eig`` natively via ``enable_eigvec_derivs`` (assumes distinct
eigenvalues); PyTorch's ``torch.linalg.eig`` is differentiable; NumPy needs no gradient.
"""
import array_api_compat as ac


def array_namespace(*xs):
    """Return the Array-API namespace for the given array(s) (numpy / jax.numpy / torch)."""
    return ac.array_namespace(*xs)


def eig(A):
    """Eigendecomposition of a (possibly non-Hermitian) complex matrix.

    Returns ``(eigenvalues, right_eigenvectors)`` and is differentiable on JAX and PyTorch.
    """
    xp = ac.array_namespace(A)
    name = xp.__name__
    if 'torch' in name:
        import torch
        w, V = torch.linalg.eig(A)
        return w, V
    if 'jax' in name:
        import jax.lax.linalg as jll
        out = jll.eig(
            A,
            compute_left_eigenvectors=False,
            compute_right_eigenvectors=True,
            enable_eigvec_derivs=True,
        )
        return out[0], out[-1]
    import numpy as np
    w, V = np.linalg.eig(A)
    return w, V


# --- Bessel J1: the second primitive that genuinely differs per framework ---------------
#
# NumPy has SciPy's j1. PyTorch has an accurate forward (`torch.special.bessel_j1`) but no
# autograd backward, so it needs the analytic derivative attached. JAX's
# `jax.scipy.special.bessel_jn` returns NaN for small arguments, so it needs its own
# implementation entirely. Same value, three routes — exactly what this adapter is for.

_J1_THR = 12.0            # series below, asymptotic expansion above
_J1_SERIES_TERMS = 30
_J1_ASYM_TERMS = 6
_J1_A = [1.0]
for _k in range(1, 2 * _J1_ASYM_TERMS + 2):
    _J1_A.append(_J1_A[-1] * (4.0 - (2 * _k - 1) ** 2) / (_k * 8.0))


def _j1_series(xp, x):
    """``(x/2) * sum_m (-x^2/4)^m / (m! (m+1)!)`` — accurate and stable for small ``x``."""
    term = x / 2.0
    s = term
    z = -(x * x) / 4.0
    for m in range(1, _J1_SERIES_TERMS):
        term = term * z / (m * (m + 1))
        s = s + term
    return s


def _j1_asymptotic(xp, x):
    """Hankel asymptotic expansion — takes over where the series loses precision."""
    P = xp.zeros_like(x)
    Q = xp.zeros_like(x)
    for k in range(_J1_ASYM_TERMS):
        P = P + ((-1) ** k) * _J1_A[2 * k] / x ** (2 * k)
        Q = Q + ((-1) ** k) * _J1_A[2 * k + 1] / x ** (2 * k + 1)
    omega = x - 3.0 * 3.141592653589793 / 4.0
    return xp.sqrt(2.0 / (3.141592653589793 * x)) * (P * xp.cos(omega) - Q * xp.sin(omega))


def j1(x):
    """Bessel function of the first kind, order 1 — differentiable on every backend.

    Defined for ``x >= 0`` (the analytic shape transforms only ever evaluate it there).
    """
    xp = ac.array_namespace(x)
    name = xp.__name__
    if 'torch' in name:
        import torch
        return _TorchBesselJ1.apply(x)
    if 'jax' in name:
        x = xp.abs(x)
        # Clamp each branch's input so the *unselected* branch stays finite: a NaN there
        # would still poison the gradient through `where`.
        return xp.where(x < _J1_THR,
                        _j1_series(xp, xp.minimum(x, _J1_THR)),
                        _j1_asymptotic(xp, xp.maximum(x, _J1_THR)))
    from scipy.special import j1 as _scipy_j1
    import numpy as np
    return _scipy_j1(np.asarray(x))


class _TorchBesselJ1Meta(type):
    """Builds the torch autograd Function lazily so importing this module never needs torch."""
    _impl = None

    def apply(cls, x):
        if cls._impl is None:
            import torch

            class _Impl(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return torch.special.bessel_j1(x)

                @staticmethod
                def backward(ctx, grad_out):
                    # dJ1/dx = J0(x) - J1(x)/x, and 1/2 at x = 0.
                    (x,) = ctx.saved_tensors
                    safe = torch.where(x == 0, torch.ones_like(x), x)
                    dj1 = torch.where(
                        x == 0, torch.full_like(x, 0.5),
                        torch.special.bessel_j0(x) - torch.special.bessel_j1(x) / safe)
                    return grad_out * dj1

            cls._impl = _Impl
        return cls._impl.apply(x)


class _TorchBesselJ1(metaclass=_TorchBesselJ1Meta):
    """``torch.special.bessel_j1`` with the analytic backward it is missing."""


def jinc(x):
    """``2 J1(x) / x``, with the removable singularity filled at 0 (value 1)."""
    xp = ac.array_namespace(x)
    safe = xp.where(x == 0, xp.ones_like(x), x)
    return xp.where(x == 0, xp.ones_like(x), 2.0 * j1(safe) / safe)


def sinc(z):
    """Unnormalized ``sin(z)/z``, with the removable singularity filled at 0."""
    xp = ac.array_namespace(z)
    safe = xp.where(z == 0, xp.ones_like(z), z)
    return xp.where(z == 0, xp.ones_like(z), xp.sin(safe) / safe)
