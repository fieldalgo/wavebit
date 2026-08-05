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


def _is_numpy(xp):
    """True for the eager NumPy namespace — the one backend whose values are always concrete."""
    name = xp.__name__
    return 'jax' not in name and 'torch' not in name


def check_finite(xp, **named):
    """Raise if any named eager-NumPy result carries a NaN or an infinity.

    The solve reaches the edges of double precision on purpose — a strongly evanescent mode's
    ``exp(-k0*q*d)`` lands in the subnormal range — so a result that came back non-finite is
    worth stopping on rather than returning. This is the check that says so.

    Only the eager NumPy path is checked. Under JAX the values are tracers during ``jit`` /
    ``grad`` (reading them would either fail or force a concretization that breaks the trace),
    and PyTorch's are likewise graph nodes; both follow the same eager-only branch convention
    the rest of the core uses.
    """
    if not _is_numpy(xp):
        return
    import numpy as np
    bad = sorted(k for k, v in named.items() if not bool(np.all(np.isfinite(np.asarray(v)))))
    if bad:
        raise FloatingPointError(
            f"non-finite value in {', '.join(bad)} — the solve did not produce a physical "
            'result. The usual causes are a degenerate structure (a singular interface: a '
            'layer identical to its neighbour that was not detected as such, a zero or '
            'near-zero thickness, an exactly grazing order) or a truncation the geometry '
            'cannot support. Try adjusting fto, re-slicing the stack into thinner layers, or '
            'use_pinv=True.')


class UnphysicalResultWarning(UserWarning):
    """A solve returned finite but physically impossible efficiencies.

    Its own category so it can be escalated (``warnings.simplefilter('error',
    UnphysicalResultWarning)`` gives a traceback with the values still in scope) or silenced
    independently of everything else the library might warn about.
    """


def check_physical(xp, de_ri, de_ti, n_top, atol=1e-3):
    """Warn if efficiencies are finite but cannot describe a physical result.

    :func:`check_finite` catches contamination that announces itself. This catches the tier
    below it — a solve that stayed finite and is still wrong, which nothing else would notice.
    Two signals, both properties of the result rather than of the arithmetic that produced it:

    * a negative efficiency, which is negative power;
    * ``sum(R) + sum(T) > 1`` for a passive stack, which is gain out of nothing.

    Deliberately *not* checked: ``R + T < 1``, which is just absorption. And the whole check is
    skipped for a complex ``n_top``, where the incident-power normalization is
    convention-dependent and the efficiencies are not comparable to 1 at all (the same reason
    ``test_lossy_media`` declines to compare ``de_ri`` there).

    Known false positive: a gain medium (``Im(n) < 0`` in this codebase's convention) legitimately
    exceeds 1. Filter :class:`UnphysicalResultWarning` if that is what you are modelling.

    The message carries no numbers, so Python's default filter collapses a sweep's worth of
    occurrences into one report instead of re-emitting per wavelength; escalate the category to
    an error when you want to catch a specific one.
    """
    if not _is_numpy(xp):
        return
    import numpy as np
    if bool(np.any(np.imag(np.asarray(n_top)) != 0)):
        return

    r = np.real(np.asarray(de_ri))
    t = np.real(np.asarray(de_ti))
    problems = []
    if float(min(np.min(r), np.min(t))) < -atol:
        problems.append('a negative efficiency (negative power)')
    if float(np.sum(r) + np.sum(t)) > 1 + atol:
        problems.append('R+T above 1 for a passive stack (gain out of nothing)')
    if problems:
        import warnings
        warnings.warn(
            f"solve returned {' and '.join(problems)}. The result is finite but not physical, "
            'so treat it as a failed solve: check for a degenerate structure, and try adjusting '
            'fto, re-slicing the stack into thinner layers, or use_pinv=True.',
            UnphysicalResultWarning, stacklevel=2)


def as_array(xp, v, dtype=None, device=None):
    """``v`` as an ``xp`` array, keeping any autograd graph intact.

    ``xp.asarray`` detaches under PyTorch, and does something worse when the dtype already
    matches: it hands back *the same tensor* with ``requires_grad`` flipped off in place, so a
    caller's optimizer leaf stops carrying gradient after one forward pass, silently. Values
    that are already arrays of this namespace are therefore cast with ``astype``, which is
    differentiable; everything else (Python scalars, NumPy arrays, lists) goes through
    ``asarray`` as before, where there is no graph to lose.

    Use this anywhere a *user-supplied* quantity enters the algebra — an index, an angle, a
    period, a wavelength, an edge position. `xp.asarray` is still right for values the core
    builds itself, such as index arrays and constants.
    """
    if hasattr(v, 'dtype') and hasattr(v, 'shape'):
        try:
            same_ns = ac.array_namespace(v) is xp
        except Exception:
            same_ns = False
        if same_ns:
            return v if dtype is None or v.dtype == dtype else xp.astype(v, dtype)
    if dtype is None:
        return xp.asarray(v, device=device)
    return xp.asarray(v, dtype=dtype, device=device)


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
