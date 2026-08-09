"""Backend adapter — the only code that dispatches on the array framework.

The single-source core (migration Phase 3+) calls these; the current per-backend solvers do
not yet. Keeping the surface tiny is the point: everything divergent between NumPy / JAX /
PyTorch lives here, so it can be tested in one place.

``eig`` is the one primitive that genuinely differs, and the one whose *gradient* this module
supplies rather than delegates: both frameworks can differentiate a non-symmetric complex
``eig`` (JAX via ``enable_eigvec_derivs``, PyTorch natively) but both assume distinct
eigenvalues and return NaN when that fails, which an ordinary structure can cause. See the
note above :func:`eig`. NumPy needs no gradient.
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
            'cannot support. Try adjusting fto, or re-slicing the stack into thinner layers.')


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
            'fto or re-slicing the stack into thinner layers.',
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


# --- eig, and the one place eigenvalue degeneracy is handled ---------------------------
#
# The eigenvector derivative carries a ``1 / (w_i - w_j)``, so it is undefined where two
# eigenvalues coincide. That is not an exotic case here: a *uniform* layer at normal
# incidence has ``q(+m) == q(-m)`` exactly, for every retained order. Both frameworks'
# native rules return NaN there — the whole gradient, not just the degenerate directions —
# and a uniform buffer or spacer layer at normal incidence is an ordinary thing to put in a
# stack, so an inverse-design run walks into it without doing anything unusual.
#
# The fix is Lorentzian broadening of that reciprocal, ``1/d -> conj(d)/(|d|**2 + eps)``,
# which is what every differentiable RCWA does; the variation is in the epsilon. Broadening
# is not a patch over the singularity so much as the correct value at it: a degenerate
# eigenspace has a gauge freedom, a physical loss cannot depend on it, and the sensitivity
# the naive expression computes as 0/0 really is zero. The regularized gradient at exact
# degeneracy comes out equal to the limit approaching it, not merely finite.
#
# This follows fmmax (MIT, google/fmmax, `utils.py`), whose eps is relative to the
# eigenvalue range with an absolute floor rather than fixed (torcwa 1e-10, meent 1e-10 on
# Torch and 1e-20 on JAX). A fixed eps is tied to the matrix scale: measured against finite
# differences, the fixed one degrades from ~2e-10 to ~1e-7 as the operator shrinks four
# decades, where the scale-aware one stays flat. That difference is the reason to prefer it.
#
# The gauge term (the third one in `rhs`) also comes from fmmax; grcwa carries it, meent and
# torcwa drop it. It is kept here on their authority rather than on evidence: for a
# gauge-invariant loss — which every physical loss is — it provably contributes nothing, and
# no measurement made here separates it from its absence. It costs one matmul.
#
# Equation 4.77 of Boeddeker et al., arXiv:1701.00392. Cross-checked against both
# frameworks' native rules where those are defined (see tests/test_eig_grad).

EIG_EPS_RELATIVE = 1e-12
EIG_EPS_MINIMUM = 1e-24


def _eig_grad(xp, w, V, gw, gV):
    """Gradient of ``A`` given the eigendecomposition's, in PyTorch's complex convention.

    ``gw`` / ``gV`` are cotangents with respect to the *conjugates* of the outputs, which is
    what PyTorch hands a custom Function and what it wants back. JAX's convention is the
    other one, so :func:`_jax_eig` conjugates on the way in and out; that asymmetry is the
    only framework-specific part of this, and taking it on faith is how the PyTorch rule
    first came out wrong by a term (native and finite differences both said so).
    """
    n = w.shape[-1]
    I = xp.eye(n, device=_device_of(w))
    eye = xp.astype(I, xp.bool)

    delta = w[..., None, :] - w[..., :, None]
    delta_sq = xp.abs(delta) ** 2
    # Scale-aware: the broadening has to shrink with the problem, or the same structure in
    # different units gets a different gradient.
    eps = xp.maximum(EIG_EPS_RELATIVE * xp.max(delta_sq), as_array(xp, EIG_EPS_MINIMUM))
    f = xp.conj(delta) / (delta_sq + eps)
    f = xp.where(eye, xp.zeros_like(f), f)            # no broadening on the diagonal

    V_H = xp.matrix_transpose(xp.conj(V))
    VHgV = V_H @ gV

    rhs = (xp.astype(I, gw.dtype) * gw[..., None, :]
           + xp.conj(f) * VHgV
           - xp.conj(f) * (V_H @ V) @ xp.where(eye, xp.astype(xp.real(VHgV), VHgV.dtype),
                                               xp.zeros_like(VHgV))) @ V_H
    return xp.linalg.solve(V_H, rhs)


def _device_of(x):
    """``device=`` argument for building an array alongside ``x`` (None on NumPy/JAX)."""
    return x.device if hasattr(x, 'device') else None


def _eig_raw(A):
    """The framework's own eigendecomposition, with no gradient rule attached."""
    xp = ac.array_namespace(A)
    name = xp.__name__
    if 'torch' in name:
        import torch
        return torch.linalg.eig(A)
    if 'jax' in name:
        import jax.lax.linalg as jll
        out = jll.eig(A, compute_left_eigenvectors=False, compute_right_eigenvectors=True,
                      enable_eigvec_derivs=False)
        return out[0], out[-1]
    import numpy as np
    return np.linalg.eig(A)


def eig(A):
    """Eigendecomposition of a (possibly non-Hermitian) complex matrix.

    Returns ``(eigenvalues, right_eigenvectors)`` and is differentiable on JAX and PyTorch,
    including where eigenvalues are degenerate — see the note above this function.
    """
    xp = ac.array_namespace(A)
    name = xp.__name__
    if 'torch' in name:
        return _torch_eig(A)
    if 'jax' in name:
        return _jax_eig(A)
    return _eig_raw(A)


_JAX_EIG = None


def _jax_eig(A):
    """JAX ``eig`` with the broadened rule, built on first use so importing stays cheap."""
    global _JAX_EIG
    if _JAX_EIG is None:
        import jax

        @jax.custom_vjp
        def _eig_jax(A):
            return _eig_raw(A)

        def _fwd(A):
            w, V = _eig_raw(A)
            return (w, V), (w, V)

        def _bwd(res, g):
            import jax.numpy as jnp
            w, V = res
            grad_w, grad_V = g
            # JAX cotangents are with respect to the parameters; _eig_grad wants them with
            # respect to the conjugates, and returns one in the same convention.
            return (jnp.conj(_eig_grad(jnp, w, V, jnp.conj(grad_w), jnp.conj(grad_V))),)

        _eig_jax.defvjp(_fwd, _bwd)
        _JAX_EIG = _eig_jax
    return _JAX_EIG(A)


_TORCH_EIG = None


def _torch_eig(A):
    """PyTorch ``eig`` with the broadened rule (same construction as the JAX one)."""
    global _TORCH_EIG
    if _TORCH_EIG is None:
        import torch

        class _Eig(torch.autograd.Function):
            @staticmethod
            def forward(ctx, A):
                w, V = torch.linalg.eig(A)
                ctx.save_for_backward(w, V)
                return w, V

            @staticmethod
            def backward(ctx, grad_w, grad_V):
                w, V = ctx.saved_tensors
                return _eig_grad(ac.array_namespace(V), w, V,
                                 grad_w.resolve_conj(), grad_V.resolve_conj())

        _TORCH_EIG = _Eig
    return _TORCH_EIG.apply(A)


# --- Bessel J1: the second primitive that genuinely differs per framework ---------------
#
# NumPy has SciPy's j1. PyTorch has an accurate forward (`torch.special.bessel_j1`) but no
# autograd backward, so it needs the analytic derivative attached. JAX's
# `jax.scipy.special.bessel_jn` returns NaN for small arguments, so it needs its own
# implementation entirely. Same value, three routes — exactly what this adapter is for.

# The two pieces are textbook, not invented here: the ascending series is DLMF 10.2.2
# (Abramowitz & Stegun 9.1.10) and the large-argument form is Hankel's expansion, DLMF 10.17.3
# (A&S 9.2.1-9.2.6), whose P/Q coefficients are the a_k(nu) recursion built just below.
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
