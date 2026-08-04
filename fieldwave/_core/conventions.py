"""Centralized sign / branch-cut conventions.

fieldwave uses the ``e^{-iωt}`` time convention (``Im(n) > 0`` = loss). Every substrate,
superstrate, and layer ``kz`` branch choice must draw from here, so a lossy case can never
silently disagree with a lossless one (the class of bug this module exists to prevent).

Backend-agnostic: pass the array namespace as ``xp`` (``numpy`` / ``jax.numpy`` / ``torch``).
The functions use only ops with matching signatures across those namespaces.
"""


def outgoing_sqrt(xp, arg):
    """``sqrt(arg)`` on the outgoing / decaying branch: ``Re(result) >= 0``.

    ``arg`` is the already-complex quantity ``eps - k_parallel**2 - 1e-20j``. The caller adds
    the ``-1e-20j`` nudge (which selects the correct root for lossless evanescent orders); the
    ``Re >= 0`` flip here makes the choice correct for lossy / absorbing media as well.

    This is the single home for the substrate/superstrate/layer kz branch. It is *not* the
    conjugated ``sqrt`` shortcut (which is a root of ``conj(arg)``, wrong for complex ``arg``),
    nor the ``-q_bot`` form (wrong on lossless evanescent orders).
    """
    kz = arg ** 0.5
    return xp.where(xp.real(kz) < 0, -kz, kz)


def substrate_eps(xp, n_bot, dtype):
    """Substrate permittivity in the convention the rest of the solve runs in.

    The internal convention is the conjugated one: ``get_kx_ky_vector`` conjugates ``kx``, and
    the factorization stores ``conj(eps)`` in the convolution matrices (a layer of ``n =
    0.18+3j`` lands as ``-8.9676-1.08j``). The substrate has to enter the boundary the same
    way, or a layer and a substrate of *the same material* end up with conjugate mode
    constants — which is exactly what happened: the stack reflected as though the substrate
    were ``conj(n_bot)``, absorbing read as gain
    (tests_verification/test_lossy_substrate.py). Real ``n_bot`` is unaffected, which is why
    this went unnoticed.

    The same-material detection already conjugates for this reason; this is that rule applied
    to the boundary too.

    ``dtype`` is the solve's complex working dtype, and passing it is not optional: PyTorch
    reads a bare Python ``complex`` as ``complex64``, and the resulting ~1e-8 error survives
    the later promotion to ``complex128``.
    """
    return xp.conj(xp.asarray(n_bot, dtype=dtype)) ** 2


def safe_sqrt(xp, z, eps=1e-30):
    """``sqrt`` with a finite derivative at 0 (guards autodiff NaN at the DC/branch point)."""
    return (z + eps) ** 0.5
