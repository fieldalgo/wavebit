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


def internal_index(xp, n):
    """A refractive index in the convention the solve runs in.

    A Python (or NumPy) scalar stays a scalar, and stays weakly typed with it: ``asarray`` of a
    Python ``complex`` becomes ``complex64`` under PyTorch, losing ~8 digits that the later
    promotion to ``complex128`` cannot bring back.
    """
    if isinstance(n, (int, float)):      # real: conjugation is the identity
        return n
    if isinstance(n, complex):
        return n.conjugate()
    return xp.conj(n)


def boundary_kz(xp, n, kx, ky=None):
    """Half-space ``kz`` in the internal convention, for the given retained orders.

    Computes the physical root and conjugates it, which is only the same thing as conjugating
    the *expression* when the in-plane wavevector is real. It is not once ``n_top`` is complex:
    ``kx`` arrives already conjugated, so pairing it with a raw ``n**2`` mixes the two
    conventions and the reflected amplitude comes out wrong at oblique incidence.

    Written the way the physical formula reads, term by term, rather than collecting the
    in-plane terms first.
    """
    if ky is None:
        return xp.conj((n ** 2 - xp.conj(kx) ** 2) ** 0.5)
    return xp.conj((n ** 2 - xp.conj(kx) ** 2 - xp.conj(ky) ** 2) ** 0.5)


def substrate_eps(xp, n_bot):
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
    """
    return internal_index(xp, n_bot) ** 2


def safe_sqrt(xp, z, eps=1e-30):
    """``sqrt`` with a finite derivative at 0 (guards autodiff NaN at the DC/branch point)."""
    return (z + eps) ** 0.5
