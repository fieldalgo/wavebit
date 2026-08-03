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


def safe_sqrt(xp, z, eps=1e-30):
    """``sqrt`` with a finite derivative at 0 (guards autodiff NaN at the DC/branch point)."""
    return (z + eps) ** 0.5
