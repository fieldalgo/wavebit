"""Centralized sign / branch-cut conventions.

FieldWave works in the ``e^{-iωt}`` time convention throughout — inputs, internals and
returned amplitudes alike. ``Im(n) > 0`` is loss, which is what refractive-index databases
tabulate (``n + 1j*k``), and a wave travelling in ``+z`` carries ``exp(+i*k0*kz*z)``.

There is no second convention hiding underneath. Until 2026-08 there was: ``kx`` and the
permittivity were conjugated on the way in, the algebra ran on the conjugated problem, and the
returned amplitudes were never conjugated back. Every boundary added after that had to guess
which side of the seam it was on, and three of them guessed wrong (absorbing substrates and
superstrates came out as gain media). The seam is gone; this module holds what replaced it.

Two branch rules, and they are *not* the same rule:

``mode_q``       a layer's modal constant, used as ``exp(-k0*q*z)`` going down.
``boundary_kz``  a half-space's ``kz``, used as ``exp(+i*k0*kz*z)`` going down.

Both pick "downward", but they parametrize it differently, so ``q = -i*kz`` relates them and
their tie-breaks come out opposite. Selecting on ``Im`` alone is wrong for either: with a
complex in-plane wavevector (an absorbing superstrate at oblique incidence) even a lossless
layer can have ``Im(kz) < 0``, and forcing it positive returns ``|r| > 1``.

Backend-agnostic: pass the array namespace as ``xp`` (``numpy`` / ``jax.numpy`` / ``torch``).
"""

_FLAT = 1e-12      # below this, a real part counts as zero and the tie-break decides


def mode_q(xp, eigenvalue):
    """Modal decay constant ``q`` of a layer mode, from its eigenvalue ``q**2``.

    A mode varies as ``exp(-k0*q*z)`` with ``z`` measured downward, so the physical root is
    the one that does not grow: ``Re(q) >= 0``. For a lossless propagating mode ``Re(q)`` is
    zero and the root is imaginary; there the downward wave is ``q = -i*kz``, i.e. ``Im(q) <= 0``.
    """
    v = eigenvalue ** 0.5
    flip = xp.where(xp.abs(xp.real(v)) > _FLAT, xp.real(v) < 0, xp.imag(v) > 0)
    return xp.where(flip, -v, v)


def boundary_kz(xp, eps, k_par_sq):
    """``kz`` of a homogeneous half-space for the downward-going wave.

    ``exp(+i*k0*kz*z)`` travels down, so the physical root carries energy that way:
    ``Re(kz) > 0``. A purely evanescent order has no real part, and there the decaying
    choice is ``Im(kz) > 0``.
    """
    v = (eps - k_par_sq) ** 0.5
    flip = xp.where(xp.abs(xp.real(v)) > _FLAT, xp.real(v) < 0, xp.imag(v) < 0)
    return xp.where(flip, -v, v)


def safe_sqrt(xp, z, eps=1e-30):
    """``sqrt`` with a finite derivative at 0 (guards autodiff NaN at the DC/branch point)."""
    return (z + eps) ** 0.5
