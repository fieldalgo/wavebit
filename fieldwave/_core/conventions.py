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

Both pick "downward", but they parametrize it differently, so ``q = -i*kz`` relates them —
for a real in-plane wavevector. It is an identity there and nothing else: with a complex
``k_par`` the two functions return *unrelated* roots, and that is correct rather than a
defect. A half-space's root is a boundary condition, so there is one right answer; a layer's
``±q`` pair is used in full by the stack algebra, so its label is free and ``mode_q`` spends
that freedom on keeping ``exp(-k0*q*d)`` inside the unit disc. Driving a layer with
``boundary_kz``'s rule instead makes 47% of its modes grow and raises ``FloatingPointError``
by 4 um of thickness. Pinned in ``tests/test_branch_selection.py``.

Selecting on ``Im`` alone is wrong for either: with a complex in-plane wavevector (an
absorbing superstrate at oblique incidence) even a lossless layer can have ``Im(kz) < 0``,
and forcing it positive returns ``|r| > 1``. Selecting on ``Re`` alone is *also* not the
whole rule for a half-space — see :func:`_off_principal_branch`, which is what decides
between them where they part. ``docs/time-convention-audit.md`` derives all of it.

Backend-agnostic: pass the array namespace as ``xp`` (``numpy`` / ``jax.numpy`` / ``torch``).
"""

_FLAT = 1e-12      # below this, a real part counts as zero and the tie-break decides


def mode_q(xp, eigenvalue):
    """Modal decay constant ``q`` of a layer mode, from its eigenvalue ``q**2``.

    A mode varies as ``exp(-k0*q*z)`` with ``z`` measured downward, so the root taken is the
    one that does not grow: ``Re(q) >= 0``. For a lossless propagating mode ``Re(q)`` is zero
    and the root is imaginary; there the downward wave is ``q = -i*kz``, i.e. ``Im(q) <= 0``.

    Both of those are *conditioning* statements, not physics. A layer's ``±q`` pair is used in
    full by the stack algebra, so the label is free: flipping any subset of a layer's roots
    moves the solve by 1.5e-13 at 40 nm, and the Airy recursion by 2.0e-15 over 4000 random
    stacks. What ``Re(q) >= 0`` buys is ``|exp(-k0*q*d)| <= 1``; the ``Im(q) <= 0`` tie-break
    fires only where that modulus is 1 either way and so buys nothing measurable — inverting
    it moves the answer by exactly 0.

    So do not "repair" this to agree with :func:`boundary_kz` at complex ``k_par``. The two
    disagree there on purpose, and the disagreement is pinned in
    ``tests/test_branch_selection.py``. See ``docs/time-convention-audit.md`` §5.
    """
    v = eigenvalue ** 0.5
    flip = xp.where(xp.abs(xp.real(v)) > _FLAT, xp.real(v) < 0, xp.imag(v) > 0)
    return xp.where(flip, -v, v)


def boundary_kz(xp, eps, k_par_sq):
    """``kz`` of a homogeneous half-space for the downward-going wave.

    ``exp(+i*k0*kz*z)`` travels down, so the physical root carries energy that way:
    ``Re(kz) > 0``. A purely evanescent order has no real part, and there the decaying
    choice is ``Im(kz) > 0``. That pair is the principal square root, and it is the whole
    answer for every real ``k_par`` — a passive medium has ``Im(eps) >= 0``, so ``kz**2``
    lies in the closed upper half plane and its principal root is in the first quadrant,
    satisfying *both* conditions at once. The two only become separate demands when
    ``k_par`` is complex, and then a third thing decides between them; see
    :func:`_off_principal_branch` and ``docs/time-convention-audit.md``.
    """
    kz2 = eps - k_par_sq
    v = kz2 ** 0.5
    flip = xp.where(xp.abs(xp.real(v)) > _FLAT, xp.real(v) < 0, xp.imag(v) < 0)
    v = xp.where(flip, -v, v)
    return xp.where(_off_principal_branch(xp, eps, k_par_sq, kz2), -v, v)


def _off_principal_branch(xp, eps, k_par_sq, kz2):
    """True where the principal root is on the wrong sheet of ``sqrt(eps - k_par_sq)``.

    ``kz**2`` has exactly one branch point, at ``kz**2 == 0``, so the physical root is the
    one reached by deforming continuously from a configuration that is not in doubt. Take
    the in-plane wavevector to zero:

        kz2(t) = eps - t*k_par_sq,     t: 0 -> 1

    At ``t = 0`` this is ``sqrt(eps)`` — normal incidence in a homogeneous passive medium,
    where nothing is ambiguous. ``Im(kz2(t))`` is *linear* in ``t``, so the segment meets the
    real axis at most once, at ``t* = Im(eps)/Im(k_par_sq)``, and the principal root (whose
    cut is the negative real axis) stops being the continuous one exactly when that meeting
    happens on the negative side. Hence the three conditions below, and a sign flip when all
    three hold.

    It cannot fire for a real ``k_par_sq``: ``Im(kz2)`` is then constant at ``Im(eps) >= 0``
    and the segment never leaves the closed upper half plane. Measured bit-for-bit identical
    over 4e5 random passive half-spaces at real ``k_par``, lossless and lossy alike.

    What it does fire on is a half-space under a *lossier* one at oblique incidence — both
    sides absorbing, which makes ``Im(k_par_sq)`` exceed ``Im(eps)``. There the principal
    root has ``Im(kz) < 0``: a field growing without bound into an absorbing medium that has
    no source in it. The symptom is a step discontinuity in the reflected amplitude as
    ``theta`` is swept — 2.96 in ``r`` across 6.7e-05 rad for a lossy dielectric on gold,
    at an angle where ``kz**2`` stands 9.03 clear of its branch point and the physics is
    analytic. See ``docs/time-convention-audit.md`` §4.
    """
    zero = xp.zeros_like(kz2)
    eps_a, kp = eps + zero, k_par_sq + zero
    down = xp.imag(kp) > 0                       # only a descending segment can reach R-
    t = xp.imag(eps_a) / xp.where(down, xp.imag(kp), xp.ones_like(xp.imag(kp)))
    return xp.logical_and(down,
                          xp.logical_and(t < 1, (xp.real(eps_a) - t * xp.real(kp)) < 0))


def incident_kz(kz_top, index):
    """The incident order's ``kz``, read off the superstrate vector rather than recomputed.

    The boundary match sets a reflected field built from ``kz_top`` (via :func:`boundary_kz`)
    against an incident field. Both sides need the *same* number at the incident order, and
    the reliable way to guarantee that is to take it from the vector the other side already
    uses — ``kz_top[index]`` — instead of writing ``n_top * cos(theta)`` a second time.

    The two spellings agree for a real incidence angle, which is why the duplication survived:
    it is an identity there and nothing in the suite could see it. They part company as soon
    as ``theta`` is complex (evanescent illumination), where ``cos(theta)`` is
    negative-imaginary while :func:`boundary_kz` forces ``Im(kz) > 0``. The two sides then
    carry opposite signs at that one order and cancel exactly, and the solve returns
    ``R = -delta`` with ``T = 0`` — a grating that scatters nothing, which plots as a clean
    flat region rather than as the failure it is.
    """
    return kz_top[index]


def safe_sqrt(xp, z, eps=1e-30):
    """``sqrt`` with a finite derivative at 0 (guards autodiff NaN at the DC/branch point)."""
    return (z + eps) ** 0.5
