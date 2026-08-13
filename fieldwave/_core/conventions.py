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

import contextlib

from .backend import UnresolvedBranchWarning, as_array

_FLAT = 1e-12      # below this, a real part counts as zero and the tie-break decides


@contextlib.contextmanager
def branch_tolerance(value):
    """Override the "is ``Re(v)`` zero?" cutoff for the enclosed solves. Advanced, and scoped.

    Exists because at single precision there is no right answer to bake in. ``Re(v)`` for a
    root that should be pure-imaginary is round-off ~1e-7 at complex64, and a weakly absorbing
    superstrate carries a *genuine* real part at the same scale — the two overlap, so any fixed
    cutoff has to sacrifice one of them. Measured against double precision (``fto=30``,
    ``docs/branch-cuts.md``):

    ============================  ==========  ==========  ==========
    ``value``                     1e-12       1e-6        1e-5
    ============================  ==========  ==========  ==========
    lossless, complex angle       4.51        1.3e-06     3.4e-07
    weakly absorbing superstrate  1.9e-04     7.4e-01     1.75
    ============================  ==========  ==========  ==========

    The rule that follows is statable, which is the only reason this is exposed rather than
    guessed at: **raise it for a lossless superstrate at a complex incidence angle; leave it
    alone if the superstrate absorbs at all.** Any absorption makes the root genuinely complex
    and the degeneracy never arises — a solve containing both regimes wants the default.

    At complex128 nothing here applies: round-off is ~1e-16, the default already stands four
    orders clear of it, and moving this can only make things worse. :class:`UnresolvedBranchWarning`
    fires exactly when the question has become unanswerable, so reach for this only after seeing it.

    Not thread-safe — it swaps a module-level constant for the duration of the block::

        with fieldwave.branch_tolerance(1e-6):
            res = solver.conv_solve()
    """
    global _FLAT
    old = _FLAT
    _FLAT = float(value)
    try:
        yield
    finally:
        _FLAT = old

# How many eps of slack to allow for the round-off accumulated in forming the radicand and
# taking the root. Not a branch threshold — it decides nothing, it only bounds where the
# branch decision stops being answerable. See :func:`_check_resolvable`.
_NOISE_EPS = 8.0

# Anything with a coarser eps than this is single precision. complex64 is 1.2e-7 and
# complex128 is 2.2e-16, so the gap is nine orders wide and the exact value is arbitrary.
_SINGLE_EPS = 1e-10


def _check_resolvable(xp, v):
    """Warn when the branch is being decided from a real part the dtype cannot resolve.

    :func:`mode_q` and :func:`boundary_kz` both ask "is ``Re(v)`` zero?" and answer it against
    ``_FLAT``. That question has no dtype-independent answer: round-off on ``Re(v)`` is ~1e-16
    at complex128 and ~1e-7 at complex64, and ``_FLAT`` sits between them. At single precision
    the comparison therefore reads the *sign of noise*, and no other threshold fixes it —
    a lossless layer at a complex angle needs the cutoff raised, a weakly absorbing superstrate
    needs it left alone, and their real parts overlap. Measured in ``docs/branch-cuts.md``.

    So this reports rather than repairs. It is silent at complex128 (0 of 244 decisions
    unresolvable across lossless, complex-angle and weakly-absorbing cases) and fires at
    complex64 (12 to 128 of 244), which makes it a check on the precision rather than on the
    physics — and it costs nothing measurable, being two absolute values on an array the solve
    already built.

    Unlike the guards in ``_core/backend.py`` this is **not** NumPy-only, and deliberately so:
    the backends it would have skipped are the ones that need it most. Measured at the same
    complex64 point, NumPy returns ``|R| = 1.80`` and warns while JAX returns 4.12 and PyTorch
    5.68 — the two worst answers, silently, on the paths people reach for when they want
    gradients or a GPU.

    What actually blocks the check is an *abstract* value, not a namespace: a JAX tracer under
    ``jit``/``grad``, a PyTorch ``FakeTensor`` under ``compile``. So the reduction runs in the
    caller's own namespace and only the resulting scalar is read, which is one boolean of
    device sync rather than a copy of the array — and when that read is impossible the value
    was abstract and there is nothing to look at anyway.
    """
    try:
        eps = xp.finfo(v.dtype).eps
    except Exception:
        return                                  # not a float/complex dtype
    re, mag = xp.abs(xp.real(v)), xp.abs(v)
    noise = _NOISE_EPS * eps * xp.maximum(mag, xp.asarray(1e-30, dtype=mag.dtype))
    flagged = xp.any((re > _FLAT) & (re < noise))
    try:
        hit = bool(flagged)
    except Exception:
        # Traced — the value is not there to be read. Silence would be the wrong answer here:
        # the check *not running* and the check *finding nothing* are different states, and a
        # jit'd single-precision solve is exactly where a wrong branch goes unnoticed (measured:
        # |R| = 3.93 under jit, no warning, against 0.67 at double precision). The dtype is
        # known at trace time even when the values are not, so say that much.
        if eps > _SINGLE_EPS:
            import warnings
            warnings.warn(
                'this solve is traced, so whether its square-root branches are resolvable '
                'could not be checked — and it is running in single precision, where they '
                'often are not. A wrong branch here is finite and silent. Run at double '
                'precision (type_complex=0), or trace a double-precision solve if you need '
                'jit; see docs/branch-cuts.md.',
                UnresolvedBranchWarning, stacklevel=3)
        return
    if hit:
        import warnings
        warnings.warn(
            'a square-root branch was decided from a real part smaller than this dtype can '
            'resolve, so the choice came from round-off rather than from the physics. The '
            'result is finite and may be badly wrong. Re-run at double precision '
            '(type_complex=0); docs/branch-cuts.md covers the single-precision case.',
            UnresolvedBranchWarning, stacklevel=3)


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
    _check_resolvable(xp, v)
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
    _check_resolvable(xp, v)
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


def incident_sp_amplitudes(xp, C, psi, polarization=None):
    """The incident wave's electric amplitudes ``(a_s, a_p)``, normalized to unit flux.

    This is the one place the two spellings of the incident polarization meet the solves:

    * ``psi`` — the polarization angle: ``(a_s, a_p) = (sin(psi), cos(psi))``, the exact
      operations the boundary match has always applied, so a ``psi`` solve is bit-identical
      to what it was before ``polarization`` existed.
    * ``polarization`` — a pair of complex amplitudes in the same ``(s, p)`` basis, which is
      what ``psi`` cannot spell: a relative phase. ``(1, -1j)/sqrt(2)`` and ``(1, +1j)/sqrt(2)``
      are the circular states. When given, it wins and ``psi`` is not read.

    The pair's overall magnitude is *not* a degree of freedom: it is divided out here, so
    efficiencies are per unit incident flux and amplitudes per unit incident E-field whatever
    scale the caller wrote — ``(3, 4j)`` and ``(0.6, 0.8j)`` are the same incidence. The
    overall *phase* is kept (the norm is real positive), and carries into the returned
    complex amplitudes. ``psi = v`` and ``polarization = (sin(v), cos(v))`` are equivalent
    by construction; pinned in ``tests/test_polarization.py``.

    Both entries are E-field amplitudes even though the coupled paths carry the p channel as
    a magnetic boundary variable — the ``-i*n_top`` conversion belongs to the incident-column
    builders, exactly as the ``/(-i*n)`` on the way out belongs to the pack sites.
    """
    if polarization is None:
        # No normalization here, and that is not an omission. `psi` is real (the setters
        # refuse a complex one — see `engine.py::_reject_complex_angle`), so `(sin, cos)` is
        # already a unit pair by construction, and dividing it by its own norm again is not
        # free: `sqrt(sin^2 + cos^2)` differs from 1.0 by half an ulp for about one angle in
        # five, so the division moves the last bit of the amplitudes on those angles. The
        # legacy goldens pin those bits. The two spellings therefore share a rule rather
        # than a line, and `tests/test_polarization.py` is what keeps them equal.
        psi = as_array(xp, psi, dtype=C)
        return xp.sin(psi), xp.cos(psi)
    a_s = as_array(xp, polarization[0], dtype=C)
    a_p = as_array(xp, polarization[1], dtype=C)
    norm = xp.sqrt(xp.abs(a_s) ** 2 + xp.abs(a_p) ** 2)
    norm = xp.astype(norm, C)
    return a_s / norm, a_p / norm


def safe_sqrt(xp, z, eps=1e-30):
    """``sqrt`` with a finite derivative at 0 (guards autodiff NaN at the DC/branch point)."""
    return (z + eps) ** 0.5
