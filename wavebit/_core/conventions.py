"""Centralized sign / branch-cut conventions.

Wavebit works in the ``e^{-iωt}`` time convention throughout — inputs, internals and
returned amplitudes alike. ``Im(eps) > 0`` is loss — equivalently ``Im(n) > 0``, which is what
refractive-index databases tabulate (``n + 1j*k``) — and a wave travelling in ``+z`` carries
``exp(+i*k0*kz*z)``. The quantity a *user* writes is the permittivity
(:class:`~wavebit._core.domain.Material`); the sign convention is the same one either way,
since the principal root maps the upper half plane into the first quadrant.

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

_PI = 3.141592653589793
_HALF_PI = 1.5707963267948966


@contextlib.contextmanager
def branch_tolerance(value):
    """Override the "is ``Re(v)`` zero?" cutoff for the enclosed solves. Advanced, and scoped.

    Exists because at single precision there is no right answer to bake in. ``Re(v)`` for a
    root that should be pure-imaginary is round-off ~1e-7 at complex64, and a weakly absorbing
    superstrate carries a *genuine* real part at the same scale — the two overlap, so any fixed
    cutoff has to sacrifice one of them. Measured against double precision (``max_order=30``,
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

        with wavebit.branch_tolerance(1e-6):
            res = solver.solve()
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


def incident_sp_amplitudes(xp, C, psi, chi=None, polarization=None):
    """The incident electric amplitudes ``(a_s, a_p)``, at unit flux — the one resolver.

    Three spellings arrive and one leaves, so the solves never learn there was a choice. This
    is called once per solve from :meth:`~wavebit._core.engine.CoreSolveMixin._incidence_pair`,
    at the boundary, rather than from inside the numerics.

    * ``psi`` — the polarization angle. ``(a_s, a_p) = (sin(psi), cos(psi))``.
    * ``chi`` — the ellipticity angle beside it. Together they are the canonical coordinates
      of the state: ``chi = 0`` returns exactly the pair above, ``|chi| = pi/4`` the circular
      states. Naming ``chi`` is what selects the chart; leaving it ``None`` means zero.
    * ``polarization`` — a pair of complex amplitudes written directly, which is the only
      spelling that can carry an overall phase. When given it wins and the angles are unread.

    **Only the written pair is normalized**, and that is a statement about arithmetic rather
    than about spellings. The two angle forms are unit by construction --
    ``|a_s|^2 + |a_p|^2 = cos^2(chi) + sin^2(chi) = 1`` -- so dividing them by their own
    computed norm cannot remove an error, only add one: ``sqrt(...)`` differs from ``1.0`` by
    up to half an ulp for the angle and 1.5 for the chart, on 18% and 36% of angles
    respectively. A written pair has no such guarantee (``(3, 4j)`` is a legitimate way to ask
    for the same incidence), so it is divided, and its overall *phase* survives because the
    norm is real positive.

    Both entries are E-field amplitudes even though the coupled paths carry the p channel as
    a magnetic boundary variable — the ``-i*n_top`` conversion belongs to the incident-column
    builders, exactly as the ``/(-i*n)`` on the way out belongs to the pack sites.
    """
    if polarization is not None:
        a_s = as_array(xp, polarization[0], dtype=C)
        a_p = as_array(xp, polarization[1], dtype=C)
        norm = xp.astype(xp.sqrt(xp.abs(a_s) ** 2 + xp.abs(a_p) ** 2), C)
        return a_s / norm, a_p / norm
    psi = as_array(xp, psi, dtype=C)
    if chi is None:
        # `chart(psi, 0)` reduces to this exactly -- `cos(0)` is 1.0 and `sin(0)` is 0.0, both
        # exact -- so the reduction is written out rather than computed, and the plain angle
        # never pays for the two multiplications and the addition that would be no-ops.
        return xp.sin(psi), xp.cos(psi)
    chi = as_array(xp, chi, dtype=C)
    return (xp.sin(psi) * xp.cos(chi) + 1j * xp.cos(psi) * xp.sin(chi),
            xp.cos(psi) * xp.cos(chi) - 1j * xp.sin(psi) * xp.sin(chi))


def safe_sqrt(xp, z, eps=1e-30):
    """``sqrt`` with a finite derivative at 0 (guards autodiff NaN at the DC/branch point)."""
    return (z + eps) ** 0.5


def order_azimuth(xp, ky, kx, phi):
    """Azimuth of each retained order's in-plane wavevector, flattened.

    ``cos(varphi)`` and ``sin(varphi)`` rotate an order into its own s/p frame, so every
    solve that couples the two polarizations needs this. The frame is the standard one,
    ``s_hat = z_hat x k_hat_par`` — i.e. ``(cos(varphi), sin(varphi))`` is the *direction* of
    ``(kx, ky)``, not merely a line parallel to it. That is ``atan2(ky, kx)``, and it is
    ``atan(ky/kx)`` only on the ``Re(kx) >= 0`` half-plane; see **the half-turn** below.

    **The ``kx == 0`` column with ``ky != 0``** — every order with ``m == 0`` and ``n != 0``
    on a 2D grid, present at *any* incidence, not only normal. Nothing is ambiguous: the
    wavevector points along ``y`` and the azimuth is a right angle. ``+-pi/2`` by the sign of
    ``Re(ky)``, and ``+pi/2`` when that is zero.

    **The origin**, ``kx == ky == 0`` — the zeroth order at exactly normal incidence, where
    the in-plane wavevector carries no direction at all. The answer is ``phi``, and it is
    **not** a free gauge, which is the trap here. A wave going straight down does have a
    degenerate plane of incidence, but the incident *polarization* still needs a reference
    direction in the ``xy`` plane, and ``phi`` is by convention what supplies it. Filling
    this with ``0`` instead makes ``F_c = 1``, ``F_s = 0``, collapses the s/p mixing, and
    sends field components that should be O(1) to 1e-17 — silently, and only when
    ``phi != 0``. ``tests/references/test_s4_field_golden.py`` is what catches it; twelve
    ``theta = 0, phi = 35 deg`` cases fail and every ``phi = 0`` case still passes.

    Neither answer is invented; both are the limit the library used to reach the long way
    round, through a ``perturbation`` constructor argument that no longer exists. ``theta``
    was replaced by a tiny non-zero value (``1e-20`` by default) whenever it was exactly
    zero, which made ``kx_0 = eps*cos(phi)`` and ``ky_0 = eps*sin(phi)`` — so ``atan``
    recovered the origin and ``+-pi/2`` for the column, on its own.

    So the detour was not merely tidy, it was carrying the incidence azimuth into ``kx`` and
    ``ky`` infinitesimally — which is precisely why nobody could tell what the knob did. It
    is passed explicitly now.

    **The half-turn**, which is the whole reason the ``+ pi`` term below exists. ``atan``
    is handed the *ratio* ``ky/kx``, and a ratio has already thrown ``sign(kx)`` away: the
    orders ``(kx, ky)`` and ``(-kx, -ky)`` produce the same ``atan``, so its answer is wrong
    by ``pi`` over the whole ``Re(kx) < 0`` half-plane. A half-turn negates both ``cos`` and
    ``sin``, so it does not tilt the s/p frame — it reverses it, ``s_hat -> -s_hat`` and
    ``p_hat -> -p_hat``, for that order alone.

    That reversal is a diagonal ``+-1`` similarity on the whole stack algebra, so it cancels
    out of every power quantity and out of the reconstructed field, and shows up in exactly
    one place: the sign of the reported ``r_s`` / ``r_p`` / ``t_s`` / ``t_p`` of each
    ``Re(kx) < 0`` order. It was invisible for as long as nothing compared amplitudes across
    orders. Three measurements say it is a defect rather than a convention, and they are what
    ``tests/test_amplitude_sp_frame.py`` now pins:

    * **it is discontinuous.** Sweep ``theta`` so one order's own ``kx`` crosses zero and
      nothing physical happens — the order stays propagating, ``R`` and ``|r_s|`` sail
      through — but ``r_s`` changes sign. No convention can do that.
    * **it disagreed with the ``kx == 0`` column handled just below.** That column takes its
      sign from ``Re(ky)``, which is the ``z_hat x k_hat`` rule; the ``Re(kx) < 0``
      half-plane did not. The function was discontinuous at the very column it special-cases.
    * **it disagreed with this library's own reconstructed field.** Decompose the field at
      the top face into orders and project onto ``z_hat x k_hat``: the ``Re(kx) > 0`` orders
      matched ``r_s`` to 1e-15 and the ``Re(kx) < 0`` orders matched ``-r_s``.

    The flip is keyed on ``Re(kx)`` rather than on a norm so that the ``kx == 0`` column —
    including its purely-imaginary-``ky`` corner, where the branch is a genuine choice rather
    than a mistake — comes out bit-identical to what it was before.

    **The origin is the same bug**, and gets the same term. There the "``kx``" whose sign the
    ratio discards is ``cos(phi)``, so ``phi`` and ``phi + pi`` landed on one frame and a solve
    handed ``phi + pi`` quietly answered for ``phi`` — an incident field of the opposite sign
    to the one asked for.

    It survived a round longer because it looked like it bought ``phi``/``phi + pi``
    equivalence, on the argument that turning the reference azimuth flips ``s_hat`` and
    ``p_hat`` together so the sign cancels in every ratio. It cancels for the zeroth order and
    for no other: at ``theta = 0`` the grid is ``kx_m = m*wl/Px``, ``ky_n = n*wl/Py``, which
    does not contain ``phi``, so no other order's frame turns with it. ``(phi + pi, psi)`` and
    ``(phi, polarization=(-a_s, -a_p))`` are the same physical field written two ways, and they
    disagreed by a sign on all 80 off-centre orders while ``R`` agreed to 2.8e-16 and hid it.
    ``test_the_two_spellings_of_one_incidence_agree`` is that measurement.

    The complex dtype is what made the detour load-bearing rather than optional: ``ky / 0``
    is ``inf + nan*j`` for a complex array rather than the real case's ``inf``, so the NaN
    reached every element through the block algebra. In real arithmetic IEEE would have
    carried ``atan(inf) = pi/2`` by itself and no knob would ever have been needed.

    The clamp on the denominator is not cosmetic either. ``where`` evaluates both branches,
    so an unguarded ``ky/kx`` would still build that NaN in the discarded one — finite in
    the forward pass, but ``0 * NaN`` in the backward one, which is how a singular point
    silently destroys a gradient while every value-based test keeps passing.
    """
    zero_kx = kx == 0
    safe_kx = xp.where(zero_kx, xp.ones_like(kx * ky), kx)
    varphi = xp.atan(ky / safe_kx)
    # Put back the sign the ratio discarded. `pi`, not `-pi`: only cos and sin are read, and
    # they cannot tell the two apart, so the cheaper spelling is also the only one.
    varphi = varphi + _PI * xp.astype(xp.real(kx + xp.zeros_like(varphi)) < 0, varphi.dtype)

    ky_b = ky + xp.zeros_like(varphi)                 # broadcast to the order grid
    right_angle = xp.where(xp.real(ky_b) < 0,
                           xp.full_like(varphi, -_HALF_PI),
                           xp.full_like(varphi, _HALF_PI))
    varphi = xp.where(zero_kx, right_angle, varphi)

    p = as_array(xp, 0.0 if phi is None else phi, dtype=varphi.dtype)
    # Same term, same reason: at the origin the "kx" whose sign the ratio dropped is cos(phi).
    incident = (xp.atan(xp.sin(p) / xp.cos(p))
                + _PI * xp.astype(xp.real(xp.cos(p)) < 0, varphi.dtype)) + xp.zeros_like(varphi)
    return xp.reshape(xp.where(zero_kx & (ky == 0), incident, varphi), (-1,))
