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
from typing import NamedTuple

import numpy as np

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
    cutoff has to sacrifice one of them. Measured against double precision (``truncation_order=30``,
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
                'precision (complex_dtype=0), or trace a double-precision solve if you need '
                'jit; see docs/branch-cuts.md.',
                UnresolvedBranchWarning, stacklevel=3)
        return
    if hit:
        import warnings
        warnings.warn(
            'a square-root branch was decided from a real part smaller than this dtype can '
            'resolve, so the choice came from round-off rather than from the physics. The '
            'result is finite and may be badly wrong. Re-run at double precision '
            '(complex_dtype=0); docs/branch-cuts.md covers the single-precision case.',
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
    # `xp.sqrt` for the same reason as :func:`boundary_kz` -- `**0.5` leaves a 6.1e-17 real
    # part on a lossless propagating mode, where `Re(q)` is meant to be a hard 0. Nothing
    # downstream reads it at that size (the tie-break's `_FLAT` is 1e-12, and the residue moves
    # `|exp(-k0*q*d)|` by ~1e-17), so this is consistency of spelling rather than a fix; the
    # tie-break itself is untouched and still disagrees with `boundary_kz` on purpose.
    v = xp.sqrt(eigenvalue)
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
    # `xp.sqrt`, not `kz2 ** 0.5`. On a negative real `kz2` -- every purely evanescent order --
    # the two disagree exactly where it counts: `csqrt` returns `Re(kz)` as a hard 0, while
    # `**0.5` goes through `exp(0.5*log)` and leaves `cos(pi/2)` behind, 6.1e-17. NumPy and
    # Torch route `**0.5` back to `csqrt` and so hide it; JAX and Python's own `complex` do not.
    # The residue is not decorative: an evanescent order's efficiency carries a factor
    # `Re(kz)`, so the spelling alone decides whether a cut-off channel reports 0.0 or 1e-19.
    v = xp.sqrt(kz2)
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


def parse_excitation(excitation, half_width_u, half_width_v):
    """Read a declared excitation into ``((m, n), amplitude, key)`` triples, refusing what it is not.

    The original ``key`` rides along because a message that has to name an order should name it
    the way the caller wrote it -- ``5``, not the normalised ``(5, 0)``.

    One validator, called by both :func:`order_window` and :func:`lit_orders`, because the
    window is computed *before* the orders are mapped and an excitation that is going to be
    refused must not reshape the problem on its way to being refused.

    **A lit order has to sit inside the declared ``truncation_order``.** The window widens around the
    lit set (see :func:`order_window`), so without this rule that widening would be unbounded
    -- lighting order 9 at ``truncation_order=6`` would quietly solve a 22-order problem instead of
    saying that 9 is outside what was asked for. With it the widening is bounded at twice the
    declared half-width, and `truncation_order` keeps meaning what it says: the orders being modelled.
    """
    if not isinstance(excitation, dict):
        raise TypeError(
            f'excitation must be a dict of order -> complex amplitude, got '
            f'{type(excitation).__name__}. A 1D scene keys it by the x order alone '
            '({0: 1.0, 1: 0.5j}); a 2D one by the pair ({(0, 0): 1.0, (1, 0): 0.5j}).')
    if not excitation:
        raise ValueError(
            'excitation is empty, so nothing is lit and every answer would be zero. Leave it '
            'unset for the ordinary single wave on the specular order.')
    two_d = half_width_v > 0
    out = []
    for key, amplitude in excitation.items():
        pair = key if isinstance(key, tuple) else (key, 0)
        if len(pair) != 2 or not all(isinstance(v, (int, np.integer)) for v in pair):
            raise TypeError(
                f'excitation key {key!r} is not a diffraction order. Use an integer m'
                + (' pair (m, n)' if two_d else '')
                + ', which is what indexes the orders this solve retains.')
        m, n = int(pair[0]), int(pair[1])
        if not two_d and n != 0:
            raise ValueError(
                f'excitation names order {key!r}, but this solve retains only n = 0 -- '
                'truncation_order[1] is 0, so there is no out-of-plane order to light. Both 1D '
                'formulations keep the single n = 0 order.')
        if abs(m) > half_width_u or abs(n) > half_width_v:
            raise ValueError(
                f'excitation names order {key!r}, which is outside the declared '
                f'truncation_order=({half_width_u}, {half_width_v}). The retained window widens around '
                'the orders you light, so this is bounded on purpose: raise truncation_order if that '
                'order is really part of the illumination.')
        out.append(((m, n), amplitude, key))
    return out


def order_window(half_width_u, half_width_v, excitation):
    """Which orders the solve retains, as ``(centre, half_width)`` per axis.

    Without an excitation this is ``(0, truncation_order)`` -- the symmetric window centred on the
    specular order that this library has always used, and the reason the pair is spelled this
    way rather than as a range: ``n_orders = 2*half_width + 1`` stays true, so every size
    computation downstream is untouched and only the *offset* is new.

    With one, every lit order needs ``truncation_order`` neighbours on each side, or a beam sitting
    off-centre is worse resolved than it would be solved alone. That union spans
    ``[m_min - truncation_order, m_max + truncation_order]``, and the smallest symmetric window containing it
    is returned -- one order wider than the union when that span is odd, since a symmetric
    window cannot straddle a half-integer centre. Measured 3.6 to 9.1 percent, falling with the
    order count because the addition is fixed while ``n_orders`` grows.

    Without the widening the retained set would stay centred on the specular order and a beam
    at ``+1`` would see one fewer neighbour above it than below. That measures as a *tie*
    against solving beam by beam; with it, better. The extra orders are what buys the
    difference.
    """
    if excitation is None:
        return (0, int(half_width_u)), (0, int(half_width_v))
    orders = [order for order, _, _ in parse_excitation(excitation, half_width_u, half_width_v)]

    def span(lit, half):
        lo, hi = min(lit) - int(half), max(lit) + int(half)
        centre = (lo + hi) // 2
        return centre, max(centre - lo, hi - centre)

    return span([m for m, _ in orders], half_width_u), span([n for _, n in orders], half_width_v)


class LitOrder(NamedTuple):
    """One order an excitation lights: where it sits, what weight, and what to call it."""
    #: Flat position into the ``(n_orders_v, n_orders_u)`` grid, row-major.
    index: int
    #: The complex weight written for it. May be traced.
    amplitude: object
    #: The diffraction order as the caller wrote it, for messages that have to name it.
    order: object


def lit_orders(excitation, window_x, window_y):
    """Which retained orders an ``excitation`` lights, as a tuple of :class:`LitOrder`.

    The spatial twin of :func:`incident_sp_amplitudes`: that one resolves *which polarization*
    arrives, this one *which plane waves*, and both are called once at the boundary so the
    numerics never learn there was a choice. ``None`` in, ``None`` out -- a solver that named no
    excitation is the ordinary single wave on the specular order, and every site keeps the
    expression it had, which is what stops a golden moving by a rounding step for a feature the
    scene does not use.

    Keys are the diffraction order: a bare ``m`` where there is no ``y`` extent, and ``(m, n)``
    otherwise. Values are complex amplitudes and may be traced -- they are ordinary design
    variables. The **keys** may not be: they decide which entries of the incident column are
    non-zero, which is a structural question, so they are read as Python integers and never
    compared against a traced value.

    ``window_x`` / ``window_y`` are :func:`order_window`'s ``(centre, half_width)`` pairs, and
    the indices returned are flat into the ``(n_orders_v, n_orders_u)`` grid the solves carry,
    row-major -- so a caller writes them straight into a column of that length. Reading the
    window rather than ``truncation_order`` is what lets the retained set sit off the specular order,
    which is the whole point of widening it.
    """
    if excitation is None:
        return None
    (centre_u, half_u), (centre_v, half_v) = window_x, window_y
    n_orders_u = 2 * half_u + 1
    # Validated against the *declared* truncation_order, which the window was widened from: a lit
    # order is inside `centre +- half` by construction, so re-deriving the declaration is what
    # keeps the refusal meaningful rather than vacuous.
    declared_x = max(abs(centre_u - half_u), abs(centre_u + half_u))
    declared_y = max(abs(centre_v - half_v), abs(centre_v + half_v))
    resolved = []
    for (m, n), amplitude, key in parse_excitation(excitation, declared_x, declared_y):
        resolved.append(LitOrder((n - centre_v + half_v) * n_orders_u
                                 + (m - centre_u + half_u), amplitude, key))
    return tuple(resolved)


def incident_column(xp, C, lit, n_orders, specular):
    """The incident amplitude of every retained order, as a flat column.

    ``lit`` is :func:`lit_orders`' answer, and ``None`` — nothing declared — gives back exactly
    the unit vector on ``specular`` that this library has always used, to the bit. That is not
    a fast path but the definition: one wave on the specular order *is* the default excitation,
    so there is no second expression to drift from.
    """
    if lit is None:
        column = [0.0] * n_orders
        column[specular] = 1.0
        return xp.astype(xp.asarray(column), C)
    column = [xp.astype(xp.asarray(0.0), C)] * n_orders
    for one in lit:
        column[one.index] = xp.astype(as_array(xp, one.amplitude, dtype=C), C)
    return xp.stack(column)


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
