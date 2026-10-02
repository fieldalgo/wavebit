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

"Down" and ``z`` here are the propagation kernels' local ones, as in
:mod:`~wavebit._core.solve_1d` and :mod:`~wavebit._core.scattering`: ``z`` runs from the
kernel's entry face into the stack, the way its incident wave travels. It is not, in general,
the public height: it coincides with it under bottom incidence and is its mirror
(``h_total - z``, ``h_total = sum(thickness)``) under top incidence. The public ``+z`` points up
from the bottom interface, and the engine binds the kernels' entry port to the public bottom
medium (:attr:`~wavebit._core.engine.CoreSolveMixin._kernel_incidence_side`), solving public
top incidence as their z-mirrored scene. So "down" is public ``+z`` under bottom incidence and
public ``-z`` under top incidence.

Both pick "downward", but they parametrize it differently, so ``q = -i*kz`` relates them —
for a real in-plane wavevector. It is an identity there and nothing else: with a complex
``k_par`` the two functions return *unrelated* roots, and that is correct rather than a
defect. A half-space's root is a boundary condition, so there is one right answer; a layer's
``±q`` pair is used in full by the stack algebra, so its label is free and ``mode_q`` spends
that freedom on keeping ``exp(-k0*q*d)`` inside the unit disc. A half-space rule may keep a
growing root, so it is no rule for a layer. Measured in 2026-08 with the half-space rule of
that day: driving a layer with it made 47% of the layer's modes grow and raised
``FloatingPointError`` by 4 um of thickness. Pinned in ``tests/test_branch_selection.py``.

For a half-space, ``Re(kz) > 0`` (carries energy down) and ``Im(kz) >= 0`` (does not grow)
name the same root -- the principal one -- at every real in-plane wavevector in a passive
medium. They part only where ``kz**2`` falls below the real axis: a complex ``theta``, or a gain
medium. Neither alone is the rule there. ``Im`` first takes the incoming wave of a propagating
order, and ``Re`` first lets an evanescent order grow at any ``Im(k_par)``, however small.
:func:`boundary_kz` takes Maystre's cut, which is the limit of the half-space's own absorption;
``docs/branch-cuts.md`` measures it against the alternatives.

Backend-agnostic: pass the array namespace as ``xp`` (``numpy`` / ``jax.numpy`` / ``torch``).
"""

import warnings

from typing import NamedTuple

import numpy as np

from .backend import (AxisNameWarning, UnresolvedBranchWarning, _concrete, _device_of,
                      array_namespace, as_array, in_namespace)

# Below this a layer root's real part counts as zero and `mode_q`'s tie-break decides. Only the
# SMM recursion reads those labels, and it does not depend on them; the constant is kept so that
# they stay what they were, to the bit.
_FLAT = 1e-12

_PI = 3.141592653589793
_HALF_PI = 1.5707963267948966


def incident_phase_index(xp, epsilon_mu, theta):
    """Return ``|Re(k)| / k0`` for a real phase angle and real transverse k.

    Write ``epsilon_mu = a + i*b`` and the inward normal wavevector as
    ``q/k0 = N*cos(theta) + i*alpha``. Dispersion gives
    ``N**2 - alpha**2 = a`` and ``2*N*cos(theta)*alpha = b``; hence
    ``N**2 = (a + hypot(a, b/cos(theta))) / 2``.

    Both material components are retained. The rationalized negative-a branch
    avoids cancellation for a weakly absorbing metal. Inputs are backend arrays;
    the caller validates the phase angle and masks unused complex-angle inputs.
    """
    a, b = xp.real(epsilon_mu), xp.imag(epsilon_mu)
    loss = b / xp.cos(theta)
    radius = xp.hypot(a, loss)
    positive = a >= 0
    large_root = xp.sqrt((radius + xp.where(positive, a, -a)) / 2)
    denominator = xp.where(large_root > 0, large_root, xp.ones_like(large_root))
    small_root = xp.abs(loss) / (2 * denominator)
    return xp.where(positive, large_root, small_root)


# How many eps of slack to allow for the round-off accumulated in forming the radicand and
# taking the root. Not a branch threshold — it decides nothing, it only bounds where the
# branch decision stops being answerable. See :func:`_check_resolvable`.
_NOISE_EPS = 8.0

# Anything with a coarser eps than this is single precision. complex64 is 1.2e-7 and
# complex128 is 2.2e-16, so the gap is nine orders wide and the exact value is arbitrary.
_SINGLE_EPS = 1e-10


def _check_resolvable(xp, v, decision, floor):
    """Warn when a root was chosen by the sign of a value the dtype cannot resolve.

    ``decision`` is the real quantity whose sign the caller reads: ``Re(q) - Im(q)`` in
    :func:`etm_mode_q`, ``Re(kz) + Im(kz)`` in :func:`boundary_kz`. ``floor`` is the size below
    which the caller stops reading it and lets a tie-break decide, so values there were never
    read; both callers pass 0, having no tie-break.

    A sign is answerable only while the value stands clear of round-off, which is ~1e-16 of
    ``|v|`` at complex128 and ~1e-7 at complex64. Both decisions are at least ``|v|`` from zero
    for every real ``k_par`` in a passive medium, so this finds nothing unless ``q**2`` or
    ``kz**2`` sits within round-off of its cut -- which complex64 reaches about four hundred
    million times sooner. :func:`mode_q` does not call it: its ``Re(q)`` *is* round-off for an
    exactly propagating mode at single precision, but only SMM reads those labels and SMM does
    not depend on them, so a warning there was a false alarm on every single-precision SMM
    solve. Measured in ``docs/branch-cuts.md``.

    So this reports rather than repairs. It costs nothing measurable: two absolute values on an
    array the solve already built.

    Unlike the guards in ``_core/backend.py`` this is **not** NumPy-only, and deliberately so:
    the backends it would have skipped are the ones that need it most. When it was written,
    under the half-space rule of 2026-08, one complex64 point returned ``|R| = 1.80`` on NumPy,
    with a warning. JAX returned 4.12 and PyTorch 5.68 at the same point, silently. Those were
    the two worst answers, on the paths people reach for when they want gradients or a GPU.

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
    margin, mag = xp.abs(decision), xp.abs(v)
    noise = _NOISE_EPS * eps * xp.maximum(mag, xp.asarray(1e-30, dtype=mag.dtype))
    flagged = xp.any((margin > floor) & (margin < noise))
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
            'a square-root branch was decided from a value smaller than this dtype can '
            'resolve, so the choice came from round-off rather than from the physics. The '
            'result is finite and may be badly wrong. Re-run at double precision '
            '(complex_dtype=0); docs/branch-cuts.md covers the single-precision case.',
            UnresolvedBranchWarning, stacklevel=3)


def mode_q(xp, eigenvalue):
    """Modal decay constant ``q`` of a layer mode, from its eigenvalue ``q**2``.

    A mode varies as ``exp(-k0*q*z)`` with ``z`` measured downward in the kernels' local frame
    (from their entry face into the stack; not, in general, the public height: it coincides
    with it under bottom incidence and is its mirror ``h_total - z`` under top incidence, see
    the module docstring), so the root taken is the one that does not grow: ``Re(q) >= 0``.
    For a lossless propagating mode ``Re(q)`` is zero and the root is imaginary; there the
    downward wave is ``q = -i*kz``, i.e. ``Im(q) <= 0``.

    Both of those are *conditioning* statements, not physics. A layer's ``±q`` pair is used in
    full by the stack algebra, so in exact arithmetic the label is free, and the SMM connecting
    algorithm keeps it free in practice: flipping any subset of a layer's roots moves the solve
    by 1.5e-13 at 40 nm, and the Airy recursion by 2.0e-15 over 4000 random stacks. What
    ``Re(q) >= 0`` buys is ``|exp(-k0*q*d)| <= 1``; the ``Im(q) <= 0`` tie-break fires only
    where that modulus is 1 either way and so buys nothing measurable — inverting it moves the
    answer by exactly 0.

    ETM is the exception, and :func:`etm_mode_q` is its rule. That recursion inverts a
    transmission into the medium below each layer, so it needs a layer's ``+`` modes to be the
    waves that medium receives -- :func:`boundary_kz`'s root. This function's root is that one
    at every real ``k_par`` in a passive layer and is not where ``q**2`` lies above the real
    axis (a complex ``k_par``, or gain): there a propagating mode's decaying root travels up,
    and an ETM solve built on it lost 12 digits. So the two rules disagree at complex ``k_par``
    on purpose, and this is the right one for everything except the ETM recursion -- which
    itself falls back to it wherever the other root would grow. Pinned in
    ``tests/test_branch_selection.py`` and ``tests/test_etm_layer_labels.py``.
    """
    # `xp.sqrt` for the same reason as :func:`boundary_kz` -- `**0.5` leaves a 6.1e-17 real
    # part on a lossless propagating mode, where `Re(q)` is meant to be a hard 0. Nothing
    # downstream reads it at that size (the tie-break's `_FLAT` is 1e-12, and the residue moves
    # `|exp(-k0*q*d)|` by ~1e-17), so this is consistency of spelling rather than a fix; the
    # tie-break itself is untouched and still disagrees with `boundary_kz` on purpose.
    v = xp.sqrt(eigenvalue)
    # No `_check_resolvable` here: at single precision `Re(v)` is round-off for an exactly
    # propagating mode, so the label is noise -- and free, since only SMM reads it.
    flip = xp.where(xp.abs(xp.real(v)) > _FLAT, xp.real(v) < 0, xp.imag(v) > 0)
    return xp.where(flip, -v, v)


def etm_mode_q(xp, eigenvalue, k0_d):
    """A layer mode's ``q`` for the enhanced-transmittance (ETM) recursion: the half-space's root.

    ``k0_d`` is the layer's optical thickness ``k0*d``. The label is the one :func:`boundary_kz`
    would give the same plane wave -- ``q = -i*kz`` with ``Re(kz) + Im(kz) > 0``, which in
    ``q``'s own variables reads ``Re(q) > Im(q)`` -- except where that root grows across the
    layer by more than ``eps**-0.25``; there the decaying root is kept, as :func:`mode_q`
    keeps it everywhere.

    **Why ETM needs a label at all.** Its interface factor ``a = (W^-1 F + V^-1 G)/2`` maps the
    field below an interface onto the layer's ``+`` amplitudes and is then inverted, so it is
    the inverse of a *transmission*: it is well conditioned only if the ``+`` modes are the
    waves the region below receives. At a real in-plane wavevector in a passive layer
    :func:`mode_q`'s decaying root is that wave, bit for bit. A complex ``k_par`` (or a gain
    layer) puts ``q**2`` above the real axis, and there a propagating mode's decaying root
    travels *up*. Where such a mode crosses an interface with nothing to reflect it -- an
    ``eps = 25`` region resting on an ``eps = 25`` substrate -- the smallest singular value of
    ``a`` is that mode's reflection, ~1e-14, and the recursion loses ~12 digits: a 1e-14 nudge
    of a complex ``theta`` moved ``r`` by 2e-4 where SMM moved 2e-15. SMM never sees it,
    because it joins every layer through its gap medium, whose ``kz = 1`` no layer mode
    resembles. So the label is free for SMM and is not free for ETM.

    **Why the cap.** The half-space's root of such a mode grows downward, by
    ``G = |exp(-k0*q*d)|`` across the layer. The recursion carries that round trip as ``G**2``
    in its reflection state, and loses ``eps*G**2`` to it; uncapped, a 10 um layer at
    ``theta = 0.5+0.3j`` broke both algorithms, its near-cutoff modes growing by up to
    ``e**33`` across it. The decaying root costs ``eps/(r*G**2)`` instead, ``r`` being the
    mode's reflection, which is no worse than ``1/G**2`` however small ``r`` is. The two costs cross at ``G = eps**-0.25`` (8192 at double precision, 54 at
    single), so the half-space's root is taken up to there and the decaying one past it.
    Measured on two thick layers, a cap of 8192 matched SMM in both (2e-14 at 50 um, 2e-12 at
    200 um); a cap of 100 or less left 2.6e-6 at 200 um, 1e8 lost a decade and a half there,
    and no cap at all broke both. A thick layer therefore converges to :func:`mode_q`'s
    labels and a thin one to :func:`boundary_kz`'s, and in the thin limit a layer of a
    half-space's own material has exactly that half-space's waves for its ``+`` modes.

    Bit-identical to :func:`mode_q` wherever ``q**2`` lies in the closed lower half plane --
    every real ``k_par`` in a passive layer -- since the two rules then pick the same root and
    the cap cannot fire. ``docs/connecting-algorithms.md`` holds the measurements.
    """
    v = xp.sqrt(eigenvalue)
    # Its decision is the sign of `Re(q) - Im(q)`, at least `|q|` from zero for every passive layer
    # at a real `k_par`: round-off on `Re(q)` alone, which is what a propagating mode's real
    # part is at single precision, cannot flip it.
    _check_resolvable(xp, v, xp.real(v) - xp.imag(v), 0.0)
    q = xp.where(xp.real(v) - xp.imag(v) < 0, -v, v)
    # `log(1/eps)/4` nepers: `|exp(-k0*q*d)| > eps**-0.25` is `Re(q)*k0*d < -nepers`.
    nepers = -0.25 * float(np.log(xp.finfo(v.dtype).eps))
    # Placed the way `modal_gradient.exponential` places its scale, and for its reason: a
    # traced thickness is cast, never rebuilt with a `device=` (which JAX cannot batch).
    real = xp.real(v).dtype
    if in_namespace(xp, k0_d):
        k0_d = xp.astype(k0_d, real)
    else:
        k0_d = xp.asarray(k0_d, dtype=real, device=getattr(v, 'device', None))
    return xp.where(xp.real(q) * k0_d < -nepers, -q, q)


def layer_root(xp, eigenvalue, k0_d=None):
    """A layer mode's ``q``: :func:`etm_mode_q` given the layer's ``k0*d``, :func:`mode_q` without.

    The one place a layer's root is chosen, so the mode builders take the connecting algorithm's
    need as an argument instead of each spelling a rule. ETM passes ``k0_d``; SMM and everything
    else pass nothing and keep the bounded root.
    """
    return mode_q(xp, eigenvalue) if k0_d is None else etm_mode_q(xp, eigenvalue, k0_d)


def boundary_kz(xp, eps, k_par_sq):
    """``kz`` of a homogeneous half-space for the downward-going wave.

    ``exp(+i*k0*kz*z)`` travels down -- along the kernels' local ``+z``, from their entry face
    into the stack, a ``z`` that is not, in general, the public height: it coincides with it
    under bottom incidence and is its mirror ``h_total - z`` under top incidence (see the
    module docstring). At every real ``k_par`` in a passive medium the answer is not in doubt.
    ``Im(eps) >= 0`` puts ``kz**2`` in the closed upper half plane, and its principal root lies in
    the closed first quadrant, so it both carries energy down (``Re(kz) > 0``) and does not grow
    (``Im(kz) >= 0``).

    A complex ``k_par`` (from a complex ``theta``) or a gain medium (``Im(eps) < 0``) can put
    ``kz**2`` below the real axis, and there no root does both. The root taken there is the one
    with ``Re(kz) + Im(kz) > 0``. That is Maystre's cut, which runs along the negative imaginary
    axis of ``kz**2``. Give the half-space enough absorption to lift ``kz**2`` into the upper half
    plane, then take that absorption away at fixed ``k_par``: the root this limiting absorption
    reaches is Maystre's. For a gain medium it is the same root as switching the gain on from
    zero. At a real ``k_par`` that gives the causal answer for ordinary weak gain away from the
    critical angle: a forward-growing wave, and evanescent gain (``|r| > 1``) under total
    internal reflection.

    The rule agrees with the principal root wherever that root lies in the closed first
    quadrant, so it is bit-identical to it at every real ``k_par`` in a passive medium. Its cut
    meets the real ``k_par`` axis only at the Rayleigh point, so a solve is continuous as
    ``Im(k_par) -> 0`` from either side. Until 2026-10-02 the rule was the principal root
    continued from normal incidence. That rule's cut lay along the real evanescent axis, so any
    ``Im(k_par)``, even one the size of round-off, made the evanescent orders on one side grow.
    ``docs/branch-cuts.md`` measures the two rules and the alternatives. On the cut itself
    (``Re(kz**2) = 0``, ``Im(kz**2) < 0``), the root with ``Re(kz) > 0`` is kept.

    RETICOLO takes the same cut (``retsqrt(x, -1)``, its "coupure de Maystre"): J. P. Hugonin and
    P. Lalanne, *RETICOLO software for grating analysis*, Institut d'Optique (2005),
    arXiv:2101.00901. The gain behaviour is that of J. Skaar, Phys. Rev. E 73, 026605 (2006),
    and J. O. Grepstad and J. Skaar, Opt. Express 19, 21404 (2011).
    """
    kz2 = eps - k_par_sq
    # `xp.sqrt`, not `kz2 ** 0.5`. On a negative real `kz2` -- every purely evanescent order --
    # the two disagree exactly where it counts: `csqrt` returns `Re(kz)` as a hard 0, while
    # `**0.5` goes through `exp(0.5*log)` and leaves `cos(pi/2)` behind, 6.1e-17. NumPy and
    # Torch route `**0.5` back to `csqrt` and so hide it; JAX and Python's own `complex` do not.
    # The residue is not decorative: an evanescent order's efficiency carries a factor
    # `Re(kz)`, so the spelling alone decides whether a cut-off channel reports 0.0 or 1e-19.
    v = xp.sqrt(kz2)
    decision = xp.real(v) + xp.imag(v)
    _check_resolvable(xp, v, decision, 0.0)
    return xp.where(decision < 0, -v, v)


def incident_kz(kz_top, index):
    """The incident order's ``kz``, read off the superstrate vector rather than recomputed.

    The boundary match sets a reflected field built from ``kz_top`` (via :func:`boundary_kz`)
    against an incident field. Both sides need the *same* number at the incident order, and
    the reliable way to guarantee that is to take it from the vector the other side already
    uses — ``kz_top[index]`` — instead of writing ``n_top * cos(theta)`` a second time.

    The two spellings agree for a real incidence angle in a lossless medium. In an
    absorbing medium, a real phase angle instead specifies real transverse k, and
    ``n*cos(theta)`` is not its normal wavevector. They can also part company when
    ``theta`` is complex (evanescent illumination), where ``cos(theta)`` is
    negative-imaginary while :func:`boundary_kz` forces ``Im(kz) > 0``. The two sides then
    carry opposite signs at that one order and cancel exactly, and the solve returns
    ``R = -delta`` with ``T = 0`` — a grating that scatters nothing, which plots as a clean
    flat region rather than as the failure it is.
    """
    return kz_top[index]


def incident_sp_amplitudes(xp, C, psi, chi=None, polarization=None):
    """The incident electric coefficients ``(a_s, a_p)``, normalized to unit Euclidean norm.

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

    This coefficient normalization is not a unit-power normalization. The boundary
    power calculation accounts for the medium and each polarization separately.

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
    declared_u = max(abs(centre_u - half_u), abs(centre_u + half_u))
    declared_v = max(abs(centre_v - half_v), abs(centre_v + half_v))
    resolved = []
    for (m, n), amplitude, key in parse_excitation(excitation, declared_u, declared_v):
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


def centre_parity(window_u, window_v):
    """Which retained orders the cell centre negates: ``True`` where ``(-1)**(m + n) == -1``.

    The kernels' Fourier origin is the cell **corner** -- every convolution matrix the kernels read is
    expressed about it -- while every public coordinate, and every per-order phase
    since 2026-09-27, is referenced at the **centre** ``o = (a1 + a2)/2``. Order ``(m, n)``
    sits at ``k_inc + m b1 + n b2`` and ``b_i . a_j = 2 pi delta_ij``, so moving a plane wave's
    phase reference from the corner to the centre multiplies it by
    ``exp(1j * k_mn . o) = exp(1j * k_inc . o) * (-1)**(m + n)`` on **every** cell, a skewed one
    included: ``G_mn . o = pi (m + n)`` exactly, whatever the basis. The common factor cancels
    from every amplitude ratio -- an amplitude, a weight, a scattering-matrix entry -- so what is
    left per order is this sign, and a sign is exact.

    ``(m, n)`` are the order **labels** (``Result.orders_u`` / ``orders_v``), read from the
    retained window, never the array index: the two differ by the window's offset, which is
    odd whenever an ``excitation`` moved the centre by an odd step. A consistent parity by
    index cancels from the amplitudes and the scattering matrix (a sign on both sides) and
    survives only in the field, which is why ``tests/test_centre_reference.py`` lights windows
    with an odd offset.

    ``window_u`` / ``window_v`` are :func:`order_window`'s ``(centre, half_width)`` pairs. A host
    ``bool`` array of shape ``(n_orders_v, n_orders_u)``: the window is static -- it depends on
    ``truncation_order`` and the excitation's keys, never on a traced value -- so nothing here is
    traced and nothing downstream branches on a value.
    """
    (centre_u, half_u), (centre_v, half_v) = window_u, window_v
    m = np.arange(centre_u - half_u, centre_u + half_u + 1)
    n = np.arange(centre_v - half_v, centre_v + half_v + 1)
    return (n[:, None] + m[None, :]) % 2 == 1


def swap_fourier_origin(slots, window_u, window_v):
    """Restate Fourier operators between the public centre and the kernels' corner.

    Each slot is ``(..., N, N)`` on the retained order labels. Translation by
    ``(a1 + a2)/2`` gives ``D A D``, with ``D = diag((-1)**(m+n))``: the common
    incident phase cancels. This applies to inverse-rule, NVF, tensor and magnetic
    operators as well as ordinary Fourier coefficients, on any lattice.

    Negation is an exact sign bit, so this is its own inverse even for signed
    zeros. None slots and named-tuple/list/tuple containers are preserved. The
    mask is static; only the arrays and their gradients cross a trace.
    """
    if slots is None:
        return None
    odd = centre_parity(window_u, window_v).reshape(-1)
    pair = odd[:, None] ^ odd[None, :]
    converted = []
    for a in slots:
        if a is None:
            converted.append(None)
            continue
        if a.shape[-2:] != pair.shape:
            raise ValueError(f'convolution slots need ({odd.size}, {odd.size}) retained '
                             f'orders in their last two axes, got {a.shape}')
        xp = array_namespace(a)
        mask = as_array(xp, pair, device=_device_of(a))
        converted.append(xp.where(mask, -a, a))
    if isinstance(slots, tuple) and hasattr(slots, '_fields'):
        return type(slots)(*converted)
    return converted if isinstance(slots, list) else tuple(converted)


def corner_column(lit, window_u, window_v):
    """:func:`lit_orders`' public weights restated for the kernels' corner Fourier origin.

    A public weight ``w_mn`` is the amplitude of ``exp(1j * k_mn . x)`` with ``x`` measured from
    the cell centre, unit at ``x = y = 0`` -- the same thing an ``OrderExcitation`` weight means.
    About the corner that wave is ``w_mn * exp(-1j * k_inc . o) * (-1)**(m + n)``; the scalar is
    a normalisation of the whole field, which cancels from every ratio and which a field door
    puts back, so the kernels are handed ``w`` with each weight on an odd order
    negated (:func:`centre_parity`). Negation is a sign bit, traced or not, so the column the
    kernels build is exact.

    ``None`` stays ``None``: an undeclared solve keeps :func:`incident_column`'s unit vector to
    the bit -- the specular order is ``(0, 0)``, which is even.
    """
    if lit is None:
        return None
    odd = centre_parity(window_u, window_v).reshape(-1)
    return tuple(one._replace(amplitude=-one.amplitude) if odd[one.index] else one
                 for one in lit)


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

    * **it is discontinuous.** Sweep ``theta`` so one order's own ``kx`` crosses zero at
      ``ky != 0`` and nothing physical happens — the order stays propagating, ``R`` and
      ``|r_s|`` sail through, and ``k_par`` never vanishes, so ``z_hat x k_hat_par`` turns
      smoothly — but ``r_s`` changes sign. No convention can do that *there*. (On the
      ``ky == 0`` row ``k_par`` passes through zero, the frame has no limit, and a sign step at
      an order's own ``kx = 0`` is the frame's singular point rather than a defect; see
      :func:`order_frame_sign`.)
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


def order_frame_sign(xp, kx):
    """``s_hat . y_hat`` of every order on the ``ky == 0`` row: ``-1`` where ``Re(kx) < 0``, else ``+1``.

    The pure 1D path's whole s/p frame. With ``ky == 0`` and the origin azimuth ``0`` -- the only
    azimuth ``tetm`` admits (``CoreSolveMixin._refuse_illegal_tetm``) -- ``s_hat = z_hat x
    k_hat_par`` is ``+-y_hat``, and ``p_hat`` lies in the xz plane and turns over with it. This
    is exactly ``cos(order_azimuth(xp, 0*kx, kx, 0))``, the rule the conical and 2D paths use,
    spelled without the angle so that ``sin(pi)`` (1.2e-16) never reaches a component that is
    identically zero; ``tests/test_order_azimuth.py`` pins the two against each other, since a
    rule that cannot be *called* has to be *compared*.

    * ``kx == 0`` (``-0.0`` included) is ``+1``: the origin rule at ``phi = 0``, which is what
      ``order_azimuth`` answers there.
    * A purely imaginary ``kx`` (a complex ``theta``) is ``+1`` too, as there: the half-turn is
      keyed on ``Re(kx)``, never on ``sign(kx)``, which is complex-valued on a complex array.

    Returned **real**, in the dtype of ``Re(kx)``: exactly ``+-1`` on every backend and
    precision, so multiplying a complex amplitude by it is an exact sign flip and never widens
    a ``complex64`` solve. ``where`` on a comparison rather than a Python branch: nothing is
    traced through a condition, and the derivative is zero rather than a NaN.
    """
    one = xp.ones_like(xp.real(kx))
    return xp.where(xp.real(kx) < 0, -one, one)


def _order_count(value, name, which):
    """One order count: a whole number, not negative.

    Neither of the two doors into this repository got both halves. The solver's side took
    ``int(value)`` and so read ``2.7`` as ``2`` -- a silent truncation of the one number that
    decides how much of the spectrum is kept -- and took ``-3`` as ``-3``, which reaches
    ``solve`` and dies there with ``IndexError: arrays used as indices must be of integer
    type``, naming neither the argument nor the caller. The JSON side (``Numerics.from_dict``)
    caught both, and refused five spellings the solver accepts on top: ``2.0``, ``[3]``,
    ``dict(u=..., v=...)``, ``dict(x=..., y=...)`` and a NumPy integer.

    So this is the missing half of the accepting side rather than a copy of the strict one.
    ``2.0`` is a whole number written as a float and is accepted; ``2.7`` is not a count.
    """
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name}: expected two integers, got {value!r}') from None
    if count != value:
        raise ValueError(
            f'{name}: {which} is {value!r}, which is not a whole number. The count is how '
            f'many diffraction orders are retained on each side of zero.')
    if count < 0:
        raise ValueError(
            f'{name}: {which} is {value!r}. The count is how many orders are retained on '
            f'each side of zero, so it cannot be negative -- 0 keeps the specular order '
            f'alone, which is the smallest a solve can be.')
    return count


def parse_order_pair(value, name='truncation_order'):
    """``([u, v], spelling)`` from any accepted spelling of an order-count pair.

    ``spelling`` is ``'uv'``, ``'xy'`` or ``None`` (a bare sequence, which has no names to
    check). It travels because **only one of the two is always true**: the order axes run along
    the cell's reciprocal vectors, so ``u``/``v`` names them on any cell while ``x``/``y`` does
    so only on an axis-aligned one -- measured, a 60-degree hex sends ``m += 1`` to
    ``(dkx, dky) = (+0.600, -0.346)``, neither axis. The caller refuses ``'xy'`` on a skewed
    cell rather than reinterpreting it; this only reports what was written.

    A mapping is the spelling with no order to remember, which is the point of accepting one.
    Every other pair in this API is positional and every one of them is x-first, while every
    array axis is v-first -- two rules that are each consistent and meet here. A caller who
    writes ``dict(u=5, v=2)`` never has to know which.
    """
    if isinstance(value, dict):
        keys = set(value)
        if keys == {'u', 'v'}:
            pair, spelling = (value['u'], value['v']), 'uv'
        elif keys == {'x', 'y'}:
            pair, spelling = (value['x'], value['y']), 'xy'
        else:
            raise ValueError(
                f"{name}: a mapping needs exactly {{'u', 'v'}} or {{'x', 'y'}}; got {sorted(keys)}. "
                f"'u' and 'v' are the cell's two directions and are always right; 'x' and 'y' "
                f"name them only on an axis-aligned cell.")
        return [_order_count(pair[0], name, "the first entry"),
                _order_count(pair[1], name, "the second entry")], spelling

    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [_order_count(value, name, 'the count'), 0], None
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return [_order_count(value[0], name, 'the count'), 0], None
        if len(value) == 2:
            return [_order_count(value[0], name, 'the first entry'),
                    _order_count(value[1], name, 'the second entry')], None
        raise ValueError(f'{name}: expected 1 or 2 values, got {len(value)}')
    raise ValueError(
        f"{name}: expected an int, a pair, or a mapping like dict(u=5, v=2); got {value!r}")


def warn_if_xy_on_a_skewed_cell(spelling, axis_aligned, name='truncation_order'):
    """Note that an ``x``/``y`` pair was read as ``u``/``v``, on a cell where they differ.

    Deferred rather than checked where the pair is set, and holding no memory of having warned:
    ``rcwa(...)`` assigns its keywords in whatever order they arrive, a cell that was rectangular
    when the pair was written can be made skewed afterwards, and one that was skewed can be made
    rectangular again -- at which point ``x``/``y`` is true and there is nothing to say. Reading
    both facts at the solve is what makes all of those come out right.

    A warning and not a refusal; :class:`~wavebit._core.backend.AxisNameWarning` carries why.
    """
    if spelling == 'xy' and not axis_aligned:
        warnings.warn(
            f"{name} was given as x/y on a cell that is not axis-aligned, and has been read as "
            f"u/v -- u along a1, v along a2, which is what x and y meant on a rectangular cell. "
            f"The solve is the one you asked for; only the names were off. Write "
            f"{name}=dict(u=..., v=...) to say it exactly.",
            AxisNameWarning, stacklevel=3)


# --- the incidence spellings, and what makes them exclusive -----------------------------
_DIRECTION_CLASH = (
    'kx/ky and theta/phi are alternative incident directions. Pass either real '
    'Cartesian kx/ky in inverse length units, or angles in radians, not both. '
    'Clear the other direction with None before switching a solver between them.')


#: Appended when a direct component is finite but complex: the refusal is deliberate, and the
#: caller who wanted a complex transverse wavevector has a door -- the angles -- to be shown.
_COMPLEX_TRANSVERSE_K = (
    '; it was complex. A direct kx/ky states illumination of uniform intensity along the '
    'surface, which a real transverse wavevector gives and a complex one cannot -- in an '
    'absorbing medium too, where the loss belongs in kz. For absorbing incidence give a real '
    'theta/phi instead: the phase angle keeps kx and ky real. A complex transverse wavevector '
    'is reachable only through a complex theta/phi, which keep the formal '
    'kx/k0 = n*sin(theta)*cos(phi), ky/k0 = n*sin(theta)*sin(phi) with n the incident '
    'half-space index (for ky = 0: theta = arcsin(kx/(k0*n)), phi = 0); R and T are then '
    'formal ratios, power_entering and absorption are NaN, and a ComplexIncidenceWarning says '
    'so. See docs/conventions.md#phase-incidence.')


def transverse_component_problem(value, name):
    """Validate one dimensional, real scalar wavevector component without detaching AD."""
    if value is None:
        return None
    message = f'{name} must be a finite real scalar in inverse length units'
    if isinstance(value, (list, tuple, dict, str)) or getattr(value, 'shape', ()) != ():
        return message
    try:
        if hasattr(value, 'dtype'):
            xp = array_namespace(value)
            finite = xp.isfinite(value)
            if _concrete(finite & (xp.imag(value + 0j) != 0)) is True:
                return message + _COMPLEX_TRANSVERSE_K
            valid = finite & (xp.imag(value + 0j) == 0)
            return message if _concrete(valid) is False else None
        import cmath
        number = complex(value)
        if cmath.isfinite(number) and number.imag != 0:
            return message + _COMPLEX_TRANSVERSE_K
        return None if cmath.isfinite(number) else message
    except (TypeError, ValueError, OverflowError):
        return message

#
# `psi`, `chi`, `polarization` and `tetm` are four names for one input, and the rules about
# which may sit beside which live here rather than in the setters that used to hold them.
# The reason is the reason this module exists at all: the solver's setters are one door and
# `Stack` / `Incidence` are another, and a rule written twice is a rule that will disagree
# with itself -- as `truncation_order`'s two copies did, for years, in `0dde6c5`.


#: One message for both directions of the same mistake — the ``psi`` setters raise it when a
#: pair is already set, the ``polarization`` setter when ``psi`` was given explicitly.
_PSI_POLARIZATION_CLASH = (
    'psi and polarization both given: they specify the same thing (the incident '
    'polarization), so one must be dropped rather than silently overridden. '
    'psi = v is exactly polarization = (sin(v), cos(v)).')

#: ``tetm`` declares the polarization away; naming one beside it is the same clash as
#: ``psi`` with ``polarization``, one level up.
_TETM_INCIDENCE_CLASH = (
    'tetm and an incidence polarization (psi or polarization) both given: tetm declares '
    'that the polarization is not a degree of freedom of this problem, so naming one is a '
    'contradiction rather than a refinement. Drop tetm to keep the polarization free (the '
    'conical path, 2-4x slower), or drop psi/polarization to keep the declaration.')

#: ``chi`` and the pair are two charts on one slot -- the same clash as ``psi`` with the pair.
_CHI_POLARIZATION_CLASH = (
    'chi and polarization both given: they are two ways of writing the same incident state, '
    'not two states. (psi, chi) are its canonical coordinates -- two real angles, no '
    'normalization, nothing redundant -- while polarization=(a_s, a_p) is the raw pair, which '
    'additionally carries an overall phase the chart does not. Drop whichever you did not '
    'mean; psi = v with chi = 0 is exactly polarization = (sin v, cos v).')

#: ``chi`` is an angle too, and the same argument applies one coordinate over.
_COMPLEX_CHI = (
    'chi must be a real angle: it is the ellipticity angle, and tan(chi) is the '
    "ellipse's minor/major axis ratio. Every state already has real coordinates -- "
    'that is what makes (psi, chi) a chart -- so an imaginary part is not a wider '
    'spelling but an unnormalized pair in disguise. Use polarization=(a_s, a_p) to '
    'write amplitudes directly.')

#: Raised by :meth:`CoreSolveMixin._reject_complex_angle`, which says why at length.
_COMPLEX_PSI = (
    'psi must be a real angle; a complex psi is not the elliptical spelling it looks '
    'like. (sin(psi), cos(psi)) stops being a unit pair once psi has an imaginary part, '
    'so the incident flux silently becomes cosh(2*Im(psi)) and every efficiency is off by '
    'that factor. It cannot reach the circular states either: they are the two values '
    'tan(psi) never takes. Use polarization=(a_s, a_p) — (1, -1j) and (1, +1j) are the '
    'circular states, and its magnitude is normalized for you.')


def reject_complex_angle(psi, message=None):
    """``psi`` is an angle, and an angle is real. Called by the backend ``psi`` setters.

    A complex ``psi`` is not a wider spelling of the same input — it is a silently wrong
    one, and it was wrong in a way nothing downstream could catch. ``(sin(psi), cos(psi))``
    is a unit pair only for real ``psi``; for ``Im(psi) = b`` the incident flux comes out
    ``cosh(2b)``, so every efficiency is scaled by a factor the solver never applied.
    On a lossless stack ``R + T > 1`` raises ``UnphysicalResultWarning``, but an absorbing
    one absorbs the discrepancy: a measured case returned ``R + T = 0.998`` where the
    right answer was ``0.647``, with no warning anywhere.

    Nor does it buy the states it looks like it should. ``a_s/a_p = tan(psi)``, and
    ``tan(z) = ±i`` reduces to ``exp(±iz) = 0``, which has no solution — the circular
    states are the two values ``tan`` omits. A complex ``psi`` approaches them only as
    ``Im(psi) -> ±inf``, and the incident amplitude diverges at exactly the rate the
    approach converges. ``polarization=(a_s, a_p)`` is where those states live.

    Anything carrying a ``dtype`` — array, tensor, tracer — is judged on that and its
    value is never read: a trace has no value to read, and reading a torch tensor that
    requires grad warns. A plain Python scalar is judged on its imaginary part instead,
    so ``0.5+0j`` is accepted as a real angle written oddly rather than refused.
    """
    message = _COMPLEX_PSI if message is None else message
    dtype = getattr(psi, 'dtype', None)
    if dtype is not None:
        if getattr(dtype, 'kind', '') == 'c' or 'complex' in str(dtype):
            raise ValueError(message)
        return
    try:
        value = complex(psi)
    except (TypeError, ValueError):
        return
    if value.imag != 0:
        raise ValueError(message)


def incidence_clashes(tetm=None, psi=None, chi=None, polarization=None):
    """Every conflict among the four incidence spellings, as messages, or an empty tuple.

    Four names for one input, and the rules are pairwise: ``tetm`` refuses to sit beside any
    of the other three, and ``psi`` (with ``chi`` beside it) and ``polarization`` refuse each
    other. Whole-state rather than incremental -- it is handed the four values and reports
    every conflict among them at once.

    The setters enforce the same rules one assignment at a time, because that is the shape
    they are in: they see one new value against a solver that already holds the others, and
    they distinguish a ``psi`` the caller *gave* from the 45-degree default using
    ``_psi_given``. A carrier has no such history; ``psi is not None`` is the whole of it.
    So this is the same rule stated for the caller that has all four in hand, not a second
    rule -- and stating it once, here, is why the messages are the constants above rather
    than sentences written twice.

    Returns messages instead of raising because a caller checking a whole scene wants every
    problem, not the first. A setter wants the opposite and keeps raising.
    """
    problems = []
    if tetm is not None:
        for name, value in (('psi', psi), ('chi', chi), ('polarization', polarization)):
            if value is not None:
                problems.append(f'{name} beside tetm: {_TETM_INCIDENCE_CLASH}')
    if polarization is not None:
        if psi is not None:
            problems.append(_PSI_POLARIZATION_CLASH)
        if chi is not None:
            problems.append(_CHI_POLARIZATION_CLASH)
    return tuple(problems)


def wavelength_problem(wavelength, name='wavelength'):
    """Why ``wavelength`` cannot be one, or ``None``.

    The one slot in the whole incidence with no rule anywhere: there is no ``wavelength``
    property on any solver -- no getter, no setter, just ``self.wavelength = wavelength`` in
    three constructors -- so ``'abc'``, ``None``, ``0.0`` and ``-620.0`` were all accepted,
    and the negative one solved and returned energy-conserving numbers.

    Written here rather than as a setter because it is the first rule for this slot and both
    doors should get it at once. A free-space wavelength is a positive length; zero has no
    frequency and a negative one is the same wave with the sign moved into the phase, which
    every downstream expression then disagrees about.

    Anything carrying a ``dtype`` is left alone, the same way :func:`reject_complex_angle`
    leaves it alone: a traced or arrayed wavelength is a design variable being swept, and its
    value is not available to look at.
    """
    if hasattr(wavelength, 'dtype') or hasattr(wavelength, 'shape'):
        return None
    try:
        value = complex(wavelength)
    except (TypeError, ValueError):
        return (f'{name}: expected a positive length, got {wavelength!r}. It is the free-space '
                f'wavelength, in whatever unit the rest of the scene is written in.')
    if value.imag != 0:
        return (f'{name}: {wavelength!r} is complex. Loss belongs in a material '
                f'(Im(eps) > 0), not in the wavelength.')
    if value.real <= 0:
        return (f'{name}: {wavelength!r} is not a length. Zero has no frequency, and a '
                f'negative wavelength is the same wave with the sign moved into the phase, '
                f'which the solve and the order grid then disagree about.')
    return None
