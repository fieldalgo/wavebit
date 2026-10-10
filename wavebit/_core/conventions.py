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
    """Return ``|Re(k)| / k0`` for a real incidence angle and real transverse k.

    Write ``epsilon_mu = a + i*b`` and the inward normal wavevector as
    ``q/k0 = N*cos(theta) + i*alpha``. Dispersion gives
    ``N**2 - alpha**2 = a`` and ``2*N*cos(theta)*alpha = b``; hence
    ``N**2 = (a + hypot(a, b/cos(theta))) / 2``. ``b`` enters squared, so the same ``N``
    serves a negative-phase-velocity medium, whose ``Re(k)`` points the other way
    (``q/k0 = -N*cos(theta) + i*alpha``); the caller applies that sign, ``s``, to ``kx`` and
    ``ky`` (``engine.CoreSolveMixin._incident_kxy``). ``N >= 0`` always.

    Both material components are retained. The rationalized negative-a branch
    avoids cancellation for a weakly absorbing metal. Inputs are backend arrays;
    the caller validates the angle and masks unused complex-angle inputs.
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


def mode_q(xp, eigenvalue, mirrored=None):
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

    SMM's field state reads no labels at all: it reads each layer's amplitudes off the gap
    amplitude between layers rather than walking a layer's ``+`` amplitudes
    (:func:`~wavebit._core.scattering._field_state`). From 2026-10-09 to 10-10 it walked the
    ETM's state and was rebuilt on the ETM's labels, which cost a second ``eig``, still inverted
    the map into each layer, and made some real-angle fields up to four orders worse
    (``docs/connecting-algorithms.md`` §10).

    **A uniform double-negative layer mirrors the tie** (``mirrored``, from
    :func:`layer_label_rule`; ``None`` everywhere else, which is the arithmetic above to
    the bit). A lossless propagating mode of such a layer carries its power against its phase
    (V. G. Veselago, Sov. Phys. Usp. 10, 509 (1968)), so its downward wave is ``Im(q) >= 0``,
    the root :func:`boundary_kz` gives a half-space of the same medium; the rule is then
    ``conj(mode_q(conj(q**2)))``. The label stops being free there: a mode labelled against its
    power whose admittance is the negative of the gap's makes SMM's gap-to-layer factor exactly
    singular. The gap's waves have ``kz/mu = kz/eps = sqrt(1 + |k_par|**2)``
    (``scattering._gap_waves``), so ``(eps, mu) = (-3, -1)`` at ``kx = 1`` is its exact TE
    mirror, ``(-1, -3)`` its TM one, and each raises ``LinAlgError`` under the old tie (under the
    gap before 2026-10-09, ``(-2, -1)`` did). A patterned layer that holds a double negative breaks
    the tie by each mode's power flux instead (:func:`power_down_labels`). Measured in
    ``docs/connecting-algorithms.md`` section 11.
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
    if mirrored is not None:
        # A selection, not new arithmetic: a layer that is not a uniform double negative is
        # handed back the comparison above, so its root keeps its bits.
        mirror = xp.where(xp.abs(xp.real(v)) > _FLAT, xp.real(v) < 0, xp.imag(v) < 0)
        flip = mirror if mirrored is True else xp.where(mirrored, mirror, flip)
    return xp.where(flip, -v, v)


def etm_mode_q(xp, eigenvalue, k0_d, mirrored=None):
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
    because it joins every layer through its gap, whose waves no layer mode reproduces. So the
    label is free for SMM -- its cascade, and the field state it walks in the gap basis
    (:func:`~wavebit._core.scattering._field_state`) -- and is not free for ETM.

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

    **The thickness is the run's.** The kernels pass ``k0`` times the thickness of the layer's
    run of identical layers (:func:`run_k0_d`), not of the slice: a layer cut in two is one
    layer, and two slices deciding apart left a pivot on an exact zero at the plane between
    them. What the cap still cannot see is the layer below: it presumes the mode's reflection
    there is not at rounding level, and where a layer's material recurs next to it that fails
    -- ``docs/connecting-algorithms.md`` section 10, "What is left".

    Bit-identical to :func:`mode_q` wherever ``q**2`` lies in the closed lower half plane --
    every real ``k_par`` in a passive layer, but for the spurious eigenvalues a truncated metal
    operator can put above the real axis -- since the two rules then pick the same root and the
    cap cannot fire. ``docs/connecting-algorithms.md`` holds the measurements.

    **A uniform double-negative layer takes the mirror** (``mirrored``, from
    :func:`layer_label_rule`; ``None`` everywhere else, which is the arithmetic above to
    the bit). The half-space of the same medium takes the mirror of Maystre's cut
    (:func:`boundary_kz`), so this does too: ``Re(q) + Im(q) > 0``, which is
    ``conj(etm_mode_q(conj(q**2)))`` before the cap. Its passive ``q**2`` lies in the closed
    *upper* half plane at every real ``k_par``, where Maystre's root is the one that grows and
    advances its phase downward -- against the power -- so a layer of a double negative resting
    on the same medium had a singular ``a``: ``LinAlgError`` on 171 of 720 such scenes, every
    path but the first-order one. With the mirror ``a`` is the identity there, as for an
    ordinary layer on its own substrate. The cap is the same cap, since it is about growth and not
    about the cut. A *patterned* layer holding a double negative has no one medium to mirror and
    takes :func:`power_down_labels` per mode. ``docs/connecting-algorithms.md`` section 11.
    """
    v = xp.sqrt(eigenvalue)
    # Its decision is the sign of `Re(q) - Im(q)`, at least `|q|` from zero for every passive layer
    # at a real `k_par`: round-off on `Re(q)` alone, which is what a propagating mode's real
    # part is at single precision, cannot flip it. The mirror's, `Re(q) + Im(q)`, is the same
    # distance from zero for a passive double negative.
    if mirrored is True:
        decision = xp.real(v) + xp.imag(v)
    elif mirrored is None:
        decision = xp.real(v) - xp.imag(v)
    else:                                       # traced: select, so either side keeps its bits
        decision = xp.where(mirrored, xp.real(v) + xp.imag(v), xp.real(v) - xp.imag(v))
    _check_resolvable(xp, v, decision, 0.0)
    q = xp.where(decision < 0, -v, v)
    return _capped(xp, q, k0_d)


def _capped(xp, q, k0_d):
    """``q``, with every root that grows by more than ``eps**-0.25`` across ``k0_d`` flipped.

    The cap of :func:`etm_mode_q`, which says why; shared with :func:`power_down_labels`.
    """
    return xp.where(_grows(xp, q, k0_d), -q, q)


def _grows(xp, q, k0_d):
    """Where ``|exp(-k0*q*d)|`` exceeds ``eps**-0.25``: the roots :func:`_capped` turns."""
    # `log(1/eps)/4` nepers: `|exp(-k0*q*d)| > eps**-0.25` is `Re(q)*k0*d < -nepers`.
    nepers = -0.25 * float(np.log(xp.finfo(q.dtype).eps))
    # Placed the way `modal_gradient.exponential` places its scale, and for its reason: a
    # traced thickness is cast, never rebuilt with a `device=` (which JAX cannot batch).
    real = xp.real(q).dtype
    if in_namespace(xp, k0_d):
        k0_d = xp.astype(k0_d, real)
    else:
        k0_d = xp.asarray(k0_d, dtype=real, device=getattr(q, 'device', None))
    return xp.real(q) * k0_d < -nepers


def layer_root(xp, eigenvalue, k0_d=None, mirrored=None):
    """A layer mode's ``q``: :func:`etm_mode_q` given the layer's ``k0*d``, :func:`mode_q` without.

    The one place a layer's root is chosen, so the mode builders take the connecting algorithm's
    need as an argument instead of each spelling a rule. ETM passes ``k0_d``; SMM passes nothing
    and keeps the bounded root. ``mirrored`` is :func:`layer_label_rule`'s verdict on the layer,
    passed to either rule -- and only when there is one, so every other layer makes the call it
    always made.
    """
    if mirrored is None:
        return mode_q(xp, eigenvalue) if k0_d is None else etm_mode_q(xp, eigenvalue, k0_d)
    return (mode_q(xp, eigenvalue, mirrored) if k0_d is None
            else etm_mode_q(xp, eigenvalue, k0_d, mirrored))


def layer_runs(xp, slots, rules=None):
    """Which adjacent layers are one layer cut in two: one verdict per boundary.

    ``slots[i]`` is every array layer ``i``'s modes are built from (its convolution slots,
    ``None`` where one is absent), in the same order for every layer, in the kernel's list
    order; ``rules[i]`` -- when given -- the engine's :class:`DoubleNegativeLayer` for it.
    Verdict ``i`` joins layers ``i`` and ``i + 1``: ``True``, ``False`` or a traced boolean.
    Joined layers form a *run*, which decides its labels once (:func:`run_k0_d`) and is
    stepped through as the one layer it is (the kernels' ETM recursions).

    Two layers join when every slot agrees to ``sqrt(eps)`` of the largest entry over all their
    slots, and the rule agrees. Bit equality alone missed the same material written twice -- a
    tensor built as ``R D R^T`` beside ``D`` rotated by ``R``, 4.4e-16 apart; a rectangle of a
    medium drawn on a background of that medium, 5.6e-17 -- and two such slices deciding apart
    (a tilted uniaxial layer as 2.5 um + 1.5 um under a grating, ``theta = 0.2+0.6j``) put
    first-order SMM 1.6 off where it had been at 1.2e-11. Why ``sqrt(eps)``: a label two neighbours
    take differently costs ``eps/delta`` at the plane between them, ``delta`` being how far their
    slots differ, so a pair closer than ``sqrt(eps)`` must decide together. A pair farther apart
    is not bounded by that: slices that each keep a growing label under the cap compound it
    across the plane, and a metal grating at a real angle cut into slices whose backgrounds
    differ by 1e-9 (3.6e-8 apart in Li's slots) leaves SMM's field 2.4e-3 off
    (``docs/connecting-algorithms.md`` section 10). Joining never moves an answer -- labels are a
    basis, and the run's step is the same algebra -- so the threshold only decides where the
    guarantee ends. Comparisons only; no gradient passes through them.
    """
    links = []
    for i in range(len(slots) - 1):
        rule = True if rules is None else _same_rule(xp, rules[i], rules[i + 1])
        links.append(False if rule is False else _all_of(rule, _same_layer(xp, slots[i],
                                                                           slots[i + 1])))
    return links


def run_k0_d(xp, k0, thickness, links):
    """Per layer, the optical thickness a label cap decides with: ``k0`` times its run's.

    ``links`` is :func:`layer_runs`'s verdicts. **Why the run and not the layer.** Joined
    layers are one layer cut in two: the plane between them reflects nothing, and their modes
    are the same modes. A cap (:func:`etm_mode_q`'s, and the first-order split's
    :func:`~wavebit._core.solve_2d_first_order._cap_round_trips`) weighs a growing label's round
    trip against a decaying one's, and the round trip is across the whole run. A slice that
    decided on its own thickness kept the half-space's root of a mode its thicker neighbour
    had turned; the plane between them then had to turn one root into the other with a
    reflection of exactly zero, which is a pivot on that zero. Measured: a uniform
    ``eps = 12.75`` layer cut 1 um + 8 um under a grating of that material put ETM's
    amplitudes 3.0e-1 off the uncut layer at ``theta = 0.66+0.47j`` (largest over all orders,
    against the largest amplitude: a dark order came out at 0.50), and an ``eps = 9`` TM
    stack (1 um + 9 um, ``theta = 0.55+0.4j``) 7.1e+2 off, where SMM's cascade (gap medium, no
    cap) stayed at 1.4e-15 and 5.9e-16; on the first-order path, where SMM has no gap medium, a
    gyrotropic layer cut 20 um + 1.5 um put SMM 0.75 off. On the run's thickness all three are
    at the uncut layer's level.

    A layer in no run reads ``k0 * thickness[i]``, the expression the kernels passed before, so
    a stack without runs is untouched; a run's total is summed once, in list order, so every
    slice reads the same bits. At a real ``k_par`` in a passive layer no cap fires, except on
    the spurious root a truncated metal operator can hold above the real axis; a metal grating
    cut in two then had its two slices keep that root growing and its fields 1.9e-3 apart
    between ETM and SMM, and now takes the uncut layer's label. Sums of thicknesses only: where
    a verdict is traced (``jax.jit`` over a traced raster) the runs are found as segments of a
    cumulative sum and their totals as one matrix product, so the graph does not grow with the
    square of the stack.
    """
    n = len(thickness)
    if any(not isinstance(v, bool) for v in links):
        cut = xp.stack([xp.asarray(not v) if isinstance(v, bool) else ~v for v in links])
        segment = xp.concat([xp.zeros((1,), dtype=xp.int64),
                             xp.cumulative_sum(xp.astype(cut, xp.int64))])
        member = xp.astype(segment[:, None] == segment[None, :], xp.float64)
        widths = xp.stack([xp.astype(xp.asarray(t), xp.float64) for t in thickness])
        totals = member @ widths
        return [k0 * totals[i] for i in range(n)]
    if not any(links):
        return [k0 * thickness[i] for i in range(n)]
    out, start = [None] * n, 0
    for i in range(n):
        if i == n - 1 or not links[i]:
            if start == i:
                out[i] = k0 * thickness[i]
            else:
                total = thickness[start]
                for j in range(start + 1, i + 1):
                    total = total + thickness[j]
                for j in range(start, i + 1):
                    out[j] = k0 * total
            start = i + 1
    return out


def _all_of(a, b):
    """``a and b`` over ``True`` / ``False`` / a traced boolean."""
    if a is False or b is False:
        return False
    if a is True:
        return b
    if b is True:
        return a
    return a & b


def _same_layer(xp, a, b):
    """Do two layers hold the same values in every slot, to ``sqrt(eps)`` of the largest entry
    over all their slots? ``True`` / ``False`` / traced (:func:`layer_runs`).

    One scale for the pair, not one per slot: a slot that is exactly zero in one layer and
    rounding noise in the other -- a tensor rotated by ``2*pi``, or twisted by 1e-12 rad --
    differs by all of its own size, 6e-17 against a layer whose largest entry is 2.66. Scaled
    per slot that kept one material apart from itself, and first-order SMM went 22 off (80 um +
    1.5 um at ``theta = 0.5+0.05j``).
    """
    pairs = []
    for x, y in zip(a, b):
        if x is y:
            continue
        if x is None or y is None or tuple(x.shape) != tuple(y.shape):
            return False
        pairs.append((x, y))
    if not pairs:
        return True
    scale = None
    for x, y in pairs:
        size = xp.maximum(xp.max(xp.abs(x)), xp.max(xp.abs(y)))
        scale = size if scale is None else xp.maximum(scale, size)
    tol = float(np.sqrt(xp.finfo(xp.real(pairs[0][0]).dtype).eps))
    verdict = True
    for x, y in pairs:
        close = xp.max(xp.abs(x - y)) <= tol * scale
        read = _concrete(close)
        if read is False:
            return False
        verdict = _all_of(verdict, close if read is None else True)
    return verdict


def _same_rule(xp, a, b):
    """Do two layers take the same label rule? ``True`` / ``False`` / traced: each field of
    the two :class:`DoubleNegativeLayer` verdicts compared by value, a traced one selected on."""
    if a is b:
        return True
    if a is None or b is None:
        return False
    verdict = True
    for x, y in zip(a, b):
        if x is y:
            continue
        if x is None or y is None:
            return False
        if isinstance(x, bool) and isinstance(y, bool):
            return False                        # distinct Python bools differ
        verdict = _all_of(verdict, xp.asarray(x) == xp.asarray(y))
    return verdict


class DoubleNegativeLayer(NamedTuple):
    """The engine's word on one layer, which the mode builders read as ``dng``.

    Each field is ``True``, ``False`` or a traced boolean -- a raster or a material that is a
    tracer under ``jax.jit`` cannot be read, and is then selected on rather than branched on.
    """
    #: The layer holds a region where ``Re(epsilon) < 0`` and ``Re(mu) < 0``.
    holds: object
    #: The layer is one uniform medium; ``None`` when not known, and then
    #: :func:`uniform_double_negative` reads it off the layer's own matrices.
    uniform: object = None


def uniform_double_negative(xp, eps_slots, mu_slots, zero_slots=()):
    """Is this layer one uniform double-negative medium? Read off its own convolution matrices.

    ``eps_slots`` / ``mu_slots`` are the layer's diagonal slots (``epx``, ``epy``, ``epz_i`` and
    their permeability twins; ``None`` entries are skipped) and ``zero_slots`` its off-diagonal
    ones. True where every diagonal slot is ``c*I`` with ``Re(c) < 0`` -- ``epz_i`` holds
    ``1/eps_zz``, whose real part has the sign of ``Re(eps_zz)`` -- and every off-diagonal slot
    is zero: a uniform medium with every principal component of ``epsilon`` and of ``mu``
    negative, which is what :func:`double_negative` asks of a half-space. Every builder returns
    a uniform layer as exactly ``c*I``, so this classifies ``solve_conv``'s hand-built slots the
    way the engine's materials classify ``solve()``'s (:class:`DoubleNegativeLayer`); it is what
    the builders fall back on when the engine said nothing.

    ``False`` for a readable no, ``True`` for a readable yes, and the traced boolean itself
    under ``jax.jit``. Comparisons only: no gradient passes through it.
    """
    verdict = False
    for slot in (*eps_slots, *mu_slots):
        if slot is None:
            continue
        c = slot[0, 0]
        eye = xp.eye(slot.shape[0], dtype=slot.dtype, device=getattr(slot, 'device', None))
        one = xp.all(slot == c * eye) & (xp.real(c) < 0)
        verdict = one if verdict is False else verdict & one
    if verdict is False:
        return False
    for slot in zero_slots:
        if slot is not None:
            verdict = verdict & xp.all(slot == 0)
    readable = _concrete(verdict)
    return verdict if readable is None else readable


def _both(a, b):
    """``a and b`` over ``True`` / ``False`` / traced, with ``None`` for a readable no."""
    if a is False or b is False:
        return None
    if a is True:
        return b
    if b is True:
        return a
    return a & b


def layer_label_rule(xp, dng, eps_slots, mu_slots, zero_slots=()):
    """``(mirrored, patterned)``: which label rule a layer's modes take, beyond the plain one.

    ``mirrored`` -- a uniform double negative -- goes to :func:`layer_root`, which then takes the
    mirror of the plain rule, as :func:`boundary_kz` does for a half-space of the same medium.
    ``patterned`` -- a layer that holds a double-negative region and is not one uniform medium
    -- goes to :func:`power_down_labels`, which labels its modes one by one by their power
    flux. Each is ``None`` when it certainly does not apply, which is every non-magnetic layer
    (``mu_slots is None``) and every layer the engine found free of double negatives, so those
    keep their arithmetic structurally; ``True`` when it certainly does; a traced boolean
    otherwise.

    ``dng`` is the engine's :class:`DoubleNegativeLayer`, or ``None`` when it said nothing
    (``solve_conv`` without a scene to read): then only the uniform case is recognised, from the
    slots (:func:`uniform_double_negative`). Nothing here moves an exact answer -- a label is a
    choice of basis -- only how well ETM's recursion, and SMM's gap factor, are conditioned.
    """
    if mu_slots is None:
        return None, None
    if dng is None:
        uniform = uniform_double_negative(xp, eps_slots, mu_slots, zero_slots)
        return _both(uniform, True), None
    holds, uniform = dng
    if holds is False:
        return None, None
    if uniform is None:
        uniform = uniform_double_negative(xp, eps_slots, mu_slots, zero_slots)
    other = (not uniform) if isinstance(uniform, bool) else ~uniform
    return _both(holds, uniform), _both(holds, other)


#: Flux below this, on a mode normalised to unit energy, counts as none: the direction is then
#: decided by decay. The constant and the normalisation of the first-order split
#: (:func:`~wavebit._core.solve_2d_first_order.split_forward_backward`).
_FLUX_FLAT = 1e-12


def power_down_labels(xp, q, flux, energy, k0_d, active):
    """Relabel a patterned double-negative layer's modes by where their power goes.

    ``q`` are the labels the layer's rule gave, ``flux`` each mode's time-averaged Poynting
    ``z`` component under those labels (it changes sign with the label), ``energy`` its
    ``sum(|psi|**2)``, and ``active`` where the rule applies (:func:`layer_label_rule`).
    Returns ``(q, flip)``: the new labels and where they moved, so the caller can turn the
    partner field with them.

    **Why per mode.** A layer that holds a double-negative region beside an ordinary one has
    modes of both kinds and modes between them, and no single cut labels them all: Maystre's
    root advances the phase of a backward mode downward, against its power, and the mirror cut
    does the same to a forward one. What :func:`etm_mode_q` asks of a label -- that the ``+``
    mode be a wave the region below receives -- is about power, so this reads the power: a
    *propagation-dominated* mode (``|Im(q)| > |Re(q)|``) whose flux is resolvable takes the root
    that carries power down; any other mode takes the decaying root, which for a passive layer at
    a real ``k_par`` is the same root, since a mode's flux decays in the direction it travels.
    That is the rule the first-order path has always split its modes with
    (:func:`~wavebit._core.solve_2d_first_order.split_forward_backward`), so the two
    formulations now agree about which way a patterned negative-index layer's modes go.

    ETM (``k0_d`` given) then applies :func:`etm_mode_q`'s cap. SMM (``k0_d is None``) never
    takes a growing root: it keeps the decaying one and lets the flux break only the tie
    :func:`mode_q` breaks by ``Im(q) <= 0``, the lossless propagating mode.

    Measured against SMM (``docs/connecting-algorithms.md`` section 11): on 274 random
    patterned scenes at oblique incidence on the 1D TE/TM, conical and 2D paths Maystre's labels
    raised ``LinAlgError`` on 15 and lost up to 2.7e-9 on the rest, these at most 6.1e-13; on a
    high-contrast double-negative block resting on its own medium, 2.7e-4 against 1.5e-13.
    """
    re, im = xp.real(q), xp.imag(q)
    sure = xp.abs(flux) > _FLUX_FLAT * energy
    if k0_d is None:
        flip = xp.where(xp.abs(re) > _FLAT, re < 0, xp.where(sure, flux < 0, im > 0))
    else:
        decaying = xp.where(xp.abs(re) > _FLAT, re < 0, im > 0)
        flip = xp.where((xp.abs(im) > xp.abs(re)) & sure, flux < 0, decaying)
        flip = flip != _grows(xp, xp.where(flip, -q, q), k0_d)
    if active is not True:
        flip = flip & active
    return xp.where(flip, -q, q), flip


def boundary_kz(xp, eps, k_par_sq, medium=None):
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
    internal reflection. That is the answer for weak, conventional gain (J. O. Grepstad and
    J. Skaar, below), not a general one: it is contested (T. S. Mansuripur and M. Mansuripur,
    Appl. Phys. Lett. 104, 121106 (2014); A. E. Siegman, Opt. Photon. News 21(1), 38 (2010)),
    and it can be the wrong root for a strongly dispersive active medium (Y.-F. Chen, P. Fischer
    and F. W. Wise, Phys. Rev. Lett. 95, 067402 (2005); B. Nistad and J. Skaar, Opt. Express 15,
    10935 (2007)), whose root is fixed by its whole dispersion and not by ``eps`` at one
    frequency (J. Skaar, URSI EMTS 2010, arXiv:1008.2874). ``docs/negative-index-media.md``
    section 18 says what the literature does and does not settle.

    The rule agrees with the principal root wherever that root lies in the closed first
    quadrant, so it is bit-identical to it at every real ``k_par`` in a passive medium. Its cut
    meets the real ``k_par`` axis only at the Rayleigh point, so a solve is continuous as
    ``Im(k_par) -> 0`` from either side. Until 2026-10-02 the rule was the principal root
    continued from normal incidence. That rule's cut lay along the real evanescent axis, so any
    ``Im(k_par)``, even one the size of round-off, made the evanescent orders on one side grow.
    ``docs/branch-cuts.md`` measures the two rules and the alternatives. On the cut itself
    (``Re(kz**2) = 0``, ``Im(kz**2) < 0``), the root with ``Re(kz) > 0`` is kept.

    Near grazing in a gain half-space no plane-wave value is the answer, and the solve says so
    with :class:`~wavebit._core.backend.GainCutoffWarning` (checked in the engine, per side,
    with :func:`gain_cutoff_band`); this function only chooses the root.

    RETICOLO takes the same cut (``retsqrt(x, -1)``, its "coupure de Maystre"): J. P. Hugonin and
    P. Lalanne, *RETICOLO software for grating analysis*, Institut d'Optique (2005),
    arXiv:2101.00901. The name is that of RETICOLO's source; its manual does not use it. In print
    the convention is ``Re(sqrt(x)) + Im(sqrt(x)) > 0``, which P. Lalanne and T. Wu,
    arXiv:2508.10742 (2025), attribute to R. Petit (ed.), *Electromagnetic Theory of Gratings*
    (Springer, 1980). The gain behaviour is that of J. Skaar, Phys. Rev. E 73, 026605 (2006),
    and J. O. Grepstad and J. Skaar, Opt. Express 19, 21404 (2011), whose ``kz`` changes sign
    only at the critical point, where this cut jumps in a gain medium.

    **A double-negative medium takes the mirror of Maystre's cut.** ``eps`` is ``eps*mu`` and
    cannot say which way the medium's own loss moves ``kz**2``. For an ordinary medium loss
    lifts it (``Im(eps*mu)`` grows), and the limiting absorption above is Maystre's cut. With
    ``Re(epsilon) < 0`` and ``Re(mu) < 0`` loss *lowers* it, since
    ``Im(eps*mu) = Re(eps) Im(mu) + Im(eps) Re(mu)``: a passive double negative puts ``kz**2``
    in the closed lower half plane, where Maystre's root advances its phase away from the
    interface and grows wherever ``Re(kz**2) > 0`` -- the wrong root for a medium whose power
    and phase travel opposite ways (V. G. Veselago, Sov. Phys. Usp. 10, 509 (1968)). The same
    limiting absorption run from below takes the root with ``Im(kz) - Re(kz) > 0``, whose cut is
    the *positive* imaginary axis of ``kz**2``, the mirror of Maystre's under
    ``kz -> -conj(kz)``. At every real ``k_par`` in a passive double negative it is the decaying
    root, ``Im(kz) >= 0``, and a lossless propagating order gets ``Re(kz) < 0``, so
    ``Re(kz/mu) > 0`` and its power flows away. The decision is at least ``|kz|`` from zero
    there, as Maystre's is for an ordinary medium.

    Both cuts together are *the limiting-absorption root, with the loss applied to eps and mu
    separately*. M. Cassier, C. Hazard and P. Joly, Commun. PDE 42, 1707 (2017) and 47, 1217
    (2022), prove it for a lossless Drude half-space with both ``eps`` and ``mu`` dispersive: the
    root, taken with the loss on each factor, changes sign exactly in the bands where both are
    negative. H.-M. Nguyen and S. Sil, Commun. Math. Phys. 379, 145 (2020), put the loss on
    ``eps`` and ``mu`` separately (for a bounded inclusion). The trigger is, in the lossless
    limit, where a passive non-dissipative local material carries backward waves: M. Cassier,
    P. Joly and M. Kachanovska, Comput. Math. Appl. 74, 2792 (2017), Theorem 3.34, for passive
    materials only (their Remark 3.35). R. A. Depine and A. Lakhtakia, Optik 116, 31 (2005),
    state the lossless mirror for a modal (C) method's eigenvalues; M. Lequime, B. Gralak,
    S. Guenneau, M. Zerrad and C. Amra, arXiv:1312.6288 (2013), and PyMoosh's code (D. Langevin
    et al., J. Opt. Soc. Am. B 41, A67 (2024)) state the same switch. None writes it as a cut on
    the positive imaginary axis, and the literature has no name for that cut; "the mirror of
    Maystre's cut" is a description. The mirror medium ``(-conj(eps), -conj(mu))`` is
    A. Lakhtakia's conjugation symmetry, Microwave Opt. Technol. Lett. 40, 160 (2004), which
    conjugates ``E`` and ``H``; V. Grunhut and R. A. Depine, Eur. Phys. J. D 62, 227 (2011),
    apply it to a flat interface, where it changes the phase of the Fresnel coefficients and not
    their magnitude. Lit from a lossless medium with the far side mirrored, its consequence for a
    solve is ``r -> conj(r)``, to the bit (``tests/test_negative_index_half_space.py``). The rule
    assumes ``Im(eps) >= 0`` and ``Im(mu) >= 0``; the effective
    ``Im(mu)`` of a homogenised passive medium can be negative (B. Gralak, C. R. Physique 21, 343
    (2020)), and then neither cut is the outgoing wave -- such a double negative propagates and
    the engine refuses it as active.

    ``medium`` is the half-space's ``(epsilon, mu)`` and selects that rule where both real parts
    are negative; ``None`` -- a caller holding only ``eps*mu``, and every non-magnetic solve --
    takes Maystre's cut, as does any medium with one real part non-negative. Where it is given
    and the medium is not double negative the decision is Maystre's to the bit, and so is the
    root. The two rules pick the same root wherever ``Re(kz**2) < 0``, so a double negative too
    lossy to propagate anywhere (``Re(eps*mu) <= 0``) keeps its bits; they part on
    ``Re(kz**2) >= 0``, and on the negative imaginary axis itself, Maystre's own cut, where his
    tie-break keeps the growing root of a passive double negative. On the mirror cut itself
    (``Re(kz**2) = 0``, ``Im(kz**2) > 0``, reachable only with gain or a complex ``k_par``) the
    root with ``Re(kz) < 0`` is kept, Maystre's tie-break mirrored: the double-negative rule is
    ``-conj(maystre(conj(kz**2)))`` to the bit, the ordinary rule run on the mirror medium
    ``(-conj(eps), -conj(mu))``.
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
    if medium is None:
        _check_resolvable(xp, v, decision, 0.0)
        return xp.where(decision < 0, -v, v)
    # A selection, not arithmetic on `decision`: where the medium is not double negative
    # `where` hands back Maystre's decision and comparison themselves, so the root is the same
    # bits. On the double-negative side the tie goes the other way (`<=`), which makes the rule
    # the exact mirror -conj(maystre(conj(z))), signed zeros included.
    mirror = double_negative(xp, *medium, v)
    decision = xp.where(mirror, xp.imag(v) - xp.real(v), decision)
    _check_resolvable(xp, v, decision, 0.0)
    return xp.where(xp.where(mirror, decision <= 0, decision < 0), -v, v)


def double_negative(xp, epsilon, mu, like):
    """``Re(epsilon) < 0`` and ``Re(mu) < 0``, broadcast against ``like`` -- traceable.

    The material test that sends a half-space to the mirror of Maystre's cut
    (:func:`boundary_kz`). Decided on the material, not on ``eps*mu``: the product of a passive
    double negative is the product of a gain medium with both real parts positive. ``epsilon``
    and ``mu`` may be Python numbers or 0-d arrays; adding a zero of ``like``'s shape and dtype
    puts them in its namespace, as :func:`gain_cutoff_band` does.
    """
    zero = xp.zeros_like(like)
    return (xp.real(epsilon + zero) < 0) & (xp.real(mu + zero) < 0)


def _maystre_factor(factor):
    """``sqrt(factor)`` on Maystre's cut: the principal root, negated where ``Re + Im < 0``.

    The root of switching the factor's loss or gain on from its lossless value
    (:func:`boundary_kz`'s rule, with its tie: ``Re + Im == 0`` keeps the principal root). It is
    the principal root itself wherever that lies in the closed first quadrant or has ``Re > 0``
    -- every factor in the closed upper half plane, and every active factor with ``Re > 0`` --
    and differs only for an active factor with ``Re < 0`` (a gain metal).

    **Decided on the factor, not on the root's round-off.** For the principal root
    ``w = u + iv`` of ``z`` (``u >= 0``), ``Re(z) = (u - v)(u + v)`` and ``Im(z) = 2uv``: where
    ``v < 0``, ``u + v < 0`` exactly when ``Re(z) < 0``, and where ``v >= 0`` never. So
    ``Re(w) + Im(w) < 0`` is ``Re(z) < 0`` and ``Im(w) < 0``, two signs no round-off moves
    (``sqrt`` gives ``Im(w)`` the sign of a nonzero ``Im(z)``), and the tie, ``Re(z) = 0``
    with ``Im(z) < 0``, is never negated. Read off the computed sum, the tie was round-off --
    ``Re + Im`` of a root of ``-0.1j`` comes out ``-2.8e-17``, ``+2.8e-17`` or exactly 0 by
    backend and by the sign of a zero real part -- so ``(eps, mu) = (-0.1j, -3)``, no double
    negative and no gain-metal factor, had its index turned against its ``kz`` (``t_p = -t_s``
    at normal incidence) on NumPy, JAX and :mod:`cmath` and not on PyTorch, and with ``mu = -3``
    so did 67 of the 300 factors ``-0.01j .. -3.00j``, on all but PyTorch. On the negative
    real axis, ``Im(z) = 0``, the root is ``+-i*sqrt(-Re(z))`` and ``Im(w)`` alone decides, as
    the sum did. A Python number stays one; an array goes through ``where``.
    """
    from .domain import index_from_eps          # `domain` imports this module
    root = index_from_eps(factor)
    if not hasattr(root, 'dtype'):
        return -root if factor.real < 0 and root.imag < 0 else root
    xp = array_namespace(root)
    out = xp.where((xp.real(factor) < 0) & (xp.imag(root) < 0), -root, root)
    return out[()] if isinstance(root, np.generic) else out


def signed_index(epsilon, mu):
    """A magnetic half-space's index, ``n = sqrt(epsilon)*sqrt(mu)``, each root on Maystre's cut.

    The one place an ``(epsilon, mu)`` pair becomes an index. The public ``n_top`` / ``n_bot``
    (``engine._half_space_index``), and through them the kernels, the incident ``kx`` and the
    field door, read it, and so does the coherent door's plane-wave basis
    (``illumination._plane_basis``): two spellings could disagree about what a ``p`` amplitude
    or an angle means. A non-magnetic medium never comes here; its index is
    :func:`~wavebit._core.domain.index_from_eps` of ``epsilon`` alone.

    **Why a product of two roots.** ``eps*mu`` has two square roots, and only one is the index.
    For a passive medium each factor lies in the closed upper half plane, so each principal root
    lies in the closed first quadrant and their product in the closed upper half plane:
    ``Im(n) >= 0``, the loss of this ``e^{-iwt}`` convention, whatever the signs of the real
    parts. The principal root of the product does not keep that. A passive double negative
    (``eps = -2+0.1j``, ``mu = -1+0.1j``) multiplies to ``1.99-0.3j``, whose principal root
    ``1.41-0.11j`` reads as gain, and so does a passive magnetic metal (``eps = -2+0.01j``,
    ``mu = 0.1+0.5j``, product ``-0.205-0.999j``). The index of both is the *negative* of that
    root, with ``Re(n) < 0``: they are negative-phase-velocity media, in which the phase
    advances against the power. V. G. Veselago, Sov. Phys. Usp. 10, 509 (1968), writes
    ``n = p*sqrt(eps*mu)`` with ``p = -1`` for the double negative; the condition
    ``Re(sqrt(eps)*sqrt(mu)) < 0`` for the whole class is that of M. W. McCall, A. Lakhtakia and
    W. S. Weiglhofer, Eur. J. Phys. 23, 353 (2002), and R. A. Depine and A. Lakhtakia, Microwave
    Opt. Technol. Lett. 41, 315 (2004). For a passive medium off the real axis of ``eps*mu`` it
    is ``Im(eps*mu) < 0`` (R. Ruppin's form, which Depine and Lakhtakia give); a lossless double
    negative sits on that axis, ``Im(eps*mu) = 0``, and is negative-phase-velocity all the same.
    It never holds for a non-magnetic medium, and among media in which waves propagate
    (``Re(eps*mu) > 0``) it holds exactly for the double negatives. For a passive medium the
    product of principal roots is also the causal index, the one that continues analytically
    into the upper half of the complex-frequency plane (J. Skaar, URSI EMTS 2010,
    arXiv:1008.2874).

    **Each factor's root is Maystre's**, the one with ``Re + Im >= 0``: the root of switching
    that factor's loss or gain on from its lossless value, the rule :func:`boundary_kz` gives a
    half-space. A. A. Govyadinov, V. A. Podolskiy and M. A. Noginov, Appl. Phys. Lett. 91,
    191103 (2007), take ``n = sqrt(eps)*sqrt(mu)`` with each factor's root cut along the
    negative imaginary axis, which is this cut; they note that their table of which media decay
    and which grow cannot be reduced to a single cut in the complex plane. For every passive
    factor it *is* the principal root -- both lie in the closed first quadrant -- and so it is
    for an active factor with ``Re > 0``; so every passive medium, and every active one without
    a gain-metal factor, keeps the index it had to the bit. The two differ only for an active
    factor with ``Re < 0`` (``Im < 0``: a gain metal), whose principal root has ``Re + Im < 0``,
    and a medium with exactly one such factor gets the exact negation of its former index. A
    factor exactly on the negative imaginary axis is the cut's tie and keeps its principal root,
    decided on the factor's real part rather than on the root's round-off
    (:func:`_maystre_factor`).

    For an active medium the sign is a convention -- no rule at one frequency says which index
    is physical (J. Skaar, Phys. Rev. E 73, 026605 (2006), and URSI EMTS 2010) -- and this one
    is chosen to point along the root the solve takes: at normal incidence the half-space's
    ``kz`` is ``+-n``, and ``t_p = t_s`` there exactly when ``n`` points along it. On 400,000
    random media (``studies/negative_index_media.py``, section 3) it does for every passive
    medium either way and for 97.6% of active ones with no gain-metal factor either way; with a
    gain-metal factor, not double negative, for all 93,797 that are evanescent (none under
    principal factors) and for none of the 5,839 that propagate (all under principal factors);
    never for an active double negative that does not propagate (both factors flip, so nothing
    changes); and for every active double negative that propagates, which is refused. One
    consequence: a gain metal named with ``mu = 1`` (``Material(-2-0.1j, mu=1.0)``) now has the
    opposite ``n`` -- and ``t_p`` -- to the same permittivity unnamed, whose index stays
    :func:`~wavebit._core.domain.index_from_eps`, the principal root.

    **What the sign decides.** ``n**2`` does not depend on it, so no ``kz`` does. The kernels
    quote a ``p`` amplitude along a ``p_hat`` normalised by ``n``, so it sets the sign of
    ``t_p`` (with the principal root, a far side of either medium above had ``t_p = -t_s`` at
    normal incidence), and on the lit side it sets the sign of the incident ``kx`` for a real
    angle (``engine.CoreSolveMixin._incident_kxy``).

    **How it is computed.** The principal ``sqrt(eps*mu)`` -- the expression this index always
    was -- negated where it points away from ``sqrt(eps)*sqrt(mu)``: where
    ``Re(conj(r)*q) < 0``, ``r`` the principal root and ``q`` the product of the two factors'
    roots, each principal and then taken to Maystre's cut (:func:`_maystre_factor`). A medium
    that is not negative-phase-velocity gets ``r`` itself, to the bit; one that is gets ``-r``,
    an exact negation, so ``n**2`` keeps its bits. ``q`` only steers: it equals ``r`` or ``-r``
    up to round-off, so the projection is ``+-|n|**2`` and its sign is never in doubt, where the
    sign of ``Re(q)`` alone is round-off on the boundary of the class -- ``eps*mu`` on the
    negative real axis, where ``n`` is imaginary. On 199930 passive pairs placed on that
    boundary to within 3 ulp, reading ``Re(q)`` alone returned the gain-reading root for 35855
    and the projection for none; off it the two agree to the bit on 1e6 random passive pairs
    (``studies/negative_index_media.py``, section 3 of its log). ``q`` is formed from
    ``epsilon + 0j`` and ``mu + 0j``, so a signed zero in a lossless value counts as lossless,
    not as gain: a lossless double negative written ``(complex(-2, -0.0), -1+0j)`` is
    ``-sqrt(2)``, not ``+sqrt(2)``. (Spell such a zero with ``complex()``: the expression
    ``-2-0j`` has a ``+0.0`` imaginary part on Python 3.12 and 3.13 and a ``-0.0`` one from 3.14,
    which converts the real operand of mixed real-complex arithmetic to a float, CPython
    gh-69639.) The result is continuous across
    ``Re(eps) = 0`` and ``Re(mu) = 0`` for every passive medium; a test on the signs of the real
    parts steps there.

    **Except on the axis itself.** Where ``eps*mu`` lies exactly on the negative real axis the
    principal root is purely imaginary (``Re(r) == 0``), and ``-r`` is no more passive than
    ``r``: the medium carries no wave, and which lip of the cut a lossless value sits on is
    decided by the sign of a zero. There ``r`` is returned unturned, the root the index always
    was and the one the non-magnetic index takes, so ``Material(-(4+0j), mu=1.0)`` has the
    ``n_top`` -- and the ``t_p`` -- of ``-(4+0j)`` alone, as naming ``mu = 1`` promises. Off the
    axis ``Re(r) > 0`` and nothing changes. A lossy medium that lands on the axis
    (``eps = -2+0.1j``, ``mu = 1+0.05j``) has ``Im(eps*mu) = +0`` and gets the decaying root
    either way.

    Python numbers in, a Python number out (through :mod:`cmath`, as ``index_from_eps``); a NumPy
    scalar in, a NumPy scalar out, as the principal root gave. Arrays and tracers go through
    their own namespace, with ``where`` rather than a branch on a value, so the sign is decided
    under ``jax.jit`` too.
    """
    from .domain import index_from_eps          # `domain` imports this module
    n = index_from_eps(epsilon * mu)
    q = _maystre_factor(epsilon + 0j) * _maystre_factor(mu + 0j)
    if not hasattr(n, 'dtype') and not hasattr(q, 'dtype'):
        return -n if (n.conjugate() * q).real < 0 and n.real != 0 else n
    xp = array_namespace(n if hasattr(n, 'dtype') else q)
    out = xp.where((xp.real(xp.conj(n) * q) < 0) & (xp.real(n) != 0), -n, n)
    # `where` hands NumPy scalars back as 0-d arrays; the index was a scalar before it was signed.
    return out[()] if isinstance(n, np.generic) else out


def half_space_medium(epsilon, mu):
    """A kernel's ``(eps_side, mu_side)`` as :func:`boundary_kz`'s ``medium``, or ``None``.

    Structural, so a trace never branches on a value: the engine hands the kernels a half-space
    permittivity exactly when the scene is magnetic, and only then can a medium be double
    negative. A non-magnetic solve passes ``None`` and keeps Maystre's arithmetic, unchanged.
    """
    return None if epsilon is None else (epsilon, mu)


#: The band of :func:`gain_cutoff_band`: ``|Re(kz**2)| < _GAIN_CUTOFF_RATIO * |Im(kz**2)|``.
#: Measured in ``studies/gain_cutoff_band.py`` (log in ``studies/results/``), for the steady
#: response of a Gaussian beam of width ``W`` (Briggs-Bers: R. J. Briggs, Electron-Stream
#: Interaction with Plasmas, MIT Press, 1964), glass onto a Lorentz gain line pinned to
#: ``eps = 1 - 0.05i``, with the gain length ``L_g = 2*sqrt(Re eps)/(k0*|Im eps|)``:
#:
#: - Where the field at the beam centre grows with the beam's width: a ratio below about 1. At
#:   0.15, |E_r/E_inc|**2 there is 2.13 / 181 / 1.1e8 for W/L_g = 2.5 / 5 / 7.5.
#: - Beyond 1 a beam converges to the same beam under Maystre's root, but slowly. At 1.17 it is
#:   off by 4.4e-2 / 1.9e-3 / 2.0e-6 / 1.5e-16 for W/L_g = 2.5 / 5 / 10 / 20. At the edge of 2,
#:   a beam of W/L_g = 2.0 .. 2.6 comes within 1e-3 (3.4 .. 4.3 for 1e-6) on the slower side,
#:   at |Im eps| 0.005 .. 0.2.
#: - The fixed-kx causal root of a dispersive line (J. Skaar, Phys. Rev. E 73, 026605, 2006)
#:   flips at angles that move with the line, up to a ratio of 4.9 on a fine grid; a beam does
#:   not see those flips, so they do not set this constant.
_GAIN_CUTOFF_RATIO = 2.0


def gain_cutoff_band(xp, epsilon, mu, k_par_sq):
    """Per order: does a gain half-space put this order in the band near grazing?

    True where the medium has gain (``Im(epsilon) < 0`` or ``Im(mu) < 0``), ``kz**2`` lies on
    the gain side of the real axis, and ``|Re(kz**2)| < _GAIN_CUTOFF_RATIO * |Im(kz**2)|``, with
    ``kz**2 = epsilon*mu - k_par_sq`` normalised by ``k0``. There a finite beam's response
    depends on its width, and the plane-wave value is only what the root rule gives; see
    :class:`~wavebit._core.backend.GainCutoffWarning`.

    **The gain side is below the axis, or above it in a double negative.** Gain lowers ``kz**2``
    in an ordinary medium and *raises* it where ``Re(epsilon) < 0`` and ``Re(mu) < 0``, since
    ``Im(eps*mu) = Re(eps) Im(mu) + Im(eps) Re(mu)``: there the band is the mirror image, the
    one :func:`boundary_kz`'s mirrored cut runs through. The band of a double negative is that
    of its double-positive mirror ``(-conj(epsilon), -conj(mu))`` to the bit. Testing the lower
    half plane for every
    medium flagged no active double negative at all. An active double negative that
    propagates is refused where it is set (``engine._refuse_negative_index_half_space``), so it
    reaches this only where that refusal cannot read a value -- under ``jax.jit`` -- which is
    where this verdict is the one left to read.

    Keyed on the material's own gain, not on the sign of ``Im(epsilon*mu)``: a passive double
    negative has a product below the real axis without any gain. Traceable: elementwise
    comparisons only, so it is the per-order verdict a ``jax.jit`` or ``vmap`` caller can still
    read when the warning cannot run.
    """
    zero = xp.zeros_like(k_par_sq)
    epsilon, mu = epsilon + zero, mu + zero
    kz2 = epsilon*mu - k_par_sq
    gain = (xp.imag(epsilon) < 0) | (xp.imag(mu) < 0)
    near = xp.abs(xp.real(kz2)) < _GAIN_CUTOFF_RATIO*xp.abs(xp.imag(kz2))
    mirrored = (xp.real(epsilon) < 0) & (xp.real(mu) < 0)
    gain_side = xp.where(mirrored, xp.imag(kz2) > 0, xp.imag(kz2) < 0)
    return gain & gain_side & near


def incident_kz(kz_top, index):
    """The incident order's ``kz``, read off the superstrate vector rather than recomputed.

    The boundary match sets a reflected field built from ``kz_top`` (via :func:`boundary_kz`)
    against an incident field. Both sides need the *same* number at the incident order, and
    the reliable way to guarantee that is to take it from the vector the other side already
    uses — ``kz_top[index]`` — instead of writing ``n_top * cos(theta)`` a second time.

    The two spellings agree for a real incidence angle in a lossless medium, a negative-index
    one included (``n_top`` is signed, and so is its root). In an
    absorbing medium, a real angle instead specifies real transverse k, and
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
    # Beside the column atan2(ky, kx) = +-pi/2 - kx/ky + O(kx**3). On it kx is a zero, so the
    # term adds a signed zero and the value is the right angle to the bit; what it adds is the
    # derivative along kx, -1/ky, which a constant drops. Without it AD of anything that reads
    # the frame off the column -- every (0, n) order of a 2D grid at normal incidence -- came
    # out wrong at exactly theta = 0 (dR/dtheta -2.0e-4 against 5.4e-4), right at 1e-4.
    safe_ky = xp.where(ky_b == 0, xp.ones_like(ky_b), ky_b)
    right_angle = right_angle - (kx + xp.zeros_like(varphi)) / safe_ky
    varphi = xp.where(zero_kx, right_angle, varphi)

    p = as_array(xp, 0.0 if phi is None else phi, dtype=varphi.dtype)
    # Same term, same reason: at the origin the "kx" whose sign the ratio dropped is cos(phi).
    incident = (xp.atan(xp.sin(p) / xp.cos(p))
                + _PI * xp.astype(xp.real(xp.cos(p)) < 0, varphi.dtype)) + xp.zeros_like(varphi)
    return xp.reshape(xp.where(zero_kx & (ky == 0), incident, varphi), (-1,))


def order_frame_sign(xp, kx, phi=None):
    """``s_hat . y_hat`` of every order on the ``ky == 0`` row: ``-1`` where ``Re(kx) < 0``, else ``+1``.

    The pure 1D path's whole s/p frame. With ``ky == 0`` and the origin azimuth ``0`` -- the only
    azimuth ``tetm`` admits (``CoreSolveMixin._refuse_illegal_tetm``) -- ``s_hat = z_hat x
    k_hat_par`` is ``+-y_hat``, and ``p_hat`` lies in the xz plane and turns over with it. This
    is exactly ``cos(order_azimuth(xp, 0*kx, kx, 0))``, the rule the conical and 2D paths use,
    spelled without the angle so that ``sin(pi)`` (1.2e-16) never reaches a component that is
    identically zero; ``tests/test_order_azimuth.py`` pins the two against each other, since a
    rule that cannot be *called* has to be *compared*.

    * ``kx == 0`` (``-0.0`` included) is ``+1``: the origin rule at ``phi = 0``, which is what
      ``order_azimuth`` answers there. ``phi`` is that origin azimuth when it is not ``0``: the
      engine hands over ``pi`` at normal incidence from a negative-index medium, where
      ``k_par`` points the other way as ``theta -> 0+``
      (``engine.CoreSolveMixin._frame_azimuth``), and the sign at ``kx == 0`` is then that of
      ``cos(phi)``, ``-1``. ``None`` (the default) is ``phi = 0`` in the arithmetic this always
      had.
    * A purely imaginary ``kx`` (a complex ``theta``) is ``+1`` too, as there: the half-turn is
      keyed on ``Re(kx)``, never on ``sign(kx)``, which is complex-valued on a complex array.

    Returned **real**, in the dtype of ``Re(kx)``: exactly ``+-1`` on every backend and
    precision, so multiplying a complex amplitude by it is an exact sign flip and never widens
    a ``complex64`` solve. ``where`` on a comparison rather than a Python branch: nothing is
    traced through a condition, and the derivative is zero rather than a NaN.
    """
    one = xp.ones_like(xp.real(kx))
    sign = xp.where(xp.real(kx) < 0, -one, one)
    if phi is None:
        return sign
    turned = xp.real(xp.cos(as_array(xp, phi, dtype=kx.dtype, device=_device_of(kx)))) < 0
    return xp.where((kx == 0) & turned, -one, sign)


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
    'theta/phi instead: a real angle keeps kx and ky real. A complex transverse wavevector '
    'is reachable only through a complex theta/phi, which keep the formal '
    'kx/k0 = n*sin(theta)*cos(phi), ky/k0 = n*sin(theta)*sin(phi) with n the incident '
    'half-space index, signed (for ky = 0: theta = arcsin(kx/(k0*n)), phi = 0); R and T are then '
    'formal ratios, power_entering and absorption are NaN, and a ComplexIncidenceWarning says '
    'so. See docs/conventions.md#real-angle-incidence.')


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
    Nothing downstream reads the discrepancy: a measured case returned ``R + T = 0.998``
    where the right answer was ``0.647``, which is an ordinary absorbing number. Refusing
    the input here is the only place it can be caught.

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
