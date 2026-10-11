"""Scattering-matrix method (SMM) — the second connecting algorithm, single-source.

Private ``top``/``bottom`` names mean start/end ports of the positive-z list, as in
``solve_1d``. The engine maps physical bottom/top onto them. Directional descriptions
below use that local propagation frame, not the public upward height.

The *connecting algorithm* is how per-layer eigenmodes are joined into a stack response. ETM
(:mod:`wavebit._core.solve_1d` / :mod:`wavebit._core.solve_2d`) propagates an enhanced
transmittance matrix; SMM composes per-layer scattering matrices with the Redheffer star
product. Both consume the same modes — this module imports the mode builders from the ETM
solves rather than restating the eigen-algebra — and both use the same boundary conventions,
so ``layer_algorithm='SMM'`` returns the same efficiencies as ``'ETM'``, not a second opinion.

Why have it: a layer's scattering matrix depends only on that layer and the shared gap medium,
and the star product is associative. Layer responses can therefore be built once and cascaded,
reused, or reordered without touching their neighbours — where ETM's recursion carries a state
that already contains everything below the current layer. SMM is also the formulation most of
the literature states its results in, which makes it the natural cross-check on the ETM path.

What it is *not*: a rescue from transfer-matrix instability. The option is named ``'ETM'``
because it *is* the enhanced-transmittance form, which is already stable; both algorithms only
ever evaluate ``exp(-k0*q*d)`` inside bounded products, and only with ``Re(q) >= 0`` or, on the
ETM's labels at a complex ``k_par``, a growth capped at ``eps**-0.25``
(:func:`~wavebit._core.conventions.etm_mode_q`). Measured on thick, strongly evanescent and
high-truncation stacks the two agree to ~1e-14 with no energy drift either way, and SMM costs
~1.1-1.4x ETM (the per-layer ``eig`` dominates both), or ~1.3-1.6x once it also recovers the
field-reconstruction state described below.

Formulation, in the conventions this codebase already uses (see :func:`solve_1d`):

* Every region's ``(W, V)`` describe its *downward* modes: a field with amplitudes ``c`` has
  primary variable ``W @ c`` and partner ``V @ c``; an upward wave flips the partner's sign.
  With that single convention the half-space scattering matrices below are derived, not
  transcribed — no per-region ``kz`` sign flip is needed.
* Layers are joined through a fictitious *gap* of zero thickness (the standard trick; the gap
  cancels out of the composed result whatever it is, as long as it is used everywhere). Its
  down wave has admittance (s) and impedance (p) ``sqrt(1 + |k_par|^2)`` on every order, so its
  ``W`` and ``V`` are never singular and no layer's modes meet it nearly totally reflected
  (:func:`_gap_waves`).
* Incidence enters as mode amplitudes directly: ``E_s = a_s`` and ``H_p = -i*n_top*a_p``
  for the coupled path — the resolved ``(a_s, a_p)`` pair, whichever spelling the
  ``polarization`` pair — and for 1D TE/TM the unit primary amplitude signed into each order's
  own s/p frame, ``sigma_m`` of ``E_y`` or (TM) ``H_y``
  (:func:`~wavebit._core.conventions.order_frame_sign`); the TM field state is then read off
  that column rescaled once to unit incident E (``n/mu`` of the incidence medium) so that every
  path's reconstructed field answers a unit electric input.
  These are the same incident amplitudes the ETM final matching solves against. The sign on
  ``H_p`` is the one ``e^{-iωt}`` requires and the one the code has always carried; this
  sentence said ``+i`` — the conjugate — until 2026-08-11, describing the formulation that
  was retired in 2026-08 rather than the code beneath it. Checked rather than reasoned:
  ``R_p / (-i*n_top)`` is the analytic amplitude to 1e-15 on both algorithms
  (``test_lossy_media.test_reflection_amplitude``), and ``+i`` gives its negative.

Field reconstruction works from either algorithm. What :mod:`wavebit._core.fields` reads is
per-layer mode amplitudes, which the ETM recursion leaves behind as a walk over each layer's
``+`` amplitude and the SMM cascade does not leave at all — so the SMM recovers each layer's
``+`` and ``-`` amplitudes afterwards from its own split-plane amplitudes, and hands over the
amplitudes themselves, in slot 8 of the same ``layer_info_list`` (see :func:`_field_state` for
why not the ETM's walk: it inverts a map into each layer, which needs the ETM's labels, and the
cascade does not take them). They are the ETM's amplitudes wherever the two algorithms label a
layer's modes alike, because amplitudes are unique given the modes and their labels, and the
modes come from the same builders. That recovery is a second pass over the stack, done only from
``answers='field'`` up, so declaring anything lower skips it outright: it is the difference
between the two cost figures above. It keeps two vectors per layer and incidence where the ETM
keeps two transfer blocks, so SMM's field rung retains 2.25 blocks per 2D layer to the ETM's
4.25.
"""
import sys
from typing import NamedTuple

import numpy as np

from .backend import _device_of, array_namespace, as_array, check_finite
from .conventions import boundary_kz, half_space_medium, order_azimuth, order_frame_sign
from .enums import Answers, answers_of
from .conventions import incident_column
from .solve_1d import (_block, _diag, _is_backside, _layer_mu, _pack,
                       layer_modes_1d, layer_modes_1d_conical, sp_efficiencies,
                       sp_modes_1d_conical, sp_modes_1d_general)
from .solve_2d import layer_modes_2d, order_grid_kxy, sp_modes_2d
from .power import (incident_power, net_power, power_for_real_k,
                    sp_incident_power, wave_power)

_PI = 3.141592653589793


# --------------------------------------------------------------------------- S-matrix algebra
# A scattering matrix is carried as the 4-tuple of blocks ``(S11, S12, S21, S22)`` relating
# the waves entering at each port to those leaving it:
#     [out_top ; out_bot] = S @ [in_top ; in_bot]
# where ``in_top`` travels down into the structure and ``in_bot`` up into it.

def _ab_matrices(xp, W_to, V_to, W_from, V_from):
    """Interface coefficients ``(A, B)`` for a boundary crossed *into* the ``to`` basis.

    From continuity of the primary field and its partner across the interface,
    ``c_to_down = (A @ c_from_down + B @ c_from_up)/2`` (and the mirrored relation for the
    upward amplitudes), with ``A = W_to^-1 W_from + V_to^-1 V_from`` and ``B`` the difference.
    """
    P = xp.linalg.inv(W_to) @ W_from
    Q = xp.linalg.inv(V_to) @ V_from
    return P + Q, P - Q


def _gap_waves(xp, C, kx, ky=None):
    """The gap's waves on each order: ``(y_s, z_p)``, both ``sqrt(1 + |k_par|**2)``.

    A half-space here is written by two numbers per order, the s wave's admittance ``kz/mu`` and
    the p wave's impedance ``kz/eps`` -- the pair the gap's ``W`` and ``V`` are built from too.
    The gap has no thickness and cancels from every exact answer whatever its waves are, so the
    choice is about conditioning alone. A layer sits between two gaps, and a mode whose own
    ``kz/eps`` (or ``kz/mu``) differs from the gap's by a ratio ``M`` is nearly totally reflected
    there: the layer's gap-embedded transmission is ``~1/M``, the cascade recovers the physical
    one from it by summing bounces, and rounding grows by about ``M``. Only a propagating mode
    carries that across a layer; an evanescent one is damped by ``X``.

    The gap was ``eps = 1 + k_par**2``, ``mu = 1``, ``kz = 1`` until 2026-10-09: ``y_s = 1``,
    ``z_p = 1/(1 + k_par**2)``. Every medium's waves have ``|kz| ~ |k_par|`` on high orders, so
    ``z_p`` ran the wrong way by ``|k_par|**3``. Ordinary layers have no propagating high-order
    modes and never showed it. A layer that sets negative permittivity beside positive does --
    plasmonic and backward modes, ``q = 0.12-16.6i`` on a double-negative TM grating whose orders
    reach ``|kx| = 55.6``, where ``z_p`` was 1/3087 -- and there SMM's rounding floor was 4.5e-10
    at a real angle, against ETM's 1.3e-13. ``docs/connecting-algorithms.md`` section 11 and
    ``studies/smm_gap_waves.py``.

    ``sqrt(1 + |k_par|**2)`` grows like ``|k_par|`` as those waves do, is the same for both
    polarizations, and keeps what ``kz = 1`` was for: never zero, so ``W`` and ``V`` of the gap
    always invert. It is real, so the gap's down wave carries power down and section 11's
    argument -- no mode labelled by its power can make the gap factor singular -- still holds.
    ``|k_par|`` rather than ``k_par``: at a complex ``k_par`` it stays real and nonzero, where
    ``1 + k_par**2`` vanished at ``kx = +-i``.
    """
    abs_sq = xp.real(kx * xp.conj(kx))
    if ky is not None:
        abs_sq = abs_sq + xp.real(ky * xp.conj(ky))
    y = xp.astype(xp.sqrt(1 + abs_sq), C)
    return y, y


def _s_layer(xp, A, B, X):
    """Scattering matrix of one layer embedded between gap half-spaces.

    ``A``/``B`` describe the gap -> layer interface and ``X = diag(exp(-k0*q*d))`` the modal
    propagation across it. The layer is symmetric, so ``S22 == S11`` and ``S21 == S12``.
    """
    A_i = xp.linalg.inv(A)
    XB = X @ B
    common = xp.linalg.inv(A - XB @ A_i @ X @ B)
    S11 = common @ (XB @ A_i @ X @ A - B)
    S12 = common @ X @ (A - B @ A_i @ B)
    return S11, S12, S12, S11


def _s_reflection(xp, W_top, V_top, W_gap, V_gap):
    """Scattering matrix of the superstrate -> gap interface (port 1 in the superstrate)."""
    A, B = _ab_matrices(xp, W_gap, V_gap, W_top, V_top)
    A_i = xp.linalg.inv(A)
    return -A_i @ B, 2 * A_i, 0.5 * (A - B @ A_i @ B), B @ A_i


def _s_transmission(xp, W_bot, V_bot, W_gap, V_gap):
    """Scattering matrix of the gap -> substrate interface (port 2 in the substrate)."""
    A, B = _ab_matrices(xp, W_gap, V_gap, W_bot, V_bot)
    A_i = xp.linalg.inv(A)
    return B @ A_i, 0.5 * (A - B @ A_i @ B), 2 * A_i, -A_i @ B


def redheffer_star(xp, I, S_a, S_b):
    """Redheffer star product ``S_a * S_b`` — cascade two scattering matrices."""
    a11, a12, a21, a22 = S_a
    b11, b12, b21, b22 = S_b
    D_i = xp.linalg.inv(I - b11 @ a22)
    F_i = xp.linalg.inv(I - a22 @ b11)
    return (a11 + a12 @ D_i @ b11 @ a21,
            a12 @ D_i @ b12,
            b21 @ F_i @ a21,
            b22 + b21 @ F_i @ a22 @ b12)


def _n_layers(thickness):
    return len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]


def _prepare(epx_conv_all, kx, ky=None):
    """Namespace, complex working dtype and complex-promoted ``kx`` / ``ky``."""
    xp = array_namespace(epx_conv_all)
    kx = as_array(xp, kx)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        kx = xp.astype(kx, xp.complex128)
        if ky is not None:
            ky = xp.astype(as_array(xp, ky), xp.complex128)
    elif ky is not None:
        ky = as_array(xp, ky)
    return xp, kx.dtype, kx, ky


# ------------------------------------------------------------- field-reconstruction state
# The cascade above answers at the outer ports only. Field reconstruction needs what is
# inside: each layer's own down- and up-going mode amplitudes. Both are read off the
# split-plane amplitudes of the same cascade, by forward maps only -- see `_field_state` for
# how, and for why the ETM's record cannot be borrowed.

#: Read a mode's ``-`` amplitude off the layer's top face when ``|k0 q d|`` is below this
#: (:func:`~wavebit._core.modal_gradient.two_face`). Below it the mode's two amplitudes cancel in
#: the field by about ``1/|k0 q d|``, which is what reading them off two faces multiplies their
#: rounding by; reading off one face multiplies the top's rounding by ``|exp(k0 q d)|`` instead,
#: at most ``e`` here. The two costs cross near 1. Measured, not tuned: ``|exp(-k0 q d)| > 1e-2``
#: also fixed the graze, and lifted the port-continuity floor of an SMM field from ``1e-14`` to
#: ``2e-13`` (``exp(4.6)``), on evanescent orders that do not cancel at all.
_TOP_FACE = 1.0


def _below_reflections(xp, I, S_layers, S_trn):
    """``(gamma, F)``: the reflection below each split plane, and each layer's gap-to-gap map.

    ``gamma[p]`` maps the down-going gap amplitude at split plane ``p`` to the up-going one: it is
    the ``S11`` block of ``L_p * ... * L_{n-1} * S_trn``. Plane ``p`` sits directly above layer
    ``p``, so plane ``n`` faces the substrate interface alone. ``F[p] = (I - s22 gamma[p+1])^-1
    s21`` carries the down-going gap amplitude at plane ``p`` to plane ``p+1``, closed over
    everything below.

    Built bottom-up, one plane at a time, with one solve per layer serving both:
    ``gamma[p] = s11 + s12 gamma[p+1] F[p]``, the loop a Redheffer product closes
    (``(I - gamma s22)^-1 gamma = gamma (I - s22 gamma)^-1``). Sharing it is what makes the walk
    consistent with itself, and not only cheaper: the up-going amplitude at plane ``p`` is then
    the layer's reflection of the down-going one plus its transmission of the up-going one below,
    in the same rounding. Past the ETM's cap that loop is a cancellation, and with ``gamma``
    spelled apart, as :func:`redheffer_star` spells its ``S11`` block, the pure TE/TM field of
    10 um of vacuum at a complex angle jittered 5e-10 under a 1e-9 change of the permittivity and
    its gradient sat 3.4e-4 off a difference quotient; shared, 3e-15 and 2e-9. What that spelling
    kept and this one does not is the cascade's own rounding of ``T``: at a growth of ``7e7`` the
    field at a layer's exit face then differs from the transmitted wave by up to 3e-3 -- less
    than either differs from the true field there (1e-2).
    """
    n = len(S_layers)
    gamma, F = [None] * (n + 1), [None] * n
    gamma[n] = S_trn[0]
    for p in range(n - 1, -1, -1):
        s11, s12, s21, s22 = S_layers[p]
        below = gamma[p + 1]
        F[p] = xp.linalg.solve(I - s22 @ below, s21)
        gamma[p] = s11 + s12 @ (below @ F[p])
    return gamma, F


def _field_state(xp, I, S_ref, S_trn, S_layers, AB, roots, c_inc, coupled=None):
    """Each layer's ``(+, -)`` mode amplitudes, top-to-bottom, and the gap amplitude above them.

    ``roots[p]`` is layer ``p``'s ``(q, k0*d)``; ``c_inc`` the incident columns. The ``+``
    amplitude is anchored at the layer's top face and the ``-`` at its bottom, as
    :mod:`~wavebit._core.fields` expects them, so every exponential it forms decays on the
    cascade's labels.

    Everything is read off ``g_p``, the down-going amplitude in the gap at split plane ``p``,
    whose up-going partner is ``gamma[p] g_p`` (:func:`_below_reflections`):

    * ``g_0`` closes over the superstrate interface, ``g_0 = S_ref21 c_inc + S_ref22 gamma[0]
      g_0``, and ``g_{p+1} = F[p] g_p`` walks down, ending in the gap above the substrate, so the
      substrate's own basis -- singular when an order sits at cutoff, ``kz_bot = 0`` -- is never
      needed;
    * a layer's amplitudes at a face follow from the gap there by the gap -> layer relation of
      :func:`_ab_matrices`: ``(A + B gamma) g / 2`` down and ``(B + A gamma) g / 2`` up;
    * the ``+`` amplitude is the top face's down one; the ``-`` is the bottom face's up one, except
      for a mode with ``|k0 q d| < 1`` (:data:`_TOP_FACE`), which takes the top face's carried down
      by ``exp(k0 q d)`` (:func:`~wavebit._core.modal_gradient.two_face`). Such a mode's two
      amplitudes nearly cancel in the field -- at a layer light line each is ``O(1/q)`` -- and read
      off two faces their rounding did not cancel with them: an SMM field 1e-6 off at the
      ``1e-12`` landing ``auto_detune`` makes after an exact graze, where one face keeps it at the
      ETM's 1e-11;
    * in a layer ``coupled`` flags -- a per-layer list of Python booleans beside ``roots``,
      ``None`` for none: a conical x-cutoff layer, whose ``q`` couples its cluster modes in
      value -- the rows that carry the coupling are read off the bottom face, and every other
      mode as above (``two_face(coupled=True)`` says why that stays exact to first order).

    **Nothing here inverts a map into a layer, and that is why the ETM's record cannot be
    borrowed.** The ETM hands over each layer's ``+`` amplitude and advances it; read off the gap
    that means inverting ``M = (A + B gamma)/2``, which is well conditioned only when the layer's
    ``+`` modes are the waves the region below receives -- the ETM's labels
    (:func:`~wavebit._core.conventions.etm_mode_q`), not the cascade's. This function inverted ``M``
    until 2026-10-10: at a complex ``k_par`` a propagating mode's decaying root travels up, and a
    vacuum layer on vacuum at ``theta = 0.5+0.3j`` had ``M`` exactly singular, ``|E_z|/|E| = 6e-2``
    where it is zero, ``R`` and ``T`` exact. Rebuilding the layers on the ETM's labels (429f4ac)
    fixed that scene and not the inversion: it repeated an ``eig`` on an identical operator, stayed
    O(1) wrong past the ETM's cap, and lost ``log10(G**2)`` digits below it.
    ``docs/connecting-algorithms.md`` §10 has the measurements. The one inverse here is
    ``I - s22 gamma``; past the ETM's cap that loop is itself a cancellation (``1e-10`` at a growth
    of ``6e5``), and the state then carries the cascade's floor rather than the ETM's.
    """
    n = len(S_layers)
    if n == 0:
        return [], None

    gamma, F = _below_reflections(xp, I, S_layers, S_trn)
    g = xp.linalg.solve(I - S_ref[3] @ gamma[0], S_ref[2] @ c_inc)
    g_top = g
    from .modal_gradient import _stop, two_face
    amplitudes = []
    for p in range(n):
        A, B = AB[p]
        q, k0_d = roots[p]
        g_next = F[p] @ g
        up_top = gamma[p] @ g
        plus = 0.5 * (A @ g + B @ up_top)
        minus_top = 0.5 * (B @ g + A @ up_top)
        minus_bottom = 0.5 * (B @ g_next + A @ (gamma[p + 1] @ g_next))
        values = _stop(q if q.ndim == 1 else xp.linalg.diagonal(q))
        keep = xp.abs(values * _stop(as_array(xp, k0_d, dtype=values.dtype))) < _TOP_FACE
        if coupled is not None and coupled[p]:
            # A conical x-cutoff layer (`solve_1d._regular_conical_modes`), whose `q` couples
            # its cluster modes in value. `two_face` as built is first order beside a coupling
            # (measured 6.8e-3 on a field gradient with the cluster kept), so it reads the
            # coupled rows off the bottom face and adds the term it would miss
            # (`two_face(coupled=True)`). Decided by the caller's structural flag, never by
            # the value: a pytree child cannot be branched on.
            minus = two_face(xp, q, k0_d, minus_top, minus_bottom, keep, coupled=True)
        else:
            minus = two_face(xp, q, k0_d, minus_top, minus_bottom, keep)
        amplitudes.append((plus, minus))
        g = g_next
    return amplitudes, g_top


def _field_records(modes, amplitudes, muz, arrange):
    """``layer_info_list`` from the per-layer pieces, bottom-to-top as the ETM appends it.

    Slots 0-4 and 7 are where the ETM puts them, ``(epz_conv_i, W, V, q, d, ., ., muz_conv_i)``,
    the first five named by the legacy goldens. The ETM's transfer blocks (slots 5 and 6) are
    ``None`` here, and slot 8 holds the layer's ``(+, -)`` amplitudes themselves, each laid out by
    ``arrange`` as ``T1`` is: what tells :mod:`~wavebit._core.fields` which record it walks, asked
    structurally (``is None``) so a traced record can answer it. Two vectors per incidence instead
    of the ETM's two full blocks, so this record keeps 2.25 blocks per layer to the ETM's 4.25.
    """
    return [m + [None, None, z, (arrange(plus), arrange(minus))]
            for m, (plus, minus), z in zip(modes, amplitudes, muz)][::-1]


#: What each :attr:`Port.channel` counts, in the words a refused join names it by.
_CHANNELS = {
    'te': 'the pure 1D TE path, one s amplitude per order',
    'tm': 'the pure 1D TM path, one p port variable per order',
    'sp': 'a coupled path, s amplitudes over p port variables',
}


class Port(NamedTuple):
    """The basis one end of a :class:`ScatteringMatrix` expresses its amplitudes in.

    A scattering block is a bare matrix of numbers; what makes those numbers mean anything is
    the set of plane waves its rows and columns count. Two blocks may only be cascaded when the
    end they meet at counts the *same* waves, and that is what this records so
    :meth:`ScatteringMatrix.cascade` can check it instead of trusting the caller.

    ``index`` and ``mu`` are the half-space the port opens into. ``kx`` / ``ky`` are the
    retained orders' in-plane wavevectors, normalised by ``k0`` as everywhere else in the core;
    ``ky`` is ``None`` on the pure 1D TE/TM path, which has no out-of-plane order.

    ``channel`` is the polarization the port's amplitude vector counts: ``'te'`` or ``'tm'`` on
    the pure 1D path, one entry per order, and ``'sp'`` on the coupled paths -- conical, 2D and
    first-order alike -- ``n_orders`` s amplitudes over ``n_orders`` p port variables. It is the
    one field the two pure channels cannot share: a TE and a TM matrix of the same cut have the
    same orders, the same media and the same size, so until 2026-09-27 they joined, and the
    result was the stack's matrix for neither polarization.

    On JAX ``channel`` is pytree aux data, not a child (:func:`_register_pytree_if_jax_loaded`):
    a string cannot cross a ``jax.jit`` boundary as a leaf, and which waves a port counts is
    fixed by the path, never by a traced value.
    """
    kx: object
    ky: object
    index: object
    mu: object
    channel: str


#: The fields of a :class:`Port` that are pytree children, in order; ``channel`` is aux data.
_PORT_CHILDREN = ('kx', 'ky', 'index', 'mu')

_PYTREE_REGISTERED = False


def _register_pytree_if_jax_loaded():
    """Register :class:`Port` as a JAX pytree with ``channel`` in the treedef, if JAX is loaded.

    A ``NamedTuple`` is a pytree without registration, every field a child, which is how a
    ``ScatteringMatrix`` has always crossed a ``jax.jit`` boundary. A string child cannot: the
    output would be refused as ``not a valid JAX type``. Registered here, ``kx`` / ``ky`` /
    ``index`` / ``mu`` stay children -- a traced joint medium is still a leaf -- and the channel
    rides in the treedef, as it has to for a jitted function returning one of these to be
    rebuilt with it. Registered *with keys*, so a leaf's path still names its field
    (``.entry_port.kx``) as the unregistered ``NamedTuple``'s did, not a flat index.

    Lazy, and never importing JAX, for the reason
    :func:`wavebit._core.domain._register_pytree_if_jax_loaded` gives at length. Called from
    :func:`_scattering_matrix`, the one constructor of a port; on import of the JAX backend;
    and from :func:`wavebit.enable_x64`, the documented call for JAX arriving after wavebit.
    The one ordering none covers is a matrix built before ``import jax`` and handed to JAX
    before any JAX solver, any other matrix or ``enable_x64()``: there the channel is still a
    string leaf, and JAX says so loudly (``TypeError: ... of type <class 'str'> ... at path
    S.entry_port.channel``) rather than answering wrongly. Any of the three afterwards
    registers the class and repairs the matrices already built, since registration is per
    class and not per instance. (``import wavebit`` with JAX already loaded needs no call: no
    port exists yet, and the first one built registers.) Before ports carried a channel no
    such ordering existed: every field was a number or ``None``, each a valid leaf registered
    or not.
    """
    global _PYTREE_REGISTERED
    if _PYTREE_REGISTERED:
        return
    jax = sys.modules.get('jax')
    if jax is None:
        return
    tree_util = getattr(jax, 'tree_util', None)
    if tree_util is None:
        return                      # jax is mid-import; the next port will try again
    try:
        tree_util.register_pytree_with_keys(
            Port,
            lambda port: (tuple((tree_util.GetAttrKey(name), getattr(port, name))
                                for name in _PORT_CHILDREN), port.channel),
            lambda channel, children: Port(*children, channel=channel),
            lambda port: (tuple(getattr(port, name) for name in _PORT_CHILDREN), port.channel))
    except ValueError:
        pass                        # already registered; idempotent by design
    _PYTREE_REGISTERED = True


class ScatteringMatrix(NamedTuple):
    """The stack's scattering matrix -- both sides, every order -- kept instead of dropped.

    The SMM cascade builds this to answer the solve at all -- ``R`` and ``T`` are two of its
    blocks applied to the incident vector -- so keeping it adds no arithmetic and four matrices
    of memory, whatever the stack's depth. It rides no declaration of its own: it appears from
    ``answers='amplitude'`` up, because an ``S`` *is* amplitude information and the rung below
    is the one that said only power was wanted. ``solver.scattering`` is ``None`` after an ETM
    solve, which is not an ignored request but the algorithm's own shape -- the ETM fixes the
    boundary condition of a single illumination and never forms a both-sides operator at all.

    Each block is square of side ``n_orders`` on the pure 1D TE/TM path and ``2*n_orders`` on
    the coupled ones, where the s and p amplitudes are stacked in that order -- the same
    layout the incident column has, so ``S11 @ c_inc`` needs no repacking. Four blocks in
    total, whatever the stack's depth. The pure 1D matrix is the coupled one's s-s (TE) or
    p-p (TM) block at ``phi = 0``, in the same port variables (since 2026-09-27; before, it
    was in the Cartesian ``E_y`` / ``H_y`` on every order, ``D S D`` of this with TM's
    ``S11`` / ``S22`` negated besides -- see :func:`_coupled_ports_1d`).

    **Its ports are the two half-spaces, not the SMM's gap.** The cascade's internal gap basis
    (:func:`_gap_waves`) is a gauge, and it cancels between the entry and exit interface
    factors: re-running a solve with a different gap moves these blocks by ~1e-14 while it
    moves the per-layer gap-embedded ones by ~0.5. So these numbers are
    directly readable. For the **s** block, ``S11 @ delta`` is exactly the reflected ``r_s``
    the solve reported and ``S21 @ delta`` its ``t_s``. The **p** block holds port variables,
    not Fresnel amplitudes: its port variable is the p wave's magnetic field, which turns over
    for the up-going wave, so ``S11 @ delta`` is ``-r_p``, and ``S21 @ delta`` is ``t_p`` times
    ``(n_trn/mu_trn) / (n_ref/mu_ref)``. That holds on every path, the pure 1D TM one
    included. Laterally every port amplitude is referenced at the cell centre, as the
    amplitudes and the ``excitation`` weights are (since 2026-09-27; the kernels' own cascade
    inside a solve works about the corner, and the engine signs its blocks over as ``C S C``,
    ``C = diag((-1)**(m + n))``, before they reach this class), so those readings hold on every
    order as stated. :meth:`cascade` takes these published blocks as they are: one sign on
    every port commutes through the Redheffer star product. The per-layer blocks are
    deliberately not here; they would need the gap -> half-space conversion shipped beside
    them to mean anything.

    **Orientation is the hazard, not the numbers.** These are orientation-blind matrices, so
    handing them over without a stated frame is the same shape of defect as a backside field
    cell in a mirrored frame: every number plausible, every number mirrored. They are therefore
    *incidence-relative*, matching what the field cell and the ``R``/``T`` reads already do --
    :attr:`S11` reflects back into the incidence medium whichever side is lit, and
    :attr:`entry_port` is the side that was lit.

    Use :meth:`cascade` rather than multiplying two of these by hand: the ports are carried so
    that a mismatch can be refused, and a mismatch is otherwise invisible (see :meth:`cascade`).
    """
    #: Reflection back into the incidence medium.
    S11: object
    #: Transmission from the far half-space into the incidence medium.
    S12: object
    #: Transmission through the stack, incidence medium to the far half-space.
    S21: object
    #: Reflection back into the far half-space.
    S22: object
    #: The basis of the lit side -- the half-space the light arrives through.
    entry_port: Port
    #: The basis of the far side.
    exit_port: Port

    @property
    def blocks(self):
        """The four blocks alone, in ``(S11, S12, S21, S22)`` order.

        Kept because the whole tuple no longer splats into four matrices -- the ports ride in
        it -- and because :func:`redheffer_star` and every hand-rolled cascade want exactly
        these four.
        """
        return (self.S11, self.S12, self.S21, self.S22)

    def cascade(self, beyond):
        """This stack followed by ``beyond``, as one scattering matrix.

        **Incidence-relative, like every other orientation in this class.** ``beyond`` is the
        piece the light reaches *next* -- not the piece geometrically below. Lit from the top
        those are the same thing; lit from the bottom they are opposite, and the piece the
        light reaches next sits geometrically *above* this one. Getting it backwards is
        refused rather than answered, because the ports no longer meet.

        The two pieces are solved as independent problems that know nothing of each other, and
        this reproduces the joined stack's own matrix to 1e-15 -- from either side. The
        Redheffer star product's ``inv(I - beyond.S11 @ self.S22)`` is the infinite bounce at
        the cut, which is why it needs :attr:`S22`: the response to light arriving at *this*
        piece's far face from outside, a thing no single one-sided solve contains.

        The plane where they meet is a **basis, not a layer**. A half-space has no thickness
        and nothing propagates in it, so nominating one at the cut changes nothing: air,
        ``eps = 2.25`` and ``eps = 12`` all give the same composite, and inserting a real
        zero-thickness air layer into the joined stack moves it by 3.7e-16.

        What the two pieces *must* share is the port where they meet -- the same retained
        orders, the same joint medium and the same polarization channel (:class:`Port`) -- and
        this refuses when they do not. That refusal is
        the reason to call this rather than :func:`redheffer_star` by hand: a join across a
        shifted order grid does not fail loudly, it returns every number plausible and about
        ten percent wrong. ``theta`` is measured in the incidence medium, so a joint medium of
        a different index shifts the whole grid under the same angle.

        Chains, so a stack cut into more than two pieces joins in the order the light meets
        them::

            whole = first.cascade(second).cascade(third)

        Read eagerly: "are these the same basis" is a value comparison, which a traced array
        cannot answer. That is a property of the question -- it is asked once, where two
        solves are joined, not inside a differentiated loop.
        """
        if not isinstance(beyond, ScatteringMatrix):
            raise TypeError(
                f'cascade() takes a ScatteringMatrix, got {type(beyond).__name__}. It comes '
                "off `solver.scattering` after an SMM solve that declared "
                "`answers='amplitude'` or higher.")
        why = _port_disagreement(self.exit_port, beyond.entry_port)
        if why is not None:
            raise ValueError(
                f'cascade() cannot join these two: {why}. This piece\'s exit port and the '
                "next piece's entry port have to be the same basis -- and `beyond` is the "
                'piece the light reaches next, which lit from below is the one geometrically '
                'above this one.')
        xp = array_namespace(self.S11)
        I = xp.eye(self.S11.shape[0], dtype=self.S11.dtype, device=_device_of(self.S11))
        return ScatteringMatrix(*redheffer_star(xp, I, self.blocks, beyond.blocks),
                                entry_port=self.entry_port, exit_port=beyond.exit_port)


def _scattering_matrix(S, backside, kx, ky, n_ref, n_trn, mu_ref, mu_trn, channel):
    """Put the cascade's four blocks into incidence-relative order, with their ports.

    Reversing the 4-tuple is the whole job for the blocks: a cascade of symmetric layer factors
    read from the other end has its blocks in the opposite order, which is why the backside
    efficiency reads above take ``S[3]`` / ``S[1]`` where the top-side ones take ``S[0]`` /
    ``S[2]``. The callers hand in ``n_ref`` / ``n_trn`` and ``mu_ref`` / ``mu_trn``, which are
    already incidence-relative for the same reason, so the ports need no second swap.
    ``channel`` is the same at both ends -- one matrix counts one set of waves -- and the
    reversal cannot move it.
    """
    _register_pytree_if_jax_loaded()
    return ScatteringMatrix(
        *(S[::-1] if backside else S),
        entry_port=Port(kx=kx, ky=ky, index=n_ref, mu=mu_ref, channel=channel),
        exit_port=Port(kx=kx, ky=ky, index=n_trn, mu=mu_trn, channel=channel))


def _coupled_ports_1d(S, sigma, tetm_index):
    """The pure 1D cascade's blocks in the coupled path's s (TE) or p (TM) port variables.

    Two changes of basis, and after them a 1D matrix *is* the s-s or p-p block of the conical
    one at ``phi = 0`` -- the same ports, the same incident column, the same readings.

    * Every port amplitude is quoted along its order's own ``s_hat`` / ``p_hat``, which on the
      ``ky == 0`` row is ``D = diag(sigma)`` of the Cartesian primary, at both ports (the
      two share one ``kx``).
    * The coupled p variable is the p wave's magnetic field, and for an up-going wave it points
      the other way from the down-going one (:func:`_sp_scattering_result`: "the reflected p
      amplitude is reported as its magnetic field, which points the other way for the upward
      wave"), while ``H_y`` does not turn. So a TM block that maps a down-going wave onto an
      up-going one (``S11``) or back (``S22``) takes one more minus.

    Measured on the conical path before this existed, both sides, ``theta = +-0.3``, magnetic
    half-spaces included: TE ``S_c = D S D`` in all four blocks; TM ``S11, S22 = -D S D`` and
    ``S12, S21 = +D S D``, to 8e-15. The backside's block reversal commutes with this map --
    ``D`` is the same at both ports and TM negates ``S11`` and ``S22`` together.

    What follows on either path: ``S11 @ w`` is ``r_s`` for TE and ``-r_p`` for TM, and
    ``S21 @ w`` is ``t_s`` for TE and ``t_p * (n_trn/mu_trn) / (n_ref/mu_ref)`` for TM (port
    units, not Fresnel units) -- ``w`` the incident column, in each order's own frame.
    """
    flip = (-1.0, 1.0, 1.0, -1.0) if tetm_index == 1 else (1.0, 1.0, 1.0, 1.0)
    return tuple(f * (sigma[:, None] * block * sigma[None, :]) for f, block in zip(flip, S))


def _port_disagreement(a, b):
    """Why ports ``a`` and ``b`` are not the same basis, or ``None`` if they are.

    Read eagerly: every comparison here is a value comparison, which a traced array cannot
    answer. That is a property of the question, not a limitation -- "are these the same basis"
    is asked once, when two solves are joined, and never inside a differentiated loop.

    The channel is asked first because it is the one answer no conversion can fix: two pieces
    on different channels are different problems, and a pure piece against a coupled one also
    differs in ``ky`` (``None`` on the pure path), where the ``kx`` / ``ky`` clause would send the
    caller to convert ``theta`` instead. ``Port`` is public and ``channel`` a plain string, so a
    hand-built port may name a channel no solve produces; it is named as that, not looked up
    into a ``KeyError``.
    """
    if a.channel != b.channel:
        def words(channel):
            return _CHANNELS.get(channel, 'a channel no solve produces')
        return (f"they count different polarizations: {a.channel!r} ({words(a.channel)}) "
                f"against {b.channel!r} ({words(b.channel)}). A pure 1D matrix joins only "
                "one solved with the same tetm; to join across polarizations, or a pure piece to "
                "a coupled one, solve every piece on the coupled path -- polarization=(a_s, a_p) "
                'or psi, at phi=0 on a 1D grating, instead of tetm -- whose ports carry both')

    def differ(x, y, tol=1e-12):
        if (x is None) != (y is None):
            return True
        if x is None:
            return False
        xa, ya = np.asarray(x), np.asarray(y)
        if xa.shape != ya.shape:
            return True
        return bool(np.max(np.abs(xa - ya)) > tol)

    if differ(a.kx, b.kx) or differ(a.ky, b.ky):
        return ('their retained orders are different in-plane wavevectors. `theta` is measured '
                'in the incidence medium, so the same angle means a different `k_par` on either '
                'side of a material step. Choose the angles for the same transverse '
                'wavevector in both media, and check `order_kxy()` matches before composing. '
                'For lossless media this is `theta_b = arcsin(sin(theta_a) * n_a / n_b)` with '
                'the signed indices `n_top` / `n_bot` (negative for a negative-index medium)')
    if differ(a.index, b.index) or differ(a.mu, b.mu):
        return ('they open into different half-spaces. The joint medium is a basis rather than '
                'a layer -- any index will do, and a zero-thickness one changes nothing -- but '
                'both pieces have to name the same one')
    return None


# --------------------------------------------------------------------------------- 1D TE / TM

def smm_1d(tetm_index, wavelength, kx, n_top, n_bot, theta, thickness,
           epx_conv_all, epy_conv_all, epz_conv_i_all,
           incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
           mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None, phi=None, dng=None):
    """Solve a 1D grating stack by scattering matrices — SMM twin of :func:`solve_1d`.

    Same arguments and same returns as :func:`wavebit._core.solve_1d.solve_1d`, so the two are
    interchangeable behind ``layer_algorithm`` — ``mu`` and the half-space terms included, and
    that is the point of carrying them here rather than refusing: the connecting algorithm is
    a choice about how layers are joined, not about what a layer is made of
    (:func:`smm_2d`).

    ``incidence_side='bottom'`` costs nothing here: the cascade composes all four blocks
    whichever side is lit, so backside illumination is *reading* ``S22`` / ``S12`` where the
    top side reads ``S11`` / ``S21``, plus the incidence-medium normalization. That one-pass
    read and the ETM's mirrored-scene solve are the same algebra, not merely the same physics:
    the two interface matrices are block-swaps of each other (``_s_transmission`` returns
    ``_s_reflection``'s blocks in the swapped order, same ``A``/``B``), each layer's matrix is
    symmetric, and reversing a cascade of such factors block-swaps the product. Measured
    against the ETM backside solve at 1e-16 on efficiencies and 1e-14 on field cells
    (``tests/test_backside_incidence.py``), when this branch still served public bottom
    incidence. Since 2026-09-21 the engine sends public *top* incidence here, so the ETM/SMM
    comparisons that reach it are the top-lit ones, ``tests/test_scattering_core.py`` among
    them.

    From ``answers='field'`` up the returned mapping also carries the ``layer_info_list`` /
    ``T1`` state field reconstruction consumes, recovered from the cascade by
    :func:`_field_state` — for the backside, from the mirrored cascade's interface matrices,
    so the state describes the mirrored scene exactly as the ETM path's does. Below that rung
    neither the per-layer matrices nor the recovery pass happen at all.
    """
    answers = answers_of(answers)
    keep_field = answers.covers(Answers.FIELD)
    backside = _is_backside(incidence_side)
    xp, C, kx, _ = _prepare(epx_conv_all, kx)
    mu_all = (None,) * 3 if mu is None else tuple(mu)
    # Each order's s/p frame on this row, the ETM twin's `sigma` (see there); `phi` is the
    # frame's azimuth at kx == 0, as there.
    sigma = order_frame_sign(xp, kx, phi)

    n_orders_u = kx.shape[0]
    k0 = 2 * _PI / wavelength
    theta = 0.0 if theta is None else theta
    I = xp.astype(xp.eye(n_orders_u), C)

    # Boundary branches come from the same place the ETM path takes them, so the two
    # algorithms answer identically.
    kz_top = boundary_kz(xp, n_top ** 2, kx ** 2, half_space_medium(eps_top, mu_top))
    kz_bot = boundary_kz(xp, n_bot ** 2, kx ** 2, half_space_medium(eps_bot, mu_bot))
    if eps_top is None:
        eps_top = n_top ** 2 / mu_top     # n stays the index; eps carries the mu
    if eps_bot is None:
        eps_bot = n_bot ** 2 / mu_bot

    # The gap is written by its waves, the s admittance `y_s` and p impedance `z_p` on each
    # order (:func:`_gap_waves`, which says why those values). It takes **nothing from the scene**,
    # the same decision `_smm_sp`'s gap records: it is a bookkeeping device rather than a medium
    # in the scene and it cancels out of the result, so giving it the scene's `mu` would be a
    # category error that happens to cancel — harder to notice than one that does not.
    y_s, z_p = _gap_waves(xp, C, kx)

    # The half-space admittances, and the same pair the ETM twin's `G` is built from: `kz/mu`
    # where the primary is an electric amplitude (TE) and `kz/eps` where it is magnetic (TM).
    # Every one is the expression that was here at mu = 1.
    if tetm_index == 0:      # TE — primary variable E_y, partner H_x
        V_top = -1j * _diag(xp, kz_top / mu_top)
        V_bot = -1j * _diag(xp, kz_bot / mu_bot)
        V_gap = -1j * _diag(xp, y_s)
    elif tetm_index == 1:    # TM — primary variable H_y, partner scaled by eps
        V_top = -1j * _diag(xp, kz_top / eps_top)
        V_bot = -1j * _diag(xp, kz_bot / eps_bot)
        V_gap = -1j * _diag(xp, z_p)
    else:
        raise ValueError(f'tetm_index must be 0 (TE) or 1 (TM), got {tetm_index!r}')
    W_top = W_bot = W_gap = I    # homogeneous half-spaces are diagonal in this basis

    # --- cascade: superstrate interface, every layer top-to-bottom, substrate interface ----
    S_ref = _s_reflection(xp, W_top, V_top, W_gap, V_gap)
    S_trn = _s_transmission(xp, W_bot, V_bot, W_gap, V_gap)
    S_layers, AB, modes, muzs = [], [], [], []
    S = S_ref
    for layer_index in range(_n_layers(thickness)):
        layer_mu = _layer_mu(I, mu_all, layer_index)
        W, V, q = layer_modes_1d(xp, tetm_index, kx, I, epx_conv_all[layer_index],
                                 epy_conv_all[layer_index], epz_conv_i_all[layer_index], C,
                                 mu=layer_mu, dng=None if dng is None else dng[layer_index])
        d = thickness[layer_index]
        from .modal_gradient import exponential
        X = exponential(xp, q, -k0*d)
        A, B = _ab_matrices(xp, W, V, W_gap, V_gap)
        S_layer = _s_layer(xp, A, B, X)
        S = redheffer_star(xp, I, S, S_layer)
        if keep_field:
            S_layers.append(S_layer)
            AB.append((A, B))
            modes.append([epz_conv_i_all[layer_index], W, V, q, d])
            # `muz_conv_i` travels with its layer for field reconstruction to divide `H_z` by,
            # exactly as `epz_conv_i` leads the entry for `E_z`. Kept in its own list because
            # it rides in slot 7, after the ETM's two transfer-block slots and before the
            # amplitudes recovered below -- the layout `field_1d` unpacks (`_field_records`).
            # Collected here as well as in the ETM twin because the two build this list by
            # their own routes, and a stack whose `H_z` was magnetic under one algorithm and
            # not the other would have identical `R` and `T` either way.
            muzs.append(None if layer_mu is None else layer_mu[2])
    S = redheffer_star(xp, I, S, S_trn)

    # --- incidence -> amplitudes -> efficiency --------------------------------------------
    # Incidence-relative labels: reflection returns into the incidence medium, transmission
    # leaves through the far side. For the backside these are the substrate-port outputs, so
    # the reads swap to S22 / S12; everything after is the top-side arithmetic with the two
    # half-spaces in incidence-relative roles.
    # The incident column in the Cartesian primary the cascade is written in, each lit order
    # along its own s_hat / p_hat: the same two-branch column as :func:`solve_1d`'s, for the
    # same signed-zero reason.
    column = incident_column(xp, C, lit, n_orders_u, n_orders_u // 2)
    delta = xp.reshape(column * (sigma[n_orders_u // 2] if lit is None else sigma),
                       (n_orders_u, 1))
    R = xp.reshape((S[3] if backside else S[0]) @ delta, (1, n_orders_u))
    Tt = xp.reshape((S[1] if backside else S[2]) @ delta, (1, n_orders_u))

    kz_ref, kz_trn = (kz_bot, kz_top) if backside else (kz_top, kz_bot)
    n_ref, n_trn = (n_bot, n_top) if backside else (n_top, n_bot)
    # The half-space material slots take the same incidence-relative roles the indices do:
    # `ref` is whichever side the light arrives from, `trn` the far one.
    mu_ref, mu_trn = (mu_bot, mu_top) if backside else (mu_top, mu_bot)
    eps_ref, eps_trn = (eps_bot, eps_top) if backside else (eps_top, eps_bot)

    # The same per-wave power definition as the ETM pure-1D path.
    admittance_ref = kz_ref / (mu_ref if tetm_index == 0 else eps_ref)
    admittance_trn = kz_trn / (mu_trn if tetm_index == 0 else eps_trn)
    incoming_power = incident_power(xp, C, lit, admittance_ref)
    de_ri = wave_power(xp, R, admittance_ref) / incoming_power
    de_ti = wave_power(xp, Tt, admittance_trn) / incoming_power
    entering = power_for_real_k(
        xp, xp.sum(net_power(xp, xp.reshape(delta, R.shape), R, admittance_ref))
        / incoming_power, kx)

    check_finite(xp, de_ri=de_ri, de_ti=de_ti)

    if not answers.covers(Answers.AMPLITUDE):
        return de_ri, de_ti, entering

    # `ky` is None: this path carries no out-of-plane order; the channel (`Port`) is what
    # refuses to join a coupled-path port, or the other polarization's. The two
    # permeabilities *are* said, in their incidence-relative roles, because a magnetic
    # half-space changes the port's own admittance -- they were literal `1.0` while this path
    # refused a `mu`.
    # Published in the coupled path's port variables, so a 1D matrix is the s-s (TE) or p-p
    # (TM) block of the conical one at `phi = 0` -- see :func:`_coupled_ports_1d`.
    scattering = _scattering_matrix(_coupled_ports_1d(S, sigma, tetm_index), backside, kx,
                                    None, n_ref, n_trn, mu_ref, mu_trn,
                                    channel='tm' if tetm_index == 1 else 'te')

    # Same Fresnel conversion as the ETM pure path: reflection primaries are already the
    # amplitude; the TM transmission converts H back to E across the index step -- by
    # impedance `Z = mu/n` once a half-space carries a permeability, which is `n_ref/n_trn`
    # to the bit when neither does. Parenthesised as one ratio for that reason. Then each
    # order into its own s/p frame, as there.
    zeros_amp = xp.zeros_like(R)
    zeros_eff = xp.zeros_like(de_ri)
    if tetm_index == 0:
        res = _pack(R * sigma, zeros_amp, Tt * sigma, zeros_amp, de_ri, zeros_eff, de_ti,
                    zeros_eff, power_entering=entering)
    else:
        res = _pack(zeros_amp, R * sigma, zeros_amp,
                    (Tt * ((n_ref / mu_ref) / (n_trn / mu_trn))) * sigma,
                    zeros_eff, de_ri, zeros_eff, de_ti, power_entering=entering)

    if not keep_field:
        return {'incident': res, 'scattering': scattering}

    # Unit incident E on the TM path, as on every other path -- the same rescale, and the same
    # reason, as :func:`~wavebit._core.solve_1d.solve_1d`'s, put on the incident column so every
    # amplitude the state holds carries it. `n_ref` / `mu_ref` are the incidence medium whichever
    # port is lit.
    column = delta * (n_ref / mu_ref) if tetm_index == 1 else delta
    roots = [(m[3], k0 * m[4]) for m in modes]
    if backside:
        # The mirrored scene's own cascade pieces, from what this pass already built: the layer
        # matrices are symmetric, and each mirrored half-space interface is the scene's other one
        # with its blocks reversed -- the same arithmetic `_s_reflection` / `_s_transmission`
        # would repeat. The recovered state then describes the mirrored scene, the same frame the
        # ETM backside solve leaves behind, and field reconstruction un-mirrors once, at the cell
        # level. `muzs` reverses with `modes`: a `mu_zz` left in scene order against mirrored
        # modes gives each layer its neighbour's permeability -- still magnetic, still
        # energy-conserving, `R` and `T` untouched.
        amplitudes, T1 = _field_state(xp, I, S_trn[::-1], S_ref[::-1], S_layers[::-1], AB[::-1],
                                      roots[::-1], column)
        modes, muzs = modes[::-1], muzs[::-1]
    else:
        amplitudes, T1 = _field_state(xp, I, S_ref, S_trn, S_layers, AB, roots, column)

    def flat(v):                              # the 1D path carries its state as plain vectors
        return xp.reshape(v, (n_orders_u,))
    layer_info_list = _field_records(modes, amplitudes, muzs, flat)
    if T1 is not None:
        T1 = flat(T1)
    return {'incident': res, 'layer_info_list': layer_info_list, 'T1': T1,
            'scattering': scattering}


# ------------------------------------------------------------------- coupled s/p (conical, 2D)

def _sp_scattering_result(xp, C, S, kx, ky, kz_top, kz_bot, n_top, n_bot,
                          incidence, backside, answers, mu_top, mu_bot,
                          eps_top, eps_bot, lit=None):
    """Public s/p amplitudes and powers from a cascade in scalar half-space bases.

    Shared by the second- and first-order SMM paths. Also return the incident
    columns so field reconstruction uses precisely the same normalization.
    """
    keep_amplitude = answers_of(answers).covers(Answers.AMPLITUDE)
    kx_g, _ = order_grid_kxy(xp, kx, ky)
    n_orders_v, n_orders_u = kx_g.shape
    n_orders = n_orders_u * n_orders_v
    kz_ref, kz_trn = (kz_bot, kz_top) if backside else (kz_top, kz_bot)
    n_ref, n_trn = (n_bot, n_top) if backside else (n_top, n_bot)
    mu_ref, mu_trn = (mu_bot, mu_top) if backside else (mu_top, mu_bot)
    eps_ref, eps_trn = (eps_bot, eps_top) if backside else (eps_top, eps_bot)

    # --- incidence ------------------------------------------------------------------------
    # The scalar half-space basis uses (E_s, H_p), so the incident column is the
    # incident field itself: E_s = a_s, H_p = -i*n_ref/mu_ref*a_p in the incidence medium — with
    # (a_s, a_p) as resolved at the boundary — one form arrives here, and it is the same
    # amplitudes the ETM final matching solves against (see conventions.incident_sp_amplitudes).
    delta = xp.reshape(incident_column(xp, C, lit, n_orders, n_orders // 2), (n_orders, 1))

    def incident(a_s, a_p):
        # Convert the electric p coefficient to the magnetic boundary variable.
        # The actual s/p input powers are evaluated separately below.
        return xp.concat([a_s * delta, -1j * (n_ref / mu_ref) * a_p * delta], axis=0)

    def incident_psi(psi_v):
        return incident(xp.sin(xp.asarray(psi_v, dtype=C)),
                        xp.cos(xp.asarray(psi_v, dtype=C)))

    # The incidence as asked, then the pure-TE / pure-TM variants (free: same scattering
    # matrix). The variants stay spelled as angles so their columns are bit-identical to the
    # pre-`polarization` ones.
    cols = ([incident(as_array(xp, incidence[0], dtype=C),
                      as_array(xp, incidence[1], dtype=C))]
            + ([incident_psi(_PI / 2), incident_psi(0.0)] if keep_amplitude else []))
    c_inc = xp.concat(cols, axis=1)

    c_ref = (S[3] if backside else S[0]) @ c_inc
    c_trn = (S[1] if backside else S[2]) @ c_inc

    n_inc = len(cols)

    def as_orders(M):
        """(n_orders, n_inc) amplitude columns -> (n_inc, n_orders_v, n_orders_u)."""
        return xp.reshape(xp.matrix_transpose(M), (n_inc, n_orders_v, n_orders_u))

    # The reflected p amplitude is reported as its magnetic field, which points the other way
    # for the upward wave; s and the transmitted pair need no such flip.
    R_s = as_orders(c_ref[:n_orders, :])
    R_p = as_orders(-c_ref[n_orders:, :])
    T_s = as_orders(c_trn[:n_orders, :])
    T_p = as_orders(c_trn[n_orders:, :])

    kz_ref_2d = xp.reshape(kz_ref, (n_orders_v, n_orders_u))
    kz_trn_2d = xp.reshape(kz_trn, (n_orders_v, n_orders_u))
    s_power, p_power = sp_incident_power(xp, C, lit, kz_ref, n_ref, eps_ref, mu_ref)
    a_s = as_array(xp, incidence[0], dtype=C)
    a_p = as_array(xp, incidence[1], dtype=C)
    powers = [xp.abs(a_s)**2 * s_power + xp.abs(a_p)**2 * p_power]
    if keep_amplitude:
        powers.extend([s_power, p_power])
    denom = xp.reshape(xp.stack(powers), (n_inc, 1, 1))
    eff = sp_efficiencies(xp, kz_ref_2d, kz_trn_2d, denom, n_ref, n_trn, R_s, R_p, T_s, T_p,
                          mu_top=mu_ref, mu_bot=mu_trn, eps_top=eps_ref, eps_bot=eps_trn)
    flux = (net_power(xp, as_orders(c_inc[:n_orders]), R_s, kz_ref_2d / mu_ref)
            + net_power(xp, as_orders(c_inc[n_orders:]), R_p, kz_ref_2d / eps_ref))
    entering = power_for_real_k(
        xp, xp.sum(flux, axis=(-2, -1)) / xp.stack(powers), kx, ky)

    check_finite(xp, de_ri=eff[0][0] + eff[1][0], de_ti=eff[2][0] + eff[3][0])

    if not keep_amplitude:
        return (eff[0][0] + eff[1][0], eff[2][0] + eff[3][0], entering[0]), c_inc

    scattering = _scattering_matrix(S, backside, kx, ky, n_ref, n_trn, mu_ref, mu_trn,
                                    channel='sp')

    # Coupled-path Fresnel conversion, identical to the ETM final matching's.
    amps = (R_s, R_p / (-1j * n_ref / mu_ref), T_s, T_p / (-1j * n_trn / mu_trn))
    packed = [_pack(*(a[i] for a in amps), *(e[i] for e in eff),
                    power_entering=entering[i]) for i in range(3)]

    return {'incident': packed[0], 's_inc': packed[1], 'p_inc': packed[2],
            'scattering': scattering}, c_inc


def _smm_sp(xp, C, wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
            layer_modes, epz_conv_i_all, backside, answers, mu_top=1.0, mu_bot=1.0,
            eps_top=None, eps_bot=None, muz_conv_i_all=None, lit=None):
    """Shared SMM cascade + matching for the coupled s/p paths (conical and 2D).

    ``layer_modes(layer_index)`` returns that layer's ``(big_W, big_V, q, W, V, coupled)``: the
    mode matrices in the s/p boundary basis that the cascade joins layers through, then the raw
    (unrotated) pair that field reconstruction expands the same amplitudes with, and a Python
    boolean saying whether ``q``'s value couples modes -- a conical x-cutoff layer
    (:func:`~wavebit._core.solve_1d._regular_conical_modes`), whose exponential then takes an
    exact Frechet tangent (:func:`~wavebit._core.modal_gradient.exponential`) and whose coupled
    rows the field state reads off the bottom face (:func:`_field_state`). Structural, so it is
    never read off a traced value; under an abstract trace every conical layer arrives coupled.
    Everything
    else — half-spaces, gap, cascade, incidence, efficiencies — is common.

    ``muz_conv_i_all`` is the per-layer ``mu_zz^-1``, carried for one reason: field
    reconstruction recovers ``H_z`` algebraically from it, exactly as it recovers ``E_z`` from
    ``epz_conv_i_all``, and the cascade itself never looks at either. Absent on the conical
    caller, which has no permeability at all. Dropping it here would leave the SMM answering a
    magnetic stack's ``H_z`` non-magnetically while the ETM twin answered it correctly — the
    one asymmetry :func:`smm_2d`'s docstring exists to rule out.

    ``backside`` selects the substrate-port reads (see :func:`smm_1d` for why that read *is*
    the mirrored-scene solve): the incident column is built with the incidence medium's index,
    reflection comes out of ``S22`` and transmission out of ``S12``, and the p sign flip stays
    with the reflection port — in the mirror's frame the reflected wave is the upward one,
    whichever scene medium it returns into.

    ``incidence`` is the resolved electric coefficient pair ``(a_s, a_p)`` of unit norm — see
    :func:`~wavebit._core.conventions.incident_sp_amplitudes`.

    ``answers`` is the rung this solve was declared for; it decides how many incidence
    columns the cascade is asked about (one, or three once the pure-TE/TM variants are
    wanted) and whether the per-layer pieces are kept for :func:`_field_state` to walk.
    """
    answers = answers_of(answers)
    keep_field = answers.covers(Answers.FIELD)
    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    n_orders_v, n_orders_u = kx_g.shape
    n_orders = n_orders_u * n_orders_v
    k0 = 2 * _PI / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(n_orders), C)
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)
    big_I = xp.astype(xp.eye(2 * n_orders), C)

    k_par_sq = kx_g ** 2 + ky_g ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq,
                                    half_space_medium(eps_top, mu_top)), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq,
                                    half_space_medium(eps_bot, mu_bot)), (-1,))
    # `n` stays the refractive index and `eps` carries the `mu`, so the two are recoverable
    # from each other -- but only up to rounding, since the index reached this call through a
    # square root. The engine holds both numbers and passes them; a caller that holds only an
    # index leaves these `None` and gets the recovery, which is what this always did.
    if eps_top is None:
        eps_top = n_top ** 2 / mu_top
    if eps_bot is None:
        eps_bot = n_bot ** 2 / mu_bot
    y_s, z_p = (xp.reshape(w, (-1,)) for w in _gap_waves(xp, C, kx_g, ky_g))   # see there

    Kz_top = _diag(xp, kz_top)
    Kz_bot = _diag(xp, kz_bot)

    # Half-space and gap modes in the (E_s, F_p) / (H_s, H_p) boundary basis — the same form
    # the ETM substrate seed (big_F, big_G) has, and the same two admittances: `kz/eps` on the
    # magnetic (p) amplitude, `kz/mu` on the electric (s) one.
    #
    # The **gap takes nothing from the scene**, deliberately: not its `mu`, not its media. It is
    # not a medium in the scene; it is a fictitious layer the cascade inserts so every interface
    # is computed the same way, and it cancels out of the result. Giving a bookkeeping device a
    # material property would be a category error that happens to cancel, which is worse than
    # one that does not. Its waves fill the slots a half-space's `kz/mu` and `kz/eps` fill.
    W_top = _block(xp, [[I, O], [O, -1j * Kz_top / eps_top]])
    V_top = _block(xp, [[-1j * Kz_top / mu_top, O], [O, I]])
    W_bot = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])
    V_bot = _block(xp, [[-1j * Kz_bot / mu_bot, O], [O, I]])
    W_gap = _block(xp, [[I, O], [O, -1j * _diag(xp, z_p)]])
    V_gap = _block(xp, [[-1j * _diag(xp, y_s), O], [O, I]])

    S_ref = _s_reflection(xp, W_top, V_top, W_gap, V_gap)
    S_trn = _s_transmission(xp, W_bot, V_bot, W_gap, V_gap)
    S_layers, AB, modes, muz_conv, coupled = [], [], [], [], []
    S = S_ref
    for layer_index in range(_n_layers(thickness)):
        big_W, big_V, q, W, V, rebuilt = layer_modes(layer_index)
        d = thickness[layer_index]
        from .modal_gradient import exponential
        big_X = exponential(xp, q, -k0*d, coupled=rebuilt)
        A, B = _ab_matrices(xp, big_W, big_V, W_gap, V_gap)
        S_layer = _s_layer(xp, A, B, big_X)
        S = redheffer_star(xp, big_I, S, S_layer)
        if keep_field:
            S_layers.append(S_layer)
            AB.append((A, B))
            modes.append([epz_conv_i_all[layer_index], W, V, q, d])
            muz_conv.append(None if muz_conv_i_all is None else muz_conv_i_all[layer_index])
            coupled.append(rebuilt)
    S = redheffer_star(xp, big_I, S, S_trn)

    out, c_inc = _sp_scattering_result(
        xp, C, S, kx, ky, kz_top, kz_bot, n_top, n_bot, incidence, backside,
        answers, mu_top, mu_bot, eps_top, eps_bot, lit)
    if not keep_field:
        return out
    n_inc = c_inc.shape[1]

    roots = [(m[3], k0 * m[4]) for m in modes]
    if backside:
        # Mirrored-scene field state, same construction as `smm_1d`'s backside branch.
        amplitudes, T1 = _field_state(xp, big_I, S_trn[::-1], S_ref[::-1], S_layers[::-1],
                                      AB[::-1], roots[::-1], c_inc, coupled=coupled[::-1])
        modes = modes[::-1]
        muz_conv = muz_conv[::-1]
    else:
        amplitudes, T1 = _field_state(xp, big_I, S_ref, S_trn, S_layers, AB, roots, c_inc,
                                      coupled=coupled)

    def by_incidence(v):
        # One column per incidence variant (psi, TE, TM), as the ETM stacks them.
        return xp.stack([v[:, i:i + 1] for i in range(n_inc)])
    layer_info_list = _field_records(modes, amplitudes, muz_conv, by_incidence)
    if T1 is not None:
        T1 = by_incidence(T1)
    out.update(layer_info_list=layer_info_list, T1=T1)
    return out


def smm_1d_conical(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                   epx_conv_all, epy_conv_all, epz_conv_i_all,
                   *, phi, incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
                   mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None, dng=None):
    """Solve a conical 1D grating by scattering matrices — SMM twin of :func:`solve_1d_conical`.

    Same arguments and returns as :func:`smm_1d`; ``polarization`` is the complex amplitude
    pair ``(a_s, a_p)``, resolved at the boundary as in the ETM twin.

    ``mu`` is the three permeability stacks
    (:func:`~wavebit._core.solve_1d.layer_modes_1d_conical`), handed to the *same* layer-mode
    function the ETM twin calls, and the half-space terms go to the shared :func:`_smm_sp`.
    Carried here rather than refused for the reason :func:`smm_2d`'s docstring gives: which
    connecting algorithm joins the layers is not a fact about what a layer is made of, and an
    algorithm that quietly dropped a ``mu`` would answer a magnetic stack plausibly and
    wrongly while its twin answered it correctly.
    """
    backside = _is_backside(incidence_side)
    xp, C, kx, ky = _prepare(epx_conv_all, kx, ky)
    mu_all = (None,) * 3 if mu is None else tuple(mu)

    n_orders_u = kx.shape[0]
    I = xp.astype(xp.eye(n_orders_u), C)
    ky_col = xp.reshape(ky, (-1, 1))
    varphi = order_azimuth(xp, ky_col, kx, phi)

    Kx = _diag(xp, kx)
    ky_vec = ky[0] * xp.astype(xp.ones((n_orders_u,)), C)      # n_orders_v == 1: Ky = diag(ky0 * 1)
    Ky = _diag(xp, ky_vec)
    Kx2 = _diag(xp, kx * kx)
    Ky2 = _diag(xp, ky_vec * ky_vec)
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

    def layer_modes(layer_index):
        modes = layer_modes_1d_conical(
            xp, Kx, Ky, Kx2, Ky2, I, epx_conv_all[layer_index],
            epy_conv_all[layer_index], epz_conv_i_all[layer_index], C,
            mu=_layer_mu(I, mu_all, layer_index), dng=None if dng is None else dng[layer_index])
        if len(modes) == 3:
            # An x-cutoff layer, in the general layout (`solve_1d._regular_conical_modes`).
            E, U, q = modes
            big_W, big_V = sp_modes_1d_general(xp, F_c, F_s, E, U)
            return big_W, big_V, q, E, U, True
        W1, W2, q1, q2, V_11, V_12, V_21, V_22 = modes
        big_W, big_V = sp_modes_1d_conical(xp, F_c, F_s, W1, W2, V_11, V_12, V_21, V_22)
        W = _block(xp, [[W1, W2]])
        V = _block(xp, [[V_11, V_12], [V_21, V_22]])
        from .modal_gradient import combine
        return big_W, big_V, combine(xp, q1, q2), W, V, False

    return _smm_sp(xp, C, wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                   layer_modes, epz_conv_i_all, backside, answers,
                   mu_top=mu_top, mu_bot=mu_bot, eps_top=eps_top, eps_bot=eps_bot,
                   muz_conv_i_all=mu_all[2], lit=lit)


def smm_2d(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
           epx_conv_all, epy_conv_all, epz_conv_i_all,
           epxy_conv_all=None, epyx_conv_all=None,
           *, phi, incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
           mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None, dng=None):
    """Solve a 2D grating stack by scattering matrices — SMM twin of :func:`solve_2d`.

    Same arguments and returns as :func:`smm_1d`; ``incidence`` is the resolved complex
    amplitude pair ``(a_s, a_p)``, as in the ETM twin. ``mu`` likewise: the five permeability
    stacks, handed to the same :func:`~wavebit._core.solve_2d.layer_modes_2d`.

    Carried here rather than refused because the connecting algorithm is a choice about how
    layers are joined, not about what a layer is made of. A path that accepted ``mu`` under
    ``'ETM'`` and dropped it under ``'SMM'`` would return a plausible non-magnetic answer for a
    magnetic stack, and the two algorithms agreeing is one of the things the suite checks.

    ``eps_top`` / ``eps_bot`` are the half-spaces' permittivities, as in :func:`solve_2d`:
    optional, recovered as ``n**2/mu`` when absent, and supplied together with the ``mu``
    pair by callers that hold both.
    """
    backside = _is_backside(incidence_side)
    xp, C, kx, ky = _prepare(epx_conv_all, kx, ky)

    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    n_orders_v, n_orders_u = kx_g.shape
    n_orders = n_orders_u * n_orders_v
    I = xp.astype(xp.eye(n_orders), C)
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)
    varphi = order_azimuth(xp, ky_g, kx_g, phi)
    mu_all = (None,) * 5 if mu is None else tuple(mu)

    Kx = _diag(xp, xp.reshape(kx_g, (-1,)))
    Ky = _diag(xp, xp.reshape(ky_g, (-1,)))
    F_c, F_s = xp.cos(varphi), xp.sin(varphi)

    def layer_modes(layer_index):
        epxy_conv = O if epxy_conv_all is None else epxy_conv_all[layer_index]
        epyx_conv = O if epyx_conv_all is None else epyx_conv_all[layer_index]
        layer_mu = None if all(t is None for t in mu_all) else (
            I if mu_all[0] is None else mu_all[0][layer_index],
            I if mu_all[1] is None else mu_all[1][layer_index],
            *(None if t is None else t[layer_index] for t in mu_all[2:]))
        W, V, q = layer_modes_2d(xp, Kx, Ky, I, epx_conv_all[layer_index],
                                 epy_conv_all[layer_index], epz_conv_i_all[layer_index],
                                 epxy_conv, epyx_conv, C, mu=layer_mu,
                                 dng=None if dng is None else dng[layer_index])
        big_W, big_V = sp_modes_2d(xp, W, V, F_c, F_s, n_orders)
        return big_W, big_V, q, W, V, False

    return _smm_sp(xp, C, wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                   layer_modes, epz_conv_i_all, backside, answers,
                   mu_top=mu_top, mu_bot=mu_bot, eps_top=eps_top, eps_bot=eps_bot,
                   muz_conv_i_all=mu_all[2], lit=lit)
