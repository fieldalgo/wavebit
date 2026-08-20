"""Scattering-matrix method (SMM) — the second connecting algorithm, single-source.

The *connecting algorithm* is how per-layer eigenmodes are joined into a stack response. ETM
(:mod:`wavebit._core.solve_1d` / :mod:`wavebit._core.solve_2d`) propagates an enhanced
transmittance matrix; SMM composes per-layer scattering matrices with the Redheffer star
product. Both consume the same modes — this module imports the mode builders from the ETM
solves rather than restating the eigen-algebra — and both use the same boundary conventions,
so ``connecting_algo='SMM'`` returns the same efficiencies as ``'ETM'``, not a second opinion.

Why have it: a layer's scattering matrix depends only on that layer and the shared gap medium,
and the star product is associative. Layer responses can therefore be built once and cascaded,
reused, or reordered without touching their neighbours — where ETM's recursion carries a state
that already contains everything below the current layer. SMM is also the formulation most of
the literature states its results in, which makes it the natural cross-check on the ETM path.

What it is *not*: a rescue from transfer-matrix instability. The option is named ``'ETM'``
because it *is* the enhanced-transmittance form, which is already stable; both algorithms only ever evaluate ``exp(-k0*q*d)`` with ``Re(q) >= 0``
(principal root) and only inside bounded products. Measured on thick, strongly evanescent and
high-truncation stacks the two agree to ~1e-14 with no energy drift either way, and SMM costs
~1.1-1.4x ETM (the per-layer ``eig`` dominates both), or ~1.3-1.6x once it also recovers the
field-reconstruction state described below.

Formulation, in the conventions this codebase already uses (see :func:`solve_1d`):

* Every region's ``(W, V)`` describe its *downward* modes: a field with amplitudes ``c`` has
  primary variable ``W @ c`` and partner ``V @ c``; an upward wave flips the partner's sign.
  With that single convention the half-space scattering matrices below are derived, not
  transcribed — no per-region ``kz`` sign flip is needed.
* Layers are joined through a fictitious *gap medium* with ``eps_gap = 1 + kx^2 + ky^2``, so
  ``kz_gap == 1`` exactly and the gap's ``V`` can never be singular (the standard trick; the
  gap cancels out of the composed result whatever it is, as long as it is used everywhere).
* Incidence enters as mode amplitudes directly: ``E_s = a_s`` and ``H_p = -i*n_top*a_p``
  for the coupled path — the resolved ``(a_s, a_p)`` pair, whichever spelling the
  ``polarization`` pair — and unit primary amplitude for 1D TE/TM.
  These are the same incident amplitudes the ETM final matching solves against. The sign on
  ``H_p`` is the one ``e^{-iωt}`` requires and the one the code has always carried; this
  sentence said ``+i`` — the conjugate — until 2026-08-11, describing the formulation that
  was retired in 2026-08 rather than the code beneath it. Checked rather than reasoned:
  ``R_p / (-i*n_top)`` is the analytic amplitude to 1e-15 on both algorithms
  (``test_lossy_media.test_reflection_amplitude``), and ``+i`` gives its negative.

Field reconstruction works from either algorithm. What :mod:`wavebit._core.fields` walks is
per-layer mode amplitudes, which the ETM recursion happens to leave behind and the SMM cascade
does not — so the SMM recovers them afterwards from its own split-plane amplitudes and hands
over the same ``layer_info_list`` / ``T1`` state (see :func:`_field_state`). Same is close to
literal: the recovered blocks match the ETM's to ~1e-15 for every layer but the bottom one,
where the two differ by a basis the reconstruction never resolves. They have to match, because
the amplitudes they describe are unique given the modes, and the modes come from the same
builders. That recovery is a second pass over the stack, done only from ``answers='field'``
up, so declaring anything lower skips it outright — it is where the extra ~10-20% in the cost
quoted above goes, and it is the one rung of the ladder that buys time rather than memory.
"""
from .backend import array_namespace, as_array, check_finite, check_physical
from .conventions import boundary_kz, order_azimuth
from .enums import Answers, answers_of
from .solve_1d import (_block, _diag, _is_backside, _pack, _unit, layer_modes_1d,
                       layer_modes_1d_conical, sp_efficiencies, sp_modes_1d_conical)
from .solve_2d import layer_modes_2d, order_grid_kxy, sp_modes_2d

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
# inside: each layer's own down- and up-going mode amplitudes. Both are recovered from the
# split-plane amplitudes of the same cascade, and then handed over in the exact form
# `fields` already consumes from the ETM — see `_field_state` for that contract.

def _below_reflections(xp, I, S_layers, S_trn):
    """Reflection of everything below each split plane, in the gap basis.

    ``gamma[p]`` maps the down-going gap amplitude at split plane ``p`` to the up-going one:
    it is the ``S11`` block of ``L_p * ... * L_{n-1} * S_trn``. Plane ``p`` sits directly above
    layer ``p``, so plane ``n`` faces the substrate interface alone.

    Built bottom-up, one plane at a time. Only the ``S11`` block of each partial cascade is
    ever used, and that block costs a single inverse — a fraction of a full Redheffer star.
    """
    n = len(S_layers)
    gamma = [None] * (n + 1)
    gamma[n] = S_trn[0]
    for p in range(n - 1, -1, -1):
        s11, s12, s21, s22 = S_layers[p]
        below = gamma[p + 1]
        gamma[p] = s11 + s12 @ xp.linalg.inv(I - below @ s22) @ below @ s21
    return gamma


def _field_state(xp, I, S_ref, S_trn, S_layers, AB, c_inc):
    """Per-layer ``(a_i, b)`` blocks and the top-layer amplitude ``T1``, top-to-bottom.

    Field reconstruction walks the stack downward carrying one vector: ``T_i``, layer ``i``'s
    down-going mode amplitude at its own top plane. It asks each layer for two matrices — one
    that advances that vector to the next layer (``T_{i+1} = a_i X_i T_i``) and one that turns
    it into the layer's up-going amplitude at its bottom plane (``c2_i = b_i T_{i+1}``). The
    ETM produces both as by-products of its recursion; the SMM has to reconstruct them, and
    away from the bottom layer they come out as the *same* matrices, because the down-going
    amplitude at a plane determines everything below it uniquely.

    At the plane below layer ``i`` the gap amplitudes are ``(g, gamma[i+1] g)`` for some ``g``,
    so a layer's amplitudes there are linear in ``g``: down is ``M_i g`` and up is ``N_i g``,
    with ``M_i = (A_i + B_i gamma)/2`` and ``N_i = (B_i + A_i gamma)/2`` from the gap -> layer
    interface. Eliminating ``g`` between neighbours gives the two matrices asked for.

    The bottom layer has no layer below it and carries ``g`` itself onward, which is why its
    two blocks are the one place this state differs from the ETM's (their product, the only
    combination the reconstruction forms, still agrees). Going further and expressing that
    vector in the substrate's own basis — matching the ETM exactly — would mean inverting the
    substrate's ``V``, and that matrix is singular whenever a retained order sits exactly at
    cutoff, ``kz_bot == 0``. The gap basis is the one basis this formulation guarantees
    invertible, by its choice of ``eps_gap``; the value handed past the bottom layer is never
    read anyway.

    Reference planes matter here too: ``b_i`` is anchored at the layer *bottom*, which is what
    keeps every exponential in it decaying. Anchoring the up-going amplitude at the top plane
    instead would need ``X^-1``, and that is exactly the growing exponential both connecting
    algorithms are built to avoid.
    """
    n = len(S_layers)
    if n == 0:
        return [], None

    gamma = _below_reflections(xp, I, S_layers, S_trn)

    # Down-going gap amplitude just below the superstrate interface, closing the loop over
    # everything below it: g = S_ref21 c_inc + S_ref22 (gamma[0] g).
    g_top = xp.linalg.inv(I - S_ref[3] @ gamma[0]) @ (S_ref[2] @ c_inc)
    A_0, B_0 = AB[0]
    T1 = 0.5 * (A_0 + B_0 @ gamma[0]) @ g_top

    blocks = []
    for i in range(n):
        A, B = AB[i]
        below = gamma[i + 1]
        M = 0.5 * (A + B @ below)
        N = 0.5 * (B + A @ below)
        if i + 1 < n:
            A_next, B_next = AB[i + 1]
            M_next = 0.5 * (A_next + B_next @ below)
            blocks.append((M_next @ xp.linalg.inv(M), N @ xp.linalg.inv(M_next)))
        else:
            blocks.append((xp.linalg.inv(M), N))    # carried on in the gap basis, M_next = I
    return blocks, T1


# --------------------------------------------------------------------------------- 1D TE / TM

def smm_1d(tetm_index, wavelength, kx, n_top, n_bot, theta, thickness,
           epx_conv_all, epy_conv_all, epz_conv_i_all,
           incidence_side='top', answers=Answers.EFFICIENCY):
    """Solve a 1D grating stack by scattering matrices — SMM twin of :func:`solve_1d`.

    Same arguments and same returns as :func:`wavebit._core.solve_1d.solve_1d`, so the two are
    interchangeable behind ``connecting_algo``.

    ``incidence_side='bottom'`` costs nothing here: the cascade composes all four blocks
    whichever side is lit, so backside illumination is *reading* ``S22`` / ``S12`` where the
    top side reads ``S11`` / ``S21``, plus the incidence-medium normalization. That one-pass
    read and the ETM's mirrored-scene solve are the same algebra, not merely the same physics:
    the two interface matrices are block-swaps of each other (``_s_transmission`` returns
    ``_s_reflection``'s blocks in the swapped order, same ``A``/``B``), each layer's matrix is
    symmetric, and reversing a cascade of such factors block-swaps the product. Measured
    against the ETM backside solve at 1e-16 on efficiencies and 1e-14 on field cells
    (``tests/test_backside_incidence.py``).

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

    n_orders_x = kx.shape[0]
    k0 = 2 * _PI / wavelength
    theta = 0.0 if theta is None else theta
    I = xp.astype(xp.eye(n_orders_x), C)

    # Boundary branches come from the same place the ETM path takes them, so the two
    # algorithms answer identically.
    kz_top = boundary_kz(xp, n_top ** 2, kx ** 2)
    kz_bot = boundary_kz(xp, n_bot ** 2, kx ** 2)
    eps_bot = n_bot ** 2

    # Gap medium: eps chosen per order so kz_gap == 1, which keeps V_gap trivially invertible.
    eps_gap = 1 + kx ** 2

    if tetm_index == 0:      # TE — primary variable E_y, partner H_x
        V_top = -1j * _diag(xp, kz_top)
        V_bot = -1j * _diag(xp, kz_bot)
        V_gap = -1j * I
    elif tetm_index == 1:    # TM — primary variable H_y, partner scaled by eps
        V_top = -1j * _diag(xp, kz_top / n_top ** 2)
        V_bot = -1j * _diag(xp, kz_bot / eps_bot)
        V_gap = -1j * _diag(xp, 1 / eps_gap)
    else:
        raise ValueError(f'tetm_index must be 0 (TE) or 1 (TM), got {tetm_index!r}')
    W_top = W_bot = W_gap = I    # homogeneous half-spaces are diagonal in this basis

    # --- cascade: superstrate interface, every layer top-to-bottom, substrate interface ----
    S_ref = _s_reflection(xp, W_top, V_top, W_gap, V_gap)
    S_trn = _s_transmission(xp, W_bot, V_bot, W_gap, V_gap)
    S_layers, AB, modes = [], [], []
    S = S_ref
    for layer_index in range(_n_layers(thickness)):
        W, V, q = layer_modes_1d(xp, tetm_index, kx, I, epx_conv_all[layer_index],
                                 epy_conv_all[layer_index], epz_conv_i_all[layer_index], C)
        d = thickness[layer_index]
        X = _diag(xp, xp.exp(-k0 * q * d))
        A, B = _ab_matrices(xp, W, V, W_gap, V_gap)
        S_layer = _s_layer(xp, A, B, X)
        S = redheffer_star(xp, I, S, S_layer)
        if keep_field:
            S_layers.append(S_layer)
            AB.append((A, B))
            modes.append([epz_conv_i_all[layer_index], W, V, q, d])
    S = redheffer_star(xp, I, S, S_trn)

    # --- incidence -> amplitudes -> efficiency --------------------------------------------
    # Incidence-relative labels: reflection returns into the incidence medium, transmission
    # leaves through the far side. For the backside these are the substrate-port outputs, so
    # the reads swap to S22 / S12; everything after is the top-side arithmetic with the two
    # half-spaces in incidence-relative roles.
    delta = xp.reshape(xp.astype(_unit(xp, n_orders_x, n_orders_x // 2), C), (n_orders_x, 1))
    R = xp.reshape((S[3] if backside else S[0]) @ delta, (1, n_orders_x))
    Tt = xp.reshape((S[1] if backside else S[2]) @ delta, (1, n_orders_x))

    kz_ref, kz_trn = (kz_bot, kz_top) if backside else (kz_top, kz_bot)
    n_ref, n_trn = (n_bot, n_top) if backside else (n_top, n_bot)

    cos_t = xp.cos(xp.asarray(theta, dtype=C))
    de_ri = xp.real(R * xp.conj(R) * xp.real(kz_ref / (n_ref * cos_t)))
    if tetm_index == 0:
        de_ti = xp.real(Tt * xp.conj(Tt) * xp.real(kz_trn / (n_ref * cos_t)))
    else:
        # Re(kz/eps), not Re(conj(kz)/eps): for TM the downward flux is Re(E_x H_y*) with E_x
        # proportional to (kz/eps) H_y. The two agree for a real exit index, which is why a
        # stray conjugate here survived every lossless test; against an absorbing substrate it
        # returned negative transmitted power. See `conventions`: this codebase ran on a
        # conjugated problem until 2026-08, and this was the last boundary still on the far
        # side of that seam.
        de_ti = xp.real(Tt * xp.conj(Tt)
                        * xp.real(kz_trn / n_trn ** 2 / (cos_t / n_ref)))

    check_finite(xp, de_ri=de_ri, de_ti=de_ti)
    check_physical(xp, de_ri, de_ti, n_ref)

    if not answers.covers(Answers.AMPLITUDE):
        return de_ri, de_ti

    # Same Fresnel conversion as the ETM pure path: reflection primaries are already the
    # amplitude; the TM transmission converts H back to E across the index step.
    zeros_amp = xp.zeros_like(R)
    zeros_eff = xp.zeros_like(de_ri)
    if tetm_index == 0:
        res = _pack(R, zeros_amp, Tt, zeros_amp, de_ri, zeros_eff, de_ti, zeros_eff)
    else:
        res = _pack(zeros_amp, R, zeros_amp, Tt * (n_ref / n_trn),
                    zeros_eff, de_ri, zeros_eff, de_ti)

    if not keep_field:
        return {'res': res}

    if backside:
        # The mirrored scene's own cascade pieces, assembled from what this pass already
        # built: the layer matrices are symmetric and the interface matrices come from the
        # same A/B, so only the two half-space interfaces are formed anew. The recovered
        # state then describes the mirrored scene — the same frame the ETM backside solve
        # leaves behind — and field reconstruction un-mirrors once, at the cell level.
        S_ref_m = _s_reflection(xp, W_bot, V_bot, W_gap, V_gap)
        S_trn_m = _s_transmission(xp, W_top, V_top, W_gap, V_gap)
        blocks, T1 = _field_state(xp, I, S_ref_m, S_trn_m, S_layers[::-1], AB[::-1], delta)
        modes = modes[::-1]
    else:
        blocks, T1 = _field_state(xp, I, S_ref, S_trn, S_layers, AB, delta)
    # Bottom-to-top, the order field reconstruction walks in reverse (as the ETM appends it).
    layer_info_list = [m + list(b) for m, b in zip(modes, blocks)][::-1]
    if T1 is not None:
        T1 = xp.reshape(T1, (n_orders_x,))    # the 1D path carries T1 as a plain vector
    return {'res': res, 'layer_info_list': layer_info_list, 'T1': T1}


# ------------------------------------------------------------------- coupled s/p (conical, 2D)

def _smm_sp(xp, C, wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
            layer_modes, epz_conv_i_all, backside, answers, mu_top=1.0, mu_bot=1.0,
            eps_top=None, eps_bot=None, muz_conv_i_all=None):
    """Shared SMM cascade + matching for the coupled s/p paths (conical and 2D).

    ``layer_modes(layer_index)`` returns that layer's ``(big_W, big_V, q, W, V)``: the mode
    matrices in the s/p boundary basis that the cascade joins layers through, then the raw
    (unrotated) pair that field reconstruction expands the same amplitudes with. Everything
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

    ``incidence`` is the resolved amplitude pair ``(a_s, a_p)`` at unit flux — see
    :func:`~wavebit._core.conventions.incident_sp_amplitudes`.

    ``answers`` is the rung this solve was declared for; it decides how many incidence
    columns the cascade is asked about (one, or three once the pure-TE/TM variants are
    wanted) and whether the per-layer pieces are kept for :func:`_field_state` to walk.
    """
    answers = answers_of(answers)
    keep_amplitude = answers.covers(Answers.AMPLITUDE)
    keep_field = answers.covers(Answers.FIELD)
    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    n_orders_y, n_orders_x = kx_g.shape
    n_orders = n_orders_x * n_orders_y
    k0 = 2 * _PI / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(n_orders), C)
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)
    big_I = xp.astype(xp.eye(2 * n_orders), C)

    k_par_sq = kx_g ** 2 + ky_g ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq), (-1,))
    # `n` stays the refractive index and `eps` carries the `mu`, so the two are recoverable
    # from each other -- but only up to rounding, since the index reached this call through a
    # square root. The engine holds both numbers and passes them; a caller that holds only an
    # index leaves these `None` and gets the recovery, which is what this always did.
    if eps_top is None:
        eps_top = n_top ** 2 / mu_top
    if eps_bot is None:
        eps_bot = n_bot ** 2 / mu_bot
    eps_gap = xp.reshape(1 + k_par_sq, (-1,))                 # kz_gap == 1 by construction

    Kz_top = _diag(xp, kz_top)
    Kz_bot = _diag(xp, kz_bot)

    # Half-space and gap modes in the (E_s, F_p) / (H_s, H_p) boundary basis — the same form
    # the ETM substrate seed (big_F, big_G) has, and the same two admittances: `kz/eps` on the
    # magnetic (p) amplitude, `kz/mu` on the electric (s) one.
    #
    # The **gap is left non-magnetic**, deliberately. It is not a medium in the scene; it is a
    # fictitious layer the cascade inserts so every interface is computed the same way, and it
    # cancels out of the result. Giving a bookkeeping device a material property would be a
    # category error that happens to cancel, which is worse than one that does not.
    W_top = _block(xp, [[I, O], [O, -1j * Kz_top / eps_top]])
    V_top = _block(xp, [[-1j * Kz_top / mu_top, O], [O, I]])
    W_bot = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])
    V_bot = _block(xp, [[-1j * Kz_bot / mu_bot, O], [O, I]])
    W_gap = _block(xp, [[I, O], [O, -1j * _diag(xp, 1 / eps_gap)]])
    V_gap = _block(xp, [[-1j * I, O], [O, I]])

    S_ref = _s_reflection(xp, W_top, V_top, W_gap, V_gap)
    S_trn = _s_transmission(xp, W_bot, V_bot, W_gap, V_gap)
    S_layers, AB, modes, muz_conv = [], [], [], []
    S = S_ref
    for layer_index in range(_n_layers(thickness)):
        big_W, big_V, q, W, V = layer_modes(layer_index)
        d = thickness[layer_index]
        big_X = _diag(xp, xp.exp(-k0 * q * d))
        A, B = _ab_matrices(xp, big_W, big_V, W_gap, V_gap)
        S_layer = _s_layer(xp, A, B, big_X)
        S = redheffer_star(xp, big_I, S, S_layer)
        if keep_field:
            S_layers.append(S_layer)
            AB.append((A, B))
            modes.append([epz_conv_i_all[layer_index], W, V, q, d])
            muz_conv.append(None if muz_conv_i_all is None else muz_conv_i_all[layer_index])
    S = redheffer_star(xp, big_I, S, S_trn)

    kz_ref, kz_trn = (kz_bot, kz_top) if backside else (kz_top, kz_bot)
    n_ref, n_trn = (n_bot, n_top) if backside else (n_top, n_bot)
    mu_ref, mu_trn = (mu_bot, mu_top) if backside else (mu_top, mu_bot)

    # --- incidence ------------------------------------------------------------------------
    # Amplitudes are (E_s, H_p) by construction of W/V above, so the incident column is the
    # incident field itself: E_s = a_s, H_p = -i*n_ref*a_p in the incidence medium — with
    # (a_s, a_p) as resolved at the boundary — one form arrives here, and it is the same
    # amplitudes the ETM final matching solves against (see conventions.incident_sp_amplitudes).
    delta = xp.reshape(xp.astype(_unit(xp, n_orders, n_orders // 2), C), (n_orders, 1))

    def incident(a_s, a_p):
        # `n_ref/mu_ref`, not `n_ref`: `a_s` and `a_p` are defined at equal incident flux,
        # and the s flux carries `kz/mu`. Same scale the ETM final matching uses.
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
        """(n_orders, n_inc) amplitude columns -> (n_inc, n_orders_y, n_orders_x)."""
        return xp.reshape(xp.matrix_transpose(M), (n_inc, n_orders_y, n_orders_x))

    # The reflected p amplitude is reported as its magnetic field, which points the other way
    # for the upward wave; s and the transmitted pair need no such flip.
    R_s = as_orders(c_ref[:n_orders, :])
    R_p = as_orders(-c_ref[n_orders:, :])
    T_s = as_orders(c_trn[:n_orders, :])
    T_p = as_orders(c_trn[n_orders:, :])

    kz_ref_2d = xp.reshape(kz_ref, (n_orders_y, n_orders_x))
    kz_trn_2d = xp.reshape(kz_trn, (n_orders_y, n_orders_x))
    denom = n_ref * xp.cos(xp.asarray(theta, dtype=C)) / mu_ref
    eff = sp_efficiencies(xp, kz_ref_2d, kz_trn_2d, denom, n_ref, n_trn, R_s, R_p, T_s, T_p,
                          mu_top=mu_ref, mu_bot=mu_trn)

    check_finite(xp, de_ri=eff[0][0] + eff[1][0], de_ti=eff[2][0] + eff[3][0])
    check_physical(xp, eff[0][0] + eff[1][0], eff[2][0] + eff[3][0], n_ref)

    if not keep_amplitude:
        return eff[0][0] + eff[1][0], eff[2][0] + eff[3][0]

    # Coupled-path Fresnel conversion, identical to the ETM final matching's.
    amps = (R_s, R_p / (-1j * n_ref / mu_ref), T_s, T_p / (-1j * n_trn / mu_trn))
    packed = [_pack(*(a[i] for a in amps), *(e[i] for e in eff)) for i in range(3)]

    if not keep_field:
        return {'res': packed[0], 'res_te_inc': packed[1], 'res_tm_inc': packed[2]}

    if backside:
        # Mirrored-scene field state, same construction as `smm_1d`'s backside branch.
        S_ref_m = _s_reflection(xp, W_bot, V_bot, W_gap, V_gap)
        S_trn_m = _s_transmission(xp, W_top, V_top, W_gap, V_gap)
        blocks, T1 = _field_state(xp, big_I, S_ref_m, S_trn_m,
                                  S_layers[::-1], AB[::-1], c_inc)
        modes = modes[::-1]
        muz_conv = muz_conv[::-1]
    else:
        blocks, T1 = _field_state(xp, big_I, S_ref, S_trn, S_layers, AB, c_inc)
    # `muz_conv_i` last, after the transfer blocks, so slots 0..6 stay where the ETM twin
    # puts them: `tests/test_scattering_core.py` compares the two algorithms' `A_i` / `B` by index,
    # and the legacy goldens name the first seven.
    layer_info_list = [m + list(b) + [z]
                       for m, b, z in zip(modes, blocks, muz_conv)][::-1]
    if T1 is not None:
        # One column per incidence variant (psi, TE, TM), as the ETM stacks them.
        T1 = xp.stack([T1[:, i:i + 1] for i in range(n_inc)])
    return {'res': packed[0], 'res_te_inc': packed[1], 'res_tm_inc': packed[2],
            'layer_info_list': layer_info_list, 'T1': T1}


def smm_1d_conical(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                   epx_conv_all, epy_conv_all, epz_conv_i_all,
                   *, phi, incidence_side='top', answers=Answers.EFFICIENCY):
    """Solve a conical 1D grating by scattering matrices — SMM twin of :func:`solve_1d_conical`.

    Same arguments and returns as :func:`smm_1d`; ``polarization`` is the complex amplitude
    pair ``(a_s, a_p)``, resolved at the boundary as in the ETM twin.
    """
    backside = _is_backside(incidence_side)
    xp, C, kx, ky = _prepare(epx_conv_all, kx, ky)

    n_orders_x = kx.shape[0]
    I = xp.astype(xp.eye(n_orders_x), C)
    ky_col = xp.reshape(ky, (-1, 1))
    varphi = order_azimuth(xp, ky_col, kx, phi)

    Kx = _diag(xp, kx)
    ky_vec = ky[0] * xp.astype(xp.ones((n_orders_x,)), C)      # n_orders_y == 1: Ky = diag(ky0 * 1)
    Ky = _diag(xp, ky_vec)
    Kx2 = _diag(xp, kx * kx)
    Ky2 = _diag(xp, ky_vec * ky_vec)
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

    def layer_modes(layer_index):
        W1, W2, q1, q2, V_11, V_12, V_21, V_22 = layer_modes_1d_conical(
            xp, Kx, Ky, Kx2, Ky2, I, epx_conv_all[layer_index],
            epy_conv_all[layer_index], epz_conv_i_all[layer_index], C)
        big_W, big_V = sp_modes_1d_conical(xp, F_c, F_s, W1, W2, V_11, V_12, V_21, V_22)
        W = _block(xp, [[W1, W2]])
        V = _block(xp, [[V_11, V_12], [V_21, V_22]])
        return big_W, big_V, xp.concat([q1, q2]), W, V

    return _smm_sp(xp, C, wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                   layer_modes, epz_conv_i_all, backside, answers)


def smm_2d(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
           epx_conv_all, epy_conv_all, epz_conv_i_all,
           epxy_conv_all=None, epyx_conv_all=None,
           *, phi, incidence_side='top', answers=Answers.EFFICIENCY,
           mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None):
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
    n_orders_y, n_orders_x = kx_g.shape
    n_orders = n_orders_x * n_orders_y
    I = xp.astype(xp.eye(n_orders), C)
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)
    varphi = order_azimuth(xp, ky_g, kx_g, phi)
    mu_all = (None,) * 5 if mu is None else tuple(mu)

    Kx = _diag(xp, xp.reshape(kx_g, (-1,)))
    Ky = _diag(xp, xp.reshape(ky_g, (-1,)))
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

    def layer_modes(layer_index):
        epxy_conv = O if epxy_conv_all is None else epxy_conv_all[layer_index]
        epyx_conv = O if epyx_conv_all is None else epyx_conv_all[layer_index]
        layer_mu = None if all(t is None for t in mu_all) else (
            I if mu_all[0] is None else mu_all[0][layer_index],
            I if mu_all[1] is None else mu_all[1][layer_index],
            *(None if t is None else t[layer_index] for t in mu_all[2:]))
        W, V, q = layer_modes_2d(xp, Kx, Ky, I, epx_conv_all[layer_index],
                                 epy_conv_all[layer_index], epz_conv_i_all[layer_index],
                                 epxy_conv, epyx_conv, C, mu=layer_mu)
        big_W, big_V = sp_modes_2d(xp, W, V, F_c, F_s, n_orders)
        return big_W, big_V, q, W, V

    return _smm_sp(xp, C, wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
                   layer_modes, epz_conv_i_all, backside, answers,
                   mu_top=mu_top, mu_bot=mu_bot, eps_top=eps_top, eps_bot=eps_bot,
                   muz_conv_i_all=mu_all[2])
