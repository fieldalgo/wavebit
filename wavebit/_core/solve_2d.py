"""Single-source 2D RCWA solve — Phase 3c of the rewrite.

As in ``solve_1d``, private ``top``/``bottom`` port names mean the start/end of the
positive-z layer list. The public engine maps physical media and illumination onto them.

Port of the triplicated ``transfer_2d_1..4`` chain into one backend-agnostic solve against the
Array-API namespace, sharing the boundary/branch conventions and the final s/p matching with
:mod:`wavebit._core.solve_1d`. The only per-backend primitive is
:func:`wavebit._core.backend.eig`.

The 2D layer couples both polarizations *and* both in-plane directions, so per-layer modes and
the boundary system are ``2*n_orders`` block matrices (``n_orders = n_orders_u*n_orders_v``).
Optional off-axis convolution blocks (``epxy`` / ``epyx``, produced by the NVF factorization;
``None`` = zeros) enter the mode operator. ``transfer_2d_1`` and ``transfer_2d_4`` are identical
to the conical versions, so this module reuses ``solve_1d``'s boundary init and
``final_matching`` verbatim.
"""
from .backend import (_concrete, _readable, array_namespace, as_array, carries_autodiff,
                      check_mode_cutoff, checked_float, division_at_a_cutoff, eig)
from .conventions import boundary_kz, layer_root, order_azimuth
from .enums import Answers, answers_of
from .solve_1d import _block, _diag, _is_backside, _mirrored_scene, final_matching

# Engage the coupled-cluster rebuild (see `_rebuild_grazing_cluster`) below this
# min|w|/max|w| ratio. Well above `backend._CUTOFF_REL` (the warn threshold) on purpose,
# for two measured reasons: the product-eigvec construction degrades as ~eps/ratio through
# the whole band (2e-6 at 1e-12, per ``docs/rayleigh-points.md`` §3) while the rebuild is
# exact everywhere in it; and `solve`'s default detune step (1e-12 relative) lands a
# rescued solve at ratio ~1e-11 — with a 1e-12 gate the retry fell back into the shipped
# construction's degraded band and returned 1e-6..8e-6 errors on the harder graze stacks,
# with 1e-9 it lands in the rebuild's regime and matches the extrapolated exact-point
# oracle to its own uncertainty. Above the gate the ordinary product modes are returned;
# a random non-grazing solve reaches ratio 1e-9 with probability ~1e-8
# per solve (an order within 4e-9 of a cutoff), measured as 0 engagements over the
# 40-config validation sweep.
_COUPLED_REL = 1e-9

# Below this modal rate (relative to sqrt(max|w|), the layer's largest rate) a rebuilt
# cluster mode is numerically indistinguishable from the exactly-defective ``{1, z}`` pair,
# and no exponential mode basis represents the layer: the intrinsic error of any such basis
# is ~eps/(k0*d*|mu|). Measured on exact-graze stacks: |mu| = 2e-9 solves to 6.5e-9, while
# |mu| ~ 5e-16 is wrong at 1e-4..1e-3 across every basis choice tried (eigvec pairing,
# volume-maximal selection, |mu| floors — the floor repeats the epsilon-floor dead end of
# ``docs/rayleigh-points.md`` §5 in miniature: any q inconsistent with its own mode pair is
# amplified by exactly the conditioning that consistency lets cancel). 1e-10 puts the
# representable side at <= ~1e-6 for k0*d >~ 0.4; below it the only honest outcome is the
# loud refusal, which `solve`'s default ladder then retries off the condition.
_DEFECTIVE_REL = 1e-10


def order_grid_kxy(xp, kx, ky):
    """``kx`` / ``ky`` on the ``(n_orders_v, n_orders_u)`` order grid, from either spelling.

    Two ways in, because a lattice basis decides which one is even expressible:

    * **1-D**, lengths ``(n_orders_u,)`` and ``(n_orders_v,)`` — the separable form an
      axis-aligned cell
      admits, where ``kx`` depends only on ``m`` and ``ky`` only on ``n``. Broadcast to the
      grid here, which is what both solves used to do inline.
    * **2-D**, ``(n_orders_v, n_orders_u)`` each — the per-order Cartesian components, which is
      the only
      form a skewed cell has: with ``G = m*b1 + n*b2`` non-orthogonal, *both* components
      depend on *both* indices and no pair of 1-D vectors can hold them. See
      :func:`wavebit._core.lattice.order_kxy`.

    Everything downstream reads the grid, so the two spellings converge here and the algebra
    below never branches. The 1-D route broadcasts exactly as before, element for element.
    """
    if kx.ndim == 2:
        return kx, ky
    n_orders_u, n_orders_v = kx.shape[0], ky.shape[0]
    return (xp.broadcast_to(kx[None, :], (n_orders_v, n_orders_u)),
            xp.broadcast_to(xp.reshape(ky, (-1, 1)), (n_orders_v, n_orders_u)))


def _rebuild_grazing_cluster(xp, w, W, V, q, Omega_L, Omega_R, k0_d=None):
    """Re-derive the grazing eigenvalue cluster from the coupled first-order problem.

    Returns ``(W, V, q)`` as NumPy arrays with the cluster columns replaced, ``None`` to
    bail (caller keeps the shipped construction and its warning), or raises
    ``FloatingPointError`` when the cluster is numerically defective.

    Why the shipped construction cannot be repaired in place: the cluster eigvecs of the
    product ``Omega2 = Omega_L @ Omega_R`` are determined by round-off (the eigenvalues are
    separated by ~eps), so no choice of ``V`` makes the pair ``(W_j, V_j, q_j)`` satisfy
    both first-order relations ``Omega_R W = V q`` and ``Omega_L V = W q`` — and the
    interface algebra inverts ``big_W`` and ``big_V`` separately, so it amplifies whichever
    relation is broken by the cluster's 1/q conditioning. What does hold to eps is the
    *invariant subspace pair*: ``S_E`` (from the product's cluster, gap O(1) to the rest of
    the spectrum) and ``S_H`` (same for the swapped product ``Omega_R @ Omega_L``). Restrict
    the factors to those subspaces —

        ``R_r = Q_H^H Omega_R Q_E``,   ``L_r = Q_E^H Omega_L Q_H``      (k x k)

    — and take eigenpairs of the small first-order matrix ``[[0, L_r], [R_r, 0]]``: each
    eigenvector ``[gE; gH]`` with eigenvalue ``mu`` satisfies *both* relations to
    eps*||M_r|| by the standard eigenpair residual, so ``(Q_E gE, Q_H gH, mu)`` is a mode of
    a problem within eps of the requested one. Measured on the exact-graze repro stacks:
    the exact point lands inside the two-sided sqrt-extrapolated oracle bracket (~6e-9
    wide), where the shipped construction was off by 4.5e-3 .. 4.1e0.

    The eigenvectors come in ``+-mu`` pairs sharing one field pattern (one physical chain
    each), and near-degenerate chains make ``eig`` return an arbitrary — possibly nearly
    parallel — basis, so the k modes are chosen greedily to keep the E-parts *and* H-parts
    jointly independent (both matrices are inverted downstream); a ``+-`` twin scores ~0 on
    both sides once its partner is in and is never double-picked. Per column the member is
    branch-fixed to the root :func:`~wavebit._core.conventions.layer_root` takes for the rest
    of the layer (``k0_d`` is passed through, so an ETM layer keeps one rule), and kept
    at ``eig``'s joint normalization: a mode may then carry a tiny E-part (a pure-H graze)
    or a tiny H-part (the classic E-graze) but never both, which is the structure the
    interface inversions tolerate — normalizing either part to unit instead was measured to
    plant an inconsistent O(1/eps) column and lose three digits.

    Everything here runs in NumPy: the caller only enters when the eigenvalues were
    concretely readable, which is also the regime where no autograd graph is at stake.
    """
    import numpy as np

    w, W, V, q, Omega_L, Omega_R = (np.asarray(a) for a in (w, W, V, q, Omega_L, Omega_R))
    aw = np.abs(w)
    J = np.where(aw < _COUPLED_REL * aw.max())[0]
    wh, Vh = np.linalg.eig(Omega_R @ Omega_L)
    Jh = np.where(np.abs(wh) < _COUPLED_REL * np.abs(wh).max())[0]
    if len(Jh) != len(J):
        return None                              # structure not understood — detect, don't guess
    k = len(J)
    Q_E = np.linalg.qr(W[:, J])[0]
    Q_H = np.linalg.qr(Vh[:, Jh])[0]
    R_r = Q_H.conj().T @ (Omega_R @ Q_E)
    L_r = Q_E.conj().T @ (Omega_L @ Q_H)
    Z = np.zeros((k, k), dtype=W.dtype)
    mu_e, g = np.linalg.eig(np.block([[Z, L_r], [R_r, Z]]))
    g = g / np.linalg.norm(g, axis=0, keepdims=True)

    sel = []
    residE, residH = g[:k, :].copy(), g[k:, :].copy()
    for _ in range(k):
        score = (np.linalg.norm(residE, axis=0) ** 2
                 + np.linalg.norm(residH, axis=0) ** 2)
        score[sel] = -1.0
        pick = int(np.argmax(score))
        sel.append(pick)
        for resid in (residE, residH):
            col = resid[:, pick]
            n = np.linalg.norm(col)
            if n > 1e-150:
                d = col / n
                resid -= d[:, None] * (d.conj() @ resid)[None, :]

    if np.abs(mu_e[sel]).min() < _DEFECTIVE_REL * np.sqrt(aw.max()):
        raise FloatingPointError(
            'a 2D layer eigenmode pair is numerically defective at a grazing condition (an '
            'order sitting on the layer light line, so its q**2 is 0): the coupled mode '
            'rates split by less than 1e-10 of the layer modal scale, and the exponential '
            'mode basis this formulation uses does not exist there at working precision — '
            'the true limit basis is {1, z}, and any exponential spelling returns quietly '
            'wrong efficiencies (measured 1e-4 up to 1e0). Move off the condition: one ULP '
            'of wavelength, period or angle is enough, and solve(auto_detune=True), '
            'the default, does that retry for you.')

    Wc = np.empty((W.shape[0], k), dtype=W.dtype)
    Vc = np.empty_like(Wc)
    qc = np.empty(k, dtype=W.dtype)
    # Read as a number, the way the rest of this function reads its inputs: a thickness can be a
    # torch tensor that requires grad while the operator does not, and then this is entered.
    k0_d = None if k0_d is None else checked_float(k0_d)
    for i, col in enumerate(sel):
        mu_i = mu_e[col]
        q_i = layer_root(np, np.asarray([mu_i ** 2]), k0_d)[0]
        gE, gH = g[:k, col], g[k:, col]
        if abs(q_i - mu_i) > abs(q_i + mu_i):
            gH = -gH                             # took the -q member; its H part flips
        Wc[:, i] = Q_E @ gE
        Vc[:, i] = Q_H @ gH
        qc[i] = q_i

    W, V, q = W.copy(), V.copy(), q.copy()
    W[:, J], V[:, J], q[J] = Wc, Vc, qc
    return W, V, q


def layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i, epxy_conv, epyx_conv, C,
                   mu=None, k0_d=None):
    """Raw (unrotated) eigenmodes ``(W, V, q)`` of one 2D layer.

    ``W`` is the ``[Ex; Ey]`` mode matrix and ``q`` the modal decay constants; without ``k0_d``
    the root keeps ``Re(q) >= 0`` so ``exp(-k0*q*d)`` never grows. Shared by the ETM
    (:func:`solve_2d`) and SMM (:mod:`wavebit._core.scattering`) connecting algorithms, which
    differ only in which of ``±q`` they label "down": ETM passes the layer's optical thickness
    ``k0_d`` (:func:`~wavebit._core.conventions.layer_root`, as in
    :func:`~wavebit._core.solve_1d.layer_modes_1d`).

    The two blocks are the factors of ``d2E/dz2 = P Q E``: ``Omega2_LR`` is the product written
    out, ``Omega_R`` the right factor kept so ``V`` can recover ``H``. **They must carry the
    off-diagonal pair in opposite orders** — ``Omega2_LR`` pairs ``epyx`` with the first column
    and ``epxy`` with the second, ``Omega_R`` the reverse. That is not a symmetry to tidy up: it
    is what makes ``Omega_L = Omega2_LR @ inv(Omega_R)`` come out as geometry alone,
    ``[[-Kx Ky epz_i, Kx^2 epz_i - 1], [-Ky^2 epz_i + 1, Kx Ky epz_i]]``, for an arbitrary
    tensor at any incidence. Align them and an asymmetric tensor takes its modes from one medium
    and its H-field from that medium's transpose, silently: every tensor the library builds is
    symmetric, so nothing in a normal run can tell the difference. See
    ``docs/off-diagonal-slot-convention.md`` §4, and ``tests/test_anisotropic_berreman.py`` /
    ``tests/references/test_reticolo_anisotropic_golden.py`` for what pins it.

    ``mu`` is an optional ``(mux_conv, muy_conv, muz_conv_i, muxy_conv, muyx_conv)``
    permeability tuple in **the same five slots and the same convention as the
    permittivity** — ``muz_conv_i`` inverted, the off-diagonals negated. Left ``None`` the
    layer is non-magnetic and the same non-magnetic Maxwell equations apply. ``Kx`` and
    ``Ky`` are diagonal wavevector matrices; their action on the longitudinal inverse is
    evaluated by row and column scaling, including during a grazing-cluster rebuild.

    Where it enters is a duality, not a patch. Under ``(E, H, eps, mu) -> (H, -E, mu, eps)``
    the two curl equations map onto each other, so ``Omega_L`` — which eliminates ``E_z``
    through ``eps_zz`` — carries the *transverse* ``mu``, and ``Omega_R`` — which eliminates
    ``H_z`` through ``mu_zz`` — carries the transverse ``eps``. Concretely the four blocks
    below each acquire one slot, uniformly, and ``Omega_R``'s three geometry products gain
    ``muz_conv_i`` between their ``K`` factors::

        B  = Kx epz_i Kx - muy          Cc = Kx epz_i Ky - muyx
        D  = Ky epz_i Ky - mux          E  = Ky epz_i Kx - muxy

    (Abbreviated on both sides — ``epz_i`` is ``epz_conv_i`` and ``muy`` is ``muy_conv``.
    Every one of these eight operands is a convolution matrix; the permeability is built by
    the same builder as the permittivity, which is what the shared ``_conv`` says.)

    Note the **cross** pairing: ``B`` takes ``mu_yy`` and ``D`` takes ``mu_xx``, not the other
    way round. ``docs/anisotropic-materials.md`` §5 pointed at the two ``- I`` terms and stopped
    there, which would have got the isotropic case right and silently transposed a uniaxial
    ``mu``. The placement was settled by requiring the derived ``Omega_L @ Omega_R`` to
    reproduce the shipped ``Omega2_LR`` term for term at ``mu = 1``.
    """
    mux_conv, muy_conv, muz_conv_i, muxy_conv, muyx_conv = (
        (I, I, None, None, None) if mu is None else mu)
    kx, ky = xp.linalg.diagonal(Kx), xp.linalg.diagonal(Ky)

    def left_blocks():
        # diag(k) A diag(l) scales rows and columns. Keep that multiplication
        # order, but do not spend eight dense matrix products on diagonal factors.
        B = (kx[:, None] * epz_conv_i) * kx[None, :] - muy_conv
        D = (ky[:, None] * epz_conv_i) * ky[None, :] - mux_conv
        Cc = (kx[:, None] * epz_conv_i) * ky[None, :]
        E = (ky[:, None] * epz_conv_i) * kx[None, :]
        if muyx_conv is not None:
            Cc = Cc - muyx_conv
        if muxy_conv is not None:
            E = E - muxy_conv
        return B, D, Cc, E

    # `Omega_R`'s three geometry products carry `muz_conv_i` between the K factors, exactly as
    # `epz_conv_i` sits between them in `Omega_L`. Skipped entirely when non-magnetic rather
    # than multiplying an identity through, so the arithmetic stays the operations it was.
    def right_factor():
        if muz_conv_i is None:
            KxKy, KxKx, KyKy, KyKx = Kx @ Ky, Kx @ Kx, Ky @ Ky, Ky @ Kx
        else:
            KxKy, KxKx = Kx @ muz_conv_i @ Ky, Kx @ muz_conv_i @ Kx
            KyKy, KyKx = Ky @ muz_conv_i @ Ky, Ky @ muz_conv_i @ Kx
        return _block(xp, [
            [-KxKy + epyx_conv, KxKx - epy_conv],
            [epx_conv - KyKy, KyKx - epxy_conv],
        ])

    B, D, Cc, E = left_blocks()
    Omega_R = None
    if mu is None:
        # The hand-expanded product. Its four bare ``K @ K`` terms are not geometry that
        # survived the multiplication untouched — they are what two *other* terms collapse to
        # when the K factors commute past a `muz_conv_i` of one. Written out, the (0,0) block is
        #   Cc Kx muz_i Ky - B Ky muz_i Ky + B epx - Cc epyx,          (abbreviated, as above)
        # where the first pair leaves `muy Ky muz_i Ky + muyx Kx muz_i Ky` behind, which is
        # `Ky @ Ky` only when `muy = 1`, `muyx = 0` and `muz_i = 1`. So this form cannot be
        # reused with a permeability, and reusing it is not a small error: it costs energy
        # conservation outright on a gyromagnetic layer (R + T came out 1.035).
        Omega2_LR = _block(xp, [
            [Ky @ Ky + B @ epx_conv - Cc @ epyx_conv, Cc @ epy_conv - Kx @ Ky - B @ epxy_conv],
            [E @ epx_conv - Ky @ Kx - D @ epyx_conv, Kx @ Kx + D @ epy_conv - E @ epxy_conv],
        ])
    else:
        Omega_R = right_factor()
        Omega2_LR = _block(xp, [[-Cc, B], [-D, E]]) @ Omega_R

    # The eigensolve needs only the product. Reconstruct the left factor
    # later if the existing grazing guard requires it; ordinary modes do
    # not retain its four dense blocks or an unused right factor here.
    del B, D, Cc, E
    from .modal_gradient import _stop
    differentiated = carries_autodiff(Omega2_LR)
    w, W = eig(_stop(Omega2_LR))
    if not differentiated:
        # Only autodiff needs the original operator after the eigensolve.
        # Preserve the eager forward path's early release of this dense matrix.
        del Omega2_LR
    if Omega_R is None:
        Omega_R = right_factor()
    q = layer_root(xp, xp.astype(w, C), k0_d)
    V = Omega_R @ W
    with division_at_a_cutoff():                 # q == 0 on a light line; rebuilt or refused below
        V = V / q

    # At an exact Rayleigh point (an order on a layer's light line, q -> 0) this V is wrong
    # with no arithmetic to blame: the grazing cluster's eigvecs of the *product* Omega2 are
    # eps-unstable with respect to the factor split, so even the bit-exact `Omega_R @ W`
    # (measured: it agrees with 200-bit arithmetic to 1e-22 on the repro stack) yields a
    # column that satisfies `Omega_R W = V q` while violating `Omega_L V = W q` at order 1 —
    # the layer is then the mode set of *no* nearby problem, and the answers were quietly
    # wrong at 1e-5 (band edge) up to 1e0 (exact point). The rebuild below re-derives the
    # cluster from the coupled first-order problem, which restores both relations to eps;
    # where the cluster is numerically defective (the `{1, z}` regime) it refuses loudly
    # instead. Traced or otherwise unreadable values keep the old detect-and-warn contract.
    aw = xp.abs(_readable(w, xp))
    grazing = _concrete(xp.min(aw) < _COUPLED_REL * xp.max(aw))
    if grazing and not differentiated:
        B, D, Cc, E = left_blocks()
        Omega_L = _block(xp, [[-Cc, B], [-D, E]])
        del B, D, Cc, E
        try:
            rebuilt = _rebuild_grazing_cluster(xp, w, W, V, q, Omega_L, Omega_R, k0_d)
        except FloatingPointError:
            raise                                # the defective-cluster refusal, on purpose
        except Exception:
            # Concrete but not materializable to NumPy (a torch tensor carrying grad, an
            # un-jitted jax.grad tracer): keep the detect-and-warn contract; the default
            # auto_detune ladder then re-solves off the point with the graph intact.
            rebuilt = None
        if rebuilt is not None:
            return tuple(xp.astype(xp.asarray(a), C) for a in rebuilt)
    check_mode_cutoff(xp, w)
    if differentiated:
        from .modal_gradient import generator, inverse
        q = generator(xp, Omega2_LR, w, W, q, square_root=True)
        V = Omega_R @ W @ inverse(xp, q)
    return W, V, q


def sp_modes_2d(xp, W, V, F_c, F_s, n_orders):
    """Rotate 2D layer modes into the s/p boundary basis: ``(big_W, big_V)``.

    ``big_W`` maps mode amplitudes to the boundary variables ``(E_s, F_p)`` and ``big_V`` to
    ``(H_s, H_p)`` — the pair the substrate seed ``big_F`` / ``big_G`` is written in.
    ``F_c`` and ``F_s`` are per-order cosine/sine vectors or their diagonal matrices.
    Rotation only scales rows, so it needs neither dense products nor dense storage
    for those diagonals. The matrix spelling remains available to the full-tensor seed.
    """
    c = (F_c if F_c.ndim == 1 else xp.linalg.diagonal(F_c))[:, None]
    s = (F_s if F_s.ndim == 1 else xp.linalg.diagonal(F_s))[:, None]
    W_11 = W[:n_orders, :n_orders]
    W_12 = W[:n_orders, n_orders:]
    W_21 = W[n_orders:, :n_orders]
    W_22 = W[n_orders:, n_orders:]
    V_11 = V[:n_orders, :n_orders]
    V_12 = V[:n_orders, n_orders:]
    V_21 = V[n_orders:, :n_orders]
    V_22 = V[n_orders:, n_orders:]

    W_ss = c * W_21 - s * W_11
    W_sp = c * W_22 - s * W_12
    W_ps = c * W_11 + s * W_21
    W_pp = c * W_12 + s * W_22

    V_ss = c * V_11 + s * V_21
    V_sp = c * V_12 + s * V_22
    V_ps = c * V_21 - s * V_11
    V_pp = c * V_22 - s * V_12

    big_W = _block(xp, [[W_ss, W_sp], [W_ps, W_pp]])
    big_V = _block(xp, [[V_ss, V_sp], [V_ps, V_pp]])
    return big_W, big_V


def solve_2d(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
             epx_conv_all, epy_conv_all, epz_conv_i_all,
             epxy_conv_all=None, epyx_conv_all=None,
             *, phi, incidence_side='top', answers=Answers.EFFICIENCY, lit=None,
             mu=None, mu_top=1.0, mu_bot=1.0, eps_top=None, eps_bot=None):
    """Solve a 2D grating stack for the given incidence.

    ``incidence`` is the resolved incident amplitude pair ``(a_s, a_p)`` at unit flux — whatever
    when given — see :func:`~wavebit._core.conventions.incident_sp_amplitudes`.

    See :func:`wavebit._core.solve_1d.solve_1d` for the shared argument conventions —
    including ``incidence_side``, handled by the same mirror (the off-diagonal blocks
    reverse with the rest of the stack; a layer's in-plane tensor is untouched by ``z -> -z``).
    ``epxy_conv_all`` / ``epyx_conv_all`` are the optional off-diagonal permittivity blocks
    (``None`` -> zero).

    ``mu`` is the optional per-layer permeability, as five stacks
    ``(mux_conv_all, muy_conv_all, muz_conv_i_all, muxy_conv_all, muyx_conv_all)`` in the
    same slots and the same convention as the permittivity — see :func:`layer_modes_2d`. Any
    entry may be ``None``.

    ``mu_top`` / ``mu_bot`` are the **half-spaces'** scalar permeabilities. ``n_top`` / ``n_bot``
    keep meaning the *refractive index* ``sqrt(eps*mu)``, so nothing about ``kz``, Snell or the
    order grid changes and every existing call means what it did; ``mu`` enters only through
    the boundary admittances (:func:`~wavebit._core.solve_1d.sp_efficiencies`).

    ``eps_top`` / ``eps_bot`` are those half-spaces' permittivities. Optional, and recovered as
    ``n**2/mu`` when they are not given -- which is what this function did unconditionally, and
    is still right for a caller holding only an index. The engine passes them because it holds
    both numbers already: the recovery is a round trip through ``sqrt`` and back, exact only up
    to rounding, and there is no reason to take it when the value is in hand. Supply them with
    ``mu_top`` / ``mu_bot`` or with neither -- half a pair would describe a medium whose index
    and permittivity disagree.

    Each is read exactly once, as the p-channel admittance ``kz/eps`` of its own half-space:
    ``eps_bot`` in the substrate seed ``big_F`` below, ``eps_top`` in
    :func:`~wavebit._core.solve_1d.final_matching`'s top row. Both are the *same* two reads the
    SMM twin makes in ``W_bot`` / ``W_top``, which is the property that keeps the two
    connecting algorithms answering one question -- ``eps_top`` reached this signature before
    it reached the ETM matching, and for a while the two algorithms honoured different inputs.

    ``kx`` / ``ky`` are the retained in-plane orders, in either spelling :func:`order_grid_kxy`
    accepts: 1-D of lengths ``n_orders_u`` / ``n_orders_v`` for an axis-aligned cell, or
    ``(n_orders_v, n_orders_u)`` per-order Cartesian components for any lattice.

    Returns ``(de_ri, de_ti, power_entering)``: two efficiency arrays of shape
    ``(n_orders_v, n_orders_u)`` and a scalar net entering power fraction.
    """
    xp = array_namespace(epx_conv_all)
    mu_all = (None,) * 5 if mu is None else tuple(mu)
    if _is_backside(incidence_side):
        # mu reverses with the layers it belongs to. `z -> -z` leaves a layer's in-plane tensor
        # alone whether it is eps or mu, so the mirror is the same one, with five more stacks.
        (n_top, n_bot, thickness, epx_conv_all, epy_conv_all, epz_conv_i_all,
         epxy_conv_all, epyx_conv_all, *mu_all) = _mirrored_scene(
            n_top, n_bot, thickness, epx_conv_all, epy_conv_all, epz_conv_i_all,
            epxy_conv_all, epyx_conv_all, *mu_all)
        # The half-spaces swap with the indices they belong to -- all three quantities of
        # each, or the mirrored scene would take its permittivity from the far side.
        mu_top, mu_bot = mu_bot, mu_top
        eps_top, eps_bot = eps_bot, eps_top
    kx = as_array(xp, kx)
    ky = as_array(xp, ky)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        kx = xp.astype(kx, xp.complex128)
        ky = xp.astype(ky, xp.complex128)
    C = kx.dtype

    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    n_orders_v, n_orders_u = kx_g.shape
    n_orders = n_orders_u * n_orders_v
    k0 = 2 * 3.141592653589793 / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(n_orders), C)
    O = xp.astype(xp.zeros((n_orders, n_orders)), C)

    # --- stage 1: boundary init (transfer_2d_1 == transfer_1d_conical_1) ----------------
    k_par_sq = kx_g ** 2 + ky_g ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq), (-1,))
    varphi = order_azimuth(xp, ky_g, kx_g, phi)

    if eps_bot is None:
        eps_bot = n_bot ** 2 / mu_bot      # n stays the refractive index; eps carries the mu
    # The substrate's two admittances: kz/eps on the magnetic (p) amplitude, kz/mu on the
    # electric (s) one. Delay their dense storage until the first modes are
    # ready. The empty stack uses the same seed before final matching.
    def substrate_state():
        Kz_bot = _diag(xp, kz_bot)
        return (_block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]]),
                _block(xp, [[-1j * Kz_bot / mu_bot, O], [O, I]]),
                xp.astype(xp.eye(2 * n_orders), C))
    big_F = big_G = big_T = None

    # The order grid flattened row-major: m fastest, matching how the ``fourier2d_*`` builders
    # lay out the convolution matrices. Diagonal either way — the eigenproblem below reads
    # only these two diagonals, which is why a skewed lattice never reaches it.
    Kx = _diag(xp, xp.reshape(kx_g, (-1,)))
    Ky = _diag(xp, xp.reshape(ky_g, (-1,)))
    F_c, F_s = xp.cos(varphi), xp.sin(varphi)
    # --- stage 2/3: per-layer eig + coupled block transfer (transfer_2d_2 / transfer_2d_3)
    keep_field = answers_of(answers).covers(Answers.FIELD)
    layer_info_list = []
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]
    for layer_index in range(n_layers)[::-1]:
        epx_conv = epx_conv_all[layer_index]
        epy_conv = epy_conv_all[layer_index]
        epz_conv_i = epz_conv_i_all[layer_index]
        epxy_conv = O if epxy_conv_all is None else epxy_conv_all[layer_index]
        epyx_conv = O if epyx_conv_all is None else epyx_conv_all[layer_index]
        d = thickness[layer_index]

        # `mux_conv`/`muy_conv` fall back to I and the other three to None, which is what
        # `layer_modes_2d` reads as non-magnetic -- so a stack with mu on some layers and not
        # others is expressible, and the layers without it take the original arithmetic.
        layer_mu = None if all(s is None for s in mu_all) else (
            I if mu_all[0] is None else mu_all[0][layer_index],
            I if mu_all[1] is None else mu_all[1][layer_index],
            *(None if s is None else s[layer_index] for s in mu_all[2:]))

        # The ETM's labels (:func:`~wavebit._core.conventions.etm_mode_q`): this recursion
        # inverts a transmission into the medium below, so a layer's `+` modes have to be
        # the waves that medium receives.
        W, V, q = layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i,
                                 epxy_conv, epyx_conv, C, mu=layer_mu, k0_d=k0*d)

        big_W, big_V = sp_modes_2d(xp, W, V, F_c, F_s, n_orders)
        if not keep_field:
            del W, V
        if big_F is None:
            big_F, big_G, big_T = substrate_state()

        big_W_i = xp.linalg.inv(big_W)
        big_V_i = xp.linalg.inv(big_V)
        big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)
        big_A_i = xp.linalg.inv(0.5 * (big_W_i @ big_F + big_V_i @ big_G))
        del big_W_i, big_V_i

        from .modal_gradient import exponential
        big_X = exponential(xp, q, -k0*d)

        Y = big_X @ big_B @ big_A_i @ big_X
        big_I = xp.astype(xp.eye(2 * n_orders), C)
        big_F = big_W @ (big_I + Y)
        big_G = big_V @ (big_I - Y)
        del big_W, big_V, big_I, Y
        big_T = big_T @ big_A_i @ big_X

        # Bottom-to-top, matching the legacy append order; consumed by field reconstruction.
        # Conditional for the reason `solve_1d`'s append is -- and this is the path where it
        # costs the most: `W`, `V`, `big_A_i` and `big_B` are each a full `2*n_orders` square,
        # 4.25 of them per layer once `epz_conv_i` is counted at its quarter (4.5 on a
        # magnetic layer, where `muz_conv_i` is the second quarter).
        #
        # `muz_conv_i` rides last, and it is there for the same reason `epz_conv_i` is: the two
        # longitudinal components are algebraic rather than propagated, and each is recovered
        # by dividing out the z slot of its own tensor. `H_z = muz_conv_i (Kx Ey - Ky Ex)` is
        # Faraday's z row, dual to Ampere's z row that gives `E_z` -- see
        # :func:`wavebit._core.fields._field_coupled`. `None` is "non-magnetic", the spelling
        # `layer_mu` and `Material.mu` already use.
        if keep_field:
            layer_info_list.append([epz_conv_i, W, V, q, d, big_A_i, big_B,
                                    None if layer_mu is None else layer_mu[2]])
        # Field declarations own the saved references above. Other solves
        # release them before the next layer or final boundary elimination.
        del big_X, big_A_i, big_B

    # --- stage 3: final matching -> efficiency (transfer_2d_4 == transfer_1d_conical_4) --
    if big_F is None:
        big_F, big_G, big_T = substrate_state()
    out = final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, incidence, theta,
                         n_top, n_bot, n_orders_u, n_orders_v, answers=answers,
                         mu_top=mu_top, mu_bot=mu_bot, eps_top=eps_top, eps_bot=eps_bot,
                         lit=lit, kx=kx, ky=ky)
    if keep_field:
        out['layer_info_list'] = layer_info_list
    return out
