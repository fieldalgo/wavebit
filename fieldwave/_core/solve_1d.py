"""Single-source 1D (TE / TM / conical) RCWA solve — Phase 3 of the rewrite (``<a scratch note outside this repository>``).

One backend-agnostic implementation of the transfer-matrix solve for a 1D grating stack,
written against the Array-API namespace (``numpy`` / ``jax.numpy`` / ``torch``). It is the
port of the triplicated ``transfer_1d_*`` chains into a single source: the only per-backend
primitive it reaches for is :func:`fieldwave._core.backend.eig`, and every sign/branch choice
comes from :mod:`fieldwave._core.conventions`.

``solve_1d`` handles the pure TE / TM (no azimuth) path; ``solve_1d_conical`` handles the
general 1D grating (arbitrary ``psi`` / ``phi``) with its coupled s/p block system.

Boundary contract — each consumes what the matching legacy ``solve_1d`` / ``solve_1d_conical``
consumes (the per-layer convolution matrices already built by the factorization stage, plus
the incidence/geometry scalars) and returns the same ``(de_ri, de_ti)`` diffraction
efficiencies. Feeding the exact inputs the legacy path uses reproduces the legacy result
bit-for-bit; feeding JAX / PyTorch arrays runs the identical algebra with autodiff intact.

Stages (matching the legacy chain, kept as one function each to make bit-parity auditable):
  1. boundary init      — kz for efficiency, substrate admittance G, F = T = I
  2. per-layer eig+couple— Redheffer/transfer propagation from the bottom layer up
  3. final matching      — incident field -> R, T -> diffraction efficiency
"""
from .backend import array_namespace, as_array, check_finite, check_physical, eig
from .conventions import boundary_kz, mode_q


def _diag(xp, v):
    """Construct a diagonal matrix from a 1D vector (Array-API ``diag`` build form)."""
    return xp.diag(v)


def _block(xp, rows):
    """Assemble a block matrix from a 2D list of blocks (Array-API stand-in for ``np.block``).

    ``rows`` is a list of block-rows; each block-row is a list of equal-height matrices that
    concatenate horizontally, and the rows then concatenate vertically.
    """
    return xp.concat([xp.concat(r, axis=1) for r in rows], axis=0)


def layer_modes_1d(xp, pol, kx, I, epx_conv, epy_conv, epz_conv_i, C):
    """Eigenmodes ``(W, V, q)`` of one 1D TE / TM layer.

    ``W`` expands the continuous primary field (``E_y`` for TE, ``H_y`` for TM), ``V`` its
    partner, and ``q`` are the modal decay constants used as ``exp(-k0*q*d)``. The principal
    square root keeps ``Re(q) >= 0``, so that exponential never grows.

    Shared by the ETM (:func:`solve_1d`) and SMM (:mod:`fieldwave._core.scattering`) connecting
    algorithms — the mode algebra is a property of the layer, not of how layers are joined.
    """
    if pol == 0:  # TE
        A = _diag(xp, kx) @ _diag(xp, kx) - epy_conv
        w, W = eig(A)
        q = mode_q(xp, xp.astype(w, C))
        V = W * q
    elif pol == 1:  # TM
        B = kx[:, None] * epz_conv_i * kx[None, :] - I
        w, W = eig(epx_conv @ B)
        q = mode_q(xp, xp.astype(w, C))
        V = xp.linalg.inv(epx_conv) @ W * q
    else:
        raise ValueError(f'pol must be 0 (TE) or 1 (TM), got {pol!r}')
    return W, V, q


def layer_modes_1d_conical(xp, Kx, Ky, Kx2, Ky2, I, epx_conv, epy_conv, epz_conv_i, C):
    """Raw (unrotated) eigenmodes of one conical 1D layer.

    Returns ``(W1, W2, q1, q2, V_11, V_12, V_21, V_22)`` — the two decoupled mode families and
    the four blocks that couple them into the s/p system built by :func:`sp_modes_1d_conical`.
    Shared by the ETM and SMM connecting algorithms.
    """
    A = Kx2 - epy_conv
    B = Kx @ epz_conv_i @ Kx - I
    w1, W1 = eig(Ky2 + A)
    w2, W2 = eig(Ky2 + B @ epx_conv)
    q1 = mode_q(xp, xp.astype(w1, C))
    q2 = mode_q(xp, xp.astype(w2, C))

    A_i = xp.linalg.inv(A)
    B_i = xp.linalg.inv(B)
    V_11 = (A_i @ W1) * q1
    V_12 = Ky @ A_i @ Kx @ W2
    V_21 = Ky @ B_i @ Kx @ epz_conv_i @ W1
    V_22 = (B_i @ W2) * q2
    return W1, W2, q1, q2, V_11, V_12, V_21, V_22


def sp_modes_1d_conical(xp, F_c, F_s, W1, W2, V_11, V_12, V_21, V_22):
    """Rotate conical layer modes into the s/p boundary basis: ``(big_W, big_V)``.

    ``big_W`` maps mode amplitudes to the boundary variables ``(E_s, F_p)`` and ``big_V`` to
    ``(H_s, H_p)`` — the same pair the substrate seed ``big_F`` / ``big_G`` is written in.
    """
    V_ss = F_c @ V_11
    V_sp = F_c @ V_12 - F_s @ W2
    W_ss = F_c @ W1 + F_s @ V_21
    W_sp = F_s @ V_22
    W_ps = F_s @ V_11
    W_pp = F_c @ W2 + F_s @ V_12
    V_ps = F_c @ V_21 - F_s @ W1
    V_pp = F_c @ V_22

    big_W = _block(xp, [[V_ss, V_sp], [W_ps, W_pp]])
    big_V = _block(xp, [[W_ss, W_sp], [V_ps, V_pp]])
    return big_W, big_V


def solve_1d(pol, wavelength, kx, n_top, n_bot, theta, thickness,
             epx_conv_all, epy_conv_all, epz_conv_i_all,
             full=False):
    """Solve a 1D grating stack for reflection/transmission diffraction efficiency.

    Args:
        pol: ``0`` (TE) or ``1`` (TM).
        wavelength: incident wavelength (sets ``k0 = 2*pi/wavelength``).
        kx: in-plane wavevectors of the retained orders (length ``ff = 2*fto+1``).
        n_top, n_bot: superstrate / substrate refractive indices.
        theta: incident polar angle (``None`` -> 0; matches the legacy ``cos(theta)`` factor).
        thickness: per-layer thicknesses, top-to-bottom (``ucell``/first-axis order).
        epx_conv_all, epy_conv_all, epz_conv_i_all: per-layer convolution matrices,
            indexed ``[layer]`` on the first axis, as produced by the factorization stage.

    Returns:
        ``(de_ri, de_ti)`` — reflected / transmitted efficiency per order, shape ``(1, ff)``.
        With ``full=True``, the complete legacy result mapping instead (amplitudes and s/p
        split; the pure TE/TM path carries all of one polarization, the other side is zero).
    """
    xp = array_namespace(epx_conv_all)
    kx = as_array(xp, kx)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        # Promote a real kx to the complex working dtype the algebra needs.
        kx = xp.astype(kx, xp.complex128)
    C = kx.dtype

    ff_x = kx.shape[0]
    k0 = 2 * 3.141592653589793 / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(ff_x), C)

    # --- stage 1: boundary init (port of transfer_1d_1) --------------------------------
    # One kz per half-space on the downward branch; it serves both the substrate admittance
    # and the efficiency factors. Scalars promote against the complex kx.
    kz_top = boundary_kz(xp, n_top ** 2, kx ** 2)
    kz_bot = boundary_kz(xp, n_bot ** 2, kx ** 2)

    eps_bot = n_bot ** 2
    if pol == 0:      # TE
        G = -1j * _diag(xp, kz_bot)
    elif pol == 1:    # TM
        G = -1j * _diag(xp, kz_bot / eps_bot)
    else:
        raise ValueError(f'pol must be 0 (TE) or 1 (TM), got {pol!r}')

    F = I
    T = I

    # --- stage 2: per-layer eig + transfer coupling (transfer_1d_2 / transfer_1d_3) ----
    layer_info_list = []
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]
    for layer_index in range(n_layers)[::-1]:
        epx_conv = epx_conv_all[layer_index]
        epy_conv = epy_conv_all[layer_index]
        epz_conv_i = epz_conv_i_all[layer_index]
        d = thickness[layer_index]

        W, V, q = layer_modes_1d(xp, pol, kx, I, epx_conv, epy_conv, epz_conv_i, C)

        X = _diag(xp, xp.exp(-k0 * q * d))
        W_i = xp.linalg.inv(W)
        V_i = xp.linalg.inv(V)
        b = 0.5 * (W_i @ F - V_i @ G)

        # A layer of the same material as the medium below shares its mode basis, which makes
        # `a` exactly I -- there used to be a branch substituting that and skipping this
        # inverse. It bought one inverse and it is gone; see the module docstring.
        a_i = xp.linalg.inv(0.5 * (W_i @ F + V_i @ G))

        Y = X @ b @ a_i @ X
        F = W @ (I + Y)
        G = V @ (I - Y)
        T = T @ a_i @ X

        # Bottom-to-top, matching the legacy append order; consumed by field reconstruction.
        layer_info_list.append([epz_conv_i, W, V, q, d, a_i, b])

    # --- stage 3: final matching -> efficiency (transfer_1d_4) -------------------------
    Kz_top = _diag(xp, kz_top)
    delta = xp.astype(_unit(xp, ff_x, ff_x // 2), C)
    cos_t = xp.cos(as_array(xp, theta, dtype=C))

    if pol == 0:  # TE
        inc = -1j * n_top * cos_t * delta
        T1 = xp.linalg.inv(G - 1j * Kz_top @ F) @ (-1j * Kz_top @ delta + inc)
    else:         # TM
        Kz_scaled = Kz_top / (n_top ** 2)
        inc = -1j * delta * cos_t / n_top
        T1 = xp.linalg.inv(G - 1j * Kz_scaled @ F) @ (-1j * Kz_scaled @ delta + inc)

    R = (F @ T1 - delta)[None, :]
    Tt = (T @ T1)[None, :]

    de_ri = xp.real(R * xp.conj(R) * xp.real(kz_top / (n_top * cos_t)))
    if pol == 0:  # TE
        de_ti = xp.real(Tt * xp.conj(Tt) * xp.real(kz_bot / (n_top * cos_t)))
    else:         # TM
        de_ti = xp.real(Tt * xp.conj(Tt)
                        * xp.real(kz_bot / n_bot ** 2 / (cos_t / n_top)))

    check_finite(xp, de_ri=de_ri, de_ti=de_ti)
    check_physical(xp, de_ri, de_ti, n_top)

    if not full:
        return de_ri, de_ti

    # TE carries everything on s, TM everything on p; the opposite side is identically zero.
    zeros_amp = xp.zeros_like(R)
    zeros_eff = xp.zeros_like(de_ri)
    if pol == 0:
        res = _pack(R, zeros_amp, Tt, zeros_amp, de_ri, zeros_eff, de_ti, zeros_eff)
    else:
        res = _pack(zeros_amp, R, zeros_amp, Tt, zeros_eff, de_ri, zeros_eff, de_ti)
    return {'res': res, 'layer_info_list': layer_info_list, 'T1': T1}


def solve_1d_conical(wavelength, kx, ky, n_top, n_bot, theta, psi, thickness,
                     epx_conv_all, epy_conv_all, epz_conv_i_all,
                     full=False):
    """Solve a general 1D grating (arbitrary azimuth/polarization) for the given ``psi``.

    The conical case couples the s and p polarizations, so the per-layer modes and the
    boundary system are ``2*ff`` block matrices. Port of ``transfer_1d_conical_1..4``; returns
    the ``(de_ri, de_ti)`` of the ``psi``-specified incidence (shape ``(ff_y, ff_x)`` with
    ``ff_y == 1``). See :func:`solve_1d` for the argument conventions; ``ky`` is the single
    out-of-plane order and ``psi`` the polarization angle.
    """
    xp = array_namespace(epx_conv_all)
    kx = as_array(xp, kx)
    ky = as_array(xp, ky)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        kx = xp.astype(kx, xp.complex128)
        ky = xp.astype(ky, xp.complex128)
    C = kx.dtype

    ff_x = kx.shape[0]
    ff_y = ky.shape[0]
    ff_xy = ff_x * ff_y
    k0 = 2 * 3.141592653589793 / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(ff_xy), C)
    O = xp.astype(xp.zeros((ff_xy, ff_xy)), C)
    ky_col = xp.reshape(ky, (-1, 1))

    # --- stage 1: boundary init (transfer_1d_conical_1) --------------------------------
    k_par_sq = kx ** 2 + ky_col ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq), (-1,))
    varphi = xp.reshape(xp.atan(ky_col / kx), (-1,))

    eps_bot = n_bot ** 2
    Kz_bot = _diag(xp, kz_bot)

    big_F = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])
    big_G = _block(xp, [[-1j * Kz_bot, O], [O, I]])
    big_T = xp.astype(xp.eye(2 * ff_xy), C)

    # --- stage 2/3: per-layer eig + coupled s/p transfer (conical_2 / conical_3) --------
    Kx = _diag(xp, kx)
    ky_vec = ky[0] * xp.astype(xp.ones((ff_x,)), C)      # ff_y == 1: Ky = diag(ky0 * 1)
    Ky = _diag(xp, ky_vec)
    Kx2 = _diag(xp, kx * kx)
    Ky2 = _diag(xp, ky_vec * ky_vec)
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

    layer_info_list = []
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]
    for layer_index in range(n_layers)[::-1]:
        epx_conv = epx_conv_all[layer_index]
        epy_conv = epy_conv_all[layer_index]
        epz_conv_i = epz_conv_i_all[layer_index]
        d = thickness[layer_index]

        W1, W2, q1, q2, V_11, V_12, V_21, V_22 = layer_modes_1d_conical(
            xp, Kx, Ky, Kx2, Ky2, I, epx_conv, epy_conv, epz_conv_i, C)

        X_1 = _diag(xp, xp.exp(-k0 * q1 * d))
        X_2 = _diag(xp, xp.exp(-k0 * q2 * d))
        big_I = xp.astype(xp.eye(2 * ff_xy), C)
        big_X = _block(xp, [[X_1, O], [O, X_2]])

        W = _block(xp, [[W1, W2]])
        V = _block(xp, [[V_11, V_12], [V_21, V_22]])
        q = xp.concat([q1, q2])

        big_W, big_V = sp_modes_1d_conical(xp, F_c, F_s, W1, W2,
                                           V_11, V_12, V_21, V_22)

        big_W_i = xp.linalg.inv(big_W)
        big_V_i = xp.linalg.inv(big_V)
        big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)
        big_A_i = xp.linalg.inv(0.5 * (big_W_i @ big_F + big_V_i @ big_G))

        Y = big_X @ big_B @ big_A_i @ big_X
        big_F = big_W @ (big_I + Y)
        big_G = big_V @ (big_I - Y)
        big_T = big_T @ big_A_i @ big_X

        layer_info_list.append([epz_conv_i, W, V, q, d, big_A_i, big_B])

    # --- stage 3: final matching -> efficiency (transfer_1d_conical_4) ------------------
    out = final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, psi, theta,
                         n_top, n_bot, ff_x, ff_y, full=full)
    if full:
        out['layer_info_list'] = layer_info_list
    return out


def _unit(xp, n, i):
    """Unit vector ``e_i`` of length ``n`` (real)."""
    v = [0.0] * n
    v[i] = 1.0
    return xp.asarray(v)


def final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, psi, theta,
                   n_top, n_bot, ff_x, ff_y, full=False):
    """Boundary match the assembled s/p system to the incident field.

    Shared, byte-identical port of ``transfer_1d_conical_4`` / ``transfer_2d_4`` (their final
    equations are the same); used by both :func:`solve_1d_conical` and
    :func:`fieldwave._core.solve_2d.solve_2d`.

    Returns ``(de_ri, de_ti)`` for the ``psi``-specified incidence, shape ``(ff_y, ff_x)``.
    With ``full=True`` returns the complete legacy result mapping instead: the reflected and
    transmitted amplitudes and their s/p-resolved efficiencies for the given ``psi``, plus the
    pure-TE and pure-TM incidence variants (which reuse the same factorized ``final_A``, so
    they cost one extra solve rather than two full passes).
    """
    ff_xy = ff_x * ff_y
    I = xp.astype(xp.eye(ff_xy), C)
    O = xp.astype(xp.zeros((ff_xy, ff_xy)), C)

    Kz_top = _diag(xp, kz_top)
    kz_top_2d = xp.reshape(kz_top, (ff_y, ff_x))
    kz_bot_2d = xp.reshape(kz_bot, (ff_y, ff_x))
    cos_t = xp.cos(as_array(xp, theta, dtype=C))
    sin_psi = xp.sin(as_array(xp, psi, dtype=C))
    cos_psi = xp.cos(as_array(xp, psi, dtype=C))

    delta = xp.reshape(xp.astype(_unit(xp, ff_xy, ff_xy // 2), C), (ff_xy, 1))

    bF = [[big_F[:ff_xy, :ff_xy], big_F[:ff_xy, ff_xy:]],
          [big_F[ff_xy:, :ff_xy], big_F[ff_xy:, ff_xy:]]]
    bG = [[big_G[:ff_xy, :ff_xy], big_G[:ff_xy, ff_xy:]],
          [big_G[ff_xy:, :ff_xy], big_G[ff_xy:, ff_xy:]]]

    final_A = _block(xp, [
        [I, O, -bF[0][0], -bF[0][1]],
        [O, 1j * Kz_top / (n_top ** 2), -bF[1][0], -bF[1][1]],
        [1j * Kz_top, O, -bG[0][0], -bG[0][1]],
        [O, I, -bG[1][0], -bG[1][1]],
    ])
    final_B = _block(xp, [
        [-sin_psi * delta],
        [cos_psi * cos_t * delta],
        [1j * sin_psi * n_top * cos_t * delta],
        [1j * n_top * cos_psi * delta],
    ])

    final_A_inv = xp.linalg.inv(final_A)
    final_RT = final_A_inv @ final_B
    R_s = xp.reshape(final_RT[:ff_xy, :], (ff_y, ff_x))
    R_p = xp.reshape(final_RT[ff_xy:2 * ff_xy, :], (ff_y, ff_x))

    big_T1 = final_RT[2 * ff_xy:, :]
    big_T_tetm = big_T
    big_T = big_T @ big_T1
    T_s = xp.reshape(big_T[:ff_xy, :], (ff_y, ff_x))
    T_p = xp.reshape(big_T[ff_xy:, :], (ff_y, ff_x))

    denom = n_top * cos_t

    def efficiencies(R_s_, R_p_, T_s_, T_p_):
        return sp_efficiencies(xp, kz_top_2d, kz_bot_2d, denom, n_top, n_bot,
                               R_s_, R_p_, T_s_, T_p_)

    de_ri_s, de_ri_p, de_ti_s, de_ti_p = efficiencies(R_s, R_p, T_s, T_p)

    check_finite(xp, de_ri=de_ri_s + de_ri_p, de_ti=de_ti_s + de_ti_p)
    check_physical(xp, de_ri_s + de_ri_p, de_ti_s + de_ti_p, n_top)

    if not full:
        return de_ri_s + de_ri_p, de_ti_s + de_ti_p

    res = _pack(R_s, R_p, T_s, T_p, de_ri_s, de_ri_p, de_ti_s, de_ti_p)

    # Pure-TE (psi = pi/2) and pure-TM (psi = 0) incidence, solved together against the same
    # already-inverted final_A.
    half_pi = 3.141592653589793 / 2
    cols = []
    for psi_v in (half_pi, 0.0):
        s_p = xp.sin(xp.asarray(psi_v, dtype=C))
        c_p = xp.cos(xp.asarray(psi_v, dtype=C))
        cols.append(_block(xp, [[-s_p * delta],
                                [c_p * cos_t * delta],
                                [1j * s_p * n_top * cos_t * delta],
                                [1j * n_top * c_p * delta]]))
    final_RT_tetm = final_A_inv @ xp.concat(cols, axis=1)

    def split2(M):
        """Two incidence columns -> (2, ff_y, ff_x), matching the legacy ``.T.reshape``."""
        return xp.reshape(xp.matrix_transpose(M), (2, ff_y, ff_x))

    R_s_2 = split2(final_RT_tetm[:ff_xy, :])
    R_p_2 = split2(final_RT_tetm[ff_xy:2 * ff_xy, :])
    big_T1_tetm = final_RT_tetm[2 * ff_xy:, :]
    big_T_tetm = big_T_tetm @ big_T1_tetm
    T_s_2 = split2(big_T_tetm[:ff_xy, :])
    T_p_2 = split2(big_T_tetm[ff_xy:, :])

    eff2 = efficiencies(R_s_2, R_p_2, T_s_2, T_p_2)
    variants = [_pack(R_s_2[i], R_p_2[i], T_s_2[i], T_p_2[i], *(e[i] for e in eff2))
                for i in (0, 1)]

    # Field reconstruction consumes T1 for the given psi plus the two incidence variants.
    T1 = xp.stack([big_T1, big_T1_tetm[:, 0:1], big_T1_tetm[:, 1:2]])

    return {'res': res, 'res_te_inc': variants[0], 'res_tm_inc': variants[1], 'T1': T1}


def sp_efficiencies(xp, kz_top_2d, kz_bot_2d, denom, n_top, n_bot, R_s, R_p, T_s, T_p):
    """Power-normalized diffraction efficiencies from the s/p boundary amplitudes.

    ``denom = n_top*cos(theta)`` is the incident-power factor; the s channel carries the
    electric amplitude and the p channel the magnetic one, hence the ``1/n**2`` on p.
    Shared by the ETM final matching and the SMM connecting algorithm.
    """
    return (xp.real(R_s * xp.conj(R_s) * xp.real(kz_top_2d / denom)),
            xp.real(R_p * xp.conj(R_p) * xp.real(kz_top_2d / n_top ** 2 / denom)),
            xp.real(T_s * xp.conj(T_s) * xp.real(kz_bot_2d / denom)),
            xp.real(T_p * xp.conj(T_p) * xp.real(kz_bot_2d / n_bot ** 2 / denom)))


def _pack(R_s, R_p, T_s, T_p, de_ri_s, de_ri_p, de_ti_s, de_ti_p):
    """Assemble one incidence's results in the legacy ``ResultSub`` field layout."""
    return {'R_s': R_s, 'R_p': R_p, 'T_s': T_s, 'T_p': T_p,
            'de_ri': de_ri_s + de_ri_p, 'de_ri_s': de_ri_s, 'de_ri_p': de_ri_p,
            'de_ti': de_ti_s + de_ti_p, 'de_ti_s': de_ti_s, 'de_ti_p': de_ti_p}
