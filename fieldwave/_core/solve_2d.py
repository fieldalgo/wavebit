"""Single-source 2D RCWA solve — Phase 3c of the rewrite (see ``<a scratch note outside this repository>``).

Port of the triplicated ``transfer_2d_1..4`` chain into one backend-agnostic solve against the
Array-API namespace, sharing the boundary/branch conventions and the final s/p matching with
:mod:`fieldwave._core.solve_1d`. The only per-backend primitive is
:func:`fieldwave._core.backend.eig`.

The 2D layer couples both polarizations *and* both in-plane directions, so per-layer modes and
the boundary system are ``2*ff_xy`` block matrices (``ff_xy = ff_x*ff_y``). Optional off-axis
convolution blocks (``epxy`` / ``epyx``, produced by the NVF factorization; ``None`` = zeros)
enter the mode operator. ``transfer_2d_1`` and ``transfer_2d_4`` are identical to the conical
versions, so this module reuses ``solve_1d``'s boundary init and ``final_matching`` verbatim.
"""
from .backend import array_namespace, eig
from .conventions import outgoing_sqrt
from .solve_1d import _block, _diag, _diagonal, _same_material, final_matching


def solve_2d(wavelength, kx, ky, n_top, n_bot, theta, psi, thickness,
             epx_conv_all, epy_conv_all, epz_conv_i_all,
             epxy_conv_all=None, epyx_conv_all=None,
             same_material=None, same_material_rtol=1e-3, use_pinv=False, full=False):
    """Solve a 2D grating stack for the given ``psi`` incidence.

    See :func:`fieldwave._core.solve_1d.solve_1d` for the shared argument conventions. ``kx`` /
    ``ky`` are the retained in-plane orders (lengths ``ff_x`` / ``ff_y``); ``epxy_conv_all`` /
    ``epyx_conv_all`` are the optional off-diagonal permittivity blocks (``None`` -> zero).

    Returns ``(de_ri, de_ti)`` for the ``psi``-specified incidence, shape ``(ff_y, ff_x)``.
    """
    xp = array_namespace(epx_conv_all)
    _inv = xp.linalg.pinv if use_pinv else xp.linalg.inv
    kx = xp.asarray(kx)
    ky = xp.asarray(ky)
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

    # --- stage 1: boundary init (transfer_2d_1 == transfer_1d_conical_1) ----------------
    kz_top = xp.reshape(xp.conj((n_top ** 2 - kx ** 2 - ky_col ** 2) ** 0.5), (-1,))
    kz_bot = xp.reshape(xp.conj((n_bot ** 2 - kx ** 2 - ky_col ** 2) ** 0.5), (-1,))
    varphi = xp.reshape(xp.atan(ky_col / kx), (-1,))

    eps_bot = n_bot ** 2
    kz_sub = outgoing_sqrt(xp, xp.reshape(eps_bot - kx ** 2 - ky_col ** 2 - 1e-20j, (-1,)))
    Kz_bot = _diag(xp, kz_sub)

    big_F = _block(xp, [[I, O], [O, 1j * Kz_bot / eps_bot]])
    big_G = _block(xp, [[1j * Kz_bot, O], [O, I]])
    big_T = xp.astype(xp.eye(2 * ff_xy), C)

    eps_below = xp.asarray(complex(n_bot).conjugate() ** 2, dtype=C)

    # Kx = diag(tile(kx, ff_y)) ; Ky = diag(tile(ky[:,None], ff_x)) — Array-API broadcast form.
    kx_full = xp.reshape(xp.broadcast_to(kx[None, :], (ff_y, ff_x)), (-1,))
    ky_full = xp.reshape(xp.broadcast_to(ky_col, (ff_y, ff_x)), (-1,))
    Kx = _diag(xp, kx_full)
    Ky = _diag(xp, ky_full)

    # --- stage 2/3: per-layer eig + coupled block transfer (transfer_2d_2 / transfer_2d_3)
    big_I = xp.astype(xp.eye(2 * ff_xy), C)
    layer_info_list = []
    n_layers = len(thickness) if not hasattr(thickness, 'shape') else thickness.shape[0]
    for layer_index in range(n_layers)[::-1]:
        epx_conv = epx_conv_all[layer_index]
        epy_conv = epy_conv_all[layer_index]
        epz_conv_i = epz_conv_i_all[layer_index]
        epxy_conv = O if epxy_conv_all is None else epxy_conv_all[layer_index]
        epyx_conv = O if epyx_conv_all is None else epyx_conv_all[layer_index]
        d = thickness[layer_index]

        B = Kx @ epz_conv_i @ Kx - I
        D = Ky @ epz_conv_i @ Ky - I
        Cc = Kx @ epz_conv_i @ Ky
        E = Ky @ epz_conv_i @ Kx

        Omega2_LR = _block(xp, [
            [Ky @ Ky + B @ epx_conv - Cc @ epyx_conv, Cc @ epy_conv - Kx @ Ky - B @ epxy_conv],
            [E @ epx_conv - Ky @ Kx - D @ epyx_conv, Kx @ Kx + D @ epy_conv - E @ epxy_conv],
        ])
        w, W = eig(Omega2_LR)
        q = xp.astype(w, C) ** 0.5

        Omega_R = _block(xp, [
            [-Kx @ Ky + epxy_conv, Kx @ Kx - epy_conv],
            [epx_conv - Ky @ Ky, Ky @ Kx - epyx_conv],
        ])
        V = (Omega_R @ W) / q

        if same_material is None:
            layer_eps_diag = xp.mean(_diagonal(xp, epx_conv))
            is_same = _same_material(xp, layer_eps_diag, eps_below, same_material_rtol)
            eps_below = layer_eps_diag
        else:
            is_same = bool(same_material[layer_index])

        q_1 = q[:ff_xy]
        q_2 = q[ff_xy:]
        X_1 = _diag(xp, xp.exp(-k0 * q_1 * d))
        X_2 = _diag(xp, xp.exp(-k0 * q_2 * d))
        big_X = _block(xp, [[X_1, O], [O, X_2]])

        if is_same:
            W_i = _inv(W)
            big_T = big_T @ (W @ big_X @ W_i)
            big_A_i = big_B = xp.zeros_like(big_F)
        else:
            W_11 = W[:ff_xy, :ff_xy]
            W_12 = W[:ff_xy, ff_xy:]
            W_21 = W[ff_xy:, :ff_xy]
            W_22 = W[ff_xy:, ff_xy:]
            V_11 = V[:ff_xy, :ff_xy]
            V_12 = V[:ff_xy, ff_xy:]
            V_21 = V[ff_xy:, :ff_xy]
            V_22 = V[ff_xy:, ff_xy:]

            F_c = _diag(xp, xp.cos(varphi))
            F_s = _diag(xp, xp.sin(varphi))

            W_ss = F_c @ W_21 - F_s @ W_11
            W_sp = F_c @ W_22 - F_s @ W_12
            W_ps = F_c @ W_11 + F_s @ W_21
            W_pp = F_c @ W_12 + F_s @ W_22

            V_ss = F_c @ V_11 + F_s @ V_21
            V_sp = F_c @ V_12 + F_s @ V_22
            V_ps = F_c @ V_21 - F_s @ V_11
            V_pp = F_c @ V_22 - F_s @ V_12

            big_W = _block(xp, [[W_ss, W_sp], [W_ps, W_pp]])
            big_V = _block(xp, [[V_ss, V_sp], [V_ps, V_pp]])

            big_W_i = _inv(big_W)
            big_V_i = _inv(big_V)
            big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
            big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)
            big_A_i = _inv(big_A)

            big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
            big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)
            big_T = big_T @ big_A_i @ big_X

        # Bottom-to-top, matching the legacy append order; consumed by field reconstruction.
        layer_info_list.append([epz_conv_i, W, V, q, d, big_A_i, big_B])

    # --- stage 3: final matching -> efficiency (transfer_2d_4 == transfer_1d_conical_4) --
    out = final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, psi, theta,
                         n_top, n_bot, ff_x, ff_y, use_pinv=use_pinv, full=full)
    if full:
        out['layer_info_list'] = layer_info_list
    return out
