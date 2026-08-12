"""Single-source 2D RCWA solve — Phase 3c of the rewrite.

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
from .backend import array_namespace, as_array, eig
from .conventions import boundary_kz, mode_q
from .solve_1d import _block, _diag, final_matching


def order_grid_kxy(xp, kx, ky):
    """``kx`` / ``ky`` on the ``(ff_y, ff_x)`` order grid, from either spelling.

    Two ways in, because a lattice basis decides which one is even expressible:

    * **1-D**, lengths ``(ff_x,)`` and ``(ff_y,)`` — the separable form an axis-aligned cell
      admits, where ``kx`` depends only on ``m`` and ``ky`` only on ``n``. Broadcast to the
      grid here, which is what both solves used to do inline.
    * **2-D**, ``(ff_y, ff_x)`` each — the per-order Cartesian components, which is the only
      form a skewed cell has: with ``G = m*b1 + n*b2`` non-orthogonal, *both* components
      depend on *both* indices and no pair of 1-D vectors can hold them. See
      :func:`fieldwave._core.lattice.order_kxy`.

    Everything downstream reads the grid, so the two spellings converge here and the algebra
    below never branches. The 1-D route broadcasts exactly as before, element for element.
    """
    if kx.ndim == 2:
        return kx, ky
    ff_x, ff_y = kx.shape[0], ky.shape[0]
    return (xp.broadcast_to(kx[None, :], (ff_y, ff_x)),
            xp.broadcast_to(xp.reshape(ky, (-1, 1)), (ff_y, ff_x)))


def layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i, epxy_conv, epyx_conv, C):
    """Raw (unrotated) eigenmodes ``(W, V, q)`` of one 2D layer.

    ``W`` is the ``[Ex; Ey]`` mode matrix and ``q`` the modal decay constants; the principal
    square root keeps ``Re(q) >= 0`` so ``exp(-k0*q*d)`` never grows. Shared by the ETM
    (:func:`solve_2d`) and SMM (:mod:`fieldwave._core.scattering`) connecting algorithms.

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
    """
    B = Kx @ epz_conv_i @ Kx - I
    D = Ky @ epz_conv_i @ Ky - I
    Cc = Kx @ epz_conv_i @ Ky
    E = Ky @ epz_conv_i @ Kx

    Omega2_LR = _block(xp, [
        [Ky @ Ky + B @ epx_conv - Cc @ epyx_conv, Cc @ epy_conv - Kx @ Ky - B @ epxy_conv],
        [E @ epx_conv - Ky @ Kx - D @ epyx_conv, Kx @ Kx + D @ epy_conv - E @ epxy_conv],
    ])
    w, W = eig(Omega2_LR)
    q = mode_q(xp, xp.astype(w, C))

    Omega_R = _block(xp, [
        [-Kx @ Ky + epyx_conv, Kx @ Kx - epy_conv],
        [epx_conv - Ky @ Ky, Ky @ Kx - epxy_conv],
    ])
    V = (Omega_R @ W) / q
    return W, V, q


def sp_modes_2d(xp, W, V, F_c, F_s, ff_xy):
    """Rotate 2D layer modes into the s/p boundary basis: ``(big_W, big_V)``.

    ``big_W`` maps mode amplitudes to the boundary variables ``(E_s, F_p)`` and ``big_V`` to
    ``(H_s, H_p)`` — the pair the substrate seed ``big_F`` / ``big_G`` is written in.
    """
    W_11 = W[:ff_xy, :ff_xy]
    W_12 = W[:ff_xy, ff_xy:]
    W_21 = W[ff_xy:, :ff_xy]
    W_22 = W[ff_xy:, ff_xy:]
    V_11 = V[:ff_xy, :ff_xy]
    V_12 = V[:ff_xy, ff_xy:]
    V_21 = V[ff_xy:, :ff_xy]
    V_22 = V[ff_xy:, ff_xy:]

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
    return big_W, big_V


def solve_2d(wavelength, kx, ky, n_top, n_bot, theta, psi, thickness,
             epx_conv_all, epy_conv_all, epz_conv_i_all,
             epxy_conv_all=None, epyx_conv_all=None,
             full=False):
    """Solve a 2D grating stack for the given ``psi`` incidence.

    See :func:`fieldwave._core.solve_1d.solve_1d` for the shared argument conventions.
    ``epxy_conv_all`` / ``epyx_conv_all`` are the optional off-diagonal permittivity blocks
    (``None`` -> zero).

    ``kx`` / ``ky`` are the retained in-plane orders, in either spelling
    :func:`order_grid_kxy` accepts: 1-D of lengths ``ff_x`` / ``ff_y`` for an axis-aligned
    cell, or ``(ff_y, ff_x)`` per-order Cartesian components for any lattice.

    Returns ``(de_ri, de_ti)`` for the ``psi``-specified incidence, shape ``(ff_y, ff_x)``.
    """
    xp = array_namespace(epx_conv_all)
    kx = as_array(xp, kx)
    ky = as_array(xp, ky)
    if not xp.isdtype(kx.dtype, 'complex floating'):
        kx = xp.astype(kx, xp.complex128)
        ky = xp.astype(ky, xp.complex128)
    C = kx.dtype

    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    ff_y, ff_x = kx_g.shape
    ff_xy = ff_x * ff_y
    k0 = 2 * 3.141592653589793 / wavelength
    theta = 0.0 if theta is None else theta

    I = xp.astype(xp.eye(ff_xy), C)
    O = xp.astype(xp.zeros((ff_xy, ff_xy)), C)

    # --- stage 1: boundary init (transfer_2d_1 == transfer_1d_conical_1) ----------------
    k_par_sq = kx_g ** 2 + ky_g ** 2
    kz_top = xp.reshape(boundary_kz(xp, n_top ** 2, k_par_sq), (-1,))
    kz_bot = xp.reshape(boundary_kz(xp, n_bot ** 2, k_par_sq), (-1,))
    varphi = xp.reshape(xp.atan(ky_g / kx_g), (-1,))

    eps_bot = n_bot ** 2
    Kz_bot = _diag(xp, kz_bot)

    big_F = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])
    big_G = _block(xp, [[-1j * Kz_bot, O], [O, I]])
    big_T = xp.astype(xp.eye(2 * ff_xy), C)


    # The order grid flattened row-major: m fastest, matching how ``dfs2d`` lays out the
    # convolution matrices. Diagonal either way — the eigenproblem below reads only these two
    # diagonals, which is why a skewed lattice never reaches it.
    Kx = _diag(xp, xp.reshape(kx_g, (-1,)))
    Ky = _diag(xp, xp.reshape(ky_g, (-1,)))
    F_c = _diag(xp, xp.cos(varphi))
    F_s = _diag(xp, xp.sin(varphi))

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

        W, V, q = layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i,
                                 epxy_conv, epyx_conv, C)

        q_1 = q[:ff_xy]
        q_2 = q[ff_xy:]
        X_1 = _diag(xp, xp.exp(-k0 * q_1 * d))
        X_2 = _diag(xp, xp.exp(-k0 * q_2 * d))
        big_X = _block(xp, [[X_1, O], [O, X_2]])

        big_W, big_V = sp_modes_2d(xp, W, V, F_c, F_s, ff_xy)

        big_W_i = xp.linalg.inv(big_W)
        big_V_i = xp.linalg.inv(big_V)
        big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)
        big_A_i = xp.linalg.inv(0.5 * (big_W_i @ big_F + big_V_i @ big_G))

        Y = big_X @ big_B @ big_A_i @ big_X
        big_F = big_W @ (big_I + Y)
        big_G = big_V @ (big_I - Y)
        big_T = big_T @ big_A_i @ big_X

        # Bottom-to-top, matching the legacy append order; consumed by field reconstruction.
        layer_info_list.append([epz_conv_i, W, V, q, d, big_A_i, big_B])

    # --- stage 3: final matching -> efficiency (transfer_2d_4 == transfer_1d_conical_4) --
    out = final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, psi, theta,
                         n_top, n_bot, ff_x, ff_y, full=full)
    if full:
        out['layer_info_list'] = layer_info_list
    return out
