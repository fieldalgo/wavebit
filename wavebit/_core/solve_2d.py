"""Single-source 2D RCWA solve — Phase 3c of the rewrite.

Port of the triplicated ``transfer_2d_1..4`` chain into one backend-agnostic solve against the
Array-API namespace, sharing the boundary/branch conventions and the final s/p matching with
:mod:`wavebit._core.solve_1d`. The only per-backend primitive is
:func:`wavebit._core.backend.eig`.

The 2D layer couples both polarizations *and* both in-plane directions, so per-layer modes and
the boundary system are ``2*ff_xy`` block matrices (``ff_xy = ff_x*ff_y``). Optional off-axis
convolution blocks (``epxy`` / ``epyx``, produced by the NVF factorization; ``None`` = zeros)
enter the mode operator. ``transfer_2d_1`` and ``transfer_2d_4`` are identical to the conical
versions, so this module reuses ``solve_1d``'s boundary init and ``final_matching`` verbatim.
"""
from .backend import array_namespace, as_array, eig
from .conventions import boundary_kz, mode_q, order_azimuth
from .solve_1d import _block, _diag, _is_backside, _mirrored_scene, final_matching


def order_grid_kxy(xp, kx, ky):
    """``kx`` / ``ky`` on the ``(ff_y, ff_x)`` order grid, from either spelling.

    Two ways in, because a lattice basis decides which one is even expressible:

    * **1-D**, lengths ``(ff_x,)`` and ``(ff_y,)`` — the separable form an axis-aligned cell
      admits, where ``kx`` depends only on ``m`` and ``ky`` only on ``n``. Broadcast to the
      grid here, which is what both solves used to do inline.
    * **2-D**, ``(ff_y, ff_x)`` each — the per-order Cartesian components, which is the only
      form a skewed cell has: with ``G = m*b1 + n*b2`` non-orthogonal, *both* components
      depend on *both* indices and no pair of 1-D vectors can hold them. See
      :func:`wavebit._core.lattice.order_kxy`.

    Everything downstream reads the grid, so the two spellings converge here and the algebra
    below never branches. The 1-D route broadcasts exactly as before, element for element.
    """
    if kx.ndim == 2:
        return kx, ky
    ff_x, ff_y = kx.shape[0], ky.shape[0]
    return (xp.broadcast_to(kx[None, :], (ff_y, ff_x)),
            xp.broadcast_to(xp.reshape(ky, (-1, 1)), (ff_y, ff_x)))


def layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i, epxy_conv, epyx_conv, C,
                   mu=None):
    """Raw (unrotated) eigenmodes ``(W, V, q)`` of one 2D layer.

    ``W`` is the ``[Ex; Ey]`` mode matrix and ``q`` the modal decay constants; the principal
    square root keeps ``Re(q) >= 0`` so ``exp(-k0*q*d)`` never grows. Shared by the ETM
    (:func:`solve_2d`) and SMM (:mod:`wavebit._core.scattering`) connecting algorithms.

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

    ``mu`` is an optional ``(mux, muy, muz_i, muxy, muyx)`` permeability tuple in **the same
    five slots and the same convention as the permittivity** — ``muz_i`` inverted, the
    off-diagonals negated. Left ``None`` the layer is non-magnetic and every expression below
    is the one that was here before, unchanged, so a non-magnetic solve is bit-identical.

    Where it enters is a duality, not a patch. Under ``(E, H, eps, mu) -> (H, -E, mu, eps)``
    the two curl equations map onto each other, so ``Omega_L`` — which eliminates ``E_z``
    through ``eps_zz`` — carries the *transverse* ``mu``, and ``Omega_R`` — which eliminates
    ``H_z`` through ``mu_zz`` — carries the transverse ``eps``. Concretely the four blocks
    below each acquire one slot, uniformly, and ``Omega_R``'s three geometry products gain
    ``muz_i`` between their ``K`` factors::

        B  = Kx epz_i Kx - muy          Cc = Kx epz_i Ky - muyx
        D  = Ky epz_i Ky - mux          E  = Ky epz_i Kx - muxy

    Note the **cross** pairing: ``B`` takes ``mu_yy`` and ``D`` takes ``mu_xx``, not the other
    way round. ``docs/anisotropic-materials.md`` §5 pointed at the two ``- I`` terms and stopped
    there, which would have got the isotropic case right and silently transposed a uniaxial
    ``mu``. The placement was settled by requiring the derived ``Omega_L @ Omega_R`` to
    reproduce the shipped ``Omega2_LR`` term for term at ``mu = 1``.
    """
    mux, muy, muz_i, muxy, muyx = (I, I, None, None, None) if mu is None else mu

    B = Kx @ epz_conv_i @ Kx - muy
    D = Ky @ epz_conv_i @ Ky - mux
    Cc = Kx @ epz_conv_i @ Ky
    E = Ky @ epz_conv_i @ Kx
    if muyx is not None:
        Cc = Cc - muyx
    if muxy is not None:
        E = E - muxy

    # `Omega_R`'s three geometry products carry `muz_i` between the K factors, exactly as
    # `epz_conv_i` sits between them in `Omega_L`. Skipped entirely when non-magnetic rather
    # than multiplying an identity through, so the arithmetic stays the operations it was.
    if muz_i is None:
        KxKy, KxKx, KyKy, KyKx = Kx @ Ky, Kx @ Kx, Ky @ Ky, Ky @ Kx
    else:
        KxKy, KxKx = Kx @ muz_i @ Ky, Kx @ muz_i @ Kx
        KyKy, KyKx = Ky @ muz_i @ Ky, Ky @ muz_i @ Kx

    Omega_R = _block(xp, [
        [-KxKy + epyx_conv, KxKx - epy_conv],
        [epx_conv - KyKy, KyKx - epxy_conv],
    ])

    if mu is None:
        # The hand-expanded product. Its four bare ``K @ K`` terms are not geometry that
        # survived the multiplication untouched — they are what two *other* terms collapse to
        # when the K factors commute past a `muz_i` of one. Written out, the (0,0) block is
        #   Cc Kx muz_i Ky - B Ky muz_i Ky + B epx - Cc epyx,
        # where the first pair leaves `muy Ky muz_i Ky + muyx Kx muz_i Ky` behind, which is
        # `Ky @ Ky` only when `muy = 1`, `muyx = 0` and `muz_i = 1`. So this form cannot be
        # reused with a permeability, and reusing it is not a small error: it costs energy
        # conservation outright on a gyromagnetic layer (R + T came out 1.035).
        Omega2_LR = _block(xp, [
            [Ky @ Ky + B @ epx_conv - Cc @ epyx_conv, Cc @ epy_conv - Kx @ Ky - B @ epxy_conv],
            [E @ epx_conv - Ky @ Kx - D @ epyx_conv, Kx @ Kx + D @ epy_conv - E @ epxy_conv],
        ])
    else:
        Omega2_LR = _block(xp, [[-Cc, B], [-D, E]]) @ Omega_R

    w, W = eig(Omega2_LR)
    q = mode_q(xp, xp.astype(w, C))

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


def solve_2d(wavelength, kx, ky, n_top, n_bot, theta, incidence, thickness,
             epx_conv_all, epy_conv_all, epz_conv_i_all,
             epxy_conv_all=None, epyx_conv_all=None,
             *, phi, incidence_side='top', full=False, mu=None, mu_top=1.0, mu_bot=1.0):
    """Solve a 2D grating stack for the given incidence.

    ``incidence`` is the resolved incident amplitude pair ``(a_s, a_p)`` at unit flux — whatever
    when given — see :func:`~wavebit._core.conventions.incident_sp_amplitudes`.

    See :func:`wavebit._core.solve_1d.solve_1d` for the shared argument conventions —
    including ``incidence_side``, handled by the same mirror (the off-diagonal blocks
    reverse with the rest of the stack; a layer's in-plane tensor is untouched by ``z -> -z``).
    ``epxy_conv_all`` / ``epyx_conv_all`` are the optional off-diagonal permittivity blocks
    (``None`` -> zero).

    ``mu`` is the optional per-layer permeability, as five stacks
    ``(mux_all, muy_all, muz_i_all, muxy_all, muyx_all)`` in the same slots and the same
    convention as the permittivity — see :func:`layer_modes_2d`. Any entry may be ``None``.

    ``mu_top`` / ``mu_bot`` are the **half-spaces'** scalar permeabilities. ``n_top`` / ``n_bot``
    keep meaning the *refractive index* ``sqrt(eps*mu)``, so nothing about ``kz``, Snell or the
    order grid changes and every existing call means what it did; ``mu`` enters only through
    the boundary admittances (:func:`~wavebit._core.solve_1d.sp_efficiencies`).

    ``kx`` / ``ky`` are the retained in-plane orders, in either spelling
    :func:`order_grid_kxy` accepts: 1-D of lengths ``ff_x`` / ``ff_y`` for an axis-aligned
    cell, or ``(ff_y, ff_x)`` per-order Cartesian components for any lattice.

    Returns ``(de_ri, de_ti)`` for that incidence, shape ``(ff_y, ff_x)``.
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
        mu_top, mu_bot = mu_bot, mu_top      # the half-spaces swap with the indices they belong to
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
    varphi = order_azimuth(xp, ky_g, kx_g, phi)

    eps_bot = n_bot ** 2 / mu_bot          # n stays the refractive index; eps carries the mu
    Kz_bot = _diag(xp, kz_bot)

    # The substrate's two admittances: kz/eps on the magnetic (p) amplitude, kz/mu on the
    # electric (s) one. Both are the previous expression at mu_bot = 1.
    big_F = _block(xp, [[I, O], [O, -1j * Kz_bot / eps_bot]])
    big_G = _block(xp, [[-1j * Kz_bot / mu_bot, O], [O, I]])
    big_T = xp.astype(xp.eye(2 * ff_xy), C)


    # The order grid flattened row-major: m fastest, matching how the ``fourier2d_*`` builders
    # lay out the convolution matrices. Diagonal either way — the eigenproblem below reads
    # only these two diagonals, which is why a skewed lattice never reaches it.
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

        # `mux`/`muy` fall back to I and the other three to None, which is what
        # `layer_modes_2d` reads as non-magnetic -- so a stack with mu on some layers and not
        # others is expressible, and the layers without it take the original arithmetic.
        layer_mu = None if all(s is None for s in mu_all) else (
            I if mu_all[0] is None else mu_all[0][layer_index],
            I if mu_all[1] is None else mu_all[1][layer_index],
            *(None if s is None else s[layer_index] for s in mu_all[2:]))

        W, V, q = layer_modes_2d(xp, Kx, Ky, I, epx_conv, epy_conv, epz_conv_i,
                                 epxy_conv, epyx_conv, C, mu=layer_mu)

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
    out = final_matching(xp, C, big_F, big_G, big_T, kz_top, kz_bot, incidence, theta,
                         n_top, n_bot, ff_x, ff_y, full=full,
                         mu_top=mu_top, mu_bot=mu_bot)
    if full:
        out['layer_info_list'] = layer_info_list
    return out
