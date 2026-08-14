"""Single-source near-field reconstruction.

Fields come back in the same ``e^{-iωt}`` convention as everything else (see
:mod:`wavebit._core.conventions`): a wave travelling in ``+z`` carries ``exp(+i*k0*kz*z)``,
and the in-plane kernel matches the order-index sign the convolution matrices are built with.

Axes follow the array, not the picture: row ``0`` of the returned cell is ``y = 0``, the same
row of ``ucell`` that the factorization reads as ``y = 0``. Turning that into a conventional
image is :func:`field_plot`'s job, and it flips there.

**The vector components are in the library's own frame, where ``+z`` points down, into the
stack** (``docs/conventions.md``) — the same ``+z`` the phase, the layer order and the
``Im(kz) > 0`` decay rule are stated against. Until 2026-08-14 they were not: every path
returned ``E_z``, ``H_x`` and ``H_y`` with the opposite sign.

It is worth being precise about what that was, because it was not an undeclared convention.
``docs/formulation.md`` §1 already carried the answer as its sanity anchor — a vacuum wave
``E = y exp(i k0 z)`` travelling ``+z`` has ``Hx = -exp(i k0 z)``, the pair ``(Ey, Hx) =
(1, -kz)`` — and ``tests/test_anisotropic_berreman.py::_vacuum_modes`` writes that pair, and
:mod:`wavebit._core.solve_2d_4ff` derives in it. The formulation, the oracle and the solver
agreed; **only this module disagreed with them.** Measured on a three-layer slab at 33
degrees, the curl-equation residual was 2.8 as shipped and 8.0e-4 (converging like the step)
once the parity was undone, and a vacuum slab returned a Poynting vector pointing back out of
the stack.

The three signs are applied at each entry point below rather than in the mode algebra, because
the mode algebra is shared with the solve and the solve was never wrong. See
``tests/test_field_core.py::test_every_path_returns_a_maxwell_field_in_the_declared_frame``.

Turns a solved stack back into the field inside it. Each layer's mode amplitudes are
propagated to a set of z-slices, then inverse-Fourier-transformed onto an (x, y) grid.

Inputs are the ``layer_info_list`` and ``T1`` the solve leaves behind — see
:mod:`wavebit._core.solve_1d`. Layers arrive bottom-to-top and are walked in reverse, so the
returned cell runs top-to-bottom in z.

The entry points mirror the solve paths: :func:`field_1d` (pure TE/TM, 3 field components),
:func:`field_1d_conical` and :func:`field_2d` (coupled s/p, 6 components, with a leading axis
selecting the incidence variants requested via ``set_field_input``), and
:func:`field_2d_4ff` for the first-order path, which carries a different per-layer state and a
different ``E_z``. ``calculate_field`` picks between the last two on the *formulation*, not on
the grating type — the first-order path sets that to 2 as well.
"""
from .backend import array_namespace


def _dev(x):
    """Device of ``x``, so arrays the core creates land beside the ones it was given."""
    return x.device if hasattr(x, 'device') else None


def _require_solved(T1, layer_info_list):
    """Reject the states that cannot produce a field, with the reason.

    Both connecting algorithms fill this state (the SMM recovers it from its cascade), so an
    empty one means no solve has run — or a stack with no layers, which has no interior.
    """
    if T1 is None or len(layer_info_list) == 0:
        raise RuntimeError(
            'field reconstruction needs the per-layer amplitudes a solve leaves behind: '
            'run conv_solve (or solve) first, on a stack with at least one layer.')


def _diag_exp(xp, diag_vals, n):
    """``exp`` of a diagonal matrix stack, given its diagonals — the legacy ``d_exp``.

    Off-diagonal entries stay exactly zero (the legacy built this by assigning into a zero
    array; multiplying an identity achieves the same without in-place mutation).
    """
    eye = xp.eye(n, dtype=diag_vals.dtype, device=_dev(diag_vals))
    return eye * xp.exp(diag_vals)[..., None, :]


def _inv_fourier(xp, kx, ky, period, res_x, res_y, k0, C, lattice=None):
    """Inverse-transform kernel ``exp(-i k.r)`` sampled on the cell, flattened over orders.

    With ``lattice`` given (and ``kx`` / ``ky`` therefore per-order, ``(ff_y, ff_x)``) the
    samples are taken on the cell's own parallelogram, ``r = u*a1 + v*a2`` for ``u, v`` in
    ``[0, 1]``, and the returned grid is indexed by ``(v, u)`` rather than ``(y, x)``.

    Separability survives that, which is the whole reason this is a few lines rather than a
    rewrite::

        k.r = k.(u*a1 + v*a2) = u*(k.a1) + v*(k.a2)

    so the kernel is still an outer product, just of ``exp(i*k0*u*(k.a1))`` and
    ``exp(i*k0*v*(k.a2))`` instead of ``exp(i*kx*x)`` and ``exp(i*ky*y)``. Nothing extra has to
    be carried: the incident offset is already inside ``kx`` / ``ky``, so ``k.a1`` and ``k.a2``
    absorb it. On an axis-aligned cell ``k.a1 = kx*Lx`` and ``u = x/Lx``, which is the 1-D
    branch below, and that branch is left alone so its samples stay bit-identical.
    """
    dev = _dev(kx)
    # Branch on the *layout*, not on whether a basis was supplied: an axis-aligned cell may
    # carry an explicit `diag(period)` and still hand over separable 1-D vectors, in which case
    # the old path is both correct and bit-identical.
    if kx.ndim == 2:
        a1x, a2x = float(lattice[0][0]), float(lattice[0][1])
        a1y, a2y = float(lattice[1][0]), float(lattice[1][1])
        kx_f = xp.reshape(kx, (-1,))
        ky_f = xp.reshape(ky, (-1,))
        proj_a1 = (kx_f * a1x + ky_f * a1y) * k0          # k . a1, per order
        proj_a2 = (kx_f * a2x + ky_f * a2y) * k0          # k . a2
        u_1d = xp.linspace(0, 1, res_x, dtype=C, device=dev)
        v_1d = xp.linspace(0, 1, res_y, dtype=C, device=dev)
        u_ph = xp.exp(1j * xp.reshape(u_1d, (1, res_x, 1)) * proj_a1)
        v_ph = xp.exp(1j * xp.reshape(v_1d, (res_y, 1, 1)) * proj_a2)
        return xp.broadcast_to(u_ph * v_ph, (res_y, res_x, kx_f.shape[0]))
    # `period` stays an array rather than a Python float: all three backends take an array
    # endpoint and give bit-identical samples, and narrowing it would break tracing whenever
    # the solver itself is jitted (period is a traced child of the JAX solver pytree).
    x_1d = xp.linspace(0, period[0], res_x, dtype=C, device=dev)
    x_2d = xp.reshape(x_1d, (1, res_x, 1)) * kx * k0
    x_2d = xp.reshape(xp.broadcast_to(x_2d, (res_y, res_x, kx.shape[0])),
                      (res_y, res_x, 1, kx.shape[0]))
    if ky is None:
        return xp.reshape(xp.exp(1j * x_2d), (res_y, res_x, -1))

    y_1d = xp.linspace(0, period[1], res_y, dtype=C, device=dev)
    y_2d = xp.reshape(y_1d, (res_y, 1, 1)) * ky * k0
    y_2d = xp.reshape(xp.broadcast_to(y_2d, (res_y, res_x, ky.shape[0])),
                      (res_y, res_x, ky.shape[0], 1))
    return xp.reshape(xp.exp(1j * x_2d) * xp.exp(1j * y_2d), (res_y, res_x, -1))


def _z_slices(xp, res_z, d, C, dev=None):
    """Sample points through one layer's thickness, shaped for broadcasting over modes."""
    return xp.reshape(xp.linspace(0, res_z, res_z, dtype=C, device=dev), (-1, 1, 1)) / res_z * d


def field_1d(wavelength, kx, T1, layer_info_list, period, tetm_index,
             res_x=20, res_y=1, res_z=20):
    """Field of a pure TE/TM 1D stack. Returns ``(res_z*n_layers, res_y, res_x, 3)``."""
    _require_solved(T1, layer_info_list)
    xp = array_namespace(T1)
    C = T1.dtype
    k0 = 2 * 3.141592653589793 / wavelength
    ff = kx.shape[0]
    Kx = xp.diag(kx)

    T_layer = T1
    cells = []
    for epz_conv_i, W, V, q, d, A_i, B in layer_info_list[::-1]:
        X = xp.diag(xp.exp(-k0 * q * d))
        c1 = T_layer[:, None]
        c2 = B @ A_i @ X @ T_layer[:, None]
        z = _z_slices(xp, res_z, d, C, _dev(T1))
        Qz = xp.reshape(q, (1, 1, ff)) * z            # diagonals of Q*z, shape (res_z, 1, ff)
        Qz = xp.reshape(Qz, (res_z, ff))
        Qd = xp.reshape(q, (1, ff)) * xp.reshape(xp.reshape(z, (res_z,)) - d, (res_z, 1))

        fwd = _diag_exp(xp, -k0 * Qz, ff)
        bwd = _diag_exp(xp, k0 * Qd, ff)
        My = W @ (fwd @ c1 + bwd @ c2)
        Mx = V @ (-fwd @ c1 + bwd @ c2)
        Mz = 1j * (epz_conv_i @ Kx @ My if tetm_index else Kx @ My)

        inv_f = _inv_fourier(xp, kx, None, period, res_x, res_y, k0, C)
        sgn = -1j if tetm_index == 0 else 1j
        Fy = inv_f[:, :, None, :] @ My[:, None, None, :, :]
        Fx = sgn * (inv_f[:, :, None, :] @ Mx[:, None, None, :, :])
        Fz = sgn * (inv_f[:, :, None, :] @ Mz[:, None, None, :, :])
        # Into the declared frame (module docstring): E_z, H_x, H_y flip, E_t and H_z do not.
        # The two layouts fill the same three slots with different components, so the pattern
        # is not the same one twice -- TE is (Ey, Hx, Hz) and only its middle slot is magnetic
        # and transverse; TM is (Hy, Ex, Ez) and its *outer* two are what the parity moves.
        par_y, par_x, par_z = (1, -1, 1) if tetm_index == 0 else (-1, 1, -1)
        cells.append(xp.concat([par_y * Fy[..., 0], par_x * Fx[..., 0],
                                par_z * Fz[..., 0]], axis=-1))

        T_layer = A_i @ X @ T_layer

    return xp.concat(cells, axis=0)


def _field_coupled(wavelength, kx, ky, T1, layer_info_list, period, res_x, res_y, res_z,
                   set_field_input, conical, lattice=None):
    """Shared body of the conical and 2D reconstructions.

    They differ only in how the layer's mode matrices assemble into ``(Sx, Sy, Ux, Uy)``; the
    propagation, the z-sampling and the inverse transform are identical.
    """
    _require_solved(T1, layer_info_list)
    xp = array_namespace(T1)
    C = T1.dtype
    k0 = 2 * 3.141592653589793 / wavelength
    if kx.ndim == 2:
        ff_y, ff_x = kx.shape
    else:
        ff_x, ff_y = kx.shape[0], ky.shape[0]
    ff_xy = ff_x * ff_y

    # Same two spellings the solve accepts, resolved by the same helper — 1-D separable for an
    # axis-aligned cell (broadcast exactly as before), per-order for any lattice.
    from .solve_2d import order_grid_kxy
    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    Kx, Ky = xp.diag(xp.reshape(kx_g, (-1,))), xp.diag(xp.reshape(ky_g, (-1,)))

    dev = _dev(T1)
    keep = [i for i, on in enumerate(set_field_input) if on]
    T_layer = xp.take(T1, xp.asarray(keep, device=dev), axis=0)
    big_I = xp.eye(T1.shape[1], dtype=C, device=dev)

    cells = []
    for epz_conv_i, W, V, q, d, big_A_i, big_B in layer_info_list[::-1]:
        q_1, q_2 = q[:ff_xy], q[ff_xy:]
        big_X = xp.diag(xp.exp(-k0 * q * d))

        c = xp.concat([big_I, big_B @ big_A_i @ big_X], axis=0) @ T_layer
        c1p, c2p = c[:, 0:ff_xy], c[:, ff_xy:2 * ff_xy]
        c1m, c2m = c[:, 2 * ff_xy:3 * ff_xy], c[:, 3 * ff_xy:4 * ff_xy]

        z = xp.reshape(xp.linspace(0, res_z, res_z, dtype=C, device=dev), (-1,)) / res_z * d
        def prop(qi):
            fwd = _diag_exp(xp, -k0 * xp.reshape(qi, (1, ff_xy)) * xp.reshape(z, (res_z, 1)), ff_xy)
            bwd = _diag_exp(xp, k0 * xp.reshape(qi, (1, ff_xy)) * xp.reshape(z - d, (res_z, 1)), ff_xy)
            return fwd[:, None, :, :], bwd[:, None, :, :]

        f1, b1 = prop(q_1)
        f2, b2 = prop(q_2)
        a1p, a1m = f1 @ c1p, b1 @ c1m          # + / - going, mode family 1
        a2p, a2m = f2 @ c2p, b2 @ c2m

        if conical:
            W_1, W_2 = W[:, :ff_xy], W[:, ff_xy:]
            V_11, V_12 = V[:ff_xy, :ff_xy], V[:ff_xy, ff_xy:]
            V_21, V_22 = V[ff_xy:, :ff_xy], V[ff_xy:, ff_xy:]
            Sx = W_2 @ (a2p + a2m)
            Sy = V_11 @ (a1p + a1m) + V_12 @ (a2p + a2m)
            Ux = W_1 @ (-a1p + a1m)
            Uy = V_21 @ (-a1p + a1m) + V_22 @ (-a2p + a2m)
        else:
            W_11, W_12 = W[:ff_xy, :ff_xy], W[:ff_xy, ff_xy:]
            W_21, W_22 = W[ff_xy:, :ff_xy], W[ff_xy:, ff_xy:]
            V_11, V_12 = V[:ff_xy, :ff_xy], V[:ff_xy, ff_xy:]
            V_21, V_22 = V[ff_xy:, :ff_xy], V[ff_xy:, ff_xy:]
            Sx = W_11 @ (a1p + a1m) + W_12 @ (a2p + a2m)
            Sy = W_21 @ (a1p + a1m) + W_22 @ (a2p + a2m)
            Ux = V_11 @ (-a1p + a1m) + V_12 @ (-a2p + a2m)
            Uy = V_21 @ (-a1p + a1m) + V_22 @ (-a2p + a2m)

        Sz = 1j * epz_conv_i @ (Kx @ Uy - Ky @ Ux)
        Uz = 1j * (Kx @ Sy - Ky @ Sx)

        inv_f = _inv_fourier(xp, kx, ky, period, res_x, res_y, k0, C,
                             lattice)[:, :, None, :]
        # `-1j` converts out of `U = i H`; the three minus signs on top of it are the parity
        # into the declared frame (module docstring), so `H_z` keeps the bare conversion.
        comps = [inv_f @ M[:, :, None, None, :, :] for M in (Sx, Sy, -Sz)]
        comps += [-1j * (inv_f @ M[:, :, None, None, :, :]) for M in (-Ux, -Uy, Uz)]
        val = xp.concat([c[..., 0] for c in comps], axis=-1)
        cells.append(xp.moveaxis(val, 1, 0))

        T_layer = big_A_i @ big_X @ T_layer

    return xp.concat(cells, axis=1)


def field_2d_4ff(wavelength, kx, ky, T1, layer_info_list, period,
                 res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                 lattice=None):
    """Field of a stack solved on the first-order (4ff) path. Same shape as :func:`field_2d`.

    The transverse half is *simpler* than the second-order reconstruction, not harder. There
    the layer's state has to be reassembled from ``W`` (the transverse ``E``) and ``V`` (its
    partner) with a sign per direction; here the eigenvector **is** the state — reading
    ``(Ex, Ey, Hx, Hy)`` off the four row blocks of ``psi`` is the whole of it. The two
    directions are separate mode sets rather than one set with a sign, so each is propagated
    from its own reference face: forward amplitudes are carried at the layer's top, backward
    ones at its bottom, which is the same split that makes the stack solve stable
    (:func:`wavebit._core.solve_2d_4ff._propagate_s`).

    ``E_z`` is the one component that cannot be carried over. Eliminating it from the curl
    equations with a **populated** permittivity ``z`` row keeps two terms the second-order form
    never has::

        Ez = ezz^-1 [ (Ky Hx - Kx Hy) - ezx Ex - ezy Ey ]

    :func:`_field_coupled`'s ``Sz = 1j * epz_conv_i @ (Kx @ Uy - Ky @ Ux)`` is that expression
    at ``ezx = ezy = 0`` — the same line with ``U = i H`` substituted — so the 2ff spelling is a
    special case of this one, not a shared one. ``H_z = Kx Ey - Ky Ex`` is unchanged while
    ``mu = 1``, which is all this path admits.

    The returned components are ``(Ex, Ey, Ez, Hx, Hy, Hz)``, matching :func:`field_2d`. This
    path needs **no** parity at the output, and it is the only one that does not:
    :mod:`wavebit._core.solve_2d_4ff` already derives everything with ``+z`` down along the
    propagation direction — which is what makes ``Im(lam_f) >= 0`` mean "does not grow" — and
    that is the library's declared frame. The state vector is therefore already in it.

    This is the reverse of how it shipped for one day. When reconstruction landed on this path
    (2026-08-14) the other three paths were returning ``E_z``, ``H_x``, ``H_y`` in the opposite
    frame, so this one applied the parity to match them rather than break a caller who flips
    ``formulation``. The vacuum slab that measurement rested on is what then showed the other
    three were wrong rather than merely different, and the parity moved to where it belongs.
    """
    _require_solved(T1, layer_info_list)
    xp = array_namespace(T1)
    C = T1.dtype
    k0 = 2 * 3.141592653589793 / wavelength
    if kx.ndim == 2:
        ff_y, ff_x = kx.shape
    else:
        ff_x, ff_y = kx.shape[0], ky.shape[0]
    ff = ff_x * ff_y

    from .solve_2d import order_grid_kxy
    kx_g, ky_g = order_grid_kxy(xp, kx, ky)
    Kx, Ky = xp.diag(xp.reshape(kx_g, (-1,))), xp.diag(xp.reshape(ky_g, (-1,)))

    dev = _dev(T1)
    keep = [i for i, on in enumerate(set_field_input) if on]
    T_layer = xp.take(T1, xp.asarray(keep, device=dev), axis=0)

    cells = []
    for eps, psi_f, lam_f, psi_b, lam_b, d, P_i, QP_i in layer_info_list[::-1]:
        # The layer's unknown is its forward amplitude at its own top face -- that is what the
        # ETM redefinition `t' = X_f^-1 P t` bought. The backward half is referenced at the
        # bottom face, so `QP_i @ X_f` is the map to it.
        X_f = _diag_exp(xp, 1j * k0 * lam_f * d, 2 * ff)
        a_f = T_layer
        a_b = (QP_i @ X_f) @ T_layer

        z = xp.reshape(xp.linspace(0, res_z, res_z, dtype=C, device=dev), (-1,)) / res_z * d
        fwd = _diag_exp(xp, 1j * k0 * xp.reshape(lam_f, (1, 2 * ff))
                        * xp.reshape(z, (res_z, 1)), 2 * ff)
        bwd = _diag_exp(xp, 1j * k0 * xp.reshape(lam_b, (1, 2 * ff))
                        * xp.reshape(z - d, (res_z, 1)), 2 * ff)
        psi = psi_f @ (fwd[:, None] @ a_f) + psi_b @ (bwd[:, None] @ a_b)

        Ex, Ey = psi[:, :, :ff], psi[:, :, ff:2 * ff]
        Hx, Hy = psi[:, :, 2 * ff:3 * ff], psi[:, :, 3 * ff:]
        Ez = eps['ezz_i'] @ (Ky @ Hx - Kx @ Hy - eps['ezx'] @ Ex - eps['ezy'] @ Ey)
        Hz = Kx @ Ey - Ky @ Ex

        inv_f = _inv_fourier(xp, kx, ky, period, res_x, res_y, k0, C,
                             lattice)[:, :, None, :]
        # No parity here, deliberately: the 4ff state vector is already in the declared frame.
        comps = [inv_f @ M[:, :, None, None, :, :]
                 for M in (Ex, Ey, Ez, Hx, Hy, Hz)]
        val = xp.concat([c[..., 0] for c in comps], axis=-1)
        cells.append(xp.moveaxis(val, 1, 0))

        T_layer = (P_i @ X_f) @ T_layer

    return xp.concat(cells, axis=1)


def field_1d_conical(wavelength, kx, ky, T1, layer_info_list, period,
                     res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                     lattice=None):
    """Field of a general 1D (conical) stack. ``(n_selected, res_z*n_layers, res_y, res_x, 6)``."""
    return _field_coupled(wavelength, kx, ky, T1, layer_info_list, period,
                          res_x, res_y, res_z, set_field_input, conical=True, lattice=lattice)


def field_2d(wavelength, kx, ky, T1, layer_info_list, period,
             res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
             lattice=None):
    """Field of a 2D stack. ``(n_selected, res_z*n_layers, res_y, res_x, 6)``."""
    return _field_coupled(wavelength, kx, ky, T1, layer_info_list, period,
                          res_x, res_y, res_z, set_field_input, conical=False, lattice=lattice)


def field_plot(field_cell, tetm_index=0, plot_indices=(1, 1, 1, 1, 1, 1), y_slice=0, z_slice=-1,
               zx=True, yx=True):
    """Plot |field|^2 slices of a reconstructed cell (side and/or top view).

    Presentation rather than physics, but there is no reason for it to exist per backend:
    it only needs the array as NumPy.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('field_plot() needs matplotlib installed') from e
    import numpy as np

    field_cell = np.asarray(field_cell)

    if field_cell.shape[-1] == 6:                     # 2D / conical grating
        title = ['2D Ex', '2D Ey', '2D Ez', '2D Hx', '2D Hy', '2D Hz']
    elif tetm_index == 0:                             # 1D TE
        title = ['1D Ey', '1D Hx', '1D Hz']
    else:                                             # 1D TM
        title = ['1D Hy', '1D Ex', '1D Ez']

    for enabled, sl, view, ylabel in ((zx, (slice(None), y_slice), 'Side View', 'Z'),
                                      (yx, (z_slice,), 'Top View', 'Y')):
        if not enabled:
            continue
        for idx, name in enumerate(title):
            if not plot_indices[idx]:
                continue
            image = abs(field_cell[sl + (slice(None), idx)]) ** 2
            if ylabel == 'Y':
                # the cell's y runs with the array index; an image reads top-down, so the
                # flip that used to live in the reconstruction lives here instead
                image = image[::-1]
            plt.imshow(image, cmap='jet', aspect='auto')
            plt.colorbar()
            plt.title(f'{name}, {view}')
            plt.xlabel('X')
            plt.ylabel(ylabel)
            plt.show()
