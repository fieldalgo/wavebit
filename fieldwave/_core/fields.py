"""Single-source near-field reconstruction.

Turns a solved stack back into the field inside it. Each layer's mode amplitudes are
propagated to a set of z-slices, then inverse-Fourier-transformed onto an (x, y) grid.

Inputs are the ``layer_info_list`` and ``T1`` the solve leaves behind — see
:mod:`fieldwave._core.solve_1d`. Layers arrive bottom-to-top and are walked in reverse, so the
returned cell runs top-to-bottom in z.

The three entry points mirror the three solve paths: :func:`field_1d` (pure TE/TM, 3 field
components), :func:`field_1d_conical` and :func:`field_2d` (coupled s/p, 6 components, with a
leading axis selecting the incidence variants requested via ``set_field_input``).
"""
from .backend import array_namespace


def _dev(x):
    """Device of ``x``, so arrays the core creates land beside the ones it was given."""
    return x.device if hasattr(x, 'device') else None


def _diag_exp(xp, diag_vals, n):
    """``exp`` of a diagonal matrix stack, given its diagonals — the legacy ``d_exp``.

    Off-diagonal entries stay exactly zero (the legacy built this by assigning into a zero
    array; multiplying an identity achieves the same without in-place mutation).
    """
    eye = xp.eye(n, dtype=diag_vals.dtype, device=_dev(diag_vals))
    return eye * xp.exp(diag_vals)[..., None, :]


def _inv_fourier(xp, kx, ky, period, res_x, res_y, k0, C):
    """Inverse-transform kernel ``exp(-i k.r)`` sampled on the (x, y) grid, flattened over orders."""
    dev = _dev(kx)
    # `period` stays an array rather than a Python float: all three backends take an array
    # endpoint and give bit-identical samples, and narrowing it would break tracing whenever
    # the solver itself is jitted (period is a traced child of the JAX solver pytree).
    x_1d = xp.linspace(0, period[0], res_x, dtype=C, device=dev)
    x_2d = xp.reshape(x_1d, (1, res_x, 1)) * kx * k0
    x_2d = xp.reshape(xp.broadcast_to(x_2d, (res_y, res_x, kx.shape[0])),
                      (res_y, res_x, 1, kx.shape[0]))
    if ky is None:
        return xp.reshape(xp.exp(-1j * x_2d), (res_y, res_x, -1))

    y_1d = xp.flip(xp.linspace(0, period[1], res_y, dtype=C, device=dev))
    y_2d = xp.reshape(y_1d, (res_y, 1, 1)) * ky * k0
    y_2d = xp.reshape(xp.broadcast_to(y_2d, (res_y, res_x, ky.shape[0])),
                      (res_y, res_x, ky.shape[0], 1))
    return xp.reshape(xp.exp(-1j * x_2d) * xp.exp(-1j * y_2d), (res_y, res_x, -1))


def _z_slices(xp, res_z, d, C, dev=None):
    """Sample points through one layer's thickness, shaped for broadcasting over modes."""
    return xp.reshape(xp.linspace(0, res_z, res_z, dtype=C, device=dev), (-1, 1, 1)) / res_z * d


def field_1d(wavelength, kx, T1, layer_info_list, period, pol,
             res_x=20, res_y=1, res_z=20):
    """Field of a pure TE/TM 1D stack. Returns ``(res_z*n_layers, res_y, res_x, 3)``."""
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
        Mz = -1j * (epz_conv_i @ Kx @ My if pol else Kx @ My)

        inv_f = _inv_fourier(xp, kx, None, period, res_x, res_y, k0, C)
        sgn = 1j if pol == 0 else -1j
        Fy = inv_f[:, :, None, :] @ My[:, None, None, :, :]
        Fx = sgn * (inv_f[:, :, None, :] @ Mx[:, None, None, :, :])
        Fz = sgn * (inv_f[:, :, None, :] @ Mz[:, None, None, :, :])
        cells.append(xp.concat([Fy[..., 0], Fx[..., 0], Fz[..., 0]], axis=-1))

        T_layer = A_i @ X @ T_layer

    return xp.concat(cells, axis=0)


def _field_coupled(wavelength, kx, ky, T1, layer_info_list, period, res_x, res_y, res_z,
                   set_field_input, conical):
    """Shared body of the conical and 2D reconstructions.

    They differ only in how the layer's mode matrices assemble into ``(Sx, Sy, Ux, Uy)``; the
    propagation, the z-sampling and the inverse transform are identical.
    """
    xp = array_namespace(T1)
    C = T1.dtype
    k0 = 2 * 3.141592653589793 / wavelength
    ff_x, ff_y = kx.shape[0], ky.shape[0]
    ff_xy = ff_x * ff_y

    kx_full = xp.reshape(xp.broadcast_to(kx[None, :], (ff_y, ff_x)), (-1,))
    ky_full = xp.reshape(xp.broadcast_to(xp.reshape(ky, (-1, 1)), (ff_y, ff_x)), (-1,))
    Kx, Ky = xp.diag(kx_full), xp.diag(ky_full)

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

        Sz = -1j * epz_conv_i @ (Kx @ Uy - Ky @ Ux)
        Uz = -1j * (Kx @ Sy - Ky @ Sx)

        inv_f = _inv_fourier(xp, kx, ky, period, res_x, res_y, k0, C)[:, :, None, :]
        comps = [inv_f @ M[:, :, None, None, :, :] for M in (Sx, Sy, Sz)]
        comps += [1j * (inv_f @ M[:, :, None, None, :, :]) for M in (Ux, Uy, Uz)]
        val = xp.concat([c[..., 0] for c in comps], axis=-1)
        cells.append(xp.moveaxis(val, 1, 0))

        T_layer = big_A_i @ big_X @ T_layer

    return xp.concat(cells, axis=1)


def field_1d_conical(wavelength, kx, ky, T1, layer_info_list, period,
                     res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False)):
    """Field of a general 1D (conical) stack. ``(n_selected, res_z*n_layers, res_y, res_x, 6)``."""
    return _field_coupled(wavelength, kx, ky, T1, layer_info_list, period,
                          res_x, res_y, res_z, set_field_input, conical=True)


def field_2d(wavelength, kx, ky, T1, layer_info_list, period,
             res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False)):
    """Field of a 2D stack. ``(n_selected, res_z*n_layers, res_y, res_x, 6)``."""
    return _field_coupled(wavelength, kx, ky, T1, layer_info_list, period,
                          res_x, res_y, res_z, set_field_input, conical=False)


def field_plot(field_cell, pol=0, plot_indices=(1, 1, 1, 1, 1, 1), y_slice=0, z_slice=-1,
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
    elif pol == 0:                                    # 1D TE
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
            plt.imshow(abs(field_cell[sl + (slice(None), idx)]) ** 2, cmap='jet', aspect='auto')
            plt.colorbar()
            plt.title(f'{name}, {view}')
            plt.xlabel('X')
            plt.ylabel(ylabel)
            plt.show()
