import numpy as np
from .fourier_analysis import dfs2d, circulant
from .primitives import meeinv


def get_normal_vector_field_analytical(grid_shape, shapes, period=(1000.0, 1000.0)):
    """
    Compute analytical unit normal vector field (nx, ny) from geometric shape definitions.
    Supports circles, ellipses, and rotated rectangles.
    """
    Ny, Nx = grid_shape
    Py, Px = period

    x = (np.arange(Nx) + 0.5) * (Px / Nx)
    y = (np.arange(Ny) + 0.5) * (Py / Ny)
    X, Y = np.meshgrid(x, y)

    nx_total = np.zeros((Ny, Nx), dtype=float)
    ny_total = np.zeros((Ny, Nx), dtype=float)

    for s in shapes:
        stype = s.get('type', 'ellipse')
        cx, cy = s.get('cx', Px / 2.0), s.get('cy', Py / 2.0)
        lx, ly = s.get('lx', 500.0), s.get('ly', 500.0)
        angle = s.get('angle', 0.0)

        dx = X - cx
        dy = Y - cy

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a

        if stype in ('ellipse', 'circle'):
            a = lx / 2.0
            b = ly / 2.0
            gxr = 2.0 * xr / (a ** 2)
            gyr = 2.0 * yr / (b ** 2)

            gx = gxr * cos_a - gyr * sin_a
            gy = gxr * sin_a + gyr * cos_a

            norm = np.hypot(gx, gy)
            norm[norm == 0] = 1.0

            nx_total += gx / norm
            ny_total += gy / norm
        elif stype == 'rectangle':
            a = lx / 2.0
            b = ly / 2.0
            dist_x = np.abs(xr) - a
            dist_y = np.abs(yr) - b
            mask_x = dist_x >= dist_y
            
            gxr = np.zeros_like(xr)
            gyr = np.zeros_like(yr)
            gxr[mask_x] = np.sign(xr[mask_x])
            gyr[~mask_x] = np.sign(yr[~mask_x])

            gx = gxr * cos_a - gyr * sin_a
            gy = gxr * sin_a + gyr * cos_a

            norm = np.hypot(gx, gy)
            norm[norm == 0] = 1.0

            nx_total += gx / norm
            ny_total += gy / norm

    norm_t = np.hypot(nx_total, ny_total)
    norm_t[norm_t == 0] = 1.0
    return nx_total / norm_t, ny_total / norm_t


def get_normal_vector_field_raster(eps_matrix, method='sdf', shapes=None, period=(1000.0, 1000.0)):
    """
    Compute unit normal vector field (nx, ny) from a 2D dielectric spatial map eps_matrix.
    Supports 'sdf' (Signed Distance Field), 'analytical' (Continuous geometric equations),
    and 'sobel' (Discrete Sobel gradient operators).
    """
    if method == 'analytical' and shapes is not None:
        return get_normal_vector_field_analytical(eps_matrix.shape, shapes, period=period)

    if method == 'sdf':
        from scipy.ndimage import distance_transform_edt
        # Find material boundary interfaces
        border = np.zeros(eps_matrix.shape, dtype=bool)
        border[1:, :] |= (eps_matrix[1:, :] != eps_matrix[:-1, :])
        border[:-1, :] |= (eps_matrix[:-1, :] != eps_matrix[1:, :])
        border[:, 1:] |= (eps_matrix[:, 1:] != eps_matrix[:, :-1])
        border[:, :-1] |= (eps_matrix[:, :-1] != eps_matrix[:, 1:])

        # Compute continuous distance transform map (Fast NVF)
        dist_map = distance_transform_edt(~border)

        # Spatial gradient of continuous distance field
        gy, gx = np.gradient(dist_map)
        norm = np.hypot(gx, gy)
        norm[norm == 0] = 1.0

        nx = gx / norm
        ny = gy / norm
        return nx, ny

    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float) / 8.0

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=float) / 8.0

    # Periodic boundary padding
    eps_padded = np.pad(eps_matrix.real, ((1, 1), (1, 1)), mode='wrap')

    # Compute spatial gradients
    grad_x = (
        eps_padded[1:-1, 2:] * sobel_x[1, 2] + eps_padded[1:-1, :-2] * sobel_x[1, 0] +
        eps_padded[:-2, 2:] * sobel_x[0, 2] + eps_padded[:-2, :-2] * sobel_x[0, 0] +
        eps_padded[2:, 2:] * sobel_x[2, 2] + eps_padded[2:, :-2] * sobel_x[2, 0]
    )

    grad_y = (
        eps_padded[2:, 1:-1] * sobel_y[2, 1] + eps_padded[:-2, 1:-1] * sobel_y[0, 1] +
        eps_padded[2:, :-2] * sobel_y[2, 0] + eps_padded[:-2, :-2] * sobel_y[0, 0] +
        eps_padded[2:, 2:] * sobel_y[2, 2] + eps_padded[:-2, 2:] * sobel_y[0, 2]
    )

    norm = np.hypot(grad_x, grad_y)
    mask = norm > 1e-8

    nx = np.zeros_like(grad_x)
    ny = np.zeros_like(grad_y)

    nx[mask] = grad_x[mask] / norm[mask]
    ny[mask] = grad_y[mask] / norm[mask]

    return nx, ny


def to_conv_mat_nvf_2d(eps_matrix, fto_x, fto_y, nx=None, ny=None, type_complex=np.complex128, use_pinv=False):
    """
    Compute Normal Vector Field (NVF) Fourier convolution matrices for the anisotropic
    in-plane permittivity operator, using the Li/Schuster normal-vector factorization.

    The five returned matrices plug directly into the ``transfer_2d_2`` slots
    ``(epx_conv, epy_conv, epxy_conv, epyx_conv, epz_conv_i)`` via
    ``to_conv_mat_raster_nvf``. The slot convention required by ``transfer_2d_2``
    was reverse-derived from its eigen-operators (``Omega2_LR = -P_E @ P_H``,
    ``Omega_R = P_H``, verified to machine precision) and is::

        epx  = eps_xx            epy  = eps_yy
        epxy = -eps_xy           epyx = -eps_yx          epz_i = inv(<eps>)

    where ``[[eps_xx, eps_xy],[eps_yx, eps_yy]]`` is the physical in-plane tensor.

    Construction (Hermitian projector form). With
    ``E = <eps>`` (Laurent / direct rule, tangential),
    ``A = <1/eps>^-1`` (inverse rule, normal), ``Delta = E - A``, and the normal
    projector blocks ``C_xx = <nx^2>``, ``C_xy = <nx ny>``, ``C_yy = <ny^2>``
    (these satisfy ``C_xx + C_yy = I`` since ``nx^2 + ny^2 = 1``), the physical
    tensor is::

        eps_xx = E - 1/2 (Delta @ C_xx + C_xx @ Delta)
        eps_yy = E - 1/2 (Delta @ C_yy + C_yy @ Delta)
        eps_xy = eps_yx = -1/2 (Delta @ C_xy + C_xy @ Delta)

    The 1/2-symmetrization is essential: the plain product ``Delta @ C`` is not
    Hermitian, which yields complex eigenvalues ``q^2`` and gross energy-conservation
    violation for curved boundaries. Symmetrizing restores a Hermitian operator
    (hence real ``q^2``), recovering ``|R+T-1| -> 0`` and converging faster than the
    standard DFS path for smooth interfaces (see ``benchmarks/benchmark_nvf.py``).
    """
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1
    ff_xy = ff_x * ff_y

    eps_matrix = eps_matrix.astype(type_complex)

    if nx is None or ny is None:
        nx, ny = get_normal_vector_field_raster(eps_matrix)

    C_xx_grid = nx ** 2
    C_xy_grid = nx * ny
    C_yy_grid = ny ** 2

    # Direct Fourier matrix for eps and 1/eps
    ep_dir = dfs2d(eps_matrix, 1, 1, fto_x, fto_y, type_complex=type_complex)
    ep_inv = dfs2d(1.0 / eps_matrix, 1, 1, fto_x, fto_y, type_complex=type_complex)

    ep_inv_i = meeinv(ep_inv, use_pinv=use_pinv)
    delta_ep = ep_dir - ep_inv_i

    # Fourier matrices for normal vector projection components
    C_xx = dfs2d(C_xx_grid.astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)
    C_xy = dfs2d(C_xy_grid.astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)
    C_yy = dfs2d(C_yy_grid.astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)

    # Hermitian (symmetrized) projector factorization -> physical in-plane tensor.
    eps_xx = ep_dir - 0.5 * (delta_ep @ C_xx + C_xx @ delta_ep)
    eps_yy = ep_dir - 0.5 * (delta_ep @ C_yy + C_yy @ delta_ep)
    eps_xy = -0.5 * (delta_ep @ C_xy + C_xy @ delta_ep)  # == eps_yx (symmetric)

    # Map physical tensor onto transfer_2d_2 slots: epxy = -eps_xy, epyx = -eps_yx.
    Exx = eps_xx
    Eyy = eps_yy
    Exy = -eps_xy
    Eyx = -eps_xy  # eps_yx == eps_xy
    Epz_i = meeinv(ep_dir, use_pinv=use_pinv)  # inv(<eps>), NOT <1/eps>

    return Exx, Eyy, Exy, Eyx, Epz_i
