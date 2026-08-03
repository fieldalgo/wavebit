"""
fourier_analytic.py — analytic (staircase-free) Fourier coefficients of smooth primitives.

Standard fieldwave modeling represents every non-axis-aligned shape as a *staircase* of
axis-aligned rectangles.  That injects a purely geometric truncation error that only
vanishes with very high Fourier order / very fine rasters, and it deprives the NVF
factorization of a true smooth boundary to work with.

This module instead supplies the **exact** 2D Fourier coefficients of the true shape,
so the solver sees the real oblique / curved interface directly:

* rotated rectangle : axis-aligned ``sinc`` evaluated in rotated frequency coordinates,
                      times a center phase (no ``n_split``).
* circle / disk     : analytic transform of a disk (``jinc`` / Bessel ``J1``).
* ellipse           : affine-transformed disk.

The coefficient convention matches ``fourier_analysis.dfs2d`` (verified to machine
precision), so the resulting Laurent (Toeplitz) matrices plug straight into the same
``transfer_2d_2`` slots the NVF path uses.  See ``to_conv_mat_analytic_nvf`` and
``ANALYTIC_FINDINGS.md``.

Coefficient definition (same as ``dfs2d``)::

    c(ky, kx) = 1/(Px*Py) * integral_cell f(x, y) exp(-i 2pi (ky y/Py + kx x/Px)) dx dy

so with u = 2pi kx/Px, v = 2pi ky/Py the integral is the continuous 2D Fourier
transform of ``f`` sampled on the reciprocal grid.
"""
import numpy as np
from scipy.special import j1

from .fourier_analysis import dfs2d
from .nvf import get_normal_vector_field_analytical
from .primitives import meeinv


def _sinc(z):
    """Unnormalized sinc, ``sin(z)/z`` with the removable singularity filled at 0."""
    z = np.asarray(z)
    out = np.ones_like(z, dtype=float if not np.iscomplexobj(z) else complex)
    nz = z != 0
    out[nz] = np.sin(z[nz]) / z[nz]
    return out


def _jinc(kap):
    """``2 J1(kap) / kap`` with the removable singularity filled at 0 (value 1)."""
    kap = np.asarray(kap, dtype=float)
    out = np.ones_like(kap)
    nz = kap != 0
    out[nz] = 2.0 * j1(kap[nz]) / kap[nz]
    return out


def shape_ft(shape, ky, kx, period):
    """
    Continuous Fourier transform of a single shape's indicator function, divided by the
    unit-cell area, evaluated on the integer reciprocal grid ``(ky, kx)``.

    Returns ``c(ky, kx) = 1/(Px*Py) * FT{1_shape}(u, v)`` — the analytic analogue of one
    Fourier coefficient of the indicator, matching the ``dfs2d`` convention.

    ``shape`` keys: ``type`` in {rectangle, square, circle, ellipse, disk}, center
    ``cx, cy``, full lengths ``lx, ly`` (diameter for circle), rotation ``angle`` (rad).
    """
    Px, Py = float(period[0]), float(period[1])
    stype = shape.get('type', 'rectangle')
    cx = shape.get('cx', Px / 2.0)
    cy = shape.get('cy', Py / 2.0)
    lx = shape.get('lx', shape.get('l', 0.0))
    ly = shape.get('ly', lx if stype in ('circle', 'square') else shape.get('l', 0.0))
    angle = shape.get('angle', 0.0)

    ky = np.asarray(ky, dtype=float)
    kx = np.asarray(kx, dtype=float)

    u = 2.0 * np.pi * kx / Px
    v = 2.0 * np.pi * ky / Py

    # Rotated frequency coordinates: kappa = Rot(angle)^T @ k
    ct, st = np.cos(angle), np.sin(angle)
    kxr = u * ct + v * st
    kyr = -u * st + v * ct

    phase = np.exp(-1j * (u * cx + v * cy))

    if stype in ('rectangle', 'square'):
        s = lx * _sinc(kxr * lx / 2.0) * ly * _sinc(kyr * ly / 2.0)
    elif stype in ('circle', 'disk'):
        R = lx / 2.0
        rho = np.sqrt(u * u + v * v)
        # 2 pi R J1(R rho)/rho  ==  (pi R^2) * jinc(R rho)
        s = (np.pi * R * R) * _jinc(R * rho)
    elif stype == 'ellipse':
        a = lx / 2.0
        b = ly / 2.0
        # safe sqrt: guard the DC point (kap2==0) so d/d(shape) of sqrt(0) is not NaN
        kap2 = (a * kxr) ** 2 + (b * kyr) ** 2
        nz = kap2 != 0
        kap = np.sqrt(np.where(nz, kap2, 1.0))
        jinc = np.where(nz, 2.0 * j1(kap) / np.where(nz, kap, 1.0), 1.0)
        s = (np.pi * a * b) * jinc
    else:
        raise ValueError(f"Unknown analytic shape type: {stype!r}")

    return phase * s / (Px * Py)


def assemble_eps_coeffs(shapes, n_base, fto_x, fto_y, period, type_complex=np.complex128):
    """
    Analytic Fourier-coefficient grids of ``eps`` and ``1/eps`` for a layer built from a
    background index plus a list of (non-overlapping) analytic shapes.

    Permittivity uses ``conj(n)**2`` to match the ``e^{-iwt}`` / Im(n)>0-loss convention
    used elsewhere in fieldwave (see ``to_conv_mat_raster_nvf``).

    Returns ``(eps_c, inv_eps_c)``, each an array of shape ``(4*fto_y+1, 4*fto_x+1)``
    indexed ``[ky + 2*fto_y, kx + 2*fto_x]`` — the double range needed to build a
    ``(2*fto+1)**2`` Laurent convolution matrix.
    """
    KY = np.arange(-2 * fto_y, 2 * fto_y + 1)
    KX = np.arange(-2 * fto_x, 2 * fto_x + 1)
    KKY, KKX = np.meshgrid(KY, KX, indexing='ij')  # (nky, nkx)

    eps_bg = np.conj(np.asarray(n_base, dtype=type_complex)) ** 2
    inv_bg = 1.0 / eps_bg

    eps_c = np.zeros(KKY.shape, dtype=type_complex)
    inv_eps_c = np.zeros(KKY.shape, dtype=type_complex)

    # DC (constant background over the whole cell)
    dc = (KY == 0)[:, None] & (KX == 0)[None, :]
    eps_c[dc] = eps_bg
    inv_eps_c[dc] = inv_bg

    for shape in shapes:
        n_s = np.asarray(shape['n_index'], dtype=type_complex)
        eps_s = np.conj(n_s) ** 2
        ft = shape_ft(shape, KKY, KKX, period).astype(type_complex)
        eps_c += (eps_s - eps_bg) * ft
        inv_eps_c += (1.0 / eps_s - inv_bg) * ft

    return eps_c, inv_eps_c


def laurent_conv_2d(coeff2d, fto_x, fto_y):
    """
    Build the block-Toeplitz (Laurent) convolution matrix from a 2D coefficient grid.

    ``coeff2d`` has shape ``(4*fto_y+1, 4*fto_x+1)`` indexed ``[ky+2fto_y, kx+2fto_x]``.
    The output ``(ff_y*ff_x, ff_y*ff_x)`` matrix satisfies
    ``M[(r,s),(cc,d)] = coeff(ky=cc-r, kx=d-s)`` — identical to the ordering produced by
    ``fourier_analysis.dfs2d`` (verified to machine precision).
    """
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    oy = np.arange(ff_y)
    ox = np.arange(ff_x)
    dy = oy[None, :] - oy[:, None]  # [r, cc] = cc - r
    dx = ox[None, :] - ox[:, None]  # [s, d] = d - s

    iy = (dy + 2 * fto_y)[:, None, :, None]
    ix = (dx + 2 * fto_x)[None, :, None, :]
    m4 = coeff2d[iy, ix]  # [r, s, cc, d]
    return m4.reshape(ff_y * ff_x, ff_y * ff_x)


def to_conv_mat_analytic(shapes, n_base, fto_x, fto_y, period, type_complex=np.complex128):
    """
    Analytic Laurent convolution matrices ``(eps_conv, inv_eps_conv)`` for one layer.

    These are the *direct-rule* Fourier matrices of the true shape; they are the building
    blocks consumed by the NVF factorization (``to_conv_mat_analytic_nvf``).
    """
    eps_c, inv_eps_c = assemble_eps_coeffs(shapes, n_base, fto_x, fto_y, period,
                                           type_complex=type_complex)
    eps_conv = laurent_conv_2d(eps_c, fto_x, fto_y)
    inv_eps_conv = laurent_conv_2d(inv_eps_c, fto_x, fto_y)
    return eps_conv, inv_eps_conv


def to_conv_mat_analytic_nvf(shapes, n_base, fto_x, fto_y, period,
                             normal_grid=256, type_complex=np.complex128, use_pinv=False):
    """
    Build the five ``transfer_2d_2`` convolution matrices for one analytic layer using the
    NVF (Li / Götz-Schuster) normal-vector factorization.

    ``eps`` and ``1/eps`` Laurent matrices come from the *exact* shape Fourier transform
    (no staircase); the normal-vector projectors ``<nx^2>, <nx ny>, <ny^2>`` come from the
    *analytic* unit normal field of the same shapes (``get_normal_vector_field_analytical``)
    sampled on a ``normal_grid x normal_grid`` grid.  The Hermitian symmetrization is the
    same as ``nvf.to_conv_mat_nvf_2d``; see that function and ``NVF_FINDINGS.md`` for the
    slot-convention derivation.

    Returns ``(Exx, Eyy, Exy, Eyx, Epz_i)`` mapped onto the transfer slots
    ``(epx, epy, epxy, epyx, epz_i)``.
    """
    eps_conv, inv_eps_conv = to_conv_mat_analytic(shapes, n_base, fto_x, fto_y, period,
                                                  type_complex=type_complex)

    ep_dir = eps_conv
    ep_inv = inv_eps_conv
    ep_inv_i = meeinv(ep_inv, use_pinv=use_pinv)
    delta_ep = ep_dir - ep_inv_i

    # Analytic unit normal field (true boundary normals), sampled then Laurent-projected.
    nx, ny = get_normal_vector_field_analytical((normal_grid, normal_grid), shapes,
                                                period=(float(period[1]), float(period[0])))
    C_xx = dfs2d((nx ** 2).astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)
    C_xy = dfs2d((nx * ny).astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)
    C_yy = dfs2d((ny ** 2).astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)

    eps_xx = ep_dir - 0.5 * (delta_ep @ C_xx + C_xx @ delta_ep)
    eps_yy = ep_dir - 0.5 * (delta_ep @ C_yy + C_yy @ delta_ep)
    eps_xy = -0.5 * (delta_ep @ C_xy + C_xy @ delta_ep)  # == eps_yx (symmetric)

    Exx = eps_xx
    Eyy = eps_yy
    Exy = -eps_xy
    Eyx = -eps_xy
    Epz_i = meeinv(ep_dir, use_pinv=use_pinv)

    return Exx, Eyy, Exy, Eyx, Epz_i


def to_conv_mat_analytic_layers(layer_shapes, fto_x, fto_y, period,
                                normal_grid=256, type_complex=np.complex128, use_pinv=False):
    """
    Stack the five analytic-NVF convolution matrices over layers.

    ``layer_shapes`` is a list (one entry per layer) of ``dict(n_base=..., shapes=[...])``.
    Returns ``(epx, epy, epxy, epyx, epz_i)`` each of shape ``(n_layers, ff_xy, ff_xy)``.
    """
    n_layers = len(layer_shapes)
    ff_xy = (2 * fto_x + 1) * (2 * fto_y + 1)

    epx = np.zeros((n_layers, ff_xy, ff_xy), dtype=type_complex)
    epy = np.zeros((n_layers, ff_xy, ff_xy), dtype=type_complex)
    epxy = np.zeros((n_layers, ff_xy, ff_xy), dtype=type_complex)
    epyx = np.zeros((n_layers, ff_xy, ff_xy), dtype=type_complex)
    epz_i = np.zeros((n_layers, ff_xy, ff_xy), dtype=type_complex)

    for i, layer in enumerate(layer_shapes):
        n_base = layer.get('n_base', layer.get('base', 1.0))
        shapes = layer.get('shapes', [])
        Exx, Eyy, Exy, Eyx, Epz_i = to_conv_mat_analytic_nvf(
            shapes, n_base, fto_x, fto_y, period,
            normal_grid=normal_grid, type_complex=type_complex, use_pinv=use_pinv)
        epx[i] = Exx
        epy[i] = Eyy
        epxy[i] = Exy
        epyx[i] = Eyx
        epz_i[i] = Epz_i

    return epx, epy, epxy, epyx, epz_i
