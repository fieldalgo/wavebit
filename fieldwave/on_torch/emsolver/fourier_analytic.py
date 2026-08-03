"""
fourier_analytic.py (torch port) — analytic (staircase-free) Fourier coefficients of
smooth primitives (rotated rectangle, circle/disk, ellipse).

See ``fieldwave/on_numpy/emsolver/fourier_analytic.py`` for the full derivation and the
coefficient convention (matches ``fourier_analysis.dfs2d``).  This backend uses
``torch.special.bessel_j1`` for the disk / ellipse transform and is differentiable.
"""
import torch

from .fourier_analysis import dfs2d
from .nvf import get_normal_vector_field_analytical
from .primitives import meeinv


class _BesselJ1(torch.autograd.Function):
    """torch.special.bessel_j1 has an accurate forward but no autograd backward.

    Wrap it with the analytic derivative ``dJ1/dx = J0(x) - J1(x)/x`` (== 1/2 at x=0)
    so gradients flow through circle / ellipse shape parameters.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.bessel_j1(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        safe = torch.where(x == 0, torch.ones_like(x), x)
        dj1 = torch.where(x == 0, torch.full_like(x, 0.5),
                          torch.special.bessel_j0(x) - torch.special.bessel_j1(x) / safe)
        return grad_out * dj1


def _j1(x):
    return _BesselJ1.apply(x)


def _sinc(z):
    safe = torch.where(z == 0, torch.ones_like(z), z)
    return torch.where(z == 0, torch.ones_like(z), torch.sin(z) / safe)


def _jinc(kap):
    safe = torch.where(kap == 0, torch.ones_like(kap), kap)
    j = _j1(kap)
    return torch.where(kap == 0, torch.ones_like(kap), 2.0 * j / safe)


def shape_ft(shape, ky, kx, period, type_float=torch.float64):
    """Continuous FT of one shape indicator / cell area, on the reciprocal grid (ky, kx)."""
    Px, Py = float(period[0]), float(period[1])
    stype = shape.get('type', 'rectangle')
    cx = shape.get('cx', Px / 2.0)
    cy = shape.get('cy', Py / 2.0)
    lx = shape.get('lx', shape.get('l', 0.0))
    ly = shape.get('ly', lx if stype in ('circle', 'square') else shape.get('l', 0.0))
    angle = shape.get('angle', 0.0)

    ky = ky.to(type_float)
    kx = kx.to(type_float)

    u = 2.0 * torch.pi * kx / Px
    v = 2.0 * torch.pi * ky / Py

    angle_t = torch.as_tensor(angle, dtype=type_float)
    ct, st = torch.cos(angle_t), torch.sin(angle_t)
    kxr = u * ct + v * st
    kyr = -u * st + v * ct

    phase = torch.exp(-1j * (u * cx + v * cy).to(torch.complex128))

    if stype in ('rectangle', 'square'):
        s = lx * _sinc(kxr * lx / 2.0) * ly * _sinc(kyr * ly / 2.0)
    elif stype in ('circle', 'disk'):
        R = lx / 2.0
        rho = torch.sqrt(u * u + v * v)
        s = (torch.pi * R * R) * _jinc(R * rho)
    elif stype == 'ellipse':
        a = lx / 2.0
        b = ly / 2.0
        # safe sqrt: guard the DC point (kap2==0) so d/d(shape) of sqrt(0) is not NaN
        kap2 = (a * kxr) ** 2 + (b * kyr) ** 2
        nz = kap2 != 0
        kap = torch.sqrt(torch.where(nz, kap2, torch.ones_like(kap2)))
        jinc = torch.where(nz, 2.0 * _j1(kap) / kap, torch.ones_like(kap))
        s = (torch.pi * a * b) * jinc
    else:
        raise ValueError(f"Unknown analytic shape type: {stype!r}")

    return phase * s.to(torch.complex128) / (Px * Py)


def assemble_eps_coeffs(shapes, n_base, fto_x, fto_y, period, type_complex=torch.complex128):
    type_float = torch.float64 if type_complex == torch.complex128 else torch.float32
    KY = torch.arange(-2 * fto_y, 2 * fto_y + 1)
    KX = torch.arange(-2 * fto_x, 2 * fto_x + 1)
    KKY, KKX = torch.meshgrid(KY, KX, indexing='ij')

    eps_bg = torch.conj(torch.as_tensor(n_base, dtype=type_complex)) ** 2
    inv_bg = 1.0 / eps_bg

    dc = ((KKY == 0) & (KKX == 0)).to(type_complex)
    eps_c = eps_bg * dc
    inv_eps_c = inv_bg * dc

    for shape in shapes:
        n_s = torch.as_tensor(shape['n_index'], dtype=type_complex)
        eps_s = torch.conj(n_s) ** 2
        ft = shape_ft(shape, KKY, KKX, period, type_float=type_float).to(type_complex)
        eps_c = eps_c + (eps_s - eps_bg) * ft
        inv_eps_c = inv_eps_c + (1.0 / eps_s - inv_bg) * ft

    return eps_c, inv_eps_c


def laurent_conv_2d(coeff2d, fto_x, fto_y):
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    oy = torch.arange(ff_y)
    ox = torch.arange(ff_x)
    dy = oy[None, :] - oy[:, None]
    dx = ox[None, :] - ox[:, None]

    iy = (dy + 2 * fto_y)[:, None, :, None].expand(ff_y, ff_x, ff_y, ff_x)
    ix = (dx + 2 * fto_x)[None, :, None, :].expand(ff_y, ff_x, ff_y, ff_x)
    m4 = coeff2d[iy, ix]
    return m4.reshape(ff_y * ff_x, ff_y * ff_x)


def to_conv_mat_analytic(shapes, n_base, fto_x, fto_y, period, type_complex=torch.complex128):
    eps_c, inv_eps_c = assemble_eps_coeffs(shapes, n_base, fto_x, fto_y, period, type_complex=type_complex)
    return laurent_conv_2d(eps_c, fto_x, fto_y), laurent_conv_2d(inv_eps_c, fto_x, fto_y)


def to_conv_mat_analytic_nvf(shapes, n_base, fto_x, fto_y, period, normal_grid=256,
                             device=torch.device('cpu'), type_complex=torch.complex128, use_pinv=False):
    type_float = torch.float64 if type_complex == torch.complex128 else torch.float32
    eps_conv, inv_eps_conv = to_conv_mat_analytic(shapes, n_base, fto_x, fto_y, period, type_complex=type_complex)

    ep_dir = eps_conv
    ep_inv = inv_eps_conv
    ep_inv_i = meeinv(ep_inv, use_pinv=use_pinv)
    delta_ep = ep_dir - ep_inv_i

    nx, ny = get_normal_vector_field_analytical((normal_grid, normal_grid), shapes,
                                                period=(float(period[1]), float(period[0])),
                                                device=device, type_float=type_float)
    C_xx = dfs2d((nx ** 2).to(type_complex), 1, 1, fto_x, fto_y, device=device, type_complex=type_complex)
    C_xy = dfs2d((nx * ny).to(type_complex), 1, 1, fto_x, fto_y, device=device, type_complex=type_complex)
    C_yy = dfs2d((ny ** 2).to(type_complex), 1, 1, fto_x, fto_y, device=device, type_complex=type_complex)

    eps_xx = ep_dir - 0.5 * (delta_ep @ C_xx + C_xx @ delta_ep)
    eps_yy = ep_dir - 0.5 * (delta_ep @ C_yy + C_yy @ delta_ep)
    eps_xy = -0.5 * (delta_ep @ C_xy + C_xy @ delta_ep)

    Exx = eps_xx
    Eyy = eps_yy
    Exy = -eps_xy
    Eyx = -eps_xy
    Epz_i = meeinv(ep_dir, use_pinv=use_pinv)
    return Exx, Eyy, Exy, Eyx, Epz_i


def to_conv_mat_analytic_layers(layer_shapes, fto_x, fto_y, period, normal_grid=256,
                                device=torch.device('cpu'), type_complex=torch.complex128, use_pinv=False):
    n_layers = len(layer_shapes)
    ff_xy = (2 * fto_x + 1) * (2 * fto_y + 1)

    epx = torch.zeros((n_layers, ff_xy, ff_xy), dtype=type_complex, device=device)
    epy = torch.zeros((n_layers, ff_xy, ff_xy), dtype=type_complex, device=device)
    epxy = torch.zeros((n_layers, ff_xy, ff_xy), dtype=type_complex, device=device)
    epyx = torch.zeros((n_layers, ff_xy, ff_xy), dtype=type_complex, device=device)
    epz_i = torch.zeros((n_layers, ff_xy, ff_xy), dtype=type_complex, device=device)

    for i, layer in enumerate(layer_shapes):
        n_base = layer.get('n_base', layer.get('base', 1.0))
        shapes = layer.get('shapes', [])
        Exx, Eyy, Exy, Eyx, Epz_i = to_conv_mat_analytic_nvf(
            shapes, n_base, fto_x, fto_y, period, normal_grid=normal_grid,
            device=device, type_complex=type_complex, use_pinv=use_pinv)
        epx[i] = Exx
        epy[i] = Eyy
        epxy[i] = Exy
        epyx[i] = Eyx
        epz_i[i] = Epz_i

    return epx, epy, epxy, epyx, epz_i
