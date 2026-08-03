"""
fourier_analytic.py (jax port) — analytic (staircase-free) Fourier coefficients of
smooth primitives (rotated rectangle, circle/disk, ellipse).

See ``fieldwave/on_numpy/emsolver/fourier_analytic.py`` for the full derivation and the
coefficient convention (matches ``fourier_analysis.dfs2d``).

jax has no reliable native Bessel J1 (``jax.scipy.special.bessel_jn`` returns NaN for
small arguments), so this backend ships a self-contained, differentiable ``_j1`` built
from the ascending power series (small argument) and the asymptotic expansion (large
argument).  It agrees with ``scipy.special.j1`` to < 1e-12 over the arguments that occur
here (``|kappa| <~ 100``); see the validation in ``tests_verification``.
"""
import jax.numpy as jnp

from .fourier_analysis import dfs2d
from .nvf import get_normal_vector_field_analytical
from .primitives import meeinv

_J1_THR = 12.0
_J1_SERIES_TERMS = 36
_J1_ASYM_TERMS = 12

# Asymptotic coefficients a_k(nu=1), mu = 4*nu^2 = 4  (static python floats).
_J1_A = [1.0]
for _k in range(1, 2 * _J1_ASYM_TERMS + 2):
    _J1_A.append(_J1_A[-1] * (4.0 - (2 * _k - 1) ** 2) / (_k * 8.0))


def _j1_series(x):
    # (x/2) * sum_m (-x^2/4)^m / (m! (m+1)!)
    half = x / 2.0
    term = half
    s = term
    z = -(x * x) / 4.0
    for m in range(1, _J1_SERIES_TERMS):
        term = term * z / (m * (m + 1))
        s = s + term
    return s


def _j1_asym(x):
    P = jnp.zeros_like(x)
    Q = jnp.zeros_like(x)
    for k in range(_J1_ASYM_TERMS):
        P = P + ((-1) ** k) * _J1_A[2 * k] / x ** (2 * k)
    for k in range(_J1_ASYM_TERMS):
        Q = Q + ((-1) ** k) * _J1_A[2 * k + 1] / x ** (2 * k + 1)
    omega = x - 3.0 * jnp.pi / 4.0
    return jnp.sqrt(2.0 / (jnp.pi * x)) * (P * jnp.cos(omega) - Q * jnp.sin(omega))


def _j1(x):
    """Differentiable Bessel J1 for x >= 0 (odd extension not needed here)."""
    x = jnp.abs(x)
    xs = jnp.minimum(x, _J1_THR)   # clamp so each branch only sees safe inputs
    xl = jnp.maximum(x, _J1_THR)
    return jnp.where(x < _J1_THR, _j1_series(xs), _j1_asym(xl))


def _sinc(z):
    safe = jnp.where(z == 0, 1.0, z)
    return jnp.where(z == 0, 1.0, jnp.sin(z) / safe)


def _jinc(kap):
    safe = jnp.where(kap == 0, 1.0, kap)
    return jnp.where(kap == 0, 1.0, 2.0 * _j1(kap) / safe)


def shape_ft(shape, ky, kx, period):
    """Continuous FT of one shape indicator / cell area, on the reciprocal grid (ky, kx)."""
    Px, Py = float(period[0]), float(period[1])
    stype = shape.get('type', 'rectangle')
    cx = shape.get('cx', Px / 2.0)
    cy = shape.get('cy', Py / 2.0)
    lx = shape.get('lx', shape.get('l', 0.0))
    ly = shape.get('ly', lx if stype in ('circle', 'square') else shape.get('l', 0.0))
    angle = shape.get('angle', 0.0)

    ky = ky.astype(jnp.float64)
    kx = kx.astype(jnp.float64)

    u = 2.0 * jnp.pi * kx / Px
    v = 2.0 * jnp.pi * ky / Py

    ct, st = jnp.cos(jnp.asarray(angle)), jnp.sin(jnp.asarray(angle))
    kxr = u * ct + v * st
    kyr = -u * st + v * ct

    phase = jnp.exp(-1j * (u * cx + v * cy))

    if stype in ('rectangle', 'square'):
        s = lx * _sinc(kxr * lx / 2.0) * ly * _sinc(kyr * ly / 2.0)
    elif stype in ('circle', 'disk'):
        R = lx / 2.0
        rho = jnp.sqrt(u * u + v * v)
        s = (jnp.pi * R * R) * _jinc(R * rho)
    elif stype == 'ellipse':
        a = lx / 2.0
        b = ly / 2.0
        # safe sqrt: guard the DC point (kap2==0) so d/d(shape) of sqrt(0) is not NaN
        kap2 = (a * kxr) ** 2 + (b * kyr) ** 2
        nz = kap2 != 0
        kap = jnp.sqrt(jnp.where(nz, kap2, 1.0))
        jinc = jnp.where(nz, 2.0 * _j1(kap) / kap, 1.0)
        s = (jnp.pi * a * b) * jinc
    else:
        raise ValueError(f"Unknown analytic shape type: {stype!r}")

    return phase * s / (Px * Py)


def assemble_eps_coeffs(shapes, n_base, fto_x, fto_y, period, type_complex=jnp.complex128):
    KY = jnp.arange(-2 * fto_y, 2 * fto_y + 1)
    KX = jnp.arange(-2 * fto_x, 2 * fto_x + 1)
    KKY, KKX = jnp.meshgrid(KY, KX, indexing='ij')

    eps_bg = jnp.conj(jnp.asarray(n_base, dtype=type_complex)) ** 2
    inv_bg = 1.0 / eps_bg

    dc = ((KKY == 0) & (KKX == 0)).astype(type_complex)
    eps_c = eps_bg * dc
    inv_eps_c = inv_bg * dc

    for shape in shapes:
        n_s = jnp.asarray(shape['n_index'], dtype=type_complex)
        eps_s = jnp.conj(n_s) ** 2
        ft = shape_ft(shape, KKY, KKX, period).astype(type_complex)
        eps_c = eps_c + (eps_s - eps_bg) * ft
        inv_eps_c = inv_eps_c + (1.0 / eps_s - inv_bg) * ft

    return eps_c, inv_eps_c


def laurent_conv_2d(coeff2d, fto_x, fto_y):
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    oy = jnp.arange(ff_y)
    ox = jnp.arange(ff_x)
    dy = oy[None, :] - oy[:, None]
    dx = ox[None, :] - ox[:, None]

    iy = (dy + 2 * fto_y)[:, None, :, None]
    ix = (dx + 2 * fto_x)[None, :, None, :]
    m4 = coeff2d[iy, ix]
    return m4.reshape(ff_y * ff_x, ff_y * ff_x)


def to_conv_mat_analytic(shapes, n_base, fto_x, fto_y, period, type_complex=jnp.complex128):
    eps_c, inv_eps_c = assemble_eps_coeffs(shapes, n_base, fto_x, fto_y, period, type_complex=type_complex)
    return laurent_conv_2d(eps_c, fto_x, fto_y), laurent_conv_2d(inv_eps_c, fto_x, fto_y)


def to_conv_mat_analytic_nvf(shapes, n_base, fto_x, fto_y, period, normal_grid=256,
                             type_complex=jnp.complex128, use_pinv=False):
    eps_conv, inv_eps_conv = to_conv_mat_analytic(shapes, n_base, fto_x, fto_y, period, type_complex=type_complex)

    ep_dir = eps_conv
    ep_inv = inv_eps_conv
    ep_inv_i = meeinv(ep_inv, use_pinv=use_pinv)
    delta_ep = ep_dir - ep_inv_i

    nx, ny = get_normal_vector_field_analytical((normal_grid, normal_grid), shapes,
                                                period=(float(period[1]), float(period[0])))
    C_xx = dfs2d((nx ** 2).astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)
    C_xy = dfs2d((nx * ny).astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)
    C_yy = dfs2d((ny ** 2).astype(type_complex), 1, 1, fto_x, fto_y, type_complex=type_complex)

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
                                type_complex=jnp.complex128, use_pinv=False):
    epx, epy, epxy, epyx, epz_i = [], [], [], [], []
    for layer in layer_shapes:
        n_base = layer.get('n_base', layer.get('base', 1.0))
        shapes = layer.get('shapes', [])
        Exx, Eyy, Exy, Eyx, Epz_i = to_conv_mat_analytic_nvf(
            shapes, n_base, fto_x, fto_y, period, normal_grid=normal_grid,
            type_complex=type_complex, use_pinv=use_pinv)
        epx.append(Exx)
        epy.append(Eyy)
        epxy.append(Exy)
        epyx.append(Eyx)
        epz_i.append(Epz_i)

    return (jnp.stack(epx), jnp.stack(epy), jnp.stack(epxy), jnp.stack(epyx), jnp.stack(epz_i))
