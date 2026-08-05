"""Single-source analytic (staircase-free) Fourier modeling.

A layer is described by a background index plus a list of parametric shapes — rectangles,
circles, ellipses — and its permittivity is Fourier-transformed *analytically* rather than
sampled onto a raster. That removes staircase error entirely, so the same accuracy needs far
fewer harmonics on curved or tilted boundaries.

A layer is ``dict(n_base=..., shapes=[...])`` and a shape is
``dict(type='rectangle'|'circle'|'ellipse', cx=, cy=, lx=, ly=, angle=, n_index=)``.

**Shapes superpose; they do not paint over each other.** Each one contributes
``(eps_shape - eps_base) * FT(indicator)``, so where two overlap the layer gets *both*
contributions and the permittivity there is neither shape's. Two 120x120 rods of index 2 and 3
in a background of 1, overlapping by 7200 of a 90000 cell, come out at a mean permittivity of
2.76 where painting the second over the first gives 2.52 -- and the efficiency follows, 0.9856
against the 0.9792 a raster and the vector path agree on. Nothing raises: the layer is simply a
different one. The vector path (``ucell=[[n_base, [[shape, ...], ...]]]``) is the one that
paints, later object over earlier, and it is what to use when shapes are meant to overlap.
Non-overlapping shapes are unaffected, which is the case this path is built for.

**Shape parameters are absolute lengths, in whatever unit ``period`` is given in** -- not
fractions of the period. A rod filling half a 0.77-long period is ``lx=0.385``, centred at
``cx=0.385``; the defaults say the same thing (``cx`` defaults to ``period_x / 2``). Passing
fractions builds a rod ``1/period`` times too large and says nothing about it, so
``test_analytic_shape_parameters_are_absolute_lengths`` pins the distinction.

Everything here is differentiable with respect to the shape parameters (``cx``, ``cy``, ``lx``,
``ly``, ``angle``, ``n_index``) — those are what an inverse-design user actually optimizes. The
one primitive that made this hard, Bessel ``J1``, lives in :mod:`fieldwave._core.backend`: SciPy
has it, PyTorch's has no autograd backward, and JAX's returns NaN near zero, so each backend
reaches it a different way behind one name.

The convolution matrices are built with the NVF (Li / Götz-Schuster) normal-vector
factorization: ``eps`` and ``1/eps`` come from the exact shape transform, while the normal
projectors come from the shapes' *analytic* unit normal field sampled on a grid.
"""
from .backend import array_namespace, as_array, j1, jinc, sinc
from .factorization import dfs2d


def _grid(xp, fto_x, fto_y, dev, dtype):
    """Integer reciprocal grid ``(KY, KX)`` spanning the doubled range a Laurent matrix needs."""
    ky = xp.asarray([float(m) for m in range(-2 * fto_y, 2 * fto_y + 1)], dtype=dtype, device=dev)
    kx = xp.asarray([float(m) for m in range(-2 * fto_x, 2 * fto_x + 1)], dtype=dtype, device=dev)
    return ky[:, None] + 0.0 * kx[None, :], 0.0 * ky[:, None] + kx[None, :]


def shape_geometry(shape, Px, Py):
    """``(type, cx, cy, lx, ly, angle)`` for one shape dict, resolved once for everybody.

    It exists because the transform and the normal field used to resolve them separately and
    disagree. Two ways that showed:

    * ``circle`` is round. Its transform only ever used ``lx`` as the diameter, while the
      normal field read ``ly`` as a second semi-axis — so ``ly != lx`` on a circle silently
      built one structure for ``eps`` and a different one for the projector (14% on
      ``sum(de_ri)``). ``ly`` is now ``lx`` for a circle wherever it is read; an oval is an
      ``ellipse``.
    * the defaults differed (``ly`` fell back to ``lx`` in one and to ``500.0`` in the other),
      so *omitting* ``ly`` on a circle — the natural way to write it — degenerated the normal
      field and moved the answer 6%.

    Lengths are absolute, in the units of ``period`` (see the module docstring).
    """
    stype = shape.get('type', 'rectangle')
    cx = shape.get('cx', Px / 2.0)
    cy = shape.get('cy', Py / 2.0)
    lx = shape.get('lx', shape.get('l', 0.0))
    round_ = stype in ('circle', 'disk', 'square')
    ly = lx if round_ else shape.get('ly', shape.get('l', 0.0))
    return stype, cx, cy, lx, ly, shape.get('angle', 0.0)


def shape_ft(xp, shape, ky, kx, period):
    """Continuous Fourier transform of one shape's indicator, divided by the cell area.

    Returns ``c(ky, kx) = 1/(Px*Py) * FT{1_shape}(u, v)`` on the integer reciprocal grid —
    the analytic analogue of one ``dfs2d`` Fourier coefficient.
    """
    Px, Py = period[0], period[1]
    stype, cx, cy, lx, ly, angle = shape_geometry(shape, Px, Py)

    pi = 3.141592653589793
    u = 2.0 * pi * kx / Px
    v = 2.0 * pi * ky / Py

    # Rotated frequency coordinates: kappa = Rot(angle)^T @ k
    ct = xp.cos(as_array(xp, angle, dtype=u.dtype))
    st = xp.sin(as_array(xp, angle, dtype=u.dtype))
    kxr = u * ct + v * st
    kyr = -u * st + v * ct

    phase = xp.exp(-1j * (u * cx + v * cy))

    if stype in ('rectangle', 'square'):
        s = lx * sinc(kxr * lx / 2.0) * ly * sinc(kyr * ly / 2.0)
    elif stype in ('circle', 'disk'):
        R = lx / 2.0
        rho = (u * u + v * v) ** 0.5
        s = (pi * R * R) * jinc(R * rho)          # 2 pi R J1(R rho)/rho
    elif stype == 'ellipse':
        a = lx / 2.0
        b = ly / 2.0
        # Guard the DC point so d/d(shape) of sqrt(0) is not NaN.
        kap2 = (a * kxr) ** 2 + (b * kyr) ** 2
        nz = kap2 != 0
        kap = xp.sqrt(xp.where(nz, kap2, xp.ones_like(kap2)))
        j = xp.where(nz, 2.0 * j1(kap) / xp.where(nz, kap, xp.ones_like(kap)),
                     xp.ones_like(kap))
        s = (pi * a * b) * j
    else:
        raise ValueError(f'Unknown analytic shape type: {stype!r}')

    return phase * s / (Px * Py)


def assemble_eps_coeffs(xp, shapes, n_base, fto_x, fto_y, period, dtype):
    """Analytic Fourier coefficients of ``eps`` and ``1/eps`` for one layer.

    Each grid has shape ``(4*fto_y+1, 4*fto_x+1)``, indexed ``[ky+2*fto_y, kx+2*fto_x]`` — the
    doubled range a ``(2*fto+1)**2`` Laurent matrix needs. The ``n**2`` maps refractive index
    to permittivity: FieldWave works in ``e^{-iwt}`` throughout (``Im(n) > 0`` = loss), with no
    conjugation on the way in.
    """
    dev = getattr(n_base, 'device', None) if hasattr(n_base, 'device') else None
    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32
    KKY, KKX = _grid(xp, fto_x, fto_y, dev, real_dtype)

    eps_bg = as_array(xp, n_base, dtype=dtype) ** 2
    inv_bg = 1.0 / eps_bg

    # DC term only: the background is constant over the whole cell.
    is_dc = (KKY == 0) & (KKX == 0)
    zero = xp.zeros(KKY.shape, dtype=dtype)
    eps_c = xp.where(is_dc, eps_bg, zero)
    inv_eps_c = xp.where(is_dc, inv_bg, zero)

    for shape in shapes:
        eps_s = as_array(xp, shape['n_index'], dtype=dtype) ** 2
        ft = xp.astype(shape_ft(xp, shape, KKY, KKX, period), dtype)
        eps_c = eps_c + (eps_s - eps_bg) * ft
        inv_eps_c = inv_eps_c + (1.0 / eps_s - inv_bg) * ft

    return eps_c, inv_eps_c


def laurent_conv_2d(xp, coeff2d, fto_x, fto_y):
    """Block-Toeplitz (Laurent) convolution matrix from a 2D coefficient grid.

    ``M[(r,s),(c,d)] = coeff(ky=r-c, kx=s-d)`` — the same ordering ``dfs2d`` produces. The
    sign is the order-index convention (see ``_circulant_index``); the two builders have to
    agree with each other and with ``kx``, which ``test_laurent_matches_dfs2d_convention``
    pins.
    """
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1
    dev = _device(coeff2d)

    iy = xp.asarray([[[[r - c + 2 * fto_y]]] for r in range(ff_y) for c in range(ff_y)],
                    device=dev)
    iy = xp.reshape(iy, (ff_y, ff_y))
    ix = xp.asarray([[s - d + 2 * fto_x for d in range(ff_x)] for s in range(ff_x)], device=dev)

    m4 = coeff2d[iy[:, None, :, None], ix[None, :, None, :]]      # [r, s, c, d]
    return xp.reshape(m4, (ff_y * ff_x, ff_y * ff_x))


def _device(x):
    return x.device if hasattr(x, 'device') else None


def normal_vector_field_analytical(xp, grid_shape, shapes, period, dtype):
    """Unit normal field of the shapes, sampled on ``grid_shape`` — analytic, not from a raster.

    ``period`` is unpacked as ``(Py, Px)``. Written without in-place mutation so it stays
    differentiable in the shape parameters.
    """
    Ny, Nx = grid_shape
    Py, Px = period[0], period[1]

    x = (xp.asarray([float(i) for i in range(Nx)], dtype=dtype) + 0.5) * (Px / Nx)
    y = (xp.asarray([float(i) for i in range(Ny)], dtype=dtype) + 0.5) * (Py / Ny)
    X = x[None, :] + 0.0 * y[:, None]
    Y = y[:, None] + 0.0 * x[None, :]

    nx_total = xp.zeros((Ny, Nx), dtype=dtype)
    ny_total = xp.zeros((Ny, Nx), dtype=dtype)

    for s in shapes:
        stype, cx, cy, lx, ly, angle = shape_geometry(s, Px, Py)

        cos_a = xp.cos(as_array(xp, angle, dtype=dtype))
        sin_a = xp.sin(as_array(xp, angle, dtype=dtype))
        dx, dy = X - cx, Y - cy
        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a

        a, b = lx / 2.0, ly / 2.0
        if stype in ('ellipse', 'circle'):
            gxr = 2.0 * xr / (a ** 2)
            gyr = 2.0 * yr / (b ** 2)
        elif stype == 'rectangle':
            # Nearest face decides the normal direction.
            mask_x = (xp.abs(xr) - a) >= (xp.abs(yr) - b)
            zeros = xp.zeros_like(xr)
            gxr = xp.where(mask_x, xp.sign(xr), zeros)
            gyr = xp.where(mask_x, zeros, xp.sign(yr))
        else:
            continue

        gx = gxr * cos_a - gyr * sin_a
        gy = gxr * sin_a + gyr * cos_a
        norm = _hypot_safe(xp, gx, gy)
        nx_total = nx_total + gx / norm
        ny_total = ny_total + gy / norm

    norm_t = _hypot_safe(xp, nx_total, ny_total)
    return nx_total / norm_t, ny_total / norm_t


def _hypot_safe(xp, a, b):
    """``hypot`` with zeros replaced by 1, so the division that follows is well defined."""
    n = (a * a + b * b) ** 0.5
    return xp.where(n == 0, xp.ones_like(n), n)


def to_conv_mat_analytic_nvf(xp, shapes, n_base, fto_x, fto_y, period, dtype,
                             normal_grid=256):
    """The five 2D-solve convolution matrices for one analytic layer, via NVF.

    ``eps`` / ``1/eps`` come from the exact shape transform (no staircase); the normal
    projectors come from the shapes' analytic normal field. Same Hermitian symmetrization as
    the raster NVF path — see :func:`fieldwave._core.factorization.nvf_conv_matrices`.
    """
    eps_c, inv_eps_c = assemble_eps_coeffs(xp, shapes, n_base, fto_x, fto_y, period, dtype)
    ep_dir = laurent_conv_2d(xp, eps_c, fto_x, fto_y)
    ep_inv = laurent_conv_2d(xp, inv_eps_c, fto_x, fto_y)

    delta_ep = ep_dir - xp.linalg.inv(ep_inv)

    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32
    nx, ny = normal_vector_field_analytical(
        xp, (normal_grid, normal_grid), shapes,
        (period[1], period[0]), real_dtype)

    C_xx = dfs2d(xp, xp.astype(nx ** 2, dtype), 1, 1, fto_x, fto_y)
    C_xy = dfs2d(xp, xp.astype(nx * ny, dtype), 1, 1, fto_x, fto_y)
    C_yy = dfs2d(xp, xp.astype(ny ** 2, dtype), 1, 1, fto_x, fto_y)

    eps_xx = ep_dir - 0.5 * (delta_ep @ C_xx + C_xx @ delta_ep)
    eps_yy = ep_dir - 0.5 * (delta_ep @ C_yy + C_yy @ delta_ep)
    eps_xy = -0.5 * (delta_ep @ C_xy + C_xy @ delta_ep)          # == eps_yx (symmetric)

    return eps_xx, eps_yy, -eps_xy, -eps_xy, xp.linalg.inv(ep_dir)


def to_conv_mat_analytic_layers(layer_shapes, fto_x, fto_y, period,
                                normal_grid=256, type_complex=None, xp=None):
    """Stack the analytic-NVF convolution matrices over layers.

    ``layer_shapes`` is one ``dict(n_base=..., shapes=[...])`` per layer, with the shapes'
    ``cx`` / ``cy`` / ``lx`` / ``ly`` as absolute lengths in the units of ``period`` (see the
    module docstring). Returns ``(epx, epy, epxy, epyx, epz_i)``, each
    ``(n_layers, ff_xy, ff_xy)``.

    The array namespace is taken from the first differentiable shape parameter found, so
    passing JAX or PyTorch scalars for e.g. ``lx`` makes the whole build run — and
    differentiate — on that backend. Pass ``xp`` to force it.
    """
    if xp is None:
        xp = _infer_namespace(layer_shapes)
    dtype = xp.complex128 if type_complex is None else type_complex

    out = [[], [], [], [], []]
    for layer in layer_shapes:
        n_base = layer.get('n_base', layer.get('base', 1.0))
        mats = to_conv_mat_analytic_nvf(
            xp, layer.get('shapes', []), n_base, fto_x, fto_y, period, dtype,
            normal_grid=normal_grid)
        for slot, m in zip(out, mats):
            slot.append(m)

    return tuple(xp.stack(slot) for slot in out)


def _infer_namespace(layer_shapes):
    """Namespace of the first array-like shape parameter; NumPy when everything is a scalar."""
    for layer in layer_shapes:
        candidates = [layer.get('n_base'), layer.get('base')]
        for shape in layer.get('shapes', []):
            candidates.extend(shape.get(k) for k in
                              ('cx', 'cy', 'lx', 'ly', 'l', 'angle', 'n_index'))
        for c in candidates:
            if c is not None and hasattr(c, 'dtype') and hasattr(c, 'shape'):
                return array_namespace(c)
    import numpy as np
    return array_namespace(np.zeros(1))
