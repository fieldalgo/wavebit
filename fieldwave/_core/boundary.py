# -*- coding: utf-8 -*-
"""Convolution matrices for a layer whose overlap has been resolved into polygons.

The fourth modeling path, and the only one that gets both halves of the problem right at once:

===============  ==================  =====================
path             tilted edge         overlapping shapes
===============  ==================  =====================
raster           staircase           as drawn
vector           staircase           **painted**
analytic         **exact**           added (superposed)
**this one**     **exact**           **painted**
===============  ==================  =====================

The vector path paints later shapes over earlier ones but decomposes them into axis-aligned
blocks, so a tilted edge becomes a staircase. The analytic path transforms a tilted shape in
closed form but *sums* the shapes, so an overlap gets both contributions and ends up as neither
shape's material. Neither can be fixed by mixing them: the analytic normal field never asks
which boundary survived.

This path resolves the painting **once** into disjoint polygons (:mod:`.clipper`), transforms
those in closed form (:func:`.polygon.polygon_ft`), and builds the normal field from the same
polygons -- so the boundary the field follows is the boundary that survived.

Split in two halves, which is what makes it differentiable:

* **planning** -- which vertices a loop has, which edges cross. Discrete, NumPy, outside any
  trace (:mod:`.clipper`).
* **replay** -- everything here. Arithmetic only, no value-dependent control flow, no in-place
  writes, so it traces and differentiates on every backend.

Validated against S4 (an independent RCWA implementation): the permittivity coefficients agree
to 2.2e-16, and on a case with both tilt and overlap this path sits 4x closer to it than the
vector path, while tying on axis-aligned overlap where the vector path has no staircase to lose.
"""
from .analytic import laurent_conv_2d, _sampled_coeffs
from .polygon import polygon_ft


def _sum(xp, a, axis=-1):
    return xp.sum(a, axis=axis)


def segments_of(xp, polys):
    """Concatenated (start, edge vector, |edge|^2) for every loop's every edge."""
    A = xp.concat([V for V in polys], axis=0)
    B = xp.concat([xp.roll(V, -1, axis=0) for V in polys], axis=0)
    E = B - A
    L2 = _sum(xp, E * E)
    # a zero-length edge would divide by zero; it also cannot be anyone's nearest point
    L2 = xp.where(L2 > 0, L2, xp.ones_like(L2))
    return A, E, L2


def _seg_dist(xp, pts, A, E, L2):
    """Squared point-to-segment distance and the foot, for every (point, segment) pair.

    Elementwise across the segment axis on purpose: column ``k`` depends only on segment ``k``,
    never on how many segments were passed, which is what lets a pruned subset give bit-identical
    numbers.
    """
    d = pts[:, None, :] - A[None, :, :]
    t = _sum(xp, d * E[None, :, :]) / L2[None, :]
    t = xp.clip(t, 0.0, 1.0)
    foot = A[None, :, :] + t[:, :, None] * E[None, :, :]
    diff = pts[:, None, :] - foot
    return _sum(xp, diff * diff), foot


def _nearest(xp, pts, A, E, L2):
    dist2, foot = _seg_dist(xp, pts, A, E, L2)
    j = xp.argmin(dist2, axis=1)
    rows = xp.arange(pts.shape[0])
    return j, foot[rows, j]


def distance_gradient_normal_field(xp, polys, grid_shape, period, dtype, chunk=None):
    """Unit normal from the direction to the closest boundary *point*.

    The closest *edge* is not well defined -- beyond a corner both edges meeting there have the
    corner as their closest point, at exactly equal distance, so `argmin`'s tie-break decides on
    29-45% of the cell. The closest point is unique except on the medial axis (measure zero).

    Sign flips inside/outside, which does not matter: NVF only uses ``nx**2``, ``nx*ny``,
    ``ny**2``.

    ``chunk`` is how many pixels are handled at once. It bounds the ``pixels x segments``
    temporaries without changing any result -- the arithmetic per pixel is independent, so a
    chunked run is bit-identical to a flat one. ``None`` targets 4M entries per temporary;
    measured peak is about 5x that, since `_seg_dist` has several ``(chunk, M, 2)`` arrays
    alive at once. Peak then stops growing with the grid, which is the point.
    """
    Ny, Nx = grid_shape
    Py, Px = period[0], period[1]
    A, E, L2 = segments_of(xp, polys)
    M = A.shape[0]

    x = (xp.asarray([float(i) for i in range(Nx)], dtype=dtype) + 0.5) * (Px / Nx)
    y = (xp.asarray([float(i) for i in range(Ny)], dtype=dtype) + 0.5) * (Py / Ny)
    X = x[None, :] + 0.0 * y[:, None]
    Y = y[:, None] + 0.0 * x[None, :]
    pts = xp.stack([xp.reshape(X, (-1,)), xp.reshape(Y, (-1,))], axis=-1)

    npix = Ny * Nx
    if chunk is None:
        chunk = max(1, min(npix, 4_000_000 // max(M, 1)))

    nxs, nys = [], []
    for lo in range(0, npix, chunk):
        p = pts[lo:lo + chunk]
        j, foot_j = _nearest(xp, p, A, E, L2)

        v = p - foot_j
        mag = xp.sqrt(_sum(xp, v * v))

        # A pixel centre exactly on the boundary has no direction; grid-aligned designs hit it.
        # Fall back to that edge's own normal, which is what the limit approaches.
        ej = E[j]
        n_edge = xp.stack([ej[:, 1], -ej[:, 0]], axis=-1)
        n_edge = n_edge / xp.sqrt(_sum(xp, n_edge * n_edge))[:, None]

        safe = xp.where(mag > 1e-13, mag, xp.ones_like(mag))
        n_far = v / safe[:, None]
        on = (mag <= 1e-13)[:, None]
        n = xp.where(on, n_edge, n_far)

        nxs.append(n[:, 0])
        nys.append(n[:, 1])

    nx = xp.reshape(xp.concat(nxs, axis=0), (Ny, Nx))
    ny = xp.reshape(xp.concat(nys, axis=0), (Ny, Nx))
    return nx, ny


# ============================================ convolution matrices for one layer
def _k_grids(xp, fto_x, fto_y, dtype):
    ky = xp.asarray([float(k) for k in range(-2 * fto_y, 2 * fto_y + 1)], dtype=dtype)
    kx = xp.asarray([float(k) for k in range(-2 * fto_x, 2 * fto_x + 1)], dtype=dtype)
    KY = ky[:, None] + 0.0 * kx[None, :]
    KX = 0.0 * ky[:, None] + kx[None, :]
    return KY, KX


def _centroid(xp, V):
    """Area centroid of a simple polygon (not the vertex mean)."""
    x, y = V[:, 0], V[:, 1]
    xn, yn = xp.roll(x, -1), xp.roll(y, -1)
    cr = x * yn - xn * y
    a = 0.5 * xp.sum(cr)
    return xp.stack([xp.sum((x + xn) * cr) / (6.0 * a),
                     xp.sum((y + yn) * cr) / (6.0 * a)])


def check_normal_grid(grid_shape, fto_x, fto_y):
    """`4*fto+1` points per axis, or `_sampled_coeffs` aliases distinct orders onto one bin.

    Structural, not gradual: a *constant* normal field has to transform to the identity for NVF
    to collapse onto Li's rules, and on a short grid it picks up off-diagonals at +/-n instead.
    Same floor `factorization._normal_field_floor` enforces on the raster path.
    """
    Ny, Nx = grid_shape
    need_y, need_x = 4 * fto_y + 1, 4 * fto_x + 1
    if Ny < need_y or Nx < need_x:
        raise ValueError(f'normal-field grid {Ny}x{Nx} aliases at fto=({fto_x}, {fto_y}): '
                         f'needs at least {need_y}x{need_x}')


def normal_projector_coeffs(xp, loops, grid_shape, period, fto_x, fto_y, dtype, cdtype):
    """Coefficient grids of ``nx**2``, ``nx*ny``, ``ny**2``, built with the layer CENTRED.

    The normal field lives on the whole plane, so sampling it in a fixed cell clips a different
    part of it depending on where the shape sits -- a bias, not a discretisation: it does not
    shrink with the grid. Building at the centre and carrying the *coefficients* by the shift
    theorem makes the clipping identical every time, and the position dependence exact.

    It only works when the whole field translates as one, i.e. **a single loop**. With several
    loops, moving one reshapes the field between them and no phase describes that.
    """
    check_normal_grid(grid_shape, fto_x, fto_y)
    Px, Py = period

    if len(loops) == 1:
        c = _centroid(xp, loops[0])
        shift = xp.stack([c[0] - Px / 2.0, c[1] - Py / 2.0])
        built = [loops[0] - shift[None, :]]
        KY, KX = _k_grids(xp, fto_x, fto_y, dtype)
        arg = -2.0 * 3.141592653589793 * (KY * (shift[1] / Py) + KX * (shift[0] / Px))
        phase = xp.astype(xp.cos(arg), cdtype) + 1j * xp.astype(xp.sin(arg), cdtype)
    else:
        built = loops
        phase = xp.asarray(1.0 + 0j, dtype=cdtype)

    nx, ny = distance_gradient_normal_field(xp, built, grid_shape, period, dtype)
    out = []
    for f in (nx * nx, nx * ny, ny * ny):
        out.append(_sampled_coeffs(xp, xp.astype(f, cdtype), fto_x, fto_y) * phase)
    return out


def conv_mats(xp, regions, n_base, fto_x, fto_y, period, normal_grid, dtype, cdtype):
    """``(epx, epy, epxy, epyx, epz_i)`` for one boundary-resolved layer.

    ``regions`` is ``[(n_index, [loop, ...]), ...]`` -- disjoint, painting already resolved by
    the clipper. Hole loops carry a reversed winding, which needs no special case: the polygon
    FT is a signed contour integral, so a hole subtracts by itself.
    """
    Px, Py = period
    KY, KX = _k_grids(xp, fto_x, fto_y, dtype)
    is_dc = (KY == 0) & (KX == 0)
    zero = xp.zeros(KY.shape, dtype=cdtype)

    eps_bg = xp.conj(xp.astype(xp.asarray(n_base, dtype=cdtype), cdtype)) ** 2
    eps_c = xp.where(is_dc, eps_bg, zero)
    inv_c = xp.where(is_dc, 1.0 / eps_bg, zero)

    for n_index, polys in regions:
        eps_s = xp.conj(xp.astype(xp.asarray(n_index, dtype=cdtype), cdtype)) ** 2
        ft = zero
        for V in polys:
            ft = ft + polygon_ft(xp, V, KY, KX, (Px, Py))
        eps_c = eps_c + (eps_s - eps_bg) * ft
        inv_c = inv_c + (1.0 / eps_s - 1.0 / eps_bg) * ft

    ep_dir = laurent_conv_2d(xp, eps_c, fto_x, fto_y)
    ep_inv = laurent_conv_2d(xp, inv_c, fto_x, fto_y)
    delta = ep_dir - xp.linalg.inv(ep_inv)

    all_loops = [V for _, polys in regions for V in polys]
    c_xx, c_xy, c_yy = normal_projector_coeffs(
        xp, all_loops, (normal_grid, normal_grid), (Px, Py), fto_x, fto_y, dtype, cdtype)
    C_xx = laurent_conv_2d(xp, c_xx, fto_x, fto_y)
    C_xy = laurent_conv_2d(xp, c_xy, fto_x, fto_y)
    C_yy = laurent_conv_2d(xp, c_yy, fto_x, fto_y)

    eps_xx = ep_dir - 0.5 * (delta @ C_xx + C_xx @ delta)
    eps_yy = ep_dir - 0.5 * (delta @ C_yy + C_yy @ delta)
    eps_xy = -0.5 * (delta @ C_xy + C_xy @ delta)
    return eps_xx, eps_yy, -eps_xy, -eps_xy, xp.linalg.inv(ep_dir)
