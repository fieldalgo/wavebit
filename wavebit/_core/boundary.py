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

The block path paints later shapes over earlier ones but decomposes them into axis-aligned
blocks, so a tilted edge becomes a staircase. The smooth path transforms a tilted shape in
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
block path, while tying on axis-aligned overlap where the block path has no staircase to lose.
"""
from . import clipper as _clipper
from .analytic import nvf_from_coeffs, shape_geometry, _sampled_coeffs
from .lattice import sample_grid
from .polygon import ellipse_verts, polygon_ft, rect_verts


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


def distance_gradient_normal_field(xp, polys, grid_shape, period, dtype, chunk=None,
                                   lattice=None):
    """Unit normal from the direction to the closest boundary *point*.

    The closest *edge* is not well defined -- beyond a corner both edges meeting there have the
    corner as their closest point, at exactly equal distance, so `argmin`'s tie-break decides on
    29-45% of the cell. The closest point is unique except on the medial axis (measure zero).

    Sign flips inside/outside, which does not matter: NVF only uses ``nx**2``, ``nx*ny``,
    ``ny**2``.

    The sample points come from :func:`wavebit._core.lattice.sample_grid`, shared with the
    analytic builder — this body was written by copying that one, and copying it is how it
    inherited a reversed ``period`` unpack that sampled over a transposed cell.

    ``chunk`` is how many pixels are handled at once. It bounds the ``pixels x segments``
    temporaries without changing any result -- the arithmetic per pixel is independent, so a
    chunked run is bit-identical to a flat one. ``None`` targets 4M entries per temporary;
    measured peak is about 5x that, since `_seg_dist` has several ``(chunk, M, 2)`` arrays
    alive at once. Peak then stops growing with the grid, which is the point.
    """
    Ny, Nx = grid_shape
    A, E, L2 = segments_of(xp, polys)
    M = A.shape[0]

    X, Y = sample_grid(xp, grid_shape, period, dtype, lattice)
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


def normal_projector_coeffs(xp, loops, grid_shape, period, fto_x, fto_y, dtype, cdtype,
                            lattice=None):
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
        if lattice is None:
            mid_x, mid_y = Px / 2.0, Py / 2.0
            frac_x, frac_y = 1.0 / Px, 1.0 / Py           # fractions of a period
        else:
            a1x, a2x = float(lattice[0][0]), float(lattice[0][1])
            a1y, a2y = float(lattice[1][0]), float(lattice[1][1])
            mid_x, mid_y = 0.5 * (a1x + a2x), 0.5 * (a1y + a2y)
        shift = xp.stack([c[0] - mid_x, c[1] - mid_y])
        built = [loops[0] - shift[None, :]]
        KY, KX = _k_grids(xp, fto_x, fto_y, dtype)
        if lattice is None:
            su, sv = shift[0] * frac_x, shift[1] * frac_y
        else:
            # The displacement in *fractional lattice* coordinates, A^-1 @ shift. The phase is
            # already exp(-i G.d) written that way; it was only ever handed Px / Py fractions.
            det = a1x * a2y - a2x * a1y
            su = (a2y * shift[0] - a2x * shift[1]) / det
            sv = (-a1y * shift[0] + a1x * shift[1]) / det
        arg = -2.0 * 3.141592653589793 * (KY * sv + KX * su)
        phase = xp.astype(xp.cos(arg), cdtype) + 1j * xp.astype(xp.sin(arg), cdtype)
    else:
        built = loops
        phase = xp.asarray(1.0 + 0j, dtype=cdtype)

    nx, ny = distance_gradient_normal_field(xp, built, grid_shape, period, dtype,
                                            lattice=lattice)
    out = []
    for f in (nx * nx, nx * ny, ny * ny):
        out.append(_sampled_coeffs(xp, xp.astype(f, cdtype), fto_x, fto_y) * phase)
    return out


def conv_mats(xp, regions, n_base, fto_x, fto_y, period, normal_grid, dtype, cdtype,
              lattice=None):
    """``(epx, epy, epxy, epyx, epz_i)`` for one boundary-resolved layer.

    ``regions`` is ``[(n_index, [(loop, sign), ...]), ...]`` -- disjoint, painting already
    resolved by the clipper, ``sign`` being ``+1`` material and ``-1`` hole.

    **The sign has to be applied here.** It used to be dropped, on the reasoning that a hole
    carries a reversed winding and so subtracts itself through the signed contour integral.
    The polygon FT is indeed signed, but :func:`~.clipper.subtract_loop` *normalises every loop
    to counter-clockwise* and carries the orientation in ``sign`` instead -- precisely so the
    next subtraction can hand the loop straight back to the clipper. A hole therefore arrives
    with a **positive** ``signed_area``, and dropping the sign adds it where it should have
    been taken away. :func:`~.clipper.region_area` and :func:`~.clipper.region_ft` both
    multiply by it; this was the one reader that did not.

    Nothing catches that downstream: the resulting matrices are still Hermitian and the solve
    still conserves energy exactly. A shape painted *entirely inside* another one was solved as
    a different structure -- measured ``<eps> = 2.4256`` against an exact 2.2371 for a disc of
    index 3 inside a square of index 2, at every ``fto`` alike, and ``sum(R) = 0.1798`` against
    a converged raster's 0.4164.

    The normal field takes the loops **unsigned**, and that is not an oversight. It is the
    direction to the nearest boundary *point*, so it depends on which segments exist and not on
    how they are wound; only ``nx**2``, ``nx*ny``, ``ny**2`` are used, so even the overall sign
    of the vector cancels. Measured by reversing every hole loop before building the field:
    the coefficient grids move by 5.99e-6 (``c_xx``, ``c_yy``) and 7.0e-18 (``c_xy``), which is
    ``argmin`` tie-breaking on the medial axis rather than a different field. That measurement
    is also why the sign is applied to the transform rather than by reversing the winding: both
    give the same permittivity to 2e-16, but reversing perturbs the normal field by that 6e-6.
    """
    Px, Py = period
    KY, KX = _k_grids(xp, fto_x, fto_y, dtype)
    is_dc = (KY == 0) & (KX == 0)
    zero = xp.zeros(KY.shape, dtype=cdtype)

    # `n**2`, not `conj(n)**2`. Wavebit works in `e^{-iwt}` throughout (`Im(n) > 0` = loss)
    # with no conjugation on the way in — the same sentence `analytic.assemble_eps_coeffs`
    # carries, and this builder is its later copy. It kept the conjugate that the smooth path
    # was moved off in 2026-08, which turned an absorbing painted layer into a gain medium:
    # `R + T` up to 1.11 and reflectance off by 0.16 on gold. Invisible to `test_boundary.py`,
    # which builds every shape with a real `n_index`, where the two spellings are identical.
    # See `docs/time-convention-audit.md` §6.
    eps_bg = xp.astype(xp.asarray(n_base, dtype=cdtype), cdtype) ** 2
    eps_c = xp.where(is_dc, eps_bg, zero)
    inv_c = xp.where(is_dc, 1.0 / eps_bg, zero)

    for n_index, polys in regions:
        eps_s = xp.astype(xp.asarray(n_index, dtype=cdtype), cdtype) ** 2
        ft = zero
        for V, sgn in polys:
            ft = ft + sgn * polygon_ft(xp, V, KY, KX, (Px, Py), lattice)
        eps_c = eps_c + (eps_s - eps_bg) * ft
        inv_c = inv_c + (1.0 / eps_s - 1.0 / eps_bg) * ft

    all_loops = [V for _, polys in regions for V, _sgn in polys]
    c_xx, c_xy, c_yy = normal_projector_coeffs(
        xp, all_loops, (normal_grid, normal_grid), (Px, Py), fto_x, fto_y, dtype, cdtype,
        lattice=lattice)
    return nvf_from_coeffs(xp, eps_c, inv_c, c_xx, c_xy, c_yy, fto_x, fto_y)


# ================================================== one painted layer, end to end

#: Vertices used to polygonise a curved primitive for this path. A polygonised ellipse
#: converges on the true one at O(1/n^2), and 256 puts that error far below the normal
#: field's, which is this path's floor. Straight-edged primitives are exact and ignore it.
POLYGON_VERTICES = 256


def shape_vertices(xp, shape, period, n_poly=POLYGON_VERTICES):
    """Counter-clockwise vertices of one analytic primitive.

    Same shape dictionaries the smooth path takes, and the same convention: ``lx`` / ``ly``
    are full lengths, so a circle is ``lx = ly = diameter``.

    The dict is read by :func:`~wavebit._core.analytic.shape_geometry` and nowhere else, for
    the reason that function was written: it used to be read here as well, directly, and the
    two readings disagreed. ``ly`` had no default here at all, so the natural spelling of a
    circle — give the diameter once — summed fine and raised ``KeyError`` the moment the same
    layer was painted; ``cx`` / ``cy`` had no default either, where the transform centres a
    shape in the cell; and ``square`` kept whatever ``ly`` it was handed instead of being
    square. ``period`` is taken for the same reason the transform takes it: it is what an
    omitted centre falls back to.
    """
    kind, cx, cy, lx, ly, angle = shape_geometry(shape, period[0], period[1])
    if kind in ('rectangle', 'square'):
        return rect_verts(xp, cx, cy, lx, ly, angle)
    if kind in ('circle', 'disk', 'ellipse'):
        return ellipse_verts(xp, cx, cy, lx, ly, angle, n_poly)
    raise NotImplementedError(
        f"The boundary path cannot draw a {kind!r} yet; it needs the shape as a polygon. "
        f"Supported: rectangle, square, circle, disk, ellipse.")


def _concrete_shape(shape):
    """Every numeric value of a shape dict as a plain Python float.

    :func:`plan_layer` is documented "concrete values only" and was relying on the caller to
    have honoured it. NumPy and JAX let that slide -- a JAX array satisfies ``__array__`` and
    converts on contact -- while a PyTorch tensor survives all the way into ``rect_verts`` and
    multiplies against a NumPy array, which raises. Enforcing the contract here is what makes
    the planning stage namespace-agnostic, which is what it always claimed to be.

    Values that are not numbers are passed through untouched: a shape's ``type``, and a complex
    ``n_index``, which ``float()`` would refuse and which the planner never reads anyway.
    """
    out = {}
    for k, v in shape.items():
        detach = getattr(v, 'detach', None)
        if detach is not None:
            v = detach()
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def plan_layer(shapes, period, n_poly=POLYGON_VERTICES):
    """Resolve one layer's painting order into disjoint loops. **Concrete values only.**

    This is the discrete half — which loop holds which vertex, which edges cross — so it runs
    on NumPy outside any trace, exactly like the block modeler's grid plan. Freeze it once
    (:meth:`~wavebit._core.engine.RCWAEngine.freeze_geometry`) and every later step replays
    it against traced vertices.
    """
    import numpy as _np
    concrete = [_np.asarray(shape_vertices(_np, _concrete_shape(s), period, n_poly), float)
                for s in shapes]
    return _clipper.resolve_painting(concrete)


def to_conv_mat_boundary_layer(xp, shapes, n_base, fto_x, fto_y, period, dtype,
                               normal_grid=256, n_poly=POLYGON_VERTICES, plan=None,
                               lattice=None):
    """The five convolution matrices for one layer whose shapes are *painted*.

    ``shapes`` is in painting order — a later shape covers an earlier one — which is what the
    block path means by painting and what a raster of the same instruction agrees with. Pass
    ``plan`` from a frozen geometry to keep the call traceable; leave it ``None`` and the
    painting is resolved eagerly, which needs concrete shape parameters.
    """
    verts = [shape_vertices(xp, s, period, n_poly) for s in shapes]
    recs, regions = plan_layer(shapes, period, n_poly) if plan is None else plan
    vals = _clipper.eval_table(xp, recs, verts)
    regs = [(s['n_index'], [(_clipper.loop_verts(xp, lp, vals), sgn) for lp, sgn in region])
            for s, region in zip(shapes, regions)]
    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32
    return conv_mats(xp, regs, n_base, fto_x, fto_y, period, normal_grid, real_dtype, dtype,
                     lattice=lattice)
