# -*- coding: utf-8 -*-
"""A layer given as a level set: ``eps = eps_in`` where ``phi > 0``, ``eps_base`` elsewhere.

The fifth modeling path, and the one a topology optimizer wants. The four before it all take
the geometry as *shapes* — a raster of them, a block decomposition, closed-form transforms, or
polygons with the overlap resolved — so all four fix the topology in the recipe. A level set
puts the geometry in a field, where merging and splitting cost nothing, which is why it is the
standard representation for metasurface topology optimisation.

The obvious way to feed one to this solver is to threshold ``phi`` onto the raster path. That
gives up both things this path exists to keep:

===============  ==========================  ====================================
                 boundary                    gradient w.r.t. the design
===============  ==========================  ====================================
raster + NVF     staircase, and the normal   ``d/d eps_pixel`` — a *density*
                 field is a distance-        gradient, and the boundary's own
                 transform guess at the      motion lives in pixel-scale jumps
                 smooth boundary behind it   that autodiff steps over
**this one**     the zero contour itself,    ``d/d phi`` analytically, through
                 transformed in closed form  the contour vertices
===============  ==========================  ====================================

Same split as :mod:`.boundary`, and for the same reason:

* **planning** — which triangles the contour crosses, which side of it each node is on.
  Discrete, NumPy, outside any trace. Records **indices**, never coordinates.
* **replay** — everything the plan is then evaluated with. Arithmetic only, so it traces and
  differentiates on every backend, and ``d(answer)/d phi`` comes back exact.

That split is what marching squares is *made of*, which is the whole reason this is cheap: the
discrete half is the case table, and the continuous half is one linear interpolation per
crossing, ``t = phi_a / (phi_a - phi_b)``. A vertex is therefore a differentiable function of
the two node values either side of it, and the shape derivative — the boundary velocity a
level-set update needs — arrives without a smoothed delta anywhere.

Three choices worth knowing about.

**Triangles, not squares.** Marching *squares* has two ambiguous cases (the saddles), where the
contour and the enclosed area have to be resolved by a separate rule and can then disagree with
each other. A triangle has three corners and eight cases, none ambiguous, so the contour and the
area are two readings of one decomposition and agree by construction — which matters here
because the area *is* the ``k = 0`` Fourier coefficient. The diagonal alternates on a
checkerboard so the triangulation carries no directional bias.

**No loops are assembled.** The transform is a divergence-theorem contour integral, i.e. a sum
over boundary segments; the region's boundary is closed *on the torus*, and ``exp(-i G.r)`` is a
function on that torus. So which loop a segment belongs to never enters, and a design that
percolates across the cell — the case a polygon representation has to special-case — needs
nothing extra here. See :func:`fieldwave._core.polygon.edge_ft`. The one order that would have
needed the loops is ``k = 0``, and the marching-triangle area supplies it directly.

**The normal field is ``grad phi``, not a distance transform.** The raster path has to *guess*
the normal of the smooth boundary its staircase stands for, and :mod:`.boundary` has to
reconstruct a distance field by scanning every pixel against every segment (the cost that makes
that path expensive on a many-segment boundary). A level set already carries the field whose
gradient is that normal, so here it is one central difference: exact where ``phi`` is a signed
distance function, free, and differentiable in ``phi`` like everything else. Reinitialising
``phi`` to a signed distance — which level-set schemes do anyway, for their own reasons — is
what keeps ``|grad phi|`` away from zero and the field meaningful in the bulk.
"""
import numpy as _np

from .analytic import _sampled_coeffs, nvf_from_coeffs
from .boundary import check_normal_grid
from .polygon import edge_ft

# Corner pairs of a triangle's three edges, in CCW order: edge e runs from corner A[e] to B[e].
_EDGE_A = (0, 1, 2)
_EDGE_B = (1, 2, 0)

# Per sign case (bit c set = corner c has phi > 0), the edge the boundary segment starts on and
# the one it ends on. Directed so the *material* is on the left, i.e. the outward normal is
# (ey, -ex) — the orientation `polygon.edge_ft` integrates against.
#
# One corner inside: from the crossing on (c -> c_next) to the one on (c_prev -> c). Two inside
# is the same rule read off the single *outside* corner and reversed. Cases 0 and 7 have no
# segment; they are listed as (-1, -1) and never indexed.
_SEG_START = _np.array([-1, 0, 1, 1, 2, 0, 2, -1])
_SEG_END = _np.array([-1, 2, 0, 2, 1, 1, 0, -1])

# Per sign case, the material sub-polygon inside the triangle, CCW, as four vertex slots.
# A slot is a corner index 0..2, or -(e + 1) for the crossing on edge e. Cases needing only
# three vertices repeat the last one, which contributes nothing to the shoelace; case 0 is four
# copies of one corner, i.e. zero area.
_AREA_POLY = _np.array([
    [0, 0, 0, 0],           # 0: nothing inside
    [0, -1, -3, -3],        # 1: c0            -> (c0, X01, X20)
    [1, -2, -1, -1],        # 2: c1            -> (c1, X12, X01)
    [0, 1, -2, -3],         # 3: c0, c1        -> (c0, c1, X12, X20)
    [2, -3, -2, -2],        # 4: c2            -> (c2, X20, X12)
    [2, 0, -1, -2],         # 5: c2, c0        -> (c2, c0, X01, X12)
    [1, 2, -3, -1],         # 6: c1, c2        -> (c1, c2, X20, X01)
    [0, 1, 2, 2],           # 7: everything inside
])


class LevelSetPlan:
    """The discrete half of one level-set layer: which triangles the contour crosses.

    Pure indices into ``phi`` plus constant node coordinates — no ``phi`` *values*, which is
    what lets the replay carry a gradient to them. Compares and hashes **by content**, like the
    block and boundary plans, so re-freezing a design whose sign pattern has not moved hits the
    JAX compilation cache instead of recompiling (see :class:`~._core.geometry.StaticGeometry`).

    A level-set step that *does* move the sign pattern gets a different plan, and must: that
    change is a topology change, and it is exactly what the discrete half is there to record.
    """

    __slots__ = ('shape', 'period', 'seg', 'area', 'const_area', 'signs', '_key')

    def __init__(self, shape, period, seg, area, const_area, signs):
        self.shape = shape                  #: (Ny, Nx) of the phi grid
        self.period = period                #: (Px, Py) the node coordinates were built on
        self.seg = seg                      #: (start_spec, end_spec) for every boundary segment
        self.area = area                    #: four vertex specs per partly-filled triangle
        self.const_area = const_area        #: area of the wholly-filled triangles, a constant
        self.signs = signs                  #: phi > 0 per node, packed — what staleness reads
        self._key = (shape, period, signs.tobytes())

    def __eq__(self, other):
        return isinstance(other, LevelSetPlan) and self._key == other._key

    def __hash__(self):
        return hash(self._key)

    def __repr__(self):
        return (f'LevelSetPlan(shape={self.shape}, segments={self.seg[0][0].shape[0]}, '
                f'mixed_triangles={self.area[0].shape[0]})')

    def is_stale_for(self, phi):
        """Has the sign pattern moved since this plan was taken?

        The plan replays the *same* case table and the *same* crossing edges. On a ``phi`` whose
        signs have changed, those edges no longer carry a crossing: ``t = phi_a / (phi_a -
        phi_b)`` leaves ``[0, 1]``, and the answer comes back wrong without looking wrong. Same
        contract as every other frozen record — checked eagerly, never under a trace.
        """
        return bool((_np.asarray(phi) > 0).reshape(-1).astype(_np.uint8).tobytes()
                    != self.signs.tobytes())


# ------------------------------------------------------------------ planning (NumPy, eager)

def _triangles(Ny, Nx, Px, Py):
    """Node indices and node positions of the ``2 * Ny * Nx`` triangles, all CCW.

    ``phi`` is sampled at pixel centres, ``((j + 0.5) / Nx * Px, (i + 0.5) / Ny * Py)`` — the
    same convention :func:`fieldwave._core.boundary.distance_gradient_normal_field` samples its
    normal field on, so the two paths' fields land on the same points and can be compared.

    Positions are **local to each triangle and never wrapped**: a cell on the last row uses
    ``y + dy``, which is outside the cell. That is deliberate and it is what makes the periodic
    boundary a non-event — every segment is then short and correct in its own frame, and
    ``exp(-i G.r)`` maps it back onto the torus by itself.
    """
    I, J = _np.meshgrid(_np.arange(Ny), _np.arange(Nx), indexing='ij')
    dx, dy = Px / Nx, Py / Ny
    Ip, Jp = (I + 1) % Ny, (J + 1) % Nx

    # Corner order [SW, SE, NE, NW] is CCW with x right and y up.
    nodes = _np.stack([I * Nx + J, I * Nx + Jp, Ip * Nx + Jp, Ip * Nx + J], axis=-1)
    x0, y0 = (J + 0.5) * dx, (I + 0.5) * dy
    pos = _np.stack([_np.stack([x0, y0], -1), _np.stack([x0 + dx, y0], -1),
                     _np.stack([x0 + dx, y0 + dy], -1), _np.stack([x0, y0 + dy], -1)], axis=-2)

    # Alternate the diagonal so the triangulation has no preferred direction.
    even = ((I + J) % 2 == 0)[..., None, None]
    sel = _np.where(even, _np.array([[0, 1, 2], [0, 2, 3]]), _np.array([[0, 1, 3], [1, 2, 3]]))

    tri_nodes = _np.take_along_axis(_np.broadcast_to(nodes[:, :, None, :], (Ny, Nx, 2, 4)),
                                    sel, axis=-1)
    tri_pos = _np.take_along_axis(_np.broadcast_to(pos[:, :, None, :, :], (Ny, Nx, 2, 4, 2)),
                                  sel[..., None], axis=-2)
    return tri_nodes.reshape(-1, 3), tri_pos.reshape(-1, 3, 2)


def _vertex_spec(tri_nodes, tri_pos, slot):
    """``(node_a, node_b, base, delta)`` for one vertex slot of every triangle.

    Uniform over both kinds of vertex: a crossing interpolates between two nodes, a corner is
    the degenerate case ``node_a == node_b`` with ``delta == 0``, so the replay is one
    expression and no branch. ``slot`` is a corner index, or ``-(e + 1)`` for a crossing on
    edge ``e``.
    """
    rows = _np.arange(tri_nodes.shape[0])
    corner = slot >= 0
    a = _np.where(corner, _np.maximum(slot, 0), _np.take(_EDGE_A, _np.maximum(-slot - 1, 0)))
    b = _np.where(corner, _np.maximum(slot, 0), _np.take(_EDGE_B, _np.maximum(-slot - 1, 0)))
    na, nb = tri_nodes[rows, a], tri_nodes[rows, b]
    pa, pb = tri_pos[rows, a], tri_pos[rows, b]
    delta = _np.where(corner[:, None], 0.0, pb - pa)
    return na, nb, pa, delta


def plan_layer(phi, period):
    """Take the discrete decisions of one level-set layer. **Concrete values only.**

    ``phi`` is an (Ny, Nx) array; material is ``phi > 0``. Exact zeros count as *outside*, which
    makes the crossing denominator ``phi_a - phi_b`` non-zero on every edge a crossing was found
    on — the one degeneracy that would otherwise divide by zero.
    """
    # Detaching is the statement, not a workaround: planning reads *which side of zero*, a
    # discrete fact that carries no derivative, and every coordinate it goes on to record is a
    # constant. The gradient reaches phi through the replay instead.
    phi = _np.asarray(phi.detach() if hasattr(phi, 'detach') else phi, float)
    if phi.ndim != 2:
        raise ValueError(f'A level-set layer needs a 2D phi; got shape {phi.shape}.')
    Ny, Nx = phi.shape
    Px, Py = float(period[0]), float(period[1])

    tri_nodes, tri_pos = _triangles(Ny, Nx, Px, Py)
    inside = (phi.reshape(-1) > 0)[tri_nodes]
    case = inside[:, 0] + 2 * inside[:, 1] + 4 * inside[:, 2]

    crossed = (case != 0) & (case != 7)
    cn, cp, cc = tri_nodes[crossed], tri_pos[crossed], case[crossed]
    seg = (_vertex_spec(cn, cp, -_SEG_START[cc] - 1),
           _vertex_spec(cn, cp, -_SEG_END[cc] - 1))

    area = tuple(_vertex_spec(cn, cp, _AREA_POLY[cc, k]) for k in range(4))

    full = tri_pos[case == 7]
    x, y = full[:, :, 0], full[:, :, 1]
    xn, yn = _np.roll(x, -1, axis=-1), _np.roll(y, -1, axis=-1)
    const_area = float(0.5 * _np.sum(x * yn - xn * y))

    signs = (phi.reshape(-1) > 0).astype(_np.uint8)
    return LevelSetPlan((Ny, Nx), (Px, Py), seg, area, const_area, signs)


# ------------------------------------------------------------------ replay (traced, any backend)

def _device(x):
    return x.device if hasattr(x, 'device') else None


def _interp(xp, phi_flat, spec, dtype):
    """``base + t * delta`` for one vertex slot, with ``t = phi_a / (phi_a - phi_b)``.

    A corner slot has ``node_a == node_b``, so the denominator is exactly zero. The guard
    replaces the **denominator** rather than the result, for the reason
    ``tests/test_nvf_zero_norm_gradients.py`` records: patching a result leaves autodiff
    differentiating the branch that was discarded, and ``0 * inf`` is NaN.
    """
    na, nb, base, delta = spec
    dev = _device(phi_flat)
    ia = xp.asarray(na.reshape(-1), device=dev)
    ib = xp.asarray(nb.reshape(-1), device=dev)
    fa = xp.reshape(xp.take(phi_flat, ia, axis=0), na.shape)
    fb = xp.reshape(xp.take(phi_flat, ib, axis=0), nb.shape)

    den = fa - fb
    is_corner = xp.asarray(na == nb, device=dev)
    safe = xp.where(is_corner, xp.ones_like(den), den)
    t = xp.where(is_corner, xp.zeros_like(den), fa / safe)

    B = xp.asarray(base, dtype=dtype, device=dev)
    D = xp.asarray(delta, dtype=dtype, device=dev)
    return B + t[..., None] * D


def boundary_and_area(xp, phi, plan, dtype):
    """``(A, E, area)`` — the directed boundary segments and the material area.

    Both are read off the *same* marching-triangle decomposition, so the area is exactly the one
    enclosed by the segments. That is what lets ``k = 0`` and ``k != 0`` come from two different
    formulas without disagreeing.
    """
    phi_flat = xp.reshape(phi, (-1,))
    A = _interp(xp, phi_flat, plan.seg[0], dtype)
    B = _interp(xp, phi_flat, plan.seg[1], dtype)

    P = xp.stack([_interp(xp, phi_flat, spec, dtype) for spec in plan.area], axis=-2)
    x, y = P[..., 0], P[..., 1]
    xn, yn = xp.roll(x, -1, axis=-1), xp.roll(y, -1, axis=-1)
    area = plan.const_area + 0.5 * xp.sum(x * yn - xn * y)
    return A, B - A, area


def normal_field(xp, phi, period):
    """Unit normal ``(nx, ny)`` of the level set: ``grad phi / |grad phi|``, periodic.

    Unit magnitude across the **whole** cell, not just at the interface — that is what makes
    ``C_xx + C_yy = I`` in the factorization, and it is the same requirement the raster path's
    distance transform exists to meet. Where the gradient vanishes outright (an extremum of
    ``phi``, or its medial axis) there is no direction to report and ``(1, 0)`` is substituted;
    with ``phi`` a signed distance function that locus is measure zero, exactly as it is for the
    distance transform.

    The radicand, not the root, is what the guard replaces — see :func:`_interp`.
    """
    Ny, Nx = phi.shape
    Px, Py = float(period[0]), float(period[1])          # axis 1 is x, axis 0 is y
    gy = (xp.roll(phi, -1, axis=0) - xp.roll(phi, 1, axis=0)) / (2.0 * Py / Ny)
    gx = (xp.roll(phi, -1, axis=1) - xp.roll(phi, 1, axis=1)) / (2.0 * Px / Nx)

    r2 = gx * gx + gy * gy
    ok = r2 > 1e-30
    inv = 1.0 / xp.sqrt(xp.where(ok, r2, xp.ones_like(r2)))
    return (xp.where(ok, gx * inv, xp.ones_like(gx)),
            xp.where(ok, gy * inv, xp.zeros_like(gy)))


# ------------------------------------------------------------------ one layer, end to end

def _reject_skewed(lattice):
    """A rectangular basis is fine; a sheared one is not, and the difference is where.

    ``diag(period)`` is what an axis-aligned cell is handed everywhere in this codebase, and it
    changes nothing here: :func:`fieldwave._core.polygon._frequencies` reduces to the plain
    ``2*pi*k/P``, and the normal field already carries ``Px/Nx`` and ``Py/Ny`` per axis, so a
    merely *non-square* cell is served exactly. A **sheared** one is not: the central difference
    below is taken in index steps, which only a rectangular basis makes Cartesian, and pulling it
    back through ``J^-T`` (the correction
    :func:`fieldwave._core.factorization.normal_vector_field` applies on the raster path) is not
    written here.
    """
    if lattice is None:
        return
    if float(lattice[0][1]) == 0.0 and float(lattice[1][0]) == 0.0:
        return
    raise NotImplementedError(
        'The level-set path is rectangular-lattice only: its normal field is a central '
        'difference on the phi grid, which a sheared basis would have to pull back through '
        'J^-T before normalising, and nothing here does that yet.')


def to_conv_mat_levelset_layer(xp, layer, fto_x, fto_y, period, dtype, plan=None,
                               lattice=None):
    """The five convolution matrices for one ``dict(n_base=..., phi=..., n_index=...)`` layer.

    Pass ``plan`` from a frozen geometry to keep the call traceable; leave it ``None`` and the
    marching-triangle plan is taken eagerly, which needs a concrete ``phi``.

    The normal field is taken on the ``phi`` grid itself — no resampling, and therefore no
    interpolation error between the field and the contour drawn from the same values. That puts
    ``4 * fto + 1`` on the *design* grid, which is the same floor
    :func:`fieldwave._core.boundary.check_normal_grid` enforces everywhere else and for the same
    structural reason.
    """
    _reject_skewed(lattice)

    phi = layer['phi']
    n_index = layer['n_index']              # presence checked by analytic.check_layer_spec
    n_base = layer.get('n_base', layer.get('base', 1.0))

    check_normal_grid(tuple(phi.shape), fto_x, fto_y)
    if plan is None:
        plan = plan_layer(phi, period)
    elif tuple(phi.shape) != plan.shape:
        raise ValueError(f'Level-set plan was taken on a {plan.shape} grid, '
                         f'phi is {tuple(phi.shape)}.')
    else:
        # The plan holds absolute node coordinates, so a period it was not built on puts every
        # vertex in the wrong place -- and quietly, since the transform accepts any coordinate.
        # Readable except when `period` is itself the variable under differentiation, and the
        # freeze that produced the plan was eager, so an unreadable one is left alone.
        try:
            here = (float(period[0]), float(period[1]))
        except Exception:                                  # noqa: BLE001 - traced period
            here = plan.period
        if here != plan.period:
            raise ValueError(f'Level-set plan was taken at period {plan.period}, '
                             f'this call is at {here}.')

    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32
    A, E, area = boundary_and_area(xp, phi, plan, real_dtype)

    ky = xp.asarray([float(k) for k in range(-2 * fto_y, 2 * fto_y + 1)], dtype=real_dtype)
    kx = xp.asarray([float(k) for k in range(-2 * fto_x, 2 * fto_x + 1)], dtype=real_dtype)
    KY = ky[:, None] + 0.0 * kx[None, :]
    KX = 0.0 * ky[:, None] + kx[None, :]
    ft = edge_ft(xp, A, E, area, KY, KX, (period[0], period[1]), lattice)

    eps_in = xp.conj(xp.astype(xp.asarray(n_index, dtype=dtype), dtype)) ** 2
    eps_bg = xp.conj(xp.astype(xp.asarray(n_base, dtype=dtype), dtype)) ** 2
    is_dc = xp.astype((KY == 0) & (KX == 0), dtype)
    eps_c = eps_bg * is_dc + (eps_in - eps_bg) * ft
    inv_c = (1.0 / eps_bg) * is_dc + (1.0 / eps_in - 1.0 / eps_bg) * ft

    nx, ny = normal_field(xp, phi, period)
    c_xx, c_xy, c_yy = (_sampled_coeffs(xp, xp.astype(f, dtype), fto_x, fto_y)
                        for f in (nx * nx, nx * ny, ny * ny))
    return nvf_from_coeffs(xp, eps_c, inv_c, c_xx, c_xy, c_yy, fto_x, fto_y)
