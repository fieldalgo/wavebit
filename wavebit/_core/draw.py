"""Constant-z pictures of the *structure*, one recipe per modeling path.

:func:`~wavebit._core.fields.plot_field` draws what came out of a solve. This draws what went
in, which had no helper at all: every tutorial that shows a layer builds the array itself,
before Wavebit sees it, so the two paths where rendering is not trivial -- the block modeler
and the analytic shapes -- were never pictured by anything the library ships.

Presentation rather than physics, and like ``plot_field`` there is no reason for it to exist
per backend: it only needs the geometry as NumPy. Everything here converts on the way in.

**Why a hand-rolled version is worth replacing.** The obvious reading of
:meth:`~wavebit._core.modeler.Modeling.draw` -- edges are right-hand edges, paint the bands in
order -- is silently wrong for any shape that crosses the cell boundary. A ``draw()`` edge list
is not guaranteed increasing: a 200x160 rod at ``cx = 960`` in a 1000x1000 cell comes back as
``x = [860., 1060., 1000.]``, and the segment closed by the last edge has a *negative* width.
The solver is right about this -- :func:`~wavebit._core.factorization._fourier1d_integrated`
weights every segment by its **signed** width, so the wrapped part subtracts and rides on the
Fourier phase: at ``cx`` = 400, 50, 960 and 550 that rod's ``sum(R)`` spreads by 1.4e-13 and
its ``sum(T)`` by 5.4e-14, which is one answer four times over. The picture was not the same
four times: painting the bands in order drops 25.0% / 30.0% / 43.8% of the rod (straddling in
x from the low side, from the high side, and in both axes at once) and puts
``<eps>`` at 1.2700 / 1.2520 / 1.2025 where the solver integrates 1.3600 in all three. At
``cx = 960`` the edge array actually *decreases*, so ``pcolormesh`` draws overlapping quads
without complaining. :func:`_render_block` below does the signed accumulation instead, and
``tests/test_plot_structure.py`` pins it at max\\|render - truth\\| = 0 on those positions --
with the naive render checked in alongside, failing, so the hazard stays demonstrated.
"""
import array_api_compat.numpy as xnp
import numpy as np

from . import analytic as _analytic
from . import boundary as _boundary
from . import clipper as _clipper
from . import factorization as _fac
from .domain import as_material, index_from_eps
from .lattice import sample_grid

#: Periodic images drawn around the cell. A shape's transform is periodic by construction, so
#: the part that leaves one edge returns at the other and the picture has to show it. One ring
#: covers anything up to about a full period across; wider than that is not drawn, which is a
#: limit of the picture and not of the solve.
_IMAGES = (-1, 0, 1)


def _np_real(x):
    """Concrete real NumPy array from any backend's array."""
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    return np.real(np.asarray(x))


def _np_complex(x):
    """Concrete complex NumPy array from any backend's array."""
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    return np.asarray(x, dtype=complex)


def _eps_of(material):
    """One material slot's relative permittivity, as a plain Python ``complex``.

    Through :func:`~wavebit._core.domain.as_material`, so the scalar shorthand and a real
    :class:`~wavebit._core.domain.Material` are read the same way here as in the builder --
    which is the point. A picture that read a shape dict by its own rules is how this module's
    two predecessors came to disagree with the transform about what a dict meant (see
    :func:`~wavebit._core.analytic.shape_geometry`), and a material is the slot where the
    disagreement would be invisible: the shape would be in the right place, made of the wrong
    stuff.
    """
    return complex(np.asarray(as_material(material).epsilon, dtype=complex))


def _passive(eps):
    """``eps`` with round-off-level *negative* imaginary parts flushed to ``+0``.

    Only for the ``show='n'`` conversion, and only because this is the one caller that can
    honestly tell round-off from a medium. :func:`~wavebit._core.domain.index_from_eps` takes
    the principal root and deliberately does not repair the branch -- a helper cannot know
    whether a tiny negative ``Im`` is noise or a deliberately active medium -- and hands that
    responsibility to callers that build a permittivity by subtraction. This is exactly such a
    caller: the band-limited reconstruction is a truncated Fourier sum, so a *lossless* layer
    comes back with imaginary parts of both signs at the 1e-16 level, and the ones that land on
    a pixel whose real part went negative (Gibbs overshoot below zero, which the docstring of
    :func:`slice_structure` measures at ``[-0.186, 13.850]`` for a lossless pair of shapes)
    take the lower lip of the cut. Their root is then ``-0.43j`` -- **gain**, drawn on a
    structure that has none, on 24 of 16384 pixels of the measured case.

    The flush is relative and tiny: an imaginary part below ``1e-12`` of the value's own
    magnitude is noise at double precision, and a real absorber or a real gain medium is orders
    above that. Nothing is clamped on the positive side, so an absorbing layer is untouched.
    """
    eps = np.asarray(eps, dtype=complex)
    noise = (eps.imag < 0) & (np.abs(eps.imag) <= 1e-12 * np.maximum(np.abs(eps), 1.0))
    return np.where(noise, eps.real + 0j, eps)


def _basis(solver):
    """The cell basis as a plain ``(2, 2)`` float array, columns ``a1`` / ``a2``."""
    A = solver.lattice
    if A is None:
        Px, Py = float(np.real(solver.period[0])), float(np.real(solver.period[1]))
        return np.array([[Px, 0.0], [0.0, Py]])
    return np.asarray([[float(A[0][0]), float(A[0][1])],
                       [float(A[1][0]), float(A[1][1])]], dtype=float)


def _periods(solver):
    return float(np.real(solver.period[0])), float(np.real(solver.period[1]))


# ================================================================ path 1: the block modeler

def _segments(edges):
    """``(lo, hi)`` for the segments a ``draw()`` edge list describes.

    Segment ``j`` runs from ``edges[j-1]`` to ``edges[j]``, with the cell's origin closing the
    first -- the same reading :func:`~wavebit._core.factorization._fourier1d_integrated` takes
    when it forms its DC term. **Oriented**, not sorted: ``hi < lo`` is a real case (a shape
    that crossed the cell boundary) and means the segment subtracts.
    """
    hi = _np_real(edges).reshape(-1)
    lo = np.concatenate([[0.0], hi[:-1]])
    return lo, hi


def _signed_coverage(lo, hi, s, L):
    """Signed count of periodic images of the oriented interval ``(lo, hi]`` containing ``s``.

    ``floor((hi - s)/L) - floor((lo - s)/L)`` counts the integers ``k`` with
    ``lo < s + k*L <= hi``, which is negative exactly when the interval is reversed. That is the
    signed width of :func:`~wavebit._core.factorization._fourier1d_integrated` resolved back
    into space, and it telescopes: the segments of one axis sum to 1 at every sample, because
    the edge list ends at the period and starts at 0.

    Samples sit at segment *centres*, so a boundary landing exactly on a sample -- where the
    two floors would be one ulp from disagreeing -- takes a coincidence rather than an aligned
    design. A shape whose edge sits at a sample centre may still show a one-pixel seam; raise
    ``n`` and it moves, which is what identifies it.
    """
    return (np.floor((hi[:, None] - s[None, :]) / L)
            - np.floor((lo[:, None] - s[None, :]) / L))


def _render_block(cell, x_list, y_list, period, xs, ys):
    """``eps`` on the grid ``(ys, xs)`` for one entry of :meth:`.Modeling.draw`.

    ``cell`` is permittivity already -- the modeler takes Materials and writes out their
    ``epsilon`` -- indexed ``[row, col]`` = ``[y, x]``, with row 0 at ``y = 0``
    (``docs/conventions.md``). There is no squaring here for the same reason there is none in
    the builder this mirrors: a picture that squared what the solve does not would show a
    different structure from the one being solved, which is the one thing a structure picture
    must not do. Exact for the staircase the modeler built -- which is not the same thing as
    exact for the shape the user typed; see :func:`slice_structure`.
    """
    Px, Py = float(period[0]), float(period[1])
    Cx = _signed_coverage(*_segments(x_list), xs, Px)
    Cy = _signed_coverage(*_segments(y_list), ys, Py)
    return Cy.T @ _np_complex(cell) @ Cx


# =========================================================== path 2: analytic shape sampling

def _shape_inside(shape, X, Y, basis, Px, Py):
    """Signed cover count of one shape and its periodic images at every sample point."""
    kind, cx, cy, lx, ly, angle = _analytic.shape_geometry(shape, Px, Py)
    cx, cy, lx, ly, angle = (float(np.real(np.asarray(v, dtype=complex)))
                             for v in (cx, cy, lx, ly, angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    count = np.zeros(X.shape, dtype=int)
    for m in _IMAGES:
        for k in _IMAGES:
            ox = cx + m * basis[0][0] + k * basis[0][1]
            oy = cy + m * basis[1][0] + k * basis[1][1]
            dx, dy = X - ox, Y - oy
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            if kind in ('circle', 'disk', 'ellipse'):
                inside = (u / (lx / 2.0)) ** 2 + (v / (ly / 2.0)) ** 2 <= 1.0
            elif kind in ('rectangle', 'square'):
                inside = (np.abs(u) <= lx / 2.0) & (np.abs(v) <= ly / 2.0)
            else:
                raise NotImplementedError(
                    f"plot_structure cannot draw a {kind!r} yet. Supported: rectangle, "
                    f"square, circle, disk, ellipse.")
            count = count + inside
    return count


def _render_analytic_add(layer, X, Y, basis, Px, Py):
    """The ``combine='add'`` structure: ``eps_base + sum_s (eps_s - eps_base) * indicator_s``.

    Read through :func:`~wavebit._core.analytic.shape_geometry` rather than off the shape dict,
    which is the whole reason that function exists -- a circle takes its diameter once, ``ly``
    and the centres have defaults, and a second reader is a second chance to disagree with the
    transform about what the dict meant.
    """
    eps_b = _eps_of(layer.get('background', 1.0))
    out = np.full(X.shape, eps_b, dtype=complex)
    for shape in layer.get('shapes', []):
        eps_s = _eps_of(shape['material'])
        out = out + (eps_s - eps_b) * _shape_inside(shape, X, Y, basis, Px, Py)
    return out


# ================================================ path 2: analytic shapes, combine='paint'

def _resolved_regions(shapes, period):
    """``[(eps, [vertices, ...]), ...]`` -- the disjoint loops the boundary path holds.

    Painting order resolved once by :func:`~wavebit._core.boundary.plan_layer`, exactly as the
    solver resolves it. **The orientation is the value**: a loop comes back counter-clockwise
    for material and *clockwise* for a hole, so a circle painted inside a rectangle leaves the
    rectangle a two-loop region whose hole has ``signed_area`` **negative**. Everything here
    reads that directly -- a plain sum over the loops, no sign to carry alongside and none to
    forget. Ignoring the orientation would put ``<eps>`` at 2.4256 where the geometry is 2.2371.
    """
    recs, regions = _boundary.plan_layer(shapes, period)
    verts = [_boundary.shape_vertices(xnp, _boundary._concrete_shape(s), period)
             for s in shapes]
    vals = _clipper.eval_table(xnp, recs, verts)
    out = []
    for shape, region in zip(shapes, regions):
        eps_s = _eps_of(shape['material'])
        loops = [np.asarray(_clipper.loop_verts(xnp, lp, vals), dtype=float)
                 for lp in region]
        out.append((eps_s, loops))
    return out


def _winding(V, xs, ys):
    """Winding number of the closed polygon ``V`` at every point of the grid ``(ys, xs)``.

    **Signed, and that is the point.** Upward crossings count ``+1`` and downward ``-1``, so a
    counter-clockwise loop gives ``+1`` inside and a clockwise one gives ``-1`` -- which is
    exactly how :mod:`~wavebit._core.clipper` states a hole. The renderer therefore sums its
    loops with no sign of its own, the same way the polygon transform does.

    Crossing count along each row, accumulated as a difference array and summed once -- so the
    cost is ``O(edges * ny + ny * nx)`` rather than a point-in-polygon per sample, which
    matters when the ellipse arriving here is a 256-gon.
    """
    x0, y0 = V[:, 0], V[:, 1]
    if y0.max() < ys[0] or y0.min() > ys[-1]:
        # The image ring puts eight of every nine copies outside the cell, and a copy that
        # spans no sample row crosses nothing. Skipping on the y extent alone is enough: the
        # ray runs to +x, so a copy sitting entirely to one side in x already sums to zero.
        return np.zeros((ys.size, xs.size), dtype=int)
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    Y = ys[None, :]
    up = (y0[:, None] <= Y) & (y1[:, None] > Y)
    down = (y1[:, None] <= Y) & (y0[:, None] > Y)
    hit = up | down
    if not hit.any():
        return np.zeros((ys.size, xs.size), dtype=int)
    e_idx, r_idx = np.nonzero(hit)
    dy = (y1 - y0)[e_idx]
    t = (ys[r_idx] - y0[e_idx]) / dy
    x_cross = x0[e_idx] + t * (x1 - x0)[e_idx]
    w = np.where(up[e_idx, r_idx], 1, -1)

    # Columns strictly left of the crossing get `w`; express that as a difference array so
    # every (edge, row) pair is two scatter-adds instead of a row-wide comparison.
    diff = np.zeros((ys.size, xs.size + 1), dtype=int)
    np.add.at(diff, (r_idx, np.zeros_like(r_idx)), w)
    np.add.at(diff, (r_idx, np.searchsorted(xs, x_cross, side='left')), -w)
    return np.cumsum(diff, axis=1)[:, :xs.size]


def _render_paint(layer, xs, ys, basis, period):
    """The painted structure, from the same disjoint loops the boundary path integrates."""
    eps_b = _eps_of(layer.get('background', 1.0))
    out = np.full((ys.size, xs.size), eps_b, dtype=complex)
    for eps_s, loops in _resolved_regions(layer.get('shapes', []), period):
        cover = np.zeros((ys.size, xs.size), dtype=int)
        for V in loops:
            for m in _IMAGES:
                for k in _IMAGES:
                    shift = np.array([m * basis[0][0] + k * basis[0][1],
                                      m * basis[1][0] + k * basis[1][1]])
                    cover = cover + _winding(V + shift, xs, ys)
        out = out + (eps_s - eps_b) * cover
    return out


# ============================================================ band-limited reconstruction

def _grid_from_conv_matrix(M, max_order_x, max_order_y):
    """The ``(4*max_order_y+1, 4*max_order_x+1)`` coefficient grid a convolution matrix came
    from.

    ``M[(r,s),(c,d)] = coeff(ky = r-c, kx = s-d)`` (see
    :func:`~wavebit._core.analytic.conv_matrix_2d`), and ``r``, ``c`` each run over
    ``2*max_order_y+1`` values, so every difference in ``[-2*max_order_y, 2*max_order_y]`` is
    reachable. Reading the grid back out of the builder's own output beats re-deriving it here:
    there is then no second opinion about the order-index sign, which is the convention this
    codebase has had to re-pin more than once.
    """
    n_orders_x, n_orders_y = 2 * max_order_x + 1, 2 * max_order_y + 1
    ky = np.arange(-2 * max_order_y, 2 * max_order_y + 1)[:, None]
    kx = np.arange(-2 * max_order_x, 2 * max_order_x + 1)[None, :]
    r, c = np.maximum(ky, 0), np.maximum(ky, 0) - ky
    s, d = np.maximum(kx, 0), np.maximum(kx, 0) - kx
    return np.asarray(M)[(r * n_orders_x + s), (c * n_orders_x + d)]


def _inverse_series(coeffs, max_order_x, max_order_y, nx, ny):
    """Evaluate a coefficient grid on ``ny x nx`` fractional samples.

    ``eps(u, v) = sum c[ky, kx] * exp(+2i*pi*(kx*u + ky*v))``, sampled at cell centres. The
    ``+`` is the order-index convention the grids are built in; a sign slip here mirrors the
    picture, which ``test_band_limited_converges_on_the_geometry`` would catch (a mirrored
    reconstruction does not converge on an asymmetric layer).
    """
    u = (np.arange(nx) + 0.5) / nx
    v = (np.arange(ny) + 0.5) / ny
    my = np.arange(-2 * max_order_y, 2 * max_order_y + 1)
    mx = np.arange(-2 * max_order_x, 2 * max_order_x + 1)
    Ey = np.exp(2j * np.pi * np.outer(v, my))
    Ex = np.exp(2j * np.pi * np.outer(u, mx))
    return Ey @ np.asarray(coeffs) @ Ex.T


def _band_limited(solver, layer, n):
    """The layer's own eps coefficients, evaluated on an ``n x n`` fractional grid."""
    max_order_x, max_order_y = int(solver.max_order[0]), int(solver.max_order[1])
    mt = solver.modeling_type_assigned
    period = _periods(solver)

    if mt == 0:
        eps = _np_complex(solver.structure[layer])
        grid = _grid_from_conv_matrix(
            _fac.fourier2d_piecewise_constant(xnp, eps, max_order_x, max_order_y, inverse_rule=None),
            max_order_x, max_order_y)
    elif mt == 1:
        cell, x_list, y_list = solver.modeling_block_instruction(solver.structure)[layer]
        grid = _grid_from_conv_matrix(
            _fac.fourier2d_integrated(xnp, _np_complex(cell), _np_complex(x_list),
                                      _np_complex(y_list), max_order_x, max_order_y, inverse_rule=None),
            max_order_x, max_order_y)
    else:
        lay = solver.structure[layer]
        # The solver's own choice of basis, not `solver.lattice`: an axis-aligned cell gets
        # `diag(period)` here rather than `None`, and taking it from the same accessor is what
        # keeps this reconstruction the transform the solve holds instead of a lookalike.
        basis = solver._geometry_lattice_for(None)
        if _analytic.layer_needs_resolving(lay, period, basis):
            KY, KX = _boundary._k_grids(xnp, max_order_x, max_order_y, np.float64)
            eps_b = _eps_of(lay.get('background', 1.0))
            grid = np.zeros(KY.shape, dtype=complex)
            grid[2 * max_order_y, 2 * max_order_x] = eps_b
            for eps_s, loops in _resolved_regions(lay.get('shapes', []), period):
                # Plain sum: `polygon_ft` is a signed contour integral and a hole arrives
                # wound clockwise, so it subtracts itself.
                ft = sum(_boundary.polygon_ft(xnp, V, KY, KX, period, basis) for V in loops)
                grid = grid + (eps_s - eps_b) * ft
        else:
            # `combine=` is passed, not assumed: `assemble_eps_coeffs` builds the summed rule
            # and refuses an overlapping painted layer rather than answering for it. The branch
            # above means the refusal never fires from here -- which is the point of stating it.
            grid = np.asarray(_analytic.assemble_eps_coeffs(
                xnp, lay.get('shapes', []), lay.get('background', 1.0), max_order_x, max_order_y, period,
                np.complex128, lattice=basis, combine=lay.get('combine'))[0])
    return _inverse_series(grid, max_order_x, max_order_y, n, n)


# ==================================================================== the two public entries

def _refuse_layer(solver, layer):
    if solver.modeling_type_assigned is None or solver.structure is None:
        raise ValueError('this solver has no structure to draw.')
    n_layers = len(solver.structure)
    if not (-n_layers <= layer < n_layers):
        raise IndexError(
            f'layer {layer} is out of range for a {n_layers}-layer stack. `layer` indexes the '
            f'way `structure` does -- 0 is the layer light meets first '
            f'(docs/conventions.md).')


def slice_structure(solver, layer=0, n=512, show='eps', band_limited=False):
    """The structure of one layer, as a NumPy array. **Geometry, not the truncated series.**

    ``layer`` indexes the way ``structure`` does: ``0`` is the layer light meets first
    (``docs/conventions.md``). Row 0 of the result is ``y = 0``, so it draws with
    ``origin='lower'`` -- the same way up ``plot_field``'s top view ends at.

    Returns complex values -- ``show='eps'`` for the relative permittivity the solve works in,
    ``show='n'`` for the refractive index derived from it. An absorbing layer has a non-zero
    imaginary part and :func:`plot_structure` draws the real part and says so.

    What ``n`` means depends on the path, and the difference is deliberate:

    * **raster** (``modeling_type=0``) -- ``structure[layer]`` itself, at the raster's own
      resolution. ``n`` is ignored, because resampling a raster invents detail it does not
      carry.
    * **block** (``modeling_type=1``) -- ``n x n`` samples of the modeler's staircase, by
      signed segment accumulation (see the module docstring; the naive read is wrong on any
      shape that crosses the cell boundary).
    * **analytic** (``modeling_type=2``) -- ``n x n`` samples of the shapes themselves, through
      :func:`~wavebit._core.analytic.shape_geometry`, periodic images included. A painted layer
      whose shapes overlap is resolved into disjoint loops by
      :mod:`~wavebit._core.boundary` first, exactly as the solver resolves it: falling back to
      the summed rule there is a 5.4% lie in ``<eps>`` (1.9134 against 1.8158 for a 300x300
      rod of permittivity 4 under a 320-diameter disc of permittivity 9 in a 1000x1000
      cell).

    Three things this picture is *not*, each measured:

    1. **It is the geometry; the solver holds a truncated transform of it.** Pass
       ``band_limited=True`` to see that instead -- the layer's own coefficients, evaluated
       rather than sampled. A ``circle(lx=260, material=12.25)`` plus a rectangle rotated 0.45
       rad in a 1000x1000 cell comes back at ``max_order=9`` with ``eps`` in ``[-0.186,
       13.850]`` where the geometry is ``[1.0, 12.25]``: negative permittivity and 13%
       overshoot, pure Gibbs. Relative L2 against the sampled geometry is 28.4% / 16.8% / 10.2%
       at ``max_order`` 3 / 9 / 25, while ``<eps>`` agrees to 1e-4 throughout -- so an
       integrated quantity can be converged while the picture still rings.
    2. **On the block path this draws the blocks, not the shape.** That is what the solver has,
       and it is not what the user typed: a 240x180 rectangle at ``angle=0.4`` with
       ``n_split=16`` differs from the true rotated rectangle over 1.9% of the rod's area (7.7%
       at ``n_split=4``, 0.5% at 64). Adding blocks narrows that, but it is *harmonics* that
       remove the kink the staircase costs in the answer, not ``n_split`` --
       :meth:`~wavebit._core.modeler.Modeling.rectangle` measures both.
    3. **A curved shape on the painted path is a 256-gon** before anything is drawn or
       transformed (:data:`~wavebit._core.boundary.POLYGON_VERTICES`), while the summed path
       keeps the exact ellipse -- so one ellipse has two pictures. Comparing them by ``<eps>``
       will not show it: :func:`~wavebit._core.polygon.ellipse_verts` matches the polygon's
       *area* to the ellipse's rather than inscribing it, precisely so the DC coefficient does
       not move. What differs is the outline, which is what a picture is for.

    A shape wider than about one period has periodic images this draws only one ring of; and a
    ``kind`` neither path can draw raises rather than being skipped.
    """
    _refuse_layer(solver, layer)
    if show not in ('eps', 'n'):
        raise ValueError(f"show must be 'eps' or 'n', not {show!r}.")

    if band_limited:
        values = _band_limited(solver, layer, n)
    else:
        mt = solver.modeling_type_assigned
        period = _periods(solver)
        basis = _basis(solver)
        if mt == 0:
            values = _np_complex(solver.structure[layer])
        elif mt == 1:
            xs = (np.arange(n) + 0.5) * period[0] / n
            ys = (np.arange(n) + 0.5) * period[1] / n
            # Re-run rather than read `modeler.block_info_list`, which may be left over from
            # an earlier `structure`. This is the same call every solve makes, and it leaves
            # the same marks on the modeler (`grid_plans`, `shape_choices`) that a solve would -- a
            # frozen `StaticGeometry` is a copy and is not disturbed.
            cell, x_list, y_list = solver.modeling_block_instruction(solver.structure)[layer]
            values = _render_block(cell, x_list, y_list, period, xs, ys)
        elif mt == 2:
            lay = solver.structure[layer]
            if _analytic.layer_needs_resolving(lay, period,
                                               solver._geometry_lattice_for(None)):
                # Fractional samples mapped through the basis, so the loops -- which are
                # absolute Cartesian positions -- are tested where they actually are.
                if not solver.lattice_is_axis_aligned:
                    raise NotImplementedError(
                        'a painted, overlapping layer on a skewed cell is not drawn yet: the '
                        'loops are Cartesian polygons and the scanline fill runs on cell rows, '
                        'which stop being rows once the cell is sheared. Draw it with '
                        "combine='add' if the shapes do not overlap, or raise the structure "
                        'as a raster structure.')
                xs = (np.arange(n) + 0.5) * period[0] / n
                ys = (np.arange(n) + 0.5) * period[1] / n
                values = _render_paint(lay, xs, ys, basis, period)
            else:
                X, Y = sample_grid(xnp, (n, n), period, np.float64, solver.lattice)
                values = _render_analytic_add(lay, X, Y, basis, *period)

    return values if show == 'eps' else index_from_eps(_passive(values))


def plot_structure(solver, layer=0, n=512, show='eps', ax=None, band_limited=False):
    """Draw one layer of the structure. Arguments are :func:`slice_structure`'s; read it first.

    Returns the axes. An axis-aligned cell is drawn with ``imshow(origin='lower')`` and a
    skewed one with ``pcolormesh`` on ``u*a1 + v*a2``, as ``docs/oblique-lattice.md`` does for
    fields -- an index-space ``imshow`` of a skewed cell draws a parallelogram as a square.

    Complex values are drawn as their real part, and the title says so.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('plot_structure() needs matplotlib installed') from e

    values = slice_structure(solver, layer=layer, n=n, show=show, band_limited=band_limited)
    complex_valued = bool(np.max(np.abs(values.imag)) > 1e-12 * max(
        1.0, float(np.max(np.abs(values.real)))))
    image = values.real

    if ax is None:
        _, ax = plt.subplots()

    basis = _basis(solver)
    if solver.lattice_is_axis_aligned:
        Px, Py = _periods(solver)
        art = ax.imshow(image, origin='lower', extent=(0.0, Px, 0.0, Py), cmap='viridis')
    else:
        ny, nx = image.shape
        u = (np.arange(nx) + 0.5) / nx
        v = (np.arange(ny) + 0.5) / ny
        U, V = np.meshgrid(u, v)
        X = U * basis[0][0] + V * basis[0][1]
        Y = U * basis[1][0] + V * basis[1][1]
        art = ax.pcolormesh(X, Y, image, shading='nearest', cmap='viridis')

    what = {0: 'raster', 1: 'blocks', 2: 'shapes'}.get(solver.modeling_type_assigned, '?')
    title = f'layer {layer} ({what}), {show}'
    if band_limited:
        title += f' band-limited to max_order={tuple(int(f) for f in solver.max_order)}'
    if complex_valued:
        title += ' (real part)'
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.figure.colorbar(art, ax=ax)
    return ax
