"""Constant-z pictures of the *structure*, one recipe per modeling path.

:func:`~wavebit._core.fields.plot_field` draws what came out of a solve. This draws what went
in, which had no helper at all: every tutorial that shows a layer builds the array itself,
before Wavebit sees it, so the path where rendering is not trivial -- the parametric shapes --
was never pictured by anything the library ships.

Presentation rather than physics, and like ``plot_field`` there is no reason for it to exist
per backend: it only needs the geometry as NumPy. Everything here converts on the way in.

A shape recipe is drawn from the shapes themselves rather than from anything the modeller
built, which is what the picture being *the geometry* means -- see :func:`slice_structure`.
An earlier version of this module read the removed block modeller's own ``draw()`` edge list,
where the hazard was that the list is not guaranteed increasing: a 200x160 rod at ``cx = 960``
in a 1000x1000 cell came back as ``x = [860., 1060., 1000.]``, whose last segment has a
*negative* width. Painting those bands in order dropped up to 43.8% of the rod while the solver
integrated it correctly, because the integrated series (``_fourier1d_integrated`` then,
:func:`~wavebit._core.factorization._segment_series` since 2026-09-25) weights each segment by
its **signed** width. That path is gone and so is the reader for it.
"""
import array_api_compat.numpy as xnp
import numpy as np

from . import parametric as _parametric
from . import boundary as _boundary
from . import clipper as _clipper
from . import factorization as _fac
from .domain import as_material, index_from_eps
from .enums import ModelingPath
from .lattice import meshgrid_xy, reduced_basis, sample_grid

#: Periodic images drawn around the cell. A shape's transform is periodic by construction, so
#: the part that leaves one edge returns at the other and the picture has to show it. One ring
#: covers anything up to about a full period across; wider than that is not drawn, which is a
#: limit of the picture and not of the solve.
#: One cell a side, counted on the *reduced* basis (`lattice.reduced_basis`) as every image
#: enumeration is since 2026-09-06: on `a2 = 10 a1 + (50, 700)` the neighbour that overlaps the
#: picture sits at (-10, 1) of the basis as given, and a 3x3 there drew the cell with holes.
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
    :func:`~wavebit._core.parametric.shape_geometry`), and a material is the slot where the
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


# =========================================================== path 1: parametric shape sampling

def _shape_inside(shape, X, Y, basis, Px, Py):
    """Signed cover count of one shape and its periodic images at every sample point."""
    kind, cx, cy, lx, ly, angle = _parametric.shape_geometry(shape, basis)
    cx, cy, lx, ly, angle = (float(np.real(np.asarray(v, dtype=complex)))
                             for v in (cx, cy, lx, ly, angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    count = np.zeros(X.shape, dtype=int)
    images_on = np.asarray(reduced_basis(basis))       # the origin above is the basis as given
    for m in _IMAGES:
        for k in _IMAGES:
            ox = cx + m * images_on[0][0] + k * images_on[0][1]
            oy = cy + m * images_on[1][0] + k * images_on[1][1]
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


def _render_summed(layer, X, Y, basis, Px, Py):
    """The summed structure: ``eps_base + sum_s (eps_s - eps_base) * indicator_s``.

    Read through :func:`~wavebit._core.parametric.shape_geometry` rather than off the shape dict,
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


# ============================================ path 2: parametric shapes, painted where they overlap

def _resolved_regions(shapes, period, basis=None):
    """``[(eps, [vertices, ...]), ...]`` -- the disjoint loops the boundary route holds.

    Painting order resolved once by :func:`~wavebit._core.boundary.plan_layer`, exactly as the
    solver resolves it. **The orientation is the value**: a loop comes back counter-clockwise
    for material and *clockwise* for a hole, so a circle painted inside a rectangle leaves the
    rectangle a two-loop region whose hole has ``signed_area`` **negative**. Everything here
    reads that directly -- a plain sum over the loops, no sign to carry alongside and none to
    forget. Ignoring the orientation would put ``<eps>`` at 2.4256 where the geometry is 2.2371.
    """
    recs, regions, _, _ = _boundary.plan_layer(shapes, basis)
    verts = [_boundary.shape_vertices(xnp, _boundary._concrete_shape(s), basis)
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
    cost is ``O(edges * n_v + n_v * n_u)`` rather than a point-in-polygon per sample, which
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
    """The painted structure, from the same disjoint loops the boundary route integrates."""
    eps_b = _eps_of(layer.get('background', 1.0))
    out = np.full((ys.size, xs.size), eps_b, dtype=complex)
    images_on = np.asarray(reduced_basis(basis))
    for eps_s, loops in _resolved_regions(layer.get('shapes', []), period, basis):
        cover = np.zeros((ys.size, xs.size), dtype=int)
        for V in loops:
            for m in _IMAGES:
                for k in _IMAGES:
                    shift = np.array([m * images_on[0][0] + k * images_on[0][1],
                                      m * images_on[1][0] + k * images_on[1][1]])
                    cover = cover + _winding(V + shift, xs, ys)
        out = out + (eps_s - eps_b) * cover
    return out


# ============================================================ band-limited reconstruction

def _grid_from_conv_matrix(M, half_width_u, half_width_v):
    """The ``(4*half_width_v+1, 4*half_width_u+1)`` coefficient grid a convolution matrix came
    from.

    ``M[(r,s),(c,d)] = coeff(ky = r-c, kx = s-d)`` (see
    :func:`~wavebit._core.parametric.conv_matrix_2d`), and ``r``, ``c`` each run over
    ``2*half_width_v+1`` values, so every difference in ``[-2*half_width_v, 2*half_width_v]`` is
    reachable. Reading the grid back out of the builder's own output beats re-deriving it here:
    there is then no second opinion about the order-index sign, which is the convention this
    codebase has had to re-pin more than once.
    """
    n_orders_u, n_orders_v = 2 * half_width_u + 1, 2 * half_width_v + 1
    ky = np.arange(-2 * half_width_v, 2 * half_width_v + 1)[:, None]
    kx = np.arange(-2 * half_width_u, 2 * half_width_u + 1)[None, :]
    r, c = np.maximum(ky, 0), np.maximum(ky, 0) - ky
    s, d = np.maximum(kx, 0), np.maximum(kx, 0) - kx
    return np.asarray(M)[(r * n_orders_u + s), (c * n_orders_u + d)]


def _inverse_series(coeffs, half_width_u, half_width_v, n_u, n_v):
    """Evaluate a coefficient grid on ``n_v x n_u`` fractional samples.

    ``eps(u, v) = sum c[ky, kx] * exp(+2i*pi*(kx*u + ky*v))``, sampled at cell centres. The
    ``+`` is the order-index convention the grids are built in; a sign slip here mirrors the
    picture, which ``test_band_limited_converges_on_the_geometry`` would catch (a mirrored
    reconstruction does not converge on an asymmetric layer).
    """
    u = (np.arange(n_u) + 0.5) / n_u
    v = (np.arange(n_v) + 0.5) / n_v
    my = np.arange(-2 * half_width_v, 2 * half_width_v + 1)
    mx = np.arange(-2 * half_width_u, 2 * half_width_u + 1)
    Ey = np.exp(2j * np.pi * np.outer(v, my))
    Ex = np.exp(2j * np.pi * np.outer(u, mx))
    return Ey @ np.asarray(coeffs) @ Ex.T


def _band_limited(solver, layer, n):
    """The layer's own eps coefficients, evaluated on an ``n x n`` fractional grid."""
    half_width_u, half_width_v = int(solver.truncation_order[0]), int(solver.truncation_order[1])
    mt = solver.modeling_path
    period = _periods(solver)

    if mt == ModelingPath.RASTER:
        eps = _np_complex(solver.structure[layer])
        grid = _grid_from_conv_matrix(
            _fac.fourier2d_piecewise_constant(xnp, eps, half_width_u, half_width_v, inverse_rule=None),
            half_width_u, half_width_v)
    else:
        lay = solver.structure[layer]
        # The solver's own choice of basis, not `solver.lattice`: an axis-aligned cell gets
        # `diag(period)` here rather than `None`, and taking it from the same accessor is what
        # keeps this reconstruction the transform the solve holds instead of a lookalike.
        basis = solver._geometry_lattice_for(None)
        if _parametric.layer_needs_resolving(lay, basis):
            KY, KX = _boundary._k_grids(xnp, half_width_u, half_width_v, np.float64)
            eps_b = _eps_of(lay.get('background', 1.0))
            grid = np.zeros(KY.shape, dtype=complex)
            grid[2 * half_width_v, 2 * half_width_u] = eps_b
            for eps_s, loops in _resolved_regions(lay.get('shapes', []), period, basis):
                # Plain sum: `polygon_ft` is a signed contour integral and a hole arrives
                # wound clockwise, so it subtracts itself.
                ft = sum(_boundary.polygon_ft(xnp, V, KY, KX, basis) for V in loops)
                grid = grid + (eps_s - eps_b) * ft
        else:
            # Disjoint shapes, by the branch above: `assemble_eps_coeffs` sums, and where
            # nothing overlaps the sum and the painting are the same structure. It refuses an
            # overlap unconditionally, so a branch that stopped being right would raise here
            # rather than draw a picture the solver does not build.
            grid = np.asarray(_parametric.assemble_eps_coeffs(
                xnp, lay.get('shapes', []), lay.get('background', 1.0), half_width_u,
                half_width_v, basis, np.complex128)[0])
    return _inverse_series(grid, half_width_u, half_width_v, n, n)


# ==================================================================== the two public entries

def _refuse_layer(solver, layer):
    if solver.modeling_path is None or solver.structure is None:
        raise ValueError('this solver has no structure to draw.')
    n_layers = len(solver.structure)
    if not (-n_layers <= layer < n_layers):
        raise IndexError(
            f'layer {layer} is out of range for a {n_layers}-layer stack. `layer` indexes the '
            f'way `structure` does -- 0 is the layer nearest `bot`, whichever side the light '
            f'arrives from (docs/conventions.md).')
    if solver._has_tensor_materials():
        # The tensor sampler validates its own primitives and supports periodic
        # painting, including overlaps the integrated scalar builder refuses.
        return
    if solver.modeling_path == ModelingPath.PARAMETRIC:
        # The same refusals the solve makes, so a picture cannot be drawn of a structure the
        # solver will not build. Only the layer asked for: drawing layer 0 should not be
        # stopped by layer 3, and a renderer is where someone goes to *find out* what a
        # refusal is about.
        index = layer % n_layers
        _parametric.check_layer_spec(solver.structure[index], index)
        _parametric.check_layer_shapes(solver.structure[index], _basis(solver), index)

        # A picture is a picture: every renderer below samples on a NumPy grid, so a traced
        # shape parameter has nowhere to go. Said here rather than left to surface as a
        # `TracerArrayConversionError` out of `numpy.stack`, naming nothing the caller wrote.
        # It became reachable on 2026-08-28: `layer_needs_resolving` used to raise for any
        # traced multi-shape layer and incidentally covered this, and now that it answers
        # under a gradient the renderer has to say so itself.
        for shape in solver.structure[index].get('shapes', []):
            for key, value in shape.items():
                if key in ('type', 'material'):
                    continue
                try:
                    # `np.asarray`, not `checked_float`. The renderer needs the value as an
                    # *array* to sample against a NumPy grid, and that is a strictly stronger
                    # ask: `checked_float` reaches a `jax.grad` tracer's primal one scalar at a
                    # time, which is why the geometry checks can run under a gradient, while a
                    # whole-array conversion is refused under any JAX trace. `detach` first so
                    # a torch leaf under autograd -- which can be drawn -- still is.
                    probe = value.detach() if hasattr(value, 'detach') else value
                    np.asarray(probe, dtype=float)
                except Exception:                  # noqa: BLE001 - a tracer of some kind
                    raise ValueError(
                        f'a traced structure cannot be drawn: layer {index} has a {key!r} that '
                        f'carries no number this can sample. `slice_structure` and '
                        f'`plot_structure` render the geometry on a grid, which is not '
                        f'something a jax.jit or vmap trace can be asked for.\n'
                        f'  Draw it outside the trace, at the values you want to look at.'
                    ) from None


def slice_structure(solver, layer=0, n=512, show='eps', band_limited=False, *, component=None):
    """The structure of one layer, as a NumPy array. **Geometry, not the truncated series.**

    ``layer`` indexes the way ``structure`` does: ``0`` is the layer nearest ``bot``,
    whichever side the light arrives from (``docs/conventions.md``). Row 0 of the result is
    the negative-y edge, so it draws with ``origin='lower'`` -- the same way up ``plot_field``'s top
    view ends at.

    Returns complex values -- ``show='eps'`` for the relative permittivity the solve works in,
    ``show='n'`` for the refractive index derived from it. An absorbing layer has a non-zero
    imaginary part and :func:`plot_structure` draws the real part and says so.

    Tensor scenes require ``component='xx'`` (or another xyz component pair).
    Only physical epsilon components are shown, with ``show='eps'`` and
    ``band_limited=False``; there is no single scalar index for a tensor.

    What ``n`` means depends on the path, and the difference is deliberate:

    * **raster** (``modeling_path='raster'``) -- ``structure[layer]`` itself, at the raster's own
      resolution. ``n`` is ignored, because resampling a raster invents detail it does not
      carry.
    * **parametric** (``modeling_path='parametric'``) -- ``n x n`` samples of the shapes themselves, through
      :func:`~wavebit._core.parametric.shape_geometry`, periodic images included. A painted layer
      whose shapes overlap is resolved into disjoint loops by
      :mod:`~wavebit._core.boundary` first, exactly as the solver resolves it: falling back to
      the summed rule there is a 5.4% lie in ``<eps>`` (1.9134 against 1.8158 for a 300x300
      rod of permittivity 4 under a 320-diameter disc of permittivity 9 in a 1000x1000
      cell).

    Two things this picture is *not*, each measured:

    1. **It is the geometry; the solver holds a truncated transform of it.** Pass
       ``band_limited=True`` to see that instead -- the layer's own coefficients, evaluated
       rather than sampled. A ``circle(lx=260, material=12.25)`` plus a rectangle rotated 0.45
       rad in a 1000x1000 cell comes back at ``truncation_order=9`` with ``eps`` in ``[-0.186,
       13.850]`` where the geometry is ``[1.0, 12.25]``: negative permittivity and 13%
       overshoot, pure Gibbs. Relative L2 against the sampled geometry is 28.4% / 16.8% / 10.2%
       at ``truncation_order`` 3 / 9 / 25, while ``<eps>`` agrees to 1e-4 throughout -- so an
       integrated quantity can be converged while the picture still rings.
    2. **A curved shape on the painted path is a 256-gon** before anything is drawn or
       transformed (:data:`~wavebit._core.boundary.POLYGON_VERTICES`), while the summed path
       keeps the exact ellipse -- so one ellipse has two pictures. Comparing them by ``<eps>``
       will not show it: :func:`~wavebit._core.polygon.ellipse_verts` matches the polygon's
       *area* to the ellipse's rather than inscribing it, precisely so the DC coefficient does
       not move. What differs is the outline, which is what a picture is for.

    A shape wider than about one period has periodic images this draws only one ring of; and a
    ``kind`` neither path can draw raises rather than being skipped.
    """
    _refuse_layer(solver, layer)
    if solver._has_tensor_materials():
        from . import anisotropy
        if component not in ('xx','xy','xz','yx','yy','yz','zx','zy','zz'):
            raise ValueError("tensor structure pictures require component='xx', 'xy', ..., or 'zz'")
        if show != 'eps' or band_limited:
            raise NotImplementedError('tensor pictures show individual permittivity components, '
                                      "with show='eps' and band_limited=False")
        shaped = isinstance(solver.structure, list)
        snapshot = anisotropy._shape_snapshot(solver, n=n) if shaped and any(
            l.get('shapes') for l in solver.structure) else None
        t = anisotropy._field(solver, solver.structure, 'epsilon', snapshot)[layer]
        values = _np_complex(t)[..., 'xyz'.index(component[0]), 'xyz'.index(component[1])]
        return np.broadcast_to(values, (n,n)).copy() if shaped and values.shape == (1,1) else values
    if component is not None:
        raise ValueError('component is only used for tensor structure pictures')
    if show not in ('eps', 'n'):
        raise ValueError(f"show must be 'eps' or 'n', not {show!r}.")

    if band_limited:
        values = _band_limited(solver, layer, n)
    else:
        mt = solver.modeling_path
        period = _periods(solver)
        basis = _basis(solver)
        if mt == ModelingPath.RASTER:
            values = _np_complex(solver.structure[layer])
        elif mt == ModelingPath.PARAMETRIC:
            lay = solver.structure[layer]
            if _parametric.layer_needs_resolving(lay, solver._geometry_lattice_for(None)):
                # Fractional samples mapped through the basis, so the loops -- which are
                # absolute Cartesian positions -- are tested where they actually are.
                if not solver.lattice_is_axis_aligned:
                    raise NotImplementedError(
                        'a painted, overlapping layer on a skewed cell is not drawn yet: the '
                        'loops are Cartesian polygons and the scanline fill runs on cell rows, '
                        'which stop being rows once the cell is sheared. State the layer '
                        'as a raster structure instead.')
                xs = (np.arange(n) + 0.5) * period[0] / n
                ys = (np.arange(n) + 0.5) * period[1] / n
                values = _render_paint(lay, xs, ys, basis, period)
            else:
                X, Y = sample_grid(xnp, (n, n), solver.lattice, np.float64)
                values = _render_summed(lay, X, Y, basis, *period)

    return values if show == 'eps' else index_from_eps(_passive(values))


def plot_structure(solver, layer=0, n=512, show='eps', ax=None, band_limited=False, *, component=None):
    """Draw one layer of the structure. Arguments are :func:`slice_structure`'s; read it first.

    Returns the axes. An axis-aligned cell is drawn with ``imshow(origin='lower')`` and a
    skewed one with ``pcolormesh`` on ``(u - 0.5)*a1 + (v - 0.5)*a2``, as
    ``docs/oblique-lattice.md`` does for fields -- an index-space ``imshow`` of a skewed cell
    draws a parallelogram as a square. The axes run from ``-P/2`` to ``+P/2``: the cell is
    centred at the origin, the same frame a shape's ``cx`` and :func:`wavebit.meshgrid_xy` use.

    Complex values are drawn as their real part, and the title says so.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('plot_structure() needs matplotlib installed') from e

    values = slice_structure(solver, layer=layer, n=n, show=show, band_limited=band_limited,
                             component=component)
    complex_valued = bool(np.max(np.abs(values.imag)) > 1e-12 * max(
        1.0, float(np.max(np.abs(values.real)))))
    image = values.real

    if ax is None:
        _, ax = plt.subplots()

    # Both branches label the axes in the frame the caller states geometry in: the cell is
    # centred at the origin, so a centred feature sits at 0 in the picture as it does in the
    # recipe. The image itself is untouched -- only the coordinates written on it moved.
    basis = _basis(solver)
    if solver.lattice_is_axis_aligned:
        # Off the basis, not off `period`: this branch has proved the cell is a rectangle, so
        # the two agree -- but `periods_of` is `(|a1|, |a2|)` with the angle already gone, and
        # `test_cell_shape_gate` is right that arithmetic on that pair should not spread. The
        # basis is in hand and the `else` below already reads it.
        half_x, half_y = 0.5 * basis[0][0], 0.5 * basis[1][1]
        art = ax.imshow(image, origin='lower', cmap='viridis',
                        extent=(-half_x, half_x, -half_y, half_y))
    else:
        # `meshgrid_xy` is exactly these five lines, and it is the function whose frame
        # this branch has to agree with -- spelling them out again is how the two drift.
        X, Y = meshgrid_xy(image.shape, lattice=basis)
        art = ax.pcolormesh(X, Y, image, shading='nearest', cmap='viridis')

    what = solver.modeling_path.value if solver.modeling_path is not None else '?'
    title = f'layer {layer} ({what}), {show}'
    if component is not None:
        title += f'_{component}'
    if band_limited:
        title += f' band-limited to truncation_order={tuple(int(f) for f in solver.truncation_order)}'
    if complex_valued:
        title += ' (real part)'
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.figure.colorbar(art, ax=ax)
    return ax
