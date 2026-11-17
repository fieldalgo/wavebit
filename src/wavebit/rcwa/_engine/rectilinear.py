"""Exact Cartesian Li integration of axis-aligned rectangular paintings.

The grid follows live rectangle edges. Plans contain only edge order, cell
ownership and the whole cells each rectangle is written away from the cell (a
rectangle written ``m`` periods over is the same structure, and the edges are
read in the cell); materials, lengths and positions are replayed on the active backend.
No normal field or pixel raster is constructed. The rules are nested as
``nesting`` says (``enums.Nesting``, ``'fixed'`` by default), by the same
integrated builder the raster path uses, so a plan serves both nestings.
"""
from collections import namedtuple
import math

from wavebit._common.backend import as_array
from wavebit._common.backend import checked_float
from wavebit.materials.response import as_material
from wavebit.rcwa._engine.factorization import ConvMatrices
from wavebit.rcwa._engine.factorization import StaleGeometryError
from wavebit.rcwa._engine.factorization import fourier2d_integrated
from wavebit.rcwa._engine.factorization import _conv_matrix
from wavebit.rcwa._engine.factorization import _li_slot
from wavebit.lattice import is_axis_aligned
import wavebit.rcwa._engine.parametric as P

Plan = namedtuple('RectilinearPlan', 'x y owners count folds')


def validate(layers, lattice):
    if not is_axis_aligned(lattice):
        raise NotImplementedError("factorization='li' on shapes requires a rectangular, axis-aligned cell")
    for index, layer in enumerate(layers):
        P.check_layer_spec(layer, index)
        for shape in layer.get('shapes', []):
            P.check_shape_spec(shape)
            try:
                angle = P._static_float(shape.get('angle', 0.))
            except (TypeError, ValueError):
                raise NotImplementedError("factorization='li' requires a fixed zero angle; omit angle in traced shapes") from None
            if shape.get('type', 'rectangle') != 'rectangle' or angle != 0.:
                raise NotImplementedError("factorization='li' on shapes supports only axis-aligned rectangles; use factorization='nvf' (NVF) for other shapes")


def _coordinate(shapes, axis, node, period, folds):
    owner, side = node
    if owner == -1:
        return side * period
    shape = shapes[owner]
    fold = folds[owner]['xy'.index(axis)]
    return period/2 + shape.get('c'+axis, 0.) - fold*period + side*shape['l'+axis]/2


def _primal_axes(shapes, lattice):
    """Each axis's edge positions in the cell, and each rectangle's whole-cell fold.

    A centre written ``m`` periods from the cell (``cx`` and ``cx + m Px`` are one structure)
    is read ``m`` periods back, to the nearest cell; the fold is an integer of the plan, so a
    replay reads the same cell and a rectangle moved by whole periods needs a new plan.
    """
    periods = [checked_float(lattice[i][i]) for i in (0, 1)]
    if any(not math.isfinite(v) or v <= 0 for v in periods):
        raise ValueError('Cartesian shape Li requires positive finite cell periods')
    result, folds = [], [[0, 0] for _ in shapes]
    for a, (axis, period) in enumerate(zip(('x', 'y'), periods)):
        values = {(-1, 0): 0., (-1, 1): period}
        for i, shape in enumerate(shapes):
            length = checked_float(shape.get('l'+axis, 0.))
            center = checked_float(shape.get('c'+axis, 0.))
            if not math.isfinite(length) or not math.isfinite(center) or length <= 0:
                raise ValueError('Cartesian shape Li requires positive finite sizes and finite centers')
            fold = round(center / period)
            low, high = period/2+center-fold*period-length/2, period/2+center-fold*period+length/2
            if low < 0 or high > period:
                raise ValueError('Cartesian shape Li currently requires rectangles inside one cell, including its edges: '
                                 'a rectangle that crosses the cell edge, wherever it is written, is refused')
            values[i, -1], values[i, 1] = low, high
            folds[i][a] = fold
        result.append(values)
    return result, tuple(tuple(f) for f in folds)


def plan_layer(shapes, lattice):
    try:
        values, folds = _primal_axes(shapes, lattice)
    except TypeError as error:
        # Only a traced topology cannot be sorted; invalid concrete values keep
        # the diagnosis from _primal_axes.
        raise ValueError('freeze_geometry() on concrete Cartesian Li shapes before jit') from error
    axes = []
    for coords in values:
        def key(node):
            i, side = node
            priority = (0 if side == 0 else 3) if i == -1 else (1 if side == 1 else 2)
            return coords[node], priority, i
        axes.append(tuple(sorted(coords, key=key)))
    x, y = axes
    owners = []
    # Symbolic containment retains zero-width intervals at contact. The upper
    # edge precedes the next lower edge, so an opening gap keeps background.
    bounds = [(x.index((i, -1)), x.index((i, 1)), y.index((i, -1)), y.index((i, 1)))
              for i in range(len(shapes))]
    for j in range(len(y)-1):
        row = []
        for k in range(len(x)-1):
            owner = -1
            for i, (x0, x1, y0, y1) in enumerate(bounds):
                if x0 <= k < x1 and y0 <= j < y1:
                    owner = i  # later shapes paint over earlier ones
            row.append(owner)
        owners.append(tuple(row))
    return Plan(x, y, tuple(owners), len(shapes), folds)


def check_plan(shapes, lattice, plan):
    if len(shapes) != plan.count:
        raise StaleGeometryError('Cartesian Li shape count changed; freeze_geometry() again')
    try:
        values, folds = _primal_axes(shapes, lattice)
    except TypeError:
        return  # Frozen replay: ordering is checked on concrete calls, as for other plans.
    if folds != plan.folds:
        raise StaleGeometryError('a Cartesian Li rectangle moved by whole cells; freeze_geometry() again')
    for nodes, coords in zip((plan.x, plan.y), values):
        positions = [coords[n] for n in nodes]
        if any(a > b for a, b in zip(positions, positions[1:])):
            raise StaleGeometryError('Cartesian Li edge order changed; freeze_geometry() again')


def layer_partition(xp, shapes, lattice, dtype, plan=None, device=None):
    """Replay live Cartesian edges and return their material-owner table."""
    if plan is None:
        plan = plan_layer(shapes, lattice)
    else:
        check_plan(shapes, lattice, plan)
    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32
    edges = []
    for axis, nodes, period in zip(('x', 'y'), (plan.x, plan.y), (lattice[0][0], lattice[1][1])):
        edges.append(xp.stack([as_array(xp, _coordinate(shapes, axis, n, period, plan.folds)/period,
                                        dtype=real_dtype, device=device) for n in nodes[1:]]).reshape((-1, 1)))
    return edges, plan.owners


def _table_series(xp, table, dtype):
    """``series(a)`` of a weighted coefficient table (:meth:`stretch.Axis.segment_table`).

    ``a[..., j]`` holds segment ``j``'s value; the result is ``[..., k]`` like
    :func:`~wavebit.rcwa._engine.factorization._segment_series`, with the stretch's slope as the
    weight of every segment's integral.
    """
    table = xp.matrix_transpose(xp.asarray(table, dtype=dtype))            # [j, k]

    def series(a):
        return xp.astype(a, dtype) @ table

    return series


def wall_positions(layers, lattice, plans=None):
    """Every rectangle edge of a stack, per axis, in cell coordinates (the corner at 0).

    A rectangle as long as the period along an axis has no wall across it: its two edges are
    one line of the lattice with the same material on both sides, and they are left out.
    """
    walls = ([], [])
    for index, layer in enumerate(layers):
        shapes = layer.get('shapes', [])
        plan = plan_layer(shapes, lattice) if plans is None else plans[index]
        for a, (axis, nodes) in enumerate(zip(('x', 'y'), (plan.x, plan.y))):
            period = checked_float(lattice[a][a])
            spans = {i for i, shape in enumerate(shapes)
                     if checked_float(shape['l' + axis]) >= period * (1 - 1e-12)}
            walls[a].extend(checked_float(_coordinate(shapes, axis, n, period, plan.folds))
                            for n in nodes if n[0] != -1 and n[0] not in spans)
    return walls


def stretched_layer_matrices(xp, shapes, background, hu, hv, lattice, dtype, stretch, plan=None,
                             nesting=None):
    """:func:`layer_matrices` in the stretched coordinates: ``eps~ = eps diag(g/f, f/g, f g)``.

    The layer is still constant on the rectangles its edges cut, now of the computational
    cell, and every slot is that constant times the slope along ``u`` and the slope along
    ``v`` (``'u'``: ``[[eps g/f]]`` with the inverse rule along ``u`` -- the reciprocal the rule
    takes turns ``1/f`` into ``f``; ``'v'`` alike; the direct rule holds ``eps f g``). So Li's
    nested operators are the unstretched ones with each segment's Fourier integral weighted by
    the slope, which is all :func:`_table_series` changes. A uniform layer is one segment per
    axis and comes out as the stretched uniform medium.
    """
    import numpy as _np
    edges, owners = layer_partition(xp, shapes, lattice, dtype, plan)
    materials = [as_array(xp, as_material(s.get('material', 1.)).epsilon, dtype=dtype) for s in shapes]
    materials.append(as_array(xp, as_material(background).epsilon, dtype=dtype))
    cell = xp.stack([xp.stack([materials[i] for i in row]) for row in owners])
    host = [_np.asarray(_to_host(e), dtype=float).reshape(-1) for e in edges]
    series_u = _table_series(xp, stretch.x.segment_table(host[0], hu), dtype)
    series_v = _table_series(xp, stretch.y.segment_table(host[1], hv), dtype)
    ex = _li_slot(xp, cell, hu, hv, series_u, series_v, 'u', nesting)
    ey = _li_slot(xp, cell, hu, hv, series_u, series_v, 'v', nesting)
    ez = xp.linalg.inv(_li_slot(xp, cell, hu, hv, series_u, series_v, None, nesting))
    return ex, ey, ez


def _to_host(array):
    """A backend array of concrete numbers as NumPy (edges are geometry, never traced here)."""
    for name in ('detach',):
        if hasattr(array, name):
            array = getattr(array, name)()
    if hasattr(array, 'cpu'):
        array = array.cpu()
    import numpy as _np
    return _np.asarray(array)


def stretched_mu(xp, hu, hv, dtype, stretch):
    """The stretched vacuum's permeability slots ``(mux, muy, muz_i)``: ``diag(g/f, f/g, f g)``.

    Smooth in both coordinates, so the direct rule: each slot is the Kronecker product of two
    one-axis Toeplitz matrices, of the slope or of its reciprocal.
    """
    one = xp.ones((1, 1), dtype=dtype)

    def series(axis, half_width, weight):
        return _table_series(xp, axis.period_table(half_width, weight)[:, None], dtype)

    def slot(weight_u, weight_v):
        return _conv_matrix(xp, one, hu, hv, series(stretch.x, hu, weight_u),
                            series(stretch.y, hv, weight_v))

    return (slot('reciprocal', 'slope'), slot('slope', 'reciprocal'),
            xp.linalg.inv(slot('slope', 'slope')))


def layer_matrices(xp, shapes, background, hu, hv, lattice, dtype, plan=None, nesting=None):
    edges, owners = layer_partition(xp, shapes, lattice, dtype, plan)
    materials = [as_array(xp, as_material(s.get('material', 1.)).epsilon, dtype=dtype) for s in shapes]
    materials.append(as_array(xp, as_material(background).epsilon, dtype=dtype))
    cell = xp.stack([xp.stack([materials[i] for i in row]) for row in owners])
    ex = fourier2d_integrated(xp, cell, *edges, hu, hv, inverse_rule='u', nesting=nesting)
    ey = fourier2d_integrated(xp, cell, *edges, hu, hv, inverse_rule='v', nesting=nesting)
    ez = xp.linalg.inv(fourier2d_integrated(xp, cell, *edges, hu, hv, inverse_rule=None))
    return ex, ey, ez


def layer_stack(xp, layers, hu, hv, lattice, dtype, plans=None, nesting=None, stretch=None):
    validate(layers, lattice)
    if plans is not None and len(plans) != len(layers):
        raise StaleGeometryError('Cartesian Li layer count changed; freeze_geometry() again')
    if stretch is not None:
        matrices = [stretched_layer_matrices(
            xp, layer.get('shapes', []), layer.get('background', 1.), hu, hv, lattice, dtype,
            stretch, None if plans is None else plans[i], nesting)
            for i, layer in enumerate(layers)]
        return ConvMatrices(*(xp.stack([m[i] for m in matrices]) for i in range(3)), None, None)
    matrices = [layer_matrices(xp, layer.get('shapes', []), layer.get('background', 1.),
                               hu, hv, lattice, dtype, None if plans is None else plans[i],
                               nesting)
                for i, layer in enumerate(layers)]
    return ConvMatrices(*(xp.stack([m[i] for m in matrices]) for i in range(3)), None, None)
