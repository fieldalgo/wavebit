"""Successive Li factorization of Cartesian material tensors.

The mixed (normal displacement, tangential field) variables are formed before
each one-dimensional Fourier transform. The inner transform is v, the outer
transform u: the scalar Li path's fixed nesting (``Nesting.FIXED``), the only
one built here. No normal field is sampled. The one-dimensional series are the
scalar builders' own (``factorization._raster_series`` for pixels,
``factorization._segment_series`` for segment edges), so a scalar tensor
reproduces the scalar operator.

Reference: L. Li, Fourier Modal Method, Gratings: Theory and Numeric
Applications (2014), sections 13.3.3.1 and 13.4.3, especially (13.132).
This implementation derives the block elimination independently; it contains
no third-party source code. Until the Li builders were rewritten (2026-09-25)
its row transform went through that module's earlier helpers ``_wrap``,
``_circulant_gather`` and ``_fourier1d_integrated``, whose history LICENSE
records, and through the JAX workaround ``_normalized_row_fft``. See
docs/theory/rcwa/tensor-li-factorization.md.
"""
import wavebit.rcwa._engine.factorization as F
import wavebit.rcwa._engine.rectilinear as rectilinear
from wavebit.materials.response import as_material
from wavebit.rcwa._engine.geometry import StaticGeometry
from wavebit.rcwa._engine.geometry import StaleGeometryError


def _exchange(xp, blocks, axis, *, pointwise=False):
    """Exchange E_axis with D_axis; this block Schur map is its own inverse."""
    a = blocks[axis][axis]
    inverse = 1 / a if pointwise else xp.linalg.inv(a)
    multiply = (lambda x, y: x * y) if pointwise else (lambda x, y: x @ y)
    size = len(blocks)
    out = [[None] * size for _ in range(size)]
    out[axis][axis] = inverse
    for j in range(size):
        if j != axis:
            out[axis][j] = -multiply(inverse, blocks[axis][j])
            out[j][axis] = multiply(blocks[j][axis], inverse)
    for i in range(size):
        for j in range(size):
            if i != axis and j != axis:
                out[i][j] = blocks[i][j] + multiply(blocks[i][axis], out[axis][j])
    return out


def _rows(xp, values, half_width, edges):
    """Fourier coefficients of piecewise constant rows, then Toeplitz gather.

    ``values`` is ``(B, N)``; the result is ``(B, 2h+1, 2h+1)`` with ``[b, r, c]`` the order
    ``r - c`` coefficient of row ``b``. Without ``edges`` the rows fill ``N`` equal pixels (the
    exact box transform); with them, ``edges`` are the ``(N, 1)`` right-hand segment edges, the
    last closing the period.
    """
    dev = F._device(xp, values)
    if edges is None:
        series = F._raster_series(xp, half_width, F._pixel_correction)
    else:
        real = xp.float32 if values.dtype == xp.complex64 else xp.float64
        series = F._segment_series(xp, *F._segment_bounds(xp, edges, real, dev), half_width)
    coefficients = series(values)
    n = 2 * half_width + 1
    gathered = xp.take(coefficients, F._toeplitz_positions(xp, half_width, dev), axis=-1)
    return xp.reshape(gathered, (coefficients.shape[0], n, n))


def layer_matrices(xp, tensor, hu, hv, edges=None):
    """Five or nine slots for a piecewise-constant (rows, columns, 3, 3) layer."""
    from wavebit.rcwa._engine.anisotropy import full_tensor
    from wavebit.rcwa._engine.anisotropy import tensor_slots
    size = 3 if full_tensor(xp, tensor) else 2
    u_edges, v_edges = (None, None) if edges is None else edges
    block = [[tensor[..., i, j] for j in range(size)] for i in range(size)]
    # Normal D_y and tangential E_x are continuous through a y interface.
    mixed_v = _exchange(xp, block, 1, pointwise=True)
    inner = [[_rows(xp, xp.matrix_transpose(a), hv, v_edges) for a in row]
             for row in mixed_v]
    inner = _exchange(xp, inner, 1)
    # These are matrix-valued functions of x; product order must be retained.
    mixed_u = _exchange(xp, inner, 0)
    nv, nu = 2 * hv + 1, 2 * hu + 1
    def outer(a):
        rows = xp.matrix_transpose(xp.reshape(a, (a.shape[0], nv * nv)))
        out = _rows(xp, rows, hu, u_edges)
        out = xp.reshape(out, (nv, nv, nu, nu))
        return xp.reshape(xp.moveaxis(out, 1, 2), (nv * nu, nv * nu))
    effective = _exchange(xp, [[outer(a) for a in row] for row in mixed_u], 0)
    if size == 3:
        return tensor_slots(xp, effective)
    if edges is None:
        zz = F.fourier2d_piecewise_constant(xp, tensor[..., 2, 2], hu, hv, inverse_rule=None)
    else:
        zz = F.fourier2d_integrated(xp, tensor[..., 2, 2], *edges, hu, hv, inverse_rule=None)
    return F.ConvMatrices(effective[0][0], effective[1][1], xp.linalg.inv(zz),
                         -effective[0][1], -effective[1][0])


def _settings(solver):
    return (tuple(tuple(v) for v in solver.order_window), solver._fourier_coefficients)


def freeze(solver):
    shaped = isinstance(solver.structure, list)
    plans = (tuple(rectilinear.plan_layer(l.get('shapes', []), solver.lattice)
                   for l in solver.structure) if shaped else None)
    return StaticGeometry(tensor_geometry={
        'mode': 'li-shapes' if shaped else 'li-raster',
        'plans': plans, 'settings': _settings(solver),
        'shape': None if shaped else tuple(solver.structure.shape),
    })


def check_geometry(solver, geometry):
    """Validate the Li record without constructing Fourier matrices."""
    shaped = isinstance(solver.structure, list)
    record = None if geometry is None else geometry.tensor_geometry
    mode = 'li-shapes' if shaped else 'li-raster'
    if geometry is not None and (record is None or record.get('mode') != mode):
        raise StaleGeometryError('tensor factorization changed; freeze_geometry() for Li again')
    if record is not None and record['settings'] != _settings(solver):
        raise StaleGeometryError('tensor Li orders or Fourier coefficients changed; freeze_geometry() again')
    if record is not None and not shaped and record['shape'] != tuple(solver.structure.shape):
        raise StaleGeometryError('tensor Li raster shape changed; freeze_geometry() again')
    if shaped and record is not None:
        plans = record['plans']
        if plans is not None and len(plans) != len(solver.structure):
            raise StaleGeometryError('tensor Li layer count changed; freeze_geometry() again')
        if plans is not None:
            for layer, plan in zip(solver.structure, plans):
                rectilinear.check_plan(layer.get('shapes', []), solver.lattice, plan)
    return record


def build(solver, geometry=None, quantity='epsilon'):
    import wavebit.rcwa._engine.anisotropy as A
    from wavebit.rcwa._engine.enums import FourierCoefficients
    shaped = isinstance(solver.structure, list)
    record = check_geometry(solver, geometry)
    xp, dtype, dev = solver.xp, solver.complex_dtype, solver._array_device
    (_, hu), (_, hv) = solver.order_window
    count = (2 * hu + 1) * (2 * hv + 1)
    def value(v):
        material = as_material(v)
        return A.as_tensor(xp, material.epsilon if quantity == 'epsilon' else material.mu_value, dtype, dev)
    if shaped:
        plans = None if record is None else record['plans']
        fields = []
        for i, layer in enumerate(solver.structure):
            shapes = layer.get('shapes', [])
            edges, owners = rectilinear.layer_partition(
                xp, shapes, solver.lattice, dtype, None if plans is None else plans[i], dev)
            materials = [value(s.get('material', 1.)) for s in shapes] + [value(layer.get('background', 1.))]
            fields.append((xp.stack([xp.stack([materials[j] for j in row]) for row in owners]), edges))
    else:
        source = solver.structure if quantity == 'epsilon' else solver.mu
        fields = [(t, None) for t in A._field(solver, source, quantity)]
        if solver._fourier_coefficients == FourierCoefficients.INTEGRATED:
            fields = [(t, [xp.reshape(xp.arange(1, n + 1, dtype=solver.float_dtype, device=dev) / n, (-1, 1))
                           for n in (t.shape[1], t.shape[0])]) for t, _ in fields]
    layers, any_full = [], False
    for i, (tensor, edges) in enumerate(fields):
        if tensor.shape[:2] != (1, 1):
            slots = list(layer_matrices(xp, tensor, hu, hv, edges))
            layers.append(slots)
            any_full |= len(slots) == 9
            continue
        # Uniform material has no in-plane boundary to factorize.
        region = as_material(solver.structure[i].get('background', 1.)) if shaped else None
        raw = (region.epsilon if quantity == 'epsilon' else region.mu_value) if shaped else None
        declared_diagonal = shaped and A.tensor_shape(raw) in ((), (3,))
        full = not declared_diagonal and A.full_tensor(xp, tensor)
        layers.append(A._uniform_slots(xp, tensor[0, 0], count, full))
        any_full |= full
    if any_full:
        for layer in layers:
            if len(layer) == 5:
                layer.extend([xp.zeros_like(layer[0]) for _ in range(4)])
    cls = A.TensorConvMatrices if any_full else F.ConvMatrices
    return cls(*(xp.stack([layer[i] for layer in layers]) for i in range(9 if any_full else 5)))
