"""Single-source Fourier factorization — Phase 4a of the rewrite (see ``<a scratch note outside this repository>``).

Builds the per-layer convolution matrices (``epx`` / ``epy`` / ``inv(epz)``) that the solve
core consumes, written once against the Array-API namespace instead of once per backend.

Why this matters beyond de-duplication: until the factorization is inside the differentiable
chain, autodiff can only reach the *convolution matrices*, not the permittivity that produced
them. With this module the gradient runs end-to-end from the design variable — the ``ucell``
raster — through to the diffraction efficiency, which is what an inverse-design user actually
differentiates.

All the strategies live here. The raster builders are the cross product of two independent
choices — how the Fourier coefficients are obtained, and which rule closes the products of
discontinuous functions:

- ``('sampled', 'li')``       -> :func:`to_conv_mat_raster_discrete`, discrete Fourier series
  off the raster with Li's rules. The default, and the only path that carries a gradient to
  every raster cell.
- ``('integrated', 'li')``    -> :func:`to_conv_mat_raster_continuous`, closed-form
  (staircase-free) Fourier of the compressed layer with Li's rules.
- ``('sampled', 'nvf')``      -> :func:`to_conv_mat_raster_nvf`, DFS with the
  normal-vector-field factorization.
- ``('integrated', 'nvf')``   -> :func:`to_conv_mat_raster_continuous_nvf`, closed-form
  Fourier with NVF.

Vector-modelled layers (:func:`to_conv_mat_block`) are the closed-form series on explicitly
given segment edges, with Li's rules; the analytic *shape* path lives in
:mod:`fieldwave._core.analytic` and is always NVF.

A recurring rule runs through this module: **discrete geometry decisions are computed eagerly
and passed in as static inputs, while material values flow through differentiably.** Two
decisions here are geometry, not material — the NVF normal field (where the interfaces point) and the CFS block
compression (where the piecewise-constant blocks start and stop). Both branch on values, and
the compression's output shape depends on the answer, so neither survives ``jax.jit``. Each is
therefore split into an eager step that yields a static description
(:func:`normal_field_for_ucell`, :func:`compression_indices`) and a differentiable step that
consumes it. Pass those precomputed descriptions to stay traceable; the permittivity riding on
top of them is what gradients see.

``jax.grad`` alone is the weaker bar, and both decisions clear it, because a grad tracer
carries a concrete primal and so permits narrowing one value at a time even though it refuses a
whole-array NumPy conversion (:func:`_as_bools`). The compression compares neighbouring
rows and columns that way directly. The normal field cannot — SciPy wants the mask as an array
— so it derives that mask from the *base* raster rather than reading it off the upsampled one
(:func:`_upsampled_interface_mask`); the upsample is a block repeat, so the two are the same
mask, and the base has ``repeat_y * repeat_x`` fewer values to narrow. **Neither raster
strategy needs a frozen geometry to differentiate**, which is what makes a fit whose pattern
moves every step safe: with no record there is nothing to go stale.

``jax.jit`` is the bar neither clears, and that is what a frozen geometry is for.

A precomputed description belongs to the structure it was taken on. Replaying one onto a
raster whose *pattern* has moved describes a layer that is no longer there, so each of the two
is checked against the layer it is handed — see :func:`_check_normal_field_still_fits` and
:func:`_check_compression_still_fits`, and :mod:`fieldwave._core.geometry` for the rule they
share. Material values never move a pattern, so the sweeps this split exists to serve pass
untouched. A fit whose *pattern* moves every step is a different matter: on the compression it
should simply drop ``geometry=`` and differentiate, which cannot go stale because it records
nothing.
"""
from .backend import array_namespace
from .geometry import StaleGeometryError


def _wrap(orders, n):
    """``orders`` reduced mod ``n`` — the DFT bin each Fourier order lands in."""
    return [[k % n for k in row] for row in orders]


def _circulant_index(fto):
    """Circular index matrix selecting Fourier orders ``m - n`` (pure integer, data-free).

    The sign here *is* the order-index convention, and it has to agree with the sign ``kx``
    carries: a convolution matrix built for one and paired with the other mirrors the
    diffraction orders (+1 and -1 swap) while conserving energy, so nothing catches it except
    a grating comparison. Uniform layers cannot expose it at all -- their convolution matrix
    is ``eps * I``, which is its own transpose.
    """
    ff = 2 * fto + 1
    return [[(r - c) for c in range(ff)] for r in range(ff)]


def _pixel_correction(xp, orders, n, dtype, device=None):
    """Turn an ``n``-point DFT coefficient into the exact one for ``n`` constant pixels.

    A raster layer is not a set of samples, it is a piecewise-constant function: pixel ``j``
    holds its value across ``[j/n, (j+1)/n)``. Integrating that against ``exp(-2*pi*i*k*x)``
    gives the ``n``-point DFT coefficient times ``sinc(k/n) * exp(-i*pi*k/n)`` — a box
    transform and the half-pixel offset of the cell centre. Exactly, for any ``k``, including
    ``|k| > n`` where the DFT itself has wrapped.

    That last part is what makes the upsampling unnecessary rather than merely cheaper. The
    previous approach repeated each pixel until the grid resolved ``4*fto+1`` harmonics and
    took the DFT of *that*, which converges to this expression as ``1/R`` — so it was an
    approximation of a closed form, at a grid cost of ``(4*fto+1)**2`` per layer.
    """
    k = xp.astype(xp.asarray(orders, device=device), dtype)
    t = k / n
    pi = 3.141592653589793
    # sinc(t) with the removable singularity at t = 0 handled without a branch, so this stays
    # differentiable and traceable.
    num = xp.sin(pi * t)
    den = xp.where(t == 0, xp.ones_like(t), pi * t)
    sinc = xp.where(t == 0, xp.ones_like(t), num / den)
    return sinc * xp.exp(-1j * pi * t)


def dfs2d(xp, cell, conti_x, conti_y, fto_x, fto_y, piecewise_constant=False):
    """2D discrete Fourier series of ``cell`` with Li's factorization rules.

    ``conti_x`` / ``conti_y`` flag the direction as continuous (1) or discontinuous (0); a
    discontinuous direction takes the inverse rule. Port of the legacy ``dfs2d``.

    ``piecewise_constant`` says each value of ``cell`` holds across its whole pixel, rather
    than being a sample of something that varies inside it. When that is true the coefficients
    have an exact closed form and this computes it (see :func:`_pixel_correction`); when it is
    false there is no exact value to reach from the grid at all, and the transform is left
    alone.

    A permittivity raster is the first case — ``ucell`` names the material *occupying* each
    cell, and a coarser or finer spelling of the same structure is the same function. A normal
    field is the second: it is a continuous vector field this code evaluates on a grid, so
    refining the grid changes the field itself. Applying the correction there is harmless but
    pointless (measured end-to-end at ~7e-7, four decades under the field's own discretization
    error), and it would assert a shape the data does not have.
    """
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    cell = xp.matrix_transpose(cell)

    if conti_y == 0:      # discontinuous in Y (row): inverse rule
        cell = 1 / cell

    dfs1d = xp.fft.fft(cell / cell.shape[1])

    dev = _device(xp, cell)
    ord_y = _circulant_index(fto_y)
    ord_x = _circulant_index(fto_x)
    n_y = cell.shape[1]

    # Which DFT bin an order lands in, stated rather than left to Python's negative indexing.
    # Identical wherever the old form worked, and defined where it did not: raw order indices
    # need a grid of at least 4*fto+1 to stay in range, which is the whole reason the raster
    # used to be upsampled before anything else was considered.
    idx_y = xp.asarray(_wrap(ord_y, n_y), device=dev)

    conv1d = dfs1d[:, idx_y]
    if piecewise_constant:
        conv1d = conv1d * _pixel_correction(xp, ord_y, n_y, conv1d.dtype, dev)

    if conti_x ^ conti_y:
        conv1d = xp.linalg.inv(conv1d)

    conv1d = xp.reshape(conv1d, (-1, ff_y ** 2))

    conv1d_t = xp.matrix_transpose(conv1d)
    n_x = conv1d_t.shape[1]
    dfs_2d = xp.fft.fft(conv1d_t / n_x)

    idx_x = xp.asarray(_wrap(ord_x, n_x), device=dev)
    conv2d = dfs_2d[:, idx_x]
    if piecewise_constant:
        conv2d = conv2d * _pixel_correction(xp, ord_x, n_x, conv2d.dtype, dev)
    conv2d = xp.reshape(conv2d, (ff_y, ff_y, ff_x, ff_x))
    conv2d = xp.moveaxis(conv2d, 1, 2)
    conv2d = xp.reshape(conv2d, (ff_y * ff_x, ff_y * ff_x))

    if conti_x == 0:      # discontinuous in X (column): inverse rule
        conv2d = xp.linalg.inv(conv2d)

    return conv2d


def _device(xp, x):
    """Device of ``x`` (Array-API ``device`` attribute), for placing index arrays alongside."""
    return x.device if hasattr(x, 'device') else None


def _layer_eps(xp, ucell):
    """Yield each layer's permittivity grid, on the raster exactly as given.

    The ``n**2`` maps refractive index to permittivity: FieldWave works in ``e^{-iwt}``
    throughout (``Im(n) > 0`` = loss), with no conjugation on the way in.

    Nothing is resampled. :func:`dfs2d` obtains the raster's Fourier coefficients in closed
    form (:func:`_pixel_correction`), so a finer grid of the same staircase is the same
    function and buys nothing. The NVF path is the one that still needs a floor, and states
    that itself — see :func:`_layer_eps_for_normal_field`.
    """
    for i in range(ucell.shape[0]):
        layer = ucell[i]
        yield xp.astype(layer, _complex_dtype(xp, layer)) ** 2


def _normal_field_floor(shape, fto_x, fto_y):
    """Repeat factors that lift a layer to the grid the normal field's transform needs.

    The normal field is a continuous vector field this code evaluates on a grid, so unlike the
    permittivity it has no closed-form transform and :func:`dfs2d` reads it off the grid
    directly. That requires ``4*fto+1`` points: the circulant asks for orders up to
    ``+/-2*fto``, and on a shorter grid ``k % n`` folds distinct orders onto one bin. The
    damage is structural rather than gradual — a *constant* normal field must transform to the
    identity for NVF to collapse onto Li's rules on a one-cell axis, and on a short grid it
    picks up off-diagonals at ``+/-n`` instead.

    Only up to that floor. Going past it was the old ``enhanced_dfs=True``, which multiplied it
    by the raster size; measured across 32 shape/material/polarization/order combinations it
    lost accuracy in 27 and gained in none that stood up (see ``docs/nvf-upsampling.md``).
    """
    return (max(1, (4 * fto_y + 1) // shape[0] + 1) if shape[0] < 4 * fto_y + 1 else 1,
            max(1, (4 * fto_x + 1) // shape[1] + 1) if shape[1] < 4 * fto_x + 1 else 1)


def _layer_eps_for_normal_field(xp, ucell, fto_x, fto_y):
    """Each layer's permittivity grid for the NVF path, with the base it was repeated from.

    Yields ``(eps, base_eps, repeat_y, repeat_x)``. The base is what
    :func:`_upsampled_interface_mask` derives the interface mask from when the values cannot be
    read whole, which is what lets NVF differentiate without a frozen geometry.
    """
    for i in range(ucell.shape[0]):
        layer = ucell[i]
        base = xp.astype(layer, _complex_dtype(xp, layer)) ** 2
        repeat_y, repeat_x = _normal_field_floor(layer.shape, fto_x, fto_y)
        if repeat_y > 1:
            layer = xp.repeat(layer, repeat_y, axis=0)
        if repeat_x > 1:
            layer = xp.repeat(layer, repeat_x, axis=1)
        yield xp.astype(layer, _complex_dtype(xp, layer)) ** 2, base, repeat_y, repeat_x


def to_conv_mat_raster_discrete(ucell, fto_x, fto_y):
    """Per-layer convolution matrices for a raster ``ucell`` via the DFS path.

    Args:
        ucell: refractive-index raster, shape ``(n_layers, Ny, Nx)``. This is the design
            variable — gradients flow back to it.
        fto_x, fto_y: Fourier truncation orders.

    Returns:
        ``(epx_conv_all, epy_conv_all, epz_conv_i_all)``, each ``(n_layers, ff_xy, ff_xy)``.
    """
    xp = array_namespace(ucell)

    epx_list, epy_list, epz_i_list = [], [], []

    for eps_matrix in _layer_eps(xp, ucell):
        epz_conv = dfs2d(xp, eps_matrix, 1, 1, fto_x, fto_y, piecewise_constant=True)
        epy_conv = dfs2d(xp, eps_matrix, 1, 0, fto_x, fto_y, piecewise_constant=True)
        epx_conv = dfs2d(xp, eps_matrix, 0, 1, fto_x, fto_y, piecewise_constant=True)

        epx_list.append(epx_conv)
        epy_list.append(epy_conv)
        epz_i_list.append(xp.linalg.inv(epz_conv))

    return xp.stack(epx_list), xp.stack(epy_list), xp.stack(epz_i_list)


def _complex_dtype(xp, x):
    """The complex dtype matching ``x``'s precision (complex64 for 32-bit inputs)."""
    if xp.isdtype(x.dtype, 'complex floating'):
        return x.dtype
    return xp.complex64 if x.dtype == xp.float32 else xp.complex128


# ============================================================ normal vector field (NVF)

def _to_numpy(x):
    """Concrete NumPy view of ``x`` (detaching a grad-tracking torch tensor if needed)."""
    import numpy as np
    if hasattr(x, 'detach'):
        x = x.detach()
    return np.asarray(x)


def normal_vector_field(eps_matrix, interface_mask=None, lattice=None):
    """Unit normal vector field ``(nx, ny)`` of a raster layer, via a signed-distance field.

    This is a **geometry** quantity — it depends on where the material interfaces sit, not on
    the permittivity values carried between them — so it is computed eagerly in NumPy/SciPy
    and then held fixed while gradients flow through the permittivity.

    ``interface_mask`` is the one input the whole construction reads (see
    :func:`_interface_mask`). Supplying it lets a caller that can obtain the mask some other
    way skip the whole-array NumPy conversion this would otherwise need, which is what lets the
    NVF paths differentiate without a frozen geometry — see
    :func:`_upsampled_interface_mask`. Left ``None``, the mask is read off ``eps_matrix``,
    which needs concrete values.

    The signed-distance field is the only raster method offered, deliberately. The
    factorization needs a normal field that is unit magnitude across the *whole* cell — that
    is what makes ``C_xx + C_yy = I`` in :func:`nvf_conv_matrices`. The distance transform
    extends the interface normal into the bulk and delivers exactly that; the legacy
    ``method='sobel'`` alternative (a raw ``grad(eps)``) is nonzero only in the one-pixel band
    at the interface, so it is not a unit field and collapses the projector toward 0, which
    degenerates the scheme back toward the plain direct rule. It was never reachable from the
    solver and was dropped in the port. See ``tests/test_nvf_normal_field.py``.
    For the shape-based (non-raster) normal field, see
    :func:`fieldwave._core.analytic.normal_vector_field_analytical`.

    ``lattice`` is the cell basis the raster indexes into. It matters because this is the one
    piece of the NVF construction that is *Cartesian-geometric* rather than index-based: the
    distance transform and ``np.gradient`` both work in index steps, and only a basis whose
    pixels are square in real space makes an index-space normal a Cartesian one. Left ``None``
    the index grid is taken to be isotropic Cartesian, which is what this has always assumed;
    given a basis, distances pick up the per-axis physical step and the gradient is carried
    through ``J^-T`` (:func:`fieldwave._core.lattice.jacobian`) before normalising, so the
    direction is exact at the interface for a skewed *or* merely non-square cell.

    That correction is what lets NVF serve a non-orthogonal lattice at all, and it is also why
    NVF rather than Li's nested rule is the factorization that ports there: the tensor this
    feeds (:func:`_nvf_assemble`) is written in Cartesian components, so once the normal is
    Cartesian nothing else in the factorization learns the cell was skewed.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    shape = tuple(eps_matrix.shape)

    # A one-cell axis carries no interface of its own: the structure cannot vary along it, so
    # every normal points along the other axis. Say so directly -- `np.gradient` needs three
    # cells and would otherwise raise, which is what made NVF unusable on 1D gratings. With a
    # constant normal the NVF projector reduces to Li's rules exactly (C_xx = I, C_xy = C_yy =
    # 0 gives eps_xx = <1/eps>^-1 and eps_yy = <eps>), so this is not an approximation.
    if 1 in shape:
        along_x = np.ones(shape), np.zeros(shape)
        return along_x if shape[0] == 1 else (along_x[1], along_x[0])

    if interface_mask is None:
        interface_mask = _interface_mask(_to_numpy(eps_matrix))

    if lattice is None:
        dist_map = distance_transform_edt(~interface_mask)
        g_j, g_i = np.gradient(dist_map)
    else:
        from .lattice import jacobian
        ny, nx = shape
        J = jacobian(lattice, ny, nx)
        # Work in units of one x-step: divide J through by |a1|/nx before anything reads it.
        # A uniform scale cannot survive the normalisation at the end, so this changes no
        # direction — but it makes a square-pixel raster land on exactly `sampling=(1, 1)` and
        # a `J^-T` of the identity, which is bit-identical to taking no metric at all. Without
        # it those cells moved by ~7e-15 purely from multiplying a factor in and dividing it
        # back out.
        J = J / (float(np.hypot(J[0, 0], J[1, 0])) or 1.0)
        # Physical length of one index step along each axis: axis 0 walks a2/ny, axis 1 a1/nx.
        # Exact for an axis-aligned cell; for a sheared one it is the best `sampling` can say,
        # since scipy's transform has no way to express a non-diagonal metric. The direction
        # that survives normalisation is set by the gradient below, which is exact regardless.
        step_j = float(np.hypot(J[0, 1], J[1, 1]))
        step_i = float(np.hypot(J[0, 0], J[1, 0]))
        if step_j == step_i == 1.0:
            dist_map = distance_transform_edt(~interface_mask)
        else:
            dist_map = distance_transform_edt(~interface_mask, sampling=(step_j, step_i))
        g_j, g_i = np.gradient(dist_map)
        # grad_xy f = J^-T @ (df/di, df/dj) -- the inverse transpose, not the Jacobian: index
        # steps push forward, gradients pull back.
        Jinv_T = np.linalg.inv(J).T
        g_i, g_j = (Jinv_T[0, 0] * g_i + Jinv_T[0, 1] * g_j,
                    Jinv_T[1, 0] * g_i + Jinv_T[1, 1] * g_j)

    gx, gy = g_i, g_j
    norm = np.hypot(gx, gy)
    norm[norm == 0] = 1.0
    return gx / norm, gy / norm


def _interface_mask(eps):
    """Cells that differ from a 4-neighbour — where the material interfaces sit."""
    import numpy as np
    border = np.zeros(eps.shape, dtype=bool)
    border[1:, :] |= (eps[1:, :] != eps[:-1, :])
    border[:-1, :] |= (eps[:-1, :] != eps[1:, :])
    border[:, 1:] |= (eps[:, 1:] != eps[:, :-1])
    border[:, :-1] |= (eps[:, :-1] != eps[:, 1:])
    return border


def _upsampled_interface_mask(xp, base_eps, repeat_y, repeat_x):
    """The interface mask of a block-repeated grid, from the base grid's neighbour comparisons.

    :func:`_layer_eps_for_normal_field` lifts a layer to the floor by repeating each cell, so the repeated grid is
    constant inside every block. Two neighbouring cells of it can therefore only differ where
    they straddle a block boundary *and* the base differs across it — which means the whole
    mask follows from ``(base_ny-1)*base_nx + base_ny*(base_nx-1)`` comparisons on the base,
    rather than one read of the repeated grid.

    That is the whole point. Reading the repeated grid means converting it to NumPy in one
    step, which a tracer refuses; narrowing the base's comparisons one value at a time is
    something a grad tracer allows, and there are ``repeat_y * repeat_x`` times fewer of them —
    128x128 rather than 2688x2688 at ``fto=5``. It is what lets NVF differentiate with no
    frozen geometry. (Under ``jax.jit`` neither works: there is no primal to narrow.)

    Note this is *not* the base's own mask repeated. Repeating ``[1, 2]`` twice gives
    ``[1, 1, 2, 2]``, whose border cells are the middle two, not all four.
    """
    import numpy as np

    base_ny, base_nx = tuple(base_eps.shape)
    border = np.zeros((base_ny * repeat_y, base_nx * repeat_x), dtype=bool)

    if base_ny > 1:
        differs = _as_bools(xp, base_eps[:-1, :] != base_eps[1:, :])
        differs = np.repeat(differs, repeat_x, axis=1)
        last_of_block = np.arange(base_ny - 1) * repeat_y + repeat_y - 1
        border[last_of_block, :] |= differs
        border[last_of_block + 1, :] |= differs

    if base_nx > 1:
        differs = _as_bools(xp, base_eps[:, :-1] != base_eps[:, 1:])
        differs = np.repeat(differs, repeat_y, axis=0)
        last_of_block = np.arange(base_nx - 1) * repeat_x + repeat_x - 1
        border[:, last_of_block] |= differs
        border[:, last_of_block + 1] |= differs

    return border


def _interface_mask_of(xp, eps, base_eps, repeat_y, repeat_x):
    """The interface mask of ``eps``, by whichever route the backend allows.

    Straight off the repeated grid when its values are concrete — the same read as before, and
    the cheaper one. From the base grid when they are not, which is the case a gradient
    presents.
    """
    try:
        return _interface_mask(_to_numpy(eps))
    except Exception:                     # a tracer: derive it from the base instead
        return _upsampled_interface_mask(xp, base_eps, repeat_y, repeat_x)


def _lattice_key(lattice):
    """A hashable, comparable stand-in for a basis — ``None`` stays ``None``.

    Scaled so ``|a1| == 1``, which makes the key record the cell's *shape* and not its size.
    That is exact rather than merely tolerant: a uniform rescale of the cell leaves the normal
    field identical (the scale multiplies into the distance transform and divides straight back
    out of the normalisation), so keying on absolute lengths would refuse replays that are in
    fact the same computation. Combined with the grid shape, which the key already carries,
    this pins the pixel geometry exactly.
    """
    if lattice is None:
        return None
    import numpy as np
    A = np.asarray(lattice, dtype=float)
    scale = float(np.hypot(A[0, 0], A[1, 0])) or 1.0
    return tuple((A / scale).reshape(-1).tolist())


def _normal_field_source(eps, lattice=None):
    """Everything :func:`normal_vector_field` reads off a layer, and nothing else.

    That is the shape, plus — unless an axis is a single cell, where the field is the constant
    along-axis one whatever the pattern — the interface mask, plus the cell basis. Two layers
    with the same source have the same normal field, bit for bit, which is what makes this the
    tell: it accepts a replay exactly when replaying is the same computation as rebuilding.

    The basis belongs here for the same reason the mask does. It is an *input* to the field
    (see :func:`normal_vector_field`), so a field frozen on one cell and replayed on a skewed
    one describes interfaces pointing the wrong way — with nothing to notice, since the NVF
    assembly stays Hermitian and energy still conserves.
    """
    if 1 in eps.shape:
        return eps.shape, None, _lattice_key(lattice)
    return eps.shape, _interface_mask(eps), _lattice_key(lattice)


def normal_field_for_ucell(ucell, fto_x, fto_y, lattice=None):
    """Per-layer ``(nx, ny)`` normal fields for ``ucell``, on the grid the NVF build uses.

    Precompute this once from the concrete geometry and hand it to
    :func:`to_conv_mat_raster_nvf` as ``normal_field=`` to keep the build traceable under a
    JAX transform (the SciPy distance transform cannot run on tracers).

    ``lattice`` is the cell basis; see :func:`normal_vector_field` for what it changes.
    """
    xp = array_namespace(ucell)
    return [normal_vector_field(eps, lattice=lattice)
            for eps, _, _, _ in _layer_eps_for_normal_field(xp, ucell, fto_x, fto_y)]


def normal_field_source_for_ucell(ucell, fto_x, fto_y, lattice=None):
    """What :func:`normal_field_for_ucell` derived its answer from, per layer.

    Freeze this alongside the normal field and hand both back to
    :func:`to_conv_mat_raster_nvf`; the replay compares it against the layer actually in front
    of it and refuses a field that was sampled from a different one.
    """
    xp = array_namespace(ucell)
    return [_normal_field_source(_to_numpy(eps), lattice)
            for eps, _, _, _ in _layer_eps_for_normal_field(xp, ucell, fto_x, fto_y)]


def _check_normal_field_still_fits(eps_matrix, source, index, lattice=None):
    """Refuse a frozen normal field sampled from interfaces that have since moved.

    The field says which way each interface points. Move the pattern in the raster and it is
    describing a layer that is no longer there — measured on a 40x40 rod frozen at 12 cells
    wide and replayed at 20, the transmitted orders come back 0.55 off and the total reads
    0.710 against a true 0.346. Nothing raised, and nothing looked unusual.

    The tell is exact rather than merely conservative, because :func:`_normal_field_source`
    captures the field's *only* input: same source, same field, so a replay it accepts is the
    same computation as a rebuild. Over 300 randomised replays no accepted one differed from a
    rebuild at all (0.0e+00), and the one rejected replay that was nonetheless exact had lost
    every interface it had — a layer gone uniform, where no normal field is wrong because none
    is used. Re-freezing costs one call.

    Only reachable eagerly. Under a trace the layer cannot be read at all, which is the point
    of having frozen it — so a jitted call carries no check, and a long fit should re-freeze on
    a schedule rather than rely on being told.
    """
    import numpy as np
    if source is None:
        return
    try:
        eps = _to_numpy(eps_matrix)
    except Exception:                     # a tracer: nothing to compare, nothing to do
        return

    entry = source[index]
    shape, mask = entry[0], entry[1]
    lat = entry[2] if len(entry) > 2 else None      # records frozen before the basis was one
    now_shape, now_mask, now_lat = _normal_field_source(eps, lattice)
    if (shape == now_shape and lat == now_lat
            and (mask is None or np.array_equal(mask, now_mask))):
        return                            # equal shapes agree on whether a mask exists at all

    if lat != now_lat:
        moved = (f'the cell basis is now {now_lat}, where the normal field was sampled on '
                 f'{lat} — the interfaces point the same way in the raster but a different '
                 f'way in space')
    elif shape != now_shape:
        moved = (f'layer {index} is now {now_shape}, where the normal field was sampled on '
                 f'{shape}')
    else:
        moved = (f"layer {index}'s material interfaces have moved — "
                 f'{int(np.count_nonzero(mask != now_mask))} of {mask.size} cells of the '
                 f'pattern the normal field was sampled from now differ')
    raise StaleGeometryError(
        f'the frozen geometry no longer fits this structure: {moved}. The recorded field '
        f'points along interfaces that are not there any more, so it describes a different '
        f'layer. Call freeze_geometry() again, or drop geometry= to rebuild by comparison. '
        f'(Changing the refractive index of a layer never trips this; moving its pattern '
        f'does.)')


def _nvf_projector(xp, nx, ny, fto_x, fto_y, dtype, device=None):
    """Convolution matrices of the normal projector: ``C_xx = <nx^2>`` etc.

    Always sampled off the raster the normal field lives on, whatever
    ``fourier_coefficients`` says. That keyword describes how the *permittivity* series is
    obtained — the part that is a design variable and that staircasing costs accuracy on. The
    normal field is static geometry (see :func:`normal_vector_field`): it is a smooth
    extension of the interface direction into the bulk, so it has no piecewise-constant block
    structure for the closed-form series to integrate over in the first place.
    """
    C_xx = dfs2d(xp, xp.asarray(nx ** 2, dtype=dtype, device=device), 1, 1, fto_x, fto_y)
    C_xy = dfs2d(xp, xp.asarray(nx * ny, dtype=dtype, device=device), 1, 1, fto_x, fto_y)
    C_yy = dfs2d(xp, xp.asarray(ny ** 2, dtype=dtype, device=device), 1, 1, fto_x, fto_y)
    return C_xx, C_xy, C_yy


def _nvf_assemble(xp, ep_dir, ep_inv, C_xx, C_xy, C_yy):
    """Close the NVF factorization from the direct/inverse-rule matrices and the projector.

    With ``E = <eps>`` (direct rule), ``A = <1/eps>^-1`` (inverse rule) and ``Delta = E - A``::

        eps_xx = E - 1/2 (Delta @ C_xx + C_xx @ Delta)
        eps_yy = E - 1/2 (Delta @ C_yy + C_yy @ Delta)
        eps_xy = eps_yx = -1/2 (Delta @ C_xy + C_xy @ Delta)

    The 1/2-symmetrization is essential: the plain product ``Delta @ C`` is not Hermitian,
    giving complex ``q^2`` and gross energy-conservation violation on curved boundaries.

    ``ep_dir`` / ``ep_inv`` are the only inputs that depend on how the coefficients are
    obtained, which is what lets the sampled and integrated builders share this step.

    Returns the five matrices in the slot convention the 2D solve expects:
    ``(epx, epy, epxy, epyx, epz_i) = (eps_xx, eps_yy, -eps_xy, -eps_yx, inv(<eps>))``.
    """
    delta_ep = ep_dir - xp.linalg.inv(ep_inv)

    eps_xx = ep_dir - 0.5 * (delta_ep @ C_xx + C_xx @ delta_ep)
    eps_yy = ep_dir - 0.5 * (delta_ep @ C_yy + C_yy @ delta_ep)
    eps_xy = -0.5 * (delta_ep @ C_xy + C_xy @ delta_ep)      # == eps_yx (symmetric)

    return eps_xx, eps_yy, -eps_xy, -eps_xy, xp.linalg.inv(ep_dir)


def nvf_conv_matrices(xp, eps_matrix, nx, ny, fto_x, fto_y):
    """NVF convolution matrices for one layer, with sampled (DFS) coefficients."""
    C = _complex_dtype(xp, eps_matrix)
    eps_matrix = xp.astype(eps_matrix, C)
    dev = _device(xp, eps_matrix)

    # eps is a staircase however finely it was sampled, so its coefficients are exact from
    # the grid it arrives on. The projector grids are not -- they sample a continuous
    # vector field -- which is why only these two carry the flag.
    ep_dir = dfs2d(xp, eps_matrix, 1, 1, fto_x, fto_y, piecewise_constant=True)
    ep_inv = dfs2d(xp, 1.0 / eps_matrix, 1, 1, fto_x, fto_y, piecewise_constant=True)

    return _nvf_assemble(xp, ep_dir, ep_inv,
                         *_nvf_projector(xp, nx, ny, fto_x, fto_y, C, dev))


def nvf_conv_matrices_continuous(xp, eps_comp, x, y, nx, ny, fto_x, fto_y):
    """NVF convolution matrices for one layer, with integrated (closed-form) coefficients.

    ``eps_comp`` is the *compressed* piecewise-constant layer and ``x`` / ``y`` its segment
    edges, exactly as :func:`cfs2d` takes them; ``nx`` / ``ny`` are the normal field on the
    uncompressed raster (see :func:`_nvf_projector` for why the two grids differ).
    """
    C = _complex_dtype(xp, eps_comp)
    eps_comp = xp.astype(eps_comp, C)
    dev = _device(xp, eps_comp)

    ep_dir = cfs2d(xp, eps_comp, x, y, 1, 1, fto_x, fto_y)
    ep_inv = cfs2d(xp, 1.0 / eps_comp, x, y, 1, 1, fto_x, fto_y)

    return _nvf_assemble(xp, ep_dir, ep_inv,
                         *_nvf_projector(xp, nx, ny, fto_x, fto_y, C, dev))


def to_conv_mat_raster_nvf(ucell, fto_x, fto_y, normal_field=None,
                           normal_field_source=None, lattice=None):
    """Per-layer NVF convolution matrices for a raster ``ucell``.

    Args:
        ucell: refractive-index raster ``(n_layers, Ny, Nx)`` — the design variable.
        fto_x, fto_y: Fourier truncation orders.
        normal_field: optional per-layer ``(nx, ny)`` pairs for the *upsampled* layer grid.
            Supply this to keep the build traceable under JAX (the default computes the field
            with SciPy, which needs concrete values). It is static geometry either way.
        normal_field_source: what that field was sampled from, from
            :func:`normal_field_source_for_ucell`. Supply it and an eager replay is checked
            against the layer in front of it (see :func:`_check_normal_field_still_fits`);
            leave it ``None`` and the field is replayed on trust.

    The normal field is taken on the raster as given, lifted only to the floor its transform
    needs (:func:`_normal_field_floor`). Lifting it further used to be offered and is not: the
    grid is a sampling of a guess at the boundary behind the staircase, and refining it makes
    the guess follow the staircase more faithfully rather than the shape — measured worse in
    27 of 32 combinations (``docs/nvf-upsampling.md``).

    Returns:
        ``(epx, epy, epxy, epyx, epz_i)``, each ``(n_layers, ff_xy, ff_xy)``.
    """
    xp = array_namespace(ucell)

    for name, record in (('normal field', normal_field),
                         ('normal-field source', normal_field_source)):
        if record is not None and len(record) != ucell.shape[0]:
            raise StaleGeometryError(
                f'the frozen geometry no longer fits this structure: it holds a {name} for '
                f'{len(record)} layers and this one has {ucell.shape[0]}. Call '
                f'freeze_geometry() again, or drop geometry= to rebuild by comparison.')

    out = [[], [], [], [], []]

    layers = _layer_eps_for_normal_field(xp, ucell, fto_x, fto_y)
    for i, (eps_matrix, base_eps, repeat_y, repeat_x) in enumerate(layers):
        if normal_field is None:
            nx, ny = normal_vector_field(
                eps_matrix, _interface_mask_of(xp, eps_matrix, base_eps, repeat_y, repeat_x),
                lattice=lattice)
        else:
            _check_normal_field_still_fits(eps_matrix, normal_field_source, i, lattice)
            nx, ny = normal_field[i]

        for slot, mat in zip(out, nvf_conv_matrices(
                xp, eps_matrix, nx, ny, fto_x, fto_y)):
            slot.append(mat)

    return tuple(xp.stack(slot) for slot in out)


# ================================================ continuous Fourier series (CFS / analytic)

def _lift(xp, v, dtype, device=None):
    """``v`` on the working backend at ``dtype``, keeping any autograd graph intact.

    ``asarray(tensor, dtype=...)`` detaches under PyTorch, so a value that *is* a design
    variable would silently stop carrying gradient. Only values that are not already arrays of
    this namespace get created from scratch.
    """
    if hasattr(v, 'dtype') and hasattr(v, 'shape'):
        try:
            same = array_namespace(v) is xp
        except Exception:
            same = False
        if same:
            return v if v.dtype == dtype else xp.astype(v, dtype)
    return xp.asarray(v, dtype=dtype, device=device)


def _cfs(xp, x, cell, fto, period):
    """1D continuous Fourier series of a piecewise-constant profile.

    ``x`` holds the right-hand edge of each constant segment (shape ``(n, 1)``) and ``cell``
    the segment values (``(m, n)``). Port of the legacy ``_cfs``, with its two in-place
    assignments rewritten: the non-DC modes divide by ``1j*2*pi*m`` and the DC mode is the
    length-weighted mean, so both are expressed as one ``where`` over a mode mask.
    """
    C = cell.dtype
    dev = _device(xp, cell)

    cell_next = xp.roll(cell, -1, axis=1)
    cell_diff = cell_next - cell

    n_modes = 4 * fto + 1
    modes = xp.asarray([float(m) for m in range(-2 * fto, 2 * fto + 1)], dtype=C, device=dev)

    # ``x`` holds the segment edges, which for a vector-modelled layer *are* the design
    # variables an OCD fit differentiates -- CD, sidewall angle, pitch all live here. Lift
    # them without detaching; ``asarray(..., dtype=)`` would cut the graph under PyTorch.
    period = _lift(xp, period, C, dev)
    x = _lift(xp, x, C, dev)

    # Segment widths: distance from each edge to the next (last wraps to the period).
    x_next = xp.concat([xp.roll(x, -1, axis=0)[:-1], xp.reshape(period, (1, 1))], axis=0) - x

    # Scalars fold into `x` *before* the matmul; hoisting them out changes the rounding and
    # costs bit-parity with the legacy builder.
    phase = (-1j * 2 * 3.141592653589793 * x) @ xp.reshape(modes, (1, n_modes))
    f = cell_diff @ xp.exp(phase / period)

    # Divisor for the non-DC modes; 1 at DC so the unused branch of `where` stays finite
    # (a NaN there would still poison the gradient).
    is_dc = xp.asarray([m == 2 * fto for m in range(n_modes)], device=dev)
    divisor = xp.where(is_dc, xp.asarray(1.0, dtype=C, device=dev),
                       1j * 2 * 3.141592653589793 * modes)

    dc = xp.reshape((cell @ xp.concat([x[0:1], x_next[:-1]], axis=0)) / period, (-1, 1))

    return xp.where(xp.reshape(is_dc, (1, n_modes)), dc, f / xp.reshape(divisor, (1, n_modes)))


def cfs2d(xp, cell, x, y, conti_x, conti_y, fto_x, fto_y):
    """2D continuous Fourier series with Li's rules — the staircase-free factorization.

    ``x`` / ``y`` are the segment edges of the compressed (piecewise-constant) layer, so the
    permittivity is Fourier-transformed analytically rather than off a raster.
    """
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    period_x, period_y = x[-1], y[-1]

    cell = xp.matrix_transpose(cell)

    if conti_y == 0:      # discontinuous in Y (row): inverse rule
        cell = 1 / cell

    cfs1d = _cfs(xp, y, cell, fto_y, period_y)

    dev = _device(xp, cell)
    idx_y = xp.asarray([[v + 2 * fto_y for v in row] for row in _circulant_index(fto_y)],
                       device=dev)
    idx_x = xp.asarray([[v + 2 * fto_x for v in row] for row in _circulant_index(fto_x)],
                       device=dev)

    conv1d = cfs1d[:, idx_y]

    if conti_x ^ conti_y:
        conv1d = xp.linalg.inv(conv1d)

    conv1d = xp.reshape(conv1d, (-1, ff_y ** 2))

    cfs_2d = _cfs(xp, x, xp.matrix_transpose(conv1d), fto_x, period_x)

    conv2d = cfs_2d[:, idx_x]
    conv2d = xp.reshape(conv2d, (ff_y, ff_y, ff_x, ff_x))
    conv2d = xp.moveaxis(conv2d, 1, 2)
    conv2d = xp.reshape(conv2d, (ff_y * ff_x, ff_y * ff_x))

    if conti_x == 0:      # discontinuous in X (column): inverse rule
        conv2d = xp.linalg.inv(conv2d)

    return conv2d


def _as_bools(xp, flags):
    """``flags`` as a NumPy bool array of the same shape, narrowed however the backend allows.

    Whole-array in one step when the values are concrete, and element by element when that
    fails. The second path is not a fallback for tidiness: under ``jax.grad`` a tracer refuses
    a whole-array NumPy conversion, but each element still narrows, because a grad tracer
    carries a concrete primal. Under ``jax.jit`` neither works — there is no primal to read,
    which is exactly what a frozen geometry is for.
    """
    import numpy as np
    try:
        return np.asarray(_to_numpy(flags), dtype=bool)
    except Exception:                     # a grad tracer: narrow one value at a time
        flat = [bool(v) for v in xp.reshape(flags, (-1,))]
        return np.asarray(flat, dtype=bool).reshape(tuple(flags.shape))


def compression_indices(cell):
    """Locate the piecewise-constant blocks of a raster layer — **static geometry**.

    Returns ``(row_idx, col_idx, x, y)``: the rows/columns that end a constant block, plus
    their normalized right-hand edges. The decision compares neighbouring rows/columns *by
    value* and its output shape depends on the answer, so it cannot run under ``jax.jit`` —
    that is what :meth:`~fieldwave._core.engine.CoreFactorizationMixin.freeze_geometry`
    records. It *can* run inside ``jax.grad``: the comparisons are narrowed through
    :func:`_as_bools`, which a grad tracer permits. So a gradient over a raster whose pattern
    moves every step — a binary topology optimization, say — needs no frozen geometry and
    therefore cannot be served a stale one. The gather this implies (:func:`compress_cell`)
    stays differentiable either way.

    Worked in the input's own namespace rather than through NumPy for that reason alone; a
    concrete array still takes the whole-array path in :func:`_as_bools` and costs the same.
    """
    import numpy as np

    xp = array_namespace(cell)
    n_row, n_col = cell.shape

    same_col = _as_bools(xp, xp.all(cell == xp.roll(cell, -1, axis=1), axis=0))
    col_idx = [c for c in range(n_col) if not same_col[c] or c == n_col - 1]

    cell_x = xp.take(cell, xp.asarray(col_idx, device=_device(xp, cell)), axis=1)
    same_row = _as_bools(xp, xp.all(cell_x == xp.roll(cell_x, -1, axis=0), axis=1))
    row_idx = [r for r in range(n_row) if not same_row[r] or r == n_row - 1]

    step_y, step_x = 1.0 / np.array((n_row, n_col), dtype=np.float64)
    x = np.array([step_x * (c + 1) for c in col_idx]).reshape((-1, 1))
    y = np.array([step_y * (r + 1) for r in row_idx]).reshape((-1, 1))
    return row_idx, col_idx, x, y


def _steps_along(cell, axis):
    """Indices along ``axis`` where ``cell`` differs from its successor — where it must be cut.

    Cyclic, matching :func:`compression_indices`: the last index compares against the first,
    and is a required cut regardless since it closes the period.
    """
    import numpy as np
    nxt = np.roll(cell, -1, axis=axis)
    same = (cell == nxt).all(axis=1 - axis)
    return {int(i) for i in np.flatnonzero(~same)} | {cell.shape[axis] - 1}


def _check_compression_still_fits(cell, record, index):
    """Refuse a frozen compression whose cut no longer separates the layer's blocks.

    The record says which rows and columns end a constant block, *and* where those blocks'
    edges sit. Move the pattern in the raster and the gather samples the new layer at the old
    indices, then pins those samples to the old edges — building a layer that is neither the
    frozen one nor the true one. Whole ranges of structures land on the same wrong profile: a
    40x40 rod frozen at 12 cells wide and replayed at 13, 16, 20 and 24 samples to the same
    three values every time, so it returns the same total every time (0.591) while the true
    answers run 0.733 down to 0.496 — and the frozen structure's own answer is 0.490, which is
    none of them. Per-order efficiencies land up to 0.24 off.

    What matters is not that the record is the one this layer would produce, but that it is
    still *a* description of it: every place the layer steps has to be cut. A cut with a
    redundant line splits one constant block into two of the same value, which is the same
    profile written differently. That distinction is not pedantic — requiring the record to
    match exactly rejected 19 of 300 randomised replays that a rebuild agreed with to
    0.0e+00, mostly material sweeps that drove two neighbouring blocks to the same index, which
    is the sweep freezing exists to serve. Requiring only that the cut still separates the
    blocks rejected none of them, and admitted nothing worse than a reordered summation:
    3.6e-15 across those 300 replays, 7.3e-15 on the transmitted efficiencies themselves.

    Only reachable eagerly, as for :func:`_check_normal_field_still_fits`.
    """
    import numpy as np
    try:
        cell = _to_numpy(cell)
    except Exception:                     # a tracer: nothing to compare, nothing to do
        return

    row_idx, col_idx, x, y = record
    n_row, n_col = cell.shape

    def refuse(detail):
        raise StaleGeometryError(
            f'the frozen geometry no longer fits this structure: layer {index} {detail}. The '
            f'recorded compression would gather blocks that are not the ones this layer has, '
            f'so it describes a different layer. Call freeze_geometry() again, or drop '
            f'geometry= to rebuild by comparison. (Changing the refractive index of a layer '
            f'never trips this on its own; moving its pattern does.)')

    if not len(col_idx) or not len(row_idx) or col_idx[-1] != n_col - 1 \
            or row_idx[-1] != n_row - 1:
        refuse(f'is {cell.shape}, which the recorded cut does not span')

    missing = sorted(_steps_along(cell, 1) - set(int(c) for c in col_idx))
    if missing:
        refuse(f'now steps at column {missing[0]}, where the recorded cut does not divide it')

    cell_x = cell[:, list(col_idx)]
    missing = sorted(_steps_along(cell_x, 0) - set(int(r) for r in row_idx))
    if missing:
        refuse(f'now steps at row {missing[0]}, where the recorded cut does not divide it')

    step_y, step_x = 1.0 / np.array(cell.shape, dtype=np.float64)
    edges = (np.asarray(x), np.asarray(y))
    want = (np.array([step_x * (c + 1) for c in col_idx]).reshape((-1, 1)),
            np.array([step_y * (r + 1) for r in row_idx]).reshape((-1, 1)))
    if not all(np.array_equal(a, b) for a, b in zip(edges, want)):
        refuse('carries block edges that are not the ones its shape implies')


def compress_cell(xp, cell, row_idx, col_idx):
    """Gather the compressed layer selected by :func:`compression_indices` (differentiable)."""
    dev = _device(xp, cell)
    cell = xp.take(cell, xp.asarray(col_idx, device=dev), axis=1)
    return xp.take(cell, xp.asarray(row_idx, device=dev), axis=0)


def to_conv_mat_raster_continuous(ucell, fto_x, fto_y, compression=None):
    """Per-layer convolution matrices via the analytic (staircase-free) CFS path.

    Args:
        ucell: refractive-index raster ``(n_layers, Ny, Nx)`` — the design variable.
        fto_x, fto_y: Fourier truncation orders.
        compression: optional per-layer ``(row_idx, col_idx, x, y)`` from
            :func:`compression_indices`. Supply it to stay traceable under JAX; it is static
            geometry either way. Replayed eagerly it is checked against the layer in front of
            it (see :func:`_check_compression_still_fits`).

    Returns:
        ``(epx_conv_all, epy_conv_all, epz_conv_i_all)``, each ``(n_layers, ff_xy, ff_xy)``.
    """
    xp = array_namespace(ucell)

    if compression is not None and len(compression) != ucell.shape[0]:
        raise StaleGeometryError(
            f'the frozen geometry no longer fits this structure: it holds a compression for '
            f'{len(compression)} layers and this one has {ucell.shape[0]}. Call '
            f'freeze_geometry() again, or drop geometry= to rebuild by comparison.')

    epx_list, epy_list, epz_i_list = [], [], []

    for i in range(ucell.shape[0]):
        layer = ucell[i]
        if compression is None:
            row_idx, col_idx, x, y = compression_indices(layer)
        else:
            _check_compression_still_fits(layer, compression[i], i)
            row_idx, col_idx, x, y = compression[i]
        n_comp = compress_cell(xp, layer, row_idx, col_idx)
        eps = xp.astype(n_comp, _complex_dtype(xp, n_comp)) ** 2

        epz_list = cfs2d(xp, eps, x, y, 1, 1, fto_x, fto_y)
        epx_list.append(cfs2d(xp, eps, x, y, 0, 1, fto_x, fto_y))
        epy_list.append(cfs2d(xp, eps, x, y, 1, 0, fto_x, fto_y))
        epz_i_list.append(xp.linalg.inv(epz_list))

    return xp.stack(epx_list), xp.stack(epy_list), xp.stack(epz_i_list)


def to_conv_mat_raster_continuous_nvf(ucell, fto_x, fto_y, normal_field=None,
                                      normal_field_source=None, compression=None,
                                      lattice=None):
    """Per-layer convolution matrices for a raster ``ucell`` via closed-form Fourier + NVF.

    The combination the deprecated ``fourier_type`` integer could not express. It buys the
    staircase-free permittivity series of the integrated path *and* the vector factorization
    that handles boundaries not aligned with x or y — which is what a shape/OCD fit of a
    tilted sidewall wants from a raster.

    Args:
        ucell: refractive-index raster ``(n_layers, Ny, Nx)`` — the design variable.
        fto_x, fto_y: Fourier truncation orders.
        normal_field: optional per-layer ``(nx, ny)`` from :func:`normal_field_for_ucell`.
        normal_field_source: what that field was sampled from, as in
            :func:`to_conv_mat_raster_nvf` — this path replays the same record and is checked
            the same way.
        compression: optional per-layer ``(row_idx, col_idx, x, y)`` from
            :func:`compression_indices`. Replayed eagerly it is checked too.
        lattice: the cell basis, as in :func:`to_conv_mat_raster_nvf`.

    Both ``normal_field`` and ``compression`` are static geometry; supply both to keep the
    build traceable under JAX (this path is the only one that needs both).

    ``lattice`` is carried here even though the solver currently refuses ``('integrated',
    'nvf')`` on a skewed cell, because the basis is part of the frozen record's staleness key
    (:func:`_normal_field_source`). A builder that replays that record without being able to
    see the basis would reject a field frozen *with* one — so the parameter is what keeps this
    path consistent with the record it consumes, not a second implementation of the skew.

    Returns:
        ``(epx, epy, epxy, epyx, epz_i)``, each ``(n_layers, ff_xy, ff_xy)``.
    """
    xp = array_namespace(ucell)

    out = [[], [], [], [], []]
    eps_grids = _layer_eps_for_normal_field(xp, ucell, fto_x, fto_y)

    for i, (eps_grid, base_eps, repeat_y, repeat_x) in enumerate(eps_grids):
        if normal_field is None:
            nx, ny = normal_vector_field(
                eps_grid, _interface_mask_of(xp, eps_grid, base_eps, repeat_y, repeat_x),
                lattice=lattice)
        else:
            _check_normal_field_still_fits(eps_grid, normal_field_source, i, lattice)
            nx, ny = normal_field[i]

        layer = ucell[i]
        if compression is None:
            row_idx, col_idx, x, y = compression_indices(layer)
        else:
            _check_compression_still_fits(layer, compression[i], i)
            row_idx, col_idx, x, y = compression[i]
        n_comp = compress_cell(xp, layer, row_idx, col_idx)
        eps_comp = xp.astype(n_comp, _complex_dtype(xp, n_comp)) ** 2

        for slot, mat in zip(out, nvf_conv_matrices_continuous(
                xp, eps_comp, x, y, nx, ny, fto_x, fto_y)):
            slot.append(mat)

    return tuple(xp.stack(slot) for slot in out)


def to_conv_mat_block(ucell_info_list, fto_x, fto_y):
    """Per-layer convolution matrices for vector-modelled layers (CFS on explicit segments).

    Each entry of ``ucell_info_list`` is ``(layer, x_list, y_list)`` — the segment values and
    their edges — so no compression step is needed.
    """
    xp = array_namespace(ucell_info_list[0][0])

    epx_list, epy_list, epz_i_list = [], [], []

    for layer, x, y in ucell_info_list:
        eps = xp.astype(layer, _complex_dtype(xp, layer)) ** 2
        epx_list.append(cfs2d(xp, eps, x, y, 0, 1, fto_x, fto_y))
        epy_list.append(cfs2d(xp, eps, x, y, 1, 0, fto_x, fto_y))
        epz_i_list.append(xp.linalg.inv(cfs2d(xp, eps, x, y, 1, 1, fto_x, fto_y)))

    return xp.stack(epx_list), xp.stack(epy_list), xp.stack(epz_i_list)
