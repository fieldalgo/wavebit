"""Single-source Fourier factorization — Phase 4a of the rewrite (see ``<a scratch note outside this repository>``).

Builds the per-layer convolution matrices (``epx`` / ``epy`` / ``inv(epz)``) that the solve
core consumes, written once against the Array-API namespace instead of once per backend.

Why this matters beyond de-duplication: until the factorization is inside the differentiable
chain, autodiff can only reach the *convolution matrices*, not the permittivity that produced
them. With this module the gradient runs end-to-end from the design variable — the ``ucell``
raster — through to the diffraction efficiency, which is what an inverse-design user actually
differentiates.

All the strategies live here:

===================  ==================  =========================================
``fourier_type``     builder             method
===================  ==================  =========================================
0 (``DFS``)          raster_discrete     discrete Fourier series, Li's rules
1 (``CONTINUOUS``)   raster_continuous   analytic (staircase-free) Fourier of the
                                         compressed piecewise-constant layer
1 (vector)           vector              same, on explicitly given segment edges
2 (``NVF``)          raster_nvf          normal-vector-field factorization
===================  ==================  =========================================

A recurring rule runs through this module, the same one the solve core follows for its
same-material short-circuit: **discrete geometry decisions are computed eagerly and passed in
as static inputs, while material values flow through differentiably.** Two decisions here are
geometry, not material — the NVF normal field (where the interfaces point) and the CFS block
compression (where the piecewise-constant blocks start and stop). Both branch on values and,
in the compression's case, produce a data-dependent shape, so neither can run under a JAX
trace. Each is therefore split into an eager step that yields a static description
(:func:`normal_field_for_ucell`, :func:`compression_indices`) and a differentiable step that
consumes it. Pass those precomputed descriptions to stay traceable; the permittivity riding on
top of them is what gradients see.
"""
from .backend import array_namespace


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


def dfs2d(xp, cell, conti_x, conti_y, fto_x, fto_y):
    """2D discrete Fourier series of ``cell`` with Li's factorization rules.

    ``conti_x`` / ``conti_y`` flag the direction as continuous (1) or discontinuous (0); a
    discontinuous direction takes the inverse rule. Port of the legacy ``dfs2d``.
    """
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    cell = xp.matrix_transpose(cell)

    if conti_y == 0:      # discontinuous in Y (row): inverse rule
        cell = 1 / cell

    dfs1d = xp.fft.fft(cell / cell.shape[1])

    idx_y = xp.asarray(_circulant_index(fto_y), device=_device(xp, cell))
    idx_x = xp.asarray(_circulant_index(fto_x), device=_device(xp, cell))

    conv1d = dfs1d[:, idx_y]

    if conti_x ^ conti_y:
        conv1d = xp.linalg.inv(conv1d)

    conv1d = xp.reshape(conv1d, (-1, ff_y ** 2))

    conv1d_t = xp.matrix_transpose(conv1d)
    dfs_2d = xp.fft.fft(conv1d_t / conv1d_t.shape[1])

    conv2d = dfs_2d[:, idx_x]
    conv2d = xp.reshape(conv2d, (ff_y, ff_y, ff_x, ff_x))
    conv2d = xp.moveaxis(conv2d, 1, 2)
    conv2d = xp.reshape(conv2d, (ff_y * ff_x, ff_y * ff_x))

    if conti_x == 0:      # discontinuous in X (column): inverse rule
        conv2d = xp.linalg.inv(conv2d)

    return conv2d


def _device(xp, x):
    """Device of ``x`` (Array-API ``device`` attribute), for placing index arrays alongside."""
    return x.device if hasattr(x, 'device') else None


def _layer_eps(xp, ucell, fto_x, fto_y, enhanced_dfs):
    """Yield each layer's permittivity grid, upsampled as the DFS/NVF builders require.

    Upsampling factors depend only on shapes, so this stays static under tracing. The
    ``n**2`` maps refractive index to permittivity: FieldWave works in ``e^{-iwt}``
    throughout (``Im(n) > 0`` = loss), with no conjugation on the way in.
    """
    if enhanced_dfs:
        min_y = (4 * fto_y + 1) * ucell.shape[1]
        min_x = (4 * fto_x + 1) * ucell.shape[2]
    else:
        min_y = 4 * fto_y + 1
        min_x = 4 * fto_x + 1

    for i in range(ucell.shape[0]):
        layer = ucell[i]
        if layer.shape[0] < min_y:
            layer = xp.repeat(layer, min_y // layer.shape[0] + 1, axis=0)
        if layer.shape[1] < min_x:
            layer = xp.repeat(layer, min_x // layer.shape[1] + 1, axis=1)
        yield xp.astype(layer, _complex_dtype(xp, layer)) ** 2


def to_conv_mat_raster_discrete(ucell, fto_x, fto_y, enhanced_dfs=True, use_pinv=False):
    """Per-layer convolution matrices for a raster ``ucell`` via the DFS path.

    Args:
        ucell: refractive-index raster, shape ``(n_layers, Ny, Nx)``. This is the design
            variable — gradients flow back to it.
        fto_x, fto_y: Fourier truncation orders.
        enhanced_dfs: upsample each layer so the raster resolves ``4*fto+1`` harmonics per
            cell (the legacy default; reduces staircase aliasing).
        use_pinv: use the pseudo-inverse for the ``epz`` inversion.

    Returns:
        ``(epx_conv_all, epy_conv_all, epz_conv_i_all)``, each ``(n_layers, ff_xy, ff_xy)``.
    """
    xp = array_namespace(ucell)

    epx_list, epy_list, epz_i_list = [], [], []

    for eps_matrix in _layer_eps(xp, ucell, fto_x, fto_y, enhanced_dfs):
        epz_conv = dfs2d(xp, eps_matrix, 1, 1, fto_x, fto_y)
        epy_conv = dfs2d(xp, eps_matrix, 1, 0, fto_x, fto_y)
        epx_conv = dfs2d(xp, eps_matrix, 0, 1, fto_x, fto_y)

        epx_list.append(epx_conv)
        epy_list.append(epy_conv)
        epz_i_list.append(xp.linalg.pinv(epz_conv) if use_pinv else xp.linalg.inv(epz_conv))

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


def normal_vector_field(eps_matrix):
    """Unit normal vector field ``(nx, ny)`` of a raster layer, via a signed-distance field.

    This is a **geometry** quantity — it depends on where the material interfaces sit, not on
    the permittivity values carried between them — so it is computed eagerly in NumPy/SciPy
    and then held fixed while gradients flow through the permittivity. Requires a concrete
    array; under a JAX trace, pass a precomputed field instead (see
    :func:`to_conv_mat_raster_nvf`).

    The signed-distance field is the only raster method offered, deliberately. The
    factorization needs a normal field that is unit magnitude across the *whole* cell — that
    is what makes ``C_xx + C_yy = I`` in :func:`nvf_conv_matrices`. The distance transform
    extends the interface normal into the bulk and delivers exactly that; the legacy
    ``method='sobel'`` alternative (a raw ``grad(eps)``) is nonzero only in the one-pixel band
    at the interface, so it is not a unit field and collapses the projector toward 0, which
    degenerates the scheme back toward the plain direct rule. It was never reachable from the
    solver and was dropped in the port. See ``tests_verification/test_nvf_normal_field.py``.
    For the shape-based (non-raster) normal field, see
    :func:`fieldwave._core.analytic.normal_vector_field_analytical`.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    eps = _to_numpy(eps_matrix)

    # Material interfaces: cells differing from any 4-neighbour.
    border = np.zeros(eps.shape, dtype=bool)
    border[1:, :] |= (eps[1:, :] != eps[:-1, :])
    border[:-1, :] |= (eps[:-1, :] != eps[1:, :])
    border[:, 1:] |= (eps[:, 1:] != eps[:, :-1])
    border[:, :-1] |= (eps[:, :-1] != eps[:, 1:])

    # A one-cell axis carries no interface of its own: the structure cannot vary along it, so
    # every normal points along the other axis. Say so directly -- `np.gradient` needs three
    # cells and would otherwise raise, which is what made NVF unusable on 1D gratings. With a
    # constant normal the NVF projector reduces to Li's rules exactly (C_xx = I, C_xy = C_yy =
    # 0 gives eps_xx = <1/eps>^-1 and eps_yy = <eps>), so this is not an approximation.
    if 1 in eps.shape:
        along_x = np.ones(eps.shape), np.zeros(eps.shape)
        return along_x if eps.shape[0] == 1 else (along_x[1], along_x[0])

    dist_map = distance_transform_edt(~border)
    gy, gx = np.gradient(dist_map)
    norm = np.hypot(gx, gy)
    norm[norm == 0] = 1.0
    return gx / norm, gy / norm


def normal_field_for_ucell(ucell, fto_x, fto_y, enhanced_dfs=True):
    """Per-layer ``(nx, ny)`` normal fields for ``ucell``, on the grid the NVF build uses.

    Precompute this once from the concrete geometry and hand it to
    :func:`to_conv_mat_raster_nvf` as ``normal_field=`` to keep the build traceable under a
    JAX transform (the SciPy distance transform cannot run on tracers).
    """
    xp = array_namespace(ucell)
    return [normal_vector_field(eps)
            for eps in _layer_eps(xp, ucell, fto_x, fto_y, enhanced_dfs)]


def nvf_conv_matrices(xp, eps_matrix, nx, ny, fto_x, fto_y, use_pinv=False):
    """NVF (Li/Schuster normal-vector) convolution matrices for one layer.

    With ``E = <eps>`` (direct rule), ``A = <1/eps>^-1`` (inverse rule), ``Delta = E - A`` and
    the normal projector blocks ``C_xx = <nx^2>``, ``C_xy = <nx ny>``, ``C_yy = <ny^2>``::

        eps_xx = E - 1/2 (Delta @ C_xx + C_xx @ Delta)
        eps_yy = E - 1/2 (Delta @ C_yy + C_yy @ Delta)
        eps_xy = eps_yx = -1/2 (Delta @ C_xy + C_xy @ Delta)

    The 1/2-symmetrization is essential: the plain product ``Delta @ C`` is not Hermitian,
    giving complex ``q^2`` and gross energy-conservation violation on curved boundaries.

    Returns the five matrices in the slot convention the 2D solve expects:
    ``(epx, epy, epxy, epyx, epz_i) = (eps_xx, eps_yy, -eps_xy, -eps_yx, inv(<eps>))``.
    """
    C = _complex_dtype(xp, eps_matrix)
    eps_matrix = xp.astype(eps_matrix, C)

    dev = _device(xp, eps_matrix)
    C_xx_grid = xp.asarray(nx ** 2, dtype=C, device=dev)
    C_xy_grid = xp.asarray(nx * ny, dtype=C, device=dev)
    C_yy_grid = xp.asarray(ny ** 2, dtype=C, device=dev)

    ep_dir = dfs2d(xp, eps_matrix, 1, 1, fto_x, fto_y)
    ep_inv = dfs2d(xp, 1.0 / eps_matrix, 1, 1, fto_x, fto_y)

    _inv = xp.linalg.pinv if use_pinv else xp.linalg.inv
    delta_ep = ep_dir - _inv(ep_inv)

    C_xx = dfs2d(xp, C_xx_grid, 1, 1, fto_x, fto_y)
    C_xy = dfs2d(xp, C_xy_grid, 1, 1, fto_x, fto_y)
    C_yy = dfs2d(xp, C_yy_grid, 1, 1, fto_x, fto_y)

    eps_xx = ep_dir - 0.5 * (delta_ep @ C_xx + C_xx @ delta_ep)
    eps_yy = ep_dir - 0.5 * (delta_ep @ C_yy + C_yy @ delta_ep)
    eps_xy = -0.5 * (delta_ep @ C_xy + C_xy @ delta_ep)      # == eps_yx (symmetric)

    return eps_xx, eps_yy, -eps_xy, -eps_xy, _inv(ep_dir)


def to_conv_mat_raster_nvf(ucell, fto_x, fto_y, normal_field=None,
                           enhanced_dfs=True, use_pinv=False):
    """Per-layer NVF convolution matrices for a raster ``ucell``.

    Args:
        ucell: refractive-index raster ``(n_layers, Ny, Nx)`` — the design variable.
        fto_x, fto_y: Fourier truncation orders.
        normal_field: optional per-layer ``(nx, ny)`` pairs for the *upsampled* layer grid.
            Supply this to keep the build traceable under JAX (the default computes the field
            with SciPy, which needs concrete values). It is static geometry either way.
        enhanced_dfs, use_pinv: as in :func:`to_conv_mat_raster_discrete`.

    Returns:
        ``(epx, epy, epxy, epyx, epz_i)``, each ``(n_layers, ff_xy, ff_xy)``.
    """
    xp = array_namespace(ucell)

    out = [[], [], [], [], []]

    for i, eps_matrix in enumerate(_layer_eps(xp, ucell, fto_x, fto_y, enhanced_dfs)):
        if normal_field is None:
            nx, ny = normal_vector_field(eps_matrix)
        else:
            nx, ny = normal_field[i]

        for slot, mat in zip(out, nvf_conv_matrices(
                xp, eps_matrix, nx, ny, fto_x, fto_y, use_pinv=use_pinv)):
            slot.append(mat)

    return tuple(xp.stack(slot) for slot in out)


# ================================================ continuous Fourier series (CFS / analytic)

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

    # ``x`` / ``period`` arrive as static geometry (often plain NumPy), so lift them onto the
    # working backend rather than casting in place.
    period = xp.asarray(period, dtype=C, device=dev)
    x = xp.asarray(x, dtype=C, device=dev)

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


def compression_indices(cell):
    """Locate the piecewise-constant blocks of a raster layer — **static geometry**.

    Returns ``(row_idx, col_idx, x, y)``: the rows/columns that end a constant block, plus
    their normalized right-hand edges. The legacy ``cell_compression`` decides this by
    comparing neighbouring rows/columns *by value* and returns a data-dependent shape, so it
    cannot run under a JAX trace. Splitting it out keeps the decision eager while the gather it
    implies (:func:`compress_cell`) stays differentiable — the same split used for the NVF
    normal field.
    """
    import numpy as np

    cell = _to_numpy(cell)
    step_y, step_x = 1.0 / np.array(cell.shape, dtype=np.float64)

    cell_next = np.roll(cell, -1, axis=1)
    col_idx = [c for c in range(cell.shape[1])
               if not (cell[:, c] == cell_next[:, c]).all() or c == cell.shape[1] - 1]

    cell_x = cell[:, col_idx]
    cell_x_next = np.roll(cell_x, -1, axis=0)
    row_idx = [r for r in range(cell_x.shape[0])
               if not (cell_x[r, :] == cell_x_next[r, :]).all() or r == cell_x.shape[0] - 1]

    x = np.array([step_x * (c + 1) for c in col_idx]).reshape((-1, 1))
    y = np.array([step_y * (r + 1) for r in row_idx]).reshape((-1, 1))
    return row_idx, col_idx, x, y


def compress_cell(xp, cell, row_idx, col_idx):
    """Gather the compressed layer selected by :func:`compression_indices` (differentiable)."""
    dev = _device(xp, cell)
    cell = xp.take(cell, xp.asarray(col_idx, device=dev), axis=1)
    return xp.take(cell, xp.asarray(row_idx, device=dev), axis=0)


def to_conv_mat_raster_continuous(ucell, fto_x, fto_y, compression=None, use_pinv=False):
    """Per-layer convolution matrices via the analytic (staircase-free) CFS path.

    Args:
        ucell: refractive-index raster ``(n_layers, Ny, Nx)`` — the design variable.
        fto_x, fto_y: Fourier truncation orders.
        compression: optional per-layer ``(row_idx, col_idx, x, y)`` from
            :func:`compression_indices`. Supply it to stay traceable under JAX; it is static
            geometry either way.
        use_pinv: use the pseudo-inverse for the ``epz`` inversion.

    Returns:
        ``(epx_conv_all, epy_conv_all, epz_conv_i_all)``, each ``(n_layers, ff_xy, ff_xy)``.
    """
    xp = array_namespace(ucell)
    _inv = xp.linalg.pinv if use_pinv else xp.linalg.inv

    epx_list, epy_list, epz_i_list = [], [], []

    for i in range(ucell.shape[0]):
        layer = ucell[i]
        row_idx, col_idx, x, y = (compression_indices(layer) if compression is None
                                  else compression[i])
        n_comp = compress_cell(xp, layer, row_idx, col_idx)
        eps = xp.astype(n_comp, _complex_dtype(xp, n_comp)) ** 2

        epz_list = cfs2d(xp, eps, x, y, 1, 1, fto_x, fto_y)
        epx_list.append(cfs2d(xp, eps, x, y, 0, 1, fto_x, fto_y))
        epy_list.append(cfs2d(xp, eps, x, y, 1, 0, fto_x, fto_y))
        epz_i_list.append(_inv(epz_list))

    return xp.stack(epx_list), xp.stack(epy_list), xp.stack(epz_i_list)


def to_conv_mat_vector(ucell_info_list, fto_x, fto_y, use_pinv=False):
    """Per-layer convolution matrices for vector-modelled layers (CFS on explicit segments).

    Each entry of ``ucell_info_list`` is ``(layer, x_list, y_list)`` — the segment values and
    their edges — so no compression step is needed.
    """
    xp = array_namespace(ucell_info_list[0][0])
    _inv = xp.linalg.pinv if use_pinv else xp.linalg.inv

    epx_list, epy_list, epz_i_list = [], [], []

    for layer, x, y in ucell_info_list:
        eps = xp.astype(layer, _complex_dtype(xp, layer)) ** 2
        epx_list.append(cfs2d(xp, eps, x, y, 0, 1, fto_x, fto_y))
        epy_list.append(cfs2d(xp, eps, x, y, 1, 0, fto_x, fto_y))
        epz_i_list.append(_inv(cfs2d(xp, eps, x, y, 1, 1, fto_x, fto_y)))

    return xp.stack(epx_list), xp.stack(epy_list), xp.stack(epz_i_list)
