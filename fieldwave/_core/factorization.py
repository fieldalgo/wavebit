"""Single-source Fourier factorization — Phase 4a of the rewrite (see ``<a scratch note outside this repository>``).

Builds the per-layer convolution matrices (``epx`` / ``epy`` / ``inv(epz)``) that the solve
core consumes, written once against the Array-API namespace instead of once per backend.

Why this matters beyond de-duplication: until the factorization is inside the differentiable
chain, autodiff can only reach the *convolution matrices*, not the permittivity that produced
them. With this module the gradient runs end-to-end from the design variable — the ``ucell``
raster — through to the diffraction efficiency, which is what an inverse-design user actually
differentiates.

This is the DFS (discrete Fourier series, Li's rules on a raster) path — the default
``fourier_type=0``. The continuous/analytic (CFS) and NVF strategies keep using the legacy
per-backend builders for now and are ported next.
"""
from .backend import array_namespace


def _circulant_index(fto):
    """Circular index matrix selecting Fourier orders ``m - n`` (pure integer, data-free)."""
    ff = 2 * fto + 1
    return [[(-r + c) for c in range(ff)] for r in range(ff)]


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

    if enhanced_dfs:
        min_y = (4 * fto_y + 1) * ucell.shape[1]
        min_x = (4 * fto_x + 1) * ucell.shape[2]
    else:
        min_y = 4 * fto_y + 1
        min_x = 4 * fto_x + 1

    epx_list, epy_list, epz_i_list = [], [], []

    for i in range(ucell.shape[0]):
        layer = ucell[i]

        # Upsampling factors come from shapes only (static), so this stays trace-safe.
        if layer.shape[0] < min_y:
            layer = xp.repeat(layer, min_y // layer.shape[0] + 1, axis=0)
        if layer.shape[1] < min_x:
            layer = xp.repeat(layer, min_x // layer.shape[1] + 1, axis=1)

        # n -> eps under the e^{-iwt} convention (Im(n) > 0 = loss).
        eps_matrix = xp.conj(xp.astype(layer, _complex_dtype(xp, layer))) ** 2

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
