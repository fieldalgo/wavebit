"""Single-source Fourier factorization — Phase 4a of the rewrite.

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

- ``('sampled', 'li')``       -> :func:`to_conv_mat_raster_sampled`, discrete Fourier series
  off the raster with Li's rules. The default, and the only path that carries a gradient to
  every raster cell.
- ``('integrated', 'li')``    -> :func:`to_conv_mat_raster_integrated`, closed-form
  (staircase-free) Fourier of the compressed layer with Li's rules.
- ``('sampled', 'nvf')``      -> :func:`to_conv_mat_raster_nvf`, sampled coefficients with the
  normal-vector-field factorization.
- ``('integrated', 'nvf')``   -> :func:`to_conv_mat_raster_integrated_nvf`, closed-form
  Fourier with NVF.

Vector-modelled layers (:func:`to_conv_mat_block`) are the closed-form series on explicitly
given segment edges, with Li's rules; the analytic *shape* path lives in
:mod:`wavebit._core.analytic` and is always NVF.

A recurring rule runs through this module: **discrete geometry decisions are computed eagerly
and passed in as static inputs, while material values flow through differentiably.** Two
decisions here are geometry, not material — the NVF normal field (where the interfaces point) and the integrated block
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
:func:`_check_compression_still_fits`, and :mod:`wavebit._core.geometry` for the rule they
share. Material values never move a pattern, so the sweeps this split exists to serve pass
untouched. A fit whose *pattern* moves every step is a different matter: on the compression it
should simply drop ``geometry=`` and differentiate, which cannot go stale because it records
nothing.
"""
from .backend import array_namespace
from .geometry import StaleGeometryError


def _inverse_axes(inverse_rule):
    """``inverse_rule`` as ``(along_x, along_y)`` booleans, refusing anything else.

    One three-valued argument rather than the pair of ``conti_x`` / ``conti_y`` integers this
    took until 2026-08-12. Those had four states and only three meanings: ``(0, 0)`` — both
    directions discontinuous — is not part of Li's scheme as used here and was never passed
    anywhere, but nothing said so and nothing refused it. A typo could reach it silently.

    The names also ran backwards. ``conti_y=0`` meant "y is discontinuous, so y takes the
    inverse rule", so the flag that was set said what was *not* true of the axis it named.
    """
    if inverse_rule not in (None, 'x', 'y'):
        raise ValueError(
            f"inverse_rule: expected None (direct rule both ways), 'x' or 'y' — the direction "
            f"whose discontinuity the inverse rule closes; got {inverse_rule!r}")
    return inverse_rule == 'x', inverse_rule == 'y'


def _wrap(orders, n):
    """``orders`` reduced mod ``n`` — the DFT bin each Fourier order lands in."""
    return [k % n for k in orders]


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


def _circulant_gather(xp, fto, dev):
    """Index expanding a ``4*fto+1`` order vector into the ``(ff, ff)`` circulant of ``r - c``.

    The same shape :func:`fourier2d_integrated` builds; ``+ 2*fto`` turns the signed order into a position in
    that vector.
    """
    return xp.asarray([[v + 2 * fto for v in row] for row in _circulant_index(fto)], device=dev)


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
    return sinc * _half_pixel_phase(xp, orders, n, dtype, device)


def _half_pixel_phase(xp, orders, n, dtype, device=None):
    """The cell-centre offset alone, for a grid holding *point samples*.

    Grid index ``j`` names the pixel spanning ``[j/n, (j+1)/n)``, so a value attached to that
    pixel sits at its centre ``(j+0.5)/n`` -- but a bare DFT places sample ``j`` at ``j/n``,
    the left edge. The difference is the half-pixel shift ``exp(-i*pi*k/n)``, and it applies to
    a sampled field exactly as it does to a staircase; what does *not* apply is the ``sinc``,
    which is the box transform of a value held across the pixel and would assert an averaging
    a point sample never had.

    Left out of the pointwise path until 2026-08-16, which put the normal field half a pixel
    off the permittivity it multiplies. Both are read off the same raster, so the offset is a
    *relative* one and no lateral shift of the whole structure can absorb it. It hid because it
    is invisible to every symmetry the suite had: mirror the layer in y and the relative offset
    must change sign, does not, and the two land a full pixel apart -- an O(1/N) error that
    reads as a broken mirror symmetry. Energy conservation, the projector identity
    ``C_xx + C_yy = I``, and the x<->y relabelling all survive it untouched.
    """
    k = xp.astype(xp.asarray(orders, device=device), dtype)
    return xp.exp(-1j * 3.141592653589793 * k / n)


def fourier2d_piecewise_constant(xp, cell, fto_x, fto_y, *, inverse_rule):
    """2D discrete Fourier series of a grid whose values *hold across* their pixels.

    A permittivity raster is this case: ``ucell`` names the material occupying each cell, so
    the layer is a staircase and its Fourier coefficients have an exact closed form, which
    :func:`_pixel_correction` supplies. Exact for any order, including past the point where
    the DFT itself wraps — which is why this path needs no grid floor.

    ``inverse_rule`` names the direction whose discontinuity the inverse rule closes, or
    ``None`` for the direct rule both ways. Those three are Li's three components of one
    layer: ``None`` builds ``epz``, ``'y'`` builds ``epy``, ``'x'`` builds ``epx``.
    """
    return _fourier2d_grid(xp, cell, fto_x, fto_y, inverse_rule=inverse_rule, piecewise_constant=True)


def fourier2d_pointwise(xp, field, fto_x, fto_y):
    """2D discrete Fourier series of a grid holding *point samples* of a continuous field.

    The NVF normal field is this case, and the only one in the package: a continuous
    unit-vector field evaluated at grid points, so refining the grid changes the field itself
    and no exact coefficient is reachable from the samples. The transform is left alone.

    ``pointwise`` rather than ``sampled`` because ``sampled`` is already taken, one level up:
    ``fourier_coefficients='sampled'`` names the whole raster/DFT path — the path in which a
    permittivity goes through :func:`fourier2d_piecewise_constant`, not through here. The pair
    that matters at *this* level is whether a value holds across its pixel or only at the
    point, and that is what these two names now say.

    **No rule argument, deliberately.** The inverse rule exists for factors that are
    discontinuous; a field continuous by construction has nothing to apply it to, and every
    call in the package asked for the direct rule. Offering the choice would only be offering
    a way to be wrong.

    Because there is no closed form, this path *does* need ``4*fto+1`` points — the caller
    lifts the grid to that floor, see :func:`_normal_field_floor`.
    """
    return _fourier2d_grid(xp, field, fto_x, fto_y, inverse_rule=None, piecewise_constant=False)


def _fourier2d_grid(xp, cell, fto_x, fto_y, *, inverse_rule, piecewise_constant):
    """2D discrete Fourier series of ``cell`` with Li's factorization rules.

    Private, and ``piecewise_constant`` has no default, because the two answers are not a
    preference: they say what kind of data the grid holds, and getting it wrong is silent.
    Call :func:`fourier2d_piecewise_constant` or :func:`fourier2d_pointwise` instead, which say it in
    the name.

    ``inverse_rule`` is the direction whose discontinuity the inverse rule closes (``'x'``,
    ``'y'``, or ``None`` for neither) — see :func:`_inverse_axes`. Port of the legacy
    ``dfs2d``; the two entry points above are what callers use.

    This is the *only* thing that selects a rule. The other name for the continuous
    case is Laurent's rule, which this codebase mostly calls ``direct`` — see
    ``docs/factorization-terms.md``, which also explains why the matrix built from these
    coefficients is BTTB rather than Toeplitz.

    Applying the rules in 2D means nesting them, and this fixes the order: inner y, outer x.
    That is Li's construction, so ``epy`` takes the inner inverse and ``epx`` the outer one —
    the mid-loop ``inv`` below is not movable to the end. It also means the result is not
    invariant under relabelling x and y: on a transpose-symmetric structure the two
    polarisations disagree by ~6e-4 at ``fto=12`` where NVF holds the symmetry to 1e-13.
    Intrinsic to the scheme rather than a defect here, and untested; the measurement is in
    ``docs/factorization-terms.md`` §6.

    ``piecewise_constant`` says each value of ``cell`` holds across its whole pixel, rather
    than being a sample of something that varies inside it. When that is true the coefficients
    have an exact closed form and this computes it (see :func:`_pixel_correction`); when it is
    false there is no exact value to reach from the grid at all, and the transform is left
    alone.

    A permittivity raster is the first case — ``ucell`` names the material *occupying* each
    cell, and a coarser or finer spelling of the same structure is the same function. A normal
    field is the second: it is a continuous vector field this code evaluates on a grid, so
    refining the grid changes the field itself. Applying the correction there would assert a
    shape the data does not have.

    **The two directions fail very differently, which is why neither is a default.** Flipping
    the projector sites to ``True`` (asserting a staircase the field does not have) moves the
    efficiencies by 1e-5 at normal incidence into an index-matched substrate, and by 2e-4 to
    2e-3 once there is index contrast or conical incidence — between one decade under the
    change from doubling the raster and comparable to it. Omitting it on a permittivity raster
    is the serious one: 1e-4 to 2e-2 per order, growing with ``fto``, and below roughly
    ``N < 2*fto`` the inverse-rule matrix is singular and the solve raises ``LinAlgError``
    outright. Energy conservation is undisturbed either way, so nothing downstream notices.

    (An earlier version of this docstring gave the first figure as ``~7e-7``, "four decades
    under the field's own discretization error". That is reproducible only at normal incidence
    with no index contrast, and is not a bound.)
    """
    inv_x, inv_y = _inverse_axes(inverse_rule)
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    cell = xp.matrix_transpose(cell)

    if inv_y:             # y discontinuous: the inverse rule starts from 1/eps
        cell = 1 / cell

    dfs1d = xp.fft.fft(cell / cell.shape[1])

    dev = _device(xp, cell)
    orders_y = list(range(-2 * fto_y, 2 * fto_y + 1))
    orders_x = list(range(-2 * fto_x, 2 * fto_x + 1))
    n_y = cell.shape[1]

    # Read the coefficients on the *order* vector, correct them there, and only then expand
    # into the circulant -- the shape `fourier2d_integrated` has always had.
    #
    # Two reasons it is this way round. It is cheaper: the circulant repeats each of the
    # 4*fto+1 distinct orders across (2*fto+1)**2 entries, so correcting afterwards evaluates
    # the same sin/exp and repeats the same multiply ~21x over at fto=20. And it is the only
    # order that is *correct* to fold: the correction depends on the order, not on the DFT bin
    # it wraps to, so it cannot be pushed any earlier into `dfs1d` -- on a short grid distinct
    # orders share a bin and need different factors.
    #
    # `_wrap` states that bin rather than leaning on Python's negative indexing. Identical
    # wherever the old form worked, and defined where it did not: raw order indices need a grid
    # of at least 4*fto+1 to stay in range, which is the whole reason the raster used to be
    # upsampled before anything else was considered.
    coeff_y = dfs1d[:, xp.asarray(_wrap(orders_y, n_y), device=dev)]
    # Both kinds of grid carry the half-pixel offset; only a staircase also carries
    # the box transform. See :func:`_half_pixel_phase`.
    _correct = _pixel_correction if piecewise_constant else _half_pixel_phase
    coeff_y = coeff_y * _correct(xp, orders_y, n_y, coeff_y.dtype, dev)
    conv1d = coeff_y[:, _circulant_gather(xp, fto_y, dev)]

    if inv_x or inv_y:    # Li's inner step, whichever direction asked for it
        conv1d = xp.linalg.inv(conv1d)

    conv1d = xp.reshape(conv1d, (-1, ff_y ** 2))

    conv1d_t = xp.matrix_transpose(conv1d)
    n_x = conv1d_t.shape[1]
    dfs_2d = xp.fft.fft(conv1d_t / n_x)

    coeff_x = dfs_2d[:, xp.asarray(_wrap(orders_x, n_x), device=dev)]
    # Both kinds of grid carry the half-pixel offset; only a staircase also carries
    # the box transform. See :func:`_half_pixel_phase`.
    _correct = _pixel_correction if piecewise_constant else _half_pixel_phase
    coeff_x = coeff_x * _correct(xp, orders_x, n_x, coeff_x.dtype, dev)
    conv2d = coeff_x[:, _circulant_gather(xp, fto_x, dev)]
    conv2d = xp.reshape(conv2d, (ff_y, ff_y, ff_x, ff_x))
    conv2d = xp.moveaxis(conv2d, 1, 2)
    conv2d = xp.reshape(conv2d, (ff_y * ff_x, ff_y * ff_x))

    if inv_x:             # x discontinuous: Li's outer step closes it
        conv2d = xp.linalg.inv(conv2d)

    return conv2d


def _device(xp, x):
    """Device of ``x`` (Array-API ``device`` attribute), for placing index arrays alongside."""
    return x.device if hasattr(x, 'device') else None


def _layer_eps(xp, ucell):
    """Yield each layer's permittivity grid, on the raster exactly as given.

    The ``n**2`` maps refractive index to permittivity: Wavebit works in ``e^{-iwt}``
    throughout (``Im(n) > 0`` = loss), with no conjugation on the way in.

    Nothing is resampled. :func:`fourier2d_piecewise_constant` obtains the raster's
    coefficients in closed form (:func:`_pixel_correction`), so a finer grid of the same
    staircase is the same
    function and buys nothing. The NVF path is the one that still needs a floor, and states
    that itself — see :func:`_layer_eps_for_normal_field`.
    """
    for i in range(ucell.shape[0]):
        layer = ucell[i]
        yield xp.astype(layer, _complex_dtype(xp, layer)) ** 2


def _normal_field_floor(shape, fto_x, fto_y):
    """Repeat factors that lift a layer to the grid the normal field's transform needs.

    The normal field is a continuous vector field this code evaluates on a grid, so unlike the
    permittivity it has no closed-form transform and :func:`fourier2d_pointwise` reads it off the grid
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


def to_conv_mat_raster_sampled(ucell, fto_x, fto_y):
    """Per-layer convolution matrices for a raster ``ucell`` via the sampled path.

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
        epz_conv = fourier2d_piecewise_constant(xp, eps_matrix, fto_x, fto_y, inverse_rule=None)
        epy_conv = fourier2d_piecewise_constant(xp, eps_matrix, fto_x, fto_y, inverse_rule='y')
        epx_conv = fourier2d_piecewise_constant(xp, eps_matrix, fto_x, fto_y, inverse_rule='x')

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


def normal_vector_field(eps_matrix, interface_mask=None, lattice=None, method=None,
                        base_shape=None):
    """Unit normal vector field ``(nx, ny)`` of a raster layer.

    This is a **geometry** quantity — it depends on where the material interfaces sit, not on
    the permittivity values carried between them — so it is computed eagerly in NumPy/SciPy
    and then held fixed while gradients flow through the permittivity. That holds for every
    ``method``: the field is a constant to the solve, whatever the backend.

    ``method`` selects the construction (:class:`~wavebit._core.enums.NormalMethod`;
    ``None`` means ``'distance'``, the original). The default body below is ``'distance'``:
    the gradient of the unsigned distance to the interface set. Its directions quantize to
    the {0, 45} degree list integer central differences can spell — ~14 degrees mean in the
    boundary band, flat in N — which is what the other three constructions remove; they
    live in :mod:`wavebit._core.normal_methods`, and ``docs/nvf-normal-methods.md`` carries
    the measurements that rank them (``'contour'`` is the one that lost nowhere).

    ``base_shape`` is the layer's shape before the ``4*fto+1`` floor lift, when the caller
    lifted it. The smoothing constructions state their pixel-unit policies on the base grid
    — a lifted raster is the same staircase spelled with more pixels, so smoothing scales
    stated on the lifted grid would shrink under the lift instead of holding still.
    ``'distance'`` has no scale and ignores it.

    ``interface_mask`` is the one input the ``'distance'`` construction reads (see
    :func:`_interface_mask`). Supplying it lets a caller that can obtain the mask some other
    way skip the whole-array NumPy conversion this would otherwise need, which is what lets the
    NVF paths differentiate without a frozen geometry — see
    :func:`_upsampled_interface_mask`. Left ``None``, the mask is read off ``eps_matrix``,
    which needs concrete values. The other three constructions read the *values* too (their
    indicators need to know which phase is which), so a traced layer without concrete values
    can take them only through ``freeze_geometry()`` — asked to anyway, they raise and say
    so rather than silently answering with ``'distance'``.

    The factorization needs a normal field that is unit magnitude across the *whole* cell —
    that is what makes ``C_xx + C_yy = I`` in :func:`nvf_conv_matrices`. Every method
    delivers that (zeros only on skeleton-type points, which the assembly collapses to the
    direct rule); the legacy ``method='sobel'`` alternative (a raw ``grad(eps)``) is nonzero
    only in the one-pixel band at the interface, so it is not a unit field and collapses the
    projector toward 0, which degenerates the scheme back toward the plain direct rule. It
    was never reachable from the solver and was dropped in the port. See
    ``tests/test_nvf_normal_field.py``.
    For the shape-based (non-raster) normal field, see
    :func:`wavebit._core.analytic.normal_vector_field_analytical`.

    ``lattice`` is the cell basis the raster indexes into. It matters because this is the one
    piece of the NVF construction that is *Cartesian-geometric* rather than index-based: the
    distance transform and ``np.gradient`` both work in index steps, and only a basis whose
    pixels are square in real space makes an index-space normal a Cartesian one. Left ``None``
    the index grid is taken to be isotropic Cartesian, which is what this has always assumed;
    given a basis, distances pick up the per-axis physical step and the gradient is carried
    through ``J^-T`` (:func:`wavebit._core.lattice.jacobian`) before normalising, so the
    direction is exact at the interface for a skewed *or* merely non-square cell.

    That correction is what lets NVF serve a non-orthogonal lattice at all, and it is also why
    NVF rather than Li's nested rule is the factorization that ports there: the tensor this
    feeds (:func:`_nvf_assemble`) is written in Cartesian components, so once the normal is
    Cartesian nothing else in the factorization learns the cell was skewed.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from .enums import NormalMethod, normal_method_of

    shape = tuple(eps_matrix.shape)
    method = NormalMethod.DISTANCE if method is None else normal_method_of(method)

    # A one-cell axis carries no interface of its own: the structure cannot vary along it, so
    # every normal points along the other axis. Say so directly -- `np.gradient` needs three
    # cells and would otherwise raise, which is what made NVF unusable on 1D gratings. With a
    # constant normal the NVF projector reduces to Li's rules exactly (C_xx = I, C_xy = C_yy =
    # 0 gives eps_xx = <1/eps>^-1 and eps_yy = <eps>), so this is not an approximation.
    # Shared across methods: nothing about the constant field depends on the construction.
    if 1 in shape:
        along_x = np.ones(shape), np.zeros(shape)
        return along_x if shape[0] == 1 else (along_x[1], along_x[0])

    if method != NormalMethod.DISTANCE:
        from .normal_methods import DISPATCH
        try:
            eps = _to_numpy(eps_matrix)
        except Exception:
            raise TypeError(
                f"normal_method={method.value!r} reads the layer's values (its indicator "
                f"needs to know which phase is which), and this layer is a tracer without "
                f"concrete ones. Precompute the field eagerly with freeze_geometry() and "
                f"hand it to conv_solve(geometry=...), or use normal_method='distance', "
                f"whose mask-only path a trace can take.") from None
        mask = (_interface_mask(eps) if interface_mask is None
                else np.asarray(interface_mask, dtype=bool))
        return DISPATCH[method.value](eps, mask, lattice, base_shape)

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


def _normal_field_source(eps, lattice=None, method=None):
    """Everything :func:`normal_vector_field` reads off a layer, and nothing else.

    That is the shape, plus — unless an axis is a single cell, where the field is the constant
    along-axis one whatever the pattern — the interface mask, plus the cell basis, plus the
    construction (``normal_method``). Two layers with the same source have the same normal
    field, bit for bit, which is what makes this the tell: it accepts a replay exactly when
    replaying is the same computation as rebuilding.

    The basis belongs here for the same reason the mask does. It is an *input* to the field
    (see :func:`normal_vector_field`), so a field frozen on one cell and replayed on a skewed
    one describes interfaces pointing the wrong way — with nothing to notice, since the NVF
    assembly stays Hermitian and energy still conserves. The method belongs here for the same
    reason again: same mask, different construction, different field — a record frozen under
    ``'contour'`` and replayed on a solver configured ``'distance'`` would be answering with
    a field the configuration never asked for.

    The smoothing constructions read the layer's values only through two
    value-relabeling-invariant facts, and both are captured: the interface mask (their
    indicators mark the not-the-corner-REGION phase, so relabeling the values moves
    nothing — see ``normal_methods._two_phase_indicator``), and the number of distinct
    phases (which decides the delegation to ``'distance'``, and which a merge of two
    non-adjacent phases can change without moving a single interface — hence the fifth
    element). The pre-lift ``base_shape`` is not captured because it follows from the
    shape the record already pins plus the solver's own ``fto``.
    """
    import numpy as np
    from .enums import NormalMethod, normal_method_of
    method = (NormalMethod.DISTANCE if method is None else normal_method_of(method)).value
    if 1 in eps.shape:
        return eps.shape, None, _lattice_key(lattice), method, None
    return (eps.shape, _interface_mask(eps), _lattice_key(lattice), method,
            int(np.unique(eps).size))


def normal_field_for_ucell(ucell, fto_x, fto_y, lattice=None, method=None):
    """Per-layer ``(nx, ny)`` normal fields for ``ucell``, on the grid the NVF build uses.

    Precompute this once from the concrete geometry and hand it to
    :func:`to_conv_mat_raster_nvf` as ``normal_field=`` to keep the build traceable under a
    JAX transform (none of the constructions can run on tracers).

    ``lattice`` is the cell basis and ``method`` the construction; see
    :func:`normal_vector_field` for what each changes.
    """
    xp = array_namespace(ucell)
    return [normal_vector_field(eps, lattice=lattice, method=method,
                                base_shape=tuple(base.shape))
            for eps, base, _, _ in _layer_eps_for_normal_field(xp, ucell, fto_x, fto_y)]


def normal_field_source_for_ucell(ucell, fto_x, fto_y, lattice=None, method=None):
    """What :func:`normal_field_for_ucell` derived its answer from, per layer.

    Freeze this alongside the normal field and hand both back to
    :func:`to_conv_mat_raster_nvf`; the replay compares it against the layer actually in front
    of it and refuses a field that was sampled from a different one — a different pattern, a
    different basis, or a different ``normal_method``.
    """
    xp = array_namespace(ucell)
    return [_normal_field_source(_to_numpy(eps), lattice, method)
            for eps, _, _, _ in _layer_eps_for_normal_field(xp, ucell, fto_x, fto_y)]


def _check_normal_field_still_fits(eps_matrix, source, index, lattice=None, method=None):
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
    meth = entry[3] if len(entry) > 3 else 'distance'   # ... or before the method was one
    phases = entry[4] if len(entry) > 4 else None       # ... or before the count was one
    now_shape, now_mask, now_lat, now_meth, now_phases = _normal_field_source(
        eps, lattice, method)
    # The phase count only decides anything off the 'distance' construction, and old
    # records carry none — compare it exactly when both sides have one and it matters.
    phases_agree = (phases is None or now_phases is None or meth != now_meth
                    or meth == 'distance' or phases == now_phases)
    if (shape == now_shape and lat == now_lat and meth == now_meth and phases_agree
            and (mask is None or np.array_equal(mask, now_mask))):
        return                            # equal shapes agree on whether a mask exists at all

    if meth != now_meth:
        moved = (f'the normal field was built with normal_method={meth!r} and this solver '
                 f'is configured normal_method={now_meth!r} — same interfaces, different '
                 f'construction, different field')
    elif lat != now_lat:
        moved = (f'the cell basis is now {now_lat}, where the normal field was sampled on '
                 f'{lat} — the interfaces point the same way in the raster but a different '
                 f'way in space')
    elif shape != now_shape:
        moved = (f'layer {index} is now {now_shape}, where the normal field was sampled on '
                 f'{shape}')
    elif not phases_agree:
        moved = (f'layer {index} now carries {now_phases} distinct phases, where the normal '
                 f'field was sampled on {phases} — merging or splitting phases can leave '
                 f'every interface where it was and still change which construction runs')
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
    C_xx = fourier2d_pointwise(xp, xp.asarray(nx ** 2, dtype=dtype, device=device), fto_x, fto_y)
    C_xy = fourier2d_pointwise(xp, xp.asarray(nx * ny, dtype=dtype, device=device), fto_x, fto_y)
    C_yy = fourier2d_pointwise(xp, xp.asarray(ny ** 2, dtype=dtype, device=device), fto_x, fto_y)
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


def _pn_fields(xp, eps33, nx, ny):
    """Pointwise ``(A, B)`` of the Popov-Neviere rearrangement, for a tensor layer.

    :func:`_nvf_assemble` cannot serve a tensor. Its whole input is ``<eps>`` and ``<1/eps>``,
    and ``1/eps`` is not a thing a tensor has: Li's rule needs a product of *one* discontinuous
    factor with a continuous one, while ``D_x = exx Ex + exy Ey + exz Ez`` is a sum of three
    such products with no per-term rule. The fix is to rearrange until only continuous
    quantities are being multiplied by discontinuous coefficients.

    At an interface with unit normal ``n`` the continuous quantities are tangential ``E`` and
    normal ``D``. Eliminating the discontinuous ``n.E`` from ``D = eps E`` gives, with
    ``a = n.eps.n``, ``p = eps n`` and ``q = eps^T n``::

        (n.E) = [ (n.D) - q.e_t ] / a           e_t = T E

        D = (eps - p q^T / a) e_t + (p / a) (n.D)
        E = (I   - n q^T / a) e_t + (n / a) (n.D)

    so with the continuous vector ``c = e_t + n (n.D)``::

        A = (eps - p q^T / a) T + (p / a) n^T           D = A c
        B = (I   - n q^T / a) T + (n / a) n^T           E = B c

    and the factorized tensor is ``[[A]] [[B]]^-1`` — see :func:`pn_conv_matrices`. Every
    product on the right is one discontinuous coefficient times one continuous quantity, which
    is exactly the case Laurent's rule converges on.

    **Do the projector algebra pointwise, here, and transform once.** The arrangement that
    looks equivalent — building ``[[T]]``, ``[[N]]`` and the transfer ``[[M]]`` separately and
    multiplying the matrices — is not. The normal field is a *unit* field extended into the
    bulk (see :func:`normal_vector_field`), so it is discontinuous itself, and ``[[T]] @ [[M]]``
    is Laurent applied to two factors that jump in the same place. Measured, that costs the one
    property that is not negotiable: a **uniform** layer came back wrong by 0.5, where this
    form is exact to 1e-15 for any normal field at all (``[[A]] = e [[B]]`` identically when
    ``eps = e I``).
    """
    # The normal field arrives real and the tensor complex. NumPy would promote on contact;
    # torch raises ("expected scalar type Double but found ComplexDouble") the moment the two
    # meet in `einsum`, so cast once here rather than at each of the five places they mix.
    dtype = _complex_dtype(xp, eps33)
    n = xp.astype(xp.stack([nx, ny, xp.zeros_like(nx)], axis=-1), dtype)
    eps33 = xp.astype(eps33, dtype)
    eye = xp.astype(xp.eye(3, device=_device(xp, eps33)), dtype)
    N_proj = n[..., :, None] * n[..., None, :]
    T_proj = eye - N_proj

    a = xp.einsum('...i,...ij,...j->...', n, eps33, n)
    p = xp.einsum('...ij,...j->...i', eps33, n)
    q = xp.einsum('...ji,...j->...i', eps33, n)

    # The normal field vanishes on the skeleton of the distance transform (`norm == 0` in
    # `normal_vector_field`). There n = 0, so p, q and n n^T are all zero and T = I -- setting
    # the divisor to 1 makes the two lines below return A = eps and B = I on their own, which
    # is the exact limit: no normal to eliminate, so the direct rule, matching what a zero
    # projector does in `_nvf_assemble`. Guard on the FIELD vanishing, not on `a`: an `a` that
    # vanishes under a nonzero normal is a null direction of the tensor itself, and patching
    # that silently would bury a real singularity instead of reporting it.
    a = xp.where(nx ** 2 + ny ** 2 == 0, xp.ones_like(a), a)[..., None, None]

    A = (eps33 - p[..., :, None] * q[..., None, :] / a) @ T_proj \
        + p[..., :, None] * n[..., None, :] / a
    B = (eye - n[..., :, None] * q[..., None, :] / a) @ T_proj \
        + n[..., :, None] * n[..., None, :] / a
    return A, B


def pn_conv_matrices(xp, eps33, nx, ny, fto_x, fto_y):
    """Popov-Neviere convolution matrices for one *patterned anisotropic* layer.

    ``eps33`` is the pointwise permittivity tensor on the layer raster, shape ``(Ny, Nx, 3, 3)``.
    Returns the same five slots as :func:`nvf_conv_matrices`,
    ``(epx, epy, epxy, epyx, epz_i)``, so the two are interchangeable at the call site.

    Scope: the **in-plane** tensor, i.e. the five slots. The derivation in :func:`_pn_fields`
    is written for a full 3x3 and stays valid for one, but the z row and column only decouple
    from the in-plane block when they are empty (with ``nz = 0`` the projectors are block
    diagonal, so ``A_zz = eps_zz`` and ``B_zz = 1` exactly). A patterned layer that fills them
    needs the 3x3 block solved together, and — more to the point — nothing available can say
    whether the answer is right: fmmax, the one oracle that handles a patterned complex
    off-diagonal, has no z-row arguments at all. That case is left unimplemented rather than
    written blind.
    """
    A, B = _pn_fields(xp, eps33, nx, ny)

    def block(field):
        return xp.concat([xp.concat([fourier2d_pointwise(xp, field[..., i, j], fto_x, fto_y)
                                     for j in range(2)], axis=-1) for i in range(2)], axis=-2)

    eps_t = block(A) @ xp.linalg.inv(block(B))
    ff = eps_t.shape[-1] // 2
    eps_xx, eps_xy = eps_t[..., :ff, :ff], eps_t[..., :ff, ff:]
    eps_yx, eps_yy = eps_t[..., ff:, :ff], eps_t[..., ff:, ff:]

    # z decouples: with nz = 0 the projectors are block diagonal, so A_zz = eps_zz and B_zz = 1
    # exactly and the slot is the plain direct rule inverted -- the same expression
    # `_nvf_assemble` closes with, and it must stay the same one: `<1/eps_zz>` is the *inverse*
    # rule and belongs to a different continuity argument. E_z is TANGENTIAL to a vertical
    # sidewall (the interface normal is in-plane), hence continuous there, and
    # continuous-times-discontinuous is precisely the product Laurent handles.
    #
    # The plain DFT, matching A and B -- **not** the closed form `nvf_conv_matrices` uses for its
    # permittivity coefficients, and that difference was measured rather than assumed.
    #
    # `eps_zz` alone is a staircase, so `fourier2d_piecewise_constant` would give its coefficients
    # exactly and looks like the better choice. It is not, because it describes a different
    # layer: closed form reads the raster as a continuum staircase, the DFT as point samples,
    # and A and B are stuck with the latter (they mix eps with the sampled normal field, which
    # has no closed form). Mixing the two descriptions of one layer costs a symmetry the true
    # answer has -- rotation covariance about z fell from 1.9e-14 to 5.2e-8. The z slot's own
    # matrix is rotation covariant under either transform; it is the disagreement between the
    # two readings that shows up. Consistency wins over per-term accuracy here.
    epz_i = xp.linalg.inv(fourier2d_pointwise(xp, eps33[..., 2, 2], fto_x, fto_y))

    return eps_xx, eps_yy, -eps_xy, -eps_yx, epz_i


def pn_mu_conv_matrices(xp, mu33, nx, ny, fto_x, fto_y):
    """The five **permeability** slots from a patterned ``mu`` tensor.

    ``pn_conv_matrices`` verbatim, in the order :func:`~wavebit._core.solve_2d.layer_modes_2d`
    takes ``mu``. That the same routine serves both is not a shortcut — it is the E/H duality
    the whole magnetic implementation rests on. The rearrangement in :func:`_pn_fields` is
    driven entirely by which quantities are continuous at an interface:

        tangential E and normal D   ->   tangential H and normal B

    and the second line is the first with every symbol replaced. ``D = eps E`` becomes
    ``B = mu H``, so ``a = n.mu.n``, ``p = mu n``, ``q = mu^T n`` and every expression after
    them is unchanged. A separate derivation for ``mu`` would be the same derivation.

    **The reorder is the whole reason this exists as a function.** ``pn_conv_matrices`` returns
    ``(x, y, xy, yx, z_i)`` and ``layer_modes_2d`` takes ``(x, y, z_i, xy, yx)`` — the inverted
    z slot is *third* there and *last* here. Passing one through to the other unchanged puts
    ``mu_zz^-1`` where ``mu_xy`` belongs, which is a plausible-looking magnetic layer rather
    than an error.

    ``nx`` / ``ny`` are the normal field of the **permeability's** boundary, which need not be
    the permittivity's — see :func:`to_conv_mat_raster_pn_mu`.
    """
    mu_xx, mu_yy, mu_xy, mu_yx, mu_z_i = pn_conv_matrices(xp, mu33, nx, ny, fto_x, fto_y)
    return mu_xx, mu_yy, mu_z_i, mu_xy, mu_yx


def to_conv_mat_raster_pn_mu(ucell_tensor, fto_x, fto_y, normal_field=None, lattice=None,
                             normal_method=None):
    """Per-layer permeability slots for a **tensor** ``mu`` raster — the ``mu`` twin of
    :func:`to_conv_mat_raster_pn`.

    Args:
        ucell_tensor: ``(n_layers, Ny, Nx, 3, 3)`` permeability tensors per pixel.
        fto_x, fto_y: Fourier truncation orders.
        normal_field: optional per-layer ``(nx, ny)``. Left ``None`` it is computed from the
            in-plane trace of **``mu``**, not of ``eps``. Reading ``mu_zz`` instead would give
            the same field for any tensor whose components share a boundary — which is every
            physical one — because :func:`normal_vector_field` reads where the interface *is*,
            not what value sits either side (measured: identical to 0 for a uniaxial pillar,
            and 2.0 apart only if ``mu_zz`` is patterned on a different shape, which this
            single-field builder could not serve anyway). The in-plane trace is named because
            it is the part the rearrangement is about, not because the other is wrong here.
        lattice: cell basis, passed to :func:`normal_vector_field`.
        normal_method: the raster construction, as in :func:`to_conv_mat_raster_nvf`;
            read only when ``normal_field`` is not supplied.

    Returns:
        ``(mux, muy, muz_i, muxy, muyx)``, each ``(n_layers, ff_xy, ff_xy)`` — ready to hand to
        ``solve_for_conv(..., mu=...)``.

    **A layer's two tensors may have different boundaries**, and this is why the normal field is
    read off ``mu`` here rather than shared with the permittivity build. A magnetic inclusion
    need not sit where the dielectric one does; the rearrangement is written around *one*
    interface normal, so each tensor gets the normal of its own. Reusing the permittivity's
    field would be correct on every case where the two shapes coincide — which is most of them,
    and is why ``tests/references/_fmmax_patterned_mu_cases.py`` carries a case where they do
    not.
    """
    xp = array_namespace(ucell_tensor)

    if ucell_tensor.shape[-2:] != (3, 3):
        raise ValueError(
            f'to_conv_mat_raster_pn_mu takes a permeability tensor per pixel, shaped '
            f'(n_layers, Ny, Nx, 3, 3); got {tuple(ucell_tensor.shape)}.')

    if normal_field is not None and len(normal_field) != ucell_tensor.shape[0]:
        raise StaleGeometryError(
            f'the frozen geometry no longer fits this structure: it holds a normal field for '
            f'{len(normal_field)} layers and this one has {ucell_tensor.shape[0]}.')

    out = [[], [], [], [], []]
    for i in range(ucell_tensor.shape[0]):
        mu33 = ucell_tensor[i]
        if normal_field is None:
            nx, ny = normal_vector_field(
                0.5 * (mu33[..., 0, 0] + mu33[..., 1, 1]), lattice=lattice,
                method=normal_method)
        else:
            nx, ny = normal_field[i]
        for slot, mat in zip(out, pn_mu_conv_matrices(xp, mu33, nx, ny, fto_x, fto_y)):
            slot.append(mat)

    return tuple(xp.stack(slot) for slot in out)


def to_conv_mat_raster_pn(ucell_tensor, fto_x, fto_y, normal_field=None, lattice=None,
                          normal_method=None):
    """Per-layer Popov-Neviere convolution matrices for a **tensor** raster.

    The counterpart of :func:`to_conv_mat_raster_nvf` for a layer stack whose permittivity is a
    tensor at every pixel rather than a scalar.

    Args:
        ucell_tensor: ``(n_layers, Ny, Nx, 3, 3)`` — the *permittivity* tensor per pixel, not a
            refractive index. This differs from ``ucell`` elsewhere in the package on purpose:
            ``n ** 2`` is a scalar operation and there is no square root of a tensor field worth
            guessing at, so the caller states eps and the ambiguity never arises.
        fto_x, fto_y: Fourier truncation orders.
        normal_field: optional per-layer ``(nx, ny)``. Left ``None`` it is computed from the
            trace of the in-plane block, which is the isotropic part of the layer and so sees
            the same geometry a scalar raster would. Supply it to keep the build traceable
            under JAX, or when the tensor's structure and its *shape* differ — an optic axis
            that rotates across an otherwise uniform slab has no permittivity interface, and a
            normal field read off the values would be reading a texture, not a boundary.
        lattice: cell basis, passed through to :func:`normal_vector_field`.
        normal_method: the raster construction, as in :func:`to_conv_mat_raster_nvf`;
            read only when ``normal_field`` is not supplied.

    Returns:
        ``(epx, epy, epxy, epyx, epz_i)``, each ``(n_layers, ff_xy, ff_xy)`` — the same five
        slots :func:`to_conv_mat_raster_nvf` returns, in the same convention.
    """
    xp = array_namespace(ucell_tensor)

    if ucell_tensor.shape[-2:] != (3, 3):
        raise ValueError(
            f'to_conv_mat_raster_pn takes a permittivity tensor per pixel, shaped '
            f'(n_layers, Ny, Nx, 3, 3); got {tuple(ucell_tensor.shape)}. A scalar raster goes '
            f'to to_conv_mat_raster_nvf instead -- and note this one takes eps, not a '
            f'refractive index.')

    if normal_field is not None and len(normal_field) != ucell_tensor.shape[0]:
        raise StaleGeometryError(
            f'the frozen geometry no longer fits this structure: it holds a normal field for '
            f'{len(normal_field)} layers and this one has {ucell_tensor.shape[0]}. Call '
            f'freeze_geometry() again, or drop geometry= to rebuild by comparison.')

    out = [[], [], [], [], []]
    for i in range(ucell_tensor.shape[0]):
        eps33 = ucell_tensor[i]
        if normal_field is None:
            # The in-plane trace: the layer's isotropic part, which is what carries the shape.
            nx, ny = normal_vector_field(
                0.5 * (eps33[..., 0, 0] + eps33[..., 1, 1]), lattice=lattice,
                method=normal_method)
        else:
            nx, ny = normal_field[i]
        for slot, mat in zip(out, pn_conv_matrices(xp, eps33, nx, ny, fto_x, fto_y)):
            slot.append(mat)

    return tuple(xp.stack(slot) for slot in out)


def nvf_conv_matrices(xp, eps_matrix, nx, ny, fto_x, fto_y):
    """NVF convolution matrices for one layer, with sampled coefficients."""
    C = _complex_dtype(xp, eps_matrix)
    eps_matrix = xp.astype(eps_matrix, C)
    dev = _device(xp, eps_matrix)

    # eps is a staircase however finely it was sampled, so its coefficients are exact from
    # the grid it arrives on. The projector grids are not -- they sample a continuous
    # vector field -- which is why only these two carry the flag.
    ep_dir = fourier2d_piecewise_constant(xp, eps_matrix, fto_x, fto_y, inverse_rule=None)
    ep_inv = fourier2d_piecewise_constant(xp, 1.0 / eps_matrix, fto_x, fto_y, inverse_rule=None)

    return _nvf_assemble(xp, ep_dir, ep_inv,
                         *_nvf_projector(xp, nx, ny, fto_x, fto_y, C, dev))


def nvf_conv_matrices_integrated(xp, eps_comp, x, y, nx, ny, fto_x, fto_y):
    """NVF convolution matrices for one layer, with integrated (closed-form) coefficients.

    ``eps_comp`` is the *compressed* piecewise-constant layer and ``x`` / ``y`` its segment
    edges, exactly as :func:`fourier2d_integrated` takes them; ``nx`` / ``ny`` are the normal field on the
    uncompressed raster (see :func:`_nvf_projector` for why the two grids differ).
    """
    C = _complex_dtype(xp, eps_comp)
    eps_comp = xp.astype(eps_comp, C)
    dev = _device(xp, eps_comp)

    ep_dir = fourier2d_integrated(xp, eps_comp, x, y, fto_x, fto_y, inverse_rule=None)
    ep_inv = fourier2d_integrated(xp, 1.0 / eps_comp, x, y, fto_x, fto_y, inverse_rule=None)

    return _nvf_assemble(xp, ep_dir, ep_inv,
                         *_nvf_projector(xp, nx, ny, fto_x, fto_y, C, dev))


def to_conv_mat_raster_nvf(ucell, fto_x, fto_y, normal_field=None,
                           normal_field_source=None, lattice=None, normal_method=None):
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
        normal_method: how the normal field is constructed from the raster
            (:class:`~wavebit._core.enums.NormalMethod`; ``None`` means ``'distance'``).
            Ignored when ``normal_field`` is supplied — the record was built under its own
            method, which the source check compares against this one.

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
                lattice=lattice, method=normal_method, base_shape=tuple(base_eps.shape))
        else:
            _check_normal_field_still_fits(eps_matrix, normal_field_source, i, lattice,
                                           normal_method)
            nx, ny = normal_field[i]

        for slot, mat in zip(out, nvf_conv_matrices(
                xp, eps_matrix, nx, ny, fto_x, fto_y)):
            slot.append(mat)

    return tuple(xp.stack(slot) for slot in out)


# ============================================ integrated Fourier series (closed form, analytic)

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


def _fourier1d_integrated(xp, x, cell, fto, period):
    """1D continuous Fourier series of a piecewise-constant profile.

    ``x`` holds the right-hand edge of each constant segment (shape ``(n, 1)``) and ``cell``
    the segment values (``(m, n)``). Port of the legacy ``_fourier1d_integrated``, with its two in-place
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


def fourier2d_integrated(xp, cell, x, y, fto_x, fto_y, *, inverse_rule):
    """2D continuous Fourier series with Li's rules — the staircase-free factorization.

    ``x`` / ``y`` are the segment edges of the compressed (piecewise-constant) layer, so the
    permittivity is Fourier-transformed analytically rather than off a raster.

    **``continuous`` describes the method, not the data.** What arrives here is piecewise
    constant, exactly as in :func:`fourier2d_piecewise_constant`; the coefficients are obtained by
    integrating over the segments rather than by a discrete sum, which is the ``integrated`` / ``sampled``
    distinction and nothing more. The public spelling of the same choice is
    ``fourier_coefficients='integrated'``.

    What that buys is *where the steps may sit*. :func:`fourier2d_piecewise_constant` places them
    on the pixel grid at ``j/n``, so an edge at ``x = 0.317`` is first snapped to a pixel and
    then transformed exactly; here the edge is a number this function integrates to. Both are
    exact for the staircase they are given — they differ in which staircases they can be
    given. Curved boundaries are still approximated by steps in both;
    :mod:`wavebit._core.analytic` is the path where the staircase itself is gone.

    ``inverse_rule`` is as in :func:`fourier2d_piecewise_constant`.
    """
    inv_x, inv_y = _inverse_axes(inverse_rule)
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    period_x, period_y = x[-1], y[-1]

    cell = xp.matrix_transpose(cell)

    if inv_y:             # y discontinuous: the inverse rule starts from 1/eps
        cell = 1 / cell

    cfs1d = _fourier1d_integrated(xp, y, cell, fto_y, period_y)

    dev = _device(xp, cell)
    idx_y = _circulant_gather(xp, fto_y, dev)
    idx_x = _circulant_gather(xp, fto_x, dev)

    conv1d = cfs1d[:, idx_y]

    if inv_x or inv_y:    # Li's inner step, whichever direction asked for it
        conv1d = xp.linalg.inv(conv1d)

    conv1d = xp.reshape(conv1d, (-1, ff_y ** 2))

    cfs_2d = _fourier1d_integrated(xp, x, xp.matrix_transpose(conv1d), fto_x, period_x)

    conv2d = cfs_2d[:, idx_x]
    conv2d = xp.reshape(conv2d, (ff_y, ff_y, ff_x, ff_x))
    conv2d = xp.moveaxis(conv2d, 1, 2)
    conv2d = xp.reshape(conv2d, (ff_y * ff_x, ff_y * ff_x))

    if inv_x:             # x discontinuous: Li's outer step closes it
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
    that is what :meth:`~wavebit._core.engine.CoreFactorizationMixin.freeze_geometry`
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


def to_conv_mat_raster_integrated(ucell, fto_x, fto_y, compression=None):
    """Per-layer convolution matrices via the analytic (staircase-free) integrated path.

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

        epz_list = fourier2d_integrated(xp, eps, x, y, fto_x, fto_y, inverse_rule=None)
        epx_list.append(fourier2d_integrated(xp, eps, x, y, fto_x, fto_y, inverse_rule='x'))
        epy_list.append(fourier2d_integrated(xp, eps, x, y, fto_x, fto_y, inverse_rule='y'))
        epz_i_list.append(xp.linalg.inv(epz_list))

    return xp.stack(epx_list), xp.stack(epy_list), xp.stack(epz_i_list)


def to_conv_mat_raster_integrated_nvf(ucell, fto_x, fto_y, normal_field=None,
                                      normal_field_source=None, compression=None,
                                      lattice=None, normal_method=None):
    """Per-layer convolution matrices for a raster ``ucell`` via closed-form Fourier + NVF.

    The pairing that needs both axes named independently. It buys the staircase-free
    permittivity series of the integrated path *and* the vector factorization that handles
    boundaries not aligned with x or y — which is what a shape/OCD fit of a tilted sidewall
    wants from a raster.

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
        normal_method: the construction, as in :func:`to_conv_mat_raster_nvf`.

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
                lattice=lattice, method=normal_method, base_shape=tuple(base_eps.shape))
        else:
            _check_normal_field_still_fits(eps_grid, normal_field_source, i, lattice,
                                           normal_method)
            nx, ny = normal_field[i]

        layer = ucell[i]
        if compression is None:
            row_idx, col_idx, x, y = compression_indices(layer)
        else:
            _check_compression_still_fits(layer, compression[i], i)
            row_idx, col_idx, x, y = compression[i]
        n_comp = compress_cell(xp, layer, row_idx, col_idx)
        eps_comp = xp.astype(n_comp, _complex_dtype(xp, n_comp)) ** 2

        for slot, mat in zip(out, nvf_conv_matrices_integrated(
                xp, eps_comp, x, y, nx, ny, fto_x, fto_y)):
            slot.append(mat)

    return tuple(xp.stack(slot) for slot in out)


def to_conv_mat_block(ucell_info_list, fto_x, fto_y):
    """Per-layer convolution matrices for vector-modelled layers (integrated over explicit segments).

    Each entry of ``ucell_info_list`` is ``(layer, x_list, y_list)`` — the segment values and
    their edges — so no compression step is needed.
    """
    xp = array_namespace(ucell_info_list[0][0])

    epx_list, epy_list, epz_i_list = [], [], []

    for layer, x, y in ucell_info_list:
        eps = xp.astype(layer, _complex_dtype(xp, layer)) ** 2
        epx_list.append(fourier2d_integrated(xp, eps, x, y, fto_x, fto_y, inverse_rule='x'))
        epy_list.append(fourier2d_integrated(xp, eps, x, y, fto_x, fto_y, inverse_rule='y'))
        epz_i_list.append(xp.linalg.inv(fourier2d_integrated(xp, eps, x, y, fto_x, fto_y, inverse_rule=None)))

    return xp.stack(epx_list), xp.stack(epy_list), xp.stack(epz_i_list)
