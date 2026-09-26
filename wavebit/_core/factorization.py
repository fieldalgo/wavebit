"""Single-source Fourier factorization — Phase 4a of the rewrite.

Builds the per-layer convolution matrices (``epx`` / ``epy`` / ``inv(epz)``) that the solve
core consumes, written once against the Array-API namespace instead of once per backend.

Why this matters beyond de-duplication: until the factorization is inside the differentiable
chain, autodiff can only reach the *convolution matrices*, not the permittivity that produced
them. With this module the gradient runs end-to-end from the design variable — the
``structure`` raster — through to the diffraction efficiency, which is what an inverse-design
user actually differentiates.

**Every raster here is relative permittivity, one sample per pixel.** Nothing in this module
squares anything on the way in: the front door takes the quantity the factorizers work in, so
there is no conversion left to place. That was not always so, and the conversion's removal is
the whole content of the flip -- see ``docs/conventions.md``.

The same builders serve the *permeability* when a structure carries one, unchanged and by
name: Maxwell's two curl equations map onto each other under
``(E, H, eps, mu) -> (H, -E, mu, eps)``, so a rule derived for one closes the other term for
term (:func:`pn_mu_conv_matrices` is :func:`pn_eps_conv_matrices` under a second name for exactly
this reason). ``CoreFactorizationMixin._mu_conv_matrices`` is what runs them twice.

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

The removed block path's layers were the closed-form series on explicitly
given segment edges, with Li's rules; the parametric shape path lives in
:mod:`wavebit._core.parametric` and is NVF, except that axis-aligned rectangles in an
axis-aligned rectangular cell also take Li's rules, through :func:`fourier2d_integrated`
(:mod:`wavebit._core.rectilinear`).

A recurring rule runs through this module: **discrete geometry decisions are computed eagerly
and passed in as static inputs, while material values flow through differentiably.** Two
decisions here are geometry, not material — the NVF normal field (where the interfaces point) and the integrated block
compression (where the piecewise-constant blocks start and stop). Both branch on values, and
the compression's output shape depends on the answer, so neither survives ``jax.jit``. Each is
therefore split into an eager step that yields a static description
(:func:`normal_field_for_raster`, :func:`compression_indices`) and a differentiable step that
consumes it. Pass those precomputed descriptions to stay traceable; the permittivity riding on
top of them is what gradients see.

``jax.grad`` alone is the weaker bar, and both decisions clear it, because a grad tracer
carries a concrete primal and so permits narrowing one value at a time even though it refuses a
whole-array NumPy conversion (:func:`_as_bools`). The compression compares neighbouring
rows and columns that way directly. The normal field cannot — SciPy wants the mask as an array
— so it derives that mask from the *base* raster rather than reading it off the upsampled one
(:func:`_upsampled_interface_mask`); the upsample is a block repeat, so the two are the same
mask, and the base has ``repeat_v * repeat_u`` fewer values to narrow. **Neither raster
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

Slot order
----------

**Every builder here returns its slots in ``solve_conv``'s parameter order**, so that

.. code-block:: python

    res = mee.solve_conv(*to_conv_mat_raster_nvf(structure, half_width_u, half_width_v))

is correct for any of them. Concretely, three slots are ``(epx, epy, epz_i)`` and five are
``(epx, epy, epz_i, epxy, epyx)`` — the inverted zz matrix is **third**, not last, because
``solve_conv`` requires it on every solve (it eliminates ``E_z``) while the off-diagonal
pair defaults to ``None``, and Python will not put a required parameter after a defaulted one.

Grouping the in-plane 2x2 block together and putting ``epz_i`` last reads better and was the
order the five-slot builders used until 2026-08-18. It is not used because the difference is
undetectable: splatting a five-slot builder into ``solve_conv`` then lands ``epz_i`` in the
``epxy`` slot, and the solve returns a full set of plausible efficiencies rather than raising —
measured ``R + T = 1.61`` against ``1.0000000000`` for the same structure. The same idiom being
correct for the three-slot builders is what makes it a trap rather than a mistake. It was found
twice: once inside this module (see :func:`pn_mu_conv_matrices`, which existed only to reorder
around it) and once through ``solve_conv`` itself, by the energy assertion in
``tests/test_manifest.py``.

``solve_conv``'s order is the one that won rather than the builders' for a second reason
besides being the forced one: the public ``eps_conv_matrices()`` accessor already returned it, so
``solve_conv(*solver.eps_conv_matrices())`` was the one splat in the package that always
worked (``tests/test_conv_matrices_accessor.py``). The builders are what disagreed with it.

``tests/test_slot_order.py`` pins the property directly, for every builder.
"""
import math
from typing import NamedTuple

from .backend import array_namespace
from .enums import DEFAULT_NESTING, Nesting, nesting_of
from .geometry import StaleGeometryError


class ConvMatrices(NamedTuple):
    """The five convolution matrices of a layer stack, in ``solve_conv`` order.

    What :meth:`~wavebit._core.engine.CoreFactorizationMixin.eps_conv_matrices` returns, and what
    ``solve_conv`` takes first — ``solver.solve_conv(*solver.eps_conv_matrices())`` is
    ``solver.solve()``. See "Slot order" above for why ``epz_i`` sits third rather than last.

    **Every builder in this module returns this**, the Li-rule ones included: they do not
    couple ``x`` and ``y``, so they set ``epxy`` and ``epyx`` to ``None``, which is what every
    consumer already reads that as. There was a separate three-slot type until 2026-08-21,
    and two arities for one quantity meant two slot orders to keep straight -- to state a fact
    the five-slot form states more directly, at the one slot that carries it.

    Splatting, unpacking, ``len``, indexing and ``zip`` all work as they did when the builders
    returned bare tuples; what it adds is ``conv.epz_i``, reaching one slot by a name instead
    of by an index that stays correct only until somebody reorders.

    Used for the permeability slots too, where the five mean ``mux_conv``, ``muy_conv``,
    ``muz_conv_i``, ``muxy_conv``, ``muyx_conv`` — the E/H duality means one layout serves
    both (see :func:`pn_mu_conv_matrices`), and the ``ep`` field names are that duality
    showing rather than a mislabelling.
    """
    epx: object
    epy: object
    epz_i: object
    epxy: object
    epyx: object

    @classmethod
    def from_energy(cls, change, gram, longitudinal):
        """Build solve slots from a supplied NVF change of field and Gram matrix.

        ``change`` is Q = C(B), ``gram`` is G = C(B.H epsilon B), and
        ``longitudinal`` is C(epsilon_zz). Q and G have shape (2H, 2H),
        or (layers, 2H, 2H); the longitudinal shape is (H, H), or
        (layers, H, H), respectively. The same Fourier orders and Cartesian
        frame must describe all three inputs and the consuming solver; Q
        and G group all x-component orders before all y-component orders.
        This five-slot closure assumes no xz, yz, zx or zy material coupling.

        Two solves form Q^(-H) G Q^(-1). No Hermitian projection, clipping
        or regularization is applied: complex, lossy Gram matrices retain
        their anti-Hermitian part. The zz slot is inverted separately, and
        the xy/yx slots carry the solver's negative off-diagonal convention.
        Inputs are preserved, including their autodiff graph and precision.

        A layer stack can be passed directly to ``solver.solve_conv(*conv)``.
        This closes supplied constitutive data; it neither constructs B from
        shapes nor certifies its discretization or optical convergence.
        """
        from .backend import _concrete

        xp = array_namespace(change, gram, longitudinal)
        if (change.ndim not in (2, 3) or change.shape[-1] == 0
                or change.shape[-1] % 2 or change.shape[-2] != change.shape[-1]
                or gram.shape != change.shape):
            raise ValueError('energy change and Gram matrices need matching nonempty even square shapes')
        count = change.shape[-1] // 2
        if longitudinal.shape != (*change.shape[:-2], count, count):
            raise ValueError('longitudinal matrix must match the energy layers and Fourier order count')
        if change.ndim == 3 and change.shape[0] == 0:
            raise ValueError('energy layer stack must not be empty')
        dtype = xp.result_type(change, gram, longitudinal)
        if not xp.isdtype(dtype, ('real floating', 'complex floating')):
            raise TypeError('energy matrices must have floating-point or complex dtype')
        change, gram, longitudinal = (xp.astype(a, dtype, copy=False)
                                      for a in (change, gram, longitudinal))
        valid = (xp.all(xp.isfinite(change)) & xp.all(xp.isfinite(gram))
                 & xp.all(xp.isfinite(longitudinal)))
        if _concrete(valid) is False:
            raise ValueError('energy matrices must be finite')
        transpose = xp.matrix_transpose
        left = xp.linalg.solve(transpose(xp.conj(change)), gram)
        effective = transpose(xp.linalg.solve(transpose(change), transpose(left)))
        identity = xp.eye(count, dtype=dtype, device=_device(xp, longitudinal))
        z_inverse = xp.linalg.solve(longitudinal, xp.broadcast_to(identity, longitudinal.shape))
        valid = valid & xp.all(xp.isfinite(effective)) & xp.all(xp.isfinite(z_inverse))
        if _concrete(valid) is False:
            raise ValueError('energy matrices must define finite nonsingular solves')
        # Under JIT, invalid data has no concrete boolean to raise on. Poison
        # every slot instead of returning a partly usable constitutive tensor.
        effective = xp.where(valid, effective, xp.nan)
        z_inverse = xp.where(valid, z_inverse, xp.nan)
        return cls(effective[..., :count, :count], effective[..., count:, count:],
                   z_inverse, -effective[..., :count, count:], -effective[..., count:, :count])


def _inverse_axes(inverse_rule):
    """``inverse_rule`` as ``(along_u, along_v)`` booleans, refusing anything else.

    One three-valued argument rather than the pair of ``conti_x`` / ``conti_y`` integers this
    took until 2026-08-12. Those had four states and only three meanings: ``(0, 0)`` — both
    directions discontinuous — is not part of Li's scheme as used here and was never passed
    anywhere, but nothing said so and nothing refused it. A typo could reach it silently.

    The names also ran backwards. ``conti_y=0`` meant "y is discontinuous, so y takes the
    inverse rule", so the flag that was set said what was *not* true of the axis it named.

    The values were ``'x'`` / ``'y'`` until 2026-08-23. The rule runs along a *lattice* axis --
    the direction the cell's own edge follows -- which is x and y only while ``a1`` and ``a2``
    are the Cartesian axes.

    **Retired, not conditionally allowed**, and the two are worth telling apart because
    ``truncation_order`` went the other way on the same day: there ``dict(x=, y=)`` is kept, and
    on a skewed cell it is *read* as ``u``/``v`` with a warning
    (:class:`~wavebit._core.backend.AxisNameWarning`) rather than refused, because the reading
    is right and only the caller's vocabulary is off. Two things make this one a rename instead.

    A **value** is not a keyword. Two spellings of a keyword cost one parse; two spellings of a
    value cost a branch at every comparison, and leave the next reader unsure which is canonical.

    And this function **cannot see the cell**. It takes a raster and two half-widths; there is no
    lattice in its signature, so "allow ``'x'`` where it is true" is not a check it can make. The
    conditional treatment ``truncation_order`` gets is available only because the solver knows
    the basis by then. Here the choice is unconditional either way, and an unconditional
    *allow* would keep a spelling that is wrong on every skewed cell.

    (An earlier version of this note argued the difference was that ``sample_x`` refuses inside
    the function that takes the name, while this is a documented out-of-tree seam that refuses
    nothing. It is not a seam: ``fourier2d_piecewise_constant`` is not exported from ``wavebit``,
    and every caller in the docs and tutorials reaches it through ``_core``. The rename needed no
    such argument.)
    """
    if inverse_rule in ('x', 'y'):
        raise ValueError(
            f"inverse_rule={inverse_rule!r}: the rule runs along a lattice axis, not a lab one — "
            f"pass {'u' if inverse_rule == 'x' else 'v'!r} instead. They coincide only on a "
            f"rectangular cell; on an oblique lattice 'x' does not name a cell direction.")
    if inverse_rule not in (None, 'u', 'v'):
        raise ValueError(
            f"inverse_rule: expected None (direct rule both ways), 'u' or 'v' — the direction "
            f"whose discontinuity the inverse rule closes; got {inverse_rule!r}")
    return inverse_rule == 'u', inverse_rule == 'v'


def _circulant_index(truncation_order):
    """Circular index matrix selecting Fourier orders ``m - n`` (pure integer, data-free).

    The sign here *is* the order-index convention, and it has to agree with the sign ``kx``
    carries: a convolution matrix built for one and paired with the other mirrors the
    diffraction orders (+1 and -1 swap) while conserving energy, so nothing catches it except
    a grating comparison. Uniform layers cannot expose it at all -- their convolution matrix
    is ``eps * I``, which is its own transpose.
    """
    n_orders = 2 * truncation_order + 1
    return [[(r - c) for c in range(n_orders)] for r in range(n_orders)]


def _pixel_correction(xp, orders, n, dtype, device=None):
    """Turn an ``n``-point DFT coefficient into the exact one for ``n`` constant pixels.

    A raster layer is not a set of samples, it is a piecewise-constant function: pixel ``j``
    holds its value across ``[j/n, (j+1)/n)``. Integrating that against ``exp(-2*pi*i*k*x)``
    gives the ``n``-point DFT coefficient times ``sinc(k/n) * exp(-i*pi*k/n)`` — a box
    transform and the half-pixel offset of the cell centre. Exactly, for any ``k``, including
    ``|k| > n`` where the DFT itself has wrapped.

    That last part is what makes the upsampling unnecessary rather than merely cheaper. The
    previous approach repeated each pixel until the grid resolved ``4*truncation_order+1`` harmonics and
    took the DFT of *that*, which converges to this expression as ``1/R`` — so it was an
    approximation of a closed form, at a grid cost of ``(4*truncation_order+1)**2`` per layer.
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


# ======================================================================== Li's rules, nested
#
# Written from Li, JOSA A 14, 2758 (1997), and Li, JOSA A 13, 1870 (1996).
#
# Everything from here to `li_conv_matrices_integrated` was first written on 2026-09-25 from
# the nested crossed-grating rules of the 1997 paper and the one-dimensional rules they nest
# from the 1996 one, without reference to RETICOLO's or meent's source or to the bodies of the
# builders it replaced. Those came from meent, whose Li-rule code was written with reference to
# RETICOLO's MATLAB source. Later changes are described in their own commit messages.
#
# The pieces, bottom up:
#
# * a *series* maps values along the last axis to their Fourier coefficients for the orders
#   ``-2*half_width .. 2*half_width`` -- `_raster_series` for ``n`` equal pixels (the DFT, made
#   exact by `_pixel_correction`), `_segment_series` for arbitrary segment edges;
# * `_conv_matrix` is the direct rule both ways: the 2D coefficients laid out as the BTTB
#   convolution matrix of docs/factorization-terms.md §3;
# * `_nested_rule` is one nested operator: Toeplitz matrices along the inner axis, inverted,
#   then transformed entrywise along the outer axis and laid out block-Toeplitz;
# * `_li_slot` picks the operator for one slot and one `Nesting`, and `li_conv_matrices` /
#   `li_conv_matrices_integrated` build a layer's three.

def _toeplitz_positions(xp, half_width, dev):
    """Where order ``r - c`` sits in a ``4*half_width+1`` coefficient vector, for every
    ``(r, c)`` of an ``(n_orders, n_orders)`` matrix, row-major — (F1) as a gather index.

    The sign is :func:`_circulant_index`'s, which is the order-index convention (see its
    docstring for what a flipped one costs).
    """
    return xp.asarray([d + 2 * half_width for row in _circulant_index(half_width) for d in row],
                      device=dev)


def _row_dft(xp, values):
    """``DFT(values / n)`` along the last axis of ``values`` ``(..., n)``, never as a degenerate
    FFT.

    A one-point transform is the identity, so none is emitted; a single row is transformed as
    the 1D vector it is and given its shape back; anything else is the batched transform. The
    values are those of ``xp.fft.fft(values / n, axis=-1)``. What the two special cases avoid
    is an operand XLA's CPU FFT refuses: under ``jax.jit`` a ``(1, n)`` or ``(n, 1)`` operand
    that a transpose or reshape produced can reach the FFT with a transposed layout, and the
    run aborts (``RET_CHECK failure ... fft_thunk.cc:168
    LayoutUtil::IsMonotonicWithDim0Major``). A one-pixel axis, or a zero half-width on one
    axis -- ``truncation_order=0``, a 1D grating -- makes exactly those shapes on a small
    raster.

    The rule is the one ``_normalized_row_fft`` applied to the builders this module replaced
    (added 2026-09-24, deleted with them), carried over from its stated contract. The jitted
    small-raster solves in ``tests/test_direct_wavevector.py``,
    ``tests/test_phase_incidence.py``, ``tests/test_half_space_power.py`` and
    ``tests/test_vertical_coordinates.py`` are what pin it.
    """
    n = values.shape[-1]
    if n == 1:
        return xp.astype(values, _complex_dtype(xp, values))
    values = values / n
    if math.prod(values.shape[:-1]) == 1:
        return xp.reshape(xp.fft.fft(xp.reshape(values, (n,))), values.shape)
    return xp.fft.fft(values, axis=-1)


def _raster_series(xp, half_width, correction):
    """Fourier coefficients along the last axis of values filling ``n`` equal cells of the
    period.

    Returns ``series(a)``: ``a`` of shape ``(..., n)`` goes to ``(..., 4*half_width+1)``, entry
    ``k + 2*half_width`` holding order ``k``. The ``n``-point DFT supplies every order -- bin
    ``k mod n``, which is what makes an order past ``n/2`` legal -- and ``correction`` turns it
    into the quantity the grid actually describes: :func:`_pixel_correction` for values held
    across their pixel (the exact box transform), :func:`_half_pixel_phase` for point samples.

    Two choices here are about bits, not values, and both are forced by
    :func:`fourier2d_pointwise`, whose output the NVF projector consumes and which this rewrite
    had to leave identical to the last bit (it did on NumPy and PyTorch; see that function for
    JAX). The ``1/n`` is applied to the
    values *before* the transform, not to the spectrum after it (they differ off powers of two);
    and the transform runs along the **last** axis. NumPy's FFT gives the same bits along any
    axis, JAX's and PyTorch's do not: measured on 21 grids, taking it along axis 0 instead left
    15 of them a few 1e-17 off on both. The transform itself is :func:`_row_dft`, which never
    hands the FFT a one-point or single-row batch.
    """
    orders = list(range(-2 * half_width, 2 * half_width + 1))

    def series(a):
        n = a.shape[-1]
        dev = _device(xp, a)
        spectrum = _row_dft(xp, a)
        picked = xp.take(spectrum, xp.asarray([k % n for k in orders], device=dev), axis=-1)
        return picked * correction(xp, xp.asarray(orders, device=dev), n, picked.dtype, dev)

    return series


def _segment_bounds(xp, edges, dtype, dev):
    """``(left, width)`` of the segments whose right-hand edges are ``edges``, on the unit cell.

    ``edges`` is what :func:`compression_indices` returns -- right-hand edges, increasing, the
    last one closing the period -- in whatever unit the caller used: dividing by the last edge
    is what puts them on the unit cell, where the coefficient integral has no ``1/period``
    left in it. A traced ``edges`` keeps its graph (:func:`_lift`).
    """
    right = xp.reshape(_lift(xp, edges, dtype, dev), (-1,))
    right = right / right[-1]
    left = xp.concat([xp.zeros((1,), dtype=dtype, device=dev), right[:-1]])
    return left, right - left


def _segment_series(xp, left, width, half_width):
    """Fourier coefficients along the last axis of a profile constant on arbitrary segments.

    Returns ``series(a)`` like :func:`_raster_series`, for segment ``j`` holding ``a[..., j]``
    across ``[left_j, left_j + width_j)`` of the unit cell. The coefficient is the segment
    integral::

        c_k = sum_j a_j (exp(-2 pi i k left_j) - exp(-2 pi i k right_j)) / (2 pi i k),  k != 0
        c_0 = sum_j a_j width_j

    and it is evaluated in the equivalent **jump** form, regrouped by edge -- segment ``j``'s
    left edge is segment ``j-1``'s right one, and ``right`` of the last is ``left_0 + 1``::

        c_k = sum_j (a_j - a_{j-1}) exp(-2 pi i k left_j) / (2 pi i k),       a_{-1} = a_{n-1}

    The two are the same number in exact arithmetic. What decides between them is a uniform
    stretch: its jumps are exactly zero, so it contributes exactly nothing off ``k = 0``,
    where the segment form leaves ``exp(-2 pi i k)`` of round-off. That is what keeps a
    uniform layer's matrix *exactly* ``eps * I``
    (``tests/test_factorization_core.py::test_a_uniform_cell_factorizes_to_exactly_eps_times_identity``),
    and why ``left_0 = 0`` exactly, so the wrap-around jump's phase is exactly one.
    """
    orders = list(range(-2 * half_width, 2 * half_width + 1))

    def series(a):
        dtype = a.dtype
        dev = _device(xp, a)
        k = xp.astype(xp.asarray(orders, device=dev), dtype)
        scale = xp.asarray([0j if o == 0 else 1.0 / (2j * math.pi * o) for o in orders],
                           dtype=dtype, device=dev)
        weights = xp.exp(-2j * math.pi * (xp.astype(left, dtype)[:, None] * k[None, :]))
        jumps = a - xp.roll(a, 1, axis=-1)
        ac = jumps @ (weights * scale[None, :])                           # (..., n_k)
        dc = xp.expand_dims(a @ xp.astype(width, dtype), axis=-1)         # (..., 1)
        return xp.where(xp.asarray([o == 0 for o in orders], device=dev), dc, ac)

    return series


def _conv_matrix(xp, grid, half_width_u, half_width_v, series_u, series_v):
    """The direct rule both ways: ``M[(n, m), (n', m')] = c(n - n', m - m')``.

    ``grid`` is ``(n_v, n_u)`` -- rows run along ``v`` -- and the 2D coefficients are taken one
    axis at a time, ``v`` first. With no inverse in between the two transforms commute, so the
    order decides nothing but rounding; ``v`` first is :func:`fourier2d_pointwise`'s order to the
    bit. The layout is the BTTB one of ``docs/factorization-terms.md`` §3, rows
    ``i = n*n_orders_u + m``.
    """
    n_u, n_v = 2 * half_width_u + 1, 2 * half_width_v + 1
    dev = _device(xp, grid)
    by_column = series_v(xp.permute_dims(grid, (1, 0)))                  # [j_u, k_v]
    coeffs = series_u(xp.permute_dims(by_column, (1, 0)))                 # [k_v, k_u]
    blocks = xp.take(xp.take(coeffs, _toeplitz_positions(xp, half_width_v, dev), axis=0),
                     _toeplitz_positions(xp, half_width_u, dev), axis=1)  # [(n, n'), (m, m')]
    out = xp.permute_dims(xp.reshape(blocks, (n_v, n_v, n_u, n_u)), (0, 2, 1, 3))
    return xp.reshape(out, (n_v * n_u, n_v * n_u))


def _nested_rule(xp, grid, half_width_u, half_width_v, series_u, series_v, *, inner,
                 invert_outer):
    """One of Li's nested operators on a piecewise-constant ``grid`` (rows along ``v``).

    With ``inner='v'``: each column of ``grid`` is transformed along ``v`` and laid out as its
    Toeplitz matrix ``T_v(g)(u)``, ``[.]_{n, n'} = c_{n - n'}``, which is then **inverted** --
    a matrix-valued function of ``u``, constant on each column. That function is transformed
    entrywise along ``u`` and laid out block-Toeplitz,
    ``out[(n, m), (n', m')] = A_hat_{m - m'}[n, n']``, and ``invert_outer`` inverts the result::

        T_u( [T_v(g)]^-1 )               invert_outer=False
        ( T_u( [T_v(g)]^-1 ) )^-1        invert_outer=True

    ``inner='u'`` swaps the roles of the two axes. The output is in the BTTB layout of
    :func:`_conv_matrix` either way -- ``u``-inner builds its blocks along ``v`` and reads them
    back in that order, so the caller never sees which axis was inside.

    The inner inverse is what makes the order of the two transforms matter, and so what the
    :class:`~wavebit._core.enums.Nesting` choice is about: without it the operator is
    :func:`_conv_matrix`, whichever axis goes first.
    """
    n_u, n_v = 2 * half_width_u + 1, 2 * half_width_v + 1
    dev = _device(xp, grid)
    at_u = _toeplitz_positions(xp, half_width_u, dev)
    at_v = _toeplitz_positions(xp, half_width_v, dev)

    if inner == 'v':
        lines = series_v(xp.permute_dims(grid, (1, 0)))                  # [j_u, k_v]
        inner_t = xp.linalg.inv(
            xp.reshape(xp.take(lines, at_v, axis=1), (lines.shape[0], n_v, n_v)))
        entries = xp.permute_dims(xp.reshape(inner_t, (lines.shape[0], n_v * n_v)), (1, 0))
        outer = xp.take(series_u(entries), at_u, axis=1)                   # [(n, n'), (m, m')]
        out = xp.permute_dims(xp.reshape(outer, (n_v, n_v, n_u, n_u)), (0, 2, 1, 3))
    elif inner == 'u':
        lines = series_u(grid)                                             # [j_v, k_u]
        inner_t = xp.linalg.inv(
            xp.reshape(xp.take(lines, at_u, axis=1), (lines.shape[0], n_u, n_u)))
        entries = xp.permute_dims(xp.reshape(inner_t, (lines.shape[0], n_u * n_u)), (1, 0))
        outer = xp.take(series_v(entries), at_v, axis=1)                   # [(m, m'), (n, n')]
        out = xp.permute_dims(xp.reshape(outer, (n_u, n_u, n_v, n_v)), (2, 0, 3, 1))
    else:
        raise ValueError(f"inner: expected 'u' or 'v', got {inner!r}")

    out = xp.reshape(out, (n_v * n_u, n_v * n_u))
    return xp.linalg.inv(out) if invert_outer else out


def _nesting(nesting):
    """``nesting`` as a :class:`~wavebit._core.enums.Nesting`, ``None`` meaning the default."""
    return DEFAULT_NESTING if nesting is None else nesting_of(nesting)


def _li_slot(xp, cell, half_width_u, half_width_v, series_u, series_v, inverse_rule, nesting):
    """One slot of Li's rules: the operator ``inverse_rule`` names, nested as ``nesting`` says.

    ``inverse_rule`` is the direction whose discontinuity the inverse rule closes
    (:func:`_inverse_axes`), and ``cell`` the permittivity -- the reciprocal is taken here where
    a rule needs it. The operators, from Li (1997), with ``T`` as in :func:`_nested_rule`:

    =================  ===================================  ===================================
    ``inverse_rule``   ``'fixed'``                          ``'symmetric'``
    =================  ===================================  ===================================
    ``'u'`` (epx)      ``( T_u( [T_v(eps)]^-1 ) )^-1``      ``T_v( [T_u(1/eps)]^-1 )``
    ``'v'`` (epy)      ``T_u( [T_v(1/eps)]^-1 )``           the same
    ``None`` (epz)     ``T_uv(eps)``, the direct rule       the same
    =================  ===================================  ===================================

    ``'fixed'`` keeps ``v`` inside for both slots, so ``epx``'s inverse rule has to be taken on
    the *outer* axis -- invert the inner matrices, transform, invert again. ``'symmetric'``
    puts each slot's inverse-rule axis inside, which makes ``epx`` the x<->y mirror image of
    ``epy``. On a y-invariant layer the two coincide: ``T_v`` of a function of ``u`` alone is
    that function times the identity, and both ``epx`` become ``[[1/eps]]_u^-1`` per ``v``
    order -- Li's 1D inverse rule.
    """
    along_u, along_v = _inverse_axes(inverse_rule)
    nesting = _nesting(nesting)
    if along_v:
        return _nested_rule(xp, 1.0 / cell, half_width_u, half_width_v, series_u, series_v,
                            inner='v', invert_outer=False)
    if along_u and nesting == Nesting.SYMMETRIC:
        return _nested_rule(xp, 1.0 / cell, half_width_u, half_width_v, series_u, series_v,
                            inner='u', invert_outer=False)
    if along_u:
        return _nested_rule(xp, cell, half_width_u, half_width_v, series_u, series_v,
                            inner='v', invert_outer=True)
    return _conv_matrix(xp, cell, half_width_u, half_width_v, series_u, series_v)


def _li_conv(xp, eps, half_width_u, half_width_v, series_u, series_v, nesting):
    """A layer's three Li slots, in ``solve_conv`` order, the off-diagonal pair ``None``."""
    def slot(inverse_rule):
        return _li_slot(xp, eps, half_width_u, half_width_v, series_u, series_v,
                        inverse_rule, nesting)

    return ConvMatrices(slot('u'), slot('v'), xp.linalg.inv(slot(None)), None, None)


def _pixel_series_pair(xp, half_width_u, half_width_v, correction):
    """``(series_u, series_v)`` for a raster: :func:`_raster_series` along each axis."""
    return (_raster_series(xp, half_width_u, correction),
            _raster_series(xp, half_width_v, correction))


def _segment_series_pair(xp, x, y, half_width_u, half_width_v, complex_dtype, dev):
    """``(series_u, series_v)`` for segments with right-hand edges ``x`` / ``y``."""
    real = xp.float32 if complex_dtype == xp.complex64 else xp.float64
    return (_segment_series(xp, *_segment_bounds(xp, x, real, dev), half_width_u),
            _segment_series(xp, *_segment_bounds(xp, y, real, dev), half_width_v))


def li_conv_matrices(xp, eps_matrix, half_width_u, half_width_v, nesting=None):
    """Li-rule convolution matrices for one layer, with sampled coefficients.

    ``eps_matrix`` is the ``(n_v, n_u)`` permittivity raster; its coefficients are the exact
    ones of the staircase it describes (:func:`_pixel_correction`), whatever its size.
    ``nesting`` is a :class:`~wavebit._core.enums.Nesting`, ``None`` meaning ``'fixed'``.

    Returns a :class:`ConvMatrices` ``(epx, epy, epz_i, None, None)`` -- Li's rules do not
    couple x and y -- each ``(n_orders, n_orders)``.
    """
    eps_matrix = xp.astype(eps_matrix, _complex_dtype(xp, eps_matrix))
    return _li_conv(xp, eps_matrix, half_width_u, half_width_v,
                    *_pixel_series_pair(xp, half_width_u, half_width_v, _pixel_correction),
                    nesting)


def li_conv_matrices_integrated(xp, eps_comp, x, y, half_width_u, half_width_v, nesting=None):
    """Li-rule convolution matrices for one layer, with integrated (closed-form) coefficients.

    ``eps_comp`` is the compressed piecewise-constant layer and ``x`` / ``y`` the right-hand
    edges of its columns and rows (:func:`compression_indices`); the coefficients are the
    segment integrals of :func:`_segment_series`. ``nesting`` as in :func:`li_conv_matrices`.
    """
    C = _complex_dtype(xp, eps_comp)
    eps_comp = xp.astype(eps_comp, C)
    return _li_conv(xp, eps_comp, half_width_u, half_width_v,
                    *_segment_series_pair(xp, x, y, half_width_u, half_width_v, C,
                                          _device(xp, eps_comp)),
                    nesting)


def fourier2d_piecewise_constant(xp, cell, half_width_u, half_width_v, *, inverse_rule,
                                 nesting=None):
    """2D Fourier series of a grid whose values *hold across* their pixels, with Li's rules.

    A permittivity raster is this case: ``structure`` names the material occupying each cell, so
    the layer is a staircase and its Fourier coefficients have an exact closed form, which
    :func:`_pixel_correction` supplies. Exact for any order, including past the point where
    the DFT itself wraps — which is why this path needs no grid floor.

    ``inverse_rule`` names the direction whose discontinuity the inverse rule closes, or
    ``None`` for the direct rule both ways. Those three are Li's three components of one
    layer: ``None`` builds ``epz`` (before it is inverted), ``'v'`` builds ``epy``, ``'u'``
    builds ``epx``. ``nesting`` is the order the rules are nested in
    (:class:`~wavebit._core.enums.Nesting`, ``None`` meaning ``'fixed'``); it changes ``epx``
    only. The operators are tabulated in :func:`_li_slot`.
    """
    return _li_slot(xp, cell, half_width_u, half_width_v, *_pixel_series_pair(
        xp, half_width_u, half_width_v, _pixel_correction), inverse_rule, nesting)


def fourier2d_pointwise(xp, field, half_width_u, half_width_v):
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

    Because there is no closed form, this path *does* need ``4*truncation_order+1`` points — the caller
    lifts the grid to that floor, see :func:`_normal_field_floor`.

    It survived the 2026-09-25 rewrite of the Li builders to the bit on NumPy and PyTorch: the
    NVF projector is built from it, so :func:`_raster_series` keeps the arithmetic order it
    always had. JAX does not give the same bits for the same arithmetic composed differently:
    measured 2026-09-26 against the previous builders on 33 grids (JAX 0.11.0, CPU), 14 moved
    by round-off -- at most 1.7e-16 relative in double precision and one single-precision ulp
    in ``complex64`` -- and none of them a one-pixel or single-row grid.
    """
    return _conv_matrix(xp, field, half_width_u, half_width_v, *_pixel_series_pair(
        xp, half_width_u, half_width_v, _half_pixel_phase))


def _device(xp, x):
    """Device of ``x`` (Array-API ``device`` attribute), for placing index arrays alongside."""
    return x.device if hasattr(x, 'device') else None


def _layer_eps(xp, structure):
    """Yield each layer's permittivity grid, on the raster exactly as given.

    **Cast, and nothing else.** The raster already holds relative permittivity, so this widens
    it to the working complex dtype and hands it on. There used to be an ``n ** 2`` here, and
    its removal is one of the sites the convention flip is *made of*: a squaring left behind
    would not raise, it would quietly solve a medium of ``eps**2`` -- legal, energy-conserving,
    and wrong by the square (``docs/conventions.md``).

    Complex either way, and untouched otherwise: Wavebit works in ``e^{-iwt}`` throughout
    (``Im(eps) > 0`` = loss), with no conjugation on the way in.

    Nothing is resampled. :func:`fourier2d_piecewise_constant` obtains the raster's
    coefficients in closed form (:func:`_pixel_correction`), so a finer grid of the same
    staircase is the same
    function and buys nothing. The NVF path is the one that still needs a floor, and states
    that itself — see :func:`_layer_eps_for_normal_field`.
    """
    for i in range(structure.shape[0]):
        layer = structure[i]
        yield xp.astype(layer, _complex_dtype(xp, layer))


def check_normal_grid(grid_shape, half_width_u, half_width_v):
    """``4*truncation_order+1`` points per axis, or reading the field off the grid aliases.

    The same floor :func:`_normal_field_floor` lifts a raster to, stated as a refusal for the
    paths that take the grid as a *number* instead of lifting one: a raster can be repeated up
    to the floor without changing what it describes, and a sampling resolution cannot.

    Structural, not gradual: a *constant* normal field has to transform to the identity for NVF
    to collapse onto Li's rules, and on a short grid it picks up off-diagonals at ``+/-n``
    instead.

    It lives here rather than beside either caller because both shape paths need it and
    :mod:`.boundary` imports :mod:`.parametric`, so the parametric one could not reach it there.
    Until 2026-08-27 only the boundary route called it, and the asymmetry was not a decision:
    at ``truncation_order=10`` the boundary route refused ``shape_nvf_grid_size=8`` while the parametric
    path built it, 4.9e-4 adrift and conserving energy to twelve digits.
    """
    Ny, Nx = grid_shape
    need_v, need_u = 4 * half_width_v + 1, 4 * half_width_u + 1
    if Ny < need_v or Nx < need_u:
        raise ValueError(
            f'normal-field grid {Ny}x{Nx} aliases at truncation_order=({half_width_u}, '
            f'{half_width_v}): needs at least {need_v}x{need_u}. The normal field has no '
            f'closed-form transform, so it is read off this grid, and the circulant asks for '
            f'orders up to +/-2*truncation_order -- below that, k % n folds distinct orders '
            f'onto one bin and a constant field stops transforming to the identity.')


def _normal_field_floor(shape, half_width_u, half_width_v):
    """Repeat factors that lift a layer to the grid the normal field's transform needs.

    The normal field is a continuous vector field this code evaluates on a grid, so unlike the
    permittivity it has no closed-form transform and :func:`fourier2d_pointwise` reads it off
    the grid directly. That requires ``4*truncation_order+1`` points: the circulant asks for orders up
    to ``+/-2*truncation_order``, and on a shorter grid ``k % n`` folds distinct orders onto one bin.
    The damage is structural rather than gradual — a *constant* normal field must transform to
    the identity for NVF to collapse onto Li's rules on a one-cell axis, and on a short grid it
    picks up off-diagonals at ``+/-n`` instead.

    Only up to that floor. Going past it was the old ``enhanced_dfs=True``, which multiplied it
    by the raster size; measured across 32 shape/material/polarization/order combinations it
    lost accuracy in 27 and gained in none that stood up (see ``docs/nvf-upsampling.md``).
    """
    return (max(1, (4 * half_width_v + 1) // shape[0] + 1) if shape[0] < 4 * half_width_v + 1 else 1,
            max(1, (4 * half_width_u + 1) // shape[1] + 1) if shape[1] < 4 * half_width_u + 1 else 1)


def _tensor_at_the_normal_field_floor(xp, t33, half_width_u, half_width_v):
    """A ``(Ny, Nx, 3, 3)`` pointwise tensor, block-repeated to the floor its field needs.

    The tensor twin of :func:`_layer_eps_for_normal_field`, and it exists because the two
    Popov-Neviere wrappers called :func:`normal_vector_field` directly and so skipped the lift
    every scalar NVF builder makes. On a **uniform** slab -- where the answer is exactly
    ``eps * I`` and no geometry is involved at all -- that put ``max|epx - 4I|`` at **2.9e+02**
    for a 8x8 raster at ``truncation_order=4``, against 0.000e+00 for the scalar path on the
    same raster. It is the aliasing :func:`_normal_field_floor` describes: the projector grids
    are read off this grid by :func:`fourier2d_pointwise`, and below ``4*truncation_order+1``
    points ``k % n`` folds distinct orders onto one bin.

    A repeat is not a resample. The tensor field it describes is unchanged -- the same
    staircase spelled with more pixels -- so the permittivity coefficients are identical and
    only the field's transform stops aliasing.
    """
    repeat_v, repeat_u = _normal_field_floor(tuple(t33.shape[:2]), half_width_u, half_width_v)
    if repeat_v > 1:
        t33 = xp.repeat(t33, repeat_v, axis=0)
    if repeat_u > 1:
        t33 = xp.repeat(t33, repeat_u, axis=1)
    return t33


def _layer_eps_for_normal_field(xp, structure, half_width_u, half_width_v):
    """Each layer's permittivity grid for the NVF path, with the base it was repeated from.

    Yields ``(eps, base_eps, repeat_v, repeat_u)``. The base is what
    :func:`_upsampled_interface_mask` derives the interface mask from when the values cannot be
    read whole, which is what lets NVF differentiate without a frozen geometry.

    Like :func:`_layer_eps` this only casts: the raster is permittivity already.
    """
    for i in range(structure.shape[0]):
        layer = structure[i]
        base = xp.astype(layer, _complex_dtype(xp, layer))
        repeat_v, repeat_u = _normal_field_floor(layer.shape, half_width_u, half_width_v)
        if repeat_v > 1:
            layer = xp.repeat(layer, repeat_v, axis=0)
        if repeat_u > 1:
            layer = xp.repeat(layer, repeat_u, axis=1)
        yield xp.astype(layer, _complex_dtype(xp, layer)), base, repeat_v, repeat_u


def to_conv_mat_raster_sampled(structure, half_width_u, half_width_v, nesting=None):
    """Per-layer convolution matrices for a raster ``structure`` via the sampled path.

    Args:
        structure: relative-permittivity raster, shape ``(n_layers, Ny, Nx)``. This is the
            design variable — gradients flow back to it. Permittivity, not an index: nothing
            here squares it (``docs/conventions.md``).
        half_width_u, half_width_v: Fourier truncation orders.
        nesting: the order Li's rules are nested in (:class:`~wavebit._core.enums.Nesting`;
            ``None`` means ``'fixed'``).

    Returns:
        ``(epx, epy, epz_i, None, None)``, each slot ``(n_layers, n_orders, n_orders)`` — one
        :func:`li_conv_matrices` per layer.
    """
    xp = array_namespace(structure)
    out = [[], [], []]
    for eps_matrix in _layer_eps(xp, structure):
        for slot, mat in zip(out, li_conv_matrices(xp, eps_matrix, half_width_u, half_width_v,
                                                   nesting)):
            slot.append(mat)
    return ConvMatrices(*(xp.stack(slot) for slot in out), None, None)


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


def _distance_gradient(interface_mask, shape, lattice=None):
    """Raw gradient of the periodic unsigned distance-to-interface map, in Cartesian components.

    Split out of :func:`normal_vector_field` because two different things read it. The
    direction is what the ``'distance'`` construction normalises into a normal field; the
    **magnitude** is ``|grad d|``, which is a property of the interface set rather than of any
    construction — 1 almost everywhere, dropping to 0 exactly on the medial axis, and the same
    array whichever of the four builders drew the directions. ``nvf_projector='circular'`` reads
    it beside every method (see :func:`normal_vector_field`'s ``with_magnitude``).

    Returns ``(g_x, g_y)`` **unnormalised**. Everything periodic about the construction lives
    here: the 3x3 tiling under the transform and the wrapping central difference. See
    :func:`normal_vector_field` for why each of those is load-bearing.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    ny_, nx_ = shape

    def _periodic_edt(sampling=None):
        """The distance transform on a 3x3 tiling, cropped back. A cell is periodic, so a
        pixel's nearest interface may sit in the next copy; the untiled transform measures
        to the in-cell one instead and the two differ near every wall.
        """
        tiled = np.tile(~interface_mask, (3, 3))
        full = (distance_transform_edt(tiled) if sampling is None
                else distance_transform_edt(tiled, sampling=sampling))
        return full[ny_:2 * ny_, nx_:2 * nx_]

    def _periodic_gradient(f):
        """Central differences that wrap. ``np.gradient`` is one-sided on the first and last
        row/column -- a different stencil there, and on its own enough to break covariance
        under a lattice translation even for a shape nowhere near the wall, because the
        distance map it differentiates is nonzero right up to the array edge.
        """
        return ((np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / 2.0,
                (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / 2.0)

    if lattice is None:
        dist_map = _periodic_edt()
        g_j, g_i = _periodic_gradient(dist_map)
    else:
        from .lattice import jacobian
        n_v, n_u = shape
        J = jacobian(lattice, n_v, n_u)
        try:
            scale = float(np.hypot(J[0, 0], J[1, 0]))
        except Exception:                   # noqa: BLE001 - a traced basis, not a mistake
            scale = None

        if scale is not None:
            # Work in units of one u-step: divide J through by |a1|/n_u before anything reads
            # it. A uniform scale cannot survive the normalisation at the end, so this changes
            # no direction -- but it makes a square-pixel raster land on exactly
            # `sampling=(1, 1)` and a `J^-T` of the identity, which is bit-identical to taking
            # no metric at all. Without it those cells moved by ~7e-15 purely from multiplying
            # a factor in and dividing it back out.
            J = J / (scale or 1.0)
            # Physical length of one index step along each axis: axis 0 walks a2/n_v, axis 1
            # a1/n_u. Exact for an axis-aligned cell; for a sheared one it is the best
            # `sampling` can say, since scipy's transform has no way to express a non-diagonal
            # metric. The direction that survives normalisation is set by the gradient below,
            # which is exact regardless.
            step_j = float(np.hypot(J[0, 1], J[1, 1]))
            step_i = float(np.hypot(J[0, 0], J[1, 0]))
            if step_j == step_i == 1.0:
                dist_map = _periodic_edt()
            else:
                dist_map = _periodic_edt(sampling=(step_j, step_i))
            g_j, g_i = _periodic_gradient(dist_map)
            # grad_xy f = J^-T @ (df/di, df/dj) -- the inverse transpose, not the Jacobian:
            # index steps push forward, gradients pull back.
            Jinv_T = np.linalg.inv(J).T
            g_i, g_j = (Jinv_T[0, 0] * g_i + Jinv_T[0, 1] * g_j,
                        Jinv_T[1, 0] * g_i + Jinv_T[1, 1] * g_j)
        else:
            # A **traced** basis. Only one thing here is genuinely discrete: *which* interface
            # pixel is nearest. The distance to it is not -- it is
            # ``hypot(step_j*dj, step_i*di)``, smooth in the metric, and its gradient is what
            # sets the normal direction. So the callback returns the index displacement and
            # the distance is rebuilt here, where the steps are traced.
            #
            # Returning the distance itself and stopping the gradient there looks equivalent
            # and is not: it drops the metric's smooth half. Measured, that cost ~10% of
            # ``d/d a2x`` on a hex cell, and the error did not shrink with raster resolution,
            # which is what said it was systematic rather than quantisation.
            import jax

            def _index_offsets(Jc):
                """``(dj, di)`` to the nearest interface pixel, on the 3x3 periodic tiling."""
                Jc = np.asarray(Jc, dtype=float)
                Jc = Jc / (float(np.hypot(Jc[0, 0], Jc[1, 0])) or 1.0)
                sj = float(np.hypot(Jc[0, 1], Jc[1, 1]))
                si = float(np.hypot(Jc[0, 0], Jc[1, 0]))
                tiled = np.tile(~interface_mask, (3, 3))
                _, idx = distance_transform_edt(tiled, sampling=(sj, si),
                                                return_indices=True)
                jj, ii = np.meshgrid(np.arange(3 * ny_), np.arange(3 * nx_), indexing='ij')
                core = (slice(ny_, 2 * ny_), slice(nx_, 2 * nx_))
                return ((idx[0] - jj)[core].astype(float),
                        (idx[1] - ii)[core].astype(float))

            spec = jax.ShapeDtypeStruct((ny_, nx_), np.float64)
            # `stop_gradient` states what is true: no tangent belongs on a choice of pixel.
            dj, di = jax.pure_callback(_index_offsets, (spec, spec),
                                       jax.lax.stop_gradient(J))
            xp_ = array_namespace(J)
            scale_ = xp_.sqrt(J[0, 0] ** 2 + J[1, 0] ** 2)
            a_, b_ = J[0, 0] / scale_, J[0, 1] / scale_
            c_, d_ = J[1, 0] / scale_, J[1, 1] / scale_
            step_j_ = xp_.sqrt(b_ ** 2 + d_ ** 2)
            step_i_ = xp_.sqrt(a_ ** 2 + c_ ** 2)
            # Guard inside the sqrt: an on-interface pixel has zero displacement, and
            # `d(sqrt)/dx` at zero is infinite whatever a `where` outside it says.
            sq_ = (step_j_ * dj) ** 2 + (step_i_ * di) ** 2
            dmap_ = xp_.sqrt(xp_.where(sq_ != 0, sq_, xp_.ones_like(sq_)))
            dmap_ = xp_.where(sq_ != 0, dmap_, xp_.zeros_like(dmap_))
            g_j = (xp_.roll(dmap_, -1, axis=0) - xp_.roll(dmap_, 1, axis=0)) / 2.0
            g_i = (xp_.roll(dmap_, -1, axis=1) - xp_.roll(dmap_, 1, axis=1)) / 2.0
            det_ = a_ * d_ - b_ * c_
            g_i, g_j = ((d_ * g_i - c_ * g_j) / det_, (-b_ * g_i + a_ * g_j) / det_)
    return g_i, g_j


def normal_vector_field(eps_matrix, interface_mask=None, lattice=None, method=None,
                        base_shape=None, with_magnitude=False):
    """Unit normal vector field ``(normal_x, normal_y)`` of a raster layer.

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

    ``base_shape`` is the layer's shape before the ``4*truncation_order+1`` floor lift, when the caller
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
    :func:`wavebit._core.parametric.normal_vector_field_analytical`.

    ``with_magnitude=True`` appends a third array: ``|grad d|`` of the periodic distance map
    (:func:`_distance_gradient`), scaled so its maximum over the cell is exactly 1. It is the
    *same* quantity for every ``method`` — the direction is what the four constructions
    disagree about, while the magnitude is a property of the interface set — so the smoothing
    constructions get it from a distance transform run beside them rather than from their own
    output, which is a unit field by construction and carries no magnitude to read.

    Only ``nvf_projector='circular'`` asks for it (:class:`~wavebit._core.enums.Projector`), and
    only that reader makes the scaling meaningful: ``phi = (pi/8)(1 + cos(pi|t|))`` is
    monotone in ``|t|`` on ``[0, 1]`` and turns back up outside it, so *unit maximum length*
    is the formula's domain rather than a normalisation anyone chose. A one-cell axis and a
    layer with no interface at all both answer 1 everywhere, which is what makes the circular
    projector reduce to ``n n^T`` exactly where the direction was never in doubt.

    **Everything here is periodic, and all three pieces of it have to be.** The cell is a torus,
    so rolling the raster by whole pixels is an exact lattice translation — the same structure
    named differently — and every efficiency must come back unchanged. It did not: the mask
    compared only inside the array, the distance transform measured to the in-cell interface
    rather than the periodically nearest one, and ``np.gradient`` is one-sided on the first and
    last row and column. Together they moved ``max|dR|`` by ~1e-2 at every ``truncation_order``, and no
    subset of the three fixes it (fixed 2026-08-17; ``docs/nvf-normal-methods.md`` has the
    table, ``tests/test_nvf_lattice_translation.py`` the guard). The defect did not shrink with
    ``truncation_order`` because the floor lift is a block repeat, so a finer grid is the same staircase.

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

    The other three constructions carry the basis too, since 2026-08-21 — the paragraph above
    describes this body, not the option. Two of them need no pullback at all (they never take
    an index-space difference), which is why they were portable in the first place and why
    they are the better field on a skewed cell than this one:
    :mod:`wavebit._core.normal_methods` says which is which, and ``docs/oblique-lattice.md``
    what it is worth (18.5 degrees against 0.42 on a disc at a 60-degree skew).
    """
    import numpy as np
    from .enums import NormalMethod, normal_method_of

    shape = tuple(eps_matrix.shape)
    method = NormalMethod.DISTANCE if method is None else normal_method_of(method)

    # A one-cell axis carries no interface of its own: the structure cannot vary along it, so
    # the normal is constant. Say so directly -- `np.gradient` needs three cells and would
    # otherwise raise, which is what made NVF unusable on 1D gratings. With a constant normal
    # the NVF projector reduces to Li's rules exactly (C_xx = I, C_xy = C_yy = 0 gives
    # eps_xx = <1/eps>^-1 and eps_yy = <eps>), so this is not an approximation.
    # Shared across methods: nothing about the constant field depends on the construction.
    #
    # *Which* constant is the part a skewed cell changes. A raster invariant along `a2` has
    # interfaces that contain `a2`, so the normal is along `b1` -- perpendicular to `a2`, not
    # parallel to `a1` -- and only on an axis-aligned cell are those the same direction. This
    # is the shortcut's version of the `J^-T` pullback the general path below applies: with
    # `df/dj == 0`, `J^-T @ (df/di, 0)` is the first row of `J^-1`, which is `b1` up to the
    # scale normalisation removes. Returning `x_hat` there instead cost R 0.132 and T 0.090 on
    # a 50% binary grating at a 60-degree skew, energy conserving throughout
    # (`tests/test_oblique_lattice.py::test_a_one_row_skewed_cell_is_the_grating_it_draws`).
    # An axis-aligned cell gives back exactly (1.0, 0.0) / (0.0, 1.0), to the bit.
    if 1 in shape:
        if lattice is None:
            ux, uy = (1.0, 0.0) if shape[0] == 1 else (0.0, 1.0)
        else:
            recip = np.linalg.inv(np.asarray(lattice, dtype=float))
            b = recip[0] if shape[0] == 1 else recip[1]
            ux, uy = b / np.hypot(b[0], b[1])
        # Magnitude 1, i.e. no circular deformation. A one-cell axis carries no interface of
        # its own, so this constant field is the exact normal everywhere and NVF reduces to
        # Li's rules; opening it into an ellipse would trade an exact answer for an
        # approximate one at the stripe centres, which are the only null `|grad d|` has here.
        field = (np.full(shape, ux), np.full(shape, uy))
        return field + (np.ones(shape),) if with_magnitude else field

    if method != NormalMethod.DISTANCE:
        from .normal_methods import DISPATCH
        try:
            eps = _to_numpy(eps_matrix)
        except Exception:
            raise TypeError(
                f"raster_normal_method={method.value!r} reads the layer's values (its indicator "
                f"needs to know which phase is which), and this layer is a tracer without "
                f"concrete ones. Precompute the field eagerly with freeze_geometry() and "
                f"hand it to solve(geometry=...), or use raster_normal_method='distance', "
                f"whose mask-only path a trace can take.") from None
        mask = (_interface_mask(eps) if interface_mask is None
                else np.asarray(interface_mask, dtype=bool))
        field = tuple(DISPATCH[method.value](eps, mask, lattice, base_shape))
        if not with_magnitude:
            return field
        # Beside the construction, not out of it: every builder here returns a *unit* field,
        # so there is no magnitude in its output to read. The interface set is what carries
        # one, and all four constructions share it.
        return field + (_field_magnitude(*_distance_gradient(mask, shape, lattice)),)

    if interface_mask is None:
        interface_mask = _interface_mask(_to_numpy(eps_matrix))

    gx, gy = _distance_gradient(interface_mask, shape, lattice)
    # Off the *raw* gradient, before the flat-pixel floor below rewrites it: the floor's whole
    # job is to keep a direction finite where there is none, and the magnitude is the number
    # that says so.
    magnitude = _field_magnitude(gx, gy) if with_magnitude else None
    if isinstance(gx, np.ndarray):
        norm = np.hypot(gx, gy)
        norm[norm < _FLAT_PIXEL] = 1.0
    else:
        # The guard goes *inside* the sqrt. `d(sqrt)/dx` is infinite at zero, and a `where`
        # outside still differentiates the branch it did not take, so the NaN comes back --
        # the same trap `shape_ft`'s DC term documents for `jinc`.
        #
        # And it is a *floor*, not `!= 0`. This used to read "every pixel with no gradient sits
        # at exactly zero, so this is not a rare corner", which is true only while the two
        # sampling steps are the literal 1 of index space. Give it the cell's real steps -- what
        # a skewed cell does -- and a pixel equidistant from two interfaces cancels to 8.9e-16
        # instead of 0, passes `!= 0`, and contributes `1/1e-15` to the gradient. Measured on
        # `hexagonal(400)` at 48x48 with a 150 nm rod, which is the shipped test's own scene:
        # `d(sum R)/d a1x` came out -2.70e+08 where the loss's own slope over a 4 nm window is
        # -8.12e-04. Eleven orders, finite, and past a test that asserted only `isfinite`.
        #
        # It needs `|a1| == |a2|` and a square raster for the cancellation to be exact enough,
        # so it never appeared through `period` and only ever on a hexagon. `|grad d|` is O(1)
        # at every real interface -- 1st percentile 4.2, median 8.3, in units of the step -- so
        # a floor nine orders below that separates the two populations with nothing in between.
        _xp = array_namespace(gx)
        sq = gx ** 2 + gy ** 2
        nz = sq > _FLAT_PIXEL ** 2
        norm = _xp.sqrt(_xp.where(nz, sq, _xp.ones_like(sq)))
        gx = _xp.where(nz, gx, _xp.zeros_like(gx))
        gy = _xp.where(nz, gy, _xp.zeros_like(gy))
    unit = (gx / norm, gy / norm)
    return unit + (magnitude,) if with_magnitude else unit


def _field_magnitude(gx, gy):
    """``|grad d|`` scaled to unit maximum over the cell — the ``|t|`` of Liu & Fan Eq. (52).

    The scale is not a taste call. ``phi = (pi/8)(1 + cos(pi|t|))`` is monotone only while
    ``|t| <= 1``: past 1 the cosine turns back up and a *stronger* field would be reported as
    a more circular one. Unit maximum length is what the formula is defined on, and it is
    what the paper states.

    A layer with no interface at all has a flat distance map and answers 1 everywhere, which
    keeps ``nvf_projector='circular'`` exactly on ``n n^T`` there — the direction is ``(0, 0)``
    on such a layer, so the linear projector is 0 and the factorization is the plain direct
    rule, which is the exact answer for a uniform layer and must not be deformed into
    something else.
    """
    xp = array_namespace(gx)
    m = xp.sqrt(gx ** 2 + gy ** 2)
    peak = xp.max(m)
    # A `where` rather than a Python branch: on the traced-basis path `peak` is a tracer, and
    # the guard has to keep the division defined without asking what its value is.
    return xp.where(peak > 0, m / xp.where(peak > 0, peak, xp.ones_like(peak)),
                    xp.ones_like(m))


#: Below this, a distance-map gradient is a flat pixel rather than an interface. See the
#: floor's own note in `normal_vector_field`: real interfaces sit at O(1) in step units.
_FLAT_PIXEL = 1e-9


def _interface_mask(eps):
    """Cells that differ from a 4-neighbour — where the material interfaces sit.

    The neighbour relation **wraps**: the cell is periodic, so the last row's neighbour is
    the first row. Comparing only inside the array misses the interface across the wall, and
    then the whole construction stops being covariant under a lattice translation of the
    raster — a shape rolled across the boundary loses the mask on one of its faces.
    """
    import numpy as np
    border = np.zeros(eps.shape, dtype=bool)
    for axis in (0, 1):
        differs = eps != np.roll(eps, 1, axis=axis)
        border |= differs
        border |= np.roll(differs, -1, axis=axis)
    return border


def _upsampled_interface_mask(xp, base_eps, repeat_v, repeat_u):
    """The interface mask of a block-repeated grid, from the base grid's neighbour comparisons.

    :func:`_layer_eps_for_normal_field` lifts a layer to the floor by repeating each cell, so
    the repeated grid is constant inside every block. Two neighbouring cells of it can therefore
    only differ where they straddle a block boundary *and* the base differs across it — which
    means the whole mask follows from ``(base_ny-1)*base_nx + base_ny*(base_nx-1)`` comparisons
    on the base, rather than one read of the repeated grid.

    That is the whole point. Reading the repeated grid means converting it to NumPy in one
    step, which a tracer refuses; narrowing the base's comparisons one value at a time is
    something a grad tracer allows, and there are ``repeat_v * repeat_u`` times fewer of them —
    128x128 rather than 2688x2688 at ``truncation_order=5``. It is what lets NVF differentiate with no
    frozen geometry. (Under ``jax.jit`` neither works: there is no primal to narrow.)

    Note this is *not* the base's own mask repeated. Repeating ``[1, 2]`` twice gives
    ``[1, 1, 2, 2]``, whose border cells are the middle two, not all four.
    """
    import numpy as np

    base_ny, base_nx = tuple(base_eps.shape)
    border = np.zeros((base_ny * repeat_v, base_nx * repeat_u), dtype=bool)

    # The wrap pair is the last base row/column against the first: on a periodic cell that is
    # a neighbour relation like any other, and leaving it out is what made the field
    # non-covariant under a lattice translation (see :func:`_interface_mask`). It marks the
    # last row of the last block and the first row of the first block.
    if base_ny > 1:
        differs = _as_bools(xp, base_eps[:-1, :] != base_eps[1:, :])
        differs = np.repeat(differs, repeat_u, axis=1)
        last_of_block = np.arange(base_ny - 1) * repeat_v + repeat_v - 1
        border[last_of_block, :] |= differs
        border[last_of_block + 1, :] |= differs

        wrap = np.repeat(_as_bools(xp, base_eps[-1, :] != base_eps[0, :]), repeat_u)
        border[-1, :] |= wrap
        border[0, :] |= wrap

    if base_nx > 1:
        differs = _as_bools(xp, base_eps[:, :-1] != base_eps[:, 1:])
        differs = np.repeat(differs, repeat_v, axis=0)
        last_of_block = np.arange(base_nx - 1) * repeat_u + repeat_u - 1
        border[:, last_of_block] |= differs
        border[:, last_of_block + 1] |= differs

        wrap = np.repeat(_as_bools(xp, base_eps[:, -1] != base_eps[:, 0]), repeat_v)
        border[:, -1] |= wrap
        border[:, 0] |= wrap

    return border


def _interface_mask_of(xp, eps, base_eps, repeat_v, repeat_u):
    """The interface mask of ``eps``, by whichever route the backend allows.

    Straight off the repeated grid when its values are concrete — the same read as before, and
    the cheaper one. From the base grid when they are not, which is the case a gradient
    presents.
    """
    try:
        return _interface_mask(_to_numpy(eps))
    except Exception:                     # a tracer: derive it from the base instead
        return _upsampled_interface_mask(xp, base_eps, repeat_v, repeat_u)


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
    construction (``raster_normal_method``). Two layers with the same source have the same normal
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
    indicators mark the smaller REGION, found from the corner, so neither relabeling the
    values nor translating the raster moves anything — see
    ``normal_methods._two_phase_indicator``), and the number of distinct
    phases (which decides the delegation to ``'distance'``, and which a merge of two
    non-adjacent phases can change without moving a single interface — hence the fifth
    element). The pre-lift ``base_shape`` is not captured because it follows from the
    shape the record already pins plus the solver's own ``truncation_order``.
    """
    import numpy as np
    from .enums import NormalMethod, normal_method_of
    method = (NormalMethod.DISTANCE if method is None else normal_method_of(method)).value
    if 1 in eps.shape:
        return eps.shape, None, _lattice_key(lattice), method, None
    return (eps.shape, _interface_mask(eps), _lattice_key(lattice), method,
            int(np.unique(eps).size))


def normal_field_for_raster(structure, half_width_u, half_width_v, lattice=None, method=None,
                            nvf_projector=None):
    """Per-layer ``(normal_x, normal_y)`` normal fields for ``structure``, on the grid the NVF build uses.

    Precompute this once from the concrete geometry and hand it to
    :func:`to_conv_mat_raster_nvf` as ``normal_field=`` to keep the build traceable under a
    JAX transform (none of the constructions can run on tracers).

    ``lattice`` is the cell basis and ``method`` the construction; see
    :func:`normal_vector_field` for what each changes. ``nvf_projector='circular'`` makes each
    entry a **triple** — the magnitude the complex-basis projector reads is part of the field
    the solve is handed, so it has to be part of what is frozen; a record without it is
    refused rather than served with the other projector (:func:`_unpack_normal_field`).
    """
    xp = array_namespace(structure)
    want = _wants_magnitude(nvf_projector)
    return [normal_vector_field(eps, lattice=lattice, method=method,
                                base_shape=tuple(base.shape), with_magnitude=want)
            for eps, base, _, _ in _layer_eps_for_normal_field(xp, structure, half_width_u, half_width_v)]


def normal_field_source_for_raster(structure, half_width_u, half_width_v, lattice=None, method=None):
    """What :func:`normal_field_for_raster` derived its answer from, per layer.

    Freeze this alongside the normal field and hand both back to
    :func:`to_conv_mat_raster_nvf`; the replay compares it against the layer actually in front
    of it and refuses a field that was sampled from a different one — a different pattern, a
    different basis, or a different ``raster_normal_method``.
    """
    xp = array_namespace(structure)
    return [_normal_field_source(_to_numpy(eps), lattice, method)
            for eps, _, _, _ in _layer_eps_for_normal_field(xp, structure, half_width_u, half_width_v)]


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
        moved = (f'the normal field was built with raster_normal_method={meth!r} and this solver '
                 f'is configured raster_normal_method={now_meth!r} — same interfaces, different '
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
        f'(Changing the permittivity of a layer never trips this; moving its pattern '
        f'does.)')


def jones_projector_fields(normal_x, normal_y, magnitude):
    """Pointwise ``(P_xx, P_xy, P_yx, P_yy)`` of the **complex-basis** projector.

    The projector NVF has always used is ``P = n n^T``, built from the field's *direction*
    alone. On the medial axis — the ridge where every boundary point is equidistant — there is
    no direction: the tie is exact, both answers are arbitrary, and which one is reported is
    an accident of where the samples fell. No better *direction* fixes that, because none
    exists.

    What does is a projector that stops reading one. Build a Jones vector from the field

    .. code-block:: text

        J   = n_hat cos(phi) + i n_perp sin(phi),    n_perp = (-n_y, n_x) / |n|
        phi = (pi/8) (1 + cos(pi |t|))

    with ``|t|`` the field magnitude scaled to unit maximum (:func:`_field_magnitude`), and
    take ``P = J J^H``. Then

    ==============  ==========================================  =========================
    ``|t| = 1``     ``phi = 0``, ``J = n_hat``                   ``P = n n^T``, exactly
    ``|t| -> 0``    ``phi -> pi/4``, ``J`` exactly circular      ``P_xx = P_yy = 1/2``,
                                                                ``P_xy = -i/2``,
                                                                ``P_yx = +i/2``
    ==============  ==========================================  =========================

    The second column is the whole point: at ``phi = pi/4`` the direction cancels out of every
    entry, so the answer on the medial axis is the same whatever arbitrary direction was
    reported there. And it is reached as a **limit** — no threshold, no "is this pixel on the
    medial axis?" test, nothing to tune. In between, ``P`` deforms continuously from one to
    the other, and ``trace(P) = 1`` throughout, which is what keeps ``C_xx + C_yy = I`` and
    :func:`_nvf_assemble` unchanged.

    ``P`` is Hermitian and rank one at every pixel, so ``P_yx = conj(P_xy)`` and the two stop
    being the same object — the first time ``eps_xy`` and ``eps_yx`` genuinely differ on this
    path.

    Written from the published formula:

    * R. Antos, *Fourier factorization with complex polarization bases in modeling optics of
      discontinuous bi-periodic structures*, Opt. Express **17**, 7269 (2009) — the complex
      polarization basis itself, and the origin of this construction.
    * V. Liu, S. Fan, *S4: A free electromagnetic solver for layered periodic structures*,
      Comput. Phys. Commun. **183**, 2233 (2012), **Eq. (52), p. 2239** — its generalization,
      and the form implemented here. S4 states ``J`` in terms of the *tangent* field ``t``;
      the two differ by a fixed 90-degree rotation, so building ``J`` from the normal
      directly gives ``R P R^-1`` of theirs, which is exactly the normal projector this
      module's ``C_xx = <P_xx>`` wants. ``e^{i theta}`` in Eq. (52) is a global phase and
      cancels out of ``J J^H``; it is not carried here.

    ``docs/nvf-normal-methods.md`` records what it is worth and what it does not fix.
    """
    xp = array_namespace(normal_x)
    # `pi` from the namespace is not portable across the three; a float literal is exact in
    # binary64 and this is multiplied, never compared.
    pi = 3.141592653589793
    phi = (pi / 8.0) * (1.0 + xp.cos(pi * magnitude))
    cos_p, sin_p = xp.cos(phi), xp.sin(phi)

    # The algebra below needs a *unit* direction, and it takes one: every construction returns
    # a unit field (that is what makes `C_xx + C_yy = I`), with `(0, 0)` at the flat pixels
    # where `normal_vector_field`'s floor says there is no direction to report. So the only
    # thing to do here is fill those in -- and **not** re-normalise the rest. Dividing an
    # already-unit field by its own length is a no-op in arithmetic and not in floating point,
    # and it would cost the property that makes this a deformation of the shipped scheme
    # rather than a rival to it: at `|t| = 1` the result is `n n^T` *to the bit*.
    #
    # Any unit vector is exact at a flat pixel. `magnitude` is 0 at precisely those pixels, so
    # `phi` is `pi/4` and the direction has cancelled out of every entry -- which is the whole
    # construction, applied to the one place it was built for. `x_hat` is the arbitrary
    # choice, and its arbitrariness is what is being made harmless.
    live = xp.sqrt(normal_x ** 2 + normal_y ** 2) > _FLAT_PIXEL
    n_x = xp.where(live, normal_x, xp.ones_like(normal_x))
    n_y = xp.where(live, normal_y, xp.zeros_like(normal_y))

    cc, ss = cos_p ** 2, sin_p ** 2
    p_xx = n_x ** 2 * cc + n_y ** 2 * ss
    p_yy = n_y ** 2 * cc + n_x ** 2 * ss
    real = n_x * n_y * (cc - ss)
    imag = cos_p * sin_p                       # == sin(2 phi) / 2
    return p_xx, real - 1j * imag, real + 1j * imag, p_yy


def _wants_magnitude(nvf_projector):
    """Whether ``nvf_projector`` asks the field for a magnitude. ``None`` is the default."""
    from .enums import DEFAULT_PROJECTOR, Projector, projector_of
    resolved = DEFAULT_PROJECTOR if nvf_projector is None else projector_of(nvf_projector)
    return resolved == Projector.CIRCULAR


def _unpack_normal_field(entry, want_magnitude, index):
    """``(normal_x, normal_y, magnitude)`` from a field the builder made or a record it replays.

    A frozen record carries a magnitude only if it was taken under ``nvf_projector='circular'``, so
    its *arity* is the tell — and a two-entry record replayed under the circular projector is
    refused rather than quietly served with the linear one. That is the same rule
    :func:`_check_normal_field_still_fits` applies to the construction: a record built under
    one configuration and replayed under another describes a field the configuration never
    asked for, and silently answering with the other projector is exactly the unread knob this
    library refuses everywhere else.

    The other direction is not an error: a record that carries a magnitude replays fine under
    ``'linear'``, which simply does not read it.
    """
    if not want_magnitude:
        return entry[0], entry[1], None
    if len(entry) < 3:
        raise StaleGeometryError(
            f"the frozen geometry no longer fits this solver: layer {index}'s normal field "
            f"was frozen under nvf_projector='linear', which records a direction and no "
            f"magnitude, and this solver is configured nvf_projector='circular', which needs "
            f'one. Call freeze_geometry() again with the projector this solve will use, or '
            f'drop geometry= to rebuild.')
    return entry[0], entry[1], entry[2]


def _nvf_projector(xp, normal_x, normal_y, half_width_u, half_width_v, dtype, device=None,
                   magnitude=None):
    """Convolution matrices of the normal projector: ``C_xx = <normal_x^2>`` etc.

    Always sampled off the raster the normal field lives on, whatever
    ``fourier_coefficients`` says. That keyword describes how the *permittivity* series is
    obtained — the part that is a design variable and that staircasing costs accuracy on. The
    normal field is static geometry (see :func:`normal_vector_field`): it is a smooth
    extension of the interface direction into the bulk, so it has no piecewise-constant block
    structure for the closed-form series to integrate over in the first place.

    ``magnitude`` selects the projector, and its absence is the default. Left ``None`` this is
    ``P = n n^T`` and returns **three** matrices, ``C_yx`` being ``C_xy`` — one transform
    saved, and the arithmetic downstream is the object-for-object identical one it has always
    been. Given a magnitude it is :func:`jones_projector_fields` and returns **four**, because
    a complex ``P`` has ``C_yx = C_xy^H`` and the two stop coinciding. Both arities splat
    straight into :func:`_nvf_assemble`, whose fourth projector argument defaults to "the same
    as the third".
    """
    def series(field):
        return fourier2d_pointwise(xp, xp.asarray(field, dtype=dtype, device=device),
                                   half_width_u, half_width_v)

    if magnitude is None:
        C_xy = series(normal_x * normal_y)
        return series(normal_x ** 2), C_xy, series(normal_y ** 2)

    p_xx, p_xy, p_yx, p_yy = jones_projector_fields(normal_x, normal_y, magnitude)
    return series(p_xx), series(p_xy), series(p_yy), series(p_yx)


def _nvf_assemble(xp, ep_dir, ep_inv, C_xx, C_xy, C_yy, C_yx=None):
    """Close the NVF factorization from the direct/inverse-rule matrices and the projector.

    With ``E = <eps>`` (direct rule), ``A = <1/eps>^-1`` (inverse rule) and ``Delta = E - A``::

        eps_xx = E - 1/2 (Delta @ C_xx + C_xx @ Delta)
        eps_yy = E - 1/2 (Delta @ C_yy + C_yy @ Delta)
        eps_xy = eps_yx = -1/2 (Delta @ C_xy + C_xy @ Delta)

    The 1/2-symmetrization is essential: the plain product ``Delta @ C`` is not Hermitian,
    giving complex ``q^2`` and gross energy-conservation violation on curved boundaries.

    ``C_yx`` left ``None`` says the projector is symmetric, which is what ``P = n n^T`` is and
    what this path assumed outright until ``nvf_projector='circular'`` (see
    :func:`jones_projector_fields`). It is a real statement about the projector rather than a
    convenience: with it, ``eps_yx`` is not merely equal to ``eps_xy`` but *the same object*,
    so the default path does one matmul pair rather than two and cannot drift from itself.

    ``ep_dir`` / ``ep_inv`` are the only inputs that depend on how the coefficients are
    obtained, which is what lets the sampled and integrated builders share this step.

    Returns the five matrices in the slot convention the 2D solve expects:
    ``(epx, epy, epz_i, epxy, epyx) = (eps_xx, eps_yy, inv(<eps>), -eps_xy, -eps_yx)`` --
    ``solve_conv``'s parameter order; see "Slot order" in this module's docstring.
    """
    delta_ep = ep_dir - xp.linalg.inv(ep_inv)

    eps_xx = ep_dir - 0.5 * (delta_ep @ C_xx + C_xx @ delta_ep)
    eps_yy = ep_dir - 0.5 * (delta_ep @ C_yy + C_yy @ delta_ep)
    eps_xy = -0.5 * (delta_ep @ C_xy + C_xy @ delta_ep)
    eps_yx = eps_xy if C_yx is None else -0.5 * (delta_ep @ C_yx + C_yx @ delta_ep)

    # Negated once and handed to both slots when the projector is symmetric, so that the two
    # really are one matrix rather than two copies of one. Spelled this way -- rather than by
    # folding the sign into the `0.5` above, which is the tidier-looking version -- because
    # that fold moves the *sign of a zero* on entries that are exactly zero. Numerically
    # nothing (0 differing elements over 1.57e6), but it breaks a byte-for-byte comparison
    # against the pre-change tree, and that comparison is the evidence that the default path
    # is untouched.
    slot_xy = -eps_xy
    slot_yx = slot_xy if eps_yx is eps_xy else -eps_yx

    return ConvMatrices(eps_xx, eps_yy, xp.linalg.inv(ep_dir), slot_xy, slot_yx)


def _pn_fields(xp, eps33, normal_x, normal_y):
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

    and the factorized tensor is ``[[A]] [[B]]^-1`` — see :func:`pn_eps_conv_matrices`. Every
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
    n = xp.astype(xp.stack([normal_x, normal_y, xp.zeros_like(normal_x)], axis=-1), dtype)
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
    a = xp.where(normal_x ** 2 + normal_y ** 2 == 0, xp.ones_like(a), a)[..., None, None]

    A = (eps33 - p[..., :, None] * q[..., None, :] / a) @ T_proj\
        + p[..., :, None] * n[..., None, :] / a
    B = (eye - n[..., :, None] * q[..., None, :] / a) @ T_proj\
        + n[..., :, None] * n[..., None, :] / a
    return A, B


def pn_eps_conv_matrices(xp, eps33, normal_x, normal_y, half_width_u, half_width_v):
    """Popov--Neviere factorization of a full constitutive tensor field.

    Form the continuous mixed field pointwise, Fourier transform A and B, then
    close all coupled components with [[A]] @ inv([[B]]). Return nine slots
    when z couples to the transverse field, otherwise the five-slot fast path.
    Epsilon and permeability use the identical interface continuity algebra.
    """
    from .anisotropy import full_tensor, tensor_slots
    size = 3 if full_tensor(xp, eps33) else 2
    A, B = _pn_fields(xp, eps33, normal_x, normal_y)
    def ft(v):
        return fourier2d_pointwise(xp, v, half_width_u, half_width_v)
    # The bare material is constant within each raster pixel. Keep the exact
    # box integral of epsilon_zz; only its projector-dependent correction is
    # sampled. This agrees continuously with the existing five-slot path.
    zz = fourier2d_piecewise_constant(
        xp, eps33[..., 2, 2], half_width_u, half_width_v, inverse_rule=None)
    def block(field, material=False):
        return xp.concat([xp.concat([
            zz + ft(field[..., 2, 2] - eps33[..., 2, 2])
            if material and i == j == 2 else ft(field[..., i, j])
            for j in range(size)], axis=-1) for i in range(size)], axis=-2)
    effective = block(A, material=True) @ xp.linalg.inv(block(B))
    n = effective.shape[-1] // size
    blocks = [[effective[i*n:(i+1)*n, j*n:(j+1)*n]
               for j in range(size)] for i in range(size)]
    if size == 3:
        return tensor_slots(xp, blocks)
    return ConvMatrices(blocks[0][0], blocks[1][1], xp.linalg.inv(zz),
                        -blocks[0][1], -blocks[1][0])


def pn_mu_conv_matrices(xp, mu33, normal_x, normal_y, half_width_u, half_width_v):
    """The five **permeability** slots from a patterned ``mu`` tensor.

    ``pn_eps_conv_matrices`` verbatim. That one routine serves both is not a shortcut — it is
    the E/H duality the whole magnetic implementation rests on. The rearrangement in
    :func:`_pn_fields` is driven entirely by which quantities are continuous at an interface:

        tangential E and normal D   ->   tangential H and normal B

    and the second line is the first with every symbol replaced. ``D = eps E`` becomes
    ``B = mu H``, so ``a = n.mu.n``, ``p = mu n``, ``q = mu^T n`` and every expression after
    them is unchanged. A separate derivation for ``mu`` would be the same derivation.

    **This used to exist in order to reorder, and that is worth keeping on the record.**
    ``pn_eps_conv_matrices`` returned ``(x, y, xy, yx, z_i)`` while
    :func:`~wavebit._core.solve_2d.layer_modes_2d` takes ``(x, y, z_i, xy, yx)``, so this
    function swapped the two groups on the way through; passing one to the other unchanged put
    ``mu_zz^-1`` where ``mu_xy`` belongs, which is a plausible-looking magnetic layer rather
    than an error. The same mismatch then bit a second time through ``solve_conv``, which
    a local fix here could not reach, so the order was made one order everywhere (see "Slot
    order" in this module's docstring) and the swap below went away. What remains is a name for
    the ``mu`` half of the duality and for the normal-field note below — nothing is rearranged.

    ``normal_x`` / ``normal_y`` are the normal field of the **permeability's** boundary, which need not be
    the permittivity's — see :func:`to_conv_mat_raster_pn_mu`.
    """
    return pn_eps_conv_matrices(xp, mu33, normal_x, normal_y, half_width_u, half_width_v)


def to_conv_mat_raster_pn_mu(tensor_raster, half_width_u, half_width_v, normal_field=None, lattice=None,
                             raster_normal_method=None):
    """Permeability counterpart of :func:`to_conv_mat_raster_pn_eps`.

    Boundaries read all nine mu components, independently of epsilon.
    A supplied normal field fixes geometry and allows differentiable replay.
    """
    xp = array_namespace(tensor_raster)

    if tensor_raster.shape[-2:] != (3, 3):
        raise ValueError(
            f'to_conv_mat_raster_pn_mu takes a permeability tensor per pixel, shaped '
            f'(n_layers, Ny, Nx, 3, 3); got {tuple(tensor_raster.shape)}.')

    if normal_field is not None and len(normal_field) != tensor_raster.shape[0]:
        raise StaleGeometryError(
            f'the frozen geometry no longer fits this structure: it holds a normal field for '
            f'{len(normal_field)} layers and this one has {tensor_raster.shape[0]}.')

    from .anisotropy import TensorConvMatrices, _labels
    out = []
    for i in range(tensor_raster.shape[0]):
        mu33 = tensor_raster[i]
        if normal_field is None:
            # Lifted only when the field is built here. A supplied one fixes the resolution,
            # and repeating the tensor under it would pair a fine tensor with a coarse field.
            mu33 = _tensor_at_the_normal_field_floor(xp, mu33, half_width_u, half_width_v)
            normal_x, normal_y = normal_vector_field(
                _labels(mu33).astype(float), lattice=lattice,
                method=raster_normal_method)
        else:
            normal_x, normal_y = normal_field[i]
        out.append(list(pn_mu_conv_matrices(xp, mu33, normal_x, normal_y, half_width_u, half_width_v)))

    full = any(len(row) == 9 for row in out)
    if full:
        for row in out:
            if len(row) == 5: row.extend([xp.zeros_like(row[0]) for _ in range(4)])
    cls = TensorConvMatrices if full else ConvMatrices
    return cls(*(xp.stack([row[j] for row in out]) for j in range(9 if full else 5)))


def to_conv_mat_raster_pn_eps(tensor_raster, half_width_u, half_width_v, normal_field=None,
                              lattice=None, raster_normal_method=None):
    """Per-layer Popov-Neviere convolution matrices for a **tensor** raster.

    The counterpart of :func:`to_conv_mat_raster_nvf` for a layer stack whose permittivity is a
    tensor at every pixel rather than a scalar.

    Args:
        tensor_raster: ``(n_layers, Ny, Nx, 3, 3)`` — the permittivity tensor per pixel. Same
            quantity as the scalar ``structure`` raster everywhere else, one rank up; this
            builder was already stated in permittivity before the scalar paths were, because
            ``n ** 2`` is a scalar operation and there is no square root of a tensor field
            worth guessing at.
        half_width_u, half_width_v: Fourier truncation orders.
        normal_field: optional per-layer ``(normal_x, normal_y)``. Left ``None`` it is computed from the
            material phase labels using all nine tensor components. Supply it to keep
            the geometry fixed under JAX tracing or to distinguish smooth material
            textures from sharp region boundaries.
        lattice: cell basis, passed through to :func:`normal_vector_field`.
        raster_normal_method: the raster construction, as in :func:`to_conv_mat_raster_nvf`;
            read only when ``normal_field`` is not supplied.

    Returns:
        Five or nine slots, each ``(n_layers, n_orders, n_orders)``. Full
        tensors append ``epxz, epyz, epzx, epzy``; all off-diagonals are negated.
    """
    xp = array_namespace(tensor_raster)

    if tensor_raster.shape[-2:] != (3, 3):
        raise ValueError(
            f'to_conv_mat_raster_pn_eps takes a permittivity tensor per pixel, shaped '
            f'(n_layers, Ny, Nx, 3, 3); got {tuple(tensor_raster.shape)}. A scalar raster goes '
            f'to to_conv_mat_raster_nvf instead; both take relative permittivity.')

    if normal_field is not None and len(normal_field) != tensor_raster.shape[0]:
        raise StaleGeometryError(
            f'the frozen geometry no longer fits this structure: it holds a normal field for '
            f'{len(normal_field)} layers and this one has {tensor_raster.shape[0]}. Call '
            f'freeze_geometry() again, or drop geometry= to rebuild by comparison.')

    from .anisotropy import TensorConvMatrices, _labels
    out = []
    for i in range(tensor_raster.shape[0]):
        eps33 = tensor_raster[i]
        if normal_field is None:
            # Lifted only when the field is built here. A supplied one fixes the resolution,
            # and repeating the tensor under it would pair a fine tensor with a coarse field.
            eps33 = _tensor_at_the_normal_field_floor(xp, eps33, half_width_u, half_width_v)
            # Equal traces can still describe distinct tensor materials.
            normal_x, normal_y = normal_vector_field(
                _labels(eps33).astype(float), lattice=lattice,
                method=raster_normal_method)
        else:
            normal_x, normal_y = normal_field[i]
        out.append(list(pn_eps_conv_matrices(xp, eps33, normal_x, normal_y, half_width_u, half_width_v)))

    full = any(len(row) == 9 for row in out)
    if full:
        for row in out:
            if len(row) == 5: row.extend([xp.zeros_like(row[0]) for _ in range(4)])
    cls = TensorConvMatrices if full else ConvMatrices
    return cls(*(xp.stack([row[j] for row in out]) for j in range(9 if full else 5)))


def nvf_conv_matrices(xp, eps_matrix, normal_x, normal_y, half_width_u, half_width_v,
                      magnitude=None):
    """NVF convolution matrices for one layer, with sampled coefficients.

    ``magnitude`` selects the projector — see :func:`_nvf_projector`. ``None`` is ``n n^T``.
    """
    C = _complex_dtype(xp, eps_matrix)
    eps_matrix = xp.astype(eps_matrix, C)
    dev = _device(xp, eps_matrix)

    # eps is a staircase however finely it was sampled, so its coefficients are exact from
    # the grid it arrives on. The projector grids are not -- they sample a continuous
    # vector field -- which is why only these two carry the flag.
    ep_dir = fourier2d_piecewise_constant(xp, eps_matrix, half_width_u, half_width_v, inverse_rule=None)
    ep_inv = fourier2d_piecewise_constant(xp, 1.0 / eps_matrix, half_width_u, half_width_v, inverse_rule=None)

    return _nvf_assemble(xp, ep_dir, ep_inv,
                         *_nvf_projector(xp, normal_x, normal_y, half_width_u, half_width_v, C, dev,
                                         magnitude=magnitude))


def nvf_conv_matrices_integrated(xp, eps_comp, x, y, normal_x, normal_y, half_width_u, half_width_v,
                                 magnitude=None):
    """NVF convolution matrices for one layer, with integrated (closed-form) coefficients.

    ``eps_comp`` is the *compressed* piecewise-constant layer and ``x`` / ``y`` its segment
    edges, exactly as :func:`fourier2d_integrated` takes them; ``normal_x`` / ``normal_y`` are the normal
    field on the uncompressed raster (see :func:`_nvf_projector` for why the two grids differ).
    ``magnitude`` selects the projector, as in :func:`nvf_conv_matrices`.
    """
    C = _complex_dtype(xp, eps_comp)
    eps_comp = xp.astype(eps_comp, C)
    dev = _device(xp, eps_comp)

    ep_dir = fourier2d_integrated(xp, eps_comp, x, y, half_width_u, half_width_v, inverse_rule=None)
    ep_inv = fourier2d_integrated(xp, 1.0 / eps_comp, x, y, half_width_u, half_width_v, inverse_rule=None)

    return _nvf_assemble(xp, ep_dir, ep_inv,
                         *_nvf_projector(xp, normal_x, normal_y, half_width_u, half_width_v, C, dev,
                                         magnitude=magnitude))


def to_conv_mat_raster_nvf(structure, half_width_u, half_width_v, normal_field=None,
                           normal_field_source=None, lattice=None, raster_normal_method=None,
                           nvf_projector=None):
    """Per-layer NVF convolution matrices for a raster ``structure``.

    Args:
        structure: relative-permittivity raster ``(n_layers, Ny, Nx)`` — the design variable.
        half_width_u, half_width_v: Fourier truncation orders.
        normal_field: optional per-layer ``(normal_x, normal_y)`` pairs for the *upsampled* layer grid.
            Supply this to keep the build traceable under JAX (the default computes the field
            with SciPy, which needs concrete values). It is static geometry either way.
        normal_field_source: what that field was sampled from, from
            :func:`normal_field_source_for_raster`. Supply it and an eager replay is checked
            against the layer in front of it (see :func:`_check_normal_field_still_fits`);
            leave it ``None`` and the field is replayed on trust.
        raster_normal_method: how the normal field is constructed from the raster
            (:class:`~wavebit._core.enums.NormalMethod`; ``None`` means ``'distance'``).
            Ignored when ``normal_field`` is supplied — the record was built under its own
            method, which the source check compares against this one.
        projector: how ``P`` is formed *from* that field
            (:class:`~wavebit._core.enums.Projector`; ``None`` means ``'linear'``, ``n n^T``).
            A separate axis from ``raster_normal_method`` rather than a fifth value of it: this
            chooses what is done with the field, so it composes with all four constructions
            instead of doubling the list.

    The normal field is taken on the raster as given, lifted only to the floor its transform
    needs (:func:`_normal_field_floor`). Lifting it further used to be offered and is not: the
    grid is a sampling of a guess at the boundary behind the staircase, and refining it makes
    the guess follow the staircase more faithfully rather than the shape — measured worse in
    27 of 32 combinations (``docs/nvf-upsampling.md``).

    Returns:
        ``(epx, epy, epz_i, epxy, epyx)``, each ``(n_layers, n_orders, n_orders)``.
    """
    xp = array_namespace(structure)
    want_magnitude = _wants_magnitude(nvf_projector)

    for name, record in (('normal field', normal_field),
                         ('normal-field source', normal_field_source)):
        if record is not None and len(record) != structure.shape[0]:
            raise StaleGeometryError(
                f'the frozen geometry no longer fits this structure: it holds a {name} for '
                f'{len(record)} layers and this one has {structure.shape[0]}. Call '
                f'freeze_geometry() again, or drop geometry= to rebuild by comparison.')

    out = [[], [], [], [], []]

    layers = _layer_eps_for_normal_field(xp, structure, half_width_u, half_width_v)
    for i, (eps_matrix, base_eps, repeat_v, repeat_u) in enumerate(layers):
        if normal_field is None:
            field = normal_vector_field(
                eps_matrix, _interface_mask_of(xp, eps_matrix, base_eps, repeat_v, repeat_u),
                lattice=lattice, method=raster_normal_method, base_shape=tuple(base_eps.shape),
                with_magnitude=want_magnitude)
        else:
            _check_normal_field_still_fits(eps_matrix, normal_field_source, i, lattice,
                                           raster_normal_method)
            field = normal_field[i]
        normal_x, normal_y, magnitude = _unpack_normal_field(field, want_magnitude, i)

        for slot, mat in zip(out, nvf_conv_matrices(
                xp, eps_matrix, normal_x, normal_y, half_width_u, half_width_v,
                magnitude=magnitude)):
            slot.append(mat)

    return ConvMatrices(*(xp.stack(slot) for slot in out))


# ============================================ integrated Fourier series (closed form, analytic)

def _lift(xp, v, dtype, device=None):
    """``v`` on the working backend at ``dtype``, keeping any autograd graph intact.

    Only values that are not already arrays of this namespace get created from scratch. That
    guarded a real defect through torch 2.11.x, where ``asarray(tensor, dtype=...)`` detached
    and a value that *is* a design variable silently stopped carrying gradient; torch 2.12.0
    fixed it — see :func:`wavebit._core.backend.as_array` for the version history and for why
    the branch is kept anyway.
    """
    if hasattr(v, 'dtype') and hasattr(v, 'shape'):
        try:
            same = array_namespace(v) is xp
        except Exception:
            same = False
        if same:
            return v if v.dtype == dtype else xp.astype(v, dtype)
    return xp.asarray(v, dtype=dtype, device=device)


def fourier2d_integrated(xp, cell, x, y, half_width_u, half_width_v, *, inverse_rule,
                         nesting=None):
    """2D continuous Fourier series with Li's rules — the staircase-free factorization.

    ``x`` / ``y`` are the segment edges of the compressed (piecewise-constant) layer, so the
    permittivity is Fourier-transformed analytically rather than off a raster. They are
    right-hand edges, the last one closing the period, in any unit: they are divided by the
    last one (:func:`_segment_bounds`).

    **``continuous`` describes the method, not the data.** What arrives here is piecewise
    constant, exactly as in :func:`fourier2d_piecewise_constant`; the coefficients are obtained
    by integrating over the segments rather than by a discrete sum, which is the ``integrated``
    / ``sampled`` distinction and nothing more. The public spelling of the same choice is
    ``fourier_coefficients='integrated'``.

    What that buys is *where the steps may sit*. :func:`fourier2d_piecewise_constant` places them
    on the pixel grid at ``j/n``, so an edge at ``x = 0.317`` is first snapped to a pixel and
    then transformed exactly; here the edge is a number this function integrates to. Both are
    exact for the staircase they are given — they differ in which staircases they can be
    given. Curved boundaries are still approximated by steps in both;
    :mod:`wavebit._core.parametric` is the path where the staircase itself is gone.

    ``inverse_rule`` and ``nesting`` are as in :func:`fourier2d_piecewise_constant`.
    """
    C = _complex_dtype(xp, cell)
    cell = xp.astype(cell, C)
    return _li_slot(xp, cell, half_width_u, half_width_v, *_segment_series_pair(
        xp, x, y, half_width_u, half_width_v, C, _device(xp, cell)), inverse_rule, nesting)


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
    """Where a raster's piecewise-constant blocks end, and the edges that places them at.

    ``cell`` is ``(N_r, N_c)``, rows along ``v`` and columns along ``u``. Returns
    ``(row_idx, col_idx, x, y)``:

    - ``col_idx``: ascending Python ints -- every column ``c < N_c - 1`` that differs from column
      ``c + 1`` in any entry, and ``N_c - 1``. That is the last column of each run of identical
      neighbouring columns.
    - ``row_idx``: the same for rows, compared on the column-reduced raster ``cell[:, col_idx]``.
    - ``x``: NumPy float64, ``(len(col_idx), 1)``, the right-hand edge of each kept column as
      ``step_u * (c + 1)`` with ``step_u = 1.0 / N_c``; ``y`` likewise from ``row_idx`` and
      ``N_r``.

    The compressed layer ``cell[row_idx][:, col_idx]`` (:func:`compress_cell`) on those edges is
    the same staircase as the raster. The comparisons are narrowed to Python booleans with
    :func:`_as_bools`, so ``cell`` may be a torch tensor that requires grad or a ``jax.grad``
    tracer. The output's shape depends on the values, so this does not run under ``jax.jit``;
    a frozen record of its result is what a traced build replays. A gradient over a raster
    whose pattern moves every step -- a binary topology optimization, say -- therefore needs no
    frozen geometry and cannot be served a stale one.

    Written with :func:`compress_cell` on 2026-09-25 from a specification of what they return,
    replacing versions that came from meent; ``LICENSE`` records the history.
    """
    import numpy as np
    xp = array_namespace(cell)
    n_row, n_col = cell.shape

    col_steps = _as_bools(xp, xp.any(cell[:, :-1] != cell[:, 1:], axis=0))
    col_idx = np.flatnonzero(col_steps).tolist() + [n_col - 1]

    reduced = xp.take(cell, xp.asarray(col_idx, device=_device(xp, cell)), axis=1)
    row_steps = _as_bools(xp, xp.any(reduced[:-1, :] != reduced[1:, :], axis=1))
    row_idx = np.flatnonzero(row_steps).tolist() + [n_row - 1]

    step_v, step_u = 1.0 / np.array((n_row, n_col), dtype=np.float64)
    x = (step_u * (np.asarray(col_idx, dtype=np.float64) + 1.0)).reshape((-1, 1))
    y = (step_v * (np.asarray(row_idx, dtype=np.float64) + 1.0)).reshape((-1, 1))
    return row_idx, col_idx, x, y


def _steps_along(cell, axis):
    """Indices along ``axis`` where ``cell`` differs from its successor — where it must be cut.

    Cyclic: the last index compares against the first, and is a required cut regardless since
    it closes the period. So it finds the cuts :func:`compression_indices` makes, although that
    one compares one-sidedly and keeps the last index unconditionally.
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
            f'geometry= to rebuild by comparison. (Changing the permittivity of a layer '
            f'never trips this on its own; moving its pattern does.)')

    if not len(col_idx) or not len(row_idx) or col_idx[-1] != n_col - 1\
            or row_idx[-1] != n_row - 1:
        refuse(f'is {cell.shape}, which the recorded cut does not span')

    missing = sorted(_steps_along(cell, 1) - set(int(c) for c in col_idx))
    if missing:
        refuse(f'now steps at column {missing[0]}, where the recorded cut does not divide it')

    cell_u = cell[:, list(col_idx)]
    missing = sorted(_steps_along(cell_u, 0) - set(int(r) for r in row_idx))
    if missing:
        refuse(f'now steps at row {missing[0]}, where the recorded cut does not divide it')

    step_v, step_u = 1.0 / np.array(cell.shape, dtype=np.float64)
    edges = (np.asarray(x), np.asarray(y))
    want = (np.array([step_u * (c + 1) for c in col_idx]).reshape((-1, 1)),
            np.array([step_v * (r + 1) for r in row_idx]).reshape((-1, 1)))
    if not all(np.array_equal(a, b) for a, b in zip(edges, want)):
        refuse('carries block edges that are not the ones its shape implies')


def compress_cell(xp, cell, row_idx, col_idx):
    """``cell[row_idx][:, col_idx]``: the compressed layer, as a differentiable gather.

    The indices come from :func:`compression_indices` or from a frozen record of it, and are
    placed on ``cell``'s device. The gather is traceable, and the gradient reaches every entry of
    ``cell`` it reads.
    """
    dev = _device(xp, cell)
    rows = xp.take(cell, xp.asarray(row_idx, device=dev), axis=0)
    return xp.take(rows, xp.asarray(col_idx, device=dev), axis=1)


def to_conv_mat_raster_integrated(structure, half_width_u, half_width_v, compression=None,
                                  nesting=None):
    """Per-layer convolution matrices via the integrated (closed-form) series.

    Args:
        structure: relative-permittivity raster ``(n_layers, Ny, Nx)`` — the design variable.
        half_width_u, half_width_v: Fourier truncation orders.
        compression: optional per-layer ``(row_idx, col_idx, x, y)`` from
            :func:`compression_indices`. Supply it to stay traceable under JAX; it is static
            geometry either way. Replayed eagerly it is checked against the layer in front of
            it (see :func:`_check_compression_still_fits`). The same record serves both
            nestings: it says where the blocks are, and the nesting only how they are combined.
        nesting: the order Li's rules are nested in (:class:`~wavebit._core.enums.Nesting`;
            ``None`` means ``'fixed'``).

    Returns:
        ``(epx, epy, epz_i, None, None)``, each slot ``(n_layers, n_orders, n_orders)`` — one
        :func:`li_conv_matrices_integrated` per layer.
    """
    xp = array_namespace(structure)
    if compression is not None and len(compression) != structure.shape[0]:
        raise StaleGeometryError(
            f'the frozen geometry no longer fits this structure: it holds a compression for '
            f'{len(compression)} layers and this one has {structure.shape[0]}. Call '
            f'freeze_geometry() again, or drop geometry= to rebuild by comparison.')

    out = [[], [], []]
    for i in range(structure.shape[0]):
        layer = structure[i]
        if compression is None:
            row_idx, col_idx, x, y = compression_indices(layer)
        else:
            _check_compression_still_fits(layer, compression[i], i)
            row_idx, col_idx, x, y = compression[i]
        comp = compress_cell(xp, layer, row_idx, col_idx)
        for slot, mat in zip(out, li_conv_matrices_integrated(xp, comp, x, y, half_width_u,
                                                              half_width_v, nesting)):
            slot.append(mat)
    return ConvMatrices(*(xp.stack(slot) for slot in out), None, None)


def to_conv_mat_raster_integrated_nvf(structure, half_width_u, half_width_v, normal_field=None,
                                      normal_field_source=None, compression=None,
                                      lattice=None, raster_normal_method=None, nvf_projector=None):
    """Per-layer convolution matrices for a raster ``structure`` via closed-form Fourier + NVF.

    The pairing that needs both axes named independently. It buys the staircase-free
    permittivity series of the integrated path *and* the vector factorization that handles
    boundaries not aligned with x or y — which is what a shape/OCD fit of a tilted sidewall
    wants from a raster.

    Args:
        structure: relative-permittivity raster ``(n_layers, Ny, Nx)`` — the design variable.
        half_width_u, half_width_v: Fourier truncation orders.
        normal_field: optional per-layer ``(normal_x, normal_y)`` from :func:`normal_field_for_raster`.
        normal_field_source: what that field was sampled from, as in
            :func:`to_conv_mat_raster_nvf` — this path replays the same record and is checked
            the same way.
        compression: optional per-layer ``(row_idx, col_idx, x, y)`` from
            :func:`compression_indices`. Replayed eagerly it is checked too.
        lattice: the cell basis, as in :func:`to_conv_mat_raster_nvf`.
        raster_normal_method: the construction, as in :func:`to_conv_mat_raster_nvf`.
        projector: how ``P`` is formed from the field, as in :func:`to_conv_mat_raster_nvf`.

    Both ``normal_field`` and ``compression`` are static geometry; supply both to keep the
    build traceable under JAX (this path is the only one that needs both).

    ``lattice`` is carried here even though the solver currently refuses ``('integrated',
    'nvf')`` on a skewed cell, because the basis is part of the frozen record's staleness key
    (:func:`_normal_field_source`). A builder that replays that record without being able to
    see the basis would reject a field frozen *with* one — so the parameter is what keeps this
    path consistent with the record it consumes, not a second implementation of the skew.

    Returns:
        ``(epx, epy, epz_i, epxy, epyx)``, each ``(n_layers, n_orders, n_orders)``.
    """
    xp = array_namespace(structure)
    want_magnitude = _wants_magnitude(nvf_projector)

    out = [[], [], [], [], []]
    eps_grids = _layer_eps_for_normal_field(xp, structure, half_width_u, half_width_v)

    for i, (eps_grid, base_eps, repeat_v, repeat_u) in enumerate(eps_grids):
        if normal_field is None:
            field = normal_vector_field(
                eps_grid, _interface_mask_of(xp, eps_grid, base_eps, repeat_v, repeat_u),
                lattice=lattice, method=raster_normal_method, base_shape=tuple(base_eps.shape),
                with_magnitude=want_magnitude)
        else:
            _check_normal_field_still_fits(eps_grid, normal_field_source, i, lattice,
                                           raster_normal_method)
            field = normal_field[i]
        normal_x, normal_y, magnitude = _unpack_normal_field(field, want_magnitude, i)

        layer = structure[i]
        if compression is None:
            row_idx, col_idx, x, y = compression_indices(layer)
        else:
            _check_compression_still_fits(layer, compression[i], i)
            row_idx, col_idx, x, y = compression[i]
        comp = compress_cell(xp, layer, row_idx, col_idx)
        eps_comp = xp.astype(comp, _complex_dtype(xp, comp))

        for slot, mat in zip(out, nvf_conv_matrices_integrated(
                xp, eps_comp, x, y, normal_x, normal_y, half_width_u, half_width_v,
                magnitude=magnitude)):
            slot.append(mat)

    return ConvMatrices(*(xp.stack(slot) for slot in out))
