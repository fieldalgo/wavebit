"""A raster normal field's discrete choices, recorded, and the rest rebuilt in a live cell.

A raster NVF normal field is a function of two things: the interface set, which moves only when
the pattern does, and the shape of the cell, which moves under a derivative with respect to the
``period`` or the ``lattice``. Its dependence on the cell has two halves. Some of it is a choice
-- which interface pixel is nearest, which band pixel a bulk pixel copies, how many pixels wide
a kernel is -- and a choice has no derivative. The rest is smooth: a distance measured in the
cell's metric, a Gaussian whose width is a length, a gradient pulled back through ``J^-T``.

A frozen record (:meth:`~wavebit.rcwa._engine.engine.CoreSolveMixin.freeze_geometry`, and every
``rcwa.prepare`` / ``Prepared.run``) used to keep the field and drop that smooth half: a replay
differentiated the solve with the field held, so a cell derivative was not the derivative of
the answer the replay computes. On an 8 x 8 rod in a skewed cell the replayed ``d/du_x`` read
2.96e-4 where the rebuilt field's own central difference reads 1.43e-4; across the three
constructions that can carry it, up to 106% of a derivative component was missing on that rod
and up to 93% on a 24 x 24 disc.

So the record keeps the choices too, as a :class:`NormalFieldPlan` per layer, and a replay whose
cell carries a derivative rebuilds the smooth half from them in the live cell. The rebuilt field
contributes **only its tangent** (:func:`_carried`): the value handed on is the recorded array,
bit for bit, so a replay answers what it always answered, and a cell with no derivative never
reaches this module at all.

Which constructions can do this:

=============  ================================================================================
``distance``   nearest interface pixel; the distance and its pulled-back gradient are rebuilt.
               This is the traced-basis branch of ``factorization._distance_gradient``, which a
               cell derivative without a record has always taken.
``spectral``   indicator, band, donors, the longer cell edge; the low-pass and the spectral
               gradient are rebuilt on the cell's Cartesian wavevectors.
``smooth``     phases, nearest pixel of the other phase, branch, kernel radii; the signed
               distance and the derivative-of-Gaussian are rebuilt -- a circulant of scipy's own
               truncated kernel on an axis-aligned cell, the transform on a skewed one.
``contour``    **refused** when the cell is differentiated. Its curve moves in index space as
               the cell's shape changes the blur, and following that needs the contour
               extraction differentiated. Holding the curve and carrying only its map to
               Cartesian coordinates was measured worse than holding the whole field (12.6%
               against 10.7% of ``d/du_x`` on the rod), so it is not offered.
=============  ================================================================================

The ``|grad d|`` magnitude ``nvf_projector='circular'`` reads is the distance construction's for
every method, so it carries on the same record.
"""
from dataclasses import dataclass
from typing import Any
from typing import Optional

import numpy as np

from wavebit._common.backend import array_namespace
from wavebit._common.backend import as_array
from wavebit._common.backend import carries_autodiff
from wavebit._common.backend import constant_array
from wavebit._common.backend import torch_autodiff


@dataclass(frozen=True, eq=False)
class NormalFieldPlan:
    """What one layer's normal field chose, so that a replay can rebuild the rest.

    Attributes:
        direction: which construction drew the direction -- ``'constant'`` (no interface, or a
            feature the smoothing erased: nothing depends on the cell), ``'axis'`` (a one-cell
            axis: the closed-form normal of the cell), ``'distance'``, ``'spectral'``,
            ``'smooth'``, or ``'contour'``, which carries no cell derivative.
        offsets: ``(d_v, d_u)``, integer index offsets from every pixel to its nearest interface
            pixel on the periodic tiling -- what the ``'distance'`` direction and the circular
            projector's magnitude are rebuilt from. ``None`` where neither reads it.
        detail: the construction's own choices (``normal_methods`` documents each), or ``None``.

    **Equal by content, and never hashed.** JAX's ``fit`` re-freezes before every step and
    keeps the previous record when the new one is the same in every field
    (``jax_backend._same_prepared_geometry``), which is what compiles the step once. A plan
    compared by identity made every fresh freeze a new record, and a fit over a fixed raster
    compiled at every step. Two plans are equal when their choices are: the same direction and
    the same arrays, dtype and value. A plan holds arrays, so it has no hash; the record that
    carries it is hashed by identity (``StaticGeometry._cache_key``).
    """
    direction: str
    offsets: Optional[Any] = None
    detail: Optional[dict] = None

    def __eq__(self, other):
        if not isinstance(other, NormalFieldPlan):
            return NotImplemented
        return (self.direction == other.direction and _same_choices(self.offsets, other.offsets)
                and _same_choices(self.detail, other.detail))

    __hash__ = None


def _same_choices(a, b):
    """Exact equality of two plans' choices: nested tuples, lists and dicts of arrays and scalars."""
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(_same_choices(a[k], b[k]) for k in a))
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return (isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b)
                and all(_same_choices(x, y) for x, y in zip(a, b)))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a, b))
    return type(a) is type(b) and a == b


#: A replay whose record cannot carry the cell's derivative, asked for one. Raised when the
#: derivative is taken -- a ``jax.jit`` or ``jax.vmap`` over the cell alone is a forward
#: evaluation, and the record serves it as it always has.
_CONTOUR = (
    "raster_normal_method='contour' cannot differentiate the cell (period or lattice): its normal "
    "field follows a contour traced in NumPy, and a frozen record would hold that field fixed, "
    "leaving its own change with the cell out of the derivative -- 11% to 37% of it on the scenes "
    "measured. raster_normal_method='distance', 'spectral' and 'smooth' carry it, through "
    "freeze_geometry(), rcwa.prepare and Prepared.run alike; a derivative with respect to the "
    "materials alone needs none of this.")
_NO_PLAN = (
    "this frozen geometry records the normal field but not the choices it was built from, so a "
    "replay cannot carry the field's own change with the cell, and a derivative with respect to "
    "the cell (period or lattice) would leave it out. Take the record with freeze_geometry(), "
    "which keeps them; a record assembled by hand replays only where the cell is not "
    "differentiated.")


def nearest_interface_offsets(interface_mask, J=None):
    """``(d_v, d_u)`` floats: the index offset to every pixel's nearest interface pixel.

    The one discrete choice in the ``'distance'`` construction, made on the 3x3 periodic tiling
    in the cell's per-axis steps, exactly as its traced-basis branch makes it (which calls this).
    ``J`` is the cell's index-to-Cartesian Jacobian (:func:`~wavebit.lattice.jacobian`); ``None``
    is square pixels.
    """
    from scipy.ndimage import distance_transform_edt
    ny_, nx_ = interface_mask.shape
    Jc = np.eye(2) if J is None else np.asarray(J, dtype=float)
    Jc = Jc / (float(np.hypot(Jc[0, 0], Jc[1, 0])) or 1.0)
    sj = float(np.hypot(Jc[0, 1], Jc[1, 1]))
    si = float(np.hypot(Jc[0, 0], Jc[1, 0]))
    tiled = np.tile(~interface_mask, (3, 3))
    _, idx = distance_transform_edt(tiled, sampling=(sj, si), return_indices=True)
    jj, ii = np.meshgrid(np.arange(3 * ny_), np.arange(3 * nx_), indexing='ij')
    core = (slice(ny_, 2 * ny_), slice(nx_, 2 * nx_))
    return ((idx[0] - jj)[core].astype(float), (idx[1] - ii)[core].astype(float))


def distance_gradient_from_offsets(xp, dj, di, J):
    """``(g_i, g_j)``, the distance map's Cartesian gradient, rebuilt in the cell ``J``.

    The distance to the recorded nearest pixel is ``hypot(step_j*dj, step_i*di)`` -- smooth in
    the metric -- and its wrapping central differences are pulled back through ``J^-T``, both in
    the units of one ``u``-step, as the eager construction works.
    """
    scale_ = xp.sqrt(J[0, 0] ** 2 + J[1, 0] ** 2)
    a_, b_ = J[0, 0] / scale_, J[0, 1] / scale_
    c_, d_ = J[1, 0] / scale_, J[1, 1] / scale_
    step_j_ = xp.sqrt(b_ ** 2 + d_ ** 2)
    step_i_ = xp.sqrt(a_ ** 2 + c_ ** 2)
    # Guard inside the sqrt: an on-interface pixel has zero displacement, and
    # `d(sqrt)/dx` at zero is infinite whatever a `where` outside it says.
    sq_ = (step_j_ * dj) ** 2 + (step_i_ * di) ** 2
    dmap_ = xp.sqrt(xp.where(sq_ != 0, sq_, xp.ones_like(sq_)))
    dmap_ = xp.where(sq_ != 0, dmap_, xp.zeros_like(dmap_))
    g_j = (xp.roll(dmap_, -1, axis=0) - xp.roll(dmap_, 1, axis=0)) / 2.0
    g_i = (xp.roll(dmap_, -1, axis=1) - xp.roll(dmap_, 1, axis=1)) / 2.0
    det_ = a_ * d_ - b_ * c_
    return (d_ * g_i - c_ * g_j) / det_, (-b_ * g_i + a_ * g_j) / det_


def unit_direction(xp, gx, gy, floor):
    """``(gx, gy) / |g|`` on traced arrays, ``(0, 0)`` where ``|g|`` is at or below ``floor``.

    The guard goes inside the sqrt, for the reason :func:`distance_gradient_from_offsets` gives.
    """
    sq = gx ** 2 + gy ** 2
    nz = sq > floor ** 2
    norm = xp.sqrt(xp.where(nz, sq, xp.ones_like(sq)))
    gx = xp.where(nz, gx, xp.zeros_like(gx))
    gy = xp.where(nz, gy, xp.zeros_like(gy))
    return gx / norm, gy / norm


def plan_for(decisions, interface_mask, lattice, shape, with_magnitude):
    """The :class:`NormalFieldPlan` of a field just built, from what its construction decided.

    ``decisions`` is the dict ``normal_vector_field`` and the constructions fill in. The offsets
    are taken here, on the same mask and cell the field was built on, wherever the direction or
    the magnitude reads them.
    """
    direction = decisions['kind']
    detail = decisions.get('detail')
    reads_distance = direction == 'distance' or (with_magnitude and direction != 'axis')
    offsets = None
    if reads_distance and interface_mask is not None and interface_mask.any():
        J = None
        if lattice is not None:
            from wavebit.lattice import jacobian
            J = constant_array(jacobian(lattice, *shape))
        d_v, d_u = nearest_interface_offsets(np.asarray(interface_mask, dtype=bool), J)
        offsets = (d_v.astype(np.int64), d_u.astype(np.int64))
    elif direction == 'distance':
        direction = 'constant'           # no interface: a flat map, nothing to differentiate
    return NormalFieldPlan(direction, offsets, detail)


def cell_carries_derivative(lattice):
    """Whether the cell is something a derivative can reach -- any JAX tracer, a tensor under
    PyTorch autodiff -- as opposed to a constant, which takes the record as it stands."""
    if lattice is None:
        return False
    cols = getattr(lattice, 'columns', lattice)
    if carries_autodiff(cols):
        return True
    try:
        constant_array(cols)
    except Exception:                     # noqa: BLE001 - a value that refuses to be a constant
        return True
    return False


def primal_cell(lattice):
    """The cell's value as a NumPy ``(2, 2)``, or ``None`` where it has none (``jax.jit``)."""
    from wavebit.rcwa._engine.factorization import _no_gradient
    cols = getattr(lattice, 'columns', lattice)
    try:
        return constant_array(_no_gradient(cols))
    except Exception:                     # noqa: BLE001 - a trace without a value
        return None


def replayed(entry, plan, lattice, shape, want_magnitude):
    """The field a replay hands the projector: the record, carrying the cell's derivative.

    ``entry`` is the recorded ``(normal_x, normal_y[, magnitude])``. A cell no derivative can
    reach gets it back untouched -- the same objects, so nothing downstream moves. Otherwise each
    component the cell moves is rebuilt from ``plan`` in the live cell and handed on as the
    recorded value plus the rebuilt one's tangent (:func:`_carried`). A plan that cannot carry
    it -- ``'contour'``, or none at all -- is refused where the derivative is taken.
    """
    if not cell_carries_derivative(lattice):
        return entry
    cols = getattr(lattice, 'columns', lattice)
    xp = array_namespace(cols)
    if plan is None or plan.direction == 'contour':
        return _refused(xp, cols, entry, _NO_PLAN if plan is None else _CONTOUR)
    rebuilt = _rebuilt(xp, plan, lattice, shape, entry, want_magnitude and len(entry) > 2)
    return tuple(rec if live is None else _carried(xp, rec, live)
                 for rec, live in zip(entry, rebuilt + (None,) * (len(entry) - len(rebuilt))))


def _carried(xp, recorded, rebuilt):
    """``recorded``, with ``rebuilt``'s tangent.

    ``recorded - (stop_gradient(rebuilt) - rebuilt)``. For finite ``rebuilt`` the bracket is
    ``+0.0`` exactly, and ``r - (+0.0)`` is ``r`` for every ``r``, ``-0.0`` included -- where
    ``r + 0.0`` would not be -- so the value is the record bit for bit and only the tangent is
    the rebuilt field's.
    """
    from wavebit.rcwa._engine.factorization import _no_gradient
    rec = as_array(xp, recorded, dtype=rebuilt.dtype, device=_device(rebuilt))
    return rec - (_no_gradient(rebuilt) - rebuilt)


def _device(x):
    """Where to put a constant beside ``x``: its device, or ``None`` for a JAX tracer.

    A tracer's ``device`` is not one to place on -- under ``jax.vmap`` it is a sharding, which
    ``jnp.asarray`` refuses -- and a constant beside it needs none: inside a trace it is part of
    the program, and eagerly JAX moves an uncommitted constant to its committed partner.
    """
    import sys
    jax = sys.modules.get('jax')
    if jax is not None and isinstance(x, jax.core.Tracer):
        return None
    return x.device if hasattr(x, 'device') else None


# ------------------------------------------------------------------------------- refusal


_JAX_REFUSAL = None


def _jax_refusal():
    """A ``jax.custom_jvp`` that is ``0`` and raises when differentiated.

    A refusal at the door could not see every derivative: under ``jax.grad(jax.jit(f))`` the
    body is traced with no tangent in sight and differentiated afterwards, as a jaxpr. A rule
    is what that differentiation calls, so the refusal sits there, and a forward evaluation
    under ``jax.jit`` or ``jax.vmap`` -- which calls no rule -- is served the record as before.
    The pattern is the floored inverse rule's (``factorization._jax_floor``).
    """
    global _JAX_REFUSAL
    if _JAX_REFUSAL is None:
        import jax
        import jax.numpy as jnp
        from functools import partial

        @partial(jax.custom_jvp, nondiff_argnums=(1,))
        def refusal(cols, message):
            return jnp.zeros((), dtype=cols.dtype)

        @refusal.defjvp
        def _rule(message, primals, tangents):
            raise TypeError(message)

        _JAX_REFUSAL = refusal
    return _JAX_REFUSAL


def _refused(xp, cols, entry, message):
    """``entry`` as it is, refused the moment the cell's derivative is taken through it."""
    if torch_autodiff(cols):
        # PyTorch is eager: a cell under its autodiff is the derivative, here and now.
        raise TypeError(message)
    if 'jax' not in getattr(xp, '__name__', ''):
        return entry
    zero = _jax_refusal()(cols, message)
    return tuple(_carried(xp, rec, zero) for rec in entry)


# ------------------------------------------------------------------------------- rebuilds


def _rebuilt(xp, plan, lattice, shape, entry, magnitude):
    """``(normal_x, normal_y[, magnitude])`` rebuilt in the live cell; ``None`` for what the
    cell does not move."""
    from wavebit.lattice import jacobian
    n_v, n_u = shape
    J = jacobian(lattice, n_v, n_u)
    cols = getattr(lattice, 'columns', lattice)
    direction = plan.direction
    g = None
    if plan.offsets is not None:
        dj, di = (as_array(xp, o, dtype=J.dtype, device=_device(J)) for o in plan.offsets)
        g = distance_gradient_from_offsets(xp, dj, di, J)

    if direction == 'distance':
        from wavebit.rcwa._engine.factorization import _FLAT_PIXEL
        field = unit_direction(xp, g[0], g[1], _FLAT_PIXEL)
    elif direction == 'spectral':
        field = _spectral(xp, plan.detail, J, shape)
    elif direction == 'smooth':
        field = _smooth(xp, plan.detail, J, shape, entry)
    elif direction == 'axis':
        field = _axis(xp, lattice, cols, shape)
    else:                                 # 'constant'
        field = (None, None)

    if not magnitude:
        return field
    if g is None:
        return field + (None,)
    from wavebit.rcwa._engine.factorization import _field_magnitude
    return field + (_field_magnitude(g[0], g[1]),)


def _axis(xp, lattice, cols, shape):
    """The one-cell axis: the normal is the cell's reciprocal vector across the stripes.

    The traced branch of ``normal_vector_field`` for the same case, which reads it as a row of
    ``A^-1``; on a cell declared axis-aligned that row stays on its axis whatever the lengths.
    """
    if getattr(lattice, 'axis_aligned', False):
        return None, None
    det = cols[0, 0] * cols[1, 1] - cols[0, 1] * cols[1, 0]
    b0, b1 = ((cols[1, 1] / det, -cols[0, 1] / det) if shape[0] == 1
              else (-cols[1, 0] / det, cols[0, 0] / det))
    norm = xp.sqrt(b0 ** 2 + b1 ** 2)
    ones = xp.ones(tuple(shape), dtype=cols.dtype, device=_device(cols))
    return ones * (b0 / norm), ones * (b1 / norm)


def _metric(xp, J):
    """``M`` (the Jacobian in units of one ``u``-step) and ``B = M^-T``, in the live cell.

    ``normal_methods._metric`` and ``_reciprocal``, written in the cell's namespace.
    """
    M = J / xp.sqrt(J[0, 0] ** 2 + J[1, 0] ** 2)
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    B = ((M[1, 1] / det, -M[1, 0] / det), (-M[0, 1] / det, M[0, 0] / det))
    return M, B, det


def _const(xp, value, like, dtype=None):
    return as_array(xp, np.asarray(value), dtype=like.dtype if dtype is None else dtype,
                    device=_device(like))


def _complex_of(xp, like):
    return xp.complex64 if like.dtype == xp.float32 else xp.complex128


def _fft2(xp, x):
    return xp.fft.fftn(x, axes=(-2, -1))


def _ifft2(xp, x):
    return xp.fft.ifftn(x, axes=(-2, -1))


def _frequencies(xp, J, shape):
    """Index-space FFT frequencies ``(fi, fj)``, as ``(1, n_u)`` and ``(n_v, 1)`` constants."""
    n_v, n_u = shape
    return (_const(xp, np.fft.fftfreq(n_u)[None, :], J),
            _const(xp, np.fft.fftfreq(n_v)[:, None], J))


def _gather(xp, field, donor_v, donor_u):
    """``field[donor_v, donor_u]`` through a flat ``take``, which every namespace spells alike."""
    n_v, n_u = donor_v.shape
    flat = as_array(xp, (np.asarray(donor_v) * field.shape[1] + np.asarray(donor_u)).ravel(),
                    dtype=xp.int64, device=_device(field))
    return xp.reshape(xp.take(xp.reshape(field, (-1,)), flat), (n_v, n_u))


def _spectral(xp, detail, J, shape):
    """``normal_methods.spectral_normal_field`` past its choices, in the live cell."""
    from wavebit.rcwa._engine.normal_methods import _SPECTRAL_CUTOFF
    n_v, n_u = shape
    M, B, _ = _metric(xp, J)
    step_v = xp.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
    step_u = xp.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    fi, fj = _frequencies(xp, J, shape)
    gxf = B[0][0] * fi + B[0][1] * fj
    gyf = B[1][0] * fi + B[1][1] * fj
    L_ref = n_u * step_u if detail['longer_u'] else n_v * step_v
    # `hypot(gxf, gyf)` squared without the root, whose derivative is infinite at the DC bin.
    H = xp.exp(-0.5 * (gxf ** 2 + gyf ** 2) * L_ref ** 2 / _SPECTRAL_CUTOFF ** 2)

    di = 2j * np.pi * np.broadcast_to(np.fft.fftfreq(n_u)[None, :], shape).astype(complex).copy()
    dj = 2j * np.pi * np.broadcast_to(np.fft.fftfreq(n_v)[:, None], shape).astype(complex).copy()
    if n_u % 2 == 0:
        di[:, n_u // 2] = 0.0
    if n_v % 2 == 0:
        dj[n_v // 2, :] = 0.0
    C_ = _complex_of(xp, J)
    di, dj = _const(xp, di, J, C_), _const(xp, dj, J, C_)
    kx = B[0][0] * di + B[0][1] * dj
    ky = B[1][0] * di + B[1][1] * dj

    chi = _const(xp, np.asarray(detail['indicator'], dtype=float), J)
    C = _fft2(xp, chi) * H
    gx = xp.real(_ifft2(xp, kx * C))
    gy = xp.real(_ifft2(xp, ky * C))
    band = _const(xp, detail['band'], J, xp.bool)
    sq = gx ** 2 + gy ** 2
    mag = xp.sqrt(xp.where(band, sq, xp.ones_like(sq)))
    nx_band = xp.where(band, gx / mag, xp.zeros_like(gx))
    ny_band = xp.where(band, gy / mag, xp.zeros_like(gy))
    return (_gather(xp, nx_band, detail['donor_v'], detail['donor_u']),
            _gather(xp, ny_band, detail['donor_v'], detail['donor_u']))


def _smooth(xp, detail, J, shape, entry):
    """``normal_methods.smoothed_distance_normal_field`` past its choices, in the live cell."""
    n_v, n_u = shape
    M, B, det = _metric(xp, J)
    d_v, d_u = (_const(xp, o, J) for o in detail['nearest'])
    sign = _const(xp, np.where(detail['inside'], -1.0, 1.0), J)
    sigma = detail['sigma_per_pixel'] * xp.sqrt(xp.abs(det))      # `_pixel_size`, live

    if detail['separable']:
        step_v = xp.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
        step_u = xp.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        sq = (step_v * d_v) ** 2 + (step_u * d_u) ** 2
        sd = sign * xp.sqrt(xp.where(sq > 0, sq, xp.ones_like(sq)))
        radius_v, radius_u = detail['radii']
        sig_v, sig_u = sigma / step_v, sigma / step_u
        smooth_v, slope_v = _circulants(xp, sig_v, radius_v, n_v, J)
        smooth_u, slope_u = _circulants(xp, sig_u, radius_u, n_u, J)
        # `gaussian_filter(sd, sig, order=(0, 1))` and `order=(1, 0)`: one 1-D filter per axis.
        g_i = (smooth_v @ sd) @ xp.permute_dims(slope_u, (1, 0))
        g_j = (slope_v @ sd) @ xp.permute_dims(smooth_u, (1, 0))
        gx = B[0][0] * g_i + B[0][1] * g_j
        gy = B[1][0] * g_i + B[1][1] * g_j
    else:
        x = M[0, 0] * d_u + M[0, 1] * d_v
        y = M[1, 0] * d_u + M[1, 1] * d_v
        sq = x ** 2 + y ** 2
        sd = sign * xp.sqrt(xp.where(sq > 0, sq, xp.ones_like(sq)))
        fi, fj = _frequencies(xp, J, shape)
        gxf = B[0][0] * fi + B[0][1] * fj
        gyf = B[1][0] * fi + B[1][1] * fj
        C = _fft2(xp, sd) * xp.exp(-0.5 * (2.0 * np.pi * sigma) ** 2 * (gxf ** 2 + gyf ** 2))
        gx = xp.real(_ifft2(xp, (2j * np.pi) * gxf * C))
        gy = xp.real(_ifft2(xp, (2j * np.pi) * gyf * C))

    # The skeleton -- an exact zero eagerly, or snapped to one on the transform branch -- is the
    # one place the construction emitted no direction, and the record says where it is.
    skeleton = _const(xp, (np.asarray(entry[0]) == 0) & (np.asarray(entry[1]) == 0), J, xp.bool)
    sq = gx ** 2 + gy ** 2
    norm = xp.sqrt(xp.where(skeleton, xp.ones_like(sq), sq))
    return (xp.where(skeleton, xp.zeros_like(gx), gx / norm),
            xp.where(skeleton, xp.zeros_like(gy), gy / norm))


def _circulants(xp, sigma, radius, n, like):
    """``(smoothing, slope)``: scipy's periodic Gaussian filter of order 0 and 1, as matrices.

    ``gaussian_filter1d(f, sigma, order=k, mode='wrap')`` is ``C @ f`` with
    ``C[i, j] = K[(i - j) mod n]``, ``K`` the kernel of ``scipy.ndimage._filters._gaussian_kernel1d``
    folded onto the axis -- ``exp(-x^2 / 2 sigma^2)`` normalised to unit sum over
    ``x = -radius..radius``, and ``-x / sigma^2`` times that for the first derivative. The fold
    is what ``'wrap'`` does with a kernel longer than the axis; measured equal to scipy to 3e-16
    with a radius three times the axis. ``radius`` is the one scipy chose eagerly, recorded.
    """
    x_np = np.arange(-radius, radius + 1, dtype=float)
    x = _const(xp, x_np, like)
    phi = xp.exp(-0.5 / sigma ** 2 * x ** 2)
    phi = phi / xp.sum(phi)
    slope = -x / sigma ** 2 * phi
    fold = np.zeros((x_np.size, n))
    fold[np.arange(x_np.size), np.arange(-radius, radius + 1) % n] = 1.0
    fold = _const(xp, fold, like)
    index = ((np.arange(n)[:, None] - np.arange(n)[None, :]) % n).ravel()

    def circulant(kernel):
        folded = xp.reshape(kernel, (1, -1)) @ fold
        flat = as_array(xp, index, dtype=xp.int64, device=_device(like))
        return xp.reshape(xp.take(xp.reshape(folded, (-1,)), flat), (n, n))
    return circulant(phi), circulant(slope)
