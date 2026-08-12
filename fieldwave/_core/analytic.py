"""Single-source analytic (staircase-free) Fourier modeling.

A layer is described by a background index plus a list of parametric shapes — rectangles,
circles, ellipses — and its permittivity is Fourier-transformed *analytically* rather than
sampled onto a raster. That removes staircase error entirely, so the same accuracy needs far
fewer harmonics on curved or tilted boundaries.

A layer is ``dict(n_base=..., shapes=[...], combine=...)`` and a shape is
``dict(type='rectangle'|'circle'|'ellipse', cx=, cy=, lx=, ly=, angle=, n_index=)``.

**Shapes superpose; they do not paint over each other.** Each one contributes
``(eps_shape - eps_base) * FT(indicator)``, so where two overlap the layer gets *both*
contributions and the permittivity there is neither shape's. Two 120x120 rods of index 2 and 3
in a background of 1, overlapping by 7200 of a 90000 cell, come out at a mean permittivity of
2.76 where painting the second over the first gives 2.52 -- and the efficiency follows, 0.9856
against the 0.9792 a raster and the block path agree on. The block path
(``ucell=[[n_base, [[shape, ...], ...]]]``) is the one that paints, later object over earlier,
so the same recipe means two different structures depending on which path it is handed to.

``combine`` is what makes that choice explicit. It is a property of the *layer*, not of this
module: both shape paths read it, and each refuses the rule it cannot build, so a recipe
carrying one cannot be quietly given the other. :func:`check_layer_combination` enforces it
here and :func:`fieldwave._core.modeler.check_block_layer_combination` there.

While no two shapes intersect the question does not arise and any spelling is accepted on
either path — that keeps ``combine`` from taxing the case this path exists for. An overlap is
where they part:

=========  ==========================  ====================================================
path       unlabelled                  refuses
=========  ==========================  ====================================================
analytic   refused                     nothing — ``'add'`` is built here, ``'paint'`` by
                                       :mod:`.boundary`, which resolves the overlap into
                                       disjoint polygons first and transforms those
vector     painted (its own rule)      ``'add'`` — a staircase of blocks holds one index
                                       per block, so an overlap can only take one of them
=========  ==========================  ====================================================

The asymmetry in the middle column is deliberate. Painting is what the block path does and
what a raster of the same recipe agrees with, so there is nothing surprising to refuse;
superposition here is the rule nobody expects, so this path asks rather than assumes. The
layer that used to come back silently different now names the two shapes and both ways
forward.

``'paint'`` used to be refused here too, on the grounds that it needs the transform of a shape
*difference*. It still does — :mod:`.boundary` computes that difference as a polygon rather
than in closed form, which is what made it buildable.

The check costs a pairwise geometry test per layer (see :mod:`fieldwave._core.overlap`) and
needs concrete numbers, so a layer whose parameters are tracers is passed through unchecked;
:meth:`freeze_geometry` is the eager call that still sees them.

**Shape parameters are absolute lengths, in whatever unit ``period`` is given in** -- not
fractions of the period. A rod filling half a 0.77-long period is ``lx=0.385``, centred at
``cx=0.385``; the defaults say the same thing (``cx`` defaults to ``period_x / 2``). Passing
fractions builds a rod ``1/period`` times too large and says nothing about it, so
``test_analytic_shape_parameters_are_absolute_lengths`` pins the distinction.

Everything here is differentiable with respect to the shape parameters (``cx``, ``cy``, ``lx``,
``ly``, ``angle``, ``n_index``) — those are what an inverse-design user actually optimizes. The
one primitive that made this hard, Bessel ``J1``, lives in :mod:`fieldwave._core.backend`: SciPy
has it, PyTorch's has no autograd backward, and JAX's returns NaN near zero, so each backend
reaches it a different way behind one name.

The convolution matrices are built with the NVF (Li / Schuster et al.) normal-vector
factorization: ``eps`` and ``1/eps`` come from the exact shape transform, while the normal
projectors come from the shapes' *analytic* unit normal field sampled on a grid.

``cx`` and ``cy`` are the exception to "sampled": a lone shape's normal field is built at the
centre of the cell and translated by a phase, because sampling it in place both kills the
position derivative and makes the answer depend on where in the cell the shape sits. See
:func:`normal_projector_coeffs`.
"""
from . import overlap as _overlap
from .backend import array_namespace, as_array, j1, jinc, sinc
from .lattice import sample_grid
from .overlap import COMBINE_MODES, OVERLAP_TOL_REL      # re-exported; both paths read them


def _grid(xp, fto_x, fto_y, dev, dtype):
    """Integer reciprocal grid ``(KY, KX)`` spanning the doubled range a convolution matrix needs.

    Doubled because the matrix is indexed by *differences*: ``r - c`` and ``s - d`` each reach
    ``+-2*fto``, so a grid of ``2*fto+1`` per axis cannot fill its corners. See
    ``docs/factorization-terms.md`` for the layout and why the result is BTTB, not Toeplitz.
    """
    ky = xp.asarray([float(m) for m in range(-2 * fto_y, 2 * fto_y + 1)], dtype=dtype, device=dev)
    kx = xp.asarray([float(m) for m in range(-2 * fto_x, 2 * fto_x + 1)], dtype=dtype, device=dev)
    return ky[:, None] + 0.0 * kx[None, :], 0.0 * ky[:, None] + kx[None, :]


def shape_geometry(shape, Px, Py):
    """``(type, cx, cy, lx, ly, angle)`` for one shape dict, resolved once for everybody.

    It exists because the transform and the normal field used to resolve them separately and
    disagree. Two ways that showed:

    * ``circle`` is round. Its transform only ever used ``lx`` as the diameter, while the
      normal field read ``ly`` as a second semi-axis — so ``ly != lx`` on a circle silently
      built one structure for ``eps`` and a different one for the projector (14% on
      ``sum(R)``). ``ly`` is now ``lx`` for a circle wherever it is read; an oval is an
      ``ellipse``.
    * the defaults differed (``ly`` fell back to ``lx`` in one and to ``500.0`` in the other),
      so *omitting* ``ly`` on a circle — the natural way to write it — degenerated the normal
      field and moved the answer 6%.

    *Everybody* means the polygon path as well:
    :func:`~fieldwave._core.boundary.shape_vertices` was a third reader with its own rules
    until 2026-08-09 — no default for ``ly``, none for ``cx`` / ``cy``, and a ``square`` that
    kept whatever ``ly`` it was handed. The first of those is how the same circle dict summed
    cleanly and raised ``KeyError: 'ly'`` the moment it was painted.

    Lengths are absolute, in the units of ``period`` (see the module docstring).
    """
    stype = shape.get('type', 'rectangle')
    cx = shape.get('cx', Px / 2.0)
    cy = shape.get('cy', Py / 2.0)
    lx = shape.get('lx', shape.get('l', 0.0))
    round_ = stype in ('circle', 'disk', 'square')
    ly = lx if round_ else shape.get('ly', shape.get('l', 0.0))
    return stype, cx, cy, lx, ly, shape.get('angle', 0.0)


def _primitives(shapes, Px, Py):
    """The layer's shapes as convex primitives, or ``None`` if the parameters are not concrete.

    The overlap test compares numbers, which a JAX tracer is not — and PyTorch tensors, which
    do convert, keep working. Any backend's refusal to hand over a value is that backend
    saying "this is traced", so it means *skip the check*, never *fail the solve*: a traced
    call already had its eager pass through :meth:`freeze_geometry`, and turning a
    differentiable sweep into an exception would be a worse bargain than the one this check
    exists to fix.
    """
    try:
        return [_overlap.primitive(*shape_geometry(shape, Px, Py)) for shape in shapes]
    except Exception:                                  # noqa: BLE001 - "not a concrete value"
        return None


def _shape_name(shapes, i):
    return f"{i} ({shapes[i].get('type', 'rectangle')})"


def _across(ox, oy):
    """Which period boundary a periodic image sits over, as a phrase or ``''`` for none."""
    axes = [name for name, off in (('x', ox), ('y', oy)) if off]
    return f" across the {' and '.join(axes)} period boundary" if axes else ''


#: Keys every shape takes, whatever its type.
_COMMON_KEYS = frozenset({'type', 'cx', 'cy', 'angle', 'n_index'})

#: The length keys each type takes, on top of :data:`_COMMON_KEYS`. ``l`` is the both-axes
#: spelling. The round types have no ``ly`` **because they have no second length** — that is
#: the whole content of this table, and the reason it is a table rather than a comment.
_LENGTH_KEYS = {
    'rectangle': frozenset({'lx', 'ly', 'l'}),
    'ellipse': frozenset({'lx', 'ly', 'l'}),
    'square': frozenset({'lx', 'l'}),
    'circle': frozenset({'lx', 'l'}),
    'disk': frozenset({'lx', 'l'}),
}


def check_shape_spec(shape, shape_index=0, layer_index=0):
    """Refuse a shape dict that says something this path will not do.

    Every key here is read with ``.get`` and a default, which is what makes
    :func:`shape_geometry` total — it never raises, so it stays usable under a trace. The cost
    of being total is that a key nobody reads is a key nobody complains about: ``angel=0.7``
    silently gave no rotation, ``radius=0.2`` silently gave a shape of size zero, and ``ly`` on
    a circle silently gave a different structure to each of the paths that read it. This is
    where that is paid for instead, once, at the funnel.

    On ``ly`` for a round type the rule is about the **key**, not the value: a ``circle`` does
    not take one, whatever it would have been set to. Refusing it only when it *differs* from
    ``lx`` was tried first and is worse in every direction —

    * it cannot see a traced ``ly`` at all, which is the case that matters most. A
      differentiable ``ly`` on a circle is dropped, so ``d(loss)/d(ly)`` is *exactly zero* and
      an optimiser reads that as a converged direction. Measured: 866.9 for ``lx`` against 0.0
      for ``ly`` on the same circle, where an ``ellipse`` gives 886.3 and 351.7.
    * comparing values then needs a tolerance (a float32 backend round-trips 0.3 to
      0.30000001) and an identity escape for ``ly=lx`` sharing one tracer — about 25 lines,
      all of it to accept a spelling that says nothing the shape did not already say.

    Key presence is a plain dict lookup, so this behaves identically eager, jitted and under
    ``grad``. That is the whole reason it is phrased this way.
    """
    where = f'Analytic layer {layer_index}, shape {shape_index}'
    stype = shape.get('type', 'rectangle')
    lengths = _LENGTH_KEYS.get(stype)
    if lengths is None:
        raise ValueError(f'{where}: type={stype!r} is not a shape this path can build. '
                         f'Known types: {sorted(_LENGTH_KEYS)}.')

    if 'ly' in shape and 'ly' not in lengths:
        # Answered here rather than by the unknown-key branch below, which would offer 'l' as
        # the near miss and send a user who wants an oval further from `ellipse`, not closer.
        alt = 'ellipse' if stype in ('circle', 'disk') else 'rectangle'
        raise ValueError(
            f"{where}: {stype!r} has one length, so 'ly' would be dropped -- silently, and "
            f"with a zero gradient if it is a parameter you are optimising. Drop it if 'lx' "
            f"is the size you meant, or use type={alt!r} to give two lengths.")

    unknown = set(shape) - (lengths | _COMMON_KEYS)
    if unknown:
        import difflib
        known = sorted(lengths | _COMMON_KEYS)
        bits = []
        for key in sorted(unknown):
            near = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
            bits.append(f'{key!r}' + (f" (did you mean {near[0]!r}?)" if near else ''))
        raise ValueError(f'{where}: {stype!r} does not take {", ".join(bits)}. '
                         f'It takes {known}. A key nobody reads is silently ignored, which is '
                         f'why this is refused rather than dropped.')


#: Everything a layer dict may say. Keep this in step with what the builders below read.
_LAYER_KEYS = frozenset({'n_base', 'base', 'shapes', 'combine'})


def check_layer_spec(layer, layer_index=0):
    """Refuse a layer dict that says something nothing here reads.

    Same argument as :func:`check_shape_spec` one level up, and the same cost of not having it:
    every key is read with ``.get`` and a default, so a misspelling is not an error, it is a
    *different structure* returned without complaint. A layer whose ``phi`` is spelled ``pih``
    has no shapes either, and an empty shape list is perfectly legal — so it comes back as a
    uniform slab of ``n_base``, solves, conserves energy, and is wrong. There is nothing
    downstream that could notice.
    """
    unknown = sorted(set(layer) - _LAYER_KEYS)
    if unknown:
        raise ValueError(
            f'Analytic layer {layer_index}: {unknown} is not read by any path here. A layer '
            f'takes {sorted(_LAYER_KEYS)}. Refused rather than ignored, because an unread key '
            f'leaves a layer that still *builds*: a dict carrying nothing this reads has no '
            f'shapes either, and an empty shape list is legal, so it comes back as a uniform '
            f'slab of n_base — which solves, conserves energy to 1e-14, and is not the '
            f'structure that was asked for.')


def check_layer_combination(layer, period, layer_index=0, tol_rel=OVERLAP_TOL_REL,
                            lattice=None):
    """Refuse a layer whose shapes overlap without saying how they should combine.

    This is where the smooth path's superposition stops being implicit. Shapes that do not
    intersect mean the same structure under either rule, so nothing is refused and ``combine``
    need not be given; shapes that *do* intersect mean two different structures, and the layer
    has to pick one. Raises :class:`ValueError` when it has not. Both answers can now be built —
    ``'add'`` here and ``'paint'`` by :mod:`.boundary`, which resolves the overlap into
    polygons first — so neither is refused, only silence is.

    Touching is not overlapping — abutting rectangles and a shape exactly filling the period
    both come out at a gap of zero — and ``tol_rel`` is measured against the period, so a
    fraction of a cell of interpenetration is still allowed as arithmetic noise.

    Also runs :func:`check_shape_spec` over every shape, ahead of everything else here. This
    function is the funnel the smooth path reaches the solver through, and — unlike
    :func:`shape_geometry`, where the check would sit closer to the mistake — it is not inside
    :func:`_primitives`, whose ``except Exception`` would catch the refusal and answer it by
    silently skipping the overlap check.
    """
    combine = layer.get('combine')
    if combine is not None and combine not in COMBINE_MODES:
        raise ValueError(
            f'Analytic layer {layer_index}: combine={combine!r} is not one of '
            f'{COMBINE_MODES}.')

    shapes = layer.get('shapes', [])
    for i, shape in enumerate(shapes):
        check_shape_spec(shape, i, layer_index)
    if len(shapes) < 1:
        return

    try:
        Px, Py = float(period[0]), float(period[1])
    except Exception:                                  # noqa: BLE001 - traced period
        return

    prims = _primitives(shapes, Px, Py)
    if prims is None:
        return

    hits = _overlap.overlapping_pairs(prims, Px, Py, tol_rel * max(Px, Py), lattice)
    if not hits or combine == 'add':
        return

    i, j, ox, oy, _ = hits[0]
    more = f' (and {len(hits) - 1} more such pair(s))' if len(hits) > 1 else ''
    if i == j:
        what = (f'shape {_shape_name(shapes, i)} is larger than the period and runs into its '
                f'own image{_across(ox, oy)}')
    else:
        what = (f'shapes {_shape_name(shapes, i)} and {_shape_name(shapes, j)} '
                f'overlap{_across(ox, oy)}')

    if combine == 'paint':
        return          # handled by the boundary path, not here — see :mod:`.boundary`

    raise ValueError(
        f'Analytic layer {layer_index}: {what}{more}. This path adds every shape\'s '
        f'(eps_shape - eps_base) * FT(indicator), so the overlap gets both contributions and '
        f"ends up as neither shape's material, while the block path paints one over the "
        f'other — the same recipe, two different structures. Say which you mean:\n'
        f"  - combine='add' in the layer dict to keep the superposition computed here;\n"
        f'  - the block path, ucell=[[n_base, [[shape, ...], ...]]], to paint later shapes '
        f'over earlier ones.')


def layer_needs_resolving(layer, period, lattice=None):
    """Does this layer actually need the boundary path — i.e. is it painted *and* overlapping?

    While nothing intersects, painting and adding describe the same structure, and the closed
    form here is both exact and cheaper than resolving a polygon that has nothing to resolve.
    Routing on the spelling alone would also make two names for one structure return different
    numbers (measured 0.8% apart on disjoint rods), which is the surprise ``combine`` exists to
    remove.

    Raises when the parameters are not concrete and no plan was frozen. That case used to
    return ``False`` on the reasoning that a traced call has already had its eager pass through
    :meth:`freeze_geometry` — but nothing enforced it, and the cost of the assumption being
    wrong is not a slower path or a warning. It is a **different structure**: the layer is
    summed instead of painted, silently. Measured under ``jax.grad`` on two overlapping rods,
    ``T`` came out 0.8743 where painting gives 0.8539, and the gradient was bit-identical to
    the one ``combine='add'`` returns — which is what identified it.

    PyTorch never hit this: its tensors convert to floats, so the check ran. JAX tracers do
    not, so it silently did not.
    """
    if layer.get('combine') != 'paint':
        return False
    shapes = layer.get('shapes', [])
    if len(shapes) < 2:
        return False
    try:
        Px, Py = float(period[0]), float(period[1])
        prims = _primitives(shapes, Px, Py)
    except Exception:                                  # noqa: BLE001 - traced period
        prims = None
    if prims is None:
        raise ValueError(
            "combine='paint' on a traced layer needs a frozen geometry. Whether the shapes "
            "overlap decides which structure this is -- painted, or summed -- and that "
            "question cannot be asked of a traced value. Taking the summed branch anyway "
            "would solve a different structure than the one asked for, without saying so.\n"
            "  Freeze once, outside the trace, and replay:\n"
            "    geometry = solver.freeze_geometry()\n"
            "    result = solver.conv_solve(geometry=geometry)\n"
            "  Re-take the freeze whenever the shapes may have started or stopped "
            "overlapping; the plan compares by content, so an unchanged one is free.")
    return bool(_overlap.overlapping_pairs(prims, Px, Py, OVERLAP_TOL_REL * max(Px, Py),
                                           lattice))


def shape_ft(xp, shape, ky, kx, period, lattice=None):
    """Continuous Fourier transform of one shape's indicator, divided by the cell area.

    Returns ``c(ky, kx) = 1/area * FT{1_shape}(u, v)`` on the integer reciprocal grid — the
    analytic analogue of one ``dfs2d`` Fourier coefficient.

    ``lattice`` is the cell basis (``None`` = the axis-aligned ``diag(period)``). It reaches
    only the two lines below, and that is the whole of what a skewed cell costs this function:
    ``(u, v)`` is the reciprocal vector ``G = m*b1 + n*b2`` in **Cartesian** components, and
    everything after it — the rotation, the phase, the sinc product, the Bessel forms — is
    already written in Cartesian frequency and so does not care how ``G`` was reached.

    Nothing about the shape deforms either, because shape lengths are absolute Cartesian
    quantities (see :func:`shape_geometry`), not fractions of the cell. A rectangle stays a
    rectangle under a skew; only the grid of frequencies its transform is *sampled at* stops
    being rectangular. That is why this path needed far less than the raster one, where the
    normal field genuinely is index-based and needs the metric.
    """
    Px, Py = period[0], period[1]
    stype, cx, cy, lx, ly, angle = shape_geometry(shape, Px, Py)

    pi = 3.141592653589793
    if lattice is None:
        u = 2.0 * pi * kx / Px
        v = 2.0 * pi * ky / Py
        area = Px * Py
    else:
        # (u, v) = 2*pi * A^-T @ (m, n), written out for the same reason as in
        # fieldwave._core.lattice.order_kxy: one expression per component, no matrix inverse.
        a1x, a2x = float(lattice[0][0]), float(lattice[0][1])
        a1y, a2y = float(lattice[1][0]), float(lattice[1][1])
        det = a1x * a2y - a2x * a1y
        u = 2.0 * pi * (a2y * kx - a1y * ky) / det
        v = 2.0 * pi * (-a2x * kx + a1x * ky) / det
        area = abs(det)

    # Rotated frequency coordinates: kappa = Rot(angle)^T @ k
    ct = xp.cos(as_array(xp, angle, dtype=u.dtype))
    st = xp.sin(as_array(xp, angle, dtype=u.dtype))
    kxr = u * ct + v * st
    kyr = -u * st + v * ct

    phase = xp.exp(-1j * (u * cx + v * cy))

    if stype in ('rectangle', 'square'):
        s = lx * sinc(kxr * lx / 2.0) * ly * sinc(kyr * ly / 2.0)
    elif stype in ('circle', 'disk'):
        R = lx / 2.0
        # Same DC guard as the ellipse below, and for the same reason: ``rho`` is zero there,
        # ``d(sqrt)/d`` is infinite, and ``jinc``'s own ``where`` turns that into NaN. Here the
        # radicand depends on ``period``, so it is ``d(R)/d(period)`` that used to come back
        # NaN on a circle while the guarded ellipse was fine.
        r2 = u * u + v * v
        nz = r2 != 0
        rho = xp.where(nz, xp.sqrt(xp.where(nz, r2, xp.ones_like(r2))), xp.zeros_like(r2))
        s = (pi * R * R) * jinc(R * rho)          # 2 pi R J1(R rho)/rho
    elif stype == 'ellipse':
        a = lx / 2.0
        b = ly / 2.0
        # Guard the DC point so d/d(shape) of sqrt(0) is not NaN.
        kap2 = (a * kxr) ** 2 + (b * kyr) ** 2
        nz = kap2 != 0
        kap = xp.sqrt(xp.where(nz, kap2, xp.ones_like(kap2)))
        j = xp.where(nz, 2.0 * j1(kap) / xp.where(nz, kap, xp.ones_like(kap)),
                     xp.ones_like(kap))
        s = (pi * a * b) * j
    else:
        raise ValueError(f'Unknown analytic shape type: {stype!r}')

    return phase * s / area


def assemble_eps_coeffs(xp, shapes, n_base, fto_x, fto_y, period, dtype, lattice=None):
    """Analytic Fourier coefficients of ``eps`` and ``1/eps`` for one layer.

    Each grid has shape ``(4*fto_y+1, 4*fto_x+1)``, indexed ``[ky+2*fto_y, kx+2*fto_x]`` — the
    doubled range a ``(2*fto+1)**2`` convolution matrix needs (``docs/factorization-terms.md``
    §2 for why it doubles). The ``n**2`` maps refractive index
    to permittivity: FieldWave works in ``e^{-iwt}`` throughout (``Im(n) > 0`` = loss), with no
    conjugation on the way in.
    """
    dev = getattr(n_base, 'device', None) if hasattr(n_base, 'device') else None
    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32
    KKY, KKX = _grid(xp, fto_x, fto_y, dev, real_dtype)

    eps_bg = as_array(xp, n_base, dtype=dtype) ** 2
    inv_bg = 1.0 / eps_bg

    # DC term only: the background is constant over the whole cell.
    is_dc = (KKY == 0) & (KKX == 0)
    zero = xp.zeros(KKY.shape, dtype=dtype)
    eps_c = xp.where(is_dc, eps_bg, zero)
    inv_eps_c = xp.where(is_dc, inv_bg, zero)

    for shape in shapes:
        eps_s = as_array(xp, shape['n_index'], dtype=dtype) ** 2
        ft = xp.astype(shape_ft(xp, shape, KKY, KKX, period, lattice), dtype)
        eps_c = eps_c + (eps_s - eps_bg) * ft
        inv_eps_c = inv_eps_c + (1.0 / eps_s - inv_bg) * ft

    return eps_c, inv_eps_c


def laurent_conv_2d(xp, coeff2d, fto_x, fto_y):
    """Convolution matrix from a 2D coefficient grid.

    ``M[(r,s),(c,d)] = coeff(ky=r-c, kx=s-d)`` — the same ordering ``dfs2d`` produces. The
    sign is the order-index convention (see ``_circulant_index``); the two builders have to
    agree with each other and with ``kx``, which ``test_laurent_matches_dfs2d_convention``
    pins.

    **Rule-neutral, despite the name.** ``Laurent`` here is the *direct* factorization rule,
    and this builder implements neither rule: :func:`to_conv_mat_smooth_nvf` calls it on
    ``eps`` and on ``1/eps`` two lines apart, and the inverse rule's inversion happens in the
    caller. It is also called on the three normal-field products.

    The result is **BTTB** (block-Toeplitz with Toeplitz blocks), not Toeplitz: with the row
    and column pairs flattened, ``i - j`` does not determine ``(r-c, s-d)``, so the diagonals
    are not constant. ``docs/factorization-terms.md`` has the check and the vocabulary.
    """
    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1
    dev = _device(coeff2d)

    iy = xp.asarray([[[[r - c + 2 * fto_y]]] for r in range(ff_y) for c in range(ff_y)],
                    device=dev)
    iy = xp.reshape(iy, (ff_y, ff_y))
    ix = xp.asarray([[s - d + 2 * fto_x for d in range(ff_x)] for s in range(ff_x)], device=dev)

    m4 = coeff2d[iy[:, None, :, None], ix[None, :, None, :]]      # [r, s, c, d]
    return xp.reshape(m4, (ff_y * ff_x, ff_y * ff_x))


def _device(x):
    return x.device if hasattr(x, 'device') else None


def normal_vector_field_analytical(xp, grid_shape, shapes, period, dtype, lattice=None):
    """Unit normal field of the shapes, sampled on ``grid_shape`` — analytic, not from a raster.

    The sample points come from :func:`fieldwave._core.lattice.sample_grid`, which is where the
    sample grid's ``(Px, Py)`` unpack lives and why it lives in exactly one place — this body
    used to carry its own copy, and ``boundary.distance_gradient_normal_field`` was written from
    that copy.

    ``period`` is still read here, though, and not only for the grid: :func:`shape_geometry`
    takes the pair as well and is reached whether or not a basis was given, so a shape that
    leaves ``cx`` / ``cy`` to default is sensitive to the order even when the sample grid is
    built from a lattice. That is what makes this path less contained than the boundary one, and
    it had to be *measured* rather than read off the shared line: reversing the pair on a 2:1
    cell moves this field by 1.96 with an explicit centre and no lattice, by 1.7 with a defaulted
    centre and a lattice, and by exactly 0 with both.

    Written without in-place mutation so it stays differentiable in the shape parameters — with
    the exception of ``cx`` and ``cy``, which the rectangle branch reaches only through
    ``sign()`` and a comparison. Both have zero derivative everywhere, so a position never
    survives this function.

    That is why :func:`normal_projector_coeffs` does not put a position in: it builds the field
    with the shape at the centre of the cell and translates the *coefficients*.
    """
    Ny, Nx = grid_shape
    Px, Py = period[0], period[1]
    X, Y = sample_grid(xp, grid_shape, period, dtype, lattice)

    nx_total = xp.zeros((Ny, Nx), dtype=dtype)
    ny_total = xp.zeros((Ny, Nx), dtype=dtype)

    for s in shapes:
        stype, cx, cy, lx, ly, angle = shape_geometry(s, Px, Py)

        cos_a = xp.cos(as_array(xp, angle, dtype=dtype))
        sin_a = xp.sin(as_array(xp, angle, dtype=dtype))
        dx, dy = X - cx, Y - cy
        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a

        a, b = lx / 2.0, ly / 2.0
        if stype in ('ellipse', 'circle'):
            gxr = 2.0 * xr / (a ** 2)
            gyr = 2.0 * yr / (b ** 2)
        elif stype == 'rectangle':
            # Nearest face decides the normal direction.
            mask_x = (xp.abs(xr) - a) >= (xp.abs(yr) - b)
            zeros = xp.zeros_like(xr)
            gxr = xp.where(mask_x, xp.sign(xr), zeros)
            gyr = xp.where(mask_x, zeros, xp.sign(yr))
        else:
            continue

        gx = gxr * cos_a - gyr * sin_a
        gy = gxr * sin_a + gyr * cos_a
        norm = _hypot_safe(xp, gx, gy)
        nx_total = nx_total + gx / norm
        ny_total = ny_total + gy / norm

    norm_t = _hypot_safe(xp, nx_total, ny_total)
    return nx_total / norm_t, ny_total / norm_t


def _hypot_safe(xp, a, b):
    """``hypot`` with zeros replaced by 1, so the division that follows is well defined.

    The guard is on the *input* to the root, not on its output. Patching the output
    (``where(n == 0, 1, n)`` on ``n = (a*a + b*b)**0.5``) fixes the forward value and nothing
    else: autograd still differentiates the branch ``where`` discarded, ``d/da`` of that root
    at ``a == b == 0`` is infinite, and ``where``'s ``0 * inf`` is NaN. Taking the root of the
    replaced radicand instead means it is only ever evaluated at 1, where the derivative is
    finite — and ``sqrt(1) == 1`` is the same replacement value, so the forward result is
    unchanged.

    Both callers reach zero on ordinary geometry: a shape's own centre gives ``g == 0``, and
    two shapes placed symmetrically make their unit normals exactly opposed on the segment
    between them, which the grid lands on exactly.
    """
    n2 = a * a + b * b
    return xp.where(n2 == 0, xp.ones_like(n2), n2) ** 0.5


def _sampled_coeffs(xp, field, fto_x, fto_y):
    """Fourier coefficients of a sampled field on the doubled reciprocal grid.

    ``c(ky, kx) = 1/(Ny*Nx) * sum f[iy, ix] exp(-2i*pi*(ky*iy/Ny + kx*ix/Nx))``, laid out the
    way :func:`assemble_eps_coeffs` lays its grids out. Feeding it to :func:`laurent_conv_2d`
    reproduces ``dfs2d(field, 1, 1, ...)`` — that equivalence is what
    ``test_laurent_matches_dfs2d_convention`` states, and it is what lets a translation be
    applied here as a phase instead of by moving the samples.
    """
    Ny, Nx = field.shape
    dev = _device(field)
    F = xp.fft.fft2(field) / (Ny * Nx)
    iy = xp.asarray([k % Ny for k in range(-2 * fto_y, 2 * fto_y + 1)], device=dev)
    ix = xp.asarray([k % Nx for k in range(-2 * fto_x, 2 * fto_x + 1)], device=dev)
    return F[iy[:, None], ix[None, :]]


def normal_projector_coeffs(xp, shapes, grid_shape, period, fto_x, fto_y, dtype, lattice=None):
    """Coefficient grids of ``nx**2``, ``nx*ny`` and ``ny**2`` — the NVF projectors' input.

    **One shape is built at the centre of the cell and carried to ``(cx, cy)`` by a phase**,
    the same way :func:`assemble_eps_coeffs` carries the permittivity. Two things follow, and
    both are the point.

    *The position becomes differentiable.* Sampling the field at the shape's actual place puts
    ``cx`` and ``cy`` through ``sign()``, whose derivative is zero, so the projectors used to
    contribute *nothing* to ``d/dcy`` — while the permittivity term contributed its exact
    share. Translation invariance holds by those two cancelling, so what was left was the
    permittivity term standing alone: -1.7e-1 on ``d(sum R)/dcy`` for a rectangle, against
    a true value of zero. Through the phase the two carry the same factor and cancel to
    machine precision.

    *And the model becomes translation-covariant, which it was not.* The normal field is
    defined over the whole plane — a rectangle's wedges and a circle's rays do not stop at the
    shape — so sampling it over a fixed cell clips a different part of it depending on where
    the shape sits. That is not a discretization: ``<nx**2>`` for a circle moves 2.4e-2 between
    0.30 and 0.50 of the period and does not budge when the grid goes from 256 to 4096. It
    reached ``sum(R)`` as 1.6e-3, so the same structure translated gave a different answer.
    Built here it reproduces the centred one to 1e-14, at every ``fto``.

    Several shapes cannot use it, and the reason is worth stating precisely because it is two
    reasons. The field is a *normalized sum* (``sum(g/|g|)``, divided by its own norm), so a
    **common** translation is not carried by a phase and the clipped window comes back; and
    moving one shape **relative** to the others genuinely reshapes the field, which no phase
    describes at all. The second is the hard half: a rectangle's level-set gradient is piecewise
    constant, so its derivative is a line measure on the wedge boundaries that no grid sample
    sees, and the closed form that would serve a lone rectangle does not exist here either —
    ``|sum|`` varies continuously between two shapes, so the combined field is not piecewise
    constant even when both of them are rectangles. Measured: two circles, whose normals are
    linear in position, already agree with central differences to 4.7e-6; two rectangles are off
    by 11% to 105%.

    Three routes were measured and rejected.

    * *Interpolating* each shape's field onto the common grid holds the forward error to 7e-5,
      but the residue oscillates over one grid cell, so its derivative is 2e-1 — worse than the
      bias being removed.
    * *Centring on the mean of the shape centres* (which reduces to the rule above for one
      shape) fixes the common-translation half exactly, 1.2e-4 to 6.6e-15 on a rectangle pair.
      It manufactures the degeneracy it needs to avoid: two shapes then sit exactly opposed
      about the centre of the cell, their unit fields cancel exactly along the segment between
      them, and for two identical circles 16 grid points reach ``|n_total| == 0`` — where
      :func:`_hypot_safe` is finite but its gradient is ``0 * inf``. The common-translation
      gradient comes back NaN. It also moved the relative-position error for rectangles from
      1.6e-1 to 2.9e-1.
    * *Mollifying* ``sign()`` (``tanh(xr/w)``) removes the zero derivative at the source and
      would serve both halves, but the width that makes the position gradient accurate is not
      the width that keeps the answer smooth through ``angle = 0``: it broke
      ``test_analytic_angle_is_smooth_through_normal``, and any value chosen between the two is
      a tuning constant.

    Multi-shape layers therefore keep the sampled field. Both halves of what that costs are
    pinned in ``test_shape_position_gradients.py``, the working one as a test and the broken one
    as a strict xfail, so neither can be inferred from the other.

    **The forward answer is affected too, not only the gradient — keep every shape inside the
    cell.** A shape straddling the boundary may be named in any period; ``cx`` and ``cx + Px``
    are one structure. A single shape agrees exactly across periods because the phase carries
    it. Several shapes do not: for a rod at ``cx`` = 0.10 / 1.10 / 2.10 beside a fixed second
    rod, total transmission comes out 0.8532 / 0.8678 / 0.8665. Splitting the layer's inputs
    puts it here rather than in the transform — across one period the permittivity coefficients
    agree to 1.1e-15 while ``<nx**2>`` differs by 3.1e-1. The boundary path inherits it
    identically, bit for bit, since it takes its field from the same place. An optimizer left
    free to move a shape will walk it over the edge and climb the artefact, so clamp the
    centres. Pinned by ``test_a_multishape_layer_is_not_periodic_across_the_cell_edge``.
    """
    Px, Py = period[0], period[1]
    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32

    if len(shapes) == 1:
        _, cx, cy, _, _, _ = shape_geometry(shapes[0], Px, Py)
        if lattice is None:
            mid_x, mid_y = Px / 2.0, Py / 2.0
            # Fractions of a period -- which is what the phase below is written in.
            shift_y, shift_x = (cy - mid_y) / Py, (cx - mid_x) / Px
        else:
            a1x, a2x = float(lattice[0][0]), float(lattice[0][1])
            a1y, a2y = float(lattice[1][0]), float(lattice[1][1])
            mid_x, mid_y = 0.5 * (a1x + a2x), 0.5 * (a1y + a2y)
            # Same quantity, stated properly: the displacement in *fractional lattice*
            # coordinates, A^-1 @ (c - mid). The phase is exp(-2i*pi*(m*u_d + n*v_d)), which is
            # exp(-i G.d) already -- it was lattice-general all along, it was only ever handed
            # its argument as a fraction of Px / Py.
            det = a1x * a2y - a2x * a1y
            dx, dy = cx - mid_x, cy - mid_y
            shift_x = (a2y * dx - a2x * dy) / det
            shift_y = (-a1y * dx + a1x * dy) / det
        built = [dict(shapes[0], cx=mid_x, cy=mid_y)]
    else:
        built, shift_y, shift_x = shapes, None, None

    nx, ny = normal_vector_field_analytical(xp, grid_shape, built, (Px, Py), real_dtype,
                                            lattice)

    if shift_y is None:
        phase = 1.0
    else:
        KKY, KKX = _grid(xp, fto_x, fto_y, _device(nx), real_dtype)
        arg = KKY * shift_y + KKX * shift_x
        phase = xp.exp(xp.astype(arg, dtype) * (-2j * 3.141592653589793))

    return tuple(_sampled_coeffs(xp, xp.astype(f, dtype), fto_x, fto_y) * phase
                 for f in (nx * nx, nx * ny, ny * ny))


def to_conv_mat_smooth_nvf(xp, shapes, n_base, fto_x, fto_y, period, dtype,
                             normal_grid=256, lattice=None):
    """The five 2D-solve convolution matrices for one analytic layer, via NVF.

    ``eps`` / ``1/eps`` come from the exact shape transform (no staircase); the normal
    projectors come from the shapes' analytic normal field. Same Hermitian symmetrization as
    the raster NVF path — see :func:`fieldwave._core.factorization.nvf_conv_matrices`.

    Every one of the five is ``D X D^-1`` for the shape's translation phase ``D`` and a
    position-free ``X`` (see :func:`normal_projector_coeffs`), so sliding a single-shape layer
    conjugates the whole solve by a diagonal and leaves each order's efficiency alone.
    """
    eps_c, inv_eps_c = assemble_eps_coeffs(xp, shapes, n_base, fto_x, fto_y, period, dtype,
                                           lattice)
    c_xx, c_xy, c_yy = normal_projector_coeffs(
        xp, shapes, (normal_grid, normal_grid), period, fto_x, fto_y, dtype, lattice)
    return nvf_from_coeffs(xp, eps_c, inv_eps_c, c_xx, c_xy, c_yy, fto_x, fto_y)


def nvf_from_coeffs(xp, eps_c, inv_eps_c, c_xx, c_xy, c_yy, fto_x, fto_y):
    """Assemble the five convolution matrices from coefficient grids — the NVF tensor itself.

    The last step of every non-raster NVF path, and the same step for all of them: what differs
    between the analytic, boundary and level-set paths is *how* the five coefficient grids were
    obtained, never what is done with them. Kept in one place so a change to the tensor cannot
    reach one path and miss another — the raster path's equivalent is
    :func:`fieldwave._core.factorization.nvf_conv_matrices`.

    ``eps_c`` / ``inv_eps_c`` are the ``eps`` and ``1/eps`` coefficients on the doubled
    reciprocal grid; ``c_xx`` / ``c_xy`` / ``c_yy`` those of ``nx**2``, ``nx*ny``, ``ny**2``.
    Returns ``(eps_xx, eps_yy, eps_xy, eps_yx, epz_i)``.
    """
    ep_dir = laurent_conv_2d(xp, eps_c, fto_x, fto_y)
    ep_inv = laurent_conv_2d(xp, inv_eps_c, fto_x, fto_y)

    delta_ep = ep_dir - xp.linalg.inv(ep_inv)

    C_xx = laurent_conv_2d(xp, c_xx, fto_x, fto_y)
    C_xy = laurent_conv_2d(xp, c_xy, fto_x, fto_y)
    C_yy = laurent_conv_2d(xp, c_yy, fto_x, fto_y)

    eps_xx = ep_dir - 0.5 * (delta_ep @ C_xx + C_xx @ delta_ep)
    eps_yy = ep_dir - 0.5 * (delta_ep @ C_yy + C_yy @ delta_ep)
    eps_xy = -0.5 * (delta_ep @ C_xy + C_xy @ delta_ep)          # == eps_yx (symmetric)

    return eps_xx, eps_yy, -eps_xy, -eps_xy, xp.linalg.inv(ep_dir)


def to_conv_mat_smooth_layers(layer_shapes, fto_x, fto_y, period,
                                normal_grid=256, type_complex=None, xp=None,
                                boundary_plans=None, lattice=None):
    """Stack the analytic-NVF convolution matrices over layers.

    ``layer_shapes`` is one ``dict(n_base=..., shapes=[...])`` per layer, with the shapes'
    ``cx`` / ``cy`` / ``lx`` / ``ly`` as absolute lengths in the units of ``period`` (see the
    module docstring). Returns ``(epx, epy, epxy, epyx, epz_i)``, each
    ``(n_layers, ff_xy, ff_xy)``.

    Every layer goes through :func:`check_layer_combination` first — this is the funnel the
    smooth path reaches the solver through, so it is the one place that can guarantee an
    overlap is never answered silently.

    The array namespace is taken from the first differentiable shape parameter found, so
    passing JAX or PyTorch scalars for e.g. ``lx`` makes the whole build run — and
    differentiate — on that backend. Pass ``xp`` to force it.
    """
    if xp is None:
        xp = _infer_namespace(layer_shapes)
    dtype = xp.complex128 if type_complex is None else type_complex

    out = [[], [], [], [], []]
    for index, layer in enumerate(layer_shapes):
        check_layer_spec(layer, index)
        # A frozen geometry carries the decision for *every* layer, so when one is present it
        # is the whole answer: `plan is None` there means "looked at while the numbers were
        # concrete, and it does not overlap". Re-asking would put the question to traced
        # values that cannot answer it.
        frozen = boundary_plans is not None
        plan = boundary_plans[index] if frozen else None
        check_layer_combination(layer, period, index, lattice=lattice)
        n_base = layer.get('n_base', layer.get('base', 1.0))
        if plan is not None or (not frozen
                                and layer_needs_resolving(layer, period, lattice)):
            # A painted layer that actually overlaps is a different structure from a summed
            # one, and this module can only sum. Resolve the painting into disjoint polygons
            # and transform those. A painted layer with nothing overlapping means the same
            # structure either way, so it stays here -- exact, and cheaper.
            from .boundary import to_conv_mat_boundary_layer
            mats = to_conv_mat_boundary_layer(
                xp, layer.get('shapes', []), n_base, fto_x, fto_y, period, dtype,
                normal_grid=normal_grid, plan=plan, lattice=lattice)
        else:
            mats = to_conv_mat_smooth_nvf(
                xp, layer.get('shapes', []), n_base, fto_x, fto_y, period, dtype,
                normal_grid=normal_grid, lattice=lattice)
        for slot, m in zip(out, mats):
            slot.append(m)

    return tuple(xp.stack(slot) for slot in out)


def _infer_namespace(layer_shapes):
    """Namespace of the first array-like shape parameter; NumPy when everything is a scalar."""
    for layer in layer_shapes:
        candidates = [layer.get('n_base'), layer.get('base')]
        for shape in layer.get('shapes', []):
            candidates.extend(shape.get(k) for k in
                              ('cx', 'cy', 'lx', 'ly', 'l', 'angle', 'n_index'))
        for c in candidates:
            if c is not None and hasattr(c, 'dtype') and hasattr(c, 'shape'):
                return array_namespace(c)
    import numpy as np
    return array_namespace(np.zeros(1))
