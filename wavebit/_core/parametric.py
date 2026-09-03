"""Single-source parametric (staircase-free) Fourier modeling.

A layer is described by a background material plus a list of parametric shapes — rectangles,
circles, ellipses — and its permittivity is Fourier-transformed *analytically* rather than
sampled onto a raster. That removes staircase error entirely, so the same accuracy needs far
fewer harmonics on curved or tilted boundaries.

A layer is ``dict(background=..., shapes=[...])`` and a shape is
``dict(type='rectangle'|'circle'|'ellipse', cx=, cy=, lx=, ly=, angle=, material=...)``.

Both material slots take a :class:`~wavebit._core.domain.Material` -- relative permittivity
and the permeability that goes with it -- or a bare number, which is the scalar shorthand for
``Material(epsilon=x)``. They were ``n_base`` and ``n_index`` and held a refractive index; both
spellings are refused by name now, because the *quantity* changed with them and a rename that
kept the number would build a different, legal, energy-conserving structure
(``docs/conventions.md``).

**Shapes superpose in this module; they do not paint over each other.** Each one contributes
``(eps_shape - eps_base) * FT(indicator)``, so where two overlap the layer would get *both*
contributions and the permittivity there is neither shape's. Two 120x120 rods of permittivity
4 and 9 in a background of 1, overlapping by 7200 of a 90000 cell, come out at a mean
permittivity of 2.76 that way, where painting the second over the first gives 2.52 -- and the
efficiency follows, 0.9856 against the 0.9792 a raster of the same recipe agrees on.

So the sum is not what an overlapping layer means, and :func:`assemble_eps_coeffs` refuses one
outright. Such a layer is dispatched to :mod:`.boundary`, which resolves the overlap into
disjoint polygons -- later shape over earlier -- and transforms those, edges still exact. Where
nothing overlaps the two rules describe the same structure, so the layer stays here: exact, and
cheaper. Nothing is asked of the recipe either way; the overlap makes the choice.

A ``combine=`` key used to make it explicit, ``'add'`` for the sum and ``'paint'`` for the
painting, and each shape path refused the rule it could not build. It was removed on 2026-08-27
together with the sum: what ``'add'`` builds in an overlap is ``eps_a + eps_b - eps_bg``, which
is not a material, and being the rule nobody expects it had to be asked for -- which taxed every
layer, including the non-overlapping ones this path exists for.

``'paint'`` used to be refused here too, on the grounds that it needs the transform of a shape
*difference*. It still does — :mod:`.boundary` computes that difference as a polygon rather
than in closed form, which is what made it buildable.

The overlap test costs a pairwise geometry test per layer (see :mod:`wavebit._core.overlap`)
and needs concrete numbers, so a layer whose parameters are tracers is passed through unchecked;
:meth:`freeze_geometry` is the eager call that still sees them.

**Shape parameters are absolute lengths, in whatever unit ``period`` is given in** -- not
fractions of the period. A rod filling half a 0.77-long period is ``lx=0.385``. Passing
fractions builds a rod ``1/period`` times too large and says nothing about it, so
``test_parametric_shape_parameters_are_absolute_lengths`` pins the distinction.

**A centre is measured from the centre of the cell**, not from its corner, so that same rod
sits at ``cx=0`` -- which is also what an omitted ``cx`` means. See :func:`shape_geometry`.

Everything here is differentiable with respect to the shape parameters (``cx``, ``cy``, ``lx``,
``ly``, ``angle``, ``material``) — those are what an inverse-design user actually optimizes. A
material is a design variable like any other, which is why :class:`Material` is a registered
JAX pytree rather than an opaque leaf. The
one primitive that made this hard, Bessel ``J1``, lives in :mod:`wavebit._core.backend`: SciPy
has it, PyTorch's has no autograd backward, and JAX's returns NaN near zero, so each backend
reaches it a different way behind one name.

The convolution matrices are built with the NVF (Li / Schuster et al.) normal-vector
factorization: ``eps`` and ``1/eps`` come from the exact shape transform, while the normal
projectors come from the shapes' *analytic* unit normal field sampled on a grid.

``cx`` and ``cy`` are the exception to "sampled": a lone shape's normal field is built at the
centre of the cell and translated by a phase, which keeps its position derivative exact. See
:func:`normal_projector_coeffs`.

**Several shapes used to be the broken case and are not any more** (2026-08-17). The field was
a normalized *sum of unit vectors*, so a shape ten cells away pushed exactly as hard as the one
underfoot: with two shapes the normal at a boundary was already 40 deg from the true surface
normal, and with three, 43. It also sampled ``X - cx``, the offset to the shape *as named*, so
naming a shape one period over slid a different part of an infinite wedge into the cell and
moved the answer by 1.9e-2. Two changes fix both, and neither carries a width to tune: the
displacement is folded to the nearest image (:func:`_shape_surface_normal`), and the shapes are
combined by distance-product weights, which collapse to one shape's own normal exactly at its
surface. Boundaries now read 0.000 deg for any number of shapes and naming a period over moves
nothing (5.7e-13).

The same weighting removed ``sign()`` from the rectangle branch, which the position gradient
had been blocked on -- see :func:`_shape_surface_normal`.

**What it costs, which the commit that made the change did not measure.** The blend evaluates a
signed distance per shape and a product over the others, and a rectangle adds four blended faces
on top, so building the field is far dearer than the sum of level-set gradients it replaced:
31.9 ms to **378.6 ms** on a tilted-rectangle pair and 47.4 ms to **161.6 ms** on a circle pair,
at the ``normal_grid`` of 1024 the solver defaults to. That is 12x and 3.4x, and it is the price
of the boundary exactness and the position gradient above rather than an accident -- but it was
paid before it was counted, and it moves what is worth optimising: the field now costs 4-5x the
three FFTs that transform it (0.36 s against 0.08 s at N = 1024), so the transform is not the
place to look. Whole-solve share falls away with ``truncation_order`` as the eigendecomposition takes
over, from 8.5% at truncation_order 4 to 0.13% at truncation_order 16, which is the range where it might
matter to an optimizer looping at low truncation. Side effect worth knowing when reading
``docs/boundary-modeling.md``: the cost ratio it quotes for swapping in the boundary route's
field fell from 12x/486x to 2x/182x because this side got slower, not because that one got
faster.
"""
from . import overlap as _overlap
from .backend import array_namespace, as_array, checked_float, j1, jinc, sinc
from .domain import as_material
from .factorization import ConvMatrices, check_normal_grid
from .lattice import periods_of, is_axis_aligned, sample_grid
from .overlap import OVERLAP_TOL_REL
from .geometry import StaleGeometryError


def _grid(xp, half_width_u, half_width_v, dev, dtype):
    """Integer reciprocal grid ``(KY, KX)`` spanning the doubled range a convolution matrix needs.

    Doubled because the matrix is indexed by *differences*: ``r - c`` and ``s - d`` each reach
    ``+-2*truncation_order``, so a grid of ``2*truncation_order+1`` per axis cannot fill its corners. See
    ``docs/factorization-terms.md`` for the layout and why the result is BTTB, not Toeplitz.
    """
    ky = xp.asarray([float(m) for m in range(-2 * half_width_v, 2 * half_width_v + 1)], dtype=dtype, device=dev)
    kx = xp.asarray([float(m) for m in range(-2 * half_width_u, 2 * half_width_u + 1)], dtype=dtype, device=dev)
    return ky[:, None] + 0.0 * kx[None, :], 0.0 * ky[:, None] + kx[None, :]


def shape_geometry(shape, lattice):
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
    :func:`~wavebit._core.boundary.shape_vertices` was a third reader with its own rules
    until 2026-08-09 — no default for ``ly``, none for ``cx`` / ``cy``, and a ``square`` that
    kept whatever ``ly`` it was handed. The first of those is how the same circle dict summed
    cleanly and raised ``KeyError: 'ly'`` the moment it was painted.

    Lengths are absolute, in the units of ``period`` (see the module docstring).

    **``cx`` / ``cy`` are measured from the centre of the cell**, ``0.5*(a1 + a2)``, so a
    centred shape is ``cx = cy = 0`` -- which is also what omitting them means. What this
    returns is the absolute position everything downstream has always consumed: the offset is
    added here, in the one place a shape dict is read, so no builder knows the difference.

    Corner-relative until 2026-08-31. It moved to match S4, the only other solver that takes
    parametric shapes, whose centres are likewise *"relative to the center of the unit cell"*.
    A centred structure is the common case and ``0`` is the natural spelling for it; the
    corner spelling of the same thing was ``0.5*(a1 + a2)`` written out by hand, which on a
    skewed cell is two components, neither of them a period. Nothing has been released, so
    there is no shim: the old numbers were re-based at every call site instead.

    **The raster path did not move, and has no origin to move.** An array *is* the geometry,
    so there is no coordinate a caller can state against the wrong origin:
    ``structure[layer][j, i]`` is still the permittivity at ``(i+0.5)/Nx * a1 +
    (j+0.5)/Ny * a2`` (``README.md``). The other surface that did move is
    ``reconstruct_field``'s ``sample_x`` / ``sample_y``, which name points in the same frame.

    The origin needs the basis, and reading it off the *periods* was a bug: it was
    ``(Px/2, Py/2)`` until 2026-08-23, and that is the centre of ``diag(|a1|, |a2|)`` rather
    than of the cell whose columns are a1, a2 -- ``periods_of`` has already thrown the angle
    away by then, so a 60-degree hex and a square both report ``(400, 400)`` and both got the
    square's answer. Measured 26% of ``|a1|`` out on that hex, and on a shear the default
    landed *outside* the cell entirely. Every shape pays that arithmetic now rather than only
    the defaulted ones, which is the one thing the move buys beyond the spelling.

    A lone shape survives such a slip, which is why it lasted: a translation is a diagonal
    phase on every conv matrix, so one misplaced shape is misplaced and not missolved
    (2.3e-15). It bites the moment a second shape fixes the *relative* geometry -- 45.8% on
    ``sum(R)``, flat across truncation order, between two spellings of the same point lattice.
    ``tests/test_boundary.py:355`` recorded the defect in prose and pinned only that the
    readers agree with each other, which they still do: there is one origin and it moves them
    together.
    """
    stype = shape.get('type', 'rectangle')
    # The origin. It used to be built only for a shape that omitted its centre, and was spelled
    # as a branch rather than as `get(key, default)` for that reason -- the default expression
    # of a `get` is evaluated whether or not it is used, so on a traced basis it would have
    # been built for nothing. Every shape needs it now, so the branch is gone and the default
    # is the constant `0.0`, which a `get` may evaluate freely.
    #
    # There used to be a `Px / 2` arm here for `lattice is None`. It was the centre of
    # `diag(|a1|, |a2|)`, which on a shear is not inside the cell (`2d7ca08`), and the solver
    # stopped reaching it once `_geometry_lattice_for` became total (`cc016fc`) -- measured:
    # driving only the public API, the one function that ever sees a `None` here is
    # `as_lattice` itself, which is where the conversion happens.
    # Two reads, not four: `Lattice.__getitem__` stacks its two vectors on every access, and
    # this now runs for every shape rather than only for a defaulted one.
    row_x, row_y = lattice[0], lattice[1]
    cx = shape.get('cx', 0.0) + 0.5 * (row_x[0] + row_x[1])
    cy = shape.get('cy', 0.0) + 0.5 * (row_y[0] + row_y[1])
    lx = shape.get('lx', shape.get('l', 0.0))
    round_ = stype in ('circle', 'disk', 'square')
    ly = lx if round_ else shape.get('ly', shape.get('l', 0.0))
    return stype, cx, cy, lx, ly, shape.get('angle', 0.0)


def _primitives(shapes, lattice):
    """The layer's shapes as convex primitives, or ``None`` if the parameters are not concrete.

    The overlap test compares numbers, which a JAX tracer is not — and PyTorch tensors, which
    do convert, keep working. Any backend's refusal to hand over a value is that backend
    saying "this is traced", so it means *skip the check*, never *fail the solve*: a traced
    call already had its eager pass through :meth:`freeze_geometry`, and turning a
    differentiable sweep into an exception would be a worse bargain than the one this check
    exists to fix.
    """
    try:
        return [_overlap.primitive(*shape_geometry(shape, lattice)) for shape in shapes]
    except Exception:                                  # noqa: BLE001 - "not a concrete value"
        return None


def _shape_name(shapes, i):
    return f"{i} ({shapes[i].get('type', 'rectangle')})"


def _length_keys_of(shape):
    """What to *call* a shape's two half-widths when refusing one of them.

    It mirrors :func:`shape_geometry`'s own fallback, so the message quotes back the key that
    actually supplied the value: a dict saying ``l=-150`` is told about ``l``, not about an
    ``ly`` it never wrote, and ``lx=120, l=-150`` is told about ``l`` for the second one.

    A round type needs no case of its own. :func:`shape_geometry` sets its second length from
    its first, so the two half-widths are equal and the first is always what raises -- an
    ``ly`` a ``circle`` does not take (and that :func:`check_shape_spec` refuses as a *key*)
    can never be named. A branch for it was written first and deleted: nothing could reach it,
    which a mutation showed by surviving its removal.
    """
    first = 'lx' if 'lx' in shape else ('l' if 'l' in shape else 'lx')
    second = 'ly' if 'ly' in shape else ('l' if 'l' in shape else 'ly')
    return (first, second)


def _across(ox, oy):
    """Which period boundary a periodic image sits over, as a phrase or ``''`` for none."""
    axes = [name for name, off in (('x', ox), ('y', oy)) if off]
    return f" across the {' and '.join(axes)} period boundary" if axes else ''


#: Keys every shape takes, whatever its type.
_COMMON_KEYS = frozenset({'type', 'cx', 'cy', 'angle', 'material'})

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
_LAYER_KEYS = frozenset({'background', 'shapes'})


def check_layer_spec(layer, layer_index=0):
    """Refuse a layer dict that says something nothing here reads.

    Same argument as :func:`check_shape_spec` one level up, and the same cost of not having it:
    every key is read with ``.get`` and a default, so a misspelling is not an error, it is a
    *different structure* returned without complaint. A layer whose ``phi`` is spelled ``pih``
    has no shapes either, and an empty shape list is perfectly legal — so it comes back as a
    uniform slab of ``background``, solves, conserves energy, and is wrong. There is nothing
    downstream that could notice.
    """
    unknown = sorted(set(layer) - _LAYER_KEYS)
    if unknown:
        raise ValueError(
            f'Analytic layer {layer_index}: {unknown} is not read by any path here. A layer '
            f'takes {sorted(_LAYER_KEYS)}. Refused rather than ignored, because an unread key '
            f'leaves a layer that still *builds*: a dict carrying nothing this reads has no '
            f'shapes either, and an empty shape list is legal, so it comes back as a uniform '
            f'slab of the background — which solves, conserves energy to 1e-14, and is not the '
            f'structure that was asked for.')


def check_layer_shapes(layer, lattice, layer_index=0):
    """Check every shape's keys, and refuse a layer that needs a shape's periodic image.

    This function is the funnel the parametric path reaches the solver through, which is why the
    per-shape check is repeated here and — unlike :func:`shape_geometry`, where it would sit
    closer to the mistake — not inside :func:`_primitives`, whose ``except Exception`` would
    catch the refusal and answer it by silently building without that shape.

    A **non-positive length** is refused first, and separately, because it is not a smaller
    shape -- it is a subtracted one. :func:`shape_ft` carries the sign straight through to the
    ``G = 0`` coefficient, which is the fraction of the cell the shape claims: ``lx = +120`` on
    a 300 period gives ``+0.200000`` and ``lx = -120`` gives ``-0.200000``, so the shape takes
    its contrast *out* of the background instead of putting it in. A 120 x 150 rod of eps 4 on
    eps 1 comes back as eps 0.4 rather than 4, ``sum(T)`` 0.444002054 against 0.980652726, and
    at ``lx = -340`` the average permittivity is **negative** -- a metal, out of a spec that
    named two dielectrics. Every backend agrees on it, and nothing downstream can notice.

    It also silently disabled the two refusals below, which is why it is checked before them:
    :func:`~wavebit._core.overlap.primitive` stores ``lx / 2`` as a half-width, so a negative
    length is a negative projection radius and the separating-axis test finds every image
    clear -- ``overlapping_pairs`` returns ``[]`` for a rod of ``lx = -340`` on a 300 period,
    exactly as it does for one that fits. ``lx = +340`` is refused; ``lx = -340`` was built,
    and answered 0.047049717.

    An optimiser reaches this by the most ordinary move it has -- shrinking a feature. There is
    no cliff at zero to stop it: ``T`` is even to second order there (0.999998510 at +0.5
    against 0.999998461 at -0.5, the derivative merely changing sign), so a descent slides
    through and starts growing anti-material on the far side. Zero itself goes with it: a
    rectangle of ``lx = 0`` is a shape that is not there, and an ``ellipse`` of ``lx = 0``
    raises ``LinAlgError`` from inside the solve with nothing said about the geometry. An
    omitted length is the same case, since :func:`shape_geometry` defaults it to ``0.0``.

    **Neither builder here reads a shape's periodic images.** The closed form sums each shape's
    own transform; :func:`~wavebit._core.boundary.resolve_painting` clips loops in unwrapped
    Cartesian coordinates and enumerates no images either. So two more things are refused, and
    they are one fact wearing two faces:

    * a shape **larger than the period**, which runs into its own image. Its transform is a
      single sinc, which is not the transform of any periodic indicator. Measured on the
      boundary builder: a 1.2x1.4-period rod gives ``<1/eps>`` = 0.186 where the geometry is
      0.250, and a pair with one rod 1.4 periods tall that does not tile the cell, 0.618
      against 0.695.
    * two shapes that **meet only through a cell edge**. An overlap inside the cell resolves
      into disjoint polygons and is built; one that happens through an edge is clipped away,
      and what comes back does not converge — ``sum(T)`` of 0.309 / 0.632 / 0.913 at truncation
      order 5 / 9 / 13 where a wrapped raster holds at 0.997.

    Both would otherwise solve and conserve energy. The second was refused by name until
    2026-08-27, when ``combine`` was removed: the refusal was reached only by a layer that had
    not named its rule, and ``combine='paint'`` walked past it into the answer above. Removing
    the keyword made that branch the default, so the refusal moved here — where it does not
    depend on what the layer says.

    The sweep is ``image_only``: quadratic in the shapes, like the routing question next door,
    and asked of the images alone so that a pair overlapping both inside the cell and through
    an edge cannot be reported as the harmless one. It needs concrete numbers, so a layer whose
    parameters are tracers passes through unchecked; :meth:`freeze_geometry` is the eager call
    that still sees them.

    This was ``check_layer_combination`` until 2026-08-27, when it also refused an overlapping
    layer that had not said whether its shapes add or paint. Nothing is asked of the layer any
    more: ``combine`` went with the summing rule, and the geometry picks the builder.
    """
    shapes = layer.get('shapes', [])
    for i, shape in enumerate(shapes):
        check_shape_spec(shape, i, layer_index)
    if not shapes:
        return

    try:
        Px, Py = (checked_float(v) for v in periods_of(lattice))
        prims = _primitives(shapes, lattice)
    except Exception:                                  # noqa: BLE001 - traced parameters
        return
    if prims is None:
        return

    # Before the sweep, not after -- not because the message would differ (a negative
    # half-width makes the sweep find nothing, so the length is reported either way; moving
    # this loop below changes no test), but because the sweep cannot answer over one. It reads
    # half-widths as projection radii, so `overlapping_pairs` on a rod of `lx = -340` on a 300
    # period returns `[]`, the same as for a rod that fits. Asking first means it is never
    # asked a question it has no way to answer.
    for i, prim in enumerate(prims):
        for half_width, key in zip(prim[3:5], _length_keys_of(shapes[i])):
            if half_width > 0.0:
                continue
            raise ValueError(
                f'Analytic layer {layer_index}: shape {_shape_name(shapes, i)} has '
                f'{key} = {2.0 * half_width:g}. A length here is a size, and the transform '
                f'carries its sign: the shape\'s coefficient at G = 0 is the fraction of the '
                f'cell it claims, so a negative length *subtracts* the shape\'s contrast from '
                f'the background instead of adding it. A 120 x 150 rod of eps 4 on eps 1 comes '
                f'back as eps 0.4, and a long enough one as a negative permittivity -- which '
                f'solves, conserves energy, and is not a structure anything asked for. Zero is '
                f'refused with it: it is a shape that is not there, and on a round type it '
                f'reaches the solve as a division by zero.\n'
                f'  Give it a positive size. If an optimiser walked it through zero, bound the '
                f'parameter or optimise a quantity that cannot go negative -- there is no '
                f'cliff at zero to stop a descent on its own.')

    # `OVERLAP_TOL_REL * max(Px, Py)` is a tolerance *scale*, not geometry: it says how much
    # interpenetration counts, in units of the cell's own size. It keeps *touching* out --
    # a rod exactly filling the period comes out at a gap of zero and is a legal structure,
    # and so are two rods abutting across an edge.
    hits = _overlap.overlapping_pairs(prims, OVERLAP_TOL_REL * max(Px, Py), lattice,
                                      image_only=True)
    if not hits:
        return

    i, j, ox, oy, _ = hits[0]
    more = f' (and {len(hits) - 1} more such pair(s))' if len(hits) > 1 else ''
    if i == j:
        what = (f'shape {_shape_name(shapes, i)} is larger than the period and runs into its '
                f'own image{_across(ox, oy)}')
        fix = 'Shrink it below the period'
    else:
        what = (f'shapes {_shape_name(shapes, i)} and {_shape_name(shapes, j)} overlap '
                f'only{_across(ox, oy)}')
        fix = ('Move them so the overlap falls inside the cell -- an overlap that does is '
               'resolved into disjoint polygons and built exactly')
    raise ValueError(
        f'Analytic layer {layer_index}: {what}{more}. Neither shape builder here reads a '
        f"shape's periodic images: the closed form sums each shape's own transform, and the "
        f'boundary route clips its polygons in unwrapped coordinates. So the layer would be '
        f'built as a different structure -- one that solves, conserves energy, and does not '
        f'converge.\n'
        f'  {fix}, or state the layer as a raster structure, which carries the wrap.')


def layer_needs_resolving(layer, lattice):
    """Does this layer actually need the boundary route — i.e. is it painted *and* overlapping?

    While nothing intersects, painting and summing describe the same structure, and the closed
    form here is both exact and cheaper than resolving a polygon that has nothing to resolve.
    Routing every shape layer through the boundary route would also move the answer on a layer
    that has no overlap at all -- measured 0.8% apart on disjoint rods.

    Raises when the parameters are not concrete and no plan was frozen. That case used to
    return ``False`` on the reasoning that a traced call has already had its eager pass through
    :meth:`freeze_geometry` — but nothing enforced it, and the cost of the assumption being
    wrong is not a slower path or a warning. It is a **different structure**: the layer is
    summed instead of painted, silently. Measured under ``jax.grad`` on two overlapping rods,
    ``T`` came out 0.8743 where painting gives 0.8539, and the gradient was bit-identical to
    the one the summing builder returns on its own — which is what identified it.

    PyTorch never hit this: its tensors convert to floats, so the check ran. JAX tracers do
    not, so it silently did not.
    """
    shapes = layer.get('shapes', [])
    if len(shapes) < 2:
        return False
    try:
        Px, Py = (checked_float(v) for v in periods_of(lattice))
        prims = _primitives(shapes, lattice)
    except Exception:                                  # noqa: BLE001 - a traced basis
        prims = None
    if prims is None:
        raise ValueError(_NEEDS_A_FREEZE.format(
            what='whether the shapes overlap, which decides how the painting resolves into '
                 'disjoint polygons'))
    return bool(_overlap.overlapping_pairs(prims, OVERLAP_TOL_REL * max(Px, Py),
                                           lattice))


#: The message for the one question a trace cannot answer.
#:
#: It briefly guarded two. `can_plan_eagerly` stood beside it for one revision, on the theory
#: that "can I read the geometry?" and "can I cut the polygons?" had come apart -- and they had
#: not: unifying the read (`boundary._concrete_shape` now uses the same `checked_float`) made
#: planning work wherever reading does. Mutation-tested as dead before it was deleted: under
#: `jit` `layer_needs_resolving` raises here first and the guard was never reached at all
#: (0 calls), while under eager and `grad` it always answered yes.
_NEEDS_A_FREEZE = (
    "a traced shape layer needs a frozen geometry: {what} cannot be asked of a traced "
    "value. Guessing it would solve a different structure than the one asked for, without "
    "saying so.\n"
    "  Freeze once, outside the trace, and replay:\n"
    "    geometry = solver.freeze_geometry()\n"
    "    result = solver.solve(geometry=geometry)\n"
    "  Re-take the freeze whenever the shapes may have started or stopped overlapping; the "
    "plan compares by content, so an unchanged one is free.")


def shape_ft(xp, shape, ky, kx, lattice):
    """Continuous Fourier transform of one shape's indicator, divided by the cell area.

    Returns ``c(ky, kx) = 1/area * FT{1_shape}(Gx, Gy)`` on the integer reciprocal grid — the
    analytic analogue of one Fourier coefficient the ``fourier2d_*`` builders read off a raster.

    ``lattice`` is the cell basis (``None`` = the axis-aligned ``diag(period)``). It reaches
    only the two lines below, and that is the whole of what a skewed cell costs this function:
    ``(Gx, Gy)`` is the reciprocal vector ``G = m*b1 + n*b2`` in **Cartesian** components, and
    everything after it — the rotation, the phase, the sinc product, the Bessel forms — is
    already written in Cartesian frequency and so does not care how ``G`` was reached.

    Nothing about the shape deforms either, because shape lengths are absolute Cartesian
    quantities (see :func:`shape_geometry`), not fractions of the cell. A rectangle stays a
    rectangle under a skew; only the grid of frequencies its transform is *sampled at* stops
    being rectangular. That is why this path needed far less than the raster one, where the
    normal field genuinely is index-based and needs the metric.
    """
    stype, cx, cy, lx, ly, angle = shape_geometry(shape, lattice)

    pi = 3.141592653589793
    # The `lattice is None` arm that stood here computed the rectangle's answer from
    # `Px`, `Py`. The solver stopped reaching it once `_geometry_lattice_for` became
    # total (`cc016fc`): driving only the public API, the one function that ever sees a
    # `None` cell is `as_lattice`, where the conversion happens. Deleted rather than
    # re-keyed on `is_axis_aligned` -- production takes *this* branch on a rectangle
    # today, and the two differ in the last place.
    # (Gx, Gy) = 2*pi * A^-T @ (m, n), written out for the same reason as in
    # wavebit._core.lattice.order_kxy: one expression per component, no matrix inverse.
    a1x, a2x = lattice[0][0], lattice[0][1]
    a1y, a2y = lattice[1][0], lattice[1][1]
    det = a1x * a2y - a2x * a1y
    Gx = 2.0 * pi * (a2y * kx - a1y * ky) / det
    Gy = 2.0 * pi * (-a2x * kx + a1x * ky) / det
    area = abs(det)

    # Rotated frequency coordinates: kappa = Rot(angle)^T @ k
    ct = xp.cos(as_array(xp, angle, dtype=Gx.dtype))
    st = xp.sin(as_array(xp, angle, dtype=Gx.dtype))
    kxr = Gx * ct + Gy * st
    kyr = -Gx * st + Gy * ct

    phase = xp.exp(-1j * (Gx * cx + Gy * cy))

    if stype in ('rectangle', 'square'):
        s = lx * sinc(kxr * lx / 2.0) * ly * sinc(kyr * ly / 2.0)
    elif stype in ('circle', 'disk'):
        R = lx / 2.0
        # Same DC guard as the ellipse below, and for the same reason: ``rho`` is zero there,
        # ``d(sqrt)/d`` is infinite, and ``jinc``'s own ``where`` turns that into NaN. Here the
        # radicand depends on ``period``, so it is ``d(R)/d(period)`` that used to come back
        # NaN on a circle while the guarded ellipse was fine.
        r2 = Gx * Gx + Gy * Gy
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
        raise ValueError(f'Unknown parametric shape type: {stype!r}')

    return phase * s / area


def _refuse_overlap(shapes, lattice):
    """Raise if these shapes overlap -- the case the superposition below cannot represent."""
    if len(shapes) < 2:
        return
    try:
        Px, Py = (checked_float(v) for v in periods_of(lattice))
        prims = _primitives(shapes, lattice)
    except Exception:                                  # noqa: BLE001 - traced parameters
        return          # the concrete pass already answered this; see `layer_needs_resolving`
    if prims is None:
        return
    hits = _overlap.overlapping_pairs(prims, OVERLAP_TOL_REL * max(Px, Py), lattice)
    if not hits:
        return          # nothing to resolve: painting and adding are the same structure here

    # `_across` is not decoration: the sweep tests the eight periodic images too, so the pair
    # a user is told about can be two shapes that plainly miss each other inside the cell.
    # Without the phrase the message reads as a false positive and the layer looks unbuildable.
    i, j, ox, oy = hits[0][0], hits[0][1], hits[0][2], hits[0][3]
    where = _across(ox, oy)
    what = (f'shape {_shape_name(shapes, i)} runs into its own periodic image{where}'
            if i == j else
            f'shapes {_shape_name(shapes, i)} and {_shape_name(shapes, j)} overlap{where}')
    raise ValueError(
        f'an overlapping layer cannot be built here: {what}, and this builder sums every '
        f"shape's "
        f'(eps_shape - eps_base) * FT(indicator). Summing an overlap is a different structure '
        f'from painting it -- measured 1.99 against 1.90 in <eps> for two 300x300 rods of '
        f'permittivity 4 and 9 -- so returning the sum under the name of the paint would '
        f'be a plausible wrong answer rather than a refusal.\n'
        f'  For the painted layer, go through wavebit._core.boundary.to_conv_mat_boundary_layer '
        f'(or parametric.to_conv_mat_parametric_layers, which routes to it), where the overlap is '
        f'resolved into disjoint polygons before anything is transformed.')


def assemble_eps_coeffs(xp, shapes, background, half_width_u, half_width_v, lattice, dtype):
    """Analytic Fourier coefficients of ``eps`` and ``1/eps`` for one layer.

    Each grid has shape ``(4*half_width_v+1, 4*half_width_u+1)``, indexed ``[ky+2*half_width_v,
    kx+2*half_width_u]`` — the doubled range a ``(2*truncation_order+1)**2`` convolution matrix needs
    (``docs/factorization-terms.md`` §2 for why it doubles).

    ``background`` and each shape's ``material`` are read through
    :func:`~wavebit._core.domain.as_material`, so a bare number means ``Material(epsilon=x)``
    and the permittivity is taken straight out -- **there is no squaring here any more**, which
    is one of the sites the convention flip is made of. Wavebit works in ``e^{-iwt}``
    throughout (``Im(eps) > 0`` = loss), with no conjugation on the way in.

    ``mu`` is not read here and must not be: this function is called a second time by
    ``CoreFactorizationMixin._mu_conv_matrices``, on a layer whose materials have had their ``mu``
    put where their ``epsilon`` was, and that second pass is what builds the permeability
    slots. Reading both here would build one of them twice and the other never.

    **This sums, and an overlapping layer is not a sum.** Where two shapes overlap, summing and
    painting describe two different structures, so reconstructing a painted layer from this
    function is wrong -- silently: two 300x300 rods of permittivity 4 and 9 in a 1000x1000 cell
    come back at ``<eps> = 1.99`` where the solver integrates ``1.90``, a 4.7% error that looks
    entirely plausible. The solver does not make that mistake -- :func:`to_conv_mat_parametric_layers`
    routes an overlapping layer to :mod:`.boundary` and only reaches this builder where the
    shapes are disjoint, and where they are disjoint the two rules coincide exactly.

    So an overlap is refused here unconditionally. It was conditional on a ``combine=`` argument
    until 2026-08-27, when ``combine='add'`` was removed: a layer could then ask for the sum,
    and the caller had to say so for the check to run. There is no such ask now, so there is
    nothing for the caller to state and no way to turn the check off.
    """
    _refuse_overlap(shapes, lattice)

    eps_base = as_material(background, 'layer background').epsilon
    dev = getattr(eps_base, 'device', None) if hasattr(eps_base, 'device') else None
    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32
    KKY, KKX = _grid(xp, half_width_u, half_width_v, dev, real_dtype)

    eps_bg = as_array(xp, eps_base, dtype=dtype)
    inv_bg = 1.0 / eps_bg

    # DC term only: the background is constant over the whole cell.
    is_dc = (KKY == 0) & (KKX == 0)
    zero = xp.zeros(KKY.shape, dtype=dtype)
    eps_c = xp.where(is_dc, eps_bg, zero)
    inv_eps_c = xp.where(is_dc, inv_bg, zero)

    for shape in shapes:
        eps_s = as_array(xp, as_material(shape['material'], 'shape material').epsilon,
                         dtype=dtype)
        ft = xp.astype(shape_ft(xp, shape, KKY, KKX, lattice), dtype)
        eps_c = eps_c + (eps_s - eps_bg) * ft
        inv_eps_c = inv_eps_c + (1.0 / eps_s - inv_bg) * ft

    return eps_c, inv_eps_c


def conv_matrix_2d(xp, coeff2d, half_width_u, half_width_v):
    """Convolution matrix from a 2D coefficient grid.

    ``M[(r,s),(c,d)] = coeff(ky=r-c, kx=s-d)`` — the ordering the ``fourier2d_*`` builders
    produce. The
    sign is the order-index convention (see ``_circulant_index``); the two builders have to
    agree with each other and with ``kx``, which ``test_conv_matrix_matches_grid_convention``
    pins.

    **Rule-neutral, which is what the name is for.** This was ``laurent_conv_2d`` until
    2026-08-12, and ``Laurent`` is the *direct* factorization rule — a rule word on a builder
    that implements neither. :func:`to_conv_mat_parametric_nvf` calls it on ``eps`` and on
    ``1/eps`` two lines apart, and on the three normal-field products; the inverse rule's
    inversion happens in the caller.

    The result is **BTTB** (block-Toeplitz with Toeplitz blocks), not Toeplitz: with the row
    and column pairs flattened, ``i - j`` does not determine ``(r-c, s-d)``, so the diagonals
    are not constant. ``docs/factorization-terms.md`` has the check and the vocabulary.
    """
    n_orders_u = 2 * half_width_u + 1
    n_orders_v = 2 * half_width_v + 1
    dev = _device(coeff2d)

    iy = xp.asarray([[[[r - c + 2 * half_width_v]]] for r in range(n_orders_v) for c in range(n_orders_v)],
                    device=dev)
    iy = xp.reshape(iy, (n_orders_v, n_orders_v))
    ix = xp.asarray([[s - d + 2 * half_width_u for d in range(n_orders_u)] for s in range(n_orders_u)], device=dev)

    m4 = coeff2d[iy[:, None, :, None], ix[None, :, None, :]]      # [r, s, c, d]
    return xp.reshape(m4, (n_orders_v * n_orders_u, n_orders_v * n_orders_u))


def _device(x):
    return x.device if hasattr(x, 'device') else None


# FROZEN 2026-08-17 from the sweep in the blend below. The exponent has a *plateau*, not a
# peak, which is what makes it a measured constant rather than the tuning the mollifier route
# was rejected for: p = 3..8 all give 0.000 deg at every surface and hold the two-circle
# position gradient under the 1e-3 the suite asks (2.5e-4 at p = 4, its best). Below it the
# blend is too soft -- p = 1 leaves 0.59 deg at the surface band and misses the gradient at
# 3.3e-3; above it the blend approaches a hard nearest-shape pick and the jump it exists to
# avoid comes back, 1.0e-3 by p = 12. Four is the interior of that window.
#
# A third axis was measured afterwards and does not move the choice: the sampling error of the
# projector coefficients, |f(N) - f(2N)|, falls monotonically as p RISES (1.9e-3, 1.5e-3,
# 1.3e-3, 1.1e-3, 7.3e-4 at p = 2, 3, 4, 6, 8 on two circles at N = 256). The reasoning that
# predicted the opposite is worth recording as wrong: a sharper crossover looks like it should
# integrate worse and integrates better, because a high p leaves the field equal to one shape's
# own smooth field over most of the cell and mixed only in a narrow band, while a low p mixes
# everywhere. Trading p = 4 for p = 8 buys 1.3x there and gives back 1.7x on the position
# gradient (2.5e-4 -> 4.3e-4), so the middle stays.
_BLEND_EXPONENT = 4


def normal_vector_field_analytical(xp, grid_shape, shapes, dtype, lattice):
    """Unit normal field of the shapes, sampled on ``grid_shape`` — analytic, not from a raster.

    The sample points come from :func:`wavebit._core.lattice.sample_grid`, which is where the
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
    X, Y = sample_grid(xp, grid_shape, lattice, dtype)

    if _tiles_pay(X, Ny, Nx, len(shapes)):
        return _blend_tiled(xp, X, Y, shapes, dtype, (Ny, Nx), lattice)

    got = [g for g in (_shape_surface_normal(xp, X, Y, s, dtype, lattice) for s in shapes)
           if g is not None]
    if not got:
        return xp.ones((Ny, Nx), dtype=dtype), xp.zeros((Ny, Nx), dtype=dtype)
    if len(got) == 1:
        return got[0][1], got[0][2]

    # Weight each shape by the product of the *other* shapes' surface distances. At shape A's
    # own surface d_A = 0, so every other weight carries that zero and the sum collapses to
    # A's normal exactly, while between the shapes the weights vary smoothly. It is `1/d_j`
    # written without a division, so there is no epsilon and nothing to tune. See :func:`_blend`
    # for the body, which the tiled traversal above shares.
    #
    # Smoothness is the point, not a bonus. Picking the nearest shape outright (`argmin`) gives
    # the same exact normals at every surface and a field that JUMPS across the selection
    # boundary -- and moving a shape sweeps that boundary, which contributes a line integral of
    # the jump to the true derivative that no grid sample and no autograd sees. Measured on two
    # circles, the hard pick's autograd sits 44-76% from central differences and does not
    # improve as h shrinks (until h drops below one pixel, where the finite difference stops
    # seeing the boundary move too and the two agree on being wrong together). This weighting
    # has no jump to sweep: the same comparison falls 1.5e-2 -> 6.5e-4 -> 6.3e-5 as h goes
    # 1e-4 -> 1e-5 -> 1e-6, which is a correct gradient with the difference converging onto it.
    return _blend(xp, got, (Ny, Nx), dtype)


#: Side of the square block :func:`_blend_tiled` works in. 64 doubles is 32 KB per array, so
#: the ten-odd live arrays of one shape's evaluation stay inside L2 and the whole pipeline runs
#: without touching memory again. Measured against 128 / 192 / 256 on two and four shapes at
#: N = 512 / 1024 / 2048: 64 wins or ties everywhere, by up to 1.3x over the next size.
_TILE = 64

#: Below this many pixels the full-grid arrays already fit in cache and tiling only adds Python
#: loop overhead — measured 1.00x at N = 256 and 1.09-1.24x at N = 512.
_TILE_MIN_PIXELS = 512 * 512


def _tiles_pay(sample, n_v, n_u, n_shapes):
    """Whether to evaluate the blend block by block instead of over the whole grid.

    **NumPy only, and that is the point rather than a limitation.** The win here is cache
    residency, which is a property of eager elementwise evaluation: each of the ~50 array
    operations per shape reads and writes the whole grid, so at N = 1024 a two-shape layer
    moves roughly 800 MB through memory to do a few tens of megaflops. JAX and Torch under a
    trace do not have that problem — XLA and the compiler fuse the same chain into one pass —
    and a Python loop over 256 tiles would instead unroll into the graph and wreck compile
    time. So the tiling is taken only where it helps and skipped where it would hurt.

    Decided on the **array**, not on the namespace's name. The obvious spelling,
    ``xp.__name__.endswith('numpy')``, is wrong and quietly so: ``jax.numpy`` ends with
    ``numpy`` too, and it would have turned the tiling on for exactly the backend the paragraph
    above says to keep it off. ``test_only_numpy_takes_the_tiled_path`` is what caught that.
    """
    import numpy as np
    return (n_shapes > 1 and n_v * n_u >= _TILE_MIN_PIXELS
            and type(sample) is np.ndarray)


def _blend_tiled(xp, X, Y, shapes, dtype, grid_shape, lattice):
    """:func:`normal_vector_field_analytical`'s blend, evaluated one cache-sized block at a time.

    Bit-identical to the whole-grid path by construction: same operations on the same values in
    the same order, only the traversal differs. Pinned by
    ``test_the_tiled_blend_is_bit_identical_to_the_whole_grid_one``.
    """
    import numpy as np
    n_v, n_u = grid_shape
    out_x = np.empty(grid_shape, dtype=dtype)
    out_y = np.empty(grid_shape, dtype=dtype)
    for i0 in range(0, n_v, _TILE):
        for j0 in range(0, n_u, _TILE):
            block = (slice(i0, min(i0 + _TILE, n_v)), slice(j0, min(j0 + _TILE, n_u)))
            bx = np.ascontiguousarray(X[block])
            by = np.ascontiguousarray(Y[block])
            got = [g for g in (_shape_surface_normal(xp, bx, by, s, dtype, lattice)
                               for s in shapes) if g is not None]
            if not got:
                out_x[block], out_y[block] = 1.0, 0.0
                continue
            if len(got) == 1:
                out_x[block], out_y[block] = got[0][1], got[0][2]
                continue
            out_x[block], out_y[block] = _blend(xp, got, bx.shape, dtype)
    return out_x, out_y


def _blend(xp, got, shape, dtype):
    """The distance-product combination of several shapes' fields — see the caller for why."""
    dists = [g[0] for g in got]
    scale = sum(dists) / len(dists)
    scale = xp.where(scale > 0, scale, xp.ones_like(scale))
    dn = [(d / scale) ** _BLEND_EXPONENT for d in dists]

    sx = xp.zeros(shape, dtype=dtype)
    sy = xp.zeros(shape, dtype=dtype)
    for j, (_, gx, gy) in enumerate(got):
        w = None
        for k, dk in enumerate(dn):
            if k != j:
                w = dk if w is None else w * dk
        sx = sx + w * gx
        sy = sy + w * gy
    norm = _hypot_safe(xp, sx, sy)
    return sx / norm, sy / norm


def _fold_to_nearest_image(xp, dx, dy, lattice):
    """``(dx, dy)`` reduced to the nearest periodic image, on the cell's own lattice.

    ``d % P - P/2`` answers "nearest image" only when the images sit at ``(m*Px, n*Py)``. They
    sit at ``m*a1 + n*a2``, so on a skewed cell the rectangular fold reduces towards a point
    where no image is: on a 60-degree cell it looks straight up while the neighbour is up and
    to the right. The field is then wrong wherever a neighbour's surface is the nearest one,
    which is why it only shows once a shape fills enough of the cell.

    Rounding the *fractional* coordinates lands in the fundamental parallelogram — the nearest
    image in the lattice metric, not the Euclidean one. The 3x3 sweep after it is what makes
    the pick Euclidean. One cell a side is enough for any basis this solver accepts.
    """
    a1x, a2x = lattice[0][0], lattice[0][1]
    a1y, a2y = lattice[1][0], lattice[1][1]
    det = a1x * a2y - a2x * a1y
    su = xp.round((a2y * dx - a2x * dy) / det)
    sv = xp.round((-a1y * dx + a1x * dy) / det)

    best_x = best_y = best_d2 = None
    for du in (-1.0, 0.0, 1.0):
        for dv in (-1.0, 0.0, 1.0):
            mu, mv = su + du, sv + dv
            ex = dx - mu * a1x - mv * a2x
            ey = dy - mu * a1y - mv * a2y
            d2 = ex * ex + ey * ey
            if best_d2 is None:
                best_x, best_y, best_d2 = ex, ey, d2
            else:
                take = d2 < best_d2
                best_x = xp.where(take, ex, best_x)
                best_y = xp.where(take, ey, best_y)
                best_d2 = xp.where(take, d2, best_d2)
    return best_x, best_y


def _shape_surface_normal(xp, X, Y, shape, dtype, lattice):
    """``(distance to this shape's surface, normal_x, normal_y)`` — the outward unit normal, periodic.

    Two things here differ from what this file did before 2026-08-17, and both are what make
    :func:`normal_vector_field_analytical` a nearest-surface field rather than a sum.

    **The displacement is folded to the nearest image.** ``X - cx`` is the offset to the shape
    as it was named, and on a torus the shape is equally at ``cx +- Px``. Folding is what makes
    one shape's field periodic, and a nearest-surface pick over periodic fields is periodic in
    turn — which is the whole reason ``cx`` and ``cx + Px`` now give the same answer for
    several shapes and not only for one.

    **Outside a rectangle's corner the normal is radial from the corner, not the nearest
    face.** The old rule extended each face's normal into an infinite wedge, which is not the
    gradient of the distance to a box: past a corner the nearest *point* is the corner itself.
    Measured against that exact gradient the wedge rule is 24.9 deg out over the corner
    quadrants and this is 0.000; against the independent raster field (``'contour'``, which is
    distance-based and periodic) 27.3 deg becomes 2.4.
    """
    stype, cx, cy, lx, ly, angle = shape_geometry(shape, lattice)
    cos_a = xp.cos(as_array(xp, angle, dtype=dtype))
    sin_a = xp.sin(as_array(xp, angle, dtype=dtype))

    # `diag(Lx, Ly)` is the one case where the two folds ARE the same fold, so it keeps the
    # cheaper spelling -- and stays bit-identical to every result predating this branch.
    if is_axis_aligned(lattice):
        Px, Py = periods_of(lattice)
        dx = (X - cx + Px / 2.0) % Px - Px / 2.0
        dy = (Y - cy + Py / 2.0) % Py - Py / 2.0
    else:
        dx, dy = _fold_to_nearest_image(xp, X - cx, Y - cy, lattice)
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    a, b = lx / 2.0, ly / 2.0

    if stype in _overlap._ELLIPSE_TYPES:                 # ellipse / circle / disk
        gxr = 2.0 * xr / (a ** 2)
        gyr = 2.0 * yr / (b ** 2)
        # The algebraic distance |f| / |grad f| -- exact for a circle, and for an ellipse the
        # first-order distance to the level set, which is all the nearest-surface pick needs.
        f = (xr / a) ** 2 + (yr / b) ** 2 - 1.0
        dist = xp.abs(f) / _hypot_safe(xp, gxr, gyr)
    elif stype in _overlap._BOX_TYPES:                   # rectangle / square
        qx, qy = xp.abs(xr) - a, xp.abs(yr) - b
        # `xp.maximum` against a Python float is a numpy convenience torch does not share, so
        # every clamp here goes through a zero array of the same dtype.
        zero = xp.zeros_like(xr)
        out_x, out_y = xp.maximum(qx, zero), xp.maximum(qy, zero)
        outside = _hypot_safe(xp, out_x, out_y)
        dist = xp.where((qx > 0) | (qy > 0), outside,
                        -xp.minimum(xp.maximum(qx, qy), zero))

        # Inside, blend the FOUR faces by the same distance-product rule, each with its own
        # outward normal and its own distance. This is what removes `sign()` from the branch
        # -- and with it the defect the module docstring called unfixable without a mollifier.
        #
        # The old rule picked the nearest face with `where(qx >= qy, sign(xr), ...)`, and both
        # halves of that jump. `sign(xr)` flips across the shape's own centre line and the
        # `>=` flips across the diagonals, so the field was piecewise constant and every bit of
        # its position dependence sat on those seams: translating the shape slid them, which a
        # grid sample cannot see. Measured, that is not a resolution problem -- the error is
        # flat at 1.8e-2 from normal_grid 1024 to 4096, while the circle pair converges.
        #
        # Treating +x and -x as separate faces makes the seam vanish rather than smoothing it.
        # At the centre line the two carry equal weight and opposite normals, so they cancel
        # exactly and continuously; at a face its own distance goes to zero, every other weight
        # carries that zero, and the sum collapses to that face's normal outright. Two circles
        # were already differentiable and stay so; the rectangle pair goes from 212-274% out
        # against central differences to 0.06 / 0.03 / 1.0%.
        faces = ((a - xr, 1.0, 0.0), (a + xr, -1.0, 0.0),
                 (b - yr, 0.0, 1.0), (b + yr, 0.0, -1.0))
        fd = [xp.maximum(f[0], zero) for f in faces]
        fscale = sum(fd) / len(fd)
        fscale = xp.where(fscale > 0, fscale, xp.ones_like(fscale))
        fdn = [(d / fscale) ** _BLEND_EXPONENT for d in fd]
        in_x = xp.zeros_like(xr)
        in_y = xp.zeros_like(xr)
        for j, (_, ux, uy) in enumerate(faces):
            w = None
            for k, dk in enumerate(fdn):
                if k != j:
                    w = dk if w is None else w * dk
            in_x = in_x + w * ux
            in_y = in_y + w * uy

        # Outside, `max(q, 0)` already points at the nearest surface point -- one face when a
        # single axis is out, the corner itself when both are. `sign` is safe here: it is only
        # read where |xr| > a, far from the flip.
        is_out = (qx > 0) | (qy > 0)
        gxr = xp.where(is_out, xp.sign(xr) * out_x, in_x)
        gyr = xp.where(is_out, xp.sign(yr) * out_y, in_y)
    else:
        # A type this does not know. `normal_vector_field_analytical` drops the None and, if
        # every shape drops out, falls back to a constant `(1, 0)` -- Li's rules with x as the
        # normal everywhere. That fallback is right for a type nothing here can transform
        # either, and was silently wrong for two that it can: until 2026-08-27 the branches
        # above read the *canonical* spellings only, so `type='disk'` and `type='square'` --
        # both accepted by `check_shape_spec`, both transformed correctly by `shape_ft` -- came
        # back with no normal field at all. Same geometry, one word apart, `sum(T)` 1.6e-3
        # adrift at truncation_order 6 and still 7.0e-4 at 14, with R+T exactly 1. The branches
        # read `overlap`'s alias tuples now, which is the one place the spellings are listed.
        return None

    gx = gxr * cos_a - gyr * sin_a
    gy = gxr * sin_a + gyr * cos_a
    norm = _hypot_safe(xp, gx, gy)
    return dist, gx / norm, gy / norm


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


def _sampled_coeffs(xp, field, half_width_u, half_width_v):
    """Fourier coefficients of a sampled field on the doubled reciprocal grid.

    ``c(ky, kx) = 1/(Ny*Nx) * sum f[iy, ix] exp(-2i*pi*(ky*iy/Ny + kx*ix/Nx))``, laid out the
    way :func:`assemble_eps_coeffs` lays its grids out. Feeding it to :func:`conv_matrix_2d`
    reproduces ``fourier2d_pointwise(field, ...)`` — that equivalence is what
    ``test_conv_matrix_matches_grid_convention`` states, and it is what lets a translation be
    applied here as a phase instead of by moving the samples.
    """
    Ny, Nx = field.shape
    dev = _device(field)
    F = xp.fft.fft2(field) / (Ny * Nx)
    iy = xp.asarray([k % Ny for k in range(-2 * half_width_v, 2 * half_width_v + 1)], device=dev)
    ix = xp.asarray([k % Nx for k in range(-2 * half_width_u, 2 * half_width_u + 1)], device=dev)
    coeff = F[iy[:, None], ix[None, :]]
    # The cell-centre half-pixel shift, the same one
    # :func:`~wavebit._core.factorization._half_pixel_phase` applies on the raster side. The
    # samples come from `sample_grid`, which places index `i` at `(i + 0.5)/N` -- so a bare DFT,
    # which places it at `i/N`, describes a field shifted half a pixel against the permittivity
    # these coefficients are multiplied by. Applied here as well as there because these two are
    # the *same transform written twice*, and `test_conv_matrix_matches_grid_convention` is what
    # says so; correcting one alone silently splits the parametric path from the raster one.
    ky = xp.asarray([float(k) for k in range(-2 * half_width_v, 2 * half_width_v + 1)], device=dev)
    kx = xp.asarray([float(k) for k in range(-2 * half_width_u, 2 * half_width_u + 1)], device=dev)
    pi = 3.141592653589793
    phase = (xp.exp(-1j * pi * xp.astype(ky, coeff.dtype) / Ny)[:, None]
             * xp.exp(-1j * pi * xp.astype(kx, coeff.dtype) / Nx)[None, :])
    return coeff * phase


_PROJECTOR_CACHE = {}
_PROJECTOR_CACHE_MAX = 8


def _static_float(v):
    """``v`` as a plain float, refusing anything a cache must not key on.

    Two refusals, and they are different. A JAX tracer cannot be converted at all, so ``float``
    raises on its own. A PyTorch tensor *can* be converted -- which is exactly the trap
    :func:`layer_needs_resolving` documents -- so one that carries ``requires_grad`` has to be
    refused explicitly, or a sweep over it would be served a result built at the first value.
    """
    if getattr(v, 'requires_grad', False):
        raise TypeError('a value under torch autograd is not a cache key')
    detach = getattr(v, 'detach', None)
    if detach is not None:
        v = detach()
    return float(v)


def _projector_cache_key(xp, built, grid_shape, lattice, half_width_u, half_width_v, dtype):
    """A content key for the coefficients built below, or ``None`` when caching would be wrong.

    Everything :func:`normal_vector_field_analytical` reads goes in and nothing else does. It
    reads each shape through :func:`shape_geometry` and never touches ``material`` or
    ``background``, which is what makes the cache worth having: a sweep over the *material*
    leaves this key alone. ``None`` means at least one value is traced or under autograd, so the
    build has to happen again -- the gradient runs through it.
    """
    try:
        shapes_key = tuple(tuple(v if isinstance(v, str) else _static_float(v)
                                 for v in shape_geometry(shape, lattice))
                           for shape in built)
        lat = tuple(_static_float(v) for row in lattice for v in row)
    except (TypeError, ValueError):
        return None

    # **A lone round shape's field is scale-free, so its size does not belong in the key.**
    # The level-set gradient of `x^2/a^2 + y^2/b^2` is `(2x/a^2, 2y/b^2)`, whose *direction*
    # is unchanged when `a` and `b` scale together -- so the projector coefficients depend on
    # the aspect and not on the size. Measured on this function: a circle at lx = 40 / 90 /
    # 220 / 280 agrees with lx = 140 to 2.8e-17, and an ellipse rescaled 2x and 0.5x at a fixed
    # aspect agrees to 0.000e+00, while changing the aspect moves it 1.2e-01.
    #
    # It matters because the commonest 1D grating sweep is over a *radius*, and with `lx` in
    # the key every step missed and rebuilt a bit-identical field: five steps, five entries,
    # where a position or material sweep leaves one.
    #
    # Only for one shape, and only for the round family. The multi-shape blend weighs absolute
    # distances between shapes, and a rectangle's four-face blend is absolute too -- measured
    # 8.4e-02 and 5.4e-02 for the same two rescalings, which is why it keeps its lengths.
    if len(shapes_key) == 1 and shapes_key[0][0] in _overlap._ELLIPSE_TYPES:
        stype, cx, cy, lx, ly, angle = shapes_key[0]
        if lx:
            shapes_key = ((stype, cx, cy, 1.0, ly / lx, angle),)
    # The namespace is redundant with ``str(dtype)``, which already spells it out
    # (``numpy.complex128`` against ``torch.complex128``) -- but that is an accident of how the
    # dtypes are named, not a guarantee, and a cross-backend collision hands one namespace's
    # array to another.
    return (getattr(xp, '__name__', ''), shapes_key, tuple(grid_shape),
            int(half_width_u), int(half_width_v), str(dtype), lat)


def normal_projector_coeffs(xp, shapes, grid_shape, lattice, half_width_u, half_width_v,
                            dtype):
    """Coefficient grids of ``normal_x**2``, ``normal_x*normal_y`` and ``normal_y**2`` — the NVF projectors' input.

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
    the shape sits. That is not a discretization: ``<normal_x**2>`` for a circle moves 2.4e-2 between
    0.30 and 0.50 of the period and does not budge when the grid goes from 256 to 4096. It
    reached ``sum(R)`` as 1.6e-3, so the same structure translated gave a different answer.
    Built here it reproduces the centred one to 1e-14, at every ``truncation_order``.

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
      It manufactured the degeneracy it needs to avoid: two shapes then sit exactly opposed
      about the centre of the cell, their unit fields cancel exactly along the segment between
      them, and for two identical circles 16 grid points reach ``|n_total| == 0`` — where
      :func:`_hypot_safe` is finite but its gradient is ``0 * inf``. The common-translation
      gradient came back NaN.

      **That rejection is dated and the route is worth re-measuring before it is dismissed
      again.** It was decided against the *sum of unit vectors*, where ``n1 + n2 = 0`` along the
      whole segment between two opposed centres — a curve the grid samples. The distance-product
      blend that replaced it (2026-08-17) does not cancel there: off the midpoint ``w1 != w2``,
      so the zero set collapses to the single midpoint, which the ``(i+0.5)/N`` grid never
      contains. Measured on two identical circles on the cell diagonal, zero grid points reach
      ``|n| == 0`` at N = 256, 512 and 1024, against 85, 144 and 303 under the old rule. What
      has not been re-measured is whether the route is *good*: it makes the common translation
      exact and appears to cost 1.5-3.4x on relative jitter. It also moved the relative-position
      error for rectangles from 1.6e-1 to 2.9e-1.
    * *Mollifying* ``sign()`` (``tanh(xr/w)``) removes the zero derivative at the source and
      would serve both halves, but the width that makes the position gradient accurate is not
      the width that keeps the answer smooth through ``angle = 0``: it broke
      ``test_analytic_angle_is_smooth_through_normal``, and any value chosen between the two is
      a tuning constant.

    A fourth is *replacing the field wholesale* for multi-shape layers with the boundary route's
    gradient-of-distance field
    (:func:`~wavebit._core.boundary.distance_gradient_normal_field`), which a disjoint layer
    can be given today by passing ``boundary_plans`` to :func:`to_conv_mat_parametric_layers`.
    Tried, and rejected on the half it was reached for. Integrate its autodiff position
    gradient across a six-pixel window on two tilted rectangles and it comes to -3.5e-4 over a
    window the loss *rises* through by +1.4e-4 -- the wrong sign, at all 25 sampled positions,
    at truncation_order 8 and again at truncation_order 12; the sampled field integrates to +1.4e-4 against
    +1.1e-4, high by 22%. A circle pair reverses it rather than widening it: there the sampled
    field accounts for the loss to 1% and the replacement is 19% out. The cost is a pixel scan
    over every segment of every loop, and ``POLYGON_VERTICES`` is 256, so a circle carries 64x
    the segments a rectangle does: at the ``normal_grid`` of 1024 this solver defaults to, the
    swap costs 12x on a rectangle pair and 486x on a circle pair, 0.4 s and 23 s per layer per
    build. Its *forward* accuracy is open rather than settled, and the two are not separable at
    reachable truncations: they still sit 2e-4 apart on a total at truncation_order 24, so neither has
    reached the limit any ratio between them would have to be measured against. Reproduced by
    ``studies/benchmark_boundary_normal_field.py``; see ``docs/boundary-modeling.md``.

    Multi-shape layers therefore keep the sampled field. Both halves of what that costs are
    pinned in ``test_shape_position_gradients.py``, the working one as a test and the broken one
    as a strict xfail, so neither can be inferred from the other. Both of those run at
    ``truncation_order = 4``, where the model's own slope is about twice its truncation_order 12 one, so they
    hold autodiff to finite differences rather than to the slope the structure has.

    **The forward answer is affected too, not only the gradient — keep every shape inside the
    cell.** A shape straddling the boundary may be named in any period; ``cx`` and ``cx + Px``
    are one structure. A single shape agrees exactly across periods because the phase carries
    it. Several shapes do not: for a rod at ``cx`` = 0.10 / 1.10 / 2.10 beside a fixed second
    rod, total transmission comes out 0.8532 / 0.8678 / 0.8665. Splitting the layer's inputs
    puts it here rather than in the transform — across one period the permittivity coefficients
    agree to 1.1e-15 while ``<normal_x**2>`` differs by 3.1e-1. The boundary route inherits it
    identically, bit for bit, since it takes its field from the same place. An optimizer left
    free to move a shape will walk it over the edge and climb the artefact, so clamp the
    centres. Pinned by ``test_a_multishape_layer_is_not_periodic_across_the_cell_edge``.

    ``grid_shape`` is checked against the ``4*truncation_order+1`` floor
    (:func:`~wavebit._core.factorization.check_normal_grid`). The boundary route has done that
    since it was written and this one did not, which was not a decision: at
    ``truncation_order=10`` the two disagreed about ``normal_grid=8``, one refusing and the
    other building it 4.9e-4 adrift with energy conserved to twelve digits. The floor is a
    property of reading a field off a grid, not of which builder put it there.
    """
    check_normal_grid(grid_shape, half_width_u, half_width_v)
    real_dtype = xp.float64 if dtype == xp.complex128 else xp.float32

    if len(shapes) == 1:
        _, cx, cy, _, _, _ = shape_geometry(shapes[0], lattice)
        # The `lattice is None` arm that stood here read `Px / 2` as the centre and
        # `1 / Px` as the fraction. The solver stopped reaching it once
        # `_geometry_lattice_for` became total (`cc016fc`), and it differs from this one in
        # the last place, so it is deleted rather than re-keyed on `is_axis_aligned`.
        a1x, a2x = lattice[0][0], lattice[0][1]
        a1y, a2y = lattice[1][0], lattice[1][1]
        mid_x, mid_y = 0.5 * (a1x + a2x), 0.5 * (a1y + a2y)
        # Same quantity, stated properly: the displacement in *fractional lattice*
        # coordinates, A^-1 @ (c - mid). The phase is exp(-2i*pi*(m*u_d + n*v_d)), which is
        # exp(-i G.d) already -- it was lattice-general all along, it was only ever handed
        # its argument as a fraction of Px / Py.
        det = a1x * a2y - a2x * a1y
        dx, dy = cx - mid_x, cy - mid_y
        shift_u = (a2y * dx - a2x * dy) / det
        shift_v = (-a1y * dx + a1x * dy) / det
        # `0.0`, not `mid`: a centre is measured from `0.5*(a1 + a2)` since 2026-08-31, so the
        # centre of the cell is what an explicit zero names. Writing `mid` here would add the
        # origin twice and build the field for a shape one cell-centre away from where the
        # phase then puts it.
        built = [dict(shapes[0], cx=0.0, cy=0.0)]
    else:
        built, shift_v, shift_u = shapes, None, None

    # The build above is the expensive half and the phase below is the cheap one, so what is
    # cached is the split: for a single shape ``built`` sits at the centre of the cell whatever
    # ``(cx, cy)`` are, which makes a position sweep -- and a position *gradient*, which runs
    # through the phase -- reuse one field instead of rebuilding it every step. A material sweep
    # reuses it for the other reason: the key holds no material at all because the build reads
    # none. Measured on a five-point sweep at ``normal_grid = 1024``, ``truncation_order = 4``: the call
    # 83% of the wall clock for one shape and 93% for six.
    key = _projector_cache_key(xp, built, grid_shape, lattice, half_width_u, half_width_v, dtype)
    coeffs = _PROJECTOR_CACHE.pop(key, None) if key is not None else None
    if coeffs is None:
        normal_x, normal_y = normal_vector_field_analytical(xp, grid_shape, built,
                                                            real_dtype, lattice)
        coeffs = tuple(_sampled_coeffs(xp, xp.astype(f, dtype), half_width_u, half_width_v)
                       for f in (normal_x * normal_x, normal_x * normal_y, normal_y * normal_y))
    if key is not None:
        # Least-recently-used: the read above is a `pop`, so a hit takes the entry out and this
        # puts it back at the end, and a full cache evicts only on a genuine miss. The case that
        # needs it is a stack holding both a moving layer and a still one -- the moving layer
        # misses every step, and without the refresh the still one ages out from under it.
        # Measured on a two-layer stack over 30 steps: the still layer is built once and the
        # cache stays pinned at its bound, against 60 builds with no cache at all.
        if len(_PROJECTOR_CACHE) >= _PROJECTOR_CACHE_MAX:
            _PROJECTOR_CACHE.pop(next(iter(_PROJECTOR_CACHE)))
        _PROJECTOR_CACHE[key] = coeffs

    if shift_v is None:
        phase = 1.0
    else:
        KKY, KKX = _grid(xp, half_width_u, half_width_v, _device(coeffs[0]), real_dtype)
        arg = KKY * shift_v + KKX * shift_u
        phase = xp.exp(xp.astype(arg, dtype) * (-2j * 3.141592653589793))

    return tuple(c * phase for c in coeffs)


def to_conv_mat_parametric_nvf(xp, shapes, background, half_width_u, half_width_v, lattice, dtype,
                           normal_grid=256):
    """The five 2D-solve convolution matrices for one parametric layer, via NVF.

    ``eps`` / ``1/eps`` come from the exact shape transform (no staircase); the normal
    projectors come from the shapes' analytic normal field. Same Hermitian symmetrization as
    the raster NVF path — see :func:`wavebit._core.factorization.nvf_conv_matrices`.

    Every one of the five is ``D X D^-1`` for the shape's translation phase ``D`` and a
    position-free ``X`` (see :func:`normal_projector_coeffs`), so sliding a single-shape layer
    conjugates the whole solve by a diagonal and leaves each order's efficiency alone.

    The coefficients come from :func:`assemble_eps_coeffs`, which sums -- see there for why
    a layer whose shapes overlap must not arrive at this function.
    """
    eps_c, inv_eps_c = assemble_eps_coeffs(xp, shapes, background, half_width_u,
                                           half_width_v, lattice, dtype)
    c_xx, c_xy, c_yy = normal_projector_coeffs(
        xp, shapes, (normal_grid, normal_grid), lattice, half_width_u, half_width_v, dtype)
    return nvf_from_coeffs(xp, eps_c, inv_eps_c, c_xx, c_xy, c_yy, half_width_u, half_width_v)


def nvf_from_coeffs(xp, eps_c, inv_eps_c, c_xx, c_xy, c_yy, half_width_u, half_width_v):
    """Assemble the five convolution matrices from coefficient grids — the NVF tensor itself.

    The last step of every non-raster NVF path, and the same step for all of them: what differs
    between the parametric, boundary and level-set routes is *how* the five coefficient grids were
    obtained, never what is done with them. Kept in one place so a change to the tensor cannot
    reach one path and miss another — the raster path's equivalent is
    :func:`wavebit._core.factorization.nvf_conv_matrices`.

    ``eps_c`` / ``inv_eps_c`` are the ``eps`` and ``1/eps`` coefficients on the doubled
    reciprocal grid; ``c_xx`` / ``c_xy`` / ``c_yy`` those of ``normal_x**2``, ``normal_x*normal_y``, ``normal_y**2``.
    Returns ``(eps_xx, eps_yy, epz_i, eps_xy, eps_yx)`` — ``solve_conv``'s parameter order,
    as every builder does; see "Slot order" in :mod:`wavebit._core.factorization`.
    """
    ep_dir = conv_matrix_2d(xp, eps_c, half_width_u, half_width_v)
    ep_inv = conv_matrix_2d(xp, inv_eps_c, half_width_u, half_width_v)

    delta_ep = ep_dir - xp.linalg.inv(ep_inv)

    C_xx = conv_matrix_2d(xp, c_xx, half_width_u, half_width_v)
    C_xy = conv_matrix_2d(xp, c_xy, half_width_u, half_width_v)
    C_yy = conv_matrix_2d(xp, c_yy, half_width_u, half_width_v)

    eps_xx = ep_dir - 0.5 * (delta_ep @ C_xx + C_xx @ delta_ep)
    eps_yy = ep_dir - 0.5 * (delta_ep @ C_yy + C_yy @ delta_ep)
    eps_xy = -0.5 * (delta_ep @ C_xy + C_xy @ delta_ep)          # == eps_yx (symmetric)

    return ConvMatrices(eps_xx, eps_yy, xp.linalg.inv(ep_dir), -eps_xy, -eps_xy)


def check_no_plan_still_fits(layer, lattice, index):
    """Refuse a frozen *"this layer does not overlap"* that the layer has since outgrown.

    A frozen record answers the overlap question for every layer, and a ``None`` plan is one of
    its two answers -- "looked at while the numbers were concrete, and nothing overlapped" --
    rather than an absence. That distinction is what
    ``test_a_frozen_geometry_answers_the_overlap_question_for_every_layer`` settled; this is its
    other half, because an answer that is recorded can also stop being true. Move a shape until
    it does overlap and the answer is exactly as stale as a plan whose loops have been re-cut.

    It used to go stale in a worse place. The dispatch below summed on the record's word, and
    :func:`_refuse_overlap` stopped it there with a ``ValueError`` about *this* builder --
    naming neither the frozen record nor the fix, and pointing at the boundary route the caller
    was already trying to reach. Reproduced by freezing a disjoint rectangle-and-circle layer and
    sliding the rectangle onto the circle; the same call without ``geometry=`` returns
    0.933226529 quite happily.

    Eagerly the question can simply be re-asked, which is the same test the unfrozen call makes
    and costs one primitive overlap check. Under a trace there is nothing to re-ask -- the limit
    every other check on this record has.
    """
    try:
        needs = layer_needs_resolving(layer, lattice)
    except ValueError:
        return                      # traced: no concrete values to re-ask with
    if not needs:
        return
    raise StaleGeometryError(
        f'the frozen geometry no longer fits this structure: it recorded layer {index} as '
        f'not overlapping, and its shapes overlap now. That record is what routes '
        f'the layer, so replaying it would sum a layer that has to be painted -- a different '
        f'structure. Call freeze_geometry() again, or drop geometry= to route by comparison. '
        f'(Moving a shape never trips this on its own; moving it into another shape does.)')


def to_conv_mat_parametric_layers(layer_shapes, half_width_u, half_width_v, lattice,
                              normal_grid=256, complex_dtype=None, xp=None,
                              boundary_plans=None):
    """Stack the parametric-NVF convolution matrices over layers.

    ``layer_shapes`` is one ``dict(background=..., shapes=[...])`` per layer, with the shapes'
    ``cx`` / ``cy`` / ``lx`` / ``ly`` as absolute lengths in the units of ``period`` (see the
    module docstring). Returns ``(epx, epy, epz_i, epxy, epyx)``, each
    ``(n_layers, n_orders, n_orders)``.

    Every layer goes through :func:`check_layer_spec` and :func:`check_layer_shapes` first —
    this is the funnel the parametric path reaches the solver through, so it is the one place that
    can guarantee a misspelled recipe is never answered with a plausible structure.

    The array namespace is taken from the first differentiable shape parameter found, so
    passing JAX or PyTorch scalars for e.g. ``lx`` makes the whole build run — and
    differentiate — on that backend. Pass ``xp`` to force it.
    """
    if xp is None:
        xp = _infer_namespace(layer_shapes)
    dtype = xp.complex128 if complex_dtype is None else complex_dtype

    out = [[], [], [], [], []]
    for index, layer in enumerate(layer_shapes):
        check_layer_spec(layer, index)
        # A frozen geometry carries the decision for *every* layer, so when one is present it
        # is the whole answer: `plan is None` there means "looked at while the numbers were
        # concrete, and it does not overlap". Re-asking would put the question to traced
        # values that cannot answer it.
        frozen = boundary_plans is not None
        plan = boundary_plans[index] if frozen else None
        if frozen and plan is None:
            check_no_plan_still_fits(layer, lattice, index)
        check_layer_shapes(layer, lattice, index)
        background = layer.get('background', 1.0)
        if plan is not None or (not frozen
                                and layer_needs_resolving(layer, lattice)):
            # Shapes that overlap paint -- later over earlier -- and this module can only
            # sum, which is a different structure. Resolve the painting into disjoint polygons
            # and transform those. Where nothing overlaps the two are the same structure, so
            # the layer stays here -- exact, and cheaper.
            from .boundary import to_conv_mat_boundary_layer
            mats = to_conv_mat_boundary_layer(
                xp, layer.get('shapes', []), background, half_width_u, half_width_v, lattice,
                dtype, normal_grid=normal_grid, plan=plan)
        else:
            # Reaching this branch means the shapes are disjoint, so the summing this
            # builder does is the painted structure. `assemble_eps_coeffs` re-checks that --
            # a second reading of the dispatch just above, on the one branch where getting it
            # wrong returns a plausible number instead of an error.
            # That failure has happened once already (see `layer_needs_resolving`). It held
            # only for the unfrozen route until `check_no_plan_still_fits` was added: a record
            # frozen while the shapes were disjoint routes by its own answer, not by the two
            # rules, and once the shapes moved together the check below is what fired -- with
            # a message about this builder rather than about the stale record.
            mats = to_conv_mat_parametric_nvf(
                xp, layer.get('shapes', []), background, half_width_u, half_width_v, lattice,
                dtype, normal_grid=normal_grid)
        for slot, m in zip(out, mats):
            slot.append(m)

    return ConvMatrices(*(xp.stack(slot) for slot in out))


def _material_value(value):
    """A material slot's ``epsilon``, or the raw value, or ``None`` -- for namespace sniffing.

    The point is to look *inside* a :class:`~wavebit._core.domain.Material`. A material is a
    legitimate design variable, so its ``epsilon`` may be the only backend array in a layer
    written entirely with Python floats; missing it would build the whole layer in NumPy and
    drop the gradient at the very parameter being optimised.
    """
    if value is None:
        return None
    return as_material(value).epsilon


def _infer_namespace(layer_shapes):
    """Namespace of the first array-like shape parameter; NumPy when everything is a scalar."""
    for layer in layer_shapes:
        candidates = [_material_value(layer.get('background'))]
        for shape in layer.get('shapes', []):
            candidates.extend(shape.get(k) for k in
                              ('cx', 'cy', 'lx', 'ly', 'l', 'angle'))
            candidates.append(_material_value(shape.get('material')))
        for c in candidates:
            if c is not None and hasattr(c, 'dtype') and hasattr(c, 'shape'):
                return array_namespace(c)
    import numpy as np
    return array_namespace(np.zeros(1))
