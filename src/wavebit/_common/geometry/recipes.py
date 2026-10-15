"""Periodic geometry recipes without Fourier operators."""
import numpy as _np
from wavebit._common.backend import checked_float
from wavebit._common.names import near_match, quantity_note

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
    :func:`~wavebit.rcwa._engine.boundary.shape_vertices` was a third reader with its own rules
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
    ``tests/rcwa/test_boundary.py:355`` recorded the defect in prose and pinned only that the
    readers agree with each other, which they still do: there is one origin and it moves them
    together.
    """
    stype = shape.get('type', 'rectangle')
    if stype == POLYGON:
        # Refused rather than read with defaults: every key below would come back `0.0` and the
        # polygon would be a zero-size box at the cell centre -- which no caller would notice,
        # because nothing here raises. A polygon is its vertices; the readers that can take one
        # read them (`boundary.shape_vertices`), and every other reader has to refuse it first.
        raise NotImplementedError(
            "a type='polygon' shape has no (cx, cy, lx, ly, angle): it is placed by its "
            "vertices. Read it with wavebit.rcwa._engine.boundary.shape_vertices; this reader "
            "is for the parametric primitives only.")
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
    ox, oy = cell_centre(lattice)
    cx = _held(shape.get('cx', 0.0)) + ox
    cy = _held(shape.get('cy', 0.0)) + oy
    lx = _held(shape.get('lx', shape.get('l', 0.0)))
    round_ = stype in ('circle', 'disk', 'square')
    ly = lx if round_ else _held(shape.get('ly', shape.get('l', 0.0)))
    return stype, cx, cy, lx, ly, _held(shape.get('angle', 0.0))


#: The shape type whose outline is a list of vertices rather than a primitive's parameters.
POLYGON = 'polygon'


def is_polygon(shape):
    """Whether a shape dict is a polygon -- a question of its *type*, so a trace can ask it."""
    return shape.get('type', 'rectangle') == POLYGON


def holds_polygon(layer):
    """Whether any shape of a layer dict is a polygon (structural, like :func:`is_polygon`)."""
    return any(is_polygon(shape) for shape in layer.get('shapes', []))


def cell_centre(lattice):
    """``0.5 * (a1 + a2)``: the point every shape coordinate is measured from.

    Shared by :func:`shape_geometry`, which adds it to ``cx`` / ``cy``, and by the polygon
    reader, which adds it to every vertex -- so a polygon and a primitive written at the same
    numbers sit at the same place on every cell, sheared ones included.
    """
    # Two reads, not four: `Lattice.__getitem__` stacks its two vectors on every access, and
    # this runs for every shape rather than only for a defaulted one.
    row_x, row_y = lattice[0], lattice[1]
    return 0.5 * (row_x[0] + row_x[1]), 0.5 * (row_y[0] + row_y[1])


def _held(v):
    """A 0-d NumPy array as the NumPy scalar it holds; any other value as it is.

    A NumPy float64 or float32 scalar for any value of a box or an ellipse solves on Torch, on
    a scalar or a tensor layer, painted or not (``polygon._scalar`` converts what is not of the
    solver's namespace), but a 0-d *array* on the left of a tensor is refused --
    ``np.asarray(80.0) * tensor`` raises ``TypeError`` -- and a box's transform writes
    ``lx * sinc(...)``: a box whose ``lx`` was ``np.asarray(80.0)`` solved on NumPy and JAX and
    raised on Torch. Indexing with ``()`` keeps the dtype, so a float32 value stays one, and
    reads no value: a JAX array or a tensor is no NumPy array and passes untouched, its graph
    with it.
    """
    return v[()] if isinstance(v, _np.ndarray) and v.ndim == 0 else v


_COMMON_KEYS = frozenset({'type', 'cx', 'cy', 'angle', 'material'})


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
    :func:`shape_geometry` total on the primitives — it raises only for a polygon, by type, so
    it stays usable under a trace. The cost
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

    A ``polygon`` takes ``vertices`` and nothing positional (:func:`_check_polygon_spec`), and
    the same holds for it: only the *layout* of its vertices is read here -- how many pairs,
    of two coordinates each -- which a trace knows. Whether the outline is simple is a question
    of values and is asked where the values are concrete
    (:func:`~wavebit.rcwa._engine.parametric.check_layer_shapes`).
    """
    where = f'Analytic layer {layer_index}, shape {shape_index}'
    stype = shape.get('type', 'rectangle')
    if stype == POLYGON:
        _check_polygon_spec(shape, where)
        return
    lengths = _LENGTH_KEYS.get(stype)
    if lengths is None:
        raise ValueError(f'{where}: type={stype!r} is not a shape this path can build. '
                         f'Known types: {sorted([*_LENGTH_KEYS, POLYGON])}.')

    if 'ly' in shape and 'ly' not in lengths:
        # Answered here rather than by the unknown-key branch below, which would offer 'l' as
        # the near miss and send a user who wants an oval further from `ellipse`, not closer.
        alt = 'ellipse' if stype in ('circle', 'disk') else 'rectangle'
        raise ValueError(
            f"{where}: {stype!r} has one length, so 'ly' would be dropped -- silently, and "
            f"with a zero gradient if it is a parameter you are optimising. Drop it if 'lx' "
            f"is the size you meant, or use type={alt!r} to give two lengths.")

    unknown = set(shape) - (lengths | _COMMON_KEYS)
    odd = [key for key in unknown if not isinstance(key, str)]
    if odd:
        # Before the guess, which matches strings only: `sorted` cannot order an int beside
        # them, and the refusal would otherwise be a TypeError from inside the hint.
        raise ValueError(f'{where}: shape keys are strings, got {", ".join(map(repr, odd))}. '
                         f'It takes {sorted(lengths | _COMMON_KEYS)}.')
    if unknown:
        known = sorted(lengths | _COMMON_KEYS)
        bits = []
        for key in sorted(unknown):
            # The package's one guess, which never sends `n_material` to `material`.
            near = near_match(key, known)
            bits.append(f'{key!r}' + (f" (did you mean {near!r}?)" if near else ''))
        raise ValueError(f'{where}: {stype!r} does not take {", ".join(bits)}.'
                         f'{quantity_note(unknown, known)} '
                         f'It takes {known}. A key nobody reads is silently ignored, which is '
                         f'why this is refused rather than dropped.')


#: Every key a polygon takes. No centre, length or angle: its vertices place, size and turn it.
_POLYGON_KEYS = frozenset({'type', 'vertices', 'material'})

#: The primitives' positional keys, which a polygon would silently drop.
_PLACING_KEYS = frozenset({'cx', 'cy', 'lx', 'ly', 'l', 'angle'})


def _check_polygon_spec(shape, where):
    """The polygon arm of :func:`check_shape_spec`: its keys, and how its vertices are laid out.

    The keys are refused for the same reason the primitives' are, and the positional ones by
    name first: ``cx=`` on a polygon reads like a move and would be nothing at all -- the
    vertices say where every corner is, so a centre beside them could only be ignored or
    double-counted, and an ignored one is a design variable with a zero gradient.
    """
    odd = [key for key in shape if not isinstance(key, str)]
    if odd:
        raise ValueError(f'{where}: shape keys are strings, got {", ".join(map(repr, odd))}. '
                         f'A polygon takes {sorted(_POLYGON_KEYS)}.')
    placing = sorted(set(shape) & _PLACING_KEYS)
    if placing:
        raise ValueError(
            f"{where}: a 'polygon' is placed, sized and turned by its vertices alone, so "
            f"{placing} would be dropped -- silently, and with a zero gradient if it is a "
            f"parameter you are optimising. Write the vertices where the corners are, measured "
            f"from the cell centre as cx and cy are.")
    unknown = sorted(set(shape) - _POLYGON_KEYS)
    if unknown:
        known = sorted(_POLYGON_KEYS)
        bits = [f'{key!r}' + (f" (did you mean {near!r}?)" if (near := near_match(key, known))
                              else '') for key in unknown]
        raise ValueError(f"{where}: 'polygon' does not take {', '.join(bits)}. It takes "
                         f"{known}. A key nobody reads is silently ignored, which is why this "
                         f"is refused rather than dropped.")
    if 'vertices' not in shape:
        raise ValueError(f"{where}: a 'polygon' needs vertices=, a sequence of (x, y) pairs "
                         f"measured from the cell centre.")
    problem = outline_layout(shape['vertices'])
    if problem is not None:
        raise ValueError(f'{where}: polygon vertices {problem}.')


def outline_layout(vertices):
    """What is wrong with how ``vertices`` is laid out, or ``None`` -- a question a trace answers.

    Only lengths and array shapes are read: ``N`` pairs of two coordinates each, ``N >= 3``.
    The values -- finite, real, a simple outline with area -- are
    :func:`~wavebit._common.geometry.polygon.polygon_defect`'s, asked where they are concrete.
    """
    shape = getattr(vertices, 'shape', None)
    if shape is not None:
        if len(shape) != 2 or shape[1] != 2:
            return f'are an (N, 2) array of (x, y) pairs; got shape {tuple(shape)}'
        count = shape[0]
    elif isinstance(vertices, (tuple, list)):
        for k, pair in enumerate(vertices):
            pair_shape = getattr(pair, 'shape', None)
            if pair_shape is not None:
                if tuple(pair_shape) != (2,):
                    return f'are (x, y) pairs; vertex {k} has shape {tuple(pair_shape)}'
            elif not isinstance(pair, (tuple, list)) or len(pair) != 2:
                return f'are (x, y) pairs; vertex {k} is {pair!r}'
        count = len(vertices)
    else:
        return (f'are a sequence of (x, y) pairs or an (N, 2) array; got '
                f'{type(vertices).__name__}')
    if count < 3:
        return f'number {count}; an outline needs at least three'
    return None


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
    odd = [key for key in layer if not isinstance(key, str)]
    if odd:
        # As in `check_shape_spec`, and before the sort below: `sorted` cannot order an int
        # beside a string, and the refusal would otherwise be a bare TypeError from inside it.
        raise ValueError(f'Analytic layer {layer_index}: layer keys are strings, got '
                         f'{", ".join(map(repr, odd))}. A layer takes {sorted(_LAYER_KEYS)}.')
    unknown = sorted(set(layer) - _LAYER_KEYS)
    if unknown:
        raise ValueError(
            f'Analytic layer {layer_index}: {unknown} is not read by any path here. A layer '
            f'takes {sorted(_LAYER_KEYS)}. Refused rather than ignored, because an unread key '
            f'leaves a layer that still *builds*: a dict carrying nothing this reads has no '
            f'shapes either, and an empty shape list is legal, so it comes back as a uniform '
            f'slab of the background — which solves, conserves energy to 1e-14, and is not the '
            f'structure that was asked for.')


def _cell_floats(lattice):
    """``(a1x, a1y, a2x, a2y)`` as plain floats, for the eager planner. Raises when traced."""
    return (checked_float(lattice[0][0]), checked_float(lattice[1][0]),
            checked_float(lattice[0][1]), checked_float(lattice[1][1]))
