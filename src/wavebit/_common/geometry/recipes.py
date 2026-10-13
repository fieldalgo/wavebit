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
    cx = _held(shape.get('cx', 0.0)) + 0.5 * (row_x[0] + row_x[1])
    cy = _held(shape.get('cy', 0.0)) + 0.5 * (row_y[0] + row_y[1])
    lx = _held(shape.get('lx', shape.get('l', 0.0)))
    round_ = stype in ('circle', 'disk', 'square')
    ly = lx if round_ else _held(shape.get('ly', shape.get('l', 0.0)))
    return stype, cx, cy, lx, ly, _held(shape.get('angle', 0.0))


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
