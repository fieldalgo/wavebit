"""What the continuity rule reads off a structure, for every door that reconstructs a field.

The rule (P. Lalanne and M. P. Jurek, J. Mod. Opt. 45, 1357 (1998)) takes the modes from the solve
and ``eps`` and Li's inverse rule from the structure, so the two must be the same scene. The solve
keeps its own copy of the structure (:func:`solved_structure`) and the rule reads only that, which
is what the Fourier sum does in effect: its field after an edit is still the solved scene's. A
derivative runs through the copy as it ran through the solve, or through neither.

Free functions over that copy, not solver methods, because a result outlives its solver and asks
the same questions (:class:`~wavebit.rcwa._engine.field_result.FieldState`).
"""

import sys
from collections.abc import Mapping
from collections.abc import Sequence
from copy import copy

import numpy as np

from wavebit._common.backend import array_namespace
from wavebit._common.backend import as_array
from wavebit._common.backend import carries_autodiff
from wavebit.lattice import Lattice
from wavebit.lattice import periods_of
from wavebit.materials.response import Material
from wavebit.materials.response import MaterialResponse
from wavebit.rcwa._engine.factorization import line_inverse_rules
from wavebit.rcwa._engine.fields import _axis_samples
from wavebit.rcwa._engine.fields import is_count
from wavebit.rcwa._engine.line_crossings import eps_at
from wavebit.rcwa._engine.line_crossings import line_operators
from wavebit.rcwa._engine.line_crossings import read_lines
from wavebit.rcwa._engine.line_crossings import slanted_wall


def _host(value):
    from wavebit.rcwa._engine.field_result import _host_value
    return _host_value(value)


def _device_of(value):
    return getattr(value, 'device', None)


def _replaced(value, **fields):
    """A frozen dataclass with ``fields`` swapped, past a constructor a subclass may have rewritten."""
    copied = copy(value)
    for name, item in fields.items():
        object.__setattr__(copied, name, item)
    return copied


def solved_structure(value):
    """A copy of a structure as a solve saw it: later edits to its arrays do not reach the copy.

    A copy of a differentiated array stays on that derivative's path, and a Torch leaf copied
    under ``no_grad`` is off it, as the solve's modes are. A container comes back plain, whatever
    its spelling was (a ``namedtuple``, a ``deque``, a read-only mapping), because the readers only
    ever walk them and a mutable one is as live as an array; a tuple stays a tuple and a sequence
    a list, which is the one difference the readers branch on. A mapping's keys are put in order,
    so the spelling of a dict is not part of a result's pytree. An array of a library the solve
    only reads through ``__array__`` (xarray's) comes back NumPy, and anything else that reads as
    a number comes back that number, since a JAX pytree cannot hold it and it may be mutable.
    """
    if value is None or isinstance(value, (str, bytes, int)):
        return value
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(value, torch.Tensor):
        return value.clone()
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        try:
            return array_namespace(value).asarray(value, copy=True)
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return {name: solved_structure(value[name]) for name in sorted(value, key=str)}
    if isinstance(value, tuple):
        return tuple(solved_structure(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [solved_structure(item) for item in value]
    if isinstance(value, (Material, MaterialResponse)):
        return _replaced(value, epsilon=solved_structure(value.epsilon), mu=solved_structure(value.mu))
    if isinstance(value, Lattice):
        return _replaced(value, u=solved_structure(value.u), v=solved_structure(value.v))
    if hasattr(value, '__array__'):
        return np.array(value)
    for number in (float, complex):
        try:
            return number(value)
        except TypeError:
            pass
    return value


def refusal(kind, structure, lattice, kx, ky, tensor, merged):
    """Why the rule cannot answer this solve, by name, or ``None``.

    ``kind`` is the solve's path as :class:`~wavebit.rcwa._engine.field_result.FieldState` names
    it, ``tensor`` whether its materials are, ``merged`` whether it merged layers into its
    half-spaces. ``structure`` is :func:`solved_structure`'s copy, ``None`` after matrices handed
    to ``solve_conv``, which need not be any structure.
    """
    if kind == 'source':
        return ("discontinuous='continuity' is built on a plane-wave solve; a solve of "
                'dipoles or other sources still takes the Fourier sum.')
    if merged:
        return ("discontinuous='continuity' is not available after a solve that merged layers "
                'into its half-spaces.')
    if kind == 'first_order':
        return ("discontinuous='continuity' is built on the second-order 2D reconstruction; the "
                'first-order formulation still takes the Fourier sum.')
    if kind == 'second_order' and np.ndim(kx) != 1:
        return ("discontinuous='continuity' applies its rule along the cell's x and y lines; a "
                'skewed lattice has no such lines and is refused, as is a lattice differentiated '
                'as a whole matrix, which can shear. A rectangular cell differentiates through '
                'period=.')
    if structure is None:
        return ("discontinuous='continuity' reads eps and the inverse rule off the structure, "
                'and solve_conv solved matrices that need not be that structure. Use solve().')
    if tensor:
        return ("discontinuous='continuity' divides D_x by a scalar eps; a tensor material "
                'couples D_x to more than E_x and is refused.')
    if isinstance(structure, list):
        return _shape_refusal(structure, lattice, kx, ky)
    return None


def _shape_refusal(structure, lattice, kx, ky):
    """The three things the rule's lines through a shape structure do not describe.

    **A slanted or curved wall.** ``D_x`` is not what is continuous across it, ``D_n`` is, so
    the rule leaves the wall's tangential ``E`` carrying the whole contrast: a jump of 0.75 at
    45 degrees that does not shrink with the truncation. A raster draws such a wall as a
    staircase, and every wall of the staircase is along an axis.

    **One order along an axis.** The solve then cannot resolve the layer along that axis, while
    the lines run through the real shape. Under Laurent's rule ``period=[400]`` with ``ly=100``,
    ``eps=4`` has the same ``R``, order by order, as ``ly=400``, ``eps=1.75``, where the rule
    would read 4 and 1.75 on different lines; Li's rules and NVF collapse it differently.

    **A negative side of the lattice.** It passes the skew check and gives the same ``R``, but
    its field is not the positive basis's (fieldalgo/wavebit#130), which the lines, read in a
    positive period, cannot answer. A trace has no sign to refuse: :func:`continuity_rows` masks
    that case instead.

    A structure with no shapes in it has none of the three to fear: every layer keeps the sum.
    """
    if not any(layer.get('shapes') for layer in structure):
        return None
    if kx.shape[0] == 1 or ky.shape[0] == 1:
        return ("discontinuous='continuity' reads lines through the real shape, and with one "
                'order along an axis the solve cannot resolve the layer along it, so the two '
                'would describe different structures. Keep more than one order along each axis, '
                'or state a grating as a one-row raster.')
    columns = _host(getattr(lattice, 'columns', lattice))
    if columns is not None and (columns[0, 0] < 0 or columns[1, 1] < 0):
        return ("discontinuous='continuity' on shapes reads its lines in a positive period; give "
                'the lattice with positive sides.')
    for i, layer in enumerate(structure):
        for j, shape in enumerate(layer.get('shapes', [])):
            if slanted_wall(shape, lattice):
                return ("discontinuous='continuity' on shapes is built for walls along x and y: "
                        'rectangles at a right angle and polygons with only such edges. '
                        f"Layer {i} shape {j} ({shape.get('type', 'rectangle')}) has a slanted or "
                        'curved wall, or one a trace hides the direction of. Across a slanted '
                        'wall the rule makes D_n continuous and carries the full contrast in E_t, '
                        "and no reference has ranked that against the Fourier sum yet. Keep "
                        "'fourier', or sample the shape on a raster for the rule on its staircase.")
    return None


def sample_pixels(spec, period, n_pixels, eps):
    """The raster pixel under each sample along one axis, for the continuity rule.

    Pixel ``j`` covers ``[j, j+1) * period / n_pixels`` from the cell's negative edge, so a
    sample is read from the pixel it falls in, periodically. A sample exactly on a wall
    between two permittivities belongs to the pixel on its positive side.
    """
    if is_count(spec):
        corner = np.linspace(0., 1., int(spec))
    else:
        points = _host(spec)
        if points is None or period is None:
            raise NotImplementedError("discontinuous='continuity' on a raster chooses the pixel "
                                      'under each sample, which explicit coordinates cannot do '
                                      'without a concrete period and concrete coordinates. Under '
                                      'jax.jit, sample by counts.')
        corner = np.asarray(points, dtype=float) / float(period) + 0.5
    # Half a part in 1e9 lifts a sample that rounding put just below a pixel edge it sits on;
    # a period held in single precision rounds further than that (3e-8 of a pixel at 8 pixels).
    lift = max(1e-9, 64.0 * eps * n_pixels)
    return np.floor(corner * n_pixels + lift).astype(int) % n_pixels


def _positive_sides(xp, lattice, dev):
    """Whether a lattice a trace hides is positive, or ``None`` where its sign is there to read."""
    columns = getattr(lattice, 'columns', lattice)
    if _host(columns) is not None:
        return None
    return (as_array(xp, columns[0, 0], device=dev) > 0) & (as_array(xp, columns[1, 1], device=dev) > 0)


def _masked(xp, eps, positive):
    """``eps``, or NaN where a trace hid a negative side: the refusal, for a trace that cannot read one."""
    return eps if positive is None else xp.where(positive, eps, xp.full_like(eps, float('nan')))


def samples_along_cell(spec, side):
    """A result's samples along a lattice side, as the lines and the pixels can read them.

    :func:`~wavebit.rcwa._engine.field_result.xy_grid` runs a count along the side as written, so
    down the cell on a negative one; the lines and pixels read a count up a positive period. As
    coordinates the two are the same points.
    """
    return np.linspace(-0.5, 0.5, int(spec)) * side if is_count(spec) and side < 0 else spec


def _shape_rows(xp, structure, lattice, half_u, half_v, real, C, dev, sample_x, sample_y):
    """:func:`continuity_rows` of a shape structure.

    The rows and columns are the lines through the samples themselves, read off the shapes
    (:mod:`wavebit.rcwa._engine.line_crossings`) rather than off pixels, and the samples are
    the ones the phases use (:func:`wavebit.rcwa._engine.fields._axis_samples`), so the two
    cannot be in different frames. 64 units of rounding put a sample that sits on a wall on
    the wall's positive side, a raster's pixel rule; they go on what is read off the shapes
    and never on the phases. They are units of the period, and a coordinate rounds in units of
    its own size, so a sample or a shape written more than about 60 periods from the cell can
    read a wall it sits on from the other side.
    """
    lift = 64.0 * xp.finfo(real).eps
    period_x, period_y = (as_array(xp, p, dtype=real, device=dev) for p in periods_of(lattice))
    x = xp.real(_axis_samples(xp, sample_x, period_x, C, dev)) + lift * period_x
    y = xp.real(_axis_samples(xp, sample_y, period_y, C, dev)) + lift * period_y
    unit_x = x / period_x - xp.floor(x / period_x)
    positive = _positive_sides(xp, lattice, dev)
    rows = []
    for layer in structure:
        if not layer.get('shapes'):
            rows.append(None)
            continue
        along_x = read_lines(xp, layer, lattice, 0, y, period_x, period_y, C)
        along_y = read_lines(xp, layer, lattice, 1, x, period_y, period_x, C)
        rows.append((line_operators(xp, along_x, half_u), line_operators(xp, along_y, half_v),
                     _masked(xp, eps_at(xp, along_x, unit_x), positive)))
    return rows


def _sampled_line_rules(xp, grid, lines, half_width, dev):
    """Li's inverse rule of the lines under the samples: each such line once, and no other.

    The samples may share lines, and a singular line no sample is on is no reason to fail.
    """
    held, back = np.unique(lines, return_inverse=True)
    rules = line_inverse_rules(xp, xp.take(grid, as_array(xp, held, device=dev), axis=0), half_width)
    return xp.take(rules, as_array(xp, back, device=dev), axis=0)


def _raster_rows(xp, structure, lattice, half_u, half_v, real, two_d, sample_x, sample_y):
    """:func:`continuity_rows` of a raster.

    A layer differentiated at a uniform point is not uniform to its derivative, so it goes through
    the rule: only its value is the sum's, and the sum's derivative is 124% off the rule's.
    """
    periods = periods_of(lattice)
    eps = float(xp.finfo(real).eps)
    columns = sample_pixels(sample_x, _host(periods[0]), structure.shape[-1], eps)
    if two_d:
        lines = sample_pixels(sample_y, _host(periods[1]), structure.shape[-2], eps)
    rows = []
    for layer in range(structure.shape[0]):
        grid = structure[layer] if two_d else structure[layer, 0]
        values = _host(grid)
        if values is not None and not carries_autodiff(grid) and np.all(values == values.flat[0]):
            rows.append(None)
            continue
        dev = _device_of(grid)
        positive = _positive_sides(xp, lattice, dev)
        at_x = as_array(xp, columns, device=dev)
        if not two_d:
            rows.append(_masked(xp, xp.take(grid, at_x, axis=0), positive))
            continue
        at_y = as_array(xp, lines, device=dev)
        ops_x = _sampled_line_rules(xp, grid, lines, half_u, dev)
        ops_y = _sampled_line_rules(xp, xp.permute_dims(grid, (1, 0)), columns, half_v, dev)
        eps_samples = xp.take(xp.take(grid, at_y, axis=0), at_x, axis=1)
        rows.append((ops_x, ops_y, _masked(xp, eps_samples, positive)))
    return rows


def continuity_rows(xp, structure, lattice, kx, ky, real, C, two_d, sample_x, sample_y):
    """What the rule reads off each layer, in the public order of the layers.

    ``None`` where the layer is uniform -- it keeps the Fourier sum, bit for bit, since there is
    nothing for the rule to change there (:func:`_raster_rows` on a derivative). Otherwise, on the 1D paths, ``eps`` at the x samples
    ``(n_x,)``; on the 2D path a triple ``(ops_x, ops_y, eps)``: ``ops_x[j]`` is Li's inverse rule
    ``[[1/eps]]_x^-1`` along the row under the ``j``-th y sample, ``ops_y[i]`` the same along the
    column under the ``i``-th x sample, and ``eps`` the ``(n_y, n_x)`` permittivity at the samples
    (:func:`wavebit.rcwa._engine.fields._continuity_2d`). A row is a raster's row of pixels, or on
    shapes the line through the sample.
    """
    half_u = (kx.shape[0] - 1) // 2
    half_v = (ky.shape[0] - 1) // 2 if two_d else 0
    if isinstance(structure, list):
        return _shape_rows(xp, structure, lattice, half_u, half_v, real, C, _device_of(kx),
                           sample_x, sample_y)
    return _raster_rows(xp, structure, lattice, half_u, half_v, real, two_d, sample_x, sample_y)
