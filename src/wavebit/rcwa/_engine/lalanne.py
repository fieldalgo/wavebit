"""What the Lalanne rule reads off a structure, for every door that reconstructs a field.

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
from collections import namedtuple
from copy import copy

import numpy as np

from wavebit._common.backend import array_namespace
from wavebit._common.backend import as_array
from wavebit._common.backend import carries_autodiff
from wavebit.lattice import Lattice
from wavebit.lattice import periods_of
from wavebit.materials.response import Material
from wavebit.materials.response import MaterialResponse
from wavebit.materials.response import as_material
from wavebit.materials.tensors import material_is_tensor
from wavebit.materials.tensors import tensor_shape
from wavebit.rcwa._engine.factorization import line_inverse_rules
from wavebit.rcwa._engine.fields import _axis_samples
from wavebit.rcwa._engine.fields import is_count
from wavebit.rcwa._engine.line_crossings import eps_at
from wavebit.rcwa._engine.line_crossings import is_magnetic
from wavebit.rcwa._engine.line_crossings import line_operators
from wavebit.rcwa._engine.line_crossings import material_paint
from wavebit.rcwa._engine.line_crossings import read_lines
from wavebit.rcwa._engine.line_crossings import SLANT_TOL
from wavebit.rcwa._engine.line_crossings import slanted_wall


#: What the rule reads off one layer. ``electric`` rebuilds ``E`` from ``D``, ``magnetic`` rebuilds
#: ``H`` from ``B``, and either is ``None`` where that sum stands: a uniform ``eps``, or a layer
#: that names no permeability, where every component of ``H`` is continuous.
LayerRule = namedtuple('LayerRule', 'electric magnetic')


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
    jax = sys.modules.get('jax')
    if jax is not None and isinstance(value, jax.Array) and not isinstance(value, jax.core.Tracer):
        # A plain copy taken inside a trace is a tracer, and the refusals read values.
        with jax.ensure_compile_time_eval():
            return jax.numpy.array(value, copy=True)
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


def _kept(merged, count):
    """The public layers a solve kept, past those it merged into its half-spaces."""
    return range(count) if merged is None else range(merged[0], count - merged[1])


def refusal(kind, structure, lattice, kx, ky, mu=None, merged=None):
    """Why the rule cannot answer this solve, by name, or ``None``.

    ``kind`` is the solve's path as :class:`~wavebit.rcwa._engine.field_result.FieldState` names
    it, and ``merged`` the ``(bottom, top, lit side)`` of a solve that merged layers into its
    half-spaces, which are uniform and so never read. ``structure`` is :func:`solved_structure`'s
    copy, ``None`` after matrices handed to ``solve_conv``, which need not be any structure, and
    ``mu`` the same copy of a raster's permeability. A source solve has no ``kx`` of the cell's
    shape to ask about the lattice, so it asks the lattice what the solve did.
    """
    if _negative_side(lattice):
        return ("reconstruction='lalanne' follows the x and y lines of a rectangular cell with "
                'positive sides. A diagonal lattice with a negative side spans the same lattice but '
                'is read as spelled, on the general path, where the rule has no lines to follow, '
                'so it is refused. Spell the cell with positive sides (a raster then reversed '
                'along that axis) to use the rule.')
    skewed = np.ndim(kx) != 1 if kind in ('second_order', 'first_order') else False
    if kind == 'source':
        skewed = not getattr(lattice, 'axis_aligned', False)
    if skewed:
        return ("reconstruction='lalanne' applies its rule along the cell's x and y lines; a "
                'skewed lattice has no such lines and is refused, as is a lattice differentiated '
                'as a whole matrix, which can shear. A rectangular cell differentiates through '
                'period=.')
    if structure is None:
        return ("reconstruction='lalanne' reads eps and the inverse rule off the structure, "
                'and solve_conv solved matrices that need not be that structure. Use solve().')
    if isinstance(structure, list):
        return _shape_refusal(structure, lattice, _kept(merged, len(structure)))
    kept = _kept(merged, structure.shape[0])
    return _raster_tensor_refusal(structure, kept) or (None if mu is None else _raster_tensor_refusal(mu, kept))


def _negative_side(lattice):
    """Whether the cell is diagonal with a negative side, read on the host; ``False`` under a trace."""
    try:
        columns = np.asarray(getattr(lattice, 'columns', None), dtype=float)
    except Exception:
        return False
    if columns.shape != (2, 2) or columns[0, 1] != 0 or columns[1, 0] != 0:
        return False
    return bool(np.any(np.diag(columns) < 0))


_COUPLING = ("reconstruction='lalanne' divides D_x by eps_xx and B_x by mu_xx, and {where} has "
             'off-diagonal entries that couple them to the other components. A diagonal tensor '
             'takes the rule, and so does a tensor of any kind that is uniform across its layer, '
             'which has no walls.')
_HIDDEN = ("reconstruction='lalanne' divides D_x by eps_xx and B_x by mu_xx, and {where} is a whole "
           'tensor under a trace or a derivative, whose zero off-diagonals can carry one the rule '
           'would drop. On shapes, give a diagonal tensor as principal values, Material([xx, yy, zz]), '
           'which keeps its derivative.')


def _off_diagonals(values):
    """Do ``(..., 3, 3)`` values couple the axes, past what ``Material.rotated`` leaves in rounding?"""
    values = np.asarray(values)
    return bool(np.max(np.abs(values[..., ~np.eye(3, dtype=bool)])) > SLANT_TOL * np.max(np.abs(values)))


def _hidden_tensor(value):
    """Is this a whole tensor whose entries cannot be read, or that a derivative runs through?

    Principal values have no off-diagonal to carry a derivative; a zero entry of a whole tensor
    can, and the rule would drop it without a word.
    """
    return value is not None and tensor_shape(value) == (3, 3) and (carries_autodiff(value) or _host(value) is None)


def _as_tensor_values(value):
    """A material value as ``(3, 3)``, ``None`` where it cannot be read. No value is one."""
    if value is None:
        return np.eye(3)
    try:
        host = np.asarray(_host(value) if hasattr(value, 'shape') else value)
    except (TypeError, RuntimeError, ValueError):
        return None
    if host.dtype == object:
        return None
    return host * np.eye(3) if host.ndim == 0 else np.diag(host) if host.ndim == 1 else host


def _quantity_refusal(values, where):
    """One layer's ``epsilon`` or ``mu``, each material's: walls need a diagonal tensor, as the raster asks."""
    whole = [value for value in values if value is not None and tensor_shape(value) == (3, 3)]
    if any(_hidden_tensor(value) for value in whole):
        return _HIDDEN.format(where=where)
    if not any(_off_diagonals(_as_tensor_values(value)) for value in whole):
        return None
    tensors = [_as_tensor_values(value) for value in values]
    if all(t is not None and np.array_equal(t, tensors[0]) for t in tensors):
        return None
    return _COUPLING.format(where=where)


def _raster_tensor_refusal(field, kept):
    """A tensor raster's layers, each uniform or diagonal, by value: it has no structural spelling."""
    if field.ndim != 5:
        return None
    # Read once, whole: an index taken inside a trace is a tracer, even of a concrete raster.
    whole = None if carries_autodiff(field) else _host(field)
    for layer in kept:
        if whole is None:
            return _HIDDEN.format(where=f'layer {layer}')
        values = whole[layer]
        if not np.all(values == values[0, 0]) and _off_diagonals(values):
            return _COUPLING.format(where=f'layer {layer}')
    return None


def _shape_refusal(structure, lattice, kept):
    """The three things the rule's lines through a shape structure do not describe.

    **A slanted or curved wall.** ``D_x`` is not what is continuous across it, ``D_n`` is, so
    the rule leaves the wall's tangential ``E`` carrying the whole contrast: a jump of 0.75 at
    45 degrees that does not shrink with the truncation. A raster draws such a wall as a
    staircase, and every wall of the staircase is along an axis. That is not a free substitute:
    against a 3D finite-element reference, a high-index circle drawn as a staircase and solved with
    Li's rules had ``R`` off by 18 to 51% on the cells whose ``R`` is 0.5 or more, unchanged by more
    steps or pixels, while NVF on the same raster came near the true circle on the one cell tried.

    **A negative side of the lattice.** It passes the skew check and gives the same ``R``, but
    its field is not the positive basis's (fieldalgo/wavebit#130), which the lines, read in a
    positive period, cannot answer. A trace has no sign to refuse: :func:`lalanne_rows` masks
    that case instead.

    **A patterned tensor with off-diagonals.** ``E_x`` then needs more than ``D_x`` and a division
    (:data:`_COUPLING`).

    A layer with no shapes in it has none of the three to fear: it keeps the sum.
    """
    held = [i for i in kept if structure[i].get('shapes')]
    if not held:
        return None
    columns = _host(getattr(lattice, 'columns', lattice))
    if columns is not None and (columns[0, 0] < 0 or columns[1, 1] < 0):
        return ("reconstruction='lalanne' on shapes reads its lines in a positive period; give "
                'the lattice with positive sides.')
    for i in held:
        layer = structure[i]
        for j, shape in enumerate(layer['shapes']):
            if slanted_wall(shape, lattice):
                return ("reconstruction='lalanne' on shapes is built for walls along x and y: "
                        'rectangles at a right angle and polygons with only such edges. '
                        f"Layer {i} shape {j} ({shape.get('type', 'rectangle')}) has a slanted or "
                        'curved wall, or one a trace hides the direction of. Across a slanted '
                        'wall it is D_n that is continuous, not D_x, so the rule leaves the full '
                        'contrast in E_t, and more orders do not shrink that jump. Keep '
                        "'plain'. A raster of the shape takes the rule on its staircase; check "
                        'its R and T against the true shape first, since with Li\'s rules a '
                        'high-index staircase can be off by tens of percent at usable orders.')
        materials = [as_material(item) for item in (
            layer.get('background', 1.0), *(shape.get('material', 1.0) for shape in layer['shapes']))]
        message = (_quantity_refusal([m.epsilon for m in materials], f'the permittivity of layer {i}')
                   or _quantity_refusal([m.mu for m in materials], f'the permeability of layer {i}'))
        if message is not None:
            return message
    return None


def layers_holding(sample_z, thickness):
    """The public layers a sample is in, or ``None`` where the heights cannot say: a count or a trace.

    The rule inverts its lines ahead of the samples, and a layer no sample is in is no reason to
    fail, on a singular line, or to hand a gradient its NaN. Within rounding of an interface a sample
    holds both layers: a mirrored height can land on either side.
    """
    if is_count(sample_z):
        return None
    z = _host(sample_z) if hasattr(sample_z, 'shape') else np.asarray(sample_z, dtype=float)
    d = _host(thickness)
    if z is None or d is None:
        return None
    edges = np.r_[0.0, np.cumsum(np.real(np.asarray(d)))]
    z, tol = np.real(np.ravel(z)), 1e-9 * max(edges[-1], 1.0)
    low = np.searchsorted(edges, z - tol, side='right') - 1
    high = np.searchsorted(edges, z + tol, side='right') - 1
    return {i for a, b in zip(low, high) for i in range(max(a, 0), min(b, d.shape[0] - 1) + 1)}


def sample_pixels(spec, period, n_pixels, eps):
    """The raster pixel under each sample along one axis, for the Lalanne rule.

    Pixel ``j`` covers ``[j, j+1) * period / n_pixels`` from the cell's negative edge, so a
    sample is read from the pixel it falls in, periodically. A sample exactly on a wall
    between two permittivities belongs to the pixel on its positive side.
    """
    if is_count(spec):
        corner = np.linspace(0., 1., int(spec))
    else:
        points = _host(spec)
        if points is None or period is None:
            raise NotImplementedError("reconstruction='lalanne' on a raster chooses the pixel "
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


def _shape_rows(xp, structure, lattice, half_u, half_v, real, C, dev, sample_x, sample_y, held):
    """:func:`lalanne_rows` of a shape structure.

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
    for index, layer in enumerate(structure):
        if not layer.get('shapes') or (held is not None and index not in held):
            rows.append(None)
            continue
        along_x = read_lines(xp, layer, lattice, 0, y, period_x, period_y, C)
        along_y = read_lines(xp, layer, lattice, 1, x, period_y, period_x, C)
        half = _shape_half(xp, along_x, along_y, layer, C, half_u, half_v, unit_x, positive)
        rows.append(LayerRule(half('epsilon'), half('mu_value') if is_magnetic(layer) else None))
    return rows


def _shape_half(xp, along_x, along_y, layer, C, half_u, half_v, unit_x, positive):
    """A layer's half for ``'epsilon'`` or ``'mu_value'``, the x entry on the rows and the y on the columns."""
    tensor = any(material_is_tensor(item) for item in (
        layer.get('background', 1.0), *(shape.get('material', 1.0) for shape in layer['shapes'])))

    def half(quantity):
        xx = material_paint(xp, along_x, layer, C, quantity, 0)
        yy = material_paint(xp, along_y, layer, C, quantity, 1)
        at_x = _masked(xp, eps_at(xp, xx, unit_x), positive)
        at_y = (_masked(xp, eps_at(xp, material_paint(xp, along_x, layer, C, quantity, 1), unit_x), positive)
                if tensor else at_x)
        return line_operators(xp, xx, half_u), line_operators(xp, yy, half_v), at_x, at_y
    return half


def _sampled_line_rules(xp, grid, lines, half_width, dev):
    """Li's inverse rule of the lines under the samples: each such line once, and no other.

    The samples may share lines, and a singular line no sample is on is no reason to fail.
    """
    held, back = np.unique(lines, return_inverse=True)
    rules = line_inverse_rules(xp, xp.take(grid, as_array(xp, held, device=dev), axis=0), half_width)
    return xp.take(rules, as_array(xp, back, device=dev), axis=0)


def _raster_rows(xp, structure, mu, lattice, half_u, half_v, real, two_d, sample_x, sample_y, held):
    """:func:`lalanne_rows` of a raster: ``eps``'s lines, and ``mu``'s where one is given."""
    electric = _field_rows(xp, structure, lattice, half_u, half_v, real, two_d, sample_x, sample_y, held)
    magnetic = ([None] * len(electric) if mu is None else
                _field_rows(xp, mu, lattice, half_u, half_v, real, two_d, sample_x, sample_y, held))
    return [None if e is None and m is None else LayerRule(e, m) for e, m in zip(electric, magnetic)]


def _field_rows(xp, structure, lattice, half_u, half_v, real, two_d, sample_x, sample_y, held):
    """What the rule reads off one raster field, ``eps`` or ``mu``, layer by layer.

    A layer differentiated at a uniform point is not uniform to its derivative, so it goes through
    the rule: only its value is the sum's, and the sum's derivative is 124% off the rule's.
    """
    periods = periods_of(lattice)
    eps = float(xp.finfo(real).eps)
    columns = sample_pixels(sample_x, _host(periods[0]), structure.shape[2], eps)
    if two_d:
        lines = sample_pixels(sample_y, _host(periods[1]), structure.shape[1], eps)
    rows = []
    for layer in range(structure.shape[0]):
        if held is not None and layer not in held:
            rows.append(None)
            continue
        grid = structure[layer] if two_d else structure[layer, 0]
        values = _host(grid)
        if values is not None and not carries_autodiff(grid) and np.all(
                values == (values[0, 0] if structure.ndim == 5 else values.flat[0])):
            rows.append(None)
            continue
        dev = _device_of(grid)
        positive = _positive_sides(xp, lattice, dev)
        at_x = as_array(xp, columns, device=dev)
        if not two_d:
            rows.append(_masked(xp, xp.take(grid, at_x, axis=0), positive))
            continue
        at_y = as_array(xp, lines, device=dev)
        # A tensor raster is diagonal or uniform by now (`_raster_tensor_refusal`): each axis has its entry.
        along_x, along_y = (grid, grid) if structure.ndim == 3 else (grid[..., 0, 0], grid[..., 1, 1])
        ops_x = _sampled_line_rules(xp, along_x, lines, half_u, dev)
        ops_y = _sampled_line_rules(xp, xp.permute_dims(along_y, (1, 0)), columns, half_v, dev)
        eps_x = _masked(xp, xp.take(xp.take(along_x, at_y, axis=0), at_x, axis=1), positive)
        eps_y = eps_x if structure.ndim == 3 else _masked(
            xp, xp.take(xp.take(along_y, at_y, axis=0), at_x, axis=1), positive)
        rows.append((ops_x, ops_y, eps_x, eps_y))
    return rows


def lalanne_rows(xp, structure, mu, lattice, kx, ky, real, C, two_d, sample_x, sample_y, held=None):
    """What the rule reads off each layer, in the public order of the layers.

    ``None`` where the layer is uniform -- it keeps the Fourier sum, bit for bit, since there is
    nothing for the rule to change there (:func:`_field_rows` on a derivative). Otherwise a
    :data:`LayerRule`, whose ``magnetic`` half is the ``electric`` half with ``mu`` for ``eps``.
    ``mu`` is a raster's permeability field; a shape names its own. Each half is, on the 1D paths,
    ``eps`` at the x samples ``(n_x,)``; on the 2D path ``(ops_x, ops_y, eps_x, eps_y)``: ``ops_x[j]`` is Li's inverse rule
    ``[[1/eps]]_x^-1`` along the row under the ``j``-th y sample, ``ops_y[i]`` the same along the
    column under the ``i``-th x sample, and ``eps_x`` and ``eps_y`` the ``(n_y, n_x)`` permittivity at
    the samples, one array on a scalar layer and a diagonal tensor's two entries on another
    (:func:`wavebit.rcwa._engine.fields._lalanne_2d`). A row is a raster's row of pixels, or on
    shapes the line through the sample. ``held``, where given, is the layers to read
    (:func:`layers_holding`), and the others are ``None``.
    """
    half_u = (kx.shape[0] - 1) // 2
    half_v = (ky.shape[0] - 1) // 2 if two_d else 0
    if isinstance(structure, list):
        return _shape_rows(xp, structure, lattice, half_u, half_v, real, C, _device_of(kx),
                           sample_x, sample_y, held)
    return _raster_rows(xp, structure, mu, lattice, half_u, half_v, real, two_d, sample_x, sample_y, held)
