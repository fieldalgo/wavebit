"""Where a line through a layer of shapes crosses their walls, for the continuity rule.

The rule (P. Lalanne and M. P. Jurek, J. Mod. Opt. 45, 1357 (1998)) needs two things along each
sample line: Li's inverse rule of that line (L. Li, J. Opt. Soc. Am. A 13, 1870 (1996)), and eps
at the samples on it (:func:`wavebit.rcwa._engine.fields._continuity_2d`). On a raster the line is
a row of pixels. Here it is the shapes' own outlines, so a wall sits where the shape says, between
any two samples, and the coefficients are those of the exact staircase along the line
(:func:`~wavebit.rcwa._engine.factorization._segment_series`).

Everything is in the corner frame the outlines are in, with ``p`` along the line and ``q`` across
it. Nothing here is planned ahead, so it replays under a trace without a frozen record: the
crossings are arithmetic on the live vertices, and sorting them and counting windings only choose
which of those numbers go where. A position therefore keeps its derivative. A wall crossing a
sample is the one thing with none: ``eps`` at a sample is locally constant in the shape.

**Inside a shape** is a nonzero winding number, counted along the line from the crossings' signs
(K. Hormann and A. Agathos, Comput. Geom. 20, 131 (2001)), so a clockwise outline and an outline
read with its axes swapped need no code of their own. Which shape holds a point is the last one
listed that encloses it, as on the painting route.

**Only walls along x and y are admitted** (:func:`slanted_wall`), so one position per edge is
enough: such a wall meets every image of a line at the same ``p``, and how many images it meets is
a count on its winding step (:func:`edge_crossings`). The count matters though
:func:`~wavebit.rcwa._engine.parametric.check_layer_shapes` refuses a shape running into its own
image, because that check has a tolerance: a ridge one period tall, written with its height rounded
up, passes it with an edge longer than a period. A slanted edge longer than a period passes it too,
and would need a position per image.
"""
from collections import namedtuple

import numpy as np

from wavebit._common.backend import _device_of
from wavebit._common.backend import as_array
from wavebit._common.backend import carries_autodiff
from wavebit._common.geometry.polygon import outline_values
from wavebit._common.geometry.recipes import is_polygon
from wavebit._common.geometry.recipes import shape_geometry
from wavebit.materials.response import as_material
from wavebit.rcwa._engine.boundary import shape_vertices
from wavebit.rcwa._engine.factorization import _inverse_of_series
from wavebit.rcwa._engine.factorization import _segment_series
from wavebit.rcwa._engine.field_result import _host_value

#: A wall is along an axis when its slant is under this: ``|sin 2 angle|`` of a rectangle, which
#: holds the rounding ``angle=pi/2`` leaves, and a polygon's lesser step along an edge against its
#: longest step anywhere.
SLANT_TOL = 1e-12

#: ``(n_lines, n_slots)`` crossings of every shape on a layer's lines. ``unit`` is the position
#: in the period, ``[0, 1)``, ``turns`` the whole periods taken off it, ``sign`` the winding step
#: and ``owner`` the slot's shape. An unused slot has ``sign`` 0.
LinePaint = namedtuple('LinePaint', 'unit turns sign owner eps background')


def slanted_wall(shape, lattice):
    """Is any wall of this shape neither along x nor along y? A value it cannot read says yes.

    A traced angle or vertex (under ``jax.jit``, ``jax.vmap`` or a ``torch.func`` transform) cannot
    show a wall's direction, and a wall in an unknown direction is not one this module may answer
    for. A differentiated angle is refused even with a value to read: it turns every wall, so its
    derivative leaves the shapes this answers for. A differentiated vertex is read by value, since
    moving both ends of a wall together is what an outline's derivative is for.
    """
    if is_polygon(shape):
        try:
            vertices = outline_values(shape['vertices'])
        except TypeError:
            return True
        steps = np.abs(np.roll(vertices, -1, axis=0) - vertices)
        return bool(np.any(np.min(steps, axis=1) > SLANT_TOL * np.max(steps)))
    kind, *_, angle = shape_geometry(shape, lattice)
    if kind in ('circle', 'disk', 'ellipse'):
        return True
    value = _host_value(angle)
    return value is None or carries_autodiff(angle) or abs(np.sin(2.0 * float(value))) > SLANT_TOL


def edge_crossings(xp, p, q, line, period_q):
    """Where each edge of an outline crosses each line, once. ``at``, ``sign``: ``(n_lines, n_v)``.

    Each vertex gets one count, the images of the line below it, and an edge's crossings are the
    difference of its two ends' counts. So the steps around a closed outline sum to zero whatever
    the rounding, and two vertices at one height see any line alike: the outline cannot leave a
    line with one wall read and its partner missed, which folding each edge on its own did within
    a few units of a vertex image, across a whole row.

    The line is folded once, to the outline's lowest vertex, and a height within rounding of a
    whole number of periods is that number: otherwise a ridge a period tall that came out one unit
    short has a gap of that unit, and a line in the gap loses the whole shape.

    A vertex is below an image at its own height, so an edge holds its lower end and not its upper
    one, and a line along an edge meets that edge nowhere. ``sign`` is the winding step times the
    images of the line the edge meets, and ``at`` where it meets the first of them at or above its
    lower end, which is where it meets them all on a wall along an axis.
    """
    p_next, q_next = xp.roll(p, -1, axis=0), xp.roll(q, -1, axis=0)
    rise = q_next - q
    lowest = xp.min(q)
    above = line - lowest
    above = (above - period_q * xp.floor(above / period_q))[:, None]
    height = q - lowest
    whole = xp.round(height / period_q) * period_q
    height = xp.where(xp.abs(height - whole) <= 8.0 * xp.finfo(q.dtype).eps * period_q, whole, height)
    below = xp.ceil((height[None, :] - above) / period_q)
    below_next = xp.roll(below, -1, axis=1)
    steps = below_next - below
    first = lowest + above + xp.minimum(below, below_next) * period_q
    slope = (p_next - p) / xp.where(rise != 0, rise, xp.ones_like(rise))
    at = p[None, :] + (first - q[None, :]) * slope[None, :]
    return xp.where(steps != 0, at, xp.zeros_like(at)), -steps


def read_lines(xp, layer, lattice, axis, line, period, period_q, C):
    """A layer's crossings on its lines: along x for ``axis=0``, along y for ``axis=1``."""
    dev = _device_of(line)
    at, sign, owner, eps = [], [], [], []
    for index, shape in enumerate(layer['shapes']):
        vertices = as_array(xp, shape_vertices(xp, shape, lattice), dtype=line.dtype, device=dev)
        crossed, step = edge_crossings(xp, vertices[:, axis], vertices[:, 1 - axis], line, period_q)
        at.append(crossed)
        sign.append(step)
        owner.extend([index] * crossed.shape[1])
        eps.append(as_array(xp, as_material(shape.get('material', 1.0)).epsilon, dtype=C, device=dev))
    at = xp.concat(at, axis=1) / period
    turns = xp.floor(at)
    return LinePaint(at - turns, turns, xp.concat(sign, axis=1), np.asarray(owner), eps,
                     as_array(xp, as_material(layer.get('background', 1.0)).epsilon,
                              dtype=C, device=dev))


def _windings(xp, paint, index):
    """Shape ``index``'s slots and the winding number it has at the start of the period.

    The line and its images repeat, so a crossing ``turns`` periods along counts ``turns`` fewer
    times by the time the line reaches any point of this period.
    """
    mine = as_array(xp, paint.owner == index, device=_device_of(paint.unit))[None, :]
    start = -xp.sum(xp.where(mine, paint.sign * paint.turns, xp.zeros_like(paint.sign)),
                    axis=1, keepdims=True)
    return mine, start


def segments(xp, paint):
    """The line as a staircase of ``1/eps``: ``left``, ``width``, ``after``, each ``(n_lines, m)``."""
    order = xp.argsort(paint.unit, axis=1, stable=True)
    left = xp.take_along_axis(paint.unit, order, axis=1)
    sign = xp.take_along_axis(paint.sign, order, axis=1)
    after = xp.ones(left.shape, dtype=paint.background.dtype, device=_device_of(left)) / paint.background
    for index, eps in enumerate(paint.eps):
        mine, start = _windings(xp, paint, index)
        mine = xp.take_along_axis(xp.broadcast_to(mine, order.shape), order, axis=1)
        winding = start + xp.cumulative_sum(xp.where(mine, sign, xp.zeros_like(sign)), axis=1)
        after = xp.where(winding != 0, 1.0 / eps, after)
    width = xp.concat([left[:, 1:], left[:, :1] + 1.0], axis=1) - left
    return left, width, after


def line_operators(xp, paint, half_width):
    """Li's inverse rule of each line, ``(n_lines, n, n)`` with ``n = 2*half_width + 1``."""
    left, width, after = segments(xp, paint)
    return _inverse_of_series(xp, _segment_series(xp, left, width, half_width)(after), half_width)


def eps_at(xp, paint, unit_samples):
    """The permittivity at each sample of each line, ``(n_lines, n_samples)``.

    ``unit_samples`` is where the samples are in the period, ``[0, 1)``. A sample exactly on a
    crossing is past it, on the crossing's positive side.
    """
    passed = paint.unit[:, :, None] <= unit_samples[None, None, :]
    sign = xp.where(passed, paint.sign[:, :, None], xp.zeros_like(paint.sign[:, :, None]))
    out = xp.ones((paint.unit.shape[0], unit_samples.shape[0]), dtype=paint.background.dtype,
                  device=_device_of(paint.unit)) * paint.background
    for index, eps in enumerate(paint.eps):
        mine, start = _windings(xp, paint, index)
        counted = xp.where(mine[:, :, None], sign, xp.zeros_like(sign))
        out = xp.where(start + xp.sum(counted, axis=1) != 0, eps, out)
    return out
