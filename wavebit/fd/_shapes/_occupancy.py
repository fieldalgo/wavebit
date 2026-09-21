"""The whole answer for a body: the periodic image ring, then the pair, per point.

``occupancy(body, points, dx, period=...)`` is what C2's lowering stage will call once per Yee
component. That stage is named after this function because it is the same question at the other
end of the wire, not a second one.
"""
import numpy as _np

from wavebit._core.backend import array_namespace

from ._bodies import Approximations, Body, _as, check_sizes
from ._cut import _mean_of_the_moment


def _checked_points(points):
    """Refuse a ``points`` array that is not ``(..., 3)`` real, by name.

    Three spellings reached the kernel before this door and each came back as an incidental
    exception from somewhere inside it: ``(N, 2)`` as ``IndexError: index 2 is out of bounds``,
    ``(N, 4)`` as a broadcast ``ValueError``, and a complex array as ``TypeError: ufunc
    'arctan2' not supported``. The shape is read from ``.shape`` and the dtype from ``str()``,
    so both are static and neither costs a device synchronisation -- the check is free under
    ``jax.jit`` as well as outside it.
    """
    shape = getattr(points, 'shape', None)
    if shape is None or len(shape) < 1 or shape[-1] != 3:
        raise ValueError(
            f'occupancy(points=...): one point per voxel centre, x first, shaped (..., 3); got '
            f'{"a value with no shape" if shape is None else shape}. A body answers for a POINT '
            'and the last axis is that point\'s three coordinates.')
    if 'complex' in str(getattr(points, 'dtype', '')):
        raise ValueError(
            f'occupancy(points=...): a real coordinate, and these are complex '
            f'({points.dtype}). A voxel centre is a place in the scene; the same refusal is on '
            'every geometry value this package takes, for the same reason -- a check reads the '
            'real part and the kernel uses the whole number.')


def _checked_period(where, period):
    """Refuse a period that is not a positive length, by name, axis by axis.

    ``period`` is the other length this package takes and the ``dx`` door's round left it open.
    Measured before it, ``Sphere(0.2)`` at its own centre with ``dx = 0.1``: ``period=(1, 1, 1)``
    reads ``fill = 1.0`` and ``period=(0, None, None)`` reads **3.0000** -- the three images of
    the ring stacked on the same point, so the fill is exactly tripled, silently, with
    ``sum_c w_c = 1`` in every voxel. ``period=(1e-300, ...)`` reads 3.0000 too, and a negative
    period was accepted and ignored.
    """
    if len(period) != 3:
        raise ValueError(f'{where}(period=...): one entry per axis, x first, each a length or '
                         f'None; got {period!r}.')
    named = {f'period[{axis}]': p for axis, p in zip('xyz', period) if p is not None}
    if named:
        check_sizes(where, **named)


def _checked_ring(rings, period):
    """Refuse a ring count that is not a non-negative whole number, by name.

    ``rings`` sizes a **static** list, so it is read as a Python integer and there is never a
    traced value to defer to. Before this door ``rings=(-1, 0, 0)`` raised a bare ``TypeError``
    from inside ``range`` and ``rings=(1.5, 0, 0)`` another one; the repository's rule is that a
    typo is refused on every door and by name.
    """
    if rings is None:
        return tuple(0 if p is None else 1 for p in period)
    if len(rings) != 3:
        raise ValueError(f'occupancy(rings=...): one count per axis, x first; got {rings!r}.')
    out = []
    for axis, ring in zip('xyz', rings):
        if isinstance(ring, bool) or not isinstance(ring, (int, _np.integer)) or ring < 0:
            raise ValueError(
                f'occupancy(rings=...): how many images to look each way along {axis} -- a '
                f'whole number and not negative; got {ring!r}. rings=0 is the pruning a caller '
                'uses when it knows the body cannot reach the neighbour, and it sizes a static '
                'list, so a fraction and a negative count have nothing to mean.')
        out.append(int(ring))
    return tuple(out)


def image_shifts(period, rings=None):
    """The static list of image offsets, ``(dx, dy, dz)``, x outer and z inner.

    ``period[k]`` is the period of a ``Periodic``/``Bloch`` axis and ``None`` for an axis that
    is not one. ``rings[k]`` is how far along that axis to look, defaulting to one image each
    way on every periodic axis -- the legacy engine's rule
    (``wavebit/_core/scene_fdtd.py::_image_shifts`` :407), which is exact while a shape's
    projected extent is smaller than its period and is refused rather than extended when it is
    not. A caller that knows a body's declared ``range=`` cannot reach the neighbour passes 0
    and pays for one evaluation instead of three; that is the whole of the pruning, and it is
    a **static** decision so a traced radius cannot change how many terms are summed.

    The iteration order is fixed here, x outer and z inner, so that a change of spelling
    elsewhere cannot regroup a floating-point sum -- the same reason the source named above
    fixes its own.
    """
    _checked_period('image_shifts', period)
    rings = _checked_ring(rings, period)
    spans = []
    for p, ring in zip(period, rings):
        if p is None or ring == 0:
            spans.append((0.,))
        else:
            spans.append(tuple(float(m) * float(p) for m in range(-ring, ring + 1)))
    return [(sx, sy, sz) for sx in spans[0] for sy in spans[1] for sz in spans[2]]


def occupancy(body, points, dx, *, period=(None, None, None), rings=None, report=None):
    """``(fill, w)`` for a voxel of side ``dx`` centred on each of ``points``.

    ``points`` is ``(..., 3)``, x-first; the answer is ``fill`` of that shape and ``w`` of
    ``(..., 3)``. Each image contributes its own fill, its own interface area and its own
    ``area * n_c^2``, and only at the end is ``w`` the area-weighted mean -- dividing per image
    would average the averages, and at a seam where two images both reach one voxel the answer
    would be wrong by the ratio of their areas.

    **Nothing is clamped.** The RCWA path ends its fill with ``where(f > 1, 1, f)``
    (``wavebit/_core/scene_fdtd.py``:802) and for a body wider than its own period that is a
    silent lie about the geometry: the shape wraps onto itself and a clamp turns the evidence
    into a plausible number. Measured here, a cylinder of diameter 1.4 periods reads
    ``max fill = 2.0000`` -- two whole images over one voxel -- and an ordinary substrate,
    ``Box((3.0, 3.0, 0.3))`` on ``period=(1, 1, None)``, reads ``max fill = 9.0000`` with
    ``sum_c w_c = 1.0000``, so the invariant does not flag it either. **9, not 2, is the number
    C5's refusal has to be written against**: a layer spanning the cell is the common case and
    the 1.4-period cylinder is the exotic one. A clamp would read 1.0000 for both.

    **Precondition: the body must lie in the cell the period describes.** The ``+/-1`` ring is
    the legacy engine's rule and is exact while the body's projected extent is smaller than its
    period *and the body is in the cell*. It is not periodic in the body's own position beyond
    one period, and the failure is silent. Measured, ``Sphere(0.25)`` at ``(0.11, -0.07, 0.03)``
    with ``period=(1, 1, 1)`` over a 20^3 cell::

        centre + 0 periods in x     V = 0.06610338
        centre + 1 period           V = 0.06610338    worst |dfill| 4.3e-15
        centre + 2 periods          V = 0.00000000    worst |dfill| 1.00
        centre + 2 periods, rings=(2, 1, 1)   V = 0.06610338

    This function is given a period and an array of points and is never told where the cell's
    origin is, so it cannot fold a body it also cannot ask for a position (a CSG node has none).
    C2 owns the cell and therefore owns the fold; what is enforceable here is the statement, and
    ``tests/test_fd_shapes.py::test_the_ring_is_periodic_in_the_body_for_one_period_only`` pins
    both halves of it so the precondition cannot be forgotten silently.

    **``report=`` is how a caller learns what was approximated.** A CSG node answers every voxel
    -- there is no per-voxel refusal in this package -- but not every voxel exactly, and where it
    cannot it brackets the filled fraction and ships the end that is safe for its own operation
    (:class:`_bodies._Combination`). Hand in an :class:`_bodies.Approximations` and it comes back
    holding how many voxels took a bound, how wide the widest bracket was, and which body it was,
    with one entry per node **per periodic image** -- so C2 and C5 can warn with a number instead
    of a solve quietly reading a bound. Leave it out, which is the default, and the two
    reductions are not even computed.

    **Every argument this function takes now has a door, and each one closed a case that used to
    come back as an incidental exception from inside the kernel.** ``period`` is the other length
    (``period=0`` stacked the three images of the ring on the same point and tripled the fill,
    silently, with ``sum_c w_c = 1`` in every voxel); ``rings`` sizes a static list and is a whole
    non-negative number; ``points`` is ``(..., 3)`` and real; ``body`` is a :class:`Body`; and
    ``dx`` is one number, not an array that would broadcast against the points. A bool is refused
    everywhere a length is read, because it passes every test a length has to pass --
    ``occupancy(..., dx=True)`` used to solve at ``dx = 1.0`` and ``Sphere(radius=True)`` was a
    unit sphere.

    **``dx`` is a length and is doored like every other one.** It was the one length in the
    package that was not, and the two failures were character for character the ones
    ``check_sizes`` was written against: at ``dx = -0.05`` a ``Sphere`` read a fill summing to
    136.0 against a true 268.08 with ``sum_c w_c = 0`` in every voxel -- a body half there with
    no interface at all -- and a ``Box`` and a ``Cylinder`` read 0.0000 everywhere, a body that
    silently is not there. Both satisfy this function's advertised contract, so nothing
    downstream could flag them, and on JAX there was not even a RuntimeWarning.
    """
    if report is not None and not isinstance(report, Approximations):
        raise ValueError(
            f'occupancy(report=...): an Approximations, which a CSG node adds two numbers to -- '
            f'how many voxels it answered with a bound and how wide the widest of them was; got '
            f'{type(report).__name__} ({report!r}). Leave it out and nothing is counted.')
    if not isinstance(body, Body):
        raise ValueError(f'occupancy(body=...): a Body -- Sphere, Cylinder, Box, Prism or a CSG '
                         f'node; got {type(body).__name__} ({body!r}).')
    _checked_points(points)
    if getattr(dx, 'ndim', 0) != 0:
        raise ValueError(
            f'occupancy(dx=...): one number -- the voxel\'s side. A non-uniform grid is not in '
            f'v1, and an array here does not mean one: it broadcasts against the POINTS and '
            f'silently reshapes the answer. Got shape {dx.shape}.')
    dx, = check_sizes('occupancy', dx=dx)
    _checked_period('occupancy', period)
    xp = array_namespace(points)
    fill = area = moment = None
    for shift in image_shifts(period, rings):
        p = points if shift == (0., 0., 0.) else points - _as(xp, shift, points)
        one_fill, one_area, one_moment = body.parts(p, dx, report=report)
        fill = one_fill if fill is None else fill + one_fill
        area = one_area if area is None else area + one_area
        moment = one_moment if moment is None else moment + one_moment
    return fill, _mean_of_the_moment(area, moment, dx, xp)
