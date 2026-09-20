"""The whole answer for a body: the periodic image ring, then the pair, per point.

``occupancy(body, points, dx, period=...)`` is what C2's lowering stage will call once per Yee
component. That stage is named after this function because it is the same question at the other
end of the wire, not a second one.
"""
from wavebit._core.backend import array_namespace

from ._bodies import _as


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
    if rings is None:
        rings = tuple(0 if p is None else 1 for p in period)
    spans = []
    for p, ring in zip(period, rings):
        if p is None or ring == 0:
            spans.append((0.,))
        else:
            spans.append(tuple(float(m) * float(p) for m in range(-ring, ring + 1)))
    return [(sx, sy, sz) for sx in spans[0] for sy in spans[1] for sz in spans[2]]


def occupancy(body, points, dx, *, period=(None, None, None), rings=None):
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
    ``max fill = 2.0000`` -- two whole images over one voxel -- and a clamp would read 1.0000.
    C5's refusal is written to fire on the first of those, so this kernel must hand it the first.
    """
    xp = array_namespace(points)
    fill = area = moment = None
    for shift in image_shifts(period, rings):
        p = points if shift == (0., 0., 0.) else points - _as(xp, shift, points)
        one_fill = body.exact_fill(p, dx)
        pair = body.exact_moment(p, dx)
        if one_fill is None or pair is None:
            plane_fill, plane_area, plane_moment = body.plane(p, dx)
            one_fill = plane_fill if one_fill is None else one_fill
            pair = (plane_area, plane_moment) if pair is None else pair
        one_area, one_moment = pair
        fill = one_fill if fill is None else fill + one_fill
        area = one_area if area is None else area + one_area
        moment = one_moment if moment is None else moment + one_moment
    live = area > 0
    safe = xp.where(live, area, xp.ones_like(area))[..., None]
    return fill, xp.where(live[..., None], moment / safe, xp.zeros_like(moment))
