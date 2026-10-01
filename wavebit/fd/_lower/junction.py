"""A flush pillar on a smoothed Box face: static membership, live material values.

The split rule is the study's arm 2. Edges read the wall's gated moment;
common nodes read the nearest transverse wall normal. The lowerer writes after all
ordinary painters, using the ground captured before the first pair member.
"""
import dataclasses
import hashlib
import json
import math

import numpy as np

SPLIT, STAIRCASE, REFUSE = 'split', 'staircase', 'refuse'
DEFAULT = SPLIT
RULES = (SPLIT, STAIRCASE, REFUSE)
TRANSVERSE = ((1, 2), (2, 0), (0, 1))
PAIRS = ((0, 1), (0, 2), (1, 2))


@dataclasses.dataclass(frozen=True)
class Junction:
    face: int
    standing: int
    axis: int
    side: int
    plane: float
    rule: str
    rows: tuple
    counts: tuple
    source: str
    media: tuple
    write: object = dataclasses.field(default=None, compare=False, metadata={'digest': True})
    inside: object = dataclasses.field(default=None, compare=False, metadata={'digest': True})
    owner: object = dataclasses.field(default=None, compare=False, metadata={'digest': True})


def _fractional(value):
    from .checks import _CLIP_ROUNDOFF
    return (value > _CLIP_ROUNDOFF) & (value < 1 - _CLIP_ROUNDOFF)


def _view(box, origin=None):
    origin = origin or ((0, 0),) * 3
    return tuple(slice(lo - base[0], hi - base[0])
                 for (lo, hi), base in zip(reversed(box), reversed(origin)))


def _points(grid, box, field, component):
    from .occupancy import yee_points
    from .constitutive import common_points
    return (yee_points(grid, box, field, component) if component < 3
            else common_points(grid, box, field))


def _section_body(shape, axis):
    from ..geometry import Box
    from .._shapes import Box as BodyBox
    if type(shape) is Box:
        return BodyBox(shape.size, center=shape.center, axis='xyz'[axis])
    return shape.body()


def _section(body, points, dx):
    _, _, box, _ = body._boxes(points, dx)
    return body._area(box, points) / dx**2


def _wall(body, points, dx, xp, node=False):
    from .._shapes import Cylinder
    from .._shapes._cut import _parts, _LIVE_AREA
    from .._shapes._occupancy import _mean_of_the_moment
    from .._shapes._exact import polygon_edge_moments
    _, (u, v, axis), q, _ = body._split(points)
    if node:
        _, normal = body._transverse_surface(q)
        zero = xp.zeros_like(normal[..., 0])
        live = _fractional(_section(body, points, dx))
        # Nodes use the nearest wall, including an unturned Box reached by
        # another body's term. Exact Box edge moments here missed its alpha=0
        # limit by .1005220005782 at four rim corners (2026-09-28 probe).
        cross = normal[..., 0]*normal[..., 1] if isinstance(body, Cylinder) else zero
        return tuple(xp.where(live, w, zero) for w in
                     (normal[..., 0]**2, normal[..., 1]**2, cross))
    if isinstance(body, Cylinder):
        distance, normal = body._transverse_surface(q)
        zero = xp.zeros_like(distance)
        plane = xp.stack([normal[..., 0], normal[..., 1], zero], axis=-1)
        _, area, moment = _parts(distance, plane, dx)
        weight = _mean_of_the_moment(area, moment, dx, xp)
        live = area > _LIVE_AREA * dx**2
        return weight[..., 0], weight[..., 1], xp.where(
            live, normal[..., 0]*normal[..., 1], zero)
    _, _, box, _ = body._boxes(points, dx)
    length, muu, mvv = polygon_edge_moments(body._vertices(points), *box, xp)
    zero = xp.zeros_like(muu)
    moment = xp.stack([muu*dx, mvv*dx, zero], axis=-1)
    weight = _mean_of_the_moment(length*dx, moment, dx, xp)
    return weight[..., 0], weight[..., 1], zero


def values(face, standing, axis, points, dx, eps_s, eps_p, eps_g, xp, node=False):
    """Live split tensor and its arithmetic inverse floor; no host reads of values."""
    from .._shapes import occupancy, Cylinder
    from .coefficients import mix
    alpha = occupancy(face, points, dx)[0]
    fu = _section(standing, points, dx)
    puu, pvv, puv = _wall(standing, points, dx, xp, node=node)
    up = fu*eps_p + (1-fu)*eps_g
    eta = fu/eps_p + (1-fu)/eps_g
    arithmetic = (1-alpha)*up + alpha*eps_s
    normal_mean = (1-alpha)/eta + alpha*eps_s
    if isinstance(standing, Cylinder):
        uu = (1-puu)/arithmetic + puu/normal_mean
        vv = (1-pvv)/arithmetic + pvv/normal_mean
        uv = (1/normal_mean - 1/arithmetic)*puv
    else:
        uu = 1/((1-alpha)*mix(xp, eps_g, eps_p, fu, puu) + alpha*eps_s)
        vv = 1/((1-alpha)*mix(xp, eps_g, eps_p, fu, pvv) + alpha*eps_s)
        uv = xp.zeros_like(alpha)
    aa = (1-alpha)/up + alpha/eps_s
    u, v = TRANSVERSE[axis]
    diagonal = [None]*3
    diagonal[u], diagonal[v], diagonal[axis] = uu, vv, aa
    cross = [xp.zeros_like(alpha) for _ in range(3)]
    cross[PAIRS.index(tuple(sorted((u, v))))] = uv
    return xp.stack(diagonal), xp.stack(cross), 1/arithmetic


def _roles(scene, grid, ia, ib):
    from ..geometry import Box, Cylinder, Sphere, Prism
    from .checks import _CLIP_ROUNDOFF
    shapes = [s.shape for s in scene.structures]
    sa, sb = shapes[ia], shapes[ib]
    box = lambda s: type(s) is Box and s.angle == 0
    if not box(sa) and not box(sb):
        return 'neither is an unturned fd.Box'
    if type(sa) is Sphere or type(sb) is Sphere:
        return 'a sphere touches a face at a point'
    if any(type(s) is Prism or (type(s) is Box and s.angle != 0) for s in (sa, sb)):
        return 'a turned fd.Box or an fd.Prism standing on a face is not covered yet'
    candidates = []
    for f, s in ((ia, ib), (ib, ia)):
        face, pillar = shapes[f], shapes[s]
        if not box(face) or not (box(pillar) or type(pillar) is Cylinder):
            continue
        for axis in range(3):
            if type(pillar) is Cylinder and pillar.axis != 'xyz'[axis]:
                continue
            u, v = TRANSVERSE[axis]
            if any(min(face.high[b], pillar.high[b]) <= max(face.low[b], pillar.low[b])
                   for b in (u, v)):
                continue
            for side in (1, -1):
                plane = face.high[axis] if side == 1 else face.low[axis]
                base = pillar.low[axis] if side == 1 else pillar.high[axis]
                if abs(plane-base) <= _CLIP_ROUNDOFF*grid.dx:
                    # A wide superstrate owns a hanging Box, not vice versa.
                    # Only if both faces cover the section does the lower Box
                    # own a film seam; role choice never reads listing order.
                    covers = all(face.low[b] <= max(pillar.low[b], grid.low[b]-grid.dx/2)
                                 and face.high[b] >= min(pillar.high[b], grid.high[b]+grid.dx/2)
                                 for b in (u, v))
                    candidates.append((not covers, side != 1, axis, plane, f, s, side))
    if not candidates:
        if any(type(s) is Cylinder for s in (sa, sb)):
            return 'does not stand flush on the face, or its axis is not normal to the face'
        return 'does not stand flush on the face'
    _, _, axis, plane, face, standing, side = min(candidates)
    return face, standing, axis, side, plane


def _signature(scene, grid):
    geometry = [(type(s.shape).__module__, type(s.shape).__qualname__,
                 s.shape.describe(), getattr(s.shape, 'axis', None),
                 s.shape.support(grid), s.junction, s.discretisation)
                for s in scene.structures]
    return hashlib.sha256(json.dumps((geometry, grid.n, grid.dx, grid.low),
                                    sort_keys=True).encode()).hexdigest()


def _material_scope(material, where):
    from ..materials import Material, concrete
    if not isinstance(material, Material) or material.poles:
        raise ValueError(f'{where}: junctions require lossless, non-dispersive media.')
    # is_magnetic means a named mu, even mu=1; keep that static contract under tracing.
    if material.is_magnetic:
        raise ValueError(f'{where}: the junction has a magnetic medium; only electric '
                         'junctions have been measured.')
    eps = material.eps
    if not (eps[0] is eps[1] is eps[2]):
        concrete_eps = [concrete(v) for v in eps]
        if any(v is None for v in concrete_eps) or len(set(concrete_eps)) != 1:
            raise ValueError(f'{where}: junctions require isotropic eps, supplied as a scalar.')


def _share(shape, points, grid):
    from .._shapes import occupancy
    body = shape.body()
    if body is not None:
        return np.asarray(occupancy(body, points, grid.dx)[0])
    cells = shape.filled_cells(grid)
    share = np.ones(points.shape[:-1])
    for a, (lo, hi) in enumerate(cells):
        low = -np.inf if lo == 0 else grid.low[a] + lo*grid.dx
        high = np.inf if hi == grid.n[a] else grid.low[a] + hi*grid.dx
        share *= np.maximum(0, np.minimum(points[..., a]+grid.dx/2, high)
                            - np.maximum(points[..., a]-grid.dx/2, low))/grid.dx
    return share


def classify(scene, grid, ia, ib, count):
    """A record, or an explanatory string retaining the original shared-surface refusal."""
    from ..geometry import Box, Cylinder
    from .._shapes import occupancy
    from .checks import _CLIP_ROUNDOFF, _shared_surface_voxels
    from .coefficients import materials_of
    roles = _roles(scene, grid, ia, ib)
    if isinstance(roles, str):
        return roles
    f, s, axis, side, plane = roles
    face, pillar = scene.structures[f], scene.structures[s]
    rule = pillar.junction
    if rule == REFUSE:
        return (f"structures[{s}] asked for junction='refuse'; leave junction= out "
                "(default 'split') to lower this flush pair")
    kp = math.floor((plane-grid.low[axis])/grid.dx)
    rows = list(pillar.shape.support(grid))
    # Half-voxel overlap plus a cylinder base-side reach < 1.21 cells. Five
    # layers include both signs and both staggerings, without a fitted margin.
    rows[axis] = (max(0, kp-2), min(grid.n[axis], kp+3))
    rows = tuple(rows)
    fb = face.shape.body()
    pb = pillar.shape.body()
    section = _section_body(pillar.shape, axis)
    total, _, marks = _shared_surface_voxels(fb, pb, grid, rows,
                                           include_common=True, masks=True)
    if total != count:
        return 'ends within a voxel of the face, or shares another surface away from that face'
    tol = _CLIP_ROUNDOFF
    reach = max(abs(v) for v in tuple(grid.low)+tuple(grid.high))/grid.dx
    slack = max(tol, 64*np.finfo(float).eps*reach)
    writes, insides, owners, counts = [], [], [], []
    background, structures = materials_of(scene)
    media = {f, s}
    _material_scope(background, 'Scene(background=)')
    for index in media:
        _material_scope(structures[index].material, f'structures[{index}]')
    for field in ('E', 'H'):
        for c in range(4):
            points = _points(grid, rows, field, c)
            alpha = np.asarray(occupancy(fb, points, grid.dx)[0])
            fill = np.asarray(occupancy(pb, points, grid.dx)[0])
            fu = np.asarray(_section(section, points, grid.dx))
            inside = marks[field, c]
            normal = np.asarray(pb.surface(points)[1])
            if type(pillar.shape) is Cylinder:
                base = side*normal[..., axis] < 0
            else:
                # Read the ORIGINAL body's exact moment, including polygon sides
                # when the face normal differs from the Box's extrusion axis.
                base = np.asarray(pb.exact_moment(points, grid.dx)[1])[..., axis] > 0
                # Reconstructing the public face and the body's cap can differ
                # by round-off. The exact moment above decides cap inclusion;
                # this distance check only excludes the far cap. Reuse the
                # flush-plane tolerance, including a half-cell tie at 0.5 dx.
                base &= side*(points[..., axis]-plane) <= (.5+tol)*grid.dx
            base &= ((side*(points[..., axis]-plane) > 0) & (alpha <= tol)
                     & _fractional(fill) & ~inside)
            write = inside | base
            section_face = np.ones(alpha.shape)
            for b in TRANSVERSE[axis]:
                section_face *= np.maximum(0, np.minimum(points[..., b]+grid.dx/2,
                                                          face.shape.high[b])
                                           - np.maximum(points[..., b]-grid.dx/2,
                                                        face.shape.low[b]))/grid.dx
            if np.any(write & (section_face < 1-slack)):
                return f'its rim meets the edge of structures[{f}]'
            expected_alpha = np.clip(.5-side*(points[..., axis]-plane)/grid.dx, 0, 1)
            if (np.any(write & (np.abs(alpha-expected_alpha) > slack))
                    or np.any(write & (np.abs(fill-(1-alpha)*fu) > slack))):
                return 'ends within a voxel of the face'
            for index, other in enumerate(structures):
                if index in (f, s) or not write.any():
                    continue
                share = _share(other.shape, points[write], grid)
                if not np.any(share > tol):
                    continue
                if index > min(f, s):
                    return f'structures[{index}] is listed after the first pair member and reaches the junction'
                if np.any(_fractional(share)):
                    return f'structures[{index}] leaves a fractional ground at the junction'
                media.add(index)
                _material_scope(other.material, f'structures[{index}]')
            counts.append((field+'xyz*'[c], int(inside.sum()),
                           int((inside & _fractional(fu)).sum()), int(base.sum())))
            if field == 'E':
                owner = np.where(side*(points[..., axis]-plane) < -tol*grid.dx, 0,
                                 np.where(np.asarray(pb.surface(points)[0]) <= 0, 1, 2))
                writes.append(write)
                insides.append(inside)
                owners.append(owner.astype(np.int8))
    arrays = [np.stack(v) for v in (writes, insides, owners)]
    for array in arrays:
        array.flags.writeable = False
    return Junction(f, s, axis, side, plane, rule, rows, tuple(counts),
                    _signature(scene, grid), tuple(sorted(media)), *arrays)


def finish(scene, records):
    from .constitutive import _intersect
    for i, record in enumerate(records):
        for other in records[:i]:
            box = _intersect(record.rows, other.rows)
            if box is not None and np.any(record.write[(slice(None),)+_view(box, record.rows)]
                                          & other.write[(slice(None),)+_view(box, other.rows)]):
                raise ValueError('JunctionOverlapRefused: two junction write sets share a point.')
    read = {r.standing for r in records}
    for i, structure in enumerate(scene.structures):
        if structure.junction != DEFAULT and i not in read:
            standing = [r.standing for r in records if r.face == i]
            hint = (f'; set it on structures[{standing[0]}], which stands on its face'
                    if standing else '')
            raise ValueError(f'structures[{i}] asks for junction={structure.junction!r} but '
                             f'stands on no smoothed face in this scene{hint}.')
    return tuple(sorted(records, key=lambda r: (r.standing, r.face)))


def refuse_mismatch(scene, plan):
    """The scene's flush pairs and rule are the plan's, checked in both directions.

    A body whose geometry is traced (a ranged body under differentiation) is certified by
    ``shape_ranges.planning_scene`` to keep its whole interpolation halo off every other
    structure, so it is never one of a flush pair, and its pairs are not asked: ``_roles``
    compares faces on the host, which a ``jax.jit`` tracer refuses. A flush junction between
    concrete partners beside a traced body is refused outright: its signature covers all
    scene geometry, and this certificate has not been extended to it.
    """
    from .coefficients import materials_of
    from .shape_ranges import leaves
    from ..geometry import _shape_traced
    moving = {i for i, s in enumerate(scene.structures)
              if any(_shape_traced(v) for v in leaves(s.shape))}
    records = plan.junctions
    if records and moving:
        raise ValueError(f'structures{sorted(moving)} carry traced geometry beside a flush '
                         'junction between concrete partners; that combination is not '
                         'certified. Differentiate in a scene without the junction, or keep '
                         'the junction scene concrete.')
    if records:
        signature = _signature(scene, plan.grid)
        if any(r.source != signature for r in records):
            raise ValueError('JunctionMismatchRefused: geometry or junction rule differs from '
                             'the plan; rebuild it with fd.plan(scene).')
        background, structures = materials_of(scene)
        _material_scope(background, 'Scene(background=)')
        for index in {i for r in records for i in r.media}:
            _material_scope(structures[index].material, f'structures[{index}]')
    recorded = {frozenset((r.face, r.standing)) for r in records}
    for i, s in enumerate(scene.structures):
        if s.junction != DEFAULT and i not in {r.standing for r in records}:
            raise ValueError('JunctionMismatchRefused: the requested junction rule is absent '
                             'from the plan; rebuild it with fd.plan(scene).')
        for j in range(i):
            if frozenset((i, j)) in recorded or i in moving or j in moving:
                continue
            if not isinstance(_roles(scene, plan.grid, i, j), str):
                raise ValueError('JunctionMismatchRefused: a flush pair is absent from the '
                                 'plan; rebuild it with fd.plan(scene).')


def tables(rt, record, scene, points, ground, inside, owner, node=False):
    xp = rt.xp
    face, pillar = scene.structures[record.face], scene.structures[record.standing]
    es = rt.array(face.material.eps[0], 'float64')
    ep = rt.array(pillar.material.eps[0], 'float64')
    result = values(face.shape.body(), _section_body(pillar.shape, record.axis),
                    record.axis, points, scene.resolution.dx, es, ep, ground, xp, node=node)
    if record.rule != STAIRCASE:
        return result
    diagonal, cross, inverse = result
    own = rt.array(np.array(owner, copy=True))
    selected = xp.where(own == 0, es, xp.where(own == 1, ep, ground))
    mask = rt.array(np.array(inside, copy=True))
    return (xp.where(mask[None], 1/selected, diagonal),
            xp.where(mask[None], xp.zeros_like(cross), cross),
            xp.where(mask, 1/selected, inverse))


def paint(rt, record, scene, plan, out, ground, component):
    from .coefficients import _paste
    xp, box = rt.xp, record.rows
    view = _view(box)
    shape = (plan.nz, plan.ny, plan.nx)
    full = xp.broadcast_to(out, shape)
    ground = xp.broadcast_to(ground, shape)[view]
    points = rt.array(_points(plan.grid, box, 'E', component), 'float64')
    diagonal, _, _ = tables(rt, record, scene, points, ground,
                             record.inside[component], record.owner[component])
    write = rt.array(np.array(record.write[component], copy=True))
    return _paste(xp, full, box, xp.where(write, 1/diagonal[component], full[view]))


def paint_nodes(rt, record, scene, plan, term, tensor, cross, held, ground):
    from .coefficients import _paste
    from .constitutive import _intersect
    box = tuple(reversed(term.box))
    cut = _intersect(box, record.rows)
    if cut is None or term.field != 'E':
        return tensor, cross, held
    record_view, view = _view(cut, record.rows), _view(cut, box)
    mask = record.write[(3,)+record_view]
    if not mask.any():
        return tensor, cross, held
    points = rt.array(_points(plan.grid, cut, 'E', 3), 'float64')
    diagonal, off, inverse = tables(rt, record, scene, points, ground[view],
        record.inside[(3,)+record_view], record.owner[(3,)+record_view], node=True)
    local = tuple((lo-origin[0], hi-origin[0]) for (lo, hi), origin in zip(cut, box))
    write = rt.array(np.array(mask, copy=True))
    tensor = _paste(rt.xp, tensor, local,
                    rt.xp.where(write[None], diagonal, tensor[(slice(None),)+view]))
    cross = _paste(rt.xp, cross, local,
                   rt.xp.where(write[None], off, cross[(slice(None),)+view]))
    held = _paste(rt.xp, held, local, rt.xp.where(write, 1/inverse, held[view]))
    return tensor, cross, held


def report_rows(plan):
    """Only electric families are written; H counts certify geometry, not work."""
    return tuple((r.standing, r.face, r.axis, r.side, r.plane, r.rule,
                  *(sum(row[k] for row in r.counts if row[0].startswith('E'))
                    for k in (1, 2, 3))) for r in plan.junctions)


def proxy_mask(plan, index, box):
    """Host-only report mask: a junction node is no longer the single-body proxy."""
    from .constitutive import _intersect
    keep = np.ones(tuple(hi-lo for lo, hi in reversed(box)), dtype=bool)
    for record in plan.junctions:
        if index not in (record.face, record.standing):
            continue
        cut = _intersect(box, record.rows)
        if cut is not None:
            keep[_view(cut, box)] &= ~record.write[(3,)+_view(cut, record.rows)]
    return keep


def notes(scene, plan):
    from ..diagnostics import SubpixelAccuracyWarning
    if not plan.junctions:
        return []
    lines = []
    for standing, face, axis, side, plane, rule, count, rim, beside in report_rows(plan):
        text = (f'structures[{standing}] stands flush on structures[{face}] at '
                f'{"xyz"[axis]} = {plane:g}: {count} electric junction voxels '
                f'({rim} on the rim), {beside} beside the base, take junction={rule!r}. ')
        if rule == SPLIT:
            text += ('The substrate is layered under the pillar/background mixture. '
                     'Measured second order on a flat film (16 cases, orders 2.00-2.03) '
                     'and consistent with RCWA on six pillar scenes. The rim error has '
                     'an unresolved order on one scene (eps 5.8). ')
        else:
            text += ('Each junction point takes its owning medium. Its junction term is '
                     'first order (0.98-1.02 on the flat film); coarse-grid cancellation '
                     "can make its total error look smaller. 'split' is the default. ")
        lines.append(text + 'These measurements are not an error bound for this scene; '
                     'see docs/fd-subpixel-occupancy.md, The junction.')
    return [(SubpixelAccuracyWarning, ' '.join(lines))]
