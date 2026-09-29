"""Fixed body allocations and the restricted public geometry-gradient certificate."""
from ..geometry import Sphere, Cylinder, Box, Prism, _shape_traced


def records(scene):
    """Structure index, kind, exact bounds and any static orientation.

    Bounds are recorded even when two ranges snap to the same cells. A plan reserves the
    range the caller declared, not any later range that happens to have the same allocation.
    Indexing also distinguishes moving a range to another painter in an overlapping scene.
    """
    out = []
    for index, structure in enumerate(scene.structures):
        shape = structure.shape
        if isinstance(shape, (Sphere, Cylinder, Box, Prism)) and shape.range is not None:
            row = (index, type(shape).__name__.lower(), shape.range)
            if isinstance(shape, Prism):
                row += (shape.axis, shape.angle, len(shape.vertices))
            elif isinstance(shape, Box):
                row += (shape.axis, shape.angle)
            elif isinstance(shape, Cylinder):
                row += (shape.axis,)
            out.append(row)
    return tuple(out)


def refuse_mismatch(scene, plan):
    """Protect direct ``lower`` callers too; ``run`` additionally checks the whole scene."""
    if records(scene) != plan.shape_ranges:
        raise ValueError('lower(scene, plan=...): Sphere/Cylinder/Box/Prism(range=) declarations differ '
                         'from the plan. Keep each range on the same structure with the same '
                         'bounds, static orientation and vertex count, or build a new plan '
                         'with fd.plan(scene).')


def leaves(shape):
    if isinstance(shape, Prism):
        return (shape.height, *shape.center, *(v for p in shape.vertices for v in p))
    if isinstance(shape, Box):
        return (*shape.size, *shape.center)
    if isinstance(shape, (Sphere, Cylinder)):
        return ((shape.radius, *shape.center)
                + ((shape.height,) if isinstance(shape, Cylinder) else ()))
    return ()


def traced(scene):
    return any(_shape_traced(v) for s in scene.structures for v in leaves(s.shape))


def name(scene):
    return next(type(s.shape).__name__ for s in scene.structures
                if any(_shape_traced(v) for v in leaves(s.shape)))


def remember(scene, plan):
    """The seed holds only concrete numbers, never backend leaves or a captured scene."""
    seeds = tuple((i, s.shape.vertices, s.shape.height, s.shape.center)
                  if isinstance(s.shape, Prism) else
                  (i, s.shape.size, s.shape.center) if isinstance(s.shape, Box) else
                  (i, s.shape.radius, s.shape.center)
                  + ((s.shape.height,) if isinstance(s.shape, Cylinder) else ())
                  for i, s in enumerate(scene.structures)
                  if isinstance(s.shape, (Sphere, Cylinder, Box, Prism)) and s.shape.range is not None)
    if seeds:
        object.__setattr__(plan, '_shape_seeds', seeds)


def planning_scene(scene, plan):
    """A concrete preflight of all unchanged inputs, after certifying the full allocation.

    A seed proves nothing about another geometry. Isolation and the fixed, interior support
    remove overlap, seam and wall cases for the entire range; runtime guards then check the
    actual sizes, containment, star positivity and CFL bound. Every other scene input is
    rechecked on the host. Multiple painters and geometry-dependent sources/monitors need
    their own range certificate and are deliberately outside this first door.
    """
    if not traced(scene):
        return scene
    where = name(scene)
    if plan is None or not getattr(plan, '_shape_seeds', ()):
        raise ValueError(f'A traced {where} requires plan= built by fd.plan(scene) from '
                         'concrete geometry before differentiation.')
    refuse_mismatch(scene, plan)
    from ..materials import Material, as_material
    from ..sources import Dipole
    from ..monitors import Record, Field
    from .._spec import static_float
    if (len(scene.structures) != 1 or len(plan._shape_seeds) != 1
            or scene.initial is not None
            or any(not isinstance(s, Dipole) for s in scene.sources)
            or any(not isinstance(m, (Record, Field)) for m in scene.monitors)):
        raise ValueError(f'A traced {where} currently supports one isolated ranged '
                         f'{where.lower()}, '
                         'Dipole sources, Record/Field monitors and no initial field.')
    for material in (as_material(scene.background), scene.structures[0].material):
        if type(material) is not Material:
            raise ValueError(f'A traced {where} requires concrete isotropic lossless Material.')
        for triple in (material.eps, material.mu_value):
            values = tuple(static_float(f'traced {where} material', 'value', v) for v in triple)
            if not values[0] == values[1] == values[2]:
                raise ValueError(f'A traced {where} requires concrete isotropic lossless Material.')
    shape = scene.structures[0].shape
    # support() includes the occupancy margin; constitutive terms add one interpolation
    # cell. Keep that whole region strictly inside the grid, including its end faces.
    if any(lo <= 1 or hi >= n - 1
           for (lo, hi), n in zip(shape.support(plan.grid), plan.grid.n)):
        raise ValueError(f'A traced {where} range, including its interpolation halo, must '
                         'stay strictly inside the domain, away from boundary masks.')
    if isinstance(shape, Prism):
        index, vertices, height, center = plan._shape_seeds[0]
        dimensions = dict(vertices=vertices, height=height, center=center)
    elif isinstance(shape, Box):
        index, size, center = plan._shape_seeds[0]
        dimensions = dict(size=size, center=center)
    else:
        index, radius, center, *height = plan._shape_seeds[0]
        dimensions = dict(radius=radius, center=center)
        if isinstance(shape, Cylinder):
            dimensions['height'] = height[0]
    seed = shape.replace(**dimensions)
    return scene.replace(structures=(scene.structures[index].replace(shape=seed),))


def validate(scene, plan):
    """Direct lower() has the same preflight as run(), not just a matching allocation."""
    concrete = planning_scene(scene, plan)
    if concrete is scene:
        return
    from .plan import plan as make_plan
    checked = make_plan(concrete, _shape_trace=True)
    if checked.skeleton != plan.skeleton:
        raise ValueError(f'lower(scene, plan=...): traced {name(scene)} plan belongs to a different '
                         'scene; rebuild fd.plan from concrete geometry.')


def _prism_outline_valid(shape, rt, minimum):
    """The concrete polygon door and hull-width floor, on backend scalar arrays.

    A hull's minimum width occurs perpendicular to one of its edges. Every hull
    edge joins two input vertices, so testing all distinct pairs includes that
    minimum without sorting a traced hull. Extra directions cannot undercut the
    global minimum. Degenerate pairs add infinity, not an unguarded division.
    The crossing signs and inclusive endpoint tests match polygon_shape; neither
    validity nor the width check contributes a derivative to the physical operator.
    """
    xp = rt.xp
    vertices = xp.stack([xp.stack([rt.array(v, 'float64') for v in p])
                         for p in shape.vertices])
    nv = len(shape.vertices)
    after = xp.roll(vertices, -1, axis=0)
    area = xp.sum(vertices[:, 0]*after[:, 1] - after[:, 0]*vertices[:, 1])
    valid = xp.all(xp.isfinite(vertices)) & (area > 0)
    width = rt.array(float('inf'), 'float64')

    def side(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def between(a, b, c):
        return xp.all((xp.minimum(a, b) <= c) & (c <= xp.maximum(a, b)))

    for i in range(nv):
        for j in range(i+1, nv):
            delta = vertices[j]-vertices[i]
            norm = xp.sqrt(xp.sum(delta*delta))
            relative = vertices-vertices[i]
            projection = relative[:, 0]*delta[1] - relative[:, 1]*delta[0]
            span = (xp.max(projection)-xp.min(projection))/xp.where(norm > 0, norm, 1.)
            width = xp.minimum(width, xp.where(norm > 0, span, float('inf')))
            if j == i+1 or (i == 0 and j == nv-1):
                continue
            p, q, r, s = vertices[i], after[i], vertices[j], after[j]
            d1, d2, d3, d4 = side(r, s, p), side(r, s, q), side(p, q, r), side(p, q, s)
            crossing = ((d1*d2 < 0) & (d3*d4 < 0)
                        | ((d1 == 0) & between(r, s, p))
                        | ((d2 == 0) & between(r, s, q))
                        | ((d3 == 0) & between(p, q, r))
                        | ((d4 == 0) & between(p, q, s)))
            valid = valid & ~crossing
    return valid & (width >= minimum)


def guard(scene, plan, rt, previous):
    """Runtime counterpart of containment and the existing four-cell resolution door."""
    if not traced(scene):
        return previous
    from .checks import CELLS_ACROSS_REFUSED, _ACROSS_SLACK
    shape = scene.structures[0].shape
    xp = rt.xp
    centers = (shape.center,)
    if isinstance(shape, Prism):
        height = rt.array(shape.height, 'float64')
        minimum = plan.dx * CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK)
        valid = xp.isfinite(height) & (height >= minimum)
        valid = valid & _prism_outline_valid(shape, rt, minimum)
        halves = tuple(height / 2 if a == 'xyz'.index(shape.axis) else 0. for a in range(3))
        centers = shape._range_points()
    elif isinstance(shape, Box):
        sizes = tuple(rt.array(s, 'float64') for s in shape.size)
        valid = xp.ones_like(sizes[0], dtype=xp.bool)
        for size in sizes:
            valid = valid & xp.isfinite(size) & (
                size / plan.dx >= CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK))
        halves = shape._range_halves()
    else:
        radius = rt.array(shape.radius, 'float64')
        valid = xp.isfinite(radius) & (2 * radius / plan.dx >=
                                      CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK))
        height = None
        if isinstance(shape, Cylinder):
            height = rt.array(shape.height, 'float64')
            valid = valid & xp.isfinite(height) & (
                height / plan.dx >= CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK))
        halves = tuple(height / 2 if height is not None and axis == 'xyz'.index(shape.axis)
                       else radius for axis in range(3))
    for center in centers:
        for axis, (c, (lo, hi)) in enumerate(zip(center, shape.range)):
            c = rt.array(c, 'float64')
            half = rt.array(halves[axis], 'float64')
            valid = valid & xp.isfinite(c) & (c - half >= lo) & (c + half <= hi)
    value = rt.constant(xp.where(valid, rt.array(1., 'float64'), rt.array(float('nan'), 'float64')))
    return value if previous is None else previous * value
