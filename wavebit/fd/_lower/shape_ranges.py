"""Fixed sphere/cylinder allocations and the restricted public geometry-gradient certificate."""
from ..geometry import Sphere, Cylinder, _shape_traced


def records(scene):
    """Structure index, kind and exact bounds; cylinders also record their static axis.

    Bounds are recorded even when two ranges snap to the same cells. A plan reserves the
    range the caller declared, not any later range that happens to have the same allocation.
    Indexing also distinguishes moving a range to another painter in an overlapping scene.
    """
    out = []
    for index, structure in enumerate(scene.structures):
        shape = structure.shape
        if isinstance(shape, (Sphere, Cylinder)) and shape.range is not None:
            row = (index, type(shape).__name__.lower(), shape.range)
            out.append(row + (shape.axis,) if isinstance(shape, Cylinder) else row)
    return tuple(out)


def refuse_mismatch(scene, plan):
    """Protect direct ``lower`` callers too; ``run`` additionally checks the whole scene."""
    if records(scene) != plan.shape_ranges:
        raise ValueError('lower(scene, plan=...): Sphere/Cylinder(range=) declarations differ from the '
                         'plan. Keep each range on the same structure with the same bounds, '
                         'or build a new plan with fd.plan(scene).')


def leaves(shape):
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
    seeds = tuple((i, s.shape.radius, s.shape.center)
                  + ((s.shape.height,) if isinstance(s.shape, Cylinder) else ())
                  for i, s in enumerate(scene.structures)
                  if isinstance(s.shape, (Sphere, Cylinder)) and s.shape.range is not None)
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


def guard(scene, plan, rt, previous):
    """Runtime counterpart of containment and the existing four-cell resolution door."""
    if not traced(scene):
        return previous
    from .checks import CELLS_ACROSS_REFUSED, _ACROSS_SLACK
    shape = scene.structures[0].shape
    xp = rt.xp
    radius = rt.array(shape.radius, 'float64')
    valid = xp.isfinite(radius) & (2 * radius / plan.dx >=
                                  CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK))
    height = None
    if isinstance(shape, Cylinder):
        height = rt.array(shape.height, 'float64')
        valid = valid & xp.isfinite(height) & (
            height / plan.dx >= CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK))
    for axis, (c, (lo, hi)) in enumerate(zip(shape.center, shape.range)):
        c = rt.array(c, 'float64')
        half = height / 2 if height is not None and axis == 'xyz'.index(shape.axis) else radius
        valid = valid & xp.isfinite(c) & (c - half >= lo) & (c + half <= hi)
    value = rt.constant(xp.where(valid, rt.array(1., 'float64'), rt.array(float('nan'), 'float64')))
    return value if previous is None else previous * value
