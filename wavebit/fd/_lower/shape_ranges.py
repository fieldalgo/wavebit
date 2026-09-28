"""Fixed sphere allocations and the restricted public geometry-gradient certificate."""
from ..geometry import Sphere, _sphere_traced


def records(scene):
    """The structure index, shape kind and exact declared bounds of every ranged sphere.

    Bounds are recorded even when two ranges snap to the same cells. A plan reserves the
    range the caller declared, not any later range that happens to have the same allocation.
    Indexing also distinguishes moving a range to another painter in an overlapping scene.
    """
    return tuple((index, 'sphere', structure.shape.range)
                 for index, structure in enumerate(getattr(scene, 'structures', ()) or ())
                 if isinstance(structure.shape, Sphere) and structure.shape.range is not None)


def refuse_mismatch(scene, plan):
    """Protect direct ``lower`` callers too; ``run`` additionally checks the whole scene."""
    if records(scene) != plan.shape_ranges:
        raise ValueError('lower(scene, plan=...): Sphere(range=) declarations differ from the '
                         'plan. Keep each range on the same structure with the same bounds, '
                         'or build a new plan with fd.plan(scene).')


def traced(scene):
    return any(isinstance(s.shape, Sphere)
               and any(_sphere_traced(v) for v in (s.shape.radius, *s.shape.center))
               for s in scene.structures)


def remember(scene, plan):
    """Only concrete numbers: no backend leaves or scene retained by the program cache."""
    seeds = tuple((i, s.shape.radius, s.shape.center) for i, s in enumerate(scene.structures)
                  if isinstance(s.shape, Sphere) and s.shape.range is not None)
    if seeds:
        object.__setattr__(plan, '_shape_seeds', seeds)


def planning_scene(scene, plan):
    """A concrete preflight of all unchanged inputs, after certifying the full allocation.

    A seed proves nothing about another geometry. Isolation and the fixed, interior support
    remove overlap, seam and wall cases for the entire range; runtime guards then check the
    actual radius, containment, star positivity and CFL bound. Every other scene input is
    rechecked on the host. Multiple painters and geometry-dependent sources/monitors need
    their own range certificate and are deliberately outside this first door.
    """
    if not traced(scene):
        return scene
    if plan is None or not getattr(plan, '_shape_seeds', ()):
        raise ValueError('A traced Sphere requires plan= built by fd.plan(scene) from '
                         'concrete radius and center before differentiation.')
    refuse_mismatch(scene, plan)
    from ..materials import Material, as_material
    from ..sources import Dipole
    from ..monitors import Record, Field
    from .._spec import static_float
    if (len(scene.structures) != 1 or len(plan._shape_seeds) != 1
            or scene.initial is not None
            or any(not isinstance(s, Dipole) for s in scene.sources)
            or any(not isinstance(m, (Record, Field)) for m in scene.monitors)):
        raise ValueError('A traced Sphere currently supports one isolated ranged sphere, '
                         'Dipole sources, Record/Field monitors and no initial field.')
    for material in (as_material(scene.background), scene.structures[0].material):
        if type(material) is not Material:
            raise ValueError('A traced Sphere requires concrete isotropic lossless Material.')
        for triple in (material.eps, material.mu_value):
            values = tuple(static_float('traced Sphere material', 'value', v) for v in triple)
            if not values[0] == values[1] == values[2]:
                raise ValueError('A traced Sphere requires concrete isotropic lossless Material.')
    shape = scene.structures[0].shape
    # support() includes the occupancy margin; constitutive terms add one interpolation
    # cell. Keep that whole region strictly inside the grid, including its end faces.
    if any(lo <= 1 or hi >= n - 1
           for (lo, hi), n in zip(shape.support(plan.grid), plan.grid.n)):
        raise ValueError('A traced Sphere range, including its interpolation halo, must '
                         'stay strictly inside the domain, away from boundary masks.')
    index, radius, center = plan._shape_seeds[0]
    seed = Sphere(radius=radius, center=center, range=shape.range)
    return scene.replace(structures=(scene.structures[index].replace(shape=seed),))


def validate(scene, plan):
    """Direct lower() has the same preflight as run(), not just a matching allocation."""
    concrete = planning_scene(scene, plan)
    if concrete is scene:
        return
    from .plan import plan as make_plan
    checked = make_plan(concrete, _shape_trace=True)
    if checked.skeleton != plan.skeleton:
        raise ValueError('lower(scene, plan=...): traced Sphere plan belongs to a different '
                         'scene; rebuild fd.plan from concrete radius and center.')


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
    for c, (lo, hi) in zip(shape.center, shape.range):
        c = rt.array(c, 'float64')
        valid = valid & xp.isfinite(c) & (c - radius >= lo) & (c + radius <= hi)
    value = rt.constant(xp.where(valid, rt.array(1., 'float64'), rt.array(float('nan'), 'float64')))
    return value if previous is None else previous * value
