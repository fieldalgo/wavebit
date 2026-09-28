"""Fixed allocations for concrete sphere sweeps; not a traced-geometry certificate."""
from ..geometry import Sphere


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
