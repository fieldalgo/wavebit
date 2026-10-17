"""Static compact-state selection after ordinary planning and public admission."""
from wavebit.fdtd._kernel.poles import CompactPoleTerm, PoleTerm


def compact_plan(scene, plan):
    """Select two histories for a fixed, source-free homogeneous ADE plan.

    The caller explicitly selects Torch CUDA; execution checks the device
    again. Every replan starts with ordinary terms and applies this same
    conversion before comparing hashes. No global factory or runtime changes.
    """
    from wavebit.fdtd import geometry, monitors
    from wavebit.fdtd._legacy_boundaries import Periodic
    from wavebit.fdtd._legacy_monitors import Record
    from wavebit.materials.time_domain import Dispersive, Drude, Lorentz, Material

    if (scene.run.decay is not None or plan.decay is not None or scene.sources
            or len(scene.structures) != 1 or type(scene.background) is not Material
            or scene.initial is None or plan.dtype != 'float64'
            or plan.cpml or plan.absorbing_faces or plan.shape_ranges
            or any(type(scene.boundaries.face(a+s)) is not Periodic
                   for a in 'xyz' for s in '-+')
            or any(type(m) not in (Record, monitors.Record) for m in scene.monitors)):
        raise ValueError('compact ADE requires source-free fixed periodic homogeneous scope')
    structure = scene.structures[0]
    material = structure.material
    if (type(structure.shape) is not geometry.Slab or structure.discretisation != 'voxel'
            or type(material) is not Dispersive or len(material.poles) != 1
            or type(material.poles[0]) not in (Lorentz, Drude)):
        raise ValueError('compact ADE requires one full-grid Slab with one Lorentz or Drude pole')
    if len(plan.terms) != 1 or type(plan.terms[0]) is not PoleTerm:
        raise ValueError('compact ADE requires one exact ordinary PoleTerm')
    term = plan.terms[0]
    if (term.n_poles != 1 or any(pair != (0, count) for pair, count in zip(
            (term.region.x, term.region.y, term.region.z), plan.grid.n))):
        raise ValueError('compact ADE requires exact full-grid pole support')
    return plan.with_static(terms=(CompactPoleTerm(term.name, term.region, term.slabs,
                                                 term.n_poles),))
