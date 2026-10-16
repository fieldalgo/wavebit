"""Public numerical-policy provenance from the validated scene and engine plan."""
from wavebit._common.execution import frozen_mapping
from . import _legacy_geometry as geometry
from . import geometry as public_geometry
from ._kernel.stencil import StencilTerm


def numerical_policies(problem, numerics, scene, plan):
    """Do not mistake a requested/default selector for a prepared operator.

    Whole-cell shapes and custom Shape implementations retain their own fill
    semantics. Only actual plan records establish stencil and junction targets.
    This reads static indices, never live traced geometry or material values.
    """
    stencil = {term.structure for term in plan.terms if isinstance(term, StencilTerm)}
    pairs = tuple(dict(face=problem.scene.structures[r.face].name,
                       standing=problem.scene.structures[r.standing].name,
                       face_index=r.face, standing_index=r.standing,
                       axis='xyz'[r.axis], side=r.side, rule=r.rule)
                  for r in plan.junctions)
    discretisation, junction = [], []
    for index, (declared, lowered) in enumerate(zip(problem.scene.structures, scene.structures)):
        shape = type(lowered.shape)
        if index in stencil:
            rule, reason = 'stencil', 'prepared_stencil_operator'
        elif shape in (geometry.Slab, geometry.Region, public_geometry.Slab, public_geometry.Region):
            rule, reason = None, 'whole_cell_fill'
        elif shape in (geometry.Sphere, geometry.Box, geometry.Cylinder, geometry.Prism):
            rule, reason = 'voxel', 'voxel_default'
        else:
            rule, reason = None, 'custom_shape_protocol'
        discretisation.append(dict(structure=declared.name, index=index, rule=rule, reason=reason))
        standing = tuple(r for r in plan.junctions if r.standing == index)
        face = any(r.face == index for r in plan.junctions)
        junction.append(dict(structure=declared.name, index=index,
            rule=standing[0].rule if standing else None,
            reason='standing_body' if standing else 'junction_face' if face else 'no_resolved_junction',
            roles=tuple(role for role, present in (('standing', bool(standing)), ('face', face)) if present)))
    return frozen_mapping(dict(
        discretisation=frozen_mapping(dict(requested=numerics.discretisation,
            structures=tuple(frozen_mapping(row) for row in discretisation))),
        junction=frozen_mapping(dict(requested=numerics.junction,
            structures=tuple(frozen_mapping(row) for row in junction),
            pairs=tuple(frozen_mapping(pair) for pair in pairs)))))
