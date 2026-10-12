"""Kill every new cylinder regression with real source edits, restoring exact bytes."""
from pathlib import Path

import fdtd_sphere_grad_mutations as controls


controls.LOGS = Path(__file__).resolve().parent/'results/fdtd_cylinder_grad_controls'
controls.PREFIX = 'tests/test_fd_cylinder_grad.py::'
controls.MUTATIONS = [
    ('center_only_range', 'geometry.py',
     'and (c - half < lo or c + half > hi)):',
     'and (c < lo or c > hi)):',
     'test_range_contains_the_whole_cylinder_and_stays_static[z]'),
    ('allow_cylinder_stencil', 'geometry.py',
     'self.discretisation == STENCIL and isinstance(self.shape, (Sphere, Cylinder))',
     'self.discretisation == STENCIL and isinstance(self.shape, Sphere)',
     'test_range_contains_the_whole_cylinder_and_stays_static[z]'),
    ('moving_support', 'geometry.py',
     'bounds = bounds if bounds is not None else tuple(zip(self.low, self.high))',
     'bounds = tuple(zip(self.low, self.high))',
     'test_reserved_support_preserves_volume_at_opposite_range_ends[z]'),
    ('moving_span_report', 'geometry.py',
     """bounds = getattr(self, 'range', None)
        if bounds is not None:""",
     """bounds = None
        if bounds is not None:""",
     'test_concrete_sweeps_reuse_the_plan_and_match_unranged_fields[numpy-z]'),
    ('drop_height_derivative', 'geometry.py',
     'return value if _shape_traced(value) else static_float(where, name, value)',
     """return value.detach() if name == 'height' and _shape_traced(value) and hasattr(value,
        'detach') else (value if _shape_traced(value) else static_float(where, name, value))""",
     'test_all_five_public_geometry_derivatives_match_numpy[torch-z]'),
    ('drop_radius_derivative', 'geometry.py',
     'return value if _shape_traced(value) else static_float(where, name, value)',
     """return value.detach() if name == 'radius' and _shape_traced(value) and hasattr(value,
        'detach') else (value if _shape_traced(value) else static_float(where, name, value))""",
     'test_all_five_public_geometry_derivatives_match_numpy[torch-z]'),
    ('detach_vector_center', 'geometry.py',
     'value = tuple(value[i] for i in range(3))',
     'value = value.tolist()',
     'test_all_five_public_geometry_derivatives_match_numpy[torch-z]'),
    ('unranged_vector_bypasses_static_door', 'geometry.py',
     'value = tuple(value[i] for i in range(3))',
     'value = value.tolist()',
     'test_unranged_vector_centers_cannot_silently_detach[Cylinder]'),
    ('skip_height_resolution', '_lower/shape_ranges.py',
     'height / plan.dx >= CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK))',
     'height / plan.dx >= 0.)',
     'test_jitted_run_marks_invalid_radius_height_and_centers_unstable[z]'),
    ('radius_instead_of_cap', '_lower/shape_ranges.py',
     "half = height / 2 if height is not None and axis == 'xyz'.index(shape.axis) else radius",
     'half = radius',
     'test_jitted_run_marks_invalid_radius_height_and_centers_unstable[z]'),
    ('forget_static_axis', '_lower/shape_ranges.py',
     'out.append(row + (shape.axis,) if isinstance(shape, Cylinder) else row)',
     'out.append(row)',
     'test_range_and_axis_changes_require_a_new_plan'),
    ('allow_multiple_painters', '_lower/shape_ranges.py',
     'len(scene.structures) != 1 or len(plan._shape_seeds) != 1',
     'False or len(plan._shape_seeds) != 1',
     'test_traced_cylinder_requires_a_certified_scene_and_concrete_seed'),
    ('wrong_seed_height', '_lower/shape_ranges.py',
     "dimensions['height'] = height[0]",
     "dimensions['height'] = height[0] + .01",
     'test_traced_cylinder_requires_a_certified_scene_and_concrete_seed'),
    ('claim_traced_uniformity', 'geometry.py',
     """# traced cylinders are certified wholly inside the domain before lowering.
            return False""",
     """# traced cylinders are certified wholly inside the domain before lowering.
            return True""",
     'test_traced_cylinder_reports_and_monitors_use_current_geometry_near_pml'),
    ('wrong_traced_report_kind', '_lower/checks.py',
     "f'{name(scene)} geometry is traced: body volume, cut-cell counts, cells across '",
     "'Sphere geometry is traced: body volume, cut-cell counts, cells across '",
     'test_traced_cylinder_reports_and_monitors_use_current_geometry_near_pml'),
]


if __name__ == '__main__':
    controls.main()
