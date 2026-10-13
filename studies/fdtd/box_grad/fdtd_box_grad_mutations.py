"""Kill every new box regression with real source edits, restoring exact source bytes."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import sys

import studies.fdtd.sphere_grad.fdtd_sphere_grad_mutations as controls


controls.LOGS = repository_path('studies/fdtd/box_grad/records/fdtd_box_grad_controls')
controls.PREFIX = 'tests/fdtd/test_fd_box_grad.py::'
controls.MUTATIONS = [
    ('ignore_rotated_extent', 'geometry.py',
     'half[u], half[v] = co*a + si*b, si*a + co*b',
     'half[u], half[v] = a, b',
     'test_range_contains_all_rotated_corners_and_is_static[z]'),
    ('allow_box_stencil', 'geometry.py',
     'self.discretisation == STENCIL and isinstance(self.shape, (Sphere, Cylinder, Box))',
     'self.discretisation == STENCIL and isinstance(self.shape, (Sphere, Cylinder))',
     'test_range_contains_all_rotated_corners_and_is_static[unturned]'),
    ('moving_support', 'geometry.py',
     'bounds = bounds if bounds is not None else tuple(zip(self.low, self.high))',
     'bounds = tuple(zip(self.low, self.high))',
     'test_reserved_support_keeps_exact_volume_at_opposite_corners[z]'),
    ('moving_span_report', 'geometry.py',
     """bounds = getattr(self, 'range', None)
        if bounds is not None:""",
     """bounds = None
        if bounds is not None:""",
     'test_sweeps_reuse_the_operator_and_match_unranged_fields[numpy-z]'),
    ('freeze_tuple_sizes', '_shapes/_bodies.py',
     'value = _as(array_namespace(*typed), value, typed[0])',
     'value = _as(array_namespace(*typed), (.43, .46, .49), typed[0])',
     'test_all_six_size_and_center_derivatives_match_numpy[jax-z]'),
    ('detach_sizes', 'geometry.py',
     'return value if _shape_traced(value) else static_float(where, name, value)',
     """return value.detach() if name.startswith('size') and _shape_traced(value) and hasattr(value,
        'detach') else (value if _shape_traced(value) else static_float(where, name, value))""",
     'test_all_six_size_and_center_derivatives_match_numpy[torch-z]'),
    ('detach_centers', 'geometry.py',
     'return value if _shape_traced(value) else static_float(where, name, value)',
     """return value.detach() if name.startswith('center') and _shape_traced(value) and hasattr(value,
        'detach') else (value if _shape_traced(value) else static_float(where, name, value))""",
     'test_all_six_size_and_center_derivatives_match_numpy[torch-z]'),
    ('check_only_first_size', '_lower/shape_ranges.py',
     'for size in sizes:', 'for size in sizes[:1]:',
     'test_jitted_run_guards_every_size_center_and_rotated_extent[0.0]'),
    ('unrotated_runtime_bounds', '_lower/shape_ranges.py',
     'halves = shape._range_halves()', 'halves = tuple(s/2 for s in shape.size)',
     'test_jitted_run_guards_every_size_center_and_rotated_extent[23.0]'),
    ('forget_static_angle', '_lower/shape_ranges.py',
     'row += (shape.axis, shape.angle)', 'row += (shape.axis,)',
     'test_exact_range_axis_and_angle_are_part_of_the_plan'),
    ('forget_static_axis', '_lower/shape_ranges.py',
     'row += (shape.axis, shape.angle)', 'row += (shape.angle,)',
     'test_exact_range_axis_and_angle_are_part_of_the_plan'),
    ('allow_multiple_painters', '_lower/shape_ranges.py',
     'len(scene.structures) != 1 or len(plan._shape_seeds) != 1',
     'False or len(plan._shape_seeds) != 1',
     'test_traced_boxes_need_a_concrete_isolated_seed_and_full_range_clearance'),
    ('wrong_seed_size', '_lower/shape_ranges.py',
     'dimensions = dict(size=size, center=center)',
     'dimensions = dict(size=tuple(s*.99 for s in size), center=center)',
     'test_traced_boxes_need_a_concrete_isolated_seed_and_full_range_clearance'),
    ('unranged_vector_detaches', 'geometry.py',
     'value = tuple(value[i] for i in range(3))', 'value = value.tolist()',
     'test_unranged_box_sizes_stay_static_in_every_torch_transform'),
    ('claim_traced_uniformity', 'geometry.py',
     """# traced bodies are certified wholly inside the domain before lowering.
                return False""",
     """# traced bodies are certified wholly inside the domain before lowering.
                return True""",
     'test_traced_reports_and_monitors_use_the_current_box_near_pml'),
    ('wrong_traced_report_kind', '_lower/checks.py',
     "f'{name(scene)} geometry is traced: body volume, cut-cell counts, cells across '",
     "'Sphere geometry is traced: body volume, cut-cell counts, cells across '",
     'test_traced_reports_and_monitors_use_the_current_box_near_pml'),
    ('signed_sine_extent', 'geometry.py',
     'abs(math.sin(math.radians(self.angle)))',
     'math.sin(math.radians(self.angle))',
     'test_range_contains_all_rotated_corners_and_is_static[x-negative]'),
]


if __name__ == '__main__':
    if sys.argv[1:] == ['quadrants']:
        controls.MUTATIONS = controls.MUTATIONS[-1:]
        controls.LOGS = controls.LOGS/'quadrants'
    controls.main()
