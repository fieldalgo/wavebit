"""Real source mutations for every prism regression; restore original source bytes."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path

import studies.fdtd.sphere_grad.fdtd_sphere_grad_mutations as controls


controls.LOGS = repository_path('studies/fdtd/prism_grad/records/fdtd_prism_grad_controls')
controls.PREFIX = 'tests/fdtd/test_fd_prism_grad.py::'
controls.MUTATIONS = [
    ('ignore_prism_containment', 'geometry.py',
     'and (c - half < lo or c + half > hi)):',
     "and where != 'Prism' and (c - half < lo or c + half > hi)):",
     'test_range_contains_every_rotated_vertex_and_both_caps[23.0-z]'),
    ('allow_prism_stencil', 'geometry.py',
     'self.discretisation == STENCIL and isinstance(self.shape, (Sphere, Cylinder, Box, Prism))',
     'self.discretisation == STENCIL and isinstance(self.shape, (Sphere, Cylinder, Box))',
     'test_range_contains_every_rotated_vertex_and_both_caps[0.0-z]'),
    ('unranged_vertices_detach', 'geometry.py',
     '''vertices = tuple(tuple(vertices[i, j] for j in range(vertices.shape[1]))
                             for i in range(vertices.shape[0]))''',
     'vertices = vertices.tolist()',
     'test_unranged_vertices_stay_static_in_every_transform_and_bad_outlines_refuse'),
    ('allow_boolean_vertex_array', 'geometry.py',
     "for kind in ('bool', 'complex')):", "for kind in ('complex',)):",
     'test_unranged_vertices_stay_static_in_every_transform_and_bad_outlines_refuse'),
    ('moving_support', 'geometry.py',
     'bounds = bounds if bounds is not None else tuple(zip(self.low, self.high))',
     'bounds = tuple(zip(self.low, self.high))',
     'test_reserved_support_recovers_volume_at_opposite_corners[concave]'),
    ('moving_span_report', 'geometry.py',
     """bounds = getattr(self, 'range', None)
        if bounds is not None:""",
     'bounds = None\n        if bounds is not None:',
     'test_convex_and_concave_sweeps_share_the_plan_capacity_and_fields[numpy-z]'),
    ('convex_capacity', 'geometry.py',
     'body._convex = body.convex = False', 'body._convex = body.convex = True',
     'test_convex_and_concave_sweeps_share_the_plan_capacity_and_fields[numpy-concave]'),
    ('detach_vertices', 'geometry.py',
     'return value if _shape_traced(value) else static_float(where, name, value)',
     """return value.detach() if name.startswith('vertices') and _shape_traced(value) and hasattr(value,
        'detach') else (value if _shape_traced(value) else static_float(where, name, value))""",
     'test_all_vertex_height_and_center_derivatives_match_numpy[torch-concave]'),
    ('detach_height', 'geometry.py',
     'return value if _shape_traced(value) else static_float(where, name, value)',
     """return value.detach() if name == 'height' and _shape_traced(value) and hasattr(value,
        'detach') else (value if _shape_traced(value) else static_float(where, name, value))""",
     'test_all_vertex_height_and_center_derivatives_match_numpy[torch-z]'),
    ('detach_centers', 'geometry.py',
     'return value if _shape_traced(value) else static_float(where, name, value)',
     """return value.detach() if name.startswith('center') and _shape_traced(value) and hasattr(value,
        'detach') else (value if _shape_traced(value) else static_float(where, name, value))""",
     'test_all_vertex_height_and_center_derivatives_match_numpy[torch-z]'),
    ('skip_height_floor', '_lower/shape_ranges.py',
     'valid = xp.isfinite(height) & (height >= minimum)', 'valid = xp.isfinite(height)',
     'test_jitted_run_guards_vertices_height_center_and_rotated_containment'),
    ('check_center_only', '_lower/shape_ranges.py',
     'centers = shape._range_points()', 'centers = (shape.center,)',
     'test_jitted_run_guards_vertices_height_center_and_rotated_containment'),
    ('ignore_runtime_rotation', '_lower/shape_ranges.py',
     'centers = shape._range_points()', 'centers = shape.replace(angle=0.)._range_points()',
     'test_jitted_run_guards_vertices_height_center_and_rotated_containment'),
    ('skip_hull_width', '_lower/shape_ranges.py',
     'return valid & (width >= minimum)', 'return valid',
     'test_runtime_outline_door_matches_topology_and_hull_width[numpy]'),
    ('allow_clockwise', '_lower/shape_ranges.py',
     'valid = xp.all(xp.isfinite(vertices)) & (area > 0)',
     'valid = xp.all(xp.isfinite(vertices)) & (area != 0)',
     'test_runtime_outline_door_matches_topology_and_hull_width[numpy]'),
    ('allow_crossing', '_lower/shape_ranges.py',
     'valid = valid & ~crossing', 'valid = valid',
     'test_runtime_outline_door_matches_topology_and_hull_width[numpy]'),
    ('allow_touching', '_lower/shape_ranges.py',
     'return xp.all((xp.minimum(a, b) <= c) & (c <= xp.maximum(a, b)))',
     'return False',
     'test_runtime_outline_door_matches_topology_and_hull_width[numpy]'),
    ('forget_vertex_count', '_lower/shape_ranges.py',
     'row += (shape.axis, shape.angle, len(shape.vertices))',
     'row += (shape.axis, shape.angle)',
     'test_exact_range_orientation_and_vertex_count_are_part_of_the_plan'),
    ('forget_prism_angle', '_lower/shape_ranges.py',
     'row += (shape.axis, shape.angle, len(shape.vertices))',
     'row += (shape.axis, len(shape.vertices))',
     'test_exact_range_orientation_and_vertex_count_are_part_of_the_plan'),
    ('forget_prism_axis', '_lower/shape_ranges.py',
     'row += (shape.axis, shape.angle, len(shape.vertices))',
     'row += (shape.angle, len(shape.vertices))',
     'test_exact_range_orientation_and_vertex_count_are_part_of_the_plan'),
    ('allow_multiple_painters', '_lower/shape_ranges.py',
     'len(scene.structures) != 1 or len(plan._shape_seeds) != 1',
     'False or len(plan._shape_seeds) != 1',
     'test_traced_prisms_need_a_concrete_isolated_seed_and_full_range_clearance'),
    ('wrong_seed_vertices', '_lower/shape_ranges.py',
     'dimensions = dict(vertices=vertices, height=height, center=center)',
     'dimensions = dict(vertices=tuple((u+.001, v) for u, v in vertices), height=height, center=center)',
     'test_traced_prisms_need_a_concrete_isolated_seed_and_full_range_clearance'),
    ('wrong_seed_height', '_lower/shape_ranges.py',
     'dimensions = dict(vertices=vertices, height=height, center=center)',
     'dimensions = dict(vertices=vertices, height=height*.99, center=center)',
     'test_traced_prisms_need_a_concrete_isolated_seed_and_full_range_clearance'),
    ('claim_traced_uniformity', 'geometry.py',
     '''# traced bodies are certified wholly inside the domain before lowering.
                return False''',
     '''# traced bodies are certified wholly inside the domain before lowering.
                return True''',
     'test_traced_reports_and_monitors_use_the_current_prism_near_pml'),
    ('wrong_report_kind', '_lower/checks.py',
     "f'{name(scene)} geometry is traced: body volume, cut-cell counts, cells across '",
     "'Sphere geometry is traced: body volume, cut-cell counts, cells across '",
     'test_traced_reports_and_monitors_use_the_current_prism_near_pml'),
]


if __name__ == '__main__':
    controls.main()
