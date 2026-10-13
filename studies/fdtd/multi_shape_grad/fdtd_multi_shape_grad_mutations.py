"""Real production mutations for every multiple-body regression, restoring source bytes."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path

import studies.fdtd.sphere_grad.fdtd_sphere_grad_mutations as controls


controls.LOGS = repository_path('studies/fdtd/multi_shape_grad/records/fdtd_multi_shape_grad_controls')
controls.PREFIX = 'tests/fdtd/test_fd_multi_shape_grad.py::'
controls.MUTATIONS = [
    ('detach_geometry_leaves', 'geometry.py',
     'return value if _shape_traced(value) else static_float(where, name, value)',
     """return value.detach() if _shape_traced(value) and hasattr(value, 'detach') else (
        value if _shape_traced(value) else static_float(where, name, value))""",
     'test_all_four_public_body_gradients_match_numpy[torch]'),
    ('moving_span_report', 'geometry.py',
     "bounds = getattr(self, 'range', None)\n        if bounds is not None:",
     'bounds = None\n        if bounds is not None:',
     'test_sweeps_keep_all_supports_and_match_unranged_and_reversed_fields[numpy]'),
    ('paint_first_body_only', '_lower/coefficients.py',
     'enumerate(zip(paint_list, structure_values))',
     'enumerate(zip(paint_list[:1], structure_values[:1]))',
     'test_separated_coefficients_equal_the_sum_of_independent_body_changes'),
    ('skip_seed_restoration', '_lower/shape_ranges.py',
     'structure = structure.replace(shape=shape.replace(**dimensions))',
     'structure = structure',
     'test_each_traced_seed_is_restored_at_its_index_and_concrete_partners_stay_current[3]'),
    ('drop_partners', '_lower/shape_ranges.py',
     'structures.append(structure)', 'structures.append(structure) if i == 0 else None',
     'test_each_traced_seed_is_restored_at_its_index_and_concrete_partners_stay_current[2]'),
    ('wrong_cylinder_seed_height', '_lower/shape_ranges.py',
     "dimensions['height'] = height[0]", "dimensions['height'] = height[0]*.99",
     'test_each_traced_seed_is_restored_at_its_index_and_concrete_partners_stay_current[1]'),
    ('ignore_reserved_overlap', '_lower/shape_ranges.py',
     'if all(min(hi, b) > max(lo, a) for (lo, hi), (a, b) in zip(halo, other)):',
     'if False:',
     'test_every_reserved_pair_checks_the_interpolation_halo_in_all_axes[0]'),
    ('omit_interpolation_halo', '_lower/shape_ranges.py',
     'halo = tuple((lo-1, hi+1) for lo, hi in support)', 'halo = support',
     'test_every_reserved_pair_checks_the_interpolation_halo_in_all_axes[1]'),
    ('compare_adjacent_pair_only', '_lower/shape_ranges.py',
     'for j, other in enumerate(halos):', 'for j, other in enumerate(halos[-1:]):',
     'test_every_reserved_pair_checks_the_interpolation_halo_in_all_axes[2]'),
    ('check_first_material_only', '_lower/shape_ranges.py',
     '*(s.material for s in scene.structures)', '*(s.material for s in scene.structures[:1])',
     'test_every_partner_material_is_concrete_isotropic_and_lossless[3]'),
    ('restore_concrete_partner', '_lower/shape_ranges.py',
     'if any(_shape_traced(v) for v in leaves(shape)):', 'if True:',
     'test_a_concrete_partner_is_rechecked_instead_of_using_its_old_seed'),
    ('guard_first_body_only', '_lower/shape_ranges.py',
     'for structure in scene.structures:\n        current = _valid(structure.shape, plan, rt)',
     'for structure in scene.structures[:1]:\n        current = _valid(structure.shape, plan, rt)',
     'test_jitted_lowering_combines_every_runtime_geometry_guard'),
    ('keep_last_guard_only', '_lower/shape_ranges.py',
     'valid = current if valid is None else valid & current', 'valid = current',
     'test_jitted_lowering_combines_every_runtime_geometry_guard'),
    ('record_first_range_only', '_lower/shape_ranges.py',
     'for index, structure in enumerate(scene.structures):',
     'for index, structure in enumerate(scene.structures[:1]):',
     'test_all_range_records_and_static_scene_inputs_protect_the_plan'),
    ('omit_absorber_halo', '_lower/shape_ranges.py',
     "if (halo[a][0] < cells if face[1] == '-' else halo[a][1] > plan.grid.n[a]-cells):",
     'if False:',
     'test_unturned_partner_halos_clear_domain_and_absorbers_too'),
    ('omit_domain_halo', '_lower/shape_ranges.py',
     'if any(lo <= 1 or hi >= n - 1 for (lo, hi), n in zip(support, plan.grid.n)):',
     'if False:',
     'test_unturned_partner_halos_clear_domain_and_absorbers_too'),
    ('hide_traced_body_kinds', '_lower/shape_ranges.py',
     """return '/'.join(dict.fromkeys(type(s.shape).__name__ for s in scene.structures
                    if any(_shape_traced(v) for v in leaves(s.shape))))""",
     "return 'Sphere'",
     'test_traced_reports_name_all_body_kinds_and_monitors_use_current_geometry'),
    ('skip_scene_preflight', '_lower/shape_ranges.py',
     'if checked.skeleton != plan.skeleton:', 'if False:',
     'test_each_traced_seed_is_restored_at_its_index_and_concrete_partners_stay_current[0]'),
]


if __name__ == '__main__':
    controls.main()
