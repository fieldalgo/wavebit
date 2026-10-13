"""Real production mutations for every traced-junction regression; restore source bytes."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path

import studies.fdtd.sphere_grad.fdtd_sphere_grad_mutations as controls


controls.LOGS = repository_path('studies/results/fdtd_traced_junction_controls')
controls.PREFIX = 'tests/fdtd/test_fd_traced_junction.py::'
controls.MUTATIONS = [
    ('live_write_without_inside', '_lower/junction.py',
     '    return inside | base', '    return base',
     'test_the_live_write_set_is_the_plans_own_at_its_seed[cylinder--0.2375]'),
    ('live_write_without_base', '_lower/junction.py',
     '    return inside | base', '    return inside',
     'test_the_live_write_set_is_the_plans_own_at_its_seed[box--0.25]'),
    ('cylinder_base_faces_up', '_lower/junction.py',
     'base = record.side*pb.surface(points)[1][..., record.axis] < 0',
     'base = record.side*pb.surface(points)[1][..., record.axis] <= 0',
     'test_the_live_write_set_is_the_plans_own_at_its_seed[cylinder--0.225]'),
    ('box_base_by_its_sides', '_lower/junction.py',
     'base = pb.exact_moment(points, grid.dx)[1][..., record.axis] > 0',
     'base = pb.exact_moment(points, grid.dx)[1][..., record.axis] >= 0',
     'test_the_live_write_set_is_the_plans_own_at_its_seed[box--0.225]'),
    ('paint_with_the_seed_mask', '_lower/junction.py',
     'write = (live_write(rt, record, scene, plan.grid, points) if moving(scene, record)',
     'write = (live_write(rt, record, scene, plan.grid, points) if False',
     'test_a_traced_pillar_lowers_and_runs_as_the_concrete_scene_of_its_geometry[box--0.2375]'),
    ('paint_nodes_with_the_seed_mask', '_lower/junction.py',
     'write = (live_write(rt, record, scene, plan.grid, points) if traced',
     'write = (live_write(rt, record, scene, plan.grid, points) if False',
     'test_a_traced_pillar_lowers_and_runs_as_the_concrete_scene_of_its_geometry'
     '[cylinder--0.2375]'),
    ('mismatch_on_the_traced_scene', '_lower/junction.py',
     '        scene = planning_scene(scene, plan)', '        pass',
     'test_jitted_lowering_follows_the_rim_without_recompiling'),
    ('mismatch_on_the_traced_scene_beside', '_lower/junction.py',
     '        scene = planning_scene(scene, plan)', '        pass',
     'tests/fdtd/test_fd_traced_partner_grad.py::'
     'test_a_flush_junction_beside_a_traced_body_lowers_as_its_concrete_scene'),
    ('junction_values_detach_the_section', '_lower/junction.py',
     '    fu = _section(standing, points, dx)',
     '    fu = _section(standing, points, dx)\n'
     "    fu = fu.detach() if hasattr(fu, 'detach') else fu",
     'test_a_pillar_on_a_substrate_differentiates_its_orders[cylinder--0.2375-None-torch]'),
    ('certificate_ignores_the_edge', '_lower/shape_ranges.py',
     'if not (face.low[b] <= grid.low[b] + (lo[b]-.5)*grid.dx',
     'if False and not (face.low[b] <= grid.low[b] + (lo[b]-.5)*grid.dx',
     'test_the_junction_certificate_covers_the_whole_range'),
    ('certificate_far_face_without_its_cell', '_lower/shape_ranges.py',
     'if lo[a] - 1 <= cell <= hi[a] + 1:', 'if lo[a] <= cell <= hi[a]:',
     'test_the_junction_certificate_covers_the_whole_range'),
    ('certificate_admits_staircase', '_lower/shape_ranges.py',
     'if record.rule != SPLIT:', 'if False:',
     'test_the_junction_certificate_covers_the_whole_range'),
    ('partner_exception_for_every_partner', '_lower/shape_ranges.py',
     'if standing.get(j) == i:', 'if j in standing:',
     'test_the_junction_certificate_covers_the_whole_range'),
    ('traced_face_admitted', '_lower/shape_ranges.py',
     'if record.face in moving:', 'if False:',
     'test_a_traced_face_is_refused'),
    ('base_guard_dropped', '_lower/shape_ranges.py',
     'valid = valid & (rt.xp.abs(base - record.plane) <= _CLIP_ROUNDOFF * plan.dx)',
     'valid = valid',
     'test_the_base_must_stay_on_the_face'),
]


if __name__ == '__main__':
    controls.main()
