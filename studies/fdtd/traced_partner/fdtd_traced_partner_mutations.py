"""Real production mutations for every concrete-partner regression; restore source bytes."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path

import studies.fdtd.sphere_grad.fdtd_sphere_grad_mutations as controls


controls.LOGS = repository_path('studies/results/fdtd_traced_partner_controls')
controls.PREFIX = 'tests/fdtd/test_fd_traced_partner_grad.py::'
controls.MUTATIONS = [
    ('detach_order_power', '_assemble.py',
     "    R = R_s + R_p\n    if monitor.side == 'reflect':",
     "    R = (R_s + R_p).detach() if hasattr(R_s, 'detach') else R_s + R_p\n"
     "    if monitor.side == 'reflect':",
     'test_a_pillar_above_a_snapped_substrate_differentiates_its_orders[torch]'),
    ('detach_pole_drive', '_kernel/poles.py',
     'drive_field = E[(slice(None),) + self.region.slices]',
     'drive_field = E[(slice(None),) + self.region.slices]\n'
     "        drive_field = drive_field.detach() if hasattr(drive_field, 'detach') else drive_field",
     'test_a_ball_above_a_dispersive_mirror_differentiates_reflection_and_leakage[torch]'),
    ('drop_partners_from_planning', '_lower/shape_ranges.py',
     '            structures.append(structure)        # checked against traced halos below',
     '            pass',
     'test_a_traced_ball_beside_three_partners_lowers_to_the_sum_of_their_changes'),
    ('guard_asks_partners', '_lower/shape_ranges.py',
     '        if not ranged_body(structure.shape):\n            continue\n'
     '        current = _valid(structure.shape, plan, rt)',
     '        current = _valid(structure.shape, plan, rt)',
     'test_jitted_lowering_beside_an_unturned_box_partner_reads_no_traced_face'),
    ('ask_traced_pairs', '_lower/junction.py',
     'if frozenset((i, j)) in recorded or i in moving or j in moving:',
     'if frozenset((i, j)) in recorded:',
     'test_jitted_lowering_beside_an_unturned_box_partner_reads_no_traced_face'),
    ('partner_reach_unwidened', '_lower/shape_ranges.py',
     'reach = tuple((lo-1, hi+1) for lo, hi in structure.shape.support(plan.grid))',
     'reach = structure.shape.support(plan.grid)',
     'test_a_concrete_partner_must_stay_a_cell_off_every_traced_halo'),
    ('overlap_at_a_boundary', '_lower/shape_ranges.py',
     'return all(min(hi, d) > max(lo, c) for (lo, hi), (c, d) in zip(a, b))',
     'return all(min(hi, d) >= max(lo, c) for (lo, hi), (c, d) in zip(a, b))',
     'test_a_concrete_partner_must_stay_a_cell_off_every_traced_halo'),
    ('skip_partner_halo_check', '_lower/shape_ranges.py',
     '        for j, halo in moving.items():',
     '        for j, halo in {}.items():',
     'test_a_concrete_partner_must_stay_a_cell_off_every_traced_halo'),
    ('partners_take_range_rules', '_lower/shape_ranges.py',
     'ranged = [ranged_body(s.shape) for s in scene.structures]',
     'ranged = [True for s in scene.structures]',
     'test_a_concrete_partner_must_stay_a_cell_off_every_traced_halo'),
    ('skip_partner_material', '_lower/shape_ranges.py',
     'if not r and _carries(as_material(structure.material)):',
     'if False:',
     'test_a_partner_material_must_be_concrete'),
    ('carries_stops_at_dataclasses', '_lower/shape_ranges.py',
     'if dataclasses.is_dataclass(value) and not isinstance(value, type):',
     'if False:',
     'test_a_partner_material_must_be_concrete'),
    ('admit_junction_beside_traced', '_lower/junction.py',
     '    if records and moving:',
     '    if False:',
     'test_a_flush_junction_beside_a_traced_body_is_refused'),
]


if __name__ == '__main__':
    controls.main()
