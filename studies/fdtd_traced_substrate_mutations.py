"""Real production mutations for every traced-substrate regression; restore source bytes."""
from pathlib import Path

import fdtd_sphere_grad_mutations as controls


controls.LOGS = Path(__file__).resolve().parent/'results/fdtd_traced_substrate_controls'
controls.PREFIX = 'tests/test_fd_traced_material.py::'
GRAZING = 'test_the_grazing_rule_holds_a_traced_substrate_at_run_time'
SUBSTRATE = 'test_a_traced_substrate_runs_and_differentiates_as_the_concrete_scene'
REFUSED = 'test_what_no_runtime_guard_covers_is_refused_by_name'
controls.MUTATIONS = [
    ('grazing_guard_dropped', '_lower/params.py',
     '    guard = grazing_guard(scene, plan, rt, guard)\n', '',
     GRAZING),
    ('band_open_above', '_lower/grazing.py',
     '                    bands.append((low, low / (1.0 - c * c)))',
     "                    bands.append((low, float('inf')))",
     GRAZING),
    ('bands_never_found', '_lower/grazing.py',
     '    if len(waves) != 1 or not diffraction:\n        return {}',
     '    if True:\n        return {}',
     GRAZING),
    ('guard_reads_another_medium', '_lower/shape_ranges.py',
     "        label = 'the background' if key < 0 else f'structures[{key}]'",
     "        label = 'the background' if key < 0 else f'structures[{key + 1}]'",
     GRAZING),
    ('accepted_without_the_default_reach', '_lower/grazing.py',
     '        lowest = max(lowest, cos_min(DEFAULT_REFLECTION, g=g))',
     '        lowest = lowest',
     GRAZING),
    ('accepted_without_the_face_reach', '_lower/grazing.py',
     '    lowest = max(cos_min(FLOOR_REFLECTION, g=g), cos_min(reflection, g=g))',
     '    lowest = cos_min(FLOOR_REFLECTION, g=g)',
     GRAZING),
    ('stabilized_door_dropped', '_lower/shape_ranges.py',
     '        elif label in stabilized:', '        elif False:',
     REFUSED),
    ('stabilized_door_dropped_partner', '_lower/shape_ranges.py',
     '        elif label in stabilized:', '        elif False:',
     'tests/test_fd_traced_partner_grad.py::'
     'test_a_traced_partner_material_in_a_stabilized_pml_or_with_poles_is_refused'),
    ('stabilized_door_for_every_face', '_lower/shape_ranges.py',
     '        if isinstance(face, PML) and _stabilize(face) > 0.0:',
     '        if isinstance(face, PML):',
     SUBSTRATE),
    ('stabilized_door_without_diffraction', '_lower/shape_ranges.py',
     '    if len(waves) != 1 or not any(isinstance(m, Diffraction) for m in scene.monitors):',
     '    if len(waves) != 1:',
     REFUSED),
    ('diffraction_detaches_its_medium', '_assemble.py',
     "    eps_a, mu_a = rt.array(eps, 'float64'), rt.array(mu, 'float64')",
     "    eps_a, mu_a = rt.constant(rt.array(eps, 'float64')), rt.array(mu, 'float64')",
     SUBSTRATE),
    ('junction_detaches_the_face_index', '_lower/junction.py',
     "    es = rt.array(face.material.eps[0], 'float64')",
     "    es = rt.constant(rt.array(face.material.eps[0], 'float64'))",
     SUBSTRATE),
]


if __name__ == '__main__':
    controls.main()
