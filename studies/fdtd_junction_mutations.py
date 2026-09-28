"""Run real source mutations against the checkout selected on the command line.

Each mutation must fail its selected regression (including tracing failures). Preserve each
failure log and restore source bytes in finally, including on unexpected errors.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
logs = root / 'studies/results/fdtd_junction_mutations'
logs.mkdir(parents=True, exist_ok=True)
J = 'wavebit/fd/_lower/junction.py'
mutants = [
    ('rule_names', [('wavebit/fd/geometry.py', '            raise ValueError(message)', '            pass')],
     'test_rule_names_are_explicit'),
    ('unread_shape', [('wavebit/fd/geometry.py', '        if self.junction != JUNCTION_DEFAULT and',
                      '        if False and self.junction != JUNCTION_DEFAULT and')],
     'test_unread_rules_on_ineligible_shapes_are_refused'),
    ('unread_scene', [(J, '        if structure.junction != DEFAULT and i not in read:',
                      '        if False and structure.junction != DEFAULT and i not in read:')],
     'test_unread_rule_on_a_lifted_pillar_and_on_the_substrate_is_refused'),
    ('refuse', [(J, '    if rule == REFUSE:', '    if False and rule == REFUSE:')],
     'test_refuse_retains_the_shared_surface_refusal'),
    ('normal_arithmetic', [(J, '    aa = (1-alpha)/up + alpha/eps_s',
                          '    aa = 1/((1-alpha)*up + alpha*eps_s)')],
     'test_split_matches_independent_laminate_and_preserves_unwritten_points'),
    ('ground_after_first', [(f, 'min(record.face, record.standing)', 'max(record.face, record.standing)')
                            for f in ('wavebit/fd/_lower/coefficients.py',
                                      'wavebit/fd/_lower/constitutive.py')],
     'test_pair_listing_order_does_not_change_the_interface'),
    ('staircase', [(J, '    if record.rule != STAIRCASE:', '    if True:')],
     'test_staircase_uses_the_medium_at_each_point'),
    ('mismatch', [(J, 'def refuse_mismatch(scene, plan):', 'def refuse_mismatch(scene, plan):\n    return')],
     'test_plan_mismatch_is_refused'),
    ('torch_values', [(J, '    return _paste(xp, full, box, xp.where(write, 1/diagonal[component], full[view]))',
                      '    return _paste(xp, full, box, xp.where(write, (1.01 if rt.name == "torch" else 1)/diagonal[component], full[view]))')],
     'test_material_backends_agree'),
    ('detached_materials', [(J, '    result = values(face.shape.body(),',
                            '    es, ep, ground = rt.constant(es), rt.constant(ep), rt.constant(ground)\n'
                            '    result = values(face.shape.body(),')],
     'test_material_gradients_follow_the_same_split'),
    ('cache_masks', [('wavebit/fd/_run.py', 'tuple(_stripped(r) for r in plan.junctions)',
                     'tuple(r for r in plan.junctions)')],
     'test_skeleton_records_the_rule_and_execution_drops_masks'),
    ('box_node_moment', [(J, '    if node:', '    if node and isinstance(body, Cylinder):')],
     'test_box_node_limit_preserves_the_existing_nearest_wall_tensor'),
    ('report_h_families', [(J, "if row[0].startswith('E')", "if True")],
     'test_report_counts_the_written_electric_families'),
    ('empty_skeleton', [('wavebit/fd/_lower/plan.py', "        static.pop('junctions')", '        pass')],
     'test_no_junction_never_classifies_or_changes_the_skeleton'),
    ('nested_empty_skeleton', [('wavebit/fd/_lower/plan.py',
                               "if not (isinstance(v, Plan) and f.name == 'junctions' and not v.junctions)",
                               'if True')],
     'tests/test_fd_stage2_backends.py::test_the_scene_is_the_one_frozen[plane_wave]'),
    ('nonflush', [(J, '                if abs(plane-base) <= _CLIP_ROUNDOFF*grid.dx:',
                   '                if abs(plane-base) <= grid.dx:')],
     'test_a_nonflush_pair_remains_refused'),
    ('magnetic', [(J, '    if material.is_magnetic:', '    if False and material.is_magnetic:')],
     'test_magnetic_and_late_overlays_remain_refused'),
    ('rim_scope', [(J, '            if np.any(write & (section_face < 1-slack)):',
                   '            if False and np.any(write & (section_face < 1-slack)):')],
     'test_unvalidated_geometries_remain_refused[edge]'),
    ('floor', [(J, 'return xp.stack(diagonal), xp.stack(cross), 1/arithmetic',
                'return xp.stack(diagonal), xp.stack(cross), 1/normal_mean')],
     'test_high_contrast_junction_exercises_the_lift_and_stays_bounded'),
    ('geometry_detached', [(J, '    fu = _section(standing, points, dx)',
                            '    fu = np.asarray(_section(standing, points, dx))')],
     'test_internal_radius_derivative_keeps_geometry_on_backend_arrays'),
    ('stepped_materials', [(J, '    result = values(face.shape.body(),',
                            '    es, ep, ground = rt.constant(es), rt.constant(ep), rt.constant(ground)\n'
                            '    result = values(face.shape.body(),')],
     'test_material_derivatives_survive_forty_coupled_steps'),
    ('half_cell_tie', [(J, 'side*(points[..., axis]-plane) <= (.5+tol)*grid.dx',
                        'side*(points[..., axis]-plane) <= .5*grid.dx')],
     'tests/test_fd_junction_physics.py::test_height_continuity_includes_the_exact_ties[0.5-box]'),
    ('study_values', [(J, '    aa = (1-alpha)/up + alpha/eps_s',
                       '    uu, vv = 1.01*uu, 1.01*vv\n    aa = (1-alpha)/up + alpha/eps_s')],
     'tests/test_fd_junction_physics.py::test_public_solver_reproduces_the_measured_study[0]'),
    ('film_values', [(J, '    aa = (1-alpha)/up + alpha/eps_s',
                      '    uu, vv = 1.01*uu, 1.01*vv\n    aa = (1-alpha)/up + alpha/eps_s')],
     'tests/test_fd_junction_physics.py::test_flat_film_retains_the_measured_second_order'),
    ('inverse_update', [(J, 'xp.where(write, 1/diagonal[component], full[view])',
                        'xp.where(write, diagonal[component], full[view])')],
     'tests/test_fd_junction_physics.py::test_a_formerly_refused_scene_agrees_with_rcwa'),
]
selected = set(sys.argv[2:])
completed = 0
manifest = logs / 'witnesses.json'
records = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
for name, changes, test in mutants:
    if selected and name not in selected:
        continue
    originals = {path: (root/path).read_bytes() for path, _, _ in changes}
    try:
        for path, before, after in changes:
            text = (root/path).read_text(encoding='utf-8')
            assert text.count(before) == 1, (name, path, text.count(before))
            (root/path).write_text(text.replace(before, after), encoding='utf-8')
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                 test if '::' in test else 'tests/test_fd_junction.py::'+test],
                                cwd=root, capture_output=True, text=True)
        output = result.stdout+'\n'+result.stderr
        (logs / (name+'.log')).write_text(output, encoding='utf-8')
        assert result.returncode == 1 and 'FAILED' in output, (name, result.returncode, output[-1500:])
        assert 'INTERNALERROR' not in output and 'ERROR collecting' not in output, (name, output)
        record = dict(mutation=name, test=test, exit=result.returncode,
                      sources={p: hashlib.sha256(b).hexdigest() for p,b in originals.items()})
        records = [r for r in records if r['mutation'] != name] + [record]
        print(name, 'killed', flush=True)
    finally:
        for path, data in originals.items():
            (root/path).write_bytes(data)
    assert all((root/path).read_bytes() == b for path,b in originals.items())
    manifest.write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
    completed += 1
print(completed, 'mutations killed this run;', len(records),
      'witnesses recorded; all source bytes restored.', flush=True)
