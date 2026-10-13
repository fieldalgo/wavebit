"""Source edits that kill every new public sphere-gradient regression; restore exact source bytes."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import hashlib
import json
import subprocess
import sys


ROOT = repository_root()
LOGS = ROOT/'studies/fdtd/sphere_grad/records/fdtd_sphere_grad_controls'
PREFIX = 'tests/fdtd/test_fd_sphere_grad.py::'
MUTATIONS = [('drop_ranged_radius',
  'geometry.py',
  "return value if _sphere_traced(value) else static_float('Sphere', name, value)",
  "return .22 if name == 'radius' else (value if _sphere_traced(value) else "
  "static_float('Sphere', name, value))",
  'tests/fdtd/test_fd_sphere_range.py::test_range_does_not_silently_detach_a_shape_derivative[torch]'),
 ('drop_radius_in_solver',
  'geometry.py',
  "return value if _sphere_traced(value) else static_float('Sphere', name, value)",
  "return value.detach() if name == 'radius' and _sphere_traced(value) and hasattr(value, "
  "'detach') else (value if _sphere_traced(value) else static_float('Sphere', name, value))",
  'test_public_radius_and_all_center_derivatives_match_numpy[torch]'),
 ('drop_center_in_solver',
  'geometry.py',
  "return value if _sphere_traced(value) else static_float('Sphere', name, value)",
  "return value.detach() if name == 'center' and _sphere_traced(value) and hasattr(value, "
  "'detach') else (value if _sphere_traced(value) else static_float('Sphere', name, value))",
  'test_public_radius_and_all_center_derivatives_match_numpy[torch]'),
 ('tensor_tuple_detach',
  '_shapes/_bodies.py',
  "if isinstance(value, (tuple, list)) and any(hasattr(v, 'dtype') for v in value):",
  "if False and isinstance(value, (tuple, list)) and any(hasattr(v, 'dtype') for v in value):",
  'test_public_radius_and_all_center_derivatives_match_numpy[torch]'),
 ('skip_resolution_guard',
  '_lower/shape_ranges.py',
  'CELLS_ACROSS_REFUSED * (1.0 - _ACROSS_SLACK))',
  '0.)',
  'test_jitted_public_run_marks_invalid_geometry_unstable'),
 ('skip_containment_guard',
  '_lower/shape_ranges.py',
  'valid = valid & xp.isfinite(c) & (c - radius >= lo) & (c + radius <= hi)',
  'valid = valid & xp.isfinite(c)',
  'test_jitted_public_run_marks_invalid_geometry_unstable'),
 ('allow_multiple_painters',
  '_lower/shape_ranges.py',
  'len(scene.structures) != 1 or len(plan._shape_seeds) != 1',
  'False or len(plan._shape_seeds) != 1',
  'test_traced_preflight_requires_a_certified_scene_and_plan'),
 ('skip_plan_preflight',
  '_lower/shape_ranges.py',
  'if checked.skeleton != plan.skeleton:',
  'if False:',
  'test_traced_preflight_requires_a_certified_scene_and_plan'),
 ('allow_boundary_halo',
  '_lower/shape_ranges.py',
  'if any(lo <= 1 or hi >= n - 1',
  'if False and any(lo <= 1 or hi >= n - 1',
  'test_traced_preflight_requires_a_certified_scene_and_plan'),
 ('drop_schedule_seed',
  '_lower/plan.py',
  "object.__setattr__(result, '_shape_seeds', self._shape_seeds)",
  "object.__setattr__(result, '_shape_seeds', ())",
  'test_traced_preflight_requires_a_certified_scene_and_plan'),
 ('hide_unknown_report',
  '_lower/checks.py',
  "'Sphere geometry is traced: body volume, cut-cell counts, cells across '",
  "'Body volume, cut-cell counts, cells across '",
  'test_traced_monitor_results_and_reports_use_the_actual_geometry')]


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    records = []
    for name, relative, before, after, test in MUTATIONS:
        target = test if test.startswith('tests/') else PREFIX+test
        source = ROOT/'src/wavebit/fd'/relative
        original = source.read_bytes()
        try:
            contents = source.read_text(encoding='utf-8')
            assert contents.count(before) == 1, (name, contents.count(before))
            source.write_text(contents.replace(before, after), encoding='utf-8')
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p',
                                     'no:cacheprovider', target], cwd=ROOT,
                                    capture_output=True, text=True)
            output = result.stdout+'\n'+result.stderr
            (LOGS/(name+'.log')).write_text(output, encoding='utf-8')
        finally:
            source.write_bytes(original)
        assert source.read_bytes() == original
        records.append(dict(mutation=name, source=relative, test=target,
                            before=before, after=after, exit=result.returncode,
                            source_sha256=hashlib.sha256(original).hexdigest()))
        (LOGS/'witnesses.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
        assert result.returncode == 1, output[-3000:]
        assert 'AssertionError' in output or 'DID NOT RAISE' in output, output[-3000:]
        assert not any(s in output for s in ('INTERNALERROR', 'ERROR collecting', 'crashed'))
        print(name, 'killed; source bytes restored', flush=True)


if __name__ == '__main__':
    main()
