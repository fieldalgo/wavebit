"""Source edits that kill every new fixed-range regression; restore exact source bytes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT/'studies/results/fdtd_sphere_range_controls'
PREFIX = 'tests/test_fd_sphere_range.py::'
MUTATIONS = [
    ('centre_only', 'geometry.py',
     'if c - radius < lo or c + radius > hi:', 'if c < lo or c > hi:',
     'test_range_is_finite_static_and_contains_the_whole_sphere'),
    ('nonfinite_range', 'geometry.py',
     'if not math.isfinite(value):', 'if False and not math.isfinite(value):',
     'test_range_is_finite_static_and_contains_the_whole_sphere'),
    ('moving_support', 'geometry.py',
     'bounds = self.range if self.range is not None else tuple(zip(self.low, self.high))',
     'bounds = tuple(zip(self.low, self.high))',
     'test_reserved_support_holds_volume_at_both_range_ends'),
    ('moving_skeleton', 'geometry.py',
     "if self.range is not None:\n            return tuple((f'sphere.range.{axis}'",
     "if False and self.range is not None:\n            return tuple((f'sphere.range.{axis}'",
     'test_sweep_reuses_plan_and_matches_unranged_fields_and_volume[numpy]'),
    ('reuse_wrong_range', '_lower/shape_ranges.py',
     'if records(scene) != plan.shape_ranges:', 'if False:',
     'test_lower_refuses_a_different_range_even_when_the_cells_match'),
    ('skip_actual_geometry_checks', '_run.py',
     'p, rows = _planned(scene, reuse=plan)',
     'p, rows = (plan, []) if plan is not None else _planned(scene, reuse=plan)',
     'test_reused_plan_still_checks_the_actual_sphere'),
    ('stencil_sweep', 'geometry.py',
     'if (self.discretisation == STENCIL and isinstance(self.shape, Sphere)',
     'if (False and self.discretisation == STENCIL and isinstance(self.shape, Sphere)',
     'test_range_refuses_a_prepared_stencil'),
    ('drop_radius_leaf', 'geometry.py',
     "radius = static_float('Sphere', 'radius', self.radius)", 'radius = .25',
     'test_range_does_not_silently_detach_a_shape_derivative'),
    ('empty_range_moves_junction_hash', '_lower/plan.py',
     "for name in ('junctions', 'shape_ranges'):", "for name in ('junctions',):",
     'tests/test_fd_junction.py::test_no_junction_never_classifies_or_changes_the_skeleton'),
    ('empty_range_moves_scene_hash', '_lower/plan.py',
     "for name in ('junctions', 'shape_ranges'):", "for name in ('junctions',):",
     'tests/test_fd_scene.py::test_mutant_a_skeleton_blind_to_one_static_field_lets_two_scenes_collide'),
]


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    records = []
    for name, relative, before, after, test in MUTATIONS:
        target = test if test.startswith('tests/') else PREFIX+test
        source = ROOT/'wavebit/fd'/relative
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
