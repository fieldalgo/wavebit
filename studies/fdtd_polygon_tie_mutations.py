"""Exercise each polygon-tie regression with a real source mutation, then restore.

Failure logs include surviving controls; only assertion failures count as witnesses.
Run with optional mutation names to select individual controls.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXACT = 'wavebit/fd/_shapes/_exact.py'
BODIES = 'wavebit/fd/_shapes/_bodies.py'
TESTS = 'tests/test_fd_polygon_ties.py::'
MUTANTS = [
    ('whole_boundary_cap', EXACT,
     'return xp.where(xp.abs(distance) <= tol, .5,',
     'return xp.where(xp.abs(distance) <= tol, 1.,',
     'test_boundary_faces_share_area_without_a_preferred_axis[x-numpy]'),
    ('whole_boundary_edge', EXACT,
     '.5*share, share)', 'share, share)',
     'test_translated_boundary_moments_keep_the_same_geometric_answer[1.0-numpy]'),
    ('first_nearest_face', BODIES,
     'g = xp.where((count > 1)[..., None], bisector, g)',
     'g = xp.where((count > 999)[..., None], bisector, g)',
     'test_equal_distance_normals_do_not_depend_on_the_first_vertex[1.0-numpy]'),
    ('first_nearest_face_fields', BODIES,
     'g = xp.where((count > 1)[..., None], bisector, g)',
     'g = xp.where((count > 999)[..., None], bisector, g)',
     'test_a_full_turn_keeps_the_same_coupled_fields[numpy]'),
    ('exact_boundary_comparison', EXACT,
     'boundary_tol = 32*xp.finfo(vertices.dtype).eps*xp.maximum(coordinate,',
     'boundary_tol = 0*xp.finfo(vertices.dtype).eps*xp.maximum(coordinate,',
     'test_boundary_aligned_mirror_reproduces_the_full_box[numpy]'),
    ('blur_distinct_faces', BODIES,
     'tie_slack = on_edge*(2*xp.where(live, norm, zero)+on_edge)',
     'tie_slack = 1e8*on_edge*(2*xp.where(live, norm, zero)+1e8*on_edge)',
     'test_equal_distance_normals_do_not_depend_on_the_first_vertex[1.0-numpy]'),
    ('bisector_support_plane', BODIES,
     'lambda q: self._transverse_surface(q, bisect=False)',
     'lambda q: self._transverse_surface(q, bisect=True)',
     'test_equal_distance_normals_do_not_depend_on_the_first_vertex[1.0-numpy]'),
    ('proxy_in_csg', BODIES,
     'faces = [body.support_surface(points) for body in self.bodies]',
     'faces = [body.surface(points) for body in self.bodies]',
     'tests/test_fd_shapes.py::test_a_union_of_flush_bodies_reads_solid_material_on_the_seam'),
]


def main():
    selected = set(sys.argv[1:])
    logs = ROOT/'studies/results/fdtd_polygon_tie_mutations'
    logs.mkdir(parents=True, exist_ok=True)
    manifest = logs/'witnesses.json'
    records = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
    for name, filename, before, after, target in MUTANTS:
        if selected and name not in selected:
            continue
        path = ROOT/filename
        original = path.read_bytes()
        try:
            source = path.read_text(encoding='utf-8')
            assert source.count(before) == 1, (name, source.count(before))
            path.write_text(source.replace(before, after), encoding='utf-8')
            test = target if target.startswith('tests/') else TESTS+target
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p',
                                     'no:cacheprovider', test],
                                    cwd=ROOT, capture_output=True, text=True)
            output = result.stdout+'\n'+result.stderr
            (logs/(name+'.log')).write_text(output, encoding='utf-8')
        finally:
            path.write_bytes(original)
        assert path.read_bytes() == original
        record = dict(mutation=name, test=test, exit=result.returncode,
                      source={filename: hashlib.sha256(original).hexdigest()})
        records = [r for r in records if r['mutation'] != name]+[record]
        manifest.write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
        assert result.returncode == 1 and 'AssertionError' in output, (name, output[-2000:])
        assert 'INTERNALERROR' not in output and 'ERROR collecting' not in output, name
        print(name, 'killed; source bytes restored', flush=True)
    print(len(records), 'controls recorded', flush=True)


if __name__ == '__main__':
    main()
