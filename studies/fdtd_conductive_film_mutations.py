"""Exclusive, byte-restoring production mutation witnesses for the nine C8 tests.

Run only while no other process edits production or runs field solves/tests. Each
accepted witness must be pytest's assertion failure or DID NOT RAISE, not an import,
syntax, fixture, or incidental execution error. Original and changed source bytes,
replacement text, raw logs, exit codes and restoration hashes are retained.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

from studies.fdtd_conductive_film_probe import source_identity


ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'studies' / 'results'
TEST = 'tests/test_fd_conductive_film.py'
FILM = 'wavebit/fd/_lower/conductive_film.py'


def mutations():
    return [
        ('normal-memory-drive-omitted', FILM,
         'drive[component, live] = 4*fl*b*b*conductivity*dt/(D*D)',
         'drive[component, live] = 0.0*fl',
         'test_lowered_response_is_the_complex_series_parallel_law[4.0-8.0-1.0-0.03-z]'),
        ('lossless-tangential-epsilon-shifted', FILM,
         'eps[component] = f*a+g*b', 'eps[component] = f*a+g*b + .125*f*g',
         'test_zero_conductivity_is_the_existing_lossless_box[z]'),
        ('interface-memory-energy-omitted', 'wavebit/fd/_kernel/observe.py',
         'total = total + rt.weighted_energy(previous + c * field, inv_delta)',
         'total = total + rt.weighted_energy(previous + c * field, 0.0 * inv_delta)',
         'test_interface_energy_uses_local_weights_before_reducing[z-numpy]'),
        ('tangential-conductivity-omitted', FILM,
         'sigma[component] = f*conductivity', 'sigma[component] = 0.0*f*conductivity',
         'test_the_oblique_tm_film_matches_complex_airy_on_each_backend[numpy]'),
        ('initial-field-door-removed', FILM,
         "if len(structures) != 1 or getattr(scene, 'initial', None) is not None:",
         'if len(structures) != 1:',
         'test_unsupported_smoothed_lossy_scenes_are_refused_by_name[initial]'),
        ('prepared-signature-check-removed', FILM,
         'if held != expected:', 'if False and held != expected:',
         'test_prepared_film_plan_refuses_changed_material_or_geometry[sigma]'),
        ('traced-conductivity-silently-detached', 'wavebit/fd/_lower/params.py',
         '    refuse_film_mismatch(scene, plan)',
         "    scene = scene.replace(structures=tuple(\n"
         "        s.replace(material=type(s.material)(eps=s.material.eps,\n"
         "            sigma=tuple(float(v.detach()) if hasattr(v, 'detach') else v\n"
         "                        for v in s.material.sigma), mu=s.material.mu))\n"
         "        if type(s.material).__name__ == 'Conductor' else s\n"
         "        for s in scene.structures))\n"
         '    refuse_film_mismatch(scene, plan)',
         'test_traced_conductivity_cannot_bypass_the_prepared_film_door[torch-reverse]'),
        ('induced-pole-cfl-rate-omitted', FILM,
         'return float(np.max(rate))', 'return 0.0 * float(np.max(rate))',
         'test_derived_normal_relaxation_tightens_the_planning_bound'),
        ('conductivity-index-omitted', 'wavebit/fd/_lower/checks.py',
         'if conductive and isinstance(material, Conductor):',
         'if False and conductive and isinstance(material, Conductor):',
         'test_conductivity_is_counted_in_the_source_band_resolution_floor'),
    ]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pytest_run(targets, path, cache_tag):
    env = dict(os.environ)
    env['PYTHONPYCACHEPREFIX'] = str(ROOT / '.c8_mutation_work' / cache_tag)
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short', *targets]
    begin = time.perf_counter()
    with path.open('xb') as stream:
        run = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                             check=False)
    return run.returncode, time.perf_counter() - begin, command


def assertion_failure(text):
    return bool(re.search(r'AssertionError|Failed: DID NOT RAISE|^E\s+assert\b',
                          text, flags=re.MULTILINE))


def run(attempt):
    directory = RESULTS / f'fdtd_conductive_film_mutations_attempt{attempt}'
    directory.mkdir(exist_ok=False)
    path = RESULTS / f'fdtd_conductive_film_mutations_attempt{attempt}.json'
    assert not path.exists()
    baseline = source_identity()
    manifest = {'attempt': attempt, 'source_identity_before': baseline,
                'harness_sha256': digest(pathlib.Path(__file__).read_bytes()),
                'test_sha256': digest((ROOT / TEST).read_bytes()),
                'baseline_log': 'studies/results/fdtd_conductive_film_controls/targeted_attempt6.log',
                'witnesses': [], 'restored': False}
    save(path, manifest)
    for name, relative, before, after, target in mutations():
        assert source_identity() == baseline, 'production changed outside this exclusive window'
        source = ROOT / relative
        original = source.read_bytes()
        decoded = original.decode('utf-8').replace('\r\n', '\n')
        assert decoded.count(before) == 1, (name, decoded.count(before))
        replacement = decoded.replace(before, after)
        if b'\r\n' in original:
            replacement = replacement.replace('\n', '\r\n')
        changed = replacement.encode('utf-8')
        original_path = directory / f'{name}.original.txt'
        mutant_path = directory / f'{name}.mutant.txt'
        original_path.write_bytes(original)
        mutant_path.write_bytes(changed)
        log = directory / f'{name}.log'
        print(f'START {name}', flush=True)
        try:
            source.write_bytes(changed)
            code, elapsed, command = pytest_run([TEST + '::' + target], log,
                                                 f'attempt{attempt}-{name}')
        finally:
            source.write_bytes(original)
            assert source.read_bytes() == original, f'byte restoration failed: {name}'
        text = log.read_text(encoding='utf-8')
        assertion = assertion_failure(text)
        accepted = (code == 1 and assertion and '1 failed' in text
                    and not re.search(r'\d+ error(?:s)?(?:,| in|\s*$)', text, flags=re.MULTILINE))
        witness = {'name': name, 'production_file': relative, 'test': target,
                   'replacement': {'before': before, 'after': after},
                   'original_sha256': digest(original), 'mutant_sha256': digest(changed),
                   'restored_sha256': digest(source.read_bytes()),
                   'original_snapshot': str(original_path.relative_to(ROOT)),
                   'mutant_snapshot': str(mutant_path.relative_to(ROOT)),
                   'raw_log': str(log.relative_to(ROOT)), 'raw_log_sha256': digest(log.read_bytes()),
                   'exit_code': code, 'wall_s': elapsed, 'command': command,
                   'assertion_failure': assertion, 'accepted': accepted}
        manifest['witnesses'].append(witness)
        manifest['restored'] = source_identity() == baseline
        save(path, manifest)
        print(f'END {name} exit={code} accepted={accepted} restored={manifest["restored"]}', flush=True)
    final = directory / 'restored_targeted_baseline.log'
    code, elapsed, command = pytest_run([TEST], final, f'attempt{attempt}-restored')
    manifest['final_baseline'] = {'raw_log': str(final.relative_to(ROOT)),
                                  'raw_log_sha256': digest(final.read_bytes()),
                                  'exit_code': code, 'wall_s': elapsed, 'command': command}
    manifest['source_identity_after'] = source_identity()
    manifest['restored'] = manifest['source_identity_after'] == baseline
    manifest['all_nine_assertion_witnesses'] = (len(manifest['witnesses']) == 9
                                               and all(w['accepted'] for w in manifest['witnesses']))
    manifest['passed'] = manifest['all_nine_assertion_witnesses'] and manifest['restored'] and code == 0
    save(path, manifest)
    if manifest['passed']:
        accepted_path = RESULTS / 'fdtd_conductive_film_mutations.json'
        assert not accepted_path.exists()
        save(accepted_path, manifest)
    print(json.dumps({'passed': manifest['passed'], 'restored': manifest['restored'],
                      'witnesses': len(manifest['witnesses']), 'manifest': str(path.relative_to(ROOT))}))
    return 0 if manifest['passed'] else 1


def audit(attempt):
    """Recheck preserved raw witnesses without rewriting a rejected original manifest."""
    path = RESULTS / f'fdtd_conductive_film_mutations_attempt{attempt}.json'
    original_bytes = path.read_bytes()
    manifest = json.loads(original_bytes)
    original_harness = RESULTS / f'fdtd_conductive_film_mutations_attempt{attempt}' / 'harness_original.py.txt'
    assert digest(original_harness.read_bytes()) == manifest['harness_sha256']
    baseline = manifest['source_identity_before']
    assert source_identity() == baseline == manifest['source_identity_after']
    assert manifest['restored'] is True
    assert digest((ROOT / TEST).read_bytes()) == manifest['test_sha256']
    reviewed = []
    for item in manifest['witnesses']:
        before = (ROOT / item['original_snapshot']).read_bytes()
        mutant = (ROOT / item['mutant_snapshot']).read_bytes()
        assert digest(before) == item['original_sha256'] == item['restored_sha256']
        assert digest((ROOT / item['production_file']).read_bytes()) == item['restored_sha256']
        assert digest(mutant) == item['mutant_sha256']
        replacement = item['replacement']
        normalized = before.decode('utf-8').replace('\r\n', '\n')
        assert normalized.count(replacement['before']) == 1
        assert normalized.replace(replacement['before'], replacement['after']) \
            == mutant.decode('utf-8').replace('\r\n', '\n')
        raw = (ROOT / item['raw_log']).read_bytes()
        assert digest(raw) == item['raw_log_sha256']
        text = raw.decode('utf-8')
        assert item['exit_code'] == 1 and assertion_failure(text) and '1 failed' in text
        assert not re.search(r'\d+ error(?:s)?(?:,| in|\s*$)', text, flags=re.MULTILINE)
        reviewed.append({**item, 'original_matcher_accepted': item['accepted'],
                         'assertion_failure': True, 'accepted': True})
    assert len(reviewed) == len({item['test'].split('[')[0] for item in reviewed}) == 9
    final = manifest['final_baseline']
    raw = (ROOT / final['raw_log']).read_bytes()
    assert digest(raw) == final['raw_log_sha256'] and final['exit_code'] == 0
    assert re.search(r'53 passed', raw.decode('utf-8'))
    accepted = {**manifest, 'witnesses': reviewed, 'passed': True,
                'all_nine_assertion_witnesses': True,
                'acceptance_audit': {
                    'audited_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'original_manifest': str(path.relative_to(ROOT)),
                    'original_manifest_sha256': digest(original_bytes),
                    'original_harness_snapshot': str(original_harness.relative_to(ROOT)),
                    'audit_harness_sha256': digest(pathlib.Path(__file__).read_bytes()),
                    'scope': 'Original matcher decisions and manifest remain unchanged. '
                             'Three pytest-rewritten E assert failures are accepted after '
                             'checking original/mutant/restored hashes, replacements, raw '
                             'logs, exit1, one failure each and the restored 53-pass baseline. '
                             'No field solve or mutation was rerun by this audit.'}}
    accepted_path = RESULTS / 'fdtd_conductive_film_mutations.json'
    assert not accepted_path.exists()
    save(accepted_path, accepted)
    assert path.read_bytes() == original_bytes
    print(json.dumps({'audit': 'passed', 'accepted_witnesses': 9, 'restored_baseline': 53,
                      'source_restored': True, 'manifest': str(accepted_path.relative_to(ROOT))}))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--attempt', type=int)
    group.add_argument('--audit-attempt', type=int)
    options = parser.parse_args()
    raise SystemExit(run(options.attempt) if options.attempt is not None
                     else audit(options.audit_attempt))
