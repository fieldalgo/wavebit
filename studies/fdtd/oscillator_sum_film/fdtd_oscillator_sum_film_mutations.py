"""Ten actual assertion witnesses across nine functions, in an exclusive byte-restoring source window.

Every attempt, replacement, source snapshot, raw pytest log and exit is retained.
Only assertion failures count; syntax/import/fixture/execution errors do not.
Invocation requires an already passed targeted baseline and explicit coordination.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
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
import uuid

from studies.fdtd.oscillator_sum_film import fdtd_oscillator_sum_film_probe as probe


ROOT = repository_root()
RESULTS = ROOT/'studies'/'results'
PREFIX = 'fdtd_oscillator_sum_film'
TEST = 'tests/fdtd/test_fd_oscillator_sum_film.py'
FILM = 'wavebit/fd/_lower/oscillator_sum_film.py'


def mutations():
    return [
        ('rank-one-runtime-coupling-omitted', FILM,
         "- coefficients['coupling'][j] * total)",
         "- 0.0 * coefficients['coupling'][j] * total)",
         'test_actual_oscillator_sum_recurrence_is_the_declared_discrete_mixture[mixed-z]'),
        ('all-poles-use-first-damping', FILM,
         'd, damping, S = map(Fraction, (resonance, gamma, strength))',
         'd, damping, S = map(Fraction, (resonance, film.gammas[0], strength))',
         'test_actual_oscillator_sum_recurrence_is_the_declared_discrete_mixture[multi-z]'),
        ('zero-strength-lossless-permittivity-shifted', FILM,
         'return _number(eps, positive=True), rank_one, tuple(slots)',
         'return _number(eps+1 if not any(film.strengths) else eps, positive=True), rank_one, tuple(slots)',
         'test_oscillator_sum_endpoints_zero_strength_and_pole_order[all-zero-z]'),
        ('rank-one-cross-energy-omitted', 'wavebit/fd/_kernel/observe.py',
         'total = total + rt.real_inner(sum_new, sum_old, weight)',
         'total = total + 0.0 * rt.real_inner(sum_new, sum_old, weight)',
         'test_oscillator_sum_storage_and_actual_recurrence_dissipation[multi-z-numpy]'),
        ('last-memory-drive-quarter-strength', FILM,
         "    return {'oscillator_sum_film[0]': coefficients}",
         "    coefficients['drive'] = coefficients['drive'][:-1]+(.25*coefficients['drive'][-1],)\n"
         "    return {'oscillator_sum_film[0]': coefficients}",
         'test_oblique_oscillator_sum_matches_complex_airy_on_each_backend[mixed-metal-numpy]'),
        ('initial-field-door-removed', FILM,
         "if len(structures) != 1 or getattr(scene, 'initial', None) is not None:",
         'if len(structures) != 1:',
         'test_unsupported_oscillator_sum_films_are_refused_by_name[initial]'),
        ('prepared-signature-check-removed', FILM,
         'if held != expected or (expected and all_poles != expected):',
         'if False and (held != expected or (expected and all_poles != expected)):',
         'test_prepared_oscillator_sum_plan_refuses_changed_identity_or_memory[gamma2]'),
        ('traced-last-damping-silently-detached', 'wavebit/fd/_lower/params.py',
         '    refuse_dispersive_film_mismatch(scene, plan)',
         "    scene = scene.replace(structures=tuple(\n"
         "        s.replace(material=s.material.replace(poles=tuple(\n"
         "            p.replace(gamma=float(p.gamma.detach()))\n"
         "            if hasattr(p, 'gamma') and hasattr(p.gamma, 'detach') else p\n"
         "            for p in s.material.poles)))\n"
         "        if type(s.material).__name__ == 'Dispersive' else s\n"
         "        for s in scene.structures))\n"
         '    refuse_dispersive_film_mismatch(scene, plan)',
         'test_traced_oscillator_sum_parameters_cannot_bypass_the_concrete_door[gamma2-torch-reverse]'),
        ('last-cfl-strength-omitted', FILM,
         'sum(Fraction(S) for _, S in live)/Fraction(film.eps)',
         'sum(Fraction(S) for _, S in live[:-1])/Fraction(film.eps)',
         'test_oscillator_sum_cfl_uses_every_active_strength_and_largest_resonance[3]'),
        ('interior-spectral-peak-omitted', 'wavebit/fd/_lower/oscillator_sum_resolution.py',
         'candidates.append(critical)', 'pass  # mutant omits the interior candidate',
         'test_oscillator_sum_spectral_guard_covers_each_band_peak_and_sample[band-interior-peak]'),
    ]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value, *, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def assertion_failure(text):
    return bool(re.search(r'AssertionError|Failed: DID NOT RAISE|^E\s+assert\b', text, flags=re.MULTILINE))


def pytest_run(targets, log, cache_tag):
    env = dict(os.environ)
    env['PYTHONPYCACHEPREFIX'] = str(ROOT/'.oscillator_sum_mutation_work'/cache_tag)
    temporary = ROOT/'.oscillator_sum_mutation_work'/('pytest-'+cache_tag+'-'+uuid.uuid4().hex)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               '--tb=short', '--basetemp', str(temporary), *targets]
    begin = time.perf_counter()
    with log.open('xb') as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 check=False)
    return process.returncode, time.perf_counter()-begin, command


def run(attempt, baseline_log):
    baseline_log = ROOT/baseline_log
    baseline_text = baseline_log.read_text(encoding='utf-8')
    assert re.search(r'\d+ passed', baseline_text), 'A passed targeted baseline is required.'
    assert not re.search(r'\d+ failed|INTERNALERROR|ERROR collecting', baseline_text)
    directory = RESULTS/f'{PREFIX}_mutations_attempt{attempt}'
    directory.mkdir(exist_ok=False)
    manifest_path = RESULTS/f'{PREFIX}_mutations_attempt{attempt}.json'
    assert not manifest_path.exists()
    identity = probe.source_identity()
    harness = pathlib.Path(__file__)
    for label, source in (('harness', harness), ('tests', ROOT/TEST)):
        with (directory/(label+'.py.txt')).open('xb') as stream:
            stream.write(source.read_bytes())
    manifest = {'attempt': attempt, 'started_utc': now(), 'source_identity_before': identity,
                'harness_sha256': digest(harness.read_bytes()), 'test_sha256': digest((ROOT/TEST).read_bytes()),
                'baseline_log': str(baseline_log.relative_to(ROOT)),
                'baseline_log_sha256': digest(baseline_log.read_bytes()),
                'witnesses': [], 'restored': False, 'passed': False}
    write(manifest_path, manifest, exclusive=True)
    # Check all replacement anchors before making the first temporary source edit.
    for name, relative, before, _after, _target in mutations():
        text = (ROOT/relative).read_text(encoding='utf-8')
        assert text.count(before) == 1, (name, text.count(before))
    collection = directory/'targeted_collection.log'
    code, elapsed, command = pytest_run([TEST, '--collect-only'], collection, f'attempt{attempt}-collection')
    output = collection.read_text(encoding='utf-8')
    nodes = [line for line in output.splitlines() if line.startswith(TEST+'::')]
    manifest['collection'] = {'raw_log': str(collection.relative_to(ROOT)),
        'raw_log_sha256': digest(collection.read_bytes()), 'exit_code': code,
        'wall_s': elapsed, 'command': command, 'collected': len(nodes)}
    write(manifest_path, manifest)
    assert code == 0 and nodes and len(nodes) == len(set(nodes))
    assert len({node.split('::')[1].split('[')[0] for node in nodes}) == 9
    assert not any(value in output for value in ('INTERNALERROR', 'crashed', 'ERROR collecting'))
    assert all(TEST+'::'+target in nodes for *_rest, target in mutations())
    for name, relative, before, after, target in mutations():
        assert probe.source_identity() == identity, 'Production changed outside exclusive window.'
        source = ROOT/relative
        original = source.read_bytes()
        text = original.decode('utf-8').replace('\r\n', '\n')
        assert text.count(before) == 1, (name, text.count(before))
        changed = text.replace(before, after)
        if b'\r\n' in original:
            changed = changed.replace('\n', '\r\n')
        changed = changed.encode('utf-8')
        original_path, mutant_path = directory/(name+'.original.txt'), directory/(name+'.mutant.txt')
        with original_path.open('xb') as stream:
            stream.write(original)
        with mutant_path.open('xb') as stream:
            stream.write(changed)
        log = directory/(name+'.log')
        print(f'{now()} START {name}', flush=True)
        try:
            source.write_bytes(changed)
            code, elapsed, command = pytest_run([TEST+'::'+target], log, f'attempt{attempt}-{name}')
        finally:
            source.write_bytes(original)
            assert source.read_bytes() == original, f'Byte restoration failed: {name}'
        output = log.read_text(encoding='utf-8')
        assertion = assertion_failure(output)
        accepted = (code == 1 and assertion and '1 failed' in output
                    and not re.search(r'\d+ error(?:s)?(?:,| in|\s*$)', output, flags=re.MULTILINE))
        manifest['witnesses'].append({
            'name': name, 'production_file': relative, 'test': target,
            'replacement': {'before': before, 'after': after},
            'original_sha256': digest(original), 'mutant_sha256': digest(changed),
            'restored_sha256': digest(source.read_bytes()),
            'original_snapshot': str(original_path.relative_to(ROOT)),
            'mutant_snapshot': str(mutant_path.relative_to(ROOT)),
            'raw_log': str(log.relative_to(ROOT)), 'raw_log_sha256': digest(log.read_bytes()),
            'exit_code': code, 'wall_s': elapsed, 'command': command,
            'assertion_failure': assertion, 'accepted': accepted})
        manifest['restored'] = probe.source_identity() == identity
        write(manifest_path, manifest)
        print(f'{now()} END {name} exit={code} accepted={accepted} restored={manifest["restored"]}', flush=True)
    log = directory/'restored_targeted_baseline.log'
    code, elapsed, command = pytest_run([TEST], log, f'attempt{attempt}-restored')
    manifest['final_baseline'] = {'raw_log': str(log.relative_to(ROOT)),
                                 'raw_log_sha256': digest(log.read_bytes()),
                                 'exit_code': code, 'wall_s': elapsed, 'command': command}
    output = log.read_text(encoding='utf-8')
    counts = {kind: int(re.findall(r'(\d+) '+kind, output)[-1])
              if re.search(r'(\d+) '+kind, output) else 0
              for kind in ('passed', 'skipped', 'xfailed', 'xpassed', 'failed', 'error')}
    manifest['final_baseline'].update(counts, collected=len(nodes))
    clean_baseline = (code == 0 and counts['passed'] == len(nodes)
                      and sum(counts.values()) == len(nodes)
                      and not any(value in output for value in
                                  ('INTERNALERROR', 'crashed', 'node down', 'ERROR collecting')))
    manifest['source_identity_after'] = probe.source_identity()
    manifest['restored'] = manifest['source_identity_after'] == identity
    manifest['all_function_assertion_witnesses'] = (
        len(manifest['witnesses']) == 10
        and len({w['test'].split('[')[0] for w in manifest['witnesses']}) == 9
        and all(w['accepted'] for w in manifest['witnesses']))
    manifest['passed'] = manifest['all_function_assertion_witnesses'] and manifest['restored'] and clean_baseline
    manifest['ended_utc'] = now()
    write(manifest_path, manifest)
    if manifest['passed']:
        write(RESULTS/f'{PREFIX}_mutations.json', manifest, exclusive=True)
    print(json.dumps({'passed': manifest['passed'], 'restored': manifest['restored'],
                      'accepted_witnesses': sum(w['accepted'] for w in manifest['witnesses']),
                      'manifest': str(manifest_path.relative_to(ROOT))}))
    return 0 if manifest['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    parser.add_argument('--baseline-log', required=True, type=pathlib.Path)
    args = parser.parse_args()
    raise SystemExit(run(args.attempt, args.baseline_log))
