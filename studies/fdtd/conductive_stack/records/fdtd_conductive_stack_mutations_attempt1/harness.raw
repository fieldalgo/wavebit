"""Ten actual assertion witnesses across nine functions, in an exclusive byte-restoring source window.

Every attempt, replacement, source snapshot, raw pytest log and exit is retained.
Only assertion failures count; syntax/import/fixture/execution errors do not.
Invocation requires an already passed targeted baseline and explicit coordination.
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
import uuid

from studies import fdtd_conductive_stack_probe as probe
from studies import fdtd_conductive_stack_support as support


ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT/'studies'/'results'
PREFIX = 'fdtd_conductive_stack'
TEST = 'tests/test_fd_conductive_stack.py'
TESTS = (TEST,)
SCRATCH = ROOT.parent/'fd-conductive-stack-work'


def mutations():
    return [('later-member-real-profile-wrong',
      'wavebit/fd/_lower/conductive_stack.py',
      '        eps[cut] = member_eps[cut]\n',
      '        eps[cut] = member_eps[cut] if film is stack.films[0] else 1.125*member_eps[cut]\n',
      'tests/test_fd_conductive_stack.py::test_every_stack_member_matches_the_independent_discrete_series_parallel_law[pair-z]'),
     ('later-memory-allocation-omitted',
      'wavebit/fd/_check.py',
      '                              for t in p.terms if isinstance(t, PoleTerm)) * itemsize\n',
      '                              for t in p.terms if isinstance(t, PoleTerm) and getattr(t, '
      "'member_index', 0) == 0) * itemsize\n",
      'tests/test_fd_conductive_stack.py::test_stack_regions_preserve_singletons_order_and_actual_state_allocation[pair-numpy]'),
     ('later-member-storage-weight-omitted',
      'wavebit/fd/_lower/conductive_stack.py',
      "    return {f'conductive_stack[{i}]':\n"
      "            single.pole_params(_member_scene(scene, i), plan, rt)['conductive[0]']\n"
      '            for i in range(len(stack.films))}\n',
      "    out = {f'conductive_stack[{i}]':\n"
      "           single.pole_params(_member_scene(scene, i), plan, rt)['conductive[0]']\n"
      '           for i in range(len(stack.films))}\n'
      "    out['conductive_stack[1]']['energy_inverse_delta'] = (rt.array(0.0, 'float64'),)\n"
      '    return out\n',
      'tests/test_fd_conductive_stack.py::test_stack_observer_sums_local_storage_and_closes_the_yee_loss_budget[pair-z-numpy]'),
     ('later-member-bulk-conductivity-omitted',
      'wavebit/fd/_lower/conductive_stack.py',
      '        sigma[cut] = member_sigma[cut]\n',
      '        sigma[cut] = member_sigma[cut] if film is stack.films[0] else 0.0\n',
      'tests/test_fd_conductive_stack.py::test_oblique_conductive_stack_matches_independent_multilayer_on_each_backend[lossy-pair-numpy]'),
     ('padded-support-overlap-admitted',
      'wavebit/fd/_lower/conductive_stack.py',
      '                 if left[0][1] > right[0][0]), None)\n',
      '                 if False and left[0][1] > right[0][0]), None)\n',
      'tests/test_fd_conductive_stack.py::test_stack_admission_requires_every_member_and_disjoint_padded_support[off-grid-two-dx]'),
     ('whole-prepared-identity-bypassed',
      'wavebit/fd/_lower/conductive_stack.py',
      '    if held != expected or (expected and all_poles != expected):\n',
      '    if False and (held != expected or (expected and all_poles != expected)):\n',
      'tests/test_fd_conductive_stack.py::test_prepared_stack_plan_binds_the_entire_ordered_scene_and_memory[second-sigma]'),
     ('foreign-ordinary-memory-ignored',
      'wavebit/fd/_lower/conductive_stack.py',
      '    if held != expected or (expected and all_poles != expected):\n',
      '    if held != expected:\n',
      'tests/test_fd_conductive_stack.py::test_prepared_stack_plan_binds_the_entire_ordered_scene_and_memory[ordinary-memory]'),
     ('torch-forward-carrier-detached',
      'wavebit/fd/_lower/shape_ranges.py',
      '    return _shape_traced(value)\n',
      '    return False\n',
      'tests/test_fd_conductive_stack.py::test_traced_later_members_and_host_cannot_bypass_the_stack_door[second-sigma-torch-forward]'),
     ('cfl-only-first-member',
      'wavebit/fd/_lower/conductive_stack.py',
      '    return max(float(np.max(single._profiles(film, plan.grid, plan.dt)[-1]))\n'
      '               for film in stack.films)\n',
      '    return max(float(np.max(single._profiles(film, plan.grid, plan.dt)[-1]))\n'
      '               for film in stack.films[:1])\n',
      'tests/test_fd_conductive_stack.py::test_stack_cfl_uses_the_maximum_local_member_rate_and_physical_curl_floor[later-fast]'),
     ('source-band-only-first-member',
      'wavebit/fd/_lower/conductive_stack.py',
      "        bounds = [(f'structures[{i}]', _band_bound(film, low, high))\n"
      '                  for i, film in enumerate(stack.films)]\n',
      "        bounds = [(f'structures[{i}]', _band_bound(film, low, high))\n"
      '                  for i, film in enumerate(stack.films[:1])]\n',
      'tests/test_fd_conductive_stack.py::test_stack_spectral_guards_check_every_member_and_the_full_source_band[band-second]')]

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
    env['PYTHONPYCACHEPREFIX'] = str(SCRATCH/('pycache-'+cache_tag+'-'+uuid.uuid4().hex))
    temporary = SCRATCH/('pytest-'+cache_tag+'-'+uuid.uuid4().hex)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    verbosity = '-q' if '--collect-only' in targets else '-vv'
    command = [sys.executable, '-B', '-m', 'pytest', verbosity, '-p', 'no:cacheprovider',
               '--tb=short', '--basetemp', str(temporary), *targets]
    begin = time.perf_counter()
    with log.open('xb') as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 check=False)
    return process.returncode, time.perf_counter()-begin, command


def run(attempt, baseline_log):
    support.start()
    targeted = support.load(RESULTS/(PREFIX+'_targeted.json'))
    assert targeted['complete']
    support.verify_guard(targeted['guard_before'])
    guard_before = support.guard()
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
        with (directory/(label+'.raw')).open('xb') as stream:
            stream.write(source.read_bytes())
    manifest = {'attempt': attempt, 'started_utc': now(), 'source_identity_before': identity,
                'harness_sha256': digest(harness.read_bytes()), 'test_sha256': digest((ROOT/TEST).read_bytes()),
                'test_file_sha256': {name: digest((ROOT/name).read_bytes()) for name in TESTS},
                'baseline_log': str(baseline_log.relative_to(ROOT)),
                'baseline_log_sha256': digest(baseline_log.read_bytes()),
                'guard_before': guard_before,
                'witnesses': [], 'restored': False, 'passed': False}
    write(manifest_path, manifest, exclusive=True)
    # Check all replacement anchors before making the first temporary source edit.
    for name, relative, before, _after, _target in mutations():
        text = (ROOT/relative).read_text(encoding='utf-8')
        assert text.count(before) == 1, (name, text.count(before))
    collection = directory/'targeted_collection.log'
    code, elapsed, command = pytest_run([*TESTS, '--collect-only'], collection, f'attempt{attempt}-collection')
    output = collection.read_text(encoding='utf-8')
    nodes = [line for line in output.splitlines() if line.startswith(tuple(name+'::' for name in TESTS))]
    manifest['collection'] = {'raw_log': str(collection.relative_to(ROOT)),
        'raw_log_sha256': digest(collection.read_bytes()), 'exit_code': code,
        'wall_s': elapsed, 'command': command, 'collected': len(nodes)}
    write(manifest_path, manifest)
    assert code == 0 and nodes and len(nodes) == len(set(nodes))
    assert len({node.split('::')[1].split('[')[0] for node in nodes}) == 9
    assert not any(value in output for value in ('INTERNALERROR', 'crashed', 'ERROR collecting'))
    assert all(target in nodes for *_rest, target in mutations())
    for name, relative, before, after, target in mutations():
        assert probe.source_identity() == identity, 'Production changed outside exclusive window.'
        support.verify_guard(guard_before)
        source = ROOT/relative
        original = source.read_bytes()
        text = original.decode('utf-8').replace('\r\n', '\n')
        assert text.count(before) == 1, (name, text.count(before))
        changed = text.replace(before, after)
        if b'\r\n' in original:
            changed = changed.replace('\n', '\r\n')
        changed = changed.encode('utf-8')
        original_path, mutant_path = directory/(name+'.original.raw'), directory/(name+'.mutant.raw')
        with original_path.open('xb') as stream:
            stream.write(original)
        with mutant_path.open('xb') as stream:
            stream.write(changed)
        log = directory/(name+'.log')
        print(f'{now()} START {name}', flush=True)
        launched_error = None
        try:
            source.write_bytes(changed)
            code, elapsed, command = pytest_run([target], log, f'attempt{attempt}-{name}')
        except BaseException as error:
            launched_error = repr(error)
            raise
        finally:
            source.write_bytes(original)
            assert source.read_bytes() == original, f'Byte restoration failed: {name}'
            if launched_error is not None:
                manifest['launcher_error'] = {'mutation': name, 'error': launched_error}
                manifest['restored'] = probe.source_identity() == identity
                write(manifest_path, manifest)
        output = log.read_text(encoding='utf-8')
        assertion = assertion_failure(output)
        try:
            support.check_log(output)
            node_status = support.reports(output)
            totals = support.summary(output)
            accepted = (code == 1 and assertion and node_status == {target: 'failed'}
                        and totals['failed'] == 1 and sum(totals.values()) == 1)
        except ValueError:
            node_status, totals, accepted = {}, {}, False
        manifest['witnesses'].append({
            'name': name, 'production_file': relative, 'test': target,
            'replacement': {'before': before, 'after': after},
            'original_sha256': digest(original), 'mutant_sha256': digest(changed),
            'restored_sha256': digest(source.read_bytes()),
            'original_snapshot': str(original_path.relative_to(ROOT)),
            'mutant_snapshot': str(mutant_path.relative_to(ROOT)),
            'raw_log': str(log.relative_to(ROOT)), 'raw_log_sha256': digest(log.read_bytes()),
            'exit_code': code, 'wall_s': elapsed, 'command': command,
            'assertion_failure': assertion, 'accepted': accepted,
            'node_status': node_status, 'totals': totals})
        manifest['restored'] = probe.source_identity() == identity
        write(manifest_path, manifest)
        print(f'{now()} END {name} exit={code} accepted={accepted} restored={manifest["restored"]}', flush=True)
    log = directory/'restored_targeted_baseline.log'
    code, elapsed, command = pytest_run(list(TESTS), log, f'attempt{attempt}-restored')
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
    baseline_status = support.reports(output)
    assert baseline_status == dict.fromkeys(nodes, 'passed')
    manifest['final_baseline']['node_status'] = baseline_status
    support.verify_guard(guard_before)
    assert manifest['test_file_sha256'] == {name: digest((ROOT/name).read_bytes()) for name in TESTS}
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
