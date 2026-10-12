"""Three actual source witnesses, exclusive window, exact finally restoration.

The witness cases must first appear in actual collection. Every mutated
child runs in a fresh bytecode-cache namespace. Only a single collected
assertion failure is accepted; launch/import/fixture/runtime errors are not.
"""
import argparse
from pathlib import Path
import re
import sys
import uuid

from fdtd_inactive_lorentz_support import (ROOT, RESULTS, PREFIX, NEW_TEST, load, require,
    guard, verify_guard, identity, attempt_record, write, run_command, nodes,
    reports, summary, check_log, snapshot, digest, relative, pytest_pair, finish)

MUTATIONS = [{'id': 'inactive-shortcut-removed',
  'before': 'if (not torch_carried(self.strength) and concrete(self.strength) == 0',
  'after': 'if (False and not torch_carried(self.strength) and concrete(self.strength) == 0',
  'function': 'test_inactive_lorentz_analytic_identity_and_active_controls',
  'case': 'inactive-resonance-term-python',
  'defect': 'Evaluate an inactive denominator at its removable singularity.',
  'expected_witness': 'Caught arithmetic error fails exact inactive identity assertion.'},
 {'id': 'forward-dual-strength-detached',
  'before': 'not torch_carried(self.strength) and concrete(self.strength) == 0',
  'after': 'concrete(self.strength) == 0',
  'function': 'test_zero_primal_lorentz_derivatives_are_preserved',
  'case': 'torch-forward-strength-term',
  'defect': 'Read a carried zero strength as an inactive primal.',
  'expected_witness': 'Reciprocal-denominator strength tangent is incorrectly zero.'},
 {'id': 'inactive-zero-graph-disconnected',
  'before': 'return 0j * omega + 0j * self.omega_0 + 0j * self.gamma + 0j * self.strength',
  'after': 'return 0j',
  'function': 'test_zero_primal_lorentz_derivatives_are_preserved',
  'case': 'torch-reverse-omega0-term',
  'defect': 'Return a detached scalar instead of a connected inactive response.',
  'expected_witness': 'Explicit Tensor/requires_grad assertion fails.'}]
SOURCE = 'wavebit/fd/materials.py'
FUNCTIONS = ('test_inactive_lorentz_analytic_identity_and_active_controls', 'test_zero_primal_lorentz_derivatives_are_preserved')


def accepted_assertion(text, node, exit_code):
    check_log(text)
    actual = reports(text)
    totals = summary(text)
    errors = re.findall(r'^E\s+([\w.]+(?:Error|Exception))\b', text, flags=re.MULTILINE)
    assertion = bool(re.search(r'AssertionError|^E\s+assert\b|Failed: DID NOT RAISE',
                               text, flags=re.MULTILINE))
    return (exit_code == 1 and actual == {node: 'failed'} and totals['failed'] == 1
            and sum(totals.values()) == 1 and assertion
            and all(error == 'AssertionError' for error in errors))


def main(attempt):
    targeted = load(RESULTS/(PREFIX+'targeted.json'))
    require(targeted['complete'] and targeted['source_after'] == identity(), 'Need current clean targeted baseline.')
    verify_guard(targeted['guard_before'])
    directory, manifest, record = attempt_record('mutations', attempt)
    source = ROOT/SOURCE
    original = source.read_bytes()
    text = original.decode('utf-8').replace('\r\n', '\n')
    require(all(text.count(m['before']) == 1 for m in MUTATIONS), 'Mutation anchors are not unique.')
    record.update(witnesses=[], restored=False, snapshots=[
        snapshot(Path(__file__), directory/'harness.py.txt.gz'),
        snapshot(ROOT/NEW_TEST, directory/'test_fd_inactive_lorentz.py.txt.gz')])
    write(manifest, record)
    common = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider']
    row, collection = run_command(record, manifest, 'new_collection',
        common+[NEW_TEST, '--collect-only'], directory)
    require(row['exit'] == 0, 'New-file collection failed.')
    collected = nodes(collection)
    require(len(collected) == 127, 'Expected 127 actual analytic cases before mutations.')
    require(all(NEW_TEST+'::'+m['function']+'['+m['case']+']' in collected for m in MUTATIONS),
            'Planned witness does not exist in actual collection.')
    for item in MUTATIONS:
        verify_guard(record['guard_before'])
        require(source.read_bytes() == original, 'Source changed before temporary mutation.')
        changed = text.replace(item['before'], item['after'])
        if b'\r\n' in original:
            changed = changed.replace('\n', '\r\n')
        changed = changed.encode('utf-8')
        node = NEW_TEST+'::'+item['function']+'['+item['case']+']'
        original_path = directory/(item['id']+'.original.raw')
        mutant_path = directory/(item['id']+'.mutant.raw')
        for path, value in ((original_path, original), (mutant_path, changed)):
            with path.open('xb') as stream:
                stream.write(value)
        witness = {'name': item['id'], 'node': node, 'source': SOURCE, 'accepted': False,
                   'replacement': {'before': item['before'], 'after': item['after']},
                   'original': {'file': relative(original_path), 'sha256': digest(original)},
                   'mutant': {'file': relative(mutant_path), 'sha256': digest(changed)}}
        record['witnesses'].append(witness)
        write(manifest, record)
        temp = ROOT/'.inactive_lorentz_work'/('mutant-'+item['id']+'-'+uuid.uuid4().hex)
        temp.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.write_bytes(changed)
            row, output = run_command(record, manifest, item['id'], common+[
                node, '-vv', '--tb=short', '--basetemp', str(temp)], directory,
                cache_tag=f'attempt{attempt}-{item["id"]}')
            witness['check_name'] = row['name']
            witness['accepted'] = accepted_assertion(output, node, row['exit'])
        except BaseException as error:
            witness['error'] = repr(error)
            raise
        finally:
            source.write_bytes(original)
            witness['restored_sha256'] = digest(source.read_bytes())
            record['restored'] = source.read_bytes() == original and guard() == record['guard_before']
            write(manifest, record)
            require(record['restored'], 'Exact source restoration failed; stop all work.')
    restored = pytest_pair(record, manifest, 'restored_baseline', [NEW_TEST], directory)
    baseline = next(c for c in targeted['checks'] if c['name'] == 'new_analytic')['observed']
    require(restored['observed'] == baseline, 'Restored analytic baseline differs from original baseline.')
    require({w['node'].split('::')[1].split('[')[0] for w in record['witnesses']} == set(FUNCTIONS),
            'Every new test function needs an actual assertion witness.')
    require(len(record['witnesses']) == 3 and all(w['accepted'] for w in record['witnesses']),
            'At least one actual mutation survived or failed improperly; preserve attempt.')
    finish('mutations', manifest, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    main(parser.parse_args().attempt)
