"""Seven actual source witnesses, exclusive window, exact finally restoration.

The witness cases must first appear in actual collection. Every mutated
child runs in a fresh bytecode-cache namespace. Only a single collected
assertion failure is accepted; launch/import/fixture/runtime errors are not.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
from pathlib import Path
import re
import sys
import uuid

from studies.fdtd.state_bytes.fdtd_state_bytes_support import ROOT, RESULTS, PREFIX, NEW_TEST, load, require, guard, verify_guard, identity, attempt_record, write, run_command, nodes, reports, summary, check_log, snapshot, digest, relative, pytest_pair, finish

MUTATIONS = [{'id': 'omit-pole-memory',
  'before': '                                   + polarization_memory),',
  'after': '                                   + 0),',
  'defect': 'Reproduce the existing missing top-level polarization memory.',
  'expected_witness': 'Every pole-bearing source-free case has strictly positive omitted array payload.',
  'case': 'voxel-mixed-real-numpy'},
 {'id': 'omit-one-history',
  'before': 'polarization_memory = sum(9 * t.n_poles * np.prod(t.region.shape)',
  'after': 'polarization_memory = sum(6 * t.n_poles * np.prod(t.region.shape)',
  'defect': 'Count only two of the three allocated polarization histories.',
  'expected_witness': 'The actual initialized P/previous/delta tree has more bytes than this estimate.',
  'case': 'voxel-mixed-real-numpy'},
 {'id': 'exclude-film-subclasses',
  'before': 'for t in p.terms if isinstance(t, PoleTerm)) * itemsize',
  'after': 'for t in p.terms if type(t) is PoleTerm) * itemsize',
  'defect': 'Count only exact ordinary PoleTerm objects, losing every film subclass.',
  'expected_witness': 'Conductive, Debye and oscillator film cases still initialize their full '
                      'memories.',
  'case': 'conductive-film-real-numpy'},
 {'id': 'count-one-pole-only',
  'before': 'polarization_memory = sum(9 * t.n_poles * np.prod(t.region.shape)',
  'after': 'polarization_memory = sum(9 * np.prod(t.region.shape)',
  'defect': 'Ignore declared pole count, including reserved collapsed/inactive slots.',
  'expected_witness': 'Mixed ordinary, collapsed two/general Debye, and oscillator-sum allocations '
                      'retain multiple slots.',
  'case': 'two-debye-collapsed-film-real-numpy'},
 {'id': 'replace-compact-support-by-domain',
  'before': 'polarization_memory = sum(9 * t.n_poles * np.prod(t.region.shape)',
  'after': 'polarization_memory = sum(9 * t.n_poles * p.cells',
  'defect': 'Charge every compact pole term for the whole domain.',
  'expected_witness': 'Every constructed support is smaller than the domain in at least one spatial '
                      'dimension.',
  'case': 'voxel-mixed-real-numpy'},
 {'id': 'force-real-pole-itemsize',
  'before': 'for t in p.terms if isinstance(t, PoleTerm)) * itemsize',
  'after': 'for t in p.terms if isinstance(t, PoleTerm)) * 8',
  'defect': 'Use float64 element width for Bloch-complex polarization arrays.',
  'expected_witness': 'Complex pole-bearing allocations have twice the real element payload.',
  'case': 'voxel-mixed-complex-numpy'},
 {'id': 'count-first-term-only',
  'before': 'for t in p.terms if isinstance(t, PoleTerm)) * itemsize',
  'after': 'for t in p.terms[:1] if isinstance(t, PoleTerm)) * itemsize',
  'defect': 'Omit material memory after the first term.',
  'expected_witness': 'The two-voxel-regions scene initializes two disjoint nonempty material memory '
                      'trees.',
  'case': 'two-voxel-regions-real-numpy'}]
SOURCE = 'wavebit/fd/_check.py'
FUNCTION = 'test_report_counts_the_initialized_main_state'


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
        snapshot(ROOT/NEW_TEST, directory/'test_fd_state_bytes.py.txt.gz')])
    write(manifest, record)
    common = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider']
    row, collection = run_command(record, manifest, 'new_collection',
        common+[NEW_TEST, '--collect-only'], directory)
    require(row['exit'] == 0, 'New-file collection failed.')
    collected = nodes(collection)
    require(len(collected) == 72, 'Expected 72 actual allocation cases before mutations.')
    require(all(NEW_TEST+'::'+FUNCTION+'['+m['case']+']' in collected for m in MUTATIONS),
            'Planned witness does not exist in actual collection.')
    for item in MUTATIONS:
        verify_guard(record['guard_before'])
        require(source.read_bytes() == original, 'Source changed before temporary mutation.')
        changed = text.replace(item['before'], item['after'])
        if b'\r\n' in original:
            changed = changed.replace('\n', '\r\n')
        changed = changed.encode('utf-8')
        node = NEW_TEST+'::'+FUNCTION+'['+item['case']+']'
        original_path = directory/(item['id']+'.original.txt')
        mutant_path = directory/(item['id']+'.mutant.txt')
        for path, value in ((original_path, original), (mutant_path, changed)):
            with path.open('xb') as stream:
                stream.write(value)
        witness = {'name': item['id'], 'node': node, 'source': SOURCE, 'accepted': False,
                   'replacement': {'before': item['before'], 'after': item['after']},
                   'original': {'file': relative(original_path), 'sha256': digest(original)},
                   'mutant': {'file': relative(mutant_path), 'sha256': digest(changed)}}
        record['witnesses'].append(witness)
        write(manifest, record)
        temp = ROOT/'.state_bytes_work'/('mutant-'+item['id']+'-'+uuid.uuid4().hex)
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
    baseline = next(c for c in targeted['checks'] if c['name'] == 'new_allocation')['observed']
    require(restored['observed'] == baseline, 'Restored allocation baseline differs from original baseline.')
    require(len(record['witnesses']) == 7 and all(w['accepted'] for w in record['witnesses']),
            'At least one actual mutation survived or failed improperly; preserve attempt.')
    finish('mutations', manifest, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    main(parser.parse_args().attempt)
