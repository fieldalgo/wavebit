"""Read-only audit of saved allocation/report gates; never run a solver or pytest.

The only writes are this audit's new attempt/canonical records. It checks
raw logs and inventories rather than trusting completion booleans. A caller
must separately preserve the observed process exit; this script cannot
claim its own external invocation completed successfully.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import re

from studies.fdtd.state_bytes.fdtd_state_bytes_support import RESULTS, PREFIX, BASE, NEW_TEST, TARGETS, write, require, relative, inside, sha, digest, now, maps, identity, verify_guard, verify_snapshot, docs_map, audit_pytest, nodes, reports
from studies.fdtd.state_bytes.fdtd_state_bytes_mutations import MUTATIONS, SOURCE, FUNCTION, accepted_assertion

INPUTS = {}


def read(path):
    path = inside(path)
    value = path.read_bytes()
    key = relative(path)
    require(key not in INPUTS or INPUTS[key] == digest(value), 'Input changed during audit: '+key)
    INPUTS[key] = digest(value)
    return value


def read_json(path):
    import json
    return json.loads(read(path))


def audit_snapshot(item, *, current=True):
    read(item['file'])
    if current:
        read(item['source'])
    return verify_snapshot(item, current=current)


def attempt(kind):
    canonical = RESULTS/(PREFIX+kind+'.json')
    value = read_json(canonical)
    saved = RESULTS/f'{PREFIX}{kind}_attempt{value["attempt"]}.json'
    require(read(canonical) == read(saved), 'Canonical manifest differs from retained attempt.')
    require(value['base'] == BASE and value['complete'], 'Incomplete or wrong-base '+kind)
    require(value['source_before'] == value['source_after'] == identity(), 'Source changed for '+kind)
    verify_guard(value['guard_before'])
    for item in value.get('snapshots', []):
        audit_snapshot(item)
    for check in value['checks']:
        raw = read(check['log'])
        require(digest(raw) == check['sha256'], 'Raw command log hash mismatch.')
        require('launcher_error' not in check, 'A child command did not launch.')
    return value


def command_pair(record, name):
    checks = {c['name']: c for c in record['checks']}
    require(len(checks) == len(record['checks']), 'Duplicate command names.')
    collection, run = checks[name+'_collection'], checks[name]
    require('--collect-only' in collection['command'] and '--collect-only' not in run['command'],
            'Gate commands are not a collection/run pair.')
    require(collection['exit'] == run['exit'] == 0, 'Gate command exited nonzero: '+name)
    result = audit_pytest(read(collection['log']).decode('utf-8'), read(run['log']).decode('utf-8'))
    require(result == run['observed'], 'Saved pytest summary differs from raw per-node replay.')
    return result


def audit_baseline():
    baseline = read_json(RESULTS/(PREFIX+'baseline.json'))
    setup = read_json(RESULTS/(PREFIX+'prevalidation.json'))
    require(baseline['base'] == setup['base'] == BASE, 'Wrong baseline/prevalidation commit.')
    require(setup['baseline_sha256'] == digest(read(RESULTS/(PREFIX+'baseline.json'))), 'Baseline hash changed.')
    require(setup['maps'] == maps(), 'Final raw inventory differs from prevalidation.')
    verify_guard(setup['guard'])
    before, after = baseline['maps'], setup['maps']
    for name in ('goldens', 'previous_film_evidence'):
        require(before[name] == after[name], 'Historical raw artifacts changed: '+name)
    require(len(after['goldens']) == 4, 'Wrong golden count.')
    require(set(before['wavebit']) == set(after['wavebit']), 'Production paths changed.')
    require({k for k in before['wavebit'] if before['wavebit'][k] != after['wavebit'][k]}
            == {'wavebit/fd/_check.py'}, 'Non-report production modification.')
    require(set(after['tests'])-set(before['tests']) == {NEW_TEST}, 'Unexpected added tests.')
    require(all(after['tests'].get(k) == v for k, v in before['tests'].items()), 'Existing tests modified.')
    original = audit_snapshot(baseline['snapshots'][0], current=False)
    require(digest(original) == before['wavebit']['wavebit/fd/_check.py'], 'Original report snapshot mismatch.')
    # Baseline helper versions document pre-application capture; later failed
    # development attempts may legitimately have a different preserved version.
    for item in baseline['snapshots'][1:]:
        audit_snapshot(item, current=False)
    for item in setup['snapshots']:
        audit_snapshot(item)
    old_full = baseline['previous_full']
    require(digest(read(old_full['manifest'])) == old_full['manifest_sha256'], 'Prior full manifest changed.')
    text = read(old_full['file'])
    require(digest(text) == old_full['sha256'], 'Prior full log changed.')
    require(reports(text.decode('utf-8')) == old_full['node_status'], 'Prior node status record differs.')
    for mapping in setup['maps'].values():
        for key, expected in mapping.items():
            require(digest(read(key)) == expected, 'Final map hash mismatch.')
    return baseline, setup


def audit_mutations(value, new_baseline):
    checks = {c['name']: c for c in value['checks']}
    collection = checks['new_collection']
    require(collection['exit'] == 0, 'Mutation collection failed.')
    require(set(nodes(read(collection['log']).decode('utf-8'))) == set(new_baseline['node_status']),
            'Mutation/new baseline collection differs.')
    require(value['restored'] and len(value['witnesses']) == 7, 'Incomplete restoration/witness list.')
    require({w['name'] for w in value['witnesses']} == {m['id'] for m in MUTATIONS}, 'Wrong mutation set.')
    for witness in value['witnesses']:
        spec = next(m for m in MUTATIONS if m['id'] == witness['name'])
        expected_node = NEW_TEST+'::'+FUNCTION+'['+spec['case']+']'
        require(witness['source'] == SOURCE and witness['node'] == expected_node,
                'Mutation witness is not its declared source/case.')
        require(witness['replacement'] == {'before': spec['before'], 'after': spec['after']},
                'Recorded mutation replacement differs from the fixed plan.')
        original, mutant = (read(witness[k]['file']) for k in ('original', 'mutant'))
        require(digest(original) == witness['original']['sha256'] == witness['restored_sha256'],
                'Original/restored source hashes differ.')
        require(original == read(witness['source']), 'Restored production bytes differ.')
        require(digest(mutant) == witness['mutant']['sha256'], 'Mutant source hash differs.')
        source = original.decode('utf-8').replace('\r\n', '\n')
        require(source.count(spec['before']) == 1, 'Mutation anchor is no longer unique.')
        expected = source.replace(spec['before'], spec['after'])
        if b'\r\n' in original:
            expected = expected.replace('\n', '\r\n')
        require(mutant == expected.encode('utf-8'), 'Mutant changes more than its declared replacement.')
        check = checks[witness['check_name']]
        require(check['command'][1:-1] == ['-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                expected_node, '-vv', '--tb=short', '--basetemp'], 'Mutation command targets another case or selection.')
        require(inside(check['command'][-1]).is_relative_to(inside('.state_bytes_work')),
                'Mutation temporary directory leaves the owned scratch scope.')
        text = read(check['log']).decode('utf-8')
        require(witness['accepted'] and accepted_assertion(text, witness['node'], check['exit']),
                'No actual assertion witness for '+witness['name'])
        require(check['environment'].get('PYTHONPYCACHEPREFIX'), 'Mutant bytecode cache was not isolated.')
    require(command_pair(value, 'restored_baseline') == new_baseline, 'Restored baseline differs.')


def main(number):
    destination = RESULTS/f'{PREFIX}final_evidence_audit_attempt{number}.json'
    result = {'base': BASE, 'started_utc': now(), 'complete': False, 'checks': {}}
    write(destination, result, exclusive=True)
    try:
        require(re.fullmatch(r'[0-9a-f]{40}', BASE), 'Resolve the pending commit before audit.')
        baseline, setup = audit_baseline()
        targeted = attempt('targeted')
        new = command_pair(targeted, 'new_allocation')
        require(new['collected'] == 72, 'Actual allocation collection differs from 72-case contract.')
        target = command_pair(targeted, 'targeted')
        target_command = next(c for c in targeted['checks'] if c['name'] == 'targeted_collection')['command']
        require(target_command[1:] == ['-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                      *TARGETS, '--collect-only'], 'Targeted selection differs from fixed report contracts.')
        require(set(new['node_status']).issubset(target['node_status']), 'Targeted tier omits new cases.')
        mutations = attempt('mutations')
        audit_mutations(mutations, new)
        validation = attempt('validation')
        full, slow = command_pair(validation, 'full'), command_pair(validation, 'slow')
        prior = baseline['previous_full']['node_status']
        require(set(full['node_status']) == set(prior)|set(new['node_status']), 'Unexpected full-tier node changes.')
        require({n for n, s in full['node_status'].items() if s == 'skipped'}
                == {n for n, s in prior.items() if s == 'skipped'}
                | {n for n, s in new['node_status'].items() if s == 'skipped'}, 'Full skip identities changed.')
        require(set(slow['node_status']).issubset(full['node_status']), 'Slow tier is outside full selection.')
        require(slow['collected'] == 81 and slow['totals']['passed'] == 81, 'Separate slow gate incomplete.')
        checks = {c['name']: c for c in validation['checks']}
        for stage, count in enumerate((143, 288, 140, 570)):
            check = checks['freezer'+str(stage)]
            require(check['exit'] == 0 and check['arrays_exact'] == count, 'Freezer failed.')
            require(f'{count} of {count} arrays reproduce bit for bit' in read(check['log']).decode('utf-8'),
                    'Freezer raw output differs.')
            require('--rebaseline' not in check['command'], 'Golden rebaseline command is forbidden.')
        docs = attempt('final_docs')
        require(docs['sources'] == docs_map(), 'Documentation changed after final gates.')
        for name, expected in docs['sources'].items():
            require(digest(read(name)) == expected, 'Final documentation input changed.')
        docs_observed = command_pair(docs, 'docs')
        sphinx = next(c for c in docs['checks'] if c['name'] == 'sphinx')
        require(sphinx['exit'] == 0 and '-W' in sphinx['command'], 'Sphinx -W did not pass.')
        result['checks'] = {'new': new, 'targeted': target, 'actual_mutations': 7, 'full': full,
                            'slow': slow, 'docs': docs_observed, 'arrays_bit_exact': 1141,
                            'raw_map_counts': {k: len(v) for k, v in setup['maps'].items()}}
        for name, expected in INPUTS.items():
            require(sha(inside(name)) == expected, 'Audit input changed during replay.')
        result.update(complete=True, ended_utc=now(), input_sha256=INPUTS,
                      qualification='Saved allocation/report evidence only; no new field-accuracy claim.')
        write(destination, result)
        write(RESULTS/(PREFIX+'final_evidence_audit.json'), result, exclusive=True)
        print('Saved-evidence audit passed; caller must record actual process completion.', flush=True)
    except BaseException as error:
        result.update(error=repr(error), ended_utc=now(), input_sha256=INPUTS)
        write(destination, result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    main(parser.parse_args().attempt)
