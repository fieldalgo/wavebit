"""Audit saved analytic-helper evidence without running a solver or pytest.

Raw logs, collected/reported node inventories, immutable source snapshots and
actual recorded child exits are replayed. Only a new audit attempt and, after
success, its canonical result are written. The caller must separately record
this auditor's observed process exit; the auditor cannot certify its own exit.
"""
import argparse
import ast
from datetime import datetime
import json
import re

from fdtd_inactive_lorentz_support import (
    ROOT, RESULTS, PREFIX, BASE, NEW_TEST, TARGETS, write, require, relative,
    inside, sha, digest, now, maps, identity, verify_guard, verify_snapshot,
    docs_map, audit_pytest, nodes,
)
from fdtd_inactive_lorentz_mutations import (
    MUTATIONS, SOURCE, FUNCTIONS, accepted_assertion,
)

INPUTS = {}
EXPECTED_BEFORE = {'wavebit': 103, 'tests': 85, 'goldens': 4,
                   'previous_film_evidence': 1620}
PREVIOUS = 'studies/results/fdtd_hybrid_film_'


def read(path):
    path = inside(path)
    raw = path.read_bytes()
    name = relative(path)
    require(name not in INPUTS or INPUTS[name] == digest(raw),
            'Input changed during audit: '+name)
    INPUTS[name] = digest(raw)
    return raw


def read_json(path):
    return json.loads(read(path))


def audit_snapshot(item, *, current=True):
    read(item['file'])
    if current:
        read(item['source'])
    return verify_snapshot(item, current=current)


def checks_by_name(value):
    answer = {item['name']: item for item in value['checks']}
    require(len(answer) == len(value['checks']), 'Duplicate child command names.')
    return answer


def attempt(kind):
    canonical = RESULTS/(PREFIX+kind+'.json')
    value = read_json(canonical)
    require(type(value['attempt']) is int and value['attempt'] > 0, 'Invalid attempt number.')
    retained = RESULTS/f'{PREFIX}{kind}_attempt{value["attempt"]}.json'
    require(read(canonical) == read(retained), 'Canonical/retained attempt differs: '+kind)
    require(value['base'] == BASE and value['kind'] == kind and value['complete'] is True,
            'Incomplete or wrong-base '+kind)
    require(value['source_before'] == value['source_after'] == identity(),
            'Numerical source changed for '+kind)
    verify_guard(value['guard_before'])
    for name, expected in value['guard_before']['helpers'].items():
        require(digest(read(name)) == expected, 'Frozen helper changed: '+name)
    for item in value.get('snapshots', []):
        audit_snapshot(item)
    checks_by_name(value)
    previous = datetime.fromisoformat(value['started_utc'])
    ended = datetime.fromisoformat(value['ended_utc'])
    for check in value['checks']:
        require('launcher_error' not in check and type(check['exit']) is int,
                'Child command did not report an actual exit.')
        require(digest(read(check['log'])) == check['sha256'], 'Raw child log hash differs.')
        start, stop = (datetime.fromisoformat(check[k]) for k in ('started_utc', 'ended_utc'))
        require(previous <= start <= stop <= ended, 'Nonsequential child chronology.')
        previous = stop
        require(not check['environment'].get('XLA_FLAGS'), 'Unexpected XLA_FLAGS.')
        require(check['environment']['PYTHONUTF8'] == '1', 'Raw log encoding was not fixed.')
        require(inside(check['environment']['PYTHONPATH']) == ROOT, 'Child used another project.')
    return value


def command_pair(record, name, targets, *, extra=(), workers=0):
    checks = checks_by_name(record)
    collection, run = checks[name+'_collection'], checks[name]
    common = [record['executable'], '-B', '-m', 'pytest', '-q', '-p',
              'no:cacheprovider', *targets, *extra]
    require(collection['command'] == common+['--collect-only'],
            'Collection selection differs: '+name)
    tail = run['command'][len(common):]
    require(run['command'][:len(common)] == common and len(tail) == (6 if workers else 4)
            and tail[:3] == ['-vv', '--tb=short', '--basetemp'],
            'Run selection/reporting differs: '+name)
    require(inside(tail[3]).is_relative_to(inside('.inactive_lorentz_work')),
            'Pytest temporary directory escapes milestone scope.')
    if workers:
        require(tail[4:] == ['-n', str(workers)], 'Unexpected worker count.')
    require(collection['exit'] == run['exit'] == 0, 'Nonzero gate child exit: '+name)
    observed = audit_pytest(read(collection['log']).decode('utf-8'),
                            read(run['log']).decode('utf-8'))
    require(observed == run['observed'], 'Saved pytest result differs from raw replay.')
    return observed


def audit_baseline():
    baseline_path = RESULTS/(PREFIX+'baseline.json')
    baseline = read_json(baseline_path)
    setup = read_json(RESULTS/(PREFIX+'prevalidation.json'))
    require(baseline['base'] == setup['base'] == BASE and baseline['raw_byte_sha256'] is True,
            'Wrong baseline/prevalidation identity.')
    require(setup['baseline_sha256'] == digest(read(baseline_path)), 'Baseline bytes changed.')
    before = {name: baseline['prior_evidence' if name == 'previous_film_evidence' else name]
              for name in EXPECTED_BEFORE}
    require({name: len(values) for name, values in before.items()} == EXPECTED_BEFORE,
            'Predecessor inventory differs from its captured contract.')
    after = setup['maps']
    require(set(after) == set(before) and after == maps(), 'Final raw inventory differs.')
    verify_guard(setup['guard'])
    for name in ('goldens', 'previous_film_evidence'):
        require(before[name] == after[name], 'Historical artifacts changed: '+name)
    require(set(before['wavebit']) == set(after['wavebit']), 'Production paths changed.')
    require({name for name in before['wavebit'] if before['wavebit'][name] != after['wavebit'][name]}
            == {'wavebit/fd/materials.py'}, 'Production change exceeds the analytic helper.')
    require(set(after['tests'])-set(before['tests']) == {NEW_TEST}, 'Unexpected new test files.')
    require(all(after['tests'].get(name) == expected for name, expected in before['tests'].items()),
            'Existing FD tests changed or disappeared.')
    for values in after.values():
        for name, expected in values.items():
            require(digest(read(name)) == expected, 'Final raw map hash differs: '+name)
    snapshots = setup['snapshots']
    needed = {SOURCE, NEW_TEST, *setup['guard']['helpers']}
    require(needed.issubset({item['source'] for item in snapshots}),
            'Prevalidation did not preserve all numerical/helper sources.')
    for item in snapshots:
        audit_snapshot(item)
    return before, setup


def predecessor_rosters(before):
    def preserved(name):
        raw = read(name)
        require(digest(raw) == before['previous_film_evidence'][name],
                'Predecessor log/manifest does not match raw baseline.')
        return raw
    manifest = json.loads(preserved(PREVIOUS+'validation.json'))
    checks = checks_by_name(manifest)
    rosters = {}
    for name, count in (('full', 3677), ('slow', 81)):
        require(checks[name]['exit'] == checks[name+'_collection']['exit'] == 0,
                'Predecessor validation child failed.')
        collection = preserved(PREVIOUS+'controls/'+name+'_collection.log').decode('utf-8')
        output = preserved(PREVIOUS+'controls/'+name+'.log').decode('utf-8')
        rosters[name] = audit_pytest(collection, output)
        require(rosters[name]['collected'] == checks[name]['collected'] == count,
                'Predecessor collected count differs.')
        for status in ('passed', 'skipped', 'xfailed', 'xpassed'):
            require(rosters[name]['totals'][status] == checks[name][status],
                    'Predecessor raw outcome differs from manifest.')
    require(rosters['full']['totals']['skipped'] == 8
            and rosters['slow']['totals']['passed'] == 81, 'Wrong predecessor skip/slow contract.')
    return rosters


def audit_mutations(value, new):
    require(SOURCE == 'wavebit/fd/materials.py' and len(MUTATIONS) == 3,
            'Wrong analytic-source mutation scope.')
    require(len(FUNCTIONS) == len(set(FUNCTIONS)) == 2, 'Wrong new function contract.')
    tree = ast.parse(read(NEW_TEST).decode('utf-8'))
    actual_functions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and n.name.startswith('test_')}
    require(actual_functions == set(FUNCTIONS), 'New functions are not fully declared.')
    checks = checks_by_name(value)
    collection = checks['new_collection']
    require(collection['command'] == [value['executable'], '-B', '-m', 'pytest', '-q', '-p',
            'no:cacheprovider', NEW_TEST, '--collect-only'] and collection['exit'] == 0,
            'Mutation collection command differs.')
    collected = set(nodes(read(collection['log']).decode('utf-8')))
    require(collected == set(new['node_status']), 'Mutation and initial collection differ.')
    require(value['restored'] is True and len(value['witnesses']) == 3,
            'Incomplete mutation/restoration evidence.')
    require({w['name'] for w in value['witnesses']} == {m['id'] for m in MUTATIONS},
            'Unexpected mutation set.')
    require({m['function'] for m in MUTATIONS} == actual_functions,
            'A new test function has no actual-source witness.')
    needed = {'studies/fdtd_inactive_lorentz_mutations.py', NEW_TEST}
    require(needed.issubset({item['source'] for item in value['snapshots']}),
            'Mutation source/test snapshots are missing.')
    for witness in value['witnesses']:
        spec = next(m for m in MUTATIONS if m['id'] == witness['name'])
        node = NEW_TEST+'::'+spec['function']+'['+spec['case']+']'
        require(node in collected and witness['source'] == SOURCE and witness['node'] == node,
                'Mutation does not target its declared collected case.')
        require(witness['replacement'] == {'before': spec['before'], 'after': spec['after']},
                'Recorded mutation differs from the frozen harness.')
        original, mutant = (read(witness[k]['file']) for k in ('original', 'mutant'))
        require(digest(original) == witness['original']['sha256'] == witness['restored_sha256']
                and original == read(SOURCE), 'Source was not restored byte for byte.')
        require(digest(mutant) == witness['mutant']['sha256'], 'Mutant snapshot hash differs.')
        source = original.decode('utf-8').replace('\r\n', '\n')
        require(source.count(spec['before']) == 1, 'Mutation anchor is not unique.')
        expected = source.replace(spec['before'], spec['after'])
        if b'\r\n' in original:
            expected = expected.replace('\n', '\r\n')
        require(mutant == expected.encode('utf-8'), 'Mutant changes more than its declared anchor.')
        check = checks[witness['check_name']]
        require(witness['check_name'] == spec['id'] and len(check['command']) == 12
                and check['command'][:-1] == [value['executable'], '-B', '-m', 'pytest', '-q', '-p',
                    'no:cacheprovider', node, '-vv', '--tb=short', '--basetemp'],
                'Mutation command differs from the intended isolated case.')
        temporary = inside(check['command'][-1])
        cache = check['environment'].get('PYTHONPYCACHEPREFIX')
        require(temporary.is_relative_to(inside('.inactive_lorentz_work'))
                and temporary.name.startswith('mutant-'+spec['id']+'-') and cache,
                'Mutation cache/temp isolation is absent.')
        require(inside(cache).is_relative_to(inside('.inactive_lorentz_work')),
                'Mutation bytecode cache escapes its scratch scope.')
        require(witness['accepted'] is True and accepted_assertion(
            read(check['log']).decode('utf-8'), node, check['exit']),
            'No actual single assertion failure for '+spec['id'])
    expected_checks = {'new_collection', 'restored_baseline_collection', 'restored_baseline',
                       *(m['id'] for m in MUTATIONS)}
    require(set(checks) == expected_checks, 'Unexpected mutation child commands.')
    require(command_pair(value, 'restored_baseline', [NEW_TEST]) == new,
            'Restored baseline differs from initial analytic baseline.')


def main(number):
    require(number > 0, 'Positive audit attempt number required.')
    destination = RESULTS/f'{PREFIX}final_evidence_audit_attempt{number}.json'
    result = {'base': BASE, 'started_utc': now(), 'complete': False, 'checks': {}}
    write(destination, result, exclusive=True)
    try:
        require(re.fullmatch('[0-9a-f]{40}', BASE), 'Expected full predecessor commit.')
        before, setup = audit_baseline()
        previous = predecessor_rosters(before)
        targeted = attempt('targeted')
        require(setup['targeted_sha256'] == digest(read(RESULTS/(PREFIX+'targeted.json'))),
                'Prepared targeted manifest changed.')
        new = command_pair(targeted, 'new_analytic', [NEW_TEST])
        require(new['collected'] == new['totals']['passed'] == 127,
                'Expected all 127 actual analytic/derivative cases to pass.')
        target = command_pair(targeted, 'targeted', list(TARGETS))
        require(set(new['node_status']).issubset(target['node_status']), 'Targeted tier omits new cases.')
        require(set(checks_by_name(targeted)) == {'new_analytic_collection', 'new_analytic',
                'targeted_collection', 'targeted'}, 'Unexpected targeted child commands.')
        mutations = attempt('mutations')
        require(setup['mutations_sha256'] == digest(read(RESULTS/(PREFIX+'mutations.json'))),
                'Prepared mutations manifest changed.')
        audit_mutations(mutations, new)
        validation = attempt('validation')
        files = sorted(setup['maps']['tests'])
        full = command_pair(validation, 'full', files+['tests/test_docs_contract.py'], workers=5)
        slow = command_pair(validation, 'slow', files, extra=['--run-slow', '-m', 'slow'], workers=5)
        old_nodes, new_nodes = previous['full']['node_status'], new['node_status']
        require(not set(old_nodes).intersection(new_nodes), 'New cases collide with predecessor nodes.')
        require(set(full['node_status']) == set(old_nodes)|set(new_nodes), 'Full roster changed unexpectedly.')
        require({n for n, s in full['node_status'].items() if s == 'skipped'}
                == {n for n, s in old_nodes.items() if s == 'skipped'}, 'Historical eight skip IDs changed.')
        require(slow == previous['slow'], 'Slow roster/outcomes differ from the predecessor 81.')
        checks = checks_by_name(validation)
        require(set(checks) == {'full_collection', 'full', 'slow_collection', 'slow',
                'freezer0', 'freezer1', 'freezer2', 'freezer3'}, 'Unexpected validation commands.')
        for stage, count in enumerate((143, 288, 140, 570)):
            check = checks['freezer'+str(stage)]
            require(check['command'] == [validation['executable'], '-B',
                    f'tests/references/interface/freeze_fd_stage{stage}.py'], 'Unexpected freezer command.')
            require(check['exit'] == 0 and check['arrays_exact'] == count, 'Freezer failed.')
            require(f'{count} of {count} arrays reproduce bit for bit'
                    in read(check['log']).decode('utf-8'), 'Freezer raw result differs.')
            require('the committed golden is reproduced; nothing written'
                    in read(check['log']).decode('utf-8'), 'Freezer is not the read-only check.')
        docs = attempt('final_docs')
        require(docs['sources'] == docs_map(), 'Final document sources changed.')
        for name, expected in docs['sources'].items():
            require(digest(read(name)) == expected, 'Document hash changed: '+name)
        docs_observed = command_pair(docs, 'docs', ['tests/test_docs_contract.py'])
        require(docs_observed['collected'] == docs_observed['totals']['passed'] == 73,
                'Final docs contract does not pass all 73 cases.')
        checks = checks_by_name(docs)
        require(set(checks) == {'docs_collection', 'docs', 'sphinx'}, 'Unexpected documentation commands.')
        sphinx = checks['sphinx']
        require(sphinx['exit'] == 0 and len(sphinx['command']) == 7
                and sphinx['command'][1:-1] == ['-W', '--keep-going', '-b', 'html', 'docs'],
                'Sphinx -W build command did not succeed.')
        require(inside(sphinx['command'][-1]).is_relative_to(inside('.inactive_lorentz_work')),
                'Sphinx output escapes milestone scope.')
        raw_sphinx = read(sphinx['log']).decode('utf-8')
        require('build succeeded.' in raw_sphinx and not re.search(
            r'(?im)^.*(?:WARNING:|ERROR:)', raw_sphinx), 'Sphinx raw output did not complete cleanly.')
        result['checks'] = {'new': new, 'targeted': target, 'actual_mutations': len(MUTATIONS),
            'witnessed_functions': sorted(FUNCTIONS), 'full': full, 'slow': slow,
            'docs': docs_observed, 'arrays_bit_exact': 1141,
            'raw_map_counts': {k: len(v) for k, v in setup['maps'].items()}}
        for name, expected in INPUTS.items():
            require(sha(inside(name)) == expected, 'Audit input changed during replay: '+name)
        result.update(complete=True, ended_utc=now(), input_sha256=INPUTS,
            qualification='Saved analytic-helper and regression evidence only; no new field-accuracy, '
                          'CFL, film-gradient or general C8 claim. Caller records auditor process exit.')
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
