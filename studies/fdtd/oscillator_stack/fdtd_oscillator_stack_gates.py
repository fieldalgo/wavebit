"""Sequential Oscillator-stack gates; retain exact rosters, raw logs and child exits.

The pre-calibration mode excludes only the new optical function and never creates
an accepted targeted record. All accepted gates include the disclosed optical
ceilings. Parallel pytest workers are opt-in; child processes never overlap.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

from studies.fdtd.oscillator_stack.fdtd_oscillator_stack_support import ROOT, RESULTS, PREFIX, SCRATCH, NEW_TEST, TARGETS, load, write, require, guard, verify_guard, attempt_record, pytest_pair, run_command, finish, docs_map, sha, snapshot, relative, inside, now, identity, audit_pytest, maps, campaign_output_matches

OPTICAL = 'test_oblique_oscillator_stack_matches_declared_multilayer_on_each_backend'
PREVIOUS = RESULTS/'fdtd_debye_stack_validation.json'


def begin(kind, number, workers=0):
    require(number > 0 and workers >= 0, 'Positive attempt and nonnegative worker count required.')
    directory, manifest, record = attempt_record(kind, number)
    record.update(passed=False, workers=workers)
    record['runner_snapshots'] = [snapshot(inside(name), directory/(name.replace('/', '__')+'.raw.gz'))
        for name in sorted(set(record['guard_before']['helpers'])|set(record['guard_before']['runner_inputs']))]
    write(manifest, record)
    return directory, manifest, record


def failed(manifest, record, error):
    record.update(complete=False, passed=False, ended_utc=now(), failure=repr(error),
                  source_after=identity(), guard_after=guard())
    write(manifest, record)


def done(kind, manifest, record):
    verify_guard(record['guard_before'])
    record.update(passed=True, guard_after=guard())
    finish(kind, manifest, record)


def ref(path):
    return {'path': relative(path), 'sha256': sha(path)}


def predecessor_skips():
    value = load(PREVIOUS)
    require(value['complete'], 'Predecessor validation is incomplete.')
    checks = {c['name']: c for c in value['checks']}
    c, r = checks['full_collection'], checks['full']
    require(c['exit'] == r['exit'] == 0 and all(sha(inside(x['log'])) == x['sha256'] for x in (c, r)),
            'Predecessor raw commands did not succeed or changed.')
    observed = audit_pytest(inside(c['log']).read_text(encoding='utf-8'),
                            inside(r['log']).read_text(encoding='utf-8'))
    require(observed == r['observed'], 'Predecessor saved and raw rosters differ.')
    nodes = sorted(k for k, v in observed['node_status'].items() if v == 'skipped')
    require(len(nodes) == 8, 'Expected eight exact predecessor skips.')
    return {'manifest': ref(PREVIOUS), 'collection': ref(inside(c['log'])),
            'run': ref(inside(r['log'])), 'nodes': nodes}


def targeted(number, workers=0, pre_calibration=False):
    kind = 'targeted_precalibration' if pre_calibration else 'targeted'
    directory, manifest, record = begin(kind, number, workers)
    try:
        extra = ('-k', 'not '+OPTICAL) if pre_calibration else ()
        record['optical_excluded'] = pre_calibration
        pytest_pair(record, manifest, 'new_stack', [NEW_TEST], directory, extra=extra, workers=workers)
        pytest_pair(record, manifest, 'targeted', list(TARGETS), directory,
                    extra=('-m', 'not slow', *extra), workers=workers)
        done(kind, manifest, record)
    except BaseException as error:
        failed(manifest, record, error)
        raise


def trailing_json(text):
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r'(?m)^\{', text))):
        try:
            value, end = decoder.raw_decode(text[match.start():])
            if not text[match.start()+end:].strip():
                return value
        except ValueError:
            continue
    raise ValueError('Missing final JSON controller output.')


def parent_receipt(path, mode):
    path = inside(Path(path).resolve(strict=True))
    record = load(path)
    require(type(record.get('exit_code')) is int and record['exit_code'] == 0
            and record['seconds'] >= 0 and 'launcher_error' not in record,
            'Parent did not observe actual successful completion.')
    require(sha(inside(record['log'])) == record['log_sha256'], 'Parent raw output changed.')
    command = record['command']
    if command[1:3] == ['-B', '-m']:
        require(command[3:] == ['studies.'+PREFIX+'campaign', mode], 'Unexpected parent module/arguments.')
    else:
        require(command[1] == '-B' and inside(command[2]) == ROOT/'studies'/(PREFIX+'campaign.py')
                and command[3:] == [mode], 'Unexpected parent script/arguments.')
    start, end = map(datetime.fromisoformat, (record['started_utc'], record['ended_utc']))
    require(start.tzinfo is not None and end.tzinfo is not None and start <= end, 'Invalid parent chronology.')
    require(not record.get('environment', {}).get('XLA_FLAGS'), 'Forbidden numerical flag.')
    return {'reference': ref(path), 'started_utc': record['started_utc'], 'ended_utc': record['ended_utc'],
            'output': trailing_json(inside(record['log']).read_text(encoding='utf-8'))}


def prepare(number, campaign_parent, audit_parent):
    directory, manifest, record = begin('prevalidation', number)
    try:
        require(campaign_parent and audit_parent, 'Explicit completed campaign and saved-audit parent receipts required.')
        parents = {'campaign': parent_receipt(campaign_parent, 'run'),
                   'audit': parent_receipt(audit_parent, 'audit')}
        # Pure saved-evidence replay, not a numerical import or a solver run.
        from studies.fdtd.oscillator_stack import fdtd_oscillator_stack_audit as audit
        targeted_value = audit.attempt('targeted')
        workers = targeted_value['workers']
        new = audit.command_pair(targeted_value, 'new_stack', [NEW_TEST], workers=workers)
        audit.command_pair(targeted_value, 'targeted', list(TARGETS), extra=('-m', 'not slow'), workers=workers)
        mutation = audit.audit_mutations(new, targeted_value)
        acceptance, calibration_value, known = audit.calibration()
        physical = audit.audit_campaign(acceptance, calibration_value, known, targeted_value, mutation)
        require(campaign_output_matches(parents['campaign']['output'],
            {'complete': True, 'stop_reason': None, 'workers': physical['workers'],
             'report': relative(RESULTS/(PREFIX+'report.json'))}),
            'Campaign parent output differs from raw physical replay.')
        audit_output = parents['audit']['output']
        require(audit_output['audit'] == 'passed' and audit_output['complete'] is True
                and audit_output['all_48_cells_recomputed'] is True
                and audit_output['workers'] == physical['workers'], 'Controller saved-audit did not pass.')
        for stored, replay in (('self_counts', 'self'), ('all_direct_counts', 'direct'), ('direct_final_counts', 'final_direct')):
            require(audit_output[stored] == physical['classifications'][replay], 'Controller/replay counts differ.')
        names = ('baseline', 'application', 'targeted', 'mutations', 'report', 'registration', 'calibration_acceptance')
        paths = set(record['guard_before']['helpers'])|set(record['guard_before']['references'])|set(record['guard_before']['runner_inputs'])
        paths |= set(record['guard_before']['maps']['wavebit'])|set(record['guard_before']['maps']['tests'])
        folder = directory/'sources'
        folder.mkdir()
        record.update(prerequisites={n: ref(RESULTS/(PREFIX+n+'.json')) for n in names}, parents=parents,
                      maps=maps(), guard=guard(), physical_replay=physical,
                      baseline_sha256=sha(RESULTS/(PREFIX+'baseline.json')),
                      snapshots=[snapshot(inside(n), folder/(n.replace('/', '__')+'.raw.gz')) for n in sorted(paths)])
        done('prevalidation', manifest, record)
    except BaseException as error:
        failed(manifest, record, error)
        raise


def validation(number, workers=0):
    setup_path = RESULTS/(PREFIX+'prevalidation.json')
    setup = load(setup_path)
    require(setup['complete'] and setup['passed'], 'Prevalidation is incomplete.')
    verify_guard(setup['guard'])
    directory, manifest, record = begin('validation', number, workers)
    try:
        record['prevalidation'] = ref(setup_path)
        record['predecessor_skip_proof'] = predecessor_skips()
        write(manifest, record)
        files = [relative(p) for p in sorted((ROOT/'tests').glob('test_fd_*.py'))]
        pytest_pair(record, manifest, 'full', files+['tests/api/test_docs_contract.py'], directory,
                    extra=('-m', 'not slow'), workers=workers,
                    allowed_skips=record['predecessor_skip_proof']['nodes'])
        verify_guard(setup['guard'])
        pytest_pair(record, manifest, 'slow', files, directory,
                    extra=('--run-slow', '-m', 'slow'), workers=workers)
        verify_guard(setup['guard'])
        for stage, expected in enumerate((143, 288, 140, 570)):
            row, text = run_command(record, manifest, 'freezer'+str(stage),
                [sys.executable, '-B', f'tests/references/interface/freeze_fd_stage{stage}.py'], directory)
            require(row['exit'] == 0 and f'{expected} of {expected} arrays reproduce bit for bit' in text
                    and 'the committed golden is reproduced; nothing written' in text,
                    'Read-only freezer failed: preserve evidence; do not rebaseline.')
            row['arrays_exact'] = expected
            write(manifest, record)
            verify_guard(setup['guard'])
        done('validation', manifest, record)
    except BaseException as error:
        failed(manifest, record, error)
        raise


def documentation(number, sphinx):
    require(sphinx is not None, 'Use an existing explicit Sphinx executable.')
    directory, manifest, record = begin('documentation', number)
    try:
        record['docs_before'] = docs_map()
        write(manifest, record)
        pytest_pair(record, manifest, 'docs', ['tests/api/test_docs_contract.py'], directory)
        output = SCRATCH/('sphinx-attempt'+str(number))
        require(not output.exists(), 'Preserve previous Sphinx output.')
        row, text = run_command(record, manifest, 'sphinx',
            [str(Path(sphinx).resolve(strict=True)), '-W', '-b', 'html', 'docs', str(output)], directory)
        require(row['exit'] == 0 and 'build succeeded' in text, 'Sphinx failed.')
        record['docs_after'] = docs_map()
        require(record['docs_before'] == record['docs_after'], 'Docs changed during checks.')
        done('documentation', manifest, record)
    except BaseException as error:
        failed(manifest, record, error)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('precalibration', 'targeted', 'prepare', 'validation', 'documentation'))
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--sphinx')
    parser.add_argument('--campaign-parent')
    parser.add_argument('--audit-parent')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.attempt, args.campaign_parent, args.audit_parent)
    elif args.mode == 'documentation':
        documentation(args.attempt, args.sphinx)
    elif args.mode in ('targeted', 'precalibration'):
        targeted(args.attempt, args.workers, args.mode == 'precalibration')
    else:
        validation(args.attempt, args.workers)
