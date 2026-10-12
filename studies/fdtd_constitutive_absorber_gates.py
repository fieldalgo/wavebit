"""Unexecuted bounded C9 numerical, preservation and documentation gate runner.

The owner runs this sequentially. Full excludes slow; the separate slow selection
runs exactly the collected slow cases. Every child exit and raw node report is
retained, including failures. No rebaseline or response-based retry is performed.
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys

from fdtd_constitutive_absorber_gate_support import (
    ROOT, RESULTS, PREFIX, SCRATCH, NEW_TEST, TARGETS, load, write, require,
    guard, verify_guard, attempt_record, pytest_pair, run_command, finish, failed,
    docs_map, sha, snapshot, relative, inside, predecessor_skips)


def targeted(attempt):
    directory, manifest, record = attempt_record('targeted', attempt)
    try:
        pytest_pair(record, manifest, 'new_constitutive_absorber', [NEW_TEST], directory)
        pytest_pair(record, manifest, 'targeted', list(TARGETS), directory,
                    extra=('-m', 'not slow'), workers=5)
        finish('targeted', manifest, record)
    except BaseException as error:
        failed(manifest, record, error)
        raise


def ref(path):
    return {'path': relative(path), 'sha256': sha(path)}


def parent_receipt(path, mode):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(ROOT.resolve()), 'Parent receipt must be preserved in the checkout.')
    record = load(path)
    require(record['exit_code'] == 0 and record['seconds'] >= 0, 'Parent did not observe success.')
    require(sha(inside(record['log'])) == record['log_sha256'], 'Parent raw output changed.')
    command = record['command']
    require(len(command) >= 4 and command[1] == '-B' and command[3] == mode,
            'Unexpected campaign/audit parent command.')
    actual = Path(command[2])
    if not actual.is_absolute():
        actual = ROOT/actual
    require(actual.resolve() == (ROOT/'studies'/(PREFIX+'campaign.py')).resolve(),
            'Parent ran a different controller.')
    start, end = map(datetime.fromisoformat, (record['started_utc'], record['ended_utc']))
    require(start.tzinfo is not None and end.tzinfo is not None and start <= end,
            'Parent chronology is invalid.')
    require(not record.get('environment', {}).get('XLA_FLAGS'), 'Forbidden numerical flag.')
    return {'reference': ref(path), 'started_utc': record['started_utc'], 'ended_utc': record['ended_utc']}


def prepare(attempt, campaign_parent, audit_parent):
    directory, manifest, record = attempt_record('prevalidation', attempt)
    try:
        names = ('targeted', 'mutations', 'report', 'campaign_audit', 'campaign_attempt1', 'registration')
        values = {name: load(RESULTS/(PREFIX+name+'.json')) for name in names}
        for label in ('targeted', 'mutations'):
            item = values[label]
            require(item['complete'] and item['passed'], label+' must pass first.')
            verify_guard(item['guard_before'])
            verify_guard(item['guard_after'])
        mutation = values['mutations']
        require(mutation['restored'] and mutation['all_function_assertion_witnesses'],
                'Actual source witnesses and exact restoration required.')
        report, audited, ledger = (values[k] for k in ('report', 'campaign_audit', 'campaign_attempt1'))
        require(report['complete'] and report['workers'] == 60 and ledger['complete']
                and len(ledger['workers']) == 60, 'Fixed physical characterization incomplete.')
        require(audited['complete'] and audited['saved_array_replay'] and audited['workers'] == 60,
                'Saved-array controller replay must succeed.')
        verify_guard(audited['guard'])
        for name, key in (('report', 'report'), ('campaign_attempt1', 'ledger'), ('registration', 'registration')):
            require(audited[key] == ref(RESULTS/(PREFIX+name+'.json')), 'Audit input link differs.')
        require(campaign_parent is not None and audit_parent is not None,
                'Pass completed campaign and saved-data audit outer receipts explicitly.')
        parents = {'campaign': parent_receipt(campaign_parent, 'run'),
                   'audit': parent_receipt(audit_parent, 'audit')}
        require(datetime.fromisoformat(parents['campaign']['started_utc']) <= datetime.fromisoformat(ledger['started_utc'])
                <= datetime.fromisoformat(ledger['ended_utc']) <= datetime.fromisoformat(parents['campaign']['ended_utc']),
                'Campaign is outside its actual parent interval.')
        require(datetime.fromisoformat(parents['audit']['started_utc']) <= datetime.fromisoformat(audited['ended_utc'])
                <= datetime.fromisoformat(parents['audit']['ended_utc']), 'Audit is outside its parent interval.')
        paths = [inside(name) for name in sorted(record['guard_before']['source'])]
        folder = directory/'sources'
        folder.mkdir()
        record.update(prerequisites={name: ref(RESULTS/(PREFIX+name+'.json')) for name in names},
                      parents=parents, snapshots=[snapshot(p, folder/(relative(p).replace('/', '__')+'.raw.gz'))
                                                  for p in paths], guard=guard())
        finish('prevalidation', manifest, record)
    except BaseException as error:
        failed(manifest, record, error)
        raise


def validation(attempt):
    setup_path = RESULTS/(PREFIX+'prevalidation.json')
    setup = load(setup_path)
    require(setup['complete'] and setup['passed'], 'Prevalidation is incomplete.')
    verify_guard(setup['guard'])
    directory, manifest, record = attempt_record('validation', attempt)
    try:
        record['prevalidation'] = ref(setup_path)
        record['predecessor_skip_proof'] = predecessor_skips()
        write(manifest, record)
        files = [relative(p) for p in sorted((ROOT/'tests').glob('test_fd_*.py'))]
        pytest_pair(record, manifest, 'full', files+['tests/test_docs_contract.py'], directory,
                    extra=('-m', 'not slow'), workers=5,
                    allowed_skips=record['predecessor_skip_proof']['nodes'])
        verify_guard(setup['guard'])
        pytest_pair(record, manifest, 'slow', files, directory,
                    extra=('--run-slow', '-m', 'slow'), workers=5)
        verify_guard(setup['guard'])
        for stage, expected in enumerate((143, 288, 140, 570)):
            row, text = run_command(record, manifest, 'freezer'+str(stage),
                [sys.executable, '-B', f'tests/references/interface/freeze_fd_stage{stage}.py'], directory)
            require(row['exit'] == 0, 'Freezer failed: preserve evidence; do not rebaseline.')
            require(f'{expected} of {expected} arrays reproduce bit for bit' in text,
                    'Read-only frozen array comparison did not match.')
            row['arrays_exact'] = expected
            write(manifest, record)
            verify_guard(setup['guard'])
        finish('validation', manifest, record)
    except BaseException as error:
        failed(manifest, record, error)
        raise


def documentation(attempt, sphinx):
    require(sphinx is not None, 'Use an existing explicit Sphinx executable.')
    directory, manifest, record = attempt_record('documentation', attempt)
    try:
        record['docs_before'] = docs_map()
        write(manifest, record)
        pytest_pair(record, manifest, 'docs', ['tests/test_docs_contract.py'], directory)
        output = SCRATCH/('sphinx-attempt'+str(attempt))
        require(not output.exists(), 'Preserve the previous Sphinx output.')
        row, text = run_command(record, manifest, 'sphinx',
            [str(Path(sphinx).resolve(strict=True)), '-W', '-b', 'html', 'docs', str(output)], directory)
        require(row['exit'] == 0 and 'build succeeded' in text, 'Sphinx failed.')
        record['docs_after'] = docs_map()
        require(record['docs_before'] == record['docs_after'], 'Docs changed during checks.')
        finish('documentation', manifest, record)
    except BaseException as error:
        failed(manifest, record, error)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('targeted', 'prepare', 'validation', 'documentation'))
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--sphinx')
    parser.add_argument('--campaign-parent')
    parser.add_argument('--audit-parent')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.attempt, args.campaign_parent, args.audit_parent)
    elif args.mode == 'documentation':
        documentation(args.attempt, args.sphinx)
    else:
        globals()[args.mode](args.attempt)
