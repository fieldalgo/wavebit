"""Recorded test, regression and documentation gates for automatic decay stopping."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import sys
from pathlib import Path
from studies.fdtd.decay_stop.fdtd_decay_stop_support import ROOT, RESULTS, PREFIX, SCRATCH, BASE, NEW_TEST, TARGETS, load, write, require, guard, verify_guard, attempt_record, pytest_pair, run_command, finish, docs_map, sha, snapshot, helpers, now


def targeted(attempt):
    directory, manifest, record = attempt_record('targeted', attempt)
    pytest_pair(record, manifest, 'new_decay_stop', [NEW_TEST], directory)
    pytest_pair(record, manifest, 'targeted', list(TARGETS), directory,
                extra=('-m', 'not slow'), workers=5)
    finish('targeted', manifest, record)


def prepare():
    target = load(RESULTS/(PREFIX+'targeted.json'))
    mutations = load(RESULTS/(PREFIX+'mutations.json'))
    campaign = load(RESULTS/(PREFIX+'campaign.json'))
    require(target['complete'], 'Targeted tests must finish first.')
    require(mutations['passed'] and mutations['restored']
            and mutations['all_function_assertion_witnesses'], 'Actual restored mutation gates required.')
    require(campaign['complete'] and campaign['physical_pass'],
            'Registered physical campaign must finish and pass first.')
    audited = load(RESULTS/(PREFIX+'campaign_audit.json'))
    require(audited['complete'] and audited['saved_array_replay'],
            'Independent saved-data replay must finish first.')
    verify_guard(audited['guard_before'])
    verify_guard(target['guard_before'])
    verify_guard(mutations['guard_before'])
    verify_guard(campaign['guard_before'])
    folder = RESULTS/(PREFIX+'prevalidation_sources')
    folder.mkdir(exist_ok=False)
    paths = sorted((ROOT/'src/wavebit').rglob('*.py'))
    paths += sorted((ROOT/'tests').glob('test_fd_*.py')) + [ROOT/'tests/api/test_docs_contract.py']
    paths += helpers()
    snapshots = [snapshot(p, folder/(p.relative_to(ROOT).as_posix().replace('/', '__')+'.raw.gz'))
                 for p in paths]
    record = {'base': BASE, 'prepared_utc': now(), 'guard': guard(), 'snapshots': snapshots,
              'prerequisites': {PREFIX+name+'.json': sha(RESULTS/(PREFIX+name+'.json'))
                                for name in ('targeted', 'mutations', 'campaign', 'campaign_audit')}}
    write(RESULTS/(PREFIX+'prevalidation.json'), record, exclusive=True)
    print('Prepared frozen sources and test inventory:', len(paths), flush=True)


def validation(attempt):
    setup = load(RESULTS/(PREFIX+'prevalidation.json'))
    verify_guard(setup['guard'])
    directory, manifest, record = attempt_record('validation', attempt)
    files = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'tests').glob('test_fd_*.py'))]
    for name, extra, targets in [('full', ['-m', 'not slow'], files+['tests/api/test_docs_contract.py']),
                                 ('slow', ['--run-slow', '-m', 'slow'], files)]:
        pytest_pair(record, manifest, name, targets, directory, extra=extra, workers=5)
        verify_guard(setup['guard'])
    for stage, expected in enumerate((143, 288, 140, 570)):
        row, text = run_command(record, manifest, 'freezer'+str(stage),
            [sys.executable, '-B', f'tests/references/interface/freeze_fd_stage{stage}.py'], directory)
        require(row['exit'] == 0, 'Freezer command failed.')
        require(f'{expected} of {expected} arrays reproduce bit for bit' in text,
                'Frozen reference moved: stop; do not rebaseline.')
        row['arrays_exact'] = expected
        write(manifest, record)
        verify_guard(setup['guard'])
    finish('validation', manifest, record)


def documentation(attempt, sphinx):
    require(sphinx is not None, 'Pass the existing Sphinx executable explicitly.')
    directory, manifest, record = attempt_record('documentation', attempt)
    record['docs_before'] = docs_map()
    write(manifest, record)
    pytest_pair(record, manifest, 'docs', ['tests/api/test_docs_contract.py'], directory)
    output = SCRATCH/('sphinx-attempt'+str(attempt))
    require(not output.exists(), 'Preserve the earlier Sphinx directory.')
    row, text = run_command(record, manifest, 'sphinx',
        [str(Path(sphinx).resolve(strict=True)), '-W', '-b', 'html', 'docs', str(output)], directory)
    require(row['exit'] == 0 and 'build succeeded' in text, 'Sphinx did not succeed.')
    require(record['docs_before'] == docs_map(), 'Docs changed during documentation checks.')
    finish('documentation', manifest, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('targeted', 'prepare', 'validation', 'documentation'))
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--sphinx')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'documentation':
        documentation(args.attempt, args.sphinx)
    else:
        globals()[args.mode](args.attempt)
