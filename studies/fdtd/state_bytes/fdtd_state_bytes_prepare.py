"""Capture a pre-application baseline, then freeze post-restoration gate inputs.

No wavebit imports, allocation tests or field solves. The baseline mode MUST
run before applying _check.py/new test; the validation mode follows successful
targeted/mutation gates. Each output is exclusive; retain partial attempts.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import subprocess

from studies.fdtd.state_bytes.fdtd_state_bytes_support import ROOT, RESULTS, CONTROL, PREFIX, BASE, NEW_TEST, TARGETS, ANTICIPATED, start, maps, helpers, guard, identity, now, relative, sha, snapshot, load, write, require, verify_guard, reports, digest, summary, check_log


def baseline():
    start()
    require(not (ROOT/NEW_TEST).exists(), 'Capture baseline BEFORE adding the new test.')
    inventory = maps()
    require(len(inventory['wavebit']) == ANTICIPATED['production_files'], 'Unexpected package count.')
    require(len(inventory['goldens']) == 4, 'Expected four frozen archives.')
    # The base commit, not a dirty after-fix file, is the pre-application source.
    for section in ('wavebit', 'tests', 'goldens', 'previous_film_evidence'):
        for name, expected in inventory[section].items():
            raw = subprocess.check_output(['git', 'cat-file', '--filters', f'{BASE}:{name}'], cwd=ROOT)
            require(digest(raw) == expected, 'Baseline differs from checked-out base bytes: '+name)
    prior_log = ROOT/'studies/fdtd/oscillator_sum_film/records/fdtd_oscillator_sum_film_controls/full.log'
    prior_text = prior_log.read_text(encoding='utf-8')
    check_log(prior_text)
    previous = reports(prior_text)
    prior_manifest = ROOT/'studies/fdtd/oscillator_sum_film/records/fdtd_oscillator_sum_film_validation.json'
    prior_full = next(c for c in load(prior_manifest)['checks'] if c['name'] == 'full')
    totals = summary(prior_text)
    require(prior_full['exit'] == 0 and prior_full['collected'] == len(previous)
            and all(prior_full[k] == totals[k] for k in ('passed', 'skipped', 'xfailed', 'xpassed')),
            'Prior full manifest/log counts differ.')
    require(all(kind in ('passed', 'skipped') for kind in previous.values()), 'Prior full tier did not pass.')
    directory = CONTROL/'baseline'
    directory.mkdir(exist_ok=False)
    snapshots = [snapshot(ROOT/'src/wavebit/fdtd/_prepare/report.py', directory/'original_check.py.txt.gz')]
    snapshots += [snapshot(p, directory/(p.name+'.txt.gz')) for p in helpers()]
    value = {'base': BASE, 'created_utc': now(), 'raw_byte_sha256': True, 'maps': inventory,
             'counts': {key: len(items) for key, items in inventory.items()},
             'source_identity': identity(), 'snapshots': snapshots,
             'previous_full': {'file': relative(prior_log), 'sha256': sha(prior_log), 'node_status': previous,
                               'manifest': relative(prior_manifest), 'manifest_sha256': sha(prior_manifest)},
             'qualification': 'Captured before the diagnostic change; no new numerical acceptance claim.'}
    require(maps() == inventory, 'Inputs changed during baseline capture.')
    write(RESULTS/(PREFIX+'baseline.json'), value, exclusive=True)
    print('Pre-application baseline captured; apply only the separately reviewed report fix.', flush=True)


def preparation():
    start()
    baseline = load(RESULTS/(PREFIX+'baseline.json'))
    current = maps()
    require(baseline['base'] == BASE, 'Wrong baseline commit.')
    for section in ('goldens', 'previous_film_evidence'):
        require(current[section] == baseline['maps'][section], 'Prior evidence or golden changed.')
    require(set(current['wavebit']) == set(baseline['maps']['wavebit']), 'Production file inventory changed.')
    changed = {k for k in current['wavebit'] if current['wavebit'][k] != baseline['maps']['wavebit'][k]}
    require(changed == {'wavebit/fd/_check.py'}, 'Only the report module may change.')
    require(set(current['tests'])-set(baseline['maps']['tests']) == {NEW_TEST}, 'Unexpected new FD tests.')
    require(all(current['tests'].get(k) == v for k, v in baseline['maps']['tests'].items()),
            'Existing FD tests changed.')
    require(len(current['wavebit']) == 101 and len(current['tests']) == 84, 'Re-review changed source inventory.')
    for kind in ('targeted', 'mutations'):
        value = load(RESULTS/(PREFIX+kind+'.json'))
        require(value['complete'] and value['source_after'] == identity(), kind+' gate is incomplete/stale.')
        verify_guard(value['guard_before'])
    directory = CONTROL/'frozen_sources'
    directory.mkdir(exist_ok=False)
    paths = [*helpers(), *(ROOT/p for p in TARGETS), ROOT/'tests/api/test_docs_contract.py',
             ROOT/'src/wavebit/fdtd/_prepare/report.py']
    paths += [ROOT/f'tests/references/interface/freeze_fd_stage{i}.py' for i in range(4)]
    paths += [ROOT/f'tests/references/_fd_stage{i}_cases.py' for i in range(4)]
    snapshots = [snapshot(p, directory/(relative(p).replace('/', '__')+'.txt.gz')) for p in paths]
    value = {'base': BASE, 'created_utc': now(), 'raw_byte_sha256': True, 'maps': current,
             'source_identity': identity(), 'guard': guard(), 'snapshots': snapshots,
             'baseline_sha256': sha(RESULTS/(PREFIX+'baseline.json')),
             'anticipated_only': ANTICIPATED,
             'qualification': 'Prepared after exact mutation restoration; no full/docs result yet.'}
    require(maps() == current, 'Inputs changed during preparation.')
    write(RESULTS/(PREFIX+'prevalidation.json'), value, exclusive=True)
    print('Final numerical source and helpers frozen; no final gate executed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('baseline', 'validation'))
    args = parser.parse_args()
    (baseline if args.phase == 'baseline' else preparation)()
