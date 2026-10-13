"""Freeze restored numerical inputs before the final regression sequence."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
from studies.fdtd.conductive_stack.fdtd_conductive_stack_support import ROOT, RESULTS, PREFIX, BASE, NEW_TEST, load, write, require, maps, guard, verify_guard, sha, snapshot, helpers, now


def main():
    baseline = load(RESULTS/(PREFIX+'baseline.json'))
    target = load(RESULTS/(PREFIX+'targeted.json'))
    mutation = load(RESULTS/(PREFIX+'mutations.json'))
    require(target['complete'] and mutation['passed'] and mutation['restored'], 'Finish development gates first.')
    verify_guard(target['guard_before'])
    verify_guard(mutation['guard_before'])
    require(mutation['all_function_assertion_witnesses'] and len({w['test'].split('[')[0] for w in mutation['witnesses']}) == 9 and all(w['accepted'] for w in mutation['witnesses']), 'Need actual mutation witnesses.')
    current = maps()
    require(current['goldens'] == baseline['goldens'] and current['previous_film_evidence'] == baseline['prior_evidence'], 'Prior raw evidence or goldens changed.')
    application = load(RESULTS/(PREFIX+'application.json'))
    changed = {item['target'] for item in application['production'] if item['kind'] == 'modify'}
    added = {item['target'] for item in application['production'] if item['kind'] == 'add'}
    require(set(current['wavebit'])-set(baseline['wavebit']) == added and set(baseline['wavebit']) <= set(current['wavebit']), 'Unexpected production inventory.')
    require({k for k,v in baseline['wavebit'].items() if current['wavebit'][k] != v} == changed, 'Unexpected production delta.')
    require(set(current['tests'])-set(baseline['tests']) == {NEW_TEST} and all(current['tests'][k] == v for k,v in baseline['tests'].items()), 'Unexpected test delta.')
    campaign = load(RESULTS/(PREFIX+'report.json'))
    require(campaign['complete'], 'Finish the registered physical campaign first.')
    output = RESULTS/(PREFIX+'prevalidation.json')
    require(not output.exists(), 'Preserve existing preparation.')
    folder = RESULTS/(PREFIX+'prevalidation_sources')
    folder.mkdir(exist_ok=False)
    sources = [*(ROOT/name for name in sorted(changed|added)), ROOT/NEW_TEST, *helpers()]
    sources += [ROOT/'tests/api/test_docs_contract.py']
    snapshots = [snapshot(p, folder/(p.name+'.raw.gz')) for p in sources]
    record = {'utc': now(), 'base': BASE, 'maps': current, 'guard': guard(), 'snapshots': snapshots,
              'baseline_sha256': sha(RESULTS/(PREFIX+'baseline.json')),
              'targeted_sha256': sha(RESULTS/(PREFIX+'targeted.json')),
              'mutations_sha256': sha(RESULTS/(PREFIX+'mutations.json')),
              'campaign_sha256': sha(RESULTS/(PREFIX+'report.json'))}
    write(output, record, exclusive=True)
    print('Prepared exact inputs and', len(snapshots), 'source snapshots; final gates not executed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
