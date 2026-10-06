"""Freeze restored numerical inputs before the final regression sequence."""
import argparse
from fdtd_inactive_lorentz_support import (ROOT, RESULTS, PREFIX, BASE, NEW_TEST,
    load, write, require, maps, guard, verify_guard, sha, snapshot, helpers, now)


def main():
    baseline = load(RESULTS/(PREFIX+'baseline.json'))
    target = load(RESULTS/(PREFIX+'targeted.json'))
    mutation = load(RESULTS/(PREFIX+'mutations.json'))
    require(target['complete'] and mutation['complete'] and mutation['restored'], 'Finish development gates first.')
    verify_guard(target['guard_before'])
    verify_guard(mutation['guard_before'])
    require(len(mutation['witnesses']) == 3 and all(w['accepted'] for w in mutation['witnesses']), 'Need actual mutation witnesses.')
    current = maps()
    require(current['goldens'] == baseline['goldens'] and current['previous_film_evidence'] == baseline['prior_evidence'], 'Prior raw evidence or goldens changed.')
    require(set(current['wavebit']) == set(baseline['wavebit']) and {k for k,v in baseline['wavebit'].items() if current['wavebit'][k] != v} == {'wavebit/fd/materials.py'}, 'Unexpected production delta.')
    require(set(current['tests'])-set(baseline['tests']) == {NEW_TEST} and all(current['tests'][k] == v for k,v in baseline['tests'].items()), 'Unexpected test delta.')
    output = RESULTS/(PREFIX+'prevalidation.json')
    require(not output.exists(), 'Preserve existing preparation.')
    folder = RESULTS/(PREFIX+'prevalidation_sources')
    folder.mkdir(exist_ok=False)
    sources = [ROOT/'wavebit/fd/materials.py', ROOT/NEW_TEST, *helpers()]
    sources += [ROOT/'tests/test_docs_contract.py']
    snapshots = [snapshot(p, folder/(p.name+'.txt.gz')) for p in sources]
    record = {'utc': now(), 'base': BASE, 'maps': current, 'guard': guard(), 'snapshots': snapshots,
              'baseline_sha256': sha(RESULTS/(PREFIX+'baseline.json')),
              'targeted_sha256': sha(RESULTS/(PREFIX+'targeted.json')),
              'mutations_sha256': sha(RESULTS/(PREFIX+'mutations.json'))}
    write(output, record, exclusive=True)
    print('Prepared exact inputs and', len(snapshots), 'source snapshots; final gates not executed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
