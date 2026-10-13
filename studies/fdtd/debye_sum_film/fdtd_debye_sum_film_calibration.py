"""One disclosed configuration on three sequential backends, with source snapshots before any response.

This is calibration, not registration of an unseen convergence study. A failed
attempt is preserved and cannot be overwritten. Invoke only after coordination.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

from studies.fdtd.debye_sum_film import fdtd_debye_sum_film_probe as probe


ROOT = repository_root()
RESULTS = ROOT/'studies'/'results'
PREFIX = 'fdtd_debye_sum_film'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(probe._json(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(attempt):
    directory = RESULTS/f'{PREFIX}_calibration_attempt{attempt}'
    directory.mkdir(exist_ok=False)
    manifest_path = RESULTS/f'{PREFIX}_calibration_attempt{attempt}.json'
    assert not manifest_path.exists()
    identity = probe.source_identity()
    snapshots = []
    for label, source in (('probe', pathlib.Path(probe.__file__)),
                           ('controller', pathlib.Path(__file__))):
        destination = directory/(label+'.py.txt')
        with destination.open('xb') as stream:
            stream.write(source.read_bytes())
        snapshots.append({'file': str(destination.relative_to(ROOT)), 'sha256': sha(destination)})
    manifest = {
        'kind': 'disclosed calibration, not a registered convergence verdict',
        'attempt': attempt, 'started_utc': now(), 'source_identity': identity,
        'analytic_preflight': probe.preflight(),
        **probe.runtime_identity(), 'frozen_sources': snapshots,
        'planned_rows': [{'family': 'debye-sum', 'backend': backend, 'n': 40, 'pol': 'p',
                          'fraction': .3, 'periods': 120, 'pml': 24,
                          'eps': 2., 'deltas': [1., 1., 2., 2.], 'taus': [.04, .08, .16, .32]}
                         for backend in ('numpy', 'jax', 'torch')],
        'candidate_airy_ceilings': {'r': .005, 't': .010},
        'ceiling_qualification': 'Inherited candidate ceilings, not accepted from unknown data. Review these disclosed responses before freezing tests.',
        'backend_pair_budget': 128*np.finfo(float).eps,
        'adaptation': 'None within this attempt. Preserve any failure and review before new work.',
        'workers': [], 'complete': False,
    }
    write(manifest_path, manifest, exclusive=True)
    for backend in ('numpy', 'jax', 'torch'):
        assert probe.source_identity() == identity, 'source changed before calibration worker'
        log, output = directory/(backend+'.log'), directory/(backend+'.json')
        command = [sys.executable, '-B', '-m', 'studies.fdtd.debye_sum_film.fdtd_debye_sum_film_probe',
                   '--family', 'debye-sum', '--backend', backend, '--n', '40', '--pol', 'p', '--fraction', '0.3',
                   '--periods', '120', '--pml', '24', '--output', str(output)]
        start, begin = now(), time.perf_counter()
        print(f'{start} START {backend}', flush=True)
        with log.open('xb') as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     check=False)
        record = {'backend': backend, 'family': 'debye-sum', 'started_utc': start, 'ended_utc': now(),
                  'wall_s': time.perf_counter()-begin, 'command': command,
                  'exit_code': process.returncode, 'raw_log': str(log.relative_to(ROOT)),
                  'raw_log_sha256': sha(log)}
        if output.exists():
            record.update(result=str(output.relative_to(ROOT)), result_sha256=sha(output))
        manifest['workers'].append(record)
        manifest['source_unchanged'] = probe.source_identity() == identity
        write(manifest_path, manifest)
        print(f'{record["ended_utc"]} END {backend} exit={process.returncode}', flush=True)
        if process.returncode != 0 or not manifest['source_unchanged']:
            return 1
    rows = {backend: json.loads((directory/(backend+'.json')).read_text(encoding='utf-8'))
            for backend in ('numpy', 'jax', 'torch')}
    comparisons = {}
    for backend in ('jax', 'torch'):
        comparisons[backend] = {}
        for coefficient in ('r', 't'):
            actual = np.array([complex(*z) for z in rows[backend][coefficient]])
            reference = np.array([complex(*z) for z in rows['numpy'][coefficient]])
            comparisons[backend][coefficient] = (abs(actual-reference)/np.maximum(1., abs(reference))).tolist()
    manifest['backend_pair_scaled_differences'] = comparisons
    manifest['backend_pairs_within_budget'] = all(
        max(values) <= manifest['backend_pair_budget']
        for comparison in comparisons.values() for values in comparison.values())
    manifest['complete'] = True
    manifest['ended_utc'] = now()
    write(manifest_path, manifest)
    print(json.dumps({'complete': True, 'workers': 3, 'manifest': str(manifest_path.relative_to(ROOT))}))
    return 0 if manifest['backend_pairs_within_budget'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    raise SystemExit(run(parser.parse_args().attempt))
