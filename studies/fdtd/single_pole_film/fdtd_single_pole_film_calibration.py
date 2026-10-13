"""Exactly two disclosed development rows, with source snapshots before any response.

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

from studies.fdtd.single_pole_film import fdtd_single_pole_film_probe as probe


ROOT = repository_root()
RESULTS = ROOT/'studies'/'results'
PREFIX = 'fdtd_single_pole_film'


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
        **probe.runtime_identity(), 'frozen_sources': snapshots,
        'planned_rows': [{'family': family, 'n': 40, 'pol': 'p', 'fraction': .3,
                          'periods': 60, 'pml': 24} for family in probe.FAMILIES],
        'adaptation': 'None within this attempt. Preserve any failure and review before new work.',
        'workers': [], 'complete': False,
    }
    write(manifest_path, manifest, exclusive=True)
    for family in probe.FAMILIES:
        assert probe.source_identity() == identity, 'source changed before calibration worker'
        log, output = directory/(family+'.log'), directory/(family+'.json')
        command = [sys.executable, '-B', '-m', 'studies.fdtd.single_pole_film.fdtd_single_pole_film_probe',
                   '--family', family, '--n', '40', '--pol', 'p', '--fraction', '0.3',
                   '--periods', '60', '--pml', '24', '--output', str(output)]
        start, begin = now(), time.perf_counter()
        print(f'{start} START {family}', flush=True)
        with log.open('xb') as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     check=False)
        record = {'family': family, 'started_utc': start, 'ended_utc': now(),
                  'wall_s': time.perf_counter()-begin, 'command': command,
                  'exit_code': process.returncode, 'raw_log': str(log.relative_to(ROOT)),
                  'raw_log_sha256': sha(log)}
        if output.exists():
            record.update(result=str(output.relative_to(ROOT)), result_sha256=sha(output))
        manifest['workers'].append(record)
        manifest['source_unchanged'] = probe.source_identity() == identity
        write(manifest_path, manifest)
        print(f'{record["ended_utc"]} END {family} exit={process.returncode}', flush=True)
        if process.returncode != 0 or not manifest['source_unchanged']:
            return 1
    manifest['complete'] = True
    manifest['ended_utc'] = now()
    write(manifest_path, manifest)
    print(json.dumps({'complete': True, 'workers': 2, 'manifest': str(manifest_path.relative_to(ROOT))}))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    raise SystemExit(run(parser.parse_args().attempt))
