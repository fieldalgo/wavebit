"""Retain backend agreement on the two already exposed calibration configurations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from studies import fdtd_single_pole_film_probe as probe


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root/'studies/results/fdtd_single_pole_film_backends'
    directory.mkdir(exist_ok=False)
    manifest = directory/'manifest.json'
    identity = probe.source_identity()
    snapshot = directory/'controller.py.txt'
    snapshot.write_bytes(Path(__file__).read_bytes())
    # The existing cross-runtime budget is 128 double eps, on a unit-or-signal scale.
    # It is fixed here before these repeated responses, not selected from them.
    record = {'source_identity': identity, 'known_configurations': 'two disclosed n40/p60/f.3 rows',
              'controller_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              'absolute_budget': 128*np.finfo(float).eps, 'workers': [], 'comparisons': [],
              'complete': False}

    def save():
        manifest.write_text(json.dumps(probe._json(record), indent=2)+'\n', encoding='utf-8')

    save()
    for family in probe.FAMILIES:
        rows = {}
        for backend in ('numpy', 'jax', 'torch'):
            name = family+'_'+backend
            output, log = directory/(name+'.json'), directory/(name+'.log')
            command = [sys.executable, '-B', '-m', 'studies.fdtd_single_pole_film_probe',
                       '--family', family, '--backend', backend, '--output', str(output)]
            print('START', name, flush=True)
            with log.open('xb') as stream:
                result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            worker = {'family': family, 'backend': backend, 'command': command,
                      'exit_code': result.returncode, 'raw_log': str(log.relative_to(root)),
                      'raw_log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}
            if output.exists():
                worker.update(result=str(output.relative_to(root)),
                              result_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            record['workers'].append(worker)
            save()
            assert result.returncode == 0 and probe.source_identity() == identity
            rows[backend] = json.loads(output.read_text(encoding='utf-8'))
            print('END', name, flush=True)
        for backend in ('jax', 'torch'):
            for name in ('r', 't'):
                reference = np.array([complex(*v) for v in rows['numpy'][name]])
                actual = np.array([complex(*v) for v in rows[backend][name]])
                difference = abs(actual-reference)
                scale = np.maximum(1., abs(reference))
                record['comparisons'].append({'family': family, 'backend': backend,
                    'coefficient': name, 'absolute_differences': difference.tolist(),
                    'max_scaled_difference': float(max(difference/scale)),
                    'within_fixed_budget': bool(np.all(difference <= record['absolute_budget']*scale))})
    record['complete'] = True
    record['all_within_fixed_budget'] = all(c['within_fixed_budget'] for c in record['comparisons'])
    record['source_unchanged'] = probe.source_identity() == identity
    save()
    assert record['source_unchanged'] and record['all_within_fixed_budget']
    print(json.dumps({'complete': True, 'all_within_fixed_budget': True,
                      'max_difference': max(c['max_scaled_difference'] for c in record['comparisons'])}))


if __name__ == '__main__':
    main()
