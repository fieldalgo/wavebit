"""Preserve collected/reported counts and every targeted test attempt."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from studies.fdtd.debye_sum_film.fdtd_debye_sum_film_probe import source_identity


def main(attempt):
    root = repository_root()
    out = root/'studies/fdtd/debye_sum_film/records/fdtd_debye_sum_film_controls'
    out.mkdir(exist_ok=True)
    stem = f'targeted_attempt{attempt}'
    manifest = out/(stem+'.json')
    assert not manifest.exists(), 'Preserve the earlier attempt.'
    tests = ['tests/fdtd/test_fd_debye_sum_film.py', 'tests/fdtd/test_fd_two_debye_film.py', 'tests/fdtd/test_fd_debye_film.py', 'tests/fdtd/test_fd_single_pole_film.py', 'tests/fdtd/test_fd_conductive_film.py',
             'tests/fdtd/test_fd_stencil.py::test_loss_and_anisotropy_meet_check_smoothed_at_plan_and_the_stencil_door_at_lowering']
    files = sorted({p.split('::')[0] for p in tests})
    snapshots = []
    for source in [Path(__file__), *(root/p for p in files)]:
        snapshot = out/(stem+'_'+source.stem+'.py.txt')
        with snapshot.open('xb') as stream:
            stream.write(source.read_bytes())
        snapshots.append({'file': str(snapshot.relative_to(root)),
                          'sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest()})
    record = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_before': source_identity(), 'complete': False,
              'test_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
              'snapshots': snapshots, 'checks': []}

    def save():
        manifest.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')

    def run(name, args):
        log, begin = out/(name+'.log'), time.perf_counter()
        command = [sys.executable, '-B', '-m', 'pytest', *args]
        with log.open('xb') as stream:
            result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        output = log.read_text(encoding='utf-8')
        record['checks'].append({'name': name, 'command': command, 'exit_code': result.returncode,
            'wall_s': time.perf_counter()-begin, 'raw_log': str(log.relative_to(root)),
            'sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
        save()
        print(output[-3000:], flush=True)
        assert result.returncode == 0
        assert not any(s in output for s in ('INTERNALERROR', 'crashed', 'node down', 'ERROR collecting'))
        return output

    save()
    args = ['-q', '-p', 'no:cacheprovider', *tests]
    collection = run(stem+'_collection', args+['--collect-only'])
    count = sum(s.startswith('tests/') and '::' in s for s in collection.splitlines())
    record['collected'] = count
    save()
    temp = root/'.debye_sum_film_work'/('pytest-'+stem)
    temp.parent.mkdir(exist_ok=True)
    output = run(stem, args+['-vv', '--tb=short', '-n', '5', '--maxfail', '1', '--basetemp', str(temp)])
    totals = {k: int(re.findall(r'(\d+) '+k, output)[-1]) if re.search(r'(\d+) '+k, output) else 0
              for k in ('passed', 'skipped', 'xfailed', 'xpassed')}
    record.update(totals=totals, source_after=source_identity())
    record['source_unchanged'] = record['source_before'] == record['source_after']
    record['complete'] = sum(totals.values()) == count and record['source_unchanged']
    save()
    assert record['complete']
    assert record['test_sha256'] == {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
    print(json.dumps({'collected': count, **totals, 'complete': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, required=True)
    main(parser.parse_args().attempt)
