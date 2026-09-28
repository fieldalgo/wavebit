"""Record the required pytest tiers and read-only freezer checks, sequentially.

The full tier deliberately uses the owner's explicit file selection, including
slow tests; the separate slow tier repeats that selection's marked cases.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT/'studies/results/fdtd_polygon_tie_controls'


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT/'wavebit/fd').rglob('*.py')):
        digest.update(path.relative_to(ROOT/'wavebit/fd').as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def main(*, base='96311b50', task='polygon_tie', maxfail=None):
    LOGS.mkdir(parents=True, exist_ok=True)
    manifest = LOGS.parent/('fdtd_'+task+'_validation.json')
    record = dict(base=base, source=source_hash(), checks=[])
    def save():
        manifest.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    def run(name, args):
        print(name, 'started', flush=True)
        start = time.perf_counter()
        logfile = LOGS/(name+'.log')
        with logfile.open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, *args], cwd=ROOT,
                                    stdout=stream, stderr=subprocess.STDOUT)
        output = logfile.read_text(encoding='utf-8')
        check = dict(name=name, exit=result.returncode, seconds=time.perf_counter()-start)
        record['checks'].append(check)
        save()
        print(name, 'exit', result.returncode, output[-400:], flush=True)
        assert result.returncode == 0, name
        assert not any(s in output for s in ('INTERNALERROR', 'crashed', 'node down',
                                              'ERROR collecting')), name
        assert source_hash() == record['source'], 'Production source changed during validation'
        return check, output
    files = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'tests').glob('test_fd_*.py'))]
    selected = set(sys.argv[1:]) or {'full', 'slow', 'freezers', 'probe'}
    for tier, extra, targets in [('full', [], files+['tests/test_docs_contract.py']),
                                 ('slow', ['--run-slow', '-m', 'slow'], files)]:
        if tier not in selected:
            continue
        args = ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', *targets, *extra]
        _, output = run(tier+'_collection', args+['--collect-only'])
        count = sum(line.startswith('tests/') and '::' in line for line in output.splitlines())
        assert count > 0, (tier, output[-1000:])
        temp = ROOT/('.'+task+'_work')/('pytest-'+tier+'-'+uuid.uuid4().hex)
        temp.parent.mkdir(parents=True, exist_ok=True)
        stop = [] if maxfail is None else ['--maxfail', str(maxfail)]
        check, output = run(tier, args+['-vv', '--tb=short', '-n', '5',
                                       '--durations=15', '--basetemp', str(temp)]+stop)
        totals = {kind: int(re.findall(r'(\d+) '+kind, output)[-1])
                  if re.search(r'(\d+) '+kind, output) else 0
                  for kind in ('passed', 'skipped', 'xfailed', 'xpassed')}
        check.update(collected=count, **totals)
        save()
        assert sum(totals.values()) == count, (tier, count, totals)
    if 'freezers' in selected:
        for stage, expected in enumerate((143, 288, 140, 570)):
            check, output = run('freezer'+str(stage),
                                [f'tests/references/interface/freeze_fd_stage{stage}.py'])
            assert f'{expected} of {expected} arrays reproduce bit for bit' in output
            check['arrays_exact'] = expected
            save()
    if 'probe' in selected:
        run('final_probe', ['studies/fdtd_polygon_ties.py', 'final'])
    save()
    print('All selected checks passed.', flush=True)


if __name__ == '__main__':
    main()
