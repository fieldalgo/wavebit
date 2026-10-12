"""Stdlib gate evidence support for the bounded constitutive Absorber route.

Unexecuted draft. The companion physical support owns the one frozen identity.
Children always have fresh pycache and pytest scratch paths. Canonical records
are exclusive publications after verified completion; retained attempts are never
removed. A publication failure does not undo an already completed child command.
"""
from collections import Counter
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid

from fdtd_constitutive_absorber_support import (
    ROOT, RESULTS, BASE, PREFIX as STEM, TEST as NEW_TEST,
    now, sha, relative, identity, guard, require_base, versions)

PREFIX = STEM + '_'
CONTROL = RESULTS/(PREFIX+'controls')
SCRATCH = ROOT.parent/'fd-c9-prep-work'
TARGETS = (NEW_TEST, 'tests/test_fd_constitutive.py',
           'tests/test_fd_constitutive_study.py', 'tests/test_fd_state_bytes.py',
           'tests/test_fd_decay_stop.py', 'tests/test_fd_decay.py',
           'tests/test_fd_contract.py', 'tests/test_fd_stage3_backends.py')
STATUSES = {'PASSED': 'passed', 'FAILED': 'failed', 'SKIPPED': 'skipped',
            'XFAIL': 'xfailed', 'XPASS': 'xpassed', 'ERROR': 'errors'}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(name):
    path = (ROOT/name).resolve()
    require(path.is_relative_to(ROOT), 'Path escapes worktree: '+str(name))
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value, *, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def snapshot(source, destination):
    raw = source.read_bytes()
    stored = gzip.compress(raw, compresslevel=9, mtime=0)
    require(gzip.decompress(stored) == raw, 'Source compression altered bytes.')
    with destination.open('xb') as stream:
        stream.write(stored)
    return {'source': relative(source), 'file': relative(destination),
            'sha256': digest(raw), 'archive_sha256': digest(stored)}


def verify_snapshot(item, *, current=True):
    path = inside(item['file'])
    require(sha(path) == item['archive_sha256'], 'Compressed snapshot hash mismatch.')
    raw = gzip.decompress(path.read_bytes())
    require(digest(raw) == item['sha256'], 'Decoded source hash mismatch.')
    if current:
        require(raw == inside(item['source']).read_bytes(), 'Live source differs from snapshot.')
    return raw


def docs_map():
    paths = [p for p in (ROOT/'docs').rglob('*') if p.is_file()
             and not {'_build', '__pycache__'}.intersection(p.relative_to(ROOT/'docs').parts)]
    paths += [ROOT/name for name in ('README.md', 'tests/README.md', 'pyproject.toml',
                                    'tests/test_docs_contract.py')]
    return {relative(p): sha(p) for p in sorted(paths)}


def check_log(text):
    require(not any(t in text for t in ('INTERNALERROR', 'node down', 'ERROR collecting',
            'xdist.dsession.Interrupted', 'Traceback (most recent call last)')), 'Abnormal pytest log.')
    require(not re.search(r'(?i)\b(?:worker|node)\b[^\n]*\bcrashed\b', text), 'Worker crashed.')


def nodes(text):
    found = [line.strip() for line in text.splitlines() if line.startswith('tests/') and '::' in line]
    require(found and len(found) == len(set(found)), 'Empty or duplicate collection.')
    summaries = re.findall(r'(?m)^(\d+)(?:/\d+)? tests? collected\b', text)
    require(len(summaries) == 1 and int(summaries[0]) == len(found), 'Collection summary mismatch.')
    return found


def reports(text):
    found = []
    status = '|'.join(STATUSES)
    parallel = re.compile(r'^\[gw\d+\]\s+\[\s*\d+%\]\s+('+status+r')\s+(tests/.+?)\s*$')
    serial = re.compile(r'^(tests/.+?)\s+('+status+r')(?:\s+\([^\n]*\))?(?:\s+\[\s*\d+%\])?\s*$')
    for line in text.splitlines():
        match = parallel.match(line)
        if match:
            kind, node = match.groups()
            found.append((node, STATUSES[kind]))
        else:
            match = serial.match(line)
            if match:
                node, kind = match.groups()
                found.append((node, STATUSES[kind]))
    require(found and len(found) == len(dict(found)), 'Empty or duplicate verbose reports.')
    return dict(found)


def summary(text):
    lines = [line for line in text.splitlines() if re.search(r'\bin \d+(?:\.\d+)?s\b', line)
             and re.search(r'\b\d+ (?:passed|failed|skipped|xfailed|xpassed|errors?)\b', line)]
    require(lines, 'No completed pytest summary.')
    out = dict.fromkeys(STATUSES.values(), 0)
    for count, kind in re.findall(r'\b(\d+) (passed|failed|skipped|xfailed|xpassed|errors?)\b', lines[-1]):
        out['errors' if kind == 'error' else kind] = int(count)
    return out


def audit_pytest(collection_text, run_text, *, allow_assertion_failure=False):
    check_log(collection_text)
    check_log(run_text)
    collected, recorded = nodes(collection_text), reports(run_text)
    require(set(collected) == set(recorded), 'Collected/reported node inventory differs.')
    totals = summary(run_text)
    observed = Counter(recorded.values())
    require(all(totals[k] == observed[k] for k in totals), 'Summary/verbose count mismatch.')
    require(not totals['errors'] and not totals['xfailed'] and not totals['xpassed'], 'Unexpected test status.')
    if not allow_assertion_failure:
        require(not totals['failed'], 'A pytest case failed.')
    return {'collected': len(collected), 'totals': totals, 'node_status': recorded}


def verify_guard(expected):
    require(guard() == expected, 'Frozen source/helper/reference/golden/prior-evidence bytes changed.')


def helpers():
    return sorted((ROOT/'studies').glob(PREFIX+'*.py'))


def run_command(record, manifest, name, command, directory, *, extra_environment=None):
    log = directory/(name+'.log')
    env = dict(os.environ)
    env.update(PYTHONUTF8='1', PYTHONPATH=os.pathsep.join((str(ROOT / 'src'), str(ROOT), str(ROOT/'studies'))),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPYCACHEPREFIX=str(SCRATCH/('pycache-'+name+'-'+uuid.uuid4().hex)))
    if extra_environment:
        require(not {'XLA_FLAGS', 'PYTHONPYCACHEPREFIX'}.intersection(extra_environment),
                'Do not override numerical flags or child cache isolation.')
        env.update(extra_environment)
    require(not env.get('XLA_FLAGS'), 'XLA_FLAGS is forbidden.')
    row = {'name': name, 'command': list(command), 'started_utc': now(),
           'environment': {k: env.get(k) for k in ('PYTHONUTF8', 'PYTHONPATH', 'OMP_NUM_THREADS',
               'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONPYCACHEPREFIX', 'XLA_FLAGS')},
           'exit': None}
    if extra_environment:
        row['extra_environment'] = extra_environment
    begin = time.perf_counter()
    try:
        with log.open('xb') as stream:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        row['exit'] = process.returncode
    except BaseException as error:
        row['launcher_error'] = repr(error)
        raise
    finally:
        row.update(ended_utc=now(), seconds=time.perf_counter()-begin)
        if log.exists():
            row.update(log=relative(log), sha256=sha(log))
        record['checks'].append(row)
        write(manifest, record)
    print(name, 'exit', row['exit'], flush=True)
    return row, log.read_text(encoding='utf-8')


def predecessor_skips():
    path = RESULTS/'fdtd_decay_stop_validation.json'
    prior = load(path)
    require(prior['complete'], 'Committed predecessor validation must be complete.')
    full = [r for r in prior['checks'] if r['name'] == 'full']
    require(len(full) == 1, 'One predecessor full result required.')
    row = full[0]
    log = inside(row['log'])
    require(row['exit'] == 0 and sha(log) == row['sha256'], 'Predecessor full raw output changed.')
    actual = reports(log.read_text(encoding='utf-8'))
    require(actual == row['observed']['node_status'], 'Predecessor skip node proof changed.')
    skipped = sorted(k for k, v in actual.items() if v == 'skipped')
    require(len(skipped) == row['observed']['totals']['skipped'], 'Predecessor skip count differs.')
    return {'manifest': relative(path), 'manifest_sha256': sha(path),
            'log': relative(log), 'log_sha256': sha(log), 'nodes': skipped}


def pytest_pair(record, manifest, name, targets, directory, *, extra=(), workers=0, allowed_skips=()):
    base = [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', *targets, *extra]
    row, collection = run_command(record, manifest, name+'_collection',
                                  base+['-q', '--collect-only'], directory)
    require(row['exit'] == 0, 'Collection command failed.')
    collected = nodes(collection)
    temp = SCRATCH/('pytest-'+name+'-'+uuid.uuid4().hex)
    temp.parent.mkdir(parents=True, exist_ok=True)
    args = ['-vv', '--tb=short', '--basetemp', str(temp)]
    if workers:
        args += ['-n', str(workers)]
    row, text = run_command(record, manifest, name, base+args, directory)
    # Preserve complete node outcomes even when the child fails. No count inferred from its exit.
    try:
        observed = audit_pytest(collection, text, allow_assertion_failure=True)
        row['observed'] = observed
        observed_skips = sorted(k for k, v in observed['node_status'].items() if v == 'skipped')
        row['skip_proof'] = {'actual': observed_skips, 'allowed': sorted(allowed_skips)}
        require(observed_skips == sorted(allowed_skips), 'Actual skipped nodes differ from allowed inventory.')
        require(row['exit'] == 0 and not observed['totals']['failed'], 'Pytest command failed.')
    finally:
        write(manifest, record)
    return row


def attempt_record(kind, attempt):
    require_base()
    require(attempt > 0, 'Attempt must be positive.')
    CONTROL.mkdir(parents=True, exist_ok=True)
    directory = CONTROL/f'{kind}_attempt{attempt}'
    directory.mkdir(exist_ok=False)
    manifest = RESULTS/f'{PREFIX}{kind}_attempt{attempt}.json'
    record = {'base': BASE, 'kind': kind, 'attempt': attempt, 'started_utc': now(),
              'versions': versions(), 'complete': False, 'passed': False,
              'source_before': identity(), 'guard_before': guard(), 'checks': []}
    write(manifest, record, exclusive=True)
    record['runner_snapshots'] = [snapshot(p, directory/(p.name+'.raw.gz')) for p in helpers()]
    write(manifest, record)
    return directory, manifest, record


def failed(manifest, record, error):
    record.update(complete=False, passed=False, ended_utc=now(), failure=repr(error),
                  source_after=identity(), guard_after=guard())
    write(manifest, record)


def finish(kind, manifest, record):
    verify_guard(record['guard_before'])
    record.update(source_after=identity(), guard_after=guard(), ended_utc=now(), complete=True, passed=True)
    require(record['source_before'] == record['source_after'], 'Source identity changed.')
    write(manifest, record)
    write(RESULTS/f'{PREFIX}{kind}.json', record, exclusive=True)
