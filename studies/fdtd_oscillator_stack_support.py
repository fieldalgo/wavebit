"""Stdlib evidence helpers for finite separated single-pole oscillator-film stacks.

Each attempt preserves source identities, raw outputs and actual subprocess exits.
No numerical code is imported by these utilities.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from importlib import metadata
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'studies/results'
PREFIX = 'fdtd_oscillator_stack_'
CONTROL = RESULTS/(PREFIX+'controls')
SCRATCH = ROOT.parent/'fd-oscillator-stack-work'
REFERENCES = ('studies/fdtd_layer_alignment.py', 'studies/fdtd_oblique_pulse.py',
              'studies/fdtd_conductive_film_probe.py', 'tests/test_fdtd_layer_alignment.py',
              'studies/fdtd_conductive_film_campaign.py')
RUNNER_INPUTS = ('tests/conftest.py', 'tests/fd_helpers.py', 'tests/test_docs_contract.py',
                 'pyproject.toml',
                 *(f'tests/references/interface/freeze_fd_stage{i}.py' for i in range(4)),
                 *(f'tests/references/_fd_stage{i}_cases.py' for i in range(4)))
BASE = 'd3ab1c74ac2de7fd8a16c6b06e6fe1c968b7b113'
NEW_TEST = 'tests/test_fd_oscillator_stack.py'
TARGETS = (NEW_TEST, 'tests/test_fd_debye_stack.py', 'tests/test_fd_conductive_stack.py',
           'tests/test_fd_single_pole_film.py', 'tests/test_fd_oscillator_sum_film.py',
           'tests/test_fd_hybrid_film.py', 'tests/test_fd_state_bytes.py',
           'tests/test_fd_materials_doors.py', 'tests/test_fd_dispersive.py',
           'tests/test_fd_decay_stop.py', 'tests/test_fd_stage1_grad.py',
           'tests/test_fd_constitutive_absorber.py')
PRIOR = ('fdtd_conductive_film_', 'fdtd_single_pole_film_', 'fdtd_debye_film_',
         'fdtd_two_debye_film_', 'fdtd_debye_sum_film_', 'fdtd_oscillator_sum_film_',
         'fdtd_state_bytes_', 'fdtd_hybrid_film_', 'fdtd_inactive_lorentz_',
         'fdtd_conductive_stack_', 'fdtd_decay_stop_', 'fdtd_constitutive_absorber_',
         'fdtd_debye_stack_')
ANTICIPATED = {'predecessor_full': 4332, 'existing_skips': 8, 'slow': 81, 'docs': 77,
               'qualification': 'Historical completed Debye-stack counts only; actual inventories are authoritative.'}
STATUSES = {'PASSED': 'passed', 'FAILED': 'failed', 'SKIPPED': 'skipped',
            'XFAIL': 'xfailed', 'XPASS': 'xpassed', 'ERROR': 'errors'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def inside(name):
    path = (ROOT/Path(str(name).replace('\\', '/'))).resolve()
    require(path.is_relative_to(ROOT), 'Path escapes worktree: '+str(name))
    return path


def campaign_output_matches(actual, expected):
    """Compare output fields exactly, allowing contained report-path spelling."""
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    if not all(actual[key] == value for key, value in expected.items() if key != 'report'):
        return False
    if not isinstance(actual.get('report'), str) or not isinstance(expected.get('report'), str):
        return False
    return inside(actual['report']) == inside(expected['report'])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value, *, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def start():
    require(re.fullmatch(r'[0-9a-f]{40}', BASE), 'Expected full predecessor commit identity.')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(head == BASE, 'Run this milestone before committing it, on the registered base.')
    require(not os.environ.get('XLA_FLAGS'), 'Do not use XLA_FLAGS for this milestone.')
    CONTROL.mkdir(parents=True, exist_ok=True)


def inventories():
    return {'wavebit': sorted((ROOT/'wavebit').rglob('*.py')),
            'tests': sorted((ROOT/'tests').glob('test_fd_*.py')),
            'goldens': sorted((ROOT/'tests/references').glob('fd_golden_stage*.npz')),
            'previous_film_evidence': sorted(p for p in RESULTS.rglob('*') if p.is_file()
                and p.relative_to(RESULTS).parts[0].startswith(PRIOR))}


def maps():
    return {name: {relative(p): sha(p) for p in paths} for name, paths in inventories().items()}


def verify_maps(expected):
    require(maps() == expected, 'Production/test/golden/prior-evidence inventory or raw bytes changed.')


def identity():
    answer = {'rule': 'sorted relative POSIX path + NUL + LF bytes + NUL'}
    for label, folder in [('fd', ROOT/'wavebit/fd'), ('package', ROOT/'wavebit')]:
        value = hashlib.sha256()
        for path in sorted(folder.rglob('*.py')):
            value.update(path.relative_to(folder).as_posix().encode()+b'\0')
            value.update(path.read_bytes().replace(b'\r\n', b'\n')+b'\0')
        answer[label] = value.hexdigest()
    return answer


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


def helpers():
    return sorted((ROOT/'studies').glob('fdtd_oscillator_stack_*.py'))


def docs_map():
    paths = [p for p in (ROOT/'docs').rglob('*') if p.is_file()
             and not {'_build', '__pycache__'}.intersection(p.relative_to(ROOT/'docs').parts)]
    paths += [ROOT/name for name in ('README.md', 'tests/README.md', 'pyproject.toml',
                                    'tests/test_docs_contract.py')]
    return {relative(p): sha(p) for p in sorted(paths)}


def guard():
    return {'maps': maps(), 'helpers': {relative(p): sha(p) for p in helpers()},
            'references': {name: sha(ROOT/name) for name in REFERENCES},
            'runner_inputs': {name: sha(ROOT/name) for name in RUNNER_INPUTS}}


def verify_guard(value):
    require(guard() == value, 'Frozen numerical inputs or runner helpers changed.')


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


def run_command(record, manifest, name, command, directory, *, cache_tag=None):
    log = directory/(name+'.log')
    env = dict(os.environ)
    env.update(PYTHONUTF8='1', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PYTHONPYCACHEPREFIX'] = str(SCRATCH/('pycache-'+(cache_tag or name)+'-'+uuid.uuid4().hex))
    require(not env.get('XLA_FLAGS'), 'XLA_FLAGS is forbidden.')
    row = {'name': name, 'command': command, 'started_utc': now(),
           'environment': {k: env.get(k) for k in ('PYTHONUTF8', 'PYTHONPATH', 'OMP_NUM_THREADS',
                          'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONPYCACHEPREFIX', 'XLA_FLAGS')},
           'exit': None}
    begin = time.perf_counter()
    try:
        with log.open('xb') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        row['exit'] = result.returncode
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
    start()
    directory = CONTROL/f'{kind}_attempt{attempt}'
    directory.mkdir(exist_ok=False)
    manifest = RESULTS/f'{PREFIX}{kind}_attempt{attempt}.json'
    require(not manifest.exists(), 'Preserve earlier attempt manifest.')
    versions = {}
    for name in ('numpy', 'jax', 'jaxlib', 'torch', 'pytest', 'pytest-xdist'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    record = {'base': BASE, 'kind': kind, 'attempt': attempt, 'started_utc': now(),
              'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'installed_versions': versions,
              'complete': False, 'source_before': identity(), 'guard_before': guard(),
              'anticipated_only': ANTICIPATED, 'checks': []}
    write(manifest, record, exclusive=True)
    return directory, manifest, record


def finish(kind, manifest, record):
    verify_guard(record['guard_before'])
    record.update(source_after=identity(), ended_utc=now(), complete=True)
    require(record['source_before'] == record['source_after'], 'Production identity changed.')
    write(manifest, record)
    write(RESULTS/f'{PREFIX}{kind}.json', record, exclusive=True)
