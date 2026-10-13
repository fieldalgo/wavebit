"""Unexecuted C9 physical-study support; stdlib only, exclusive evidence writes."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = repository_root()
PREFIX = 'fdtd_constitutive_absorber'
BASE = 'b4ed5cdb5ea3e9ec5b04d636769e332466897e24'
RESULTS = ROOT / 'studies/results'
HELPERS = tuple('studies/' + PREFIX + '_' + name + '.py' for name in
                ('support', 'probe', 'calibration', 'campaign'))
TEST = 'tests/fdtd/test_fd_constitutive_absorber.py'
PRIOR = ('fdtd_conductive_film_', 'fdtd_single_pole_film_', 'fdtd_debye_film_',
         'fdtd_two_debye_film_', 'fdtd_debye_sum_film_', 'fdtd_oscillator_sum_film_',
         'fdtd_state_bytes_', 'fdtd_hybrid_film_', 'fdtd_inactive_lorentz_',
         'fdtd_conductive_stack_', 'fdtd_decay_stop_')
REFERENCES = ('pyproject.toml', 'tests/conftest.py', 'tests/fixtures/fdtd/fd_helpers.py',
              'tests/api/test_docs_contract.py',
              'tests/references/_fd_stage0_cases.py', 'tests/references/_fd_stage1_cases.py',
              'tests/references/_fd_stage2_cases.py', 'tests/references/_fd_stage3_cases.py',
              'tests/references/interface/freeze_fd_stage0.py',
              'tests/references/interface/freeze_fd_stage1.py',
              'tests/references/interface/freeze_fd_stage2.py',
              'tests/references/interface/freeze_fd_stage3.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def identity():
    paths = sorted(ROOT.joinpath('src/wavebit').rglob('*.py'))
    paths += sorted((ROOT/'studies').glob(PREFIX+'_*.py'))
    paths += sorted((ROOT/'tests').glob('test_fd_*.py'))
    paths += [ROOT/p for p in REFERENCES]
    paths = sorted(set(paths))
    return {relative(p): sha(p) for p in paths}


def guard():
    goldens = {relative(p): sha(p) for p in sorted((ROOT/'tests/references').glob('fd_golden_stage*.npz'))}
    if len(goldens) != 4:
        raise ValueError('Exactly four unchanged golden files are required.')
    previous = {relative(p): sha(p) for p in sorted(RESULTS.rglob('*'))
                if p.is_file() and p.relative_to(RESULTS).parts[0].startswith(PRIOR)}
    if not previous:
        raise ValueError('Missing previous milestones including Decay evidence.')
    return {'source': identity(), 'goldens': goldens, 'previous_evidence': previous}


def versions():
    found = {'python': sys.version, 'platform': platform.platform(), 'executable': sys.executable}
    for name in ('numpy', 'scipy', 'jax', 'jaxlib', 'torch'):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    found['environment'] = {k: os.environ.get(k) for k in
        ('JAX_ENABLE_X64', 'JAX_PLATFORMS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS')}
    return found


def require_base():
    if len(BASE) != 40 or any(c not in '0123456789abcdef' for c in BASE):
        raise ValueError('Resolve the full committed Decay predecessor before any execution.')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != BASE:
        raise ValueError('This pre-commit experiment requires the exact registered predecessor HEAD.')
    if os.environ.get('XLA_FLAGS'):
        raise ValueError('No XLA_FLAGS are permitted.')
    return head


def run_worker(config, folder):
    """One fresh child, one solve. No retry, data replacement or stdout self-hash."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    config_path, result_path = folder / 'config.json', folder / 'result.json'
    write(config_path, config)
    command = [sys.executable, '-B', str(ROOT / HELPERS[1]), '--config', str(config_path),
               '--out', str(result_path)]
    start, clock = now(), time.perf_counter()
    log = folder / 'stdout.log'
    failure, exit_code = None, None
    try:
        with log.open('xb') as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                     check=False)
        exit_code = process.returncode
    except BaseException as error:
        failure = repr(error)
    record = {'command': command, 'started_utc': start, 'ended_utc': now(),
              'seconds': time.perf_counter() - clock, 'exit_code': exit_code, 'launcher_error': failure,
              'config': relative(config_path), 'config_sha256': sha(config_path),
              'log': relative(log), 'log_sha256': sha(log) if log.exists() else None, 'result': relative(result_path),
              'result_sha256': sha(result_path) if result_path.exists() else None}
    write(folder / 'execution.json', record)
    return record
