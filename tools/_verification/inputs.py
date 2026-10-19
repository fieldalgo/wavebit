"""Conservative input fingerprints without recording arbitrary environment values."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

DIRECTORIES = ('.github', 'src', 'tests', 'tools', 'examples', 'studies', 'tutorials', 'docs', 'assets')
EXCLUDED = {'__pycache__', '.pytest_cache', '.ruff_cache', '_build', '.ipynb_checkpoints',
            '.git', '.venv', 'artifacts', 'build', 'dist'}
ROOT_FILES = ('pyproject.toml', 'pytest.ini', 'setup.cfg', 'setup.py', 'conftest.py',
              'README.md', 'CONTRIBUTING.md', 'CLAUDE.md', 'environment.yml', 'CITATION.bib',
              'CITATION.cff',
              '.gitignore', '.gitattributes', 'LICENSE', 'LICENSE.meent', 'LICENSE.fmmax')
SAFE_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'JAX_ENABLE_X64',
            'JAX_PLATFORMS', 'XLA_FLAGS', 'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def hashes(root):
    root = Path(root)
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in DIRECTORIES:
        directory = root / name
        if not directory.exists():
            continue
        if directory.is_symlink():
            raise ValueError(f'Input symlink is not supported: {directory}')
        for parent, folders, files in os.walk(directory, followlinks=False):
            folders[:] = sorted(folder for folder in folders if folder not in EXCLUDED)
            for folder in folders:
                if (Path(parent) / folder).is_symlink():
                    raise ValueError(f'Input symlink is not supported: {Path(parent)/folder}')
            paths.extend(Path(parent) / file for file in sorted(files)
                         if not file.endswith(('.pyc', '.pyo', '.mo')))
    result = {}
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError(f'Input symlink is not supported: {path}')
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def environment():
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                prefix=sys.prefix, platform=platform.platform(), machine=platform.machine(),
                packages=sorted((distribution.metadata['Name'], distribution.version)
                                for distribution in importlib.metadata.distributions()),
                controls={name: os.environ.get(name) for name in SAFE_ENV})


def snapshot(root, destination, expected):
    destination.mkdir()
    for name in expected:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    actual = hashes(destination)
    if actual != expected:
        raise ValueError('Snapshot bytes differ from the recorded inputs.')
    return actual
