"""Build an sdist and its wheel, then check a fresh installation outside the checkout.

Requires build and twine in this Python environment. Public dependencies are downloaded
by pip. No artifacts are published. Run from any directory; --output must not exist.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd, env):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True, timeout=1200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='new directory for distributions, reports and field image')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(output / 'mpl-cache'))
    with tempfile.TemporaryDirectory(prefix='wavebit-distribution-') as tmp:
        work = Path(tmp)
        # Build from a copy: an sdist must not pick up stale egg-info/build outputs,
        # and setuptools must not mutate the developer's checkout.
        source = work / 'source'
        shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
            '.git', '.claude', '.agents', '.codex', 'AGENTS.md', '__pycache__',
            '*.egg-info', 'build', 'dist', '.venv', '.pytest_cache', '.ruff_cache'))
        # build's default builds the wheel FROM its sdist, exercising both artifacts.
        run([sys.executable, '-m', 'build', '--outdir', output], cwd=source, env=env)
        wheels = list(output.glob('*.whl'))
        sdists = list(output.glob('*.tar.gz'))
        if len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError('expected exactly one wheel and one sdist')
        run([sys.executable, '-m', 'twine', 'check', '--strict', *wheels, *sdists],
            cwd=work, env=env)
        target = work / 'venv'
        venv.EnvBuilder(with_pip=True).create(target)
        python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        run([python, '-m', 'pip', 'install', wheels[0]], cwd=work, env=env)
        run([python, '-m', 'pip', 'check'], cwd=work, env=env)
        run([python, '-I', ROOT / 'tests/check_installed_package.py'], cwd=work, env=env)
        with (output / 'fdtd-install.json').open('w', encoding='utf-8') as report:
            subprocess.run([str(python), '-I', str(ROOT / 'studies/fdtd_install_smoke.py')],
                           cwd=work, env=env, stdout=report, check=True, timeout=1200)
        # Request the extra from the same local wheel, never a name-only index release.
        run([python, '-m', 'pip', 'install', f'{wheels[0]}[plot]'], cwd=work, env=env)
        run([python, '-m', 'pip', 'check'], cwd=work, env=env)
        run([python, '-I', ROOT / 'examples/fdtd_record.py',
             '--output', output / 'fdtd-field.png'], cwd=work, env=env)
    print(f'Distribution checks passed; artifacts: {output}', flush=True)


if __name__ == '__main__':
    main()
