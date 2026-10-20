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
import tarfile
import tempfile
import venv
import zipfile


ROOT = Path(__file__).resolve().parents[1]
# Shipped at the sdist root and in the wheel's .dist-info/licenses/, byte for byte.
NOTICES = ('LICENSE', 'LICENSE.meent', 'LICENSE.fmmax', 'NOTICE')


def check_notices(wheel, sdist):
    """Both artifacts carry every license and notice file unchanged."""
    with zipfile.ZipFile(wheel) as archive:
        for name in NOTICES:
            found = [p for p in archive.namelist() if p.endswith(f'.dist-info/licenses/{name}')]
            if len(found) != 1:
                raise RuntimeError(f'{wheel.name}: expected one .dist-info/licenses/{name}, found {found}')
            if archive.read(found[0]) != (ROOT / name).read_bytes():
                raise RuntimeError(f'{wheel.name}: {name} differs from the checkout')
    with tarfile.open(sdist) as archive:
        top = archive.getnames()[0].split('/')[0]
        for name in NOTICES:
            member = archive.extractfile(f'{top}/{name}')
            if member is None or member.read() != (ROOT / name).read_bytes():
                raise RuntimeError(f'{sdist.name}: {name} is missing or differs from the checkout')


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
        check_notices(wheels[0], sdists[0])
        run([sys.executable, '-m', 'twine', 'check', '--strict', *wheels, *sdists],
            cwd=work, env=env)
        target = work / 'venv'
        venv.EnvBuilder(with_pip=True).create(target)
        python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        run([python, '-m', 'pip', 'install', wheels[0]], cwd=work, env=env)
        run([python, '-m', 'pip', 'check'], cwd=work, env=env)
        run([python, '-I', ROOT / 'tests/distribution/check_installed_package.py'], cwd=work, env=env)
        run([python, '-I', ROOT / 'tests/distribution/run_installed_smoke.py'], cwd=work, env=env)
        with (output / 'fdtd-install.json').open('w', encoding='utf-8') as report:
            subprocess.run([str(python), '-I', str(ROOT / 'tests/distribution/fdtd_install_smoke.py')],
                           cwd=work, env=env, stdout=report, check=True, timeout=1200)
        # The current physical FDTD API must work with base NumPy dependencies.
        run([python, '-I', ROOT / 'examples/fdtd/pulse_slab.py'], cwd=work, env=env)
        # Request the extra from the same local wheel, never a name-only index release.
        run([python, '-m', 'pip', 'install', f'{wheels[0]}[plot]'], cwd=work, env=env)
        run([python, '-m', 'pip', 'check'], cwd=work, env=env)
        run([python, '-I', ROOT / 'examples/fdtd/fdtd_record.py',
             '--output', output / 'fdtd-field.png'], cwd=work, env=env)
    print(f'Distribution checks passed; artifacts: {output}', flush=True)


if __name__ == '__main__':
    main()
