"""Install CI's environments at their recorded versions, and record new versions.

Required CI installs every Python distribution at the exact version listed in
`.github/constraints/<set>.txt`, so a new upstream release cannot change a run on a
commit that did not change. The `latest dependencies` workflow installs the same
environments without the files, runs a smaller set of checks, and writes what it got
with `freeze`; a pull request that replaces the files is how a version moves. See
docs/development/verification.md (Dependency versions in CI).

    python tools/ci_constraints.py install tests             # the recorded versions
    python tools/ci_constraints.py install tests --latest    # the newest releases
    python tools/ci_constraints.py freeze tests              # record what is installed
    python tools/ci_constraints.py env distribution          # PIP_* lines for $GITHUB_ENV

`ENVIRONMENTS` is the one place the CI install commands are written; the workflows call
`install` instead of spelling them out. Standard library only: `install` runs before
anything else is in the environment.
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / '.github' / 'constraints'

# The CPU wheel goes in before the extras, so that pip already sees torch as satisfied and
# does not pull the multi-gigabyte CUDA build from PyPI. That index mirrors only some
# versions of torch's own dependencies (MarkupSafe, setuptools, ...), which is one reason
# `tests` and `docs` record different versions of the same distribution and get one file
# each rather than sharing one.
CPU_TORCH = ['torch', '--index-url', 'https://download.pytorch.org/whl/cpu']

# The build backend, as pyproject.toml declares it. Builds run in an isolated environment
# that `pip freeze` cannot see, so each set installs it once where freeze can, and the
# constraint then reaches the isolated build through PIP_BUILD_CONSTRAINT (pip's own
# builds, `pip install -e .`) and PIP_CONSTRAINT (`python -m build`, whose build
# environment is installed by an ordinary pip call).
BUILD_BACKEND = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['build-system']['requires']

# name -> (constraints set, pip install argument lists, run in order)
ENVIRONMENTS = {
    # `test` and `slow`: both optional backends, contour and plotting, so that no part of
    # the suite skips for want of an import.
    'tests': ('tests', [CPU_TORCH, ['-e', '.[jax,pytorch,contour,plot]', 'pytest', 'pytest-timeout',
                                    'pytest-xdist', *BUILD_BACKEND]]),
    # `examples`: what a reader of the README installs, from the same recorded set.
    'examples': ('tests', [CPU_TORCH, ['-e', '.[jax,pytorch,plot]']]),
    # `docs` and the documentation deploy: autodoc imports the package, NumPy only.
    'docs': ('docs', [['-e', '.[docs]', *BUILD_BACKEND]]),
    # `distribution`, the job: the tools it runs. tools/run_distribution_smoke.py then
    # builds and installs into its own venv, constrained by the environment variables
    # `env` prints, so the set has to cover that venv as well.
    'distribution-tools': ('distribution', [['build', 'twine']]),
    # `distribution`, recorded: the tools, the build backend, and what the smoke installs.
    'distribution': ('distribution', [['build', 'twine', *BUILD_BACKEND, '.[plot]']]),
}

# Never recorded: pip is the installer rather than an installed requirement, and wavebit
# is the checkout under test.
UNRECORDED = {'pip', 'wavebit'}


def canonical(name):
    """PEP 503 normalized name, so that a file sorts and diffs the same however a
    distribution spells itself."""
    return re.sub(r'[-_.]+', '-', name).lower()


def constraints_file(name):
    return CONSTRAINTS / f'{ENVIRONMENTS[name][0]}.txt'


def pip_environment(name):
    """The environment variables that apply a set's file to every pip call, including the
    ones a subprocess or an isolated build makes."""
    path = str(constraints_file(name))
    return {'PIP_CONSTRAINT': path, 'PIP_BUILD_CONSTRAINT': path}


def install(name, *, latest):
    env = dict(os.environ)
    for key in ('PIP_CONSTRAINT', 'PIP_BUILD_CONSTRAINT'):
        env.pop(key, None)
    if not latest:
        if not constraints_file(name).is_file():
            raise SystemExit(f'{constraints_file(name).relative_to(ROOT)} is missing')
        env.update(pip_environment(name))
    for arguments in ENVIRONMENTS[name][1]:
        command = [sys.executable, '-m', 'pip', 'install', *arguments]
        print('+', ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def installed():
    """{canonical name: version} for every distribution in this interpreter's environment,
    local version labels dropped: the CPU index serves `torch 2.14.1+cpu`, and the
    constraint `torch==2.14.1` matches it (PEP 440 ignores a local label the specifier
    does not name) while also holding on an index that has no such label."""
    found = {}
    for dist in metadata.distributions():
        name = canonical(dist.metadata['Name'])
        if name not in UNRECORDED:
            found[name] = dist.version.split('+', 1)[0]
    return found


def render(name, versions):
    users = ', '.join(sorted(k for k, v in ENVIRONMENTS.items() if v[0] == ENVIRONMENTS[name][0]))
    lines = [
        f'# Exact versions required CI installs for: {users}.',
        f'# Written by `python tools/ci_constraints.py freeze {name}` with Python '
        f'{platform.python_version()} on {platform.system()}; replace it the same way, by pull request.',
        '# A distribution missing here (one another OS adds) installs at its newest version.',
    ]
    lines += [f'{key}=={versions[key]}' for key in sorted(versions)]
    return '\n'.join(lines) + '\n'


def freeze(name, output):
    path = Path(output) if output else constraints_file(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(name, installed()), encoding='utf-8', newline='\n')
    print(f'wrote {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('install', 'freeze', 'env'):
        sub = commands.add_parser(command)
        sub.add_argument('name', choices=sorted(ENVIRONMENTS))
        if command == 'install':
            sub.add_argument('--latest', action='store_true',
                             help='ignore the recorded versions and install the newest releases')
        if command == 'freeze':
            sub.add_argument('--output', help='write here instead of over the recorded file')
    args = parser.parse_args(argv)
    if args.command == 'install':
        install(args.name, latest=args.latest)
    elif args.command == 'freeze':
        freeze(args.name, args.output)
    else:
        for key, value in pip_environment(args.name).items():
            print(f'{key}={value}')


if __name__ == '__main__':
    main()
