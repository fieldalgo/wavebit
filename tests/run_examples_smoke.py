#!/usr/bin/env python3
"""Run every script under ``examples/`` and fail on any sign it did not work.

Not collected by pytest: the filename has no ``test_`` prefix, deliberately, so ``pytest tests
-q`` does not pick it up. The verification suite checks the physics; this only asks whether the examples still *run*.
Keeping them apart gives broken examples a separately named failure. Runtime depends
on the machine and installed backends.

It is still an assert-first check -- both conditions below can be written down before the
run -- so by the routing rule in ``CONTRIBUTING.md`` it lives here rather than in ``studies/``.

A script fails on any of:

* **a non-zero exit status** -- the ordinary signal;
* **``Traceback`` in the output** -- a script that catches its own exception, prints the
  traceback and exits 0 would otherwise pass.

There used to be a third: ``UnphysicalResultWarning`` in the output, on the reasoning that a
script can exit 0 while returning R+T > 1 on a passive stack. That warning no longer exists --
``check_physical`` was removed because it could not tell a broken solve from an active medium
or from a time-domain discretisation, and warned on both. **So this file no longer says
anything about whether an example is physically right**, only that it runs. What replaces it
is per-scene assertions in ``tests/``, where the expected number is known and can be written
down; a smoke run over scripts has no such number and never did.

Usage::

    python tests/run_examples_smoke.py                    # every example
    python tests/run_examples_smoke.py examples/shapes_2d.py  # one script, for iterating
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / 'examples'

# A per-script timeout prevents a hung example from occupying a runner indefinitely.
# This is a smoke-test budget, not a promise about solver performance.
TIMEOUT_S = 300

# Substrings that mean failure even when the script exited 0. Matched against stdout and stderr
# together, because the traceback goes to one and a print to the other.
BAD_OUTPUT = (
    ('Traceback', 'the script raised, and either caught it or printed it on the way out'),
)


def find_scripts(roots: list[Path]) -> list[Path]:
    """Every ``*.py`` under ``roots``, sorted, so the report reads the same on every run.

    ``__init__.py`` and anything under a ``_``- or ``.``-prefixed directory are skipped: those
    are packaging and scratch, not examples someone is meant to run. There are none today.
    """
    out: set[Path] = set()
    for root in roots:
        if root.is_file():
            out.add(root.resolve())
            continue
        for path in root.rglob('*.py'):
            if path.name == '__init__.py':
                continue
            if any(part.startswith(('_', '.')) for part in path.relative_to(root).parts[:-1]):
                continue
            out.add(path.resolve())
    return sorted(out)


def run_one(script: Path) -> tuple[bool, float, list[str], str]:
    """Run one script and judge it. Returns ``(ok, seconds, reasons, output)``."""
    env = dict(os.environ)

    # Plot examples must also finish on runners without an interactive display.
    env['MPLBACKEND'] = 'Agg'

    # The repository ahead of anything installed. Scripts run from their own directory, so the
    # cwd shadowing that normally makes a worktree import its own copy of `wavebit` does not
    # apply, and `import wavebit` would resolve to the editable install -- i.e. to the main
    # checkout, not to the tree being tested. That is a real trap when several worktrees share
    # one install: the examples would pass or fail on somebody else's edits.
    env['PYTHONPATH'] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, script.name],
            cwd=script.parent,  # its own directory: relative data paths and outputs resolve
            env=env,
            capture_output=True,
            text=True,
            errors='replace',
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        partial = (exc.stdout or '') + (exc.stderr or '')
        return False, elapsed, [f'timed out after {TIMEOUT_S}s'], partial

    elapsed = time.perf_counter() - start
    output = proc.stdout + proc.stderr

    reasons = []
    if proc.returncode != 0:
        reasons.append(f'exit status {proc.returncode}')
    for needle, why in BAD_OUTPUT:
        if needle in output:
            reasons.append(f'{needle} in output -- {why}')

    return not reasons, elapsed, reasons, output


def display(script: Path) -> str:
    """A path to print. Relative to the repository when it is inside it, absolute otherwise."""
    try:
        return str(script.relative_to(REPO_ROOT))
    except ValueError:
        return str(script)


def main(argv: list[str]) -> int:
    roots = [Path(a).resolve() for a in argv[1:]] or [EXAMPLES]
    for root in roots:
        if not root.exists():
            print(f'no such path: {root}', file=sys.stderr)
            return 2

    scripts = find_scripts(roots)
    if not scripts:
        # An empty glob is the one way this job can be green without having checked anything.
        print(f'found no example scripts under {", ".join(str(r) for r in roots)}', file=sys.stderr)
        return 2

    print(f'running {len(scripts)} example scripts\n', flush=True)

    failures: list[tuple[str, list[str], str]] = []
    total = 0.0
    for script in scripts:
        rel = display(script)
        print(f'  {rel} ... ', end='', flush=True)
        ok, elapsed, reasons, output = run_one(script)
        total += elapsed
        print(f'{"ok" if ok else "FAIL"} ({elapsed:.1f}s)', flush=True)
        if not ok:
            failures.append((rel, reasons, output))

    print(f'\n{len(scripts) - len(failures)}/{len(scripts)} passed in {total:.1f}s')

    for rel, reasons, output in failures:
        print(f'\n{"=" * 78}\nFAILED  {rel}')
        for reason in reasons:
            print(f'  - {reason}')
        print(f'{"-" * 78}\n{output.rstrip() or "(no output)"}')

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
