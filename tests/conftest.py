"""Suite-wide pytest configuration: the slow tier.

``@pytest.mark.slow`` marks a test that costs minutes, not seconds: every test function with a
case that took 120 s or more in the FDTD line's full suite under ``-n 5`` on the desktop
(2026-09-15), and the tests that share an expensive module fixture with one -- there it is the
fixture that costs, and unmarked it would be built in both halves. They are the FDTD line's
resolution-refinement studies, finite-difference gradient checks and comparisons with Meep, and
develop's painted-overlap polygon tests: slow because they are thorough, not because anything in
them is wrong.

A plain run over a directory (``pytest tests``) skips them, with the reason on each skip (``-rs``
or ``-v`` prints it). Anything more specific runs them: naming a file or a test, any ``-k`` or
``-m`` expression, ``--run-slow``, and a re-run of failures with ``--lf``, ``--ff`` or ``--sw``.
A skip there would also erase the failure from pytest's last-failed record, which counts a skip
as clearing it. CI runs ``-m "not slow"`` and ``-m slow`` as two jobs on every push, so this
splits the schedule, not the assertions: there is still no second, unasserted tier.
"""
import pathlib

import pytest


def pytest_addoption(parser):
    parser.addoption('--run-slow', action='store_true', default=False,
                     help='also run tests marked slow when collecting a whole directory')


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'slow: costs minutes; a directory run skips it unless --run-slow or -m asks')


def pytest_collection_modifyitems(config, items):
    # Any selection the caller wrote is theirs to make: `-m` says slow or not itself, and `-k`
    # names the tests it wants -- `-k slow` included, which matches the marker's own name.
    if (config.getoption('--run-slow') or config.getoption('markexpr')
            or config.getoption('keyword')):
        return
    # A re-run of failures asks for those tests by name, in effect.
    failed = set()
    if any(config.getoption(name, False) for name in ('lf', 'failedfirst', 'stepwise')):
        cache = getattr(config, 'cache', None)
        if cache is not None:
            failed = set(cache.get('cache/lastfailed', {}))
            stepwise = cache.get('cache/stepwise', None)
            if isinstance(stepwise, str):
                failed.add(stepwise)
    root = pathlib.Path(config.invocation_params.dir)
    named = set()
    for arg in config.args:
        path = arg.split('::', 1)[0]
        if path.endswith('.py'):
            named.add((root / path).resolve())
    skip = pytest.mark.skip(reason='slow tier: pass --run-slow or -m slow, or name the file')
    for item in items:
        if (item.get_closest_marker('slow') and item.nodeid not in failed
                and pathlib.Path(item.path).resolve() not in named):
            item.add_marker(skip)
