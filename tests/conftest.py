"""Suite-wide collection rules for the research and slow tiers.

`research` (registered in `pyproject.toml`) marks the tests whose subject is NVF research code
-- the experimental conforming module `src/wavebit/rcwa/_engine/conforming/`, the NVF study instruments
under `studies/`, and the unused passive closure kernel. They are asserted like everything else
here. They are left out of the default run because they study that experimental code rather
than the ordinary solver and because they are where the suite's time goes. They are a
development aid for that code: work on it runs them, work on the rest of the package does not.
Many drive the ordinary engine (`solve_conv`, `ConvMatrices`, `parametric`), and so do the
NVF models under `studies/rcwa/` that they import, so a product change can break one; it is repaired when
research work next runs the tier. `tests/README.md` says which files are in the tier and why.

A run that names only directories -- `pytest tests`, which is what CI runs -- deselects them.
They run when asked for, in any of three ways:

    pytest -m research tests                      # the research tier alone
    pytest -m "research or not research" tests    # everything
    pytest tests/rcwa/test_nvf_pair_galerkin.py         # a file named on the command line, whole

Any explicit `-m` decides on its own, so this hook only supplies the default. Naming a file
counts as asking for it: the published write-ups under `docs/` reproduce their numbers with
`pytest <file>`, and those commands must keep running what they name.

``slow`` marks expensive physical convergence/gradient checks. Directory runs skip them;
naming a file, ``-k``, ``-m``, ``--run-slow`` or a cached failure rerun opts them in. This
rule is independent of research selection: ``--run-slow`` does not enable research tests.
"""

import pathlib

import pytest


def _named_files(config):
    here = pathlib.Path(config.invocation_params.dir)
    named = set()
    for arg in config.args:
        path = (here / str(arg).split('::', 1)[0]).resolve()
        if path.is_file():
            named.add(path)
    return named


def pytest_addoption(parser):
    parser.addoption('--run-slow', action='store_true', default=False,
                     help='also run tests marked slow when collecting a whole directory')


def _relocate_cached_nodeids(config):
    # A folder move does not repair a failed numerical test. Preserve pytest's
    # last-failed/stepwise identities so reruns still select that same case.
    cache = getattr(config, 'cache', None)
    ledger_path = pathlib.Path(__file__).resolve().parents[1] / 'tools/relocations.json'
    if cache is None or not ledger_path.is_file():
        return
    import json
    paths = json.loads(ledger_path.read_text())['paths']

    def relocate(nodeid):
        path, separator, suffix = nodeid.partition('::')
        return paths.get(path, path) + separator + suffix

    failed = cache.get('cache/lastfailed', {})
    migrated = {relocate(nodeid): value for nodeid, value in failed.items()}
    if migrated != failed:
        cache.set('cache/lastfailed', migrated)
    stepwise = cache.get('cache/stepwise', None)
    if isinstance(stepwise, str) and relocate(stepwise) != stepwise:
        cache.set('cache/stepwise', relocate(stepwise))
    elif isinstance(stepwise, dict) and isinstance(stepwise.get('last_failed'), str):
        migrated = dict(stepwise, last_failed=relocate(stepwise['last_failed']))
        if migrated != stepwise:
            cache.set('cache/stepwise', migrated)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    _relocate_cached_nodeids(config)
    config.addinivalue_line(
        'markers', 'slow: costs minutes; a directory run skips it unless --run-slow or -m asks')
    config.addinivalue_line(
        'markers', 'shard_unit(name): CI may run this node apart from the rest of its file '
        '(tools/run_verification.py --shard); nodes of one file that share a name stay together')


def pytest_collection_modifyitems(config, items):
    # An explicit marker expression owns selection for both tiers.
    if config.option.markexpr:
        return
    named = _named_files(config)
    selected, deselected = [], []
    for item in items:
        asked = pathlib.Path(item.path).resolve() in named
        research = item.get_closest_marker('research') is not None
        (deselected if research and not asked else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    if config.getoption('--run-slow') or config.getoption('keyword'):
        return
    # Skipping a cached failure would clear pytest's last-failed record. Reruns
    # therefore execute those slow cases, as an explicitly named test would.
    failed = set()
    if any(config.getoption(name, False) for name in ('lf', 'failedfirst', 'stepwise')):
        cache = getattr(config, 'cache', None)
        if cache is not None:
            failed = set(cache.get('cache/lastfailed', {}))
            stepwise = cache.get('cache/stepwise', None)
            if isinstance(stepwise, str):
                failed.add(stepwise)
    skip = pytest.mark.skip(reason='slow tier: pass --run-slow or -m slow, or name the file')
    for item in items:
        if (item.get_closest_marker('slow') and item.nodeid not in failed
                and pathlib.Path(item.path).resolve() not in named):
            item.add_marker(skip)


def pytest_sessionstart(session):
    """Refuse an editable installation belonging to another worktree."""
    import importlib.util
    from pathlib import Path
    expected = Path(__file__).resolve().parents[1] / 'src/wavebit/__init__.py'
    if not expected.is_file():
        return  # Collection-policy unit fixtures do not contain the package.
    spec = importlib.util.find_spec('wavebit')
    origin = None if spec is None or spec.origin is None else Path(spec.origin).resolve()
    if origin != expected:
        raise pytest.UsageError(
            'Tests must import this worktree: install it with python -m pip install -e . '
            'in its own environment, or set PYTHONPATH to its src directory. '
            f'Expected {expected}, found {origin}.')
