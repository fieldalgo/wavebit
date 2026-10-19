"""Pytest plugin loaded explicitly by the runner, including xdist workers."""
import json
import os
from pathlib import Path
import time

import pytest

CONTRACT = json.loads(Path(os.environ['WAVEBIT_VERIFICATION_PHASE']).read_text())
DIRECTORY = Path(CONTRACT['directory'])
WORKER = os.environ.get('PYTEST_XDIST_WORKER', 'controller')
MUTATIONS = (
    'tests/rcwa/test_boundary.py::test_the_reduced_basis_mutant_is_killed',
    'tests/rcwa/test_gamma_field.py::test_the_mutant_is_killed',
)


def write(name, value):
    (DIRECTORY / name).write_text(json.dumps(value, indent=2) + '\n')


def event(value):
    if WORKER == 'controller':
        with (DIRECTORY / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
            stream.flush()


def pytest_configure(config):
    config.addinivalue_line('markers', 'source_mutation: temporarily rewrites package sources; run in an isolated serial snapshot')


def shard_unit(item):
    """The CI shard unit of a node marked `shard_unit`, or None for its file's unit.

    A bare marker makes each node its own unit; `shard_unit('name')` puts the nodes of one
    file that carry the same name in one unit, for nodes that share a cache or fixture.
    """
    marker = item.get_closest_marker('shard_unit')
    if marker is None:
        return None
    if marker.kwargs or len(marker.args) > 1 or (
            marker.args and (not isinstance(marker.args[0], str) or not marker.args[0])):
        raise pytest.UsageError(f'{item.nodeid}: shard_unit takes no argument or one name')
    if not marker.args:
        return item.nodeid
    return f"{item.nodeid.split('::', 1)[0]}::@{marker.args[0]}"


def pytest_sessionstart(session):
    root = Path(CONTRACT['root'])
    expected = root / 'src/wavebit/__init__.py'
    if expected.exists():
        import wavebit
        actual = Path(wavebit.__file__).resolve()
        if actual != expected.resolve():
            raise pytest.UsageError(f'Wrong package origin: {actual}; expected {expected}')
        write(f'import-origin-{WORKER}.json', dict(wavebit=str(actual)))


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    nodes = [item.nodeid for item in items]
    if len(nodes) != len(set(nodes)):
        raise pytest.UsageError('Duplicate collected nodeids cannot be recorded safely.')
    mutations = [item.nodeid for item in items
                 if item.get_closest_marker('source_mutation') or
                 any(item.nodeid.startswith(prefix) for prefix in MUTATIONS)]
    units = {item.nodeid: unit for item in items if (unit := shard_unit(item)) is not None}
    write(f'collection-{WORKER}.json', dict(nodeids=nodes, mutation_nodes=mutations, shard_units=units))
    expected = CONTRACT.get('expected')
    if expected is not None and nodes != expected:
        raise pytest.UsageError('Collection changed between verification phases; no partial reuse is allowed.')
    selected = CONTRACT.get('selected')
    if selected is not None:
        selected = set(selected)
        if not selected <= set(nodes):
            raise pytest.UsageError('Requested nodeids are missing from collection.')
        rejected = [item for item in items if item.nodeid not in selected]
        items[:] = [item for item in items if item.nodeid in selected]
        if rejected:
            config.hook.pytest_deselected(items=rejected)
        if not CONTRACT.get('snapshot') and selected.intersection(mutations):
            raise pytest.UsageError('Source mutation tests must run in an isolated snapshot.')


def pytest_collectreport(report):
    if report.failed:
        with (DIRECTORY / f'collection-errors-{WORKER}.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(nodeid=report.nodeid, message=str(report.longrepr))) + '\n')


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):
    write(f'dispatched-{node.gateway.id}.json', list(ids))


def pytest_runtest_logreport(report):
    event(dict(event='report', nodeid=report.nodeid, when=report.when,
               outcome=report.outcome, duration=report.duration,
               wasxfail=getattr(report, 'wasxfail', None),
               message=str(report.longrepr) if report.failed or report.skipped else None,
               time=time.time()))


def pytest_keyboard_interrupt(excinfo):
    event(dict(event='interruption', message=str(excinfo.value), time=time.time()))


def pytest_internalerror(excrepr, excinfo):
    event(dict(event='internal_error', message=str(excrepr), time=time.time()))


def pytest_sessionfinish(session, exitstatus):
    event(dict(event='session_finish', exit_code=int(exitstatus), time=time.time()))
