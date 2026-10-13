"""Exercise the recorder on tiny isolated suites, never the large numerical suite."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'checkout'
    (root / 'tests').mkdir(parents=True)
    (root / 'src' / 'wavebit').mkdir(parents=True)
    (root / 'src/wavebit/__init__.py').write_text('VALUE = 7\n')
    (root / 'tools').mkdir()
    shutil.copy2(ROOT / 'tools/run_verification.py', root / 'tools/run_verification.py')
    shutil.copytree(ROOT / 'tools/_verification', root / 'tools/_verification',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'pytest.ini').write_text('[pytest]\nmarkers =\n    slow: slow\n    research: research\n')
    return root


def run(root, output, *arguments):
    env = dict(os.environ)
    env['UNRELATED_SECRET_FOR_TEST'] = 'DO_NOT_RECORD_THIS_SECRET'
    command = [sys.executable, str(root/'tools/run_verification.py'), '--root', str(root),
               '--profile', 'full', '--output', str(output), *arguments]
    result = subprocess.run(command, text=True, capture_output=True, env=env, timeout=60)
    receipt = json.loads((output/'receipt.json').read_text()) if (output/'receipt.json').exists() else None
    return result, receipt


def suite(root, body):
    (root/'tests/test_tiny.py').write_text(textwrap.dedent(body))


def test_outcomes_have_exact_phases_and_no_secret_environment(checkout):
    suite(checkout, '''
        import pytest
        def test_pass(): pass
        def test_fail(): assert False
        @pytest.mark.skip(reason='optional')
        def test_skip(): pass
        @pytest.mark.xfail(reason='known')
        def test_xfail(): assert False
        @pytest.fixture
        def broken(): raise ValueError('setup failure')
        def test_setup(broken): pass
        @pytest.fixture
        def bad_teardown():
            yield
            raise ValueError('teardown failure')
        def test_teardown(bad_teardown): pass
    ''')
    output = checkout/'artifacts/first'
    result, receipt = run(checkout, output)
    assert result.returncode == 1, result.stdout + result.stderr
    assert receipt['counts'] == dict(passed=1, failed=1, skipped=1, xfailed=1, error=2)
    assert receipt['outcomes']['tests/test_tiny.py::test_setup']['failed_phase'] == 'setup'
    assert receipt['outcomes']['tests/test_tiny.py::test_teardown']['failed_phase'] == 'teardown'
    assert all(row['complete'] for row in receipt['outcomes'].values())
    assert (output/'tests/junit.xml').exists()
    assert 'DO_NOT_RECORD_THIS_SECRET' not in (output/'inputs.json').read_text()
    assert receipt['inputs_unchanged'] and not receipt['integrity_errors']
    again, _ = run(checkout, output)
    assert again.returncode == 2 and 'exists' in again.stderr


def test_resume_reuses_only_completed_nonfailures_and_preserves_chain(checkout):
    suite(checkout, '''
        from pathlib import Path
        def test_pass(): pass
        def test_retry():
            sentinel = Path('artifacts/attempt')
            if not sentinel.exists():
                sentinel.parent.mkdir(exist_ok=True)
                sentinel.write_text('attempted')
                assert False
    ''')
    first = checkout/'artifacts/first'
    _, one = run(checkout, first)
    result, two = run(checkout, checkout/'artifacts/second', '--resume', str(first))
    assert result.returncode == 0, result.stdout + result.stderr
    assert two['counts'] == {'passed': 2}
    assert two['reused_nodeids'] == ['tests/test_tiny.py::test_pass']
    assert two['collected_nodeids'] == one['collected_nodeids']
    reports = (checkout/'artifacts/second/tests/events.jsonl').read_text()
    assert 'test_pass' not in reports and 'test_retry' in reports
    _, three = run(checkout, checkout/'artifacts/third', '--resume', str(checkout/'artifacts/second'))
    assert len(three['reused_nodeids']) == 2 and len(three['phases']) == 1


def test_interruption_keeps_finished_nodes_and_resumes_the_unfinished_case(checkout):
    suite(checkout, '''
        from pathlib import Path
        def test_first(): pass
        def test_interrupt():
            flag = Path('artifacts/interrupted')
            if not flag.exists():
                flag.parent.mkdir(exist_ok=True)
                flag.touch()
                raise KeyboardInterrupt
        def test_last(): pass
    ''')
    first = checkout/'artifacts/first'
    result, one = run(checkout, first)
    assert result.returncode == 130
    assert one['interrupted']
    assert one['outcomes']['tests/test_tiny.py::test_first']['complete']
    result, two = run(checkout, checkout/'artifacts/second', '--resume', str(first))
    assert result.returncode == 0, result.stdout + result.stderr
    assert two['counts'] == {'passed': 3}
    assert two['reused_nodeids'] == ['tests/test_tiny.py::test_first']


@pytest.mark.parametrize('change', ['source', 'test', 'configuration', 'environment'])
def test_changed_resume_inputs_are_refused_before_output_creation(checkout, change):
    suite(checkout, 'def test_pass(): pass\n')
    first = checkout/'artifacts/first'
    run(checkout, first)
    extra = []
    if change == 'source':
        (checkout/'src/wavebit/__init__.py').write_text('VALUE = 8\n')
    elif change == 'test':
        (checkout/'tests/test_tiny.py').write_text('def test_other(): pass\n')
    elif change == 'configuration':
        extra = ['--workers', '1']
    else:
        # Only a safe, intentionally recorded control is changed.
        extra = []
        old = os.environ.get('JAX_ENABLE_X64')
        os.environ['JAX_ENABLE_X64'] = 'different-control'
    try:
        result, receipt = run(checkout, checkout/'artifacts/second', '--resume', str(first), *extra)
    finally:
        if change == 'environment':
            if old is None:
                os.environ.pop('JAX_ENABLE_X64')
            else:
                os.environ['JAX_ENABLE_X64'] = old
    assert result.returncode == 2 and receipt is None
    assert 'Resume refused' in result.stderr


def test_profiles_include_slow_only_when_requested(checkout):
    suite(checkout, '''
        import pytest
        def test_ordinary(): pass
        @pytest.mark.slow
        def test_slow(): pass
        @pytest.mark.research
        def test_research(): pass
    ''')
    for profile, count in [('quick', 1), ('full', 2), ('research', 1)]:
        result, receipt = run(checkout, checkout/'artifacts'/profile,
                              '--profile', profile, '--path', 'tests')
        assert result.returncode == 0, result.stdout + result.stderr
        assert receipt['counts'] == {'passed': count}


def test_collection_changes_and_duplicates_are_not_silently_reused(checkout):
    suite(checkout, 'def test_pass(): pass\n')
    (checkout/'conftest.py').write_text(textwrap.dedent('''
        from pathlib import Path
        def pytest_collection_modifyitems(items):
            if Path('artifacts/change').exists():
                items[0]._nodeid = 'tests/test_tiny.py::new_dynamic_name'
    '''))
    first = checkout/'artifacts/first'
    run(checkout, first)
    (checkout/'artifacts/change').touch()
    result, receipt = run(checkout, checkout/'artifacts/second', '--resume', str(first))
    assert result.returncode == 2
    assert any('collected nodeids changed' in error for error in receipt['integrity_errors'])
    (checkout/'conftest.py').write_text('def pytest_collection_modifyitems(items): items.append(items[0])\n')
    result, receipt = run(checkout, checkout/'artifacts/duplicate')
    assert result.returncode == 2
    assert receipt['integrity_errors']


def test_collection_error_is_not_a_green_empty_run(checkout):
    suite(checkout, 'this is not valid Python (\n')
    result, receipt = run(checkout, checkout/'artifacts/bad')
    assert result.returncode == 2
    assert receipt['integrity_errors'] and receipt['collected_nodeids'] == []


def test_source_mutations_run_only_in_a_restored_snapshot(checkout):
    suite(checkout, '''
        from pathlib import Path
        import pytest
        import wavebit
        @pytest.mark.source_mutation
        def test_mutation():
            path = Path(wavebit.__file__)
            assert 'snapshot' in path.parts
            original = path.read_bytes()
            try:
                path.write_text('VALUE = 100\\n')
                assert path.read_bytes() != original
            finally:
                path.write_bytes(original)
    ''')
    before = (checkout/'src/wavebit/__init__.py').read_bytes()
    result, receipt = run(checkout, checkout/'artifacts/run')
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt['snapshot']['initially_byte_identical'] and receipt['snapshot']['restored']
    assert receipt['counts'] == {'passed': 1}
    assert (checkout/'src/wavebit/__init__.py').read_bytes() == before
    origin = json.loads((checkout/'artifacts/run/mutations/import-origin-controller.json').read_text())
    assert '/snapshot/src/wavebit/' in origin['wavebit']


def test_xdist_records_each_phase_once_and_resume_uses_the_same_membership(checkout):
    pytest.importorskip('xdist')
    suite(checkout, 'import pytest\n@pytest.mark.parametrize("i", range(4))\ndef test_case(i): assert i >= 0\n')
    first = checkout/'artifacts/first'
    result, one = run(checkout, first, '--workers', '2')
    assert result.returncode == 0, result.stdout + result.stderr
    assert one['counts'] == {'passed': 4} and not one['integrity_errors']
    rows = [json.loads(x) for x in (first/'tests/events.jsonl').read_text().splitlines()]
    assert len([r for r in rows if r['event'] == 'report']) == 12
    result, two = run(checkout, checkout/'artifacts/second', '--workers', '2', '--resume', str(first))
    assert result.returncode == 0 and len(two['reused_nodeids']) == 4


def test_xpass_is_visible_and_strict_xpass_is_retried(checkout):
    suite(checkout, """
        import pytest
        @pytest.mark.xfail(reason='unexpected success', strict=False)
        def test_nonstrict(): pass
        @pytest.mark.xfail(reason='strict success', strict=True)
        def test_strict(): pass
    """)
    first = checkout/'artifacts/first'
    result, one = run(checkout, first)
    assert result.returncode == 1
    assert one['counts'] == {'xpassed': 1, 'failed': 1}
    result, two = run(checkout, checkout/'artifacts/second', '--resume', str(first))
    assert result.returncode == 1
    assert two['counts'] == one['counts']
    assert two['reused_nodeids'] == ['tests/test_tiny.py::test_nonstrict']
    assert 'test_strict' in (checkout/'artifacts/second/tests/events.jsonl').read_text()


@pytest.mark.skipif(os.name == 'nt', reason='POSIX signal delivery; Windows uses CTRL_BREAK_EVENT')
def test_external_interrupt_is_forwarded_and_the_receipt_is_resumable(checkout):
    suite(checkout, """
        from pathlib import Path
        import time
        def test_first(): pass
        def test_wait():
            flag = Path('artifacts/waiting')
            if not flag.exists():
                flag.parent.mkdir(exist_ok=True)
                flag.touch()
                time.sleep(60)
    """)
    first = checkout/'artifacts/first'
    command = [sys.executable, str(checkout/'tools/run_verification.py'), '--root', str(checkout),
               '--profile', 'full', '--output', str(first)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not (checkout/'artifacts/waiting').exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.02)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 130, stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    one = json.loads((first/'receipt.json').read_text())
    assert one['interrupted']
    assert one['outcomes']['tests/test_tiny.py::test_first']['outcome'] == 'passed'
    result, two = run(checkout, checkout/'artifacts/second', '--resume', str(first))
    assert result.returncode == 0, result.stdout + result.stderr
    assert two['counts'] == {'passed': 2}
    assert two['reused_nodeids'] == ['tests/test_tiny.py::test_first']


def test_unrestored_mutation_is_an_integrity_error_and_never_changes_the_checkout(checkout):
    suite(checkout, """
        from pathlib import Path
        import pytest
        import wavebit
        @pytest.mark.source_mutation
        def test_bad_mutation():
            Path(wavebit.__file__).write_text('VALUE = 99')
    """)
    original = (checkout/'src/wavebit/__init__.py').read_bytes()
    first = checkout/'artifacts/first'
    result, receipt = run(checkout, first)
    assert result.returncode == 2
    assert receipt['counts'] == {'passed': 1}
    assert receipt['snapshot']['restored'] is False
    assert any('not restored' in error for error in receipt['integrity_errors'])
    assert (checkout/'src/wavebit/__init__.py').read_bytes() == original
    result, _ = run(checkout, checkout/'artifacts/second', '--resume', str(first))
    assert result.returncode == 2 and 'Resume refused' in result.stderr


def test_session_failure_cannot_become_green_or_disappear_on_resume(checkout):
    suite(checkout, 'def test_pass(): pass\n')
    (checkout/'conftest.py').write_text("""
def pytest_sessionfinish(session, exitstatus):
    if not session.config.option.collectonly:
        session.exitstatus = 1
""")
    first = checkout/'artifacts/first'
    result, receipt = run(checkout, first)
    assert result.returncode != 0
    assert receipt['counts'] == {'passed': 1}
    assert any('without a failed test report' in error for error in receipt['integrity_errors'])
    result, _ = run(checkout, checkout/'artifacts/second', '--resume', str(first))
    assert result.returncode == 2 and 'Resume refused' in result.stderr


def test_corrupt_receipt_membership_is_refused(checkout):
    suite(checkout, 'def test_pass(): pass\n')
    first = checkout/'artifacts/first'
    _, receipt = run(checkout, first)
    receipt['outcomes']['tests/test_tiny.py::uncollected'] = dict(outcome='passed', complete=True)
    (first/'receipt.json').write_text(json.dumps(receipt))
    result, _ = run(checkout, checkout/'artifacts/second', '--resume', str(first))
    assert result.returncode == 2 and 'exact collection' in result.stderr


def test_inputs_changed_during_a_run_keep_a_nonresumable_receipt(checkout):
    suite(checkout, """
        from pathlib import Path
        def test_changes_a_recorded_file():
            Path('README.md').write_text('a new input appeared')
    """)
    first = checkout/'artifacts/first'
    result, receipt = run(checkout, first)
    assert result.returncode == 2
    assert not receipt['inputs_unchanged'] and receipt['integrity_errors']
    assert receipt['counts'] == {'passed': 1}
    result, _ = run(checkout, checkout/'artifacts/second', '--resume', str(first))
    assert result.returncode == 2 and 'Resume refused' in result.stderr
