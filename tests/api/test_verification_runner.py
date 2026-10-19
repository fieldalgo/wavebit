"""Exercise the recorder on tiny isolated suites, never the large numerical suite."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
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
    assert again.returncode == 2 and 'Output directory already exists' in again.stderr


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
    assert Path(origin['wavebit']).parts[-4:-1] == ('snapshot', 'src', 'wavebit')


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


def test_full_slow_shards_partition_the_original_full_profile(checkout):
    suite(checkout, """
        import pytest
        def test_fast(): pass
        @pytest.mark.slow
        def test_slow(): pass
        @pytest.mark.research
        def test_research(): pass
        @pytest.mark.research
        @pytest.mark.slow
        def test_slow_research(): pass
    """)
    members = {}
    for selection in ('include', 'exclude', 'only'):
        result, receipt = run(checkout, checkout/'artifacts'/selection, '--slow', selection)
        assert result.returncode == 0, result.stdout + result.stderr
        members[selection] = set(receipt['collected_nodeids'])
        assert receipt['counts'] == {'passed': len(members[selection])}
    assert members['exclude'].isdisjoint(members['only'])
    assert members['exclude'] | members['only'] == members['include']
    assert len(members['include']) == 2
    result, receipt = run(checkout, checkout/'artifacts/invalid', '--profile', 'quick', '--slow', 'only')
    assert result.returncode == 2 and receipt is None
    assert 'cannot select --slow only' in result.stderr


def test_shards_partition_the_collection_by_whole_file_and_the_check_notices_a_missing_one(checkout):
    (checkout/'tests/test_a.py').write_text('def test_a1(): pass\ndef test_a2(): pass\ndef test_a3(): pass\n')
    (checkout/'tests/test_b.py').write_text('def test_b1(): pass\ndef test_b2(): pass\n')
    (checkout/'tests/test_c.py').write_text('def test_c1(): pass\n')
    receipts = {}
    for index in (1, 2):
        result, receipt = run(checkout, checkout/'artifacts'/f'shard{index}', '--shard', f'{index}/2')
        assert result.returncode == 0, result.stdout + result.stderr
        receipts[index] = receipt
        assert receipt['shard']['index'] == index and receipt['shard']['count'] == 2
        assert receipt['shard']['nodes_total'] == 6
        assert receipt['counts'] == {'passed': receipt['shard']['nodes_in_shard']}
        full = json.loads((checkout/'artifacts'/f'shard{index}/collected-nodeids-all.json').read_text())
        assert len(full) == 6
        for node in receipt['collected_nodeids']:
            assert node.split('::')[0] in receipt['shard']['files']
    # Largest file first, then the lighter shard: a (3) alone; b (2) and c (1) together.
    assert receipts[1]['shard']['files'] == ['tests/test_a.py']
    assert receipts[2]['shard']['files'] == ['tests/test_b.py', 'tests/test_c.py']
    one, two = (set(receipts[index]['collected_nodeids']) for index in (1, 2))
    assert one.isdisjoint(two) and len(one | two) == 6
    # The same run without --shard is the union, in the same order.
    result, whole = run(checkout, checkout/'artifacts/whole')
    assert result.returncode == 0
    assert set(whole['collected_nodeids']) == one | two
    assert 'shard' not in whole
    check = [sys.executable, str(ROOT/'tools/verify_shards.py'), '--count', '2']
    both = [str(checkout/'artifacts/shard1'), str(checkout/'artifacts/shard2')]
    result = subprocess.run(check + both, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.rsplit('\n}', 1)[0] + '\n}')['counts'] == {'passed': 6}
    result = subprocess.run(check + both[:1], text=True, capture_output=True, timeout=60)
    assert result.returncode == 1 and 'missing shard(s): 2/2' in result.stdout
    result = subprocess.run(check + both + both[:1], text=True, capture_output=True, timeout=60)
    assert result.returncode == 1 and 'also recorded by' in result.stdout
    result = subprocess.run(check + [str(checkout/'artifacts/whole'), both[1]], text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 1 and 'not from a --shard run' in result.stdout
    for bad in ('0/2', '3/2', '2', 'a/b'):
        result, receipt = run(checkout, checkout/'artifacts'/('bad' + bad.replace('/', '_')), '--shard', bad)
        assert result.returncode == 2 and receipt is None, bad
        assert '--shard' in result.stderr
    result, receipt = run(checkout, checkout/'artifacts/toomany', '--shard', '4/4')
    assert result.returncode == 2 and 'selects no tests' in receipt['integrity_errors'][0]


@pytest.mark.parametrize('options', [
    ['--slow', 'exclude'], ['--timeout', '2'], ['--timeout-method', 'signal'],
    ['--verbose'], ['--live-output'], ['--shard', '1/2'],
])
def test_ci_configuration_changes_refuse_resume(checkout, options):
    if '--timeout' in options:
        pytest.importorskip('pytest_timeout')
    suite(checkout, 'def test_pass(): pass\n')
    first = checkout/'artifacts/first'
    result, _ = run(checkout, first)
    assert result.returncode == 0
    result, receipt = run(checkout, checkout/'artifacts/second', '--resume', str(first), *options)
    assert result.returncode == 2 and receipt is None
    assert 'Resume refused' in result.stderr


def test_duration_shards_preserve_coverage_and_reject_changed_timing_inputs(checkout):
    # Count balancing would put the three cheap cases alone; measured time must
    # instead isolate the single expensive case, including an unmeasured file.
    (checkout/'tests/test_a.py').write_text('def test_a1(): pass\ndef test_a2(): pass\ndef test_a3(): pass\n')
    (checkout/'tests/test_b.py').write_text('def test_b(): pass\n')
    (checkout/'tests/test_c.py').write_text('def test_c(): pass\n')
    timing_path = checkout.parent/'timings.json'  # also fingerprint external tables
    table = dict(schema='wavebit.shard-timings/v1', partitions=dict(exclude={
        'tests/test_a.py': dict(nodes=3, seconds=3),
        'tests/test_b.py': dict(nodes=1, seconds=100),
    }))
    timing_path.write_text(json.dumps(table))
    options = ['--slow', 'exclude', '--shard-timings', str(timing_path)]
    outputs, receipts = [], []
    for index in (1, 2):
        output = checkout/'artifacts'/f'timed{index}'
        result, receipt = run(checkout, output, '--shard', f'{index}/2', *options)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(str(output))
        receipts.append(receipt)
    assert receipts[0]['shard']['files'] == ['tests/test_b.py']
    assert receipts[1]['shard']['files'] == ['tests/test_a.py', 'tests/test_c.py']
    command = [sys.executable, str(ROOT/'tools/verify_shards.py'), '--count', '2', *outputs]
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    table['partitions']['exclude']['tests/test_b.py']['seconds'] = 110
    timing_path.write_text(json.dumps(table))
    result, _ = run(checkout, checkout/'artifacts/resume-timed', '--shard', '1/2',
                    '--resume', outputs[0], *options)
    assert result.returncode == 2 and 'Resume refused' in result.stderr
    receipts[1]['shard']['timings_digest'] = 'different-table'
    Path(outputs[1], 'receipt.json').write_text(json.dumps(receipts[1]))
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 1 and 'different timing tables' in result.stdout


def test_marked_units_leave_their_file_and_named_units_stay_together(checkout):
    # One heavy file: two independent marked cases, a named pair that shares state, and the
    # unmarked rest. As one unit the whole file would be one shard's load.
    suite(checkout, '''
        import pytest
        @pytest.mark.shard_unit
        @pytest.mark.parametrize('n', [1, 2])
        def test_alone(n): pass
        @pytest.mark.shard_unit('pair')
        def test_pair_fill(): pass
        def test_rest_one(): pass
        @pytest.mark.shard_unit('pair')
        def test_pair_read(): pass
        def test_rest_two(): pass
    ''')
    heavy = 'tests/test_tiny.py'
    timing_path = checkout.parent/'timings.json'
    # A unit's own row wins; `test_alone[2]` has none and costs the file's mean, 10.
    timing_path.write_text(json.dumps(dict(schema='wavebit.shard-timings/v1', partitions=dict(exclude={
        heavy: dict(nodes=6, seconds=60),
        f'{heavy}::test_alone[1]': dict(nodes=1, seconds=40),
        f'{heavy}::@pair': dict(nodes=2, seconds=30)}))))
    options = ['--slow', 'exclude', '--shard-timings', str(timing_path)]
    outputs, selected = [], []
    for index in (1, 2, 3):
        output = checkout/'artifacts'/f'unit{index}'
        result, receipt = run(checkout, output, '--shard', f'{index}/3', *options)
        assert result.returncode == 0, result.stdout + result.stderr
        assert receipt['shard']['files'] == [heavy]
        assert receipt['shard']['units'][f'{heavy}::test_pair_read'] == f'{heavy}::@pair'
        assert f'{heavy}::test_rest_one' not in receipt['shard']['units']
        outputs.append(str(output))
        selected.append([node.split('::')[1] for node in receipt['collected_nodeids']])
    # Largest first to the lightest shard: 40 | 30 | the rest (20), then 10 beside it.
    assert selected == [['test_alone[1]'], ['test_pair_fill', 'test_pair_read'],
                        ['test_alone[2]', 'test_rest_one', 'test_rest_two']]
    command = [sys.executable, str(ROOT/'tools/verify_shards.py'), '--count', '3', *outputs]
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('marker', ["shard_unit('')", "shard_unit('a', 'b')", 'shard_unit(name="a")'])
def test_a_malformed_shard_unit_is_refused_at_collection(checkout, marker):
    suite(checkout, f'import pytest\n@pytest.mark.{marker}\ndef test_one(): pass\n')
    result, receipt = run(checkout, checkout/'artifacts/malformed', '--shard', '1/1')
    assert result.returncode != 0
    assert 'shard_unit takes no argument or one name' in (checkout/'artifacts/malformed/collection/pytest.log').read_text()


@pytest.mark.parametrize('seconds', [-1, float('nan'), float('inf')])
def test_invalid_shard_durations_are_refused_before_collection(checkout, seconds):
    suite(checkout, 'def test_pass(): pass\n')
    timing_path = checkout.parent/'timings.json'
    timing_path.write_text(json.dumps(dict(schema='wavebit.shard-timings/v1',
        partitions=dict(exclude={'tests/test_tiny.py': dict(nodes=1, seconds=seconds)}))))
    result, receipt = run(checkout, checkout/'artifacts/invalid-time', '--shard', '1/1',
                          '--slow', 'exclude', '--shard-timings', str(timing_path))
    assert result.returncode == 2 and receipt is None
    assert 'finite nonnegative seconds' in result.stderr


def test_timing_import_requires_complete_successful_shards(tmp_path):
    for label in ('default', 'slow'):
        directory = tmp_path/f'verification-{label}-1'
        directory.mkdir()
        nodes = ['tests/test_example.py::test_one', 'tests/test_example.py::test_two',
                 'tests/test_example.py::test_apart']
        receipt = dict(shard=dict(index=1, count=1, units={nodes[2]: nodes[2]}), exit_code=0,
                       integrity_errors=[], collect_only=False, inputs_unchanged=True,
                       collected_nodeids=nodes, counts=dict(passed=3), outcomes={
                           node: dict(complete=True, outcome='passed', duration=seconds)
                           for node, seconds in zip(nodes, (2.5, 3.5, 9.0))})
        (directory/'receipt.json').write_text(json.dumps(receipt))
        (directory/'collected-nodeids-all.json').write_text(json.dumps(nodes))
    output = tmp_path/'timings.json'
    command = [sys.executable, str(ROOT/'tools/update_shard_timings.py'), str(tmp_path),
               '--source-run', 'https://example.test/run/1', '--output', str(output)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    table = json.loads(output.read_text())
    # A marked unit is its own row; the rest of the file stays the file's row.
    assert table['partitions']['exclude']['tests/test_example.py'] == dict(nodes=2, seconds=6)
    assert table['partitions']['exclude'][nodes[2]] == dict(nodes=1, seconds=9)
    assert table['partitions']['only'] == table['partitions']['exclude']
    original = output.read_bytes()
    receipt['outcomes'][nodes[0]]['complete'] = False
    (directory/'receipt.json').write_text(json.dumps(receipt))
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode != 0 and 'incomplete or invalid timing' in result.stderr
    assert output.read_bytes() == original
    receipt['shard']['count'] = 2
    (directory/'receipt.json').write_text(json.dumps(receipt))
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode != 0 and 'incomplete baseline' in result.stderr
    assert output.read_bytes() == original


@pytest.mark.parametrize('seconds', ['0', '-1', 'nan', 'inf'])
def test_timeout_requires_positive_finite_seconds(checkout, seconds):
    suite(checkout, 'def test_pass(): pass\n')
    result, receipt = run(checkout, checkout/'artifacts/invalid', '--timeout', seconds)
    assert result.returncode == 2 and receipt is None
    assert 'positive finite' in result.stderr


def test_timeout_missing_dependency_is_an_explicit_preflight_error(checkout):
    suite(checkout, 'def test_pass(): pass\n')
    script = """
import importlib.util
import runpy
import sys
original = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args: None if name == 'pytest_timeout' else original(name, *args)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    output = checkout/'artifacts/missing'
    env = dict(os.environ, PYTHONPATH=str(checkout/'tools'))
    result = subprocess.run([sys.executable, '-c', script, str(checkout/'tools/run_verification.py'),
                             '--root', str(checkout), '--profile', 'full', '--timeout', '1',
                             '--output', str(output)], env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 2
    assert '--timeout requires pytest-timeout' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('mutation', [False, True], ids=['ordinary', 'snapshot'])
def test_thread_timeout_preserves_completed_events_and_cannot_report_success(checkout, mutation):
    pytest.importorskip('pytest_timeout')
    suite(checkout, """
        import pytest
        import time
        def test_first(): pass
        MARKER
        def test_timeout(): time.sleep(10)
    """.replace('MARKER', '@pytest.mark.source_mutation' if mutation else ''))
    result, receipt = run(checkout, checkout/'artifacts/timeout', '--timeout', '.5',
                          '--timeout-method', 'thread', '--verbose', '--live-output')
    assert result.returncode != 0
    assert receipt['outcomes']['tests/test_tiny.py::test_first']['outcome'] == 'passed'
    unfinished = receipt['outcomes']['tests/test_tiny.py::test_timeout']
    assert not unfinished['complete'] and unfinished['outcome'] == 'interrupted'
    phase = next(phase for phase in receipt['phases'] if phase['name'] == ('mutations' if mutation else 'tests'))
    assert phase['exit_code'] != 0
    assert '--timeout=0.5' in phase['command'] and '--timeout-method=thread' in phase['command']
    assert 'pytest_timeout' in phase['command']
    assert 'Timeout' in Path(phase['directory'], 'pytest.log').read_text()
    assert receipt['inputs_unchanged']
    if mutation:
        assert receipt['snapshot']['restored'] and '-n' not in phase['command']


def test_live_output_is_flushed_before_a_running_test_finishes(checkout):
    suite(checkout, """
        from pathlib import Path
        import time
        def test_wait_for_release():
            deadline = time.monotonic() + 15
            while not Path('artifacts/release').exists():
                assert time.monotonic() < deadline
                time.sleep(.02)
    """)
    output = checkout/'artifacts/live'
    command = [sys.executable, str(checkout/'tools/run_verification.py'), '--root', str(checkout),
               '--profile', 'full', '--output', str(output), '--verbose', '--live-output']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    chunks, seen = [], threading.Event()
    needle = b'tests/test_tiny.py::test_wait_for_release '
    def capture():
        while block := process.stdout.read1(65536):
            chunks.append(block)
            if needle in b''.join(chunks):
                seen.set()
    reader = threading.Thread(target=capture, daemon=True)
    reader.start()
    try:
        assert seen.wait(timeout=10), b''.join(chunks).decode(errors='replace')
        assert process.poll() is None
        assert needle in (output/'tests/pytest.log').read_bytes()
        (checkout/'artifacts/release').touch()
        assert process.wait(timeout=15) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        reader.join(timeout=5)
        process.stdout.close()
    assert (output/'tests/pytest.log').read_bytes() in b''.join(chunks)
    receipt = json.loads((output/'receipt.json').read_text())
    assert receipt['counts'] == {'passed': 1}
    assert '-v' in receipt['phases'][1]['command']
    assert '-q' in receipt['phases'][0]['command']
