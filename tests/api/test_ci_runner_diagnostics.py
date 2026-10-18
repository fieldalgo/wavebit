"""Runner diagnostics retain resource evidence without inspecting process secrets."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from tools import ci_runner_diagnostics as diagnostic


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def test_cgroup_ancestors_include_inherited_limits_and_reject_escape(tmp_path):
    root = tmp_path/'cgroups'
    assert diagnostic.cgroup_paths('0::/job/child', root) == [root, root/'job', root/'job/child']
    assert diagnostic.cgroup_paths('0::/../../outside', root) == []
    assert diagnostic.cgroup_paths('2:memory:/legacy', root) == []
    assert diagnostic.cgroup_paths(None, root) == []


def test_memory_sample_reports_pressure_oom_and_processes_without_secrets(tmp_path):
    proc, group = tmp_path/'proc', tmp_path/'cgroup'
    put(proc/'meminfo', 'MemTotal: 1000000 kB\nMemAvailable: 120 kB\nSwapFree: 0 kB\nSecret: hidden\n')
    put(proc/'vmstat', 'oom_kill 3\npgmajfault 7\nunrelated 9\n')
    put(proc/'pressure/memory', 'some avg10=12.5 total=1234')
    put(proc/'self/cgroup', '0::/job/child')
    put(proc/'42/status', 'Name:\tpython\nPid:\t42\nPPid:\t7\nVmRSS:\t70000 kB\nVmHWM:\t80000 kB\n')
    put(proc/'42/cgroup', '0::/job/child')
    put(proc/'42/cmdline', 'SECRET_ARGUMENT')
    put(proc/'42/environ', 'SECRET_TOKEN')
    put(group/'job/memory.max', '900000')
    put(group/'job/memory.events', 'high 1\nmax 2\noom 3\noom_kill 1')
    row = diagnostic.sample(proc, group)
    assert row['memory']['MemAvailable'] == '120 kB'
    assert row['vmstat'] == {'oom_kill': 3, 'pgmajfault': 7}
    assert row['processes'][0]['PPid'] == '7'
    assert row['cgroups'][str(group/'job')]['memory.max'] == '900000'
    assert 'oom_kill 1' in row['cgroups'][str(group/'job')]['memory.events']
    assert row['cgroups'][str(group)]['memory.max'] is None
    assert 'SECRET' not in json.dumps(row)


def test_disappearing_process_and_unreadable_files_do_not_erase_other_samples(tmp_path):
    proc = tmp_path/'proc'
    (proc/'123').mkdir(parents=True)
    put(proc/'meminfo', 'MemAvailable: 900 kB')
    row = diagnostic.sample(proc, tmp_path/'groups')
    assert row['memory'] == {'MemAvailable': '900 kB'}
    assert row['processes'] == []
    assert row['pressure'] is None


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process signal/proc integration')
def test_sidecar_flushes_memory_and_received_sigterm_without_touching_parent(tmp_path):
    output = tmp_path/'events.jsonl'
    process = subprocess.Popen([sys.executable, '-u', diagnostic.__file__, '--output', str(output),
                                '--interval', '0.05', '--no-kernel-trace'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            if output.exists() and '"event": "memory"' in output.read_text(encoding='utf-8'):
                break
            if process.poll() is not None:
                pytest.fail('sidecar exited before its first sample')
            time.sleep(.02)
        else:
            pytest.fail('sidecar did not flush a memory sample')
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
        assert rows[-1]['event'] == 'monitor_stop'
        assert rows[-1]['received_signals'] == [signal.SIGTERM]
        assert any(row['event'] == 'memory' and row['memory']['MemTotal'] for row in rows)
        assert 'RUNNER_DIAGNOSTIC' in stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_signal_sender_ancestry_is_captured_without_command_lines(tmp_path):
    put(tmp_path/'123/status', 'Name: killer\nPid: 123\nPPid: 7')
    put(tmp_path/'7/status', 'Name: service\nPid: 7\nPPid: 0')
    put(tmp_path/'123/cmdline', 'SECRET_ARGUMENT')
    rows = diagnostic.signal_sender(' killer-123 [001] .... 42.0: signal_generate: sig=15 comm=python pid=456', tmp_path)
    assert [row['Name'] for row in rows] == ['killer', 'service']
    assert 'SECRET' not in json.dumps(rows)
    assert diagnostic.signal_sender('unknown format', tmp_path) == []
