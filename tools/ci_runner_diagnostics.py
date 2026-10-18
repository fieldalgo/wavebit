"""Linux CI sidecar: bounded memory samples and optional kernel signal/OOM events.

No solver imports, command lines, environment dumps, or changes to test processes.
Kernel tracing uses a private tracefs instance and needs passwordless sudo in CI.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
import re
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time


def read(path):
    try:
        return path.read_text(encoding='utf-8').strip()
    except (OSError, UnicodeError):
        return None


def fields(path, names):
    text = read(path)
    if text is None:
        return None
    result = {}
    for line in text.splitlines():
        key, _, value = line.partition(':')
        if key in names:
            result[key] = value.strip()
    return result


def cgroup_paths(text, root):
    """Include limits/events inherited from cgroup v2 ancestors, without escape."""
    if text is None:
        return []
    result = set()
    root = root.resolve()
    for line in text.splitlines():
        if not line.startswith('0::'):
            continue
        path = (root / line[3:].lstrip('/')).resolve()
        if not path.is_relative_to(root):
            continue
        while True:
            result.add(path)
            if path == root:
                break
            path = path.parent
    return sorted(result)


def sample(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    processes = []
    groups = set(cgroup_paths(read(proc/'self/cgroup'), cgroup))
    for entry in proc.iterdir():
        if not entry.name.isdecimal():
            continue
        status = fields(entry/'status', {'Name', 'Pid', 'PPid', 'VmRSS', 'VmHWM', 'VmSwap', 'Threads'})
        if not status:
            continue
        rss = int(status.get('VmRSS', '0').split()[0])
        if rss >= 65536 or status.get('Name', '').startswith(('Runner', 'python', 'systemd-oomd')):
            processes.append(status)
            groups.update(cgroup_paths(read(entry/'cgroup'), cgroup))
    processes.sort(key=lambda row: int(row.get('VmRSS', '0').split()[0]), reverse=True)
    names = ('memory.current', 'memory.peak', 'memory.max', 'memory.high',
             'memory.swap.current', 'memory.swap.max', 'memory.events', 'memory.events.local',
             'memory.pressure')
    return dict(memory=fields(proc/'meminfo', {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'}),
                pressure=read(proc/'pressure/memory'),
                vmstat={line.split()[0]: int(line.split()[1]) for line in (read(proc/'vmstat') or '').splitlines()
                        if line.split()[0] in {'oom_kill', 'pgmajfault', 'pswpin', 'pswpout'}},
                processes=processes[:24],
                cgroups={str(path): {name: read(path/name) for name in names} for path in sorted(groups)})



def signal_sender(line, proc=Path('/proc')):
    match = re.search(r'-(\d+)\s+\[', line)
    chain = []
    pid = match.group(1) if match else None
    for _ in range(4):
        if not pid or pid == '0':
            break
        row = fields(proc/pid/'status', {'Name', 'Pid', 'PPid'})
        if not row:
            break
        chain.append(row)
        parent = row.get('PPid')
        if parent == pid:
            break
        pid = parent
    return chain


def kernel_trace():
    """Privileged helper; stdin EOF closes this helper and removes its own instance."""
    def emit(**data):
        print(json.dumps(data), flush=True)
    instance = None
    try:
        base = Path('/sys/kernel/tracing')
        if not (base/'instances').is_dir():
            mounted = subprocess.run(['mount', '-t', 'tracefs', 'tracefs', str(base)],
                                     capture_output=True, text=True, timeout=10)
            if mounted.returncode:
                raise RuntimeError('tracefs mount unavailable: '+mounted.stderr.strip()[:300])
        instance = base/'instances'/f'wavebit-signals-{os.getpid()}'
        instance.mkdir()
        (instance/'buffer_size_kb').write_text('128', encoding='ascii')
        event = instance/'events/signal/signal_generate'
        (event/'filter').write_text('sig == 1 || sig == 2 || sig == 9 || sig == 15 || sig == 18', encoding='ascii')
        (event/'enable').write_text('1', encoding='ascii')
        oom = instance/'events/oom/mark_victim/enable'
        if oom.exists():
            oom.write_text('1', encoding='ascii')
        (instance/'tracing_on').write_text('1', encoding='ascii')
        emit(event='trace_ready', pid=os.getpid(), oom_event=oom.exists(),
             note='prefix is sender task/PID; signal_generate payload identifies target; SIGCONT is self-probe')
        with (instance/'trace_pipe').open('rb', buffering=0) as stream:
            os.kill(os.getpid(), signal.SIGCONT)
            pending = b''
            while True:
                ready, _, _ = select.select([stream, sys.stdin.buffer], [], [], 1)
                if sys.stdin.buffer in ready and not os.read(sys.stdin.fileno(), 1):
                    break
                if stream in ready:
                    pending += os.read(stream.fileno(), 65536)
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        decoded = line.decode('utf-8', errors='replace')[:1200]
                        emit(event='kernel_event', trace=decoded, sender_ancestry=signal_sender(decoded))
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        emit(event='trace_unavailable', error=str(error)[:500])
        return 1
    finally:
        if instance is not None and instance.exists():
            try:
                (instance/'tracing_on').write_text('0', encoding='ascii')
                (instance/'events/enable').write_text('0', encoding='ascii')
                instance.rmdir()
            except OSError as error:
                emit(event='trace_cleanup_error', error=str(error)[:300])
    return 0


def monitor(output, interval, samples, trace):
    output.parent.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    received = []
    lock = threading.Lock()
    helper = None
    relay = None
    with output.open('x', encoding='utf-8') as log:
        def emit(**row):
            row.update(utc=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic())
            line = json.dumps(row, sort_keys=True)
            with lock:
                log.write(line+'\n')
                log.flush()
                print('RUNNER_DIAGNOSTIC '+line, flush=True)
        def stop(signum, frame):
            received.append(signum)
            stopped.set()
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, stop)
        emit(event='monitor_start', pid=os.getpid(), interval_seconds=interval)
        try:
            if trace:
                try:
                    helper = subprocess.Popen(['sudo', '-n', sys.executable, str(Path(__file__).resolve()),
                                               '--kernel-trace'], stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                    def pump():
                        for line in helper.stdout:
                            try:
                                row = json.loads(line)
                            except ValueError:
                                row = dict(event='trace_helper_message', message=line.strip()[:500])
                            emit(**row)
                    relay = threading.Thread(target=pump, daemon=True)
                    relay.start()
                except OSError as error:
                    emit(event='trace_unavailable', error=str(error))
            count = 0
            while not stopped.is_set():
                try:
                    emit(event='memory', **sample())
                except (OSError, ValueError) as error:
                    emit(event='sample_error', error=str(error))
                count += 1
                if samples and count >= samples:
                    break
                stopped.wait(interval)
        finally:
            emit(event='monitor_stop', received_signals=received)
            if helper is not None:
                helper.stdin.close()
                try:
                    code = helper.wait(timeout=5)
                    if relay is not None:
                        relay.join(timeout=2)
                    emit(event='trace_helper_exit', code=code)
                except subprocess.TimeoutExpired:
                    emit(event='trace_helper_stop_timeout')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--samples', type=int, default=0)
    parser.add_argument('--kernel-trace', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-kernel-trace', action='store_true')
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('Linux /proc and tracefs are required; this is a CI diagnostic, not a solver')
    if args.kernel_trace:
        return kernel_trace()
    if args.output is None or not math.isfinite(args.interval) or args.interval <= 0 or args.samples < 0:
        parser.error('--output, a positive interval and a nonnegative sample count are required')
    return monitor(args.output, args.interval, args.samples, not args.no_kernel_trace)


if __name__ == '__main__':
    raise SystemExit(main())
