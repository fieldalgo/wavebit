"""Bound one CPU diagnostic, retaining progress and a native macOS sample on stalls."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
child = subprocess.Popen(sys.argv[2:], start_new_session=True)
start = last_change = time.monotonic()
last_cpu = None
samples = []
reason = None
while child.poll() is None:
    try:
        status = subprocess.run(
            ["ps", "-p", str(child.pid), "-o", "time=", "-o", "rss="],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        fields = status.stdout.split()
        if len(fields) != 2:
            time.sleep(1)
            continue
        parts = fields[0].split(":")
        cpu = sum(float(part) * 60**index for index, part in enumerate(reversed(parts)))
        rss = int(fields[1]) * 1024
        now = time.monotonic()
        if cpu != last_cpu:
            last_change, last_cpu = now, cpu
        samples.append(
            dict(elapsed_seconds=now - start, cpu_seconds=cpu, rss_bytes=rss)
        )
        if now - last_change > 120:
            reason = "120s_without_process_cpu_progress"
        if now - start > 1200:
            reason = "1200s_time_guard"
        if rss > 6 * 1024**3:
            reason = "6GiB_process_rss_guard"
        if reason:
            if sys.platform == "darwin":
                subprocess.run(
                    [
                        "sample",
                        str(child.pid),
                        "3",
                        "1",
                        "-file",
                        str(output.with_suffix(".sample.txt")),
                    ],
                    timeout=20,
                    check=False,
                )
            os.kill(child.pid, signal.SIGUSR1)
            time.sleep(2)
            os.killpg(child.pid, signal.SIGKILL)
            break
    except ProcessLookupError:
        break
    time.sleep(1)
code = child.wait()
output.write_text(
    json.dumps(
        dict(
            exit_code=code,
            stop_reason=reason,
            elapsed_seconds=time.monotonic() - start,
            samples=samples,
        ),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print("guard:", code, reason, flush=True)
sys.exit(0 if code == 0 else 1)
