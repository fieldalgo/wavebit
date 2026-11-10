"""Wall-clock and peak resident memory of one solve on each arm (NumPy).

    python -m studies.rcwa.isotropic_integrated.cost      # writes cost.json beside this file

Each (case, arm) runs in a fresh process: one untimed warm-up solve, then REPS timed solves,
each under a freshly entered arm (which clears the builders' caches). Peak memory is the process's
resident high-water mark (``ru_maxrss``), interpreter and NumPy included; both arms carry the same
imports. ``tracemalloc`` is not used: it slows the Python-heavy boundary route by an order of
magnitude. The arms run interleaved (scalar, integrated, scalar, integrated).
"""
import json
import pathlib
import resource
import subprocess
import sys
import time

PLAN = [('rectangle', 3, None, 5), ('circle', 3, None, 5), ('polygon', 3, None, 5),
        ('circle', 6, None, 3), ('overlap', 3, 64, 3)]


def worker(case, m, grid, arm_name, reps):
    import numpy as np
    from studies.rcwa.isotropic_integrated import cases as C
    grid = None if grid == 'None' else int(grid)
    m, reps = int(m), int(reps)
    with C.arm(arm_name):
        C.solve(case, m, grid)
    times = []
    for _ in range(reps):
        with C.arm(arm_name):
            t = time.perf_counter()
            C.solve(case, m, grid)
            times.append(time.perf_counter() - t)
    print(json.dumps(dict(case=case, M=m, grid=grid or 'default', arm=arm_name,
                          median_s=float(np.median(times)), min_s=float(np.min(times)),
                          peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)))


if __name__ == '__main__':
    if sys.argv[1:2] == ['worker']:
        worker(*sys.argv[2:])
        sys.exit()
    root = pathlib.Path(__file__).resolve().parents[3]
    rows = []
    for case, m, grid, reps in PLAN:
        for name in ('scalar', 'integrated', 'scalar', 'integrated'):
            cmd = [sys.executable, '-m', 'studies.rcwa.isotropic_integrated.cost', 'worker',
                   case, str(m), str(grid), name, str(reps)]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=root)
                row = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else \
                    dict(case=case, M=m, grid=grid or 'default', arm=name, error=p.stderr[-400:])
            except subprocess.TimeoutExpired:
                row = dict(case=case, M=m, grid=grid or 'default', arm=name, error='timeout 1200 s')
            rows.append(row)
            print(row, flush=True)
    pathlib.Path(__file__).with_name('cost.json').write_text(json.dumps(rows, indent=1))
