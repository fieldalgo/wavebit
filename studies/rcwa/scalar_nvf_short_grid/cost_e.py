"""Section (e): wall-clock and peak resident memory of one solve, today's builder against the prototype.

    python cost_e.py            # writes results/e.json

Each (case, arm) runs in a fresh process: one untimed warm-up solve (imports, first-call costs),
then REPS timed solves with the projector cache cleared before each, so every timed solve builds
its projector. Peak memory is the process's resident high-water mark (``ru_maxrss``), which
includes the interpreter and NumPy; both arms carry the same imports, so the difference is what
the arm adds. ``tracemalloc`` is not used: it slows the Python-heavy boundary route by an order
of magnitude. The arms run interleaved (current, prototype, current, prototype) to spread drift.
The overlapping-circles case runs at grid 256: one boundary-route solve at the default grid takes
over ten minutes.
"""
import json
import pathlib
import resource
import subprocess
import sys
import time

OUT = pathlib.Path(__file__).with_name('results')
PLAN = [('rectangle', None, 3, 7), ('circle', None, 3, 7), ('ellipse', None, 3, 7),
        ('circle', None, 6, 5), ('overlap', 256, 3, 3)]


def worker(case, grid, m, arm_name, reps):
    import numpy as np
    from measure import arm_ctx, parametric, solve
    grid = None if grid == 'None' else int(grid)
    m, reps = int(m), int(reps)
    with arm_ctx(arm_name):
        solve(case, grid, (m, m))
        times = []
        for _ in range(reps):
            parametric._PROJECTOR_CACHE.clear()
            t = time.perf_counter()
            solve(case, grid, (m, m))
            times.append(time.perf_counter() - t)
    print(json.dumps(dict(case=case, grid=grid or 'default', M=m, arm=arm_name,
                          median_s=float(np.median(times)), min_s=float(np.min(times)),
                          peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)))


def coeff_bench():
    import numpy as np
    from measure import arm_ctx, parametric
    field = np.exp(1j * np.random.default_rng(0).uniform(size=(1024, 1024)))
    rows = []
    for name in ('current', 'prototype', 'current', 'prototype'):
        with arm_ctx(name):
            ts = []
            for _ in range(20):
                t = time.perf_counter()
                parametric._sampled_coeffs(np, field, 3, 3)
                ts.append(time.perf_counter() - t)
        rows.append(dict(case='_sampled_coeffs 1024x1024, M=3', arm=name, median_s=float(np.median(ts))))
        print(rows[-1], flush=True)
    return rows


if __name__ == '__main__':
    if sys.argv[1:2] == ['worker']:
        worker(*sys.argv[2:])
        sys.exit()
    rows = []
    for case, grid, m, reps in PLAN:
        for name in ('current', 'prototype', 'current', 'prototype'):
            cmd = [sys.executable, __file__, 'worker', case, str(grid), str(m), name, str(reps)]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200,
                                   cwd=pathlib.Path(__file__).parent)
                row = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else \
                    dict(case=case, grid=grid or 'default', M=m, arm=name, error=p.stderr[-400:])
            except subprocess.TimeoutExpired:
                row = dict(case=case, grid=grid or 'default', M=m, arm=name, error='timeout 1200 s')
            rows.append(row)
            print(row, flush=True)
    rows += coeff_bench()
    (OUT / 'e.json').write_text(json.dumps(rows, indent=1))
