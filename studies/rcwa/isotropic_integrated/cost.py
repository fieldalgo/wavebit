"""Wall-clock and peak traced memory of one solve on each arm (NumPy, default grid).

    python -m studies.rcwa.isotropic_integrated.cost      # writes cost.json beside this file
"""
import json
import pathlib
import time
import tracemalloc

import numpy as np

from studies.rcwa.isotropic_integrated import cases as C

rows = []
for case, orders, grid in (('rectangle', 3, None), ('circle', 3, None), ('polygon', 3, None),
                           ('circle', 6, None), ('overlap', 3, 64)):
    for name in ('scalar', 'integrated', 'scalar', 'integrated'):
        times, peaks = [], []
        for _ in range(3):
            with C.arm(name):
                t = time.perf_counter()
                C.solve(case, orders, grid)
                times.append(time.perf_counter() - t)
                tracemalloc.start()
                C.solve(case, orders, grid)
                peaks.append(tracemalloc.get_traced_memory()[1])
                tracemalloc.stop()
        rows.append(dict(case=case, M=orders, grid=grid or 'default', arm=name,
                         median_s=float(np.median(times)), peak_MiB=float(np.median(peaks)) / 2**20))
        print(rows[-1], flush=True)
pathlib.Path(__file__).with_name('cost.json').write_text(json.dumps(rows, indent=1))
