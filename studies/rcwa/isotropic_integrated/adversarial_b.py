"""Adversarial re-check B: per-order accuracy, nearby parameters, both arms, M ladder vs M=12/13 refs."""
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from studies.rcwa.isotropic_integrated import cases as C

LADDER = (2, 3, 4, 5, 6, 7, 8, 9, 10)
REFS = (12, 13)
cases = [(c, x) for c, x in (('circle', .27), ('high-contrast', .32), ('ellipse', .33), ('metallic', .28))
         if len(sys.argv) < 2 or c in sys.argv[1:]]
path = pathlib.Path(__file__).with_name('adversarial_b.json')
out = json.loads(path.read_text()) if path.exists() else {}
for case, x in cases:
    d = {}
    for a in C.ARMS:
        with C.arm(a):
            for M in LADDER + REFS:
                Rr, Tt = C.solve(case, M, x=x)
                d[f'{a}|{M}'] = float(Tt.sum())
                print(case, a, M, d[f'{a}|{M}'], flush=True)
    out[f'{case}@{x}'] = d
    path.write_text(json.dumps(out, indent=1))
