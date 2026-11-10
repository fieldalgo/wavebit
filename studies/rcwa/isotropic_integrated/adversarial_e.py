"""Adversarial re-check E: painted overlap grid convergence at another radius (x=.23), M=2."""
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from studies.rcwa.isotropic_integrated import cases as C
x = .23
grids = [int(g) for g in sys.argv[2:]] or [32, 64, 128, 256, 512]
a = sys.argv[1]
p = pathlib.Path(__file__).with_name('adversarial_e.json')
out = json.loads(p.read_text()) if p.exists() else {}
with C.arm(a):
    for g in grids:
        R_, T_ = C.solve('overlap', 2, grid=g, x=x)
        out[f'{a}|{g}'] = dict(R=R_.ravel().tolist(), T=T_.ravel().tolist(), sumRT=float(R_.sum()+T_.sum()))
        print(a, g, float(T_.sum()), out[f'{a}|{g}']['sumRT'], flush=True)
        p.write_text(json.dumps(out))
