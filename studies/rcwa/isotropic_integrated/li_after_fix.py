"""Li and Laurent on shape stacks with tensor_shapes._ISOTROPIC_INTEGRATED on, after the fix.

    python -m studies.rcwa.isotropic_integrated.li_after_fix     # writes li_after_fix.json
"""
import json
import pathlib

from studies.rcwa.isotropic_integrated import cases as C
from wavebit import rcwa

rows = []
for case in ('rectangle', 'circle', 'polygon'):
    for fac in ('li', 'laurent'):
        for name in C.ARMS:
            with C.arm(name):
                try:
                    r = rcwa.run(C.problem(case), numerics=rcwa.Numerics(orders=(2, 2), factorization=fac))
                    rows.append(dict(case=case, factorization=fac, arm=name, sum_T=float(r.T.sum())))
                except Exception as e:  # noqa: BLE001
                    rows.append(dict(case=case, factorization=fac, arm=name, error=repr(e)[:200]))
            print(rows[-1], flush=True)
pathlib.Path(__file__).with_name('li_after_fix.json').write_text(json.dumps(rows, indent=1))
