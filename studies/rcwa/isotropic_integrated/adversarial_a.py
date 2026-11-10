"""Adversarial re-check A: routing counter, energy defect at a nearby parameter, Li+switch refusal."""
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import wavebit
from wavebit import rcwa
from wavebit.rcwa._engine import tensor_shapes
from studies.rcwa.isotropic_integrated import cases as C

out = {'wavebit_file': wavebit.__file__}
calls = {'n': 0}
_orig = tensor_shapes._layer
def _counted(*a, **k):
    calls['n'] += 1
    return _orig(*a, **k)
tensor_shapes._layer = _counted

# 1) energy defect, lossless, at a NEARBY parameter (x0+0.02), both arms, s and p
res = {}
for case, x in [('high-contrast', .32), ('ellipse', .33), ('circle', .27), ('rect-tilted', .28)]:
    for armname in C.ARMS:
        for M in (2, 3, 5):
            with C.arm(armname):
                calls['n'] = 0
                Rr, Tt = C.solve(case, M, x=x)
                res[f'{case}|{armname}|M{M}'] = dict(
                    defect=float(Rr.sum() + Tt.sum() - 1), sumT=float(Tt.sum()), layer_calls=calls['n'])
                print(case, armname, M, res[f'{case}|{armname}|M{M}'], flush=True)
out['energy'] = res

# 2) p polarization (cases use s) on high-contrast
pres = {}
for armname in C.ARMS:
    with C.arm(armname):
        p = C.problem('high-contrast', .3)
        p = rcwa.Problem(stack=p.stack, excitation=rcwa.Excitation(
            wavelength=C.WAVELENGTH, sources=(rcwa.PlaneWave(theta=C.THETA, phi=C.PHI, polarization='p'),)))
        r = rcwa.run(p, numerics=C.numerics(3), backend='numpy')
        pres[armname] = float(np.sum(C.to_np(r.R)) + np.sum(C.to_np(r.T)) - 1)
        print('p-pol', armname, pres[armname], flush=True)
out['energy_p_highcontrast_M3'] = pres

# 3) Li factorization under the switch
li = {}
for armname in C.ARMS:
    with C.arm(armname):
        for fac in ('li', 'laurent'):
            try:
                r = rcwa.run(C.problem('circle'), numerics=rcwa.Numerics(orders=(2, 2), factorization=fac),
                             backend='numpy')
                li[f'{armname}|{fac}'] = ['ok', float(np.sum(C.to_np(r.T)))]
            except Exception as e:
                li[f'{armname}|{fac}'] = [type(e).__name__, str(e)[:200]]
            print(armname, fac, li[f'{armname}|{fac}'], flush=True)
out['li_switch'] = li
pathlib.Path(__file__).with_name('adversarial_a.json').write_text(json.dumps(out, indent=1))
