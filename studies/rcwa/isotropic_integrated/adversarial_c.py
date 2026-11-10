"""Adversarial re-check C: Li under the switch (rectangle), metal lamella vs Li (other width/eps), odd/even grid."""
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from wavebit import rcwa, Material
from studies.rcwa.isotropic_integrated import cases as C

out = {}
def prob(regs, bg):
    return rcwa.Problem(
        stack=rcwa.Stack(layers=(rcwa.Layer(thickness=C.THICKNESS, background=Material(bg), regions=tuple(regs)),),
                         bottom=Material(C.SUBSTRATE), period=(1., 1.)),
        excitation=rcwa.Excitation(wavelength=C.WAVELENGTH,
                                   sources=(rcwa.PlaneWave(theta=C.THETA, phi=C.PHI, polarization='s'),)))
def run(p, o, fac='nvf', grid=None):
    r = rcwa.run(p, numerics=rcwa.Numerics(orders=o, factorization=fac, shape_nvf_grid_size=grid), backend='numpy')
    return np.asarray(r.R), np.asarray(r.T)

# 1) li on an axis-aligned rectangle with the switch on
rect = prob([rcwa.Region(geometry=rcwa.Rectangle(size=(.3, .4)), material=Material(4.))], 1.5)
li = {}
for a in C.ARMS:
    with C.arm(a):
        for fac in ('li', 'laurent', 'nvf'):
            try:
                R_, T_ = run(rect, (2, 2), fac)
                li[f'{a}|{fac}'] = ['ok', float(T_.sum())]
            except Exception as e:
                li[f'{a}|{fac}'] = [type(e).__name__, str(e)[:160]]
            print(a, fac, li[f'{a}|{fac}'], flush=True)
out['li_rect_switch'] = li

# 2) metal lamella, different width/eps/centre, vs Li (computed on scalar arm)
lam = prob([rcwa.Region(geometry=rcwa.Rectangle(size=(.45, 1.), center=(.1, 0.)), material=Material(-15. + 1.j))], 1.)
rows = []
for o in ((2, 2), (3, 3), (10, 0), (20, 0)):
    with C.arm('scalar'):
        lr, lt = run(lam, o, 'li')
    for a in C.ARMS:
        with C.arm(a):
            r, t = run(lam, o)
        d = float(max(np.abs(r - lr).max(), np.abs(t - lt).max()))
        rows.append(dict(orders=o, arm=a, max_d_vs_li=d))
        print('lamella', o, a, d, flush=True)
out['metal_lamella'] = rows

# 3) odd vs even grids, circle at x=.27 and high-contrast at x=.32, M=3
odd = {}
for case, x in (('circle', .27), ('high-contrast', .32)):
    for a in C.ARMS:
        with C.arm(a):
            ref = C.solve(case, 3, grid=1024, x=x)
            for g in (63, 64, 65, 127, 128, 129, 255, 256, 257):
                R_, T_ = C.solve(case, 3, grid=g, x=x)
                e = float(max(np.abs(R_ - ref[0]).max(), np.abs(T_ - ref[1]).max()))
                odd[f'{case}|{a}|{g}'] = e
                print(case, a, g, e, flush=True)
out['odd_even'] = odd
pathlib.Path(__file__).with_name('adversarial_c.json').write_text(json.dumps(out, indent=1))
