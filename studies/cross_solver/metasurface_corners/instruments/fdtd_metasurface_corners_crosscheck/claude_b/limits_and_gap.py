"""Limits, reductions, random cross-checks, and the 0.702 hunt."""
import math
import numpy as np
from scipy.optimize import brentq
from corner2d import (parity_roots, nodal_det, direct_det, scan_roots, fe_extrapolated,
                      nu_two_media_90, nu_three_media, D2R, PI)


def smallest(widths_deg, eps, route='nodal'):
    w = [d * D2R for d in widths_deg]
    f = (lambda t: nodal_det(t, w, eps)) if route == 'nodal' else (lambda t: direct_det(t, w, eps))
    r = scan_roots(f, 1e-4, 1 - 1e-9, 20001)
    return min(r) if r else float('nan'), r


print("=== 1. infinite-contrast limit (Dirichlet wedge nu = pi/exterior angle) ===")
for alpha in [30, 45, 60, 90, 120, 150]:
    for big in [1e3, 1e6, 1e9]:
        ev, od = parity_roots(alpha * D2R, big, 1.0)     # conductor-like sector of width alpha
        cand = sorted(ev + od)
        nu_d = 180.0 / (360.0 - alpha)
        ndl, _ = smallest([alpha, 360 - alpha], [big, 1.0])
        print(f"   alpha={alpha:3d}  eps_in={big:.0e}  parity {cand[0]:.9f}  nodal {ndl:.9f}"
              f"  Dirichlet pi/(2pi-alpha)={nu_d:.9f}  diff {cand[0]-nu_d:+.2e}")
    # the same sector as the LOW-eps one (Neumann on the big sector): odd Neumann mode of 2pi-alpha
    ev, od = parity_roots(alpha * D2R, 1.0, 1e9)
    print(f"   alpha={alpha:3d}  reversed (air sector inside eps=1e9): smallest {min(ev+od):.9f}"
          f"  (Neumann sector of 2pi-alpha: pi/(2pi-alpha) = {180/(360-alpha):.9f})")

print("\n=== 2. no-contrast and small-contrast limits ===")
for name, wd, eps in [('two 90/270', [90, 270], [12.0, 1.0]),
                      ('two 60/300', [60, 300], [12.0, 1.0]),
                      ('three E', [90, 90, 180], [12.0, 1.0, 2.25]),
                      ('three F', [90, 90, 180], [5.8, 1.0, 2.13]),
                      ('three asym 70/110/180', [70, 110, 180], [12.0, 1.0, 2.25])]:
    w = np.array(wd) * D2R
    # first-order perturbation around the degenerate nu = 1 pair:
    # nu ~ 1 - |sum_j ln(eps_j) int_sector e^{2i theta}| / (2 pi)
    starts = np.concatenate([[0], np.cumsum(w)[:-1]])
    C = sum(math.log(e) * (np.exp(2j * (s + ww)) - np.exp(2j * s)) / 2j for e, s, ww in zip(eps, starts, w))
    slope_pred = abs(C) / (2 * PI)
    for s in [1e-2, 1e-3, 1e-4]:
        es = [e ** s for e in eps]
        r = scan_roots(lambda t: nodal_det(t, list(w), es), 0.5, 1 - 1e-12, 20001)
        nu = min(r) if r else float('nan')
        print(f"   {name:22s} eps^s, s={s:.0e}: nu={nu:.12f}  (1-nu)/s={(1-nu)/s:.6f}  predicted {slope_pred:.6f}")
    es = [1.0] * len(eps)
    r = scan_roots(lambda t: nodal_det(t, list(w), es), 1e-3, 1 - 1e-9, 20001)
    print(f"   {name:22s} uniform eps: roots in (0,1) = {r}")

print("\n=== 3. reductions of the three-media junction ===")
for ep in [12.0, 5.8, 4.0]:
    print(f"   eps_p={ep}: E(eps_s=1)={nu_three_media(ep,1,1):.12f}  vs B {nu_two_media_90(ep,1):.12f};"
          f"  E(eps_s=eps_p)={nu_three_media(ep,1,ep):.12f}  vs A {nu_two_media_90(1,ep):.12f}")
# conductor pillar: Dirichlet on a 270-degree composite (air 90 + substrate 180):
#   eps_s tan(nu pi/2) + eps_a tan(nu pi) = 0
for es_ in [1.0, 2.13, 2.25, 4.0]:
    f = lambda t: es_ * math.sin(t*PI/2) * math.cos(t*PI) + 1.0 * math.sin(t*PI) * math.cos(t*PI/2)
    nd = brentq(f, 0.51, 0.99)
    print(f"   eps_p=1e9, eps_s={es_}: three-media {nu_three_media(1e9,1,es_):.9f}  nodal "
          f"{smallest([90,90,180],[1e9,1.0,es_])[0]:.9f}  Dirichlet-composite eq {nd:.9f}")
for big in [1e3, 1e6, 1e9]:
    print(f"   eps_s={big:.0e} (conducting substrate): nu={nu_three_media(12,1,big):.9f} -> 1")

print("\n=== 4. random cross-check: parity vs nodal vs direct vs FE, two media, any angle ===")
rng = np.random.default_rng(1)
worst = 0
for _ in range(12):
    a = rng.uniform(20, 340); e1 = 10 ** rng.uniform(-1.5, 1.5); e2 = 1.0
    ev, od = parity_roots(a * D2R, e1, e2)
    p = sorted(ev + od)
    n = smallest([a, 360 - a], [e1, e2])[1]
    d = smallest([a, 360 - a], [e1, e2], 'direct')[1]
    sp = math.sin  # closed form check: sin(nu pi) = +-K sin(nu(pi-alpha))
    K = (e1 - e2) / (e1 + e2)
    res = min(abs(math.sin(p[0]*PI) - s*K*math.sin(p[0]*(PI - a*D2R))) for s in (1, -1))
    worst = max(worst, abs(p[0] - min(n)), abs(p[0] - min(d)), res)
    print(f"   alpha={a:7.3f} eps={e1:8.4f}: parity {p[0]:.12f} nodal {min(n):.12f} direct {min(d):.12f}"
          f"  sin-form residual {res:.1e}")
print(f"   worst disagreement {worst:.1e}")
print("   random three-media (nodal vs direct vs FE):")
for _ in range(5):
    w1, w2 = rng.uniform(30, 150, 2); w3 = 360 - w1 - w2
    eps = list(10 ** rng.uniform(-1, 1.2, 3))
    n = smallest([w1, w2, w3], eps)[1]; d = smallest([w1, w2, w3], eps, 'direct')[1]
    _, _, r2 = fe_extrapolated([w1*D2R, w2*D2R, w3*D2R], eps)
    print(f"   w=({w1:.1f},{w2:.1f},{w3:.1f}) eps=({eps[0]:.3f},{eps[1]:.3f},{eps[2]:.3f}):"
          f" nodal {[round(x,10) for x in n]} direct {[round(x,10) for x in d]} FE {r2[-1][0]:.10f}")

print("\n=== 5. the 0.702 hunt (Farjadpour 2006) ===")
target = 0.702
for e in [11.56, 12.0, 12.25, 13.0]:
    print(f"   90-deg two-media corner, eps={e}: nu={nu_two_media_90(e,1):.6f}  2nu={2*nu_two_media_90(e,1):.4f}")
e_need = brentq(lambda e: nu_two_media_90(e, 1) - target, 2, 1e4)
print(f"   eps needed for nu=0.702 at 90 deg: {e_need:.4f}")
K = 11 / 13
nu_min = 1 - math.asin(K) / PI
print(f"   eps=12: minimum over corner angle of nu = 1 - asin(11/13)/pi = {nu_min:.6f}"
      f" at alpha = {180 - 90/nu_min:.3f} deg")
f = lambda a: min(sorted(sum(parity_roots(a*D2R, 12.0, 1.0), []))) - target
grid = np.linspace(5, 175, 341); vals = [f(a) for a in grid]
for i in range(len(grid)-1):
    if vals[i]*vals[i+1] < 0:
        print(f"   eps=12 corner angle giving nu=0.702: {brentq(f, grid[i], grid[i+1]):.4f} deg (dielectric sector)")
# pillar-on-substrate junction with eps 12
print(f"   eps 12 / air / 2.25 junction (case E): {nu_three_media(12,1,2.25):.6f}")
for es_ in [1.5, 2.0, 2.25, 3.0]:
    print(f"   eps 12 / air / {es_} junction: {nu_three_media(12,1,es_):.6f}")
# checkerboard vertex: 4 x 90, alternating
cb = smallest([90, 90, 90, 90], [12.0, 1.0, 12.0, 1.0])
print(f"   checkerboard 12/1/12/1 vertex: roots in (0,1) {cb[1]}")
# field components: all of E ~ r^(nu-1); TM (E_z) has no r^nu singularity in the principal part
# the second (even) exponent of a 90-deg corner and the 'wrong-branch' formulas
print(f"   second exponent of case A/B: {2 - nu_two_media_90(12,1):.6f}")
for label, val in [("(2/pi) asin(K)", 2/PI*math.asin(K)), ("1 - asin(K)/pi", nu_min),
                   ("(1/pi) acos(-K/2)", math.acos(-K/2)/PI), ("(2/pi) acos(K/2) [correct]", 2/PI*math.acos(K/2)),
                   ("(2/pi) acos((n-1)/(2(n+1))), n=sqrt12", 2/PI*math.acos((math.sqrt(12)-1)/(2*(math.sqrt(12)+1)))),
                   ("1 - K/pi (small-contrast, linear in K)", 1 - K/PI),
                   ("1 - ln(12)/(2pi) (first-order in ln eps)", 1 - math.log(12)/(2*PI))]:
    print(f"   {label:42s} = {val:.6f}")
print(f"   slope fit: 2*0.7219 = {2*0.7219:.4f}; Farjadpour fit 1.4 -> p = 0.70")
