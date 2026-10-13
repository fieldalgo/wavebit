"""Exponents for cases A-G: SEM p-convergence, P1 h-convergence + Richardson."""
import numpy as np
from angular import CASES, sectors_from_deg, sem, p1, singular

np.set_printoptions(precision=12, linewidth=160)

print("=== SEM, one element per sector, degree p (exponents nu = sqrt(lam), first 6) ===")
best = {}
for name, spec in CASES.items():
    ang, eps = sectors_from_deg(spec)
    print(f"\ncase {name}: {spec}")
    prev = None
    for p in [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30]:
        nu = sem(ang, eps, p, nsub=1, nev=6)
        s = singular(nu)
        d = '' if prev is None else f"  |d nu1|={abs(s[0]-prev):.2e}"
        print(f"  p={p:2d}  nu[1:6]={np.array2string(nu[1:6], precision=10)}{d}")
        prev = s[0]
    # cross-check: 3 elements per sector, p = 20
    nu3 = sem(ang, eps, 20, nsub=3, nev=8)
    print(f"  nsub=3,p=20 nu[1:6]={np.array2string(nu3[1:6], precision=12)}")
    best[name] = nu3
    print(f"  singular nu in (0,1): {[f'{v:.12f}' for v in singular(nu3)]}")

print("\n=== P1 FEM, uniform mesh, nodes on sector edges: h-convergence of nu_1 ===")
for name, spec in CASES.items():
    ang, eps = sectors_from_deg(spec)
    ref = singular(best[name])[0]
    vals = []
    print(f"\ncase {name}  (SEM reference {ref:.12f})")
    for n in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        nu, h = p1(ang, eps, n)
        v = singular(nu)[0]
        vals.append(v)
        print(f"  N={n:5d}  nu1={v:.12f}  err={v-ref:+.3e}", end='')
        if len(vals) >= 3:
            a, b, c = vals[-3:]
            order = np.log2((a - b) / (b - c))
            rich = c + (c - b) / 3.0
            print(f"  obs.order={order:.3f}  richardson={rich:.12f} (err {rich-ref:+.1e})")
        else:
            print()
