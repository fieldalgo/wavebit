"""Validation of the angular solver: limits, symmetries, and (as a check only)
the classical two-sector closed form.  The exponents themselves come from
angular.sem; nothing here feeds back into them."""
import numpy as np
from scipy.optimize import brentq
from angular import CASES, sectors_from_deg, sem, singular

P, NS = 20, 3
def nus(spec, nev=8):
    a, e = sectors_from_deg(spec)
    return sem(a, e, P, NS, nev)

print("1) homogeneous medium -> integers, each double (cos/sin n theta)")
print("   ", np.array2string(nus([(90, 7.0), (90, 7.0), (180, 7.0)], 9), precision=12))

print("\n2) contrast -> 1: nu1 -> 1 continuously (B-type, 90 deg of eps in air)")
for e in [1.001, 1.01, 1.1, 2.0]:
    print(f"    eps={e:6.3f}  nu1={singular(nus([(90, e), (270, 1.0)]))[0]:.10f}")

print("\n3) conductor limit: wedge alpha of eps->inf in air -> nu = pi/(2pi-alpha)")
for adeg in [30, 60, 90, 120, 150]:
    v = nus([(adeg, 1e9), (360 - adeg, 1.0)])
    v1 = singular(v)[0]
    print(f"    alpha={adeg:3d}  nu1={v1:.10f}  pi/(2pi-alpha)={180/(360-adeg):.10f}")
print("   air wedge 90 deg in eps->inf (case A extreme):",
      f"{singular(nus([(90, 1.0), (270, 1e9)]))[0]:.10f}  (2/3 = {2/3:.10f})")

print("\n4) eps <-> 1/eps (A vs B) and global scaling invariance")
a = nus(CASES['A']); b = nus(CASES['B']); c = nus([(90, 3.0), (270, 36.0)])
print(f"    A {a[1]:.14f}  B {b[1]:.14f}  3*A {c[1]:.14f}")

print("\n5) three-media reductions of E (90 eps_p | 90 air | 180 eps_s)")
print(f"    eps_p=1   (flat air/glass interface) nu1={nus([(90,1.0),(90,1.0),(180,2.25)])[1]:.12f}  (expect 1)")
print(f"    eps_s=12  (= case A)                 nu1={nus([(90,12.0),(90,1.0),(180,12.0)])[1]:.12f}")
print(f"    eps_s=1   (= case B)                 nu1={nus([(90,12.0),(90,1.0),(180,1.0)])[1]:.12f}")

print("\n6) two-sector closed form, used ONLY as a check (wedge alpha of eps1 in eps2)")
print("   even: -eps1 tan(nu a/2) = eps2 tan(nu(pi-a/2));  odd: eps1 tan(nu(pi-a/2)) = -eps2 tan(nu a/2)")
def closed_roots(alpha, e1, e2, numax=3.0, n=200000):
    f_even = lambda v: e1*np.sin(v*alpha/2)*np.cos(v*(np.pi-alpha/2)) + e2*np.sin(v*(np.pi-alpha/2))*np.cos(v*alpha/2)
    f_odd = lambda v: e1*np.sin(v*(np.pi-alpha/2))*np.cos(v*alpha/2) + e2*np.sin(v*alpha/2)*np.cos(v*(np.pi-alpha/2))
    roots = []
    x = np.linspace(1e-6, numax, n)
    for f in (f_even, f_odd):
        y = f(x)
        for k in np.nonzero(np.sign(y[:-1]) != np.sign(y[1:]))[0]:
            roots.append(brentq(f, x[k], x[k+1], xtol=1e-15))
    return np.sort(roots)
for name in 'ABCD':
    (a1, e1), (a2, e2) = CASES[name]
    r = closed_roots(np.deg2rad(a1), e1, e2)
    s = nus(CASES[name])
    s = s[(s > 1e-3) & (s < 3)]
    print(f"    {name}: closed {np.array2string(r[:4], precision=12)}")
    print(f"       SEM    {np.array2string(s[:4], precision=12)}   max|diff|={np.max(abs(r[:4]-s[:4])):.1e}")
for adeg, e1, e2 in [(37, 12, 1), (200, 3.3, 1), (75, 1, 12)]:
    r = closed_roots(np.deg2rad(adeg), e1, e2)
    s = nus([(adeg, e1), (360 - adeg, e2)], 10)
    s = s[(s > 1e-3) & (s < 3)]
    m = min(len(r), len(s), 5)
    print(f"    alpha={adeg} eps {e1}/{e2}: max|closed-SEM| over {m} roots = {np.max(abs(r[:m]-s[:m])):.1e}")

print("\n7) observed structure for sectors that are multiples of 90 deg: nu1 + nu2 = 2")
for name in CASES:
    v = nus(CASES[name])
    print(f"    {name}: nu1={v[1]:.12f}  nu2={v[2]:.12f}  sum={v[1]+v[2]:.12f}")
