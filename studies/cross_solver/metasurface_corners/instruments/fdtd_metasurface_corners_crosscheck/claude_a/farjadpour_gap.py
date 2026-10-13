"""What would it take to get p = 0.702 instead of 0.7219?  All values from the
numerical angular solver (SEM), roots found with brentq on that solver."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from angular import sectors_from_deg, sem, singular

def nu1(spec):
    a, e = sectors_from_deg(spec)
    s = singular(sem(a, e, 20, 3, 8))
    return s[0] if s else 1.0

def nu_ratio(r, adeg=90.0):
    return nu1([(adeg, 1.0), (360 - adeg, r)])

print("90-deg corner, contrast r = eps_high/eps_low (both A and B):")
for r in [np.sqrt(12), 4, 5.8, 10, 11.4, 12, 13, 14, 16, 20, 25, 144, 1e3, 1e6]:
    print(f"   r={r:10.4f}  nu1={nu_ratio(r):.6f}  2nu={2*nu_ratio(r):.4f}")
rstar = brentq(lambda r: nu_ratio(r) - 0.702, 12, 100, xtol=1e-10)
print(f"   contrast giving nu1 = 0.702 at 90 deg: r* = {rstar:.4f}")

print("\nContrast 12, corner angle alpha of the AIR (or, equivalently, of the eps-12) sector:")
for adeg in [10, 20, 30, 45, 60, 70, 75, 80, 85, 90, 100, 120, 150, 179]:
    print(f"   alpha={adeg:3d}  nu1={nu_ratio(12, adeg):.6f}")
res = minimize_scalar(lambda a: nu_ratio(12, a), bounds=(5, 175), method='bounded', options={'xatol': 1e-6})
print(f"   min over alpha: nu1={res.fun:.6f} at alpha={res.x:.3f} deg")
if res.fun < 0.702:
    a_lo = brentq(lambda a: nu_ratio(12, a) - 0.702, 5, res.x, xtol=1e-9)
    a_hi = brentq(lambda a: nu_ratio(12, a) - 0.702, res.x, 179.9, xtol=1e-9)
    print(f"   nu1 = 0.702 at alpha = {a_lo:.3f} deg and {a_hi:.3f} deg (and 360 minus these)")

print("\nOther candidate corners in an eps-12 / air structure:")
print(f"   checkerboard junction (air|12|air|12, 4x90):          {nu1([(90,1),(90,12),(90,1),(90,12)]):.6f}")
print(f"   two holes' corners touching across 90 deg of eps 12:  {nu1([(90,1),(90,12),(90,1),(90,12)]):.6f}")
print(f"   T-junction air 90 | eps12 90 | air 180:               {nu1([(90,1),(90,12),(180,1)]):.6f}")
print(f"   T-junction eps12 90 | air 90 | eps12 180:             {nu1([(90,12),(90,1),(180,12)]):.6f}")
print(f"   hole corner at 45 deg wedge in eps 12 (alpha=45):     {nu_ratio(12,45):.6f}")
print(f"   135 deg corner (octagon-like):                        {nu_ratio(12,135):.6f}")

print("\nFit-resolution of a convergence slope: 2*0.7219 = 1.4438 vs 2*0.702 = 1.404")
