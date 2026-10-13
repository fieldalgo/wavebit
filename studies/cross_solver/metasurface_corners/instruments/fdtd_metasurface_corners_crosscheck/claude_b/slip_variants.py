"""Which plausible mis-evaluations of the 90-deg, eps-12 corner equation give 0.702?"""
import math
import numpy as np
from scipy.optimize import brentq

PI = math.pi
e = 12.0
K = (e - 1) / (e + 1)


def root(f, lo=0.05, hi=0.999):
    xs = np.linspace(lo, hi, 4000)
    v = [f(x) for x in xs]
    for i in range(len(xs) - 1):
        if v[i] * v[i + 1] < 0 and abs(v[i]) < 50 and abs(v[i + 1]) < 50:
            return brentq(f, xs[i], xs[i + 1])
    return float('nan')


variants = {
    "correct odd:  e tan(nu pi/4) + tan(3 nu pi/4) = 0": lambda n: e*math.sin(n*PI/4)*math.cos(3*n*PI/4) + math.sin(3*n*PI/4)*math.cos(n*PI/4),
    "sin form, K=(e-1)/(e+1)": lambda n: math.sin(n*PI) - K*math.sin(n*PI/2),
    "sin form, K=(e-1)/e": lambda n: math.sin(n*PI) - (e-1)/e*math.sin(n*PI/2),
    "sin form, K=(e^2-1)/(e^2+1)": lambda n: math.sin(n*PI) - (e*e-1)/(e*e+1)*math.sin(n*PI/2),
    "sin form, K=(n-1)/(n+1), n=sqrt(e)": lambda n: math.sin(n*PI) - (math.sqrt(e)-1)/(math.sqrt(e)+1)*math.sin(n*PI/2),
    "sin form with 2pi-alpha for pi-alpha": lambda n: math.sin(n*PI) - K*math.sin(n*1.5*PI),
    "sin form with alpha for pi-alpha (90 deg)": lambda n: math.sin(n*PI) - K*math.sin(n*PI/2),
    "half-angle used as full angle (alpha=45)": lambda n: math.sin(n*PI) - K*math.sin(n*0.75*PI),
    "full angle used as half (alpha=180)": lambda n: math.sin(n*PI) - K*math.sin(n*0.0),
    "angle 60 deg": lambda n: math.sin(n*PI) - K*math.sin(n*2*PI/3),
    "angle 120 deg": lambda n: math.sin(n*PI) - K*math.sin(n*PI/3),
    "tan sign slip: e tan a = tan b": lambda n: e*math.sin(n*PI/4)*math.cos(3*n*PI/4) - math.sin(3*n*PI/4)*math.cos(n*PI/4),
    "cot form: e cot a + cot b = 0": lambda n: e*math.cos(n*PI/4)*math.sin(3*n*PI/4) + math.cos(3*n*PI/4)*math.sin(n*PI/4),
}
for k, f in variants.items():
    r = root(f)
    print(f"   {k:48s} nu = {r:.6f}")
print(f"   eps that gives 0.702 at 90 deg = "
      f"{brentq(lambda x: 2/PI*math.acos((x-1)/(2*(x+1))) - 0.702, 2, 1e3):.4f}")
