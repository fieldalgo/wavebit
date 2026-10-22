"""Independent 120-digit ordinary Fresnel slab reference; no Wavebit imports.

Run manually with mpmath 1.3.0 (BSD-3-Clause, LICENSE in its distribution).
mpmath is a reference-generation tool only, not a runtime or CI dependency.
Inputs retain their binary64 values, including the two neighbors of epsilon=9.
"""
import json
import math
from pathlib import Path

import mpmath as mp

mp.mp.dps = 120
ANGLE = .9+.2j
Z = (0., 5000., 10000., 20000.)


def oracle(e, pol):
    eps = (mp.mpf(9), e, mp.mpf(1))
    kx = 3*mp.sin(mp.mpc(ANGLE.real, ANGLE.imag))
    def root(v):
        q = mp.sqrt(v-kx*kx)
        return q if q.real+q.imag > 0 else -q
    a, b, c = [root(v) for v in eps]
    adm = (a, b, c) if pol == 's' else (a/9, b/e, c)
    r01, r12 = [(u-v)/(u+v) for u,v in zip(adm, adm[1:])]
    t01, t12 = [2*u/(u+v) for u,v in zip(adm, adm[1:])]
    k0 = 2*mp.pi/1000
    phase = mp.exp(1j*k0*b*20000)
    den = 1+r01*r12*phase**2
    r = (r01+r12*phase**2)/den
    t = t01*t12*phase/den * (1 if pol == 's' else 3)
    fields = []
    for z in Z:
        f = t01/den*mp.exp(1j*k0*b*z)
        back = t01/den*r12*mp.exp(1j*k0*b*(40000-z))
        # E and Z0 H, the latter avoiding an unrelated SI constant in the oracle.
        fields.extend((f+back, -b*(f-back), kx*(f+back)) if pol == 's'
                      else (3*b/e*(f-back), -3*kx/e*(f+back), 3*(f+back)))
    return [r, t, *fields]


def main():
    rows = []
    for pol in ('s', 'p'):
        for eps in (9., math.nextafter(9., math.inf), math.nextafter(9., -math.inf), 8.5, 9.5):
            e = mp.mpf(eps)
            values = oracle(e, pol)
            derivatives = [mp.diff(lambda v: oracle(v, pol)[i], e) for i in range(len(values))]
            encode = lambda seq: [[float(mp.re(v)), float(mp.im(v))] for v in seq]
            rows.append(dict(polarization=pol, epsilon=eps, values=encode(values),
                             derivatives=encode(derivatives)))
    Path(__file__).with_name('values.json').write_text(json.dumps(
        dict(reference='ordinary Fresnel, mpmath 1.3.0, 120 decimal digits',
             angle=[ANGLE.real,ANGLE.imag], z=list(Z), rows=rows), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
