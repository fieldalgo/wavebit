"""Dispersive lossy film: table lookup, both connecting algorithms, finite-film Airy check."""

import json
import numpy as np
from wavebit.compat import legacy as wavebit

silicon = wavebit.materials.catalog.load('Si')
wl, d = 900., 120.
material = silicon.at(wl)
# Index is needed only for the independent closed-form Airy calculation below.
n = wavebit.index_from_eps(material.epsilon)
rows = []
for algorithm in ('ETM', 'SMM'):
    solver = wavebit.rcwa(
        backend='numpy', wavelength=wl, period=(700., 700.),
        structure=np.array([[[material.epsilon]]]), thickness=[d],
        top=1., bot=2.25, tetm='te', truncation_order=(0, 0),
        layer_algorithm=algorithm,
    )
    r = solver.solve()
    rows.append(dict(algorithm=algorithm, R=float(r.R.sum()), T=float(r.T.sum()),
        A=float(1-r.R.sum()-r.T.sum()),
        has_scattering=(solver.scattering is not None) if algorithm == 'SMM' else False))
# A separate closed-form nonmagnetic film, normal incidence, e^(-i*w*t).
n0, ns = 1., 1.5
r01, r12 = (n0-n)/(n0+n), (n-ns)/(n+ns)
t01, t12 = 2*n0/(n0+n), 2*n/(n+ns)
phase = np.exp(1j*2*np.pi*n*d/wl)
denom = 1+r01*r12*phase**2
airy_R = abs((r01+r12*phase**2)/denom)**2
airy_T = ns/n0*abs(t01*t12*phase/denom)**2
print(json.dumps(dict(wavelength_nm=wl, index=[float(n.real), float(n.imag)],
    epsilon=[float(material.epsilon.real), float(material.epsilon.imag)],
    rows=rows, airy=dict(R=float(airy_R), T=float(airy_T))), indent=2))
