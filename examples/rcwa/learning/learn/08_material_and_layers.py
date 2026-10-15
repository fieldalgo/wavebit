"""A physical lossy film: catalog, both algorithms and an independent Airy check."""
import json
import numpy as np
from wavebit import Material, Units, materials, rcwa

units = Units(length='nm')
silicon = materials.load('Si')
wl, d = 900., 120.
material = silicon.at(wl, units=units)
n = materials.refractive_index_from_eps(material.epsilon)
# Independent finite-film expression, normal incidence, exp(-i omega t).
n0, ns = 1., 1.5
r01, r12 = (n0-n)/(n0+n), (n-ns)/(n+ns)
t01, t12 = 2*n0/(n0+n), 2*n/(n+ns)
phase = np.exp(1j*2*np.pi*n*d/wl)
denom = 1+r01*r12*phase**2
airy_R = abs((r01+r12*phase**2)/denom)**2
airy_T = ns/n0*abs(t01*t12*phase/denom)**2
rows = []
for algorithm in ('etm', 'smm'):
    problem = rcwa.Problem(units=units,
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=d, background=silicon),)),
        excitation=rcwa.Excitation(wavelength=wl, sources=(rcwa.PlaneWave(polarization='s'),)),
        outputs=rcwa.Outputs(scattering=algorithm == 'smm'))
    r = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0), algorithm=algorithm))
    np.testing.assert_allclose([r.R.sum(), r.T.sum()], [airy_R, airy_T], atol=2e-12)
    rows.append(dict(algorithm=algorithm.upper(), R=float(r.R.sum()), T=float(r.T.sum()),
        A=float(r.absorption[0]), has_scattering=bool(r.scattering) if algorithm == 'smm' else False))
print(json.dumps(dict(wavelength_nm=wl, index=[float(n.real), float(n.imag)],
    epsilon=[float(material.epsilon.real), float(material.epsilon.imag)],
    rows=rows, airy=dict(R=float(airy_R), T=float(airy_T))), indent=2))
