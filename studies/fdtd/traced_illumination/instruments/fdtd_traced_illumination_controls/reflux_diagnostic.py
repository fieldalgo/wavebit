"""Is the oblique energy residual in R0's order algebra or in the solve? Flux planes both sides."""
import dataclasses, sys, warnings
import numpy as np
sys.path.insert(0, sys.argv[1])
import wavebit.fd as fd
import fdtd_traced_illumination_probe as P
warnings.filterwarnings('ignore')
gap = .5
for theta, eps in ((25., 2.25), (25., 1.0), (0., 2.25)):
    pw = fd.PlaneWave('pw', z=.45 + gap, direction='-z', theta=theta, phi=30., polarization='p',
                      band=fd.Band(wavelength=1., df=.05))
    base = P.pw_scene()
    s = dataclasses.replace(base, sources=(pw,), run=fd.Run(steps=6000, courant=.5),
        structures=(base.structures[0].replace(material=fd.Material(eps=eps)),),
        domain=fd.Domain(size=(.6, .6, 2.1 + 2 * gap), center=(0., 0., .05)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=P._layer()),
        monitors=(fd.Flux('t', plane=fd.Plane(z=-.45 - gap), wavelengths=P.WAVELENGTHS),
                  fd.Flux('r', plane=fd.Plane(z=.55 + gap), wavelengths=P.WAVELENGTHS),
                  fd.Flux('m', plane=fd.Plane(z=.3), wavelengths=P.WAVELENGTHS),
                  fd.Diffraction('T', z=-.45 - gap, side='transmit', wavelengths=P.WAVELENGTHS, orders=0),
                  fd.Diffraction('R', z=.55 + gap, side='reflect', wavelengths=P.WAVELENGTHS, orders=0)))
    res = fd.run(s, 'numpy')
    I = np.asarray(res.sources['pw'].intensity(P.WAVELENGTHS)) * .36
    ft, fr, fm = (np.asarray(res[k].flux) / I for k in 'trm')
    T0, R0 = np.asarray(res['T'].orders.given.T[:, 0, 0]), np.asarray(res['R'].orders.given.R[:, 0, 0])
    print(f'theta {theta} eps {eps}: -flux_t {-ft} flux_r {fr} -flux_mid {-fm}')
    print(f'   T0 {T0} R0 {R0} T0+R0-1 {T0 + R0 - 1} (-ft)+fr-1 {-ft + fr - 1}', flush=True)
