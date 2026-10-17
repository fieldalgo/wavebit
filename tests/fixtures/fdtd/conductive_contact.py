"""Independent continuum reference and bounded touching-film wave fixture."""
import math
import numpy as np
import wavebit.fdtd._legacy as fd
from studies.fdtd.layer_alignment.fdtd_layer_alignment import multilayer

WAVELENGTHS = (.9, 1., 1.1)
FILMS = ((4., 8., .125), (2.25, 4., .125))


def reference(pol='p', films=FILMS):
    omega = 2*np.pi/np.asarray(WAVELENGTHS)
    return multilayer(WAVELENGTHS, [a+1j*u/omega for a, u, h in films],
                      [h for a, u, h in films], theta=math.radians(35.))['tm' if pol == 'p' else 'te']


def build(n=40, pol='p', fraction=.375, periods=60, pml=24, courant=.2):
    dx = 1./n
    depth = sum(h for a, u, h in FILMS)
    kx = 2*math.pi*math.sin(math.radians(35.))
    low = -(math.ceil((depth+.8)/dx)+pml+fraction)*dx
    nz = math.ceil((1.4+pml*dx-low)/dx)
    structures = []
    top = 0.
    for a, u, h in FILMS:
        structures.append(fd.Structure(fd.Box(size=(5*dx, 5*dx, h), center=(0., 0., top-h/2)),
                                       fd.Conductor(eps=a, sigma=u)))
        top -= h
    return fd.Scene(domain=fd.Domain(size=(dx, dx, nz*dx), center=(0., 0., low+nz*dx/2)),
        resolution=fd.Resolution(dx=dx), boundaries=fd.Boundaries(x=fd.Axis1D(k=kx),
            y=fd.Periodic(), z=fd.PML(cells=pml, wavelength=1., reflection=1e-12)),
        structures=tuple(structures), sources=(fd.PlaneWave('pw', z=.6, direction='-z',
            theta=35., phi=0., polarization=pol, band=fd.Band(wavelength=1., df=.1, cutoff=1e-6)),),
        monitors=(fd.Diffraction('r', z=1.1, side='reflect', wavelengths=WAVELENGTHS,
            reference_plane=0., orders=(0, 0)), fd.Diffraction('t', z=-depth-.5,
            side='transmit', wavelengths=WAVELENGTHS, reference_plane=-depth,
            incident_plane=0., orders=(0, 0))), run=fd.Run(periods=periods, courant=courant))


def measure(**kwargs):
    scene = build(**kwargs)
    result = fd.run(scene, backend='numpy')
    pol = kwargs.get('pol', 'p')
    r = np.asarray(getattr(result['r'].orders.order(0, 0), 'r_'+pol))
    t = np.asarray(getattr(result['t'].orders.order(0, 0), 't_'+pol))
    R, T = [np.asarray(result[k].orders.order(0, 0).__getattribute__(k.upper())) for k in ('r', 't')]
    energy = np.asarray(result.diagnostics['interior_energy'][1])
    return dict(r=r, t=t, R=R, T=T, A=1-R-T, reference=reference(pol),
                decay=result.diagnostics['decay'], stable=result.diagnostics['stable'],
                energy_finite=bool(np.isfinite(energy).all()), energy_max=float(energy.max()),
                dt=result.plan.dt, steps=result.plan.steps, shape=result.plan.shape,
                bytes_per_state=result.report.bytes_per_state)


def score(rows):
    """Registered interval gates; rows are coarse, middle, long, boundary, fine, half-dt."""
    value = lambda row: np.asarray([row['r'], row['t']], complex)
    coarse, middle, long, boundary, fine, halfdt = [value(row) for row in rows]
    proxy = abs(long-middle)+abs(boundary-long)
    changes = [abs(coarse-middle), abs(middle-fine)]
    errors = [abs(v-rows[0]['reference']) for v in (coarse, middle, fine)]
    minimum = np.minimum.reduce(changes+errors)
    resolved = (minimum > 1e-10) & (2*proxy <= .1*minimum)
    def interval(a, b, uncertainty):
        if np.any(a <= uncertainty) or np.any(b <= uncertainty):
            return np.full(a.shape+(2,), np.nan)
        return np.stack((np.log2((a-uncertainty)/(b+uncertainty)),
                         np.log2((a+uncertainty)/(b-uncertainty))), axis=-1)
    self_order = interval(*changes, 2*proxy)
    direct = [interval(a, b, proxy) for a, b in zip(errors, errors[1:])]
    time_error = abs(halfdt-middle)
    temporal_ok = time_error <= .1*np.minimum(*changes)
    intervals = [self_order]+direct
    order_ok = all(np.all((v[..., 0] >= 1.8) & (v[..., 1] <= 2.3)) for v in intervals)
    stable = all(row['stable'] and row['energy_finite'] and row['energy_max'] > 0
                 and max(row['decay'].values()) < 1e-5 for row in rows)
    return dict(passed=bool(resolved.all() and temporal_ok.all() and order_ok and stable),
                resolved=resolved, temporal_ok=temporal_ok, stable=stable,
                noise_proxy=proxy, self_order=self_order, direct_order=direct,
                time_error=time_error, changes=changes, errors=errors)
