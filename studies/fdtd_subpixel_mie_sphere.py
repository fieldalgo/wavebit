"""Public Sphere scattering and independent Mie/power/optical closures.

This is an accuracy study, not evidence that C3 already passes. Defaults are
the design fixture; numerical control results are in docs/fd-mie-validation.md.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
import numpy as np
import wavebit.fd as fd
from references.mie import sphere_efficiencies, sphere_amplitudes
from fd_helpers import sphere_integral, to_numpy



def sphere_scene(cpw=20, eps=2.25, periods=24., offset=(0., 0., 0.),
                 gap=.2, pml=12, reflection=1e-8, radius=.25, propagation='continuum'):
    dx = 1/cpw
    center = tuple(v*dx for v in offset)
    # Nested half-widths: sphere .25, absorption .4, source .55,
    # scattering .7, far field .75, then gap and the absorbing layer.
    size = 2*(.75+gap+pml*dx)
    monitors = (fd.Flux('abs', box=fd.Region(size=(.8,)*3), wavelengths=(1.,)),
                fd.Flux('sca', box=fd.Region(size=(1.4,)*3), wavelengths=(1.,)),
                fd.FarField('ff', box=fd.Region(size=(1.5,)*3), wavelengths=(1.,), propagation=propagation))
    return fd.Scene(domain=fd.Domain(size=(size,)*3),
                    boundaries=fd.Boundaries.open(fd.PML(cells=pml, reflection=reflection)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                    structures=(fd.Structure(fd.Sphere(radius=radius, center=center), eps),),
                    sources=(fd.TFSF('t', box=fd.Region(size=(1.1,)*3), direction='+z',
                                     polarization='x', band=fd.Band(wavelength=1., df=.1)),),
                    monitors=monitors)


def read_result(res, eps, radius=.25):
    incident = to_numpy(res.sources['t'].intensity((1.,)))[0]
    amplitude = to_numpy(res.sources['t'].incident_amplitude((1.,), plane=0.))[0]
    area = math.pi*radius**2
    qsca = float(to_numpy(res['sca'].flux)[0]/incident/area)
    qabs = float(-to_numpy(res['abs'].flux)[0]/incident/area)
    ffq = float(to_numpy(res['ff'].flux)[0]/incident/area)
    theta = np.linspace(0.,180.,37)
    expected = sphere_amplitudes(math.sqrt(eps), 2*math.pi*radius, theta)
    pattern = {}
    for phi in (0.,90.):
        measured = to_numpy(res['ff'].intensity(theta,phi))[0]/incident
        reference = abs(expected[1 if phi==0 else 0])**2/(2*math.pi)**2
        pattern[str(phi)] = dict(values=measured.tolist(), reference=reference.tolist(),
                                 peak_error=float(max(abs(measured-reference))/max(reference)))
    forward = to_numpy(res['ff'].fields(0.,0.)[0])[0]/amplitude
    # With exp(-i omega t), the outgoing Mie amplitude is i*S2/k.
    # Keep both quadratures: a small phase error can dominate Im(f) when
    # scattering is weak, although the total complex-field error is small.
    reference_forward = 1j*expected[1][0]/(2*math.pi)
    optical_denominator = 2*math.pi
    if res['ff'].propagation == 'yee':
        # Derived normal-incidence vacuum lattice optical theorem. Omega is
        # the temporal difference symbol, and cos(k*h/2) the incident flux
        # weight. This is NOT a substitution of k in the continuum theorem.
        Omega = 2*math.sin(math.pi*res.plan.dt)/res.plan.dt
        optical_denominator = Omega*math.sqrt(1-(Omega*res.plan.dx/2)**2)
    optical = 4*math.pi/optical_denominator*forward.imag/area
    qmie = sphere_efficiencies(math.sqrt(eps),2*math.pi*radius)
    # On the 20-cell glass fixture, doubling both angular orders to 48/64
    # changes the integral by 7.78e-15 relative (recorded in the study log).
    integral = float(sphere_integral(res['ff'],24,32)[0]/incident/area)
    return dict(qsca=qsca,qabs=qabs,qmie=qmie.tolist(), error=qsca/qmie[1]-1,
                ffq=ffq, farfield_integral=integral, optical=float(optical),
                optical_closure=float(optical/(qsca+qabs)-1),
                forward=dict(value=[float(forward.real), float(forward.imag)],
                             reference=[float(reference_forward.real), float(reference_forward.imag)],
                             relative_error=float(abs(forward/reference_forward-1))),
                # The staggered grid's incident flux includes its H-plane
                # average. Recording the ratio separates that real factor
                # from errors in the complex forward-scattering phase.
                incident_flux_ratio=float(incident/(abs(amplitude)**2/2)),
                decay={k:float(v) for k,v in res.diagnostics['decay'].items()},
                pattern=pattern, shape=res.plan.grid.n, propagation=res['ff'].propagation)


def forward_controls(res, eps):
    """Locate a forward-phase defect without claiming a corrected transform.

    The additional half-widths .65/.85 leave exactly two cells between the
    source/layer and the diagnostic surfaces on the coarsest 20-cell fixture.
    Changing only k in a continuum transform is an instrument perturbation,
    not a derived discrete Green function, and must not replace reported power.
    """
    from wavebit.fd._post import far_field
    amplitude = to_numpy(res.sources['t'].incident_amplitude((1.,), plane=0.))[0]
    exact = 1j*sphere_amplitudes(math.sqrt(eps), math.pi/2, 0.)[1]/(2*math.pi)
    dx, dt = res.plan.dx, res.plan.dt
    # Exact normal-incidence vacuum Yee dispersion, not a fitted phase shift.
    kappa = 2/dx*np.arcsin(dx/dt*np.sin(math.pi*dt))
    rows = []
    for name, half in (('near', .65), ('ff', .75), ('far', .85)):
        value = to_numpy(res[name].fields(0., 0.)[0])[0]/amplitude
        rt, accs, faces, grid, _, eps_bg, mu_bg = res[name]._args
        altered = to_numpy(far_field(rt, accs, faces, grid, (kappa/(2*math.pi),),
                                    eps_bg, mu_bg, 0., 0.)[0])[0]/amplitude
        rows.append(dict(monitor=name, half=half, propagation=res[name].propagation,
            original=[float(value.real), float(value.imag)],
            adjusted_k=[float(altered.real), float(altered.imag)],
            original_complex_error=float(abs(value/exact-1)),
            original_phase_degrees=float(np.degrees(np.angle(value/exact))),
            adjusted_complex_error=float(abs(altered/exact-1)),
            adjusted_phase_degrees=float(np.degrees(np.angle(altered/exact))),
            propagation_phase_degrees=float(np.degrees((kappa-2*math.pi)*half))))
    return dict(kappa=float(kappa), exact=[float(exact.real), float(exact.imag)], rows=rows,
                note='The adjusted-k transform is a diagnostic perturbation, not a correction.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cpw',type=int,default=20)
    p.add_argument('--eps',type=float,default=2.25)
    p.add_argument('--periods',type=float,default=24.)
    p.add_argument('--backend',default='jax')
    p.add_argument('--propagation', choices=('continuum','yee'), default='continuum')
    p.add_argument('--gap',type=float,default=.2)
    p.add_argument('--pml',type=int,default=12)
    p.add_argument('--offset',default='0,0,0')
    p.add_argument('--energy',action='store_true',
                   help='retain the interior-energy history for time-tail studies')
    p.add_argument('--phase-control', action='store_true',
                   help='compare three radiation surfaces and a diagnostic k perturbation')
    p.add_argument('output',type=Path)
    args=p.parse_args()
    start=time.time()
    s=sphere_scene(args.cpw,args.eps,args.periods,
                   tuple(map(float,args.offset.split(','))),args.gap,args.pml, propagation=args.propagation)
    if args.phase_control:
        extra = tuple(fd.FarField(name, box=fd.Region(size=(2*half,)*3), wavelengths=(1.,),
                                 propagation=args.propagation)
                      for name, half in (('near', .65), ('far', .85)))
        s = s.replace(monitors=s.monitors+extra)
    with warnings.catch_warnings(record=True) as notes:
        res=fd.run(s,backend=args.backend)
    row=read_result(res,args.eps)
    if args.energy:
        times, energy = res.diagnostics['interior_energy']
        row['interior_energy'] = dict(time=to_numpy(times).tolist(), value=to_numpy(energy).tolist())
    if args.phase_control:
        row['forward_controls'] = forward_controls(res, args.eps)
    metadata={k:v for k,v in vars(args).items() if k!='output'}
    row.update(metadata,seconds=time.time()-start,warnings=[str(n.message) for n in notes])
    args.output.write_text(json.dumps(row,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in row.items() if k not in ('pattern', 'interior_energy')},indent=2),flush=True)
