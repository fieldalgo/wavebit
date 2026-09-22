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
sys.path.insert(0, str(ROOT / 'tests'))
import numpy as np
import wavebit.fd as fd
from references.mie import sphere_efficiencies, sphere_amplitudes
from fd_helpers import sphere_integral, to_numpy



def sphere_scene(cpw=20, eps=2.25, periods=24., offset=(0., 0., 0.),
                 gap=.2, pml=12, reflection=1e-8, radius=.25):
    dx = 1/cpw
    center = tuple(v*dx for v in offset)
    # Nested half-widths: sphere .25, absorption .4, source .55,
    # scattering .7, far field .75, then gap and the absorbing layer.
    size = 2*(.75+gap+pml*dx)
    monitors = (fd.Flux('abs', box=fd.Box(size=(.8,)*3), wavelengths=(1.,)),
                fd.Flux('sca', box=fd.Box(size=(1.4,)*3), wavelengths=(1.,)),
                fd.FarField('ff', box=fd.Box(size=(1.5,)*3), wavelengths=(1.,)))
    return fd.Scene(domain=fd.Domain(size=(size,)*3),
                    boundaries=fd.Boundaries.open(fd.PML(cells=pml, reflection=reflection)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                    structures=(fd.Structure(fd.Sphere(radius=radius, center=center), eps),),
                    sources=(fd.TFSF('t', box=fd.Box(size=(1.1,)*3), direction='+z',
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
    optical = 4*math.pi/(2*math.pi)*forward.imag/area
    qmie = sphere_efficiencies(math.sqrt(eps),2*math.pi*radius)
    # On the 20-cell glass fixture, doubling both angular orders to 48/64
    # changes the integral by 7.78e-15 relative (recorded in the study log).
    integral = float(sphere_integral(res['ff'],24,32)[0]/incident/area)
    return dict(qsca=qsca,qabs=qabs,qmie=qmie.tolist(), error=qsca/qmie[1]-1,
                ffq=ffq, farfield_integral=integral, optical=float(optical),
                optical_closure=float(optical/(qsca+qabs)-1),
                decay={k:float(v) for k,v in res.diagnostics['decay'].items()},
                pattern=pattern, shape=res.plan.grid.n)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cpw',type=int,default=20)
    p.add_argument('--eps',type=float,default=2.25)
    p.add_argument('--periods',type=float,default=24.)
    p.add_argument('--backend',default='jax')
    p.add_argument('--gap',type=float,default=.2)
    p.add_argument('--pml',type=int,default=12)
    p.add_argument('--offset',default='0,0,0')
    p.add_argument('output',type=Path)
    args=p.parse_args()
    start=time.time()
    s=sphere_scene(args.cpw,args.eps,args.periods,
                   tuple(map(float,args.offset.split(','))),args.gap,args.pml)
    with warnings.catch_warnings(record=True) as notes:
        res=fd.run(s,backend=args.backend)
    row=read_result(res,args.eps)
    metadata={k:v for k,v in vars(args).items() if k!='output'}
    row.update(metadata,seconds=time.time()-start,warnings=[str(n.message) for n in notes])
    args.output.write_text(json.dumps(row,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in row.items() if k!='pattern'},indent=2),flush=True)
