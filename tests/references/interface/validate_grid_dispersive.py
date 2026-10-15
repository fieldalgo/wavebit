"""Live CPML/Drude/Lorentz slab validation; Meep public API only.

The finite domain uses each solver's own PML. Compare normalized complex E/H
transfer and power, plus each solver's convergence to the passive Airy solution.
Wavebit point sampling matches Meep's documented unsmoothed dispersive term;
average sampling checks the tangential half-cell alternative against Airy.
The pulse has zero DC by symmetry to avoid long CPML low-frequency tails.
"""

from wavebit.fdtd._periodic.grid import fdtd_grid as _legacy_wavebit_fdtd_grid
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.tracebacklimit=0
WAVELENGTHS=np.array([.6,.7,.8])
MODELS={'dielectric':(2.25,()), 'lorentz':(1.5,((12.,1.5,60.),)),
        'drude':(1.,((0.,1.2,100.),)),
        'mixed':(1.5,((0.,1.2,40.),(12.,1.5,60.))),
        'low-loss-mixed':(1.5,((0.,.08,160.),(10.,.12,40.)))}


def pulse(t, wavelength=.7):
    return np.exp(-((t-5*wavelength)/wavelength)**2)*np.sin(2*np.pi*t/wavelength)


def worker(backend,resolution,model,pml_width=1.,duration=24.,pole_sampling="average"):
    eps_inf,poles=MODELS[model]
    wavelengths=np.array([.9,1.,1.1]) if model=='low-loss-mixed' else WAVELENGTHS
    centre=float((wavelengths.min()+wavelengths.max())/2)
    dt=.4/resolution
    steps=round(duration/dt)
    if backend=='meep':
        import meep as mp
        mp.verbosity(0)
        def solve(empty):
            susceptibilities=[]
            for omega,gamma,strength in (() if empty else poles):
                frequency=omega/(2*np.pi) if omega else 1.
                cls=mp.LorentzianSusceptibility if omega else mp.DrudeSusceptibility
                susceptibilities.append(cls(frequency=frequency,gamma=gamma/(2*np.pi),
                                            sigma=strength/(2*np.pi*frequency)**2))
            medium=mp.Medium(epsilon=eps_inf,E_susceptibilities=susceptibilities)
            geometry=[] if empty else [mp.Block(center=mp.Vector3(z=1.25),
                                               size=mp.Vector3(mp.inf,mp.inf,.5),material=medium)]
            sim=mp.Simulation(cell_size=mp.Vector3(z=12),dimensions=1,resolution=resolution,
                              Courant=.4,eps_averaging=True,geometry=geometry,
                              boundary_layers=[mp.PML(pml_width)],
                              sources=[mp.Source(mp.CustomSource(lambda t:pulse(t,centre),end_time=8*centre/.7),component=mp.Ex,
                                                 center=mp.Vector3(z=-3))])
            monitor=sim.add_dft_fields([mp.Ex,mp.Hy],list(1/wavelengths),center=mp.Vector3(z=3),
                                      size=mp.Vector3(),decimation_factor=1)
            sim.run(until=steps*dt)
            values=np.array([[np.asarray(sim.get_dft_array(monitor,c,i)).item()
                              for c in (mp.Ex,mp.Hy)] for i in range(len(wavelengths))])
            sim.reset_meep()
            return values
        version=mp.__version__
    else:
        import wavebit
        import jax
        jax.config.update('jax_enable_x64',True)
        def solve(empty):
            shape=(12*resolution,2,2)
            fill=np.zeros(shape)
            start,stop=7*resolution,int(7.5*resolution)
            if not empty:
                fill[start:stop+1]=1
                fill[[start,stop]]=.5
            source=np.zeros((3,)+shape);source[0,3*resolution]=1
            out=_legacy_wavebit_fdtd_grid(1+(eps_inf-1)*fill,spacing=(1/resolution,)*3,dt=dt,steps=steps,
                                 source=source,pulse=pulse((np.arange(steps)+.5)*dt,centre),backend='jax',
                                 poles=tuple((w,g,s*(fill if pole_sampling=="average" else (fill>0))) for w,g,s in poles),
                                 pml_cells=round(pml_width*resolution),pml_wavelength=centre,
                                 wavelengths=wavelengths,monitor_z=[9*resolution])
            return np.stack((np.asarray(out['electric_spectrum'])[:,0,0].mean(axis=(1,2)),
                             np.asarray(out['magnetic_spectrum'])[:,0,1].mean(axis=(1,2))),axis=1)
        version=wavebit.__version__
    empty,slab=solve(True),solve(False)
    fields=slab/empty
    transmission=np.real(slab[:,0]*slab[:,1].conj())/np.real(empty[:,0]*empty[:,1].conj())
    w=2*np.pi/wavelengths
    epsilon=eps_inf+sum((s/(w0*w0-w*w-1j*g*w) for w0,g,s in poles),0j)
    n=np.sqrt(epsilon+0j);r=(1-n)/(1+n)
    phase=w*n*.5
    airy=(1-r*r)*np.exp(1j*phase)/(1-r*r*np.exp(2j*phase))*np.exp(-1j*w*.5)
    return dict(backend=backend,version=version,model=model,resolution=resolution,pml_width=pml_width,
                duration=duration,wavelengths=wavelengths.tolist(),pole_sampling="point" if backend=="meep" else pole_sampling,field_real=fields.real.tolist(),field_imag=fields.imag.tolist(),
                transmission=transmission.tolist(),airy_real=airy.real.tolist(),airy_imag=airy.imag.tolist(),
                airy_error=float(np.max(abs(fields[:,0]-airy))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['meep','wavebit'],required=True)
    parser.add_argument('--model',choices=list(MODELS),required=True)
    parser.add_argument('--resolution',type=int,required=True)
    parser.add_argument('--pml-width',type=float,default=1.)
    parser.add_argument('--duration',type=float,default=24.)
    parser.add_argument('--pole-sampling',choices=['average','point'],default='average')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(worker(args.backend,args.resolution,args.model,args.pml_width,args.duration,args.pole_sampling),indent=2)+'\n')


if __name__=='__main__':
    main()
