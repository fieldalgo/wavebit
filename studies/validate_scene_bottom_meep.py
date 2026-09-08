"""Bottom-illuminated asymmetric stack through Meep's public API only.

The source physically sits BELOW the original layers and emits toward -z.
No reflected-coordinate solve is used by this independent reference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.tracebacklimit = 0  # Never display Meep implementation frames.


def worker(resolution=80, duration=30.):
    import meep as mp
    mp.verbosity(0)
    wl = np.array([.9, 1., 1.1])
    thickness = [.13, .21]
    depth = sum(thickness)
    dt = .4/resolution
    def pulse(t):
        return np.exp(-((t-3.5)/.7)**2)*np.sin(2*np.pi*t)
    def solve(material):
        geometry = [] if not material else [
            mp.Block(center=mp.Vector3(z=.065),size=mp.Vector3(mp.inf,mp.inf,.13),
                     material=mp.Medium(epsilon=2.)),
            mp.Block(center=mp.Vector3(z=.13+.105),size=mp.Vector3(mp.inf,mp.inf,.21),
                     material=mp.Medium(epsilon=3.))]
        sim=mp.Simulation(cell_size=mp.Vector3(z=12),dimensions=1,
                          boundary_layers=[mp.PML(1.)],resolution=resolution,Courant=.4,
                          k_point=mp.Vector3(),eps_averaging=True,geometry=geometry,
                          sources=[mp.Source(mp.CustomSource(pulse,end_time=8),
                                             component=mp.Ex,center=mp.Vector3(z=3.))])
        monitors=[sim.add_dft_fields([mp.Ex],list(1/wl),center=mp.Vector3(z=z),
                                    size=mp.Vector3(),decimation_factor=1) for z in (2.,-2.)]
        sim.run(until=duration)
        fields=np.array([[np.asarray(sim.get_dft_array(m,mp.Ex,i)).item()
                          for i in range(len(wl))] for m in monitors])
        sim.reset_meep()
        return fields
    empty,actual=solve(False),solve(True)
    # De-embed native Yee propagation in the vacuum half spaces to z=depth/0.
    k=2*resolution*np.arcsin(np.sin(np.pi*dt/wl)/(dt*resolution))
    r=(actual[0]/empty[0]-1)*np.exp(-2j*k*(2.-depth))
    t=actual[1]/empty[1]*np.exp(1j*k*depth)
    return dict(backend='meep',version=mp.__version__,resolution=resolution,duration=duration,
                wavelengths=wl.tolist(),epsilon=[2.,3.],thickness=thickness,
                source_z=3.,monitor_z=[2.,-2.],incidence_side='bottom',
                r_real=r.real.tolist(),r_imag=r.imag.tolist(),
                t_real=t.real.tolist(),t_imag=t.imag.tolist(),R=(abs(r)**2).tolist(),T=(abs(t)**2).tolist())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolution',type=int,default=80)
    parser.add_argument('--duration',type=float,default=30.)
    args=parser.parse_args()
    args.output.write_text(json.dumps(worker(args.resolution,args.duration),indent=2)+'\n')
