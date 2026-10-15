"""Live native-grid slab comparison using only Meep's documented public API.

Run each worker in its own environment; no Meep implementation is read/imported
into Wavebit. Geometry is periodic with a time window preceding wraparound.
"""

from wavebit.fdtd._periodic.grid import fdtd_grid as _legacy_wavebit_fdtd_grid
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.tracebacklimit = 0  # Do not display third-party implementation frames.
WAVELENGTHS = np.array([.6, .7, .8])


def pulse(t):
    return np.exp(-((t-1.5)/.35)**2)*np.sin(2*np.pi*t/.7)


def worker(backend, resolution, duration=18.):
    dt = .4/resolution
    steps = round(duration/dt)
    if backend == 'meep':
        import meep as mp
        mp.verbosity(0)
        def solve(eps):
            geometry = [] if eps == 1 else [mp.Block(
                center=mp.Vector3(z=-8.75), size=mp.Vector3(mp.inf, mp.inf, .5),
                material=mp.Medium(epsilon=eps))]
            sim = mp.Simulation(cell_size=mp.Vector3(z=32), dimensions=1,
                                resolution=resolution, Courant=.4, k_point=mp.Vector3(),
                                eps_averaging=True, geometry=geometry,
                                sources=[mp.Source(mp.CustomSource(pulse, end_time=4),
                                                   component=mp.Ex, center=mp.Vector3(z=-13))])
            monitor = sim.add_dft_fields([mp.Ex, mp.Hy], list(1/WAVELENGTHS),
                                        center=mp.Vector3(z=-7), size=mp.Vector3(),
                                        decimation_factor=1)
            sim.run(until=steps*dt)
            fields = np.array([[np.asarray(sim.get_dft_array(monitor, c, i)).item()
                                for c in (mp.Ex, mp.Hy)] for i in range(len(WAVELENGTHS))])
            sim.reset_meep()
            return fields
        version = mp.__version__
    else:
        import wavebit
        def solve(eps):
            shape = (32*resolution, 2, 2)
            permittivity = np.ones(shape)
            # E nodes represent physical integer grid coordinates. Include half
            # tangential material weights at the two planar interface nodes.
            start, stop = 7*resolution, int(7.5*resolution)
            permittivity[start:stop+1] = eps
            permittivity[[start, stop]] = (1+eps)/2
            source = np.zeros((3,)+shape)
            source[0, 3*resolution] = 1
            out = _legacy_wavebit_fdtd_grid(permittivity, spacing=(1/resolution,)*3,
                                   dt=dt, steps=steps, source=source,
                                   pulse=pulse((np.arange(steps)+.5)*dt),
                                   wavelengths=WAVELENGTHS, monitor_z=[9*resolution],
                                   backend='jax')
            return np.stack((np.asarray(out['electric_spectrum'])[:, 0, 0].mean(axis=(1, 2)),
                             np.asarray(out['magnetic_spectrum'])[:, 0, 1].mean(axis=(1, 2))), axis=1)
        import jax
        jax.config.update('jax_enable_x64', True)
        version = wavebit.__version__
    empty, slab = solve(1.), solve(2.25)
    transfer = slab/empty
    power = np.real(slab[:, 0]*slab[:, 1].conj())/np.real(empty[:, 0]*empty[:, 1].conj())
    return dict(backend=backend, version=version, resolution=resolution, duration=duration,
                wavelengths=WAVELENGTHS.tolist(), field_real=transfer.real.tolist(),
                field_imag=transfer.imag.tolist(), transmission=power.tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['meep','wavebit'],required=True)
    parser.add_argument('--resolution',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=18.)
    args=parser.parse_args()
    args.output.write_text(json.dumps(worker(args.backend,args.resolution,args.duration),indent=2)+'\n')


if __name__ == '__main__':
    main()
