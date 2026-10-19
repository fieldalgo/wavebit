"""Measure instantaneous energy flow inside a dielectric film.

Requires wavebit[plot]. The probe interpolates native Yee space/time coordinates.
The instantaneous flux is in W/m²; positive Sz points upward; top illumination flows toward negative z.
"""

from wavebit import Units, Material
from wavebit.fdtd import periodic as fdtd
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('fdtd-poynting.png'))
    args = parser.parse_args()
    units=Units(length='nm')
    problem=fdtd.Problem(units=units,
        stack=fdtd.Stack(period=(100.,),bottom=Material(1.44),
            layers=(fdtd.Layer(thickness=100.,epsilon=np.full((1,2),2.25)),)),
        excitation=fdtd.Excitation(wavelength=900.,sources=(fdtd.PlaneWave(),)),
        outputs=fdtd.Outputs(snapshots=fdtd.Snapshots(every=1)))
    result=fdtd.run(problem,numerics=fdtd.Numerics(spacing=1/.03,
        duration_seconds=80*900*units.time_seconds,algorithm='reference'))
    probe=result.sample(x=0.,y=0.,z=50.)
    flux=np.cross(probe.values[:,:3].real,probe.values[:,3:].real)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(probe.time/(900*units.time_seconds), flux[:,2], color='#297fc1')
    ax.set(xlabel='Time (optical cycles)', ylabel='Sz (W/m²)',
           title='Instantaneous energy flow at z=50 nm inside a dielectric film')
    ax.axhline(0., color='gray', linewidth=.7)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Probe region: {probe.metadata['region']}; samples: {len(probe.time)}")
    print(f'Flux trace: {args.output.resolve()}')


if __name__ == '__main__':
    main()
