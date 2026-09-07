"""Measure instantaneous energy flow inside a dielectric film.

Requires wavebit[plot]. The probe interpolates native Yee space/time coordinates.
The flux is in squared source-field units; positive Sz points into the stack.
"""
import argparse
from pathlib import Path

import numpy as np
import wavebit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('fdtd-poynting.png'))
    args = parser.parse_args()
    _, diagnostics = wavebit.fdtd(
        structure=np.full((1, 1, 2), 2.25), thickness=[100.],
        top=1., bot=1.44, period=[100.], wavelength=900., tetm='te',
        resolution=.03, n_periods=80, record_field=1,
        record_components='all', return_diagnostics=True)
    probe = wavebit.fdtd_poynting(diagnostics, x=0., y=0., z=50.)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(probe['t']/900., probe['S'][:, 2], color='#297fc1')
    ax.set(xlabel='Time (optical cycles)', ylabel='Sz (squared source-field units)',
           title='Instantaneous energy flow at z=50 nm inside a dielectric film')
    ax.axhline(0., color='gray', linewidth=.7)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Probe region: {probe['region']}; samples: {len(probe['t'])}")
    print(f'Flux trace: {args.output.resolve()}')


if __name__ == '__main__':
    main()
