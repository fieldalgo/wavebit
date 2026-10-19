"""Take field snapshots of a conical FDTD pulse and plot one, using physical declarations.

Install with ``pip install 'wavebit[plot]'``. Lengths are in nm; material values
are relative permittivities, angles are radians. The explicit reference stepping route preserves this pulse calculation.
This is not a grid-convergence or accuracy claim.
"""

from wavebit import Units, Material
from wavebit.fdtd import periodic as fdtd
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('fdtd-field.png'))
    parser.add_argument('--component', choices=('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'),
                        default='Ex', help='electric (V/m) or magnetic (A/m) field component')
    parser.add_argument('--psi', type=float, default=.37, help='polarization angle in radians')
    parser.add_argument('--chi', type=float, default=None,
                        help='ellipticity in radians; pi/4 gives a circular pulse')
    args = parser.parse_args()
    units=Units(length='nm')
    chi=0. if args.chi is None else args.chi
    # Convert polarization angles to the electric Jones pair explicitly.
    pair=(np.sin(args.psi)*np.cos(chi)+1j*np.cos(args.psi)*np.sin(chi),
          np.cos(args.psi)*np.cos(chi)-1j*np.sin(args.psi)*np.sin(chi))
    problem=fdtd.Problem(units=units,
        stack=fdtd.Stack(period=(300.,),bottom=Material(1.44),
            layers=(fdtd.Layer(thickness=100.,epsilon=np.array([[1.,2.25]])),)),
        excitation=fdtd.Excitation(wavelength=900.,sources=(fdtd.PlaneWave(theta=.3,phi=.4,polarization=pair),)),
        outputs=fdtd.Outputs(snapshots=fdtd.Snapshots(every=5,components=(args.component,))))
    result=fdtd.run(problem,numerics=fdtd.Numerics(spacing=1/.025,
        duration_seconds=80*900*units.time_seconds,algorithm='reference'))
    series=result.snapshots[args.component]
    strength=np.max(np.abs(series.values.real),axis=(1,2,3))
    frame=int(np.argmax(strength))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(constrained_layout=True)
    mesh=ax.pcolormesh(series.coordinates['x'],series.coordinates['z'],series.values[frame,:,0,:].real,shading='auto')
    fig.colorbar(mesh,ax=ax,label=f"{args.component} ({series.units['field']})")
    ax.set(xlabel='x (nm)',ylabel='z (nm)',title=f't = {float(series.time[frame])*1e15:.3f} fs')
    fig.savefig(args.output, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'R={float(result.R.sum()):.6f}, T={float(result.T.sum()):.6f}, '
          f"residual={result.diagnostics['residual']:.3g}")
    print(f'Field snapshot: {args.output.resolve()}')


if __name__ == '__main__':
    main()
