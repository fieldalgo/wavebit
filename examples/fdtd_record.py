"""Record and plot a conical FDTD pulse using the public scene API.

Install with ``pip install 'wavebit[plot]'``. Lengths are in nm; material values
are relative permittivities, angles are radians. This research-alpha example
illustrates the API, not a grid-convergence or accuracy claim.
"""
import argparse
from pathlib import Path

import numpy as np
import wavebit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('fdtd-field.png'))
    parser.add_argument('--psi', type=float, default=.37, help='polarization angle in radians')
    parser.add_argument('--chi', type=float, default=None,
                        help='ellipticity in radians; pi/4 gives a circular pulse')
    args = parser.parse_args()
    scene = wavebit.Stack(
        layers=[wavebit.Layer(thickness=100., structure=np.array([[1., 2.25]]))],
        top=1., bot=1.44, period=(300.,))
    incidence = wavebit.Incidence(wavelength=900., theta=.3, phi=.4,
                                 psi=args.psi, chi=args.chi)
    result, diagnostics = wavebit.fdtd(
        stack=scene, incidence=incidence, resolution=.025, n_periods=80,
        record_field=5, return_diagnostics=True)
    # The pulse has left by the final frame; show the strongest recorded Ex slice.
    strength = np.max(np.abs(diagnostics['field_ex'].real), axis=(1, 2, 3))
    frame = int(np.argmax(strength))
    fig, _, _ = wavebit.plot_fdtd_field(diagnostics, frame=frame, component='ex')
    fig.savefig(args.output, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'R={float(result.R.sum()):.6f}, T={float(result.T.sum()):.6f}, '
          f"residual={diagnostics['residual']:.3g}")
    print(f'Field snapshot: {args.output.resolve()}')


if __name__ == '__main__':
    main()
