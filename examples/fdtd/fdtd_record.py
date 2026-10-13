"""Record and plot a conical FDTD pulse using the explicit compatibility API.

Install with ``pip install 'wavebit[plot]'``. Lengths are in nm; material values
are relative permittivities, angles are radians. This research-alpha example
preserves the historical plotting workflow. For the current physical API, see
``examples/fdtd/pulse_slab.py``. This is not a grid-convergence or accuracy claim.
"""

from wavebit.compat import legacy as _legacy_wavebit
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('fdtd-field.png'))
    parser.add_argument('--component', choices=('ex', 'ey', 'ez', 'hx', 'hy', 'hz'),
                        default='ex', help='electric or normalized magnetic field component')
    parser.add_argument('--psi', type=float, default=.37, help='polarization angle in radians')
    parser.add_argument('--chi', type=float, default=None,
                        help='ellipticity in radians; pi/4 gives a circular pulse')
    args = parser.parse_args()
    scene = _legacy_wavebit.Stack(
        layers=[_legacy_wavebit.Layer(thickness=100., structure=np.array([[1., 2.25]]))],
        top=1., bot=1.44, period=(300.,))
    incidence = _legacy_wavebit.Incidence(wavelength=900., theta=.3, phi=.4,
                                 psi=args.psi, chi=args.chi)
    result, diagnostics = _legacy_wavebit.fdtd(
        stack=scene, incidence=incidence, resolution=.025, n_periods=80,
        record_field=5, record_components=(args.component,), return_diagnostics=True)
    # The pulse has left by the final frame; show the strongest recorded slice.
    strength = np.max(np.abs(diagnostics['fields'][args.component]['values'].real), axis=(1, 2, 3))
    frame = int(np.argmax(strength))
    fig, _, _ = _legacy_wavebit.plot_fdtd_field(diagnostics, frame=frame, component=args.component)
    fig.savefig(args.output, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'R={float(result.R.sum()):.6f}, T={float(result.T.sum()):.6f}, '
          f"residual={diagnostics['residual']:.3g}")
    print(f'Field snapshot: {args.output.resolve()}')


if __name__ == '__main__':
    main()
