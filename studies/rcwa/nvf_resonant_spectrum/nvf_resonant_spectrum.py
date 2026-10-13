"""Where does the tilted contact fixture's NVF/direct-rule gap come from: a resonance shift?

The fixture is the one in ``studies/rcwa/nvf_resonant_spectrum/resonant_nvf_probe.py`` (square 700, box 640 x 700/sqrt5 at
atan2(1, 2), eps 12.25 on 1, substrate 2.25, thickness 300, theta .2 rad, p polarization).
The probe compares total transmission at one wavelength, 900 nm. This study adds a wavelength
argument and records every propagating order separately, because two things the total cannot
show decide the question:

* which diffraction channels carry the circular-NVF/direct-rule difference, and
* whether 900 nm sits on the flank of a resonance whose *position* moves with truncation --
  in which case the difference in T(900) is a wavelength shift, and the converged reference has
  to be stated as a resonance position, not as a number at one wavelength.

The substrate (1, 0) order has its Rayleigh line at ``700 (1.5 - sin .2) = 910.93`` nm, 11 nm
from the operating point, so the scan brackets that line. Extrapolation of the resonance
markers is done by the reader of the JSON lines, not here; see
``docs/validation/rcwa/nvf-contact-followup.md`` for the fits and what they decided.

Run from the repository root:
    PYTHONPATH=$PWD python studies/rcwa/nvf_resonant_spectrum/nvf_resonant_spectrum.py --orders 6 10 14 --wavelengths 900
    PYTHONPATH=$PWD python studies/rcwa/nvf_resonant_spectrum/nvf_resonant_spectrum.py --orders 6 10 14 18 \
        --methods circular wavebit-direct --wavelengths $(seq 905 925)
"""

from wavebit.compat import legacy as _legacy_wavebit

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import argparse
import json
import sys
import time
from unittest.mock import patch

import numpy as np
from wavebit.rcwa._engine import parametric as P

sys.path.insert(0, str(repository_path('tests')))
from studies.rcwa.nvf_resonant_spectrum.resonant_nvf_probe import direct_from_coeffs, fixture_shape, polygon_extension  # noqa: E402

PERIOD, THETA, N_SUBSTRATE = 700., .2, 1.5


def make_solver(order, grid, method, wavelength):
    return _legacy_wavebit.rcwa(
        backend='numpy', period=(PERIOD, PERIOD), wavelength=wavelength,
        top=1., bot=2.25, theta=THETA, phi=0., psi=0., thickness=(300.,),
        truncation_order=(order, order), shape_nvf_grid_size=grid,
        nvf_projector='circular' if method == 'circular' else None,
        structure=[dict(background=1., shapes=[fixture_shape()])])


def solve_full(order, grid, method, wavelength):
    """The probe's method dispatch, returning the whole result instead of the totals."""
    if method == 'wavebit-direct':
        target, builder = 'nvf_from_coeffs', direct_from_coeffs
    else:
        target = 'to_conv_mat_parametric_nvf'
        builder = polygon_extension if method == 'polygon' else P.to_conv_mat_parametric_nvf
    with patch.object(P, target, builder):
        return make_solver(order, grid, method, wavelength).solve()


def per_order(result, order, wavelength):
    """Every propagating order: T in the substrate, R in air. Result arrays are v-first."""
    sin = np.sin(THETA)
    out = {}
    for m in range(-2, 3):
        for n in range(-2, 3):
            kx, ky = sin + m*wavelength/PERIOD, n*wavelength/PERIOD
            if kx*kx + ky*ky < N_SUBSTRATE**2:
                out[f'T({m},{n})'] = float(result.T[n + order, m + order])
            if kx*kx + ky*ky < 1.:
                out[f'R({m},{n})'] = float(result.R[n + order, m + order])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[6, 10, 14])
    parser.add_argument('--grid', type=int, default=512)
    parser.add_argument('--methods', nargs='+', default=['circular', 'wavebit-direct'],
                        choices=['pieces', 'polygon', 'circular', 'wavebit-direct'])
    parser.add_argument('--wavelengths', nargs='+', type=float, default=[900.])
    parser.add_argument('--out', default=None, help='append JSON lines here as well')
    args = parser.parse_args()
    sink = open(args.out, 'a') if args.out else None
    try:
        for order in args.orders:
            for method in args.methods:
                for wavelength in args.wavelengths:
                    P._PROJECTOR_CACHE.clear()
                    start = time.monotonic()
                    result = solve_full(order, args.grid, method, wavelength)
                    T, R = float(np.sum(result.T)), float(np.sum(result.R))
                    line = json.dumps(dict(order=order, grid=args.grid, method=method,
                                           wavelength=wavelength, T=T, R=R,
                                           energy_error=abs(T + R - 1.),
                                           seconds=round(time.monotonic() - start, 2),
                                           **per_order(result, order, wavelength)))
                    print(line, flush=True)
                    if sink:
                        sink.write(line + '\n')
                        sink.flush()
    finally:
        P._PROJECTOR_CACHE.clear()
        if sink:
            sink.close()


if __name__ == '__main__':
    main()
