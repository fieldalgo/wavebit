"""Reproduce the closing-gap sensitivity decomposition (not a pytest test).

Run from the checkout:
    PYTHONPATH=$PWD python studies/rcwa/nvf_contact_derivatives/contact_derivative_probe.py --orders 3 5

All modes share the Maxwell solve. Li independently replaces its constitutive
matrices; material/normal freeze only one coefficient group for attribution.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import argparse
import json
from unittest.mock import patch

import numpy as np
from wavebit.rcwa._engine import parametric as P
from tests.fixtures.rcwa.contact_instruments import rectangular_li, split_nvf_derivative


def total(height, order, grid):
    P._PROJECTOR_CACHE.clear()
    solver = _legacy_wavebit_rcwa(
        backend='numpy', period=(700., 700.), wavelength=900., top=1., bot=2.25,
        theta=.2, phi=0., psi=0., thickness=(300.,),
        truncation_order=(order, order), shape_nvf_grid_size=grid,
        structure=[dict(background=1., shapes=[dict(
            type='rectangle', cx=0., cy=0., lx=350., ly=height,
            angle=0., material=12.25)])])
    return float(np.sum(solver.solve().T))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[3, 5])
    parser.add_argument('--gaps', nargs='+', type=float, default=[1e-2, 1e-4, 1e-6])
    parser.add_argument('--grid', type=int, default=256)
    args = parser.parse_args()
    if args.grid < 1 or any(n < 0 for n in args.orders):
        parser.error('grid must be positive and orders nonnegative')
    if any(not np.isfinite(g) or not 0 < g < 700/1.1 for g in args.gaps):
        parser.error('gaps must be finite, positive, and retain a positive height')
    original = P.to_conv_mat_parametric_nvf
    try:
        for gap in args.gaps:
            center = 700. - gap
            for order in args.orders:
                row = dict(gap=gap, order=order, grid=args.grid,
                           steps=[gap/10, gap/100])
                for mode, builder in (
                    ('full', original),
                    ('material', split_nvf_derivative(center, 'material')),
                    ('normal', split_nvf_derivative(center, 'normal')),
                    ('li', rectangular_li),
                ):
                    with patch.object(P, 'to_conv_mat_parametric_nvf', builder):
                        row[mode] = [
                            (total(center+h, order, args.grid) -
                             total(center-h, order, args.grid))/(2*h)
                            for h in row['steps']]
                print(json.dumps(row), flush=True)
    finally:
        P._PROJECTOR_CACHE.clear()


if __name__ == '__main__':
    main()
