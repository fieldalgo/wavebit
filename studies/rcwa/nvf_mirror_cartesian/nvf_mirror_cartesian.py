"""Independent Cartesian interval control in the same mirror Maxwell sectors.

The one-dimensional material inversions and x-interval assembly are restricted
before the large inversion. This is algebraically the existing Cartesian Li
instrument, not a fallback inside the NVF energy closure.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import json
import time

import numpy as np
from scipy.linalg import block_diag, solve

from studies.rcwa.nvf_fourier_orders.nvf_fourier_orders import orders_xy, parse_order
from studies.rcwa.nvf_harmonic_energy.nvf_harmonic_energy import PERIOD
from studies.rcwa.nvf_mirror_convergence.nvf_mirror_convergence import parity_basis, powers_from_sectors, propagation_method


def cartesian_sector(order, gap=17., width=350., seams=(31.,), values=(4., 9.), *, polarization='p'):
    seams, values = np.asarray(seams), np.asarray(values)
    edges = np.r_[-width/2, seams, width/2]
    if (seams.ndim != 1 or values.shape != (len(seams)+1,)
            or not np.isfinite(np.r_[edges, values, gap]).all() or np.min(values) <= 0
            or not np.all(np.diff(edges) > 0) or not 0 < width < PERIOD or not 0 <= gap < PERIOD):
        raise ValueError('invalid rectangular material profile')
    if polarization not in ('p', 's'):
        raise ValueError('polarization must be p or s')
    ox, oy = orders_xy(order)
    m, n = np.arange(-ox, ox+1), np.arange(-oy, oy+1)
    dm, dn = m[:, None]-m[None], n[:, None]-n[None]
    height = 1-gap/PERIOD
    # profile_li uses intervals centered at fractional .5 in both directions.
    y_body = height*np.sinc(height*dn)*(-1.)**dn
    x_parts = [((hi-lo)/PERIOD*np.sinc(dm*(hi-lo)/PERIOD)
                 *np.exp(-2j*np.pi*dm*(.5+(hi+lo)/(2*PERIOD)))) for lo, hi in zip(edges[:-1], edges[1:])]

    def assembly(parity, rule):
        basis = parity_basis((0, oy), parity)
        size = len(basis['n'])
        if size == 0:
            return np.empty((0, 0), complex)
        u = np.zeros((2*oy+1, size))
        u[oy+basis['n'], np.arange(size)] += basis['primary']
        u[oy-basis['n'], np.arange(size)] += basis['reflected']
        indicator = u.T@y_body@u
        identity = np.eye(size)
        result = np.eye(size*(2*ox+1), dtype=complex)
        for epsilon, x_part in zip(values, x_parts):
            material = 1/epsilon if rule == 'y' else epsilon
            inner = identity+(material-1)*indicator
            if rule in ('x', 'y'):
                inner = solve(inner, identity, assume_a='pos')
            result += np.kron(inner-identity, x_part)
        if rule in ('x', 'z'):
            result = solve(result, np.eye(len(result)), assume_a='pos')
        return result

    parity = 1 if polarization == 'p' else -1
    ex, ey = assembly(parity, 'x'), assembly(-parity, 'y')
    return dict(a=parity_basis(order, parity), b=parity_basis(order, -parity),
                effective=block_diag(ex, ey), z_inverse=assembly(parity, 'z'),
                polarization=polarization, mirror_error=0.)


def cartesian_powers(order, gap=17., width=350., seams=(31.,), values=(4., 9.), progress=None,
                     *, propagation='modal', doublings=6, terms=12):
    return powers_from_sectors(lambda pol: cartesian_sector(order, gap, width, seams, values, polarization=pol), progress,
                               propagate=propagation_method(propagation, doublings, terms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=parse_order, nargs='+', default=[24, 32])
    parser.add_argument('--gap', type=float, default=17.)
    parser.add_argument('--width', type=float, default=350.)
    parser.add_argument('--seams', type=float, nargs='*', default=[31.])
    parser.add_argument('--values', type=float, nargs='+', default=[4., 9.])
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--propagation', choices=('modal', 'doubling'), default='modal')
    parser.add_argument('--doublings', type=int, default=6)
    parser.add_argument('--terms', type=int, default=12)
    args = parser.parse_args()
    for order in args.orders:
        start = time.monotonic()
        def report(stage):
            print(json.dumps(dict(kind='cartesian_mirror_progress', order=order, **stage,
                                  seconds=time.monotonic()-start)), flush=True)
        result = cartesian_powers(order, args.gap, args.width, args.seams, args.values, report if args.progress else None,
                                  propagation=args.propagation, doublings=args.doublings, terms=args.terms)
        print(json.dumps(dict(kind='cartesian_mirror', order=order, gap=args.gap, width=args.width,
                              propagation=args.propagation, doublings=args.doublings, terms=args.terms,
                              seams=args.seams, values=args.values, **result, seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
