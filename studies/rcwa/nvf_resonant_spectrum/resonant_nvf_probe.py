"""Separate Fourier truncation and normal quadrature in the tilted contact fixture.

The three Wavebit NVF extensions use exact rectangle transforms and the same solve.
``wavebit-direct`` removes only the NVF correction to check that solve against S4.
It is a scoped diagnostic, not a production factorization option.
The optional S4 direct rule is an independent implementation, also subject to
Fourier truncation. Its actual reciprocal orders must match Wavebit's square cap.
S4 runs in its own interpreter through ``tests/references/interface/s4_resonant_nvf.py``;
set ``S4_PYTHON`` to that interpreter to use the S4 methods.
Run: PYTHONPATH=$PWD python studies/rcwa/nvf_resonant_spectrum/resonant_nvf_probe.py --orders 6 10 14 18
Add --methods circular s4-direct to compare only the new NVF and S4.
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import argparse
import json
import os
import subprocess
import time
from unittest.mock import patch
import numpy as np
from wavebit.rcwa._engine import boundary as B, factorization as F, parametric as P
from tests.references.interface.s4_resonant_nvf import BASES as S4_METHODS, fixture_shape  # noqa: F401
from tools.repository_paths import repository_path

S4_BRIDGE = repository_path('tests/references/interface/s4_resonant_nvf.py')


def polygon_extension(xp, shapes, background, hu, hv, lattice, dtype,
                      shape_nvf_grid_size=256, plan=None):
    return B.to_conv_mat_boundary_layer(xp, shapes, background, hu, hv, lattice,
                                       dtype, shape_nvf_grid_size=shape_nvf_grid_size)


def make_solver(order, grid, method='pieces', shape=None):
    return _legacy_wavebit_rcwa(
        backend='numpy', period=(700., 700.), wavelength=900.,
        top=1., bot=2.25, theta=.2, phi=0., psi=0., thickness=(300.,),
        truncation_order=(order, order), shape_nvf_grid_size=grid,
        nvf_projector='circular' if method == 'circular' else None,
        structure=[dict(background=1., shapes=[fixture_shape() if shape is None else shape])])


def s4_solve_case(order, method):
    """Total (T, R) of the fixture from S4, solved in S4's own interpreter.

    S4 is GPL-licensed and never shares this process: the solve runs in the
    bridge ``tests/references/interface/s4_resonant_nvf.py`` under the
    interpreter named by the ``S4_PYTHON`` environment variable.
    """
    interpreter = os.environ.get('S4_PYTHON')
    if not interpreter:
        raise RuntimeError('Set S4_PYTHON to an interpreter with S4 installed to run the S4 methods.')
    process = subprocess.run([interpreter, str(S4_BRIDGE), '--orders', str(order), '--methods', method],
                             capture_output=True, text=True, encoding='utf-8', check=True)
    row = json.loads(process.stdout.strip().splitlines()[-1])
    return row['T'], row['R']


def direct_from_coeffs(xp, eps_c, inv_eps_c, c_xx, c_xy, c_yy,
                       half_width_u, half_width_v, c_yx=None, eps_bound=None):
    """Keep the exact material series, with Laurent's rule in both transverse slots.

    ``eps_bound`` is accepted to match :func:`wavebit.rcwa._engine.parametric.nvf_from_coeffs` and
    ignored: the inverse rule it bounds is not used here."""
    ep_dir = P.conv_matrix_2d(xp, eps_c, half_width_u, half_width_v)
    zero = xp.zeros_like(ep_dir)
    return F.ConvMatrices(ep_dir, ep_dir, xp.linalg.inv(ep_dir), zero, zero)


def solve_case(order, grid, method='pieces', shape=None):
    """Return total (T, R), restoring every diagnostic override before returning."""
    if method in S4_METHODS:
        if shape is not None:
            raise ValueError('The S4 methods solve the probe fixture only.')
        return s4_solve_case(order, method)
    if method not in ('pieces', 'polygon', 'circular', 'wavebit-direct'):
        raise ValueError(f'unknown comparison method: {method!r}')
    if method == 'wavebit-direct':
        target, builder = 'nvf_from_coeffs', direct_from_coeffs
    else:
        target = 'to_conv_mat_parametric_nvf'
        builder = polygon_extension if method == 'polygon' else P.to_conv_mat_parametric_nvf
    with patch.object(P, target, builder):
        result = make_solver(order, grid, method, shape).solve()
    return float(np.sum(result.T)), float(np.sum(result.R))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[6, 10, 14])
    parser.add_argument('--grids', nargs='+', type=int, default=[256, 512])
    parser.add_argument('--methods', nargs='+',
                        choices=['pieces', 'polygon', 'circular', 'wavebit-direct',
                                 's4-direct', 's4-default', 's4-normal', 's4-jones'],
                        default=['pieces', 'polygon', 'circular'])
    args = parser.parse_args()
    try:
        for order in args.orders:
            for method in args.methods:
                for grid in ([None] if method.startswith('s4-') else args.grids):
                    P._PROJECTOR_CACHE.clear()
                    start = time.monotonic()
                    transmission, reflection = solve_case(order, grid, method)
                    print(json.dumps(dict(order=order, grid=grid, method=method,
                                          T=transmission, R=reflection,
                                          energy_error=abs(transmission+reflection-1.),
                                          seconds=time.monotonic()-start)), flush=True)
    finally:
        P._PROJECTOR_CACHE.clear()


if __name__ == '__main__':
    main()
