"""Explicitly select ordinary NVF, Cartesian Li, or the experimental pair cell.

This small example demonstrates selection, not optical convergence or a
fair performance comparison. The auxiliary path is never selected implicitly.
"""

from wavebit.compat import legacy as _legacy_wavebit
import argparse
import json
import time

import numpy as np


def solve_pair(method='nvf', *, order=1, gap=0., auxiliary_options=None):
    """Select one construction for the same fixed equal-height rectangle pair."""
    if method not in ('li', 'nvf', 'auxiliary'):
        raise ValueError("method must be 'li', 'nvf' or 'auxiliary'")
    if method != 'auxiliary' and auxiliary_options is not None:
        raise ValueError('auxiliary_options require method=auxiliary')
    if not np.isfinite(gap) or gap < 0:
        raise ValueError('gap must be finite and nonnegative')
    epsilon = (12.25, -10.+1j)
    basis = np.diag([700., 700.])
    orders = (order, order) if isinstance(order, (int, np.integer)) else order
    common = dict(backend='numpy', lattice=basis, wavelength=900., thickness=(300.,),
                  top=1., bot=2.25, theta=.2, phi=0., truncation_order=orders)
    started = time.perf_counter()
    if method == 'auxiliary':
        options = dict(degree=3, quadrature=80, angular_quadrature=48,
                       coercivity_angle=1.53)
        options.update(auxiliary_options or {})
        refinement = options.pop('refinement', 0)
        cell = _legacy_wavebit.CartesianPairTransmissionGalerkinGapCell(
            _legacy_wavebit.CartesianPairOpeningPlan(0, refinement), orders,
            reference=(*epsilon, 1.), **options)
        matrices, tangent, metadata = cell.closure(
            gap=gap, basis=basis, widths=(300., 300.), height=300.,
            epsilon=epsilon, background=1., gap_tangent=1.)
        answer = _legacy_wavebit.rcwa(**common).solve_conv(*matrices)
    else:
        shapes = [dict(type='rectangle', lx=300., ly=300.,
                       cx=sign*(150.+gap/2), cy=0., material=material)
                  for sign, material in zip((-1., 1.), epsilon)]
        options = dict(nvf_projector='circular', shape_nvf_grid_size=64) if method == 'nvf' else {}
        answer = _legacy_wavebit.rcwa(
            **common, structure=[dict(background=1., shapes=shapes)],
            factorization=method, **options).solve()
        tangent, metadata = None, dict(factorization=method)
    return answer, tangent, metadata, time.perf_counter()-started


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--method', choices=('li', 'nvf', 'auxiliary'), default='nvf')
    parser.add_argument('--order', type=int, default=1)
    parser.add_argument('--gap', type=float, default=0.)
    parser.add_argument('--degree', type=int, choices=range(3, 9))
    parser.add_argument('--refinement', type=int, choices=range(4))
    args = parser.parse_args()
    if args.order < 0:
        parser.error('order must be nonnegative')
    options = {key: value for key, value in
               (('degree', args.degree), ('refinement', args.refinement)) if value is not None}
    if args.method != 'auxiliary' and options:
        parser.error('--degree and --refinement require --method auxiliary')
    print(json.dumps(dict(method=args.method, auxiliary_enabled=args.method == 'auxiliary',
                          experimental=args.method == 'auxiliary', order=args.order,
                          gap=args.gap, optical_convergence_certified=False)), flush=True)
    answer, tangent, metadata, seconds = solve_pair(
        args.method, order=args.order, gap=args.gap,
        auxiliary_options=options if args.method == 'auxiliary' else None)
    output = dict(method=args.method, seconds=seconds, polarizations=['p', 's'],
                  T=[float(result.T.sum()) for result in (answer.p_inc, answer.s_inc)],
                  R=[float(result.R.sum()) for result in (answer.p_inc, answer.s_inc)],
                  constitutive_gap_jvp_available=tangent is not None)
    if args.method == 'auxiliary':
        output.update(closure=metadata['closure'], degree=metadata['degree'],
                      refinement=metadata['refinement'],
                      energy_residual=float(metadata['energy_residual']),
                      energy_derivative_residual=float(metadata['energy_derivative_residual']))
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
