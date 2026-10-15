"""Explicitly select ordinary NVF, Cartesian Li, or the experimental pair cell.

This small example demonstrates selection, not optical convergence or a
fair performance comparison. The auxiliary path is never selected implicitly.
"""

from wavebit import Material, Units, rcwa
from wavebit.rcwa import experimental, operators
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
    regions = tuple(rcwa.Region(geometry=rcwa.Rectangle(size=(300.,300.),
        center=(sign*(150.+gap/2),0.)), material=Material(material))
        for sign,material in zip((-1.,1.),epsilon))
    problem = rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(lattice=basis, bottom=Material(2.25),
            layers=(rcwa.Layer(thickness=300.,background=Material(1.),regions=regions),)),
        excitation=rcwa.Excitation(wavelength=900.,sources=(rcwa.PlaneWave(theta=.2,phi=0.,polarization=(1.,0.)),)))
    started = time.perf_counter()
    if method == 'auxiliary':
        options = dict(degree=3, quadrature=80, angular_quadrature=48,
                       coercivity_angle=1.53)
        options.update(auxiliary_options or {})
        refinement = options.pop('refinement', 0)
        cell = experimental.CartesianPairTransmissionGalerkinGapCell(
            experimental.CartesianPairOpeningPlan(0, refinement), orders,
            reference=(*epsilon, 1.), **options)
        matrices, tangent, metadata = cell.closure(
            gap=gap, basis=basis, widths=(300., 300.), height=300.,
            epsilon=epsilon, background=1., gap_tangent=1.)
        assembled = operators.assemble(problem, numerics=rcwa.Numerics(orders=orders,factorization='li'))
        assembled = assembled.replace(epsilon=(operators.ConvMatrices(*matrices),))
        answers = {pol: operators.solve(problem.replace(excitation=problem.excitation.replace(
            sources=(problem.excitation.sources[0].replace(polarization=(0.,1.) if pol=='p' else (1.,0.)),))), assembled) for pol in ('p','s')}
    else:
        options = dict(nvf_projector='circular', shape_nvf_grid_size=64) if method == 'nvf' else {}
        numerics = rcwa.Numerics(orders=orders,factorization=method,**options)
        prepared = rcwa.prepare(problem,numerics=numerics)
        answers = {pol: prepared.run(problem.replace(excitation=problem.excitation.replace(
            sources=(problem.excitation.sources[0].replace(polarization=(0.,1.) if pol=='p' else (1.,0.)),)))) for pol in ('p','s')}
        tangent, metadata = None, dict(factorization=method)
    return answers, tangent, metadata, time.perf_counter()-started


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
                  T=[float(answer[pol].T.sum()) for pol in ('p','s')],
                  R=[float(answer[pol].R.sum()) for pol in ('p','s')],
                  constitutive_gap_jvp_available=tangent is not None)
    if args.method == 'auxiliary':
        output.update(closure=metadata['closure'], degree=metadata['degree'],
                      refinement=metadata['refinement'],
                      energy_residual=float(metadata['energy_residual']),
                      energy_derivative_residual=float(metadata['energy_derivative_residual']))
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
