"""Does raising interface and interior polynomial degree improve the closed error bound?"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from wavebit.rcwa._engine.conforming import galerkin_error, cartesian_pair_polynomial_galerkin, triangle_polynomials, cartesian_pair_galerkin, cartesian_pair_opening, cartesian_pair_quadratic, cartesian_pair, periodic_rectangle_galerkin, triangle_moments, material_sector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--spaces', nargs='+', default=['2:0', '3:0', '4:0', '2:1', '3:1', '4:1', '2:2'],
                        help='polynomial degree:mesh refinement pairs')
    parser.add_argument('--out', type=Path, default=repository_path('studies/rcwa/nvf_polynomial_galerkin/records/nvf_polynomial_galerkin.log'))
    args = parser.parse_args()
    if any(n < 0 for n in args.orders):
        parser.error('orders must be nonnegative')
    try:
        spaces = [tuple(map(int, item.split(':'))) for item in args.spaces]
        if any(len(pair) != 2 or pair[0] not in (2, 3, 4) or pair[1] not in range(4)
               or pair[0]*2**pair[1] > 16 for pair in spaces):
            raise ValueError
    except ValueError:
        parser.error('spaces must be supported degree:refinement pairs with edge resolution <= 16')
    modules = (galerkin_error, cartesian_pair_polynomial_galerkin, triangle_polynomials,
               cartesian_pair_galerkin, cartesian_pair_opening, cartesian_pair_quadratic,
               cartesian_pair, periodic_rectangle_galerkin, triangle_moments, material_sector)
    sources = [Path(m.__file__) for m in modules]+[Path(__file__)]
    failed = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as sink:
        def emit(row):
            line = json.dumps(row, allow_nan=False)
            sink.write(line+'\n'); sink.flush(); print(line, flush=True)
        emit(dict(kind='header', numpy=np.__version__, orders=args.orders, spaces=spaces,
                  period=[700., 700.], widths=[300., 300.], height=300., gap=0., background=1.,
                  scope='closed transverse constitutive bound at fixed Fourier order; no optical certificate',
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        for order in args.orders:
            for material, epsilon in (('dielectric', (4., 4.)), ('metal', (12.25, -10+1j))):
                for degree, refinement in spaces:
                    start = time.perf_counter()
                    family = (cartesian_pair_galerkin.CartesianPairGalerkinGapCell if degree == 2
                              else cartesian_pair_polynomial_galerkin.CartesianPairPolynomialGalerkinGapCell)
                    cell = family(cartesian_pair_opening.CartesianPairOpeningPlan(0, refinement), order,
                                  degree=degree, coercivity_angle=0. if material == 'dielectric' else 1.53)
                    try:
                        result = galerkin_error.estimate_closed_error(cell, epsilon=epsilon,
                            basis=np.diag([700., 700.]), widths=(300., 300.), height=300.)
                        norm = float(np.linalg.norm(result.tensor, 2))
                        emit(dict(kind='estimate', order=order, material=material, degree=degree,
                                  refinement=refinement, open_potential_count=len(cell.data['pmap'])-1,
                                  closed_potential_count=cell.data['C'].shape[1], tensor_norm=norm,
                                  field_error_bound=result.field_error_bound,
                                  operator_error_bound=result.operator_error_bound,
                                  bound_over_approximate_norm=result.operator_error_bound/norm,
                                  relative_error_bound=(result.relative_error_bound
                                      if np.isfinite(result.relative_error_bound) else None),
                                  hypercircle_defect=(None if material == 'metal' else float(np.linalg.norm(
                                      result.residual_gram-result.tensor+result.dual_tensor))),
                                  diagnostics=result.diagnostics, seconds=time.perf_counter()-start))
                    except ValueError as error:
                        failed = True
                        emit(dict(kind='failure', order=order, material=material, degree=degree,
                                  refinement=refinement, error=str(error), seconds=time.perf_counter()-start))
        emit(dict(kind='coverage', complete=not failed, requested_rows=2*len(args.orders)*len(spaces)))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
