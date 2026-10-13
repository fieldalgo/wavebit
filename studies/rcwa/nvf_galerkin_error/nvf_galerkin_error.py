"""Does a small weak residual also imply a resolved closed constitutive cell?"""

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

from wavebit.rcwa._engine.conforming import galerkin_error, cartesian_pair_galerkin, cartesian_pair_opening, cartesian_pair_quadratic, cartesian_pair, periodic_rectangle_galerkin, triangle_moments, material_sector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--refinements', type=int, nargs='+', choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument('--out', type=Path, default=repository_path('studies/rcwa/nvf_galerkin_error/records/nvf_galerkin_error.log'))
    args = parser.parse_args()
    if any(n < 0 for n in args.orders):
        parser.error('orders must be nonnegative')
    modules = (galerkin_error, cartesian_pair_galerkin, cartesian_pair_opening,
               cartesian_pair_quadratic, cartesian_pair, periodic_rectangle_galerkin,
               triangle_moments, material_sector)
    sources = [Path(m.__file__) for m in modules]+[Path(__file__)]
    failed = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as sink:
        def emit(row):
            line = json.dumps(row, allow_nan=False)
            sink.write(line+'\n'); sink.flush(); print(line, flush=True)
        emit(dict(kind='header', numpy=np.__version__, orders=args.orders, refinements=args.refinements,
                  gap=0., period=[700., 700.], pair_widths=[300., 300.], pair_height=300.,
                  periodic_width=350., background=1., degree=2,
                  scope='closed transverse constitutive error at fixed retained Fourier modes',
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        for n in args.orders:
            for kind in ('pair', 'periodic'):
                for material in ('dielectric', 'metal'):
                    previous = None
                    for refinement in args.refinements:
                        start = time.perf_counter()
                        angle = 0. if material == 'dielectric' else 1.53
                        if kind == 'pair':
                            obj = cartesian_pair_galerkin.CartesianPairGalerkinGapCell(
                                cartesian_pair_opening.CartesianPairOpeningPlan(0, refinement), n,
                                degree=2, coercivity_angle=angle)
                            settings = dict(basis=np.diag([700., 700.]), widths=(300., 300.), height=300.,
                                            epsilon=(4., 4.) if material == 'dielectric' else (12.25, -10.+1j))
                        else:
                            obj = periodic_rectangle_galerkin.PeriodicRectangleGalerkinGapCell(
                                n, degree=2, refinement=refinement, coercivity_angle=angle)
                            settings = dict(period=(700., 700.), width=350.,
                                            epsilon=4. if material == 'dielectric' else -10.+1j)
                        try:
                            result = galerkin_error.estimate_closed_error(obj, **settings)
                            norm = float(np.linalg.norm(result.tensor, 2))
                            relative = result.relative_error_bound
                            emit(dict(kind='estimate', geometry=kind, material=material, order=n,
                                      refinement=refinement, coercivity_angle=angle, tensor_norm=norm,
                                      field_error_bound=result.field_error_bound,
                                      operator_error_bound=result.operator_error_bound,
                                      bound_over_approximate_norm=result.operator_error_bound/norm,
                                      relative_error_bound=relative if np.isfinite(relative) else None,
                                      relative_bound_unbounded=not np.isfinite(relative),
                                      tensor_change=None if previous is None else float(np.linalg.norm(result.tensor-previous, 2)),
                                      hypercircle_defect=None if material == 'metal' else float(np.linalg.norm(
                                          result.residual_gram-result.tensor+result.dual_tensor)),
                                      diagnostics=result.diagnostics, seconds=time.perf_counter()-start))
                            previous = result.tensor
                        except ValueError as error:
                            failed = True
                            emit(dict(kind='failure', geometry=kind, material=material, order=n,
                                      refinement=refinement, error=str(error), seconds=time.perf_counter()-start))
        emit(dict(kind='coverage', complete=not failed,
                  requested_rows=len(args.orders)*len(args.refinements)*4))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
