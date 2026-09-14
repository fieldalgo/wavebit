"""How much does an optimized admissible flux tighten the closed error bound?"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from wavebit._core.conforming import (
    galerkin_error, equilibrated_flux, cartesian_pair_galerkin,
    cartesian_pair_opening, cartesian_pair_quadratic, cartesian_pair,
    cartesian_pair_polynomial_galerkin, triangle_polynomials, triangle_moments,
    material_sector,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[4])
    parser.add_argument('--degrees', type=int, nargs='+', choices=(2, 3, 4), default=[3, 4])
    parser.add_argument('--refinements', type=int, nargs='+', choices=range(3), default=[0, 1])
    parser.add_argument('--materials', nargs='+', choices=('dielectric', 'metal'), default=['metal'])
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results/nvf_equilibrated_flux.log')
    args = parser.parse_args()
    if any(n < 0 for n in args.orders):
        parser.error('orders must be nonnegative')
    modules = (galerkin_error, equilibrated_flux, cartesian_pair_galerkin,
               cartesian_pair_opening, cartesian_pair_quadratic, cartesian_pair,
               cartesian_pair_polynomial_galerkin, triangle_polynomials, triangle_moments,
               material_sector)
    sources = [Path(m.__file__) for m in modules]+[Path(__file__)]
    failed = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as sink:
        def emit(row):
            line = json.dumps(row, allow_nan=False)
            sink.write(line+'\n'); sink.flush(); print(line, flush=True)
        emit(dict(kind='header', numpy=np.__version__, orders=args.orders, degrees=args.degrees,
                  refinements=args.refinements, materials=args.materials, gap=0.,
                  period=[700., 700.], widths=[300., 300.], height=300., background=1.,
                  scope='closed transverse constitutive error at fixed retained Fourier modes',
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        for n in args.orders:
            for degree in args.degrees:
                for material in args.materials:
                    for refinement in args.refinements:
                        start = time.perf_counter()
                        key = dict(order=n, degree=degree, material=material, refinement=refinement)
                        angle = 0. if material == 'dielectric' else 1.53
                        obj = cartesian_pair_polynomial_galerkin.CartesianPairPolynomialGalerkinGapCell(
                            cartesian_pair_opening.CartesianPairOpeningPlan(0, refinement), n,
                            degree=degree, coercivity_angle=angle)
                        settings = dict(basis=np.diag([700., 700.]), widths=(300., 300.), height=300.,
                                        epsilon=(4., 4.) if material == 'dielectric' else (12.25, -10.+1j))
                        estimates = []
                        try:
                            for flux in ('surrogate', 'least_squares'):
                                result = galerkin_error.estimate_closed_error(obj, flux=flux, **settings)
                                estimates.append(result)
                                relative = result.relative_error_bound
                                emit(dict(kind='estimate', **key, flux=flux, coercivity_angle=angle,
                                          tensor_norm=float(np.linalg.norm(result.tensor, 2)),
                                          field_error_bound=result.field_error_bound,
                                          operator_error_bound=result.operator_error_bound,
                                          relative_error_bound=relative if np.isfinite(relative) else None,
                                          relative_bound_unbounded=not np.isfinite(relative),
                                          diagnostics=result.diagnostics, seconds=time.perf_counter()-start))
                            old, new = estimates
                            difference = old.residual_gram-new.residual_gram
                            emit(dict(kind='comparison', **key,
                                      tensor_change=float(np.linalg.norm(old.tensor-new.tensor)),
                                      minimum_gram_improvement=float(np.linalg.eigvalsh(
                                          (difference+difference.conj().T)/2)[0]),
                                      combined_roundoff_allowance=sum(r.diagnostics[
                                          'residual_gram_roundoff_allowance'] for r in estimates),
                                      bound_ratio=new.operator_error_bound/old.operator_error_bound))
                        except (ValueError, np.linalg.LinAlgError) as error:
                            failed = True
                            emit(dict(kind='failure', **key, error=str(error), seconds=time.perf_counter()-start))
        emit(dict(kind='coverage', complete=not failed,
                  requested_comparisons=len(args.orders)*len(args.degrees)*len(args.refinements)*len(args.materials)))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
