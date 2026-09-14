"""Does coupling real seam potentials resolve both periodic-contact polarizations?"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from wavebit._core.conforming import (
    periodic_seam_galerkin, periodic_seam_trace, periodic_rectangle_galerkin,
    cartesian_pair_galerkin, cartesian_pair_quadratic, cartesian_pair_opening,
    cartesian_pair, triangle_moments, material_sector,
)
from studies import nvf_periodic_galerkin, nvf_pair_material_sector


GAPS = (0., .001, 1e-5, 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--refinements', type=int, nargs='+', choices=range(4), default=[0, 1])
    parser.add_argument('--quadrature', type=int, default=32)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results/nvf_coupled_seam_galerkin.log')
    args = parser.parse_args()
    if any(n < 0 for n in args.orders):
        parser.error('orders must be nonnegative')
    modules = (periodic_seam_galerkin, periodic_seam_trace, periodic_rectangle_galerkin,
               cartesian_pair_galerkin, cartesian_pair_quadratic, cartesian_pair_opening,
               cartesian_pair, triangle_moments, material_sector,
               nvf_periodic_galerkin, nvf_pair_material_sector)
    sources = [Path(module.__file__) for module in modules]+[Path(__file__)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as sink:
        def emit(row):
            line = json.dumps(row)
            sink.write(line+'\n'); sink.flush(); print(line, flush=True)
        emit(dict(kind='header', numpy=np.__version__, orders=args.orders,
                  refinements=args.refinements, quadrature=args.quadrature, gaps=GAPS,
                  period=[700., 700.], width=350., epsilon=12.25, background=1.,
                  wavelength=900., thickness=300., theta=.2, phi=0., top=1., bot=2.25,
                  source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}))
        for n in args.orders:
            for refinement in args.refinements:
                obj = periodic_seam_galerkin.PeriodicSeamGalerkinGapCell(
                    (n, n), refinement=refinement, quadrature=args.quadrature)
                start = time.perf_counter()
                totals = {pol: [] for pol in ('p', 's')}
                residuals = []
                for gap in GAPS:
                    value, tangent, metadata = obj.closure(gap=gap)
                    residuals.append({k: float(v) if v is not None else None for k, v in metadata.items()
                                      if k.endswith('residual') or k in ('condition', 'reciprocity',
                                          'derivative_reciprocity', 'minimum_rotated_storage', 'minimum_dissipation')})
                    if gap == 0:
                        closed, derivative = np.asarray(value), np.asarray(tangent)
                    for pol in ('p', 's'):
                        totals[pol].append(nvf_periodic_galerkin.power(n, pol, value)[0])
                for pol in ('p', 's'):
                    reference = [nvf_periodic_galerkin.comparison(n, pol, g, 'li') for g in GAPS]
                    slopes = [(nvf_periodic_galerkin.power(n, pol, closed+h*derivative)[0]
                               -nvf_periodic_galerkin.power(n, pol, closed-h*derivative)[0])/(2*h)
                              for h in (.0005, .00025)]
                    emit(dict(kind='contact', order=n, refinement=refinement, polarization=pol,
                              gaps=GAPS, T=totals[pol], li_T=reference,
                              gap_secants=[(t-totals[pol][0])/g for g, t in zip(GAPS[1:3], totals[pol][1:3])],
                              tangent_gap_slopes=slopes, diagnostics=residuals,
                              seconds=time.perf_counter()-start))


if __name__ == '__main__':
    main()
