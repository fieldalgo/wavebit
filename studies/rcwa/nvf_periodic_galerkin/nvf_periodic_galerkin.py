"""Does the real Galerkin gap chart resolve the periodic-contact derivative?"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from wavebit.rcwa._engine import parametric
from wavebit.rcwa._engine.conforming import periodic_rectangle_galerkin, cartesian_pair_galerkin, cartesian_pair, cartesian_pair_opening, cartesian_pair_quadratic, triangle_moments, material_sector
from studies.rcwa.nvf_pair_material_sector.nvf_pair_material_sector import solver


GAPS = (0., .0005, .001, .002, 1e-9)


@lru_cache(None)
def comparison(order, polarization, gap, factorization):
    options = dict(shape_nvf_grid_size=256) if factorization == 'nvf' else {}
    obj = solver(order, polarization, factorization=factorization, **options,
                 structure=[dict(background=1., shapes=[dict(
                     type='rectangle', lx=350., ly=700.-gap, material=12.25)])])
    result = obj.solve()
    return float(result.T.real.sum())


def power(order, polarization, slots):
    result = solver(order, polarization)._solve_conv(*slots)
    t, r = float(result.T.real.sum()), float(result.R.real.sum())
    if not np.isfinite(t+r) or min(t, r) < -1e-9 or abs(t+r-1) > 1e-9:
        raise RuntimeError('lossless physical power guard failed')
    return t, r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[3, 5, 7, 9])
    parser.add_argument('--refinements', type=int, nargs='+', choices=range(4), default=[1, 2, 3])
    parser.add_argument('--degree', type=int, choices=(1, 2), default=2)
    parser.add_argument('--out', type=Path, default=repository_path('studies/rcwa/nvf_periodic_galerkin/records/nvf_periodic_galerkin.log'))
    args = parser.parse_args()
    if any(n < 0 for n in args.orders):
        parser.error('orders must be nonnegative')
    sources = [m.__file__ for m in (periodic_rectangle_galerkin, cartesian_pair_galerkin,
                                    cartesian_pair, cartesian_pair_opening, cartesian_pair_quadratic,
                                    triangle_moments, material_sector, parametric)]
    sources += [__file__, repository_path('studies/rcwa/nvf_pair_material_sector/nvf_pair_material_sector.py')]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as sink:
        def emit(row):
            line = json.dumps(row)
            sink.write(line+'\n'); sink.flush(); print(line, flush=True)
        emit(dict(kind='header', numpy=np.__version__, orders=args.orders, refinements=args.refinements,
                  degree=args.degree, gaps=GAPS, period=[700., 700.], width=350., epsilon=12.25,
                  background=1., top=1., bot=2.25, wavelength=900., thickness=300., theta=.2, phi=0.,
                  original_normal_grid=256,
                  source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}))
        for n in args.orders:
            for r in args.refinements:
                obj = periodic_rectangle_galerkin.PeriodicRectangleGalerkinGapCell(
                    (n, n), refinement=r, degree=args.degree)
                start = time.perf_counter()
                totals = {'p': [], 's': []}; reflections = {'p': [], 's': []}
                for gap in GAPS:
                    slots, derivative, metadata = obj.closure(gap=gap)
                    if gap == 0:
                        closed, tangent = np.asarray(slots), np.asarray(derivative)
                        diagnostics = {k: float(v) for k, v in metadata.items()
                                       if k.endswith('residual') or k in (
                                           'solve_condition', 'weak_condition', 'relative_reciprocity_defect',
                                           'derivative_reciprocity_defect', 'minimum_dissipation',
                                           'minimum_rotated_storage')}
                    for pol in ('p', 's'):
                        t, reflection = power(n, pol, slots)
                        totals[pol].append(t); reflections[pol].append(reflection)
                for pol in ('p', 's'):
                    li = [comparison(n, pol, gap, 'li') for gap in GAPS]
                    original = [comparison(n, pol, gap, 'nvf') for gap in GAPS]
                    slopes = lambda v: [(-3*v[0]+4*v[1]-v[2])/.001, (-3*v[0]+4*v[2]-v[3])/.002]
                    tangent_slopes = [(power(n, pol, closed+h*tangent)[0]
                                       -power(n, pol, closed-h*tangent)[0])/(2*h)
                                      for h in (.001, .0005)]
                    emit(dict(kind='contact', order=n, refinement=r, degree=args.degree, polarization=pol,
                              gaps=GAPS, T=totals[pol], R=reflections[pol], li_T=li, original_T=original,
                              gap_slopes=slopes(totals[pol]), li_gap_slopes=slopes(li),
                              original_gap_slopes=slopes(original), tangent_gap_slopes=tangent_slopes,
                              elimination=metadata['elimination'], diagnostics=diagnostics,
                              open_free_dofs=metadata['open_free_dofs'], contact_free_dofs=metadata['contact_free_dofs'],
                              seconds=time.perf_counter()-start))


if __name__ == '__main__':
    main()
