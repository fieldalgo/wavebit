"""Does a real constrained-gradient response improve finite contact optics?"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from wavebit._core.conforming import cartesian_pair_galerkin as galerkin
from wavebit._core.conforming import cartesian_pair, cartesian_pair_opening, cartesian_pair_quadratic, triangle_moments, material_sector
from studies.nvf_pair_material_sector import GEOMETRY, MATERIALS, solver, li_value

GAPS = (0., .0005, .001, .002, 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--degree', type=int, choices=(1, 2), default=1)
    parser.add_argument('--refinements', type=int, nargs='+', choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument('--derivative-orders', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results/nvf_pair_galerkin.log')
    args = parser.parse_args()
    if any(n < 0 for n in args.orders+args.derivative_orders):
        parser.error('orders must be nonnegative')
    sources = [m.__file__ for m in (galerkin, cartesian_pair, cartesian_pair_opening,
                                   cartesian_pair_quadratic, triangle_moments, material_sector)]
    sources += [__file__, Path(__file__).with_name('nvf_pair_material_sector.py')]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as sink:
        def emit(row):
            line = json.dumps(row)
            sink.write(line+'\n'); sink.flush(); print(line, flush=True)
        emit(dict(kind='header', numpy=np.__version__, degree=args.degree, orders=args.orders, refinements=args.refinements,
                  derivative_orders=args.derivative_orders, gaps=GAPS,
                  cell=700., width=300., height=300., thickness=300., wavelength=900.,
                  theta=.2, phi=0., top=1., bot=2.25, background=1., coercivity_angle=1.53,
                  source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}))
        for order in args.orders:
            for refinement in args.refinements:
                plan = cartesian_pair_opening.CartesianPairOpeningPlan.from_geometry(**GEOMETRY, refinement=refinement)
                cell = galerkin.CartesianPairGalerkinGapCell(plan, (order, order), degree=args.degree, coercivity_angle=1.53)
                for name, epsilon in MATERIALS.items():
                    start = time.perf_counter()
                    gaps = GAPS if order in args.derivative_orders else (0.,)
                    values = {'p': [], 's': []}; reflections = {'p': [], 's': []}
                    for gap in gaps:
                        slots, derivative, metadata = cell.closure(gap=gap, epsilon=epsilon, **GEOMETRY)
                        if gap == 0:
                            diagnostics = {k: float(v) for k, v in metadata.items()
                                           if k.endswith('residual') or k in ('solve_condition', 'weak_condition',
                                               'relative_reciprocity_defect', 'derivative_reciprocity_defect',
                                               'minimum_rotated_storage', 'minimum_dissipation')}
                            derivative_maximum = float(max(np.max(np.abs(v)) for v in derivative))
                        for pol in ('p', 's'):
                            result = solver(order, pol)._solve_conv(*slots)
                            t, r = float(result.T.real.sum()), float(result.R.real.sum())
                            if not np.isfinite(t+r) or min(t, r) < -1e-9 or t+r > 1+1e-9:
                                raise RuntimeError('nonfinite or active optical output')
                            if name == 'equal' and abs(t+r-1) > 1e-9:
                                raise RuntimeError('lossless power balance failed')
                            values[pol].append(t); reflections[pol].append(r)
                    for pol in ('p', 's'):
                        li = [li_value(order, pol, name, gap) for gap in gaps]
                        row = dict(kind='contact', material=name, degree=args.degree, order=order, refinement=refinement, polarization=pol,
                                   gaps=gaps, T=values[pol], R=reflections[pol], li_T=li,
                                   elimination=metadata['elimination'], diagnostics=diagnostics,
                                   maximum_constitutive_gap_derivative=derivative_maximum,
                                   seconds=time.perf_counter()-start)
                        if len(gaps) > 1:
                            slopes = lambda v: [(-3*v[0]+4*v[2]-v[3])/.002, (-3*v[0]+4*v[1]-v[2])/.001]
                            row.update(contact_slopes=slopes(values[pol]), li_contact_slopes=slopes(li))
                        emit(row)


if __name__ == '__main__':
    main()
