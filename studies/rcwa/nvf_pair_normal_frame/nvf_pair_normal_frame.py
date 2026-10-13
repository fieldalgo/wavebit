"""Does a coercive material-referenced normal frame resolve contact optics?"""

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

from wavebit.rcwa._engine import parametric, factorization
from wavebit.rcwa._engine.conforming import cartesian_pair_normal_frame as normal_frame
from wavebit.rcwa._engine.conforming import material_sector
from studies.rcwa.nvf_pair_material_sector.nvf_pair_material_sector import GEOMETRY, MATERIALS, solver, li_value


GAPS = (0., .0005, .001, .002, 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--quadratures', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--out', type=Path, default=repository_path('studies/rcwa/nvf_pair_normal_frame/records/nvf_pair_normal_frame.log'))
    args = parser.parse_args()
    if any(n < 0 for n in args.orders):
        parser.error('orders must be nonnegative')
    if any(q % 2 or q < max(8, 2*max(args.orders)+4) for q in args.quadratures):
        parser.error('quadratures must be even and at least max(8, 2*max(orders)+4)')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as sink:
        def emit(row):
            line = json.dumps(row)
            sink.write(line+'\n')
            sink.flush()
            print(line, flush=True)
        sources = [normal_frame.__file__, parametric.__file__, factorization.__file__,
                   material_sector.__file__, __file__, repository_path('studies/rcwa/nvf_pair_material_sector/nvf_pair_material_sector.py')]
        emit(dict(kind='header', numpy=np.__version__, orders=args.orders, quadratures=args.quadratures,
                  gaps_at_maximum_quadrature=GAPS, cell=700., width=300., height=300., thickness=300.,
                  wavelength=900., theta=.2, phi=0., top=1., bot=2.25, background=1., coercivity_angle=1.53,
                  source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}))
        for order in args.orders:
            for q in args.quadratures:
                cell = normal_frame.CartesianPairNormalFrame(**GEOMETRY, order=(order, order),
                                                            quadrature=q, coercivity_angle=1.53)
                gaps = GAPS if q == max(args.quadratures) else (0.,)
                for material, epsilon in MATERIALS.items():
                    start = time.perf_counter()
                    observed, reflected, diagnostics = {}, {}, {}
                    for gap in gaps:
                        _, slots, meta = cell.evaluate(gap=gap, epsilon=epsilon)
                        if gap == 0:
                            diagnostics = {k: float(meta[k]) for k in (
                                'frame_condition', 'frame_minimum_rotated_storage', 'frame_storage_lower_bound',
                                'constant_integral_error', 'minimum_dissipation', 'minimum_rotated_storage',
                                'inverse_residual', 'z_inverse_residual')}
                        for pol in ('p', 's'):
                            result = solver(order, pol)._solve_conv(*slots)
                            observed[pol, gap] = float(np.sum(result.T).real)
                            reflected[pol, gap] = float(np.sum(result.R).real)
                    for pol in ('p', 's'):
                        values = [observed[pol, g] for g in gaps]
                        reference = [li_value(order, pol, material, g) for g in gaps]
                        row = dict(kind='contact', order=order, quadrature=q, material=material,
                                   polarization=pol, gaps=gaps, T=values,
                                   R=[reflected[pol, g] for g in gaps], li_T=reference,
                                   diagnostics=diagnostics, seconds=time.perf_counter()-start)
                        if len(gaps) > 1:
                            slopes = lambda v: [(-3*v[0]+4*v[2]-v[3])/.002,
                                                (-3*v[0]+4*v[1]-v[2])/.001]
                            row.update(contact_slopes=slopes(values), li_contact_slopes=slopes(reference),
                                       delta_T=[v-values[0] for v in values])
                        emit(row)


if __name__ == '__main__':
    main()
