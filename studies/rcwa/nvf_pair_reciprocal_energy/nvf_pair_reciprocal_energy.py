"""Does conjugate-paired energy restore reciprocity and improve metal optics?"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wavebit.rcwa._engine import parametric, factorization
from wavebit.rcwa._engine.conforming import cartesian_pair_normal_frame as frame
from wavebit.rcwa._engine.conforming import material_sector
from studies.rcwa.nvf_pair_material_sector.nvf_pair_material_sector import GEOMETRY, MATERIALS, solver, li_value


GAPS = (0., .0005, .001, .002, 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[4, 8, 12, 16])
    parser.add_argument('--quadratures', type=int, nargs='+', default=[128, 256])
    parser.add_argument('--derivative-orders', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--out', type=Path, default=repository_path('studies/rcwa/nvf_pair_reciprocal_energy/records/nvf_pair_reciprocal_energy.log'))
    args = parser.parse_args()
    if any(n < 0 for n in args.orders+args.derivative_orders):
        parser.error('orders must be nonnegative')
    if any(q % 2 or q < max(8, 2*max(args.orders)+4) for q in args.quadratures):
        parser.error('quadratures must satisfy normal-frame admission')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sources = [frame.__file__, parametric.__file__, factorization.__file__, material_sector.__file__,
               __file__, repository_path('studies/rcwa/nvf_pair_material_sector/nvf_pair_material_sector.py')]
    with args.out.open('a') as sink:
        def emit(row):
            line = json.dumps(row)
            sink.write(line+'\n')
            sink.flush()
            print(line, flush=True)
        emit(dict(kind='header', numpy=np.__version__, orders=args.orders, quadratures=args.quadratures,
                  derivative_orders=args.derivative_orders, gaps_at_maximum_quadrature=GAPS,
                  cell=700., width=300., height=300., thickness=300., wavelength=900.,
                  theta=.2, phi=0., top=1., bot=2.25, background=1., coercivity_angle=1.53,
                  source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}))
        for n in args.orders:
            for q in args.quadratures:
                cell = frame.CartesianPairNormalFrame(**GEOMETRY, order=(n, n), quadrature=q, coercivity_angle=1.53)
                gaps = GAPS if q == max(args.quadratures) and n in args.derivative_orders else (0.,)
                for material, epsilon in MATERIALS.items():
                    observed, reflected, diagnostics = {}, {}, {}
                    for gap in gaps:
                        _, raw_slots, raw = cell.evaluate(gap=gap, epsilon=epsilon)
                        paired_slots, paired = cell.paired_closure(gap=gap, epsilon=epsilon)
                        for name, slots, meta in [('raw', raw_slots, raw), ('paired', paired_slots, paired)]:
                            if gap == 0:
                                diagnostics[name] = {k: float(meta[k]) for k in (
                                    'reciprocity_error', 'relative_reciprocity_defect',
                                    'minimum_dissipation', 'minimum_rotated_storage')}
                                if name == 'paired':
                                    diagnostics[name].update(maximum_pairing_change=float(meta['maximum_pairing_change']),
                                                             relative_pairing_change=float(meta['relative_pairing_change']),
                                                             raw_inverse_residual=float(meta['raw']['inverse_residual']))
                            for pol in ('p', 's'):
                                result = solver(n, pol)._solve_conv(*slots)
                                t, r = float(np.sum(result.T).real), float(np.sum(result.R).real)
                                if not np.isfinite(t+r) or t+r > 1+1e-9:
                                    raise RuntimeError('nonfinite or active optical output is not comparison data')
                                if material == 'equal' and abs(t+r-1) > 1e-9:
                                    raise RuntimeError('lossless power balance failed')
                                observed[name, pol, gap], reflected[name, pol, gap] = t, r
                    for pol in ('p', 's'):
                        reference = [li_value(n, pol, material, g) for g in gaps]
                        for name in ('raw', 'paired'):
                            values = [observed[name, pol, g] for g in gaps]
                            row = dict(kind='contact', closure=name, order=n, quadrature=q, material=material,
                                       polarization=pol, gaps=gaps, T=values,
                                       R=[reflected[name, pol, g] for g in gaps], li_T=reference,
                                       diagnostics=diagnostics[name])
                            if len(gaps) > 1:
                                slopes = lambda v: [(-3*v[0]+4*v[2]-v[3])/.002,
                                                    (-3*v[0]+4*v[1]-v[2])/.001]
                                row.update(contact_slopes=slopes(values), li_contact_slopes=slopes(reference))
                            emit(row)


if __name__ == '__main__':
    main()
