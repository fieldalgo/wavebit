"""Does a quadratic auxiliary potential improve metal optics at matched DOFs?"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from wavebit._core.conforming import cartesian_pair_quadratic as quadratic
from wavebit._core.conforming import triangle_moments
from wavebit._core.conforming import cartesian_pair_gap as linear
from wavebit._core.conforming import cartesian_pair_opening as opening
from wavebit._core.conforming import cartesian_pair, material_sector
from studies.nvf_pair_material_sector import GEOMETRY, MATERIALS, solver, li_value


GAPS = (0., .0005, .001, .002, 1e-9)


def field_condition(value, order):
    n, m = np.meshgrid(np.arange(-order, order+1), np.arange(-order, order+1), indexing='ij')
    dn, dm = n.ravel()[:, None]-n.ravel()+2*order, m.ravel()[:, None]-m.ravel()+2*order
    c = value[:, dn, dm]
    return float(np.linalg.cond(np.block([[c[0], c[1]], [c[2], c[3]]])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--refinements', type=int, nargs='+', choices=range(3), default=[0, 1, 2])
    parser.add_argument('--degrees', type=int, nargs='+', choices=[1, 2], default=[1, 2])
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results/nvf_pair_quadratic.log')
    args = parser.parse_args()
    if any(n < 0 for n in args.orders):
        parser.error('orders must be nonnegative')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as sink:
        def emit(row):
            line = json.dumps(row)
            sink.write(line+'\n')
            sink.flush()
            print(line, flush=True)
        sources = [quadratic.__file__, triangle_moments.__file__, linear.__file__, opening.__file__,
                   cartesian_pair.__file__, material_sector.__file__, __file__,
                   Path(__file__).with_name('nvf_pair_material_sector.py')]
        emit(dict(kind='header', numpy=np.__version__, orders=args.orders, refinements=args.refinements,
                  degrees=args.degrees, gaps=GAPS, cell=700., width=300., height=300., thickness=300.,
                  wavelength=900., theta=.2, phi=0., top=1., bot=2.25, background=1., coercivity_angle=1.53,
                  source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}))
        for order in args.orders:
            for refinement in args.refinements:
                for degree in args.degrees:
                    actual_refinement = refinement+1 if degree == 1 else refinement
                    plan = opening.CartesianPairOpeningPlan.from_geometry(**GEOMETRY, refinement=actual_refinement)
                    cls = linear.CartesianPairGapCell if degree == 1 else quadratic.CartesianPairQuadraticGapCell
                    cell = cls(plan, order, coercivity_angle=1.53)
                    for name, epsilon in MATERIALS.items():
                        start = time.perf_counter()
                        values, diagnostics = {}, {}
                        for gap in GAPS:
                            value, derivative, meta, _ = cell.coefficients(gap=gap, epsilon=epsilon, **GEOMETRY)
                            slots, _, cm = opening.energy_opening(value, derivative, cell.order, epsilon, 1.,
                                                                  phase=np.exp(-1j*1.53))
                            if gap == 0:
                                diagnostics = {key: float(meta[key]) for key in (
                                    'matrix_condition', 'original_flux_residual', 'original_flux_derivative_residual')}
                                diagnostics.update(field_condition=field_condition(value, order),
                                                   minimum_dissipation=float(cm['minimum_dissipation']))
                            for pol in ('p', 's'):
                                result = solver(order, pol).solve_conv(*slots)
                                values[pol, gap] = [float(np.sum(result.T).real), float(np.sum(result.R).real)]
                        for pol in ('p', 's'):
                            observed = [values[pol, g][0] for g in GAPS]
                            reference = [li_value(order, pol, name, g) for g in GAPS]
                            slopes = lambda v: [(-3*v[0]+4*v[2]-v[3])/.002, (-3*v[0]+4*v[1]-v[2])/.001]
                            emit(dict(kind='contact', degree=degree, matched_level=refinement,
                                      refinement=actual_refinement, order=order, material=name, polarization=pol,
                                      open_free_dofs=meta['open_free_dofs'], T=observed,
                                      R=[values[pol, g][1] for g in GAPS], li_T=reference,
                                      contact_slopes=slopes(observed), li_contact_slopes=slopes(reference),
                                      diagnostics=diagnostics, seconds=time.perf_counter()-start))


if __name__ == '__main__':
    main()
