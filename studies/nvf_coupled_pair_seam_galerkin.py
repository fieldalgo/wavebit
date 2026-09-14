"""Do continuous real pair gap traces improve metal contact optical convergence?"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from wavebit._core.conforming import (
    cartesian_pair_seam_galerkin, real_trace_coupling, periodic_seam_trace,
    cartesian_pair_galerkin, cartesian_pair_quadratic, cartesian_pair_opening,
    cartesian_pair, triangle_moments, material_sector,
)
from studies import nvf_pair_material_sector as fixture

GAPS = (0., .001, 1e-5, 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--refinements', type=int, nargs='+', choices=range(4), default=[0, 1])
    parser.add_argument('--materials', nargs='+', choices=tuple(fixture.MATERIALS), default=list(fixture.MATERIALS))
    parser.add_argument('--contact-only', action='store_true', help='record exact contact and its tensor JVP without positive-gap qualification')
    parser.add_argument('--gaps', type=float, nargs='+', help='explicit gap samples, starting with exact contact')
    parser.add_argument('--quadrature', type=int, default=32)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results/nvf_coupled_pair_seam_galerkin.log')
    args = parser.parse_args()
    if any(n < 0 for n in args.orders):
        parser.error('orders must be nonnegative')
    if args.gaps is not None and (args.contact_only or args.gaps[0] != 0
                                  or any(not np.isfinite(g) or g < 0 for g in args.gaps)):
        parser.error('explicit gaps must start at zero, be finite and nonnegative, and exclude --contact-only')
    gaps = (0.,) if args.contact_only else (tuple(args.gaps) if args.gaps is not None else GAPS)
    modules = (cartesian_pair_seam_galerkin, real_trace_coupling, periodic_seam_trace,
               cartesian_pair_galerkin, cartesian_pair_quadratic, cartesian_pair_opening,
               cartesian_pair, triangle_moments, material_sector, fixture)
    sources = [Path(m.__file__) for m in modules]+[Path(__file__)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with args.out.open('a') as sink:
        def emit(row):
            line = json.dumps(row)
            sink.write(line+'\n'); sink.flush(); print(line, flush=True)
        emit(dict(kind='header', numpy=np.__version__, orders=args.orders, refinements=args.refinements,
                  materials=args.materials, quadrature=args.quadrature, gaps=gaps,
                  period=[700.,700.], widths=[300.,300.], height=300., background=1.,
                  wavelength=900., thickness=300., theta=.2, phi=0., top=1., bot=2.25,
                  coercivity_angle=1.53,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        for n in args.orders:
            for refinement in args.refinements:
                plan = cartesian_pair_opening.CartesianPairOpeningPlan.from_geometry(
                    **fixture.GEOMETRY, refinement=refinement)
                obj = cartesian_pair_seam_galerkin.CartesianPairSeamGalerkinGapCell(
                    plan, (n,n), quadrature=args.quadrature, coercivity_angle=1.53)
                for name in args.materials:
                    start = time.perf_counter()
                    totals = {pol: [] for pol in ('p','s')}
                    reflections = {pol: [] for pol in ('p','s')}
                    diagnostics = []
                    def power(pol, slots):
                        output = fixture.solver(n,pol).solve_conv(*slots)
                        t, r = float(output.T.real.sum()), float(output.R.real.sum())
                        if not np.isfinite(t+r) or min(t,r) < -1e-9 or t+r > 1+1e-9:
                            raise ValueError('nonfinite or active optical output')
                        if name == 'equal' and abs(t+r-1) > 1e-9:
                            raise ValueError('lossless physical power balance failed')
                        return t,r
                    try:
                        for gap in gaps:
                            value, tangent, metadata = obj.closure(
                                gap=gap, epsilon=fixture.MATERIALS[name], **fixture.GEOMETRY)
                            diagnostics.append({k: float(v) if v is not None else None for k,v in metadata.items()
                                                if k.endswith('residual') or k in ('condition','reciprocity',
                                                    'derivative_reciprocity','minimum_rotated_storage','minimum_dissipation')})
                            if gap == 0:
                                closed, derivative = np.asarray(value), np.asarray(tangent)
                            for pol in ('p','s'):
                                t,r = power(pol,value)
                                totals[pol].append(t); reflections[pol].append(r)
                        for pol in ('p','s'):
                            comparison = [fixture.li_value(n,pol,name,gap) for gap in gaps]
                            slopes = [(power(pol,closed+h*derivative)[0]-power(pol,closed-h*derivative)[0])/(2*h)
                                      for h in (.00005,.000025)]
                            emit(dict(kind='contact', order=n, refinement=refinement, material=name,
                                      polarization=pol, gaps=gaps, T=totals[pol], R=reflections[pol],
                                      li_T=comparison, tangent_gap_slopes=slopes,
                                      gap_secants=[(v-totals[pol][0])/g for g,v in zip(gaps[1:3],totals[pol][1:3])],
                                      diagnostics=diagnostics, seconds=time.perf_counter()-start))
                    except ValueError as error:
                        failed = True
                        emit(dict(kind='failure', order=n, refinement=refinement, material=name,
                                  gap=gap, error=str(error), completed_gap_diagnostics=diagnostics,
                                  seconds=time.perf_counter()-start))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
