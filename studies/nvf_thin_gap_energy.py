"""Study optical readouts over the shared conforming NVF cell implementation."""
import numpy as np
import argparse
import json
import time

from wavebit._core.conforming.thin_gap import (
    ThinGapEnergyModel as _Cell,
    PERIOD as PERIOD,
    _periodic_line as _periodic_line,
    _line_mass as _line_mass,
    _line_stiffness as _line_stiffness
)
from studies.nvf_gap_optics import GapOptics
from studies.nvf_harmonic_energy import powers


class ThinGapEnergyModel(GapOptics, _Cell):
    """The shared cell with the original fixed-fixture optical study methods."""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order', type=int, default=7)
    parser.add_argument('--rows', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--edge-fraction', type=float, default=1e-5)
    parser.add_argument('--quadrature', type=int, default=12)
    parser.add_argument('--uniform-y', action='store_true')
    parser.add_argument('--autodiff', action='store_true', help='also check JAX forward and parameter derivatives')
    parser.add_argument('--gaps', type=float, nargs='+', default=[0., 1e-4, 1e-3])
    args = parser.parse_args()
    if args.autodiff:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
    for rows in args.rows:
        model = ThinGapEnergyModel(args.order, rows, args.edge_fraction, args.quadrature,
                                   grade_y=not args.uniform_y)
        if args.autodiff:
            functions = {pol: jax.jit(jax.value_and_grad(
                lambda p, pol=pol: model.transmission(p[0], p[1], p[2], jnp, polarization=pol)['T']))
                for pol in ('p', 's')}
        for gap in args.gaps:
            start = time.monotonic()
            slots, metadata = model.matrices(gap)
            results = {}
            for polarization in ('p', 's'):
                t, r = powers(args.order, slots, polarization=polarization)
                results[polarization] = dict(T=float(t), R=float(r), energy_error=float(abs(t+r-1)))
                if args.autodiff:
                    value, derivative = functions[polarization](jnp.array([gap, 12.25, 350.]))
                    if not np.isfinite(np.asarray([value, *derivative])).all():
                        raise FloatingPointError('nonfinite forward or derivative is not comparison data')
                    results[polarization].update(jax_T=float(value),
                                                  derivative_gap_epsilon_width=np.asarray(derivative).tolist())
            print(json.dumps(dict(kind='thin_gap_energy', order=args.order, rows=rows,
                                  edge_fraction=args.edge_fraction, quadrature=args.quadrature,
                                  grade_y=not args.uniform_y, body_rows=model.rows,
                                  gap=gap, **metadata, **results,
                                  seconds=time.monotonic()-start)), flush=True)
        if args.autodiff:
            jax.clear_caches()


if __name__ == '__main__':
    main()
