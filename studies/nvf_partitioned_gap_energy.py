"""Study optical readouts over the shared conforming NVF cell implementation."""
import numpy as np
import argparse
import json

from wavebit._core.conforming.partitioned_gap import (
    PartitionedGapEnergyModel as _Cell,
    interval_moments as interval_moments,
    moment_matrices as moment_matrices,
    _project_strip as _project_strip
)
from studies.nvf_gap_optics import GapOptics
from wavebit._core.conforming.thin_gap import PERIOD


class PartitionedGapEnergyModel(GapOptics, _Cell):
    """The shared cell with the original fixed-fixture optical study methods."""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order', type=int, default=7)
    parser.add_argument('--gap', type=float, default=0.)
    parser.add_argument('--width', type=float, default=350.)
    parser.add_argument('--epsilon', type=float, default=12.25)
    parser.add_argument('--subdivisions', type=int, default=1)
    parser.add_argument('--shape-gates', action='store_true', help='use translated rectangle dictionaries')
    parser.add_argument('--edge-fraction', type=float, default=1e-5)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = PartitionedGapEnergyModel(args.order, rows=8, edge_fraction=args.edge_fraction,
                                      x_subdivisions=args.subdivisions)
    parameters = jnp.array([args.gap, args.epsilon, args.width, 31., 37., -29.])
    for pol in ('p', 's'):
        def objective(p, split):
            gap, epsilon, width, seam, cx, cy = p
            if args.shape_gates:
                if split:
                    shapes = [dict(lx=seam+width/2, ly=PERIOD-gap, epsilon=epsilon,
                                   cx=cx+(seam-width/2)/2, cy=cy),
                              dict(lx=width/2-seam, ly=PERIOD-gap, epsilon=epsilon,
                                   cx=cx+(seam+width/2)/2, cy=cy)]
                else:
                    shapes = [dict(lx=width, ly=PERIOD-gap, epsilon=epsilon, cx=cx, cy=cy)]
                return model.shape_transmission(shapes, jnp, polarization=pol)['T']
            profile = dict(seams=jnp.array([seam]), values=jnp.array([epsilon, epsilon])) if split else epsilon
            return model.transmission(gap, profile, width, jnp, polarization=pol)['T']
        merged = jax.jit(jax.value_and_grad(lambda p: objective(p, False)))(parameters)
        split = jax.jit(jax.value_and_grad(lambda p: objective(p, True)))(parameters)
        if not np.isfinite(np.r_[merged[0], merged[1], split[0], split[1]]).all():
            raise FloatingPointError('nonfinite values/derivatives are not comparison data')
        print(json.dumps(dict(kind='partitioned_gap', shape_gates=args.shape_gates,
                              polarization=pol, order=args.order, gap=args.gap, width=args.width,
                              epsilon=args.epsilon, subdivisions=args.subdivisions,
                              edge_fraction=args.edge_fraction,
                              gradient_parameters=['gap', 'epsilon', 'width', 'seam', 'cx', 'cy'],
                              merged_T=float(merged[0]), split_T=float(split[0]),
                              merged_gradient=np.asarray(merged[1]).tolist(),
                              split_gradient=np.asarray(split[1]).tolist())), flush=True)
        jax.clear_caches()


if __name__ == '__main__':
    main()
