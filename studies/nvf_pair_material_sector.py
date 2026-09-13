"""Metal admission and contact behavior of the exact finite pair cell.

Compare the same physical rectangular pair using an independent Cartesian Li
factorization. Smooth contact values do not certify patterned optical accuracy.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import wavebit
from wavebit._core.conforming import cartesian_pair_gap as gap_module
from wavebit._core.conforming import cartesian_pair_opening as opening_module
from wavebit._core.conforming import material_sector


GEOMETRY = dict(basis=np.diag([700., 700.]), widths=(300., 300.), height=300.)
MATERIALS = {"equal": (4., 4.), "metal": (12.25, -10.+1j)}
GAPS = (0., .0005, .001, .002, 1e-6, 1e-9)


def solver(order, polarization, **kwargs):
    return wavebit.rcwa(
        backend=0, lattice=GEOMETRY["basis"], truncation_order=(order, order),
        wavelength=900., thickness=[300.], top=1., bot=2.25, theta=.2,
        phi=0., psi=0. if polarization == "p" else np.pi/2, **kwargs,
    )


@lru_cache(None)
def li_value(order, polarization, material, gap):
    shapes = [dict(type="rectangle", cx=sign*(150.+gap/2), cy=0.,
                   lx=300., ly=300., material=epsilon)
              for sign, epsilon in zip((-1., 1.), MATERIALS[material])]
    result = solver(order, polarization, factorization="li",
                    structure=[dict(background=1., shapes=shapes)]).solve()
    return float(np.sum(result.T).real)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent/"results/nvf_pair_material_sector.log")
    args = parser.parse_args()
    if any(order < 0 for order in args.orders):
        parser.error("orders must be nonnegative")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        def emit(row):
            line = json.dumps(row)
            sink.write(line+"\n")
            sink.flush()
            print(line, flush=True)

        sources = [gap_module.__file__, opening_module.__file__, material_sector.__file__, __file__]
        emit(dict(kind="header", numpy=np.__version__, orders=args.orders,
                  source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
                  coercivity_angle=1.53, gaps=GAPS,
                  cell=700., width=300., height=300., thickness=300., wavelength=900.,
                  theta=.2, phi=0., top=1., bot=2.25, background=1.))
        for order in args.orders:
            for refinement in (0, 1):
                plan = opening_module.CartesianPairOpeningPlan.from_geometry(**GEOMETRY, refinement=refinement)
                cell = gap_module.CartesianPairGapCell(plan, order, coercivity_angle=1.53)
                for material, epsilon in MATERIALS.items():
                    start = time.perf_counter()
                    values, diagnostics = {}, None
                    for gap in GAPS:
                        slots, _, metadata, _ = cell.closure(gap=gap, epsilon=epsilon, **GEOMETRY)
                        if gap == 0:
                            diagnostics = {name: metadata[name] for name in (
                                "matrix_condition", "minimum_storage", "minimum_rotated_storage",
                                "minimum_dissipation", "original_flux_residual",
                                "original_flux_derivative_residual",
                            )}
                        for polarization in ("p", "s"):
                            result = solver(order, polarization).solve_conv(*slots)
                            values[polarization, gap] = dict(T=float(np.sum(result.T).real), R=float(np.sum(result.R).real))
                    for polarization in ("p", "s"):
                        reference = [li_value(order, polarization, material, gap) for gap in GAPS]
                        observed = [values[polarization, gap]["T"] for gap in GAPS]
                        # Independent second-order one-sided secants at two windows.
                        slopes = lambda v: [(-3*v[0]+4*v[2]-v[3])/.002,
                                            (-3*v[0]+4*v[1]-v[2])/.001]
                        emit(dict(kind="contact", order=order, refinement=refinement,
                                  material=material, polarization=polarization,
                                  T=observed, R=[values[polarization, gap]["R"] for gap in GAPS],
                                  delta_T=[value-observed[0] for value in observed],
                                  li_T=reference, contact_slopes=slopes(observed),
                                  li_contact_slopes=slopes(reference),
                                  diagnostics=diagnostics, seconds=time.perf_counter()-start))


if __name__ == "__main__":
    main()
