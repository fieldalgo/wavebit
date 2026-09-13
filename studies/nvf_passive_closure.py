"""Does a passive NVF energy closure improve the metal and contact controls?

The normal field and exact material transforms are held fixed. This explicit
experiment compares finite constitutive closures, not independent Maxwell codes.
Cartesian Li is a separate factorization on the same exact rectangular geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import wavebit
from wavebit._core import parametric as P, passive_nvf as passive
from wavebit._core.factorization import _nvf_assemble
from wavebit._core.lattice import as_lattice
from studies.nvf_seam_contrast import pair, opened, SOLVE_KW, li_T

CELL = as_lattice(period=(700., 700.))


def matrices(shapes, order=4, grid=512, closure='passive'):
    """Single-layer slots, preserving the same field on both closures."""
    eps, reciprocal = P.assemble_eps_coeffs(np, shapes, 1., order, order, CELL, np.complex128)
    projectors = P.normal_projector_coeffs(np, shapes, (grid, grid), CELL,
                                           order, order, np.complex128)
    convert = lambda values: P.conv_matrix_2d(np, values, order, order)
    coefficients = [convert(values) for values in projectors]
    if closure not in ('symmetric', 'passive'):
        raise ValueError('closure must be symmetric or passive')
    assemble = _nvf_assemble if closure == 'symmetric' else passive.passive_nvf_assemble
    return assemble(np, convert(eps), convert(reciprocal), *coefficients)


def response(shapes, order=4, grid=512, closure='passive', spectrum=False):
    slots = matrices(shapes, order, grid, closure)
    solver = wavebit.rcwa(lattice=CELL, truncation_order=(order, order), **SOLVE_KW)
    result = solver.solve_conv(*[m[None, :, :] for m in slots])
    row = dict(T=float(np.real(np.sum(result.T))), R=float(np.real(np.sum(result.R))))
    if spectrum:
        tensor = np.block([[slots[0], -slots[3]], [-slots[4], slots[1]]])
        row['minimum_dissipation'] = float(np.linalg.eigvalsh((tensor-tensor.conj().T)/(2j)).min())
    return row


def emit(sink, row):
    sink.write(json.dumps(row)+'\n')
    sink.flush()
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--orders', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--sections', nargs='+', choices=['optical', 'contact'],
                        default=['optical', 'contact'])
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results/nvf_passive_closure.log')
    args = parser.parse_args()
    if any(order < 0 or order > 127 for order in args.orders):
        parser.error('orders must be between 0 and 127 for the fixed normal grid')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as sink:
        digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        emit(sink, dict(kind='header', numpy=np.__version__, orders=args.orders, sections=args.sections,
                        source_sha256=dict(closure=digest(passive.__file__), parametric=digest(P.__file__),
                                           instrument=digest(__file__))))
        if 'optical' in args.sections:
            for label, shapes in [('metal', pair(12.25, -10.+1j)), ('dielectric', pair(4., 9.)),
                                  ('lossy', pair(4., 4.+.5j))]:
                for order in args.orders:
                    start = time.perf_counter()
                    row = dict(kind='optical', material=label, order=order, grid=512,
                               li_T=li_T(shapes, order))
                    for closure in ('symmetric', 'passive'):
                        row[closure] = response(shapes, order, closure=closure, spectrum=True)
                    row['seconds'] = time.perf_counter()-start
                    emit(sink, row)
        if 'contact' in args.sections:
            shapes = pair(4., 4.)
            for order in args.orders:
                for grid in (128, 512):
                    closed = {method: response(shapes, order, grid, method)['T']
                              for method in ('symmetric', 'passive')}
                    reference = li_T(shapes, order)
                    for gap in (1e-5, 1e-7, 1e-9, 1e-10):
                        shape = opened(shapes, gap)
                        row = dict(kind='contact', order=order, grid=grid, gap=gap,
                                   li_delta=li_T(shape, order)-reference)
                        for method in closed:
                            row[method+'_delta'] = response(shape, order, grid, method)['T']-closed[method]
                        emit(sink, row)


if __name__ == '__main__':
    main()
