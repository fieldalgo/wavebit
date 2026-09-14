"""Can tiled normal-frame integration resolve the inner ladder at lower memory?"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

from wavebit._core import parametric, factorization
from wavebit._core.conforming import cartesian_pair_normal_frame as frame
from wavebit._core.conforming import cartesian_pair_opening, material_sector
from wavebit._core.conforming.cartesian_pair_opening import energy_opening
from studies.nvf_pair_material_sector import GEOMETRY, MATERIALS, solver


def worker(args):
    if args.worker == 'whole':
        frame._QUADRATURE_TILE_ROWS = 3*args.quadrature
    cell = frame.CartesianPairNormalFrame(**GEOMETRY, order=(args.cap, args.cap),
                                         quadrature=args.quadrature, coercivity_angle=1.53)
    start = time.perf_counter()
    value, meta = cell.coefficients(gap=0., epsilon=MATERIALS['metal'])
    seconds = time.perf_counter()-start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB. Includes imports and node setup.
    peak_bytes = rss if sys.platform == 'darwin' else rss*1024
    np.save(args.array, value)
    print(json.dumps(dict(kind='memory', quadrature=args.quadrature, mode=args.worker,
                          cap=args.cap, peak_process_bytes=peak_bytes, coefficient_seconds=seconds,
                          sample_count=meta['sample_count'], maximum_field_samples=meta['maximum_field_samples'],
                          constant_integral_error=float(meta['constant_integral_error']))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quadratures', type=int, nargs='+', default=[128, 256, 512])
    parser.add_argument('--orders', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results/nvf_pair_normal_frame_quadrature.log')
    parser.add_argument('--worker', choices=['whole', 'tiled'], help=argparse.SUPPRESS)
    parser.add_argument('--quadrature', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--cap', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--array', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if any(n < 0 for n in args.orders):
        parser.error('orders must be nonnegative')
    if any(q % 2 or q < max(8, 2*max(args.orders)+4) for q in args.quadratures):
        parser.error('quadratures must satisfy the normal-frame admission rule')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sources = [frame.__file__, parametric.__file__, factorization.__file__,
               cartesian_pair_opening.__file__, material_sector.__file__, __file__,
               Path(__file__).with_name('nvf_pair_material_sector.py')]
    with args.out.open('a') as sink, tempfile.TemporaryDirectory(prefix='nvf-frame-quadrature-') as scratch:
        def emit(row):
            line = json.dumps(row)
            sink.write(line+'\n')
            sink.flush()
            print(line, flush=True)
        emit(dict(kind='header', numpy=np.__version__, platform=platform.system(), machine=platform.machine(),
                  quadratures=args.quadratures, orders=args.orders, geometry='touching 300x300 pair in 700x700 cell',
                  epsilon=[[12.25, 0.], [-10., 1.]], background=1.,
                  source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
                  memory_scope='fresh-process peak including imports; coefficient integration only'))
        cap, previous = max(args.orders), None
        coefficients = {}
        # Finish every coefficient-only process before any optical propagation.
        for q in args.quadratures:
            arrays = {}
            for mode in ('whole', 'tiled'):
                path = Path(scratch)/f'{mode}-{q}.npy'
                command = [sys.executable, '-m', 'studies.nvf_pair_normal_frame_quadrature',
                           '--worker', mode, '--quadrature', str(q), '--cap', str(cap), '--array', str(path)]
                completed = subprocess.run(command, check=True, capture_output=True, text=True)
                row = json.loads(completed.stdout)
                arrays[mode] = np.load(path)
                if mode == 'tiled':
                    row['maximum_coefficient_difference_from_whole'] = float(np.max(abs(arrays[mode]-arrays['whole'])))
                    if previous is not None:
                        row['maximum_coefficient_change_from_previous_quadrature'] = float(np.max(abs(arrays[mode]-previous)))
                emit(row)
            previous = arrays['tiled']
            coefficients[q] = previous
        for q, value in coefficients.items():
            for n in args.orders:
                c = value[:, 2*(cap-n):2*(cap+n)+1, 2*(cap-n):2*(cap+n)+1]
                slots, _, meta = energy_opening(c, np.zeros_like(c), (n, n), MATERIALS['metal'], 1.,
                                                phase=np.exp(-1j*1.53))
                tensor = np.block([[slots.epx[0], -slots.epxy[0]], [-slots.epyx[0], slots.epy[0]]])
                reciprocity = {k: float(v) for k, v in frame._reciprocity_defects(tensor, np).items()}
                for pol in ('p', 's'):
                    result = solver(n, pol).solve_conv(*slots)
                    emit(dict(kind='optical', quadrature=q, order=n, polarization=pol,
                              T=float(np.sum(result.T).real), R=float(np.sum(result.R).real),
                              minimum_dissipation=float(meta['minimum_dissipation']), **reciprocity))


if __name__ == '__main__':
    main()
