"""Isolated NumPy weak-block assembly cost on actual finite-gap inputs.

Legacy and current contractions run in fresh child processes with the same
inputs. This excludes planning, Fourier integration, cell solves and optics.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
from wavebit.rcwa._engine.conforming import cartesian_pair_gap_autodiff as implementation
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan


NAMES = ("x", "x1", "y", "factor", "thin_factor", "thin", "s")


def legacy(x, x1, y, factor, thin_factor, thin, s):
    """The former element-wise outer products, retained only as a cost control."""
    def product(a, b):
        return np.einsum("eac,ead->ecd", a, b)
    def blend(a, b):
        return np.sum(np.where(thin[:, None, None], b, a), axis=0)
    return (
        blend(factor[:, None, None]*product(x, x), (thin_factor*s)[:, None, None]*product(x1, x1)),
        blend(factor[:, None, None]*product(x, y), thin_factor[:, None, None]*product(x1, y)),
        blend((factor*s)[:, None, None]*product(y, y), thin_factor[:, None, None]*product(y, y)),
    )


def captured_inputs(refinement):
    geometry = dict(basis=np.diag([700., 700.]), widths=(300., 300.), height=300.)
    plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=refinement)
    cell = implementation.CartesianPairGapEvaluator(plan, order=0, coercivity_angle=1.53)
    original, captured = implementation._gap_weak_blocks, {}
    def observe(xp, *args):
        captured.update(zip(NAMES, args))
        return original(xp, *args)
    try:
        implementation._gap_weak_blocks = observe
        cell.evaluate(gap=7., epsilon=(12.25, -10.+1j), **geometry)
    finally:
        implementation._gap_weak_blocks = original
    return captured


def child(args):
    with np.load(args.input) as archive:
        values = [archive[name] for name in NAMES]
    calculate = legacy if args.kernel == "legacy" else lambda *v: implementation._gap_weak_blocks(np, *v)
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        result = calculate(*values)
        timings.append(time.perf_counter()-start)
    np.savez(args.output, **dict(zip(("H00", "H01", "H11"), result)))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(kernel=args.kernel, seconds=timings,
                          median_seconds=float(np.median(timings)),
                          peak_process_rss_bytes=int(rss if sys.platform == "darwin" else rss*1024))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refinements", type=int, nargs="+", choices=[1, 2], default=[1, 2])
    parser.add_argument("--out", type=Path, default=repository_path('studies/rcwa/nvf_pair_assembly_cost/records/nvf_pair_assembly_cost.log'))
    parser.add_argument("--kernel", choices=["legacy", "contracted"], help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.kernel:
        child(args)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink, tempfile.TemporaryDirectory(prefix="nvf-assembly-") as scratch:
        def emit(row):
            line = json.dumps(row)
            sink.write(line+"\n")
            sink.flush()
            print(line, flush=True)
        emit(dict(kind="header", numpy=np.__version__, repeats=5,
                  source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in (implementation.__file__, __file__)},
                  scope="NumPy weak-block assembly only; fresh child process RSS includes runtime/input overhead"))
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        for refinement in args.refinements:
            inputs = captured_inputs(refinement)
            input_path = Path(scratch)/"input.npz"
            np.savez(input_path, **inputs)
            records, matrices = {}, {}
            for kernel in ("legacy", "contracted"):
                output = Path(scratch)/(kernel+".npz")
                run = subprocess.run([sys.executable, __file__, "--kernel", kernel,
                                      "--input", str(input_path), "--output", str(output)],
                                     env=env, capture_output=True, text=True, check=True, timeout=60)
                records[kernel] = json.loads(run.stdout)
                with np.load(output) as archive:
                    matrices[kernel] = [archive[name] for name in ("H00", "H01", "H11")]
            differences = [float(np.max(np.abs(a-b))/max(1., np.max(np.abs(b))))
                           for a, b in zip(matrices["contracted"], matrices["legacy"])]
            if max(differences) > 3e-13:
                raise ValueError("assembly representation comparison failed")
            emit(dict(kind="cost", refinement=refinement, elements=len(inputs["x"]),
                      contact_unknowns=inputs["x"].shape[-1], difference_unknowns=inputs["y"].shape[-1],
                      gap=7., records=records, normalized_maximum_differences=differences))


if __name__ == "__main__":
    main()
