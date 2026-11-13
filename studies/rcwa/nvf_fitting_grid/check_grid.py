"""Check N5 spectra/Jacobians at fixed parameters before interpreting a fit.

Each command writes one grid's values and Jacobian. Optionally compare against a
previous grid and Richardson-extrapolated central differences. The fixed-point
checks are separate from the two consecutive fitted-parameter checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import jax
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from studies.rcwa.nvf_fitting_grid.nvf_fitting_grid import Model, TRUE  # noqa: E402


def relative_columns(candidate, reference):
    """Column maximum errors; None denotes a changed zero reference column."""
    differences = np.max(np.abs(candidate - reference), axis=0)
    scales = np.max(np.abs(reference), axis=0)
    return [float(d / s) if s > 0 else 0.0 if d == 0 else None
            for d, s in zip(differences, scales)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--solver-revision', required=True,
                        help='revision of the solver actually imported for this measurement')
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--fd', action='store_true')
    parser.add_argument('--batch-size', type=int, default=5)
    args = parser.parse_args()
    if args.grid <= 0 or args.batch_size <= 0:
        parser.error('grid and batch-size must be positive')
    if args.output.exists():
        parser.error('output already exists; preserve the earlier measurement')
    previous = None
    if args.previous:
        previous = json.loads(args.previous.read_text(encoding='utf-8'))
        if (not previous.get('completed') or previous['grid'] >= args.grid
                or previous['solver_revision'] != args.solver_revision
                or previous['parameters'] != TRUE.tolist() or previous['orders'] != [4, 4]):
            parser.error('previous must be a completed lower grid of the same solver and case')
    record = dict(completed=False, solver_revision=args.solver_revision,
                  instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  grid=args.grid, parameters=TRUE.tolist(), orders=[4, 4],
                  batch_size=args.batch_size, execution='bounded_scalar_calls',
                  versions=dict(jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__),
                  spectrum_tolerance=1e-5, jacobian_relative_tolerance=1e-3,
                  fd_criterion=1e-3, columns=[], checks={})

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=args.output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write('\n')
        try:
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)

    start = time.monotonic()
    model = Model(args.grid, batch_size=args.batch_size)
    values, jacobian = model(TRUE)
    record.update(values=values.tolist(), jacobian=jacobian.tolist(),
                  ad_seconds=time.monotonic() - start)
    if previous:
        difference = float(np.max(np.abs(values - np.asarray(previous['values']))))
        relative = relative_columns(np.asarray(previous['jacobian']), jacobian)
        record['previous_grid'] = previous['grid']
        record['spectrum_max'] = difference
        record['jacobian_column_relative'] = relative
        record['checks'].update(adjacent_spectrum=difference <= 1e-5,
            adjacent_jacobian=all(v is not None and v <= 1e-3 for v in relative))
    save()
    if args.fd:
        for axis, h in enumerate([.01, .0001, .01, .01]):
            central = []
            for step in [h, h / 2]:
                delta = np.zeros(4)
                delta[axis] = step
                central.append((model(TRUE + delta, derivative=False)
                                - model(TRUE - delta, derivative=False)) / (2 * step))
            richardson = (4 * central[1] - central[0]) / 3
            relative = relative_columns(richardson[:, None], jacobian[:, axis:axis+1])[0]
            record['columns'].append(dict(axis=axis, step=h, central=central[0].tolist(),
                central_half=central[1].tolist(), richardson=richardson.tolist(),
                relative_max=relative, passed=relative is not None and relative <= 1e-3))
            save()
            print(f'FD column {axis}: relative error {relative}', flush=True)
        record['checks']['derivative_fidelity'] = all(c['passed'] for c in record['columns'])
    record.update(completed=True, total_seconds=time.monotonic() - start)
    save()
    print(record['checks'], flush=True)
    if any(not passed for passed in record['checks'].values()):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
