"""Measure spectral vmap Jacobian memory, or its independent scalar-call control.

Run each measurement in a fresh process; compare JSON outputs at identical inputs.
The default is 30 s-polarized wavelengths, not the fitting study's 15s + 15p.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from studies.rcwa.nvf_fitting_grid.nvf_fitting_grid import TRUE, problem  # noqa: E402
from wavebit import rcwa  # noqa: E402
from wavebit.rcwa._engine import integrated_normal  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lanes', type=int, default=30)
    parser.add_argument('--grid', type=int, default=128)
    parser.add_argument('--execution', choices=('vmap', 'scalar'), default='vmap')
    parser.add_argument('--mixed', action='store_true', help='equal s/p wavelength pairs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.lanes < 1 or args.grid < 1 or (args.mixed and args.lanes % 2):
        parser.error('positive lanes/grid required; mixed polarization needs even lanes')
    if args.output.exists():
        parser.error('output already exists; preserve the earlier measurement')
    wavelengths = np.linspace(800., 1200., args.lanes // 2 if args.mixed else args.lanes)
    jones = np.repeat(np.eye(2), len(wavelengths), axis=0) if args.mixed else np.tile([1., 0.], (args.lanes, 1))
    if args.mixed:
        wavelengths = np.tile(wavelengths, 2)
    wavelengths, jones = jnp.asarray(wavelengths), jnp.asarray(jones)
    q = jnp.asarray(TRUE)
    prepared = rcwa.prepare(problem(TRUE, 800., (1., 0.)), backend='jax',
                            numerics=rcwa.Numerics(orders=(4, 4), shape_nvf_grid_size=args.grid,
                                                   topology_check='host'))
    prepared.report.raise_if_invalid()

    def lane(q, wavelength, jones):
        result = prepared.run(problem(q, wavelength, (jones[0], jones[1])))
        value = jnp.real(result.reflection.order(0, 0).efficiency)[0]
        valid = jnp.all(jnp.stack([jnp.all(v) for v in result.validity.checks.values()]))
        return value, valid

    def values(q):
        return jax.vmap(lambda wavelength, jo: lane(q, wavelength, jo))(wavelengths, jones)

    # Keep the measured vmap graph Jacobian-only, as in the original reproducer.
    derivative = (jax.jit(jax.jacfwd(lambda q: values(q)[0])) if args.execution == 'vmap'
                  else jax.jit(jax.jacfwd(lambda q, wl, jo: lane(q, wl, jo)[0])))
    arguments = (q,) if args.execution == 'vmap' else (q, wavelengths[0], jones[0])
    start = time.monotonic()
    compiled = derivative.lower(*arguments).compile()
    record = dict(completed=False, execution=args.execution, value_execution='scalar', grid=args.grid,
                  integral_source_sha256=hashlib.sha256(Path(integrated_normal.__file__).read_bytes()).hexdigest(),
                  wavelengths=np.asarray(wavelengths).tolist(), jones=np.asarray(jones).tolist(),
                  parameters=TRUE.tolist(), orders=[4, 4],
                  versions=dict(python=platform.python_version(), jax=jax.__version__, numpy=np.__version__),
                  compile_seconds=time.monotonic()-start,
                  compiled_temporary_bytes=compiled.memory_analysis().temp_size_in_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write():
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    write()
    print('compiled temporary bytes:', record['compiled_temporary_bytes'], flush=True)
    start = time.monotonic()
    if args.execution == 'vmap':
        jacobian = np.asarray(compiled(q))
    else:
        jacobian = np.stack([np.asarray(compiled(q, wl, jo)) for wl, jo in zip(wavelengths, jones)])
    record['run_seconds'] = time.monotonic()-start
    # Independent scalar values retain validity checks without enlarging the
    # measured Jacobian graph. These are excluded from run_seconds and peak RSS.
    try:
        import resource
        record['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    except ImportError:
        record['peak_rss_bytes'] = None  # resource is unavailable on Windows.
    scalar = jax.jit(lane)
    pairs = [jax.tree.map(np.asarray, scalar(q, wl, jo)) for wl, jo in zip(wavelengths, jones)]
    spectrum, valid = np.array([v for v, _ in pairs]), np.array([v for _, v in pairs])
    if not valid.all() or not np.isfinite(spectrum).all() or not np.isfinite(jacobian).all():
        raise RuntimeError('invalid value or Jacobian; incomplete record retained')
    record.update(completed=True, jacobian=jacobian.tolist(), values=spectrum.tolist(), valid=valid.tolist())
    write()
    print('completed', args.execution, args.lanes, flush=True)


if __name__ == '__main__':
    main()
