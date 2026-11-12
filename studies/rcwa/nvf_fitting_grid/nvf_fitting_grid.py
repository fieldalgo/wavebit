"""Measure normal-grid bias in a four-parameter overlapping-shape spectral fit.

Run from the source checkout with wavebit[jax]. The target is synthetic at grid
128, NOT an independent optical reference. The fitted dimensions need not tend
to its generating dimensions when the model grid is refined. Fourier order and
ellipse polygonization stay fixed. See the accompanying validation document.
"""
import argparse
from dataclasses import asdict
import json
import os
import tempfile
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.rcwa.spectral_fit_grid import fit_grid_ladder  # noqa: E402
from wavebit import Material, Units, rcwa  # noqa: E402

jax.config.update('jax_enable_x64', True)
TRUE = np.array([420., .3, 90., 300.])
WAVELENGTHS = np.linspace(800., 1200., 15)


def problem(p, wavelength, jones):
    width, angle, center_x, thickness = p
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), top=Material(epsilon=1.),
            bottom=Material(epsilon=2.25), layers=(rcwa.Layer(thickness=thickness,
                background=Material(epsilon=1.), regions=(
                    rcwa.Region(geometry=rcwa.Rectangle(size=(width, 210.),
                        center=(-60., 0.), angle=angle), material=Material(epsilon=12.25)),
                    rcwa.Region(geometry=rcwa.Ellipse(radii=(160., 100.),
                        center=(center_x, 40.), angle=-.4), material=Material(epsilon=4.)))),)),
        excitation=rcwa.Excitation(wavelength=wavelength,
            sources=(rcwa.PlaneWave(theta=.2, phi=.3, polarization=jones),)))


class Model:
    """Host-validated prepared reuse; JIT values and forward Jacobian together."""
    def __init__(self, grid, batch_size=5):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.grid = grid
        self.batch_size = batch_size
        self.prepared = self.compiled = None
        self.key = self.cached = None
        self.evaluations = 0

    def __call__(self, p):
        key = tuple(float(x) for x in p)
        if self.key == key:
            return self.cached
        start = time.monotonic()
        concrete = problem(key, 800., (1., 0.))
        if self.prepared is None or not self.prepared.validate(concrete).ok:
            prepared = rcwa.prepare(concrete, numerics=rcwa.Numerics(
                orders=(4, 4), shape_nvf_grid_size=self.grid, topology_check='host'), backend='jax')
            prepared.report.raise_if_invalid()

            def lane(q, wavelength, jones):
                result = prepared.run(problem(q, wavelength, (jones[0], jones[1])))
                flags = jnp.stack([jnp.all(v) for v in result.validity.checks.values()])
                return jnp.real(result.reflection.order(0, 0).efficiency)[0], jnp.all(flags)

            def values(q, wavelengths, jones):
                y, valid = jax.vmap(lambda wl, pol: lane(q, wl, pol))(wavelengths, jones)
                return y, (y, valid)

            self.prepared = prepared
            self.compiled = jax.jit(jax.jacfwd(values, has_aux=True))
        # Bound the simultaneous optical batch without changing any observation.
        # Consume each result before dispatching the next chunk to bound memory.
        wavelengths = np.tile(WAVELENGTHS, 2)
        polarizations = np.repeat(np.eye(2), len(WAVELENGTHS), axis=0)
        values, derivatives, validity = [], [], []
        for offset in range(0, len(wavelengths), self.batch_size):
            jac, (y, valid) = self.compiled(jnp.asarray(p),
                jnp.asarray(wavelengths[offset:offset+self.batch_size]),
                jnp.asarray(polarizations[offset:offset+self.batch_size]))
            values.append(np.asarray(y))
            derivatives.append(np.asarray(jac))
            validity.append(np.asarray(valid))
        y, jac = np.concatenate(values), np.concatenate(derivatives)
        if not all(np.all(v) for v in validity) or not np.isfinite(y).all() or not np.isfinite(jac).all():
            raise RuntimeError(f'invalid solve at grid {self.grid}, parameters {key}')
        self.key, self.cached = key, (y, jac)
        self.evaluations += 1
        print(f'grid {self.grid}: evaluation {self.evaluations}, '
              f'{time.monotonic()-start:.1f} s, p={key}', flush=True)
        return self.cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grids', type=int, nargs='+', default=[128, 256, 512, 1024])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=5,
                        help='maximum simultaneous spectral lanes (default: 5)')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.output.exists():
        parser.error('output already exists; preserve the earlier measurement')
    target_model = Model(128, batch_size=args.batch_size)
    data, target_jac = target_model(TRUE)
    models = {128: target_model}
    del target_model

    def factory(grid):
        if grid not in models:
            models[grid] = Model(grid, batch_size=args.batch_size)
        # Only adjacent grids are needed by the convergence check.
        for old in list(models):
            if old < grid and old != max(g for g in models if g < grid):
                del models[old]
        return models[grid]

    completed_stages = []

    def write_record(converged, reason, completed):
        stages = []
        for stage in completed_stages:
            row = asdict(stage)
            if not np.isfinite(stage.noise_std).all():
                row['noise_std'] = None
            stages.append(row)
        record = dict(completed=completed, converged=converged, reason=reason,
                      data_grid=128, lane_batch_size=args.batch_size, orders=[4, 4],
                      precision='double', data=data.tolist(), target_parameters=TRUE.tolist(),
                      target_jacobian=target_jac.tolist(), stages=stages,
                      versions=dict(jax=jax.__version__, numpy=np.__version__))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=args.output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, indent=2, allow_nan=False, default=lambda x: x.tolist())
            handle.write('\n')
        try:
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)

    def checkpoint(stage):
        completed_stages.append(stage)
        write_record(False, 'in progress; no convergence decision yet', False)

    report = fit_grid_ladder(factory, TRUE, data, grids=args.grids,
                            noise_sigma=1e-4, parameter_tolerance=[.02, 4e-5, .015, .004],
                            spectrum_tolerance=1e-5, on_stage=checkpoint)
    write_record(report.converged, report.reason, True)
    print(f'grid converged={report.converged}: {report.reason}')
    if not report.converged:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
