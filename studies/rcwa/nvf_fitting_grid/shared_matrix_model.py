"""Share this nondispersive N5 layer's matrices across spectral observations.

This is a study-specific execution path, not a general dispersive-material cache.
Matrices are rebuilt at every parameter vector, inside the differentiated graph.
The scalar Prepared.run model remains an independent execution-path control.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from wavebit import rcwa
from wavebit.rcwa._lifecycle import _runtime_checks
from wavebit.rcwa.results import pack

from studies.rcwa.nvf_fitting_grid.nvf_fitting_grid import WAVELENGTHS, problem


class SharedMatrixModel:
    """The fixed N5 case, with one matrix build per complete spectrum.

    All materials in ``problem`` are constant, scalar and nonmagnetic. Thus the
    layer matrices depend on geometry, but neither wavelength nor polarization.
    Share only within a call; geometry derivatives still pass through that build.
    ``eps_conv_matrices`` and ``solve_conv`` use the same Fourier origin. The
    prepared contract, topology checks and result validity remain in the path.
    Lifecycle internals here bind this repository study to its recorded revision;
    this class does not extend the installed fitting or preparation API.
    """

    def __init__(self, grid, *, wavelengths=None, sequential_jvp=False):
        self.grid = grid
        self.sequential_jvp = sequential_jvp
        self.wavelengths = np.asarray(
            WAVELENGTHS if wavelengths is None else wavelengths, dtype=float).copy()
        if (self.wavelengths.ndim != 1 or not self.wavelengths.size
                or not np.isfinite(self.wavelengths).all() or np.any(self.wavelengths <= 0)):
            raise ValueError('wavelengths must be a finite positive vector')
        self.wavelengths.flags.writeable = False
        self.prepared = None
        self.key = self.cached = None
        self.evaluations = 0

    def _prepare(self, p):
        concrete = problem(p, float(self.wavelengths[0]), (1., 0.))
        if self.prepared is not None and self.prepared.validate(concrete).ok:
            return
        prepared = rcwa.prepare(concrete, numerics=rcwa.Numerics(
            orders=(4, 4), shape_nvf_grid_size=self.grid, topology_check='host'), backend='jax')
        prepared.report.raise_if_invalid()
        wavelengths = jnp.asarray(np.tile(self.wavelengths, 2))
        polarizations = jnp.asarray(np.repeat(np.eye(2), len(self.wavelengths), axis=0))
        first = float(self.wavelengths[0])

        def values(q):
            with prepared.execution._scope():
                geometry_solver = prepared._solvers(problem(q, first, (1., 0.)))[0]
                matrices = geometry_solver.eps_conv_matrices(geometry=prepared._geometry[0])

                def lane(wavelength, jones):
                    live = problem(q, wavelength, (jones[0], jones[1]))
                    solver = prepared._solvers(live)[0]
                    checks = _runtime_checks(live, prepared.execution, prepared._partitions)
                    checks.update(prepared._topology_checks(live))
                    solved = solver.solve_conv(*matrices)
                    result = pack(live, prepared.numerics, prepared.execution,
                                  [(solver, solved)], checks, prepared.report)
                    flags = jnp.stack([jnp.all(v) for v in result.validity.checks.values()])
                    return jnp.real(result.reflection.order(0, 0).efficiency)[0], jnp.all(flags)

                # Only the optical solves are batched. The normal-field integral
                # and its tangents were built once above, outside this vmap.
                y, valid = jax.vmap(lane)(wavelengths, polarizations)
            return y, (y, valid)

        self.prepared = prepared
        if self.sequential_jvp:
            def directional(q, tangent):
                _, derivative, auxiliary = jax.jvp(values, (q,), (tangent,), has_aux=True)
                return derivative, auxiliary
            self.compiled = jax.jit(directional)
        else:
            self.compiled = jax.jit(jax.jacfwd(values, has_aux=True))
        self.compiled_values = jax.jit(values)

    def __call__(self, p, *, derivative=True):
        key = tuple(float(x) for x in p)
        if self.key == key:
            return self.cached if derivative else self.cached[0]
        start = time.monotonic()
        self._prepare(key)
        if derivative and self.sequential_jvp:
            columns, flags = [], []
            for axis, tangent in enumerate(np.eye(len(key))):
                column, (y, valid) = self.compiled(jnp.asarray(p), jnp.asarray(tangent))
                # Consume before dispatching another direction, bounding live
                # normal-field and FFT tangent arrays to a single direction.
                columns.append(np.asarray(column))
                flags.append(np.asarray(valid))
                print(f'grid {self.grid}: completed Jacobian column {axis + 1}/{len(key)}',
                      flush=True)
            jac, valid = np.stack(columns, axis=1), np.stack(flags)
        elif derivative:
            jac, (y, valid) = self.compiled(jnp.asarray(p))
            jac = np.asarray(jac)
        else:
            y, (_, valid) = self.compiled_values(jnp.asarray(p))
            jac = None
        y, valid = np.asarray(y), np.asarray(valid)
        if (not valid.all() or not np.isfinite(y).all()
                or (derivative and not np.isfinite(jac).all())):
            raise RuntimeError(f'invalid solve at grid {self.grid}, parameters {key}')
        if derivative:
            self.key, self.cached = key, (y, jac)
        self.evaluations += 1
        print(f'grid {self.grid}: shared-matrix evaluation {self.evaluations}, '
              f'{time.monotonic()-start:.1f} s, p={key}', flush=True)
        return (y, jac) if derivative else y
