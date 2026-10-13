"""Measure fixed-range sweeps against the identical sphere without a range."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import json
import time
import warnings

import numpy as np

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._runtime import runtime


RANGE = ((-.36, .36),)*3
POINTS = ((.22, (-.073, .017, -.009)), (.27, (.071, -.037, .023)))


def scene(radius, center, bounds=RANGE):
    return fd.Scene(
        domain=fd.Domain(size=(1.6,)*3), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        run=fd.Run(steps=24), structures=(fd.Structure(
            fd.Sphere(radius=radius, center=center, range=bounds),
            fd.Material(eps=2.25, mu=1.2)),), sources=(fd.Dipole(
                'drive', center=(-.55, .03, -.02), component='Ey',
                band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def measure():
    rows = []
    numpy_fields = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = fd.plan(scene(*POINTS[0]))
        for backend in ('numpy', 'jax', 'torch'):
            rt = runtime(backend)
            for radius, center in POINTS:
                started = time.perf_counter()
                ranged = fd.run(scene(radius, center), backend, plan=p)
                for field in ('E', 'H'):
                    rt.numpy(getattr(ranged.final, field))
                run_seconds = time.perf_counter()-started
                direct = fd.run(scene(radius, center, None), backend)
                row = dict(backend=backend, radius=radius, center=center,
                           skeleton=ranged.plan.skeleton, seconds=time.perf_counter()-started,
                           ranged_run_seconds=run_seconds,
                           volume=ranged.report.bodies[0].asked)
                for field in ('E', 'H'):
                    got = rt.numpy(getattr(ranged.final, field))
                    want = rt.numpy(getattr(direct.final, field))
                    row[field+'_absolute'] = float(np.max(abs(got-want)))
                    row[field+'_relative'] = float(np.max(abs(got-want))/np.max(abs(want)))
                    key = (radius, field)
                    if backend == 'numpy':
                        numpy_fields[key] = want
                    baseline = numpy_fields[key]
                    row[field+'_backend_relative'] = float(np.max(abs(got-baseline))
                                                            /np.max(abs(baseline)))
                if backend == 'jax':
                    row['compiled_variants'] = rt._compiled[p]._cache_size()
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


if __name__ == '__main__':
    destination = repository_path('studies/fdtd/sphere_range/records/fdtd_sphere_range_probe.json')
    destination.write_text(json.dumps(measure(), indent=2)+'\n', encoding='utf-8')
