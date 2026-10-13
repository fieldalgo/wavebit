"""Measure public box size/centre derivatives before selecting regression tolerances."""
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._runtime import runtime


RANGE = ((-.36, .36),)*3
POINT = np.array((.43, .46, .49, -.031, .017, -.013))
SECOND = np.array((.46, .43, .51, .023, -.019, .031))
CASES = (('z', 0.), ('x', 23.), ('y', 23.), ('z', 23.))


def scene(v=POINT, axis='z', angle=23., bounds=RANGE):
    return fd.Scene(
        domain=fd.Domain(size=(1.6,)*3), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        run=fd.Run(steps=24), structures=(fd.Structure(
            fd.Box(size=v[:3], center=v[3:], axis=axis, angle=angle, range=bounds),
            fd.Material(eps=2.25, mu=1.2)),), sources=(fd.Dipole(
                'drive', center=(-.55, .03, -.02), component='Ey',
                band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def objective(v, axis, angle, backend, plan):
    result = fd.run(scene(v, axis, angle), backend, plan=plan)
    return (result.final.E**2).sum()


def measure():
    warnings.filterwarnings('ignore')
    rows = []
    for axis, angle in CASES:
        p = fd.plan(scene(axis=axis, angle=angle))
        row = dict(axis=axis, angle=angle, point=POINT.tolist(), differences=[])
        row['numpy_value'] = float(objective(POINT, axis, angle, 'numpy', p))
        for h in (1e-4, 1e-5, 3e-6, 1e-6):
            g = [(objective(POINT+d, axis, angle, 'numpy', p)
                  - objective(POINT-d, axis, angle, 'numpy', p))/(2*h) for d in np.eye(6)*h]
            row['differences'].append(dict(step=h, gradient=[float(v) for v in g]))
        print(axis, angle, row['differences'], flush=True)
        for backend in ('jax', 'torch'):
            rt = runtime(backend)
            started = time.perf_counter()
            fn = lambda v: objective(v, axis, angle, backend, p)
            if backend == 'jax':
                import jax
                import jax.numpy as jnp
                value, derivative = jax.value_and_grad(fn)(jnp.asarray(POINT))
                gradient = np.asarray(derivative).tolist()
            else:
                import torch
                v = torch.tensor(POINT, dtype=torch.float64, requires_grad=True)
                value = fn(v)
                derivative, = torch.autograd.grad(value, v)
                gradient = derivative.detach().numpy().tolist()
                value = value.detach()
                _, tangent = torch.func.jvp(fn, (v.detach(),), (torch.ones_like(v),))
                row['torch_jvp'] = float(tangent)
            row[backend] = dict(value=float(value), gradient=gradient,
                                seconds=time.perf_counter()-started)
            current = fd.run(scene(SECOND, axis, angle), backend, plan=p)
            plain = fd.run(scene(SECOND, axis, angle, None), 'numpy')
            row[backend]['sweep_relative'] = {
                f: float(np.max(abs(rt.numpy(getattr(current.final, f))-getattr(plain.final, f)))
                         / np.max(abs(getattr(plain.final, f)))) for f in ('E', 'H')}
            print(axis, angle, backend, row[backend], flush=True)
        rows.append(row)
        Path('studies/fdtd/box_grad/records/fdtd_box_grad_probe.json').write_text(
            json.dumps(rows, indent=2)+'\n', encoding='utf-8')


def volumes():
    from wavebit.fdtd._kernel.grid import Grid
    from wavebit.fdtd._prepare.occupancy import component_occupancy
    grid = Grid.from_domain((1.6,)*3, (0.,)*3, .1)
    rows = []
    for axis, angle in CASES:
        for side in (-1, 1):
            base = fd.Box(size=POINT[:3], axis=axis, angle=angle)
            center = tuple(side*(.4-h) for h in base.high)
            shape = base.replace(center=center, range=((-.4, .4),)*3)
            errors = []
            for field in ('E', 'H'):
                for c in range(3):
                    fill, _ = component_occupancy(shape.body(), grid, shape.support(grid),
                                                  field, c, runtime('numpy'))
                    errors.append(float(fill.sum()*grid.dx**3/shape.volume()-1))
            rows.append(dict(axis=axis, angle=angle, side=side, support=shape.support(grid),
                             errors=errors))
    print(rows, flush=True)
    Path('studies/fdtd/box_grad/records/fdtd_box_grad_volumes.json').write_text(
        json.dumps(rows, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    volumes() if sys.argv[1:] == ['volumes'] else measure()
