"""Measure public prism vertex/height/centre derivatives before selecting test budgets."""
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._runtime import runtime


OUTLINE = np.array(((-.25, -.21), (.18, -.24), (.28, -.03),
                    (.22, .23), (-.04, .27), (-.24, .14)))
CONCAVE = OUTLINE.copy()
CONCAVE[3] = (.02, .04)
RANGE = ((-.36, .36),)*3
POINT = np.concatenate((OUTLINE.ravel(), (.47, -.031, .017, -.013)))
CASES = (('x', 23., False), ('y', 23., False), ('z', 23., False), ('z', -23., True))
FULL_FIELD = sys.argv[1:] == ['full-field']


def point(concave=False):
    value = POINT.copy()
    if concave:
        value[:12] = CONCAVE.ravel()
    return value


def scene(v=POINT, axis='z', angle=23., bounds=RANGE):
    return fd.Scene(
        domain=fd.Domain(size=(1.6,)*3), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        run=fd.Run(steps=24), structures=(fd.Structure(
            fd.Prism(vertices=v[:12].reshape(6, 2), height=v[12], center=v[13:],
                     axis=axis, angle=angle, range=bounds),
            fd.Material(eps=2.25, mu=1.2)),), sources=(fd.Dipole(
                'drive', center=(-.55, .03, -.02), component='Ey',
                band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def objective(v, axis, angle, backend, plan):
    result = fd.run(scene(v, axis, angle), backend, plan=plan)
    # The full-field probe is source-dominated (~1070): sensitivities down to
    # 0.00177 lose relative accuracy in central differences. Observe a fixed
    # central region, excluding the drive, before selecting any derivative gate.
    field = result.final.E if FULL_FIELD else result.final.E[:, 5:11, 5:11, 5:11]
    return (field**2).sum()


def measure():
    warnings.filterwarnings('ignore')
    rows = []
    suffix = 'full_field_probe' if FULL_FIELD else 'probe'
    path = Path('studies/fdtd/prism_grad/records/fdtd_prism_grad_'+suffix+'.json')
    for axis, angle, concave in CASES:
        at = point(concave)
        p = fd.plan(scene(at, axis, angle))
        row = dict(axis=axis, angle=angle, concave=concave, point=at.tolist(), differences=[],
                   observation='full field' if FULL_FIELD else 'E[:, 5:11, 5:11, 5:11]')
        rows.append(row)
        row['numpy_value'] = float(objective(at, axis, angle, 'numpy', p))
        for h in (1e-4, 1e-5, 3e-6, 1e-6):
            g = [(objective(at+d, axis, angle, 'numpy', p)
                  - objective(at-d, axis, angle, 'numpy', p))/(2*h) for d in np.eye(16)*h]
            row['differences'].append(dict(step=h, gradient=[float(v) for v in g]))
            path.write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
        print(axis, angle, concave, row['differences'], flush=True)
        for backend in ('jax', 'torch'):
            rt = runtime(backend)
            started = time.perf_counter()
            fn = lambda v: objective(v, axis, angle, backend, p)
            if backend == 'jax':
                import jax
                import jax.numpy as jnp
                value, derivative = jax.value_and_grad(fn)(jnp.asarray(at))
                gradient = np.asarray(derivative).tolist()
            else:
                import torch
                v = torch.tensor(at, dtype=torch.float64, requires_grad=True)
                value = fn(v)
                derivative, = torch.autograd.grad(value, v)
                gradient = derivative.detach().numpy().tolist()
                value = value.detach()
                _, tangent = torch.func.jvp(fn, (v.detach(),), (torch.ones_like(v),))
                row['torch_jvp'] = float(tangent)
            row[backend] = dict(value=float(value), gradient=gradient,
                                seconds=time.perf_counter()-started)
            other = at.copy()
            other[12:] = (.49, .023, -.019, .031)
            current = fd.run(scene(other, axis, angle), backend, plan=p)
            plain = fd.run(scene(other, axis, angle, None), 'numpy')
            row[backend]['sweep_relative'] = {
                f: float(np.max(abs(rt.numpy(getattr(current.final, f))-getattr(plain.final, f)))
                         / np.max(abs(getattr(plain.final, f)))) for f in ('E', 'H')}
            print(axis, angle, backend, row[backend], flush=True)
            path.write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')


def volumes():
    from wavebit.fdtd._kernel.grid import Grid
    from wavebit.fdtd._prepare.occupancy import component_occupancy
    grid = Grid.from_domain((1.6,)*3, (0.,)*3, .1)
    rows = []
    for axis, angle, concave in CASES:
        for side in (-1, 1):
            base = scene(point(concave), axis, angle).structures[0].shape.replace(center=(0.,)*3)
            center = tuple(.4-hi if side > 0 else -.4-lo for lo, hi in zip(base.low, base.high))
            shape = base.replace(center=center, range=((-.4, .4),)*3)
            errors = []
            for field in ('E', 'H'):
                for c in range(3):
                    fill, _ = component_occupancy(shape.body(), grid, shape.support(grid),
                                                  field, c, runtime('numpy'))
                    errors.append(float(fill.sum()*grid.dx**3/shape.volume()-1))
            rows.append(dict(axis=axis, angle=angle, concave=concave, side=side,
                             support=shape.support(grid), errors=errors))
    print(rows, flush=True)
    Path('studies/fdtd/prism_grad/records/fdtd_prism_grad_volumes.json').write_text(
        json.dumps(rows, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    volumes() if sys.argv[1:] == ['volumes'] else measure()
