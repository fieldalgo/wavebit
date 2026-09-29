"""Measure a four-body public geometry gradient before selecting regression steps."""
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

import wavebit.fd as fd
from wavebit.fd._runtime import runtime


POINT = np.array((.223, .239, .45, .49))
CENTERS = ((-2.131, .017, -.013), (-.719, -.013, .023),
           (.713, .019, -.017), (2.117, -.019, .013))
RANGES = tuple(((x-.36, x+.36), (-.36, .36), (-.36, .36))
               for x in (-2.1, -.7, .7, 2.1))
OUTLINE = ((-.25, -.21), (.18, -.24), (.28, -.03),
           (.02, .04), (-.04, .27), (-.24, .14))


def scene(v=POINT):
    shapes = (fd.Sphere(radius=v[0], center=CENTERS[0], range=RANGES[0]),
              fd.Cylinder(radius=v[1], height=.47, axis='y', center=CENTERS[1], range=RANGES[1]),
              fd.Box(size=(v[2], .47, .49), angle=23., center=CENTERS[2], range=RANGES[2]),
              fd.Prism(vertices=OUTLINE, height=v[3], angle=-23., center=CENTERS[3], range=RANGES[3]))
    return fd.Scene(
        domain=fd.Domain(size=(6., 1.6, 1.6)), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        run=fd.Run(steps=32), structures=tuple(fd.Structure(shape, fd.Material(eps=eps, mu=mu))
            for shape, eps, mu in zip(shapes, (2.25, 2., 1.8, 2.1), (1.2, 1.1, 1., 1.15))),
        sources=tuple(fd.Dipole('drive'+str(i), center=(x-.1, -.55, -.02), component='Ey',
                      band=fd.Band(wavelength=3., df=.03), delay=-18.)
                      for i, x in enumerate((-2.1, -.7, .7, 2.1))))


def objective(v, backend, plan):
    result = fd.run(scene(v), backend, plan=plan)
    # Fixed central y/z slab excludes the four drives, following the prism probe.
    return (result.final.E[:, 5:11, 5:11, :]**2).sum()


def measure():
    warnings.filterwarnings('ignore')
    path = Path('studies/results/fdtd_multi_shape_grad_probe.json')
    p = fd.plan(scene())
    row = dict(point=POINT.tolist(), supports=[s.shape.support(p.grid) for s in scene().structures],
               observation='E[:, 5:11, 5:11, :]', differences=[])
    row['numpy_value'] = float(objective(POINT, 'numpy', p))
    for h in (1e-4, 1e-5, 3e-6, 1e-6):
        gradient = [(objective(POINT+d, 'numpy', p)-objective(POINT-d, 'numpy', p))/(2*h)
                    for d in np.eye(4)*h]
        row['differences'].append(dict(step=h, gradient=[float(g) for g in gradient]))
        path.write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8')
        print('central difference', row['differences'][-1], flush=True)
    for backend in ('jax', 'torch'):
        rt = runtime(backend)
        started = time.perf_counter()
        fn = lambda v: objective(v, backend, p)
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
        row[backend] = dict(value=float(value), gradient=gradient, seconds=time.perf_counter()-started)
        print(backend, row[backend], flush=True)
        at = POINT+np.array((.004, -.003, .013, -.009))
        current = fd.run(scene(at), backend, plan=p)
        direct = fd.run(scene(at), 'numpy', plan=p)
        row[backend]['fields_relative'] = {
            f: float(np.max(abs(rt.numpy(getattr(current.final, f))-getattr(direct.final, f)))
                     / np.max(abs(getattr(direct.final, f)))) for f in ('E', 'H')}
        path.write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8')


def controls():
    from wavebit.fd._lower.constitutive import common_points
    warnings.filterwarnings('ignore')
    current = scene()
    p = fd.plan(current)
    cylinder = current.structures[1].shape.body()
    rows = []
    for term in p.constitutive:
        box = tuple(reversed(term.box))
        support = current.structures[1].shape.support(p.grid)
        if not all(min(b, d) > max(a, c) for (a, b), (c, d) in zip(box, support)):
            continue
        points = common_points(p.grid, box, term.field)
        xp, axes, q, s = cylinder._split(points)
        radial, _ = cylinder._transverse_surface(q)
        axial = abs(s)-cylinder.height/2
        fill = cylinder.exact_fill(points, p.dx)
        live = (fill > 1e-9) & (fill < 1-1e-9) & ~((radial > 0) & (axial > 0))
        gap = np.abs(radial-axial)[live]
        rows.append(dict(field=term.field, minimum_wall_cap_gap=float(gap.min()),
                         switching_counts={str(h): int(np.sum(gap < h))
                             for h in (1e-4, 1e-5, 3e-6, 1e-6)}))
    at = POINT+np.array((.004, -.003, .013, -.009))
    current = scene(at)
    ranged = fd.run(current, plan=p)
    plain = fd.run(current.replace(structures=tuple(s.replace(shape=s.shape.replace(range=None))
                                                   for s in current.structures)))
    reversed_result = fd.run(current.replace(structures=current.structures[::-1]))
    fields = {label: {f: float(np.max(abs(getattr(ranged.final, f)-getattr(other.final, f)))
                               / np.max(abs(getattr(ranged.final, f)))) for f in ('E', 'H')}
              for label, other in [('unranged', plain), ('reversed', reversed_result)]}
    rt = runtime('numpy')
    all_params = fd.lower(current, p, rt)
    empty = current.replace(structures=())
    base = fd.lower(empty, fd.plan(empty), rt)
    singles = [fd.lower(current.replace(structures=(s,)), fd.plan(current.replace(structures=(s,))), rt)
               for s in current.structures]
    coefficient_errors = {}
    for key in ('be', 'bh'):
        expected = base[key]+sum(one[key]-base[key] for one in singles)
        coefficient_errors[key] = float(np.max(abs(all_params[key]-expected))/np.max(abs(expected)))
    output = dict(cylinder_wall_cap=rows, fields_relative=fields, coefficients_relative=coefficient_errors)
    print(output, flush=True)
    Path('studies/results/fdtd_multi_shape_grad_controls_probe.json').write_text(
        json.dumps(output, indent=2)+'\n', encoding='utf-8')


def small_steps():
    warnings.filterwarnings('ignore')
    p = fd.plan(scene())
    base = float(objective(POINT, 'numpy', p))
    rows = []
    for h in (2e-6, 1e-6, 3e-7, 1e-7):
        d = np.array((0., h, 0., 0.))
        plus = float(objective(POINT+d, 'numpy', p))
        minus = float(objective(POINT-d, 'numpy', p))
        rows.append(dict(step=h, plus=plus, minus=minus, base=base,
                         gradient=(plus-minus)/(2*h)))
    print(rows, flush=True)
    Path('studies/results/fdtd_multi_shape_grad_small_steps.json').write_text(
        json.dumps(rows, indent=2)+'\n', encoding='utf-8')


def coefficient_diagnostic():
    from wavebit.fd._lower.coefficients import build
    warnings.filterwarnings('ignore')
    p, rt = fd.plan(scene()), runtime('numpy')
    rows, reference = [], None
    for h in (1e-6, 3e-6, 1e-5, 1e-4):
        samples = []
        for sign in (-1, 1):
            at = POINT.copy()
            at[1] += sign*h
            current = scene(at)
            raw = build(current, p, rt)
            lifted = fd.lower(current, p, rt)
            samples.append({prefix+key: np.asarray(values[key]) for prefix, values in
                            [('raw_', raw), ('lifted_', lifted)] for key in ('be', 'bh')})
        derivatives = {key: (samples[1][key]-samples[0][key])/(2*h) for key in samples[0]}
        if reference is None:
            reference = derivatives
        errors = {}
        for key, derivative in derivatives.items():
            delta = abs(derivative-reference[key])
            at = np.unravel_index(np.argmax(delta), delta.shape)
            errors[key] = dict(max_derivative_difference=float(delta[at]), index=list(map(int, at)),
                               derivative=float(derivative[at]), reference=float(reference[key][at]))
        rows.append(dict(step=h, errors=errors))
    print(rows, flush=True)
    Path('studies/results/fdtd_multi_shape_grad_coefficients.json').write_text(
        json.dumps(rows, indent=2)+'\n', encoding='utf-8')


def component_switches():
    from wavebit.fd._kernel.grid import Grid
    from wavebit.fd._lower.occupancy import yee_points
    current = scene()
    grid = Grid.from_domain(current.domain.size, current.domain.center, current.resolution.dx)
    shape = current.structures[1].shape
    body, support = shape.body(), shape.support(grid)
    rows = []
    for field in ('E', 'H'):
        for component in range(3):
            points = yee_points(grid, support, field, component)
            xp, axes, q, s = body._split(points)
            radial, _ = body._transverse_surface(q)
            axial = abs(s)-body.height/2
            fill = body.exact_fill(points, grid.dx)
            live = (fill > 1e-9) & (fill < 1-1e-9) & ~((radial > 0) & (axial > 0))
            distances = np.where(live, abs(radial-axial), np.inf)
            at = np.unravel_index(np.argmin(distances), distances.shape)
            rows.append(dict(field=field+'xyz'[component], point=points[at].tolist(),
                minimum_wall_cap_gap=float(distances[at]), fill=float(fill[at]),
                switching_counts={str(h): int(np.sum(distances < h))
                                  for h in (1e-4, 1e-5, 3e-6, 2e-6, 1e-6)}))
    print(rows, flush=True)
    Path('studies/results/fdtd_multi_shape_grad_component_switches.json').write_text(
        json.dumps(rows, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    {'controls': controls, 'small-steps': small_steps,
     'coefficient-diagnostic': coefficient_diagnostic,
     'component-switches': component_switches}.get(' '.join(sys.argv[1:]), measure)()
