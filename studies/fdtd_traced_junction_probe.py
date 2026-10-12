"""A traced pillar standing flush on a smoothed substrate: the range-aware 'split' junction.

The scene is a metasurface cell: a 12 x 12-cell Bloch cell at ``dx = .05``, 42 cells along z
with eight-cell PMLs, a glass ``fd.Box`` substrate spanning the cell and running through the
bottom PML with its face at ``z = PLANE`` (a quarter cell above a node plane), and a ranged
Cylinder pillar standing on that face. The pillar's leaves are its radius, its height (with
its centre at ``PLANE + height/2``, so the base stays on the face) and its centre x.

Modes:

``masks``    the junction's live write set at the planning seed against the plan's own masks;
``equal``    a traced Torch lowering and run against the concrete NumPy plan and run of the same
             geometry, at radii and positions across the reservation where rim voxels enter
             and leave the junction;
``measure``  NumPy central differences at four steps against JAX and Torch reverse mode, for
             the transmitted and reflected zero orders under a normal s PlaneWave;
``settled``  energy conservation of those orders and of their derivatives in a long run with
             24-cell PMLs and a narrow band.

Results go to ``studies/results/fdtd_traced_junction_<mode>.json``.
"""
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

import wavebit.fd as fd
from wavebit.fd._runtime import runtime


RESULTS = Path(__file__).resolve().parent / 'results'
DX = .05
CELLS = 8
PLANE = -.2375
WAVELENGTHS = (.95, 1., 1.05)
#: Radius, height, centre x.
POINT = np.array((.12, .3, .011))
def ranges(plane=PLANE):
    return ((-.145, .145), (-.145, .145), (plane - .001, plane + .401))


RANGE = ranges()
GLASS = fd.Material(eps=2.1)


def _band():
    return fd.Band(wavelength=1., df=.15)


#: The Box pillar's leaves: x size, y size, height (centre at PLANE + height/2, x at .011).
BOX_POINT = np.array((.23, .21, .3))


def scene(v=None, theta=0., steps=600, ranged=True, kind='cylinder', plane=PLANE):
    v = tuple((POINT if kind == 'cylinder' else BOX_POINT) if v is None else v)
    pw = fd.PlaneWave('pw', z=.45, direction='-z', theta=theta, phi=30.,
                      polarization='s' if theta == 0. else 'p', band=_band())
    substrate = fd.Box(size=(10., 10., 2.), center=(0., 0., plane - 1.))
    if kind == 'cylinder':
        pillar = fd.Cylinder(radius=v[0], height=v[1], axis='z',
                             center=(v[2], -.007, plane + v[1] / 2),
                             range=ranges(plane) if ranged else None)
    else:
        pillar = fd.Box(size=(v[0], v[1], v[2]), center=(.011, -.007, plane + v[2] / 2),
                        range=ranges(plane) if ranged else None)
    return fd.Scene(
        domain=fd.Domain(size=(.6, .6, 2.1), center=(0., 0., .05)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                 z=(fd.PML(cells=CELLS, wavelength=1., background=GLASS,
                                           allow_thin=True),
                                    fd.PML(cells=CELLS, wavelength=1., allow_thin=True))),
        resolution=fd.Resolution(dx=DX), run=fd.Run(steps=steps, courant=.5), sources=(pw,),
        structures=(fd.Structure(substrate, GLASS),
                    fd.Structure(pillar, fd.Material(eps=2.25))),
        monitors=(fd.Diffraction('T', z=-.55, side='transmit', wavelengths=WAVELENGTHS,
                                 orders=0),
                  fd.Diffraction('R', z=.55, side='reflect', wavelengths=WAVELENGTHS,
                                 orders=0)))


def objectives(v, backend, plan, **kwargs):
    result = fd.run(scene(v, **kwargs), backend, plan=plan)
    return (result['T'].orders.incident.T[:, 0, 0].sum(), result['R'].orders.incident.R[:, 0, 0].sum())


def masks_mode(kind='cylinder', plane=PLANE):
    from wavebit.fd._lower.junction import _points, live_write
    warnings.filterwarnings('ignore')
    rt = runtime('numpy')
    plane = float(plane)
    current = scene(kind=kind, plane=plane)
    p = fd.plan(current)
    (record,) = p.junctions
    out = {}
    for c in range(4):
        points = _points(p.grid, record.rows, 'E', c)
        live = np.asarray(live_write(rt, record, current, p.grid, points))
        out['E' + 'xyz*'[c]] = dict(written=int(record.write[c].sum()),
                                     live=int(live.sum()),
                                     equal=bool(np.array_equal(live, record.write[c])))
    out['rows'] = [list(r) for r in record.rows]
    out['base_counts'] = [row[3] for row in record.counts]
    print(out, flush=True)
    (RESULTS / f'fdtd_traced_junction_masks_{kind}_{plane:g}.json').write_text(json.dumps(out, indent=2) + '\n',
                                                             encoding='utf-8')


def equal_mode(kind='cylinder', plane=PLANE):
    """Traced Torch lowering and run against the concrete NumPy plan of the same geometry."""
    import torch
    warnings.filterwarnings('ignore')
    plane = float(plane)
    p = fd.plan(scene(steps=40, kind=kind, plane=plane))
    rows = []
    points = (((.12, .3, .011), (.1, .25, -.013), (.135, .37, .003), (.127, .21, .0))
              if kind == 'cylinder' else
              ((.23, .21, .3), (.2, .22, .25), (.25, .205, .37), (.214, .233, .21)))
    for v in points:
        traced_v = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in v]
        traced = scene(traced_v, steps=40, kind=kind, plane=plane)
        concrete = scene(v, steps=40, kind=kind, plane=plane)
        q = fd.plan(concrete)
        rtt, rtn = runtime('torch'), runtime('numpy')
        a, b = fd.lower(traced, p, rtt), fd.lower(concrete, q, rtn)
        row = dict(point=v, written=int(sum(r.write.sum() for r in q.junctions)),
                   base=int(sum(row[3] for r in q.junctions for row in r.counts)))
        for key in ('be',):
            want = np.asarray(b[key])
            row[key] = float(np.max(abs(rtt.numpy(a[key].detach()) - want)) / np.max(abs(want)))
        for name in b.get('constitutive', {}):
            want = np.asarray(b['constitutive'][name])
            got = rtt.numpy(a['constitutive'][name].detach())
            row['cross_' + name] = float(np.max(abs(got - want))
                                         / max(np.max(abs(want)), 1e-300))
        ta = fd.run(traced, 'torch', plan=p)
        tb = fd.run(concrete, 'numpy', plan=q)
        for f in ('E', 'H'):
            want = getattr(tb.final, f)
            row[f] = float(np.max(abs(rtt.numpy(getattr(ta.final, f).detach()) - want))
                           / np.max(abs(want)))
        rows.append(row)
        print(row, flush=True)
    (RESULTS / f'fdtd_traced_junction_equal_{kind}_{plane:g}.json').write_text(json.dumps(rows, indent=2) + '\n',
                                                             encoding='utf-8')


def numpy_values(v, kind='cylinder', plane=PLANE):
    """The concrete solver as a user runs it: planned (and classified) at ``v`` itself.

    A concrete ranged pillar's junction record (its counts, masks and geometry signature) is
    its own geometry's, so a plan does not carry over to another geometry the way a lone
    ranged body's does; ``fd.run(scene, plan=p)`` refuses that by its skeleton.
    """
    return np.array([float(x) for x in objectives(v, 'numpy', None, kind=kind, plane=plane)])


def measure_mode(kind='cylinder', plane=PLANE, height=None):
    """``height`` overrides the point's: with the face on a node or half-node plane, height .3
    puts the top cap on one too, where the cap's moment shares the voxel face half and half
    and the value is the mean of its two one-sided limits, a point discontinuity."""
    warnings.filterwarnings('ignore')
    plane = float(plane)
    point = (POINT if kind == 'cylinder' else BOX_POINT).copy()
    if height is not None:
        point[1 if kind == 'cylinder' else 2] = float(height)
    p = fd.plan(scene(point, kind=kind, plane=plane))
    row = dict(kind=kind, plane=plane, point=point.tolist(), differences=[])
    started = time.perf_counter()
    row['numpy_values'] = numpy_values(point, kind, plane).tolist()
    row['numpy_seconds'] = time.perf_counter() - started
    path = RESULTS / f'fdtd_traced_junction_measure_{kind}_{plane:g}.json'
    for h in (1e-4, 1e-5, 3e-6, 1e-6):
        jac = np.array([(numpy_values(point + d, kind, plane) - numpy_values(point - d, kind, plane))
                        / (2 * h)
                        for d in np.eye(3) * h]).T
        row['differences'].append(dict(step=h, jacobian=jac.tolist()))
        path.write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
        print('central difference', h, jac.tolist(), flush=True)
    for backend in ('jax', 'torch'):
        started = time.perf_counter()
        runtime(backend)
        if backend == 'jax':
            import jax
            import jax.numpy as jnp

            def fn(v):
                return jnp.stack(objectives(v, 'jax', p, kind=kind, plane=plane))
            x = jnp.asarray(point)
            values, jac = np.asarray(fn(x)), np.asarray(jax.jacrev(fn)(x))
        else:
            import torch
            v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            out = objectives(v, 'torch', p, kind=kind, plane=plane)
            jac = np.array([torch.autograd.grad(o, v, retain_graph=True)[0].numpy()
                            for o in out])
            values = np.array([float(o.detach()) for o in out])
        row[backend] = dict(values=values.tolist(), jacobian=jac.tolist(),
                            seconds=time.perf_counter() - started)
        path.write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
        print(backend, row[backend], flush=True)
    reference = np.array(row['jax']['jacobian'])
    row['relative_errors'] = {str(d['step']): float(np.max(abs(np.array(d['jacobian'])
                                                              - reference) / abs(reference)))
                              for d in row['differences']}
    row['jax_torch_max_abs'] = float(np.max(abs(np.array(row['torch']['jacobian'])
                                                - reference)))
    path.write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
    print('relative errors', row['relative_errors'], 'jax-torch', row['jax_torch_max_abs'],
          flush=True)


def settled_mode(cells=24, steps=6000, df=.05, h=3e-6):
    """Energy conservation of the pillar's zero orders and of their derivatives, settled.

    The lossless cell must give T + R = 1 per wavelength and dT + dR = 0. The regression
    scene's thin PMLs and short run are a route test; here both PMLs have ``cells`` cells at
    the default reflection, the band is narrowed to ``df`` (``fdtd_traced_illumination_probe``
    found the wide band keeps an oblique cell ringing) and the run is ``steps`` long. The
    derivatives are central differences of the concrete solver, each geometry planned on its
    own, at the regression step.
    """
    import dataclasses
    warnings.filterwarnings('ignore')
    cells, steps, df = int(cells), int(steps), float(df)

    def make(v):
        base = scene(v)
        pw = base.sources[0].replace(band=fd.Band(wavelength=1., df=df))
        grow = (cells - CELLS) * DX
        return dataclasses.replace(
            base, sources=(pw,), run=fd.Run(steps=steps, courant=.5),
            domain=fd.Domain(size=(.6, .6, 2.1 + 2 * grow), center=(0., 0., .05)),
            boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=(
                fd.PML(cells=cells, wavelength=1., background=GLASS),
                fd.PML(cells=cells, wavelength=1.))))

    def rows(v):
        result = fd.run(make(v), 'numpy')
        return (np.asarray(result['T'].orders.incident.T[:, 0, 0]),
                np.asarray(result['R'].orders.incident.R[:, 0, 0]), result.diagnostics['decay'])
    T, R, decay = rows(POINT)
    out = dict(cells=cells, steps=steps, df=df, step=h, decay=decay, T0=T.tolist(),
               R0=R.tolist(), energy_residual=(T + R - 1).tolist(), derivatives=[])
    for i, d in enumerate(np.eye(3)[:2] * h):
        plus, minus = rows(POINT + d), rows(POINT - d)
        dT, dR = (plus[0] - minus[0]) / (2 * h), (plus[1] - minus[1]) / (2 * h)
        out['derivatives'].append(dict(leaf=('radius', 'height')[i], dT0=dT.tolist(),
                                       dR0=dR.tolist(), dT_plus_dR=(dT + dR).tolist()))
        print(out['derivatives'][-1], flush=True)
    print(out, flush=True)
    (RESULTS / 'fdtd_traced_junction_settled.json').write_text(
        json.dumps(out, indent=2, default=str) + chr(10), encoding='utf-8')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'masks'
    {'masks': masks_mode, 'equal': equal_mode, 'measure': measure_mode,
     'settled': settled_mode}[mode](
        *sys.argv[2:5])
