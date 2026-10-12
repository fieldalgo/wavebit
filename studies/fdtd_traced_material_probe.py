"""Traced material values beside traced geometry: a pillar's permittivity with its shape.

The metasurface cell is ``studies/fdtd_traced_junction_probe.py``'s: a glass ``fd.Box``
substrate running through the bottom PML and a ranged Cylinder (or unturned Box) pillar
standing flush on it under the default 'split' junction, a normal s PlaneWave, and the
transmitted and reflected zero orders as objectives. Here the pillar's ``eps`` is a leaf too.
The substrate stays concrete: it reaches the bottom PML, whose grazing survey is judged on the
host only (``shape_ranges._seeded_materials``).

Modes:

``equal``     a traced Torch lowering and a 40-step run from the seed's plan against the
              concrete NumPy scene planned at each of four (geometry, eps) points, for both
              pillar kinds; and the same with the geometry concrete and unranged, eps alone
              traced (the material-only path through ``fd.run``);
``measure``   NumPy central differences at four steps against JAX and Torch reverse mode for
              (radius, height, centre x, eps) -- (x size, y size, height, eps) for the Box;
``multi``     the four separated magnetic bodies of ``fdtd_multi_shape_grad_probe``: each one's
              size, eps and mu traced together (12 leaves) under Dipole drives;
``sphere``    an unranged smoothed sphere's eps and mu alone through ``fd.run`` (the
              material-only path), against central differences;
``settled``   energy conservation of the cylinder cell's zero orders and of their eps and radius
              derivatives with 24-cell PMLs, a narrow band and a long run;
``guards``    where the runtime guards stop a traced value the concrete path refuses: the
              resolution floor, the time-step bound, positivity.

Results go to ``studies/results/fdtd_traced_material_<mode>.json``.
"""
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

import wavebit.fd as fd
from wavebit.fd._runtime import runtime
from studies import fdtd_traced_junction_probe as junction


RESULTS = Path(__file__).resolve().parent / 'results'
#: Radius, height, centre x, eps (Cylinder); x size, y size, height, eps (Box).
POINT = np.array((.12, .3, .011, 2.25))
BOX_POINT = np.array((.23, .21, .3, 2.25))
#: Across the reservation and the pillar's index, below the resolution floor (eps 2.576 at
#: dx .05 and the band's shortest wavelength .642).
POINTS = {'cylinder': ((.12, .3, .011, 2.25), (.1, .25, -.013, 1.7), (.135, .37, .003, 2.5),
                       (.127, .21, 0., 1.3)),
          'box': ((.23, .21, .3, 2.25), (.2, .22, .25, 1.7), (.25, .205, .37, 2.5),
                  (.214, .233, .21, 1.3))}


def scene(v=None, kind='cylinder', ranged=True, **kwargs):
    """The junction probe's cell with the pillar's permittivity ``v[3]``."""
    v = tuple((POINT if kind == 'cylinder' else BOX_POINT) if v is None else v)
    base = junction.scene(v[:3], kind=kind, ranged=ranged, **kwargs)
    substrate, pillar = base.structures
    return base.replace(structures=(substrate, pillar.replace(material=fd.Material(eps=v[3]))))


def objectives(v, backend, plan, **kwargs):
    result = fd.run(scene(v, **kwargs), backend, plan=plan)
    return (result['T'].orders.incident.T[:, 0, 0].sum(), result['R'].orders.incident.R[:, 0, 0].sum())


def numpy_values(v, **kwargs):
    """Planned at ``v`` itself: a concrete ranged pillar's junction record is its geometry's."""
    return np.array([float(x) for x in objectives(v, 'numpy', None, **kwargs)])


def _write(name, row):
    (RESULTS / f'fdtd_traced_material_{name}.json').write_text(
        json.dumps(row, indent=2, default=str) + '\n', encoding='utf-8')


def equal_mode():
    import torch
    warnings.filterwarnings('ignore')
    rows = []
    for kind in ('cylinder', 'box'):
        for ranged in (True, False):
            seed = POINTS[kind][0]
            p = fd.plan(scene(seed, kind=kind, ranged=ranged, steps=40))
            # Material-only: the geometry stays at the seed and only eps moves.
            points = POINTS[kind] if ranged else tuple(seed[:3] + (e,) for e in (2.25, 1.7, 2.5,
                                                                                  1.3))
            for v in points:
                leaves = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in v]
                if not ranged:
                    leaves = list(v[:3]) + leaves[3:]
                traced = scene(leaves, kind=kind, ranged=ranged, steps=40)
                concrete = scene(v, kind=kind, ranged=ranged, steps=40)
                q = fd.plan(concrete)
                rtt, rtn = runtime('torch'), runtime('numpy')
                a, b = fd.lower(traced, p, rtt), fd.lower(concrete, q, rtn)
                row = dict(kind=kind, ranged=ranged, point=v,
                           written=int(sum(r.write.sum() for r in q.junctions)),
                           requires_grad=bool(a['be'].requires_grad))
                want = np.asarray(b['be'])
                row['be'] = float(np.max(abs(rtt.numpy(a['be'].detach()) - want))
                                  / np.max(abs(want)))
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
                row['stable'] = bool(ta.diagnostics['stable'])
                rows.append(row)
                print(row, flush=True)
                _write('equal', rows)


def measure_mode(kind='cylinder', plane=junction.PLANE):
    warnings.filterwarnings('ignore')
    plane = float(plane)
    point = (POINT if kind == 'cylinder' else BOX_POINT).copy()
    p = fd.plan(scene(point, kind=kind, plane=plane))
    row = dict(kind=kind, plane=plane, point=point.tolist(), differences=[])
    started = time.perf_counter()
    row['numpy_values'] = numpy_values(point, kind=kind, plane=plane).tolist()
    row['numpy_seconds'] = time.perf_counter() - started
    name = f'measure_{kind}_{plane:g}'
    for h in (1e-4, 1e-5, 3e-6, 1e-6):
        jac = np.array([(numpy_values(point + d, kind=kind, plane=plane)
                         - numpy_values(point - d, kind=kind, plane=plane)) / (2 * h)
                        for d in np.eye(4) * h]).T
        row['differences'].append(dict(step=h, jacobian=jac.tolist()))
        _write(name, row)
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
        _write(name, row)
        print(backend, row[backend], flush=True)
    reference = np.array(row['jax']['jacobian'])
    row['relative_errors'] = {str(d['step']): float(np.max(abs(np.array(d['jacobian'])
                                                              - reference) / abs(reference)))
                              for d in row['differences']}
    row['relative_errors_eps'] = {str(d['step']): float(np.max(
        abs(np.array(d['jacobian'])[:, 3] - reference[:, 3]) / abs(reference[:, 3])))
        for d in row['differences']}
    row['jax_torch_max_abs'] = float(np.max(abs(np.array(row['torch']['jacobian'])
                                                - reference)))
    _write(name, row)
    print('relative errors', row['relative_errors'], 'eps', row['relative_errors_eps'],
          'jax-torch', row['jax_torch_max_abs'], flush=True)


# The four separated magnetic bodies of fdtd_multi_shape_grad_probe: size, eps and mu of each.
MULTI = np.array((.223, .239, .45, .49, 2.25, 2., 1.8, 2.1, 1.2, 1.1, 1., 1.15))


def multi_scene(v=MULTI):
    from studies import fdtd_multi_shape_grad_probe as multi
    base = multi.scene(v[:4])
    return base.replace(structures=tuple(
        s.replace(material=fd.Material(eps=v[4 + i], mu=v[8 + i]))
        for i, s in enumerate(base.structures)))


def multi_objective(v, backend, plan):
    result = fd.run(multi_scene(v), backend, plan=plan)
    return (result.final.E[:, 5:11, 5:11, :]**2).sum()


def _gradient_rows(name, point, objective, plan, steps=(1e-4, 1e-5, 3e-6, 1e-6)):
    """Central differences (fresh concrete runs, each at ``plan``) against JAX and Torch."""
    row = dict(point=list(point), differences=[])
    row['numpy_value'] = float(objective(point, 'numpy', plan))
    for h in steps:
        gradient = [(objective(point + d, 'numpy', plan) - objective(point - d, 'numpy', plan))
                    / (2 * h) for d in np.eye(len(point)) * h]
        row['differences'].append(dict(step=h, gradient=[float(g) for g in gradient]))
        _write(name, row)
        print('central difference', row['differences'][-1], flush=True)
    for backend in ('jax', 'torch'):
        runtime(backend)
        started = time.perf_counter()
        if backend == 'jax':
            import jax
            import jax.numpy as jnp
            value, derivative = jax.value_and_grad(lambda v: objective(v, 'jax', plan))(
                jnp.asarray(point))
            gradient = np.asarray(derivative)
        else:
            import torch
            v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            value = objective(v, 'torch', plan)
            value.backward()
            gradient = v.grad.numpy()
        row[backend] = dict(value=float(value), gradient=gradient.tolist(),
                            seconds=time.perf_counter() - started)
        _write(name, row)
        print(backend, row[backend], flush=True)
    reference = np.array(row['jax']['gradient'])
    row['relative_errors'] = {str(d['step']): float(np.max(abs(np.array(d['gradient'])
                                                              - reference) / abs(reference)))
                              for d in row['differences']}
    row['jax_torch_max_abs'] = float(np.max(abs(np.array(row['torch']['gradient'])
                                                - reference)))
    _write(name, row)
    print('relative errors', row['relative_errors'], 'jax-torch', row['jax_torch_max_abs'],
          flush=True)


def multi_mode():
    warnings.filterwarnings('ignore')
    p = fd.plan(multi_scene())
    _gradient_rows('multi', MULTI, multi_objective, p)


#: An unranged glass-and-magnetic sphere in a PEC box under one Dipole: eps and mu.
SPHERE = np.array((2.25, 1.3))


def sphere_scene(v=SPHERE):
    return fd.Scene(
        domain=fd.Domain(size=(1.6, 1.6, 1.6)), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()), run=fd.Run(steps=40),
        structures=(fd.Structure(fd.Sphere(radius=.33, center=(.031, -.017, .012)),
                                 fd.Material(eps=v[0], mu=v[1])),),
        sources=(fd.Dipole('d', center=(-.5, -.45, .05), component='Ey',
                           band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def sphere_objective(v, backend, plan):
    result = fd.run(sphere_scene(v), backend, plan=plan)
    return (result.final.E[:, 4:12, 4:12, 4:12]**2).sum()


def sphere_mode():
    warnings.filterwarnings('ignore')
    p = fd.plan(sphere_scene())
    _gradient_rows('sphere', SPHERE, sphere_objective, p)


def settled_mode(cells=24, steps=6000, df=.05, h=3e-6):
    """``T + R = 1`` and ``dT + dR = 0`` for eps and the radius, as the junction probe does."""
    import dataclasses
    warnings.filterwarnings('ignore')
    cells, steps, df = int(cells), int(steps), float(df)

    def make(v):
        base = scene(v)
        pw = base.sources[0].replace(band=fd.Band(wavelength=1., df=df))
        grow = (cells - junction.CELLS) * junction.DX
        return dataclasses.replace(
            base, sources=(pw,), run=fd.Run(steps=steps, courant=.5),
            domain=fd.Domain(size=(.6, .6, 2.1 + 2 * grow), center=(0., 0., .05)),
            boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=(
                fd.PML(cells=cells, wavelength=1., background=junction.GLASS),
                fd.PML(cells=cells, wavelength=1.))))

    def rows(v):
        result = fd.run(make(v), 'numpy')
        return (np.asarray(result['T'].orders.incident.T[:, 0, 0]),
                np.asarray(result['R'].orders.incident.R[:, 0, 0]), result.diagnostics['decay'])
    T, R, decay = rows(POINT)
    out = dict(cells=cells, steps=steps, df=df, step=h, decay=decay, T0=T.tolist(),
               R0=R.tolist(), energy_residual=(T + R - 1).tolist(), derivatives=[])
    for i in (3, 0):
        d = np.eye(4)[i] * h
        plus, minus = rows(POINT + d), rows(POINT - d)
        dT, dR = (plus[0] - minus[0]) / (2 * h), (plus[1] - minus[1]) / (2 * h)
        out['derivatives'].append(dict(leaf=('radius', 'height', 'x', 'eps')[i], dT0=dT.tolist(),
                                       dR0=dR.tolist(), dT_plus_dR=(dT + dR).tolist()))
        print(out['derivatives'][-1], flush=True)
        _write('settled', out)
    print(out, flush=True)
    _write('settled', out)


def guards_mode():
    """A traced value past each door the concrete path keeps: the run is ``nan``, ``stable`` is
    False, and the concrete scene at that value is refused by name."""
    import torch
    warnings.filterwarnings('ignore')
    p = fd.plan(scene(POINT, steps=20))
    rows = []
    for label, eps in (('inside every door', 2.25), ('resolution floor', 2.7),
                       ('time-step bound', .2), ('positivity', -1.)):
        v = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in POINT[:3]]
        v.append(torch.tensor(eps, dtype=torch.float64, requires_grad=True))
        result = fd.run(scene(v, steps=20), 'torch', plan=p)
        E = result.final.E.detach().numpy()
        row = dict(label=label, eps=eps, stable=bool(result.diagnostics['stable']),
                   finite=bool(np.isfinite(E).all()))
        try:
            fd.plan(scene(tuple(POINT[:3]) + (eps,), steps=20))
            row['concrete'] = 'planned'
        except ValueError as err:
            row['concrete'] = str(err)[:200]
        rows.append(row)
        print(row, flush=True)
    _write('guards', rows)


def closed_scene(background=1., eps=2.25, mirror=False, extra=()):
    """An unranged smoothed sphere in a PEC box under one Dipole (the regressions' closed cell):
    nothing reads the background but the update. ``mirror`` centres the sphere on a Symmetry
    face; ``extra`` adds structures."""
    faces = dict(x=fd.PEC(), y=fd.PEC(), z=fd.PEC())
    if mirror:
        faces['x'] = (fd.Symmetry(parity=-1), fd.PEC())
    centre = (-.8 if mirror else .031, -.017, .012)
    return fd.Scene(
        domain=fd.Domain(size=(1.6, 1.6, 1.6)), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(**faces), run=fd.Run(steps=40), background=background,
        structures=(fd.Structure(fd.Sphere(radius=.33, center=centre),
                                 fd.Material(eps=eps)),) + tuple(extra),
        sources=(fd.Dipole('d', center=(.5, .45, .05), component='Ey',
                           band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def pml_box(background=1., own=True, film=None):
    """The closed cell with PMLs on z (each its own concrete background, or matched to the
    scene's), and optionally a Slab film of permittivity ``film`` running into the bottom one:
    a Dipole scene, so no host rule reads either medium's value."""
    s = closed_scene(background=background).replace(
        domain=fd.Domain(size=(1.6, 1.6, 2.4)),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PML(
            cells=4, wavelength=3., allow_thin=True,
            background=fd.Material(eps=1.) if own else None)))
    if film is not None:
        s = s.replace(structures=s.structures + (fd.Structure(fd.Slab(z=(-1.2, -.6)),
                                                              fd.Material(eps=film)),))
    return s


def dimer(eps=2.25):
    """The sphere on the Symmetry face's side of it, its surface a cell clear: the term box
    touches the face, and no fill there is fractional."""
    s = closed_scene(eps=eps, mirror=True)
    shape = s.structures[0].shape.replace(center=(-.8 + .33 + .1 + .002, -.017, .012))
    return s.replace(structures=(s.structures[0].replace(shape=shape),))


def open_box(background=1.):
    """The sphere in an Absorber box with a lattice FarField around it and the Dipole."""
    return closed_scene(background=background).replace(
        domain=fd.Domain(size=(2.4, 2.4, 2.4)),
        boundaries=fd.Boundaries(x=fd.Absorber(cells=4), y=fd.Absorber(cells=4),
                                 z=fd.Absorber(cells=4)),
        sources=(fd.Dipole('d', center=(.15, .45, .05), component='Ey',
                           band=fd.Band(wavelength=3., df=.03), delay=-18.),),
        monitors=(fd.FarField('ff', box=fd.Region(size=(1.4, 1.4, 1.4)), wavelengths=(3.,),
                              propagation='yee'),))


def junction_scene(radius=.2, substrate=2.25, pillar=4.):
    """``tests/test_fd_traced_partner_grad.py``'s scene: a concrete pillar flush on a concrete
    unturned Box substrate (a 'split' junction) and a ranged ball clear of both, in a
    32 x 16 x 20 cell with periodic x and y and PEC z -- no PML, so the partners' materials may
    be traced beside the traced ball."""
    return fd.Scene(domain=fd.Domain(size=(3.2, 1.6, 2.)), resolution=fd.Resolution(dx=.1),
                    boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.PEC()),
                    run=fd.Run(steps=1),
                    structures=(fd.Structure(fd.Box(size=(10., 10., .5),
                                                    center=(0., 0., -.45)),
                                             fd.Material(eps=substrate)),
                                fd.Structure(fd.Cylinder(radius=.2, height=.4,
                                                         center=(0., 0., 0.)),
                                             fd.Material(eps=pillar)),
                                fd.Structure(fd.Sphere(radius=radius, center=(1., 0., .45),
                                                       range=((.75, 1.25), (-.22, .22),
                                                              (.22, .68))),
                                             fd.Material(eps=2.25))))


#: Ball radius, substrate eps, pillar eps.
PARTNERS = np.array((.21, 2.25, 4.))


def partners_loss(v, backend, plan):
    """``sum(be)`` and the cross coefficients' sum: every coefficient the three leaves move.
    ``plan=None`` plans the point itself, as the concrete reference does."""
    current = junction_scene(*v)
    params = fd.lower(current, fd.plan(current) if plan is None else plan, backend)
    xp = runtime(backend).xp
    return xp.sum(params['be']) + sum(xp.sum(c) for c in params['constitutive'].values())


def extras_mode():
    """The regressions' further comparisons: what each equality is within, and the partner
    gradient against central differences."""
    import torch
    warnings.filterwarnings('ignore')
    rtt = runtime('torch')
    t = lambda x: torch.tensor(x, dtype=torch.float64, requires_grad=True)

    def gap(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.max(abs(a - b)) / np.max(abs(b)))
    out = {}
    # A traced background in the closed cell, by fd.run, against the concrete run.
    p = fd.plan(closed_scene())
    ran = fd.run(closed_scene(background=fd.Material(eps=t(1.2))), 'torch', plan=p)
    out['closed_background_E'] = gap(rtt.numpy(ran.final.E.detach()),
                                     fd.run(closed_scene(background=1.2)).final.E)
    # A traced snapped Conductor's sigma beside the sphere, by a direct fd.lower.
    block = lambda sigma: fd.Structure(fd.Region(size=(.1, .1, .1), center=(.7, -.7, -.7)),
                                       fd.Conductor(eps=1., sigma=sigma))
    lowered = fd.lower(closed_scene(extra=(block(t(.3)),)), p, 'torch')
    concrete = closed_scene(extra=(block(.3),))
    want = fd.lower(concrete, fd.plan(concrete))
    out['conductor_be'] = gap(rtt.numpy(lowered['be'].detach()), want['be'])
    out['conductor_ae'] = gap(rtt.numpy(lowered['ae'].detach()), want['ae'])
    # A traced diagonal on a snapped Region clear of the sphere, by fd.run.
    crystal = lambda e: fd.Structure(fd.Region(size=(.1, .1, .1), center=(.7, -.7, -.7)),
                                     fd.Material(eps=e))
    q = fd.plan(closed_scene(extra=(crystal((2., 2., 2.)),)))
    diagonal = (t(2.), t(2.3), t(2.6))
    ran = fd.run(closed_scene(extra=(crystal(diagonal),)), 'torch', plan=q)
    out['diagonal_E'] = gap(rtt.numpy(ran.final.E.detach()),
                            fd.run(closed_scene(extra=(crystal((2., 2.3, 2.6)),))).final.E)
    # The admitted neighbours of three doors, by fd.run: a traced background beside PMLs that
    # carry their own, the dimer a cell clear of its mirror, a traced film in a Dipole scene's
    # PML.
    for label, make, traced, value in (
            ('pml_own_background_E', lambda x: pml_box(background=x), fd.Material(eps=t(1.2)),
             1.2),
            ('dimer_E', lambda x: dimer(eps=x), t(2.25), 2.25),
            ('film_E', lambda x: pml_box(film=x), t(2.3), 2.3)):
        q = fd.plan(make(1. if label.startswith('pml') else 2.25 if label == 'dimer_E'
                         else 2.))
        ran = fd.run(make(traced), 'torch', plan=q)
        out[label] = gap(rtt.numpy(ran.final.E.detach()), fd.run(make(value)).final.E)
    # The partners scene: traced ball radius, substrate and pillar eps, lowered.
    pp = fd.plan(junction_scene())
    point = (.22, 2.4, 3.5)
    lowered = fd.lower(junction_scene(*[t(x) for x in point]), pp, 'torch')
    concrete = junction_scene(*point)
    want = fd.lower(concrete, fd.plan(concrete))
    out['partners_be'] = gap(rtt.numpy(lowered['be'].detach()), want['be'])
    out['partners_cross'] = max(gap(rtt.numpy(lowered['constitutive'][k].detach()),
                                    want['constitutive'][k]) for k in want['constitutive'])
    print(out, flush=True)
    _write('extras', out)
    # The partners' gradient: central differences, each point planned on its own (the
    # junction record belongs to its geometry; the ball's radius does not change it).
    row = dict(point=PARTNERS.tolist(), differences=[])
    for h in (1e-4, 1e-5, 3e-6, 1e-6):
        gradient = [(float(partners_loss(PARTNERS + d, 'numpy', None))
                     - float(partners_loss(PARTNERS - d, 'numpy', None))) / (2 * h)
                    for d in np.eye(3) * h]
        row['differences'].append(dict(step=h, gradient=gradient))
    v = torch.tensor(PARTNERS, dtype=torch.float64, requires_grad=True)
    partners_loss(v, 'torch', pp).backward()
    row['torch'] = v.grad.numpy().tolist()
    reference = np.array(row['torch'])
    row['relative_errors'] = {str(d['step']): float(np.max(abs(np.array(d['gradient'])
                                                              - reference) / abs(reference)))
                              for d in row['differences']}
    out['partners_gradient'] = row
    print(row, flush=True)
    _write('extras', out)
    # JAX: a jitted fd.run at two pillar indices against the concrete runs.
    runtime('jax')
    import jax
    import jax.numpy as jnp
    point = POINTS['cylinder'][0]
    p = fd.plan(scene(point, steps=40))
    compiled = jax.jit(lambda eps: fd.run(scene(point[:3] + (eps,), steps=40), 'jax',
                                          plan=p).final.E)
    fields = {}
    for eps in (2.25, 1.7):
        got = np.asarray(compiled(jnp.asarray(eps)))
        fields[eps] = got
        out[f'jit_E_{eps}'] = gap(got, fd.run(scene(point[:3] + (eps,), steps=40)).final.E)
    out['jit_between_indices'] = gap(fields[1.7], fields[2.25])
    out['jit_compiles'] = compiled._cache_size()
    print(out, flush=True)
    _write('extras', out)


#: Radius, height, centre x, pillar eps, substrate eps.
SUBSTRATE = np.array((.12, .3, .011, 2.25, 2.1))
#: The substrate eps the first orders begin to propagate in at wavelength .95 (period .6):
#: (.95 / .6)^2; with the 8-cell PMLs every such order is refused up to the resolution floor.
GRAZING_ONSET = (.95 / .6) ** 2


def substrate_scene(v=SUBSTRATE, **kwargs):
    """The cylinder cell with the substrate's permittivity ``v[4]`` a leaf too. The bottom PML
    keeps its concrete ``background=GLASS``: the stretch's match, not the medium in it."""
    base = scene(v[:4], **kwargs)
    substrate, pillar = base.structures
    return base.replace(structures=(substrate.replace(material=fd.Material(eps=v[4])),
                                    pillar))


def substrate_objectives(v, backend, plan, **kwargs):
    result = fd.run(substrate_scene(v, **kwargs), backend, plan=plan)
    return (result['T'].orders.incident.T[:, 0, 0].sum(), result['R'].orders.incident.R[:, 0, 0].sum())


def substrate_mode():
    """A traced substrate: equal to the concrete scene, the five-leaf Jacobian, the grazing
    guard against the plan's own refusal across the onset, and settled energy."""
    import dataclasses
    import torch
    warnings.filterwarnings('ignore')
    rtt = runtime('torch')
    out = dict(equal=[], differences=[], onset=GRAZING_ONSET)
    t = lambda x: torch.tensor(x, dtype=torch.float64, requires_grad=True)
    p = fd.plan(substrate_scene(SUBSTRATE, steps=40))
    for v in (tuple(SUBSTRATE), (.127, .21, 0., 2., 2.3), (.1, .25, -.013, 2.4, 1.8)):
        ran = fd.run(substrate_scene([t(x) for x in v], steps=40), 'torch', plan=p)
        want = fd.run(substrate_scene(v, steps=40)).final.E
        out['equal'].append(dict(point=v, E=float(np.max(abs(rtt.numpy(ran.final.E.detach())
                                                              - want)) / np.max(abs(want)))))
        print(out['equal'][-1], flush=True)
    _write('substrate', out)
    point = SUBSTRATE.copy()
    p = fd.plan(substrate_scene(point))

    def values(v):
        return np.array([float(x) for x in substrate_objectives(v, 'numpy', None)])
    for h in (1e-4, 1e-5, 3e-6, 1e-6):
        jac = np.array([(values(point + d) - values(point - d)) / (2 * h)
                        for d in np.eye(5) * h]).T
        out['differences'].append(dict(step=h, jacobian=jac.tolist()))
        _write('substrate', out)
        print('central difference', h, jac.tolist(), flush=True)
    for backend in ('jax', 'torch'):
        started = time.perf_counter()
        runtime(backend)
        if backend == 'jax':
            import jax
            import jax.numpy as jnp
            fn = lambda v: jnp.stack(substrate_objectives(v, 'jax', p))
            jac = np.asarray(jax.jacrev(fn)(jnp.asarray(point)))
        else:
            v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            outs = substrate_objectives(v, 'torch', p)
            jac = np.array([torch.autograd.grad(o, v, retain_graph=True)[0].numpy()
                            for o in outs])
        out[backend] = dict(jacobian=jac.tolist(), seconds=time.perf_counter() - started)
        _write('substrate', out)
    reference = np.array(out['jax']['jacobian'])
    out['relative_errors'] = {str(d['step']): float(np.max(abs(np.array(d['jacobian'])
                                                              - reference) / abs(reference)))
                              for d in out['differences']}
    out['relative_errors_substrate'] = {str(d['step']): float(np.max(
        abs(np.array(d['jacobian'])[:, 4] - reference[:, 4]) / abs(reference[:, 4])))
        for d in out['differences']}
    out['jax_torch_max_abs'] = float(np.max(abs(np.array(out['torch']['jacobian'])
                                                - reference)))
    print('relative errors', out['relative_errors'], 'substrate',
          out['relative_errors_substrate'], 'jax-torch', out['jax_torch_max_abs'], flush=True)
    _write('substrate', out)
    # The guard against the plan's own refusal, across the onset and up to the floor.
    q = fd.plan(substrate_scene(SUBSTRATE, steps=20))
    out['sweep'] = []
    for eps in (2.4, 2.5, 2.506, 2.5069, 2.5070, 2.508, 2.52, 2.55, 2.575):
        traced = substrate_scene([t(x) for x in SUBSTRATE[:4]] + [t(eps)], steps=20)
        stable = bool(fd.run(traced, 'torch', plan=q).diagnostics['stable'])
        try:
            fd.plan(substrate_scene(tuple(SUBSTRATE[:4]) + (eps,), steps=20))
            planned = True
        except ValueError as err:
            planned = 'Grazing order refused' not in str(err) and str(err)[:120]
        out['sweep'].append(dict(eps=eps, stable=stable, planned=planned))
        print(out['sweep'][-1], flush=True)
    _write('substrate', out)
    # Settled: dT + dR for the substrate's eps, 24-cell PMLs, a .05 band, 6000 steps.
    cells, steps, df, h = 24, 6000, .05, 3e-6

    def make(v):
        base = substrate_scene(v)
        pw = base.sources[0].replace(band=fd.Band(wavelength=1., df=df))
        grow = (cells - junction.CELLS) * junction.DX
        return dataclasses.replace(
            base, sources=(pw,), run=fd.Run(steps=steps, courant=.5),
            domain=fd.Domain(size=(.6, .6, 2.1 + 2 * grow), center=(0., 0., .05)),
            boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=(
                fd.PML(cells=cells, wavelength=1., background=junction.GLASS),
                fd.PML(cells=cells, wavelength=1.))))

    def rows(v):
        result = fd.run(make(v), 'numpy')
        return (np.asarray(result['T'].orders.incident.T[:, 0, 0]),
                np.asarray(result['R'].orders.incident.R[:, 0, 0]))
    T, R = rows(SUBSTRATE)
    d = np.eye(5)[4] * h
    plus, minus = rows(SUBSTRATE + d), rows(SUBSTRATE - d)
    dT, dR = (plus[0] - minus[0]) / (2 * h), (plus[1] - minus[1]) / (2 * h)
    out['settled'] = dict(cells=cells, steps=steps, df=df, step=h,
                          energy_residual=(T + R - 1).tolist(), dT0=dT.tolist(),
                          dR0=dR.tolist(), dT_plus_dR=(dT + dR).tolist())
    print(out['settled'], flush=True)
    _write('substrate', out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'equal'
    {'equal': equal_mode, 'measure': measure_mode, 'multi': multi_mode,
     'sphere': sphere_mode, 'settled': settled_mode, 'guards': guards_mode,
     'extras': extras_mode, 'substrate': substrate_mode}[mode](*sys.argv[2:5])
