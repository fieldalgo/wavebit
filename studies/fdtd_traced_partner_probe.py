"""Measure geometry gradients of ranged bodies beside concrete partners without a range.

Two families, both a 12 x 12 x 42-cell Bloch cell at ``dx = .05`` with eight-cell PMLs and
600 steps. ``'substrate'``: a ranged Cylinder pillar three cells above a cell-snapped glass
``fd.Slab`` that runs into the bottom PML, under a normally incident s PlaneWave (a real run);
read by the zero order transmitted into the glass and the reflected one. ``'mirror'``: a
ranged Sphere above a finite Drude-metal ``fd.Slab`` film, under an oblique p PlaneWave (a
complex run); read by the reflected zero order and the flux through a plane below the film,
normalised by the incident intensity. Leaves: the pillar's radius and height; the ball's radius
and its height above the mirror (no translation invariance there).

Modes: ``measure`` (default) writes ``studies/results/fdtd_traced_partner_probe.json``;
``fields`` compares a traced Torch run with the concrete NumPy run and with the same scene with
its ranges removed; ``coefficients`` checks that a ranged ball's coefficient change adds to
concrete partners' (an unranged Sphere, a Region and a Slab) exactly.
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
WAVELENGTHS = (.95, 1., 1.05)
#: The ball's centre is .103, not .1: at .1 it lies exactly on a node plane, where reverse mode
#: gave a centre derivative of R0 of .1196 (JAX) and .1641 (Torch) against a central difference
#: of .1267 that is the same at every step from 1e-4 to 1e-6, while at .1013 and .0987 all three
#: agree to eight digits (``fdtd_traced_partner_controls/center_tie_diagnostic.log``). That
#: defect predates this probe and is not what it measures.
POINTS = {'substrate': np.array((.12, .3)), 'mirror': np.array((.12, .103))}
OBJECTIVES = {'substrate': ('T0', 'R0'), 'mirror': ('R0', 'transmitted_flux')}
#: The pillar's and the ball's reservation: z cells 17..27 of 42, three cells above the
#: substrate's last cell (14) and the film's (14), so the partner widened by one cell (to 15)
#: meets the traced halo (16..28) at an index boundary without sharing a cell.
RANGE = ((-.145, .145), (-.145, .145), (-.095, .305))
METAL = fd.Dispersive(eps_inf=1., poles=(fd.Drude(omega_p=20., gamma=.5),))


def _band():
    return fd.Band(wavelength=1., df=.15)


def _cell(pw, structures, monitors):
    return fd.Scene(
        domain=fd.Domain(size=(.6, .6, 2.1), center=(0., 0., .05)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                 z=fd.PML(cells=CELLS, wavelength=1., allow_thin=True)),
        resolution=fd.Resolution(dx=DX), run=fd.Run(steps=600, courant=.5), sources=(pw,),
        structures=structures, monitors=monitors)


def substrate_scene(v=POINTS['substrate'], ranged=True):
    pw = fd.PlaneWave('pw', z=.55, direction='-z', polarization='s', band=_band())
    pillar = fd.Cylinder(radius=v[0], height=v[1], axis='z', center=(.01, -.007, .06),
                         range=RANGE if ranged else None)
    return _cell(pw, (fd.Structure(fd.Slab(z=(-1.5, -.25)), fd.Material(eps=2.1)),
                      fd.Structure(pillar, fd.Material(eps=2.25))),
                 (fd.Diffraction('T', z=-.5, side='transmit', wavelengths=WAVELENGTHS,
                                 orders=0),
                  fd.Diffraction('R', z=.65, side='reflect', wavelengths=WAVELENGTHS,
                                 orders=0)))


def mirror_scene(v=POINTS['mirror'], ranged=True):
    pw = fd.PlaneWave('pw', z=.55, direction='-z', theta=25., phi=30., polarization='p',
                      band=_band())
    ball = fd.Sphere(radius=v[0], center=(.01, -.007, v[1]), range=RANGE if ranged else None)
    return _cell(pw, (fd.Structure(fd.Slab(z=(-.4, -.25)), METAL),
                      fd.Structure(ball, fd.Material(eps=2.25))),
                 (fd.Flux('t', plane=fd.Plane(z=-.5), wavelengths=WAVELENGTHS),
                  fd.Diffraction('R', z=.65, side='reflect', wavelengths=WAVELENGTHS,
                                 orders=0)))


SCENES = {'substrate': substrate_scene, 'mirror': mirror_scene}


def scene(family, v=None, ranged=True):
    return SCENES[family](POINTS[family] if v is None else v, ranged)


def objectives(family, v, backend, plan):
    """The family's objectives as a tuple of backend scalars, differentiable in ``v``."""
    result = fd.run(scene(family, v), backend, plan=plan)
    reflected = result['R'].orders.given.R[:, 0, 0].sum()
    if family == 'substrate':
        return (result['T'].orders.given.T[:, 0, 0].sum(), reflected)
    intensity = result.sources['pw'].intensity(WAVELENGTHS)
    return (reflected, (-result['t'].flux / (intensity * .36)).sum())


def numpy_values(family, v, plan):
    return np.array([float(x) for x in objectives(family, v, 'numpy', plan)])


def central(family, plan, h, point=None):
    point = POINTS[family] if point is None else point
    rows = [(numpy_values(family, point + d, plan) - numpy_values(family, point - d, plan))
            / (2 * h) for d in np.eye(len(point)) * h]
    return np.array(rows).T


def jacobian(family, backend, plan, point=None):
    point = POINTS[family] if point is None else point
    runtime(backend)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp

        def fn(v):
            return jnp.stack(objectives(family, v, 'jax', plan))
        x = jnp.asarray(point)
        return np.asarray(fn(x)), np.asarray(jax.jacrev(fn)(x))
    import torch
    v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    out = objectives(family, v, 'torch', plan)
    rows = [torch.autograd.grad(o, v, retain_graph=True)[0].detach().numpy() for o in out]
    return np.array([float(o.detach()) for o in out]), np.array(rows)


def measure(families=('substrate', 'mirror')):
    warnings.filterwarnings('ignore')
    path = RESULTS / 'fdtd_traced_partner_probe.json'
    record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for family in families:
        started = time.perf_counter()
        p = fd.plan(scene(family))
        row = dict(point=POINTS[family].tolist(), objectives=OBJECTIVES[family],
                   supports=[s.shape.support(p.grid) for s in scene(family).structures],
                   cells=list(p.grid.n), steps=p.steps, dtype=p.dtype)
        row['numpy_values'] = numpy_values(family, POINTS[family], p).tolist()
        row['numpy_seconds'] = time.perf_counter() - started
        row['differences'] = []
        record[family] = row
        for h in (1e-4, 1e-5, 3e-6, 1e-6):
            row['differences'].append(dict(step=h, jacobian=central(family, p, h).tolist()))
            path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            print(family, 'central difference', h, row['differences'][-1], flush=True)
        for backend in ('jax', 'torch'):
            started = time.perf_counter()
            values, jac = jacobian(family, backend, p)
            row[backend] = dict(values=values.tolist(), jacobian=jac.tolist(),
                                seconds=time.perf_counter() - started)
            path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            print(family, backend, row[backend], flush=True)
        reference = np.array(row['jax']['jacobian'])
        row['relative_errors'] = {
            str(d['step']): float(np.max(abs(np.array(d['jacobian']) - reference)
                                         / abs(reference)))
            for d in row['differences']}
        row['jax_torch_max_abs'] = float(np.max(abs(np.array(row['torch']['jacobian'])
                                                    - reference)))
        path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        print(family, 'relative errors', row['relative_errors'], 'jax-torch',
              row['jax_torch_max_abs'], flush=True)


def fields():
    """A traced Torch run against the concrete NumPy run, and NumPy with the ranges removed."""
    import torch
    warnings.filterwarnings('ignore')
    out = {}
    for family in ('substrate', 'mirror'):
        p = fd.plan(scene(family))
        at = POINTS[family] + np.array((.004, -.006))
        traced = fd.run(scene(family, torch.tensor(at, dtype=torch.float64,
                                                   requires_grad=True)), 'torch', plan=p)
        direct = fd.run(scene(family, at), 'numpy', plan=p)
        plain = fd.run(scene(family, at, ranged=False), 'numpy')
        rt = runtime('torch')
        out[family] = {}
        for f in ('E', 'H'):
            want = getattr(direct.final, f)
            peak = np.max(abs(want))
            out[family][f] = dict(
                torch_relative=float(np.max(abs(rt.numpy(getattr(traced.final, f).detach())
                                                - want)) / peak),
                unranged_relative=float(np.max(abs(getattr(plain.final, f) - want)) / peak))
        print(family, out[family], flush=True)
    (RESULTS / 'fdtd_traced_partner_fields.json').write_text(
        json.dumps(out, indent=2) + '\n', encoding='utf-8')


def partners_scene(radius=.22):
    """A ranged ball beside three concrete partners in a 32 x 16 x 16 PEC box: an unranged
    smoothed Sphere, a cell-snapped Region and a cell-snapped Slab, each a cell or more off
    the ball's halo."""
    ball = fd.Sphere(radius=radius, center=(-.75, .02, -.03),
                     range=((-1.1, -.4), (-.36, .36), (-.36, .36)))
    return fd.Scene(
        domain=fd.Domain(size=(3.2, 1.6, 1.6)), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()), run=fd.Run(steps=32),
        structures=(fd.Structure(ball, fd.Material(eps=2.25, mu=1.2)),
                    fd.Structure(fd.Sphere(radius=.23, center=(.55, -.01, .02)),
                                 fd.Material(eps=2.)),
                    fd.Structure(fd.Region(size=(.3, .4, .3), center=(1.25, .1, -.2)),
                                 fd.Material(eps=3.)),
                    fd.Structure(fd.Slab(x=(1.45, 1.6)), fd.Material(eps=1.5))),
        sources=(fd.Dipole('drive', center=(-.85, -.55, -.02), component='Ey',
                           band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def coefficients():
    """Combined update coefficients against the background plus each structure's own change."""
    warnings.filterwarnings('ignore')
    rt = runtime('numpy')
    current = partners_scene(.237)

    def lower(s):
        return fd.lower(s, fd.plan(s), rt)
    combined = lower(current)
    base = lower(current.replace(structures=()))
    singles = [lower(current.replace(structures=(s,))) for s in current.structures]
    out = {}
    for key in ('be', 'bh'):
        want = base[key] + sum(one[key] - base[key] for one in singles)
        out[key] = float(np.max(abs(np.asarray(combined[key]) - want)) / np.max(abs(want)))
    print(out, flush=True)
    (RESULTS / 'fdtd_traced_partner_coefficients.json').write_text(
        json.dumps(out, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'measure'
    if mode == 'measure':
        measure(tuple(sys.argv[2:]) or ('substrate', 'mirror'))
    elif mode == 'fields':
        fields()
    elif mode == 'coefficients':
        coefficients()
    else:
        raise SystemExit(f'unknown mode {mode!r}')
