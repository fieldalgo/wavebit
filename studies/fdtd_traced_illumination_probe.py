"""Measure public geometry gradients under plane-wave illumination before choosing test steps.

Two families. ``'pw'``: a ranged Cylinder pillar in a Bloch cell under an oblique p
PlaneWave (a complex run), read by a transmitted Flux plane normalised by the source's own
intensity, the transmitted zero order and the reflected zero order. ``'tfsf'``: a turned,
ranged Box inside an axis-aligned TFSF box in an open PML scene, read by the scattered flux
through a closed Flux box (normalised by the incident intensity: a cross-section) and by a
FarField box. Every objective goes through a source normalisation, so the plane-wave medium
reads that ``shape_ranges.plane_media`` certifies are on every route.

Modes: ``measure`` (default) writes ``studies/results/fdtd_traced_illumination_probe.json``;
``fields`` compares a traced Torch run's final fields with the concrete NumPy run;
``settled`` checks energy conservation of the pillar's orders and their derivatives in a run
ten times longer.
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
#: Cylinder radius and height; the Box's three sizes. A lone body's centre is no test leaf:
#: every objective here is a power, invariant under translation in the continuum, so its
#: derivative is a small discretisation residue (measured 1e-4 against 0.13 for the
#: pillar's centre x), which a relative budget cannot judge.
POINTS = {'pw': np.array((.12, .3)), 'tfsf': np.array((.26, .22, .24))}
OBJECTIVES = {'pw': ('transmitted_flux', 'T0', 'R0'), 'tfsf': ('cross_section', 'far_field')}
THETA = np.array([20., 75., 150.])
PHI = np.array([10., 130., 250.])


def _band():
    return fd.Band(wavelength=1., df=.15)


def _layer():
    return fd.PML(cells=CELLS, wavelength=1., allow_thin=True)


def pw_scene(v=POINTS['pw']):
    """A 12 x 12-cell Bloch cell, 42 cells along z with eight-cell PMLs. The sheet is at
    z = .45, the reflect plane at .55 and the transmit planes at -.45; the pillar's reserved
    support spans z cells 14..24 of 42, clear of all three planes' layer pairs."""
    pw = fd.PlaneWave('pw', z=.45, direction='-z', theta=25., phi=30., polarization='p',
                      band=_band())
    pillar = fd.Cylinder(radius=v[0], height=v[1], axis='z', center=(.01, -.007, .004),
                         range=((-.145, .145), (-.145, .145), (-.2, .2)))
    return fd.Scene(
        domain=fd.Domain(size=(.6, .6, 2.1), center=(0., 0., .05)),
        boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=_layer()),
        resolution=fd.Resolution(dx=DX), run=fd.Run(steps=600, courant=.5), sources=(pw,),
        structures=(fd.Structure(pillar, fd.Material(eps=2.25)),),
        monitors=(fd.Flux('t', plane=fd.Plane(z=-.45), wavelengths=WAVELENGTHS),
                  fd.Diffraction('T', z=-.45, side='transmit', wavelengths=WAVELENGTHS,
                                 orders=0),
                  fd.Diffraction('R', z=.55, side='reflect', wavelengths=WAVELENGTHS,
                                 orders=0)))


def tfsf_scene(v=POINTS['tfsf']):
    """A 38-cubed open scene: a 12-cell TFSF box, a 16-cell scattered-flux box and an 18-cell
    far-field box, all centred; the turned Box's reserved support spans cells 14..23."""
    tfsf = fd.TFSF('t', box=fd.Region(size=(.6, .6, .6)), direction='-z', polarization='x',
                   band=_band())
    body = fd.Box(size=(v[0], v[1], v[2]), angle=20., center=(.01, -.012, .007),
                  range=((-.19, .19), (-.19, .19), (-.15, .15)))
    return fd.Scene(
        domain=fd.Domain(size=(1.9,) * 3), boundaries=fd.Boundaries.open(_layer()),
        resolution=fd.Resolution(dx=DX), run=fd.Run(steps=500, courant=.5), sources=(tfsf,),
        structures=(fd.Structure(body, fd.Material(eps=2.25)),),
        monitors=(fd.Flux('s', box=fd.Region(size=(.8,) * 3), wavelengths=WAVELENGTHS),
                  fd.FarField('ff', box=fd.Region(size=(.9,) * 3), wavelengths=WAVELENGTHS)))


SCENES = {'pw': pw_scene, 'tfsf': tfsf_scene}


def scene(family, v=None):
    return SCENES[family](POINTS[family] if v is None else v)


def objectives(family, v, backend, plan):
    """The family's objectives as a tuple of backend scalars, differentiable in ``v``."""
    result = fd.run(scene(family, v), backend, plan=plan)
    if family == 'pw':
        intensity = result.sources['pw'].intensity(WAVELENGTHS)
        area = .6 * .6
        transmitted = (-result['t'].flux / (intensity * area)).sum()
        return (transmitted, result['T'].orders.incident.T[:, 0, 0].sum(),
                result['R'].orders.incident.R[:, 0, 0].sum())
    intensity = result.sources['t'].intensity(WAVELENGTHS)
    return ((result['s'].flux / intensity).sum(), result['ff'].intensity(THETA, PHI).sum())


def numpy_values(family, v, plan):
    return np.array([float(x) for x in objectives(family, v, 'numpy', plan)])


def central(family, plan, h, point=None):
    """``(objective, leaf)`` central differences on NumPy at relative step ``h``."""
    point = POINTS[family] if point is None else point
    rows = []
    for d in np.eye(len(point)) * h:
        rows.append((numpy_values(family, point + d, plan)
                     - numpy_values(family, point - d, plan)) / (2 * h))
    return np.array(rows).T


def jacobian(family, backend, plan, point=None):
    """``(values, jacobian)`` by reverse mode on ``backend``."""
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


def measure(families=('pw', 'tfsf')):
    warnings.filterwarnings('ignore')
    path = RESULTS / 'fdtd_traced_illumination_probe.json'
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
    """A traced Torch run (every leaf requires grad) against the concrete NumPy run."""
    import torch
    warnings.filterwarnings('ignore')
    out = {}
    for family in ('pw', 'tfsf'):
        p = fd.plan(scene(family))
        at = POINTS[family] + np.array((.004, -.006, .003))[:len(POINTS[family])]
        traced = fd.run(scene(family, torch.tensor(at, dtype=torch.float64,
                                                   requires_grad=True)), 'torch', plan=p)
        direct = fd.run(scene(family, at), 'numpy', plan=p)
        rt = runtime('torch')
        out[family] = {f: float(np.max(abs(rt.numpy(getattr(traced.final, f).detach())
                                           - getattr(direct.final, f)))
                                / np.max(abs(getattr(direct.final, f)))) for f in ('E', 'H')}
        print(family, out[family], flush=True)
    (RESULTS / 'fdtd_traced_illumination_fields.json').write_text(
        json.dumps(out, indent=2) + '\n', encoding='utf-8')


def settled(steps=6000, incidence='oblique', cells=CELLS, df=None, gap=0., h=3e-6):
    """The pillar's derivatives in a longer run: a physical check, not a test.

    The regression scenes stop at 600 steps, where the zero orders' DFT is not yet the steady
    state (T + R summed over the three wavelengths is 2.9972, not 3). Once the run has rung
    down, the lossless cell must conserve energy per wavelength, T + R = 1, and so must its
    derivative, dT + dR = 0. ``incidence='normal'`` puts the same pillar under a normally
    incident wave (Periodic faces, a real run) and ``cells`` sets both z PMLs, keeping the
    interior; ``df`` narrows the band (the probe's own is .15); ``gap`` moves the sheet, the
    monitor planes and the PMLs that much further from the pillar. The derivatives are NumPy
    central differences at the regression step, which the measure mode shows reverse mode
    reproduces to 1.8e-9.
    """
    import dataclasses
    warnings.filterwarnings('ignore')

    def make(v):
        base = pw_scene(v)
        pw = base.sources[0]
        if df is not None:
            pw = pw.replace(band=fd.Band(wavelength=1., df=df))
        if incidence == 'normal':
            pw = fd.PlaneWave('pw', z=.45, direction='-z', polarization='p', band=pw.band)
        pw = pw.replace(z=.45 + gap)
        monitors = (fd.Flux('t', plane=fd.Plane(z=-.45 - gap), wavelengths=WAVELENGTHS),
                    fd.Diffraction('T', z=-.45 - gap, side='transmit', wavelengths=WAVELENGTHS,
                                   orders=0),
                    fd.Diffraction('R', z=.55 + gap, side='reflect', wavelengths=WAVELENGTHS,
                                   orders=0))
        layer = fd.PML(cells=cells, wavelength=1., allow_thin=cells < 12)
        return dataclasses.replace(
            base, sources=(pw,), run=fd.Run(steps=steps, courant=.5), monitors=monitors,
            domain=fd.Domain(size=(.6, .6, 2.1 + 2 * (cells - CELLS) * DX + 2 * gap),
                             center=(0., 0., .05)),
            boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=layer))
    p = fd.plan(make(POINTS['pw']))

    def rows(v):
        result = fd.run(make(v), 'numpy', plan=p)
        intensity = result.sources['pw'].intensity(WAVELENGTHS)
        return (np.asarray(-result['t'].flux / (intensity * .36)),
                np.asarray(result['T'].orders.incident.T[:, 0, 0]),
                np.asarray(result['R'].orders.incident.R[:, 0, 0]),
                result.diagnostics['decay'])
    flux, T, R, decay = rows(POINTS['pw'])
    out = dict(steps=steps, incidence=incidence, cells=cells, df=df, gap=gap, dtype=p.dtype,
               step=h,
               wavelengths=WAVELENGTHS, decay=decay, flux_T=flux.tolist(), T0=T.tolist(),
               R0=R.tolist(), energy_residual=(T + R - 1).tolist(),
               flux_minus_T0=(flux - T).tolist(), derivatives=[])
    for i, d in enumerate(np.eye(2) * h):
        plus, minus = rows(POINTS['pw'] + d), rows(POINTS['pw'] - d)
        dT = (plus[1] - minus[1]) / (2 * h)
        dR = (plus[2] - minus[2]) / (2 * h)
        out['derivatives'].append(dict(leaf=('radius', 'height')[i], dT0=dT.tolist(),
                                       dR0=dR.tolist(), dT_plus_dR=(dT + dR).tolist()))
        print(out['derivatives'][-1], flush=True)
    print(out, flush=True)
    suffix = '' if (incidence, cells) == ('oblique', CELLS) else f'_{incidence}_{cells}'
    suffix += '' if df is None else f'_df{df:g}'
    suffix += '' if not gap else f'_gap{gap:g}'
    (RESULTS / f'fdtd_traced_illumination_settled_{steps}{suffix}.json').write_text(
        json.dumps(out, indent=2, default=str) + '\n', encoding='utf-8')


def empty(steps=6000, df=.05):
    """The settled cell without its pillar: what the zero orders read with nothing to scatter.

    T = 1 and R = 0 exactly in the continuum. An R here is the sheet's own leakage into the
    reflect side and T - 1 the normalisation's error, for each incidence.
    """
    import dataclasses
    warnings.filterwarnings('ignore')
    out = dict(steps=steps, df=df, rows={})
    for incidence, theta in (('oblique', 25.), ('normal', 0.)):
        pw = fd.PlaneWave('pw', z=.45, direction='-z', theta=theta, phi=30., polarization='p',
                          band=fd.Band(wavelength=1., df=df))
        base = dataclasses.replace(
            pw_scene(), structures=(), sources=(pw,), run=fd.Run(steps=steps, courant=.5),
            boundaries=fd.Boundaries(**fd.Bloch.from_source(pw), z=_layer()))
        result = fd.run(base, 'numpy')
        T = np.asarray(result['T'].orders.incident.T[:, 0, 0])
        R = np.asarray(result['R'].orders.incident.R[:, 0, 0])
        out['rows'][incidence] = dict(T0=T.tolist(), R0=R.tolist(), T_minus_1=(T - 1).tolist(),
                                      decay=result.diagnostics['decay'])
        print(incidence, out['rows'][incidence], flush=True)
    (RESULTS / f'fdtd_traced_illumination_empty_{steps}_df{df:g}.json').write_text(
        json.dumps(out, indent=2, default=str) + '\n', encoding='utf-8')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'measure'
    if mode == 'measure':
        measure(tuple(sys.argv[2:]) or ('pw', 'tfsf'))
    elif mode == 'fields':
        fields()
    elif mode == 'empty':
        empty()
    elif mode == 'settled':
        settled(*(int(a) if a.isdigit() else a for a in sys.argv[2:5]),
                *(float(a) for a in sys.argv[5:7]))
    else:
        raise SystemExit(f'unknown mode {mode!r}')
