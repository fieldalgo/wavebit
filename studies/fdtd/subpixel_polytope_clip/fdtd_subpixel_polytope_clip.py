"""Stage 4, C10 -- does the polytope clip converge where it writes? (registered Gate 2)

The scenes, grids and pass rules are registered in
``docs/validation/fdtd/fd-polytope-clip.md`` ("Registered acceptance", and "Amendment 1",
both committed before any run of this script measured a field). This file computes; that page
records what the numbers decided.

Arms
----
``stack``   Arm L: two smoothed, unturned Boxes spanning the transverse cell -- ``eps`` 6 on
            ``z in [-0.30, 0]`` and ``eps`` 2.25 above a vacuum gap of 0.003 -- whose gap is
            thinner than every grid, so the clip writes at every grid. Zeroth-order reflectance
            against the continuum transfer-matrix reflectance of the same stack, normal
            incidence and p polarisation at 30 degrees. ``--gap 0.2`` is the control L0.
``dimer``   Arm D: the S7 dimer (radius 0.25, centres at x = -/+0.2625, z = 0.021), ``eps`` 12
            and 2.25 in vacuum, scattering cross-section from a closed flux box around a TFSF
            source, one narrow-band run per wavelength.
``plan``    Plans every (grid, wavelength) of both arms and prints which plan and how many
            electric voxels the clip writes. No field is computed.

Run from the repository root, e.g.::

    python studies/fdtd/subpixel_polytope_clip/fdtd_subpixel_polytope_clip.py plan
    python studies/fdtd/subpixel_polytope_clip/fdtd_subpixel_polytope_clip.py stack --pol s
    python studies/fdtd/subpixel_polytope_clip/fdtd_subpixel_polytope_clip.py dimer --cells 40

Each run appends JSON rows to ``studies/fdtd/subpixel_polytope_clip/records/`` (one file per
arm); ``report`` reads them and prints the registered statistics.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import argparse
import json
import math
import sys
import time
import warnings

ROOT = repository_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np                                                               # noqa: E402
import wavebit.fdtd._legacy as fd                                                 # noqa: E402

RECORDS = ROOT / 'studies' / 'fdtd' / 'subpixel_polytope_clip' / 'records'
SOURCE = None

# ---- Arm L ---------------------------------------------------------------------------------
STACK_GRIDS = (28, 40, 56, 80, 112, 160)               # Amendment 1 (was 20 .. 160)
STACK_WAVELENGTHS = tuple(np.round(np.linspace(0.8, 1.2, 21), 12))
STACK_CENTRE = 0.0137                                   # the domain's z centre: the faces fall
STACK_THETA = 30.0                                      # at a different sub-cell place per grid
EPS_A, EPS_B = 6.0, 2.25

# ---- Arm D ---------------------------------------------------------------------------------
DIMER_GRIDS = (20, 24, 28, 32, 40, 56)
DIMER_REFERENCE = 80
DIMER_WAVELENGTHS = tuple(np.round(np.linspace(1.0, 2.0, 11), 12))
DIMER_CENTRES = ((-0.2625, 0.0, 0.021), (0.2625, 0.0, 0.021))
DIMER_EPS = (12.0, 2.25)
DIMER_RADIUS = 0.25


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


# ---------------------------------------------------------------------------------------------
# Arm L
# ---------------------------------------------------------------------------------------------

def stack_layers(gap):
    """``[(z_lo, z_hi, eps), ...]`` bottom to top, vacuum outside."""
    return ((-0.30, 0.0, EPS_A), (gap, gap + 0.2, EPS_B))


def stack_scene(cells, pol, gap=0.003, periods=80.):
    dx = 1.0 / cells
    pml = max(12, int(round(0.5 / dx)))
    size_z = 3.0 + 2 * pml * dx
    structures = tuple(fd.Structure(fd.Box(size=(4., 4., hi - lo), center=(0., 0., (lo + hi) / 2)),
                                    eps) for lo, hi, eps in stack_layers(gap))
    band = fd.Band(wavelength=1.0, df=0.1)
    if pol == 's':
        source = fd.PlaneWave('pw', z=0.9, direction='-z', polarization='s', band=band)
        faces = dict(x=fd.Periodic(), y=fd.Periodic())
    else:
        source = fd.PlaneWave('pw', z=0.9, direction='-z', theta=STACK_THETA, polarization='p',
                              band=band)
        kx = 2 * math.pi * math.sin(math.radians(STACK_THETA))
        faces = dict(x=fd.Axis1D(k=kx), y=fd.Axis1D(k=0.0))
    return fd.Scene(domain=fd.Domain(size=(dx, dx, size_z), center=(0., 0., STACK_CENTRE)),
                    boundaries=fd.Boundaries(**faces, z=fd.PML(cells=pml)),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                    structures=structures, sources=(source,),
                    monitors=(fd.Diffraction('r', z=1.1, side='reflect',
                                             wavelengths=STACK_WAVELENGTHS),))


def transfer_reflectance(wavelength, pol, gap):
    """Continuum reflectance of the stack, light incident from +z, fixed ``kx`` (p) or 0 (s)."""
    k0 = 2 * math.pi / wavelength
    kx = 0.0 if pol == 's' else 2 * math.pi * math.sin(math.radians(STACK_THETA))
    # top to bottom: vacuum, B, gap, A, vacuum
    (a_lo, a_hi, ea), (b_lo, b_hi, eb) = stack_layers(gap)
    layers = [(1.0, None), (eb, b_hi - b_lo), (1.0, b_lo - a_hi), (ea, a_hi - a_lo), (1.0, None)]

    def kz(eps):
        return np.sqrt(complex(eps * k0 * k0 - kx * kx))

    def admittance(eps):
        return kz(eps) / (eps if pol == 'p' else 1.0)
    # characteristic matrices (Born & Wolf), from the incidence side down
    m = np.eye(2, dtype=complex)
    for eps, d in layers[1:-1]:
        phase = kz(eps) * d
        y = admittance(eps)
        m = m @ np.array([[np.cos(phase), -1j * np.sin(phase) / y],
                          [-1j * y * np.sin(phase), np.cos(phase)]])
    y0, ys = admittance(layers[0][0]), admittance(layers[-1][0])
    b, c = m @ np.array([1.0, ys])
    r = (y0 * b - c) / (y0 * b + c)
    return float(abs(r) ** 2)


def run_stack(cells, pol, gap, backend):
    scene = stack_scene(cells, pol, gap)
    plan = quiet(fd.plan, scene)
    t0 = time.perf_counter()
    res = quiet(fd.run, scene, plan=plan, backend=backend)
    reflect = np.asarray(res['r'].orders.R)
    reflect = np.real(np.nansum(reflect.reshape(len(STACK_WAVELENGTHS), -1), axis=1))
    exact = [transfer_reflectance(w, pol, gap) for w in STACK_WAVELENGTHS]
    clip = sum(r[1] for record in plan.clips for r in record.counts if r[0] != 'E*')
    decay = res.diagnostics.get('decay') or {}
    return dict(arm='L0' if gap > 0.1 else 'L', cells=cells, pol=pol, gap=gap, backend=backend,
                wavelengths=list(STACK_WAVELENGTHS), R=[float(v) for v in reflect], exact=exact,
                clip_voxels=int(clip), steps=int(plan.steps), stable=res.diagnostics.get('stable'),
                decay={k: (None if v is None else float(v)) for k, v in decay.items()},
                wall=time.perf_counter() - t0)


# ---------------------------------------------------------------------------------------------
# Arm D
# ---------------------------------------------------------------------------------------------

#: Amendment 2: narrow-band runs (one wavelength) at the clip-writing grids, a broadband run
#: covering all eleven wavelengths where it plans. Amendment 3: every run stops by the
#: engine's state-amplitude decay rule inside a hard time cap, not after a fixed length.
BROAD_BAND = dict(wavelength=4.0 / 3.0, df=0.12)
BROAD_GRIDS = (40, 56, 80)
DECAY_THRESHOLD, DECAY_CAP = 1e-3, 600.0


def dimer_scene(cells, wavelength=None, pml=12):
    """``wavelength`` a number: the narrow band at it; ``None``: the broadband run."""
    dx = 1.0 / cells
    centre = (0.0, 0.0, 0.021)
    tfsf = (1.3, 0.8, 0.8)
    flux = (1.6, 1.1, 1.1)
    size = tuple(f + 2 * (0.2 + pml * dx) for f in flux)
    structures = tuple(fd.Structure(fd.Sphere(radius=DIMER_RADIUS, center=c), e)
                       for c, e in zip(DIMER_CENTRES, DIMER_EPS))
    if wavelength is None:
        band, wavelengths = fd.Band(**BROAD_BAND), DIMER_WAVELENGTHS
    else:
        band = fd.Band(wavelength=wavelength, df=0.04 / wavelength)
        wavelengths = (wavelength,)
    window = 2.0 * max(wavelengths)
    run = fd.Run(time=DECAY_CAP, decay=fd.Decay(threshold=DECAY_THRESHOLD, window=window))
    return fd.Scene(domain=fd.Domain(size=size, center=centre),
                    boundaries=fd.Boundaries.open(fd.PML(cells=pml, reflection=1e-8)),
                    resolution=fd.Resolution(dx=dx), run=run,
                    structures=structures,
                    sources=(fd.TFSF('t', box=fd.Region(size=tfsf, center=centre), direction='+z',
                                     polarization='x', band=band),),
                    monitors=(fd.Flux('sca', box=fd.Region(size=flux, center=centre),
                                      wavelengths=tuple(wavelengths)),))


def run_dimer(cells, wavelength, backend):
    """Rows, one per wavelength the run measured."""
    from tests.fixtures.fdtd.fd_helpers import to_numpy
    scene = dimer_scene(cells, wavelength)
    plan = quiet(fd.plan, scene)
    t0 = time.perf_counter()
    res = quiet(fd.run, scene, plan=plan, backend=backend)
    wavelengths = DIMER_WAVELENGTHS if wavelength is None else (wavelength,)
    incident = np.asarray(to_numpy(res.sources['t'].intensity(tuple(wavelengths))), float)
    flux = np.asarray(to_numpy(res['sca'].flux), float)
    clip = sum(r[1] for record in plan.clips for r in record.counts if r[0] != 'E*')
    decay = res.diagnostics.get('decay') or {}
    stop = res.diagnostics.get('stopping')
    common = dict(arm='D', cells=cells, backend=backend, executed_steps=int(res.plan.steps),
                  stopped=None if stop is None else str(stop)[:300],
                  instrument='broad' if wavelength is None else 'narrow',
                  clip_voxels=int(clip), steps=int(plan.steps), cells_total=int(plan.cells),
                  stable=res.diagnostics.get('stable'),
                  decay={k: (None if v is None else float(v)) for k, v in decay.items()},
                  wall=time.perf_counter() - t0)
    return [dict(common, wavelength=float(w), sigma=float(f / i))
            for w, f, i in zip(wavelengths, flux, incident)]


# ---------------------------------------------------------------------------------------------
# Feasibility: plans only
# ---------------------------------------------------------------------------------------------

def plan_table():
    rows = []
    for pol in ('s', 'p'):
        for cells in (20,) + STACK_GRIDS + (224, 320):
            try:
                plan = quiet(fd.plan, stack_scene(cells, pol))
                clip = sum(r[1] for c in plan.clips for r in c.counts if r[0] != 'E*')
                rows.append(f'L {pol} {cells:4d} plans   clip voxels {clip}')
            except ValueError as err:
                rows.append(f'L {pol} {cells:4d} REFUSED {str(err)[:110]}')
    for cells in DIMER_GRIDS + (DIMER_REFERENCE,):
        ok = []
        clip = None
        for w in DIMER_WAVELENGTHS:
            try:
                plan = quiet(fd.plan, dimer_scene(cells, w))
                ok.append(w)
                clip = sum(r[1] for c in plan.clips for r in c.counts if r[0] != 'E*')
            except ValueError:
                pass
        rows.append(f'D   {cells:4d} plans at {len(ok)}/11 wavelengths '
                    f'{[float(w) for w in ok]}  clip voxels {clip}')
    return rows


def source_digest():
    """SHA-256 over every ``src/wavebit`` Python file's path and bytes: what this row ran on."""
    import hashlib
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'src' / 'wavebit').rglob('*.py')):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fit(h, e):
    """Least-squares ``log e = log C + p log h``: ``(p, C, scatter)``, scatter as registered."""
    x, y = np.log(np.asarray(h, float)), np.log(np.asarray(e, float))
    p, logc = np.polyfit(x, y, 1)
    scatter = float(np.exp(np.max(np.abs(y - (p * x + logc)))) - 1)
    return float(p), float(np.exp(logc)), scatter


def _rows(name):
    path = RECORDS / f'{name}.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def report():
    out = []
    stack = _rows('stack')
    for arm in ('L', 'L0'):
        for pol in ('s', 'p'):
            rows = {r['cells']: r for r in stack if r['arm'] == arm and r['pol'] == pol}
            grids = [c for c in STACK_GRIDS if c in rows]
            if len(grids) < 2:
                continue
            h = [1.0 / c for c in grids]
            err = np.array([[abs(a - b) for a, b in zip(rows[c]['R'], rows[c]['exact'])]
                            for c in grids])
            band = np.sqrt(np.mean(err ** 2, axis=1))
            p, _, scatter = _fit(h, band)
            verdict = '' if arm == 'L0' else ('  PASS' if p >= 1.8 else '  FAIL')
            out.append(f'arm {arm} pol {pol}: band-RMS order {p:.4f} scatter {100 * scatter:.2f}%'
                       f' over {grids}{verdict}')
            out.append('  cells  clip  band-RMS |R-R_exact|  stable  decay')
            for c, b in zip(grids, band):
                decay = max((v for v in (rows[c].get('decay') or {}).values() if v is not None),
                            default=None)
                out.append(f'  {c:5d} {rows[c]["clip_voxels"]:5d}  {b:.4e}         '
                           f'{rows[c]["stable"]}  {decay:.1e}')
            orders = []
            for k, w in enumerate(STACK_WAVELENGTHS):
                orders.append(_fit(h, err[:, k])[0])
            out.append('  per-wavelength order: ' + ' '.join(
                f'{w:.2f}:{o:.2f}' for w, o in zip(STACK_WAVELENGTHS, orders)))
            out.append('  finest-grid error per wavelength: ' + ' '.join(
                f'{e:.1e}' for e in err[-1]))
    dimer = [r for r in _rows('dimer') if 'sigma' in r]
    refused = [r for r in _rows('dimer') if 'refused' in r]
    if dimer:
        sig = {(r['cells'], r['wavelength']): r for r in dimer}
        out.append('arm D: relative error of the scattering cross-section against 80 cells')
        writes = {r['cells']: r['clip_voxels'] for r in dimer}
        out.append('  grid clip-voxels: ' + ', '.join(f'{c}:{writes.get(c)}' for c in DIMER_GRIDS))
        out.append('  refused (grid, wavelength): ' + ', '.join(
            f'({r["cells"]}, {r["wavelength"]})' for r in refused))
        verdicts = []
        header = '  wavelength ' + ' '.join(f'{c:>9d}' for c in DIMER_GRIDS) + '   order  b-check'
        out.append(header)
        common = []
        for w in DIMER_WAVELENGTHS:
            ref = sig.get((DIMER_REFERENCE, w))
            if ref is None:
                continue
            row, errs = [], {}
            for c in DIMER_GRIDS:
                r = sig.get((c, w))
                if r is None:
                    row.append(f'{"--":>9s}')
                    continue
                e = abs(r['sigma'] - ref['sigma']) / abs(ref['sigma'])
                errs[c] = e
                settled = max((r.get('decay') or {}).values(), default=0.0) < 1e-4
                row.append(f'{e:9.2e}' if settled else f'{e:8.2e}*')
            inactive = [c for c in errs if writes.get(c) == 0]
            active = [c for c in errs if writes.get(c)]
            order = _fit([1 / c for c in errs], list(errs.values()))[0] if len(errs) > 1 else None
            check = ''
            if len(inactive) >= 2:
                p, cst, _ = _fit([1 / c for c in inactive], [errs[c] for c in inactive])
                ratios = {c: errs[c] / (cst * (1 / c) ** p) for c in active}
                ok = all(0.5 <= v <= 2.0 for v in ratios.values())
                verdicts.append(ok)
                check = ('ok ' if ok else 'FAIL ') + ' '.join(f'{c}:{v:.2f}' for c, v in
                                                            ratios.items())
            if all(c in errs for c in DIMER_GRIDS):
                common.append(w)
            out.append(f'  {w:10.2f} ' + ' '.join(row) +
                       f'   {order if order is None else round(order, 3)}  {check}')
        if common:
            band = [np.mean([abs(sig[(c, w)]['sigma'] - sig[(DIMER_REFERENCE, w)]['sigma'])
                             / abs(sig[(DIMER_REFERENCE, w)]['sigma']) for w in common])
                    for c in DIMER_GRIDS]
            p, _, scatter = _fit([1 / c for c in DIMER_GRIDS], band)
            out.append(f'  band (mean over {common}): ' + ' '.join(f'{b:.2e}' for b in band) +
                       f'  order {p:.3f} scatter {100 * scatter:.1f}%')
        unsettled = [(r['cells'], r['wavelength']) for r in dimer
                     if max((r.get('decay') or {}).values(), default=0.0) >= 1e-4]
        out.append(f'  * unsettled (flux decay >= 1e-4): {unsettled}')
        reference_unsettled = [w for (c, w) in unsettled if c == DIMER_REFERENCE]
        out.append(f'  reference rows unsettled: {reference_unsettled}')
        check = [r for r in _rows('dimer_check') if 'sigma' in r]
        for r in check:
            broad = sig.get((40, r['wavelength']))
            if broad is not None:
                out.append(f'  instrument check, 40 cells, wavelength {r["wavelength"]}: narrow '
                           f'{r["sigma"]:.10f} broad {broad["sigma"]:.10f} relative '
                           f'{abs(r["sigma"] - broad["sigma"]) / abs(broad["sigma"]):.2e}')
        out.append(f'  criterion (a): {"FAIL" if refused else "pass"} '
                   f'({len(refused)} refused (grid, wavelength) pairs); criterion (b): '
                   f'{"pass" if verdicts and all(verdicts) else "FAIL"} '
                   f'({sum(verdicts)}/{len(verdicts)} wavelengths)')
    return out


def append(name, row):
    row = dict(row, source=SOURCE)
    RECORDS.mkdir(parents=True, exist_ok=True)
    with open(RECORDS / f'{name}.jsonl', 'a', encoding='utf-8') as out:
        out.write(json.dumps(row) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='arm', required=True)
    sub.add_parser('plan')
    sub.add_parser('report')
    s = sub.add_parser('stack')
    s.add_argument('--pol', choices=('s', 'p'), required=True)
    s.add_argument('--gap', type=float, default=0.003)
    s.add_argument('--cells', type=int, nargs='*', default=list(STACK_GRIDS))
    s.add_argument('--backend', default='jax')
    d = sub.add_parser('dimer')
    d.add_argument('--cells', type=int, nargs='*', default=list(DIMER_GRIDS) + [DIMER_REFERENCE])
    d.add_argument('--wavelengths', type=float, nargs='*', default=list(DIMER_WAVELENGTHS))
    d.add_argument('--backend', default='jax')
    d.add_argument('--check', action='store_true', help='Amendment 2 instrument check')
    args = parser.parse_args(argv)
    global SOURCE
    SOURCE = source_digest()
    if args.arm == 'plan':
        print('\n'.join(plan_table()))
        return
    if args.arm == 'report':
        print('\n'.join(report()))
        return
    if args.arm == 'stack':
        for cells in args.cells:
            row = run_stack(cells, args.pol, args.gap, args.backend)
            append('stack', row)
            print(json.dumps({k: row[k] for k in ('arm', 'pol', 'cells', 'clip_voxels', 'stable',
                                                  'wall')}), flush=True)
        return
    if args.check:
        for w in (1.0, 1.5, 2.0):
            for row in run_dimer(40, w, args.backend):
                append('dimer_check', row)
                print(json.dumps({k: row[k] for k in ('cells', 'wavelength', 'sigma', 'decay')}),
                      flush=True)
        return
    for cells in args.cells:
        if cells in BROAD_GRIDS:
            rows = run_dimer(cells, None, args.backend)
        else:
            rows = []
            for w in args.wavelengths:
                try:
                    quiet(fd.plan, dimer_scene(cells, w))
                except ValueError as err:
                    append('dimer', dict(arm='D', cells=cells, wavelength=w,
                                         refused=str(err)[:400]))
                    print(json.dumps(dict(cells=cells, wavelength=w, refused=True)), flush=True)
                    continue
                rows += run_dimer(cells, w, args.backend)
                append('dimer', rows[-1])
                print(json.dumps({k: rows[-1][k] for k in ('cells', 'wavelength', 'sigma',
                                                           'clip_voxels', 'decay', 'wall')}),
                      flush=True)
            continue
        for row in rows:
            append('dimer', row)
        print(json.dumps(dict(cells=cells, broad=True, decay=rows[0]['decay'],
                              wall=rows[0]['wall'])), flush=True)

if __name__ == '__main__':
    main()
