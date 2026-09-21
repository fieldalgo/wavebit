"""S2: measure the current cylinder occupancy against an independent Bessel series.

The one-cell x axis is invariant. A wide periodic y cell and a finite time gate
isolate one cylinder before a neighbour can arrive; the control runs vary both.
A paired empty run removes the incident field before the closed-contour flux.
Until public fd.Cylinder lands in C4, this study substitutes coefficients after
lowering an empty scene. Geometry, component sampling and mixing use the actual
C2 functions. This is an instrument, not a public scene bypass: in particular,
the declared-material resolution refusal is not exercised by this substitution.
The 3-D sphere study uses the public fd.Sphere path without that substitution.

The contour construction follows the earlier S2 study, with obsolete prototype
fill/moment rules removed. Fits report placement RMS as well as signed mean;
a small signed mean caused by cancellation is not sufficient evidence of order.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import wavebit.fd as fd
from wavebit.fd._lower import lower, plan as _plan
from wavebit.fd._lower.occupancy import component_occupancy
from wavebit.fd._lower.coefficients import mix
from wavebit.fd._run import execute, _results
from wavebit.fd._runtime import runtime
from wavebit.fd._shapes import Cylinder
from references.mie import cylinder_cross_sections
from fd_helpers import to_numpy
from studies.fdtd_subpixel_tilted_tensor import fit

# Baseline experiment dimensions, not accuracy thresholds. At TE eps=12 and
# 40 cells, multiplying gate/period by 1.4 changes sigma by 7.04e-7 relative;
# multiplying them by .55 changes it by 2.17e-6. See the committed result log.
# TM eps=12 has a longer tail: the sweep retains each row's measured decay.
WAVELENGTH, RADIUS, DF = 1., .15, .25
PML, SOURCE_Z, HALF_Z, CONTOUR = 12, .9, 1.4, .55
SCHEDULE = {1.2: (26., 24.), 12.: (50., 45.)}
POLARIZATION = {'TM': 'p', 'TE': 's'}
FLUX_COMPONENTS = {'TM': ('Ex', 'Hy', 'Ex', 'Hz'), 'TE': ('Ey', 'Hx', 'Ez', 'Hx')}
FLUX_SIGN = {'TM': (1., -1.), 'TE': (-1., 1.)}
# Fixed stratified sampling of one voxel; these are sampling locations, not
# accuracy thresholds. All eight locations are retained, with no best-case pick.
PLACEMENTS = ((0., 0.), (.5, 1/3), (.25, 2/3), (.75, 1/9),
              (.125, 4/9), (.625, 7/9), (.375, 2/9), (.875, 5/9))

def scene_for(cpw, pol, period, gate, monitor_half=0.8, half_z=HALF_Z, pml=PML,
              reflection=None, df=DF, source_z=SOURCE_Z):
    dx = WAVELENGTH / cpw
    band = fd.Band(wavelength=WAVELENGTH, df=df)
    pw = fd.PlaneWave('pw', z=source_z, direction='-z', theta=0.0,
                      polarization=POLARIZATION[pol], band=band)
    size = (dx, period, 2.0 * (half_z + pml * dx))
    mon = fd.Field('box', box=fd.Box(size=(dx, 2 * monitor_half, 2 * monitor_half)),
                   wavelengths=(WAVELENGTH,))
    face = fd.PML(cells=pml) if reflection is None else fd.PML(cells=pml, reflection=reflection)
    return fd.Scene(domain=fd.Domain(size=size),
                    boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=face),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(periods=gate),
                    sources=(pw,), monitors=(mon,))

def solve(scene, eps=None, backend='numpy'):
    p = _plan(scene)
    rt = runtime(backend)
    params = lower(scene, p, rt)
    if eps is not None:
        params = dict(params, be=params['be'] / rt.array(eps, 'float64'))
    state = execute(p, params, rt)
    return (_results(scene, p, state, rt, params), p)

def _near(values, target):
    a = np.asarray(values)
    return int(np.argmin(np.abs(a - target)))

def _tangential(values, lo, hi, h, closed=True):
    """Node-box quadrature: trapezoid on nodes, midpoint on half cells.

    Node and half-cell lattices differ by h/2; h/4 is their geometric separator,
    not a fitted tolerance. closed=False deliberately opens the contour.
    """
    a = np.asarray(values)
    if not closed:
        i0, i1 = (_near(a, lo), _near(a, hi))
        return (slice(i0, i1 + 1), np.ones(i1 - i0 + 1))
    on_node = abs(a[_near(a, lo)] - lo) < 0.25 * h
    if on_node:
        i0, i1 = (_near(a, lo), _near(a, hi))
        w = np.ones(i1 - i0 + 1)
        w[0] = w[-1] = 0.5
    else:
        i0, i1 = (_near(a, lo + h / 2), _near(a, hi - h / 2))
        w = np.ones(i1 - i0 + 1)
    return (slice(i0, i1 + 1), w)

def _face(fields, coords, e_name, h_name, sign_of_s, normal_axis, box, h, closed=True):
    tangential = 1 if normal_axis == 2 else 2
    plane = box[normal_axis - 1][0 if sign_of_s < 0 else 1]
    ke = _near(coords[e_name][normal_axis], plane)
    kh0 = _near(coords[h_name][normal_axis], plane - h / 2)
    kh1 = _near(coords[h_name][normal_axis], plane + h / 2)
    sl, w = _tangential(coords[e_name][tangential], *box[tangential - 1], h, closed=closed)
    if normal_axis == 2:
        e = fields[e_name][ke, sl, 0]
        hh = 0.5 * (fields[h_name][kh0, sl, 0] + fields[h_name][kh1, sl, 0])
    else:
        e = fields[e_name][sl, ke, 0]
        hh = 0.5 * (fields[h_name][sl, kh0, 0] + fields[h_name][sl, kh1, 0])
    # Postprocessing is on the host. In particular, np.sum(Torch tensor) would
    # dispatch unsupported axis/out keywords instead of doing a NumPy reduction.
    e, hh = to_numpy(e), to_numpy(hh)
    return float(np.sum(0.5 * np.real(e * np.conj(hh)) * w) * h)

def contour_box(grid, half):
    out = []
    for axis in (1, 2):
        nodes = grid.axis(axis, 0)
        out.append((float(nodes[_near(nodes, -half)]), float(nodes[_near(nodes, half)])))
    return tuple(out)

def scattered_flux(res, ref, pol, half, grid, closed=True):
    ez, hz, ey, hy = FLUX_COMPONENTS[pol]
    sz, sy = FLUX_SIGN[pol]
    fields = {c: res['box'].fields[c][0] - ref['box'].fields[c][0]
              for c in {ez, hz, ey, hy}}
    coords = {c: res['box'].coords[c] for c in fields}
    box = contour_box(grid, half)
    h = grid.dx
    parts = {}
    for side, sign in (('-', -1.0), ('+', +1.0)):
        parts[f'z{side}'] = sign * sz * _face(fields, coords, ez, hz, sign, 2,
                                             box, h, closed=closed)
        parts[f'y{side}'] = sign * sy * _face(fields, coords, ey, hy, sign, 1,
                                             box, h, closed=closed)
    return (sum(parts.values()), parts)

def incident_intensity(ref, pol, half, grid):
    ez, hz = FLUX_COMPONENTS[pol][:2]
    fields = {c: ref['box'].fields[c][0] for c in FLUX_COMPONENTS[pol][:2]}
    coords = {c: ref['box'].coords[c] for c in fields}
    box = contour_box(grid, half)
    flux = _face(fields, coords, ez, hz, -1.0, 2, box, grid.dx)
    return abs(flux) / (box[0][1] - box[0][0])


def component_eps(grid, eps, center=(0., 0.)):
    """Actual C2 geometry/sampling/mixing on the invariant x slice."""
    # Height 1 spans the one-cell x slice with both caps far outside it. The
    # scalar field is invariant in x; this slice is the infinite-cylinder limit.
    node = Cylinder(RADIUS, height=1., axis='x', center=(0., *center))
    box = tuple((0, n) for n in grid.n)
    out = []
    for c in range(3):
        fill, moment = component_occupancy(node, grid, box, 'E', c, runtime('numpy'))
        out.append(mix(np, np.ones_like(fill), eps, fill, moment))
    return np.stack(out)


def convergence(cells, errors):
    """Report signed-mean and RMS fits plus four-placement subset sensitivity.

    errors has shape (resolutions, placements). A signed mean near zero can be
    cancellation; its RMS and spread remain visible. Subset sensitivity is not
    the regression's confidence interval and is never used to exclude an order.
    """
    errors = np.asarray(errors, dtype=float)
    def fitted(values):
        return fit([dict(cells_actual=c, error=e) for c, e in zip(cells, values)])
    means = errors.mean(axis=1)
    rms = np.sqrt(np.mean(errors**2, axis=1))
    result = dict(mean_errors=means.tolist(), rms_errors=rms.tolist(),
                  spread=np.ptp(errors, axis=1).tolist(), mean=fitted(means), rms=fitted(rms))
    if errors.shape[1] >= 4:
        orders = [fitted(errors[:, sub].mean(axis=1))['order']
                  for sub in itertools.combinations(range(errors.shape[1]), 4)]
        finite = [p for p in orders if p is not None and math.isfinite(p)]
        result['subset_orders'] = dict(count=len(finite),
                                      minimum=min(finite) if finite else None,
                                      maximum=max(finite) if finite else None,
                                      sd=float(np.std(finite)) if finite else None)
    return result


def run_case(cpw, pol, eps, backend='jax', placements=PLACEMENTS, **changes):
    period, gate = SCHEDULE[eps]
    period *= changes.pop('period_scale', 1.)
    gate *= changes.pop('gate_scale', 1.)
    scene = scene_for(cpw, pol, period, gate, **changes)
    with warnings.catch_warnings(record=True) as notes:
        ref, plan = solve(scene, backend=backend)
        incident = incident_intensity(ref, pol, CONTOUR, plan.grid)
        rows = []
        for place in placements:
            coefficients = component_eps(plan.grid, eps, tuple(v/cpw for v in place))
            result, _ = solve(scene, coefficients, backend)
            sigma = scattered_flux(result, ref, pol, CONTOUR, plan.grid)[0]/incident
            other = [scattered_flux(result, ref, pol, CONTOUR + n/cpw, plan.grid)[0]/incident
                     for n in (-4, 4)]
            opened = scattered_flux(result, ref, pol, CONTOUR, plan.grid, closed=False)[0]/incident
            exact = float(cylinder_cross_sections(RADIUS, eps, WAVELENGTH, pol)[1])
            rows.append(dict(cpw=cpw, pol=pol, eps=eps, placement=place, sigma=sigma,
                             exact=exact, error=sigma/exact-1,
                             decay=float(result.diagnostics['decay']['box']),
                             contours=other, open_contour=opened))
    return dict(rows=rows, warnings=list(dict.fromkeys(str(w.message) for w in notes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--backend', default='jax')
    parser.add_argument('--cells', default='20,28,40,56')
    parser.add_argument('--polarizations', default='TM,TE')
    parser.add_argument('--eps', default='1.2,12')
    parser.add_argument('--placements', type=int, choices=range(1, len(PLACEMENTS)+1),
                        default=len(PLACEMENTS))
    parser.add_argument('--period-scale', type=float, default=1.)
    parser.add_argument('--gate-scale', type=float, default=1.)
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    cells = tuple(int(c) for c in args.cells.split(','))
    output = {}
    started = time.perf_counter()
    for pol in args.polarizations.split(','):
        for eps in map(float, args.eps.split(',')):
            key = f'{pol}|{eps:g}'
            output[key] = {}
            if args.controls:
                for name, changes in (
                    ('base', {}), ('longer', dict(period_scale=1.4, gate_scale=1.4)),
                    ('wider', dict(period_scale=1.4)),
                    ('short', dict(period_scale=.55, gate_scale=.55)),
                    ('source', dict(source_z=SOURCE_Z+.2)),
                    ('layer', dict(half_z=HALF_Z+1.)),
                ):
                    value = run_case(cells[0], pol, eps, args.backend, placements=PLACEMENTS[:1],
                                     **changes)
                    output[key][name] = value
                    print(json.dumps(dict(case=key, control=name, **value)), flush=True)
                    args.output.write_text(json.dumps(output, indent=2), encoding='utf8')
            else:
                errors = []
                for cpw in cells:
                    value = run_case(cpw, pol, eps, args.backend,
                                     placements=PLACEMENTS[:args.placements],
                                     period_scale=args.period_scale, gate_scale=args.gate_scale)
                    output[key][cpw] = value
                    errors.append([row['error'] for row in value['rows']])
                    print(json.dumps(dict(case=key, **value)), flush=True)
                    args.output.write_text(json.dumps(output, indent=2), encoding='utf8')
                output[key]['fit'] = convergence(cells, errors)
                print(key, json.dumps(output[key]['fit']), flush=True)
    output['seconds'] = time.perf_counter()-started
    args.output.write_text(json.dumps(output, indent=2), encoding='utf8')
    print(f'elapsed {output["seconds"]:.2f} s', flush=True)


if __name__ == '__main__':
    main()
