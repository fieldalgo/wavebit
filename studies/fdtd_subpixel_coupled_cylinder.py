"""Does production inverse-material cross coupling repair the TE cylinder fit?

This retains the historical S2 grid, source, paired empty run and contour.
Material coefficients, compact flux memory, stepping and positive-energy/CFL
checks are the production implementations. Written before public Cylinder landed (C4),
the custom shape is placed into an empty scene's prepared plan. Its periodic
invariant axis and declared-material resolution are NOT public-scene checks.
Result reporting uses that same empty-scene context; public refusals stay intact.
See docs/fd-mie-validation.md for the measurements and limitations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
import numpy as np
import wavebit.fd as fd
from wavebit.fd._lower import lower, plan
from wavebit.fd._lower.constitutive import terms, validate
from wavebit.fd._run import execute, _results
from wavebit.fd._runtime import runtime
from wavebit.fd._shapes import Cylinder
from studies.fdtd_subpixel_mie_cylinder import (
    CONTOUR, RADIUS, WAVELENGTH, SCHEDULE, PLACEMENTS, SOURCE_Z, HALF_Z, scene_for,
    cylinder_cross_sections, scattered_flux, incident_intensity, convergence,
)


class CylinderSlice(fd.Shape):
    """Research shape for one invariant x cell, not the public Cylinder API."""

    def __init__(self, center):
        self.center = tuple(center)

    def body(self):
        # Historical unit-height primitive: both caps are outside the one-cell
        # slice at every study resolution. Only the radial wall is sampled.
        return Cylinder(RADIUS, height=1., axis='x', center=(0., *self.center))

    def support(self, grid):
        box = [(0, grid.n[0])]
        for a in (1, 2):
            c = self.center[a-1]
            # One cell includes every component's voxel at the radial wall.
            lo = math.floor((c-RADIUS-grid.low[a])/grid.dx)-1
            hi = math.ceil((c+RADIUS-grid.low[a])/grid.dx)+1
            box.append((max(0, lo), min(grid.n[a], hi)))
        return tuple(box)

    def bounded_axes(self, grid):
        return (1, 2)

    def uniform_along(self, axis, cells, grid):
        if axis == 0:
            return True
        lo, hi = self.support(grid)[axis]
        return hi <= cells[0] or lo >= cells[1]

    def describe(self):
        return f'ResearchCylinderSlice(radius={RADIUS}, center={self.center})'

    def span_rows(self, grid):
        return tuple((f'cylinder.{a}', c-RADIUS, c+RADIUS, c-RADIUS, c+RADIUS)
                     for a, c in zip(('y', 'z'), self.center))

    def min_extent(self, grid):
        return 2*RADIUS


def prepare(empty, base, eps, center):
    scene = empty.replace(structures=(fd.Structure(CylinderSlice(center), eps),))
    coupled = base.with_static(terms=base.terms+terms(scene, base))
    validate(scene, coupled)
    return scene, coupled


def solve(empty, material_scene, p, rt):
    params = lower(material_scene, p, rt)
    state = execute(p, params, rt)
    # The report uses the historical experiment's context. In particular,
    # epsilon=12 at 40 cells is correctly refused by public preparation at
    # the pulse's shortest wavelength. This experiment does not open that door.
    return _results(empty, p, state, rt, params)


def run_case(cpw, eps=12., pol='TE', backend='jax', placements=PLACEMENTS,
             diagonal_control=False, energy=False, **changes):
    period, gate = SCHEDULE[eps]
    period *= changes.pop('period_scale', 1.)
    gate *= changes.pop('gate_scale', 1.)
    empty = scene_for(cpw, pol, period, gate, **changes)
    rt = runtime(backend)
    rows = []
    with warnings.catch_warnings(record=True) as notes:
        base = plan(empty)
        ref = solve(empty, empty, base, rt)
        incident = incident_intensity(ref, pol, CONTOUR, base.grid)
        exact = float(cylinder_cross_sections(RADIUS, eps, WAVELENGTH, pol)[1])
        for placement in placements:
            scene, coupled = prepare(empty, base, eps, tuple(v/cpw for v in placement))
            variants = [('coupled', coupled)]
            if diagonal_control:
                variants.insert(0, ('diagonal', base))
            for name, p in variants:
                started = time.perf_counter()
                result = solve(empty, scene, p, rt)
                sigma = scattered_flux(result, ref, pol, CONTOUR, p.grid)[0]/incident
                zero = {'box': SimpleNamespace(
                    fields={c: 0*v for c,v in result['box'].fields.items()},
                    coords=result['box'].coords)}
                absorption = -scattered_flux(result, zero, pol, CONTOUR, p.grid)[0]/incident
                times, energies = result.diagnostics['interior_energy']
                values = np.asarray(energies)
                row = dict(cpw=cpw, eps=eps, pol=pol, backend=backend, variant=name,
                    placement=placement, sigma=sigma, exact=exact, error=sigma/exact-1,
                    constitutive_terms=len(p.constitutive),
                    absorption_cross_section=absorption,
                    decay=float(result.diagnostics['decay']['box']),
                    final_energy_over_peak=float(values[-1]/max(values)),
                    # The historical instrument moves the contour by four cells
                    # in either direction; these are control locations, not fits.
                    contours=[scattered_flux(result, ref, pol, CONTOUR+n/cpw, p.grid)[0]/incident
                              for n in (-4, 4)], seconds=time.perf_counter()-started)
                if energy:
                    row['interior_energy'] = dict(time=np.asarray(times).tolist(), value=values.tolist())
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='interior_energy'}), flush=True)
    return dict(rows=rows, warnings=list(dict.fromkeys(str(w.message) for w in notes)),
                grid=base.grid.n, steps=base.steps, period=period, gate=gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cells', default='40,56,80')
    parser.add_argument('--eps', type=float, choices=tuple(SCHEDULE), default=12.)
    parser.add_argument('--pol', choices=('TE', 'TM'), default='TE')
    parser.add_argument('--backend', choices=('numpy', 'jax', 'torch'), default='jax')
    parser.add_argument('--placements', type=int, choices=range(1,len(PLACEMENTS)+1), default=len(PLACEMENTS))
    parser.add_argument('--diagonal-control', action='store_true')
    parser.add_argument('--energy', action='store_true')
    parser.add_argument('--period-scale', type=float, default=1.)
    parser.add_argument('--gate-scale', type=float, default=1.)
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; use a new path to preserve measurements')
    cells = tuple(map(int,args.cells.split(',')))
    if args.controls and (len(cells) != 1 or args.placements != 1):
        parser.error('controls requires one --cells value and --placements 1')
    files = sorted((ROOT/'src/wavebit/fd').rglob('*.py')) + [
        Path(__file__), ROOT/'studies/fdtd_subpixel_mie_cylinder.py',
        ROOT/'studies/fdtd_subpixel_tilted_tensor.py',
        ROOT/'tests/fd_helpers.py', ROOT/'tests/references/mie.py']
    def hashes():
        return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files}
    source = hashes()
    def verify_source():
        if hashes() != source:
            raise RuntimeError('MeasurementSourceChanged: source changed during the experiment')
    output = dict(status='running', public_scene_preparation=False, cases={}, fits={},
                  source_sha256=source)
    started = time.perf_counter()
    if args.controls:
        # Identical perturbations to the original cylinder instrument; their
        # effects must be remeasured after adding production cross coupling.
        for name,changes in (
            ('base', {}), ('longer', dict(period_scale=1.4,gate_scale=1.4)),
            ('wider', dict(period_scale=1.4)),
            ('short', dict(period_scale=.55,gate_scale=.55)),
            ('short_time_only', dict(gate_scale=.55)),
            ('source', dict(source_z=SOURCE_Z+.2)),
            ('layer', dict(half_z=HALF_Z+1.)),
        ):
            verify_source()
            print('CONTROL',name,flush=True)
            output['cases'][name] = run_case(cells[0],args.eps,args.pol,args.backend,
                PLACEMENTS[:1],args.diagonal_control,args.energy,**changes)
            verify_source()
            args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
        output.update(status='completed_controls_not_C3_pass',seconds=time.perf_counter()-started)
        args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
        return
    for cpw in cells:
        verify_source()
        value = run_case(cpw, args.eps, args.pol, args.backend, PLACEMENTS[:args.placements],
                         args.diagonal_control, args.energy,
                         period_scale=args.period_scale, gate_scale=args.gate_scale)
        verify_source()
        output['cases'][cpw] = value
        args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    if len(cells) >= 3:
        for name in (('diagonal','coupled') if args.diagonal_control else ('coupled',)):
            errors = [[r['error'] for r in output['cases'][c]['rows'] if r['variant']==name] for c in cells]
            output['fits'][name] = convergence(cells, errors)
    output.update(status='completed_measurements_not_C3_pass', seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(json.dumps(output['fits']),flush=True)


if __name__ == '__main__':
    main()
