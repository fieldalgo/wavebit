"""The problems of tests/api/test_check_run_consistency.py section 12, run with the prototype on.

    python section12_runs.py      # writes results/section12_runs.json

Section 12 asserts that a shape NVF grid below 4*max(orders)+1 is refused at check, prepare and
run. With the prototype's switch on it is admitted; this records what each of those problems then
returns at every door: whether check is clean, whether prepare(...).run and run agree, whether R
and T are finite, the lossless energy balance, and how far the short-grid answer is from the same
problem at the floor grid and at the default grid. The problem definitions are copied from the
test so this script does not import the test suite.
"""
import contextlib
import json
import pathlib

import numpy as np

from cases import arm, to_np
from wavebit import Material, rcwa

OUT = pathlib.Path(__file__).with_name('results')

_REGION = (rcwa.Region(geometry=rcwa.Rectangle(size=(.3, .4)), material=Material(4.)),)
_TENSOR = (rcwa.Region(geometry=rcwa.Rectangle(size=(.3, .4)), material=Material(np.diag([4., 3., 2.]))),)
_BEAM = dict(sources=(rcwa.GaussianBeam(waist=1., angular_spectrum_extent=3., sampling=(3, 3)),))
_BEAM_OUT = rcwa.Outputs(diffraction=False, power_region=((-.4, .4), (-.45, .45)))
GRID_STACKS = {
    'region': (rcwa.Layer(thickness=.2, background=Material(1.5), regions=_REGION),),
    'region-free': (rcwa.Layer(thickness=.2, background=Material(1.5)),),
    'no-layers': (),
    'tensor-beside-region': (rcwa.Layer(thickness=.2, background=Material(1.5), regions=_TENSOR),
                             rcwa.Layer(thickness=.2, background=Material(1.5), regions=_REGION)),
}


def _grid_problem(layers, beam=False):
    excitation = rcwa.Excitation(wavelength=1.3, **(_BEAM if beam else dict(sources=(rcwa.PlaneWave(theta=.2),))))
    return rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=layers), excitation=excitation,
                        **(dict(outputs=_BEAM_OUT) if beam else {}))


def _x64(backend):
    return __import__('jax').enable_x64(True) if backend == 'jax' else contextlib.nullcontext()


def _arrays(result):
    out = {}
    for name in ('R', 'T'):
        try:
            value = getattr(result, name)
        except Exception:
            continue
        if value is not None and not callable(value):
            out[name] = to_np(value)
    for port, power in result.outputs.items():
        if hasattr(power, 'incoming'):
            out[f'{port}_incoming'], out[f'{port}_outgoing'] = to_np(power.incoming), to_np(power.outgoing)
    return out


def doors(problem, numerics, backend='numpy'):
    row = {}
    with _x64(backend):
        report = rcwa.check(problem, numerics=numerics, backend=backend)
        row['check_ok'] = bool(report.ok)
        row['issues'] = [f'{i.code} {i.path}' for i in report.issues]
        prepared = _arrays(rcwa.prepare(problem, numerics=numerics, backend=backend).run(problem))
        direct = _arrays(rcwa.run(problem, numerics=numerics, backend=backend))
    row['outputs'] = sorted(direct)
    row['finite'] = all(np.isfinite(v).all() for v in direct.values())
    row['prepare_vs_run'] = max((float(np.max(np.abs(prepared[k] - direct[k]))) for k in direct), default=0.)
    if 'R' in direct and 'T' in direct:
        row['sum_R'], row['sum_T'] = float(np.sum(direct['R'])), float(np.sum(direct['T']))
        row['one_minus_R_minus_T'] = 1. - row['sum_R'] - row['sum_T']
    return row, direct


rows = []
with arm(True):
    for stack in sorted(GRID_STACKS):
        for orders, grid in [((2, 2), 8), ((3, 1), 12), ((1, 0), 4)]:
            problem = _grid_problem(GRID_STACKS[stack])
            row, short = doors(problem, rcwa.Numerics(orders=orders, shape_nvf_grid_size=grid))
            need = 4*max(orders) + 1
            for label, g in (('floor', need), ('default', None)):
                ref = _arrays(rcwa.run(problem, numerics=rcwa.Numerics(orders=orders, shape_nvf_grid_size=g)))
                row[f'max_dT_vs_{label}'] = float(np.max(np.abs(short['T'] - ref['T'])))
            rows.append(dict(test='refused_at_every_door', stack=stack, orders=orders, grid=grid, **row))
            print(rows[-1], flush=True)
    for beam in (False, True):
        for backend in ('numpy', 'jax', 'torch'):
            problem = _grid_problem(GRID_STACKS['region'], beam=beam)
            row, short = doors(problem, rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=8), backend)
            ref = _arrays(rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=9)))
            row['max_rel_d_vs_floor_numpy'] = max(float(np.max(np.abs(short[k] - ref[k]))/max(np.max(np.abs(ref[k])), 1e-300))
                                                  for k in ref)
            rows.append(dict(test='refused_alike_on_each_backend', beam=beam, backend=backend,
                             orders=(2, 2), grid=8, **row))
            print(rows[-1], flush=True)
(OUT / 'section12_runs.json').write_text(json.dumps(rows, indent=1, default=str))
