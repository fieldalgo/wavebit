"""Unexecuted bounded finite-guide instrument, with no open-domain truth claim.

The two domains share a declared lattice; actual returned indices and coefficients
are checked. Early traces are measured comparisons, not an exact causal theorem.
Sparse corrected energy and source work are retained, not treated as a full loss
budget. The independent dense-operator tests establish that budget separately.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time as timer
import traceback
import warnings

import numpy as np
from wavebit.fdtd import _legacy as fd
from studies.fdtd.constitutive_absorber.fdtd_constitutive_absorber_support import identity, now, relative, sha, versions, write

FAMILIES = ('cylinder', 'rotated-box')
BACKENDS = ('numpy', 'jax', 'torch')
GRIDS = (20, 30, 40)
PLACEMENTS = (0., .3)
POLARIZATIONS = ('u', 'v')
LENGTHS = {'short': 3.2, 'reference': 6.4, 'farther': 9.6}
ABSORBER_CELLS = {20: 8, 30: 12, 40: 16}
WAVELENGTHS = (.9, 1., 1.1)
WINDOWS = ((0., 4.), (4., 8.), (8., 12.), (12., 16.))
AXES = {'cylinder': (0, 1, 2), 'rotated-box': (1, 2, 0)}
OFFSETS = {'E': ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
           'H': ((0, 1, 1), (1, 0, 1), (1, 1, 0))}


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    return value


def component_max(value):
    value = array(value)
    return float(max(np.max(np.abs(value.real), initial=0.),
                     np.max(np.abs(value.imag), initial=0.)))


def difference(actual, reference):
    actual, reference = array(actual), array(reference)
    if actual.shape != reference.shape:
        return {'same_shape': False, 'actual_shape': actual.shape,
                'reference_shape': reference.shape}
    absolute, scale = component_max(actual-reference), component_max(reference)
    return {'same_shape': True, 'absolute': absolute, 'reference_scale': scale,
            'normalization_status': 'finite-reference' if scale else 'unresolved-zero-reference-scale',
            'relative': absolute/scale if scale else None,
            'roundoff_scaled': absolute/max(1., scale),
            'equal_bits': actual.dtype == reference.dtype and actual.tobytes() == reference.tobytes(),
            'finite': bool(np.isfinite(actual).all() and np.isfinite(reference).all())}


def point(axes, u, v, w):
    answer = [0., 0., 0.]
    for a, value in zip(axes, (u, v, w)):
        answer[a] = value
    return tuple(answer)


def build(family, *, n=20, placement=.3, polarization='u', length='short', time=16., contrast=True, homogeneous=False):
    if family not in FAMILIES or n not in GRIDS or placement not in PLACEMENTS:
        raise ValueError('Only the prospectively fixed family/grid/placement roster is admitted.')
    if polarization not in POLARIZATIONS or length not in LENGTHS or time not in (16., 32.):
        raise ValueError('Only the fixed polarization/length/time roster is admitted.')
    if type(contrast) is not bool or type(homogeneous) is not bool:
        raise TypeError('contrast and homogeneous are explicit Booleans.')
    if homogeneous and contrast:
        raise ValueError('The homogeneous control requires contrast=False.')
    axes = AXES[family]
    u, v, w = axes
    dx, axial = 1./n, LENGTHS[length]
    center = point(axes, placement*dx, .5*placement*dx, 0.)
    size = point(axes, .8, .8, axial)
    if family == 'cylinder':
        body = fd.Cylinder(radius=.16, height=axial+4*dx, axis='xyz'[w], center=center)
    else:
        body = fd.Box(size=point(axes, .32, .44, axial+4*dx), center=center,
                      angle=25., axis='xyz'[w])
    boundaries = {a: fd.PEC() for a in 'xyz'}
    boundaries['xyz'[w]] = fd.Absorber(cells=ABSORBER_CELLS[n])
    components = ('E'+'xyz'[u], 'E'+'xyz'[v], 'H'+'xyz'[u], 'H'+'xyz'[v])
    monitors = []
    for side, position in (('minus', -.4), ('plus', .4)):
        at = point(axes, center[u], center[v], position)
        monitors.extend(fd.Record(side+'_'+c, at=at, component=c) for c in components)
    monitors.append(fd.Field('plane', plane=fd.Plane(**{'xyz'[w]: .4}),
                             components=components, wavelengths=WAVELENGTHS))
    return fd.Scene(domain=fd.Domain(size=size), boundaries=fd.Boundaries(**boundaries),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(time=time, courant=.5),
                    background=fd.Material(eps=1.),
                    structures=(() if homogeneous else (fd.Structure(body, fd.Material(eps=2.25 if contrast else 1.)),)),
                    sources=(fd.Dipole('drive', center=center,
                        component='E'+'xyz'[u if polarization == 'u' else v], kind='electric',
                        band=fd.Band(wavelength=1., df=.15, cutoff=1e-3)),),
                    monitors=tuple(monitors))


def preflight():
    """Scalar geometry/band facts; this does not import or lower a production operator."""
    shortest = 1./(1.+math.sqrt(2*math.log(1000))*.15)
    cpw = {str(n): shortest*n/1.5 for n in GRIDS}
    if min(cpw.values()) < 8.:
        raise ValueError('The fixed source band fails the predeclared eight-cell floor.')
    for n in GRIDS:
        if Fraction(ABSORBER_CELLS[n], n) != Fraction(2, 5):
            raise ValueError('Physical absorber thickness changed across the grid roster.')
        for length in (Fraction(16, 5), Fraction(32, 5), Fraction(48, 5)):
            if (length*n).denominator != 1 or int(length*n) % 2:
                raise ValueError('Centered domain does not preserve an integer cell extension.')
    return {'shortest_wavelength': shortest, 'cells_per_medium_wavelength': cpw,
            'absorber_physical_thickness': .4, 'integer_even_extensions': True,
            'scope': 'finite axial terminations with unchanged transverse PEC walls',
            'contrast_false': 'equal-epsilon shaped scene retains the new term',
            'homogeneous_true': 'omit the shape for the ordinary diagonal control',
            'not_certified': ['zero reflection', 'continuum convergence', 'open-domain truth',
                              'source-free sparse energy budget', 'exact causal prefix']}


def extract(result):
    data = {}
    for name, value in result.items():
        if hasattr(value, 'values') and hasattr(value, 't'):
            data[name+'_values'], data[name+'_times'] = array(value.values), array(value.t)
    for c, values in result['plane'].fields.items():
        data['plane_'+c] = array(values)
        for a, coords in zip('xyz', result['plane'].coords[c]):
            data['plane_'+c+'_coords_'+a] = array(coords)
    for key in ('t', 'drive', 'field', 'work', 't_field'):
        data['source_'+key] = array(getattr(result.sources['drive'], key))
    times, energy = result.diagnostics['interior_energy']
    data['energy_times'], data['energy_values'] = array(times), array(energy)
    return data


def plan_receipt(scene, plan, params, data):
    """Expose actual placement/source stamp/central coefficients for saved-data checks."""
    from wavebit.fdtd._kernel.constitutive_absorber import AbsorberConstitutiveTerm
    grid = plan.grid
    axis = next(i for i, a in enumerate('xyz') if isinstance(getattr(scene.boundaries, a)[0], fd.Absorber))
    records = {}
    for monitor in scene.monitors:
        if not isinstance(monitor, fd.Record):
            continue
        c = 'xyz'.index(monitor.component[1])
        index_zyx = grid.nearest(monitor.component[0], c, monitor.at)
        xyz = tuple(reversed(index_zyx))
        coords = tuple(grid.low[a]+(xyz[a]+.5*OFFSETS[monitor.field][c][a])*grid.dx for a in range(3))
        records[monitor.name] = {'component': monitor.component, 'index_xyz': xyz,
                                 'actual_coordinates': coords, 'requested': monitor.at}
    source = next(t for t in plan.terms if t.name == 'drive')
    source_span = [getattr(source.slab, a) for a in 'xyz']
    data['source_stamp'] = array(params['sources']['drive']['stamp'])
    # The central shared normal interval excludes both short-domain damping profiles.
    central = (int(round((-.8-grid.low[axis])/grid.dx)),
               int(round((.8-grid.low[axis])/grid.dx)))
    view = [slice(None)]*4
    view[3-axis] = slice(*central)
    for key in ('ae', 'be', 'ah', 'bh'):
        data['central_'+key] = np.broadcast_to(array(params[key]), plan.shape)[tuple(view)].copy()
    cross = np.zeros(plan.shape, dtype=np.float64)
    for term in plan.constitutive:
        if term.field == 'E':
            cross[term.view] += array(params['constitutive'][term.name])
    data['central_cross'] = cross[tuple(view)].copy()
    return {'grid_n': grid.n, 'grid_low': grid.low, 'grid_high': grid.high, 'axis': axis,
            'dx': plan.dx, 'dt': plan.dt, 'steps': plan.steps, 'skeleton': plan.skeleton,
            'source_span_xyz': source_span, 'records': records, 'central_indices': central,
            'constitutive_terms': [{'name': t.name, 'type': type(t).__name__, 'field': t.field}
                                   for t in plan.constitutive],
            'promoted_terms': sum(isinstance(t, AbsorberConstitutiveTerm) for t in plan.constitutive),
            'energy_type': type(plan.energy).__name__}


def solve(config):
    before, start, clock = identity(), now(), timer.perf_counter()
    scene_config = {k: config[k] for k in ('family', 'n', 'placement', 'polarization', 'length', 'time', 'contrast', 'homogeneous')}
    scene = build(**scene_config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        plan = fd.plan(scene)
        # Public lowering is inspected before a run; no private kernel is substituted.
        params = fd.lower(scene, plan, config['backend'])
        result = fd.run(scene, backend=config['backend'], plan=plan)
        data = extract(result)
        receipt = plan_receipt(scene, result.plan, params, data)
    finite = all(np.isfinite(v).all() for v in data.values())
    expected_promoted = receipt['promoted_terms'] > 0 if not config['homogeneous'] else receipt['promoted_terms'] == 0
    checks = {'finite': bool(finite), 'source_unchanged': identity() == before,
              'expected_promoted_path': bool(expected_promoted),
              'steps_positive': result.plan.steps > 1,
              'energy_present': len(data['energy_values']) > 1,
              'correct_energy_observer': receipt['energy_type'] == (
                  'InteriorEnergy' if config['homogeneous'] else 'AbsorberConstitutiveEnergy'),
              'no_decay': result.plan.decay is None,
              'source_stamp_positive': component_max(data['source_stamp']) > 0,
              'source_drive_nonzero': component_max(data['source_drive']) > 0,
              'recorded_response_nonzero': any(component_max(value) > 0 for key, value in data.items()
                                             if key.endswith('_values') and key != 'energy_values'),
              'stable': result.diagnostics.get('stable') is True}
    return {'config': config, 'started_utc': start, 'ended_utc': now(),
            'seconds': timer.perf_counter()-clock, 'source_before': before,
            'source_after': identity(), 'versions': versions(), 'plan': receipt,
            'checks': checks, 'stable': result.diagnostics.get('stable'),
            'warnings': [{'category': type(w.message).__name__, 'message': str(w.message)} for w in caught],
            'diagnostics': jsonable(result.diagnostics), 'arrays': data}


def compare_pair(short, reference, a, b):
    pa, pb = short['plan'], reference['plan']
    axis = pa['axis']
    shift = [(j-i)//2 for i, j in zip(pa['grid_n'], pb['grid_n'])]
    common = {'same_dx_dt_steps': all(pa[k] == pb[k] for k in ('dx', 'dt', 'steps')),
              'integer_even_extension': all((j-i) % 2 == 0 and j >= i for i, j in zip(pa['grid_n'], pb['grid_n'])),
              'only_axial_extension': all(shift[i] == 0 for i in range(3) if i != axis),
              'record_indices': all(all(y-x == shift[i] for i, (x, y) in enumerate(zip(
                  pa['records'][name]['index_xyz'], pb['records'][name]['index_xyz']))) for name in pa['records']),
              'same_axis': axis == pb['axis'],
              'source_indices': all(all(y-x == shift[i] for x, y in zip(pa['source_span_xyz'][i], pb['source_span_xyz'][i])) for i in range(3))}
    coordinate_errors = {name: difference(pa['records'][name]['actual_coordinates'],
                                         pb['records'][name]['actual_coordinates']) for name in pa['records']}
    lowering = {k: difference(a[k], b[k]) for k in a if k.startswith('central_') or k == 'source_stamp'}
    curves = {}
    for name in pa['records']:
        key = name+'_values'
        times = a[name+'_times']
        if not np.array_equal(times, b[name+'_times']):
            raise ValueError('Same-grid record clocks differ.')
        windows = list(WINDOWS) + ([(16., 32.)] if short['config']['time'] == 32. else [])
        curves[name] = {str(lo)+'_'+str(hi): difference(a[key][(times >= lo) & (times < hi)],
                                                     b[key][(times >= lo) & (times < hi)]) for lo, hi in windows}
        curves[name]['whole'] = difference(a[key], b[key])
    spectra = {k: [difference(a[k][j], b[k][j]) for j in range(len(WAVELENGTHS))]
               for k in a if k.startswith('plane_') and '_coords_' not in k}
    coords = {k: difference(a[k], b[k]) for k in a if '_coords_' in k}
    return {'mapping_checks': common, 'coordinate_errors': coordinate_errors,
            'shared_lowering': lowering, 'plane_coordinates': coords,
            'trace_windows': curves, 'spectra_by_wavelength': spectra,
            'interpretation': 'finite-domain sensitivity, not a reflection coefficient',
            'exact_causal_prefix_claim': False}


def measure_pair(family, *, backend='numpy', n=20, placement=.3, polarization='u',
                 length='short', time=16., contrast=True, homogeneous=False):
    if length != 'short':
        raise ValueError('The public calibration/test instrument compares short and reference.')
    common = dict(family=family, backend=backend, n=n, placement=placement,
                  polarization=polarization, time=time, contrast=contrast, homogeneous=homogeneous)
    runs = {key: solve(dict(common, length=key)) for key in ('short', 'reference')}
    arrays = {k: runs[k].pop('arrays') for k in runs}
    comparison = compare_pair(runs['short'], runs['reference'], arrays['short'], arrays['reference'])
    checks = {k+'_'+c: v for k, run in runs.items() for c, v in run['checks'].items()}
    checks.update({'mapping_'+k: v for k, v in comparison['mapping_checks'].items()})
    alignment = [v for group in ('coordinate_errors', 'shared_lowering', 'plane_coordinates')
                 for v in comparison[group].values()]
    checks['all_alignment_finite_same_shape'] = bool(alignment) and all(
        v.get('same_shape') and v.get('finite') for v in alignment)
    if not checks['all_alignment_finite_same_shape']:
        raise ValueError('Missing, nonfinite or differently shaped shared-grid alignment arrays.')
    rounding = [v['roundoff_scaled'] for v in alignment]
    return {'runs': runs, 'comparisons': comparison, 'checks': checks,
            'roundoff_metrics': {'shared_grid_lowering': max(rounding)}, 'arrays': arrays}


def zero_comparison(shaped, legacy, a, b):
    keys = [k for k in a if (k.endswith('_values') and k != 'energy_values')
            or (k.startswith('plane_') and '_coords_' not in k) or k.startswith('source_')]
    parity = {k: difference(a[k], b[k]) for k in keys}
    checks = {'new_path': shaped['plan']['promoted_terms'] > 0,
              'legacy_path': legacy['plan']['promoted_terms'] == 0,
              'shaped_cross_zero': bool((a['central_cross'] == 0).all()),
              'legacy_cross_zero': bool((b['central_cross'] == 0).all()),
              'same_clock': shaped['plan']['dt'] == legacy['plan']['dt']
                            and shaped['plan']['steps'] == legacy['plan']['steps'],
              'all_finite_same_shape': all(v['same_shape'] and v['finite'] for v in parity.values())}
    return {'checks': checks, 'parity': parity,
            'max_roundoff_scaled': max(v['roundoff_scaled'] for v in parity.values()),
            'energy_equality_claim': False}


def measure_zero_pair(family, *, backend='numpy', n=20, placement=.3, polarization='u', time=16.):
    common = dict(family=family, backend=backend, n=n, placement=placement,
                  polarization=polarization, time=time, contrast=False, length='short')
    runs = {key: solve(dict(common, homogeneous=homogeneous))
            for key, homogeneous in (('shaped', False), ('legacy', True))}
    arrays = {k: runs[k].pop('arrays') for k in runs}
    compared = zero_comparison(runs['shaped'], runs['legacy'], arrays['shaped'], arrays['legacy'])
    compared['checks'].update({key+'_'+label: value for key, run in runs.items()
                               for label, value in run['checks'].items()})
    return dict(compared, runs=runs, arrays=arrays,
                qualification='Known six-row calibration fixture; no energy equality or continuum accuracy claim.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    result = {'started_utc': now(), 'config': config, 'complete': False}
    try:
        result = solve(config)
        data = result.pop('arrays')
        raw = args.out.with_suffix('.npz')
        with raw.open('xb') as handle:
            np.savez_compressed(handle, **data)
        result['raw'] = relative(raw)
        result['raw_sha256'] = sha(raw)
        result['array_sha256'] = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in data.items()}
        result['complete'] = all(result['checks'].values())
        write(args.out, jsonable(result))
        if not result['complete']:
            raise SystemExit(1)
    except BaseException as error:
        if not args.out.exists():
            result.update(ended_utc=now(), exception=repr(error), traceback=traceback.format_exc())
            write(args.out, jsonable(result))
        raise


if __name__ == '__main__':
    main()
