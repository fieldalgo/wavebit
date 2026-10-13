"""Bounded physical instruments for optional automatic decay stopping.

No continuum/spatial-accuracy or universal temporal-error claim. Every adaptive
run is compared with the unchanged fixed-duration executor on the same grid.
The uniform conductor additionally has an independent exact discrete recurrence.
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
import time
import warnings

import numpy as np
from wavebit.fdtd import _legacy as fd
from studies.fdtd.decay_stop.fdtd_decay_stop_support import ROOT, identity, now, relative, sha, write

BACKENDS = ('numpy', 'jax', 'torch')
CASES = ('uniform_conductor', 'pec_cavity', 'late_dipoles',
         'debye_memory', 'weak_pml_echo', 'open_dipole')
CALIBRATION_CASES = ('uniform_conductor', 'open_dipole')
WAVELENGTHS = (.9, 1., 1.1)
CAP_TIMES = {'uniform_conductor': 20., 'pec_cavity': 12., 'late_dipoles': 30.,
             'debye_memory': 12., 'weak_pml_echo': 24., 'open_dipole': 16.}


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


def build(case, *, threshold, window, adaptive=True, steps=None, cap_scale=1.):
    """Return a fixed public scene; no fitting, random geometry or backend knob."""
    if case not in CASES:
        raise ValueError(case)
    dx, courant = .05, .5
    run = (fd.Run(steps=steps, courant=courant) if steps is not None else
           fd.Run(time=CAP_TIMES[case]*cap_scale, courant=courant))
    if adaptive:
        run = run.replace(decay=fd.Decay(threshold=threshold, window=window))
    record = (fd.Record('e', at=(0., 0., .1), component='Ey'),
              fd.Record('h', at=(0., 0., .1), component='Hx'))
    initial = fd.InitialField(E=(None, lambda x, y, z: np.cos(math.pi*z), None))
    if case == 'uniform_conductor':
        return fd.Scene(domain=fd.Domain(size=(.2, .2, .2)),
            boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
            resolution=fd.Resolution(dx=dx), run=run,
            background=fd.Conductor(eps=2., sigma=4.),
            initial=fd.InitialField(E=(None, 1., None)),
            monitors=(fd.Record('e', at=(0., 0., 0.), component='Ey'),
                      fd.Record('h', at=(0., 0., 0.), component='Hx'),
                      fd.Field('field', plane=fd.Plane(z=0.), components=('Ey',),
                               wavelengths=WAVELENGTHS)))
    if case == 'debye_memory':
        return fd.Scene(domain=fd.Domain(size=(.2, .2, .2)),
            boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
            resolution=fd.Resolution(dx=dx), run=run,
            structures=(fd.Structure(fd.Slab(z=(-.1,.1)),
                fd.Dispersive(eps_inf=1., poles=(fd.Debye(delta_eps=1e6,tau=5e5),))),),
            initial=fd.InitialField(E=(None,1.,None)),
            monitors=(fd.Record('e',at=(0.,0.,0.),component='Ey'),
                      fd.Record('h',at=(0.,0.,0.),component='Hx')))
    if case == 'pec_cavity':
        return fd.Scene(domain=fd.Domain(size=(dx, dx, 1.)),
            boundaries=fd.Boundaries(x=fd.Axis1D(k=0.), y=fd.Axis1D(k=0.), z=fd.PEC()),
            resolution=fd.Resolution(dx=dx), run=run, initial=initial, monitors=record)
    if case == 'weak_pml_echo':
        cells = 64
        return fd.Scene(domain=fd.Domain(size=(dx, dx, 6.+2*cells*dx)),
            boundaries=fd.Boundaries(x=fd.Axis1D(k=0.), y=fd.Axis1D(k=0.),
                z=fd.PML(cells=cells, reflection=1e-3, wavelength=1., allow_thin=True)),
            resolution=fd.Resolution(dx=dx), run=run,
            initial=fd.InitialField(E=(None,
                lambda x, y, z: np.exp(-z*z)*np.cos(2*math.pi*z), None)),
            monitors=record+(fd.Field('field', plane=fd.Plane(z=.5),
                components=('Ey', 'Hx'), wavelengths=WAVELENGTHS),
                fd.Flux('flux', plane=fd.Plane(z=.5), wavelengths=WAVELENGTHS)))
    # Dipole stamps are genuinely 3-D; Axis1D deliberately refuses them.
    band = fd.Band(wavelength=1., df=.18, cutoff=1e-6)
    source = fd.Dipole('d', center=(.013, -.02, .007), component='Ez', band=band)
    sources = (source,) if case == 'open_dipole' else (source, source.replace(name='late', delay=12.))
    side, pml = .4+2*12*dx, 12
    return fd.Scene(domain=fd.Domain(size=(side,)*3),
        boundaries=fd.Boundaries.open(fd.PML(cells=pml, wavelength=1.)),
        resolution=fd.Resolution(dx=dx), run=run, sources=sources,
        monitors=(fd.Record('e', at=(.1, .05, 0.), component='Ez'),
                  fd.Record('h', at=(.1, .05, 0.), component='Hy'),
                  fd.Field('field', box=fd.Region(size=(.1,)*3, center=(.1, .05, 0.)),
                           components=('Ez', 'Hy'), wavelengths=WAVELENGTHS),
                  fd.Flux('flux', box=fd.Region(size=(.3,)*3), wavelengths=WAVELENGTHS)))


def component_max(values):
    values = np.asarray(values)
    if not values.size:
        return 0.
    return float(max(np.max(np.abs(values.real)), np.max(np.abs(values.imag))))


def error(actual, reference):
    """Raw max error and a scale, never silently map a zero reference to a pass."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        return {'same_shape': False, 'actual_shape': actual.shape, 'reference_shape': reference.shape}
    absolute = component_max(actual-reference)
    scale = component_max(reference)
    return {'same_shape': True, 'absolute': absolute, 'scale': scale,
            'scaled': absolute/scale if scale else (0. if absolute == 0. else None),
            'equal_bits': bool(actual.dtype == reference.dtype and actual.tobytes() == reference.tobytes()),
            'finite': bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)))}


def outputs(result):
    answer = {'final_E': array(result.final.E), 'final_H': array(result.final.H)}
    for name in ('e', 'h'):
        answer[name+'_values'] = array(result[name].values)
        answer[name+'_times'] = array(result[name].t)
    if 'field' in result:
        for component, values in result['field'].fields.items():
            answer['field_'+component] = array(values)
    if 'flux' in result:
        answer['flux'] = array(result['flux'].flux)
    for name, source in result.sources.items():
        for key in ('t', 'drive', 'field', 'work', 't_field'):
            value = getattr(source, key, None)
            if value is not None:
                answer['source_'+name+'_'+key] = array(value)
    interior = result.diagnostics.get('interior_energy')
    if interior is not None:
        answer['energy_times'], answer['energy_values'] = map(array, interior)
    return answer


def solve(scene, backend):
    started, start = now(), time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fd.run(scene, backend=backend)
    return result, {'started_utc': started, 'ended_utc': now(),
        'seconds': time.perf_counter()-start, 'steps': result.plan.steps,
        'report_steps': result.report.steps, 'dt': result.plan.dt,
        'stable': result.diagnostics.get('stable'),
        'stopping': result.diagnostics.get('stopping'),
        'warnings': [{'category': type(w.message).__name__, 'message': str(w.message)} for w in caught]}


def uniform_oracle(dt, cap_steps, threshold, window):
    """Independent Yee cell with zero curl and trapezoidal Ohmic current."""
    eps, sigma = 2., 4.
    q = (eps-sigma*dt/2)/(eps+sigma*dt/2)
    quotient = Fraction(window)/Fraction(dt)
    width = -(-quotient.numerator//quotient.denominator)
    stop = cap_steps
    for end in range(width, cap_steps+1, width):
        if q**(end-width+1) <= threshold:
            stop = end
            break
    omega = 2*math.pi/np.asarray(WAVELENGTHS)
    phase = np.exp(1j*omega*dt)
    z = q*phase
    summed = dt*z*(1-z**stop)/(1-z)
    symbol = 2*np.sin(omega*dt/2)/dt
    corrected = summed+(1j/symbol)*np.exp(1j*omega*dt/2)*(q**stop*phase**stop-1.)
    return {'q': q, 'expected_steps': stop, 'window_steps': width,
            'expected_record': q**np.arange(1, stop+1), 'expected_phasor': corrected}


def field_only_counterfactual(e, h, threshold, width):
    """Exact represented-product criterion on this uniform fixture's two records."""
    peaks = [1.,0.]  # The declared initial uniform Ey=1, H=0.
    for end in range(width,len(e)+1,width):
        maxima = [component_max(values[end-width:end]) for values in (e,h)]
        peaks = [max(old,new) for old,new in zip(peaks,maxima)]
        if all(Fraction(value) <= Fraction(threshold)*Fraction(peak)
               for value,peak in zip(maxima,peaks)):
            return end
    return None


def debye_oracle(dt, steps, threshold, window):
    """Closed uniform split-Debye map, including zero memory and the first-step shift."""
    delta, tau, eps = 1e6,5e5,1.
    c = delta*dt/(2*tau+dt)
    gain = (2*tau-dt)/(2*tau+dt)
    drive = c*(1+gain)
    update_eps = eps+c
    ratio = gain-drive/update_eps
    equilibrium = update_eps/(eps+delta)
    if not 0 < ratio < 1:
        raise ValueError('The declared retained-memory fixture lost its monotone discrete mode.')
    # E1=1 because initial delta=0. After n steps P leads the field by one update.
    electric = equilibrium+(1-equilibrium)*ratio**np.arange(steps+1)
    current = update_eps*(1-electric[1:])
    previous = update_eps*(1-electric[:-1])
    quotient = Fraction(window)/Fraction(dt)
    width = -(-quotient.numerator//quotient.denominator)
    counterfactual = field_only_counterfactual(electric[:-1],np.zeros(steps),threshold,width)
    return {'c':c, 'gain':gain, 'drive':drive, 'update_eps':update_eps,
            'ratio':ratio, 'equilibrium':equilibrium, 'record':electric[:-1],
            'P':current, 'previous':previous, 'field_only_steps':counterfactual,
            'window_steps':width}


def stopping_receipt(stopping, *, n, dt, cap, threshold, window):
    """Independently audit the public receipt, including a claimed successful decision."""
    quotient = Fraction(window)/Fraction(dt)
    width = (quotient.numerator+quotient.denominator-1)//quotient.denominator
    groups = tuple(stopping['groups'])
    peaks, maxima = stopping['peaks'], stopping['window_max']
    matching = len(groups) == len(set(groups)) and {'E','H'} <= set(groups) and all(
        set(stopping[key]) == set(groups) for key in ('peaks','window_max','ratios'))
    finite = matching and all(math.isfinite(value) and value >= 0
        for mapping in (peaks,maxima,stopping['ratios']) for value in mapping.values())
    seen = peaks.get('E',0.) > 0 or peaks.get('H',0.) > 0
    checks = {
        'reason':stopping['reason'] in ('cap','decay'),
        'reached':stopping['reached'] is (stopping['reason'] == 'decay'),
        'steps':stopping['steps'] == n and stopping['cap_steps'] == cap and 0 < n <= cap,
        'parameters':stopping['threshold'] == threshold and stopping['window'] == window,
        'window_steps':stopping['window_steps'] == width,
        'complete_windows':stopping['complete_windows'] == n//width,
        'groups':matching, 'finite_nonnegative':finite,
        'field_seen':stopping['field_seen'] is seen,
        'eligibility':math.isfinite(stopping['eligible_time']) and stopping['eligible_time'] >= 0,
        'maxima_below_peaks':finite and all(maxima[k] <= peaks[k] for k in groups),
        'display_ratios':finite and all(stopping['ratios'][k] == (maxima[k]/peaks[k] if peaks[k] else 0.) for k in groups),
    }
    if stopping['reason'] == 'cap':
        checks['cap_is_actual_cap'] = n == cap
    elif stopping['reason'] == 'decay':
        checks['decay_complete_window'] = n % width == 0
        checks['decay_eligible'] = Fraction(n-width)*Fraction(dt) >= Fraction(stopping['eligible_time'])
        checks['decay_field_signal'] = seen
        checks['decay_finite'] = not stopping['nonfinite'] and finite
        checks['decay_all_groups'] = finite and all(Fraction(maxima[k]) <=
            Fraction(threshold)*Fraction(peaks[k]) for k in groups)
    return {'receipt_'+key:bool(value) for key,value in checks.items()}


def measure(case, *, backend, threshold, window, checkpoint=None):
    before, started = identity(), now()
    if backend not in BACKENDS:
        raise ValueError('The campaign covers three concrete CPU backends only.')
    scene = build(case, threshold=threshold, window=window)
    adaptive, a_meta = solve(scene, backend)
    if checkpoint is not None:
        checkpoint('adaptive', adaptive, a_meta)
    n = adaptive.plan.steps
    stopping = adaptive.diagnostics['stopping']
    # Same published fixed Run(time) rounding contract, independently evaluated.
    cap = max(1,int(math.ceil(CAP_TIMES[case]/adaptive.plan.dt-1e-9)))
    # The fixed actual-prefix run is also the fixed-cap control when n == cap.
    prefix, p_meta = solve(build(case, threshold=threshold, window=window,
                                 adaptive=False, steps=n), backend)
    p_meta['also_fixed_cap'] = n == cap
    if checkpoint is not None:
        checkpoint('prefix', prefix, p_meta)
    # Twice the explicit cap: one grid/absorber, separate temporal truncation only.
    long_result, l_meta = solve(build(case, threshold=threshold, window=window,
                                      adaptive=False, steps=2*cap), backend)
    if checkpoint is not None:
        checkpoint('long', long_result, l_meta)
    a, p, long = map(outputs, (adaptive, prefix, long_result))
    prefix_comparisons = {key: error(a[key], p[key]) for key in a if not key.startswith('energy_')}
    # Fixed cadence includes its forced final completed row. Match exact timestamps,
    # not an assumed every-sixteenth slice (the final row can be off cadence).
    energy_lookup = {float(t): i for i, t in enumerate(a.get('energy_times', ()))}
    energy_indices = [energy_lookup.get(float(t), -1) for t in p.get('energy_times', ())]
    energy_matched = all(i >= 0 for i in energy_indices)
    if energy_matched:
        prefix_comparisons['energy_common'] = error(a['energy_values'][energy_indices], p['energy_values'])
    else:
        prefix_comparisons['energy_common'] = {'same_shape': False, 'missing_common_times': True}
    spectral = {}
    for key in a:
        if key.startswith('field_') or key == 'flux':
            spectral[key] = [error(a[key][j], long[key][j]) for j in range(len(WAVELENGTHS))]
    source_end = 0.
    for src in scene.sources:
        tau = 1/(2*math.pi*float(src.band.df))
        bound = min(src.band.cutoff, threshold)
        source_end = max(source_end, float(src.delay)+tau*(src.band.sigmas+math.sqrt(-2*math.log(bound))))
    # Metadata contracts are exact; numerical equality is recorded before choosing budgets.
    contracts = {
        'actual_steps_consistent': n == stopping['steps'] == adaptive.report.steps,
        'hard_cap_respected': 0 < n <= cap,
        'record_lengths': all(len(a[key+'_values']) == n for key in ('e', 'h')),
        'electric_clock': bool(np.array_equal(a['e_times'], (np.arange(n)+1.)*adaptive.plan.dt)),
        'magnetic_clock': bool(np.array_equal(a['h_times'], (np.arange(n)+.5)*adaptive.plan.dt)),
        'source_eligibility_at_least_independent': float(stopping['eligible_time']) >= source_end-8*np.finfo(float).eps*max(1.,source_end),
        'energy_rows': len(a.get('energy_times', ())) == max(0,n-1),
        'energy_clock': bool(np.array_equal(a.get('energy_times', ()), np.arange(1,n)*adaptive.plan.dt)),
        'fixed_energy_timestamps_present': energy_matched,
        'all_saved_arrays_finite': all(np.all(np.isfinite(v)) for item in (a,p,long) for v in item.values()),
    }
    contracts.update(stopping_receipt(stopping,n=n,dt=adaptive.plan.dt,cap=cap,threshold=threshold,window=window))
    if stopping['reason'] == 'decay':
        width = stopping['window_steps']
        contracts['complete_window'] = n % width == 0
        contracts['post_source_window'] = Fraction(n-width)*Fraction(adaptive.plan.dt) >= Fraction(stopping['eligible_time'])
        contracts['field_seen'] = bool(stopping['field_seen'])
        contracts['not_nonfinite'] = not stopping['nonfinite']
    if case in ('pec_cavity', 'debye_memory'):
        contracts['required_cap'] = stopping['reason'] == 'cap' and n == cap
    analytic = None
    if case == 'uniform_conductor':
        oracle = uniform_oracle(adaptive.plan.dt, cap, threshold, window)
        analytic = {'q': oracle['q'], 'expected_steps': oracle['expected_steps'],
            'observed_steps': n, 'stop_exact': n == oracle['expected_steps'],
            'record': error(a['e_values'], oracle['expected_record']),
            'phasor': [error(a['field_Ey'][j], np.full(a['field_Ey'][j].shape,
                oracle['expected_phasor'][j])) for j in range(len(WAVELENGTHS))]}
        contracts['uniform_stops_early_exactly'] = analytic['stop_exact'] and n < cap
    retained = None
    if case == 'debye_memory':
        oracle = debye_oracle(adaptive.plan.dt,n,threshold,window)
        counterfactual = field_only_counterfactual(a['e_values'],a['h_values'],threshold,oracle['window_steps'])
        memory = {key: kind for key in stopping['groups'] for kind in ('P','previous')
                  if key.endswith(':'+kind+'[0]')}
        memory_metrics = {key: {'peak':error(stopping['peaks'][key],oracle[kind][-1]),
                               'window_max':error(stopping['window_max'][key],oracle[kind][-1])}
                          for key,kind in memory.items()}
        retained = {'record':error(a['e_values'],oracle['record']),
            'field_only_steps':counterfactual, 'oracle_field_only_steps':oracle['field_only_steps'],
            'scalar_map':{key:oracle[key] for key in ('c','gain','drive','update_eps','ratio','equilibrium')},
            'memory_metrics':memory_metrics,
            'qualification':'Counterfactual contract witness: static retained polarization forces the strict criterion to cap; not a demonstrated later spectral error.'}
        contracts['field_only_would_pass_before_cap'] = counterfactual is not None and counterfactual < cap
        contracts['field_only_matches_discrete_oracle'] = counterfactual == oracle['field_only_steps']
        contracts['retained_memory_blocks'] = len(memory) == 2 and all(stopping['ratios'][key] > threshold for key in memory)
    record = {'kind': 'decay-stop-physical-row', 'case': case, 'backend': backend,
        'threshold': threshold, 'window': window, 'wavelengths': WAVELENGTHS,
        'started_utc': started, 'ended_utc': now(), 'source_before': before,
        'source_after': identity(), 'helper_sha256': sha(Path(__file__)),
        'adaptive': a_meta, 'fixed_prefix': p_meta, 'long_reference': l_meta,
        'cap_time': CAP_TIMES[case], 'independent_source_eligibility': source_end,
        'contracts': contracts, 'prefix': prefix_comparisons, 'spectral_vs_twice_cap': spectral,
        'uniform_oracle': analytic, 'debye_oracle': retained, 'fixed_energy_indices_in_adaptive': energy_indices,
        'qualification': 'Temporal comparison on one fixed spatial grid and one absorber; no spatial or universal error certificate.'}
    if record['source_before'] != record['source_after']:
        raise RuntimeError('Production source changed during this physical row.')
    saved = {}
    for label, values in (('adaptive', a), ('prefix', p), ('long', long)):
        for key, value in values.items():
            saved[label+'__'+key] = value
    return jsonable(record), saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=CASES)
    parser.add_argument('--backend', required=True, choices=BACKENDS)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--window', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arrays', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.out, args.arrays):
        if path.exists() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Fresh evidence paths inside the checkout are required.')
    partials = []
    def checkpoint(role, result, meta):
        values = outputs(result)
        data = args.arrays.with_name(args.arrays.stem+'.'+role+'.npz')
        receipt = args.out.with_name(args.out.stem+'.'+role+'.json')
        with data.open('xb') as stream:
            np.savez_compressed(stream, **values)
        item = {'role': role, 'metadata': jsonable(meta), 'file': relative(data), 'sha256': sha(data)}
        write(receipt, item, exclusive=True)
        partials.append({'file': relative(receipt), 'sha256': sha(receipt)})
    record, arrays = measure(args.case, backend=args.backend, threshold=args.threshold,
                             window=args.window, checkpoint=checkpoint)
    record['completed_stage_receipts'] = partials
    with args.arrays.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    record['arrays'] = {'file': relative(args.arrays), 'sha256': sha(args.arrays),
        'members': {key: {'shape': value.shape, 'dtype': str(value.dtype),
            'raw_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}
                    for key, value in arrays.items()}}
    write(args.out, record, exclusive=True)
    print(json.dumps({'case': args.case, 'backend': args.backend,
                      'result': relative(args.out), 'sha256': sha(args.out)}), flush=True)


if __name__ == '__main__':
    main()
