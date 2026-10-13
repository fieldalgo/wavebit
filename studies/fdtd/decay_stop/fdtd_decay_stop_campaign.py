"""Sequential, preregistered physical evidence for bounded decay stopping.

Calibration discloses results before any empirical spectral ceiling is accepted.
The campaign has no replacements, sweep, aggregate worst-cell selection or
extension after looking at a response. Raw complete/partial attempts are kept.
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
import sys

from studies.fdtd.decay_stop.fdtd_decay_stop_support import ROOT, RESULTS, PREFIX, NEW_TEST, load, write, sha, relative, inside, require, now, attempt_record, finish, run_command, verify_guard, guard, snapshot, helpers

CASES = ('uniform_conductor', 'pec_cavity', 'late_dipoles',
         'debye_memory', 'weak_pml_echo', 'open_dipole')
BACKENDS = ('numpy', 'jax', 'torch')
CALIBRATION = ('uniform_conductor', 'open_dipole')
REGISTRATION = RESULTS/(PREFIX+'registration.json')
ACCEPTANCE = RESULTS/(PREFIX+'calibration_acceptance.json')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def protocol():
    return {'cases': list(CASES), 'backends': list(BACKENDS),
        'sequence': [[case, backend] for case in CASES for backend in BACKENDS],
        'solves_per_row': ['adaptive', 'fixed_actual_prefix', 'fixed_twice_cap'],
        'no_replacements': True, 'no_response_based_extension': True,
        'wavelengths': [.9, 1., 1.1],
        'cap_times': {'uniform_conductor':20.,'pec_cavity':12.,'late_dipoles':30.,
                      'debye_memory':12.,'weak_pml_echo':24.,'open_dipole':16.},
        'quantities': 'Every declared field component and flux at every wavelength; every saved actual-prefix array.',
        'cap_rows': ['pec_cavity', 'debye_memory'],
        'required_early_stop': ['uniform_conductor'],
        'other_rows': 'A cap is an explicit nonconvergence outcome, never a success claim.',
        'spectral_comparison': 'Same grid/material/PML, adaptive versus fixed twice-cap. Compare every cell; no spatial theorem.',
        'error': 'max componentwise real/imag difference / max componentwise real/imag reference; zero denominator is unresolved unless exact zero.',
        'accepted_limits': 'External post-disclosure acceptance JSON binds threshold, window and prefix/spectral budgets.',
        'source_gate': 'Gaussian envelope eligibility min(cutoff,threshold), not zero forcing.',
        'refusals': 'Independent targeted tests require CPU concrete-only initial/Dipole scope; unsupported source/AD/device routes never enter this campaign.',
        'procedural_failure': 'Preserve the raw output, stop on worker failure; do not replace it.'}


def archived_sources(directory):
    folder = directory/'sources'
    folder.mkdir(exist_ok=False)
    paths = sorted((ROOT/'src/wavebit').rglob('*.py'))+helpers()+[ROOT/NEW_TEST]
    return [snapshot(p, folder/(relative(p).replace('/', '__')+'.raw.gz')) for p in paths]


def worker(record, manifest, directory, case, backend, threshold, window):
    key = case+'__'+backend
    output, arrays = directory/(key+'.json'), directory/(key+'.npz')
    command = [sys.executable, '-B', 'studies/'+PREFIX+'probe.py',
        '--case', case, '--backend', backend, '--threshold', repr(threshold),
        '--window', repr(window), '--out', relative(output), '--arrays', relative(arrays)]
    stage, _ = run_command(record, manifest, key, command, directory, cache_tag=key)
    artifacts = {relative(p): sha(p) for p in sorted(directory.glob(key+'*')) if p.is_file()}
    item = {'case': case, 'backend': backend, 'check_index': len(record['checks'])-1,
            'artifacts': artifacts, 'exit': stage['exit']}
    if output.exists():
        item.update(result=relative(output), result_sha256=sha(output))
    record.setdefault('workers', []).append(item)
    write(manifest, record)
    verify_guard(record['guard_before'])
    require(stage['exit'] == 0, 'Physical worker failed; keep all artifacts and do not replace it.')
    require(output.exists() and arrays.exists(), 'Successful worker must save both raw forms.')
    return load(output)


def calibrate(attempt, threshold, window):
    require(0 < threshold < 1 and math.isfinite(window) and window > 0, 'Explicit valid criterion required.')
    directory, manifest, record = attempt_record('calibration', attempt)
    record.update(threshold=threshold, window=window,
        qualification='Development responses, not unseen validation and no accepted spectral tolerance yet.',
        source_snapshots=archived_sources(directory))
    write(manifest, record)
    for case in CALIBRATION:
        for backend in BACKENDS:
            worker(record, manifest, directory, case, backend, threshold, window)
    finish('calibration', manifest, record)


def linked(value):
    path = inside(value['path'])
    require(sha(path) == value['sha256'], 'Accepted prerequisite hash moved.')
    return load(path)


def register(attempt):
    require(not REGISTRATION.exists(), 'Preserve the original registration; no replacement.')
    accepted = load(ACCEPTANCE)
    require(accepted['decision'] == 'accepted', 'Explicit post-disclosure acceptance required.')
    calibration = linked(accepted['calibration_manifest'])
    require(calibration['complete'] and len(calibration['workers']) == 6, 'All six calibration rows required.')
    require([[w['case'], w['backend']] for w in calibration['workers']] ==
            [[case, backend] for case in CALIBRATION for backend in BACKENDS], 'Calibration roster moved.')
    for worker_row in calibration['workers']:
        require(worker_row['exit'] == 0, 'Calibration worker did not complete.')
        for name, digest in worker_row['artifacts'].items():
            require(sha(inside(name)) == digest, 'Disclosed calibration artifact changed.')
    require(calibration['ended_utc'] <= accepted['accepted_utc'] <= now(), 'Calibration/acceptance chronology invalid.')
    for name in ('threshold', 'window'):
        require(accepted[name] == calibration[name], 'Criterion changed; disclose a fresh calibration first.')
    for name in ('prefix_scaled_ceiling', 'spectral_relative_ceiling', 'analytic_scaled_ceiling'):
        require(math.isfinite(accepted[name]) and accepted[name] > 0, 'A finite positive reviewed budget is required.')
    target_path, mutation_path = (RESULTS/(PREFIX+name+'.json') for name in ('targeted', 'mutations'))
    targeted, mutations = load(target_path), load(mutation_path)
    require(targeted['complete'], 'Targeted gates have not completed.')
    require(mutations['passed'] and mutations['restored'] and mutations['all_function_assertion_witnesses'],
            'Actual production assertion witnesses and exact restoration required.')
    verify_guard(targeted['guard_before'])
    verify_guard(mutations['guard_before'])
    current = guard()
    original_guard = calibration['guard_before']
    for name in ('wavebit', 'goldens', 'previous_film_evidence'):
        require(original_guard['maps'][name] == current['maps'][name],
                'Frozen non-test inventory changed after calibration: '+name)
    require(original_guard['helpers'] == current['helpers'] and
            original_guard['references'] == current['references'],
            'A numerical helper or reference changed after calibration.')
    before_tests, after_tests = original_guard['maps']['tests'], current['maps']['tests']
    require(set(before_tests) == set(after_tests), 'Test roster changed after calibration.')
    changed = {path for path in before_tests if before_tests[path] != after_tests[path]}
    declared = accepted['test_budget_changes']
    require(changed == set(declared) and changed <= {NEW_TEST},
            'Only a declared new-test measured assertion-budget edit may follow calibration.')
    require(isinstance(accepted['budget_justification'], str) and accepted['budget_justification'].strip(),
            'Disclosed measurements must justify the selected budgets.')
    for path, change in declared.items():
        require(change['before_sha256'] == before_tests[path] and
                change['after_sha256'] == after_tests[path] and change['reason'].strip(),
                'Assertion-budget change lacks exact before/after identities or explanation.')
    directory, manifest, record = attempt_record('registration', attempt)
    payload = {'protocol': protocol(), 'accepted': accepted,
        'acceptance': {'path': relative(ACCEPTANCE), 'sha256': sha(ACCEPTANCE)},
        'prerequisites': {relative(p): sha(p) for p in (target_path, mutation_path)},
        'known_responses': [w for w in calibration['workers']],
        'guard': current, 'registered_utc': now()}
    for prerequisite in (targeted, mutations):
        require(prerequisite['ended_utc'] <= payload['registered_utc'], 'Development gate ended after registration.')
    payload['fingerprint'] = hashlib.sha256(canonical(payload)).hexdigest()
    record.update(registration=payload, source_snapshots=archived_sources(directory))
    write(manifest, record)
    finish('registration', manifest, record)


def registration():
    row = load(REGISTRATION)
    require(row['complete'], 'Incomplete registration.')
    payload = row['registration']
    fingerprint = payload['fingerprint']
    unsigned = {k: v for k, v in payload.items() if k != 'fingerprint'}
    require(hashlib.sha256(canonical(unsigned)).hexdigest() == fingerprint, 'Registration fingerprint mismatch.')
    require(payload['protocol'] == protocol(), 'Protocol changed after registration.')
    verify_guard(payload['guard'])
    require(sha(ACCEPTANCE) == payload['acceptance']['sha256'], 'Accepted criteria moved.')
    return payload


def replay_receipt(row, accepted):
    """Reconstruct the public stopping claims without importing the probe/detector."""
    stop = row['adaptive']['stopping']
    n,dt = row['adaptive']['steps'],row['adaptive']['dt']
    threshold,window = accepted['threshold'],accepted['window']
    cap = max(1,int(math.ceil(protocol()['cap_times'][row['case']]/dt-1e-9)))
    ratio = Fraction(window)/Fraction(dt)
    width = -(-ratio.numerator//ratio.denominator)
    names = tuple(stop['groups'])
    p,m = stop['peaks'],stop['window_max']
    keys = len(names) == len(set(names)) and {'E','H'} <= set(names) and all(
        set(stop[key]) == set(names) for key in ('peaks','window_max','ratios'))
    finite = keys and all(math.isfinite(v) and v >= 0 for values in (p,m,stop['ratios']) for v in values.values())
    seen = p.get('E',0.) > 0 or p.get('H',0.) > 0
    fields = {'reason':stop['reason'] in ('cap','decay'),
        'reached':stop['reached'] is (stop['reason'] == 'decay'),
        'steps':stop['steps'] == n and stop['cap_steps'] == cap and 0 < n <= cap,
        'parameters':stop['threshold'] == threshold and stop['window'] == window,
        'window_steps':stop['window_steps'] == width,'complete_windows':stop['complete_windows'] == n//width,
        'groups':keys,'finite_nonnegative':finite,'field_seen':stop['field_seen'] is seen,
        'eligibility':math.isfinite(stop['eligible_time']) and stop['eligible_time'] >= 0,
        'maxima_below_peaks':finite and all(m[k] <= p[k] for k in names),
        'display_ratios':finite and all(stop['ratios'][k] == (m[k]/p[k] if p[k] else 0.) for k in names)}
    if stop['reason'] == 'cap':
        fields['cap_is_actual_cap'] = n == cap
    elif stop['reason'] == 'decay':
        fields.update(decay_complete_window=n % width == 0,
            decay_eligible=Fraction(n-width)*Fraction(dt) >= Fraction(stop['eligible_time']),
            decay_field_signal=seen,decay_finite=not stop['nonfinite'] and finite,
            decay_all_groups=finite and all(Fraction(m[k]) <= Fraction(threshold)*Fraction(p[k]) for k in names))
    return {'receipt_'+key:bool(value) for key,value in fields.items()}


def bounded_pass(metric, limit):
    return bool(metric.get('same_shape') and metric.get('finite') and
                metric.get('scaled') is not None and metric['scaled'] <= limit)


def assess(row, accepted):
    receipt = replay_receipt(row,accepted)
    require(receipt == {key:value for key,value in row['contracts'].items() if key.startswith('receipt_')},
            'Stopping receipt replay disagrees with the physical row.')
    require(row['threshold'] == accepted['threshold'] and row['window'] == accepted['window'],
            'Physical row criterion differs from acceptance.')
    failed = [key for key, value in row['contracts'].items() if value is not True]
    prefix = {key: bounded_pass(value, accepted['prefix_scaled_ceiling']) for key, value in row['prefix'].items()}
    failed += ['prefix:'+key for key, ok in prefix.items() if not ok]
    spectral = {}
    if row['adaptive']['stopping']['reason'] == 'decay':
        spectral = {key: [bounded_pass(metric, accepted['spectral_relative_ceiling']) for metric in values]
                    for key, values in row['spectral_vs_twice_cap'].items()}
        failed += [f'spectral:{key}:{j}' for key, values in spectral.items() for j, ok in enumerate(values) if not ok]
    analytic = row['uniform_oracle']
    if analytic is not None:
        if not bounded_pass(analytic['record'], accepted['analytic_scaled_ceiling']):
            failed.append('analytic:record')
        failed += ['analytic:phasor:'+str(j) for j, metric in enumerate(analytic['phasor'])
                   if not bounded_pass(metric, accepted['analytic_scaled_ceiling'])]
    retained = row['debye_oracle']
    if retained is not None:
        if not bounded_pass(retained['record'],accepted['analytic_scaled_ceiling']):
            failed.append('debye:record')
        failed += ['debye:'+key+':'+label for key,values in retained['memory_metrics'].items()
                   for label,metric in values.items()
                   if not bounded_pass(metric,accepted['analytic_scaled_ceiling'])]
    return {'case': row['case'], 'backend': row['backend'], 'passed': not failed, 'failed': failed,
            'reason': row['adaptive']['stopping']['reason'], 'steps': row['adaptive']['steps'],
            'cap_steps': row['adaptive']['stopping']['cap_steps'],
            'prefix': prefix, 'spectral': spectral,
            'spectral_qualification': 'Checked only a claimed decay; cap retains the raw comparison but does not claim convergence.'}


def run(attempt):
    registered = registration()
    accepted = registered['accepted']
    directory, manifest, record = attempt_record('campaign', attempt)
    record.update(fingerprint=registered['fingerprint'], registration_sha256=sha(REGISTRATION),
                  protocol=protocol(), assessments=[])
    write(manifest, record)
    for case, backend in registered['protocol']['sequence']:
        row = worker(record, manifest, directory, case, backend, accepted['threshold'], accepted['window'])
        record['assessments'].append(assess(row, accepted))
        if case in CALIBRATION:
            known = next(item for item in registered['known_responses']
                         if item['case'] == case and item['backend'] == backend)
            require(sha(inside(known['result'])) == known['result_sha256'], 'Known response changed.')
            original = load(inside(known['result']))
            record.setdefault('known_repeats', []).append({'case': case, 'backend': backend,
                'original_result': known['result'], 'original_sha256': known['result_sha256'],
                'same_array_members': row['arrays']['members'] == original['arrays']['members'],
                'same_stopping': row['adaptive']['stopping'] == original['adaptive']['stopping'],
                'qualification': 'Previously disclosed calibration response, not unseen validation.'})
        write(manifest, record)
    record['procedural_complete'] = True
    record['physical_pass'] = all(row['passed'] for row in record['assessments'])
    write(manifest, record)
    require(record['physical_pass'], 'A registered physical criterion failed; preserve it, do not widen or replace.')
    finish('campaign', manifest, record)


def audit(attempt):
    """Saved-data replay only; no field solve or import of the physical probe."""
    import numpy as np
    registered = registration()
    path = RESULTS/(PREFIX+'campaign.json')
    campaign = load(path)
    require(campaign['complete'] and campaign['physical_pass'], 'Campaign did not complete and pass.')
    require([[w['case'],w['backend']] for w in campaign['workers']] == protocol()['sequence'], 'Worker roster moved.')
    require(len(campaign['assessments']) == len(campaign['workers']) == len(campaign['checks']),
            'Each physical worker needs exactly one actual call and one assessment.')
    require([w['check_index'] for w in campaign['workers']] == list(range(len(campaign['workers']))),
            'Actual command indices are missing, reordered or duplicated.')
    directory, manifest, record = attempt_record('campaign_audit', attempt)
    evidence = {relative(path): sha(path), relative(REGISTRATION): sha(REGISTRATION)}
    for worker_row, saved in zip(campaign['workers'], campaign['assessments']):
        check = campaign['checks'][worker_row['check_index']]
        require(check['exit'] == worker_row['exit'] == 0, 'Worker command did not finish successfully.')
        require(registered['registered_utc'] <= check['started_utc'] <= check['ended_utc'],
                'Worker ran before registration or has an invalid clock.')
        expected_tail = ['-B', 'studies/'+PREFIX+'probe.py', '--case', worker_row['case'],
            '--backend', worker_row['backend'], '--threshold', repr(registered['accepted']['threshold']),
            '--window', repr(registered['accepted']['window']), '--out', worker_row['result'],
            '--arrays', worker_row['result'].removesuffix('.json')+'.npz']
        require(check['command'][1:] == expected_tail, 'Worker command differs from registered controls.')
        for name, digest in worker_row['artifacts'].items():
            require(sha(inside(name)) == digest, 'Raw physical artifact changed.')
            evidence[name] = digest
        row = load(inside(worker_row['result']))
        require(sha(inside(worker_row['result'])) == worker_row['result_sha256'], 'Raw result changed.')
        require(row['case'] == worker_row['case'] and row['backend'] == worker_row['backend'], 'Worker metadata mismatch.')
        arrays = row['arrays']
        require(sha(inside(arrays['file'])) == arrays['sha256'], 'Raw NPZ changed.')
        with np.load(inside(arrays['file']), allow_pickle=False) as raw:
            require(set(raw.files) == set(arrays['members']), 'Raw array member inventory changed.')
            require(all(np.all(np.isfinite(raw[key])) for key in raw.files), 'Nonfinite saved physical array.')
            for key, meta in arrays['members'].items():
                value = raw[key]
                require(list(value.shape) == meta['shape'] and str(value.dtype) == meta['dtype'], 'Array metadata mismatch.')
                require(hashlib.sha256(value.tobytes(order='C')).hexdigest() == meta['raw_sha256'], 'Array bytes mismatch.')
            for key, metric in row['prefix'].items():
                if key == 'energy_common':
                    lookup = {float(t): i for i,t in enumerate(raw['adaptive__energy_times'])}
                    indices = [lookup[float(t)] for t in raw['prefix__energy_times']]
                    require(indices == row['fixed_energy_indices_in_adaptive'], 'Energy timestamp index map moved.')
                    a, b = raw['adaptive__energy_values'][indices], raw['prefix__energy_values']
                else:
                    a, b = raw['adaptive__'+key], raw['prefix__'+key]
                require(a.shape == b.shape, 'Actual prefix shape differs.')
                diff = float(max(np.max(np.abs((a-b).real)), np.max(np.abs((a-b).imag)))) if a.size else 0.
                scale = float(max(np.max(np.abs(b.real)), np.max(np.abs(b.imag)))) if b.size else 0.
                replay = {'same_shape': True, 'absolute': diff, 'scale': scale,
                    'scaled': diff/scale if scale else (0. if diff == 0. else None),
                    'equal_bits': bool(a.dtype == b.dtype and a.tobytes() == b.tobytes()),
                    'finite': bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))}
                require(replay == metric, 'Complete prefix metric replay disagrees.')
            for key, metrics in row['spectral_vs_twice_cap'].items():
                for j, metric in enumerate(metrics):
                    a, b = raw['adaptive__'+key][j], raw['long__'+key][j]
                    diff = float(max(np.max(np.abs((a-b).real)), np.max(np.abs((a-b).imag))))
                    scale = float(max(np.max(np.abs(b.real)), np.max(np.abs(b.imag))))
                    replay = {'same_shape': a.shape == b.shape, 'absolute': diff, 'scale': scale,
                        'scaled': diff/scale if scale else (0. if diff == 0. else None),
                        'equal_bits': bool(a.dtype == b.dtype and a.tobytes() == b.tobytes()),
                        'finite': bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))}
                    require(replay == metric, 'Complete spectral metric replay disagrees.')
            if row['case'] == 'uniform_conductor':
                dt = row['adaptive']['dt']
                threshold, window = row['threshold'], row['window']
                q = (2.-4.*dt/2)/(2.+4.*dt/2)
                quotient = Fraction(window)/Fraction(dt)
                width = (quotient.numerator+quotient.denominator-1)//quotient.denominator
                stop = row['adaptive']['stopping']['cap_steps']
                for end in range(width, stop+1, width):
                    if q**(end-width+1) <= threshold:
                        stop = end
                        break
                oracle = row['uniform_oracle']
                require(oracle['q'] == q and oracle['expected_steps'] == stop and
                        oracle['observed_steps'] == row['adaptive']['steps'] and
                        oracle['stop_exact'] == (stop == row['adaptive']['steps']), 'Scalar oracle metadata mismatch.')
                record_reference = q**np.arange(1,stop+1)
                omega = 2*math.pi/np.asarray(protocol()['wavelengths'])
                phase = np.exp(1j*omega*dt)
                z = q*phase
                summed = dt*z*(1-z**stop)/(1-z)
                symbol = 2*np.sin(omega*dt/2)/dt
                phasor = summed+(1j/symbol)*np.exp(1j*omega*dt/2)*(q**stop*phase**stop-1.)
                pairs = [(raw['adaptive__e_values'], record_reference, oracle['record'])]
                pairs += [(raw['adaptive__field_Ey'][j],
                           np.full(raw['adaptive__field_Ey'][j].shape,phasor[j]), oracle['phasor'][j])
                          for j in range(len(phasor))]
                for a,b,metric in pairs:
                    require(a.shape == b.shape, 'Analytic reference has a wrong clock/shape.')
                    diff = float(max(np.max(np.abs((a-b).real)),np.max(np.abs((a-b).imag))))
                    scale = float(max(np.max(np.abs(b.real)),np.max(np.abs(b.imag))))
                    replay = {'same_shape': True, 'absolute': diff, 'scale': scale,
                        'scaled': diff/scale if scale else (0. if diff == 0. else None),
                        'equal_bits': bool(a.dtype == b.dtype and a.tobytes() == b.tobytes()),
                        'finite': bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))}
                    require(replay == metric, 'Complete analytic metric replay disagrees.')
            if row['case'] == 'debye_memory':
                dt,n = row['adaptive']['dt'],row['adaptive']['steps']
                delta,tau,eps = 1e6,5e5,1.
                c = delta*dt/(2*tau+dt)
                g = (2*tau-dt)/(2*tau+dt)
                d = c*(1+g)
                u = eps+c
                r,equilibrium = g-d/u,u/(eps+delta)
                reference = equilibrium+(1-equilibrium)*r**np.arange(n+1)
                q = Fraction(row['window'])/Fraction(dt)
                width = (q.numerator+q.denominator-1)//q.denominator
                def field_only(e,h):
                    peaks = [1.,0.]
                    for end in range(width,len(e)+1,width):
                        maxima = [float(max(np.max(np.abs(v[end-width:end].real)),
                                           np.max(np.abs(v[end-width:end].imag)))) for v in (e,h)]
                        peaks = [max(old,new) for old,new in zip(peaks,maxima)]
                        if all(Fraction(v) <= Fraction(row['threshold'])*Fraction(p)
                               for v,p in zip(maxima,peaks)):
                            return end
                    return None
                oracle = row['debye_oracle']
                require(oracle['scalar_map'] == {'c':c,'gain':g,'drive':d,'update_eps':u,
                        'ratio':r,'equilibrium':equilibrium}, 'Debye map changed.')
                require(oracle['field_only_steps'] == field_only(raw['adaptive__e_values'],raw['adaptive__h_values'])
                        and oracle['oracle_field_only_steps'] == field_only(reference[:-1],np.zeros(n)),
                        'Field-only counterfactual replay disagrees.')
                stopping = row['adaptive']['stopping']
                groups = {key:kind for key in stopping['groups'] for kind in ('P','previous')
                          if key.endswith(':'+kind+'[0]')}
                require(len(groups) == 2 and set(groups) == set(oracle['memory_metrics']), 'Retained-memory roster mismatch.')
                pairs = [(raw['adaptive__e_values'],reference[:-1],oracle['record'])]
                for key,kind in groups.items():
                    expected = u*(1-reference[-1 if kind == 'P' else -2])
                    pairs += [(np.asarray(stopping[label][key]),np.asarray(expected),oracle['memory_metrics'][key][field])
                              for label,field in (('peaks','peak'),('window_max','window_max'))]
                for a,b,metric in pairs:
                    require(a.shape == b.shape, 'Debye oracle shape changed.')
                    diff = float(max(np.max(np.abs((a-b).real)),np.max(np.abs((a-b).imag))))
                    scale = float(max(np.max(np.abs(b.real)),np.max(np.abs(b.imag))))
                    replay = {'same_shape':True,'absolute':diff,'scale':scale,
                        'scaled':diff/scale if scale else (0. if diff == 0. else None),
                        'equal_bits':bool(a.dtype == b.dtype and a.tobytes() == b.tobytes()),
                        'finite':bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))}
                    require(replay == metric, 'Debye oracle metric replay disagrees.')
        require(assess(row, registered['accepted']) == saved, 'Saved verdict replay disagrees.')
    record.update(inputs=evidence, workers=len(campaign['workers']), saved_array_replay=True,
                  qualification='Controller arithmetic replay; independent reviewer still checks fixture/oracle and all normalization clocks.')
    finish('campaign_audit', manifest, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('calibrate', 'register', 'run', 'audit'))
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--window', type=float)
    args = parser.parse_args()
    if args.mode == 'calibrate':
        require(args.threshold is not None and args.window is not None, 'Calibration criterion must be explicit.')
        calibrate(args.attempt, args.threshold, args.window)
    else:
        globals()[args.mode](args.attempt)
