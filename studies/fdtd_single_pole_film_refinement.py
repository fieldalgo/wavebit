"""Prospectively register a post-hoc selected, bounded single-pole film follow-up.

The original forty-row study remains unchanged, including every low, mixed and
unresolved classification. Only its two exceptional groups are selected, retaining
all three wavelengths and both complex coefficients. The known n80/120 responses
are recomputed; agreement with them is provenance, not independent corroboration.
No field response may precede this controller's separate registration.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
import pathlib
import os
import subprocess
import sys
import time

import numpy as np

from studies import fdtd_single_pole_film_campaign as original
from studies import fdtd_single_pole_film_probe as probe


ROOT, RESULTS = original.ROOT, original.RESULTS
PREFIX = 'fdtd_single_pole_film_refinement_v1'
REGISTRATION = RESULTS/f'{PREFIX}_registration.json'
LEDGER = RESULTS/f'{PREFIX}_campaign.jsonl'
REPORT = RESULTS/f'{PREFIX}_report.json'
JOBS = RESULTS/f'{PREFIX}_campaign_jobs'
AUDIT = RESULTS/f'{PREFIX}_audit.json'
GROUPS = (('drude', .3, 's'), ('lorentz', .3, 'p'))
RUNGS, BASE, LONG, MIDDLE = (80, 160, 320), 120, 240, 160
criteria = original.criteria
read, write, sha, canonical, now, key = (original.read, original.write, original.sha,
                                       original.canonical, original.now, original.key)


def identity():
    return {**probe.source_identity(), **probe.runtime_identity(),
            'refinement_controller_sha256': sha(pathlib.Path(__file__)),
            'original_controller_sha256': sha(pathlib.Path(original.__file__)),
            'criteria_dependency_sha256': sha(pathlib.Path(criteria.__file__)),
            'test_sha256': sha(ROOT/'tests/test_fd_single_pole_film.py')}


def protocol():
    return {**original.protocol(),
        'scope': 'Two post-hoc selected groups from the original single-pole film study: '
                 'Drude fraction.3 s and Lorentz fraction.3 p. All three wavelengths and '
                 'both complex coefficients remain. No extension of the physical scope.',
        'selection': 'Selection followed the complete original forty-row report: the '
                     'selected groups contain its four non-second self-order records. '
                     'This selection is post-hoc; the present follow-up is prospectively '
                     'registered before its new responses. Original verdicts remain unchanged.',
        'groups': [{'family': family, 'fraction': fraction, 'polarization': pol}
                   for family, fraction, pol in GROUPS],
        'group_rule': 'Only the two listed tuples, not their Cartesian product.',
        'rungs': list(RUNGS), 'fractions': [.3], 'base_periods': BASE, 'long_periods': LONG,
        'base_worker_count': 10,
        'sequence_per_placement_polarization': ['80 base120 PML24', '160 base120 PML24',
            '160 long240 PML24', '160 long240 PML48', '320 base120 PML24'],
        'replacement': 'Only an otherwise valid but unsettled 120-period base may receive '
                       'one 240-period replacement. The 160/240/PML24 row is reused if it '
                       'was already required as that replacement. Stop globally if a '
                       '240-period replacement or fixed control is invalid; no 320 row '
                       'follows an invalid earlier selected row/control. No duration beyond '
                       '240, extra rung, changed geometry or further adaptation.',
        'noise_proxy': '|160,long24 - 160,base24| + |160,long48 - 160,long24| per complex '
                       'cell; retain base120 in this proxy even if selected160 is long240.',
        'self_order': 'log2(|v80-v160| / |v160-v320|), with each difference interval +/-2*proxy.',
        'cells': 'Every selected group x wavelength x complex r/t coefficient (12 total) '
                 'is reported independently; no worst-cell substitute.',
        'known_repeated_rows': 'Both n80/120/PML24 configurations were already computed as '
                               'original-study controls. Run them again into new raw files. '
                               'Their exact physical-payload comparison is reported only as '
                               'a provenance/reproducibility control, not corroboration or '
                               'an acceptance gate, and cannot trigger an adaptive rerun.',
        'interpretation': 'Report all resulting second/low/mixed/unresolved labels. Do not '
                          'replace the original study classification, reinterpret an '
                          'unresolved point estimate as an order, or extend the run after '
                          'mixed/low/unresolved outcomes. The middle-grid proxy remains '
                          'empirical and is not a rigorous bound on fine-grid noise.',
    }


def register():
    assert not any(path.exists() for path in (REGISTRATION, LEDGER, REPORT, JOBS, AUDIT))
    prior_registration = original.verify()
    prior_audit = original.audit()
    prior_report = read(original.REPORT)
    assert prior_audit['complete'] and prior_audit['workers'] == 40 and len(prior_report['cells']) == 48
    known = {}

    def add(path, scope):
        filename = str(path.relative_to(RESULTS))
        known[filename] = {'file': filename, 'sha256': sha(path), 'scope': scope}

    for item in prior_registration['known_exposures']+prior_registration['frozen_sources']:
        assert sha(RESULTS/item['file']) == item['sha256']
        known[item['file']] = item
    for path in (original.REGISTRATION, original.LEDGER, original.REPORT,
                 RESULTS/'fdtd_single_pole_film_campaign_audit.log',
                 RESULTS/'fdtd_single_pole_film_campaign_controller.log'):
        add(path, 'Complete original study known before this post-hoc group selection; preserved unchanged.')
    prior_entries = original.ledger()
    for item in prior_entries:
        for name in ('raw_log', 'result'):
            path = ROOT/item[name]
            assert sha(path) == item[name+'_sha256']
            add(path, 'Known original response/raw stdout; all forty rows are disclosed.')
    for path in sorted(RESULTS.glob('fdtd_single_pole_film_backends*')):
        for item in sorted(path.rglob('*')) if path.is_dir() else (path,):
            if item.is_file():
                add(item, 'Known supplementary backend comparison of the repeated calibration configurations.')
    exception_cells = [cell for cell in prior_report['cells']
                       if cell['self_order']['classification'] != 'second']
    assert len(exception_cells) == 4
    assert {(c['family'], c['fraction'], c['polarization']) for c in exception_cells} == set(GROUPS)
    prior_summary = {
        'workers': 40, 'self_counts': dict(Counter(c['self_order']['classification'] for c in prior_report['cells'])),
        'all_direct_counts': dict(Counter(o['classification'] for c in prior_report['cells'] for o in c['direct_orders'])),
        'final_direct_counts': dict(Counter(c['direct_orders'][-1]['classification'] for c in prior_report['cells'])),
        'selection_exception_cells': exception_cells,
    }
    old_map = {entry['key']: entry for entry in prior_entries}
    repeated = [{name: old_map[key(family, fraction, pol, 80, BASE, 24)][name]
                 for name in ('key', 'result', 'result_sha256', 'raw_log', 'raw_log_sha256')}
                for family, fraction, pol in GROUPS]
    frozen = []
    for label, source in (('controller', pathlib.Path(__file__)), ('probe', pathlib.Path(probe.__file__)),
                           ('original_controller', pathlib.Path(original.__file__)),
                           ('criteria', pathlib.Path(criteria.__file__))):
        destination = RESULTS/f'{PREFIX}_registered_{label}.py.txt'
        with destination.open('xb') as stream:
            stream.write(source.read_bytes())
        frozen.append({'file': destination.name, 'sha256': sha(destination)})
    data = {'registered_utc': now(), 'identity': identity(), 'protocol': protocol(),
            'analytic_preflight': {**probe.preflight(), **criteria.order_preflight()},
            'original_registration_fingerprint': prior_registration['fingerprint'],
            'known_original_outcomes': prior_summary, 'known_repeated_rows': repeated,
            'known_exposures': list(known.values()), 'frozen_sources': frozen,
            'qualification': 'Post-hoc selection from disclosed original outcomes, followed '
                             'by a separately registered prospective follow-up. Fixed '
                             'criteria, material/source geometry and capped adaptations. '
                             'No independent corroboration is inferred from repeated rows.'}
    data['fingerprint'] = hashlib.sha256(canonical(data)).hexdigest()
    write(REGISTRATION, data, exclusive=True)
    verify()
    return {'registered': str(REGISTRATION.relative_to(ROOT)), 'fingerprint': data['fingerprint'],
            'known_original_workers': 40, 'planned_base_workers': 10, 'cells': 12}


def verify():
    data = read(REGISTRATION)
    bare = {k: v for k, v in data.items() if k != 'fingerprint'}
    assert hashlib.sha256(canonical(bare)).hexdigest() == data['fingerprint']
    assert data['identity'] == identity(), 'Registered source/runtime/environment changed.'
    assert data['protocol'] == protocol(), 'Registered follow-up protocol changed.'
    for item in data['known_exposures']+data['frozen_sources']:
        assert sha(RESULTS/item['file']) == item['sha256'], item['file']
    return data


def ledger():
    entries = [json.loads(line) for line in LEDGER.read_text(encoding='utf-8').splitlines()] \
        if LEDGER.exists() else []
    assert len({item['key'] for item in entries}) == len(entries), 'Duplicate worker attempt.'
    return entries


def rows():
    available = {}
    for item in ledger():
        if item['exit_code'] == 0:
            path = ROOT/item['result']
            assert sha(path) == item['result_sha256']
            available[item['key']] = read(path)
    return available


def worker(family, fraction, pol, n, periods=BASE, pml=24):
    registration = verify()
    job = key(family, fraction, pol, n, periods, pml)
    existing = {entry['key']: entry for entry in ledger()}
    if job in existing:
        if existing[job]['exit_code'] != 0:
            raise RuntimeError(f'Prior failed worker {job}; no silent retry.')
        return rows()[job]
    JOBS.mkdir(exist_ok=True)
    output, log = JOBS/(job+'.json'), JOBS/(job+'.log')
    if output.exists() or log.exists():
        raise RuntimeError(f'Orphan attempt exists for {job}; preserve and audit it.')
    command = [sys.executable, '-B', '-m', 'studies.fdtd_single_pole_film_probe',
               '--family', family, '--fraction', str(fraction), '--pol', pol, '--n', str(n),
               '--periods', str(periods), '--pml', str(pml), '--output', str(output)]
    start, begin = now(), time.perf_counter()
    print(f'{start} START {job}', flush=True)
    with log.open('xb') as stream:
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    entry = {'key': job, 'started_utc': start, 'ended_utc': now(), 'wall_s': time.perf_counter()-begin,
             'exit_code': process.returncode, 'command': command, 'registration': registration['fingerprint'],
             'raw_log': str(log.relative_to(ROOT)), 'raw_log_sha256': sha(log)}
    if output.exists():
        entry.update(result=str(output.relative_to(ROOT)), result_sha256=sha(output))
    with LEDGER.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(entry, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(f'{entry["ended_utc"]} END {job} exit={process.returncode}', flush=True)
    if process.returncode != 0:
        raise RuntimeError(f'Worker {job} failed; preserve its raw attempt.')
    verify()
    return read(output)


def selected(available, group, n):
    for periods in (BASE, LONG):
        row = available.get(key(*group, n, periods, 24))
        if row is not None and criteria.validity(row)['valid']:
            return row
    return None


def make_report(stop_reason=None):
    registered, available = verify(), rows()
    groups, cells, complete = [], [], True
    for group in GROUPS:
        family, fraction, pol = group
        chosen = [selected(available, group, n) for n in RUNGS]
        controls = [available.get(key(*group, MIDDLE, periods, pml))
                    for periods, pml in ((BASE, 24), (LONG, 24), (LONG, 48))]
        ready = (all(row is not None for row in chosen+controls)
                 and all(criteria.validity(row)['valid'] for row in controls[1:]))
        complete &= ready
        groups.append({'family': family, 'fraction': fraction, 'polarization': pol,
                       'complete': bool(ready), 'selected_periods': [
                           None if row is None else row['config']['periods'] for row in chosen]})
        for name in ('r', 't'):
            for index, wavelength in enumerate(probe.WAVELENGTHS):
                cell = {'family': family, 'fraction': fraction, 'polarization': pol,
                        'coefficient': name, 'wavelength': wavelength, 'resolved_input': bool(ready)}
                if ready:
                    samples = [criteria.values(row)[name][index] for row in chosen]
                    base, long, pml = (criteria.values(row)[name][index] for row in controls)
                    time_noise, pml_noise = abs(long-base), abs(pml-long)
                    proxy = time_noise+pml_noise
                    differences = [abs(samples[0]-samples[1]), abs(samples[1]-samples[2])]
                    errors = [float(row['error_'+name][index]) for row in chosen]
                    cell.update(values=[[v.real, v.imag] for v in samples],
                        continuum=chosen[0]['reference_'+name][index], errors=errors, differences=differences,
                        absorption=[float(row['absorption'][index]) for row in chosen],
                        time_proxy=float(time_noise), pml_proxy=float(pml_noise), proxy=float(proxy),
                        self_order=criteria.order(*differences, proxy, difference=True),
                        direct_orders=[criteria.order(errors[i], errors[i+1], proxy,
                                                     difference=False) for i in (0, 1)])
                else:
                    cell.update(self_order={'classification': 'unresolved',
                                            'reason': 'Missing or invalid registered input.'}, direct_orders=[])
                cells.append(cell)
    return probe._json({'complete': bool(complete and stop_reason is None), 'stop_reason': stop_reason,
        'registration': registered['fingerprint'], 'rungs': list(RUNGS), 'groups': groups, 'cells': cells,
        'worker_rows': len(ledger()), 'scope': protocol()['scope'], 'selection': protocol()['selection'],
        'noise_scope': protocol()['noise_scope'], 'interpretation': protocol()['interpretation']})


def run():
    verify()
    if REPORT.exists() and read(REPORT).get('stop_reason'):
        raise RuntimeError('The follow-up already stopped under its registered cap.')
    stop = None
    try:
        for group in GROUPS:
            for n in RUNGS:
                state = criteria.validity(worker(*group, n))
                if not state['valid']:
                    if not state['otherwise_valid']:
                        raise RuntimeError(f'Invalid base {key(*group, n, BASE, 24)}')
                    if not criteria.validity(worker(*group, n, LONG))['valid']:
                        raise RuntimeError(f'Invalid at registered cap: {key(*group, n, LONG, 24)}')
                if n == MIDDLE:
                    for pml in (24, 48):
                        if not criteria.validity(worker(*group, n, LONG, pml))['valid']:
                            raise RuntimeError(f'Invalid fixed control: {key(*group, n, LONG, pml)}')
    except RuntimeError as error:
        stop = str(error)
    report = make_report(stop)
    write(REPORT, report)
    return {'complete': report['complete'], 'stop_reason': stop, 'workers': report['worker_rows'],
            'report': str(REPORT.relative_to(ROOT))}


def audit_sequence(entries, available):
    index, consumed = 0, {}

    class EndOfAttempts(Exception):
        pass

    def demand(group, n, periods=BASE, pml=24):
        nonlocal index
        job = key(*group, n, periods, pml)
        if job in consumed:
            return consumed[job]
        if index == len(entries):
            raise EndOfAttempts
        entry = entries[index]
        assert entry['key'] == job, ('Unexpected job, replacement or order', entry['key'], job)
        index += 1
        if entry['exit_code'] != 0:
            raise EndOfAttempts
        consumed[job] = available[job]
        return consumed[job]

    try:
        for group in GROUPS:
            for n in RUNGS:
                state = criteria.validity(demand(group, n))
                if not state['valid']:
                    if not state['otherwise_valid']:
                        raise EndOfAttempts
                    if not criteria.validity(demand(group, n, LONG))['valid']:
                        raise EndOfAttempts
                if n == MIDDLE:
                    for pml in (24, 48):
                        if not criteria.validity(demand(group, n, LONG, pml))['valid']:
                            raise EndOfAttempts
    except EndOfAttempts:
        pass
    assert index == len(entries), 'Additional job after the registered stop or outside the fixed plan.'


def audit():
    registered, entries, available = verify(), ledger(), rows()
    observed = set()
    defaults = inspect.signature(probe.build).bind()
    defaults.apply_defaults()
    for entry in entries:
        assert entry['registration'] == registered['fingerprint']
        assert entry['started_utc'] >= registered['registered_utc']
        raw = ROOT/entry['raw_log']
        assert sha(raw) == entry['raw_log_sha256']
        observed.add(raw.name)
        if entry['exit_code'] != 0:
            continue
        output = ROOT/entry['result']
        assert sha(output) == entry['result_sha256']
        observed.add(output.name)
        row = read(output)
        assert read(raw) == row
        assert row['source_identity'] == probe.source_identity() and row['source_unchanged_during_run']
        assert row['source_band'] == protocol()['source_band'] and row['wavelengths'] == list(probe.WAVELENGTHS)
        for name, value in probe.runtime_identity().items():
            assert row[name] == value, name
        config, expected = row['config'], dict(defaults.arguments)
        for name in ('family', 'fraction', 'pol', 'n', 'periods', 'pml'):
            expected[name] = config[name]
            assert entry['command'][entry['command'].index('--'+name)+1] == str(config[name])
        assert config == {'backend': 'numpy', **expected}
        assert (config['family'], config['fraction'], config['pol']) in GROUPS
        assert config['n'] in RUNGS and config['periods'] in (BASE, LONG)
        assert config['pml'] == 24 or (config['n'] == MIDDLE and config['periods'] == LONG and config['pml'] == 48)
        assert key(*(config[name] for name in ('family', 'fraction', 'pol', 'n', 'periods', 'pml'))) == entry['key']
        assert pathlib.Path(entry['command'][entry['command'].index('--output')+1]) == output
        kx = 2*math.pi*math.sin(math.radians(config['theta']))
        assert row['kx'] == kx
        material = {name: config[name] for name in ('family', 'eps', 'gamma', 'omega_p', 'omega0', 'strength')}
        reference = probe.airy(probe.WAVELENGTHS, **material, kx=kx, pol=config['pol'], thickness=config['thickness'])
        np.testing.assert_array_equal(np.asarray([complex(*v) for v in row['epsilon']]),
                                      probe.epsilon(probe.WAVELENGTHS, **material))
        for name, value in zip(('r', 't'), reference):
            np.testing.assert_array_equal(np.asarray([complex(*v) for v in row['reference_'+name]]), value)
            np.testing.assert_array_equal(np.asarray(row['error_'+name]), abs(criteria.values(row)[name]-value))
        np.testing.assert_array_equal(np.asarray(row['absorption']), 1-np.asarray(row['R'])-np.asarray(row['T']))
        np.testing.assert_array_equal(np.asarray(row['reference_absorption']), 1-abs(reference[0])**2-abs(reference[1])**2)
        assert row['started_utc'] >= registered['registered_utc'] and row['ended_utc'] >= row['started_utc']
    assert {path.name for path in JOBS.iterdir()} == observed, 'Unledgered raw attempt.'
    audit_sequence(entries, available)
    saved = read(REPORT)
    assert saved == make_report(saved['stop_reason']), 'Report does not reconstruct exactly.'
    assert len(saved['cells']) == 12
    physical_fields = ('config', 'grid', 'dt', 'steps', 'kx', 'source_band', 'wavelengths', 'epsilon',
        'r', 't', 'reference_r', 'reference_t', 'error_r', 'error_t', 'R', 'T', 'absorption',
        'reference_absorption', 'stable', 'decay', 'energy_shape', 'energy_rows', 'energy_finite', 'energy_max')
    comparisons = []
    for prior in registered['known_repeated_rows']:
        current = available.get(prior['key'])
        if current is None:
            comparisons.append({'key': prior['key'], 'status': 'not reached under registered cap'})
            continue
        old = read(ROOT/prior['result'])
        comparisons.append({'key': prior['key'], 'status': 'compared',
            'exact_numeric_agreement': {name: old[name] == current[name] for name in physical_fields},
            'canonical_physical_payload_identical': canonical({name: old[name] for name in physical_fields})
                == canonical({name: current[name] for name in physical_fields})})
    summary = {'audit': 'passed', 'audited_utc': now(), 'registration': registered['fingerprint'],
        'workers': len(entries), 'complete': saved['complete'], 'all_12_cells_recomputed': True,
        'stop_reason': saved['stop_reason'], 'self_counts': dict(Counter(
            c['self_order']['classification'] for c in saved['cells'])),
        'all_direct_counts': dict(Counter(o['classification'] for c in saved['cells'] for o in c['direct_orders'])),
        'final_direct_counts': dict(Counter(c['direct_orders'][-1]['classification']
            for c in saved['cells'] if c['direct_orders'])), 'known_row_comparisons': comparisons,
        'comparison_qualification': 'Repeated known rows are provenance controls, not independent '
                                    'corroboration or acceptance gates; no adaptive rerun follows a difference.',
        'original_verdict_unchanged': registered['known_original_outcomes'],
        'noise_scope': protocol()['noise_scope']}
    write(AUDIT, summary, exclusive=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'report', 'audit'))
    args = parser.parse_args()
    if args.action == 'report':
        result = make_report(read(REPORT).get('stop_reason') if REPORT.exists() else None)
    else:
        result = {'register': register, 'run': run, 'audit': audit}[args.action]()
    print(json.dumps(probe._json(result), indent=2, allow_nan=False))
    if args.action == 'run' and not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
