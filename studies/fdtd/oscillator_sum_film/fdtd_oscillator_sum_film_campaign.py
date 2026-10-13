"""Register, run and audit the bounded 40-row finite oscillator-sum film characterization.

The procedural order/noise functions are reused unchanged from the preserved
conductive-film controller and hashed as a dependency. No original artifact is
modified. Registration discloses the six backend calibration rows and all development
exposures; every new raw attempt is retained under this study's own prefix.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import argparse
from collections import Counter
import datetime
import hashlib
import inspect
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

from studies.fdtd.conductive_film import fdtd_conductive_film_campaign as criteria
from studies.fdtd.oscillator_sum_film import fdtd_oscillator_sum_film_probe as probe


ROOT = repository_root()
RESULTS = ROOT/'studies'/'results'
PREFIX = 'fdtd_oscillator_sum_film'
REGISTRATION = RESULTS/f'{PREFIX}_registration.json'
LEDGER = RESULTS/f'{PREFIX}_campaign.jsonl'
REPORT = RESULTS/f'{PREFIX}_report.json'
JOBS = RESULTS/f'{PREFIX}_campaign_jobs'
FRACTIONS, POLS, RUNGS = (.3, .1), ('s', 'p'), (40, 80, 160)
BASE, LONG = 120, 240


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(probe._json(value), sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def write(path, value, *, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(probe._json(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def identity():
    return {**probe.source_identity(), **probe.runtime_identity(),
            'campaign_sha256': sha(pathlib.Path(__file__)),
            'criteria_dependency_sha256': sha(pathlib.Path(criteria.__file__)),
            'test_sha256': sha(ROOT/'tests/fdtd/test_fd_oscillator_sum_film.py')}


def protocol():
    inherited = criteria.protocol()
    return {**inherited,
        'scope': 'Two fixed synthetic oscillator-sum films in vacuum, eps_inf1, '
                 'thickness.2 theta35, s/p, exact Axis1D transverse derivative. '
                 'The mixed-metal and three-Lorentz examples are parameter-specific; '
                 'other counts and material parameters require their own validation.',
        'materials': {family: {'eps_inf': 1., 'pole_specs': probe._json(probe.POLE_SPECS[family])}
                      for family in probe.FAMILIES},
        'base_commit': '47541723',
        'rungs': list(RUNGS), 'fractions': list(FRACTIONS), 'polarizations': list(POLS),
        'wavelengths': list(probe.WAVELENGTHS), 'base_worker_count': 40,
        'family_sequence': list(probe.FAMILIES), 'base_periods': BASE, 'long_periods': LONG,
        'sequence_per_placement_polarization': ['40 base 120 PML24', '80 base 120 PML24',
            '80 long240 PML24', '80 long240 PML48', '160 base 120 PML24'],
        'replacement': 'Only an otherwise valid but unsettled 120-period base row may receive '
                       'one 240-period replacement. Reuse the fixed 80/240/PML24 control '
                       'if already computed. Stop globally if the 240-period replacement '
                       'or fixed control is invalid. No duration beyond 240, additional '
                       'grid rung, geometry change or adaptation to order classifications.',
        'noise_proxy': '|80,long24-80,base24|+|80,long48-80,long24| per complex cell; '
                       'retain base 120 in this proxy even if selected 80 is long240.',
        'cells': 'Every placement x polarization x wavelength x complex r/t (48 total) '
                 'is reported independently; no worst-cell substitute.',
        'reference': 'Independent declared-sum Drude/Lorentz rational epsilon and continuum Airy '
                     'admittances, with no production eps_at or split-memory helper.',
        'known_exposure': 'Two coarse p/fraction.3 material configurations are disclosed on all three '
                          'backends before registration; targeted and mutation exposures '
                          'repeat those configurations. Each matching NumPy campaign row '
                          'is a repeated known response, not independent corroboration.',
        'known_warning': 'Retain every calibration/source warning. The fixed geometry '
                         'is not adapted to a warning or outcome; registered time/PML '
                         'controls measure response sensitivity.',
    }


def register():
    assert not any(path.exists() for path in (REGISTRATION, LEDGER, REPORT, JOBS))
    analytic = {**probe.preflight(), **criteria.order_preflight()}
    known = {}

    def add(path, scope):
        filename = str(path.relative_to(RESULTS))
        known[filename] = {'file': filename, 'sha256': sha(path), 'scope': scope}

    calibrations = sorted(RESULTS.glob(f'{PREFIX}_calibration_attempt*.json'))
    assert calibrations, 'The six disclosed backend calibration rows must exist before registration.'
    complete = False
    for path in calibrations:
        data = read(path)
        add(path, 'Disclosed development calibration; not unseen-outcome study data.')
        complete |= (data['complete'] and len(data['workers']) == 6
                     and data['backend_pairs_within_budget'])
        for item in data['frozen_sources']:
            source = ROOT/item['file']
            assert sha(source) == item['sha256']
            add(source, 'Exact calibration helper/controller snapshot recorded before field responses.')
        for item in data['workers']:
            for name in ('raw_log', 'result'):
                if name in item:
                    source = ROOT/item[name]
                    assert sha(source) == item[name+'_sha256']
                    add(source, 'Preserved known calibration response or failed attempt.')
    assert complete
    acceptance_path = RESULTS/f'{PREFIX}_calibration_acceptance.json'
    acceptance = read(acceptance_path)
    accepted_path = ROOT/acceptance['calibration']
    assert accepted_path in calibrations and sha(accepted_path) == acceptance['calibration_sha256']
    accepted = read(accepted_path)
    assert accepted['complete'] and accepted['source_unchanged'] and accepted['backend_pairs_within_budget']
    assert accepted['source_identity'] == acceptance['source_identity'] == probe.source_identity()
    assert [(w['family'], w['backend']) for w in accepted['workers']] == [
        (family, backend) for family in probe.FAMILIES for backend in ('numpy', 'jax', 'torch')]
    assert acceptance['accepted_ceilings'] == accepted['candidate_airy_ceilings'] == {'r': .005, 't': .010}
    assert acceptance['inherited_candidates_unchanged'] is True
    assert datetime.datetime.fromisoformat(accepted['ended_utc']) < datetime.datetime.fromisoformat(acceptance['accepted_utc'])
    assert all(acceptance['measured_max_absolute_complex_errors'][name] < ceiling
               for name, ceiling in acceptance['accepted_ceilings'].items())
    add(acceptance_path, 'Explicit post-measurement acceptance of unchanged candidate ceilings for both disclosed configurations.')
    for path in sorted((RESULTS/f'{PREFIX}_controls').glob('targeted*.log')):
        add(path, 'Known targeted development exposure; optical tests repeat the two disclosed calibration configurations on three backends.')
    for pattern in ('targeted*.json', 'targeted*.py.txt', '*calibration*.log', '*launcher.log'):
        for path in sorted((RESULTS/f'{PREFIX}_controls').glob(pattern)):
            add(path, 'Preserved development controller provenance, including failed launcher attempts.')
    for path in sorted(RESULTS.glob(f'{PREFIX}_calibration*.log')):
        add(path, 'Disclosed calibration controller stdout, including failures.')
    for pattern in ('calibration_acceptance*.json', 'algebra_attempt*'):
        for path in sorted(RESULTS.glob(f'{PREFIX}_{pattern}')):
            if path.is_file():
                add(path, 'Disclosed calibration acceptance or isolated algebra development record; no new scattering response.')
    for path in sorted(RESULTS.glob(f'{PREFIX}_*launcher.log')):
        add(path, 'Preserved development launcher stdout, hashed after completed attempts.')
    mutations = RESULTS/f'{PREFIX}_mutations.json'
    data = read(mutations)
    assert data['passed'] and data['restored'] and data['all_function_assertion_witnesses']
    assert data['source_identity_before'] == data['source_identity_after'] == probe.source_identity()
    assert data['test_sha256'] == sha(ROOT/'tests/fdtd/test_fd_oscillator_sum_film.py')
    assert data['harness_sha256'] == sha(ROOT/'studies/fdtd/oscillator_sum_film/fdtd_oscillator_sum_film_mutations.py')
    assert data['final_baseline']['exit_code'] == 0
    assert data['final_baseline']['passed'] == data['collection']['collected']
    for name in ('baseline_log',):
        assert sha(ROOT/data[name]) == data[name+'_sha256']
    for entry in (data['collection'], data['final_baseline']):
        assert sha(ROOT/entry['raw_log']) == entry['raw_log_sha256']
    for witness in data['witnesses']:
        assert witness['accepted'] and witness['assertion_failure'] and witness['exit_code'] == 1
        for name, hash_name in (('raw_log', 'raw_log_sha256'),
                                ('original_snapshot', 'original_sha256'),
                                ('mutant_snapshot', 'mutant_sha256')):
            assert sha(ROOT/witness[name]) == witness[hash_name], witness['name']
        assert sha(ROOT/witness['production_file']) == witness['restored_sha256'] == witness['original_sha256']
    add(mutations, 'Ten actual production assertion witnesses across nine functions and restored baseline; known exposures.')
    for filename in [data['final_baseline']['raw_log']] + [
            w[name] for w in data['witnesses'] for name in ('raw_log', 'original_snapshot', 'mutant_snapshot')]:
        add(ROOT/filename, 'Preserved mutation evidence; runtime mutants reuse disclosed configurations.')
    for path in sorted(RESULTS.glob(f'{PREFIX}_mutations_attempt*')):
        for item in sorted(path.rglob('*')) if path.is_dir() else (path,):
            if item.is_file():
                add(item, 'Complete preserved mutation attempt, including rejected attempts and harness snapshots.')
    frozen = []
    for label, path in (('campaign', pathlib.Path(__file__)), ('probe', pathlib.Path(probe.__file__)),
                        ('criteria', pathlib.Path(criteria.__file__)),
                        ('tests', ROOT/'tests/fdtd/test_fd_oscillator_sum_film.py')):
        destination = RESULTS/f'{PREFIX}_registered_{label}.py.txt'
        with destination.open('xb') as stream:
            stream.write(path.read_bytes())
        frozen.append({'file': destination.name, 'sha256': sha(destination)})
    data = {'registered_utc': now(), 'identity': identity(), 'protocol': protocol(),
            'analytic_preflight': analytic, 'known_exposures': list(known.values()),
            'frozen_sources': frozen,
            'qualification': 'Two known calibration configurations on three backends and repeated regression/mutation '
                             'exposures are explicitly disclosed. New 40-row results use fixed '
                             'criteria. Procedural audit does not imply independent agent review.'}
    data['fingerprint'] = hashlib.sha256(canonical(data)).hexdigest()
    write(REGISTRATION, data, exclusive=True)
    return {'registered': str(REGISTRATION.relative_to(ROOT)), 'fingerprint': data['fingerprint']}


def verify():
    data = read(REGISTRATION)
    bare = {k: v for k, v in data.items() if k != 'fingerprint'}
    assert hashlib.sha256(canonical(bare)).hexdigest() == data['fingerprint']
    assert data['identity'] == identity(), 'Registered source/runtime/environment changed.'
    assert data['protocol'] == protocol(), 'Registered protocol changed.'
    for item in data['known_exposures']+data['frozen_sources']:
        assert sha(RESULTS/item['file']) == item['sha256'], item['file']
    return data


def key(family, fraction, pol, n, periods, pml):
    return f'{family}_f{fraction:.1f}_{pol}_n{n}_t{periods}_pml{pml}'


def ledger():
    entries = [json.loads(line) for line in LEDGER.read_text(encoding='utf-8').splitlines()] \
        if LEDGER.exists() else []
    assert len({item['key'] for item in entries}) == len(entries), 'Duplicate attempt.'
    return entries


def rows():
    result = {}
    for entry in ledger():
        if entry['exit_code'] == 0:
            path = ROOT/entry['result']
            assert sha(path) == entry['result_sha256']
            result[entry['key']] = read(path)
    return result


def worker(family, fraction, pol, n, periods=BASE, pml=24):
    registered = verify()
    job = key(family, fraction, pol, n, periods, pml)
    previous = {entry['key']: entry for entry in ledger()}
    if job in previous:
        if previous[job]['exit_code'] != 0:
            raise RuntimeError(f'Prior worker failed: {job}; no silent retry.')
        return rows()[job]
    JOBS.mkdir(exist_ok=True)
    output, log = JOBS/(job+'.json'), JOBS/(job+'.log')
    if output.exists() or log.exists():
        raise RuntimeError(f'Orphan raw attempt exists for {job}; preserve and audit it.')
    command = [sys.executable, '-B', '-m', 'studies.fdtd.oscillator_sum_film.fdtd_oscillator_sum_film_probe',
               '--family', family, '--fraction', str(fraction), '--pol', pol,
               '--n', str(n), '--periods', str(periods), '--pml', str(pml), '--output', str(output)]
    start, begin = now(), time.perf_counter()
    print(f'{start} START {job}', flush=True)
    with log.open('xb') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    entry = {'key': job, 'started_utc': start, 'ended_utc': now(), 'wall_s': time.perf_counter()-begin,
             'exit_code': result.returncode, 'command': command, 'registration': registered['fingerprint'],
             'raw_log': str(log.relative_to(ROOT)), 'raw_log_sha256': sha(log)}
    if output.exists():
        entry.update(result=str(output.relative_to(ROOT)), result_sha256=sha(output))
    with LEDGER.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(entry, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(f'{entry["ended_utc"]} END {job} exit={result.returncode}', flush=True)
    if result.returncode != 0:
        raise RuntimeError(f'Worker {job} failed; preserve the raw attempt.')
    verify()
    return read(output)


def selected(available, family, fraction, pol, n):
    for periods in (BASE, LONG):
        row = available.get(key(family, fraction, pol, n, periods, 24))
        if row is not None and criteria.validity(row)['valid']:
            return row
    return None


def make_report(stop_reason=None):
    registered = verify()
    available, groups, cells = rows(), [], []
    complete = True
    for family in probe.FAMILIES:
        for fraction in FRACTIONS:
            for pol in POLS:
                chosen = [selected(available, family, fraction, pol, n) for n in RUNGS]
                controls = [available.get(key(family, fraction, pol, 80, periods, pml))
                            for periods, pml in ((BASE, 24), (LONG, 24), (LONG, 48))]
                ready = (all(row is not None for row in chosen+controls)
                         and all(criteria.validity(row)['valid'] for row in controls[1:]))
                complete &= ready
                group = {'family': family, 'fraction': fraction, 'polarization': pol,
                         'complete': bool(ready), 'selected_periods': [
                             None if row is None else row['config']['periods'] for row in chosen]}
                groups.append(group)
                for name in ('r', 't'):
                    for index, wavelength in enumerate(probe.WAVELENGTHS):
                        cell = {'family': family, 'fraction': fraction, 'polarization': pol,
                                'coefficient': name, 'wavelength': wavelength, 'resolved_input': bool(ready)}
                        if ready:
                            samples = [criteria.values(row)[name][index] for row in chosen]
                            base, long, pml = (criteria.values(row)[name][index] for row in controls)
                            time_noise, pml_noise = abs(long-base), abs(pml-long)
                            noise = time_noise+pml_noise
                            differences = [abs(samples[0]-samples[1]), abs(samples[1]-samples[2])]
                            errors = [float(row['error_'+name][index]) for row in chosen]
                            cell.update(values=[[v.real, v.imag] for v in samples],
                                continuum=chosen[0]['reference_'+name][index], errors=errors,
                                differences=differences,
                                absorption=[float(row['absorption'][index]) for row in chosen],
                                time_proxy=float(time_noise), pml_proxy=float(pml_noise), proxy=float(noise),
                                self_order=criteria.order(*differences, noise, difference=True),
                                direct_orders=[criteria.order(errors[i], errors[i+1], noise,
                                                              difference=False) for i in (0, 1)])
                        else:
                            cell.update(self_order={'classification': 'unresolved',
                                        'reason': 'Missing or invalid registered input.'}, direct_orders=[])
                        cells.append(cell)
    return probe._json({'complete': bool(complete and stop_reason is None), 'stop_reason': stop_reason,
            'registration': registered['fingerprint'], 'rungs': list(RUNGS), 'groups': groups,
            'cells': cells, 'worker_rows': len(ledger()), 'scope': protocol()['scope'],
            'noise_scope': protocol()['noise_scope']})


def run():
    verify()
    if REPORT.exists() and read(REPORT).get('stop_reason'):
        raise RuntimeError('The campaign already stopped under its registered cap.')
    stop = None
    try:
        for family in probe.FAMILIES:
            for fraction in FRACTIONS:
                for pol in POLS:
                    for n in RUNGS:
                        row = worker(family, fraction, pol, n)
                        state = criteria.validity(row)
                        if not state['valid']:
                            if not state['otherwise_valid']:
                                raise RuntimeError(f'Invalid base {key(family, fraction, pol, n, BASE, 24)}')
                            row = worker(family, fraction, pol, n, LONG)
                            if not criteria.validity(row)['valid']:
                                raise RuntimeError(f'Invalid at registered cap: {key(family, fraction, pol, n, LONG, 24)}')
                        if n == 80:
                            for pml in (24, 48):
                                row = worker(family, fraction, pol, n, LONG, pml)
                                if not criteria.validity(row)['valid']:
                                    raise RuntimeError(f'Invalid fixed control: {key(family, fraction, pol, n, LONG, pml)}')
    except RuntimeError as error:
        stop = str(error)
    report = make_report(stop)
    write(REPORT, report)
    return {'complete': report['complete'], 'stop_reason': stop, 'workers': report['worker_rows'],
            'report': str(REPORT.relative_to(ROOT))}


def audit_sequence(entries, available):
    """Replay policy decisions, including replacement eligibility and stop ordering."""
    index, consumed = 0, {}

    class EndOfAttempts(Exception):
        pass

    def demand(family, fraction, pol, n, periods=BASE, pml=24):
        nonlocal index
        job = key(family, fraction, pol, n, periods, pml)
        if job in consumed:
            return consumed[job]
        if index == len(entries):
            raise EndOfAttempts
        entry = entries[index]
        assert entry['key'] == job, ('Unexpected job, ordering or optional replacement', entry['key'], job)
        index += 1
        if entry['exit_code'] != 0:
            raise EndOfAttempts
        consumed[job] = available[job]
        return consumed[job]

    try:
        for family in probe.FAMILIES:
            for fraction in FRACTIONS:
                for pol in POLS:
                    for n in RUNGS:
                        state = criteria.validity(demand(family, fraction, pol, n))
                        if not state['valid']:
                            if not state['otherwise_valid']:
                                raise EndOfAttempts
                            if not criteria.validity(demand(family, fraction, pol, n, LONG))['valid']:
                                raise EndOfAttempts
                        if n == 80:
                            for pml in (24, 48):
                                if not criteria.validity(demand(family, fraction, pol, n, LONG, pml))['valid']:
                                    raise EndOfAttempts
    except EndOfAttempts:
        pass
    assert index == len(entries), 'Additional worker after registered stop or outside requested job set.'


def audit():
    registered, entries, observed = verify(), ledger(), set()
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
        assert read(raw) == row, 'Raw stdout differs from saved response.'
        assert row['source_identity'] == probe.source_identity() and row['source_unchanged_during_run']
        assert row['source_band'] == protocol()['source_band'] and row['wavelengths'] == list(probe.WAVELENGTHS)
        for name, value in probe.runtime_identity().items():
            assert row[name] == value, name
        config = row['config']
        expected = probe._json(dict(defaults.arguments))
        expected['pole_specs'] = probe._json(probe.POLE_SPECS[config['family']])
        for name in ('family', 'fraction', 'pol', 'n', 'periods', 'pml'):
            expected[name] = config[name]
            flag = '--'+name
            assert entry['command'][entry['command'].index(flag)+1] == str(config[name])
        assert config == {'backend': 'numpy', **expected}
        assert config['family'] in probe.FAMILIES and config['fraction'] in FRACTIONS and config['pol'] in POLS
        assert config['n'] in RUNGS and config['periods'] in (BASE, LONG)
        assert config['pml'] == 24 or (config['n'] == 80 and config['periods'] == LONG and config['pml'] == 48)
        assert key(*(config[name] for name in ('family', 'fraction', 'pol', 'n', 'periods', 'pml'))) == entry['key']
        assert pathlib.Path(entry['command'][entry['command'].index('--output')+1]) == output
        kx = 2*math.pi*math.sin(math.radians(config['theta']))
        assert row['kx'] == kx
        material = {name: config[name] for name in ('family', 'eps', 'pole_specs')}
        reference = probe.airy(probe.WAVELENGTHS, **material, kx=kx, pol=config['pol'],
                               thickness=config['thickness'])
        np.testing.assert_array_equal(np.asarray([complex(*v) for v in row['epsilon']]),
                                      probe.epsilon(probe.WAVELENGTHS, **material))
        for name, value in zip(('r', 't'), reference):
            np.testing.assert_array_equal(np.asarray([complex(*v) for v in row['reference_'+name]]), value)
            np.testing.assert_array_equal(np.asarray(row['error_'+name]), abs(criteria.values(row)[name]-value))
        np.testing.assert_array_equal(np.asarray(row['absorption']), 1-np.asarray(row['R'])-np.asarray(row['T']))
        np.testing.assert_array_equal(np.asarray(row['reference_absorption']), 1-abs(reference[0])**2-abs(reference[1])**2)
        assert row['started_utc'] >= registered['registered_utc'] and row['ended_utc'] >= row['started_utc']
    assert {path.name for path in JOBS.iterdir()} == observed, 'Unledgered raw attempt.'
    audit_sequence(entries, rows())
    saved = read(REPORT)
    assert saved == make_report(saved['stop_reason']), 'Report does not reproduce from raw records.'
    assert len(saved['cells']) == 48
    summary = {'audit': 'passed', 'workers': len(entries), 'complete': saved['complete'],
               'all_48_cells_recomputed': True, 'stop_reason': saved['stop_reason'],
               'self_counts': dict(Counter(c['self_order']['classification'] for c in saved['cells'])),
               'all_direct_counts': dict(Counter(o['classification'] for c in saved['cells'] for o in c['direct_orders'])),
               'direct_final_counts': dict(Counter(c['direct_orders'][-1]['classification']
                    for c in saved['cells'] if c['direct_orders']))}
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
