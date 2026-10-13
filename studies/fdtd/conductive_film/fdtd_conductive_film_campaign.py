"""Registered, bounded characterization of the first C8 conductive-film scope.

Commands: register (before new responses), run (one worker at a time), report, audit.
Every raw worker attempt is retained. This campaign characterizes two placements of
one film, not arbitrary conductors, curved interfaces, or declared dispersive media.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_root
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

from studies.fdtd.conductive_film import fdtd_conductive_film_probe as probe


ROOT = repository_root()
RESULTS = ROOT / 'studies' / 'results'
PREFIX = 'fdtd_conductive_film'
REGISTRATION = RESULTS / f'{PREFIX}_registration.json'
LEDGER = RESULTS / f'{PREFIX}_campaign.jsonl'
REPORT = RESULTS / f'{PREFIX}_report.json'
JOBS = RESULTS / f'{PREFIX}_campaign_jobs'
FRACTIONS, POLS, RUNGS = (.3, .1), ('s', 'p'), (40, 80, 160)
BASE, LONG = 60, 120
SAME, DECAY_LINE = 1e-10, 1e-5
ENV_NAMES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONUTF8')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def write_json(path, value, *, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def identity():
    versions = {}
    for name in ('numpy', 'jax', 'jaxlib', 'torch'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {**probe.source_identity(), 'campaign_sha256': sha(pathlib.Path(__file__)),
            'python': sys.version, 'executable': sys.executable,
            'backend_versions': versions,
            'environment': {name: os.environ.get(name) for name in ENV_NAMES}}


def protocol():
    return {
        'scope': 'One concrete full-width unturned eps=4 sigma=8 film, thickness .2 in vacuum; '
                 'theta35, s/p, exact Axis1D transverse derivative; no general C8 claim.',
        'rungs': list(RUNGS), 'fractions': list(FRACTIONS), 'polarizations': list(POLS),
        'wavelengths': list(probe.WAVELENGTHS), 'backend': 'numpy',
        'source_band': {'wavelength': 1., 'df': .1, 'cutoff': 1e-6},
        'base_periods': BASE, 'long_periods': LONG, 'pml_cells': [24, 48],
        'base_worker_count': 20,
        'sequence_per_placement_polarization': ['40 base', '80 base', '80 long',
                                               '80 long PML48', '160 base'],
        'replacement': 'Only an otherwise valid but unsettled 60-period base row may receive '
                       'one 120-period replacement. Stop if that replacement or a fixed '
                       '120-period control is invalid. No duration beyond 120 or extra rung.',
        'decay_limit': DECAY_LINE,
        'noise_proxy': '|80,long24 - 80,base24| + |80,long48 - 80,long24| per complex cell; '
                       'retain the base60 value in this proxy even if selected80 is long120.',
        'noise_scope': 'The middle-grid time/PML proxy is empirical, not a rigorous fine-grid bound.',
        'self_order': 'log2(|v40-v80| / |v80-v160|), with each difference interval +/-2*proxy.',
        'self_resolved': 'Both changes >1e-10 and 2*proxy <= .1*min(adjacent changes).',
        'direct_order': 'log2(error_coarse/error_fine), each error interval +/-proxy; '
                        'resolved only if min(errors)>1e-10 and 2*proxy<=.1*min(errors).',
        'classification': {'second': 'entire interval inside [1.8,2.3]',
                           'low': 'interval upper <=1.75', 'mixed': 'other resolved intervals',
                           'unresolved': 'noise gate, vanishing change/error, or missing/invalid row'},
        'cells': 'Every placement x polarization x wavelength x complex r/t coefficient '
                 '(24 total) is reported; no worst-cell aggregate replaces them.',
        'reference': 'Independent continuum complex Airy expression; no truncation ladder.',
        'absorptance': 'Preserve measured 1-R-T and continuum value at each wavelength; '
                       'do not treat loss as an energy-closure error.',
    }


def register():
    if REGISTRATION.exists() or LEDGER.exists() or JOBS.exists():
        raise RuntimeError('Registration requires no prior campaign registration or worker data.')
    analytic = {**probe.preflight(), **order_preflight()}
    known = []
    for filename, scope in (
        (f'{PREFIX}_probe_dev_n40p60.json',
         'Known narrow calibration: n40 p60 PML24; df=.035; wavelengths .95/1/1.05. '
         'Original helper/production hashes were not captured. Annotated helper reconstruction '
         'is not an original snapshot.'),
        (f'{PREFIX}_probe_calibration_n40p60.json',
         'Known broad calibration of n40 p60 PML24, fraction.3, df=.1; '
         'wavelengths .9/1/1.1. Helper captured unchanged afterward; production hash not recorded.')):
        path = RESULTS / filename
        known.append({'file': filename, 'sha256': sha(path), 'scope': scope})
    for filename in (f'{PREFIX}_probe_dev_narrow.py.txt',
                     f'{PREFIX}_probe_calibration_broad.py.txt',
                     f'{PREFIX}_calibration_provenance_note.json'):
        known.append({'file': filename, 'sha256': sha(RESULTS / filename),
                      'scope': 'Calibration provenance supporting file; see its own qualification.'})
    for tests in sorted((RESULTS / f'{PREFIX}_controls').glob('targeted_attempt*.log')):
        known.append({'file': str(tests.relative_to(RESULTS)), 'sha256': sha(tests),
                      'scope': 'Known development/validation exposure. Attempts1/2 preserve '
                               'refused source setups, not field responses; attempts3/4 include '
                               'broad n40,p60,fraction.3 on NumPy/JAX/Torch; subsequent attempts '
                               'reuse that case with additional door/metadata checks. These logs '
                               'report assertions, not saved response datasets.'})
    mutation_path = RESULTS / f'{PREFIX}_mutations.json'
    mutations = json.loads(mutation_path.read_text(encoding='utf-8'))
    assert mutations['passed'] and mutations['restored'] and mutations['all_nine_assertion_witnesses']
    known.append({'file': mutation_path.name, 'sha256': sha(mutation_path),
                  'scope': 'Known mutation and restored-baseline exposures. The runtime witness '
                           'uses the same n40,p60,fraction.3 calibration case with intentionally '
                           'wrong production; byte restoration is verified. Other witnesses '
                           'exercise material or door assertions, not new scattering scenes.'})
    evidence = [mutations['final_baseline']['raw_log']]
    for witness in mutations['witnesses']:
        evidence += [witness['raw_log'], witness['original_snapshot'], witness['mutant_snapshot']]
    for filename in evidence:
        path = ROOT / filename
        known.append({'file': str(path.relative_to(RESULTS)), 'sha256': sha(path),
                      'scope': 'Preserved mutation or restored-baseline evidence; no campaign response.'})
    frozen = []
    for label, path in (('campaign', pathlib.Path(__file__)),
                        ('probe', pathlib.Path(probe.__file__))):
        destination = RESULTS / f'{PREFIX}_registered_{label}.py.txt'
        with destination.open('xb') as stream:
            stream.write(path.read_bytes())
        frozen.append({'file': destination.name, 'sha256': sha(destination)})
    data = {'registered_utc': now(), 'identity': identity(), 'protocol': protocol(),
            'analytic_preflight': analytic, 'known_exposures': known, 'frozen_sources': frozen,
            'qualification': 'Independent analytic oracle and procedural audit; no independent '
                             'agent review is implied by this registration.'}
    data['fingerprint'] = hashlib.sha256(canonical(data)).hexdigest()
    write_json(REGISTRATION, data, exclusive=True)
    return {'registered': str(REGISTRATION.relative_to(ROOT)), 'fingerprint': data['fingerprint']}


def verify():
    data = json.loads(REGISTRATION.read_text(encoding='utf-8'))
    bare = {key: value for key, value in data.items() if key != 'fingerprint'}
    assert hashlib.sha256(canonical(bare)).hexdigest() == data['fingerprint']
    assert data['identity'] == identity(), 'registered source/runtime/environment changed'
    assert data['protocol'] == protocol(), 'registered protocol changed'
    for item in data['known_exposures'] + data['frozen_sources']:
        assert sha(RESULTS / item['file']) == item['sha256'], item['file']
    return data


def key(fraction, pol, n, periods, pml):
    return f'f{fraction:.1f}_{pol}_n{n}_t{periods}_pml{pml}'


def ledger():
    entries = []
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding='utf-8').splitlines():
            entries.append(json.loads(line))
    assert len({entry['key'] for entry in entries}) == len(entries), 'duplicate worker attempt'
    return entries


def values(row):
    return {name: np.array([complex(*pair) for pair in row[name]]) for name in ('r', 't')}


def validity(row):
    numeric = list(values(row).values()) + [np.asarray(row[name]) for name in
                                          ('error_r', 'error_t', 'R', 'T', 'absorption')]
    finite = all(np.all(np.isfinite(value)) for value in numeric)
    finite &= row['energy_finite'] and math.isfinite(row['energy_max']) and row['energy_max'] > 0
    hard = [warning['category'] for warning in row['warnings']
            if warning['category'] in ('GrowingFieldWarning', 'UnstableFieldWarning')]
    stable = row['stable'] is True and finite and not hard
    decay = row['decay']
    settled = (bool(decay) and all(math.isfinite(v) and v <= DECAY_LINE for v in decay.values()))
    return {'valid': stable and settled, 'otherwise_valid': stable,
            'settled': settled, 'hard_warnings': hard}


def rows():
    result = {}
    for entry in ledger():
        if entry['exit_code'] == 0:
            path = ROOT / entry['result']
            assert sha(path) == entry['result_sha256'], entry['key']
            result[entry['key']] = json.loads(path.read_text(encoding='utf-8'))
    return result


def worker(fraction, pol, n, periods=BASE, pml=24):
    registered = verify()
    job = key(fraction, pol, n, periods, pml)
    previous = {entry['key']: entry for entry in ledger()}
    if job in previous:
        if previous[job]['exit_code'] != 0:
            raise RuntimeError(f'Prior worker failed: {job}; no silent retry.')
        return rows()[job]
    JOBS.mkdir(exist_ok=True)
    output, log = JOBS / (job + '.json'), JOBS / (job + '.log')
    if output.exists() or log.exists():
        raise RuntimeError(f'Orphan attempt exists for {job}; preserve it and audit before retry.')
    command = [sys.executable, '-m', 'studies.fdtd.conductive_film.fdtd_conductive_film_probe',
               '--n', str(n), '--pol', pol, '--fraction', str(fraction),
               '--periods', str(periods), '--pml', str(pml), '--output', str(output)]
    start = now()
    print(f'{start} START {job}', flush=True)
    begin = time.perf_counter()
    with log.open('xb') as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   check=False)
    entry = {'key': job, 'started_utc': start, 'ended_utc': now(),
             'wall_s': time.perf_counter()-begin, 'exit_code': completed.returncode,
             'command': command, 'registration': registered['fingerprint'],
             'raw_log': str(log.relative_to(ROOT)), 'raw_log_sha256': sha(log)}
    if output.exists():
        entry.update(result=str(output.relative_to(ROOT)), result_sha256=sha(output))
    with LEDGER.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(entry, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(f'{entry["ended_utc"]} END {job} exit={completed.returncode}', flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f'Worker {job} exited {completed.returncode}; raw attempt preserved.')
    verify()
    return json.loads(output.read_text(encoding='utf-8'))


def selected(available, fraction, pol, n):
    base = available.get(key(fraction, pol, n, BASE, 24))
    if base is not None and validity(base)['valid']:
        return base
    longer = available.get(key(fraction, pol, n, LONG, 24))
    if longer is not None and validity(longer)['valid']:
        return longer
    return None


def classification(interval):
    if interval is None:
        return 'unresolved'
    low, high = interval
    if 1.8 <= low and high <= 2.3:
        return 'second'
    return 'low' if high <= 1.75 else 'mixed'


def order(first, second, noise, *, difference):
    margin = 2 * noise if difference else noise
    resolved = (min(first, second) > SAME and 2 * noise <= .1 * min(first, second))
    point = math.log2(first/second) if min(first, second) > 0 else None
    interval = ([math.log2((first-margin)/(second+margin)),
                 math.log2((first+margin)/(second-margin))] if resolved else None)
    return {'point': point, 'interval': interval, 'classification': classification(interval),
            'resolved': resolved}


def order_preflight():
    """Signed power laws, the vanishing-change case, and a noise-dominated instrument."""
    for exponent, label in ((-1., 'low'), (1., 'low'), (1.444, 'low'), (2., 'second')):
        sequence = [n ** (-exponent) for n in RUNGS]
        changes = [abs(sequence[0]-sequence[1]), abs(sequence[1]-sequence[2])]
        got = order(*changes, 0., difference=True)
        assert abs(got['point'] - exponent) < 2e-14
        assert got['classification'] == label
    assert order(0., 0., 0., difference=True)['classification'] == 'unresolved'
    assert order(.04, .01, .001, difference=True)['classification'] == 'unresolved'
    assert order(.04, .01, .00001, difference=False)['classification'] == 'second'
    return {'signed_order_and_noise_controls': 'passed'}


def make_report(stop_reason=None):
    verify()
    available, cells, groups = rows(), [], []
    complete = True
    for fraction in FRACTIONS:
        for pol in POLS:
            chosen = [selected(available, fraction, pol, n) for n in RUNGS]
            controls = [available.get(key(fraction, pol, 80, periods, pml))
                        for periods, pml in ((BASE, 24), (LONG, 24), (LONG, 48))]
            ready = (all(row is not None for row in chosen + controls)
                     and all(validity(row)['valid'] for row in controls[1:]))
            complete &= ready
            group = {'fraction': fraction, 'polarization': pol, 'complete': bool(ready),
                     'selected_periods': [None if row is None else row['config']['periods']
                                          for row in chosen]}
            groups.append(group)
            for name in ('r', 't'):
                for index, wavelength in enumerate(probe.WAVELENGTHS):
                    cell = {'fraction': fraction, 'polarization': pol, 'coefficient': name,
                            'wavelength': wavelength, 'resolved_input': bool(ready)}
                    if ready:
                        samples = [values(row)[name][index] for row in chosen]
                        base, long, pml = (values(row)[name][index] for row in controls)
                        time_noise, pml_noise = abs(long-base), abs(pml-long)
                        noise = time_noise + pml_noise
                        changes = [abs(samples[0]-samples[1]), abs(samples[1]-samples[2])]
                        errors = [float(row['error_' + name][index]) for row in chosen]
                        cell.update(values=[[v.real, v.imag] for v in samples],
                                    continuum=chosen[0]['reference_' + name][index],
                                    errors=errors, differences=changes,
                                    absorption=[float(row['absorption'][index]) for row in chosen],
                                    time_proxy=float(time_noise), pml_proxy=float(pml_noise),
                                    proxy=float(noise),
                                    self_order=order(*changes, noise, difference=True),
                                    direct_orders=[order(errors[i], errors[i+1], noise,
                                                         difference=False) for i in (0, 1)])
                    else:
                        cell.update(self_order={'classification': 'unresolved',
                                                'reason': 'missing or invalid registered input'},
                                    direct_orders=[])
                    cells.append(cell)
    return {'complete': bool(complete and stop_reason is None), 'stop_reason': stop_reason,
            'registration': verify()['fingerprint'], 'rungs': list(RUNGS),
            'groups': groups, 'cells': cells, 'worker_rows': len(ledger()),
            'scope': protocol()['scope'], 'noise_scope': protocol()['noise_scope']}


def run():
    verify()
    if REPORT.exists():
        previous = json.loads(REPORT.read_text(encoding='utf-8'))
        if previous.get('stop_reason'):
            raise RuntimeError('Campaign already stopped under its registered cap; do not resume silently.')
    stop = None
    try:
        for fraction in FRACTIONS:
            for pol in POLS:
                for n in RUNGS:
                    row = worker(fraction, pol, n)
                    state = validity(row)
                    if not state['valid']:
                        if not state['otherwise_valid']:
                            raise RuntimeError(f'Invalid base response {key(fraction, pol, n, BASE, 24)}')
                        row = worker(fraction, pol, n, LONG)
                        if not validity(row)['valid']:
                            raise RuntimeError(f'Unsettled/invalid at registered cap '
                                               f'{key(fraction, pol, n, LONG, 24)}')
                    if n == 80:
                        for pml in (24, 48):
                            row = worker(fraction, pol, 80, LONG, pml)
                            if not validity(row)['valid']:
                                raise RuntimeError(f'Invalid fixed control '
                                                   f'{key(fraction, pol, 80, LONG, pml)}')
    except RuntimeError as error:
        stop = str(error)
    report = make_report(stop)
    write_json(REPORT, report)
    return {'complete': report['complete'], 'stop_reason': stop,
            'workers': report['worker_rows'], 'report': str(REPORT.relative_to(ROOT))}


def audit():
    reg = verify()
    entries = ledger()
    observed = set()
    for entry in entries:
        assert entry['registration'] == reg['fingerprint']
        assert entry['started_utc'] >= reg['registered_utc']
        raw = ROOT / entry['raw_log']
        assert sha(raw) == entry['raw_log_sha256']
        observed.add(raw.name)
        if entry['exit_code'] == 0:
            result = ROOT / entry['result']
            assert sha(result) == entry['result_sha256']
            observed.add(result.name)
            row = json.loads(result.read_text(encoding='utf-8'))
            assert json.loads(raw.read_text(encoding='utf-8')) == row, 'raw stdout/result mismatch'
            assert row['source_identity'] == probe.source_identity()
            assert row['source_unchanged_during_run'] is True
            assert row['source_band'] == reg['protocol']['source_band']
            assert row['wavelengths'] == list(probe.WAVELENGTHS)
            assert row['config']['backend'] == 'numpy'
            config = row['config']
            assert key(config['fraction'], config['pol'], config['n'], config['periods'],
                       config['pml']) == entry['key']
            assert config['fraction'] in FRACTIONS and config['pol'] in POLS
            assert config['n'] in RUNGS and config['periods'] in (BASE, LONG)
            assert config['pml'] == 24 or (config['n'] == 80 and config['periods'] == LONG
                                            and config['pml'] == 48)
            assert {name: config[name] for name in ('theta', 'eps', 'sigma', 'thickness', 'courant')} \
                == dict(theta=35., eps=4., sigma=8., thickness=.2, courant=.5)
            command = entry['command']
            for flag, name in (('--n', 'n'), ('--pol', 'pol'), ('--fraction', 'fraction'),
                               ('--periods', 'periods'), ('--pml', 'pml')):
                assert command[command.index(flag) + 1] == str(config[name])
            assert pathlib.Path(command[command.index('--output') + 1]) == result
            kx = 2 * math.pi * math.sin(math.radians(config['theta']))
            assert row['kx'] == kx
            ref_r, ref_t = probe.airy(probe.WAVELENGTHS, kx=kx, pol=config['pol'],
                                      eps=config['eps'], sigma=config['sigma'],
                                      thickness=config['thickness'])
            for name, reference in (('r', ref_r), ('t', ref_t)):
                saved_reference = np.array([complex(*pair) for pair in row['reference_' + name]])
                np.testing.assert_array_equal(saved_reference, reference)
                np.testing.assert_array_equal(np.asarray(row['error_' + name]),
                                              abs(values(row)[name] - reference))
            np.testing.assert_array_equal(np.asarray(row['absorption']),
                                          1 - np.asarray(row['R']) - np.asarray(row['T']))
            np.testing.assert_array_equal(np.asarray(row['reference_absorption']),
                                          1 - abs(ref_r) ** 2 - abs(ref_t) ** 2)
            assert row['started_utc'] >= reg['registered_utc']
            assert row['ended_utc'] >= row['started_utc']
    assert {path.name for path in JOBS.iterdir()} == observed, 'unledgered raw attempts'
    saved = json.loads(REPORT.read_text(encoding='utf-8'))
    rebuilt = make_report(saved['stop_reason'])
    assert saved == rebuilt, 'report does not reproduce exactly from immutable raw responses'
    assert len(saved['cells']) == 24
    return {'audit': 'passed', 'workers': len(entries), 'complete': saved['complete'],
            'all_24_cell_records_recomputed': True, 'stop_reason': saved['stop_reason']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'report', 'audit'))
    args = parser.parse_args()
    if args.action == 'register':
        result = register()
    elif args.action == 'run':
        result = run()
    elif args.action == 'audit':
        result = audit()
    else:
        prior = json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {}
        result = make_report(prior.get('stop_reason'))
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.action == 'run' and not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
