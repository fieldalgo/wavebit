"""Register, run and audit the bounded 40-row separated Debye-stack characterization.

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
import ast
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
from studies.fdtd.debye_stack import fdtd_debye_stack_probe as probe
from studies.fdtd.debye_stack import fdtd_debye_stack_support as support


ROOT = repository_root()
RESULTS = ROOT/'studies'/'results'
PREFIX = 'fdtd_debye_stack'
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
            'support_sha256': sha(pathlib.Path(support.__file__)),
            'criteria_dependency_sha256': sha(pathlib.Path(criteria.__file__)),
            'test_sha256': sha(ROOT/'tests/fdtd/test_fd_debye_stack.py'),
            'allocation_test_sha256': sha(ROOT/'tests/fdtd/test_fd_state_bytes.py')}


def protocol():
    inherited = criteria.protocol()
    return {**inherited,
        'scope': 'Two fixed asymmetric Debye stacks in vacuum: K2 debye-pair and K3 debye-triple '
                 'including exact zero-strength declarations; theta35 fixed kx, exact Axis1D derivative. '
                 'No general C8 or arbitrary-stack accuracy claim.',
        'materials': {family: {'films': probe._json(probe.FILMS[family]),
                               'gaps': probe._json(probe.GAPS[family])} for family in probe.FAMILIES},
        'base_commit': support.BASE,
        'criteria_provenance': 'The preserved conductive-film campaign supplies validity, order '
            'and order_preflight unchanged: 1e-5 state-decay line, 1e-10 same-value line, '
            'empirical middle-grid duration/PML proxy, and its fixed classification intervals. '
            'These are inherited procedural criteria, not independent numerical error bounds.',
        'rungs': list(RUNGS), 'fractions': list(FRACTIONS), 'polarizations': list(POLS),
        'wavelengths': list(probe.WAVELENGTHS), 'base_worker_count': 40,
        'family_sequence': list(probe.FAMILIES), 'base_periods': BASE, 'long_periods': LONG,
        'sequence_per_placement_polarization': ['40 base120 PML24', '80 base120 PML24',
            '80 long240 PML24', '80 long240 PML48', '160 base120 PML24'],
        'replacement': 'Only an otherwise valid but unsettled120 base may receive one240 replacement. '
            'Reuse the fixed80/240/PML24 control. Invalid cap/control stops globally. '
            'No duration beyond240, extra rung, geometry adaptation or outcome-selected cell.',
        'noise_proxy': '|80,long24-80,base24|+|80,long48-80,long24| per complex cell; '
                       'retain base120 even if selected80 is240.',
        'self_order': 'log2(|v40-v80|/|v80-v160|), constructing sensitivity intervals by +/-2*proxy.',
        'direct_order': 'log2(error_coarse/error_fine), sensitivity intervals by +/-proxy; '
                        'resolved only if min(errors)>1e-10 and2*proxy<=.1*min(errors).',
        'cells': 'Every family x placement x polarization x wavelength x complex r/t (48cells) '
                 'is reported; 48self,96direct,48final-direct. No aggregate substitute.',
        'reference': 'Existing independently validated multilayer characteristic matrix, '
            'declared eps_inf+sum(delta/(1-i*omega*tau)) and explicit vacuum gaps; '
            'independent recursive scattering and complex single-film Airy preflight controls. '
            'Central theta passed once '
            'because the oracle internally preserves kx across wavelengths. Outer-face r/t.',
        'known_exposure': 'Two coarse p/fraction.3 configurations on all three backends disclosed '
            'before registration, including test/mutation reuse. Matching NumPy campaign rows '
            'are known repeats, not independent corroboration.',
        'known_warning': 'Retain all warnings without moving fixed geometry in response.',
        'optical_regression_ceilings': read(RESULTS/f'{PREFIX}_calibration_acceptance.json')['accepted_ceilings'],
        'ceiling_scope': 'Disclosed coarse calibration rows only; not a campaign absolute-error gate.',
    }



def register():
    support.start()
    assert not any(path.exists() for path in (REGISTRATION, LEDGER, REPORT, JOBS))
    analytic = {**probe.preflight(), **criteria.order_preflight()}
    known = {}
    def add(path, scope):
        filename = str(path.relative_to(RESULTS))
        known[filename] = {'file': filename, 'sha256': sha(path), 'scope': scope}
    acceptance_path = RESULTS/f'{PREFIX}_calibration_acceptance.json'
    acceptance = read(acceptance_path)
    accepted_path = ROOT/acceptance['calibration']
    accepted = read(accepted_path)
    assert sha(accepted_path) == acceptance['calibration_sha256']
    assert accepted['base_commit'] == support.BASE
    assert accepted['support_sha256'] == sha(pathlib.Path(support.__file__))
    assert accepted['complete'] and accepted['source_unchanged'] and accepted['production_unchanged']
    assert accepted['finite_stable_all_rows'] and accepted['backend_pairs_within_budget']
    assert accepted['source_identity'] == acceptance['source_identity'] == probe.source_identity()
    assert [(w['family'], w['backend']) for w in accepted['workers']] == [
        (family, backend) for family in probe.FAMILIES for backend in ('numpy', 'jax', 'torch')]
    assert accepted['optical_ceilings'] is None, 'Ceilings must follow disclosure, not be preclaimed.'
    ceilings = acceptance['accepted_ceilings']
    assert set(ceilings) == {'r', 't'} and all(math.isfinite(v) and v > 0 for v in ceilings.values())
    assert datetime.datetime.fromisoformat(accepted['ended_utc']) < datetime.datetime.fromisoformat(acceptance['accepted_utc'])
    measured = {name: max(max(read(ROOT/w['result'])['error_'+name]) for w in accepted['workers'])
                for name in ('r', 't')}
    assert measured == accepted['measured_max_absolute_complex_errors'] == acceptance['measured_max_absolute_complex_errors']
    assert all(measured[name] < ceilings[name] for name in ceilings)
    for worker in accepted['workers']:
        row = read(ROOT/worker['result'])
        assert worker['exit_code'] == 0 and read(ROOT/worker['raw_log']) == row
        assert sha(ROOT/worker['result']) == worker['result_sha256']
        assert sha(ROOT/worker['raw_log']) == worker['raw_log_sha256']
        assert row['source_identity'] == probe.source_identity() and criteria.validity(row)['valid']
    for path in sorted(RESULTS.glob(f'{PREFIX}_calibration_attempt*.json')):
        data = read(path)
        add(path, 'Disclosed calibration, including failed attempts; not unseen response.')
        for item in data['frozen_sources']:
            assert sha(ROOT/item['file']) == item['sha256']
            add(ROOT/item['file'], 'Source snapshot taken before calibration responses.')
        for item in data['workers']:
            for name in ('raw_log', 'result'):
                if name in item:
                    assert sha(ROOT/item[name]) == item[name+'_sha256']
                    add(ROOT/item[name], 'Raw disclosed calibration response or failure.')
        if 'production_archive' in data:
            assert sha(ROOT/data['production_archive']) == data['production_archive_sha256']
            add(ROOT/data['production_archive'], 'Exact pre-response production archive.')
    add(acceptance_path, 'Post-disclosure optical regression acceptance, scoped to six known rows.')
    targeted_path = RESULTS/f'{PREFIX}_targeted.json'
    targeted = read(targeted_path)
    expected_source = {'fd': probe.source_identity()['fd_source_sha256'],
                       'package': probe.source_identity()['wavebit_source_sha256']}
    assert targeted['complete'] and all(c['exit'] == 0 for c in targeted['checks'])
    assert targeted['source_before'] == targeted['source_after']
    assert all(targeted['source_after'][name] == value for name, value in expected_source.items())
    assert {c['name'] for c in targeted['checks']} == {'new_stack_collection', 'new_stack', 'targeted_collection', 'targeted'}
    add(targeted_path, 'Complete restored-source targeted checks; actual counts are retained.')
    mutations = RESULTS/f'{PREFIX}_mutations.json'
    data = read(mutations)
    assert data['passed'] and data['restored'] and data['all_function_assertion_witnesses']
    assert data['source_identity_before'] == data['source_identity_after'] == probe.source_identity()
    assert data['test_sha256'] == sha(ROOT/'tests/fdtd/test_fd_debye_stack.py')
    assert data['harness_sha256'] == sha(ROOT/'studies/fdtd/debye_stack/fdtd_debye_stack_mutations.py')
    assert data['final_baseline']['exit_code'] == 0
    assert data['final_baseline']['passed'] == data['collection']['collected']
    functions = {f'tests/fdtd/test_fd_debye_stack.py::{node.name}' for node in ast.parse(
        (ROOT/'tests/fdtd/test_fd_debye_stack.py').read_text(encoding='utf-8')).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')}
    assert functions <= {w['test'].split('[')[0] for w in data['witnesses']}
    assert len({w['name'] for w in data['witnesses']}) == len(data['witnesses'])
    for witness in data['witnesses']:
        assert witness['accepted'] and witness['assertion_failure'] and witness['exit_code'] == 1
        assert witness['test'] in witness['command']
        for name, key_name in (('raw_log', 'raw_log_sha256'), ('original_snapshot', 'original_sha256'),
                               ('mutant_snapshot', 'mutant_sha256')):
            assert sha(ROOT/witness[name]) == witness[key_name]
        assert sha(ROOT/witness['production_file']) == witness['restored_sha256'] == witness['original_sha256']
    add(mutations, 'Actual production assertion witnesses and exact restored baseline; known exposures.')
    # Include every complete or failed pre-registration development artifact.
    for path in sorted(RESULTS.glob(f'{PREFIX}_*')):
        stem = path.name[len(PREFIX)+1:]
        if stem.startswith(('calibration', 'mutations', 'targeted', 'algebra', 'controls')):
            for item in sorted(path.rglob('*')) if path.is_dir() else (path,):
                if stem.startswith('controls') and not any(
                        part.startswith(('targeted', 'mutations', 'calibration'))
                        for part in item.relative_to(path).parts):
                    continue  # In particular, never fingerprint this registration's open stdout.
                if item.is_file() and str(item.relative_to(RESULTS)) not in known:
                    add(item, 'Complete preserved development provenance, including rejected attempts.')
    # Registration stdout is intentionally excluded: hashing an open stream is not stable.
    frozen = []
    sources = [('campaign', pathlib.Path(__file__)), ('probe', pathlib.Path(probe.__file__)),
               ('criteria', pathlib.Path(criteria.__file__)),
               ('support', pathlib.Path(support.__file__)),
               ('tests', ROOT/'tests/fdtd/test_fd_debye_stack.py'),
               ('allocation_tests', ROOT/'tests/fdtd/test_fd_state_bytes.py')]
    sources += [('reference_'+str(i), ROOT/name) for i, name in enumerate(probe.REFERENCE_DEPENDENCIES)]
    for label, path in sources:
        destination = RESULTS/f'{PREFIX}_registered_{label}.py.raw'
        with destination.open('xb') as stream:
            stream.write(path.read_bytes())
        frozen.append({'file': destination.name, 'sha256': sha(destination)})
    registered_utc = now()
    assert all(datetime.datetime.fromisoformat(value) < datetime.datetime.fromisoformat(registered_utc)
               for value in (acceptance['accepted_utc'], targeted['ended_utc'], data['ended_utc']))
    record = {'registered_utc': registered_utc, 'identity': identity(), 'protocol': protocol(),
              'analytic_preflight': analytic, 'known_exposures': list(known.values()),
              'frozen_sources': frozen,
              'qualification': 'Six disclosed calibration rows and all targeted/mutation attempts are known. '
                  'Fixed40-worker campaign; procedural replay is not independent physics review.'}
    record['fingerprint'] = hashlib.sha256(canonical(record)).hexdigest()
    write(REGISTRATION, record, exclusive=True)
    return {'registered': str(REGISTRATION.relative_to(ROOT)), 'fingerprint': record['fingerprint']}



def verify():
    support.start()
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
    command = [sys.executable, '-B', '-m', 'studies.fdtd.debye_stack.fdtd_debye_stack_probe',
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
    previous_end = registered['registered_utc']
    for entry in entries:
        assert previous_end <= entry['started_utc'] <= entry['ended_utc']
        previous_end = entry['ended_utc']
        assert entry['registration'] == registered['fingerprint']
        assert entry['started_utc'] >= registered['registered_utc']
        raw = ROOT/entry['raw_log']
        assert sha(raw) == entry['raw_log_sha256']
        observed.add(raw.name)
        output = ROOT/entry['result'] if 'result' in entry else None
        if output is not None:
            assert sha(output) == entry['result_sha256']
            observed.add(output.name)
        if entry['exit_code'] != 0:
            continue
        assert output is not None
        row = read(output)
        assert read(raw) == row, 'Raw stdout differs from saved response.'
        assert row['source_identity'] == probe.source_identity() and row['source_unchanged_during_run']
        assert row['source_band'] == protocol()['source_band'] and row['wavelengths'] == list(probe.WAVELENGTHS)
        for name, value in probe.runtime_identity().items():
            assert row[name] == value, name
        config = row['config']
        expected = probe._json(dict(defaults.arguments))
        expected['films'] = probe._json(probe.FILMS[config['family']])
        expected['gaps'] = probe._json(probe.GAPS[config['family']])
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
        material = {name: config[name] for name in ('family', 'films', 'gaps')}
        reference = probe.reference(probe.WAVELENGTHS, **material, theta=config['theta'], pol=config['pol'])
        np.testing.assert_array_equal(np.asarray([[complex(*v) for v in member] for member in row['epsilon']]),
                                      probe.epsilon(probe.WAVELENGTHS, **material))
        faces, depth = probe.geometry(**material)
        assert row['film_faces'] == probe._json(faces) and row['total_depth'] == depth
        assert row['outer_faces'] == [0., -depth]
        dx = 1./config['n']
        low = -(math.ceil((depth+.8)/dx)+config['pml']+config['fraction'])*dx
        nz = math.ceil((1.4+config['pml']*dx-low)/dx)
        assert row['grid'] == [3, nz, 1, 1]
        np.testing.assert_allclose(row['grid_low'], [-dx/2, -dx/2, low], rtol=0., atol=2e-13*dx)
        phases = [((face-row['grid_low'][2])/dx) % 1 for pair in faces for face in pair]
        np.testing.assert_allclose(phases, config['fraction'], rtol=0., atol=2e-13)
        assert row['dt'] == config['courant']*dx/math.sqrt(3.)
        assert row['steps'] == math.ceil(config['periods']/row['dt'])
        theta_at = np.degrees(np.arcsin(kx/(2*math.pi/np.asarray(probe.WAVELENGTHS))))
        np.testing.assert_array_equal(row['reference_theta_at'], theta_at)
        np.testing.assert_allclose(row['theta_at'], theta_at, atol=2e-12, rtol=0.)
        for name, value in zip(('r', 't'), reference):
            np.testing.assert_array_equal(np.asarray([complex(*v) for v in row['reference_'+name]]), value)
            np.testing.assert_array_equal(np.asarray(row['error_'+name]), abs(criteria.values(row)[name]-value))
        np.testing.assert_array_equal(np.asarray(row['absorption']), 1-np.asarray(row['R'])-np.asarray(row['T']))
        np.testing.assert_array_equal(np.asarray(row['reference_absorption']), 1-abs(reference[0])**2-abs(reference[1])**2)
        assert entry['started_utc'] <= row['started_utc'] <= row['ended_utc'] <= entry['ended_utc']
    assert {path.name for path in JOBS.iterdir()} == observed, 'Unledgered raw attempt.'
    audit_sequence(entries, rows())
    saved = read(REPORT)
    assert saved == make_report(saved['stop_reason']), 'Report does not reproduce from raw records.'
    assert len(saved['cells']) == 48
    acceptance = read(RESULTS/f'{PREFIX}_calibration_acceptance.json')
    calibration = read(ROOT/acceptance['calibration'])
    comparison_keys = ('config', 'grid', 'source_band', 'dt', 'steps', 'kx', 'wavelengths',
        'theta_at', 'reference_theta_at', 'film_faces', 'total_depth', 'outer_faces', 'grid_low',
        'epsilon', 'r', 't', 'reference_r', 'reference_t', 'error_r', 'error_t', 'R', 'T',
        'absorption', 'reference_absorption', 'stable', 'decay', 'energy_shape', 'energy_rows',
        'energy_finite', 'energy_max', 'bytes_per_state', 'warnings')
    repeats = {}
    available = rows()
    for family in probe.FAMILIES:
        job = key(family, .3, 'p', 40, BASE, 24)
        original = next(w for w in calibration['workers'] if w['family'] == family and w['backend'] == 'numpy')
        known = read(ROOT/original['result'])
        repeats[family] = {'calibration': original['result'], 'campaign_key': job,
                          'available': job in available,
                          'payload_identical': None if job not in available else all(
                              known[name] == available[job][name] for name in comparison_keys),
                          'qualification': 'Repeated known calibration, not independent corroboration.'}
    summary = {'audit': 'passed', 'workers': len(entries), 'complete': saved['complete'],
               'all_48_cells_recomputed': True, 'known_calibration_repeats': repeats,
               'stop_reason': saved['stop_reason'],
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
