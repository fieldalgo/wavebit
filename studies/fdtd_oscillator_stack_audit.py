"""Audit saved Oscillator-stack evidence without running a solver or pytest.

Raw logs, collected/reported node inventories, immutable source snapshots and
actual recorded child exits are replayed. Only a new audit attempt and, after
success, its canonical result are written. The caller must separately record
this auditor's observed process exit; the auditor cannot certify its own exit.
"""
import argparse
import ast
from datetime import datetime
import json
import re

from studies.fdtd_oscillator_stack_support import (
    ROOT, RESULTS, PREFIX, BASE, NEW_TEST, TARGETS, write, require, relative,
    inside, sha, digest, now, maps, identity, verify_guard, verify_snapshot,
    docs_map, audit_pytest, nodes, SCRATCH, reports, summary, check_log, campaign_output_matches,
)
import cmath
from collections import Counter
import io
import math
from pathlib import Path
import zipfile

INPUTS = {}
MAP_KEYS = ('wavebit', 'tests', 'goldens', 'previous_film_evidence')
PREVIOUS = 'studies/results/fdtd_debye_stack_'
EXPECTED_TEST_FUNCTIONS = ('test_each_member_advance_matches_the_declared_series_parallel_response', 'test_composition_preserves_singletons_declared_slots_and_actual_allocation', 'test_all_members_storage_and_closed_yee_dissipation_use_independent_oscillators', 'test_oblique_oscillator_stack_matches_declared_multilayer_on_each_backend', 'test_every_member_must_pass_scope_and_exact_padded_support_separation', 'test_prepared_stack_binds_all_members_memories_and_source_allocation', 'test_traced_later_poles_geometry_and_host_cannot_detach_into_stack_admission', 'test_cfl_uses_each_complete_oscillator_rate_then_the_spatial_maximum', 'test_spectral_certificate_includes_later_member_interior_resonances_and_samples')


def read(path):
    path = inside(path)
    raw = path.read_bytes()
    name = relative(path)
    require(name not in INPUTS or INPUTS[name] == digest(raw),
            'Input changed during audit: '+name)
    INPUTS[name] = digest(raw)
    return raw


def read_json(path):
    return json.loads(read(path))


def audit_snapshot(item, *, current=True):
    read(item['file'])
    if current:
        read(item['source'])
    return verify_snapshot(item, current=current)


def checks_by_name(value):
    answer = {item['name']: item for item in value['checks']}
    require(len(answer) == len(value['checks']), 'Duplicate child command names.')
    return answer


def attempt(kind):
    canonical = RESULTS/(PREFIX+kind+'.json')
    value = read_json(canonical)
    require(type(value['attempt']) is int and value['attempt'] > 0, 'Invalid attempt number.')
    retained = RESULTS/f'{PREFIX}{kind}_attempt{value["attempt"]}.json'
    require(read(canonical) == read(retained), 'Canonical/retained attempt differs: '+kind)
    require(value['base'] == BASE and value['kind'] == kind and value['complete'] is True,
            'Incomplete or wrong-base '+kind)
    require(value['source_before'] == value['source_after'] == identity(),
            'Numerical source changed for '+kind)
    verify_guard(value['guard_before'])
    for family in ('helpers', 'references', 'runner_inputs'):
        for name, expected in value['guard_before'][family].items():
            require(digest(read(name)) == expected, 'Frozen dependency changed: '+name)
    for item in value.get('snapshots', [])+value.get('runner_snapshots', []):
        audit_snapshot(item)
    checks_by_name(value)
    previous = datetime.fromisoformat(value['started_utc'])
    ended = datetime.fromisoformat(value['ended_utc'])
    for check in value['checks']:
        require('launcher_error' not in check and type(check['exit']) is int,
                'Child command did not report an actual exit.')
        require(digest(read(check['log'])) == check['sha256'], 'Raw child log hash differs.')
        start, stop = (datetime.fromisoformat(check[k]) for k in ('started_utc', 'ended_utc'))
        require(previous <= start <= stop <= ended, 'Nonsequential child chronology.')
        previous = stop
        require(not check['environment'].get('XLA_FLAGS'), 'Unexpected XLA_FLAGS.')
        require(check['environment']['PYTHONUTF8'] == '1', 'Raw log encoding was not fixed.')
        require(inside(check['environment']['PYTHONPATH']) == ROOT, 'Child used another project.')
    return value


def command_pair(record, name, targets, *, extra=(), workers=0):
    checks = checks_by_name(record)
    collection, run = checks[name+'_collection'], checks[name]
    common = [record['executable'], '-B', '-m', 'pytest', '-p',
              'no:cacheprovider', *targets, *extra]
    require(collection['command'] == common+['-q', '--collect-only'],
            'Collection selection differs: '+name)
    tail = run['command'][len(common):]
    require(run['command'][:len(common)] == common and len(tail) == (6 if workers else 4)
            and tail[:3] == ['-vv', '--tb=short', '--basetemp'],
            'Run selection/reporting differs: '+name)
    scratch_path(tail[3])
    if workers:
        require(tail[4:] == ['-n', str(workers)], 'Unexpected worker count.')
    require(collection['exit'] == run['exit'] == 0, 'Nonzero gate child exit: '+name)
    observed = audit_pytest(read(collection['log']).decode('utf-8'),
                            read(run['log']).decode('utf-8'))
    require(observed == run['observed'], 'Saved pytest result differs from raw replay.')
    return observed


def scratch_path(value):
    """Validate the separate, fixed sibling scratch root without reading its files."""
    path = Path(value).resolve()
    require(path.is_relative_to(SCRATCH.resolve()) and path != SCRATCH.resolve(),
            'Temporary/output path escapes the exact sibling scratch directory.')
    for parent in (path, *path.parents):
        if parent.exists():
            require(not parent.is_symlink() and not (
                getattr(parent.stat(), 'st_file_attributes', 0) & 0x400),
                'Scratch path traverses a reparse point.')
        if parent == SCRATCH.resolve():
            break
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def near(actual, expected, *, atol=2e-13, rtol=2e-13):
    """Roundoff replay tolerance only; never changes acceptance classifications."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Dictionary schema differs.')
        for key in expected:
            near(actual[key], expected[key], atol=atol, rtol=rtol)
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, (tuple, list)) and len(actual) == len(expected), 'Array shape differs.')
        for a, e in zip(actual, expected):
            near(a, e, atol=atol, rtol=rtol)
    elif isinstance(expected, (float, complex)):
        require(math.isfinite(abs(actual)) and math.isfinite(abs(expected))
                and abs(actual-expected) <= atol+rtol*abs(expected), 'Numeric raw replay differs.')
    else:
        require(actual == expected, 'Literal/schema value differs.')


def probe_identity():
    current = identity()
    refs = ('studies/fdtd_layer_alignment.py', 'studies/fdtd_oblique_pulse.py')
    return {'helper_sha256': digest(read('studies/fdtd_oscillator_stack_probe.py')),
            'fd_source_sha256': current['fd'], 'wavebit_source_sha256': current['package'],
            'reference_dependencies': {name: digest(read(name)) for name in refs},
            'fd_digest_rule': 'sorted relative POSIX path + NUL + LF-normalized bytes + NUL'}


def audit_baseline():
    baseline_path = RESULTS/(PREFIX+'baseline.json')
    baseline = read_json(baseline_path)
    setup = attempt('prevalidation')
    application = read_json(RESULTS/(PREFIX+'application.json'))
    require(baseline['base'] == setup['base'] == application['base'] == BASE
            and baseline['raw_byte_sha256'] is True, 'Wrong baseline/application identity.')
    require(setup['baseline_sha256'] == digest(read(baseline_path)), 'Baseline changed.')
    before = {name: baseline['prior_evidence' if name == 'previous_film_evidence' else name]
              for name in MAP_KEYS}
    require(all(before.values()), 'Empty baseline inventory.')
    after = setup['maps']
    require(set(after) == set(before) and after == maps(), 'Final raw inventory differs.')
    verify_guard(setup['guard'])
    for name in ('goldens', 'previous_film_evidence'):
        require(before[name] == after[name], 'Historical raw artifacts changed: '+name)
    rows = application['production']
    require(len({r['target'] for r in rows}) == len(rows), 'Duplicate applied production path.')
    require(all(r['kind'] in ('modify', 'add') for r in rows), 'Unknown application kind.')
    modified = {r['target'] for r in rows if r['kind'] == 'modify'}
    added = {r['target'] for r in rows if r['kind'] == 'add'}
    require(added == {'wavebit/fd/_lower/oscillator_stack.py'}, 'Unexpected added production.')
    require(set(after['wavebit']) == set(before['wavebit'])|added, 'Production file set differs.')
    require({n for n, h in before['wavebit'].items() if after['wavebit'][n] != h} == modified,
            'Production change exceeds recorded application.')
    for row in rows:
        require(after['wavebit'][row['target']] == row['draft_sha256'], 'Applied production changed.')
        if row['kind'] == 'modify':
            require(before['wavebit'][row['target']] == row['original_sha256'], 'Wrong application base.')
    require(set(after['tests']) == set(before['tests'])|{NEW_TEST}, 'Unexpected test inventory.')
    require(all(after['tests'][n] == h for n, h in before['tests'].items()), 'Existing FD tests changed.')
    for values in after.values():
        for name, expected in values.items():
            require(digest(read(name)) == expected, 'Final raw map differs: '+name)
    needed = modified|added|{NEW_TEST}|set(setup['guard']['helpers'])|set(setup['guard']['references'])|set(setup['guard']['runner_inputs'])
    require(needed <= {r['source'] for r in setup['snapshots']}, 'Missing frozen source snapshot.')
    for item in setup['snapshots']:
        require(Path(item['file']).suffix == '.gz', 'Unsafe copied-source suffix.')
        audit_snapshot(item)
    for name, reference in setup['prerequisites'].items():
        require(reference['path'] == relative(RESULTS/(PREFIX+name+'.json'))
                and digest(read(reference['path'])) == reference['sha256'], 'Prepared prerequisite changed.')
    return before, setup


def predecessor_rosters(before):
    def preserved(name):
        raw = read(name)
        require(digest(raw) == before['previous_film_evidence'][relative(inside(name))],
                'Predecessor evidence differs from baseline.')
        return raw
    manifest = json.loads(preserved(PREVIOUS+'validation.json'))
    checks, rosters = checks_by_name(manifest), {}
    for name in ('full', 'slow'):
        c, r = checks[name+'_collection'], checks[name]
        require(c['exit'] == r['exit'] == 0, 'Predecessor command did not pass.')
        rosters[name] = audit_pytest(preserved(c['log']).decode(), preserved(r['log']).decode())
        require(rosters[name] == r['observed'],
                'Predecessor raw roster differs.')
    require(rosters['full']['totals']['skipped'] == 8
            and rosters['slow']['totals']['passed'] == 81, 'Predecessor skip/slow contract differs.')
    return rosters


def mutation_command(command, targets, *, collection=False):
    if collection:
        require(command[1:] == ['-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *targets, '--collect-only'],
                'Mutation collection flags or targets differ.')
    else:
        require(command[1:-2] == ['-B', '-m', 'pytest', '-vv', '--tb=short', '-p', 'no:cacheprovider', *targets]
                and command[-2] == '--basetemp', 'Mutation command differs.')
        scratch_path(command[-1])


def audit_mutations(new, targeted):
    value = read_json(RESULTS/(PREFIX+'mutations.json'))
    retained = RESULTS/f'{PREFIX}mutations_attempt{value["attempt"]}.json'
    require(read(retained) == read(RESULTS/(PREFIX+'mutations.json')), 'Mutation canonical differs.')
    require(value['base'] == BASE and value['complete'] is True and value['passed'] is True
            and value['restored'] is True and value['all_function_assertion_witnesses'] is True,
            'Incomplete actual mutation window.')
    require(value['source_identity_before'] == value['source_identity_after'] == probe_identity(),
            'Mutation source identity/restoration differs.')
    verify_guard(value['guard_before'])
    verify_guard(value['guard_after'])
    harness = 'studies/fdtd_oscillator_stack_mutations.py'
    require(value['harness_sha256'] == digest(read(harness)) and value['test_sha256'] == digest(read(NEW_TEST)),
            'Mutation harness/test changed.')
    require(read(value['test_snapshot']) == read(NEW_TEST)
            and digest(read(value['test_snapshot'])) == value['test_snapshot_sha256']
            and read(value['harness_snapshot']) == read(harness), 'Mutation input snapshots differ.')
    tree = ast.parse(read(harness))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mutations')
    require(len(function.body) == 1 and isinstance(function.body[0], ast.Return), 'Mutation table is not literal.')
    specs = ast.literal_eval(function.body[0].value)
    require(len(specs) > 0 and len(specs) == len({s[0] for s in specs}),
            'Every frozen literal mechanism needs a unique actual witness.')
    funcs = {n.name for n in ast.parse(read(NEW_TEST)).body
             if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}
    require(funcs == set(EXPECTED_TEST_FUNCTIONS)
            and {s[4].split('::')[1].split('[')[0] for s in specs} == funcs,
            'Every new function needs an actual assertion witness.')
    base = checks_by_name(targeted)['new_stack']
    require(inside(value['baseline_log']) == inside(base['log']) and value['baseline_log_sha256'] == base['sha256']
            and digest(read(value['targeted_manifest'])) == value['targeted_manifest_sha256']
            and read(value['targeted_manifest']) == read(RESULTS/(PREFIX+'targeted.json')), 'Wrong targeted baseline link.')
    collection = value['collection']
    require(collection['exit_code'] == collection['exit'] == 0
            and digest(read(collection['log'])) == collection['sha256'], 'Mutation collection failed/changed.')
    mutation_command(collection['command'], [NEW_TEST], collection=True)
    require(Path(collection['command'][0]).resolve() == Path(value['executable']).resolve(), 'Collection interpreter differs.')
    collected = nodes(read(collection['log']).decode())
    require(set(collected) == set(new['node_status']) and collected == collection['nodes']
            and len(collected) == collection['collected'], 'Mutation/new collection differs.')
    require(len(value['witnesses']) == len(specs), 'Missing mutation witness.')
    checks = checks_by_name(value)
    previous = datetime.fromisoformat(collection['ended_utc'])
    for witness, spec in zip(value['witnesses'], specs):
        name, source, before, after, node = spec
        require(witness['name'] == name and witness['production_file'] == source
                and witness['test'] == node and node in collected, 'Wrong witness/node binding.')
        require(witness['replacement'] == {'before': before, 'after': after}, 'Replacement changed.')
        mutation_command(witness['command'], [node])
        require(Path(witness['command'][0]).resolve() == Path(value['executable']).resolve(), 'Mutation interpreter differs.')
        original, mutant, restored = (read(witness[k]) for k in ('original_snapshot', 'mutant_snapshot', 'restored_snapshot'))
        require(all(Path(witness[k]).suffix == '.raw' for k in ('original_snapshot', 'mutant_snapshot', 'restored_snapshot')),
                'Unsafe mutation source suffix.')
        require(original == restored == read(source) and digest(original) == witness['original_sha256']
                == witness['restored_sha256'] == witness['restored_snapshot_sha256'], 'Original source not byte-restored.')
        require(digest(mutant) == witness['mutant_sha256'], 'Mutant snapshot changed.')
        text = original.decode().replace('\r\n', '\n')
        require(text.count(before) == 1 and before != after, 'Mutation anchor differs.')
        changed = text.replace(before, after, 1)
        if b'\r\n' in original:
            changed = changed.replace('\n', '\r\n')
        require(mutant == changed.encode(), 'Mutant exceeds the declared replacement.')
        ast.parse(mutant.decode())
        expected_guard = json.loads(json.dumps(value['guard_before']))
        expected_guard['maps']['wavebit'][source] = digest(mutant)
        require(witness['guard_mutated'] == expected_guard, 'A second source/fixture changed during mutation.')
        require(witness['source_identity_before'] == witness['source_identity_after'] == probe_identity()
                and witness['restored'] and witness['source_mtime_before_ns'] == witness['source_mtime_restored_ns'],
                'Witness source identity/mtime restoration differs.')
        raw = read(witness['raw_log'])
        require(digest(raw) == witness['raw_log_sha256'], 'Mutation raw log changed.')
        text = raw.decode()
        check_log(text)
        observed, totals = reports(text), summary(text)
        require(witness['accepted'] is True and witness['assertion_failure'] is True
                and witness['observed']['assertion_failure'] is True
                and witness['exit_code'] == 1 and observed == {node: 'failed'}
                and totals['failed'] == sum(totals.values()) == 1 and assertion_failure(text),
                'Mutation lacks exactly the intended assertion failure.')
        failed = [line for line in text.splitlines() if line.startswith('FAILED ')]
        require(len(failed) == 1 and failed[0].split(' - ', 1)[0] == 'FAILED '+node, 'Failure summary node differs.')
        require(observed == witness['observed']['node_status'] and totals == witness['observed']['totals'],
                'Saved witness counts differ from raw.')
        child = checks[name]
        require(child['command'] == witness['command'] and child['exit'] == witness['exit_code']
                and inside(child['log']) == inside(witness['raw_log']) and child['sha256'] == witness['raw_log_sha256'],
                'Witness and actual child receipt differ.')
        stamp = lambda k: datetime.fromisoformat(witness[k])
        require(previous <= stamp('started_utc') <= stamp('ended_utc') <= stamp('restored_utc'),
                'Witness source windows overlap or chronology differs.')
        previous = stamp('restored_utc')
    require(set(checks) == {'mutation_collection', 'final_baseline_collection', 'final_baseline'}|{s[0] for s in specs},
            'Unreviewed/missing mutation child command.')
    for child in value['checks']:
        require(type(child['exit']) is int and 'launcher_error' not in child
                and digest(read(child['log'])) == child['sha256'], 'Missing actual child exit or changed raw output.')
        require(not child['environment'].get('XLA_FLAGS') and inside(child['environment']['PYTHONPATH']) == ROOT,
                'Mutation child environment differs.')
    final = value['final_baseline']
    require(final['exit_code'] == 0 and digest(read(final['raw_log'])) == final['raw_log_sha256']
            and previous <= datetime.fromisoformat(final['started_utc']), 'Restored baseline did not complete after restoration.')
    replay = command_pair(value, 'final_baseline', [NEW_TEST])
    require(replay == new and final['observed'] == replay and final['passed'] == new['collected'],
            'Restored full new-test baseline differs.')
    return {'mechanisms': len(specs), 'functions': sorted(funcs), 'restored_cases': new['collected'],
            'ended_utc': value['ended_utc']}


FILMS = {'mixed-pair': ((1., 'drude', 9., 1., None, .10),
                        (1., 'lorentz', 8., 1., 40., .15)),
         'lorentz-triple': ((1.5, 'lorentz', 5., .8, 10., .10),
                            (1., 'lorentz', 8., 1.5, 20., .15),
                            (2.25, 'lorentz', 2*math.pi, 0., 0., .10))}
GAPS = {'mixed-pair': (.15,), 'lorentz-triple': (.125, .175)}
WAVELENGTHS = (.9, 1., 1.1)
FRACTIONS, POLS = (.3, .1), ('s', 'p')
SOURCE_BAND = {'wavelength': 1., 'df': .1, 'cutoff': 1e-6}



def film_rows(family):
    return [list(row) for row in FILMS[family]]


def material_epsilon(film, omega):
    """Independent declared continuum response; no modal/lowering helper."""
    eps, kind, frequency, gamma, strength, _depth = film
    if kind == 'drude':
        return eps-frequency**2/(omega**2+1j*gamma*omega)
    require(kind == 'lorentz', 'Unexpected declared material family.')
    if strength == 0:
        return complex(eps)
    return eps+strength/(frequency**2-omega**2-1j*gamma*omega)


def band_triangle_bound(film, low, high):
    """Independent sufficient denominator-minimum bound, not the runtime critical-point formula."""
    eps, kind, frequency, gamma, strength, _depth = film
    d, strength = (0., frequency**2) if kind == 'drude' else (frequency**2, strength)
    if strength == 0:
        return high*math.sqrt(eps)
    x = min(high**2, max(low**2, d-gamma**2/2))
    minimum = math.sqrt((d-x)**2+gamma**2*x)
    require(minimum > 0, 'Unbounded oscillator band is not an eligible physical fixture.')
    return high*math.sqrt(eps+strength/minimum)


def assertion_failure(text):
    errors = [re.sub(r'^E\s+', '', line).strip() for line in text.splitlines() if re.match(r'^E\s+', line)]
    require(errors, 'No raw pytest failure evidence.')
    missing = {"Failed: DID NOT RAISE <class 'ValueError'>", 'Failed: DID NOT RAISE ValueError'}
    terminals = [line for line in errors if re.match(r'^(?:[\w.]+(?:Error|Exception)|Failed):', line)]
    require(any(line.startswith('AssertionError:') or line.startswith('assert ') or line in missing for line in errors),
            'No assertion or named missing-refusal failure.')
    require(all(line.startswith('AssertionError:') or line in missing for line in terminals),
            'Collateral exception/import/fixture/runtime failure is not an assertion witness.')
    return True


def optical_reference(family, wavelength, pol):
    """Separate scalar characteristic-matrix replay; no project/NumPy import."""
    omega = 2*math.pi/wavelength
    transverse = math.sin(math.radians(35.))*wavelength
    q0 = math.sqrt(1-transverse**2)
    y0 = q0 if pol == 's' else 1/q0
    layers = []
    for index, film in enumerate(FILMS[family]):
        layers.append((material_epsilon(film, omega), film[-1]))
        if index < len(GAPS[family]):
            layers.append((1.+0j, GAPS[family][index]))
    a, b, c, d = 1.+0j, 0j, 0j, 1.+0j
    for eps, depth in layers:
        q = cmath.sqrt(eps-transverse**2)
        y = q if pol == 's' else eps/q
        phase = omega*q*depth
        u, v, w = cmath.cos(phase), -1j*cmath.sin(phase)/y, -1j*cmath.sin(phase)*y
        a, b, c, d = a*u+b*w, a*v+b*u, c*u+d*w, c*v+d*u
    electric, magnetic = a+b*y0, c+d*y0
    denominator = y0*electric+magnetic
    r = (y0*electric-magnetic)/denominator
    return (r if pol == 's' else -r), 2*y0/denominator


def zvalues(row, name):
    require(len(row[name]) == 3 and all(len(v) == 2 for v in row[name]), 'Complex row shape differs.')
    return [complex(*v) for v in row[name]]


def validity(row):
    finite = all(math.isfinite(abs(v)) for name in ('r', 't') for v in zvalues(row, name))
    finite &= all(math.isfinite(v) for name in ('error_r', 'error_t', 'R', 'T', 'absorption')
                  for v in row[name])
    finite &= row['energy_finite'] and math.isfinite(row['energy_max']) and row['energy_max'] > 0
    hard = any(w['category'] in ('GrowingFieldWarning', 'UnstableFieldWarning') for w in row['warnings'])
    stable = row['stable'] is True and finite and not hard
    settled = bool(row['decay']) and all(math.isfinite(v) and v <= 1e-5 for v in row['decay'].values())
    return {'valid': stable and settled, 'otherwise_valid': stable, 'settled': settled}


def raw_response(entry, *, runtime, start, end, calibration=False):
    """Validate recorded worker exit/command, raw serialization and independent optics."""
    require(type(entry['exit_code']) is int and entry['exit_code'] == 0, 'Physical worker failed.')
    raw, saved = read(entry['raw_log']), read(entry['result'])
    require(digest(raw) == entry['raw_log_sha256']
            and digest(saved) == entry['result_sha256'], 'Physical raw hash differs.')
    row = json.loads(saved)
    require(json.loads(raw) == row, 'Raw stdout and response JSON disagree.')
    require(row['kind'] == 'Oscillator-stack-probe' and row['source_identity'] == probe_identity()
            and row['source_unchanged_during_run'] is True, 'Worker source identity differs.')
    for key in ('python', 'executable', 'backend_versions', 'environment'):
        require(row[key] == runtime[key], 'Physical runtime changed: '+key)
    stamp = lambda value: datetime.fromisoformat(value)
    require(stamp(start) <= stamp(entry['started_utc']) <= stamp(row['started_utc'])
            <= stamp(row['ended_utc']) <= stamp(entry['ended_utc']) <= stamp(end), 'Worker chronology differs.')
    config = row['config']
    family = config['family']
    require(family in FILMS and config['pol'] in POLS and config['fraction'] in FRACTIONS
            and config['n'] in (40, 80, 160) and config['periods'] in (120, 240), 'Unexpected physical row.')
    expected = {'backend': config['backend'], 'n': config['n'], 'pol': config['pol'], 'family': family,
                'theta': 35., 'periods': config['periods'], 'pml': config['pml'],
                'fraction': config['fraction'], 'courant': .5,
                'films': film_rows(family), 'gaps': list(GAPS[family])}
    require(config == expected, 'Physical config has undeclared settings.')
    if calibration:
        require(config == {**expected, 'n': 40, 'pol': 'p', 'fraction': .3, 'periods': 120, 'pml': 24}
                and config['backend'] in ('numpy', 'jax', 'torch'), 'Wrong calibration row.')
        flags = ['--family', family, '--backend', config['backend'], '--n', '40', '--pol', 'p',
                 '--fraction', '0.3', '--periods', '120', '--pml', '24']
    else:
        require(config['backend'] == 'numpy' and (config['pml'] == 24 or
                (config['n'] == 80 and config['periods'] == 240 and config['pml'] == 48)),
                'Unexpected campaign backend/PML control.')
        flags = ['--family', family, '--fraction', str(config['fraction']), '--pol', config['pol'],
                 '--n', str(config['n']), '--periods', str(config['periods']),
                 '--pml', str(config['pml'])]
    command = entry['command']
    require(command[:4] == [runtime['executable'], '-B', '-m', 'studies.fdtd_oscillator_stack_probe']
            and command[4:-2] == flags and command[-2] == '--output'
            and inside(command[-1]) == inside(entry['result']), 'Physical worker command differs.')
    require(row['source_band'] == SOURCE_BAND and row['wavelengths'] == list(WAVELENGTHS),
            'Physical frequencies changed.')
    faces, top = [], 0.
    for i, film in enumerate(FILMS[family]):
        thickness = film[-1]
        bottom = top-thickness
        faces.append([top, bottom])
        top = bottom-(GAPS[family][i] if i < len(GAPS[family]) else 0.)
    depth = -top
    near(row['film_faces'], faces)
    near(row['total_depth'], depth)
    near(row['outer_faces'], [0., -depth])
    dx, kx = 1/config['n'], 2*math.pi*math.sin(math.radians(35.))
    low = -(math.ceil((depth+.8)/dx)+config['pml']+config['fraction'])*dx
    nz = math.ceil((1.4+config['pml']*dx-low)/dx)
    require(row['grid'] == [3, nz, 1, 1], 'Unexpected grid shape.')
    near(row['grid_low'], [-dx/2, -dx/2, low], atol=2e-13*dx, rtol=0.)
    near(row['dt'], .5*dx/math.sqrt(3.), atol=0., rtol=3e-16)
    require(row['steps'] == math.ceil(config['periods']/row['dt']-1e-9), 'Step count differs.')
    near(row['kx'], kx)
    theta = [math.degrees(math.asin(kx/(2*math.pi/w))) for w in WAVELENGTHS]
    near(row['reference_theta_at'], theta, atol=2e-12, rtol=0.)
    near(row['theta_at'], theta, atol=2e-12, rtol=0.)
    near([[complex(*v) for v in eps] for eps in row['epsilon']],
         [[material_epsilon(film, 2*math.pi/w) for w in WAVELENGTHS]
          for film in FILMS[family]])
    for i, wavelength in enumerate(WAVELENGTHS):
        reference = optical_reference(family, wavelength, config['pol'])
        for name, oracle in zip(('r', 't'), reference):
            near(zvalues(row, 'reference_'+name)[i], oracle, atol=3e-13, rtol=3e-13)
            # Stored complex oracle determines exact registered error; independent
            # scalar reference only checks its correctness within arithmetic roundoff.
            near(row['error_'+name][i], abs(zvalues(row, name)[i]-zvalues(row, 'reference_'+name)[i]),
                 atol=2e-15, rtol=3e-13)
        near(row['absorption'][i], 1-row['R'][i]-row['T'][i])
        near(row['reference_absorption'][i], 1-abs(reference[0])**2-abs(reference[1])**2)
    require(row['energy_finite'] is True and row['energy_rows'] > 0
            and row['energy_shape'][0] == row['energy_rows']
            and type(row['bytes_per_state']) is int and row['bytes_per_state'] > 0,
            'Missing finite energy/storage evidence.')
    require(all(set(w) == {'category', 'message'} for w in row['warnings']), 'Warning payload changed.')
    return row


def audit_preflight(value):
    """Replay recorded analytic instruments and coarse band/placement certificates."""
    require(value['analytic_controls'] == 'passed' and value['field_solves'] == 0,
            'Analytic preflight did not pass without field responses.')
    require(set(value['bulk_bounds']) == set(FILMS)
            and set(value['support_controls']) == set(FILMS)
            and set(value['sampled_cells']) == set(FILMS), 'Preflight family coverage differs.')
    width = math.sqrt(2*math.log(1e6))*.1
    low, high = 2*math.pi*(1-width), 2*math.pi*(1+width)
    for family, films in FILMS.items():
        bounds = value['bulk_bounds'][family]
        expected = [band_triangle_bound(film, low, high) for film in films]
        near(bounds, {'angular_band': [low, high], 'member_modulus_bounds': expected,
                      'host_modulus_bound': high, 'qualification': 'independent denominator-minimum triangle bound; ordinary floating evaluation; no endpoint-monotonicity assumption'})
        require(low > 0 and max([high, *expected])/40 < 2*math.pi/8, 'Coarse whole-band floor failed.')
        depth, faces = 0., []
        for i, film in enumerate(films):
            thickness = film[-1]
            faces.append([-depth, -depth-thickness])
            depth += thickness
            if i < len(GAPS[family]):
                depth += GAPS[family][i]
        supports = value['support_controls'][family]
        require([(v['n'], v['fraction']) for v in supports] ==
                [(n, f) for n in (40, 80, 160) for f in FRACTIONS], 'Placement preflight roster differs.')
        for row in supports:
            dx, fraction = 1/row['n'], row['fraction']
            grid_low = -(math.ceil((depth+.8)/dx)+24+fraction)*dx
            spans = [[math.floor((bottom-grid_low)/dx)-1, math.ceil((top-grid_low)/dx)+1]
                     for top, bottom in faces]
            phases = [((face-grid_low)/dx) % 1 for pair in faces for face in pair]
            require(row['spans'] == spans, 'Independent padded support changed.')
            ordered = sorted(spans)
            require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), 'Padded supports intersect.')
            near(row['face_fractions'], phases, atol=2e-14, rtol=0.)
            near(phases, [fraction]*len(phases), atol=2e-13, rtol=2e-13)
        require(len(value['sampled_cells'][family]) == len(films), 'Sampled member coverage differs.')
        for row, film in zip(value['sampled_cells'][family], films):
            omega = [2*math.pi/w for w in WAVELENGTHS]
            index = [cmath.sqrt(material_epsilon(film, w)) for w in omega]
            phase = [40*2*math.pi/(w*z.real) for w, z in zip(omega, index)]
            skin = [None if z.imag == 0 else 40/(w*z.imag) for w, z in zip(omega, index)]
            near(row, {'phase_cells': phase, 'skin_cells': skin})
            require(min(phase) >= 8 and all(v is None or v >= 1 for v in skin), 'Sampled floor failed.')
    keys = {'fixed-kx-transverse'}
    for family, films in FILMS.items():
        keys |= {f'{family}-n{n}-f{f}-phases' for n in (40, 80, 160) for f in FRACTIONS}
        keys |= {f'{family}-normal-r-sign', f'{family}-normal-t'}
        for pol in POLS:
            keys |= {f'{family}-{pol}-{part}' for part in
                     ('matched-r', 'matched-t', 'lossless-power', 'reciprocal-t', 'reflection-asymmetry',
                      'recursive', 'reverse-recursive')}
            keys |= {f'{family}-{pol}-{part}-{i}' for i in range(len(films)) for part in ('single', 'zero-depth')}
    require(set(value['residuals']) == keys, 'Analytic instrument control is omitted.')
    for key, residual in value['residuals'].items():
        require(math.isfinite(residual) and residual >= 0, 'Invalid analytic residual.')
        if key.endswith('reflection-asymmetry'):
            require(residual > 1e-10, 'Asymmetry control became insensitive.')
        else:
            # Stored maximum must respect the original elementwise 2e-13+2e-13*|expected|;
            # all these expected magnitudes are <=1, so 4e-13 is a necessary replay bound.
            require(residual <= 4e-13, 'Analytic control residual exceeds original arithmetic tolerance.')
    near(value['reference_theta_at'],
         [math.degrees(math.asin(math.sin(math.radians(35.))*w)) for w in WAVELENGTHS],
         atol=2e-12, rtol=0.)



def calibration_test_transition(acceptance):
    def linked(item):
        raw = read(item['path'])
        require(digest(raw) == item['sha256'], 'Optical threshold transition link changed.')
        return raw
    original, final = linked(acceptance['test_before']), linked(acceptance['test_after'])
    linked(acceptance['test_transition'])
    require(final == read(NEW_TEST), 'Actual test differs from accepted test snapshot.')
    def masked(raw, expected):
        text = raw.decode('utf-8')
        tree = ast.parse(text)
        rows = [n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CALIBRATED_CEILINGS' for t in n.targets)]
        require(len(rows) == 1 and ast.literal_eval(rows[0].value) == expected, 'Wrong optical threshold transition.')
        doc = tree.body[0]
        require(isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)
                and isinstance(doc.value.value, str), 'Expected an explicit module docstring.')
        lines = text.splitlines(keepends=True)
        spans = []
        for node in (doc.value, rows[0].value):
            lo = sum(map(len, lines[:node.lineno-1]))+node.col_offset
            hi = sum(map(len, lines[:node.end_lineno-1]))+node.end_col_offset
            spans.append((lo, hi))
        for lo, hi in sorted(spans, reverse=True):
            text = text[:lo]+'<accepted-transition>'+text[hi:]
        return text
    ceilings = acceptance['accepted_ceilings']
    require(masked(original, None) == masked(final, (ceilings['r'], ceilings['t'])),
            'Test changed outside its disclosed module docstring and optical constant.')


def calibration():
    acceptance = read_json(RESULTS/(PREFIX+'calibration_acceptance.json'))
    calibration_test_transition(acceptance)
    path = inside(acceptance['calibration'])
    value = read_json(path)
    require(digest(read(path)) == acceptance['calibration_sha256'], 'Accepted calibration changed.')
    require(value['complete'] and value['source_unchanged'] and value['production_unchanged']
            and value['source_identity'] == acceptance['source_identity'] == probe_identity(),
            'Calibration incomplete or source changed.')
    require(value['base_commit'] == BASE and value['support_sha256'] == digest(read('studies/fdtd_oscillator_stack_support.py')),
            'Calibration support/base changed.')
    audit_preflight(value['analytic_preflight'])
    require(value['optical_ceilings'] is None, 'Calibration preclaimed an optical ceiling.')
    ceilings = acceptance['accepted_ceilings']
    require(set(ceilings) == {'r', 't'} and all(math.isfinite(v) and v > 0 for v in ceilings.values()),
            'Invalid post-disclosure ceilings.')
    constants = [ast.literal_eval(n.value) for n in ast.parse(read(NEW_TEST)).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CALIBRATED_CEILINGS' for t in n.targets)]
    require(constants == [(ceilings['r'], ceilings['t'])], 'Actual optical test ceilings differ from accepted disclosure.')
    require(datetime.fromisoformat(value['ended_utc']) < datetime.fromisoformat(acceptance['accepted_utc']),
            'Acceptance precedes disclosure.')
    require(value['production_sources'] == maps()['wavebit'], 'Calibration source map changed.')
    archive = read(value['production_archive'])
    require(digest(archive) == value['production_archive_sha256'], 'Production archive changed.')
    with zipfile.ZipFile(io.BytesIO(archive)) as stream:
        require(set(stream.namelist()) == set(value['production_sources'])
                and len(stream.namelist()) == len(value['production_sources']), 'Archive member roster differs.')
        require(all(digest(stream.read(n)) == h for n, h in value['production_sources'].items()),
                'Pre-response production archive differs.')
    sources = ['studies/fdtd_oscillator_stack_probe.py', 'studies/fdtd_oscillator_stack_calibration.py',
               'studies/fdtd_oscillator_stack_support.py', *probe_identity()['reference_dependencies']]
    require(len(value['frozen_sources']) == len(sources), 'Missing calibration source archives.')
    for item, source in zip(value['frozen_sources'], sources):
        require(Path(item['file']).suffix in ('.raw', '.gz'), 'Unsafe source archive suffix.')
        raw = read(item['file'])
        require(digest(raw) == item['sha256'] and raw == read(source), 'Calibration helper archive changed.')
    require([(w['family'], w['backend']) for w in value['workers']] ==
            [(f, b) for f in FILMS for b in ('numpy', 'jax', 'torch')], 'Six-row sequence differs.')
    rows, previous = {}, value['started_utc']
    for worker in value['workers']:
        row = raw_response(worker, runtime=value, start=previous, end=value['ended_utc'], calibration=True)
        previous = worker['ended_utc']
        require(validity(row)['valid'], 'Calibration physical row is invalid.')
        rows[worker['family'], worker['backend']] = row
    maxima = {name: max(max(r['error_'+name]) for r in rows.values()) for name in ('r', 't')}
    near(value['measured_max_absolute_complex_errors'], maxima)
    near(acceptance['measured_max_absolute_complex_errors'], maxima)
    require(all(maxima[n] < ceilings[n] for n in ceilings), 'Calibration ceiling violated.')
    by_family = {f: {n: max(max(rows[f, b]['error_'+n]) for b in ('numpy', 'jax', 'torch'))
                      for n in ('r', 't')} for f in FILMS}
    near(value['measured_max_errors_by_family'], by_family)
    comparisons = {f: {b: {n: [abs(a-e)/max(1., abs(e))
                     for a, e in zip(zvalues(rows[f, b], n), zvalues(rows[f, 'numpy'], n))]
                     for n in ('r', 't')} for b in ('jax', 'torch')} for f in FILMS}
    near(value['backend_pair_scaled_differences'], comparisons, atol=3e-16, rtol=1e-12)
    require(value['backend_pair_budget'] == 128*2**-52 and value['backend_pairs_within_budget'] is True
            and all(max(v) <= 128*2**-52 for f in comparisons.values() for b in f.values() for v in b.values()),
            'Backend comparison exceeded its fixed arithmetic budget.')
    require(value['finite_stable_all_rows'] and value['all_decay_records_present'], 'Calibration diagnostics missing.')
    near(value['max_decay'], max(v for row in rows.values() for v in row['decay'].values()), atol=0.)
    return acceptance, value, rows


def job_key(family, fraction, pol, n, periods=120, pml=24):
    return f'{family}_f{fraction:.1f}_{pol}_n{n}_t{periods}_pml{pml}'


def order(first, second, noise, difference):
    margin = (2 if difference else 1)*noise
    resolved = min(first, second) > 1e-10 and 2*noise <= .1*min(first, second)
    point = math.log2(first/second) if min(first, second) > 0 else None
    interval = [math.log2((first-margin)/(second+margin)),
                math.log2((first+margin)/(second-margin))] if resolved else None
    classification = ('unresolved' if interval is None else
                      'second' if 1.8 <= interval[0] and interval[1] <= 2.3 else
                      'low' if interval[1] <= 1.75 else 'mixed')
    return {'point': point, 'interval': interval, 'classification': classification, 'resolved': resolved}


def audit_campaign(acceptance, calibration_value, known_rows, targeted, mutation):
    reg = read_json(RESULTS/(PREFIX+'registration.json'))
    bare = {k: v for k, v in reg.items() if k != 'fingerprint'}
    require(digest(canonical(bare)) == reg['fingerprint'], 'Registration fingerprint differs.')
    source = probe_identity()
    require(all(reg['identity'][k] == v for k, v in source.items()), 'Registered source changed.')
    for label, name in (('campaign_sha256', 'studies/fdtd_oscillator_stack_campaign.py'),
                        ('support_sha256', 'studies/fdtd_oscillator_stack_support.py'),
                        ('criteria_dependency_sha256', 'studies/fdtd_conductive_film_campaign.py'),
                        ('test_sha256', NEW_TEST), ('allocation_test_sha256', 'tests/test_fd_state_bytes.py')):
        require(reg['identity'][label] == digest(read(name)), 'Registered helper/test changed.')
    for stamp in (acceptance['accepted_utc'], targeted['ended_utc'], mutation['ended_utc']):
        require(datetime.fromisoformat(stamp) < datetime.fromisoformat(reg['registered_utc']),
                'Registration precedes its disclosed development results.')
    for item in reg['known_exposures']+reg['frozen_sources']:
        require(digest(read(RESULTS/item['file'])) == item['sha256'], 'Registered input changed.')
    frozen_sources = ['studies/fdtd_oscillator_stack_campaign.py', 'studies/fdtd_oscillator_stack_probe.py',
        'studies/fdtd_conductive_film_campaign.py', 'studies/fdtd_oscillator_stack_support.py',
        NEW_TEST, 'tests/test_fd_state_bytes.py',
        *source['reference_dependencies']]
    require(len(reg['frozen_sources']) == len(frozen_sources), 'Registered source snapshots missing.')
    for item, name in zip(reg['frozen_sources'], frozen_sources):
        require(Path(item['file']).suffix in ('.raw', '.gz')
                and read(RESULTS/item['file']) == read(name), 'Frozen registered source differs.')
    known = {relative(RESULTS/r['file']): r['sha256'] for r in reg['known_exposures']}
    require(len(known) == len(reg['known_exposures']), 'Duplicate known exposure.')
    for path in sorted(RESULTS.glob(PREFIX+'*')):
        stem = path.name[len(PREFIX):]
        if not stem.startswith(('calibration', 'mutations', 'targeted', 'algebra', 'controls')):
            continue
        for file in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if not file.is_file():
                continue
            if stem.startswith('controls') and not any(p.startswith(('targeted', 'mutations', 'calibration'))
                    for p in file.relative_to(path).parts):
                continue
            require(relative(file) in known and digest(read(file)) == known[relative(file)],
                    'Undisclosed or changed development attempt: '+relative(file))
    protocol = reg['protocol']
    checks = {'base_commit': BASE, 'rungs': [40, 80, 160], 'fractions': list(FRACTIONS),
              'polarizations': list(POLS), 'wavelengths': list(WAVELENGTHS),
              'base_worker_count': 40, 'family_sequence': list(FILMS), 'base_periods': 120,
              'long_periods': 240, 'source_band': SOURCE_BAND, 'decay_limit': 1e-5,
              'optical_regression_ceilings': acceptance['accepted_ceilings']}
    require(all(protocol[k] == v for k, v in checks.items()), 'Registered numeric protocol differs.')
    require(protocol['materials'] == {f: {'films': film_rows(f), 'gaps': list(GAPS[f])}
                                      for f in FILMS}, 'Registered materials differ.')
    audit_preflight({k: v for k, v in reg['analytic_preflight'].items() if k in calibration_value['analytic_preflight']})
    require(all(reg['analytic_preflight'][k] == v for k, v in calibration_value['analytic_preflight'].items()), 'Registered/calibration preflight differs.')
    entries = [json.loads(line) for line in read(RESULTS/(PREFIX+'campaign.jsonl')).decode().splitlines()]
    require(entries and len({e['key'] for e in entries}) == len(entries), 'Empty/duplicate campaign ledger.')
    report = read_json(RESULTS/(PREFIX+'report.json'))
    require(report['complete'] is True and report['stop_reason'] is None
            and report['registration'] == reg['fingerprint'] and report['rungs'] == [40, 80, 160]
            and report['worker_rows'] == len(entries), 'Campaign did not complete its declared protocol.')
    available, observed, previous = {}, set(), reg['registered_utc']
    for entry in entries:
        require(entry['registration'] == reg['fingerprint'], 'Worker belongs to another registration.')
        row = raw_response(entry, runtime=reg['identity'], start=previous,
                           end=entry['ended_utc'])
        previous = entry['ended_utc']
        conf = row['config']
        require(entry['key'] == job_key(*(conf[n] for n in
                ('family', 'fraction', 'pol', 'n', 'periods', 'pml'))), 'Wrong worker key.')
        available[entry['key']] = row
        observed.update((inside(entry['raw_log']), inside(entry['result'])))
    jobs = RESULTS/(PREFIX+'campaign_jobs')
    require(set(jobs.iterdir()) == observed, 'Unledgered or missing campaign raw artifact.')
    cursor, consumed, groups, cells = 0, {}, [], []
    replacements = []
    def demand(family, fraction, pol, n, periods=120, pml=24):
        nonlocal cursor
        key = job_key(family, fraction, pol, n, periods, pml)
        if key in consumed:
            return consumed[key]
        require(cursor < len(entries) and entries[cursor]['key'] == key,
                'Unexpected worker/order/replacement.')
        cursor += 1
        consumed[key] = available[key]
        return consumed[key]
    for family in FILMS:
        for fraction in FRACTIONS:
            for pol in POLS:
                chosen = []
                for n in (40, 80, 160):
                    row = demand(family, fraction, pol, n)
                    state = validity(row)
                    if not state['valid']:
                        require(state['otherwise_valid'], 'Hard failure cannot receive replacement.')
                        replacements.append(job_key(family, fraction, pol, n, 240))
                        row = demand(family, fraction, pol, n, 240)
                        require(validity(row)['valid'], 'Replacement at registered cap invalid.')
                    chosen.append(row)
                    if n == 80:
                        for pml in (24, 48):
                            require(validity(demand(family, fraction, pol, n, 240, pml))['valid'],
                                    'Fixed middle-grid control invalid.')
                periods = [r['config']['periods'] for r in chosen]
                groups.append({'family': family, 'fraction': fraction, 'polarization': pol,
                               'complete': True, 'selected_periods': periods})
                controls = [available[job_key(family, fraction, pol, 80, t, p)]
                            for t, p in ((120, 24), (240, 24), (240, 48))]
                for coefficient in ('r', 't'):
                    for i, wavelength in enumerate(WAVELENGTHS):
                        samples = [zvalues(row, coefficient)[i] for row in chosen]
                        b, t, p = [zvalues(row, coefficient)[i] for row in controls]
                        time_noise, pml_noise = abs(t-b), abs(p-t)
                        noise = time_noise+pml_noise
                        diffs = [abs(samples[0]-samples[1]), abs(samples[1]-samples[2])]
                        errors = [r['error_'+coefficient][i] for r in chosen]
                        cells.append({'family': family, 'fraction': fraction, 'polarization': pol,
                            'coefficient': coefficient, 'wavelength': wavelength, 'resolved_input': True,
                            'values': [[z.real, z.imag] for z in samples],
                            'continuum': chosen[0]['reference_'+coefficient][i], 'errors': errors,
                            'differences': diffs, 'absorption': [r['absorption'][i] for r in chosen],
                            'time_proxy': time_noise, 'pml_proxy': pml_noise, 'proxy': noise,
                            'self_order': order(*diffs, noise, True),
                            'direct_orders': [order(errors[j], errors[j+1], noise, False) for j in (0, 1)]})
    require(cursor == len(entries) and len(entries) == 40+sum(
            '_n80_' not in key for key in replacements), 'Extra/missing workers after completed protocol.')
    require(report['groups'] == groups and len(report['cells']) == len(cells) == 48,
            'Missing registered group/cell.')
    near(report['cells'], cells, atol=2e-12, rtol=3e-12)
    counts = {'self': dict(Counter(c['self_order']['classification'] for c in cells)),
              'direct': dict(Counter(o['classification'] for c in cells for o in c['direct_orders'])),
              'final_direct': dict(Counter(c['direct_orders'][-1]['classification'] for c in cells))}
    repeats = {}
    payload = ('config', 'grid', 'source_band', 'dt', 'steps', 'kx', 'wavelengths', 'theta_at',
        'reference_theta_at', 'film_faces', 'total_depth', 'outer_faces', 'grid_low', 'epsilon',
        'r', 't', 'reference_r', 'reference_t', 'error_r', 'error_t', 'R', 'T', 'absorption',
        'reference_absorption', 'stable', 'decay', 'energy_shape', 'energy_rows',
        'energy_finite', 'energy_max', 'bytes_per_state', 'warnings')
    for family in FILMS:
        old = known_rows[family, 'numpy']
        new = available[job_key(family, .3, 'p', 40)]
        repeats[family] = all(old[k] == new[k] for k in payload)
    require(datetime.fromisoformat(calibration_value['ended_utc']) < datetime.fromisoformat(reg['registered_utc']),
            'Calibration after registration.')
    return {'registration': reg['fingerprint'], 'workers': len(entries), 'base_workers': 40,
            'replacement_keys': replacements, 'cells': 48, 'direct_cells': 96, 'final_direct_cells': 48,
            'classifications': counts, 'known_calibration_payload_identical': repeats,
            'exception_cells': [c for c in cells if c['self_order']['classification'] != 'second'
                               or any(o['classification'] != 'second' for o in c['direct_orders'])],
            'max_errors_by_rung': [max(c['errors'][i] for c in cells) for i in range(3)],
            'max_proxy': max(c['proxy'] for c in cells),
            'qualification': 'All per-cell outcomes retained; no required all-second verdict. '
                             'Middle-grid proxy is empirical, not a rigorous error bound.'}

def outer_receipt(reference, *, module, arguments, inner=None):
    value = read_json(reference['path'])
    require(digest(read(reference['path'])) == reference['sha256'], 'Outer receipt link changed.')
    require(type(value.get('exit_code')) is int and value['exit_code'] == 0
            and value['seconds'] >= 0 and 'launcher_error' not in value, 'No actual successful outer exit.')
    command = value['command']
    if command[1:3] == ['-B', '-m']:
        require(command[3] == module, 'Outer module differs.')
        tail = command[4:]
    else:
        require(command[1] == '-B' and inside(command[2]) == inside(module.replace('.', '/')+'.py'),
                'Outer script differs.')
        tail = command[3:]
    require(tail == arguments, 'Outer arguments differ from the selected stage.')
    if inner is not None and 'executable' in inner:
        require(Path(command[0]).resolve() == Path(inner['executable']).resolve(), 'Outer interpreter differs.')
    require(digest(read(value['log'])) == value['log_sha256'], 'Outer raw log changed.')
    require(value.get('source_before') == value.get('source_after') == maps()['wavebit'],
            'Outer production source inventory changed or differs.')
    require(value.get('complete') is True and 'failure' not in value, 'Outer wrapper did not complete cleanly.')
    receipt_folder = inside(reference['path']).parent
    require(digest(read(receipt_folder/'runner.py.raw')) == value['runner_sha256'], 'Outer wrapper source snapshot differs.')
    environment = value.get('environment', {})
    require(not environment.get('XLA_FLAGS') and environment['PYTHONUTF8'] == '1'
            and inside(environment['PYTHONPATH']) == ROOT, 'Outer environment differs.')
    stamp = lambda v: datetime.fromisoformat(v)
    start, end = stamp(value['started_utc']), stamp(value['ended_utc'])
    require(start.tzinfo is not None and end.tzinfo is not None and start <= end, 'Invalid outer chronology.')
    if inner is not None:
        require(start <= stamp(inner['started_utc']) <= stamp(inner['ended_utc']) <= end,
                'Outer/inner chronology differs.')
    return value


def audit_outer_exits(targeted, validation, docs, calibration_value, physical, parents_path):
    refs = read_json(parents_path)
    require(set(refs) == {'calibration', 'targeted', 'mutations', 'registration', 'campaign',
                         'campaign_audit', 'prepare', 'validation', 'documentation'}, 'Parent receipt roster differs.')
    stages = {}
    for stage, value in (('targeted', targeted), ('validation', validation)):
        stages[stage] = outer_receipt(refs[stage], module='studies.'+PREFIX+'gates',
            arguments=[stage, '--attempt', str(value['attempt']), '--workers', str(value['workers'])], inner=value)
    stages['calibration'] = outer_receipt(refs['calibration'], module='studies.'+PREFIX+'calibration',
        arguments=['--attempt', str(calibration_value['attempt'])], inner=calibration_value)
    mutation = read_json(RESULTS/(PREFIX+'mutations.json'))
    stages['mutations'] = outer_receipt(refs['mutations'], module='studies.'+PREFIX+'mutations',
        arguments=['--attempt', str(mutation['attempt'])], inner=mutation)
    sphinx = checks_by_name(docs)['sphinx']['command'][0]
    doc_parent = read_json(refs['documentation']['path'])
    doc_command = doc_parent['command']
    doc_tail = doc_command[4:] if doc_command[1:3] == ['-B', '-m'] else doc_command[3:]
    require(len(doc_tail) == 5 and doc_tail[:4] == ['documentation', '--attempt', str(docs['attempt']), '--sphinx']
            and Path(doc_tail[4]).resolve() == Path(sphinx).resolve(), 'Sphinx parent executable differs.')
    stages['documentation'] = outer_receipt(refs['documentation'], module='studies.'+PREFIX+'gates',
        arguments=doc_tail, inner=docs)
    reg = read_json(RESULTS/(PREFIX+'registration.json'))
    report = read_json(RESULTS/(PREFIX+'report.json'))
    for stage, action in (('registration', 'register'), ('campaign', 'run'), ('campaign_audit', 'audit')):
        stages[stage] = outer_receipt(refs[stage], module='studies.'+PREFIX+'campaign', arguments=[action])
    from studies.fdtd_oscillator_stack_gates import trailing_json
    require(trailing_json(read(stages['registration']['log']).decode())['fingerprint'] == reg['fingerprint'],
            'Registration outer response differs.')
    require(campaign_output_matches(trailing_json(read(stages['campaign']['log']).decode()),
            {'complete': True, 'stop_reason': None, 'workers': report['worker_rows'],
             'report': relative(RESULTS/(PREFIX+'report.json'))}),
            'Campaign outer response differs.')
    observed = trailing_json(read(stages['campaign_audit']['log']).decode())
    require(observed['audit'] == 'passed' and observed['complete'] is True and observed['stop_reason'] is None
            and observed['all_48_cells_recomputed'] is True and observed['workers'] == physical['workers'],
            'Controller saved-data audit failed.')
    for stored, replay in (('self_counts', 'self'), ('all_direct_counts', 'direct'), ('direct_final_counts', 'final_direct')):
        require(observed[stored] == physical['classifications'][replay], 'Controller classification differs.')
    for family, identical in physical['known_calibration_payload_identical'].items():
        require(observed['known_calibration_repeats'][family]['available']
                and observed['known_calibration_repeats'][family]['payload_identical'] == identical,
                'Known calibration comparison differs.')
    setup = attempt('prevalidation')
    campaign_ref, audit_ref = (setup['parents'][k]['reference'] for k in ('campaign', 'audit'))
    require(campaign_ref == refs['campaign'] and audit_ref == refs['campaign_audit'], 'Prepared parent links differ.')
    # Parent command preserves the provided absolute/relative paths literally.
    prepare_value = read_json(refs['prepare']['path'])
    tail = prepare_value['command'][4:] if prepare_value['command'][1:3] == ['-B', '-m'] else prepare_value['command'][3:]
    require(len(tail) == 7 and tail[:3] == ['prepare', '--attempt', str(setup['attempt'])]
            and tail[3] == '--campaign-parent' and inside(tail[4]) == inside(campaign_ref['path'])
            and tail[5] == '--audit-parent' and inside(tail[6]) == inside(audit_ref['path']), 'Prepare parents differ.')
    stages['prepare'] = outer_receipt(refs['prepare'], module='studies.'+PREFIX+'gates', arguments=tail, inner=setup)
    stamp = lambda v: datetime.fromisoformat(v)
    sequence = ['calibration', 'targeted', 'mutations', 'registration', 'campaign', 'campaign_audit',
                'prepare', 'validation', 'documentation']
    require(all(stamp(stages[a]['ended_utc']) <= stamp(stages[b]['started_utc'])
                for a, b in zip(sequence, sequence[1:])), 'Overlapping/out-of-order accepted stages.')
    require(stamp(stages['registration']['started_utc']) <= stamp(reg['registered_utc'])
            <= stamp(stages['registration']['ended_utc']), 'Registration outside parent interval.')
    return {k: {'receipt': refs[k], 'exit_code': v['exit_code'], 'log': v['log'],
                'started_utc': v['started_utc'], 'ended_utc': v['ended_utc']} for k, v in stages.items()}


def preserve_attempts(number, open_parent):
    """Bind every retained attempt; exclude only this explicitly named open launch."""
    control = RESULTS/(PREFIX+'controls')
    opened = inside(open_parent)
    require(opened.parent == control and opened.is_dir(), 'Open audit parent is not an exact launch directory.')
    started = read_json(opened/'started.json')
    command = started['command']
    if command[1:3] == ['-B', '-m']:
        require(command[3] == 'studies.'+PREFIX+'audit', 'Excluded launch is another module.')
    else:
        require(command[1] == '-B' and inside(command[2]) == ROOT/'studies'/(PREFIX+'audit.py'),
                'Excluded launch is another script.')
    require('--attempt' in command and command[command.index('--attempt')+1] == str(number),
            'Excluded launch belongs to another audit attempt.')
    records = {}
    for path in sorted(RESULTS.glob(PREFIX+'*')):
        if path.name in {PREFIX+'final_evidence_audit.json', PREFIX+f'final_evidence_audit_attempt{number}.json'}:
            continue
        for file in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if not file.is_file() or file.is_relative_to(opened):
                continue
            raw = read(file)
            if file.name == 'started.json' and file.parent.parent == control:
                require((file.parent/'exit.json').is_file(), 'Another launch has no actual completion receipt.')
            if file.name == 'exit.json' and file.parent.parent == control:
                receipt = json.loads(raw)
                require(type(receipt.get('exit_code')) is int and 'ended_utc' in receipt,
                        'A retained launch has no actual child exit.')
                require(digest(read(receipt['log'])) == receipt['log_sha256'], 'Prior launch output changed.')
                records[relative(file)] = {'exit_code': receipt['exit_code'], 'log': receipt['log']}
            require(file.suffix != '.txt', 'New copied-source/text archives require .raw/.gz: '+relative(file))
    return records


def main(number, parents_path, open_parent):
    require(number > 0, 'Positive audit attempt required.')
    destination = RESULTS/f'{PREFIX}final_evidence_audit_attempt{number}.json'
    result = {'base': BASE, 'started_utc': now(), 'complete': False, 'checks': {}}
    write(destination, result, exclusive=True)
    try:
        require(re.fullmatch('[0-9a-f]{40}', BASE), 'Full predecessor commit required.')
        before, setup = audit_baseline()
        previous = predecessor_rosters(before)
        targeted = attempt('targeted')
        new = command_pair(targeted, 'new_stack', [NEW_TEST], workers=targeted['workers'])
        require(new['totals']['passed'] == new['collected'] and new['collected'] > 0,
                'New actual case roster did not all pass.')
        target = command_pair(targeted, 'targeted', list(TARGETS), extra=('-m', 'not slow'), workers=targeted['workers'])
        require(set(new['node_status']) <= set(target['node_status'])
                and target['totals']['passed'] == target['collected'], 'Targeted baseline failed/omits new cases.')
        require(set(checks_by_name(targeted)) == {'new_stack_collection', 'new_stack',
                'targeted_collection', 'targeted'}, 'Unexpected targeted commands.')
        mutation = audit_mutations(new, targeted)
        acceptance, calibration_value, known = calibration()
        physical = audit_campaign(acceptance, calibration_value, known, targeted, mutation)
        validation = attempt('validation')
        files = sorted(setup['maps']['tests'])
        full = command_pair(validation, 'full', files+['tests/test_docs_contract.py'],
                            extra=('-m', 'not slow'), workers=validation['workers'])
        slow = command_pair(validation, 'slow', files, extra=['--run-slow', '-m', 'slow'], workers=validation['workers'])
        old_nodes = previous['full']['node_status']
        docs_node = 'tests/test_docs_contract.py::test_every_target_a_document_names_exists[fd-oscillator-stack.md]'
        require(not set(old_nodes).intersection(new['node_status']) and docs_node not in old_nodes,
                'New node collides with predecessor.')
        require(set(full['node_status']) == set(old_nodes)|set(new['node_status'])|{docs_node},
                'Full per-node roster differs from old plus actual new cases plus the new document.')
        require({n for n, s in full['node_status'].items() if s == 'skipped'}
                == {n for n, s in old_nodes.items() if s == 'skipped'}, 'Historical eight skip IDs differ.')
        require(slow == previous['slow'], 'Existing slow roster or outcomes changed.')
        checks = checks_by_name(validation)
        require(set(checks) == {'full_collection', 'full', 'slow_collection', 'slow',
                'freezer0', 'freezer1', 'freezer2', 'freezer3'}, 'Unexpected regression command.')
        for stage, count in enumerate((143, 288, 140, 570)):
            check = checks['freezer'+str(stage)]
            require(check['command'] == [validation['executable'], '-B',
                    f'tests/references/interface/freeze_fd_stage{stage}.py'], 'Unexpected freezer invocation.')
            output = read(check['log']).decode()
            require(check['exit'] == 0 and check['arrays_exact'] == count
                    and f'{count} of {count} arrays reproduce bit for bit' in output
                    and 'the committed golden is reproduced; nothing written' in output,
                    'Read-only freezer failed or rewrote a golden.')
        docs = attempt('documentation')
        require(docs['docs_before'] == docs['docs_after'] == docs_map(), 'Final documentation sources changed.')
        for name, expected in docs['docs_after'].items():
            require(digest(read(name)) == expected, 'Final document hash changed.')
        docs_observed = command_pair(docs, 'docs', ['tests/test_docs_contract.py'])
        require(docs_observed['collected'] == docs_observed['totals']['passed']
                and set(docs_observed['node_status']) ==
                {n for n in full['node_status'] if n.startswith('tests/test_docs_contract.py::')},
                'Final docs roster/outcomes differ.')
        checks = checks_by_name(docs)
        require(set(checks) == {'docs_collection', 'docs', 'sphinx'}, 'Unexpected final documentation command.')
        sphinx = checks['sphinx']
        require(sphinx['exit'] == 0 and len(sphinx['command']) == 6
                and sphinx['command'][1:-1] == ['-W', '-b', 'html', 'docs'],
                'Sphinx -W command failed/differs.')
        scratch_path(sphinx['command'][-1])
        output = read(sphinx['log']).decode()
        require('build succeeded.' in output and not re.search(r'(?im)^.*(?:WARNING:|ERROR:)', output),
                'Sphinx output did not complete cleanly.')
        outer = audit_outer_exits(targeted, validation, docs, calibration_value, physical, parents_path)
        preserved = preserve_attempts(number, open_parent)
        result['checks'] = {'new': new, 'targeted': target, 'mutations': mutation,
            'physical': physical, 'full': full, 'slow': slow, 'docs': docs_observed,
            'arrays_bit_exact': 1141, 'outer_exits': outer, 'preserved_attempt_receipts': preserved,
            'raw_map_counts': {k: len(v) for k, v in setup['maps'].items()},
            'actual_new_cases': new['collected']}
        for name, expected in INPUTS.items():
            require(sha(inside(name)) == expected, 'Audit input changed during replay: '+name)
        result.update(complete=True, ended_utc=now(), input_sha256=INPUTS,
            qualification='Saved evidence replay only; no solver, project numerical import or pytest run. '
            'Scalar optics verifies the recorded reference; all per-cell verdicts remain visible, including '
            'mixed/low/unresolved outcomes. Known calibration repeats are not independent corroboration. '
            'No arbitrary-stack, general C8 or C7 completion claim. The caller separately records this '
            'auditor process exit; its own open launch stdout/receipt is intentionally excluded.')
        write(destination, result)
        write(RESULTS/(PREFIX+'final_evidence_audit.json'), result, exclusive=True)
        print('Saved-evidence audit passed; caller must record actual process completion.', flush=True)
    except BaseException as error:
        result.update(error=repr(error), ended_utc=now(), input_sha256=INPUTS)
        write(destination, result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    parser.add_argument('--parents', required=True)
    parser.add_argument('--open-parent', required=True)
    args = parser.parse_args()
    main(args.attempt, args.parents, args.open_parent)
