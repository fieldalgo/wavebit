"""Prospective 60-solve finite-guide characterization, never a reflection/order gate.

Draft only: the full predecessor and three root-owned prerequisite interfaces must
be reviewed before first execution. Registration precedes every campaign response;
all six known four-solve calibration rows and every failed development attempt are disclosed.
"""
import argparse
import ast
from datetime import datetime
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import traceback
import zipfile

from fdtd_constitutive_absorber_support import (
    ROOT, RESULTS, PREFIX, TEST, guard, identity, now, relative, require_base, run_worker, sha, write)
from fdtd_constitutive_absorber_calibration import config, freeze, read_worker, require

REGISTRATION = RESULTS/(PREFIX+'_registration.json')
FAMILIES = ('cylinder', 'rotated-box')
BACKENDS = ('numpy', 'jax', 'torch')


def load_ref(item):
    path = (ROOT/item['path']).resolve()
    require(path.is_relative_to(ROOT.resolve()) and sha(path) == item['sha256'], 'Input reference/hash differs.')
    return json.loads(path.read_text())


def ref(path):
    return {'path': relative(path), 'sha256': sha(path)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def jobs():
    out = []
    for family, n, placement, polarization, length in itertools.product(
            FAMILIES, (20, 30, 40), (0., .3), ('u', 'v'), ('short', 'reference')):
        out.append({'kind': 'base', 'config': config(family, 'numpy', n=n, placement=placement,
                    polarization=polarization, length=length)})
    for family in FAMILIES:
        for length, time in (('short', 32.), ('reference', 32.), ('farther', 16.), ('farther', 32.)):
            out.append({'kind': 'designated-control', 'config': config(family, 'numpy', n=30,
                        length=length, time=time)})
    for family, homogeneous in itertools.product(FAMILIES, (False, True)):
        out.append({'kind': 'zero-legacy' if homogeneous else 'zero-shaped',
                    'config': config(family, 'numpy', n=30, contrast=False, homogeneous=homogeneous)})
    require(len(out) == 60 and len({canonical(v['config']) for v in out}) == 60,
            'Exactly60 unique fixed solves required.')
    return [{'index': index, **row} for index, row in enumerate(out)]


def protocol():
    return {'sequence': jobs(), 'no_replacements': True, 'no_response_based_extension': True,
            'backends': {'calibration': list(BACKENDS), 'campaign': ['numpy']},
            'rungs': [20, 30, 40], 'placements': [0., .3], 'polarizations': ['u', 'v'],
            'absorber_cells': {'20': 8, '30': 12, '40': 16}, 'physical_absorber_thickness': .4,
            'lengths': {'short': 3.2, 'reference': 6.4, 'farther': 9.6},
            'base_time': 16., 'control_time': 32., 'wavelengths': [.9, 1., 1.1],
            'trace_windows': [[0., 4.], [4., 8.], [8., 12.], [12., 16.]],
            'designated_controls': {'n': 30, 'placement': .3, 'polarization': 'u',
                'qualification': 'Controls quantify only these two configurations; no transfer to all other rows.'},
            'zero_controls': 'shortT16 n30,f.3,u: retained eps1 body versus omitted body; fields/source only, not energy equality',
            'reference_statement': 'Longer finite domains are comparisons, never exact/open-domain truth.',
            'numerical_acceptance': 'All mapping/path/finite/source guards; measured physical_alignment budget for shared-grid and zero-field parity only.',
            'classification': 'Every termination/time/length sensitivity is reported per channel/window/wavelength; no small-error, monotonicity or order pass gate.',
            'stop': 'Stop on the first failed worker, source/guard change, mapping or numerical-instrument assertion; preserve all completed/failed outputs.'}


def contained(name):
    path = (ROOT/name).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Evidence path escapes checkout.')
    return path


def same_path(left, right):
    return (ROOT/left).resolve() == (ROOT/right).resolve()


def timestamps(*values):
    stamps = [datetime.fromisoformat(t) for t in values]
    require(all(t.tzinfo is not None for t in stamps)
            and all(a <= b for a, b in zip(stamps, stamps[1:])), 'Evidence chronology differs.')


def parent_success(item, earliest, latest, helper, arguments):
    record = load_ref(item)
    require(record['exit_code'] == 0 and record['seconds'] >= 0, 'Prerequisite parent did not observe success.')
    require(sha(contained(record['log'])) == record['log_sha256'], 'Parent log changed.')
    timestamps(record['started_utc'], earliest, latest, record['ended_utc'])
    command = record['command']
    require(len(command) == 3 + len(arguments) and command[1] == '-B'
            and same_path(command[0], sys.executable)
            and same_path(command[2], 'studies/'+PREFIX+'_'+helper+'.py')
            and command[3:] == arguments, 'Wrong prerequisite launch command.')
    require(not record.get('environment', {}).get('XLA_FLAGS'), 'Unreviewed XLA flags.')
    return record


def verify_zip(item, expected):
    path = contained(item['path'])
    require(sha(path) == item['sha256'], 'Source ZIP hash differs.')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(expected), 'Source ZIP inventory differs.')
        require(all(hashlib.sha256(archive.read(k)).hexdigest() == v for k, v in expected.items()),
                'Source ZIP member bytes differ.')


def archive_bytes(item):
    path = contained(item['file'])
    require(sha(path) == item['archive_sha256'], 'Compressed source hash differs.')
    raw = gzip.decompress(path.read_bytes())
    require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'Decoded source hash differs.')
    return raw


def replay_calibration(calibration):
    from fdtd_constitutive_absorber_probe import compare_pair, difference, zero_comparison, preflight
    schedule = []
    for family, backend in itertools.product(FAMILIES, BACKENDS):
        schedule.extend([config(family, backend), config(family, backend, length='reference'),
                         config(family, backend, contrast=False),
                         config(family, backend, contrast=False, homogeneous=True)])
    require(calibration['schedule'] == schedule and len(calibration['workers']) == 24,
            'The disclosed calibration must contain exactly six fixed four-solve rows.')
    require(calibration['ceilings'] is None and calibration['analytic_preflight'] == preflight(),
            'Calibration must disclose data before budget acceptance.')
    verify_zip(calibration['source_archive'], calibration['source_before'])
    loaded, previous = [], calibration['started_utc']
    for row, cfg in zip(calibration['workers'], schedule):
        result, arrays = read_worker(row, cfg, calibration['source_before'])
        timestamps(previous, row['started_utc'], result['started_utc'], result['ended_utc'], row['ended_utc'])
        previous = row['ended_utc']
        loaded.append((result, arrays))
    timestamps(previous, calibration['ended_utc'])
    pairs = []
    for index in range(0, 24, 4):
        (short, a), (reference, b), (shaped, c), (legacy, d) = loaded[index:index+4]
        comparison = compare_pair(short, reference, a, b)
        zero = zero_comparison(shaped, legacy, c, d)
        require(all(comparison['mapping_checks'].values()) and all(zero['checks'].values()),
                'Calibration mapping/zero-route structure does not replay.')
        pairs.append({'family': short['config']['family'], 'backend': short['config']['backend'],
                      'worker_indices': list(range(index, index+4)),
                      'comparison': comparison, 'zero_comparison': zero})
    comparisons = []
    for family in FAMILIES:
        for role, length, contrast, homogeneous in (
                ('short', 'short', True, False), ('reference', 'reference', True, False),
                ('zero-shaped', 'short', False, False), ('zero-legacy', 'short', False, True)):
            selected = {r['config']['backend']: a for r, a in loaded
                        if r['config']['family'] == family and r['config']['length'] == length
                        and r['config']['contrast'] == contrast and r['config']['homogeneous'] == homogeneous}
            for backend in ('jax', 'torch'):
                require(set(selected[backend]) == set(selected['numpy']), 'Calibration backend roster differs.')
                comparisons.append({'family': family, 'role': role, 'length': length, 'backend': backend,
                    'arrays': {key: difference(selected[backend][key], selected['numpy'][key])
                               for key in selected['numpy']}})
    require(calibration['pairs'] == pairs and calibration['backend_comparisons'] == comparisons,
            'Calibration summaries differ from saved arrays.')
    metrics = [v for row in comparisons for v in row['arrays'].values()]
    alignment = [v for pair in pairs for key in ('coordinate_errors', 'shared_lowering', 'plane_coordinates')
                 for v in pair['comparison'][key].values()]
    require(all(v['same_shape'] and v['finite'] for v in metrics+alignment),
            'Calibration silently omitted an invalid numerical comparison.')
    expected = {'max_backend_scaled': max(v['roundoff_scaled'] for v in metrics),
                'max_shared_alignment_scaled': max(v['roundoff_scaled'] for v in alignment),
                'max_zero_scaled': max(pair['zero_comparison']['max_roundoff_scaled'] for pair in pairs)}
    require(all(calibration[key] == value for key, value in expected.items()),
            'Calibration maxima differ from raw replay.')


def literal_assignment(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    found = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    require(len(found) == 1, 'Expected one frozen literal assignment: '+name)
    return ast.literal_eval(found[0])


def replay_algebra(algebra, item, calibrated_source):
    import numpy as np
    folder = contained(item['path']).parent
    before = json.loads((folder/'before.json').read_text())
    require(before['before'] == algebra['guard_before'] and all(
                calibrated_source.get(k) == v for k, v in before['before'].items()),
            'Independent algebra source map differs from calibration.')
    require(before['base'] == algebra['base'] and before['branch'] == algebra['branch']
            and before['started_utc'] == algebra['started_utc'], 'Algebra source receipt differs.')
    verify_zip({'path': relative(folder/before['sources_archive']),
                'sha256': before['sources_sha256']}, before['before'])
    raw = gzip.decompress((folder/'test.py.raw.gz').read_bytes())
    require(hashlib.sha256(raw).hexdigest() == calibrated_source[TEST], 'Algebra test snapshot differs.')
    fixed = literal_assignment(ROOT/'studies'/(PREFIX+'_algebra.py'), 'SCHEDULE')
    schedule = [{'function': name, 'arguments': dict(zip(names, values))}
                for name, names, choices in fixed for values in itertools.product(*choices)]
    require(len(schedule) == 81 and algebra['schedule'] == before['schedule'] == schedule
            and len(algebra['rows']) == len(schedule), 'Algebra instrument inventory differs.')
    descriptor = algebra['arrays']
    payload = folder/descriptor['file']
    require(sha(payload) == descriptor['sha256'], 'Algebra raw comparisons changed.')
    with np.load(payload, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), 'Duplicate algebra array names.')
        arrays = {k: archive[k] for k in archive.files}
    require(len(arrays) == descriptor['members'] == 2*len(algebra['metrics']), 'Algebra array count differs.')
    used, values, cursor, previous = set(), {'algebra': [], 'step': []}, 0, algebra['started_utc']
    for index, (expected, row) in enumerate(zip(schedule, algebra['rows'])):
        require(row['index'] == index and {k: row[k] for k in expected} == expected
                and row['passed'] is True and row['checks'] > 0 and row['first_check'] == cursor,
                'Algebra row or check boundaries differ.')
        timestamps(previous, row['started_utc'], row['ended_utc'])
        previous = row['ended_utc']
        for number in range(cursor, cursor+row['checks']):
            metric = algebra['metrics'][number]
            require(metric['check_index'] == number and metric['row_index'] == index
                    and metric['kind'] in values, 'Algebra comparison row binding differs.')
            stem = 'check'+str(number).zfill(5)
            require(metric['actual'] == stem+'_actual' and metric['expected'] == stem+'_expected',
                    'Algebra raw member binding differs.')
            a, b = arrays[metric['actual']], arrays[metric['expected']]
            used.update((metric['actual'], metric['expected']))
            require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                    and metric['finite'] and metric['same_shape'], 'Invalid algebra comparison.')
            absolute = float(np.max(np.abs(a-b), initial=0.0))
            scale = max(1.0, float(np.max(np.abs(b), initial=0.0)))
            expected_metric = {'absolute': absolute, 'scale': scale, 'scaled': absolute/scale,
                'shape_actual': list(a.shape), 'shape_expected': list(b.shape),
                'dtype_actual': str(a.dtype), 'dtype_expected': str(b.dtype),
                'normalization': ('absolute residual; zero reference, scale=1' if not np.any(b)
                    else 'absolute error / max(1, reference component magnitude)')}
            require(all(metric[k] == v for k, v in expected_metric.items()), 'Algebra metric differs from raw arrays.')
            values[metric['kind']].append(absolute/scale)
        cursor += row['checks']
    timestamps(previous, algebra['ended_utc'])
    require(cursor == len(algebra['metrics']) and used == set(arrays)
            and algebra['max_scaled'] == {k: max(v) for k, v in values.items()},
            'Algebra maximum or complete raw inventory differs.')


def replay_gates(accepted, current):
    from fdtd_constitutive_absorber_gate_support import TARGETS, SCRATCH, audit_pytest, nodes, reports, summary, check_log
    targeted, mutation = load_ref(accepted['targeted']), load_ref(accepted['mutations'])
    for kind, record in (('targeted', targeted), ('mutations', mutation)):
        require(record['complete'] and record['passed'] and record['guard_before'] == record['guard_after'] == current
                and record['source_before'] == record['source_after'] == current['source'],
                'Pending or differently frozen gate: '+kind)
        arguments = (['targeted', '--attempt', str(record['attempt'])] if kind == 'targeted'
                     else ['--attempt', str(record['attempt'])])
        parent_success(accepted[kind+'_parent'], record['started_utc'], record['ended_utc'],
                       'gates' if kind == 'targeted' else 'mutations', arguments)
        snapshots = record['runner_snapshots']
        expected = {k for k in current['source'] if k.startswith('studies/'+PREFIX+'_') and k.endswith('.py')}
        require(len(snapshots) == len(expected) and {v['source'] for v in snapshots} == expected,
                'Gate helper snapshot inventory differs.')
        for item in snapshots:
            raw = archive_bytes(item)
            require(raw == contained(item['source']).read_bytes(), 'Frozen gate helper bytes differ.')
    caches = set()
    def row_text(row, start, end):
        require(row['seconds'] >= 0 and sha(contained(row['log'])) == row['sha256'], 'Gate raw output differs.')
        timestamps(start, row['started_utc'], row['ended_utc'], end)
        command = row['command']
        require(len(command) > 4 and same_path(command[0], sys.executable)
                and command[1:4] == ['-B', '-m', 'pytest'], 'Gate did not execute the expected Python pytest.')
        environment = row['environment']
        cache = (ROOT/environment['PYTHONPYCACHEPREFIX']).resolve()
        require(cache.is_relative_to(SCRATCH.resolve()) and cache not in caches
                and not environment['XLA_FLAGS'], 'Unowned/reused cache or numerical flag.')
        caches.add(cache)
        text = contained(row['log']).read_text(encoding='utf-8')
        check_log(text)
        return text
    def pair(collection, run, targets, extra, workers, start, end):
        require(collection['exit'] == run['exit'] == 0, 'Baseline pytest exit is not successful.')
        left, right = row_text(collection, start, end), row_text(run, start, end)
        base = ['-B', '-m', 'pytest', '-p', 'no:cacheprovider', *targets, *extra]
        require(collection['command'][1:] == base+['-q', '--collect-only'], 'Collection arguments differ.')
        actual = run['command'][1:]
        tail = ['-n', str(workers)] if workers else []
        require(actual[:len(base)] == base and actual[len(base):len(base)+3] == ['-vv', '--tb=short', '--basetemp']
                and actual[len(base)+4:] == tail, 'Run arguments differ.')
        temp = (ROOT/actual[len(base)+3]).resolve()
        require(temp.is_relative_to(SCRATCH.resolve()), 'Unowned pytest basetemp.')
        timestamps(collection['ended_utc'], run['started_utc'])
        observed = audit_pytest(left, right)
        require(observed == run['observed'] and observed['totals']['passed'] == observed['collected']
                and sum(observed['totals'].values()) == observed['collected'], 'Baseline raw node outcome differs.')
        return observed
    rows = targeted['checks']
    require([r['name'] for r in rows] == ['new_constitutive_absorber_collection', 'new_constitutive_absorber',
            'targeted_collection', 'targeted'], 'Targeted gate sequence differs.')
    baseline = pair(rows[0], rows[1], [TEST], [], 0, targeted['started_utc'], targeted['ended_utc'])
    pair(rows[2], rows[3], list(TARGETS), ['-m', 'not slow'], 5, targeted['started_utc'], targeted['ended_utc'])
    timestamps(rows[1]['ended_utc'], rows[2]['started_utc'], targeted['ended_utc'], mutation['started_utc'])
    require(mutation['targeted_manifest_sha256'] == accepted['targeted']['sha256']
            and same_path(mutation['targeted_manifest'], accepted['targeted']['path'])
            and same_path(mutation['baseline_log'], rows[1]['log'])
            and mutation['baseline_log_sha256'] == rows[1]['sha256'], 'Mutation baseline links differ.')
    harness = ROOT/'studies'/(PREFIX+'_mutations.py')
    require(mutation['harness_sha256'] == sha(harness) and mutation['test_file_sha256'] == {TEST: sha(ROOT/TEST)}
            and archive_bytes(mutation['test_snapshot']) == (ROOT/TEST).read_bytes(), 'Mutation test/harness differs.')
    specs = literal_assignment(harness, 'SPECS')
    plugin = literal_assignment(harness, 'COLLECTION_PLUGIN')
    require(archive_bytes(mutation['collection_plugin']) == plugin.encode(), 'Collection plugin snapshot differs.')
    functions = sorted(n.name for n in ast.parse((ROOT/TEST).read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name.startswith('test_'))
    require(len(specs) == len({s['name'] for s in specs}) == mutation['expected_mechanisms']
            and functions == mutation['expected_functions']
            and set(functions) == {s['test_function'] for s in specs}, 'Actual function/mechanism inventory differs.')
    checks = mutation['checks']
    require([r['name'] for r in checks] == ['mutation_collection']+['witness']*len(specs)
            +['restored_baseline_collection', 'restored_baseline'], 'Mutation command sequence differs.')
    collection = mutation['collection']
    require(all(collection[k] == v for k, v in checks[0].items()), 'Mutation collection receipt differs.')
    text = row_text(checks[0], mutation['started_utc'], mutation['ended_utc'])
    require(checks[0]['exit'] == 0 and checks[0]['command'][1:] == [
        '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-p', 'c9_collection_inventory', TEST, '--collect-only'],
        'Mutation collection command differs.')
    require(sha(contained(collection['nodes'])) == collection['nodes_sha256'], 'Actual callspec inventory changed.')
    descriptors = json.loads(contained(collection['nodes']).read_text())
    require([d['nodeid'] for d in descriptors] == nodes(text)
            and set(nodes(text)) == set(baseline['node_status']), 'Mutation callspec/node inventories differ.')
    require(len(mutation['witnesses']) == len(specs) and len(mutation['bound_nodes']) == len(specs),
            'Missing source mutation witness.')
    previous = checks[0]['ended_utc']
    for spec, witness, check, bound in zip(specs, mutation['witnesses'], checks[1:-2], mutation['bound_nodes']):
        selected = [d['nodeid'] for d in descriptors if d['function'] == spec['test_function']
                    and d['parameters'] == spec['case_parameters']]
        require(len(selected) == 1, 'Mutation does not select one actual parameterized test.')
        node = selected[0]
        require(bound == {'name': spec['name'], 'node': node, 'parameters': spec['case_parameters']}
                and all(witness[k] == v for k, v in {'name': spec['name'], 'production_file': spec['production_file'],
                    'test': node, 'function': spec['test_function'], 'parameters': spec['case_parameters'],
                    'replacement': spec['replacements']}.items()), 'Witness differs from the frozen source plan.')
        original, mutant = archive_bytes(witness['original_snapshot']), archive_bytes(witness['mutant_snapshot'])
        source = contained(spec['production_file'])
        require(original == source.read_bytes() and hashlib.sha256(original).hexdigest() == spec['source_raw_sha256'],
                'Actual source is not exactly restored to reviewed bytes.')
        crlf = b'\r\n' in original
        require(not crlf or b'\n' not in original.replace(b'\r\n', b''), 'Mixed original source EOL.')
        changed = original.decode().replace('\r\n', '\n')
        for hunk in spec['replacements']:
            require(changed.count(hunk['before']) == 1, 'Mutation anchor is absent or ambiguous.')
            changed = changed.replace(hunk['before'], hunk['after'], 1)
        ast.parse(changed)
        if crlf:
            changed = changed.replace('\n', '\r\n')
        require(mutant == changed.encode() and mutant != original, 'Actual mutant is not the exact reviewed replacement.')
        expected_source = dict(current['source'])
        expected_source[spec['production_file']] = hashlib.sha256(mutant).hexdigest()
        require(witness['source_before'] == witness['source_restored'] == current['source']
                and witness['source_mutated'] == expected_source and witness['restored'],
                'Mutation changed unrelated source or did not restore it.')
        require(witness['execution'] == check and witness['command'] == check['command']
                and witness['raw_log'] == check['log'] and witness['raw_log_sha256'] == check['sha256']
                and witness['exit_code'] == check['exit'] == 1, 'Mutant child/raw linkage differs.')
        command = check['command']
        require(len(command) == 11 and command[1:9] == [
            '-B', '-m', 'pytest', '-vv', '-p', 'no:cacheprovider', '--tb=short', '--basetemp']
            and command[-1] == node and (ROOT/command[9]).resolve().is_relative_to(SCRATCH.resolve()),
            'Mutant command does not target the exact node in owned scratch.')
        timestamps(previous, witness['started_utc'], check['started_utc'], check['ended_utc'], witness['ended_utc'])
        previous = witness['ended_utc']
        raw = row_text(check, mutation['started_utc'], mutation['ended_utc'])
        statuses, totals = reports(raw), summary(raw)
        errors = [line for line in raw.splitlines() if re.match(r'^E\s+', line)]
        forbidden = r'\b(?:ValueError|TypeError|RuntimeError|ImportError|ModuleNotFoundError|NameError|KeyError|AttributeError|IndexError|SyntaxError|OSError|MemoryError|SystemExit|ZeroDivisionError)\b'
        require(statuses == {node: 'failed'} and totals['failed'] == 1 and sum(totals.values()) == 1
                and any('AssertionError' in line or 'assert ' in line or 'Failed: DID NOT RAISE' in line for line in errors)
                and not any(re.search(forbidden, line) and 'Failed: DID NOT RAISE' not in line for line in errors),
                'Mutation failed only through runtime/import artifacts or a different test.')
        require(witness['accepted'] and witness['assertion_failure'] and witness['node_status'] == statuses
                and witness['totals'] == totals, 'Stored assertion acceptance does not match raw output.')
    timestamps(previous, checks[-2]['started_utc'])
    restored = pair(checks[-2], checks[-1], [TEST], [], 0, mutation['started_utc'], mutation['ended_utc'])
    require(mutation['final_baseline'] == checks[-1] and restored['node_status'] == baseline['node_status']
            and mutation['restored'] and mutation['all_function_assertion_witnesses'],
            'Restored complete baseline/function witness contract does not replay.')
    return targeted, mutation


def budget_edit(accepted, calibration):
    """Only one explicit ACCEPTED_BUDGETS assignment may change after calibration."""
    before_item, after_item = accepted['test_before'], accepted['test_after']
    before_path, after_path = ROOT/before_item['path'], ROOT/after_item['path']
    require(sha(before_path) == before_item['sha256'] and sha(after_path) == after_item['sha256'],
            'Budget-edit source snapshots differ.')
    before, after = before_path.read_bytes(), after_path.read_bytes()
    require(hashlib.sha256(before).hexdigest() == calibration['source_before'][TEST]
            and after == (ROOT/TEST).read_bytes(), 'Budget source endpoints differ.')
    left, right = ast.parse(before.decode()), ast.parse(after.decode())
    def assignment(tree):
        rows = [n for n in tree.body if isinstance(n, ast.Assign)
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == 'ACCEPTED_BUDGETS']
        require(len(rows) == 1, 'Expected one budget assignment.')
        return rows[0]
    old, new = assignment(left), assignment(right)
    require(ast.literal_eval(old.value) is None and ast.literal_eval(new.value) == accepted['budgets'],
            'Only measured accepted budget assignment is permitted.')
    new.value = ast.Constant(value=None)
    require(ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False),
            'Undisclosed test logic changed after calibration.')
    current = identity()
    changed = {p for p in set(current) | set(calibration['source_before'])
               if current.get(p) != calibration['source_before'].get(p)}
    require(changed == {TEST}, 'Non-budget numerical/helper source changed after calibration.')


def prerequisites(acceptance_path):
    accepted = json.loads(acceptance_path.read_text())
    require(accepted['decision'] == 'accepted' and accepted['reason'].strip(), 'Measured acceptance is missing.')
    budgets = accepted['budgets']
    require(set(budgets) == {'algebra', 'step', 'physical_alignment'}
            and all(math.isfinite(v) and v > 0 for v in budgets.values()), 'Unaccepted numerical budgets.')
    calibration = load_ref(accepted['calibration_manifest'])
    algebra = load_ref(accepted['algebra_manifest'])
    require(calibration['complete'] and calibration['exit_code'] == 0
            and calibration['source_before'] == calibration['source_after'], 'Calibration incomplete/source changed.')
    require(algebra['complete'] and algebra['passed'] and algebra['exit_code'] == 0
            and algebra['source_unchanged'], 'Independent algebra disclosure incomplete.')
    require(algebra['guard_before'] == algebra['guard_after']
            and all(calibration['source_before'].get(path) == expected
                    for path, expected in algebra['guard_before'].items()),
            'Algebra disclosure is not tied to the calibrated None-budget numerical source.')
    calibration_parent = parent_success(accepted['calibration_parent'], calibration['started_utc'], calibration['ended_utc'],
                   'calibration', ['--attempt', str(calibration['attempt'])])
    algebra_parent = parent_success(accepted['algebra_parent'], algebra['started_utc'], algebra['ended_utc'],
                   'algebra', ['--attempt', str(algebra['attempt']), '--expected-base', algebra['base'],
                               '--expected-branch', algebra['branch']])
    replay_calibration(calibration)
    replay_algebra(algebra, accepted['algebra_manifest'], calibration['source_before'])
    require(accepted['budgets']['physical_alignment'] >= max(
        calibration['max_backend_scaled'], calibration['max_shared_alignment_scaled'], calibration['max_zero_scaled']),
        'Accepted physical alignment allowance is below the disclosed measured error.')
    require(all(accepted['budgets'][k] >= algebra['max_scaled'][k] for k in ('algebra', 'step')),
            'Accepted algebra allowance is below the disclosed measured error.')
    budget_edit(accepted, calibration)
    current = guard()
    require(calibration['guard_before'] == calibration['guard_after'], 'Calibration preservation guard changed.')
    require(all(current[k] == calibration['guard_before'][k] for k in ('goldens', 'previous_evidence')),
            'Golden/prior records changed after calibration.')
    targeted, mutation = replay_gates(accepted, current)
    timestamps(calibration_parent['ended_utc'], accepted['accepted_utc'], targeted['started_utc'])
    timestamps(algebra_parent['ended_utc'], accepted['accepted_utc'], targeted['started_utc'])
    return accepted, calibration


def known_development():
    """Completed development inventory; never capture this registration's live stdout."""
    out = {}
    stems = tuple(PREFIX+'_'+p for p in ('calibration', 'algebra', 'targeted', 'mutations', 'structural', 'acceptance'))
    for path in sorted(RESULTS.rglob('*')):
        if not path.is_file():
            continue
        parts = path.relative_to(RESULTS).parts
        wanted = parts[0].startswith(stems)
        if parts[0] == PREFIX+'_controls':
            wanted = len(parts) > 1 and not any(x.startswith(('registration', 'campaign', 'final')) for x in parts[1:])
        if wanted:
            out[relative(path)] = sha(path)
    return out


def register(acceptance_path):
    base = require_base()
    accepted, calibration = prerequisites(acceptance_path)
    folder = RESULTS/(PREFIX+'_registered_sources')
    folder.mkdir(exist_ok=False)
    before, source_archive = freeze(folder)
    from fdtd_constitutive_absorber_probe import preflight
    payload = {'base': base, 'registered_utc': now(), 'protocol': protocol(), 'guard': guard(),
               'source_identity': before, 'source_archive': source_archive,
               'acceptance': ref(acceptance_path), 'accepted': accepted,
               'known_development': known_development(), 'analytic_preflight': preflight(),
               'known_calibration_workers': calibration['workers'],
               'calibration_scope': 'six family/CPU rows, four solves each,24 total; two positive domains and zero-shaped/legacy short',
               'qualification': 'The six four-solve development rows were already observed. All60 campaign rows are fixed before their new responses.'}
    require(payload['guard']['source'] == before, 'Source changed during registration.')
    payload['fingerprint'] = hashlib.sha256(canonical(payload)).hexdigest()
    write(REGISTRATION, payload)
    print(json.dumps({'registered': True, 'fingerprint': payload['fingerprint'], 'workers': 60}), flush=True)


def registered():
    item = json.loads(REGISTRATION.read_text())
    require(hashlib.sha256(canonical({k: v for k, v in item.items() if k != 'fingerprint'})).hexdigest()
            == item['fingerprint'], 'Registration fingerprint differs.')
    require(item['protocol'] == protocol() and item['guard'] == guard(), 'Registered protocol/guard differs.')
    for path, expected in item['known_development'].items():
        require(sha(ROOT/path) == expected, 'Known development artifact changed.')
    require(sha(ROOT/item['source_archive']['path']) == item['source_archive']['sha256'], 'Registered source archive changed.')
    return item


def check_ready_comparisons(records, registration):
    """Stop before the next solve whenever a now-complete instrument fails."""
    from fdtd_constitutive_absorber_probe import compare_pair, difference, zero_comparison
    ceiling = registration['accepted']['budgets']['physical_alignment']
    receipts = []
    def have(family, **kwargs):
        return records.get(canonical(config(family, 'numpy', **kwargs)))
    def aligned(left, right, label):
        if left is None or right is None:
            return
        a, x = left
        b, y = right
        pair = compare_pair(a, b, x, y)
        require(all(pair['mapping_checks'].values()), 'Shared-grid mapping failed: '+label)
        for group in ('coordinate_errors', 'shared_lowering', 'plane_coordinates'):
            require(all(v['same_shape'] and v['finite'] and v['roundoff_scaled'] <= ceiling
                        for v in pair[group].values()), 'Shared-grid instrument failed: '+label)
        receipts.append(label)
    for family, n, placement, polarization in itertools.product(FAMILIES, (20,30,40), (0.,.3), ('u','v')):
        options = dict(n=n, placement=placement, polarization=polarization)
        aligned(have(family, **options), have(family, length='reference', **options),
                repr((family,n,placement,polarization,'base')))
    for family in FAMILIES:
        for time in (16.,32.):
            aligned(have(family,n=30,length='reference',time=time),
                    have(family,n=30,length='farther',time=time), repr((family,time,'reference-length')))
        for length in ('short','reference','farther'):
            first, last = have(family,n=30,length=length), have(family,n=30,length=length,time=32.)
            if first is None or last is None:
                continue
            a, b = first[1], last[1]
            keys = [k for k in a if (k.endswith('_values') and k != 'energy_values')
                    or (k.startswith('source_') and k != 'source_stamp')]
            parity = {key:difference(a[key],b[key][:len(a[key])]) for key in keys}
            require(all(v['same_shape'] and v['finite'] and v['roundoff_scaled'] <= ceiling
                        for v in parity.values()), 'Same-domain fixed-prefix changed.')
            receipts.append(repr((family,length,'duration-prefix')))
        shaped, legacy = have(family,n=30,contrast=False), have(family,n=30,contrast=False,homogeneous=True)
        if shaped is not None and legacy is not None:
            zero = zero_comparison(shaped[0],legacy[0],shaped[1],legacy[1])
            require(all(zero['checks'].values()) and zero['max_roundoff_scaled'] <= ceiling,
                    'Zero-cross new/legacy parity failed.')
            receipts.append(repr((family,'zero-legacy')))
    return receipts


def run(attempt):
    require(attempt == 1, 'No automatic second campaign or response-based replacement is registered.')
    require_base()
    registration = registered()
    folder = RESULTS/f'{PREFIX}_campaign_attempt{attempt}'
    folder.mkdir(exist_ok=False)
    ledger = {'attempt': attempt, 'started_utc': now(), 'registration_sha256': sha(REGISTRATION),
              'fingerprint': registration['fingerprint'], 'workers': [], 'complete': False}
    write(folder/'before.json', ledger)
    loaded = {}
    try:
        for job in jobs():
            registered()
            execution = run_worker(job['config'], folder/f'worker{job["index"]:02d}')
            ledger['workers'].append({'job': job, 'execution': execution})
            write(folder/f'ledger{job["index"]:02d}.json', ledger)
            loaded[canonical(job['config'])] = read_worker(execution, job['config'], registration['source_identity'])
            comparison_start = now()
            comparisons = check_ready_comparisons(loaded, registration)
            comparison_path = folder/f'comparisons{job["index"]:02d}.json'
            write(comparison_path, {'completed': comparisons, 'through_worker': job['index'],
                                   'started_utc': comparison_start, 'ended_utc': now()})
            ledger['workers'][-1]['comparison_receipt'] = ref(comparison_path)
            write(folder/f'checked_ledger{job["index"]:02d}.json', ledger)
        ledger['complete'] = True
    except BaseException:
        ledger['failure'] = traceback.format_exc()
        raise
    finally:
        ledger['ended_utc'] = now()
        write(RESULTS/f'{PREFIX}_campaign_attempt{attempt}.json', ledger)
    report = make_report(ledger, registration)
    write(RESULTS/(PREFIX+'_report.json'), report)
    return 0


def make_report(ledger, registration):
    from fdtd_constitutive_absorber_probe import compare_pair, difference, jsonable
    require(ledger['complete'] and ledger['registration_sha256'] == sha(REGISTRATION)
            and len(ledger['workers']) == 60, 'Incomplete campaign ledger.')
    records = {}
    previous_comparison_end = registration['registered_utc']
    for expected, row in zip(jobs(), ledger['workers']):
        require(row['job'] == expected, 'Worker sequence or complete configuration differs.')
        result, arrays = read_worker(row['execution'], expected['config'], registration['source_identity'])
        require(datetime.fromisoformat(result['started_utc']) >= datetime.fromisoformat(registration['registered_utc']),
                'A campaign response predates registration.')
        records[canonical(expected['config'])] = (result, arrays)
        compared = load_ref(row['comparison_receipt'])
        require(compared['through_worker'] == expected['index']
                and compared['completed'] == check_ready_comparisons(records, registration),
                'Incremental comparison receipt does not match its first eligible worker prefix.')
        sequence = [previous_comparison_end, row['execution']['started_utc'],
                    row['execution']['ended_utc'], compared['started_utc'], compared['ended_utc']]
        stamps = [datetime.fromisoformat(t) for t in sequence]
        require(all(t.tzinfo is not None for t in stamps) and all(a <= b for a,b in zip(stamps,stamps[1:])),
                'Worker/comparison order differs from immediate-stop execution.')
        previous_comparison_end = compared['ended_utc']
    def get(family, **kwargs):
        return records[canonical(config(family, 'numpy', **kwargs))]
    ready = check_ready_comparisons(records, registration)
    require(len(ready) == 36, 'Expected24 base,4 length,6 duration,2 zero comparisons.')
    report = {'registration_fingerprint': registration['fingerprint'], 'workers': 60,
              'base_comparisons': [], 'designated_controls': [], 'zero_controls': [],
              'no_replacements': True, 'all_warnings': [], 'known_calibration_repeats': []}
    ceiling = registration['accepted']['budgets']['physical_alignment']
    def aligned(pair):
        require(all(pair['mapping_checks'].values()), 'Shared-grid mapping failed.')
        for group in ('coordinate_errors', 'shared_lowering', 'plane_coordinates'):
            require(all(v['same_shape'] and v['finite'] and v['roundoff_scaled'] <= ceiling for v in pair[group].values()),
                    'Shared-grid lowering/coordinate alignment exceeds its measured budget.')
    for family, n, placement, polarization in itertools.product(FAMILIES, (20, 30, 40), (0., .3), ('u', 'v')):
        a, x = get(family, n=n, placement=placement, polarization=polarization)
        b, y = get(family, n=n, placement=placement, polarization=polarization, length='reference')
        pair = compare_pair(a, b, x, y)
        aligned(pair)
        report['base_comparisons'].append({'family': family, 'n': n, 'placement': placement,
                                           'polarization': polarization, 'comparison': pair})
    for family in FAMILIES:
        control = {'family': family, 'n': 30, 'placement': .3, 'polarization': 'u',
                   'qualification': 'These controls do not bound the other22 base configurations.'}
        for time in (16., 32.):
            a, x = get(family, n=30, length='reference', time=time)
            b, y = get(family, n=30, length='farther', time=time)
            pair = compare_pair(a, b, x, y)
            aligned(pair)
            control['reference_vs_farther_T'+str(int(time))] = pair
        for length in ('short', 'reference', 'farther'):
            short, a = get(family, n=30, length=length)
            long, b = get(family, n=30, length=length, time=32.)
            # The discrete fixed-run trajectory before T16 must be the same; DFTs
            # intentionally use different endpoints and quantify time sensitivity.
            prefix = {k: difference(a[k], b[k][:len(a[k])]) for k in a
                      if (k.endswith('_values') and k != 'energy_values')
                      or (k.startswith('source_') and k != 'source_stamp')}
            require(all(v['same_shape'] and v['finite'] and v['roundoff_scaled'] <= ceiling for v in prefix.values()),
                    'Same-domain fixed-duration prefix changed.')
            spectra = {k: [difference(a[k][j], b[k][j]) for j in range(3)] for k in a
                       if k.startswith('plane_') and '_coords_' not in k}
            control[length+'_T16_vs_T32'] = {'prefix': prefix, 'spectra_by_wavelength': spectra,
                                            'energy_comparison': 'raw sparse curves retained, no exact budget asserted'}
        report['designated_controls'].append(control)
        shaped, a = get(family, n=30, contrast=False)
        legacy, b = get(family, n=30, contrast=False, homogeneous=True)
        keys = [k for k in a if k.endswith('_values') and k != 'energy_values'
                or (k.startswith('plane_') and '_coords_' not in k)
                or k.startswith('source_')]
        parity = {k: difference(a[k], b[k]) for k in keys}
        require(shaped['plan']['promoted_terms'] > 0 and legacy['plan']['promoted_terms'] == 0,
                'Zero control silently exercised the same path twice.')
        require(all(v['same_shape'] and v['finite'] and v['roundoff_scaled'] <= ceiling for v in parity.values()),
                'Zero-cross new/legacy field or source parity exceeds measured budget.')
        report['zero_controls'].append({'family': family, 'comparison': parity,
            'shaped_central_cross_is_zero': bool((a['central_cross'] == 0).all()),
            'legacy_central_cross_is_zero': bool((b['central_cross'] == 0).all()),
            'energy_equality_claim': False})
    calibration = load_ref(registration['accepted']['calibration_manifest'])
    for family in FAMILIES:
        for length in ('short', 'reference'):
            cfg = config(family, 'numpy', length=length)
            index = calibration['schedule'].index(cfg)
            previous, a = read_worker(calibration['workers'][index], cfg, calibration['source_before'])
            current, b = records[canonical(cfg)]
            require(set(a) == set(b), 'Known calibration array inventory differs.')
            parity = {k: difference(b[k], a[k]) for k in a}
            report['known_calibration_repeats'].append({'family': family, 'length': length,
                'all_payloads_equal_bits': all(v['equal_bits'] for v in parity.values()), 'arrays': parity,
                'qualification': 'Previously exposed response; rerun equality is provenance, not independent corroboration.'})
    for row, arrays in records.values():
        report['all_warnings'].append({'config': row['config'], 'stable': row['stable'], 'warnings': row['warnings']})
    report['complete'] = True
    report['qualification'] = 'Finite-domain termination and duration sensitivity only. No reflection coefficient, continuum order, exact open-domain reference or source-free sparse energy theorem.'
    return jsonable(report)


def audit():
    registration = registered()
    ledger = json.loads((RESULTS/(PREFIX+'_campaign_attempt1.json')).read_text())
    expected = make_report(ledger, registration)
    saved = json.loads((RESULTS/(PREFIX+'_report.json')).read_text())
    require(saved == expected, 'Saved campaign report differs from raw-array replay.')
    result = {'complete': True, 'saved_array_replay': True, 'guard': guard(),
              'registration': ref(REGISTRATION), 'ledger': ref(RESULTS/(PREFIX+'_campaign_attempt1.json')),
              'report': ref(RESULTS/(PREFIX+'_report.json')), 'ended_utc': now(), 'workers': 60}
    write(RESULTS/(PREFIX+'_campaign_audit.json'), result)
    print(json.dumps({'complete': True, 'workers': 60, 'saved_array_replay': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'run', 'audit'))
    parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.mode == 'register':
        require(args.acceptance is not None, 'Pass the reviewed measured acceptance explicitly.')
        register(args.acceptance)
    elif args.mode == 'run':
        run(args.attempt)
    else:
        audit()
