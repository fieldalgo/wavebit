"""Thirteen actual-source assertion witnesses for nine Debye-stack functions.

Root-only execution after a complete targeted baseline. Every attempt remains;
no syntax/import/runtime-only error counts. Snapshots use .raw to avoid treating
implementation comments as published prose. Original bytes and mtime are restored
in finally before any next witness or the complete restored test baseline.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import ast
import copy
import os
from pathlib import Path
import re
import sys
import uuid

from studies.fdtd.debye_stack import fdtd_debye_stack_support as support

ROOT, RESULTS = support.ROOT, support.RESULTS
PREFIX, TEST = 'fdtd_debye_stack', 'tests/fdtd/test_fd_debye_stack.py'
PROBE = 'studies/fdtd/debye_stack/fdtd_debye_stack_probe.py'


def mutations():
    return [('later-real-profile-wrong',
  'wavebit/fd/_lower/debye_stack.py',
  '        eps[cut] = member_eps[cut]\n',
  '        eps[cut] = member_eps[cut] if film is stack.films[0] else 1.125*member_eps[cut]\n',
  'tests/fdtd/test_fd_debye_stack.py::test_each_member_advance_matches_the_declared_series_parallel_response[pair-z]'),
 ('later-memory-allocation-omitted',
  'wavebit/fd/_check.py',
  '                              for t in p.terms if isinstance(t, PoleTerm)) * itemsize\n',
  '                              for t in p.terms if isinstance(t, PoleTerm) and getattr(t, '
  "'member_index', 0) == 0) * itemsize\n",
  'tests/fdtd/test_fd_debye_stack.py::test_composition_preserves_singletons_declared_slots_and_actual_allocation[pair-numpy]'),
 ('later-storage-weight-omitted',
  'wavebit/fd/_lower/debye_stack.py',
  "    return {f'debye_stack[{index}]': _MODULES[kind].pole_params(\n"
  '                _member_scene(scene, index), plan, rt)[_KEYS[kind]]\n'
  '            for index, kind in enumerate(stack.kinds)}\n',
  "    out = {f'debye_stack[{index}]': _MODULES[kind].pole_params(\n"
  '                _member_scene(scene, index), plan, rt)[_KEYS[kind]]\n'
  '            for index, kind in enumerate(stack.kinds)}\n'
  "    out['debye_stack[1]']['energy_inverse_delta'] = tuple(0*v for v in "
  "out['debye_stack[1]']['energy_inverse_delta'])\n"
  '    return out\n',
  'tests/fdtd/test_fd_debye_stack.py::test_all_members_storage_and_closed_yee_dissipation_use_independent_residues[pair-z-numpy]'),
 ('later-optical-drive-omitted',
  'wavebit/fd/_lower/debye_stack.py',
  "    return {f'debye_stack[{index}]': _MODULES[kind].pole_params(\n"
  '                _member_scene(scene, index), plan, rt)[_KEYS[kind]]\n'
  '            for index, kind in enumerate(stack.kinds)}\n',
  "    out = {f'debye_stack[{index}]': _MODULES[kind].pole_params(\n"
  '                _member_scene(scene, index), plan, rt)[_KEYS[kind]]\n'
  '            for index, kind in enumerate(stack.kinds)}\n'
  "    out['debye_stack[1]']['drive'] = tuple(0*v for v in out['debye_stack[1]']['drive'])\n"
  '    return out\n',
  'tests/fdtd/test_fd_debye_stack.py::test_oblique_debye_stack_matches_declared_multilayer_on_each_backend[debye-pair-numpy]'),
 ('padded-overlap-admitted',
  'wavebit/fd/_lower/debye_stack.py',
  '                 if left[0][1] > right[0][0]), None)\n',
  '                 if False and left[0][1] > right[0][0]), None)\n',
  'tests/fdtd/test_fd_debye_stack.py::test_every_member_must_pass_scope_and_exact_padded_support_separation[off-grid-two-dx]'),
 ('unvalidated-decay-admitted',
  'wavebit/fd/_lower/debye_stack.py',
  "    if getattr(scene.run, 'decay', None) is not None:\n",
  "    if False and getattr(scene.run, 'decay', None) is not None:\n",
  'tests/fdtd/test_fd_debye_stack.py::test_every_member_must_pass_scope_and_exact_padded_support_separation[decay]'),
 ('whole-prepared-material-identity-bypassed',
  'wavebit/fd/_lower/debye_stack.py',
  '    expected = terms(scene, plan.grid)\n',
  '    expected = held if held else terms(scene, plan.grid)\n',
  'tests/fdtd/test_fd_debye_stack.py::test_prepared_stack_binds_all_members_memories_and_source_allocation[later-delta]'),
 ('foreign-material-source-memory-ignored',
  'wavebit/fd/_lower/debye_stack.py',
  '        if (len({term.name for term in complete}) != len(complete)\n'
  '                or tuple(plan.terms) != complete):\n',
  '        if False and (len({term.name for term in complete}) != len(complete)\n'
  '                or tuple(plan.terms) != complete):\n',
  'tests/fdtd/test_fd_debye_stack.py::test_prepared_stack_binds_all_members_memories_and_source_allocation[foreign-flux]'),
 ('torch-forward-carrier-detached',
  'wavebit/fd/_lower/shape_ranges.py',
  '    return _shape_traced(value)\n',
  '    return False\n',
  'tests/fdtd/test_fd_debye_stack.py::test_traced_later_poles_geometry_and_host_cannot_detach_into_stack_admission[last-delta-torch-forward]'),
 ('cfl-only-first-member',
  'wavebit/fd/_lower/debye_stack.py',
  '    return max(_MODULES[kind].policy_rate(_member_scene(scene, index), plan)\n'
  '               for index, kind in enumerate(stack.kinds))\n',
  '    return max(_MODULES[kind].policy_rate(_member_scene(scene, index), plan)\n'
  '               for index, kind in enumerate(stack.kinds[:1]))\n',
  'tests/fdtd/test_fd_debye_stack.py::test_cfl_uses_summed_local_moments_then_the_maximum_over_separated_members[later-fast]'),
 ('cfl-spatial-sum-instead-of-maximum',
  'wavebit/fd/_lower/debye_stack.py',
  '    return max(_MODULES[kind].policy_rate(_member_scene(scene, index), plan)\n'
  '               for index, kind in enumerate(stack.kinds))\n',
  '    return sum(_MODULES[kind].policy_rate(_member_scene(scene, index), plan)\n'
  '               for index, kind in enumerate(stack.kinds))\n',
  'tests/fdtd/test_fd_debye_stack.py::test_cfl_uses_summed_local_moments_then_the_maximum_over_separated_members[four-members]'),
 ('band-only-first-member',
  'wavebit/fd/_lower/debye_stack.py',
  "        bounds = [(f'structures[{index}]', _band_bound(film, low, high))\n"
  '                  for index, film in enumerate(stack.films)]\n',
  "        bounds = [(f'structures[{index}]', _band_bound(film, low, high))\n"
  '                  for index, film in enumerate(stack.films[:1])]\n',
  'tests/fdtd/test_fd_debye_stack.py::test_spectral_certificate_checks_every_declared_pole_and_later_member[band-later]'),
 ('band-only-first-declared-pole',
  'wavebit/fd/_lower/debye_stack.py',
  '    for delta, tau in zip(deltas, taus):\n',
  '    for delta, tau in zip(deltas[:1], taus[:1]):\n',
  'tests/fdtd/test_fd_debye_stack.py::test_spectral_certificate_checks_every_declared_pole_and_later_member[band-later]')]


def source_identity():
    """The probe's exact identity schema, read without importing a field solver."""
    parsed = ast.parse((ROOT/PROBE).read_text(encoding='utf-8'))
    declared = [ast.literal_eval(node.value) for node in parsed.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                    and t.id == 'REFERENCE_DEPENDENCIES' for t in node.targets)]
    support.require(len(declared) == 1, 'Probe reference inventory must be a literal tuple.')
    identity = support.identity()
    return {'helper_sha256': support.sha(ROOT/PROBE), 'fd_source_sha256': identity['fd'],
            'wavebit_source_sha256': identity['package'],
            'reference_dependencies': {name: support.sha(ROOT/name) for name in declared[0]},
            'fd_digest_rule': 'sorted relative POSIX path + NUL + LF-normalized bytes + NUL'}


def archive(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
    support.require(path.read_bytes() == raw, 'Raw archive differs.')
    return support.relative(path), support.digest(raw)


def mutate(original, before, after):
    text = original.decode('utf-8')
    crlf = '\r\n' in text
    support.require(not crlf or '\n' not in text.replace('\r\n', ''), 'Mixed source newlines require explicit review.')
    text = text.replace('\r\n', '\n')
    support.require(text.count(before) == 1 and before != after, 'Expected exactly one reviewed mutation anchor.')
    text = text.replace(before, after, 1)
    ast.parse(text)
    return text.replace('\n', '\r\n').encode() if crlf else text.encode()


def assertion_witness(text, node, exit_code):
    support.check_log(text)
    actual, totals = support.reports(text), support.summary(text)
    support.require(actual == {node: 'failed'} and totals['failed'] == 1
                    and sum(totals.values()) == 1 and exit_code == 1,
                    'Expected exactly one intended failed node and actual exit1.')
    lines = [line for line in text.splitlines() if re.match(r'^E\s+', line)]
    support.require(lines, 'No pytest error evidence.')
    missing = ("Failed: DID NOT RAISE <class 'ValueError'>", 'Failed: DID NOT RAISE ValueError')
    missing_lines = [re.sub(r'^E\s+', '', line).strip() for line in lines if 'Failed: DID NOT RAISE' in line]
    support.require(not missing_lines or all(line in missing for line in missing_lines),
                    'Unexpected missing-refusal type or spelling.')
    support.require(any('AssertionError' in line or re.match(r'^E\s+assert\b', line) for line in lines)
                    or bool(missing_lines), 'Only actual assertions or the named missing ValueError count.')
    forbidden = r'\b(?:ValueError|TypeError|RuntimeError|ImportError|ModuleNotFoundError|NameError|KeyError|AttributeError|IndexError|SyntaxError|OSError|MemoryError|SystemExit|ZeroDivisionError)\b'
    support.require(not any(re.search(forbidden, line) and 'Failed: DID NOT RAISE' not in line
                            for line in lines), 'Runtime-only or collateral exception is not a witness.')
    failed = [line for line in text.splitlines() if line.startswith('FAILED ')]
    support.require(len(failed) == 1 and failed[0].split(' - ', 1)[0] == 'FAILED '+node,
                    'Final failed-node summary does not match the selected witness.')
    return {'node_status': actual, 'totals': totals, 'assertion_failure': True,
            'error_lines': lines, 'error_marker': missing_lines[-1] if missing_lines else 'AssertionError/assert'}


def saved_targeted(path):
    data = support.load(path)
    support.require(data['complete'] and data['source_before'] == data['source_after'] == support.identity(),
                    'A complete restored-source targeted baseline is required.')
    support.verify_guard(data['guard_before'])
    if 'guard_after' in data:
        support.verify_guard(data['guard_after'])
    by_name = {row['name']: row for row in data['checks']}
    support.require(len(by_name) == len(data['checks']) and set(by_name) ==
        {'new_stack_collection', 'new_stack', 'targeted_collection', 'targeted'}, 'Unexpected targeted checks.')
    for row in by_name.values():
        support.require(row['exit'] == 0 and support.sha(support.inside(row['log'])) == row['sha256'],
                        'Targeted child exit or raw hash differs.')
    for name in ('new_stack', 'targeted'):
        observed = support.audit_pytest(support.inside(by_name[name+'_collection']['log']).read_text(encoding='utf-8'),
                                       support.inside(by_name[name]['log']).read_text(encoding='utf-8'))
        support.require(observed == by_name[name]['observed'], 'Targeted saved node outcomes differ.')
        if name == 'new_stack':
            support.require(observed['totals']['passed'] == observed['collected'], 'New baseline must all pass.')
    return data, by_name['new_stack'], by_name['new_stack_collection']


def run(attempt):
    directory, manifest, record = support.attempt_record('mutations', attempt)
    canonical = RESULTS/f'{PREFIX}_mutations.json'
    support.require(not canonical.exists(), 'Never replace a previously published accepted mutation record.')
    record.update(passed=False, restored=False, all_function_assertion_witnesses=False,
                  source_identity_before=source_identity(), witnesses=[])
    caught = None
    originals, stats = {}, {}
    try:
        target = RESULTS/f'{PREFIX}_targeted.json'
        targeted, baseline, collection = saved_targeted(target)
        specs = mutations()
        functions = {TEST+'::'+node.name for node in ast.parse((ROOT/TEST).read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')}
        support.require(len(specs) == 13 and len({s[0] for s in specs}) == 13,
                        'Exactly thirteen unique reviewed mechanisms are required.')
        support.require(len(functions) == 9 and {s[4].split('[')[0] for s in specs} == functions,
                        'All nine actual new functions need a source witness.')
        record.update(targeted_manifest=support.relative(target), targeted_manifest_sha256=support.sha(target),
            baseline_log=baseline['log'], baseline_log_sha256=baseline['sha256'],
            test_sha256=support.sha(ROOT/TEST), harness_sha256=support.sha(Path(__file__)),
            expected_functions=sorted(functions))
        record['test_snapshot'], record['test_snapshot_sha256'] = archive(directory/'test.py.raw', (ROOT/TEST).read_bytes())
        record['harness_snapshot'], _ = archive(directory/'harness.py.raw', Path(__file__).read_bytes())
        row, raw_collection = support.run_command(record, manifest, 'mutation_collection',
            [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', TEST, '--collect-only'], directory)
        support.require(row['exit'] == 0, 'Mutation collection failed.')
        support.check_log(raw_collection)
        collected = support.nodes(raw_collection)
        old_nodes = support.nodes(support.inside(collection['log']).read_text(encoding='utf-8'))
        support.require(collected == old_nodes, 'Actual new-test inventory changed since targeted baseline.')
        support.require(all(s[4] in collected for s in specs), 'Reviewed mutation node is absent from real collection.')
        record['collection'] = {**row, 'exit_code': row['exit'], 'collected': len(collected), 'nodes': collected}
        prepared = []
        for name, filename, before, after, node in specs:
            source = support.inside(filename)
            support.require(filename.startswith('wavebit/') and source.is_file(), 'Mutation is outside owned production.')
            if filename not in originals:
                originals[filename], stats[filename] = source.read_bytes(), source.stat()
            broken = mutate(originals[filename], before, after)
            prepared.append((name, filename, before, after, node, broken))
        record['source_originals'] = {}
        for index, filename in enumerate(sorted(originals)):
            file, value = archive(directory/f'original_{index}.py.raw', originals[filename])
            record['source_originals'][filename] = {'file': file, 'sha256': value,
                'mtime_ns': stats[filename].st_mtime_ns, 'atime_ns': stats[filename].st_atime_ns}
        support.write(manifest, record)
        for index, (name, filename, before, after, node, broken) in enumerate(prepared):
            support.verify_guard(record['guard_before'])
            source, original, stat = support.inside(filename), originals[filename], stats[filename]
            support.require(source.read_bytes() == original and source.stat().st_mtime_ns == stat.st_mtime_ns,
                            'Original source bytes/mtime changed before mutation.')
            folder = directory/f'{index:02d}_{name}'
            folder.mkdir(exist_ok=False)
            broken_path, broken_hash = archive(folder/'mutant.py.raw', broken)
            witness = {'name': name, 'test': node, 'production_file': filename,
                'replacement': {'before': before, 'after': after}, 'original_snapshot': record['source_originals'][filename]['file'],
                'original_sha256': support.digest(original), 'mutant_snapshot': broken_path, 'mutant_sha256': broken_hash,
                'source_mtime_before_ns': stat.st_mtime_ns, 'source_identity_before': source_identity(),
                'accepted': False, 'assertion_failure': False, 'restored': False}
            record['witnesses'].append(witness)
            support.write(manifest, record)
            altered = copy.deepcopy(record['guard_before'])
            altered['maps']['wavebit'][filename] = broken_hash
            temporary = support.SCRATCH/('pytest-mutant-'+name+'-'+uuid.uuid4().hex)
            temporary.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-B', '-m', 'pytest', '-vv', '--tb=short',
                       '-p', 'no:cacheprovider', node, '--basetemp', str(temporary)]
            try:
                atomic_source(source, broken)
                support.verify_guard(altered)
                witness.update(source_identity_broken=source_identity(), guard_mutated=altered,
                               source_mtime_broken_ns=source.stat().st_mtime_ns)
                support.write(manifest, record)
                row, text = support.run_command(record, manifest, name, command, folder, cache_tag=name)
                witness.update(command=command, exit_code=row['exit'], raw_log=row['log'], raw_log_sha256=row['sha256'],
                               started_utc=row['started_utc'], ended_utc=row['ended_utc'], environment=row['environment'])
                try:
                    observed = assertion_witness(text, node, row['exit'])
                    witness.update(observed=observed, accepted=True, assertion_failure=True)
                except (ValueError, AssertionError) as error:
                    witness['not_accepted_reason'] = str(error)
                support.verify_guard(altered)
            finally:
                current = source.read_bytes()
                support.require(current in (original, broken), 'Unexpected third-party source bytes: refusing overwrite.')
                if current != original:
                    atomic_source(source, original)
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                witness['restored_snapshot'], witness['restored_snapshot_sha256'] = archive(folder/'restored.py.raw', source.read_bytes())
                witness.update(restored=source.read_bytes() == original,
                    restored_sha256=support.sha(source), source_mtime_restored_ns=source.stat().st_mtime_ns,
                    source_identity_after=source_identity(), restored_utc=support.now())
                support.write(manifest, record)
                support.require(witness['restored'] and witness['source_mtime_restored_ns'] == stat.st_mtime_ns,
                                'Source bytes or original mtime were not restored exactly.')
                support.verify_guard(record['guard_before'])
        support.verify_guard(record['guard_before'])
        baseline_row = support.pytest_pair(record, manifest, 'final_baseline', (TEST,), directory)
        baseline_observed = baseline_row['observed']
        support.require(set(baseline_observed['node_status']) == set(collected), 'Restored baseline inventory changed.')
        support.require(baseline_observed['totals']['passed'] == len(collected), 'Restored baseline did not all pass.')
        record['final_baseline'] = {**baseline_row, 'exit_code': baseline_row['exit'],
            'passed': baseline_observed['totals']['passed'], 'skipped': baseline_observed['totals']['skipped'],
            'raw_log': baseline_row['log'], 'raw_log_sha256': baseline_row['sha256']}
        record['all_function_assertion_witnesses'] = functions == {
            w['test'].split('[')[0] for w in record['witnesses'] if w['accepted']}
        record['passed'] = (len(record['witnesses']) == 13
            and all(w['accepted'] and w['restored'] for w in record['witnesses'])
            and record['all_function_assertion_witnesses'])
        record['complete'] = True
    except BaseException as error:
        caught = error
        record['error'] = repr(error)
    finally:
        record['restored'] = all(support.inside(name).read_bytes() == raw
            and support.inside(name).stat().st_mtime_ns == stats[name].st_mtime_ns
            for name, raw in originals.items())
        record.update(source_after=support.identity(), source_identity_after=source_identity(),
                      guard_after=support.guard(), ended_utc=support.now())
        support.write(manifest, record)
    if caught is not None:
        raise caught
    support.require(record['restored'] and record['guard_after'] == record['guard_before'], 'Final frozen-source guard failed.')
    support.require(record['source_identity_before'] == record['source_identity_after'], 'Source identity changed.')
    if record['passed']:
        support.write(canonical, record, exclusive=True)
    return 0 if record['passed'] else 1


def atomic_source(path, raw):
    temporary = path.with_name(path.name+'.debye-stack-'+uuid.uuid4().hex+'.tmp')
    support.require(temporary.parent == path.parent and path.is_relative_to(ROOT/'src/wavebit'), 'Unowned source write.')
    try:
        with temporary.open('xb') as stream:
            stream.write(raw)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    raise SystemExit(run(parser.parse_args().attempt))
