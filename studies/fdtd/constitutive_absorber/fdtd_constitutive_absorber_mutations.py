"""Unexecuted actual-source mutation witnesses for bounded constitutive Absorber.

Every node is selected from the real pytest collection by its full parameter map.
Only assertion failures or an explicit missing-refusal failure count. Original and
mutant raw bytes are archived before each mutation; restoration is in finally and
only replaces bytes still owned by this attempt. No numerical allowance is changed.
Canonical publication occurs after restored-baseline success; earlier failed or
surviving attempts remain on disk. No automatic retry is performed.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
import ast
import copy
import gzip
import os
from pathlib import Path
import re
import sys
import uuid

from studies.fdtd.constitutive_absorber.fdtd_constitutive_absorber_gate_support import ROOT, RESULTS, PREFIX, SCRATCH, NEW_TEST, require, guard, identity, verify_guard, attempt_record, run_command, pytest_pair, audit_pytest, nodes, reports, summary, check_log, load, write, sha, digest, relative, inside, now, finish, failed

SPECS = [{'name': 'flux-no-damping',
  'production_file': 'wavebit/fd/_kernel/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_kernel/constitutive_absorber.py.draft',
  'source_raw_sha256': 'ee0ba7f96b20fd7ef5be234b90367a059b17cf23ee8d5d7292d7c98c80104b70',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_damped_flux_and_field_match_independent_dense_operator',
  'case_parameters': {'backend': 'numpy', 'axis': 2, 'complex_': False},
  'replacements': [{'before': "flux = factors['keep'] * mem + self.ratio * driven",
                    'after': 'flux = mem + self.ratio * driven'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'cross-no-drive',
  'production_file': 'wavebit/fd/_kernel/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_kernel/constitutive_absorber.py.draft',
  'source_raw_sha256': 'ee0ba7f96b20fd7ef5be234b90367a059b17cf23ee8d5d7292d7c98c80104b70',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_damped_flux_and_field_match_independent_dense_operator',
  'case_parameters': {'backend': 'numpy', 'axis': 2, 'complex_': True},
  'replacements': [{'before': "cross_action(driven, params['constitutive'][self.name]",
                    'after': "cross_action(local, params['constitutive'][self.name]"}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'flux-wrong-drive-sign',
  'production_file': 'wavebit/fd/_kernel/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_kernel/constitutive_absorber.py.draft',
  'source_raw_sha256': 'ee0ba7f96b20fd7ef5be234b90367a059b17cf23ee8d5d7292d7c98c80104b70',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_staggered_storage_obeys_independent_dissipation_identity',
  'case_parameters': {'backend': 'numpy', 'complex_': True, 'damped': True, 'work': True},
  'replacements': [{'before': "flux = factors['keep'] * mem + self.ratio * driven",
                    'after': "flux = factors['keep'] * mem - self.ratio * driven"}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'energy-no-magnetic-loss-correction',
  'production_file': 'wavebit/fd/_kernel/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_kernel/constitutive_absorber.py.draft',
  'source_raw_sha256': 'ee0ba7f96b20fd7ef5be234b90367a059b17cf23ee8d5d7292d7c98c80104b70',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_observer_uses_whole_absorber_and_both_magnetic_clocks',
  'case_parameters': {'backend': 'numpy', 'complex_': True, 'large_rate': True},
  'replacements': [{'before': '+ rt.real_inner(old, old + following, (self.dt / 2.0) * rate)',
                    'after': '+ rt.real_inner(old, old + following, 0.0 * rate)'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'energy-magnetic-clock-shift',
  'production_file': 'wavebit/fd/_kernel/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_kernel/constitutive_absorber.py.draft',
  'source_raw_sha256': 'ee0ba7f96b20fd7ef5be234b90367a059b17cf23ee8d5d7292d7c98c80104b70',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_observer_uses_whole_absorber_and_both_magnetic_clocks',
  'case_parameters': {'backend': 'torch', 'complex_': False, 'large_rate': False},
  'replacements': [{'before': '+ rt.real_inner(old, old + following, (self.dt / 2.0) * rate)',
                    'after': '+ rt.real_inner(following, old + following, (self.dt / 2.0) * rate)'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'commutation-pointwise-only',
  'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_every_live_interpolation_link_is_certified',
  'case_parameters': {'field': 'E', 'pair': [0, 1], 'link': 'i-neighbour'},
  'replacements': [{'before': 'bad = active & ((rates[i] != rates[j])\n'
                              '                        | (valid_i & (other_i != rates[j]))\n'
                              '                        | (valid_j & (rates[i] != other_j))\n'
                              '                        | (valid_i & valid_j & (other_i != other_j)))',
                    'after': 'bad = active & (rates[i] != rates[j])'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'physical-diagonal-uses-damped-value',
  'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_lift_and_cfl_read_physical_not_damped_diagonal',
  'case_parameters': {'backend': 'numpy', 'axis': 2},
  'replacements': [{'before': "value = value / params['constitutive_absorber'][term.name]['drive']",
                    'after': 'value = value * 1.0'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'star-lift-no-drive',
  'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_lift_and_cfl_read_physical_not_damped_diagonal',
  'case_parameters': {'backend': 'numpy', 'axis': 1},
  'replacements': [{'before': "value = params['constitutive_absorber'][term.name]['drive'] * value",
                    'after': 'value = value * 1.0'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'prepared-missing-term-allowed',
  'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_prepared_plan_cannot_bypass_new_scope',
  'case_parameters': {'change': 'missing'},
  'replacements': [{'before': 'if expected != held or (expected and tuple(plan.constitutive) != '
                              'all_expected):',
                    'after': 'if False:'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'prepared-foreign-ordinary-term-allowed',
  'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_prepared_plan_cannot_bypass_new_scope',
  'case_parameters': {'change': 'foreign'},
  'replacements': [{'before': 'if expected != held or (expected and tuple(plan.constitutive) != '
                              'all_expected):',
                    'after': 'if expected != held:'},
                   {'before': 'if tuple(plan.terms) != all_expected + source_terms(scene, plan):',
                    'after': 'if False:'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'magnetic-source-allowed',
  'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_scope_refuses_unproved_material_source_and_boundary_combinations',
  'case_parameters': {'kind': 'magnetic-source'},
  'replacements': [{'before': "if type(source) is not Dipole or source.kind != 'electric':",
                    'after': 'if type(source) is not Dipole:'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'gpu-host-copy-allowed',
  'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_concrete_cpu_and_finite_undamped_coefficients_are_required',
  'case_parameters': {'case': 'torch-gpu'},
  'replacements': [{'before': "if rt.name == 'torch' and rt.device.type != 'cpu':",
                    'after': 'if False:'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'nonfinite-physical-diagonal-allowed',
  'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_concrete_cpu_and_finite_undamped_coefficients_are_required',
  'case_parameters': {'case': 'nonfinite'},
  'replacements': [{'before': 'if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):',
                    'after': 'if False:'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'nonoverlap-path-promoted',
  'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_nonoverlapping_scene_keeps_ordinary_term_and_memory_accounting',
  'case_parameters': {'backend': 'numpy'},
  'replacements': [{'before': 'if not touched:', 'after': 'if not terms:'},
                   {'before': 'selected = {t.name for t in touched}',
                    'after': 'selected = {t.name for t in terms}'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'masked-flux-not-projected',
  'production_file': 'wavebit/fd/_kernel/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_kernel/constitutive_absorber.py.draft',
  'source_raw_sha256': 'ee0ba7f96b20fd7ef5be234b90367a059b17cf23ee8d5d7292d7c98c80104b70',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_projected_yee_step_and_sampled_energy_match_independent_flux_oracle',
  'case_parameters': {'backend': 'numpy', 'wall': 'PEC', 'axis': 2},
  'replacements': [{'before': 'if self.masks:', 'after': 'if False:'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'name': 'observer-selected-as-legacy',
  'production_file': 'wavebit/fd/_lower/plan.py',
  'source_draft': 'drafts/integration/wavebit/fd/_lower/plan.py.draft',
  'source_raw_sha256': 'acbb2b7a3d4c6f6cd4705afb33e4dc2cc97f28ef4f87f200ea956c63501f66b8',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_projected_yee_step_and_sampled_energy_match_independent_flux_oracle',
  'case_parameters': {'backend': 'numpy', 'wall': 'PMC', 'axis': 1},
  'replacements': [{'before': 'any(\n'
                              '            isinstance(t, AbsorberConstitutiveTerm) for t in '
                              'self.constitutive)',
                    'after': 'False'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False},
 {'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_prepared_plan_cannot_bypass_new_scope',
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False,
  'name': 'new-source-silently-ignored',
  'case_parameters': {'change': 'added-source'},
  'replacements': [{'before': 'if tuple(plan.terms) != all_expected + source_terms(scene, plan):',
                    'after': 'if False:'}]},
 {'production_file': 'wavebit/fd/_lower/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_lower/constitutive_absorber.py.draft',
  'source_raw_sha256': '48ce46cd238cdac16accc1457babb6e23ad713efe5da780c24964d1d04945d1b',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_prepared_plan_cannot_bypass_new_scope',
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False,
  'name': 'foreign-memory-admitted',
  'case_parameters': {'change': 'foreign-memory'},
  'replacements': [{'before': 'if tuple(plan.terms) != all_expected + source_terms(scene, plan):',
                    'after': 'if False:'}]},
 {'name': 'zero-cross-duplicates-diagonal-curl',
  'production_file': 'wavebit/fd/_kernel/constitutive_absorber.py',
  'source_draft': 'drafts/wavebit/fd/_kernel/constitutive_absorber.py.draft',
  'source_raw_sha256': 'ee0ba7f96b20fd7ef5be234b90367a059b17cf23ee8d5d7292d7c98c80104b70',
  'test_file': 'tests/fdtd/test_fd_constitutive_absorber.py',
  'test_function': 'test_equal_material_new_route_matches_legacy_fields',
  'case_parameters': {'backend': 'numpy', 'family': 'cylinder'},
  'replacements': [{'before': 'return flux, delta',
                    'after': 'return flux, delta + self.ratio * driven'}],
  'matching': 'one decoded-LF occurrence per hunk, preserve original raw EOL',
  'mutant_ast_parsed': True,
  'executed': False}]

COLLECTION_PLUGIN = r'''"""Saved stdlib collection inventory; never alters collection or test execution."""
import json
import os
from pathlib import Path

def encode(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, dict) and all(type(k) is str for k in value):
        return {k: encode(v) for k, v in value.items()}
    raise TypeError('Non-concrete collection parameter: '+type(value).__name__)

def pytest_collection_finish(session):
    rows = [{'nodeid': item.nodeid, 'function': item.originalname,
             'parameters': encode(item.callspec.params) if hasattr(item, 'callspec') else {}}
            for item in session.items]
    with Path(os.environ['C9_NODE_EXPORT']).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
        stream.write('\n')
'''


def archive(raw, destination):
    stored = gzip.compress(raw, compresslevel=9, mtime=0)
    require(gzip.decompress(stored) == raw, 'Archive round trip failed.')
    with destination.open('xb') as stream:
        stream.write(stored)
    return {'file': relative(destination), 'sha256': digest(raw), 'archive_sha256': digest(stored)}


def replace_owned(path, expected, replacement):
    """Atomic raw replacement; never clobber bytes no longer owned by this attempt."""
    require(path.read_bytes() == expected, 'Source is no longer owned: '+relative(path))
    temporary = path.with_name('.'+path.name+'.c9-'+uuid.uuid4().hex+'.raw.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(replacement)
            stream.flush()
            os.fsync(stream.fileno())
        require(path.read_bytes() == expected, 'Source changed before atomic replacement.')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            require(temporary.read_bytes() == replacement, 'Unexpected temporary bytes; preserve for review.')
            temporary.unlink()


def make_mutant(original, spec):
    crlf = b'\r\n' in original
    require(not crlf or b'\n' not in original.replace(b'\r\n', b''), 'Mixed source line endings.')
    text = original.decode('utf-8').replace('\r\n', '\n')
    for hunk in spec['replacements']:
        require(text.count(hunk['before']) == 1, 'Mutation anchor is absent or ambiguous: '+spec['name'])
        text = text.replace(hunk['before'], hunk['after'], 1)
    ast.parse(text, filename=spec['production_file'])
    if crlf:
        text = text.replace('\n', '\r\n')
    mutated = text.encode('utf-8')
    require(mutated != original, 'Mutation must change actual production bytes.')
    return mutated


def assertion_witness(text, node, exit_code):
    check_log(text)
    actual = reports(text)
    totals = summary(text)
    require(actual == {node: 'failed'} and totals['failed'] == 1
            and sum(totals.values()) == 1 and exit_code == 1,
            'Expected exactly the selected test assertion failure.')
    error_lines = [line for line in text.splitlines() if re.match(r'^E\s+', line)]
    require(error_lines, 'No raw pytest failure evidence.')
    require(any('AssertionError' in line or 'assert ' in line or 'Failed: DID NOT RAISE' in line
                for line in error_lines), 'No assertion or missing-refusal witness.')
    forbidden = r'\b(?:ValueError|TypeError|RuntimeError|ImportError|ModuleNotFoundError|NameError|KeyError|AttributeError|IndexError|SyntaxError|OSError|MemoryError|SystemExit|ZeroDivisionError)\b'
    require(not any(re.search(forbidden, line) and 'Failed: DID NOT RAISE' not in line
                    for line in error_lines), 'A collateral exception is not an assertion witness.')
    return {'node_status': actual, 'totals': totals, 'assertion_failure': True}


def main(attempt):
    directory, manifest, record = attempt_record('mutations', attempt)
    record.update(witnesses=[], restored=False, all_function_assertion_witnesses=False,
                  expected_mechanisms=len(SPECS))
    try:
        targetpath = RESULTS/(PREFIX+'targeted.json')
        targeted = load(targetpath)
        require(targeted['complete'] and targeted['passed'], 'Complete targeted baseline required.')
        verify_guard(targeted['guard_before'])
        verify_guard(targeted['guard_after'])
        for row in targeted['checks']:
            require(row['exit'] == 0 and sha(inside(row['log'])) == row['sha256'], 'Targeted output differs.')
        baseline = [row for row in targeted['checks'] if row['name'] == 'new_constitutive_absorber']
        collection = [row for row in targeted['checks'] if row['name'] == 'new_constitutive_absorber_collection']
        require(len(baseline) == len(collection) == 1, 'Missing exact new-test baseline pair.')
        baseline, collection = baseline[0], collection[0]
        observed = audit_pytest(inside(collection['log']).read_text(encoding='utf-8'),
                                inside(baseline['log']).read_text(encoding='utf-8'))
        require(observed == baseline['observed'] and observed['totals']['passed'] == observed['collected'],
                'Targeted baseline has unverified or nonpassed cases.')
        functions = sorted(n.name for n in ast.parse(inside(NEW_TEST).read_text()).body
                           if isinstance(n, ast.FunctionDef) and n.name.startswith('test_'))
        require(set(functions) == {s['test_function'] for s in SPECS}, 'Every actual test function needs a witness.')
        record.update(targeted_manifest_sha256=sha(targetpath), targeted_manifest=relative(targetpath),
                      baseline_log=baseline['log'], baseline_log_sha256=baseline['sha256'],
                      test_file_sha256={NEW_TEST: sha(inside(NEW_TEST))},
                      harness_sha256=sha(Path(__file__)), expected_functions=functions)
        record['test_snapshot'] = archive(inside(NEW_TEST).read_bytes(), directory/'tests.py.raw.gz')
        record['collection_plugin'] = archive(COLLECTION_PLUGIN.encode(), directory/'collection_plugin.py.raw.gz')
        pluginfolder = SCRATCH/('collection-plugin-'+uuid.uuid4().hex)
        pluginfolder.mkdir(parents=True, exist_ok=False)
        plugin = pluginfolder/'c9_collection_inventory.py'
        plugin.write_bytes(COLLECTION_PLUGIN.encode())
        nodefile = directory/'collected_nodes.json'
        row, text = run_command(record, manifest, 'mutation_collection',
            [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
             '-p', 'c9_collection_inventory', NEW_TEST, '--collect-only'], directory,
            extra_environment={'PYTHONPATH': os.pathsep.join((str(ROOT / 'src'), str(pluginfolder), str(ROOT), str(ROOT/'studies'))),
                               'C9_NODE_EXPORT': str(nodefile)})
        require(row['exit'] == 0, 'Mutation node collection failed.')
        collected = nodes(text)
        descriptors = load(nodefile)
        require([d['nodeid'] for d in descriptors] == collected
                and set(collected) == set(observed['node_status']), 'Actual collection inventory changed.')
        require(sha(plugin) == record['collection_plugin']['sha256'], 'Collection plugin changed.')
        record['collection'] = {**row, 'nodes': relative(nodefile), 'nodes_sha256': sha(nodefile)}
        originals = {s['production_file']: inside(s['production_file']).read_bytes() for s in SPECS}
        bound = []
        for spec in SPECS:
            require(spec['test_file'] == NEW_TEST, 'Unregistered test file.')
            require(digest(originals[spec['production_file']]) == spec['source_raw_sha256'],
                    'Production differs from reviewed exact mutation anchors.')
            candidates = [d['nodeid'] for d in descriptors if d['function'] == spec['test_function']
                          and d['parameters'] == spec['case_parameters']]
            require(len(candidates) == 1, 'Expected one exact parameter-selected node: '+spec['name'])
            bound.append((spec, candidates[0], make_mutant(originals[spec['production_file']], spec)))
        record['bound_nodes'] = [{'name': spec['name'], 'node': node, 'parameters': spec['case_parameters']}
                                 for spec, node, _ in bound]
        write(manifest, record)
        for index, (spec, node, mutant) in enumerate(bound):
            verify_guard(record['guard_before'])
            source = inside(spec['production_file'])
            original = originals[spec['production_file']]
            folder = directory/f'{index:02d}_{spec["name"]}'
            folder.mkdir()
            witness = {'name': spec['name'], 'production_file': spec['production_file'],
                       'test': node, 'function': spec['test_function'], 'parameters': spec['case_parameters'],
                       'replacement': spec['replacements'], 'started_utc': now(), 'accepted': False,
                       'assertion_failure': False, 'exit_code': None, 'restored': False,
                       'original_snapshot': archive(original, folder/'original.py.raw.gz'),
                       'mutant_snapshot': archive(mutant, folder/'mutant.py.raw.gz'),
                       'source_before': identity()}
            record['witnesses'].append(witness)
            write(manifest, record)
            command = [sys.executable, '-B', '-m', 'pytest', '-vv', '-p', 'no:cacheprovider',
                       '--tb=short', '--basetemp', str(SCRATCH/('mutant-'+uuid.uuid4().hex)), node]
            witness['command'] = command
            try:
                replace_owned(source, original, mutant)
                expected = copy.deepcopy(record['guard_before'])
                expected['source'][spec['production_file']] = digest(mutant)
                require(guard() == expected, 'Mutation changed unrelated frozen bytes.')
                witness['source_mutated'] = identity()
                result, output = run_command(record, manifest, 'witness', command, folder)
                witness.update(exit_code=result['exit'], raw_log=result['log'],
                               raw_log_sha256=result['sha256'], execution=result)
                require(guard() == expected, 'Numerical child changed frozen mutant source/evidence.')
                try:
                    witness.update(assertion_witness(output, node, result['exit']), accepted=True)
                except (ValueError, KeyError) as error:
                    witness['not_accepted_reason'] = str(error)
            finally:
                current = source.read_bytes()
                if current == mutant:
                    replace_owned(source, mutant, original)
                elif current != original:
                    witness['restoration_error'] = 'Foreign source bytes preserved; owner action required.'
                witness.update(restored=source.read_bytes() == original,
                               source_restored=identity(), ended_utc=now())
                write(manifest, record)
            require(witness['restored'], 'Cannot continue with unowned source modification.')
            verify_guard(record['guard_before'])
        record['restored'] = all(w['restored'] for w in record['witnesses'])
        record['final_baseline'] = pytest_pair(record, manifest, 'restored_baseline', [NEW_TEST], directory)
        record['all_function_assertion_witnesses'] = (
            len(record['witnesses']) == len(SPECS)
            and all(w['accepted'] and w['assertion_failure'] for w in record['witnesses'])
            and {w['function'] for w in record['witnesses'] if w['accepted']} == set(functions))
        require(record['all_function_assertion_witnesses'] and record['restored'],
                'A planned mechanism survived or failed without its intended assertion.')
        finish('mutations', manifest, record)
    except BaseException as error:
        record['restored'] = guard() == record['guard_before']
        failed(manifest, record, error)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, required=True)
    main(parser.parse_args().attempt)
