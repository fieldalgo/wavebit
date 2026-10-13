"""Final-source confirmation after the C8 stencil-door compatibility repair.

This adapter preserves the original registered campaign, all prior responses and
the original canonical mutation evidence. It changes output destinations, adds
final-source/prior-outcome provenance, and normalizes NumPy scalar JSON types.
The imported protocol, worker, numerical analysis and audit criteria are unchanged.
Run mutations only in an explicitly coordinated exclusive production window.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import argparse
from collections import Counter
import hashlib
import json
import pathlib
import re

from studies.fdtd.conductive_film import fdtd_conductive_film_campaign as campaign
from studies.fdtd.conductive_film import fdtd_conductive_film_mutations as mutation
from studies.fdtd.conductive_film import fdtd_conductive_film_probe as probe


ROOT, RESULTS = campaign.ROOT, campaign.RESULTS
OLD_PREFIX = campaign.PREFIX
PREFIX = OLD_PREFIX + '_final_source_v1'
ORIGINAL_REGISTRATION = campaign.REGISTRATION
ORIGINAL_LEDGER = campaign.LEDGER
ORIGINAL_REPORT = campaign.REPORT
MUTATIONS = RESULTS / f'{PREFIX}_mutations_attempt2.json'
MUTATION_DIRECTORY = RESULTS / f'{PREFIX}_mutations_attempt2'
BASELINE = RESULTS / f'{OLD_PREFIX}_controls' / 'targeted_post_compat.log'
ADAPTER = pathlib.Path(__file__)
ORIGINAL_IDENTITY = campaign.identity
ORIGINAL_WRITE_JSON = campaign.write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value, *, exclusive=False):
    """Serialization-only repair: preserve every numerical value and criterion."""
    return ORIGINAL_WRITE_JSON(path, probe._json(value), exclusive=exclusive)


def identity():
    return {**ORIGINAL_IDENTITY(), 'confirmation_adapter_sha256': campaign.sha(ADAPTER),
            'mutation_harness_sha256': campaign.sha(pathlib.Path(mutation.__file__)),
            'test_sha256': campaign.sha(ROOT / mutation.TEST)}


def configure():
    # RESULTS intentionally remains the original root for provenance references.
    campaign.REGISTRATION = RESULTS / f'{PREFIX}_registration.json'
    campaign.LEDGER = RESULTS / f'{PREFIX}_campaign.jsonl'
    campaign.REPORT = RESULTS / f'{PREFIX}_report.json'
    campaign.JOBS = RESULTS / f'{PREFIX}_campaign_jobs'
    campaign.identity = identity
    campaign.write_json = write


def original_provenance():
    original = read(ORIGINAL_REGISTRATION)
    bare = {key: value for key, value in original.items() if key != 'fingerprint'}
    assert hashlib.sha256(campaign.canonical(bare)).hexdigest() == original['fingerprint']
    assert original['protocol'] == campaign.protocol()
    current = ORIGINAL_IDENTITY()
    for key in ('helper_sha256', 'campaign_sha256', 'python', 'executable',
                'backend_versions', 'environment'):
        assert current[key] == original['identity'][key], key
    linked = list(original['known_exposures']) + list(original['frozen_sources'])
    for item in linked:
        assert campaign.sha(RESULTS / item['file']) == item['sha256'], item['file']
    return original, linked


def mutations():
    """The same nine actual assertion witnesses, with final-source provenance."""
    original, linked = original_provenance()
    assert not MUTATIONS.exists() and not MUTATION_DIRECTORY.exists()
    assert re.search(r'54 passed', BASELINE.read_text(encoding='utf-8'))
    MUTATION_DIRECTORY.mkdir()
    baseline = probe.source_identity()
    manifest = {
        'attempt': 2, 'kind': 'final-source confirmation', 'started_utc': campaign.now(),
        'source_identity_before': baseline, 'adapter_sha256': campaign.sha(ADAPTER),
        'harness_sha256': campaign.sha(pathlib.Path(mutation.__file__)),
        'test_sha256': campaign.sha(ROOT / mutation.TEST),
        'baseline_log': str(BASELINE.relative_to(ROOT)),
        'baseline_log_sha256': campaign.sha(BASELINE),
        'original_registration_fingerprint': original['fingerprint'],
        'prior_mutation_manifest': f'studies/results/{OLD_PREFIX}_mutations.json',
        'prior_mutation_manifest_sha256': campaign.sha(RESULTS / f'{OLD_PREFIX}_mutations.json'),
        'scope': 'Same nine mutant definitions and assertion acceptance criteria imported '
                 'from the existing harness. Prior canonical evidence remains unchanged. '
                 'The 54-case pre-mutation baseline includes the existing stencil test; '
                 'the restored baseline runs the unchanged 53-case new test file.',
        'witnesses': [], 'restored': False,
    }
    write(MUTATIONS, manifest, exclusive=True)
    for name, relative, before, after, target in mutation.mutations():
        assert probe.source_identity() == baseline, 'source changed outside exclusive window'
        source = ROOT / relative
        original_bytes = source.read_bytes()
        decoded = original_bytes.decode('utf-8').replace('\r\n', '\n')
        assert decoded.count(before) == 1, (name, decoded.count(before))
        changed = decoded.replace(before, after)
        if b'\r\n' in original_bytes:
            changed = changed.replace('\n', '\r\n')
        changed = changed.encode('utf-8')
        before_path = MUTATION_DIRECTORY / f'{name}.original.txt'
        after_path = MUTATION_DIRECTORY / f'{name}.mutant.txt'
        log = MUTATION_DIRECTORY / f'{name}.log'
        before_path.write_bytes(original_bytes)
        after_path.write_bytes(changed)
        print(f'{campaign.now()} START {name}', flush=True)
        try:
            source.write_bytes(changed)
            code, elapsed, command = mutation.pytest_run(
                [mutation.TEST + '::' + target], log, f'final-source-attempt2-{name}')
        finally:
            source.write_bytes(original_bytes)
            assert source.read_bytes() == original_bytes, f'byte restoration failed: {name}'
        text = log.read_text(encoding='utf-8')
        assertion = mutation.assertion_failure(text)
        accepted = (code == 1 and assertion and '1 failed' in text
                    and not re.search(r'\d+ error(?:s)?(?:,| in|\s*$)', text, flags=re.MULTILINE))
        manifest['witnesses'].append({
            'name': name, 'production_file': relative, 'test': target,
            'replacement': {'before': before, 'after': after},
            'original_sha256': mutation.digest(original_bytes),
            'mutant_sha256': mutation.digest(changed), 'restored_sha256': campaign.sha(source),
            'original_snapshot': str(before_path.relative_to(ROOT)),
            'mutant_snapshot': str(after_path.relative_to(ROOT)),
            'raw_log': str(log.relative_to(ROOT)), 'raw_log_sha256': campaign.sha(log),
            'exit_code': code, 'wall_s': elapsed, 'command': command,
            'assertion_failure': assertion, 'accepted': accepted,
        })
        manifest['restored'] = probe.source_identity() == baseline
        write(MUTATIONS, manifest)
        print(f'{campaign.now()} END {name} exit={code} accepted={accepted} '
              f'restored={manifest["restored"]}', flush=True)
    final = MUTATION_DIRECTORY / 'restored_targeted_baseline.log'
    code, elapsed, command = mutation.pytest_run(
        [mutation.TEST], final, 'final-source-attempt2-restored')
    manifest['final_baseline'] = {
        'raw_log': str(final.relative_to(ROOT)), 'raw_log_sha256': campaign.sha(final),
        'exit_code': code, 'wall_s': elapsed, 'command': command,
    }
    manifest['source_identity_after'] = probe.source_identity()
    manifest['restored'] = manifest['source_identity_after'] == baseline
    manifest['all_nine_assertion_witnesses'] = (
        len(manifest['witnesses']) == 9 and all(w['accepted'] for w in manifest['witnesses']))
    manifest['passed'] = bool(manifest['all_nine_assertion_witnesses'] and manifest['restored']
                              and code == 0 and re.search(r'53 passed', final.read_text()))
    manifest['ended_utc'] = campaign.now()
    for item in linked:
        assert campaign.sha(RESULTS / item['file']) == item['sha256'], item['file']
    write(MUTATIONS, manifest)
    return {'passed': manifest['passed'], 'restored': manifest['restored'],
            'accepted_witnesses': sum(w['accepted'] for w in manifest['witnesses']),
            'manifest': str(MUTATIONS.relative_to(ROOT))}


def mutation_evidence():
    manifest = read(MUTATIONS)
    assert manifest['passed'] and manifest['restored'] and manifest['all_nine_assertion_witnesses']
    assert manifest['source_identity_before'] == manifest['source_identity_after'] \
        == probe.source_identity()
    assert manifest['adapter_sha256'] == campaign.sha(ADAPTER)
    assert manifest['harness_sha256'] == campaign.sha(pathlib.Path(mutation.__file__))
    assert manifest['test_sha256'] == campaign.sha(ROOT / mutation.TEST)
    assert campaign.sha(ROOT / manifest['baseline_log']) == manifest['baseline_log_sha256']
    files = [MUTATIONS, ROOT / manifest['baseline_log']]
    definitions = {item[0]: item[1:] for item in mutation.mutations()}
    for item in manifest['witnesses']:
        relative, before, after, target = definitions[item['name']]
        assert item['production_file'] == relative and item['test'] == target
        assert item['replacement'] == {'before': before, 'after': after}
        original_path, mutant_path = ROOT / item['original_snapshot'], ROOT / item['mutant_snapshot']
        original_bytes, mutant_bytes = original_path.read_bytes(), mutant_path.read_bytes()
        assert campaign.sha(original_path) == item['original_sha256'] == item['restored_sha256']
        assert campaign.sha(ROOT / relative) == item['restored_sha256']
        assert campaign.sha(mutant_path) == item['mutant_sha256']
        decoded = original_bytes.decode('utf-8').replace('\r\n', '\n')
        assert decoded.count(before) == 1
        assert decoded.replace(before, after) == mutant_bytes.decode('utf-8').replace('\r\n', '\n')
        raw = ROOT / item['raw_log']
        assert campaign.sha(raw) == item['raw_log_sha256']
        text = raw.read_text(encoding='utf-8')
        assert item['exit_code'] == 1 and item['accepted'] and mutation.assertion_failure(text)
        assert '1 failed' in text
        assert not re.search(r'\d+ error(?:s)?(?:,| in|\s*$)', text, flags=re.MULTILINE)
        files.extend((original_path, mutant_path, raw))
    assert len(manifest['witnesses']) == len({w['test'].split('[')[0] for w in manifest['witnesses']}) == 9
    final = manifest['final_baseline']
    log = ROOT / final['raw_log']
    assert final['exit_code'] == 0 and campaign.sha(log) == final['raw_log_sha256']
    assert re.search(r'53 passed', log.read_text(encoding='utf-8'))
    return files + [log]


def register():
    assert not any(path.exists() for path in
                   (campaign.REGISTRATION, campaign.LEDGER, campaign.REPORT, campaign.JOBS))
    original, known = original_provenance()
    linked = {item['file']: item for item in known}

    def add(path, scope):
        filename = str(path.relative_to(RESULTS))
        linked[filename] = {'file': filename, 'sha256': campaign.sha(path), 'scope': scope}

    for path in (ORIGINAL_REGISTRATION, ORIGINAL_LEDGER, ORIGINAL_REPORT):
        add(path, 'Original campaign completed before this confirmation; outcomes are known.')
    prior_entries = [json.loads(line) for line in ORIGINAL_LEDGER.read_text().splitlines()]
    assert len(prior_entries) == 20 and all(entry['exit_code'] == 0 for entry in prior_entries)
    for entry in prior_entries:
        for name in ('raw_log', 'result'):
            path = ROOT / entry[name]
            assert campaign.sha(path) == entry[name + '_sha256']
            add(path, 'Known original campaign worker; no unseen-outcome claim is made.')
    for path in sorted(RESULTS.glob(f'{OLD_PREFIX}_report_recovery*')):
        if path.is_file():
            add(path, 'Original controller JSON failure and serialization-only recovery evidence.')
    for path in (RESULTS / f'{OLD_PREFIX}_pre_compat_fix.py.txt',
                 RESULTS / f'{OLD_PREFIX}_validation_attempt1.json',
                 RESULTS / f'{OLD_PREFIX}_controls' / 'full_attempt1.log',
                 RESULTS / f'{OLD_PREFIX}_controls' / 'full_collection_attempt1.log'):
        add(path, 'Prior full validation found one stencil refusal-message compatibility defect.')
    for path in mutation_evidence():
        add(path, 'Final-source nine assertion witnesses and exactly restored baselines.')
    summary = read(RESULTS / f'{OLD_PREFIX}_report_recovery_summary.json')
    prior = read(ORIGINAL_REPORT)
    assert prior['complete'] and len(prior['cells']) == 24
    assert summary['self_counts'] == dict(Counter(cell['self_order']['classification']
                                                 for cell in prior['cells']))
    assert summary['direct_final_counts'] == dict(Counter(
        cell['direct_orders'][-1]['classification'] for cell in prior['cells']))
    snapshot = RESULTS / f'{PREFIX}_registered_adapter.py.txt'
    with snapshot.open('xb') as stream:
        stream.write(ADAPTER.read_bytes())
    data = {
        'registered_utc': campaign.now(), 'identity': identity(),
        'protocol': campaign.protocol(),
        'analytic_preflight': {**probe.preflight(), **campaign.order_preflight()},
        'known_exposures': list(linked.values()),
        'frozen_sources': [{'file': snapshot.name, 'sha256': campaign.sha(snapshot)}],
        'original_registration_fingerprint': original['fingerprint'],
        'known_original_outcomes': summary,
        'qualification': 'Final-source confirmation with all original 20 outcomes known. '
                         'The only production change after the original campaign scopes the '
                         'unsupported prepared-plan conductor guard to voxel scenes, preserving '
                         'the existing stencil refusal. No formulas or physical criteria change. '
                         'The original controller, helper, protocol, rows, registration and '
                         'mutation evidence remain unmodified. The adapter redirects artifact '
                         'paths and applies probe._json before JSON writing to repair the known '
                         'NumPy-bool serialization failure without changing values. This is '
                         'a confirmation, not a fresh unseen-outcome characterization.',
    }
    data['fingerprint'] = hashlib.sha256(campaign.canonical(data)).hexdigest()
    write(campaign.REGISTRATION, data, exclusive=True)
    campaign.verify()
    return {'registered': str(campaign.REGISTRATION.relative_to(ROOT)),
            'fingerprint': data['fingerprint'], 'known_original_workers': len(prior_entries)}


def compare_and_audit():
    audited = campaign.audit()
    mutation_evidence()
    old_entries = {entry['key']: entry for entry in
                   (json.loads(line) for line in ORIGINAL_LEDGER.read_text().splitlines())}
    new_entries = {entry['key']: entry for entry in campaign.ledger()}
    assert old_entries.keys() == new_entries.keys(), 'confirmation and original job sets differ'
    fields = ('r', 't', 'R', 'T', 'absorption', 'error_r', 'error_t', 'decay', 'energy_max',
              'energy_finite', 'reference_r', 'reference_t', 'reference_absorption',
              'stable', 'dt', 'steps', 'grid', 'config', 'source_band', 'wavelengths')
    comparisons = []
    for key, entry in new_entries.items():
        old, new = read(ROOT / old_entries[key]['result']), read(ROOT / entry['result'])
        agreement = {name: old[name] == new[name] for name in fields}
        # Canonical JSON equality also distinguishes signed zero, which Python == does not.
        serialized = campaign.canonical({name: old[name] for name in fields}) \
            == campaign.canonical({name: new[name] for name in fields})
        comparisons.append({'key': key, 'exact_numeric_agreement': agreement,
                            'canonical_physical_payload_identical': serialized,
                            'original_result_sha256': old_entries[key]['result_sha256'],
                            'confirmation_result_sha256': entry['result_sha256']})
    report = read(campaign.REPORT)
    cells = report['cells']
    complete = [cell for cell in cells if cell['resolved_input']]
    result = {**audited, 'audited_utc': campaign.now(),
              'registration': campaign.verify()['fingerprint'],
              'comparison_scope': 'Decoded numeric equality and identical canonical JSON for '
                                  'the listed physical fields; source hashes, timestamps and '
                                  'wall time are intentionally separate metadata.',
              'matched_workers': len(comparisons), 'comparisons': comparisons,
              'all_physical_payloads_identical': all(
                  item['canonical_physical_payload_identical'] for item in comparisons),
              'self_counts': dict(Counter(cell['self_order']['classification'] for cell in cells)),
              'direct_final_counts': dict(Counter(
                  cell['direct_orders'][-1]['classification'] for cell in complete)),
              'self_point_range': [min(cell['self_order']['point'] for cell in complete),
                                   max(cell['self_order']['point'] for cell in complete)],
              'direct_final_point_range': [
                  min(cell['direct_orders'][-1]['point'] for cell in complete),
                  max(cell['direct_orders'][-1]['point'] for cell in complete)],
              'max_proxy': max(cell['proxy'] for cell in complete),
              'max_noise_share': max(2*cell['proxy']/min(cell['differences']) for cell in complete),
              'max_errors_by_rung': {str(n): max(cell['errors'][i] for cell in complete)
                                     for i, n in enumerate(campaign.RUNGS)}}
    destination = RESULTS / f'{PREFIX}_audit.json'
    if destination.exists():
        raise RuntimeError('Audit already published; preserve it instead of overwriting.')
    write(destination, result, exclusive=True)
    return {key: value for key, value in result.items() if key != 'comparisons'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('mutations', 'register', 'run', 'audit'))
    args = parser.parse_args()
    configure()
    functions = {'mutations': mutations, 'register': register,
                 'run': campaign.run, 'audit': compare_and_audit}
    result = functions[args.action]()
    print(json.dumps(probe._json(result), indent=2, allow_nan=False))
    if args.action == 'mutations' and not result['passed']:
        raise SystemExit(1)
    if args.action == 'run' and not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
