"""Recover only the serialization of the immutable registered C8 report.

The registered campaign, probe, production sources and every worker response remain
unchanged. The original write_json failure is reproduced into a separate preserved
file, then the original make_report result is normalized with the original probe._json.
No criterion, numeric value, field response, or solver invocation is changed.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import pathlib
import traceback

from studies import fdtd_conductive_film_campaign as campaign
from studies import fdtd_conductive_film_probe as probe


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(probe._json(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def evidence():
    paths = [campaign.REGISTRATION, campaign.LEDGER,
             pathlib.Path(campaign.__file__), pathlib.Path(probe.__file__)]
    paths.extend(sorted(campaign.JOBS.iterdir()))
    return {str(path.relative_to(campaign.ROOT)): sha(path) for path in paths}


def summary(report):
    cells = report['cells']
    out = {'complete': report['complete'], 'workers': report['worker_rows'],
           'cell_records': len(cells),
           'self_counts': dict(Counter(cell['self_order']['classification'] for cell in cells)),
           'direct_final_counts': dict(Counter(cell['direct_orders'][-1]['classification']
                                               for cell in cells)),
           'self_point_range': [min(cell['self_order']['point'] for cell in cells),
                                max(cell['self_order']['point'] for cell in cells)],
           'direct_final_point_range': [min(cell['direct_orders'][-1]['point'] for cell in cells),
                                        max(cell['direct_orders'][-1]['point'] for cell in cells)],
           'largest_noise_proxy': max(cell['proxy'] for cell in cells),
           'largest_noise_share': max(2*cell['proxy']/min(cell['differences']) for cell in cells),
           'errors_by_rung': {str(n): max(cell['errors'][i] for cell in cells)
                             for i, n in enumerate(campaign.RUNGS)},
           'groups': []}
    for fraction in campaign.FRACTIONS:
        for pol in campaign.POLS:
            group = [cell for cell in cells if cell['fraction'] == fraction
                     and cell['polarization'] == pol]
            out['groups'].append({
                'fraction': fraction, 'polarization': pol,
                'self_counts': dict(Counter(cell['self_order']['classification'] for cell in group)),
                'direct_final_counts': dict(Counter(cell['direct_orders'][-1]['classification']
                                                    for cell in group)),
                'self_points': {f'{cell["coefficient"]}@{cell["wavelength"]}':
                                cell['self_order']['point'] for cell in group},
                'max_errors_by_rung': {str(n): max(cell['errors'][i] for cell in group)
                                      for i, n in enumerate(campaign.RUNGS)}})
    return out


def recover(controller_log=None):
    reg = campaign.verify()
    before = evidence()
    root = campaign.RESULTS
    prefix = 'fdtd_conductive_film_report_recovery'
    backup = root / f'{prefix}_original_partial.json.txt'
    partial = campaign.REPORT.read_bytes()
    with backup.open('xb') as stream:
        stream.write(partial)
    try:
        json.loads(partial)
    except json.JSONDecodeError as error:
        partial_error = str(error)
    else:
        raise RuntimeError('The report already parses; this utility only recovers the preserved failure.')
    rebuilt = campaign.make_report()
    assert rebuilt['complete'] and rebuilt['worker_rows'] == 20
    assert all(entry['exit_code'] == 0 for entry in campaign.ledger())
    reproduction = root / f'{prefix}_reproduced_partial.json.txt'
    failure_log = root / f'{prefix}_serialization_failure.log'
    try:
        campaign.write_json(reproduction, rebuilt, exclusive=True)
    except TypeError as error:
        assert 'bool' in str(error) and 'JSON serializable' in str(error)
        serialization_error = str(error)
        with failure_log.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write('Faithful replay of original campaign.write_json on original make_report.\n')
            stream.write('Serialization only; no worker or field solve was invoked.\n\n')
            stream.write(traceback.format_exc())
    else:
        raise AssertionError('The registered serialization failure did not reproduce.')
    assert reproduction.read_bytes() == partial, 'reproduced partial bytes differ from original failure'
    native = probe._json(rebuilt)
    assert native == rebuilt, 'scalar normalization changed a report value'
    encoded = json.dumps(native, indent=2, allow_nan=False) + '\n'
    assert json.loads(encoded) == rebuilt
    # The original partial file is now preserved byte-for-byte above; only its completed
    # presentation replaces the canonical report path.
    with campaign.REPORT.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)
    audit = campaign.audit()
    assert audit['audit'] == 'passed' and audit['complete']
    assert evidence() == before, 'immutable source, registration, ledger, or raw worker evidence moved'
    stats = summary(native)
    controller = None
    if controller_log is not None:
        controller_log = pathlib.Path(controller_log).resolve()
        controller = {'path': str(controller_log), 'sha256': sha(controller_log),
                      'scope': 'Original controller failure log, unchanged.'}
    manifest = {
        'recovered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'utility_sha256': sha(pathlib.Path(__file__)), 'registration': reg['fingerprint'],
        'source_identity': probe.source_identity(), 'field_solves': 0, 'worker_reruns': 0,
        'scope': 'Serialization recovery only. Original criteria and all values unchanged. '
                 'The NumPy bool in order.resolved is converted to the equal native bool '
                 'by the already registered probe._json helper.',
        'original_controller_log': controller,
        'controller_provenance_limitation': {
            'scope': 'Original controller output exists only in the tool transcript; there '
                     'was no controller log file. The replay log below is a reproduction, '
                     'not a claimed copy of that original output.',
            'exec_session': 80517, 'completion_chunk': '293a6f', 'controller_exit_code': 1,
            'last_worker_end_utc': '2026-10-06T05:16:58.575236+00:00',
            'reported_exception_location': 'json.encoder.py:180'},
        'original_controller_failure': {'type': 'TypeError', 'message': serialization_error,
                                        'all_20_worker_exit_codes': [e['exit_code'] for e in campaign.ledger()]},
        'partial_json_parse_error': partial_error,
        'original_partial': {'file': str(backup.relative_to(campaign.ROOT)),
                             'sha256': sha(backup), 'bytes': len(partial)},
        'faithful_reproduction': {'partial': str(reproduction.relative_to(campaign.ROOT)),
                                  'partial_sha256': sha(reproduction),
                                  'identical_to_original_partial': True,
                                  'failure_log': str(failure_log.relative_to(campaign.ROOT)),
                                  'failure_log_sha256': sha(failure_log)},
        'report': {'file': str(campaign.REPORT.relative_to(campaign.ROOT)),
                   'sha256': sha(campaign.REPORT)},
        'immutable_evidence_sha256': before, 'original_audit': audit, 'summary': stats,
    }
    write(root / f'{prefix}.json', manifest)
    write(root / f'{prefix}_summary.json', stats)
    print(json.dumps({'recovery': 'passed', 'original_audit': audit, 'summary': stats},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-log', type=pathlib.Path)
    recover(parser.parse_args().controller_log)
