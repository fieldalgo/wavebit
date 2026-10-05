"""Audit completed fine-ridge evidence without running or changing a solver.

This is a procedural audit, not an independent physical model or agent review.
It checks raw-result correspondence, imported values, the adaptive job schedule
and agreement of the saved report with the frozen reporter.
"""
from collections import Counter
import json

import fdtd_metasurface_fine as study
from fdtd_polygon_tie_validation import source_hash


def main():
    registration = study.verify()
    report = json.loads(study.REPORT.read_text(encoding='utf-8'))
    ledger = [json.loads(line) for line in study.LOG.read_text(encoding='utf-8').splitlines()
              if line.strip()]
    rows = study.read_rows(study.LOG)
    assert len(rows) == len(ledger), 'Repeated job keys require an explicit attempt review.'
    assert report['complete'] and report['controls_valid']
    assert report['fingerprint'] == registration['fingerprint']
    raw_paths = set()
    imported = 0
    for row in ledger:
        assert row['fingerprint'] == registration['fingerprint']
        assert row.get('status', 'ok') == 'ok'
        if row.get('imported'):
            imported += 1
            origin = row['origin']
            path = study.RESULTS / origin['file']
            assert study.sha(path) == origin['sha256']
            original = study.read_rows(path)[origin['key']]
            for field, value in original.items():
                if field not in ('key', 'job', 'fingerprint'):
                    assert row[field] == value, (row['key'], field)
        else:
            assert row['worker_exit'] == 0
            path = study.JOBS / (row['key'].replace('|', '_') + '_attempt1.json')
            raw = json.loads(path.read_text(encoding='utf-8'))
            assert raw == {k: v for k, v in row.items() if k != 'worker_exit'}
            assert path.with_suffix('.log').is_file()
            raw_paths.add(path)
    assert raw_paths == set(study.JOBS.glob('*.json'))
    assert {p.with_suffix('.log') for p in raw_paths} == set(study.JOBS.glob('*.log'))
    fd_rows = [r for r in ledger if r['kind'] == 'fd']
    assert all(not study.ridge.validity(r) for r in fd_rows)
    # Check every conditional 640 control against the original 320-only proxy.
    controls = {}
    for scene in study.SCENES:
        without_640_controls = {k: r for k, r in rows.items()
                                if not (r['job'][0] == 'fd' and r['job'][2] == 640
                                        and r['job'][4] in ('long', 'pml'))}
        entries = study.cells(without_640_controls, scene)
        for pol in study.POLS:
            need = any(c['pol'] == pol and min(c['changes'][-2:]) > study.SAME
                       and not c['floor_ok'] for c in entries)
            for variant in ('long', 'pml'):
                assert (study.key(study.fd_job(scene, 640, pol, variant)) in rows) == need
            controls[scene + '|' + pol] = need
    # Reconstruct each adaptive decision with only the reference levels that
    # existed before it; later high-N rows must not hide a missed refinement.
    references = []
    for scene in ('Y12', 'Y4', 'D12'):
        for i, wavelength in enumerate(study.WLS):
            levels = []
            for level in (1000, 1200):
                before = {k: r for k, r in rows.items()
                          if not (r['job'][0] == 'rcwa' and r['job'][1:3] == [scene, i]
                                  and r['job'][3] >= level)}
                need = any(c['wavelength'] == wavelength
                           and min(c['errors'][-3:]) > study.SAME
                           and c['reference_spread'] > study.SHARE * min(c['errors'][-3:])
                           for c in study.cells(before, scene))
                assert (study.key(study.ref_job(scene, i, level)) in rows) == need
                if need:
                    levels.append(level)
            entries = [c for c in report['scenes'][scene] if c['wavelength'] == wavelength]
            shares = [c['reference_spread'] / min(c['errors'][-3:]) for c in entries
                      if min(c['errors'][-3:]) > study.SAME]
            references.append(dict(scene=scene, wavelength=wavelength, added_levels=levels,
                                   top=entries[0]['reference_top'], max_share=max(shares)))
    counts = {}
    for scene, entries in report['scenes'].items():
        assert len(entries) == 36 and all(c['status'] == 'ok' for c in entries)
        assert entries == json.loads(json.dumps(study.cells(rows, scene)))
        counts[scene] = dict(self=dict(Counter(c['self_class'] for c in entries)),
                             direct=dict(Counter(c['direct_classes'][-1] for c in entries)))
    print(json.dumps(dict(fingerprint=registration['fingerprint'], fd_source=source_hash(),
                          ledger_rows=len(ledger), imported_rows=imported,
                          new_worker_rows=len(raw_paths), valid_fd_rows=len(fd_rows),
                          replay_gap=report['replay_gap'], controls_640=controls,
                          references=references, counts=counts,
                          audit='passed; raw records and registered procedure agree'), indent=2))


if __name__ == '__main__':
    main()
