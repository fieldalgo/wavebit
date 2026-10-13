"""Audit imported evidence, raw attempts and all calibrated height-study cells."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
import datetime
import json

import studies.cross_solver.metasurface_tall.fdtd_metasurface_tall_settled as study


def calibration_audit():
    """Recompute the calibration record without rewriting a frozen input."""
    cal, parent = study.calibration, study.parent
    reg = cal.verify()
    ledger = [json.loads(s) for s in cal.LOG.read_text().splitlines() if s.strip()]
    assert len(ledger)==len(cal.PERIODS)==len(parent.read_rows(cal.LOG))
    assert {r['periods'] for r in ledger}==set(cal.PERIODS)
    raw_files, logs = set(),set()
    rows = {p:json.loads(f.read_text()) for p,f in zip((120,180),cal.INPUTS[-2:])}
    for row in ledger:
        assert row['fingerprint']==reg['fingerprint'] and row['status']=='ok'
        assert row['worker_exit']==0 and row['attempt']==1
        assert datetime.datetime.fromisoformat(reg['registered']) < datetime.datetime.fromisoformat(
            row['started']) < datetime.datetime.fromisoformat(row['finished'])
        raw = json.loads((cal.JOBS/row['raw_file']).read_text())
        assert all(row[k]==v for k,v in raw.items())
        raw_files.add(row['raw_file'])
        logs.add(row['raw_log'])
        rows[row['periods']] = row
    assert raw_files=={p.name for p in cal.JOBS.glob('*.json')}
    assert logs=={p.name for p in cal.JOBS.glob('*.log')}
    report_path = study.RESULTS/(cal.PREFIX+'_report.json')
    before = parent.sha(report_path)
    saved = json.loads(report_path.read_text())
    assert saved['fingerprint']==reg['fingerprint'] and saved['spatial_order_verdict'] is None
    assert saved['periods']==sorted(rows) and len(saved['cells'])==54
    vals = {p:parent.values(row,40) for p,row in rows.items()}
    expected_keys = {(obs,wl) for obs in study.c7.cell_keys('M3') for wl in study.c7.WAVELENGTHS}
    assert {(c['observable'],c['wavelength']) for c in saved['cells']}==expected_keys
    for cell in saved['cells']:
        k = cell['observable'],list(study.c7.WAVELENGTHS).index(cell['wavelength'])
        sequence = [vals[p][k] for p in saved['periods']]
        assert cell['values']==[[v.real,v.imag] for v in sequence]
        assert cell['changes']==[abs(a-b) for a,b in zip(sequence,sequence[1:])]
    assert saved['max_changes']==[max(c['changes'][i] for c in saved['cells']) for i in range(4)]
    for p,row in rows.items():
        assert saved['rows'][str(p)]==dict(validity=study.c7.validity('M3',row),decay=row['decay'],
            max_energy_residual=max(map(abs,row['balance'])),wall=row['wall'])
    assert parent.sha(report_path)==before
    return dict(new_workers=len(ledger),cell_records=54,frozen_report_unchanged=True)


def main():
    reg = study.verify()
    calibration = calibration_audit()
    registered = datetime.datetime.fromisoformat(reg['registered'])
    ledger = [json.loads(s) for s in study.LOG.read_text().splitlines() if s.strip()]
    sources = {p.name:study.parent.read_rows(p) for p in (study.parent.LOG,study.calibration.LOG)}
    imported,completed = 0,0
    raw_files,logs = set(),set()
    for row in ledger:
        assert row['fingerprint']==reg['fingerprint'] and row['status']=='ok'
        if row.get('imported'):
            origin = row['origin']
            assert study.parent.sha(study.RESULTS/origin['file'])==origin['sha256']
            original = sources[origin['file']][origin['key']]
            assert all(row[k]==v for k,v in original.items()
                       if k not in ('key','job','scene','origin','imported','fingerprint'))
            assert row['values']==original['values']
            imported += 1
        else:
            assert row['worker_exit']==0 and row['attempt']==1
            raw = json.loads((study.JOBS/row['raw_file']).read_text())
            assert all(row[k]==v for k,v in raw.items())
            assert datetime.datetime.fromisoformat(row['started'])>registered
            assert datetime.datetime.fromisoformat(row['finished'])>datetime.datetime.fromisoformat(row['started'])
            raw_files.add(row['raw_file'])
            logs.add(row['raw_log'])
            completed += 1
    assert len(ledger)==len(study.parent.read_rows(study.LOG))
    assert raw_files=={p.name for p in study.JOBS.glob('*.json')}
    assert logs=={p.name for p in study.JOBS.glob('*.log')}
    saved = json.loads(study.REPORT.read_text())
    got = study.report(saved.get('stop_reason'))
    assert saved==json.loads(json.dumps(got))
    assert len(got['scenes']['base'])==len(got['scenes']['tall'])==54
    if not got['complete']:
        reason = got['stop_reason']
        assert reason, 'Incomplete execution without a registered stop; do not claim completion'
        last = [r for r in ledger if not r.get('imported')][-1]
        assert last['job']==reason['job']
        assert study.c7.validity('M3',last)==reason['invalid']
    result = dict(fingerprint=reg['fingerprint'],imported_rows=imported,new_workers=completed,
                  all_108_cell_records_recomputed=True,comparison_complete=got['complete'],
                  calibration_audit=calibration,
                  registered_stop=got.get('stop_reason'),inherited_replay_gap=got['inherited_replay_gap'],
                  self_counts=got['self_counts'],
                  scope='Single-agent procedural audit; not independent physics or agent review')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
