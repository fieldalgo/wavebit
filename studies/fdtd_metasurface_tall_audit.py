"""Audit the original height campaign's preserved validity failure and early stop.

This does not turn the incomplete original experiment into a completed ladder.
Its independently valid historical-height check is retained separately.
"""
from collections import Counter
import datetime
import json

import fdtd_metasurface_tall as study


def main():
    reg = study.verify()
    halt = json.loads((study.RESULTS/'fdtd_metasurface_tall_halt.json').read_text(encoding='utf-8-sig'))
    assert halt['status']=='aborted_after_validity_failure'
    registered = datetime.datetime.fromisoformat(reg['registered'])
    ledger = [json.loads(line) for line in study.LOG.read_text().splitlines() if line.strip()]
    old = {p.name: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
           for p in (study.c7.LOG, study.c7.RCWA_LOG)}
    imported, workers, stopped = 0, 0, 0
    attempts, logs = [], []
    for row in ledger:
        assert row['fingerprint'] == reg['fingerprint']
        if row.get('imported'):
            original = next(r for r in reversed(old[row['origin']['file']])
                            if r['key']==row['origin']['key'] and r['prov']==row['prov'])
            assert study.sha(study.RESULTS/row['origin']['file']) == row['origin']['sha256']
            overrides = {'key', 'scene', 'status'}
            assert all(row[k] == v for k,v in original.items() if k not in overrides)
            assert row['values'] == original['values']
            imported += 1
        else:
            logs.append(row['raw_log'])
            if row['status']=='error':
                assert row['key']==halt['stopped_job'] and row['worker_exit']!=0
                assert row['attempt']==1
                assert (study.JOBS/row['raw_log']).exists()
                assert not (study.JOBS/row['raw_log']).with_suffix('.json').exists()
                stopped += 1
                continue
            assert row['worker_exit'] == 0 and row['status'] == 'ok'
            raw = json.loads((study.JOBS/row['raw_file']).read_text())
            assert all(row[k] == v for k,v in raw.items())
            assert datetime.datetime.fromisoformat(row['started']) > registered
            assert datetime.datetime.fromisoformat(row['finished']) > datetime.datetime.fromisoformat(row['started'])
            assert row['attempt'] == 1, 'Review preserved repeated attempts explicitly'
            attempts.append(row['raw_file'])
            workers += 1
    assert set(attempts) == {p.name for p in study.JOBS.glob('*.json')}
    assert set(logs) == {p.name for p in study.JOBS.glob('*.log')}
    assert len(ledger) == len(study.read_rows()), 'Repeated ledger jobs require explicit review'
    assert stopped==1
    rows = study.read_rows()
    failed = rows[halt['failed_job']]
    assert failed['job']==study.fd_job('tall',40,'long') and not study.c7.settled(failed)
    assert study.c7.validity('M3',failed)
    assert study.report() is None and not study.REPORT.exists()
    baseline = study.cells(rows,'base')
    assert len(baseline)==54 and not study.validity(rows,'base')
    assert study.replay_gap(rows)<=study.SAME
    summary = dict(registration=reg['registered'],fingerprint=reg['fingerprint'],
                   ledger_rows=len(ledger),imported_rows=imported,new_workers=workers,
                   intentionally_stopped_workers=stopped,replay_gap=study.replay_gap(rows),
                   original_campaign_complete=False,original_tall_convergence_verdict='withheld',
                   failed_selected_row_validity=study.c7.validity('M3',failed),
                   baseline_self_counts=dict(Counter(c['self_class'] for c in baseline)),
                   baseline_direct_final_counts=dict(Counter(c['direct_classes'][-1] for c in baseline)),
                   baseline_max_noise_share=max(c['noise_share'] for c in baseline),
                   baseline_cells=baseline,
                   scope='Single-agent procedural audit of an incomplete, aborted experiment; not a convergence result or independent review')
    (study.RESULTS/'fdtd_metasurface_tall_abort_report.json').write_text(
        json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='baseline_cells'},indent=2))


if __name__ == '__main__':
    main()
