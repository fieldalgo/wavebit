"""Descriptive summaries of every frozen height-study cell; no new pass rule."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from collections import Counter
import json

import studies.cross_solver.metasurface_tall.fdtd_metasurface_tall_settled as study


def main():
    study.verify()
    result = json.loads(study.REPORT.read_text())
    assert result['complete']
    rows = study.parent.read_rows(study.LOG)
    summary = dict(scope=__doc__,controls_valid=result['controls_valid'],scenes={})
    for name,entries in result['scenes'].items():
        good = [c for c in entries if c['status']=='ok']
        orders = [c['self_order'] for c in good if c['self_order'] is not None]
        per_wavelength = []
        for wl in study.c7.WAVELENGTHS:
            group = [c for c in good if c['wavelength']==wl]
            per_wavelength.append(dict(wavelength=wl,
                self_counts=dict(Counter(c['self_class'] for c in group)),
                direct_final_counts=dict(Counter(c['direct_classes'][-1] for c in group))))
        summary['scenes'][name] = dict(
            self_counts=dict(Counter(c['self_class'] for c in entries)),
            direct_final_counts=dict(Counter(c['direct_classes'][-1] for c in entries)),
            max_noise_share=max((c['noise_share'] for c in good if c['noise_share'] is not None),default=None),
            noise_gate_failures=sum(not c['floor_ok'] for c in good),
            self_order_range=[min(orders),max(orders)] if orders else None,
            reference_ladder=result['reference_ladders'][name],per_wavelength=per_wavelength)
    transitions = Counter()
    for a,b in zip(result['scenes']['base'],result['scenes']['tall']):
        transitions[a['self_class']+' -> '+b['self_class']] += 1
    summary['class_transitions'] = dict(transitions)
    ds = [c['interval'] for c in result['delta_intervals'] if c['interval'] is not None]
    summary['delta_intervals'] = dict(resolved=len(ds),strictly_positive=sum(d[0]>0 for d in ds),
                                     strictly_negative=sum(d[1]<0 for d in ds),include_zero=sum(d[0]<=0<=d[1] for d in ds))
    fresh = [r for r in rows.values() if not r.get('imported')]
    fd = [r for r in rows.values() if r['kind']=='fd' and r['status']=='ok']
    summary['work'] = dict(new_workers=len(fresh),worker_seconds=sum(r.get('wall',0) for r in fresh),
                          new_fd_seconds=sum(r.get('wall',0) for r in fresh if r['kind']=='fd'),
                          new_rcwa_seconds=sum(r.get('wall',0) for r in fresh if r['kind']=='rcwa'),
                          raw_max_abs_energy_residual=max(abs(b) for r in fd for b in r['balance']),
                          selected_max_abs_energy_residual=max(abs(b) for name in study.HEIGHTS for n in study.RUNGS
                                                               for b in study.selected(rows,name,n)['balance']))
    output = json.dumps(summary,indent=2,allow_nan=False)+'\n'
    (study.RESULTS/'fdtd_metasurface_tall_settled_summary.json').write_text(output,encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
