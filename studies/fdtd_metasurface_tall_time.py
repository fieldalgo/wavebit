"""Time calibration after the bounded tall-box campaign failed its 40-cell gate.

Known before this calibration: original tall 40-cell decays at 120 periods
(r=3.771241535e-4, t=5.342549338e-5) and 180 periods
(r=5.108708166e-5, t=7.159252215e-6). The registered line is 1e-5.
The original 180-period cap has failed. Its ongoing 80-cell worker was stopped;
the original ladder remains incomplete and has no tall convergence verdict.
All original source, results, failed/stopped logs and registration are preserved.

This separate calibration fixes three additional durations, 240/360/540, BEFORE
measuring them, at the same tall scene, s incidence, n=40, PML=24. It measures
time settling and all 54 phase-corrected outputs. It is not a spatial-order
experiment and does not rescue the original protocol. No tolerance is widened.
Report C7 validity at each time and every change between adjacent durations,
including the already known 120/180 rows. A future spatial experiment must
disclose these data and register its own durations before new grid data. The
duration choice is calibration-informed, not a preregistered result of the
original height study. Workers are sequential fresh Python processes.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import fdtd_metasurface_tall as height


ROOT, RESULTS = height.ROOT, height.RESULTS
PREFIX = 'fdtd_metasurface_tall_time'
REG = RESULTS/(PREFIX+'_registration.json')
LOG = RESULTS/(PREFIX+'.jsonl')
JOBS = RESULTS/(PREFIX+'_jobs')
PERIODS = (240,360,540)
INPUTS = [height.REG,height.LOG,RESULTS/'fdtd_metasurface_tall_halt.json',
          height.JOBS/'fd_tall_40_base_attempt1.json',height.JOBS/'fd_tall_40_long_attempt1.json']


def identity():
    return dict(source=height.sha(__file__),parent=height.identity(),
                inputs={p.relative_to(ROOT).as_posix():height.sha(p) for p in INPUTS})


def verify():
    height.verify()
    reg = json.loads(REG.read_text())
    assert reg['identity']==identity()
    assert height.sha(RESULTS/reg['copy'])==reg['identity']['source']
    return reg


def register():
    if REG.exists():
        verify()
        print('Existing time-calibration registration verified.')
        return
    height.verify()
    assert not LOG.exists() and not JOBS.exists()
    known = json.loads(INPUTS[-1].read_text())
    assert not height.c7.settled(known)
    ident = identity()
    fingerprint = hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()
    copy = RESULTS/(PREFIX+'_registered_'+ident['source'][:16]+'.py.txt')
    with copy.open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    reg = dict(registered=height.timestamp(),identity=ident,fingerprint=fingerprint,
               copy=copy.name,protocol=__doc__,periods=PERIODS,original_campaign='aborted, not reclassified')
    with REG.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(reg,indent=2)+'\n')
    print('REGISTERED',reg['registered'],fingerprint,flush=True)


def worker(periods,output):
    reg = verify()
    started,stamp = time.perf_counter(),height.timestamp()
    with height.configure('tall'):
        height.c7.PROV = height.c7.provenance()
        height.c7.PROV['digest'] = reg['fingerprint']
        row = height.c7.fd_row(height.c7.spec('M3',40,'s',periods))
    row.update(key=f'fd|tall|40|{periods}',periods=periods,scene='tall',started=stamp,
               finished=height.timestamp(),wall=time.perf_counter()-started,
               fingerprint=reg['fingerprint'],metadata_dirty_scope='wavebit source tree only')
    Path(output).write_text(json.dumps(row,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:row[k] for k in ('key','status','wall','decay') if k in row}),flush=True)
    return 0 if row['status']=='ok' else 1


def run():
    reg = verify()
    JOBS.mkdir(exist_ok=True)
    for periods in PERIODS:
        rows = height.read_rows(LOG)
        if (old:=rows.get(f'fd|tall|40|{periods}')) and old['status']=='ok':
            continue
        attempt = 1
        while (JOBS/f'periods_{periods}_attempt{attempt}.log').exists():
            attempt += 1
        output = JOBS/f'periods_{periods}_attempt{attempt}.json'
        logfile = output.with_suffix('.log')
        print(height.timestamp(),'START',periods,flush=True)
        with logfile.open('x',encoding='utf-8') as stream:
            got = subprocess.run([sys.executable,__file__,'worker',str(periods),str(output)],
                                 cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        row = json.loads(output.read_text()) if output.exists() else dict(
            key=f'fd|tall|40|{periods}',periods=periods,status='error',
            error='Worker exited without a result',fingerprint=reg['fingerprint'])
        row.update(worker_exit=got.returncode,attempt=attempt,raw_file=output.name,raw_log=logfile.name)
        with LOG.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        assert got.returncode==0 and row['status']=='ok', logfile
        print(height.timestamp(),'END',periods,row['wall'],flush=True)
    report()


def report():
    reg = verify()
    rows = {p:json.loads(f.read_text()) for p,f in zip((120,180),INPUTS[-2:])}
    new = height.read_rows(LOG)
    for p in PERIODS:
        row = new[f'fd|tall|40|{p}']
        assert row['status']=='ok' and row['worker_exit']==0
        assert row['fingerprint']==reg['fingerprint']
        assert datetime.datetime.fromisoformat(row['started']) > datetime.datetime.fromisoformat(reg['registered'])
        raw = json.loads((JOBS/row['raw_file']).read_text())
        assert all(row[k]==v for k,v in raw.items())
        rows[p] = row
    ps = sorted(rows)
    vals = {p:height.values(r,40) for p,r in rows.items()}
    cells = [dict(observable=obs,wavelength=height.c7.WAVELENGTHS[i],
                  changes=[abs(vals[a][obs,i]-vals[b][obs,i]) for a,b in zip(ps,ps[1:])],
                  values=[[vals[p][obs,i].real,vals[p][obs,i].imag] for p in ps])
             for obs,i in vals[120]]
    result = dict(fingerprint=reg['fingerprint'],scope=__doc__,periods=ps,cells=cells,
                  rows={p:dict(validity=height.c7.validity('M3',r),decay=r['decay'],
                               max_energy_residual=max(map(abs,r['balance'])),wall=r['wall']) for p,r in rows.items()},
                  max_changes=[max(c['changes'][j] for c in cells) for j in range(len(ps)-1)],
                  spatial_order_verdict=None)
    (RESULTS/(PREFIX+'_report.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','scope')},indent=2),flush=True)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        raise SystemExit(worker(int(sys.argv[2]),sys.argv[3]))
    for command in sys.argv[1:] or ['report']:
        dict(register=register,run=run,report=report)[command]()
