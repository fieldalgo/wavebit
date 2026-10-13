"""Time-calibrated, bounded height comparison on the common 40/60/80 ladder.

This is a NEW protocol, not a successful rerun of the failed 40/80/120 study.
Known data: original baseline rows/references and its new absorber control;
tall RCWA N=12/16/20/24; tall n=40 at 120/180 (invalid) and the separately
registered 240/360/540 time calibration. The original n=80 worker was stopped
without a result. No tall n=60 or n=80 response was available for registration.

At n=40 the 540-period reflection decay is 3.539264e-10 and the largest energy
residual is 6.566713e-5. The largest 360-to-540 cell move is 8.529044e-4.
These calibration data motivate 540/720 periods; they do not prove the time
error at finer grids, which the registered 60-cell control must measure.

The original 180-period cap failed at n=40. To bound cost after increasing
duration, BOTH heights now use 40/60/80, not 40/80/120. This tests a coarser
range and cannot supply the missing 120-cell result. Geometry, face fractions,
fixed-k source, wavelengths, s incidence and all 54 cells per height stay as
before. The baseline needs a new 60-cell run. The valid tall 540-period 40-cell
calibration row is reused explicitly; its values were already known.

Durations: baseline 120 periods, tall 540. A merely unsettled base run gets ONE
replacement at respectively 180 or 720. Stop immediately if the selected run
still fails C7 validity (1e-5 decay, finite complete values, angle <=1e-9, stable,
no growth/unsettled warning). A registered stop is an unresolved comparison,
not evidence of a spatial order. Do not spend later large runs on an invalid
ladder, silently extend the cap, or relax any tolerance.

Time/absorber controls are at n=60. For EACH height run its longer duration
(180 or 720) with 24 PML cells, and that SAME longer duration with 48 PML cells.
Both controls must settle; no extra control extension. Per-cell empirical
noise = |long24-base24| + |long48-long24|. This separates time and PML changes
at matched times. The proxy at 60 is not a rigorous bound on n=80 error.

Primary self-order inverts |v40-v60|/|v60-v80| using the unequal-rung one-power
ratio, with its zero limit and negative orders retained. Twice the noise must
be <= .1 of the smaller grid change, and widens each change by +/-2*noise.
The same-number line remains 1e-10. Inherited interval classes are second
inside [1.8,2.3], low if upper <=1.75, mixed otherwise, unresolved on a failed
gate/positivity/range. Print all cells and descriptive tall-minus-baseline
intervals. Three points and a noise proxy do not prove an asymptotic power.

Reuse baseline references N=20/26/28/30/32 and tall N=12/16/20/24 under hashes.
Spread is max distance of the previous three levels from the top. Direct orders
need spread <= .02 of both errors and the noise/positivity gates. An unresolved
direct order does not invalidate the reference-free statistic. No new RCWA or
reference refinement. The earlier fresh baseline 40 replay is inherited and
verified, not described as freshly executed under this protocol.

Height changes edge length, corner fields and resonances together. Results
cannot identify their cause, certify C7, or describe p-incidence convergence.
The reduced ladder and longer durations are disclosed, calibration-informed
design choices, not preregistered findings of the original failed campaign.
Freeze this file before any new spatial response. One heavy worker at a time;
all imported data, attempts and registration identities are preserved.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from scipy.optimize import brentq

import studies.cross_solver.metasurface_tall.fdtd_metasurface_tall as parent
import studies.cross_solver.metasurface_tall.fdtd_metasurface_tall_time as calibration


ROOT, RESULTS, HEIGHTS, c7 = parent.ROOT, parent.RESULTS, parent.HEIGHTS, parent.c7
PREFIX = 'fdtd_metasurface_tall_settled'
REG = RESULTS/(PREFIX+'_registration.json')
LOG = RESULTS/(PREFIX+'.jsonl')
JOBS = RESULTS/(PREFIX+'_jobs')
REPORT = RESULTS/(PREFIX+'_report.json')
RUNGS = (40,60,80)
BASE, LONG = {'base':120,'tall':540}, {'base':180,'tall':720}
MIDDLE = 60
REFS, SAME = parent.REFS, parent.SAME
INPUTS = [parent.REG,parent.LOG,RESULTS/'fdtd_metasurface_tall_halt.json',
          calibration.REG,calibration.LOG,RESULTS/'fdtd_metasurface_tall_time_report.json']
key, values = parent.key, parent.values


def identity():
    return dict(source=parent.sha(__file__),parent=parent.identity(),calibration=calibration.identity(),
                inputs={p.relative_to(ROOT).as_posix():parent.sha(p) for p in INPUTS})


def verify():
    parent.verify()
    calibration.verify()
    reg = json.loads(REG.read_text())
    assert reg['identity']==identity()
    assert parent.sha(RESULTS/reg['copy'])==reg['identity']['source']
    return reg


def ratio_at(p):
    a,b = math.log(RUNGS[1]/RUNGS[0]),math.log(RUNGS[2]/RUNGS[1])
    return a/b if p==0 else math.expm1(p*a)/(-math.expm1(-p*b))


def order_at(ratio):
    return brentq(lambda p:ratio_at(p)-ratio,-64,64,xtol=1e-12) if ratio_at(-64)<ratio<ratio_at(64) else None


def interval(a,b,u):
    if min(a,b)<=max(SAME,u):
        return None
    ends = [order_at((a-u)/(b+u)),order_at((a+u)/(b-u))]
    return ends if all(v is not None for v in ends) else None


def fd_job(name,n,periods=None,pml=24):
    return ['fd',name,n,BASE[name] if periods is None else periods,pml]


def preflight():
    parent.verify()
    calibration.verify()
    for p in (-2.,-.5,0.,1.,1.444,2.,3.):
        assert math.isclose(order_at(ratio_at(p)),p,abs_tol=1e-11)
        if p:
            vs = [complex(.3,.7)*n**(-p) for n in RUNGS]
            bounds = interval(abs(vs[0]-vs[1]),abs(vs[1]-vs[2]),0.)
            assert all(math.isclose(x,p,abs_tol=1e-10) for x in bounds)
        for a,b in zip(RUNGS,RUNGS[1:]):
            assert all(math.isclose(x,p,abs_tol=1e-10) for x in parent.direct_interval(
                (a/40)**(-p),(b/40)**(-p),0.,b/a))
    assert interval(0.,0.,0.) is None and interval(1.,1.,1.) is None
    layouts = []
    for name in HEIGHTS:
        with parent.configure(name):
            for n in RUNGS:
                scene,_ = c7.build('M3',n,'s',BASE[name])
                assert c7.faces('M3',n)==(.25,.25,.35,.35,.15,.15)
                assert scene.structures[1].shape.size==(.30,.20,HEIGHTS[name])
                grid = [round(x*n) for x in scene.domain.size]
                if n==60:
                    plan = c7.fd.plan(scene)
                    assert list(plan.grid.n)==grid
                layouts.append(dict(scene=name,n=n,grid=grid,periods=BASE[name]))
    cal = parent.read_rows(calibration.LOG)['fd|tall|40|540']
    assert not c7.validity('M3',cal), 'Selected calibrated 40-cell row is invalid'
    old = parent.read_rows(parent.LOG)
    assert parent.replay_gap(old)<=SAME
    assert not c7.validity('M3',old[key(parent.fd_job('base',40,'replay'))])
    print('Preflight passed: calibrated selected row, six layouts, two new plans, signed-order instruments.',flush=True)
    return layouts


def register():
    if REG.exists():
        verify()
        return
    assert not LOG.exists() and not JOBS.exists()
    layouts = preflight()
    ident = identity()
    fingerprint = hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()
    copy = RESULTS/(PREFIX+'_registered_'+ident['source'][:16]+'.py.txt')
    with copy.open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    reg = dict(registered=parent.timestamp(),identity=ident,fingerprint=fingerprint,
               copy=copy.name,protocol=__doc__,rungs=RUNGS,base_periods=BASE,long_periods=LONG,
               control_rung=MIDDLE,layouts=layouts,known_40_cell_calibration=True,
               original_campaign='incomplete and aborted; not retroactively rescued')
    with REG.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(reg,indent=2)+'\n')
    print('REGISTERED',reg['registered'],fingerprint,flush=True)


def append(row):
    with LOG.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')


def import_history():
    reg = verify()
    original,cal = parent.read_rows(parent.LOG),parent.read_rows(calibration.LOG)
    pairs = [(fd_job('base',n),parent.key(parent.fd_job('base',n)),parent.LOG,original) for n in (40,80)]
    pairs += [(fd_job('base',40)+['replay'],parent.key(parent.fd_job('base',40,'replay')),parent.LOG,original),
              (fd_job('tall',40),'fd|tall|40|540',calibration.LOG,cal)]
    pairs += [(parent.ref_job(name,i,n),key(parent.ref_job(name,i,n)),parent.LOG,original)
              for name in HEIGHTS for i in range(3) for n in REFS[name]]
    done = parent.read_rows(LOG)
    for job,oldkey,path,rows in pairs:
        if key(job) not in done:
            old = rows[oldkey]
            assert old['status']=='ok'
            row = dict(old,key=key(job),job=job,scene=job[1],fingerprint=reg['fingerprint'],imported=True,
                       origin=dict(file=path.name,key=oldkey,sha256=parent.sha(path)))
            append(row)
    print('Imported required references, baseline endpoints, calibrated tall40, and inherited replay.',flush=True)


def worker(job,output):
    reg = verify()
    _,name,n,periods,pml = job
    started,stamp = time.perf_counter(),parent.timestamp()
    with parent.configure(name,'pml' if pml==48 else 'base'):
        c7.PROV = c7.provenance()
        c7.PROV['digest'] = reg['fingerprint']
        row = c7.fd_row(c7.spec('M3',n,'s',periods))
    row.update(key=key(job),job=job,scene=name,fingerprint=reg['fingerprint'],started=stamp,
               finished=parent.timestamp(),wall=time.perf_counter()-started,
               metadata_dirty_scope='wavebit source tree only')
    Path(output).write_text(json.dumps(row,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:row[k] for k in ('key','status','wall','decay') if k in row}),flush=True)
    return 0 if row['status']=='ok' else 1


def execute(job):
    reg = verify()
    old = parent.read_rows(LOG).get(key(job))
    if old is not None and old['status']=='ok':
        assert old['fingerprint']==reg['fingerprint']
        return old
    JOBS.mkdir(exist_ok=True)
    stem,attempt = key(job).replace('|','_'),1
    while (JOBS/f'{stem}_attempt{attempt}.log').exists():
        attempt += 1
    output = JOBS/f'{stem}_attempt{attempt}.json'
    logfile = output.with_suffix('.log')
    print(parent.timestamp(),'START',key(job),flush=True)
    with logfile.open('x',encoding='utf-8') as stream:
        got = subprocess.run([sys.executable,__file__,'worker',json.dumps(job),str(output)],
                             cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    row = json.loads(output.read_text()) if output.exists() else dict(
        key=key(job),job=job,scene=job[1],kind='fd',status='error',
        error='Worker exited without result',fingerprint=reg['fingerprint'])
    row.update(worker_exit=got.returncode,attempt=attempt,raw_file=output.name,raw_log=logfile.name)
    append(row)
    print(parent.timestamp(),'END',key(job),row['status'],row.get('wall'),flush=True)
    assert got.returncode==0 and row['status']=='ok', logfile
    return row


def selected(rows,name,n):
    row = rows.get(key(fd_job(name,n)))
    return rows.get(key(fd_job(name,n,LONG[name]))) if row and not c7.settled(row) else row


def validity(rows,name):
    required = [(f'selected{n}',selected(rows,name,n),True) for n in RUNGS]
    required += [(f'raw{n}',rows.get(key(fd_job(name,n))),False) for n in RUNGS]
    required += [('long60',rows.get(key(fd_job(name,MIDDLE,LONG[name]))),True),
                 ('pml60',rows.get(key(fd_job(name,MIDDLE,LONG[name],48))),True)]
    return {label:bad for label,row,settle in required if (bad:=c7.validity('M3',row,settle))}


def cells(rows,name):
    bad = validity(rows,name)
    out = []
    for obs in c7.cell_keys('M3'):
        for i,wl in enumerate(c7.WAVELENGTHS):
            c = dict(observable=obs,wavelength=wl,pol='s',self_class='unresolved',
                     self_interval=None,direct_classes=['unresolved']*2)
            if bad:
                out.append(dict(c,status='invalid',invalid=bad))
                continue
            vs = [values(selected(rows,name,n),n)[obs,i] for n in RUNGS]
            base,long,pml = [values(rows[key(j)],MIDDLE)[obs,i] for j in (
                fd_job(name,MIDDLE),fd_job(name,MIDDLE,LONG[name]),fd_job(name,MIDDLE,LONG[name],48))]
            noise = abs(long-base)+abs(pml-long)
            changes = [abs(a-b) for a,b in zip(vs,vs[1:])]
            share = 2*noise/min(changes) if min(changes)>SAME else None
            floor = share is not None and share<=.1
            si = interval(*changes,2*noise) if floor else None
            ref,spread = parent.reference(rows,name,i,obs)
            errors = [abs(v-ref) for v in vs]
            orders,dis = [],[]
            for j,(a,b) in enumerate(zip(errors,errors[1:])):
                ratio = RUNGS[j+1]/RUNGS[j]
                orders.append(math.log(a/b)/math.log(ratio) if min(a,b)>SAME else None)
                dis.append(parent.direct_interval(a,b,spread+noise,ratio)
                           if floor and spread<=.02*min(a,b) else None)
            c.update(status='ok',values=[[v.real,v.imag] for v in vs],changes=changes,
                     noise=noise,noise_share=share,floor_ok=floor,
                     self_order=order_at(changes[0]/changes[1]) if min(changes)>SAME else None,
                     self_interval=si,self_class=parent.instrument.classify(si),errors=errors,
                     reference=[ref.real,ref.imag],reference_spread=spread,reference_top=REFS[name][-1],
                     direct_orders=orders,direct_intervals=dis,
                     direct_classes=[parent.instrument.classify(x) for x in dis])
            out.append(c)
    assert len(out)==54
    return out


def report(stop_reason=None):
    reg = verify()
    rows = parent.read_rows(LOG)
    assert all(r['fingerprint']==reg['fingerprint'] for r in rows.values())
    scenes = {name:cells(rows,name) for name in HEIGHTS}
    a,b = (rows[key(j)] for j in (fd_job('base',40),fd_job('base',40)+['replay']))
    gap = max(abs(complex(*a['values'][k][i])-complex(*b['values'][k][i])) for k in a['values'] for i in range(3))
    valid = gap<=SAME and all(not validity(rows,name) for name in HEIGHTS)
    if stop_reason is None and not valid and REPORT.exists():
        previous = json.loads(REPORT.read_text())
        if previous.get('stop_reason'):
            stop_reason = previous['stop_reason']
    if gap>SAME:
        for cs in scenes.values():
            for c in cs:
                c.update(self_class='unresolved',self_interval=None,
                         direct_classes=['unresolved']*2,direct_intervals=[None,None])
    deltas = []
    for a,b in zip(scenes['base'],scenes['tall']):
        assert (a['observable'],a['wavelength'])==(b['observable'],b['wavelength'])
        x,y = a['self_interval'],b['self_interval']
        deltas.append(dict(observable=a['observable'],wavelength=a['wavelength'],
                           interval=[y[0]-x[1],y[1]-x[0]] if x and y else None))
    result = dict(fingerprint=reg['fingerprint'],complete=valid,controls_valid=valid,
                  stop_reason=stop_reason,rungs=RUNGS,control_rung=MIDDLE,base_periods=BASE,long_periods=LONG,
                  scope='New calibrated 40/60/80 s-only comparison; original 40/80/120 campaign remains aborted.',
                  inherited_replay_gap=gap,reference_ladders=REFS,scenes=scenes,delta_intervals=deltas,
                  self_counts={n:dict(Counter(c['self_class'] for c in cs)) for n,cs in scenes.items()},
                  validity={n:validity(rows,n) for n in HEIGHTS})
    REPORT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = [result['scope'],json.dumps({k:result[k] for k in ('complete','stop_reason','self_counts')})]
    for name,cs in scenes.items():
        lines.append(name)
        lines.extend(json.dumps(c,allow_nan=False) for c in cs)
    (RESULTS/(PREFIX+'_report.log')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(lines[1],flush=True)
    return result


def run():
    verify()
    import_history()
    for name in HEIGHTS:
        for n in RUNGS:
            row = execute(fd_job(name,n))
            if not c7.settled(row) and not c7.validity('M3',row,settle=False):
                row = execute(fd_job(name,n,LONG[name]))
            if bad:=c7.validity('M3',row):
                report(dict(job=row['job'],invalid=bad))
                raise SystemExit(2)
            if n==MIDDLE:
                for pml in (24,48):
                    control = execute(fd_job(name,n,LONG[name],pml))
                    if bad:=c7.validity('M3',control):
                        report(dict(job=control['job'],invalid=bad))
                        raise SystemExit(2)
    assert report()['complete']


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        raise SystemExit(worker(json.loads(sys.argv[2]),sys.argv[3]))
    for command in sys.argv[1:] or ['report']:
        dict(preflight=preflight,register=register,run=run,report=report)[command]()
