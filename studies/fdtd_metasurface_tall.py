"""Registered bounded M3 height probe: s incidence, every propagating cell.

Known before registration: all C7 M3 rows/references, the lifted-box and ridge
studies, and height_preflight's post-hoc time-only orders. The new treatment is
height .60 instead of .30, with the same .30 x .20 footprint, periods, glass
contact, fractional face positions, fixed-k illumination and monitor conventions.
There are 54 cells per height (18 quantities at .95/1/1.05), s incidence only.
The historical baseline is reanalysed; only the tall response is prospective.

Protocol fixed before any new target response:
* Use 40/80/120 cells per wavelength. Import baseline base rows and 80-cell
  180-period control under verified old hashes. A fresh baseline 40 replay must
  agree in every raw value and phase-corrected cell within the old 1e-10 line.
* All selected runs must pass C7 validity: complete finite values, angle within
  1e-9, stable, no growing/unsettled warning, both decays below 1e-5. A base
  120-period run that is merely unsettled gets one 180-period replacement.
  Other invalid runs are retained and withhold that height's classifications.
* Both heights have time (180 periods) and absorber (48 vs 24 PML cells)
  controls at 80. Per cell, noise = |long80-base80| + |pml80-base80|.
  Twice this empirical proxy must be <= .1 of the smaller adjacent grid
  change. There are NO conditional finer controls in this bounded campaign:
  failures remain unresolved. An 80-cell proxy is not a rigorous bound on
  120-cell noise. PML controls run 120 periods and must themselves settle.
* Primary statistic: invert |v40-v80|/|v80-v120| for p in the unequal-rung
  power-law ratio, including the continuous p=0 limit and negative orders.
  Apply +/- 2*noise to each change for an empirical interval. Changes at or
  below 1e-10, nonpositive interval endpoints, invalid controls or a failed
  noise gate are unresolved. Every cell is printed; maxima cannot overrule it.
* Inherited classes: entire interval in [1.8,2.3] is second; upper end <=1.75
  is low; otherwise mixed. A finite-grid class does not establish an asymptote.
* Tall RCWA uses the fixed square ladder N=12/16/20/24, at all three wavelengths.
  No refinement beyond 24. Baseline retains old N=20/26/28/30/32. The maximum
  distance of the three preceding references from the top is an empirical
  per-cell spread. Reference orders need spread <= .02 of both endpoint errors,
  the same noise gate, and positive errors after subtracting spread+noise.
  Insufficient references leave direct orders unresolved; primary self-orders
  do not require RCWA. The unequal reference ladders are explicitly reported.
* Before each new reference, estimated memory = 34 complex128 matrices of side
  2*(2*N+1)^2, plus a 2-GiB reserve, must fit current physical/commit headroom.
  A failed resource guard leaves a recorded incomplete attempt; it cannot alter
  the ladder or gates. One heavy child process runs at a time.
* Report tall-minus-baseline self-intervals [tall_lo-base_hi,tall_hi-base_lo]
  descriptively, without a new significance threshold or single causal verdict.
  Height changes edge length, corner fields and resonances together. This
  experiment cannot identify edges/vertices/resonances as the cause, certify
  C7, or establish the response of p incidence.

Analytic instruments and layout plans are checked before registration. No new
pytest functions or production changes. Preserve all attempts, inputs, source
copy and hashes. This file is immutable after registration. Commands:
preflight, register, run, report; workers are private subprocess entry points.
"""
from collections import Counter
import contextlib
import datetime
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import fdtd_metasurface_rcwa as c7
import fdtd_metasurface_lifted as lifted
import fdtd_metasurface_height_preflight as instrument


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'studies/results'
PREFIX = 'fdtd_metasurface_tall'
REG = RESULTS / (PREFIX + '_registration.json')
LOG = RESULTS / (PREFIX + '.jsonl')
JOBS = RESULTS / (PREFIX + '_jobs')
REPORT = RESULTS / (PREFIX + '_report.json')
RUNGS = (40, 80, 120)
HEIGHTS = {'base': .30, 'tall': .60}
REFS = {'base': (20, 26, 28, 30, 32), 'tall': (12, 16, 20, 24)}
SAME, NOISE_SHARE, REF_SHARE = 1e-10, .1, .02
OLD_REG = RESULTS / 'fdtd_metasurface_rcwa_registration.json'
INPUTS = [c7.LOG, c7.RCWA_LOG, OLD_REG,
          RESULTS / 'fdtd_metasurface_height_preflight.json',
          RESULTS / 'fdtd_metasurface_fine_report.json']
DEPENDENCIES = [Path(m.__file__) for m in (c7, lifted, instrument)]
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


def timestamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def sha(path):
    return instrument.sha(path)


def identity():
    h = hashlib.sha256()
    for path in sorted((ROOT / 'src/wavebit').rglob('*.py')):
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return dict(source=sha(__file__), package=h.hexdigest(),
                dependencies={p.relative_to(ROOT).as_posix(): sha(p) for p in DEPENDENCIES},
                inputs={p.relative_to(ROOT).as_posix(): sha(p) for p in INPUTS},
                versions={m: importlib.metadata.version(m)
                          for m in ('numpy', 'scipy', 'jax', 'jaxlib', 'torch')},
                threads={k: os.environ.get(k) for k in THREADS}, python=sys.version)


def verify():
    reg = json.loads(REG.read_text(encoding='utf-8'))
    assert reg['identity'] == identity(), 'Frozen study identity changed'
    assert sha(RESULTS / reg['copy']) == reg['identity']['source']
    return reg


def key(job):
    return '|'.join(map(str, job))


def fd_job(name, n, variant='base'):
    return ['fd', name, n, variant]


def ref_job(name, i, n):
    return ['rcwa', name, i, n]


@contextlib.contextmanager
def configure(name, variant='base'):
    saved = c7.SCENES['M3']['H'], c7.PML_CELLS
    c7.SCENES['M3']['H'] = HEIGHTS[name]
    c7.PML_CELLS = 48 if variant == 'pml' else 24
    try:
        yield
    finally:
        c7.SCENES['M3']['H'], c7.PML_CELLS = saved


def read_rows(path=LOG):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            out[row['key']] = row
    return out


def append(row):
    with LOG.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')


def old_rows():
    c7.PROV = c7.provenance()
    old = json.loads(OLD_REG.read_text())
    assert old['sha256'] == sha(c7.__file__)
    assert old['digest'] == c7.PROV['digest']
    assert old['rcwa']['M3'] == c7.PROV['rcwa']['M3']
    return c7.load(c7.LOG), c7.load(c7.RCWA_LOG)


def direct_interval(a, b, u, ratio):
    if not all(math.isfinite(x) for x in (a,b,u)) or min(a,b) <= max(SAME,u):
        return None
    return [math.log((a-u)/(b+u))/math.log(ratio),
            math.log((a+u)/(b-u))/math.log(ratio)]


def preflight():
    rows, refs = old_rows()
    for p in (-2., -.5, 0., 1., 1.444, 2., 3.):
        assert math.isclose(instrument.order_at(instrument.ratio_at(p)), p, abs_tol=1e-11)
        if p:
            v = [complex(.3,.7)*n**(-p) for n in RUNGS]
            changes = [abs(a-b) for a,b in zip(v,v[1:])]
            bounds = instrument.interval(*changes, 0.)
            assert all(math.isclose(x,p,abs_tol=1e-10) for x in bounds)
        for a,b in zip(RUNGS,RUNGS[1:]):
            assert all(math.isclose(x,p,abs_tol=1e-10)
                       for x in direct_interval((a/40)**(-p),(b/40)**(-p),0.,b/a))
    assert instrument.interval(0.,0.,0.) is None
    assert instrument.interval(1.,1.,1.) is None
    assert direct_interval(0.,0.,0.,2.) is None
    assert instrument.classify((1.,1.)) == 'low'
    assert instrument.classify((2.,2.)) == 'second'
    assert instrument.classify((1.7,1.9)) == 'mixed'
    assert instrument.classify(None) == 'unresolved'
    layouts = []
    with contextlib.redirect_stdout(io.StringIO()):
        for name in HEIGHTS:
            with configure(name):
                for n in RUNGS:
                    scene, _ = c7.build('M3', n, 's')
                    plan = c7.fd.plan(scene)
                    old = rows[c7.key_of(c7.spec('M3',n,'s'))]
                    assert not c7.validity('M3', old)
                    assert plan.steps == old['steps']
                    assert scene.structures[1].shape.size == (.30,.20,HEIGHTS[name])
                    assert c7.faces('M3',n) == (.25,.25,.35,.35,.15,.15)
                    layouts.append(dict(scene=name,n=n,grid=list(plan.grid.n),steps=int(plan.steps)))
    assert all(f'M3|{i}|{n}x{n}' in refs for i in range(3) for n in REFS['base'])
    print('Preflight passed: old identities, six public plans, signed-order instruments.', flush=True)
    return layouts


def register():
    if REG.exists():
        verify()
        print('Existing registration verified and preserved.')
        return
    assert not LOG.exists() and not JOBS.exists(), 'Registration would postdate data'
    layouts = preflight()
    ident = identity()
    assert all(ident['threads'][k] == '1' for k in THREADS)
    fingerprint = hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()
    copy = RESULTS / (PREFIX+'_registered_'+ident['source'][:16]+'.py.txt')
    with copy.open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    record = dict(registered=timestamp(), identity=ident, fingerprint=fingerprint,
                  base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  copy=copy.name, protocol=__doc__, layouts=layouts,
                  reference_ladders=REFS, memory=instrument.memory())
    with REG.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('REGISTERED',record['registered'],fingerprint,flush=True)


def import_history():
    reg = verify()
    fd, refs = old_rows()
    expected = [(fd_job('base',n),c7.key_of(c7.spec('M3',n,'s')),c7.LOG,fd) for n in RUNGS]
    expected.append((fd_job('base',80,'long'),c7.key_of(c7.spec('M3',80,'s',180)),c7.LOG,fd))
    expected += [(ref_job('base',i,n),f'M3|{i}|{n}x{n}',c7.RCWA_LOG,refs)
                 for i in range(3) for n in REFS['base']]
    done = read_rows()
    for job, oldkey, path, source in expected:
        if key(job) not in done:
            original = source[oldkey]
            row = dict(original,key=key(job),job=job,scene=job[1],status='ok',imported=True,
                       fingerprint=reg['fingerprint'],
                       origin=dict(file=path.name,key=oldkey,sha256=sha(path)))
            append(row)
    print('Imported 4 baseline FDTD and 15 reference rows.',flush=True)


def worker(job, output):
    reg = verify()
    started = time.perf_counter()
    row = dict(key=key(job),job=job,kind=job[0],scene=job[1],
               fingerprint=reg['fingerprint'],started=timestamp())
    try:
        if job[0] == 'rcwa':
            _, name, i, n = job
            memory = instrument.memory()
            peak = 34*16*(2*(2*n+1)**2)**2
            row.update(memory=memory,estimated_peak_bytes=peak)
            if peak+2*2**30 > min(memory['available_physical'],memory['available_pagefile']):
                raise MemoryError('Reference resource guard; ladder unchanged, job incomplete')
            with configure(name):
                values = c7.rcwa_point('M3',c7.WAVELENGTHS[i],(n,n))
            assert set(values) == {'s','p'}
            assert all(set(v)==set(c7.cell_keys('M3')) for v in values.values())
            assert all(math.isfinite(z.real) and math.isfinite(z.imag)
                       for v in values.values() for z in v.values())
            row.update(status='ok',N=n,wavelength=c7.WAVELENGTHS[i],
                       values={p:{k:[v.real,v.imag] for k,v in vs.items()} for p,vs in values.items()})
        else:
            _, name, n, variant = job
            with configure(name,variant):
                c7.PROV = c7.provenance()
                c7.PROV['digest'] = reg['fingerprint']
                got = c7.fd_row(c7.spec('M3',n,'s',180 if variant=='long' else 120))
            row.update(got,key=key(job),job=job,scene=name,
                       metadata_dirty_scope='wavebit source tree only',
                       fingerprint=reg['fingerprint'])
        row['wall'] = time.perf_counter()-started
    except Exception as err:  # preserve failures instead of replacing their evidence
        row.update(status='error',error=f'{type(err).__name__}: {err}',
                   trace=traceback.format_exc(),wall=time.perf_counter()-started)
    row['finished'] = timestamp()
    Path(output).write_text(json.dumps(row,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:row[k] for k in ('key','status','wall')}),flush=True)
    return 0 if row['status']=='ok' else 1


def execute(job):
    reg = verify()
    old = read_rows().get(key(job))
    if old is not None and old['status']=='ok':
        assert old['fingerprint']==reg['fingerprint']
        return old
    JOBS.mkdir(exist_ok=True)
    stem, attempt = key(job).replace('|','_'), 1
    while (JOBS/f'{stem}_attempt{attempt}.log').exists():
        attempt += 1
    output = JOBS/f'{stem}_attempt{attempt}.json'
    logfile = output.with_suffix('.log')
    print(timestamp(),'START',key(job),'attempt',attempt,flush=True)
    with logfile.open('x',encoding='utf-8') as stream:
        proc = subprocess.run([sys.executable,__file__,'worker',json.dumps(job),str(output)],
                              cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if not output.exists():
        append(dict(key=key(job),job=job,kind=job[0],scene=job[1],status='error',
                    fingerprint=reg['fingerprint'],worker_exit=proc.returncode,
                    error='Worker exited without result',raw_log=logfile.name,attempt=attempt))
        raise RuntimeError(f'Worker exited {proc.returncode} without result: {logfile}')
    row = json.loads(output.read_text(encoding='utf-8'))
    row.update(worker_exit=proc.returncode,raw_file=output.name,raw_log=logfile.name,attempt=attempt)
    append(row)
    print(timestamp(),'END',key(job),row['status'],f'{row["wall"]:.2f}s',flush=True)
    if proc.returncode:
        raise RuntimeError(f'Worker failed; preserved {logfile}')
    return row


def selected(rows,name,n):
    base = rows.get(key(fd_job(name,n)))
    return rows.get(key(fd_job(name,n,'long'))) if base and not c7.settled(base) else base


def values(row,n):
    return lifted.cells('M3',row,n)


def validity(rows,name):
    required = [(f'selected{n}',selected(rows,name,n),True) for n in RUNGS]
    required += [(v,rows.get(key(fd_job(name,80,v))),v!='base') for v in ('base','long','pml')]
    for n in RUNGS:
        required.append((f'raw{n}',rows.get(key(fd_job(name,n))),False))
    return {label:bad for label,row,settle in required if (bad:=c7.validity('M3',row,settle))}


def reference(rows,name,i,obs):
    ladder = [rows[key(ref_job(name,i,n))] for n in REFS[name]]
    vals = [complex(*r['values']['s'][obs]) for r in ladder]
    return vals[-1],max(abs(v-vals[-1]) for v in vals[-4:-1])


def cells(rows,name):
    bad = validity(rows,name)
    out = []
    for obs in c7.cell_keys('M3'):
        for i,wl in enumerate(c7.WAVELENGTHS):
            entry = dict(observable=obs,wavelength=wl,pol='s',self_class='unresolved',
                         self_interval=None,direct_classes=['unresolved']*2)
            if bad:
                out.append(dict(entry,status='invalid',invalid=bad))
                continue
            vs = [values(selected(rows,name,n),n)[obs,i] for n in RUNGS]
            controls = [values(rows[key(fd_job(name,80,v))],80)[obs,i] for v in ('base','long','pml')]
            noise = abs(controls[1]-controls[0])+abs(controls[2]-controls[0])
            changes = [abs(a-b) for a,b in zip(vs,vs[1:])]
            share = 2*noise/min(changes) if min(changes)>SAME else None
            floor = share is not None and share<=NOISE_SHARE
            si = instrument.interval(*changes,2*noise) if floor else None
            ref,spread = reference(rows,name,i,obs)
            errors = [abs(v-ref) for v in vs]
            dis, orders = [], []
            for j,(a,b) in enumerate(zip(errors,errors[1:])):
                ratio = RUNGS[j+1]/RUNGS[j]
                orders.append(math.log(a/b)/math.log(ratio) if min(a,b)>SAME else None)
                dis.append(direct_interval(a,b,spread+noise,ratio)
                           if floor and spread<=REF_SHARE*min(a,b) else None)
            entry.update(status='ok',values=[[v.real,v.imag] for v in vs],changes=changes,
                         noise=noise,noise_share=share,floor_ok=floor,
                         self_order=instrument.order_at(changes[0]/changes[1]) if min(changes)>SAME else None,
                         self_interval=si,self_class=instrument.classify(si),errors=errors,
                         reference=[ref.real,ref.imag],reference_spread=spread,reference_top=REFS[name][-1],
                         direct_orders=orders,direct_intervals=dis,
                         direct_classes=[instrument.classify(x) for x in dis])
            out.append(entry)
    assert len(out)==54
    return out


def replay_gap(rows):
    old, new = (rows[key(fd_job('base',40,v))] for v in ('base','replay'))
    raw = max(abs(complex(*old['values'][k][i])-complex(*new['values'][k][i]))
              for k in old['values'] for i in range(3))
    a,b = values(old,40),values(new,40)
    return max(raw,max(abs(a[k]-b[k]) for k in a))


def report():
    reg = verify()
    rows = read_rows()
    assert all(r['fingerprint']==reg['fingerprint'] for r in rows.values())
    required = [fd_job(name,n) for name in HEIGHTS for n in RUNGS]
    required += [fd_job(name,80,v) for name in HEIGHTS for v in ('long','pml')]
    required += [ref_job(name,i,n) for name in HEIGHTS for i in range(3) for n in REFS[name]]
    required.append(fd_job('base',40,'replay'))
    for name in HEIGHTS:
        required += [fd_job(name,n,'long') for n in RUNGS
                     if (r:=rows.get(key(fd_job(name,n)))) and not c7.settled(r)]
    missing = [key(j) for j in required if key(j) not in rows or rows[key(j)]['status']!='ok']
    if missing:
        print('INCOMPLETE',json.dumps(missing),flush=True)
        return None
    scenes = {name:cells(rows,name) for name in HEIGHTS}
    replay = rows[key(fd_job('base',40,'replay'))]
    gap = replay_gap(rows)
    valid = gap<=SAME and not c7.validity('M3',replay) and all(not validity(rows,n) for n in HEIGHTS)
    # A failed global replay withholds classifications, never merely adds a footnote.
    if not valid:
        for entries in scenes.values():
            for c in entries:
                c.update(self_class='unresolved',self_interval=None,direct_classes=['unresolved']*2,
                         direct_intervals=[None,None])
    deltas = []
    for base,tall in zip(scenes['base'],scenes['tall']):
        assert (base['observable'],base['wavelength'])==(tall['observable'],tall['wavelength'])
        a,b = base['self_interval'],tall['self_interval']
        deltas.append(dict(observable=base['observable'],wavelength=base['wavelength'],
                           interval=[b[0]-a[1],b[1]-a[0]] if a and b else None))
    counts = {name:dict(Counter(c['self_class'] for c in cs)) for name,cs in scenes.items()}
    result = dict(fingerprint=reg['fingerprint'],complete=True,controls_valid=valid,replay_gap=gap,
                  scope='Bounded s-only height comparison; empirical 80-cell noise proxy; not a C7 pass or causal identification.',
                  reference_ladders=REFS,scenes=scenes,delta_intervals=deltas,self_counts=counts,
                  validity={name:validity(rows,name) for name in HEIGHTS})
    REPORT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = [result['scope'],'controls_valid '+str(valid),'replay_gap '+str(gap)]
    for name,entries in scenes.items():
        lines.append(name+' '+json.dumps(counts[name]))
        lines.extend(json.dumps(c,allow_nan=False) for c in entries)
    lines.extend('delta '+json.dumps(c) for c in deltas)
    (RESULTS/(PREFIX+'_report.log')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(controls_valid=valid,replay_gap=gap,self_counts=counts)),flush=True)
    return result


def run():
    verify()
    import_history()
    replay = execute(fd_job('base',40,'replay'))
    assert not c7.validity('M3',replay) and replay_gap(read_rows())<=SAME
    for name in HEIGHTS:
        for i in range(3):
            for n in REFS[name]:
                execute(ref_job(name,i,n))
    for name in HEIGHTS:
        for n in RUNGS:
            row = execute(fd_job(name,n))
            if not c7.settled(row) and not c7.validity('M3',row,settle=False):
                execute(fd_job(name,n,'long'))
            if n==80:
                for v in ('long','pml'):
                    execute(fd_job(name,n,v))
    result = report()
    if result is None or not result['controls_valid']:
        raise RuntimeError('Incomplete or invalid campaign; retained evidence, no certification')


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        raise SystemExit(worker(json.loads(sys.argv[2]),sys.argv[3]))
    commands = dict(preflight=preflight,register=register,run=run,report=report)
    for command in sys.argv[1:] or ['report']:
        commands[command]()
