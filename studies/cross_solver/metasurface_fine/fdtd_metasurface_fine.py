"""C7 follow-up: all y-ridge observables separately through 640 cells per wavelength.

This protocol is registered before its new FDTD rows. Known data are the previous
ridges/corners campaigns through 320 and the two reference-only fine_scan files.
The old Y12 worst-cell rule read second order; its wavelength-.95 s-transmitted
amplitude read 1.73, 1.62, 1.56 at the three steps. No inference of a purely 3D
cause survives that observation without further evidence.

Scenes: Y12 (epsilon 12, height .30), Y4 (epsilon 4, height .30), D12 (epsilon 12,
height .10), all width .20, period .50, phi 30, theta 25, glass epsilon 2.25.
Use the previous source, geometry placement, split junction and monitor planes.
D12 was selected from reference-only scans: height .15 moves a strong p peak
near .975, height .20 near 1.0, height .10 below .95. D12 has a smooth sampled
.95--1.05 band; this is NOT a proof of no resonance or an isolated intervention
on resonance strength. Changing height changes the field at corners too.

Rungs 80/160/320/640, s and p, wavelengths .95/1/1.05, every outgoing complex
amplitude and efficiency of the zeroth order. F12 (the old cornerless film) is
the propagation control. Reuse unchanged Y12/F12 rows through 320 and existing
Y12/film references only after checking their registered source identities.
A fresh Y12 160 s replay must reproduce all 18 numbers within 1e-10 (the
existing same-number line). New workers are sequential separate processes to
release compiled arrays between large runs. No production source is changed.

Measurement rules, fixed before the new FDTD rows:
* Each run passes the old ridge validity checks: finite complete values, angle
  within 1e-9, stable, no growth/unsettled warning, both monitor decays <1e-5.
  An unsettled 120-period run gets one 180-period replacement, then is invalid.
* At 320, every scene/polarisation also runs 180 periods and with
  48 instead of 24 PML cells. Their sum of absolute moves is an empirical
  time/absorber noise proxy for that cell. If twice this proxy exceeds one
  tenth of a resolved 320--640 change, rerun BOTH controls at 640 for that
  scene/polarisation. Use the larger proxy from the two resolutions. Failure
  to clear the proxy makes that observable unresolved; do not widen a gate.
  A proxy is not a rigorous numerical error bound.
* References start at N=300/400/600/800 (old Y12 rows may provide more).
  Their empirical uncertainty is the maximum distance of the previous three
  truncations from the top, per observable. If it exceeds .02 of a resolved
  finest error, extend that scene/wavelength to 1000, then 1200, stopping at
  the cap. This is a registered refinement rule, not a change of pass lines.
  Cells still lacking a reference remain unresolved in reference-based order;
  reference-free self-orders can remain usable. A difference <=1e-10 is
  unresolved at the study's same-number line; it is never called zero error.
* Print every cell, including unresolved and small cells, and both steps
  160--320 and 320--640. Also print the per-wavelength maxima, but they do
  not override a slower resolved cell. Reference order is log2(Ea/Eb), with
  an empirical interval using E +/- (reference spread + noise proxy).
  Self-order is log2(abs(v160-v320)/abs(v320-v640)), with +/- twice the
  noise proxy on each difference. Its interval needs no RCWA reference.
* The inherited classes are second if the entire interval is inside [1.8,2.3],
  low if its upper end is <=1.75, mixed otherwise, unresolved when a gate or
  positivity fails. The film must be second in each wavelength's worst
  amplitude and efficiency over the final step, with valid runs. It need
  not have a resolved order in exactly vanishing cross-polarised components.
* A low D12 shows the sampled Y12 peak is not necessary for low order in a
  y-ridge. It does not eliminate all resonance effects. Epsilon-4 versus
  epsilon-12 trends may be consistent with corner exponents but do not identify
  a mechanism. The previously derived exponent ranges (1.369--1.444 and
  1.591--1.612) are context, not fitted targets or new pass bands.

This bounded extension stops at 640, and may remain inconclusive about an
asymptote or M3's cause. It does not certify C7 as passed. Preserve registration,
raw worker logs, all failures, original input rows and every reported cell.

Commands: preflight, register, run, report. Do not modify this file after
registration; a mismatch with its saved copy withholds the result.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import contextlib
import datetime
import hashlib
import importlib.metadata
import io
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import studies.cross_solver.metasurface_ridges.fdtd_metasurface_ridges as ridge


HERE = repository_path('studies')
ROOT = HERE.parent
RESULTS = HERE / 'results'
PREFIX = 'fdtd_metasurface_fine'
LOG = RESULTS / (PREFIX + '.jsonl')
REG = RESULTS / (PREFIX + '_registration.json')
JOBS = RESULTS / (PREFIX + '_jobs')
REPORT = RESULTS / (PREFIX + '_report.json')
TEXT = RESULTS / (PREFIX + '_report.log')
SCENES = {'Y12': (12., .30), 'Y4': (4., .30), 'D12': (12., .10), 'F12': (12., .30)}
RUNGS = (80, 160, 320, 640)
WLS = (.95, 1., 1.05)
POLS = ('s', 'p')
SAME = 1e-10
SHARE = .02
FLOOR_SHARE = .1
INPUTS = [ridge.LOG, ridge.REGISTRATION,
          RESULTS / 'fdtd_metasurface_fine_scan.jsonl',
          RESULTS / 'fdtd_metasurface_fine_scan_more.jsonl',
          RESULTS / 'fdtd_metasurface_ridges_validation.json']
DEPENDENCIES = [Path(ridge.__file__), Path(ridge.cs.__file__), Path(ridge.c7.__file__),
                HERE / 'fdtd_metasurface_fine_scan.py', HERE / 'fdtd_metasurface_fine_scan_more.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


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
                python=sys.version)


def fingerprint():
    return hashlib.sha256(json.dumps(identity(), sort_keys=True).encode()).hexdigest()


def verify():
    record = json.loads(REG.read_text())
    if record['identity'] != identity() or sha(RESULTS / record['copy']) != record['identity']['source']:
        raise RuntimeError('Study, package, dependencies, versions or historical inputs changed.')
    return record


def read_rows(path):
    rows = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row['key']] = row
    return rows


def append(row):
    with LOG.open('a', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(row, allow_nan=False)+'\n')


def key(job):
    return '|'.join(str(x) for x in job)


def fd_job(name, n, pol, variant='base'):
    return ['fd', name, n, pol, variant]


def ref_job(name, i, n):
    return ['rcwa', name, i, n]


@contextlib.contextmanager
def configure(name, variant='base'):
    eps, height = SCENES[name]
    saved = ridge.EPS_P, ridge.H, ridge.cs.PML_CELLS
    ridge.EPS_P, ridge.H = eps, height
    if variant == 'pml':
        ridge.cs.PML_CELLS = 48
    try:
        yield
    finally:
        ridge.EPS_P, ridge.H, ridge.cs.PML_CELLS = saved


def preflight():
    old = json.loads(ridge.REGISTRATION.read_text())
    assert old['sha256'] == sha(ridge.__file__)
    assert all(sha(HERE / name) == digest for name, digest in old['dependencies'].items())
    measured = ridge.provenance()
    assert all(measured[k] == old['digests'][k] for k in ('fd', 'rcwa'))
    # Plan-only checks inspect allocation without measuring any FDTD answer.
    with contextlib.redirect_stdout(io.StringIO()):
        for name in SCENES:
            for n in RUNGS:
                with configure(name):
                    scene = ridge.build('F12' if name == 'F12' else 'Y12', n, 's')
                    plan = ridge.fd.plan(scene)
                assert math.isclose(scene.structures[1].material.eps[0], SCENES[name][0])
                assert math.isclose(scene.structures[1].shape.size[2], SCENES[name][1])
                assert plan.grid.n[0] == 1
    # Independent analytic sequences check the reporter's mathematical instrument.
    for p in (1., 1.369, 1.444, 2.):
        a, b, c = (complex(.3, .7) / n**p for n in (160, 320, 640))
        interval = order_interval(abs(a-b), abs(b-c), 0.)
        assert math.isclose(interval[0], p, abs_tol=1e-12)
        assert math.isclose(interval[1], p, abs_tol=1e-12)
    assert order_interval(0., 0., 0.) is None
    assert order_interval(1., 1., 1.) is None
    assert classify((1., 1.)) == 'low'
    assert classify((2., 2.)) == 'second'
    print('Preflight: historical identities, 16 scene plans, and order instrument verified.')


def register():
    if REG.exists():
        verify()
        print('Registration exists and matches; preserved.')
        return
    if LOG.exists() or JOBS.exists():
        raise RuntimeError('Registration would postdate campaign data.')
    preflight()
    ident = identity()
    copy = RESULTS / (PREFIX + '_registered_' + ident['source'][:16] + '.py.txt')
    with copy.open('xb') as out:
        out.write(Path(__file__).read_bytes())
    record = dict(registered=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
                  base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  identity=ident, fingerprint=fingerprint(), copy=copy.name,
                  scenes=SCENES, rungs=RUNGS, wavelengths=WLS, same_line=SAME,
                  reference_share=SHARE, floor_share=FLOOR_SHARE,
                  numerical_threads={s: os.environ.get(s) for s in
                                     ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    with REG.open('x', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(record, indent=2)+'\n')
    print('Registered', record['registered'], record['fingerprint'], flush=True)


def import_history():
    old = read_rows(ridge.LOG)
    oldreg = json.loads(ridge.REGISTRATION.read_text())
    done = read_rows(LOG)
    expected = []
    for name in ('Y12', 'F12'):
        for n in RUNGS[:-1]:
            for pol in POLS:
                expected.append((fd_job(name, n, pol), f'fd|{name}|{n}|{pol}|120', 'fd'))
        for i in range(3):
            for n in ((100, 200, 300, 400, 600, 800) if name == 'Y12' else (0,)):
                geom = 'Y' if name == 'Y12' else 'None'
                expected.append((ref_job(name, i, n), f'rcwa|{geom}|{i}|{n}', 'rcwa'))
    for job, oldkey, kind in expected:
        if key(job) in done:
            continue
        row = old[oldkey]
        assert row['prov'] == oldreg['digests'][kind]
        if kind == 'fd':
            assert not ridge.validity(row), (oldkey, ridge.validity(row))
        row = dict(row, key=key(job), job=job, origin=dict(file=ridge.LOG.name, key=oldkey,
                                                       sha256=sha(ridge.LOG)), imported=True)
        row['fingerprint'] = fingerprint()
        append(row)
    print('Historical rows imported without modifying their source files.', flush=True)


def worker(job, output):
    reg = verify()
    started = time.perf_counter()
    name = job[1]
    row = dict(key=key(job), job=job, kind=job[0], scene=name, fingerprint=reg['fingerprint'])
    try:
        if job[0] == 'rcwa':
            _, _, i, n = job
            with configure(name):
                values = ridge.rcwa_point(None if name == 'F12' else 'Y', WLS[i], n)
            row.update(status='ok', N=n, wavelength=WLS[i],
                       values={k: [v.real, v.imag] for k, v in values.items()})
        else:
            _, _, n, pol, variant = job
            periods = 180 if variant == 'long' else 120
            ridge.PROV = dict(fd=reg['fingerprint'], commit=reg['base'], dirty=False,
                              env=dict(threads=reg['numerical_threads']))
            with configure(name, variant):
                got = ridge.fd_row('F12' if name == 'F12' else 'Y12', n, pol, periods)
            row.update(got, key=key(job), job=job, scene=name, fingerprint=reg['fingerprint'])
        row['wall'] = time.perf_counter()-started
    except Exception as err:  # retain failed attempts; never silently substitute a value
        row.update(status='error', error=f'{type(err).__name__}: {err}',
                   trace=traceback.format_exc(), wall=time.perf_counter()-started)
    Path(output).write_text(json.dumps(row, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: row[k] for k in ('key', 'status', 'wall')}), flush=True)
    return 0 if row['status'] == 'ok' else 1


def execute(job):
    reg = verify()
    rows = read_rows(LOG)
    cached = rows.get(key(job))
    if cached is not None and cached.get('status', 'ok') == 'ok':
        assert cached['fingerprint'] == reg['fingerprint']
        return cached
    JOBS.mkdir(exist_ok=True)
    stem = key(job).replace('|', '_')
    attempt = 1
    while (JOBS / f'{stem}_attempt{attempt}.log').exists():
        attempt += 1
    output = JOBS / f'{stem}_attempt{attempt}.json'
    logfile = output.with_suffix('.log')
    print('START', key(job), 'attempt', attempt, flush=True)
    with logfile.open('x', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, __file__, 'worker', json.dumps(job), str(output)],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if not output.exists():
        raise RuntimeError(f'Worker exited {result.returncode} without a result: {logfile.name}')
    row = json.loads(output.read_text(encoding='utf-8'))
    row['worker_exit'] = result.returncode
    append(row)
    print('END', key(job), row['status'], f'{row["wall"]:.2f}s', flush=True)
    if result.returncode != 0:
        raise RuntimeError(f'Worker failed: {logfile.name}')
    return row


def selected(rows, name, n, pol):
    base = rows.get(key(fd_job(name, n, pol)))
    if base is not None and not ridge.settled(base):
        return rows.get(key(fd_job(name, n, pol, 'long')))
    return base


def value(row, observable, i):
    return complex(*row['values'][observable][i])


def noise(rows, name, pol, observable, i):
    proxies = []
    for n in (320, 640):
        base = rows.get(key(fd_job(name, n, pol)))
        long = rows.get(key(fd_job(name, n, pol, 'long')))
        pml = rows.get(key(fd_job(name, n, pol, 'pml')))
        if base is not None and long is not None and pml is not None:
            proxies.append(abs(value(long, observable, i)-value(base, observable, i))
                           + abs(value(pml, observable, i)-value(base, observable, i)))
    return max(proxies) if proxies else float('inf')


def reference(rows, name, i, observable):
    ladder = {job[3]: row for row in rows.values()
              if (job := row.get('job')) and job[0] == 'rcwa' and job[1:3] == [name, i]}
    ns = sorted(ladder)
    top = complex(*ladder[ns[-1]]['values'][observable])
    spread = max((abs(complex(*ladder[n]['values'][observable])-top) for n in ns[-4:-1]), default=0.)
    return top, spread, ns[-1]


def order_interval(a, b, uncertainty):
    if not all(math.isfinite(x) for x in (a, b, uncertainty)) or min(a, b) <= max(SAME, uncertainty):
        return None
    return (math.log2((a-uncertainty)/(b+uncertainty)),
            math.log2((a+uncertainty)/(b-uncertainty)))


def classify(interval):
    if interval is None:
        return 'unresolved'
    if 1.8 <= interval[0] and interval[1] <= 2.3:
        return 'second'
    if interval[1] <= 1.75:
        return 'low'
    return 'mixed'


def cells(rows, name):
    out = []
    for pol in POLS:
        for obs in ('r|r_s', 'r|r_p', 'r|R', 't|t_s', 't|t_p', 't|T'):
            for i, wl in enumerate(WLS):
                runs = [selected(rows, name, n, pol) for n in RUNGS]
                if any(r is None or ridge.validity(r) for r in runs):
                    out.append(dict(pol=pol, observable=obs, wavelength=wl, status='invalid or missing'))
                    continue
                values = [value(r, obs, i) for r in runs]
                ref, u, top = reference(rows, name, i, pol+'|'+obs)
                errors = [abs(v-ref) for v in values]
                t = noise(rows, name, pol, obs, i)
                diffs = [abs(a-b) for a, b in zip(values, values[1:])]
                floor_ok = min(diffs[-2:]) > SAME and 2*t <= FLOOR_SHARE*min(diffs[-2:])
                intervals = []
                direct = []
                for a, b in zip(errors[-3:-1], errors[-2:]):
                    direct.append(math.log2(a/b) if min(a, b) > SAME else None)
                    intervals.append(order_interval(a, b, u+t) if u <= SHARE*min(a,b) and floor_ok else None)
                si = order_interval(diffs[-2], diffs[-1], 2*t) if floor_ok else None
                out.append(dict(pol=pol, observable=obs, wavelength=wl, status='ok', errors=errors,
                                reference_spread=u, reference_top=top, noise=t, floor_ok=floor_ok,
                                direct_orders=direct, direct_intervals=intervals,
                                direct_classes=[classify(x) for x in intervals],
                                self_order=math.log2(diffs[-2]/diffs[-1]) if min(diffs[-2:])>SAME else None,
                                self_interval=si, self_class=classify(si), changes=diffs))
    return out


def run():
    verify()
    import_history()
    replay = execute(fd_job('Y12', 160, 's', 'replay'))
    old = read_rows(LOG)[key(fd_job('Y12', 160, 's'))]
    assert max(abs(value(replay,k,i)-value(old,k,i)) for k in old['values'] for i in range(3)) <= SAME
    print('Historical replay matches.', flush=True)
    for name in SCENES:
        for i in range(3):
            for n in ((0,) if name == 'F12' else (300, 400, 600, 800)):
                execute(ref_job(name, i, n))
    # Interleave small scenes by resolution; each large run gets a fresh process.
    for n in RUNGS:
        for name in SCENES:
            for pol in POLS:
                row = execute(fd_job(name, n, pol))
                if not ridge.settled(row):
                    execute(fd_job(name, n, pol, 'long'))
    for name in SCENES:
        for pol in POLS:
            for variant in ('long', 'pml'):
                execute(fd_job(name, 320, pol, variant))
            rows = read_rows(LOG)
            need = any(c.get('status')=='ok' and c['pol']==pol and not c['floor_ok']
                       and min(c['changes'][-2:])>SAME for c in cells(rows, name))
            if need:
                for variant in ('long', 'pml'):
                    execute(fd_job(name, 640, pol, variant))
    for name in ('Y12', 'Y4', 'D12'):
        for i, wl in enumerate(WLS):
            for n in (1000, 1200):
                current = cells(read_rows(LOG), name)
                need = any(c.get('status')=='ok' and c['wavelength']==wl and min(c['errors'][-3:])>SAME
                           and c['reference_spread']>SHARE*min(c['errors'][-3:]) for c in current)
                if not need:
                    break
                execute(ref_job(name, i, n))
    result = report()
    if result is None or not result['controls_valid']:
        raise RuntimeError('Campaign incomplete or controls invalid; inspect the retained report.')


def report():
    reg = verify()
    rows = read_rows(LOG)
    assert all(r['fingerprint']==reg['fingerprint'] for r in rows.values())
    required = [fd_job(name,n,pol) for name in SCENES for n in RUNGS for pol in POLS]
    required += [ref_job(name,i,n) for name in SCENES for i in range(3)
                 for n in ((0,) if name=='F12' else (300,400,600,800))]
    required += [fd_job(name,320,pol,v) for name in SCENES for pol in POLS for v in ('long','pml')]
    required.append(fd_job('Y12',160,'s','replay'))
    missing = [key(j) for j in required if key(j) not in rows or rows[key(j)].get('status','ok')!='ok']
    if missing:
        print('INCOMPLETE', len(missing), 'required rows:', ', '.join(missing), flush=True)
        return
    result = dict(fingerprint=reg['fingerprint'], complete=True,
                  scope='Diagnostic extension through 640; not a C7 pass or causal identification.',
                  scenes={name: cells(rows,name) for name in SCENES})
    replay=rows[key(fd_job('Y12',160,'s','replay'))]
    original=rows[key(fd_job('Y12',160,'s'))]
    result['replay_gap']=max(abs(value(replay,k,i)-value(original,k,i))
                             for k in original['values'] for i in range(3))
    result['per_wavelength_maxima']={}
    for name, entries in result['scenes'].items():
        maxima=[]
        for wl in WLS:
            for kind in ('amplitude','efficiency'):
                group=[c for c in entries if c.get('status')=='ok' and c['wavelength']==wl
                       and (c['observable'].split('|')[-1] in ('R','T'))==(kind=='efficiency')]
                if group:
                    errors=[max(c['errors'][j] for c in group) for j in range(4)]
                    maxima.append(dict(wavelength=wl,kind=kind,errors=errors,
                                       orders=[math.log2(a/b) if min(a,b)>SAME else None
                                               for a,b in zip(errors,errors[1:])]))
        result['per_wavelength_maxima'][name]=maxima
    lines = [result['scope']]
    for name, entries in result['scenes'].items():
        lines.append(name)
        lines.append('per-wavelength maxima '+json.dumps(result['per_wavelength_maxima'][name]))
        for c in entries:
            lines.append(json.dumps(c, allow_nan=False))
    # Film criterion reads the per-wavelength worst amplitude and power, not zero channels.
    film = []
    for wl in WLS:
        for kind in ('amplitude','efficiency'):
            group=[c for c in result['scenes']['F12'] if c.get('status')=='ok' and c['wavelength']==wl
                   and (c['observable'].split('|')[-1] in ('R','T'))==(kind=='efficiency')]
            e=[max(c['errors'][j] for c in group) for j in range(4)] if group else [float('nan')]*4
            uncertainty=max((c['noise']+c['reference_spread'] for c in group),default=float('inf'))
            interval=order_interval(e[-2],e[-1],uncertainty)
            film.append(dict(wavelength=wl, kind=kind, interval=interval, passed=classify(interval)=='second'))
    result['film']=film
    result['controls_valid']=result['replay_gap']<=SAME and all(x['passed'] for x in film) and all(
        not ridge.validity(r) for r in rows.values() if r['kind']=='fd')
    lines.append('film '+json.dumps(film))
    lines.append('controls_valid '+str(result['controls_valid']))
    REPORT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    TEXT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name, entries in result['scenes'].items():
        counts={k:sum(c.get('self_class')==k for c in entries) for k in ('low','second','mixed','unresolved')}
        print(name, 'self-order cells', counts, flush=True)
    print('Report written; controls_valid',result['controls_valid'],flush=True)
    return result


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        raise SystemExit(worker(json.loads(sys.argv[2]),sys.argv[3]))
    commands={'preflight':preflight,'register':register,'run':run,'report':report}
    for command in sys.argv[1:] or ['report']:
        commands[command]()
