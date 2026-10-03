"""Does C7's epsilon-12 box converge below second order because of its corners?

**The question.** C7 (``studies/fdtd_metasurface_rcwa.py``, ``docs/fd-metasurface-validation.md``)
found its epsilon-12 box under conical incidence (M3) agreeing with RCWA in every cell but
converging at orders 1.51 to 1.75. Subpixel smoothing is second order at a flat interface, but a
sharp dielectric corner carries a field singularity, ``E ~ r^(nu - 1)``, and the smoothing error of
an integrated quantity then falls as ``dx^(2 nu)`` (Farjadpour et al., Opt. Lett. 31, 2972 (2006),
who measured dx^1.4 for tilted-square air holes in epsilon 12; Oskooi et al., Opt. Lett. 34,
2778 (2009): corners leave an error 'between first- and second-order'). This study predicts the
order from the corners' exponents and measures it where it can be measured cleanly.

**The exponents** (:func:`corner_exponent`). Near a vertex where piecewise-constant sectors meet,
the potential is ``r^nu f(theta)`` with ``f = a cos(nu t) + b sin(nu t)`` in each sector; ``f`` and
``eps f'`` are continuous across each ray, so the state ``(f, eps f'/nu)`` crosses a sector of
opening ``alpha`` by ``[[c, s/eps], [-eps s, c]]`` (determinant 1), and a full turn must return it:
``trace(M(nu)) = 2``. The smallest root in (0, 1) is the corner's exponent. Checked: infinite
contrast gives ``2/3`` for a right-angle wedge (``pi`` over the 270-degree exterior), no contrast
gives 1, and the parity form ``eps1 tan(nu a1/2) + eps2 tan(nu a2/2) = 0`` gives the same root.
Two independent derivations agree with every exponent below to 1e-9 (a parity / nodal-system
derivation with the closed forms ``cos(nu pi/2) = |e1 - e2| / (2 (e1 + e2))`` for a right angle
and ``cos^2(nu pi/2) = e3 (e1 - e2)^2 / (2 (e1 + e2)(e1 + e3)(e2 + e3))`` for the 90/90/180
junction, and a spectral-element angular eigenproblem checked by a graded 2D finite-volume
solve; ``results/fdtd_metasurface_corners_crosscheck.md``). For Farjadpour's corner (air inside a
right angle of epsilon 12) the exact value is ``(2/pi) arccos(11/26) = 0.7219`` where the paper
quotes 0.702, which neither derivation could reproduce from the paper's geometry (it is the root
at contrast 19.5); the gap does not move the order away from about 1.4.
:data:`CORNERS` lists the corners of the scenes below:

* a right angle of the pillar against air (its top edges): epsilon 12 -> nu 0.7219, order 1.444;
  epsilon 4 -> 0.8060, 1.612;
* the pillar's base edge, where pillar, air and substrate meet (90 / 90 / 180 degrees):
  epsilon 12 on 2.25 -> 0.6845, 1.369; epsilon 4 on 2.25 -> 0.7955, 1.591.

**The scenes** (:data:`SCENES`). M3's cross-section, the box made a ridge uniform along y (the
y axis is one cell, ``Axis1D``, with the plane wave's ``k_y``), so the only corners are its four
edges and the run is two-dimensional and cheap enough to reach 320 cells per wavelength:

* ``R12`` -- the epsilon-12 ridge 0.30 wide and 0.30 high on epsilon 2.25, cell 1.30, theta 25,
  phi 30 (M3's incidence, layout and placement in cells: axis 0.35 cells past ``floor(cells/2)``,
  substrate plane 0.25 cells above a node plane). Predicted asymptotic order 1.37 (the base edge;
  the top edges 1.44).
* ``R4`` -- the same ridge at epsilon 4. Predicted 1.59 (base; top 1.61).
* ``F12`` -- an epsilon-12 film 0.30 thick on the same glass under the same wave, both transverse
  axes one cell: no corner at all. Predicted 2.

Rungs 40, 80, 160 and 320 cells per vacuum wavelength for all three; s and p incidence; the
three wavelengths of C7. The reference is ``wavebit.rcwa`` with the shape centred (a 1D lamellar
grating on its shape path, where the normal-vector-field rule is Li's), at the run's own angle,
over the truncation ladder (N, 0) for N in 100, 200, 300, 400, 600 (the film: truncation 0, which
is exact), with the lateral phase correction of C7.

**The registered rule** (written before any FDTD row of this study; the corner exponents above
were computed first, and nothing else about the answer was looked at). Cells, ``A(n)``, ``F(n)``
and ``u`` are C7's: the worst amplitude and efficiency errors over every propagating order, both
outgoing polarisations, both incident polarisations and three wavelengths, and the ladder's
spread. The local order of a statistic between the two finest rungs is
``log(X(160)/X(320))/log 2``.

1. *Validity.* Every run settles (both monitors' decay below 1e-5), every value is finite, and
   for each scene ``u_A <= A(320)/3`` and ``u_F <= F(320)/3``.
2. *The control.* ``F12``'s local orders of ``A`` and ``F`` are at least 1.8: without a corner the
   chain -- sources, absorbers, monitors, de-embedding, the substrate and film faces -- is second
   order at these rungs.
3. *The corner order.* ``R12``'s local orders of ``A`` and ``F`` lie in [1.12, 1.69], within 0.25
   of its two predicted orders; ``R4``'s in [1.34, 1.86]; and ``R4``'s exceed ``R12``'s by at
   least 0.05 in both statistics.

The explanation is *supported* when 1, 2 and 3 hold; *contradicted* when 1 holds and either
``R12``'s local orders are both at least 1.85 or ``F12``'s are below 1.8; otherwise the study is
*inconclusive*, with its numbers. The median cell's local order and the three-rung least-squares
orders are reported, not judged. The 0.25 allows for the second-order part of the error still
weighing at 320 cells: the corner term overtakes it only as ``dx^(2 nu - 2)``, slowly.

*Seen before registration.* A dry run of every command on a toy configuration (rungs 40 and 80,
40 periods, an RCWA ladder up to N = 60) was made to test this harness, and its numbers were
seen before this registration: local orders between 40 and 80 of 2.10 / 2.05 (``A`` / ``F``) for
``R12``, 2.00 / 1.99 for ``R4`` and 1.99 / 1.95 for ``F12``
(``results/fdtd_metasurface_corners_dryrun_report.log``). The rule above was written before that
run and is unchanged by it; only this paragraph and the cross-check sentence above were added.

Run: ``python studies/fdtd_metasurface_corners.py [register exponents rcwa fd report]``.
"""
import cmath
import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np                                                                # noqa: E402
import wavebit                                                                    # noqa: E402
import wavebit.fd as fd                                                           # noqa: E402
from wavebit._core.result import Result                                          # noqa: E402

RESULTS = HERE / 'results'
LOG = RESULTS / 'fdtd_metasurface_corners.jsonl'
REPORT = RESULTS / 'fdtd_metasurface_corners_report.log'
REGISTRATION = RESULTS / 'fdtd_metasurface_corners_registration.json'

WAVELENGTHS = (0.95, 1.0, 1.05)
BAND = dict(wavelength=1.0, df=0.04)
SHEET, GAP_R, GAP_T, CLEAR = 0.6, 0.5, 0.5, 0.3
PML_CELLS, PML_REFLECTION, PERIODS = 24, 1e-12, 120
JAX_FROM = 48
RUNGS = (40, 80, 160, 320)
LADDER = (100, 200, 300, 400, 600)
POLS = ('s', 'p')
DECAY_LINE, REFERENCE_SHARE, CONTROL_LINE = 1e-5, 1.0 / 3.0, 1.8
WINDOW, CONTRADICTION, ORDERING = 0.25, 1.85, 0.05
PX, WIDTH, H, EPS_S, THETA, PHI, OFFSET, ZETA = 1.30, 0.30, 0.30, 2.25, 25.0, 30.0, 0.35, 0.25

Q = math.pi / 2
#: name: the sectors (opening, permittivity) counter-clockwise round the vertex
CORNERS = {
    'eps 12 right angle in air': [(Q, 12.0), (3 * Q, 1.0)],
    'eps 12 / air / glass base': [(Q, 12.0), (Q, 1.0), (2 * Q, EPS_S)],
    'eps 4 right angle in air': [(Q, 4.0), (3 * Q, 1.0)],
    'eps 4 / air / glass base': [(Q, 4.0), (Q, 1.0), (2 * Q, EPS_S)],
}
SCENES = {
    'R12': dict(kind='ridge', eps_p=12.0, corners=('eps 12 right angle in air',
                                                   'eps 12 / air / glass base')),
    'R4': dict(kind='ridge', eps_p=4.0, corners=('eps 4 right angle in air',
                                                 'eps 4 / air / glass base')),
    'F12': dict(kind='film', eps_p=12.0, corners=()),
}


# -- the corner exponents ----------------------------------------------------------------------

def corner_trace(nu, sectors):
    M = np.eye(2)
    for alpha, eps in sectors:
        c, s = math.cos(nu * alpha), math.sin(nu * alpha)
        M = np.array([[c, s / eps], [-eps * s, c]]) @ M
    return float(np.trace(M))


def corner_exponent(sectors, n=200001):
    """The smallest ``nu`` in (0, 1) with ``trace(M(nu)) = 2`` (module docstring), by a sign scan
    and bisection; ``1.0`` when there is none (no singularity)."""
    assert abs(sum(a for a, _ in sectors) - 2 * math.pi) < 1e-12
    grid = np.linspace(1e-6, 1.0 - 1e-9, n)
    g = np.array([corner_trace(v, sectors) - 2.0 for v in grid])
    for i in range(n - 1):
        if g[i] * g[i + 1] < 0:
            a, b = grid[i], grid[i + 1]
            for _ in range(200):
                m = 0.5 * (a + b)
                if (corner_trace(a, sectors) - 2) * (corner_trace(m, sectors) - 2) <= 0:
                    b = m
                else:
                    a = m
            return 0.5 * (a + b)
    return 1.0


def predicted_orders(name):
    corners = SCENES[name]['corners']
    return {c: 2 * corner_exponent(CORNERS[c]) for c in corners} or {'none': 2.0}


# -- the scenes --------------------------------------------------------------------------------

def z_grid(n, zeta, z_min, z_max, pml):
    dx = 1.0 / n
    z_off = (-zeta) % 1.0
    k_lo = math.floor((z_min - z_off * dx) / dx + 1e-9) - pml
    k_hi = math.ceil((z_max - z_off * dx) / dx - 1e-9) + pml
    return dx, z_off * dx + k_lo * dx, z_off * dx + k_hi * dx


def centre(n):
    cells = round(PX * n)
    assert abs(PX * n - cells) < 1e-9
    return -PX / 2 + (cells // 2 + OFFSET) / n


def orders(name, side):
    """The orders propagating on ``side`` at every wavelength (a film has the zeroth only)."""
    if SCENES[name]['kind'] == 'film':
        return [0]
    s = math.sin(math.radians(THETA))
    qy = s * math.sin(math.radians(PHI))
    index = 1.0 if side == 'r' else math.sqrt(EPS_S)
    out = []
    for m in range(-5, 6):
        qx = s * math.cos(math.radians(PHI)) + m / PX
        if all(wl * math.hypot(qx, qy) < index for wl in WAVELENGTHS):
            out.append(m)
    return out


def build(name, n, pol, periods=PERIODS):
    g = SCENES[name]
    dx = 1.0 / n
    sheet = H + SHEET
    r_plane, t_plane = sheet + GAP_R, -GAP_T
    _, z_lo, z_hi = z_grid(n, ZETA, t_plane - CLEAR, r_plane + CLEAR, PML_CELLS)
    glass = fd.Material(eps=EPS_S)
    deep = z_lo - 2 * dx
    sub = fd.Structure(fd.Box(size=(10.0, 10.0, abs(deep)), center=(0.0, 0.0, deep / 2)), glass)
    if g['kind'] == 'ridge':
        body = fd.Box(size=(WIDTH, 10.0, H), center=(centre(n), 0.0, H / 2))
        size_x, axis1d = PX, 'y'
    else:
        body = fd.Box(size=(10.0, 10.0, H), center=(0.0, 0.0, H / 2))
        size_x, axis1d = dx, ('x', 'y')
    pillar = fd.Structure(body, fd.Material(eps=g['eps_p']))
    pw = fd.PlaneWave('pw', z=sheet, direction='-z', theta=THETA, phi=PHI, polarization=pol,
                      band=fd.Band(**BAND))
    monitors = (fd.Diffraction('r', z=r_plane, side='reflect', wavelengths=WAVELENGTHS,
                               reference_plane=H, incident_plane=H),
                fd.Diffraction('t', z=t_plane, side='transmit', wavelengths=WAVELENGTHS,
                               reference_plane=0.0, incident_plane=H))
    layer = dict(cells=PML_CELLS, reflection=PML_REFLECTION)
    return fd.Scene(domain=fd.Domain(size=(size_x, dx, z_hi - z_lo),
                                     center=(0.0, 0.0, 0.5 * (z_hi + z_lo))),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw, axis1d=axis1d),
                                             z=(fd.PML(background=glass, **layer),
                                                fd.PML(**layer))),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                    structures=(sub, pillar), sources=(pw,), monitors=monitors)


def phase(name, n, m):
    """C7's lateral correction: fd's order ``m`` referred to the centred ridge."""
    return 1.0 if SCENES[name]['kind'] == 'film' else cmath.exp(2j * math.pi * m * centre(n) / PX)


# -- the RCWA side -----------------------------------------------------------------------------

def rcwa_point(name, wl, N):
    g = SCENES[name]
    th = math.radians(math.degrees(math.asin(math.sin(math.radians(THETA)) * wl)))
    if g['kind'] == 'ridge':
        structure = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0,
                                                       lx=WIDTH, ly=0.5, material=g['eps_p'])])]
        trunc = [N, 0]
    else:
        structure, trunc = [dict(background=g['eps_p'], shapes=[])], [0, 0]
    res = wavebit.rcwa(backend=0, top=1.0, bot=EPS_S, theta=th, phi=math.radians(PHI),
                       psi=math.pi / 2, truncation_order=trunc, period=[PX, 0.5],
                       wavelength=wl, structure=structure, thickness=[H]).solve()
    out = {}
    for pol, sub in (('s', res.s_inc), ('p', res.p_inc)):
        one = Result(given=sub, order_window=res.order_window)
        for side, power in (('r', 'R'), ('t', 'T')):
            for m in orders(name, side):
                o = one.order(m, 0)
                for q in (f'{side}_s', f'{side}_p', power):
                    out[f'{pol}|{side}|{m}|{q}'] = complex(np.asarray(getattr(o, q)))
    return out


# -- rows ---------------------------------------------------------------------------------------

def _c(v):
    return [float(np.real(v)), float(np.imag(v))]


def emit(row):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(LOG, 'a', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(row) + '\n')
    print(json.dumps({k: row[k] for k in ('key', 'status', 'wall') if k in row}), flush=True)


def load():
    out = {}
    if LOG.exists():
        for line in open(LOG, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[row['key']] = row
    return out


def fd_row(name, n, pol):
    key = f'fd|{name}|{n}|{pol}'
    backend = 'jax' if n >= JAX_FROM else 'numpy'
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = fd.run(build(name, n, pol), backend)
        values = {}
        for side, power in (('r', 'R'), ('t', 'T')):
            d = res[side].orders
            for m in orders(name, side):
                o = d.order(m, 0)
                for q in (f'{side}_s', f'{side}_p', power):
                    values[f'{side}|{m}|{q}'] = [_c(v) for v in np.asarray(getattr(o, q))]
        decay = {k: float(v) for k, v in res.diagnostics['decay'].items()}
        return dict(key=key, status='ok', scene=name, n=n, pol=pol, backend=backend,
                    values=values, decay=decay,
                    theta_at=[float(v) for v in np.asarray(res['r'].theta_at)],
                    warnings=sorted({w.category.__name__ for w in caught}),
                    wall=time.perf_counter() - t0)
    except Exception as err:                                               # noqa: BLE001
        return dict(key=key, status='error', error=f'{type(err).__name__}: {err}'[:600],
                    wall=time.perf_counter() - t0)


# -- the commands ------------------------------------------------------------------------------

def file_hash():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def part_register():
    if REGISTRATION.exists():
        print('registered before', flush=True)
        return
    if any(k.startswith('fd|') for k in load()):
        raise SystemExit('fd rows exist: a registration now would postdate data')
    record = dict(file='studies/fdtd_metasurface_corners.py', sha256=file_hash(),
                  registered=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  predicted={name: predicted_orders(name) for name in SCENES})
    RESULTS.mkdir(parents=True, exist_ok=True)
    REGISTRATION.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    (RESULTS / 'fdtd_metasurface_corners_registered.py.txt').write_bytes(
        Path(__file__).read_bytes())
    print(json.dumps(record), flush=True)


def part_exponents():
    for name in CORNERS:
        nu = corner_exponent(CORNERS[name])
        print(f'{name:28s} nu {nu:.7f}  order {2 * nu:.4f}', flush=True)


def part_rcwa():
    done = load()
    for name in SCENES:
        for N in (LADDER if SCENES[name]['kind'] == 'ridge' else (0,)):
            for i, wl in enumerate(WAVELENGTHS):
                key = f'rcwa|{name}|{i}|{N}'
                if key in done:
                    continue
                t0 = time.perf_counter()
                vals = rcwa_point(name, wl, N)
                emit(dict(key=key, status='ok', scene=name, wavelength=wl, N=N,
                          values={k: _c(v) for k, v in vals.items()},
                          wall=time.perf_counter() - t0))


def part_fd():
    done = load()
    for name in SCENES:
        for pol in POLS:
            for n in RUNGS:
                key = f'fd|{name}|{n}|{pol}'
                if key in done and done[key]['status'] == 'ok':
                    continue
                emit(fd_row(name, n, pol))


def worst(values):
    a = np.asarray(list(values), dtype=float)
    return float(np.max(a)) if a.size and not np.any(np.isnan(a)) else float('nan')


def part_report():
    rows = load()
    lines = []

    def say(text):
        lines.append(text)
        print(text, flush=True)

    reg = json.loads(REGISTRATION.read_text(encoding='utf-8')) if REGISTRATION.exists() else {}
    say(f'corners report {time.strftime("%Y-%m-%dT%H:%M:%S%z")}: file {file_hash()[:16]}, '
        f'registered {reg.get("sha256", "never")[:16]} at {reg.get("registered")}'
        f'{"" if reg.get("sha256") == file_hash() else " -- RULE CHANGED"}')
    out = {}
    for name in SCENES:
        Ns = LADDER if SCENES[name]['kind'] == 'ridge' else (0,)
        ref = {}
        for N in Ns:
            for i in range(len(WAVELENGTHS)):
                row = rows.get(f'rcwa|{name}|{i}|{N}')
                if row is None:
                    say(f'{name}: rcwa row {N} {i} missing')
                    return None
                for k, v in row['values'].items():
                    ref.setdefault(N, {}).setdefault(k, [None] * 3)[i] = complex(*v)
        top, lower = ref[Ns[-1]], [ref[N] for N in Ns[-4:-1]] if len(Ns) > 1 else [ref[Ns[-1]]]
        A, F, med, valid = [], [], [], True
        cells = {}
        for n in RUNGS:
            errs_a, errs_f, per_cell = [], [], {}
            for pol in POLS:
                row = rows.get(f'fd|{name}|{n}|{pol}')
                if row is None or row['status'] != 'ok':
                    say(f'{name} {n} {pol}: missing or failed')
                    return None
                if not all(v < DECAY_LINE for v in row['decay'].values()):
                    valid = False
                for k, top_v in top.items():
                    p_, side, m, q = k.split('|')
                    if p_ != pol:
                        continue
                    got = row['values'].get(f'{side}|{m}|{q}')
                    for i in range(3):
                        x = complex(*got[i]) if got else complex('nan')
                        if q not in ('R', 'T'):
                            x *= phase(name, n, int(m))
                        e = abs(x - top_v[i])
                        (errs_f if q in ('R', 'T') else errs_a).append(e)
                        per_cell[(k, i)] = e
            A.append(worst(errs_a))
            F.append(worst(errs_f))
            cells[n] = per_cell
        uA = worst(abs(lo[k][i] - top[k][i]) for lo in lower for k in top for i in range(3)
                   if k.split('|')[-1] not in ('R', 'T'))
        uF = worst(abs(lo[k][i] - top[k][i]) for lo in lower for k in top for i in range(3)
                   if k.split('|')[-1] in ('R', 'T'))
        step = math.log(RUNGS[-1] / RUNGS[-2])
        local_A = math.log(A[-2] / A[-1]) / step
        local_F = math.log(F[-2] / F[-1]) / step
        ls_A = -float(np.polyfit(np.log(RUNGS), np.log(A), 1)[0])
        ls_F = -float(np.polyfit(np.log(RUNGS), np.log(F), 1)[0])
        fine, coarse = cells[RUNGS[-1]], cells[RUNGS[-2]]
        orders_cells = [math.log(coarse[c] / fine[c]) / math.log(RUNGS[-1] / RUNGS[-2])
                        for c in fine if fine[c] > 1e-12 and coarse[c] > 1e-12]
        med_order = float(np.median(orders_cells)) if orders_cells else float('nan')
        judgeable = uA <= REFERENCE_SHARE * A[-1] and uF <= REFERENCE_SHARE * F[-1]
        out[name] = dict(A=A, F=F, local_A=local_A, local_F=local_F, valid=valid and judgeable)
        say(f'{name}: predicted {predicted_orders(name)}; A {", ".join(f"{a:.3e}" for a in A)} '
            f'local {local_A:.3f} (lsq {ls_A:.3f}); F {", ".join(f"{f:.3e}" for f in F)} local '
            f'{local_F:.3f} (lsq {ls_F:.3f}); median cell local order {med_order:.3f}; u '
            f'{uA:.1e}/{uF:.1e}; settled {valid}; judgeable {judgeable}')
    ok1 = all(v['valid'] for v in out.values())
    f12, r12, r4 = out['F12'], out['R12'], out['R4']
    ok2 = f12['local_A'] >= CONTROL_LINE and f12['local_F'] >= CONTROL_LINE
    pr12 = list(predicted_orders('R12').values())
    pr4 = list(predicted_orders('R4').values())
    in12 = all(min(pr12) - WINDOW <= r12[k] <= max(pr12) + WINDOW for k in ('local_A', 'local_F'))
    in4 = all(min(pr4) - WINDOW <= r4[k] <= max(pr4) + WINDOW for k in ('local_A', 'local_F'))
    ordered = all(r4[k] - r12[k] >= ORDERING for k in ('local_A', 'local_F'))
    ok3 = in12 and in4 and ordered
    contradicted = ok1 and ((r12['local_A'] >= CONTRADICTION and r12['local_F'] >= CONTRADICTION)
                            or not ok2)
    verdict = ('supported' if ok1 and ok2 and ok3 else 'contradicted' if contradicted
               else 'inconclusive')
    say(f'1 valid {ok1}; 2 control {ok2}; 3 corner order: R12 in window {in12}, R4 in window '
        f'{in4}, R4 above R12 {ordered}')
    say(f'the corner explanation: {verdict}')
    with open(REPORT, 'a', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n\n')
    return verdict


def main(argv):
    parts = argv or ['register', 'exponents', 'rcwa', 'fd', 'report']
    actions = dict(register=part_register, exponents=part_exponents, rcwa=part_rcwa,
                   fd=part_fd, report=part_report)
    for part in parts:
        if part not in actions:
            raise SystemExit(f'unknown command {part!r}')
        actions[part]()


if __name__ == '__main__':
    main(sys.argv[1:])
