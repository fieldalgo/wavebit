"""Does the junction's rim hold C7's epsilon-12 box below second order? M3 lifted off its glass.

**The question.** C7 (``studies/fdtd_metasurface_rcwa.py``) found its epsilon-12 box under conical
incidence (M3) agreeing with RCWA in every cell but converging at orders 1.51 to 1.75; the
follow-up ``studies/fdtd_metasurface_corners.py`` found M3's cross-section, as a ridge, converging
at second order from 40 to 320 cells per wavelength, so the cause is three-dimensional. The
candidates are the box's vertical edges, its vertices, and the junction's rim where pillar, air
and glass meet along the box's base perimeter and turn its corners. The junction campaign
(``studies/fdtd_subpixel_junction.py``) had found its rim test unresolved on one scene. This study
removes the rim: M3 with its box lifted ``GAP`` off the glass, so that no voxel holds pillar and
glass together, and asks whether the order comes back. Lifting the box also gives its base four
edges and four vertices in air, so the outcome it can show is narrower than "the rim is the
cause": either the order comes back without the rim (the rim is *implicated*), or it does not
(the rim is *not necessary* for the low order -- the vertical edges and vertices suffice).

**The scene.** C7's M3 in every respect -- cell 1.30 x 0.50, epsilon-12 box 0.30 x 0.20 x 0.30 on
epsilon 2.25, theta 25, phi 30, the box's axis 0.35 / 0.15 cells past ``floor(cells/2)``, the
glass face 0.25 cells above a node plane, the same source band, absorbers, run length, monitors
and backend per rung -- except that the box's base sits at ``z = GAP = 0.10``: 4, 8 and 12 cells
at 40, 80 and 120 cells per wavelength, so every face keeps its fraction of a cell at every rung,
and at the lowest rung three voxels of pure air separate the box's smoothed base from the glass's
smoothed face. The reflected amplitudes refer to the box's top, the transmitted ones to the glass
face, both against the incident wave at the top, which is what RCWA quotes for the two-layer
stack (the box over an air layer ``GAP`` thick). The RCWA reference is C7's M3 reference -- the
raster with Li's rule and the exact series, pixel edges on the box's edges, equal half-widths --
with the air layer added under the box, on the ladder (20, 26, 28): C7's ladder without its two
top rungs, which this machine cannot hold (see the registration history below). Only s incidence
is run (C7's slower polarisation of M3).

**Why the rule does not lean on RCWA.** An earlier draft judged the lifted box by C7's statistic,
the local order of the error against the top of the RCWA ladder between 80 and 120 cells. A
review found that statistic still moving with the ladder: C7's flush M3 s reads amplitude /
efficiency local orders 1.42 / 1.23, 1.55 / 1.42, 1.60 / 1.46, 1.61 / 1.51 and 1.65 / 1.53 against
the tops 20, 26, 28, 30 and 32 -- the reference is still converging at the size where thresholds
0.05 apart would decide. So the verdict rests on a statistic with no reference in it: the
*self-convergence order*. With ``D1`` the largest change of a cell between rungs 40 and 80 and
``D2`` the largest between 80 and 120 (over every propagating order, both outgoing polarisations,
the three wavelengths; amplitudes referred to the centred box as in C7), the order ``p`` solves
``(40^-p - 80^-p) / (80^-p - 120^-p) = D1 / D2``; ``p_A`` reads the amplitude cells, ``p_F`` the
efficiency cells. RCWA keeps two jobs: it shows that the lifted runs converge toward the lifted
scene's answer, and it gives a second view of the change, compared between the lifted and the
flush box at the same truncation, where the reference's own error largely cancels.

**Known before registration.** The flush M3 s and C7's controls are C7's rows, read before this
rule was written: the flush self-convergence orders are ``p_A`` 1.633 and ``p_F`` 1.471 (median
cell 1.809), C7's eps-4 cylinder M1 reads 2.04 / 2.03 (s) and 2.14 / 1.97 (p), and M1 s one cell
off its axis reads ``p_A`` 0.92 (0.25 and 0.5 of a cell read 1.61 and 1.05). The registration
records them as computed by this file. No number of the lifted epsilon-12 box was seen: the
harness was tried first on an epsilon-6 box against toy ladders. The earlier draft (SHA-256
``235ff2cd...``, ``GAP`` 0.05, rungs 40 / 60 / 80) read local orders 1.2 / 0.8 against a toy
reference that could not judge. This file (SHA-256 ``f71c2dd2...``, identical to the first
registered one but for this paragraph; rungs 30 / 40 / 60, after 20 / 30 / 40 was refused by the
plan as under-resolved at epsilon 6) read self-convergence orders ``p_A`` 2.30 and ``p_F`` 2.48,
median cell 2.81 -- a different material at different rungs, seen after the rule above was
written and not used to set it. Both dry runs' rows and reports, and the second's copy of this
file, are kept in ``results/fdtd_metasurface_lifted_dryrun/``.

**Registration history.** The rule was first registered at 2026-10-04T11:49:29+0900 (SHA-256
``fe87f8f9...``) with C7's whole ladder, (20, 26, 28, 30, 32). Its campaign was stopped by hand
at 12:04, before any FDTD row, after four RCWA rows (rung 20 at the three wavelengths, rung 26 at
the first): the process had peaked at 16.5 GB on rung 26, and a measurement at truncation 12
(``results/fdtd_metasurface_lifted_memory/``) put a solve's peak allocation at about 34 matrices
of size 2N x 2N for N orders, for one layer or two alike -- some 31 GB at rung 30 and 39 GB at rung
32, against 21 GB of free memory and 31 GB of free commit on this machine that day. (C7's
one-layer ladder needs the same per solve; it finished on 2026-10-03.) The rule was registered
again with the ladder ending at 28 (about 23 GB) and the reference's view at the tops 20, 26 and
28, where C7's flush rows exist; nothing else changed. The four RCWA rows are reused: their key
(the ``rcwa`` digest) covers the solve, not the ladder. The first registration and the stopped
campaign's log are kept beside the new ones.

**The registered rule.**

1. *Validity.* Every rung run of the lifted box is valid by C7's criterion 1 (it settles below
   the decay line or its one 180-period repeat does, no growth or unsettled warning, not marked
   unstable, ``theta_at`` on the fixed-k angle, every cell present and finite); its middle rung
   at 180 periods differs from the 120-period run by at most ``FLOOR_SHARE`` (0.1) of ``D2``, for
   amplitudes and efficiencies -- the run length moves the cells far less than the step the
   statistic reads; the lifted box converges toward its reference: against the top of its
   ladder ``A`` and ``F`` fall at both rung steps and their local orders between 80 and 120 are
   at least ``TOWARD`` (1.0); the flush rows are C7's under C7's provenance and valid by the
   same criteria (C7's own run-length check included), at the same rungs and the same three
   ladder tops as the lifted ones.
2. *Controls* (the statistic reads second order where it is, and a first-order error as one).
   On C7's M1 rows (rungs 20 / 40 / 80) ``p_A`` and ``p_F`` are at least ``ORDER_LINE`` (1.8)
   for s and for p; with M1 s's amplitudes referred to a pillar one cell off its axis along x and
   y at every rung -- an error first order in the cell -- ``p_A`` is at most ``FIRST_LINE``
   (1.5).
3. *The primary verdict*, from ``dA = p_A(lifted) - p_A(flush)`` and ``dF`` likewise:
   *implicated* when ``dA`` and ``dF`` are both at least ``RISE`` (0.2) and the lifted ``p_A``
   and ``p_F`` both reach ``ORDER_LINE``; *partial* when both rise by ``RISE`` but the lifted
   orders do not both reach the line; *not necessary* when ``dA`` and ``dF`` are both at most
   ``FLAT`` (0.1) -- the lift does not raise the order; otherwise *inconclusive*. ``RISE`` is
   half the step from the flush box's orders to second order, ``FLAT`` a quarter of it.
4. *The reference's view.* For each of the ladder's three tops, the local orders between 80 and
   120 of ``A`` and ``F`` against that top, lifted minus flush, are ``dA_t`` and ``dF_t``. An
   *implicated* or *partial* verdict stands only if every ``dA_t`` and ``dF_t`` exceeds
   ``FLAT``; a *not necessary* verdict only if every one is below ``RISE``. Otherwise the
   verdict is *inconclusive (the reference disagrees)*.

The report always prints both boxes' errors, the self-convergence and reference orders, the
median cell's self-convergence order, C7's judgeability (``u`` at most a third of the finest
error) and C7's ``project`` line (``u`` against a third of the 40-cell error carried to 120 at
order 1.8) for information; neither of the last two gates the verdict. If this file no longer
hashes to its registration, the verdict is withheld.

Run: ``python studies/fdtd_metasurface_lifted.py [register rcwa fd report]``.
"""
import hashlib
import inspect
import json
import math
import sys
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fdtd_metasurface_rcwa as c7                                               # noqa: E402
import numpy as np                                                                # noqa: E402
import wavebit                                                                    # noqa: E402
import wavebit.fd as fd                                                           # noqa: E402
from scipy.optimize import brentq                                                 # noqa: E402
from wavebit._core.result import Result                                          # noqa: E402

RESULTS = HERE / 'results'
LOG = RESULTS / 'fdtd_metasurface_lifted.jsonl'
REPORT = RESULTS / 'fdtd_metasurface_lifted_report.log'
REGISTRATION = RESULTS / 'fdtd_metasurface_lifted_registration.json'

NAME, POL = 'M3', 's'
GAP = 0.10
RUNGS = (40, 80, 120)
LADDER = ((20, 20), (26, 26), (28, 28))
#: the scene's material (the harness was tried on 6.0 before registration)
EPS_P = 12.0
#: C7's flush M3: its rungs and the ladder tops the reference's view compares at
FLUSH_RUNGS = (40, 80, 120)
FLUSH_TOPS = ((20, 20), (26, 26), (28, 28))
#: the control scene (C7's M1) and its rungs
CONTROL = 'M1'

#: The rule's lines (module docstring).
ORDER_LINE, FIRST_LINE = 1.8, 1.5
RISE, FLAT = 0.2, 0.1
TOWARD = 1.0
FLOOR_SHARE = c7.FLOOR_SHARE
#: a cell whose change between the finer rungs is below this is left out of the median
TINY = 1e-12


def tops():
    """The three ladder tops the reference's view reads (the last three of ``LADDER``)."""
    return tuple(LADDER[-3:])


# -- the scene and its reference ---------------------------------------------------------------

def build(n, periods=c7.PERIODS):
    """C7's M3 at ``n`` cells per wavelength with the box lifted ``GAP`` off the glass."""
    g = c7.SCENES[NAME]
    (Px, Py), H = g['P'], g['H']
    top = GAP + H
    sheet = top + c7.SHEET
    r_plane, t_plane = sheet + c7.GAP_R, -c7.GAP_T
    dx, z_lo, z_hi = c7.z_grid(n, g['zeta'], t_plane - c7.CLEAR, r_plane + c7.CLEAR,
                               c7.PML_CELLS)
    xc, yc = c7.centre(NAME, n)
    glass = fd.Material(eps=g['eps_s'])
    deep = z_lo - 2 * dx                  # the substrate runs through the z- PML
    sub = fd.Structure(fd.Box(size=(10.0, 10.0, abs(deep)), center=(0.0, 0.0, deep / 2)), glass)
    box = fd.Structure(fd.Box(size=(g['size'][0], g['size'][1], H),
                              center=(xc, yc, GAP + H / 2)), fd.Material(eps=EPS_P))
    pw = fd.PlaneWave('pw', z=sheet, direction='-z', theta=g['theta'], phi=g['phi'],
                      polarization=POL, band=fd.Band(**c7.BAND))
    monitors = (fd.Diffraction('r', z=r_plane, side='reflect', wavelengths=c7.WAVELENGTHS,
                               reference_plane=top, incident_plane=top),
                fd.Diffraction('t', z=t_plane, side='transmit', wavelengths=c7.WAVELENGTHS,
                               reference_plane=0.0, incident_plane=top))
    layer = dict(cells=c7.PML_CELLS, reflection=c7.PML_REFLECTION)
    scene = fd.Scene(domain=fd.Domain(size=(Px, Py, z_hi - z_lo),
                                      center=(0.0, 0.0, 0.5 * (z_hi + z_lo))),
                     boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                              z=(fd.PML(background=glass, **layer),
                                                 fd.PML(**layer))),
                     resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                     structures=(sub, box), sources=(pw,), monitors=monitors)
    return scene, dict(dx=dx, xc=xc, yc=yc, z_lo=z_lo, z_hi=z_hi)


def rcwa_point(wl, trunc):
    """``{key: complex}`` over every cell at ``wl``: the box layer over an air layer ``GAP``."""
    g = c7.SCENES[NAME]
    box = c7.box_raster(NAME, g['size'])
    box = np.where(box > 1.0, EPS_P, 1.0)
    structure = np.concatenate([box, np.ones_like(box)])
    res = wavebit.rcwa(backend=0, top=1.0, bot=g['eps_s'],
                       theta=math.radians(c7.theta_at(NAME, wl)), phi=math.radians(g['phi']),
                       psi=math.pi / 2, truncation_order=list(trunc), period=list(g['P']),
                       wavelength=wl, structure=structure, thickness=[g['H'], GAP],
                       fourier_coefficients='integrated', factorization='li').solve()
    one = Result(given=res.s_inc, order_window=res.order_window)
    out = {}
    for side, power in (('r', 'R'), ('t', 'T')):
        for (m, n) in c7.orders(NAME, side):
            o = one.order(m, n)
            for q in (f'{side}_s', f'{side}_p', power):
                out[f'{side}|{m}|{n}|{q}'] = complex(np.asarray(getattr(o, q)))
    return out


# -- provenance and rows -----------------------------------------------------------------------

def provenance():
    """``fd`` keys the lifted fd rows (wavebit, the functions and constants an fd row's numbers
    come from, the versions), ``rcwa`` the lifted ladder rows; ``c7`` and ``c7_rcwa`` are C7's
    keys for the flush rows this study reads (C7's ``provenance()``)."""
    c7.PROV = c7.provenance()
    fd_tree, rest = c7.PROV['fd_tree'], c7.PROV['wavebit_rest']
    versions = c7.PROV['versions']
    scene = {k: v for k, v in c7.SCENES[NAME].items() if k not in c7.RCWA_ONLY}
    fd_code = [inspect.getsource(f) for f in (build, fd_row, c7.centre, c7.theta_at, c7.z_grid,
                                              c7.orders, c7.propagating, c7.order_q, c7._c)]
    constants = dict(NAME=NAME, POL=POL, GAP=GAP, EPS_P=EPS_P, scene=scene,
                     WAVELENGTHS=c7.WAVELENGTHS, BAND=c7.BAND, SHEET=c7.SHEET, GAP_R=c7.GAP_R,
                     GAP_T=c7.GAP_T, CLEAR=c7.CLEAR, PML_CELLS=c7.PML_CELLS,
                     PML_REFLECTION=c7.PML_REFLECTION, JAX_FROM=c7.JAX_FROM)
    fd_digest = hashlib.sha256(json.dumps([fd_tree, rest, fd_code, constants, versions],
                                          sort_keys=True, default=repr).encode()).hexdigest()[:16]
    rcwa_code = [inspect.getsource(f) for f in (rcwa_point, c7.box_raster, c7.orders,
                                                c7.propagating, c7.order_q, c7.theta_at)]
    rcwa_fields = {k: c7.SCENES[NAME].get(k) for k in c7.RCWA_FIELDS}
    rcwa_digest = hashlib.sha256(json.dumps(
        [rest, rcwa_code, rcwa_fields, GAP, EPS_P, repr(c7.WAVELENGTHS), versions],
        sort_keys=True, default=repr).encode()).hexdigest()[:16]
    return dict(fd=fd_digest, rcwa=rcwa_digest, c7=c7.PROV['digest'],
                c7_rcwa=c7.PROV['rcwa'][NAME], commit=c7.PROV['commit'],
                dirty=c7.PROV['dirty'], versions=versions, env=c7.PROV['env'])


PROV = None


def emit(row):
    c7.emit(LOG, row)


def load():
    """``{key: row}``, the last row per key under the current provenance; a line a killed run
    cut short (or padded with NULs) is skipped with a note."""
    out = {}
    if LOG.exists():
        for k, line in enumerate(open(LOG, encoding='utf-8')):
            line = line.replace('\0', '').strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f'note: {LOG.name} line {k + 1} is not a whole row; skipped', flush=True)
                continue
            if row.get('prov') == PROV.get(row.get('kind')):
                out[row['key']] = row
    return out


def fd_key(n, periods):
    return f'fd|{n}|{periods}|{"jax" if n >= c7.JAX_FROM else "numpy"}'


def fd_row(n, periods):
    """One public solve as a JSON row, as C7's ``fd_row`` writes it."""
    backend = 'jax' if n >= c7.JAX_FROM else 'numpy'
    s = dict(scene=f'{NAME}-lifted', n=n, pol=POL, periods=periods, backend=backend)
    base = dict(kind='fd', key=fd_key(n, periods), prov=PROV['fd'], spec=s,
                commit=PROV['commit'], dirty=PROV['dirty'], env=PROV['env'])
    t0 = time.perf_counter()
    stage = 'plan'
    try:
        scene, info = build(n, periods)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plan = fd.plan(scene)
            stage = 'run'
            res = fd.run(scene, plan=plan, backend=backend)
        stage = 'read'
        values = {}
        for side, power in (('r', 'R'), ('t', 'T')):
            d = res[side].orders
            for (m, k) in c7.orders(NAME, side):
                o = d.order(m, k)
                for q in (f'{side}_s', f'{side}_p', power):
                    values[f'{side}|{m}|{k}|{q}'] = [c7._c(v) for v in np.asarray(getattr(o, q))]
        names = sorted({f'{w.category.__name__}: {str(w.message)[:120]}' for w in caught})
        decay = {k: (None if v is None else float(v))
                 for k, v in (res.diagnostics.get('decay') or {}).items()}
        stable = res.diagnostics.get('stable')
        row = dict(base, status='ok', values=values, decay=decay, warnings=names,
                   theta_at=[float(v) for v in np.asarray(res['r'].theta_at)],
                   stable=stable if isinstance(stable, bool) else None, steps=int(plan.steps),
                   grid=list(plan.grid.n), centre=[info['xc'], info['yc']],
                   wall=time.perf_counter() - t0)
        json.dumps(row, default=c7._json)
        return row
    except Exception as err:                                               # noqa: BLE001
        return dict(base, status='error', stage=stage, error=f'{type(err).__name__}: {err}'[:600],
                    trace=traceback.format_exc()[-2000:], wall=time.perf_counter() - t0)


def file_hash():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


# -- the statistics ----------------------------------------------------------------------------

def cells(name, row, n, misplace=0.0):
    """``{(key, i): complex}`` over ``name``'s cells and wavelengths, amplitudes referred to the
    centred pillar (C7's ``phase``); ``misplace`` refers them to a pillar that many cells off its
    axis along x and y. A cell the row does not hold is NaN."""
    out = {}
    Px, Py = c7.SCENES[name]['P']
    for key in c7.cell_keys(name):
        side, m, k, q = key.split('|')
        got = row['values'].get(key)
        for i in range(len(c7.WAVELENGTHS)):
            x = complex(*got[i]) if got else complex('nan')
            if q not in ('R', 'T'):
                x *= c7.phase(name, n, int(m), int(k))
                if misplace:
                    x *= complex(np.exp(2j * math.pi * misplace / n * (int(m) / Px
                                                                        + int(k) / Py)))
            out[(key, i)] = x
    return out


def is_eff(cell):
    return cell[0].split('|')[-1] in ('R', 'T')


def ratio_order(r, rungs):
    """The order ``p`` with ``(a^-p - b^-p) / (b^-p - c^-p) = r`` for rungs ``(a, b, c)``; NaN
    when no ``p`` in (0.05, 8) gives ``r``."""
    a, b, c = rungs

    def f(p):
        return (a ** -p - b ** -p) / (b ** -p - c ** -p) - r
    lo, hi = 0.05, 8.0
    if not (math.isfinite(r) and f(lo) < 0.0 < f(hi)):
        return float('nan')
    return float(brentq(f, lo, hi, xtol=1e-12))


def self_orders(name, runs, rungs, misplace=0.0):
    """The self-convergence orders (module docstring): ``dict(pA, pF, median, D)`` with
    ``D[kind] = (D1, D2)``, the largest change of a cell between the coarser and the finer
    pair of rungs."""
    v = [cells(name, runs[n], n, misplace) for n in rungs]
    d1 = {c: abs(v[0][c] - v[1][c]) for c in v[0]}
    d2 = {c: abs(v[1][c] - v[2][c]) for c in v[0]}
    D = {}
    for kind, pick in (('amp', lambda c: not is_eff(c)), ('eff', is_eff)):
        D[kind] = (c7.worst(d1[c] for c in d1 if pick(c)), c7.worst(d2[c] for c in d2 if pick(c)))
    per = [ratio_order(d1[c] / d2[c], rungs) for c in d1 if d2[c] > TINY and d1[c] > TINY]
    per = [p for p in per if math.isfinite(p)]
    return dict(pA=ratio_order(D['amp'][0] / D['amp'][1], rungs),
                pF=ratio_order(D['eff'][0] / D['eff'][1], rungs),
                median=float(np.median(per)) if per else float('nan'), D=D)


def against(name, runs, rungs, ref):
    """``(A, F, (local A, local F))``: the largest amplitude and efficiency errors at each rung
    against ``ref`` (``{key: [complex per wavelength]}``) and their local orders between the last
    two rungs."""
    A, F = [], []
    for n in rungs:
        v = cells(name, runs[n], n)
        A.append(c7.worst(abs(x - ref[c[0]][c[1]]) for c, x in v.items() if not is_eff(c)))
        F.append(c7.worst(abs(x - ref[c[0]][c[1]]) for c, x in v.items() if is_eff(c)))
    step = math.log(rungs[-1] / rungs[-2])
    local = tuple(math.log(X[-2] / X[-1]) / step if X[-1] > 0 and X[-2] > 0 else float('nan')
                  for X in (A, F))
    return A, F, local


def spread(ref, lower):
    """``(u_A, u_F)``: the largest distance of a lower truncation's cell from ``ref``."""
    out = []
    for eff in (False, True):
        out.append(c7.worst(abs(lo[k][i] - ref[k][i]) for lo in lower for k in ref
                            for i in range(len(c7.WAVELENGTHS))
                            if (k.split('|')[-1] in ('R', 'T')) == eff))
    return tuple(out)


# -- the commands ------------------------------------------------------------------------------

def known_before():
    """The numbers of C7's rows the rule was written knowing: the flush M3 s self-convergence
    orders and the controls' (module docstring)."""
    rows = c7.load(c7.LOG)
    out = {}
    flush = {n: c7.rung_run(rows, NAME, n, POL)[0] for n in FLUSH_RUNGS}
    if all(r is not None and r.get('status') == 'ok' for r in flush.values()):
        out['flush M3 s'] = self_orders(NAME, flush, FLUSH_RUNGS)
    rungs = c7.SCENES[CONTROL]['rungs']
    for pol in c7.POLS:
        runs = {n: c7.rung_run(rows, CONTROL, n, pol)[0] for n in rungs}
        if all(r is not None and r.get('status') == 'ok' for r in runs.values()):
            out[f'{CONTROL} {pol}'] = self_orders(CONTROL, runs, rungs)
            if pol == 's':
                out[f'{CONTROL} s misplaced'] = self_orders(CONTROL, runs, rungs, misplace=1.0)
    return {k: {kk: (vv if kk != 'D' else {a: list(b) for a, b in vv.items()})
                for kk, vv in v.items()} for k, v in out.items()}


def part_register():
    """Record the rule before any lifted FDTD row: this file's SHA-256, the time, the commit,
    the digests and the numbers of C7's rows it was written knowing (written once; a superseded
    registration is moved aside by hand and kept)."""
    if REGISTRATION.exists():
        print(f'{REGISTRATION.name} exists; the rule was registered before', flush=True)
        return
    if LOG.exists() and any('"kind": "fd"' in line for line in open(LOG, encoding='utf-8')):
        raise SystemExit(f'{LOG.name} already holds fd rows: a registration now would postdate '
                         f'data')
    record = dict(file='studies/fdtd_metasurface_lifted.py', sha256=file_hash(),
                  registered=time.strftime('%Y-%m-%dT%H:%M:%S%z'), commit=PROV['commit'],
                  dirty=PROV['dirty'], digests={k: PROV[k] for k in ('fd', 'rcwa', 'c7',
                                                                       'c7_rcwa')},
                  known_before=known_before())
    RESULTS.mkdir(parents=True, exist_ok=True)
    REGISTRATION.write_text(json.dumps(record, indent=1, default=c7._json) + '\n',
                            encoding='utf-8', newline='\n')
    print(json.dumps(record, default=c7._json), flush=True)


def ladder_key(i, trunc):
    return f'rcwa|{i}|{trunc[0]}x{trunc[1]}'


def part_rcwa():
    done = load()
    for trunc in LADDER:
        for i, wl in enumerate(c7.WAVELENGTHS):
            key = ladder_key(i, trunc)
            if key in done:
                continue
            t0 = time.perf_counter()
            vals = rcwa_point(wl, trunc)
            row = dict(kind='rcwa', key=key, prov=PROV['rcwa'], wavelength=wl,
                       truncation=list(trunc), values={k: c7._c(v) for k, v in vals.items()},
                       wall=time.perf_counter() - t0)
            emit(row)
            done[key] = row


def part_fd():
    """Every rung at 120 periods, the middle rung at 180 (the run-length check), and the one
    180-period repeat of a rung whose 120-period run is over the decay line."""
    done = load()
    wanted = [(n, c7.PERIODS) for n in RUNGS] + [(RUNGS[1], c7.LONG_PERIODS)]
    for n, periods in wanted:
        key = fd_key(n, periods)
        row = done.get(key)
        if row is None or row.get('status') == 'error':
            row = fd_row(n, periods)
            emit(row)
            done[key] = row
        if periods == c7.PERIODS and row.get('status') == 'ok' and not c7.settled(row):
            again = fd_key(n, c7.LONG_PERIODS)
            if done.get(again) is None or done[again].get('status') == 'error':
                done[again] = fd_row(n, c7.LONG_PERIODS)
                emit(done[again])


def rung_run(rows, n):
    """The run that stands for a rung (C7's ``rung_run`` for the lifted rows)."""
    first = rows.get(fd_key(n, c7.PERIODS))
    if first is not None and first.get('status') == 'ok' and not c7.settled(first):
        return rows.get(fd_key(n, c7.LONG_PERIODS)), True
    return first, False


def ladder_values(rows, trunc, key_of, pick):
    """``{key: [complex per wavelength]}`` at ``trunc`` from ladder rows, ``None`` if missing."""
    out = {}
    for i in range(len(c7.WAVELENGTHS)):
        row = rows.get(key_of(i, trunc))
        if row is None:
            return None
        for k, v in pick(row).items():
            out.setdefault(k, [None] * len(c7.WAVELENGTHS))[i] = complex(*v)
    return out


def rows_digest(rows):
    """A short digest of the values a report read (which rows the verdict stands on)."""
    return hashlib.sha256(json.dumps([[r.get('key'), r.get('values')] for r in rows],
                                     sort_keys=True).encode()).hexdigest()[:16]


def part_report():
    """The report, written to ``REPORT`` whatever happens on the way."""
    lines = []

    def say(text):
        lines.append(text)
        print(text, flush=True)
    verdict = 'incomplete'
    try:
        verdict = _report(say)
    except Exception as err:                                               # noqa: BLE001
        say(f'report failed: {type(err).__name__}: {err}')
        say(traceback.format_exc()[-1500:])
        verdict = 'incomplete'
    finally:
        say(f'verdict: {verdict}')
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(REPORT, 'a', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(lines) + '\n\n')
    return verdict


def _fmt(X):
    return ', '.join(f'{x:.3e}' for x in X)


def _report(say):
    reg = json.loads(REGISTRATION.read_text(encoding='utf-8')) if REGISTRATION.exists() else {}
    same_rule = reg.get('sha256') == file_hash()
    say(f'lifted report {time.strftime("%Y-%m-%dT%H:%M:%S%z")}: file {file_hash()[:16]}, '
        f'registered {reg.get("sha256", "never")[:16]} at {reg.get("registered")}'
        f'{"" if same_rule else " -- RULE CHANGED"}; digests fd {PROV["fd"]} rcwa '
        f'{PROV["rcwa"]} c7 {PROV["c7"]} c7_rcwa {PROV["c7_rcwa"]}')
    bad = []
    if (RUNGS, tops()) != (FLUSH_RUNGS, FLUSH_TOPS):
        bad.append(f'the lifted rungs / tops {RUNGS} / {tops()} are not the flush '
                   f'{FLUSH_RUNGS} / {FLUSH_TOPS}')

    # 1. the lifted runs
    rows = load()
    runs, repeats = {}, {}
    for n in RUNGS:
        runs[n], repeats[n] = rung_run(rows, n)
        why = c7.validity(NAME, runs[n])
        if why:
            bad.append(f'lifted {n}: {why}')
    if any(not r or r.get('status') != 'ok' for r in runs.values()):
        say(f'lifted runs missing or failed: {bad}')
        return 'incomplete'
    lifted = self_orders(NAME, runs, RUNGS)
    mid = rows.get(fd_key(RUNGS[1], c7.PERIODS))
    long_ = rows.get(fd_key(RUNGS[1], c7.LONG_PERIODS))
    why = c7.validity(NAME, mid, settle=False) + c7.validity(NAME, long_)
    if why:
        bad.append(f'run-length pair: {why}')
    else:
        dA, dF = c7.gap_over_cells(NAME, mid, long_)
        lineA, lineF = (FLOOR_SHARE * lifted['D'][k][1] for k in ('amp', 'eff'))
        say(f'lifted run length: {RUNGS[1]} at {c7.PERIODS} vs {c7.LONG_PERIODS} periods moves '
            f'amplitudes {dA:.1e} (line {lineA:.1e}), efficiencies {dF:.1e} (line {lineF:.1e})')
        if not (dA <= lineA and dF <= lineF):
            bad.append('lifted run length over the line')
    say(f'lifted rows {rows_digest([runs[n] for n in RUNGS] + [long_])}; repeats '
        f'{[n for n in RUNGS if repeats[n]]}; walls '
        f'{[round(runs[n]["wall"]) for n in RUNGS]} s')

    # 2. the flush rows (C7's, under C7's provenance)
    c7rows = c7.load(c7.LOG)
    flush, frep = {}, {}
    for n in FLUSH_RUNGS:
        flush[n], frep[n] = c7.rung_run(c7rows, NAME, n, POL)
        why = c7.validity(NAME, flush[n])
        if why:
            bad.append(f'flush {n}: {why}')
    if any(not r or r.get('status') != 'ok' for r in flush.values()):
        say(f'flush rows missing: {bad}')
        return 'invalid'
    base = self_orders(NAME, flush, FLUSH_RUNGS)
    fmid = c7rows.get(c7.key_of(c7.spec(NAME, FLUSH_RUNGS[1], POL)))
    flong = c7rows.get(c7.key_of(c7.spec(NAME, FLUSH_RUNGS[1], POL, periods=c7.LONG_PERIODS)))
    why = c7.validity(NAME, fmid, settle=False) + c7.validity(NAME, flong)
    if why:
        bad.append(f'flush run-length pair: {why}')
    else:
        dA, dF = c7.gap_over_cells(NAME, fmid, flong)
        lineA, lineF = (FLOOR_SHARE * base['D'][k][1] for k in ('amp', 'eff'))
        say(f'flush run length: amplitudes {dA:.1e} (line {lineA:.1e}), efficiencies {dF:.1e} '
            f'(line {lineF:.1e})')
        if not (dA <= lineA and dF <= lineF):
            bad.append('flush run length over the line')
    say(f'flush rows {rows_digest([flush[n] for n in FLUSH_RUNGS] + [flong])}')

    # 3. the controls
    control = {}
    crungs = c7.SCENES[CONTROL]['rungs']
    for pol in c7.POLS:
        cruns = {n: c7.rung_run(c7rows, CONTROL, n, pol)[0] for n in crungs}
        why = [w for n in crungs for w in c7.validity(CONTROL, cruns[n])]
        if why:
            bad.append(f'control {CONTROL} {pol}: {why}')
            continue
        control[pol] = self_orders(CONTROL, cruns, crungs)
        if pol == 's':
            control['misplaced'] = self_orders(CONTROL, cruns, crungs, misplace=1.0)
    for pol in c7.POLS:
        c = control.get(pol)
        if c:
            ok = c['pA'] >= ORDER_LINE and c['pF'] >= ORDER_LINE
            say(f'control {CONTROL} {pol}: p_A {c["pA"]:.3f} p_F {c["pF"]:.3f} (both at least '
                f'{ORDER_LINE}): {"holds" if ok else "FAILS"}')
            if not ok:
                bad.append(f'control {CONTROL} {pol} fails')
    c = control.get('misplaced')
    if c:
        ok = c['pA'] <= FIRST_LINE
        say(f'control {CONTROL} s one cell off its axis: p_A {c["pA"]:.3f} (at most '
            f'{FIRST_LINE}): {"holds" if ok else "FAILS"}')
        if not ok:
            bad.append('the first-order control fails')

    # 4. the references
    ladder = {t: ladder_values(rows, t, ladder_key, lambda r: r['values']) for t in LADDER}
    c7rcwa = c7.load(c7.RCWA_LOG)
    fladder = {t: ladder_values(c7rcwa, t, lambda i, tr: c7.ladder_key(NAME, i, tr),
                                lambda r: r['values'][POL])
               for t in c7.SCENES[NAME]['ladder']}
    missing = [t for t, v in ladder.items() if v is None] + \
        [('flush', t) for t in FLUSH_TOPS if fladder.get(t) is None]
    if missing:
        say(f'ladder rows missing: {missing}')
        return 'incomplete' if any(not isinstance(t[0], str) for t in missing) else 'invalid'
    top = ladder[LADDER[-1]]
    A, F, local = against(NAME, runs, RUNGS, top)
    uA, uF = spread(top, [ladder[t] for t in LADDER[:-1]])
    toward = (A[0] > A[1] > A[2] and F[0] > F[1] > F[2] and local[0] >= TOWARD
              and local[1] >= TOWARD)
    say(f'lifted against its top {LADDER[-1]}: A {_fmt(A)}; F {_fmt(F)}; local orders A '
        f'{local[0]:.3f} F {local[1]:.3f}; u {uA:.1e}/{uF:.1e}; judgeable (C7) '
        f'{uA <= c7.REFERENCE_SHARE * A[-1] and uF <= c7.REFERENCE_SHARE * F[-1]}; project '
        f'(C7) u against {c7.REFERENCE_SHARE * A[0] * (RUNGS[0] / RUNGS[-1]) ** ORDER_LINE:.1e}'
        f'/{c7.REFERENCE_SHARE * F[0] * (RUNGS[0] / RUNGS[-1]) ** ORDER_LINE:.1e}; toward '
        f'{toward}')
    if not toward:
        bad.append('the lifted runs do not converge toward their reference')
    view = {}
    for t, ft in zip(tops(), FLUSH_TOPS):
        lA, lF, ll = against(NAME, runs, RUNGS, ladder[t])
        fA, fF, fl = against(NAME, flush, FLUSH_RUNGS, fladder[ft])
        view[t] = (ll[0] - fl[0], ll[1] - fl[1])
        say(f'  top {t}: lifted A {_fmt(lA)} F {_fmt(lF)} local {ll[0]:.3f}/{ll[1]:.3f}; flush '
            f'A {_fmt(fA)} F {_fmt(fF)} local {fl[0]:.3f}/{fl[1]:.3f}; lifted - flush '
            f'{view[t][0]:+.3f}/{view[t][1]:+.3f}')

    # 5. the statistic
    dA, dF = lifted['pA'] - base['pA'], lifted['pF'] - base['pF']
    for label, s in (('lifted', lifted), ('flush', base)):
        say(f'{label} self-convergence: D1/D2 amp {s["D"]["amp"][0]:.3e}/{s["D"]["amp"][1]:.3e} '
            f'eff {s["D"]["eff"][0]:.3e}/{s["D"]["eff"][1]:.3e}; p_A {s["pA"]:.3f} p_F '
            f'{s["pF"]:.3f}; median cell {s["median"]:.3f}')
    say(f'lifted - flush: dA {dA:+.3f} dF {dF:+.3f}')
    if not all(math.isfinite(x) for x in (lifted['pA'], lifted['pF'], base['pA'], base['pF'])):
        bad.append('a self-convergence order is undefined')
    if bad:
        say(f'not valid: {bad}')
    if not same_rule:
        return 'withheld (the rule changed since its registration)'
    if bad:
        return 'invalid'
    if dA >= RISE and dF >= RISE:
        primary = ('the rim is implicated' if lifted['pA'] >= ORDER_LINE
                   and lifted['pF'] >= ORDER_LINE else 'partial')
    elif dA <= FLAT and dF <= FLAT:
        primary = 'the rim is not necessary'
    else:
        return 'inconclusive'
    deltas = [d for pair in view.values() for d in pair]
    if primary == 'the rim is not necessary':
        agrees = all(d < RISE for d in deltas)
    else:
        agrees = all(d > FLAT for d in deltas)
    return primary if agrees else 'inconclusive (the reference disagrees)'


def main(argv):
    global PROV
    PROV = provenance()
    print(json.dumps({k: PROV[k] for k in ('fd', 'rcwa', 'c7', 'c7_rcwa', 'commit', 'dirty')}),
          flush=True)
    parts = argv or ['register', 'rcwa', 'fd', 'report']
    actions = dict(register=part_register, rcwa=part_rcwa, fd=part_fd, report=part_report)
    for part in parts:
        if part not in actions:
            raise SystemExit(f'unknown command {part!r}: one of {sorted(actions)}')
        if part in ('rcwa', 'fd') and not REGISTRATION.exists():
            raise SystemExit('register the rule before any row')
        actions[part]()


if __name__ == '__main__':
    main(sys.argv[1:])
