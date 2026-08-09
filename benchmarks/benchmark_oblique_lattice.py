"""
benchmark_oblique_lattice.py — non-orthogonal cells: what they cost, and whether they are right.

Every number quoted in ``docs/oblique-lattice.md`` is produced here. Run it to reproduce them,
or after changing anything the ``lattice`` basis reaches.

Five independent checks, in decreasing order of how much they prove:

    published   Li's checkerboard against the table in the literature. The only check here
                against numbers that came from neither this codebase nor a tool run by us.
    s4          S4 as an outside oracle, *with a square-lattice control* — the control is the
                point, since it separates a real disagreement from S4's own convergence.
    shear       A unimodular basis change, which describes the identical point lattice, so the
                two runs must agree. The cheapest check, and the one to run on your own
                structure.
    supercell   A native hex cell against the rectangular supercell it replaces: same physics,
                two cells.
    cost        What the feature buys — order counts, the measured cost exponent, wall clock.

Deliberately absent: energy conservation. ``R + T == 1`` holds to 1e-15 on this path with the
geometry read wrongly, because the NVF assembly is Hermitian whatever the normal field says. It
would pass every broken version of this feature.

Run from the repo root::

    python benchmarks/benchmark_oblique_lattice.py            # all of them, + log + figure
    python benchmarks/benchmark_oblique_lattice.py published  # one

Output follows the convention `benchmark_backends.py` set: the transcript streams to
``results/oblique_lattice.log`` (checked in) as it lands, the numbers to
``results/oblique_lattice.json`` (gitignored), and a full run also writes
``images/oblique-lattice.png``.

``s4`` needs the S4 python module; point ``S4_PATH`` at it (default is the py312 build beside
this repo). Every other check is self-contained.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fieldwave                                                        # noqa: E402
from fieldwave._core.lattice import is_axis_aligned                     # noqa: E402

S4_PATH = os.environ.get('S4_PATH', '<a path outside this repository>')
SQ3 = np.sqrt(3.0)


def _hex(a):
    """Hexagonal basis of pitch ``a``, columns a1 / a2."""
    return np.column_stack((np.array([a, 0.0]), np.array([a / 2, a * SQ3 / 2])))


def _rod_raster(basis, n, radius):
    """A rod on the parametric grid: ``cell[j, i]`` at ``(i+.5)/n*a1 + (j+.5)/n*a2``.

    The minimum image over the lattice is not decoration — a feature reaching past one cell edge
    has to wrap along ``a1`` / ``a2``, and wrapping it on a rectangle instead is exactly the
    mistake ``docs/oblique-lattice.md`` opens by warning about.
    """
    a1, a2 = np.asarray(basis)[:, 0], np.asarray(basis)[:, 1]
    t = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(t, t)
    pts = U[..., None] * a1 + V[..., None] * a2
    centre = 0.5 * (a1 + a2)
    inside = np.zeros(pts.shape[:2], dtype=bool)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            s = i * a1 + j * a2
            inside |= (((pts[..., 0] - centre[0] - s[0]) ** 2
                        + (pts[..., 1] - centre[1] - s[1]) ** 2) <= radius ** 2)
    cell = np.full((1, n, n), 1.0)
    cell[0][inside] = 2.4
    return cell


# ------------------------------------------------------------------ published

LI_FMM = {(-1, -1): 0.04308, (-1, 0): 0.12860, (-1, 1): 0.06196,
          (0, -1): 0.12860, (0, 0): 0.17486, (0, 1): 0.12860,
          (1, -1): 0.06196, (1, 0): 0.12860, (1, 1): 0.04308}
LI_FEM = {(-1, -1): 0.04333, (-1, 0): 0.12845, (-1, 1): 0.06176,
          (0, -1): 0.12838, (0, 0): 0.17577, (0, 1): 0.12839,
          (1, -1): 0.06177, (1, 0): 0.12843, (1, 1): 0.04332}


def published(fto=16):
    """Li's checkerboard against the published table.

    Li, JOSA A **14**(10), 2758 (1997), tabulated by Demesy et al., *Gratings: Theory and
    Numeric Applications* (2012), Table 5.4 — reflected sum 0.10040, total 1.00000.

    The case is well chosen for this feature: a checkerboard needs two blocks on a square cell
    and one on its own primitive cell, which is a square rotated 45 degrees and therefore *not*
    axis-aligned. The paper's ``dx = dy = 5*sqrt(2)/4`` is that primitive cell's edge, which is
    how the geometry was identified in the first place.

    ``psi0 = 45 deg`` in the paper is measured from *its* cell axes — the rotated basis — so here
    it is ``psi = pi/2``. That is worth stating because the wrong frame is nearly invisible in
    the zeroth order (polarisation-independent on a C4v cell) while swapping which pairs of
    orders are degenerate; see :func:`published_polarisation_frame`.
    """
    side = 1.25
    a1, a2 = np.array([side, side]), np.array([side, -side])
    basis = np.column_stack((a1, a2))
    print(f'  cell edge |a1| = {np.linalg.norm(a1):.6f}   paper 5*sqrt(2)/4 = '
          f'{5 * np.sqrt(2) / 4:.6f}')
    print(f'  axis-aligned: {is_axis_aligned(basis)}  (False = the oblique path carries this)')

    ti, ri = _checkerboard(basis, side, fto, np.pi / 2)
    c = (ti.shape[0] // 2, ti.shape[1] // 2)
    print(f'\n  {"order":>8} {"Li FMM":>9} {"FEM":>9} {"FieldWave":>10} {"vs FEM":>9}')
    worst = 0.0
    for (n, m), fem in LI_FEM.items():
        v = float(ti[c[0] + n, c[1] + m])
        worst = max(worst, abs(v - fem))
        print(f'  {f"({n},{m})":>8} {LI_FMM[(n, m)]:>9.5f} {fem:>9.5f} {v:>10.5f} {v - fem:>9.5f}')
    print(f'\n  worst over all nine : {worst:.2e}')
    print(f'  reflected sum       : {float(ri.sum()):.5f}   (paper 0.10040)')
    print(f'  R + T               : {float(ri.sum() + ti.sum()):.6f}   (paper 1.00000)')
    print('  for scale, the paper\'s own FMM and FEM differ by 9.1e-4 on T(0,0)')
    return worst


def _checkerboard(basis, side, fto, psi):
    centre = 0.5 * (basis[:, 0] + basis[:, 1])
    ucell = [{'n_base': 1.0, 'shapes': [
        {'type': 'rectangle', 'cx': float(centre[0]), 'cy': float(centre[1]),
         'lx': side, 'ly': side, 'n_index': 1.5}]}]
    s = fieldwave.rcwa(backend=0, ucell=ucell, fto=[fto, fto], thickness=[1.0], wavelength=1.0,
                       n_top=1.5, n_bot=1.0, theta=0.0, phi=0.0, psi=psi,
                       factorization='nvf', lattice=basis)
    r = s.conv_solve().res
    return np.asarray(r.de_ti), np.asarray(r.de_ri)


def published_polarisation_frame(fto=16):
    """Why the whole order table is compared and not T(0,0).

    A wrong polarisation frame leaves the zeroth order essentially unmoved and rearranges which
    pairs of orders are degenerate. Checking T(0,0) alone would accept it.
    """
    side = 1.25
    basis = np.column_stack((np.array([side, side]), np.array([side, -side])))
    print(f'  {"psi":>6} {"T(0,0)":>9} {"worst vs FEM":>14}')
    for name, psi in (('0', 0.0), ('pi/4', np.pi / 4), ('pi/2', np.pi / 2)):
        ti, _ = _checkerboard(basis, side, fto, psi)
        c = (ti.shape[0] // 2, ti.shape[1] // 2)
        worst = max(abs(float(ti[c[0] + n, c[1] + m]) - v) for (n, m), v in LI_FEM.items())
        print(f'  {name:>6} {float(ti[c]):>9.5f} {worst:>14.2e}')
    print('  T(0,0) moves by ~1e-5 across all three; the table moves by 1.9e-2')


# ------------------------------------------------------------------ s4

def s4(pitch=400.0, radius=90.0, height=300.0, wl=900.0):
    """S4 as an outside oracle — **against a square-lattice control**.

    The control is the whole design. S4's default Laurent rule converges slowly on a
    high-contrast curved boundary, so a raw hex comparison shows a gap that says nothing about
    whether the oblique path is right. Running the same comparison on a square lattice, where
    none of the oblique code executes, measures that gap independently; what matters is the
    *difference* between the two.

    Do not reach for ``PolarizationDecomposition=True`` to speed S4's convergence here — it
    diverged on both lattices in our hands (0.986 -> 0.933 over the same sweep) and is not
    usable as an oracle.
    """
    sys.path.insert(0, S4_PATH)
    import S4                                                          # noqa: N813

    n_si = 3.638751670074983

    def run_s4(basis, centres, nb):
        a1, a2 = basis[:, 0], basis[:, 1]
        S = S4.New(Lattice=((a1[0], a1[1]), (a2[0], a2[1])), NumBasis=nb)
        S.SetMaterial(Name='V', Epsilon=1.0)
        S.SetMaterial(Name='Si', Epsilon=n_si ** 2)
        S.AddLayer(Name='top', Thickness=0, Material='V')
        S.AddLayer(Name='slab', Thickness=height, Material='V')
        for cx, cy in centres:
            S.SetRegionCircle(Layer='slab', Material='Si', Center=(cx, cy), Radius=radius)
        S.AddLayer(Name='bot', Thickness=0, Material='V')
        S.SetFrequency(1.0 / wl)
        S.SetExcitationPlanewave(IncidenceAngles=(0, 0), sAmplitude=1, pAmplitude=0)
        inc, _ = S.GetPowerFlux(Layer='top')
        tr, _ = S.GetPowerFlux(Layer='bot')
        return len(S.GetBasisSet()), float(tr.real / inc.real)

    def run_fw(basis, fto):
        cell = _rod_raster(basis, 512, radius)
        cell[0][cell[0] > 1.5] = n_si
        kw = {} if is_axis_aligned(basis) else {'lattice': basis}
        s = fieldwave.rcwa(backend=0, ucell=cell, period=[pitch, pitch], fto=[fto, fto],
                           thickness=[height], wavelength=wl, n_top=1.0, n_bot=1.0,
                           theta=0.0, phi=0.0, psi=0.0, factorization='nvf', **kw)
        return float(np.sum(np.asarray(s.conv_solve().de_ti)))

    square = np.diag([pitch, pitch])
    out = {}
    for label, basis, centres in (('square (control)', square, [(0, 0)]),
                                  ('hex (new code)', _hex(pitch), [(0, 0)])):
        print(f'\n  {label}:')
        for nb in (101, 213, 401, 697):
            n, t = run_s4(basis, centres, nb)
            print(f'    S4         orders {n:>4}  T={t:.5f}')
        fw = run_fw(basis, 16)
        print(f'    FieldWave  orders {33 ** 2:>4}  T={fw:.5f}')
        out[label] = (t, fw)
    gaps = {k: abs(a - b) for k, (a, b) in out.items()}
    print(f'\n  gap on the control : {gaps["square (control)"]:.2e}')
    print(f'  gap on hex         : {gaps["hex (new code)"]:.2e}')
    print(f'  the oblique path adds: '
          f'{abs(gaps["hex (new code)"] - gaps["square (control)"]):.2e}')
    return gaps


# ------------------------------------------------------------------ shear

def shear(fto=8, n=128):
    """A unimodular basis change: ``a2' = a2 + a1`` generates the *identical* point lattice.

    Same cell area, same reciprocal set — so this is not an approximation of the orthogonal run,
    it is the same problem in a different primitive cell, and the answers must agree. The
    cheapest correctness check available, and the one worth running on your own structure.

    The residual is the two rasters staircasing the rod differently, which is why analytic shapes
    (no staircase) land two orders tighter.
    """
    p = 400.0
    ortho = np.column_stack(([p, 0.0], [0.0, p]))
    sheared = np.column_stack(([p, 0.0], [p, p]))

    def totals(basis, use_lattice):
        cell = _rod_raster(basis, n, 90.0)
        kw = {'lattice': basis} if use_lattice else {}
        s = fieldwave.rcwa(backend=0, ucell=cell, period=[p, p], fto=[fto, fto],
                           thickness=[150.0], wavelength=900.0, n_top=1.0, n_bot=1.0,
                           theta=0.0, phi=0.0, psi=0.0, factorization='nvf', **kw)
        r = s.conv_solve()
        return float(np.sum(np.asarray(r.de_ti)))

    t_o, t_s = totals(ortho, False), totals(sheared, True)
    print(f'  raster   orthogonal {t_o:.8f}   sheared {t_s:.8f}   |diff| {abs(t_o - t_s):.2e}')

    def analytic(basis, use_lattice):
        a1, a2 = basis[:, 0], basis[:, 1]
        c = 0.5 * (a1 + a2)
        kw = {'lattice': basis} if use_lattice else {}
        s = fieldwave.rcwa(backend=0, ucell=[{'n_base': 1.0, 'shapes': [
            {'type': 'circle', 'cx': float(c[0]), 'cy': float(c[1]), 'lx': 180.0,
             'n_index': 2.4}]}], period=[p, p], fto=[12, 12], thickness=[150.0],
            wavelength=900.0, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
            factorization='nvf', **kw)
        return float(np.sum(np.asarray(s.conv_solve().de_ti)))

    a_o, a_s = analytic(ortho, False), analytic(sheared, True)
    print(f'  analytic orthogonal {a_o:.8f}   sheared {a_s:.8f}   |diff| {abs(a_o - a_s):.2e}')
    print('  removing the normal field\'s metric correction drifts the raster case to ~8.5e-4')
    return abs(t_o - t_s), abs(a_o - a_s)


# ------------------------------------------------------------------ supercell

def supercell(pitch=400.0, radius=80.0, n=96):
    """A native hex cell against the rectangular supercell it replaces.

    Same infinite array, two representations. The residual is the two rasters staircasing the
    rod differently, so it tightens with raster resolution rather than with ``fto``.
    """
    hexb = _hex(pitch)
    t_hex = _rod_raster(hexb, n, radius)
    s = fieldwave.rcwa(backend=0, ucell=t_hex, period=[pitch, pitch], fto=[8, 8],
                       thickness=[150.0], wavelength=900.0, n_top=1.0, n_bot=1.0,
                       theta=0.0, phi=0.0, psi=0.0, factorization='nvf', lattice=hexb)
    a_hex = float(np.sum(np.asarray(s.conv_solve().de_ti)))

    lx, ly = pitch, pitch * SQ3
    ny = round(n * SQ3)
    x = (np.arange(n) + 0.5) / n * lx
    y = (np.arange(ny) + 0.5) / ny * ly
    X, Y = np.meshgrid(x, y)
    inside = np.zeros(X.shape, dtype=bool)
    for cx, cy in ((0.25 * lx, 0.25 * ly), (0.75 * lx, 0.75 * ly)):
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                inside |= ((X - cx - i * lx) ** 2 + (Y - cy - j * ly) ** 2) <= radius ** 2
    sup = np.full((1, ny, n), 1.0)
    sup[0][inside] = 2.4
    s2 = fieldwave.rcwa(backend=0, ucell=sup, period=[lx, ly], fto=[8, 14], thickness=[150.0],
                        wavelength=900.0, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                        factorization='nvf')
    a_sup = float(np.sum(np.asarray(s2.conv_solve().de_ti)))
    print(f'  native hex  (1 motif,  289 orders) T = {a_hex:.6f}')
    print(f'  supercell   (2 motifs, 493 orders) T = {a_sup:.6f}')
    print(f'  |diff| = {abs(a_hex - a_sup):.2e}')
    return abs(a_hex - a_sup)


# ------------------------------------------------------------------ cost

def cost(pitch=400.0, radius=90.0):
    """What the feature buys: order counts and wall clock.

    Two regimes, and quoting one for the other is misleading. The order count scales with cell
    area, so a native cell keeps exactly half — but that only becomes ~8x once the matrices are
    large enough to be in the asymptotic regime. A metasurface library at ``fto=8`` is not.

    **Read the order ratio, not the third digit of the time ratio.** The order counts are exact
    and identical on every run; the timings are not. This has no warm-up and no repetition —
    `benchmark_backends.py` spends 45 s warming the machine and times one fixed configuration
    three times across the run, and that discipline is what its numbers rest on. Here the drift
    probe below is the whole of it: one configuration timed before and after everything else. If
    those two disagree, the run measured the machine and the ratios are worth no more than
    "somewhere around 3-5x".
    """
    hexb = _hex(pitch)
    cell = _rod_raster(hexb, 384, radius)
    lx, ly = pitch, pitch * SQ3
    ny = round(384 * SQ3)
    x = (np.arange(384) + 0.5) / 384 * lx
    y = (np.arange(ny) + 0.5) / ny * ly
    X, Y = np.meshgrid(x, y)
    inside = np.zeros(X.shape, dtype=bool)
    for cx, cy in ((0.25 * lx, 0.25 * ly), (0.75 * lx, 0.75 * ly)):
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                inside |= ((X - cx - i * lx) ** 2 + (Y - cy - j * ly) ** 2) <= radius ** 2
    sup = np.full((1, ny, 384), 1.0)
    sup[0][inside] = 2.4

    def _probe():
        s = fieldwave.rcwa(backend=0, ucell=cell, period=[pitch, pitch], fto=[10, 10],
                           thickness=[300.0], wavelength=900.0, n_top=1.0, n_bot=1.0,
                           theta=0.0, phi=0.0, psi=0.0, factorization='nvf', lattice=hexb)
        t0 = time.perf_counter(); s.conv_solve(); return time.perf_counter() - t0

    drift_a = _probe()
    print(f'  [drift] start: {drift_a * 1000:.0f} ms')
    print(f'  {"":>10} {"fto":>9} {"orders":>7} {"sec":>7}')
    rows = {}
    for label, f in (('native', 8), ('native', 12), ('native', 16)):
        s = fieldwave.rcwa(backend=0, ucell=cell, period=[pitch, pitch], fto=[f, f],
                           thickness=[300.0], wavelength=900.0, n_top=1.0, n_bot=1.0,
                           theta=0.0, phi=0.0, psi=0.0, factorization='nvf', lattice=hexb)
        t0 = time.perf_counter(); s.conv_solve(); dt = time.perf_counter() - t0
        rows[('native', f)] = ((2 * f + 1) ** 2, dt)
        print(f'  {label:>10} {str((f, f)):>9} {(2*f+1)**2:>7} {dt:>7.2f}')
    for f in (8, 12, 16):
        fy = round(f * SQ3)
        s = fieldwave.rcwa(backend=0, ucell=sup, period=[lx, ly], fto=[f, fy],
                           thickness=[300.0], wavelength=900.0, n_top=1.0, n_bot=1.0,
                           theta=0.0, phi=0.0, psi=0.0, factorization='nvf')
        t0 = time.perf_counter(); s.conv_solve(); dt = time.perf_counter() - t0
        rows[('super', f)] = ((2 * f + 1) * (2 * fy + 1), dt)
        print(f'  {"supercell":>10} {str((f, fy)):>9} {(2*f+1)*(2*fy+1):>7} {dt:>7.2f}')

    print()
    ratios = {}
    for f in (8, 12, 16):
        nn, tn = rows[('native', f)]
        ns, ts = rows[('super', f)]
        ratios[str(f)] = ts / tn
        print(f'  fto={f:>2}: orders {ns/nn:.2f}x, time {ts/tn:.1f}x')
    n = np.array([rows[('super', f)][0] for f in (8, 12, 16)], float)
    t = np.array([rows[('super', f)][1] for f in (8, 12, 16)], float)
    p = float(np.polyfit(np.log(n), np.log(t), 1)[0])
    drift_b = _probe()
    spread = abs(drift_b - drift_a) / max(drift_a, drift_b)
    print(f'\n  [drift] end: {drift_b * 1000:.0f} ms   (start {drift_a * 1000:.0f} ms, '
          f'spread {spread:.0%})')
    print(f'  cost exponent from these three points: N^{p:.2f}')
    if spread > 0.15:
        print('  ^ the drift probe moved more than 15%: this run timed the machine, not the')
        print('    code. The order ratios above are still exact; the time ratios are not.')
    return {'native': {str(f): rows[('native', f)] for f in (8, 12, 16)},
            'super': {str(f): rows[('super', f)] for f in (8, 12, 16)},
            'exponent': p, 'ratios': ratios, 'drift': [drift_a, drift_b, spread]}


CHECKS = {'published': published, 'polarisation': published_polarisation_frame,
          's4': s4, 'shear': shear, 'supercell': supercell, 'cost': cost}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(HERE, 'results', 'oblique_lattice.log')
JSON = os.path.join(HERE, 'results', 'oblique_lattice.json')
FIG = os.path.join(ROOT, 'images', 'oblique-lattice.png')


class _Tee:
    """Stream the transcript to the log as it lands, so an interrupted run still leaves it."""

    def __init__(self, path):
        self.term, self.fh = sys.stdout, open(path, 'w')

    def write(self, s):
        self.term.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.term.flush()
        self.fh.flush()


def figure(data, path=FIG):
    """Three panels: is it right, what does it cost, and does it converge faster.

    Drawn from the numbers the checks just returned, so the figure cannot drift from the log
    beside it.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    keys = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    x = np.arange(len(keys))
    fw = data.get('published_orders')
    if fw:
        ax[0].plot(x, [LI_FMM[k] for k in keys], 's', ms=9, mfc='none', label='Li 1997 (FMM)')
        ax[0].plot(x, [LI_FEM[k] for k in keys], 'o', ms=9, mfc='none', label='published FEM')
        ax[0].plot(x, [fw[str(k)] for k in keys], 'x', ms=9, label='FieldWave (oblique)')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels([f'{a},{b}' for a, b in keys], fontsize=8)
        ax[0].set_xlabel('diffraction order (n,m)')
        ax[0].set_ylabel('transmittance')
        ax[0].set_title(f"Li's checkerboard vs the published table\nworst "
                        f"{data['published_worst']:.1e} over all nine", fontsize=10)
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=.3)

    c = data.get('cost')
    if c:
        # Against `fto`, not against retained orders. Both cells obey the same N^p curve, so
        # time-vs-N puts them on top of each other and hides the point — which is that the same
        # `fto` costs the supercell more, because it needs more orders to express it.
        ftos = [8, 12, 16]
        for label, mark, name in (('native', 'o-', 'native oblique'),
                                  ('super', 's--', 'rectangular supercell')):
            ax[1].semilogy(ftos, [c[label][str(f)][1] for f in ftos], mark, label=name)
        for f in ftos:
            ax[1].annotate(f'{c["ratios"][str(f)]:.1f}x', (f, c['super'][str(f)][1]),
                           textcoords='offset points', xytext=(6, 4), fontsize=9)
        ax[1].set_xticks(ftos)
        ax[1].set_xlabel('fto (per axis)')
        ax[1].set_ylabel('wall clock (s)')
        drift = c.get('drift', [0, 0, 0])[2]
        ax[1].set_title(f"cost at the same fto\norder ratio 1.7x exact; timing drift "
                        f"{drift:.0%} this run", fontsize=10)
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=.3, which='both')

    conv = data.get('convergence')
    if conv:
        ax[2].plot(conv['fw_orders'], conv['fw_T'], 'x-', label='FieldWave, native cell')
        ax[2].plot(conv['s4_orders'], conv['s4_T'], 's--', mfc='none',
                   label='S4, square supercell')
        ax[2].axhline(LI_FEM[(0, 0)], color='k', lw=.8, ls=':', label='published FEM')
        ax[2].set_xlabel('retained orders')
        ax[2].set_ylabel('T(0,0)')
        ax[2].set_title('same benchmark, two cells\nhalf the orders and it settles first',
                        fontsize=10)
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f'\nfigure: {path}')


def convergence():
    """T(0,0) against retained orders, native cell vs the supercell S4 has to use.

    The one plot that shows *why* halving the orders matters: it is not only cheaper per solve,
    the native cell reaches its answer at a truncation the supercell has not reached.
    """
    side = 1.25
    basis = np.column_stack((np.array([side, side]), np.array([side, -side])))
    fw_orders, fw_t = [], []
    for f in (6, 10, 14, 18):
        ti, _ = _checkerboard(basis, side, f, np.pi / 2)
        c = (ti.shape[0] // 2, ti.shape[1] // 2)
        fw_orders.append((2 * f + 1) ** 2)
        fw_t.append(float(ti[c]))
        print(f'  FieldWave  orders {fw_orders[-1]:>5}  T(0,0)={fw_t[-1]:.6f}')

    s4_orders, s4_t = [], []
    try:
        sys.path.insert(0, S4_PATH)
        import S4                                                      # noqa: N813
        L = 2 * side
        for ng in (101, 201, 401, 701, 1001):
            S = S4.New(Lattice=((L, 0), (0, L)), NumBasis=ng)
            S.SetMaterial(Name='D', Epsilon=2.25)
            S.SetMaterial(Name='V', Epsilon=1.0)
            S.AddLayer(Name='b', Thickness=0, Material='D')
            S.AddLayer(Name='s', Thickness=1, Material='V')
            for cx, cy in ((-0.5 * side, -0.5 * side), (0.5 * side, 0.5 * side)):
                S.SetRegionRectangle(Layer='s', Material='D', Center=(cx, cy), Angle=0,
                                     Halfwidths=(0.5 * side, 0.5 * side))
            S.AddLayer(Name='a', Thickness=0, Material='V')
            S.SetFrequency(1)
            S.SetExcitationPlanewave(IncidenceAngles=(0, 0), sAmplitude=1, pAmplitude=0)
            inc = S.GetPowerFlux(Layer='b')[0].real
            s4_orders.append(len(S.GetBasisSet()))
            s4_t.append(float(S.GetPowerFluxByOrder(Layer='a')[0][0].real / inc))
            print(f'  S4         orders {s4_orders[-1]:>5}  T(0,0)={s4_t[-1]:.6f}')
    except ImportError as exc:
        print(f'  S4 skipped: {exc}')
    print(f'  published FEM T(0,0) = {LI_FEM[(0, 0)]}')
    return dict(fw_orders=fw_orders, fw_T=fw_t, s4_orders=s4_orders, s4_T=s4_t)


if __name__ == '__main__':
    import json

    want = sys.argv[1:] or list(CHECKS) + ['convergence']
    full = not sys.argv[1:]
    if full:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        sys.stdout = _Tee(LOG)

    data = {}
    for name in want:
        fn = CHECKS.get(name) or (convergence if name == 'convergence' else None)
        if fn is None:
            print(f'unknown check {name!r}; pick from {list(CHECKS) + ["convergence"]}')
            raise SystemExit(2)
        print(f'\n=== {name} ===')
        try:
            data[name] = fn()
        except ImportError as exc:
            print(f'  skipped: {exc}')

    if full:
        side = 1.25
        basis = np.column_stack((np.array([side, side]), np.array([side, -side])))
        ti, _ = _checkerboard(basis, side, 16, np.pi / 2)
        c = (ti.shape[0] // 2, ti.shape[1] // 2)
        payload = {
            'published_orders': {str(k): float(ti[c[0] + k[0], c[1] + k[1]]) for k in LI_FEM},
            'published_worst': data.get('published'),
            'convergence': data.get('convergence'),
            'cost': data.get('cost'),
        }
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(JSON, 'w') as fh:
            json.dump(payload, fh, indent=1, sort_keys=True, default=float)
        figure(payload)
        print(f'log: {LOG}\njson: {JSON}')
