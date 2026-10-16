"""ETM's labels at a complex k_par from the state below a layer; a layer of a half-space's medium.

Two changes to the stack algebra at a complex in-plane wavevector, measured together because they
meet on the same scenes (``docs/theory/rcwa/connecting-algorithms.md`` section 10):

* **maximum-volume labels** (``rcwa/_engine/maxvol_labels.py``): which root of each ``+-q`` pair
  ETM calls ``+`` is chosen from the ETM state below the layer -- a locally maximum-volume pivot
  set of its transmitted subspace (Mehrmann and Poloni, SIAM J. Matrix Anal. Appl. 33, 780,
  2012) -- instead of from the layer alone (``etm_mode_q``'s capped half-space root, "the rule");
* **the same-half-space merge** (``rcwa/_engine/half_space_merge.py``): a layer whose material is
  the half-space's beside it is merged into it, the rest solved, and the answer carried to the
  faces analytically -- the theory value where the layer is a knife edge.

  1 table     section 10's first table and the cap rows: the rule's ETM and the new labels'
              against SMM, and how far a 1e-14 nudge of ``theta`` moves each.
  2 etched    4,000 random etched gratings whose teeth are the material of a thick slab beside
              them (the scenes the rule left O(1) off): over 1e-6 above SMM's nudge, rule against
              new labels; then three scenes built to the description section 10 gave.
  3 random    13,000 general and 13,000 thick-layer complex-angle scenes: over 1e-10 above SMM's
              nudge, and any scene where the new labels are more than 3x worse than the rule.
  4 knives    a thick layer of a half-space's own material. The uniform stacks among 1,536 seeds
              against an exact (mpmath) thin-film oracle, and 600 TE gratings on such a layer
              against an exact linear system on the solver's own convolution matrices: which are
              knife edges (4 ulps of the layer's permittivity move the exact answer by more than
              1e-6, its value finite in double precision), and on those, every algorithm unmerged
              and merged against the theory -- the stack with the layer deleted, its faces moved
              back (60 digits).
  5 fields    on the same knife edges, the merged field at five heights per layer against the
              deleted stack's field carried into the full stack by plane waves written here.
  6 develop   with ``--develop REV``: real-angle amplitude and field digests of three families,
              half-space-material layers included, on this tree and on a ``git archive`` of REV.
  7 cost      with ``--time``: eager NumPy time per solve, rule against new labels (left out of
              the log, which is byte-for-byte reproducible).

Units: lengths in metres, wavelength 1e-6; wavevectors in units of k0. Time convention
exp(-i w t). Every solve is NumPy complex128, ``auto_detune=False``, ``factorization='li'``,
``fourier_coefficients='integrated'``. Random draws are seeded. Run from the repository root:

    python studies/rcwa/etm_maxvol_labels/etm_maxvol_labels.py --develop c9ff4630

It prints the log committed as ``studies/rcwa/etm_maxvol_labels/records/etm_maxvol_labels.log``
in about ten minutes with ``--workers 4`` (sections 3 and 4 dominate); ``--only 1 2`` selects
sections. ``--develop`` extracts REV with ``git archive`` into a temporary directory and runs
section 6's scenes there in a subprocess, by this same script.
"""

import argparse
import contextlib
import hashlib
import importlib
import os
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Select the checkout before importing any wavebit module (as the other studies here do).
ROOT = str(Path(__file__).resolve().parents[3])
_root_parser = argparse.ArgumentParser(add_help=False)
_root_parser.add_argument('--root', default=ROOT)
_SELECTED_ROOT = Path(_root_parser.parse_known_args()[0].root).expanduser().resolve()
_PACKAGE_PARENT = next((path for path in (_SELECTED_ROOT / 'src', _SELECTED_ROOT)
                        if (path / 'wavebit' / '__init__.py').is_file()), None)
if _PACKAGE_PARENT is None:
    raise RuntimeError(f'No wavebit source package in selected root: {_SELECTED_ROOT}')
sys.path.insert(0, str(_PACKAGE_PARENT))
_selected_wavebit = importlib.import_module('wavebit')
if Path(_selected_wavebit.__file__).resolve() != (_PACKAGE_PARENT / 'wavebit' / '__init__.py').resolve():
    raise RuntimeError('wavebit was already imported from another tree; run in a fresh process.')

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa  # noqa: E402

try:
    from wavebit.rcwa._engine import maxvol_labels as _maxvol
    from wavebit.rcwa._engine import engine as _engine
    _HAS_MERGE = hasattr(_engine.CoreFactorizationMixin, '_half_space_merge_plan')
except ImportError:                  # an archive from before either change (section 6 only)
    _maxvol, _HAS_MERGE = None, False

NUDGE = 1 + 1e-14
ULP = 2.0 ** -52
WORKERS = 4


def header(number, title):
    print(f'\n== {number} {title}')


# ------------------------------------------------------------------------------- switches

@contextlib.contextmanager
def labels(mode):
    """``'complex'``: the new labels at a complex ``k_par``; ``'off'``: the rule everywhere."""
    if _maxvol is None:
        yield
        return
    saved = _maxvol.MODE
    _maxvol.MODE = mode
    try:
        yield
    finally:
        _maxvol.MODE = saved


@contextlib.contextmanager
def merged(on):
    """With ``on`` false, every layer is solved as a layer (the engine has no switch for it)."""
    if on or not _HAS_MERGE:
        yield
        return
    cls = _engine.CoreFactorizationMixin
    saved = cls.__dict__['_half_space_merge_plan']
    cls._half_space_merge_plan = lambda self: None
    try:
        yield
    finally:
        cls._half_space_merge_plan = saved


# --------------------------------------------------------------------------------- scenes

def _pattern(rng, n, X, groove):
    """A binary 1D raster of ``n`` cells, teeth ``X`` and grooves ``groove``, fill in (0.2, 0.8)."""
    fill = rng.uniform(0.2, 0.8)
    k = max(1, min(n - 1, int(round(fill * n))))
    start = rng.integers(0, n)
    row = np.full(n, groove, dtype=complex)
    row[(start + np.arange(k)) % n] = X
    return row


def etched_scene(seed):
    """An etched grating whose teeth are the material of a thick slab beside it."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(2, 120)
    groove = 1.0 if rng.uniform() < 0.7 else rng.uniform(1, 4)
    n = 20
    g = _pattern(rng, n, X, groove)
    t_g = float(np.exp(rng.uniform(np.log(50e-9), np.log(1e-6))))
    t_s = float(np.exp(rng.uniform(np.log(0.3e-6), np.log(200e-6))))
    slab = np.full(n, X, dtype=complex)
    bot = float(rng.choice([1.0, 2.25, rng.uniform(1, 12)]))
    layers = [(slab, t_s), (g, t_g)] if rng.uniform() < 0.7 else [(g, t_g), (slab, t_s)]
    kind = rng.choice(['te', 'tm', 'conical'])
    theta = complex(rng.uniform(0.0, 1.3), rng.uniform(0.02, 0.6))
    return dict(kind=str(kind), top=1.0, bot=bot, layers=layers, theta=theta,
                phi=float(rng.uniform(0, np.pi / 2)), psi=float(rng.uniform(0, np.pi)),
                period=float(rng.uniform(0.4e-6, 3e-6)), to=int(rng.integers(5, 26)),
                wavelength=1000e-9, side='top' if rng.uniform() < 0.8 else 'bottom', X=X)


def general_scene(seed):
    """A general complex-angle stack: 1-4 layers, uniform or grating, random media."""
    rng = np.random.default_rng(seed)
    n, nl = 16, int(rng.integers(1, 5))

    def medium():
        u = rng.uniform()
        if u < 0.6:
            return complex(rng.uniform(1, 16), 0)
        if u < 0.85:
            return complex(rng.uniform(1, 16), rng.uniform(0, 2))
        return complex(rng.uniform(-30, -1), rng.uniform(0.1, 3))
    layers = []
    for _ in range(nl):
        row = (np.full(n, medium(), dtype=complex) if rng.uniform() < 0.5
               else _pattern(rng, n, medium(), medium()))
        layers.append((row, float(np.exp(rng.uniform(np.log(10e-9), np.log(50e-6))))))
    kind = rng.choice(['te', 'tm', 'conical'])
    theta = complex(rng.uniform(0.0, 1.4), rng.uniform(0.01, 0.6))
    top = float(rng.choice([1.0, 1.0, rng.uniform(1, 4)]))
    bot = float(rng.choice([1.0, 2.25, rng.uniform(1, 12)]))
    return dict(kind=str(kind), top=top, bot=bot, layers=layers, theta=theta,
                phi=float(rng.uniform(0, np.pi / 2)), psi=float(rng.uniform(0, np.pi)),
                period=float(rng.uniform(0.3e-6, 3e-6)), to=int(rng.integers(3, 16)),
                wavelength=1000e-9, side='top' if rng.uniform() < 0.8 else 'bottom')


def thick_scene(seed):
    """Complex angle through 1-3 layers, at least one 2-200 um thick: growth near the cap."""
    rng = np.random.default_rng(seed)
    n, nl = 16, int(rng.integers(1, 4))

    def medium():
        u = rng.uniform()
        if u < 0.65:
            return complex(rng.uniform(1, 16), 0)
        if u < 0.9:
            return complex(rng.uniform(1, 16), rng.uniform(0, 0.5))
        return complex(rng.uniform(-30, -1), rng.uniform(0.1, 3))
    thick_at = int(rng.integers(0, nl))
    layers = []
    for i in range(nl):
        row = (np.full(n, medium(), dtype=complex) if rng.uniform() < 0.5
               else _pattern(rng, n, medium(), medium()))
        low, high = (2e-6, 200e-6) if i == thick_at else (20e-9, 5e-6)
        layers.append((row, float(np.exp(rng.uniform(np.log(low), np.log(high))))))
    kind = rng.choice(['te', 'tm', 'conical'])
    theta = complex(rng.uniform(0.0, 1.4), float(np.exp(rng.uniform(np.log(0.005), np.log(0.6)))))
    top = float(rng.choice([1.0, 1.0, rng.uniform(1, 4)]))
    bot = float(rng.choice([1.0, 2.25, rng.uniform(1, 12)]))
    return dict(kind=str(kind), top=top, bot=bot, layers=layers, theta=theta,
                phi=float(rng.uniform(0, np.pi / 2)), psi=float(rng.uniform(0, np.pi)),
                period=float(rng.uniform(0.3e-6, 3e-6)), to=int(rng.integers(3, 16)),
                wavelength=1000e-9, side='top' if rng.uniform() < 0.8 else 'bottom')


def class_a_scene(seed):
    """A thick layer of a half-space's own material (the far one, or the one lit through)."""
    rng = np.random.default_rng(seed)
    n = 16
    eps_h = float(rng.choice([2.25, 4.0, 9.0, rng.uniform(1.5, 16)]))
    far = rng.uniform() < 0.6
    t_thick = float(np.exp(rng.uniform(np.log(1e-6), np.log(100e-6))))
    thick = np.full(n, eps_h, dtype=complex)
    layers = [(thick, t_thick)]
    u = rng.uniform()
    if u < 0.5:
        X = rng.uniform(1.5, 16)
        g = _pattern(rng, n, X, 1.0 if rng.uniform() < 0.5 else rng.uniform(1, 4))
        layers = [(thick, t_thick), (g, float(np.exp(rng.uniform(np.log(30e-9), np.log(1e-6)))))]
    elif u < 0.7:
        layers = [(thick, t_thick), (np.full(n, rng.uniform(1, 9), dtype=complex),
                                     float(rng.uniform(50e-9, 2e-6)))]
    other = float(rng.choice([1.0, rng.uniform(1, 4)]))
    kind = rng.choice(['te', 'tm', 'conical'])
    theta = complex(rng.uniform(0.0, 1.2), rng.uniform(0.01, 0.4))
    return dict(kind=str(kind), top=other, bot=eps_h, layers=layers, theta=theta,
                phi=float(rng.uniform(0, np.pi / 2)), psi=float(rng.uniform(0, np.pi)),
                period=float(rng.uniform(0.5e-6, 3e-6)), to=int(rng.integers(3, 12)),
                wavelength=1000e-9, side='top' if far else 'bottom', eps_h=eps_h)


def class_a_grating_te(seed):
    """Class A with a grating beside the thick layer, TE, few orders (for the exact system)."""
    sc = class_a_scene(seed)
    rng = np.random.default_rng(10_000_000 + seed)
    n = 16
    X = rng.uniform(1.5, 16)
    g = _pattern(rng, n, X, 1.0 if rng.uniform() < 0.5 else rng.uniform(1, 4))
    tg = float(np.exp(rng.uniform(np.log(30e-9), np.log(1e-6))))
    sc['layers'] = [(np.full(n, sc['eps_h'], dtype=complex), sc['layers'][0][1]), (g, tg)]
    sc['kind'] = 'te'
    sc['to'] = int(rng.integers(2, 7))
    return sc


def uniform(sc):
    return all(np.allclose(np.asarray(row), np.asarray(row)[0]) for row, _ in sc['layers'])


# -------------------------------------------------------------------------------- solving

def solver(sc, algorithm, theta=None, layers=None, answers='amplitude'):
    layers = sc['layers'] if layers is None else layers
    kw = dict(tetm=sc['kind']) if sc['kind'] in ('te', 'tm') else dict(psi=sc['psi'], phi=sc['phi'])
    return _legacy_wavebit_rcwa(
        backend='numpy', top=sc['top'], bot=sc['bot'], theta=sc['theta'] if theta is None else theta,
        truncation_order=[sc['to']], wavelength=sc['wavelength'], period=[sc['period']],
        structure=np.stack([np.asarray(row, dtype=complex).reshape(1, -1) for row, _ in layers]),
        thickness=[t for _, t in layers], complex_dtype=0, fourier_coefficients='integrated',
        factorization='li', layer_algorithm=algorithm, incidence_side=sc['side'],
        answers=answers, **kw)


def amplitudes(sc, algorithm, mode='complex', merge=True, theta=None):
    """Every Fresnel amplitude (both pure incidences on the conical path), flat; or the error."""
    try:
        with labels(mode), merged(merge), warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            res = solver(sc, algorithm, theta).solve(auto_detune=False)
        subs = (res.incident,) if sc['kind'] in ('te', 'tm') else (res.s_inc, res.p_inc)
        v = np.concatenate([np.ravel(np.asarray(getattr(sub, f))) for sub in subs
                            for f in ('r_s', 'r_p', 't_s', 't_p')])
        return v if np.all(np.isfinite(v)) else 'nonfinite'
    except Exception as exc:                    # noqa: BLE001 -- the error is the measurement
        return type(exc).__name__


def rel(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return np.nan
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def evaluate(sc):
    """The rule's and the new labels' ETM against SMM, and SMM's own nudge."""
    smm = amplitudes(sc, 'SMM', 'off')
    nudge = rel(amplitudes(sc, 'SMM', 'off', theta=sc['theta'] * NUDGE), smm)
    rule, new = amplitudes(sc, 'ETM', 'off'), amplitudes(sc, 'ETM', 'complex')
    return dict(nudge=nudge, rule=rel(rule, smm), new=rel(new, smm),
                rule_raised=isinstance(rule, str) and not isinstance(smm, str),
                new_raised=isinstance(new, str) and not isinstance(smm, str))


def pool(fn, items):
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, items, chunksize=8))


# ----------------------------------------------------------------------------------- 1. table

def _table_r0(kw, algorithm, mode):
    with labels(mode), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = _legacy_wavebit_rcwa(backend='numpy', complex_dtype=0, fourier_coefficients='integrated',
                                 factorization='li', layer_algorithm=algorithm, **kw)
        try:
            res = s.solve(auto_detune=False)
        except Exception as exc:                # noqa: BLE001
            return type(exc).__name__
    a = np.asarray(res.incident.r_s if kw.get('tetm') == 'te' else res.incident.r_p
                   if 'tetm' in kw else res.s_inc.r_s).ravel()
    return complex(a[len(a) // 2])


def section_table():
    header(1, "table -- section 10's scenes: |ETM - SMM| in r0, and each one's move under a "
              "1e-14 nudge of theta")
    S = np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1
    base = dict(top=1.0, bot=25.0, truncation_order=[30], wavelength=900e-9, period=[2000e-9],
                structure=S, thickness=[100e-9])
    con = dict(base, psi=np.pi / 2, phi=0.3)
    cell = np.full((6, 6), 25.0)
    cell[0, :] = 1.0
    cell[:, 0] = 1.0
    alpha = np.arccosh(np.sqrt(1.6 ** 2 + 1.2 ** 2 + 1e-4))
    phi = np.arctan(1.2 / 1.6001)
    rows = [('1D TE 0.5+0.3j', dict(base, tetm='te', theta=0.5 + 0.3j)),
            ('1D TM 0.5+0.3j', dict(base, tetm='tm', theta=0.5 + 0.3j)),
            ('conical 0.5+0.3j', dict(con, theta=0.5 + 0.3j)),
            ('conical 0.5+0.03j', dict(con, theta=0.5 + 0.03j)),
            ('conical 1.2+0.5j', dict(con, theta=1.2 + 0.5j)),
            ('conical 0.5+0.3j, 50 um', dict(con, theta=0.5 + 0.3j, thickness=[50e-6])),
            ('cap: conical 1.2+0.5j, 200 um', dict(con, theta=1.2 + 0.5j, thickness=[200e-6])),
            ('2D 6x6, (6, 6)', dict(base, psi=np.pi / 2, phi=0.3, truncation_order=[6, 6],
                                    period=[2000e-9, 2000e-9], structure=cell[None],
                                    theta=0.5 + 0.3j))]
    rows += [(f'evanescent pi/2-{off:g}', dict(con, phi=phi, theta=(np.pi / 2 - off) + 1j * alpha))
             for off in (7.5e-8, 1e-4, 1e-2)]
    rows += [('1 mm, conical 0.5+0.3j', dict(con, theta=0.5 + 0.3j, thickness=[1e-3]))]
    print(f'  {"scene":32s} {"rule":>22s} {"new labels":>22s} {"SMM nudge":>10s}')
    for name, kw in rows:
        smm = _table_r0(kw, 'SMM', 'off')
        smm_n = _table_r0(dict(kw, theta=kw['theta'] * NUDGE), 'SMM', 'off')
        cells = []
        for mode in ('off', 'complex'):
            e = _table_r0(kw, 'ETM', mode)
            en = _table_r0(dict(kw, theta=kw['theta'] * NUDGE), 'ETM', mode)
            cells.append(e if isinstance(e, str) else f'{abs(e - smm):.1e} (nudge {abs(en - e):.0e})')
        print(f'  {name:32s} {cells[0]:>22s} {cells[1]:>22s} {abs(smm_n - smm):10.0e}')


# ---------------------------------------------------------------------------------- 2. etched

def _etched_job(seed):
    return dict(evaluate(etched_scene(seed)), seed=seed)


def _forced_etched(seed, kind, X, to):
    sc = etched_scene(seed)
    sc['kind'] = kind
    sc['layers'] = [(np.where(np.abs(np.asarray(row) - sc['X']) < 1e-9, X, np.asarray(row)), t)
                    for row, t in sc['layers']]
    if to:
        sc['to'] = to
    return sc


def summarize(rows, threshold, label):
    for key in ('rule', 'new'):
        e = np.array([r[key] if r[key] == r[key] else np.inf for r in rows])
        n = np.array([r['nudge'] for r in rows])
        raised = sum(r[key + '_raised'] for r in rows)
        finite = e[np.isfinite(e)]
        print(f'  {label} {"the rule  " if key == "rule" else "new labels"}: '
              f'{int(np.sum(e > threshold + n)):5d} over {threshold:g} above SMM\'s nudge, '
              f'{raised} raised; worst finite {np.max(finite):.1e}')


def section_etched(count):
    header(2, f'etched -- {count} etched gratings on their own slab, ETM against SMM')
    rows = pool(_etched_job, range(count))
    summarize(rows, 1e-6, 'etched')
    print('  three scenes built to section 10\'s description (seeds 100000 on, each the first '
          'whose rule is 0.2 off, or the worst of 400):')
    for kind, X, to in (('tm', 108.0, 24), ('conical', 58.0, None), ('conical', 78.0, None)):
        best = None
        for seed in range(100000, 100400):
            sc = _forced_etched(seed, kind, X, to)
            r = evaluate(sc)
            if r['rule'] == r['rule'] and (best is None or r['rule'] > best[0]):
                best = (r['rule'], seed, r, sc)
            if r['rule'] > 0.2:
                break
        _, seed, r, sc = best
        print(f'    {kind:7s} X = {X:g}, orders {sc["to"]}, seed {seed}: thicknesses '
              f'{[f"{t:.3g}" for _, t in sc["layers"]]}, theta {sc["theta"]:.3f}: rule '
              f'{r["rule"]:.2e}, new labels {r["new"]:.2e}, SMM nudge {r["nudge"]:.1e}')


# ---------------------------------------------------------------------------------- 3. random

def _general_job(seed):
    return evaluate(general_scene(seed))


def _thick_job(seed):
    return evaluate(thick_scene(seed))


def section_random(count):
    header(3, f'random -- {count} general and {count} thick-layer complex-angle scenes')
    for label, job in (('general', _general_job), ('thick  ', _thick_job)):
        rows = pool(job, range(count))
        summarize(rows, 1e-10, label)
        a = np.array([r['rule'] for r in rows])
        b = np.array([r['new'] for r in rows])
        n = np.array([r['nudge'] for r in rows])
        worse = (b > 3 * a) & (b > 1e-12) & (b > 3 * n)
        print(f'  {label} new labels over 3x worse than the rule (and over 1e-12 and 3x SMM\'s '
              f'nudge): {int(np.sum(worse))}')


# ---------------------------------------------------------------------------------- 4. knives

def _thin_film(eps_list, d_list, k_par2, k0, pol):
    """``(r, t)`` of a uniform stack, characteristic matrices in mpmath; field E for s, H for p.

    Incidence from ``eps_list[0]`` through ``eps_list[1:-1]`` into ``eps_list[-1]``; half-spaces
    on Maystre's cut. A layer's root is either one: the answer does not depend on it exactly.
    """
    def maystre(kz2):
        kz = mp.sqrt(kz2)
        return -kz if mp.re(kz) + mp.im(kz) < 0 else kz
    eps = [mp.mpc(e) for e in eps_list]
    kz = [maystre(e - k_par2) for e in eps]

    def Y(i):
        return kz[i] if pol == 's' else kz[i] / eps[i]
    M = mp.eye(2)
    for i, d in enumerate(d_list, start=1):
        phase = kz[i] * k0 * mp.mpf(d)
        c, s = mp.cos(phase), mp.sin(phase)
        M = M * mp.matrix([[c, -1j * s / Y(i)], [-1j * Y(i) * s, c]])
    B = M[0, 0] + M[0, 1] * Y(len(eps) - 1)
    C = M[1, 0] + M[1, 1] * Y(len(eps) - 1)
    return (Y(0) * B - C) / (Y(0) * B + C), 2 * Y(0) / (Y(0) * B + C)


def _uniform_exact(sc, n, kx, ky, eps_thick):
    """The exact amplitude vector of a uniform scene in ``amplitudes``' layout."""
    c = sc['to']
    eps_layers = [complex(np.asarray(row)[0]) for row, _ in sc['layers']]
    eps_layers[0] = eps_thick
    d = [t for _, t in sc['layers']]
    if sc['side'] == 'top':
        seq, ds = [sc['top']] + eps_layers[::-1] + [sc['bot']], d[::-1]
    else:
        seq, ds = [sc['bot']] + eps_layers + [sc['top']], d
    k0 = 2 * mp.pi / mp.mpf(sc['wavelength'])
    kp2 = mp.mpc(kx[c]) ** 2 + mp.mpc(ky) ** 2
    growth = sum(abs(complex(mp.sqrt(mp.mpc(e) - kp2)).imag) * float(k0) * t
                 for e, t in zip(seq[1:-1], ds))
    mp.mp.dps = int(2 * growth / np.log(10)) + 60
    rs, ts = _thin_film(seq, ds, kp2, k0, 's')
    rp, tp = _thin_film(seq, ds, kp2, k0, 'p')
    tp = tp * mp.sqrt(mp.mpc(seq[0]) / mp.mpc(seq[-1]))
    m = n // (4 if sc['kind'] in ('te', 'tm') else 8)
    v = np.zeros(n, complex)
    slots = {'te': ((0, rs), (2, ts)), 'tm': ((1, rp), (3, tp)),
             'conical': ((0, rs), (2, ts), (5, rp), (7, tp))}[sc['kind']]
    for block, value in slots:
        v[block * m + c] = complex(value)
    return v


def _te_grating_exact(kx, eps_in, eps_out, layers, k0, dps):
    """Exact TE ``(R, T)`` per order: one linear system over every layer's two waves, each
    referenced at the face where its exponential is largest, so only the physics' own
    conditioning needs digits. ``layers`` from the lit side, ``[(E, d)]`` with ``E`` the
    convolution matrix as nested lists."""
    mp.mp.dps = dps
    kx = [mp.mpc(v) for v in kx]
    n, c = len(kx), len(kx) // 2

    def maystre(kz2):
        kz = mp.sqrt(kz2)
        return -kz if mp.re(kz) + mp.im(kz) < 0 else kz
    kzt = [maystre(mp.mpc(eps_in) - k ** 2) for k in kx]
    kzb = [maystre(mp.mpc(eps_out) - k ** 2) for k in kx]
    modes = []
    for E, d in layers:
        A = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                A[i, j] = -mp.mpc(E[i][j])
            A[i, i] += kx[i] ** 2
        w, W = mp.eig(A)
        q = []
        for v in w:
            r = mp.sqrt(v)
            q.append(-r if mp.re(r) < 0 or (mp.re(r) == 0 and mp.im(r) > 0) else r)
        V = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                V[i, j] = W[i, j] * q[j]
        modes.append((W, V, [mp.exp(-k0 * qq * mp.mpf(d)) for qq in q]))
    L = len(layers)
    M = mp.matrix(2 * n * (L + 1), n * (2 * L + 2))
    rhs = mp.matrix(2 * n * (L + 1), 1)

    def col(j, sign):
        return n + 2 * n * j + n * sign
    iT = n + 2 * n * L
    for i in range(n):
        M[i, i] = 1
        M[n + i, i] = 1j * kzt[i]
        if i == c:
            rhs[i], rhs[n + i] = -1, 1j * kzt[i]
    if L == 0:
        for i in range(n):
            M[i, iT + i], M[n + i, iT + i] = -1, 1j * kzb[i]
    else:
        W, V, X = modes[0]
        for i in range(n):
            for j in range(n):
                M[i, col(0, 0) + j], M[i, col(0, 1) + j] = -W[i, j], -W[i, j] * X[j]
                M[n + i, col(0, 0) + j], M[n + i, col(0, 1) + j] = -V[i, j], V[i, j] * X[j]
    row = 2 * n
    for k in range(L):
        W, V, X = modes[k]
        for i in range(n):
            for j in range(n):
                M[row + i, col(k, 0) + j], M[row + i, col(k, 1) + j] = W[i, j] * X[j], W[i, j]
                M[row + n + i, col(k, 0) + j] = V[i, j] * X[j]
                M[row + n + i, col(k, 1) + j] = -V[i, j]
        if k + 1 < L:
            W2, V2, X2 = modes[k + 1]
            for i in range(n):
                for j in range(n):
                    M[row + i, col(k + 1, 0) + j] = -W2[i, j]
                    M[row + i, col(k + 1, 1) + j] = -W2[i, j] * X2[j]
                    M[row + n + i, col(k + 1, 0) + j] = -V2[i, j]
                    M[row + n + i, col(k + 1, 1) + j] = V2[i, j] * X2[j]
        else:
            for i in range(n):
                M[row + i, iT + i], M[row + n + i, iT + i] = -1, 1j * kzb[i]
        row += 2 * n
    sol = mp.lu_solve(M, rhs)
    return [sol[i] for i in range(n)], [sol[iT + i] for i in range(n)]


def _grating_exact(sc, layers, scale_first=None, dps=60):
    """``_te_grating_exact`` on the solver's own convolution matrices for ``layers``, in
    ``amplitudes``' layout."""
    from wavebit._common.conventions import order_frame_sign
    s = solver(sc, 'SMM', layers=layers)
    epy = np.asarray(s.eps_conv_matrices()[1])
    kx = np.asarray(s.order_kxy()[0]).ravel()
    Es = [epy[i] for i in range(len(layers))]
    if scale_first is not None:
        Es[0] = Es[0] * scale_first
    stack = list(zip(Es, [t for _, t in layers]))
    if sc['side'] == 'top':
        stack, e_in, e_out = stack[::-1], sc['top'], sc['bot']
    else:
        e_in, e_out = sc['bot'], sc['top']
    k0 = 2 * mp.pi / mp.mpf(sc['wavelength'])
    stack = [([[complex(E[i, j]) for j in range(E.shape[1])] for i in range(E.shape[0])], d)
             for E, d in stack]
    R, T = _te_grating_exact([complex(v) for v in kx], e_in, e_out, stack, k0, dps)
    sigma = np.asarray(order_frame_sign(np, kx, None)).ravel()
    zero = np.zeros(len(kx), complex)
    return np.concatenate([np.array([complex(v) for v in R]) * sigma, zero,
                           np.array([complex(v) for v in T]) * sigma, zero]), kx


def _deleted_theory(sc, kx):
    """The TE grating with the thick layer deleted, exactly, its faces moved back (60 digits)."""
    rest = sc['layers'][1:]
    exact, _ = _grating_exact(sc, rest, dps=60)
    n = len(kx)
    R, T = exact[:n], exact[2 * n:3 * n]
    mp.mp.dps = 60
    k0 = 2 * mp.pi / mp.mpf(sc['wavelength'])
    d = mp.mpf(sc['layers'][0][1])

    def maystre(kz2):
        kz = mp.sqrt(kz2)
        return -kz if mp.re(kz) + mp.im(kz) < 0 else kz
    kz = [maystre(mp.mpc(sc['eps_h']) - mp.mpc(k) ** 2) for k in kx]
    c = n // 2
    if sc['side'] == 'top':                        # the far medium is merged
        T = np.array([complex(mp.mpc(t) * mp.exp(1j * k0 * z * d)) for t, z in zip(T, kz)])
    else:                                          # the incident one
        R = np.array([complex(mp.mpc(r) * mp.exp(1j * k0 * (kz[c] + z) * d))
                      for r, z in zip(R, kz)])
        T = np.array([complex(mp.mpc(t) * mp.exp(1j * k0 * kz[c] * d)) for t in T])
    zero = np.zeros(n, complex)
    return np.concatenate([R, zero, T, zero])


def _algorithms(sc, reference):
    out = {}
    for label, algorithm, mode in (('SMM', 'SMM', 'off'), ('rule', 'ETM', 'off'),
                                   ('new', 'ETM', 'complex')):
        for merge in (False, True):
            a = amplitudes(sc, algorithm, mode, merge)
            out[label + ('+merge' if merge else '')] = (
                a if isinstance(a, str) else rel(a, reference))
    return out


def _uniform_job(seed):
    sc = class_a_scene(seed)
    if not uniform(sc):
        return None
    probe = amplitudes(sc, 'SMM', 'off', merge=False)
    if isinstance(probe, str):
        return dict(seed=seed, skipped=probe)
    s = solver(sc, 'SMM')
    kx = np.asarray(s.order_kxy()[0]).ravel()
    ky = complex(np.asarray(s.order_kxy()[1]).ravel()[0])
    ex = _uniform_exact(sc, len(probe), kx, ky, sc['eps_h'])
    up = _uniform_exact(sc, len(probe), kx, ky, float(np.float64(sc['eps_h']) * (1 + 4 * ULP)))
    down = _uniform_exact(sc, len(probe), kx, ky, float(np.float64(sc['eps_h']) * (1 - 4 * ULP)))
    row = dict(seed=seed, knife=max(rel(up, ex), rel(down, ex)))
    if row['knife'] > 1e-6 and np.all(np.isfinite(ex)):
        row.update(_algorithms(sc, ex))
    return row


def _grating_job(seed):
    sc = class_a_grating_te(seed)
    s = solver(sc, 'SMM')
    kx = np.asarray(s.order_kxy()[0]).ravel()
    k0 = 2 * np.pi / sc['wavelength']
    q = np.sqrt(kx.astype(complex) ** 2 - sc['eps_h'])
    propagating = np.abs(q.real) < np.abs(q.imag)
    growth = float(np.max(k0 * sc['layers'][0][1] * np.abs(q.real[propagating]))) \
        if propagating.any() else 0.0
    dps = int(50 + 2 * growth / np.log(10))
    if dps > 900:
        return dict(seed=seed, skipped='over 900 digits')
    ex, _ = _grating_exact(sc, sc['layers'], dps=dps)
    up, _ = _grating_exact(sc, sc['layers'], 1 + 4 * ULP, dps)
    down, _ = _grating_exact(sc, sc['layers'], 1 - 4 * ULP, dps)
    row = dict(seed=seed, knife=max(rel(up, ex), rel(down, ex)))
    if row['knife'] > 1e-6:
        theory = _deleted_theory(sc, kx)
        if np.all(np.isfinite(theory)):
            row.update(_algorithms(sc, theory))
    return row


def _knife_summary(label, rows):
    rows = [r for r in rows if r is not None]
    knives = [r for r in rows if r.get('knife', 0) > 1e-6 and 'SMM' in r]
    print(f'  {label}: {len(rows)} scenes, {sum("skipped" in r for r in rows)} skipped, '
          f'{len(knives)} knife edges with a finite theory')
    for key in ('SMM', 'rule', 'new'):
        plain = [r[key] for r in knives]
        merged_ = [r[key + '+merge'] for r in knives]
        off = sum(isinstance(v, float) and v > 1e-6 for v in plain)
        raised = sum(isinstance(v, str) for v in plain)
        worst = max(v for v in merged_ if isinstance(v, float))
        bad = sum(not isinstance(v, float) for v in merged_)
        name = {'SMM': 'SMM', 'rule': 'ETM, the rule', 'new': 'ETM, new labels'}[key]
        print(f'    {name:16s} unmerged: {off:4d} over 1e-6 off the theory, {raised:3d} raised; '
              f'merged: worst {worst:.1e}' + (f', {bad} not finite' if bad else ''))
    return knives


KNIVES = {}


def section_knives(uniform_count, grating_count):
    header(4, 'knives -- a thick layer of a half-space\'s medium against exact theory')
    KNIVES['uniform'] = _knife_summary(f'uniform, seeds 0-{uniform_count - 1}',
                                       pool(_uniform_job, range(uniform_count)))
    KNIVES['grating'] = _knife_summary(f'TE gratings, seeds 0-{grating_count - 1}',
                                       pool(_grating_job, range(grating_count)))


# ---------------------------------------------------------------------------------- 5. fields

def _field_reference(sc, plan, xs, ys, zs, edges):
    """The deleted stack's field (SMM), carried into the full stack by plane waves written here."""
    from wavebit.rcwa._engine.solve_2d import order_grid_kxy
    n = len(sc['layers'])
    kept = sc['layers'][plan.bottom:n - plan.top]
    total_kept = float(np.sum([t for _, t in kept])) if kept else 0.0
    if not kept:                     # a bare interface: a layer of zero thickness is invisible
        kept = [(np.full(len(sc['layers'][0][0]), 1.7 + 0j), 0.0)]
    with labels('off'), merged(False), warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        D = solver(sc, 'SMM', layers=kept, answers='field')
        rD = D.solve(auto_detune=False)
    k0 = 2 * np.pi / sc['wavelength']
    kx, ky = D.order_kxy()
    gx, gy = (np.asarray(a).ravel() for a in order_grid_kxy(np, np.asarray(kx), np.asarray(ky)))

    def maystre(w):
        v = np.sqrt(np.asarray(w, dtype=complex))
        return np.where(v.real + v.imag < 0, -v, v)
    eps = {'top': complex(sc['top']), 'bottom': complex(sc['bot'])}
    kz = {side: maystre(eps[side] - gx ** 2 - gy ** 2) for side in eps}
    lit = sc['side']
    d = {'bottom': edges[plan.bottom], 'top': edges[-1] - edges[n - plan.top]}
    (cx, hx), (cy, hy) = D.order_window
    f0 = np.exp(1j * k0 * kz[lit][(hy - cy) * (2 * hx + 1) + hx - cx] * d[lit])
    lo, hi = edges[plan.bottom], edges[n - plan.top]
    X, Y = np.meshgrid(xs, ys)
    lateral = np.exp(1j * k0 * (X[..., None] * gx + Y[..., None] * gy))

    def wave(spectrum, kz_z, dz):
        spectrum = np.asarray(spectrum)
        with np.errstate(all='ignore'):
            value = spectrum * np.exp(1j * k0 * kz_z * dz)[:, None]
        return np.where(spectrum == 0, 0, value)
    out = {}
    names = ('incident',) if sc['kind'] in ('te', 'tm') else ('s', 'p')
    for name in names:
        spectra = rD._field_state.data['boundary'][name]
        rows = []
        for z in zs:
            owner = min(int(np.sum(edges[1:] <= z)), n - 1)   # the layer above owns a face
            if owner < plan.bottom or owner >= n - plan.top:
                side = 'bottom' if owner < plan.bottom else 'top'
                inner = lo if side == 'bottom' else hi
                up = 1 if side == 'bottom' else -1
                incoming, outgoing = spectra[side]
                coef = (wave(incoming, up * kz[side], z - inner)
                        + wave(outgoing, -up * kz[side], z - inner))
                with np.errstate(all='ignore'):
                    rows.append(f0 * np.einsum('yxm,mc->yxc', lateral, coef))
            else:
                local = min(max(z - lo, 0.0), total_kept)
                rows.append(f0 * np.asarray(rD.reconstruct_field(
                    sample_x=xs, sample_y=ys, sample_z=[local], incidence=name))[0])
        out[name] = np.stack(rows)
    return out


def _field_job(args):
    family, seed = args
    sc = class_a_scene(seed) if family == 'uniform' else class_a_grating_te(seed)
    edges = np.r_[0., np.cumsum([t for _, t in sc['layers']])]
    zs = sorted({float(a + f * (b - a)) for a, b in zip(edges[:-1], edges[1:])
                 for f in (0, .2, .5, .8, 1)})
    xs = [-0.37 * sc['period'], 0.0, 0.21 * sc['period']]
    ys = [0.0] if sc['kind'] in ('te', 'tm') else [0.0, 0.13 * sc['period']]
    names = ('incident',) if sc['kind'] in ('te', 'tm') else ('s', 'p')
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        s = solver(sc, 'SMM', answers='field')
        s.solve(auto_detune=False)
    plan = s._half_space_merged
    if plan is None:
        return dict(seed=seed, skipped='not merged')
    ref = _field_reference(sc, plan, xs, ys, zs, edges)
    if not all(np.all(np.isfinite(v)) for v in ref.values()):
        return dict(seed=seed, skipped='theory beyond double range')
    scale = max(float(np.max(np.abs(v))) for v in ref.values())
    row = dict(seed=seed)
    for label, algorithm, mode in (('SMM', 'SMM', 'off'), ('rule', 'ETM', 'off'),
                                   ('new', 'ETM', 'complex')):
        with labels(mode), warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            res = solver(sc, algorithm, answers='field').solve(auto_detune=False)
            err = max(float(np.max(np.abs(np.asarray(res.reconstruct_field(
                sample_x=xs, sample_y=ys, sample_z=zs, incidence=name)) - ref[name])))
                for name in names)
        row[label] = err / scale
    return row


def section_fields():
    header(5, 'fields -- the merged field on the knife edges, against the deleted stack carried '
              'by plane waves')
    for family in ('uniform', 'grating'):
        if family not in KNIVES:
            print(f'  {family}: run section 4 first')
            continue
        rows = pool(_field_job, [(family, r['seed']) for r in KNIVES[family]])
        done = [r for r in rows if 'skipped' not in r]
        print(f'  {family}: {len(done)} fields ({len(rows) - len(done)} skipped: theory beyond '
              'double range or not merged); worst relative to the largest field: '
              + ', '.join(f'{name} {max(r[key] for r in done):.1e}'
                          for key, name in (('SMM', 'SMM'), ('rule', 'ETM rule'),
                                            ('new', 'ETM new labels'))))


# --------------------------------------------------------------------------------- 6. develop

def _digest(value):
    if isinstance(value, str):
        return value
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]


def develop_digests():
    """Real-angle digests: amplitudes of 400 scenes and fields of 120, per family and algorithm."""
    out = {}
    for name, gen in (('class A', class_a_scene), ('general', general_scene),
                      ('etched', etched_scene)):
        for seed in range(400):
            sc = gen(seed)
            sc['theta'] = complex(sc['theta'].real, 0.0)
            for algorithm in ('ETM', 'SMM'):
                out[f'{name} {seed} {algorithm} amplitude'] = _digest(
                    amplitudes(sc, algorithm, 'complex'))
                if seed >= 120:
                    continue
                try:
                    with warnings.catch_warnings(), np.errstate(all='ignore'):
                        warnings.simplefilter('ignore')
                        s = solver(sc, algorithm, answers='field')
                        r = s.solve(auto_detune=False)
                        total = float(np.sum([t for _, t in sc['layers']]))
                        kw = {} if sc['kind'] in ('te', 'tm') else {
                            'incidences': ('incident', 's', 'p')}
                        cells = [np.asarray(s.reconstruct_field(sample_x=5, sample_y=2,
                                                                sample_z=3, **kw)),
                                 np.asarray(r.reconstruct_field(
                                     sample_x=4, sample_y=2,
                                     sample_z=list(np.linspace(0, total, 7))))]
                        value = np.concatenate([np.ravel(np.asarray(getattr(r.incident, f)))
                                                for f in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T')]
                                               + [c.ravel() for c in cells])
                except Exception as exc:        # noqa: BLE001
                    value = type(exc).__name__
                out[f'{name} {seed} {algorithm} field'] = _digest(value)
    return out


def section_develop(rev):
    header(6, f'develop -- real-angle digests against {rev}' if rev
           else 'develop -- skipped (no --develop)')
    if not rev:
        return
    with tempfile.TemporaryDirectory() as tmp:
        tree = os.path.join(tmp, 'tree')
        os.mkdir(tree)
        archive = subprocess.run(['git', '-C', ROOT, 'archive', rev], check=True,
                                 capture_output=True).stdout
        subprocess.run(['tar', '-x', '-C', tree], input=archive, check=True)
        dump = os.path.join(tmp, 'develop.txt')
        subprocess.run([sys.executable, os.path.abspath(__file__), '--dump', dump,
                        '--root', tree], check=True, capture_output=True,
                       env=dict(os.environ, PYTHONWARNINGS='ignore'))
        theirs = dict(line.rstrip('\n').rsplit(' ', 1) for line in open(dump))
    ours = develop_digests()
    same = sum(ours[k] == theirs[k] for k in theirs)
    fields = sum(k.endswith('field') for k in theirs)
    print(f'  {same} of {len(theirs)} digests byte-identical ({len(theirs) - fields} amplitude '
          f'sets, {fields} field-rung sets of amplitudes and two field cells; class A, general '
          'and etched scenes at a real angle, ETM and SMM)')
    for key in sorted(k for k in theirs if ours[k] != theirs[k])[:10]:
        print(f'    differs: {key}')


# ------------------------------------------------------------------------------------ 7. cost

def section_cost():
    header(7, 'cost -- eager NumPy, best of 5 after a warm-up')
    S = np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1
    cell = np.full((6, 6), 25.0)
    cell[0, :] = 1.0
    cell[:, 0] = 1.0
    scenes = [('1D TE, orders 30, 3 layers', dict(tetm='te', truncation_order=[30],
                                                   period=[2000e-9], structure=np.concatenate([S] * 3))),
              ('conical, orders 30, 3 layers', dict(psi=0.3, phi=0.3, truncation_order=[30],
                                                     period=[2000e-9], structure=np.concatenate([S] * 3))),
              ('2D (6, 6), 3 layers', dict(psi=0.3, phi=0.3, truncation_order=[6, 6],
                                           period=[2000e-9, 2000e-9],
                                           structure=np.stack([cell] * 3)))]
    for name, kw in scenes:
        for theta in (0.5, 0.5 + 0.3j):
            times = {}
            for label, algorithm, mode in (('rule', 'ETM', 'off'), ('new', 'ETM', 'complex'),
                                           ('SMM', 'SMM', 'off')):
                with labels(mode), warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    s = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=25.0, theta=theta,
                                             wavelength=900e-9, thickness=[100e-9] * 3,
                                             layer_algorithm=algorithm, **kw)
                    s.solve(auto_detune=False)
                    best = np.inf
                    for _ in range(5):
                        t0 = time.perf_counter()
                        s.solve(auto_detune=False)
                        best = min(best, time.perf_counter() - t0)
                times[label] = best
            print(f'  {name:30s} theta {theta!s:>10s}: rule {1e3 * times["rule"]:7.1f} ms, new '
                  f'labels {1e3 * times["new"]:7.1f} ms ({times["new"] / times["rule"]:.2f}x), '
                  f'SMM {1e3 * times["SMM"]:7.1f} ms')


def main():
    global WORKERS
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--develop', metavar='REV', help='section 6: compare with this commit')
    parser.add_argument('--only', type=int, nargs='*', help='run these sections only')
    parser.add_argument('--time', action='store_true', help='section 7: timings')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--etched', type=int, default=4000)
    parser.add_argument('--random', type=int, default=13000)
    parser.add_argument('--uniform', type=int, default=1536)
    parser.add_argument('--gratings', type=int, default=600)
    parser.add_argument('--dump', help=argparse.SUPPRESS)
    parser.add_argument('--root', help=argparse.SUPPRESS)
    args = parser.parse_args()
    WORKERS = args.workers
    if args.dump:
        with open(args.dump, 'w') as out:
            for key, value in develop_digests().items():
                out.write(f'{key} {value}\n')
        return
    print('ETM labels from the state below; a layer of a half-space\'s medium -- '
          'studies/rcwa/etm_maxvol_labels/etm_maxvol_labels.py')
    print(f'numpy {np.__version__}, mpmath {mp.__version__}; every solve NumPy complex128')
    sections = {1: section_table, 2: lambda: section_etched(args.etched),
                3: lambda: section_random(args.random),
                4: lambda: section_knives(args.uniform, args.gratings), 5: section_fields,
                6: lambda: section_develop(args.develop)}
    if args.time:
        sections[7] = section_cost
    for number, section in sections.items():
        if not args.only or number in args.only:
            section()


if __name__ == '__main__':
    main()
