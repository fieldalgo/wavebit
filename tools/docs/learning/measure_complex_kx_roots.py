"""Record the solver numbers of Appendices A and B, choosing kz at a complex kx.

Most of the two appendices is closed-form and is computed where it is drawn. This script records
the numbers that need a solver:

1. Wavebit's ``boundary_kz`` in air at kx = 1.5 - 1e-9i, 1.5 and 1.5 + 1e-9i (A);
2. which of four root rules the installed ``boundary_kz`` equals on 600 random complex kx, also split
   into the lossless (A) and absorbing (B) media of that sample;
3. a one-layer grating lit at theta = 0.2 + i Im(theta): Wavebit, and optionally RETICOLO (A);
4. glass over a gain half-space, and glass | 2 um gain layer | air: Wavebit against Airy (B);
5. the transverse kx/k0 Wavebit makes of theta = 0.4 and 0.4 +- 1e-12i, in eps = 4 + 3i and in the
   lossless eps = 4 (B).

Wavebit's half-space rule changed on 2026-10-02 from path D to Maystre's cut. The appendices still
quote what path D gave, so the record keeps those numbers in a ``path_d`` block, measured at the last
commit with the old rule. A run carries that block forward unchanged. A run over a record that was
itself measured under path D (``boundary_kz`` equal to path D on all 600) makes the block from it.

The rules, the sampling and every setup are those of the lab notes
``complex_kx_root_choice.ko.md`` and ``complex_kx_root_choice_lossy_gain.ko.md`` and their figure
script ``_complex_kx_root_figs.py``. They are copied here, not imported, so the record does not move
when the lab script does. The numbers written here must equal that script's output.

    conda run -n fieldwave python tools/measure_complex_kx_roots.py --wavebit CHECKOUT
    RETICOLO_PATH=DIR conda run -n fieldwave python tools/measure_complex_kx_roots.py \\
        --wavebit CHECKOUT --reticolo

The RETICOLO columns need Octave, oct2py and RETICOLO V10, so they are recorded rather than
recomputed at build time. A run without ``--reticolo`` keeps the RETICOLO block that is
already in the record, provided that block was measured on the same grating.
"""
import argparse
import datetime
import importlib
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / 'artifacts/docs/learning/complex-kx-roots.json'
BRIDGE = 'tests/references/interface/Reticolo.py::run_res2_all_modes'
PROVENANCE = ('Wavebit numbers come from the checkout named in wavebit_source; rerun this script '
              'to refresh them. The RETICOLO numbers in grating.reticolo need Octave, oct2py and '
              'RETICOLO V10; they are recorded here, not recomputed at build time. path_d holds the '
              'Wavebit numbers under the half-space rule before 2026-10-02, measured at its own commit '
              'and carried forward.')
PATH_D_NOTE = ('Wavebit numbers under path D, the half-space rule before 2026-10-02 (principal root plus '
               'continuity fix), measured at the commit named here. Later runs carry this block unchanged.')


# ---------------------------------------------------------------- the four root rules
# Verbatim from the lab figure script, which checks D and E against step-by-step tracking and
# every rule's branch cut against where the grid actually jumps.
FLAT = 1e-12      # the principal root's tie-break (Wavebit's half-space `_FLAT` before 2026-10-02):
                  # a real part below this counts as zero; then Im(kz) > 0 wins


def principal(kz2):
    """Principal root: Re(kz) > 0, or Im(kz) > 0 when the real part is zero."""
    v = np.sqrt(np.asarray(kz2, dtype=complex))
    flip = np.where(np.abs(v.real) > FLAT, v.real < 0, v.imag < 0)
    return np.where(flip, -v, v)


def rule_D(eps, kx):
    """Path D: continue the root along the straight kx path from normal incidence (kx = 0).

    In the kz^2 plane the path is the segment eps -> eps - kx^2; the principal root is cut only on
    the negative real axis, so flip where the segment crosses it.
    """
    kx = np.asarray(kx, dtype=complex)
    eps = np.asarray(eps, dtype=complex) + 0 * kx
    kp2 = kx * kx
    v = principal(eps - kp2)
    down = kp2.imag > 0
    t = eps.imag / np.where(down, kp2.imag, 1.0)
    flip = down & (t < 1) & (eps.real - t * kp2.real < 0)
    return np.where(flip, -v, v)


def rule_E(eps, kx):
    """Path E: start at the real kx with the same real part and raise only the imaginary part."""
    kx = np.asarray(kx, dtype=complex)
    eps = np.asarray(eps, dtype=complex) + 0 * kx
    a, b = kx.real, kx.imag
    v = principal(eps - kx * kx)
    ab = a * b
    down = ab > 0
    s = eps.imag / np.where(down, 2 * ab, 1.0)
    flip = down & (s < 1) & (eps.real - a * a + (s * b) ** 2 < 0)
    return np.where(flip, -v, v)


def rule_M(eps, kx):
    """Maystre's cut, Re(kz) + Im(kz) > 0: RETICOLO's `retsqrt(x, -1)` default; S4 answers the same."""
    v = np.sqrt(np.asarray(eps, dtype=complex) - np.asarray(kx, dtype=complex) ** 2)
    return np.where(v.real + v.imag < 0, -v, v)


def rule_F(eps, kx):
    """Im(kz) > 0: fmmax 1.7.1 `_select_eigenvalues_sign`."""
    v = np.sqrt(np.asarray(eps, dtype=complex) - np.asarray(kx, dtype=complex) ** 2)
    return np.where(v.imag < 0, -v, v)


RULES = {'D': rule_D, 'E': rule_E, 'M': rule_M, 'F': rule_F}
RULE_TEXT = {
    'D': ('path D: continued along the straight kx path from normal incidence (Wavebit before 2026-10-02: '
          'principal root plus continuity fix)'),
    'E': 'path E: continued from the real kx with the same real part, raising only Im(kx)',
    'M': "Maystre's cut, Re(kz) + Im(kz) > 0 (Wavebit since 2026-10-02; RETICOLO retsqrt(x, -1); S4 answers the same)",
    'F': 'Im(kz) > 0 (fmmax 1.7.1)',
}


def sample_media(rng, N):
    """Half lossless; Re(eps) in 1..6, Im(eps) in 0.01..3 on the lossy half; real kx in 0..3."""
    lossless = rng.random(N) < 0.5
    eps = rng.uniform(1.0, 6.0, N) + 1j * np.where(lossless, 0.0, rng.uniform(0.01, 3.0, N))
    return eps, rng.uniform(0.0, 3.0, N) + 0j


# ---------------------------------------------------------------- closed forms for the gain example
GAIN_EPS, GLASS = 1.0 - 0.05j, 2.25        # glass (n = 1.5) onto a gain medium, TE


def te_r(k1, k2):
    return (k1 - k2) / (k1 + k2)


def gain_turned_on(eps, kx, n=20000):
    """Hold kx, switch the gain on from zero and follow the root: the causal root for weak gain."""
    root = complex(principal(eps.real - kx * kx))
    for s in np.linspace(0, 1, n + 1)[1:]:
        v = complex(np.sqrt(eps.real + 1j * eps.imag * s - kx * kx))
        root = v if abs(v - root) <= abs(-v - root) else -v
    return root


def slab_R(theta, d_um, eps2=GAIN_EPS, wavelength_um=1.0):
    """Glass | layer of thickness d | air, TE, by the Airy formula (no root choice inside the layer)."""
    kx = np.sqrt(GLASS) * np.sin(theta)
    k1 = complex(principal(GLASS - kx * kx))
    k2 = complex(principal(eps2 - kx * kx))
    k3 = complex(principal(1.0 - kx * kx))
    phase = np.exp(2j * k2 * 2 * np.pi / wavelength_um * d_um)
    r12, r23 = te_r(k1, k2), te_r(k2, k3)
    return abs((r12 + r23 * phase) / (1 + r12 * r23 * phase)) ** 2


# ---------------------------------------------------------------- helpers
def pair(z):
    z = complex(z)
    return [z.real, z.imag]


def solve(wavebit, **kw):
    """One numpy-backend solve; returns the result and the warning categories it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = wavebit.rcwa(backend='numpy', **kw).solve()
    return res, sorted({w.category.__name__ for w in caught})


def git(checkout, *args):
    try:
        out = subprocess.run(['git', '-C', str(checkout), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def import_wavebit(checkout):
    sys.path.insert(0, str(checkout / "src"))
    import wavebit
    found = Path(wavebit.__file__).resolve()
    if checkout not in found.parents:
        raise SystemExit(f'imported wavebit from {found}, not from --wavebit {checkout}')
    from wavebit.compat import legacy
    return legacy


# ---------------------------------------------------------------- the measurements
def boundary_kz_air():
    """Appendix A (lab note 1, section 6): air, kx = 1.5 + d i.

    Under path D a real kx was a tie, settled as Im(kz) > 0; under Maystre's cut it is no tie.
    """
    from wavebit._common.conventions import boundary_kz
    rows = []
    for d in (-1e-9, 0.0, 1e-9):
        kx = 1.5 + 1j * d
        rows.append({'kx': pair(kx), 'kz': pair(boundary_kz(np, 1.0, kx ** 2))})
    return {'eps': 1.0, 'function': 'wavebit._common.conventions.boundary_kz', 'rows': rows}


def rule_match():
    """The first 600 media of the lab's seed-7 draw of 3000, with Im(kx) in -0.5..0.5."""
    from wavebit._common.conventions import boundary_kz
    rng = np.random.default_rng(7)
    eps, kr = sample_media(rng, 3000)
    M = 600
    eps = eps[:M]
    kc = kr[:M] + 1j * rng.uniform(-0.5, 0.5, M)
    got = np.array([complex(boundary_kz(np, complex(e), complex(k) ** 2)) for e, k in zip(eps, kc)])
    tolerance = 1e-12
    lossless = eps.imag == 0

    def split(mask):
        same = {k: int(np.sum((np.abs(got - RULES[k](eps, kc)) < tolerance) & mask)) for k in RULES}
        same['principal'] = int(np.sum((np.abs(got - principal(eps - kc * kc)) < tolerance) & mask))
        return {'samples': int(mask.sum()), 'same_as': same}
    return {
        'samples': M,
        'sampling': ('numpy default_rng(7): 3000 media drawn as sample_media (half lossless; Re(eps) 1..6, '
                     'Im(eps) 0.01..3 on the lossy half; real kx 0..3), then Im(kx) uniform in -0.5..0.5 '
                     'for the first 600'),
        'tolerance': tolerance,
        'rules': RULE_TEXT,
        'same_as': {k: int(np.sum(np.abs(got - RULES[k](eps, kc)) < tolerance)) for k in RULES},
        'lossless': split(lossless),
        'absorbing': split(~lossless),
    }


def theta_to_kx(wavebit):
    """Appendix B: the incident kx/k0 Wavebit builds from theta, before any solve.

    A real theta is read as the phase angle (real kx); a complex theta as the formal n sin(theta).
    """
    rows = []
    for top in ([4.0, 3.0], [4.0, 0.0]):
        for theta_imag in (0.0, 1e-12, -1e-12):
            theta = 0.4 + 1j * theta_imag if theta_imag else 0.4
            solver = wavebit.rcwa(backend='numpy', tetm='te', top=complex(*top), bot=1.0, theta=theta,
                                  truncation_order=0, wavelength=1000.0, period=[2000.0], thickness=[100.0],
                                  structure=np.array([[[1.0, 1.0]]], dtype=complex))
            rows.append({'top_eps': top, 'theta': [0.4, theta_imag],
                         'kx': pair(solver._incident_kxy(1000.0)[0])})
    return {'function': 'wavebit rcwa(...)._incident_kxy(wavelength)[0], normalized by k0', 'rows': rows}


THETA_IMAG = (-1e-3, -1e-9, 0.0, 1e-9, 1e-6, 1e-3)


def grating_setup():
    return dict(tetm='te', top=1.0, bot=2.25, structure=np.array([[[9.0] * 5 + [1.0] * 5]]),
                thickness=[0.15], period=[0.7], wavelength=0.9, truncation_order=5)


def grating_record_setup(kw):
    return {'top': kw['top'], 'bot': kw['bot'],
            'structure': 'one layer of 10 pixels: 5 of eps 9, then 5 of eps 1',
            'thickness': kw['thickness'][0], 'period': kw['period'][0], 'wavelength': kw['wavelength'],
            'polarization': 'TE', 'truncation_order': kw['truncation_order'], 'theta_real': 0.2,
            'fourier_coefficients': 'integrated', 'backend': 'numpy'}


def grating_wavebit(wavebit, kw):
    """Appendix A (lab note 1, section 11), Wavebit side: zeroth-order reflection amplitude and efficiency."""
    rows = []
    for d in THETA_IMAG:
        th = 0.2 + 1j * d if d else 0.2
        res, caught = solve(wavebit, fourier_coefficients='integrated', theta=th, **kw)
        zero = int(np.flatnonzero(np.asarray(res.orders_u) == 0)[0])
        rows.append({'theta_imag': d,
                     'r0': pair(np.asarray(res.incident.r_s).ravel()[zero]),
                     'R0': float(np.real(np.asarray(res.R)).ravel()[zero]),
                     'warnings': caught})
    return rows


def grating_reticolo(checkout, reticolo_path, kw):
    """Appendix A (lab note 1, section 11), RETICOLO side, through the test tree's bridge as the lab script calls it."""
    try:
        from oct2py import octave, __version__ as oct2py_version
    except ImportError as exc:
        raise SystemExit(f'--reticolo needs oct2py and Octave: {exc}')
    octave.addpath(octave.genpath(str(reticolo_path)))
    sys.path.insert(0, str(checkout / 'tests' / 'references'))
    bridge = importlib.import_module('interface.Reticolo')
    if checkout not in Path(bridge.__file__).resolve().parents:
        raise SystemExit(f'imported the RETICOLO bridge from {bridge.__file__}, not from {checkout}')
    reti = bridge.Reticolo()
    rows = []
    for d in THETA_IMAG:
        th = 0.2 + 1j * d if d else 0.2
        refl, _ = reti.run_res2_all_modes(kw['period'], kw['truncation_order'], kw['structure'],
                                          kw['thickness'], th, 0, kw['wavelength'], kw['top'], kw['bot'])
        order = np.rint(np.real(np.asarray(refl.order).ravel())).astype(int)
        zero = int(np.flatnonzero(order == 0)[0])
        # Only order 0 is compared. With every mode reported (tolh = tolb = 1e6), RETICOLO writes
        # |amplitude|^2 as the efficiency of evanescent orders too at an exactly real theta, so the
        # sum jumps at that one point; that has nothing to do with the root choice.
        rows.append({'theta_imag': d,
                     'r0': pair(np.asarray(refl.amplitude).ravel()[zero]),
                     'R0': float(np.real(np.asarray(refl.efficiency).ravel()[zero])),
                     'sum_R': float(np.real(np.asarray(refl.efficiency)).sum())})
    try:
        octave_version = str(octave.version())
    except Exception:                                                  # pragma: no cover
        octave_version = None
    return {'date': datetime.date.today().isoformat(),
            'provenance': 'Needs Octave, oct2py and RETICOLO V10; recorded, not recomputed at build time.',
            'reticolo': '/'.join(Path(reticolo_path).parts[-2:]),
            'octave': octave_version, 'oct2py': oct2py_version,
            'bridge': BRIDGE, 'bridge_commit': git(checkout, 'rev-parse', 'HEAD'),
            'pol_argument': 0, 'rows': rows}


def gain(wavebit):
    """Appendix B (lab note 2, sections 3-4): glass over a gain half-space, and a 2 um gain layer under glass, TE."""
    def R(deg, bot, layer=GLASS, d_nm=100.0):
        res, caught = solve(wavebit, tetm='te', top=GLASS, bot=bot, theta=np.radians(deg), truncation_order=0,
                            wavelength=1000.0, period=[2000.0],
                            structure=np.array([[[layer, layer]]], dtype=complex), thickness=[d_nm])
        return float(np.sum(res.R).real), float(np.sum(res.T).real), caught

    half, layer = [], []
    for deg in (30.0, 45.0):
        kx = np.sqrt(GLASS) * np.sin(np.radians(deg))
        k1 = complex(principal(GLASS - kx * kx))
        Rg, _, wg = R(deg, GAIN_EPS)
        Rl, _, wl = R(deg, GAIN_EPS.conjugate())
        half.append({'theta_deg': deg, 'R': Rg, 'warnings': wg,
                     'R_lossy_mirror': Rl, 'warnings_lossy_mirror': wl,
                     'R_principal_root': abs(te_r(k1, complex(principal(GAIN_EPS - kx * kx)))) ** 2,
                     'R_gain_continued_root': abs(te_r(k1, gain_turned_on(GAIN_EPS, kx))) ** 2})
        Rs, Ts, ws = R(deg, 1.0, layer=GAIN_EPS, d_nm=2000.0)
        airy = slab_R(np.radians(deg), 2.0)
        layer.append({'theta_deg': deg, 'R': Rs, 'T': Ts, 'warnings': ws,
                      'R_airy': airy, 'abs_difference': abs(Rs - airy)})
    return {
        'setup': {'top': GLASS, 'gain_eps': pair(GAIN_EPS), 'polarization': 'TE', 'wavelength_nm': 1000.0,
                  'period_nm': 2000.0, 'truncation_order': 0, 'backend': 'numpy',
                  'half_space': 'a 100 nm glass layer, then the gain medium as bot',
                  'lossy_mirror': 'the same with bot = conj(gain_eps) = 1 + 0.05i',
                  'layer': 'a 2000 nm gain layer between glass (top) and air (bot)'},
        'half_space': half,
        'layer': {'thickness_um': 2.0, 'rows': layer},
    }


# ---------------------------------------------------------------- main
def carried_reticolo(output, setup):
    """The RETICOLO block already recorded, if it was measured on this grating."""
    try:
        old = json.loads(output.read_text())['grating']
    except (OSError, ValueError, KeyError):
        return None
    block = old.get('reticolo')
    if block and old.get('setup') == setup and [r['theta_imag'] for r in block['rows']] == list(THETA_IMAG):
        return block
    if block:
        print('The recorded RETICOLO block was measured on another grating; dropped. Rerun with --reticolo.')
    return None


def measured_under_path_d(record):
    match = record.get('rule_match', {})
    return match.get('samples') is not None and match.get('same_as', {}).get('D') == match['samples']


def carried_path_d(output):
    """The path-D block already recorded, or one made from a record that was measured under path D."""
    try:
        old = json.loads(output.read_text())
    except (OSError, ValueError):
        return None
    if 'path_d' in old:
        return old['path_d']
    if not measured_under_path_d(old):
        return None
    match = old['rule_match']
    return {'note': PATH_D_NOTE, 'date': old['date'], 'wavebit_source': old['wavebit_source'],
            'boundary_kz_air': old['boundary_kz_air'],
            'rule_match': {k: match[k] for k in ('samples', 'same_as', 'lossless', 'absorbing')},
            'grating_wavebit': old['grating']['wavebit'],
            'gain_half_space': old['gain']['half_space']}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--wavebit', type=Path,
                        default=Path(os.environ.get('WAVEBIT_SOURCE', ROOT)),
                        help='Wavebit source checkout to measure (default: this repository)')
    parser.add_argument('--reticolo', action='store_true',
                        help='also run RETICOLO through Octave; its location comes from RETICOLO_PATH')
    parser.add_argument('--output', type=Path, default=RECORD)
    args = parser.parse_args()
    preserved = (ROOT / 'docs/validation/learning-records').resolve()
    if args.output.resolve().is_relative_to(preserved):
        parser.error('historical learning records are immutable; choose an artifacts output path')
    checkout = args.wavebit.resolve()
    if not (checkout / 'src' / 'wavebit' / '__init__.py').is_file():
        parser.error('--wavebit must name a Wavebit source checkout')
    reticolo_path = None
    if args.reticolo:
        if not os.environ.get('RETICOLO_PATH'):
            parser.error('--reticolo needs RETICOLO_PATH set to the RETICOLO V10 folder (the one holding res2.m)')
        reticolo_path = Path(os.environ['RETICOLO_PATH']).resolve()
        if not (reticolo_path / 'res2.m').is_file():
            parser.error(f'RETICOLO_PATH has no res2.m: {reticolo_path}')

    wavebit = import_wavebit(checkout)
    dirty = git(checkout, 'status', '--porcelain', '--untracked-files=no')
    kw = grating_setup()
    setup = grating_record_setup(kw)
    record = {
        'date': datetime.date.today().isoformat(),
        'script': 'tools/docs/learning/measure_complex_kx_roots.py',
        'provenance': PROVENANCE,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'versions': {'numpy': np.__version__, 'wavebit': getattr(wavebit, '__version__', None)},
        'wavebit_source': {'commit': git(checkout, 'rev-parse', 'HEAD'),
                           'describe': git(checkout, 'describe', '--tags', '--always', '--dirty'),
                           'dirty': None if dirty is None else bool(dirty)},
        'boundary_kz_air': boundary_kz_air(),
        'rule_match': rule_match(),
        'grating': {
            'setup': setup,
            'order_kx': [{'order': m, 'kx': float(np.sin(0.2) + m * kw['wavelength'] / kw['period'][0])}
                         for m in (-1, 0, 1)],
            'theta_imag': list(THETA_IMAG),
            'wavebit': {'rows': grating_wavebit(wavebit, kw)},
        },
        'gain': gain(wavebit),
        'theta_to_kx': theta_to_kx(wavebit),
    }
    reticolo = (grating_reticolo(checkout, reticolo_path, kw) if args.reticolo
                else carried_reticolo(args.output, setup))
    if reticolo:
        record['grating']['reticolo'] = reticolo
    path_d = None if measured_under_path_d(record) else carried_path_d(args.output)
    if path_d:
        record['path_d'] = path_d
    if record['wavebit_source']['dirty']:
        print('warning: the Wavebit checkout has uncommitted changes; the commit does not name the code measured')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')

    print(f"Wavebit {record['versions']['wavebit']} at {record['wavebit_source']['describe']}")
    print('boundary_kz, air: ' + ', '.join(f"kx = {r['kx'][0]}{r['kx'][1]:+g}i -> {complex(*r['kz']):.6f}"
                                          for r in record['boundary_kz_air']['rows']))
    print('boundary_kz equals rule: ' + ', '.join(f'{k} {v}/600' for k, v in record['rule_match']['same_as'].items()))
    for part in ('lossless', 'absorbing'):
        block = record['rule_match'][part]
        print(f'   {part}: ' + ', '.join(f"{k} {v}/{block['samples']}" for k, v in block['same_as'].items()))
    print('theta -> kx/k0: ' + ', '.join(f"eps {complex(*r['top_eps'])}, theta {complex(*r['theta'])} -> "
                                        f"{complex(*r['kx']):.6f}" for r in record['theta_to_kx']['rows']))
    reti_rows = reticolo['rows'] if reticolo else [None] * len(THETA_IMAG)
    for w, r in zip(record['grating']['wavebit']['rows'], reti_rows):
        line = f"Im(theta) {w['theta_imag']:+.0e}: Wavebit r0 {complex(*w['r0']):.6f} R0 {w['R0']:.6f}"
        if r:
            line += f" | RETICOLO r0 {complex(*r['r0']):.6f} R0 {r['R0']:.6f} sum {r['sum_R']:.6f}"
        print(line + (f"  {w['warnings']}" if w['warnings'] else ''))
    for h, s in zip(record['gain']['half_space'], record['gain']['layer']['rows']):
        print(f"gain {h['theta_deg']:g} deg: half-space R {h['R']!r} {h['warnings'] or 'no warnings'} | "
              f"lossy mirror R {h['R_lossy_mirror']!r} | 2 um layer R {s['R']:.9f} (Airy {s['R_airy']:.9f}) "
              f"{s['warnings'] or 'no warnings'}")
    if path_d:
        print(f"path D block carried from Wavebit {path_d['wavebit_source']['describe']} ({path_d['date']})")
    elif not measured_under_path_d(record):
        print('warning: no path D block; the appendices quote path D numbers that this record no longer holds')
    print(f'Wrote {args.output}' + ('' if reticolo else ' (no RETICOLO block)'))


if __name__ == '__main__':
    main()
