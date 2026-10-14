"""Negative-index media: the physics, the rules, and every number ``docs/theory/rcwa/negative-index-media.md``
quotes, reproduced.

A double negative (``Re(eps) < 0`` and ``Re(mu) < 0``) carries its power against its phase
(V. G. Veselago, Sov. Phys. Usp. 10, 509 (1968)). A solver meets that in three decisions: which
root of ``kz**2`` a half-space takes, which way a real incidence angle points the in-plane
wavevector, and which of a layer's ``+-q`` modes its recursion calls "down". The document gives
the reasons; this script gives the numbers, each from a computation that needs none of the
solver's own choices where that is possible (closed-form Fresnel coefficients on a root fixed
by physics alone, the Poynting vector of the field the solve returns, a thick lossy slab that
keeps both roots, reciprocity, an outside solver's frozen answers).

   1 roots      the two roots of kz**2 at a glass / double-negative interface, what each one
                does, and the worked example: Maystre's cut on eps*mu returns R = 928.78 where
                the answer is 0.0010767, r becomes exactly 1/r, and R + T = 1 holds for both.
                The standard table (0/30/60 deg, lossy and lossless) that section 14 of the
                document compares other tools against.
   2 mirror     the mirror of Maystre's cut: the bit identity with -conj(maystre(conj z)),
                where it and Maystre's cut agree, limiting absorption on eps and mu separately
                against loss on the product, the decision margin, continuity through grazing,
                the tie (eps = mu = -1+1i), and bit identity for every other medium.
   3 index      the signed index n = sqrt(eps)*sqrt(mu): which media it negates, the
                negative-phase-velocity conditions of McCall, Lakhtakia and Weiglhofer (2002)
                and Depine and Lakhtakia (2004) against it (and where Im(eps*mu) < 0 is not
                one of them: a lossless double negative), bit identity, the class boundary,
                continuity across Re(eps) = 0 and Re(mu) = 0, a magnetic metal's t_p, and,
                for active media, whether n points along the half-space's kz with each
                factor's root on Maystre's cut against each root principal.
   4 theta      what a real incidence angle means: negative refraction, the incident Poynting
                vector against (theta, phi) in lossless and absorbing lit media, a worked
                absorbing lit example, the s/p basis from a negative-index side, the frame at
                theta = 0, complex angles, the s = +-1 seam, and direct kx / ky (and how they
                meet the theta door at normal incidence).
   5 amplitudes r_p = -r_s and t_p = t_s at normal incidence, the incident wave's phase and
                power directions, the beam direction of every order in a negative-index
                half-space, and unitarity of the scattering matrix between lossless
                negative-index ports.
   6 layers     a uniform double-negative layer on its own medium and random patterned layers:
                ETM against SMM with the label rules on, and with them switched off.
   7 refusals   what is refused or flagged, with the messages, active media that are not
                refused, and the gain band of a double negative against its double-positive
                mirror.
   8 oracles    closed forms on every path, the thick-slab and lossless limits, reciprocity,
                the coherent door, fmmax 1.7.1's frozen answers
                (``tests/references/golden_fmmax_dng.npz``), the mirror medium of
                A. Lakhtakia's conjugation symmetry, Microwave Opt. Technol. Lett. 40, 160
                (2004), through the solver, as V. Grunhut and R. A. Depine, Eur. Phys. J. D 62,
                227 (2011), apply it to a flat interface, and PyMoosh 4.0.1's frozen answers
                for lossless double negatives in flat stacks
                (``tests/references/golden_pymoosh_dng.npz``).
   9 backends   eager JAX and PyTorch against NumPy, ``jax.jit`` (one compile for both signs
                of index), gradients against central differences, the jit gain band.
  10 develop    with ``--develop REV``: every array of a harness of media with s = +1, byte for
                byte against a ``git archive`` of REV, and what REV did with a double negative.
  11 usage      the examples of the document's section 16, run as written there, and what each
                one prints.
  12 negation   two lossless half-spaces that are each other's negation: refused on NumPy, JAX
                and PyTorch where a retained order is evanescent, solved where none is; what a
                solve does with that refusal switched off; a loss delta, and the 1/delta growth
                of an evanescent order lit on its pole.
  13 literature the claims of the document's section 18 that a number can show: Re + Im > 0
                against the decaying root for a Drude-Drude double negative at a complex
                frequency, a medium whose Im(eps) and Im(mu) have opposite signs, a uniaxial
                pair no per-medium rule serves, the principal root of a signed zero on each
                backend and Wavebit's index of it, and how far cmath's root is from NumPy's.

Units: lengths in nm, wavevectors in units of k0. Time convention exp(-i w t): Im(eps) > 0 is
loss. Every solve is NumPy complex128 except in sections 9 and 11. Random draws are seeded.

Run from the repository root:

    python studies/rcwa/negative_index_media/negative_index_media.py --develop 0dfc4ce

It prints the log committed as ``studies/rcwa/negative_index_media/records/negative_index_media.log``, in about a minute
and a half on a laptop CPU (sections 1 to 9 take a quarter of it; section 10 solves its harness
twice; section 12 solves its scenes on three backends, twice). Sections 12 and 13 are printed
after 10 and 11, so that the earlier numbers keep their places.
``--develop`` needs this repository's history: it extracts REV with ``git archive`` into a
temporary directory and runs section 10's harness there in a subprocess, by this same script.
Without it section 10 says it was skipped and everything else is the same. ``--only 3 4`` runs
those sections. Sections 9 and 11 need JAX and PyTorch and say so if either is missing; section
12 then runs on NumPy alone, and section 13 leaves out their rows. A rerun
reproduces the log byte for byte on the same NumPy and BLAS (the log's second line names the
NumPy version); on another BLAS the last digits of the round-off-sized numbers can differ.
"""

import argparse
import importlib.util
from pathlib import Path
import cmath
import collections
import contextlib
import os
import subprocess
import sys
import tempfile
import warnings

# Select the checkout before importing any wavebit module. Archived comparisons may
# use the historical flat package; the maintained tree uses src/.
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
_expected_origin = (_PACKAGE_PARENT / 'wavebit' / '__init__.py').resolve()
if Path(_selected_wavebit.__file__).resolve() != _expected_origin:
    raise RuntimeError(
        f'Selected root {_SELECTED_ROOT} requires {_expected_origin}; '
        f'wavebit was already imported from {_selected_wavebit.__file__}. '
        'Run this study in a fresh Python process.')
try:
    _legacy_wavebit = importlib.import_module('wavebit.compat.legacy')
except ModuleNotFoundError as error:
    if error.name not in ('wavebit.compat', 'wavebit.compat.legacy'):
        raise
    _legacy_wavebit = _selected_wavebit
# These maintained experiments retain the historical factory/type contract.
wavebit = _legacy_wavebit


def _study_module(current, historical):
    """Load the matching private implementation in the selected checkout."""
    return importlib.import_module(
        historical if _legacy_wavebit is _selected_wavebit else current)


import numpy as np                                          # noqa: E402

WL = 500.0
N_GLASS = 1.5
PASSIVE = (-2 + 0.1j, -1 + 0.1j)          # the worked example: eps*mu = 1.99 - 0.3i
LOSSLESS = (-2 + 0j, -1 + 0j)             # n = -sqrt(2), |mu| = 1
ASYM = (-3 + 0.1j, -0.5 + 0.05j)          # n = -1.2254 + 0.0816i, |mu| != 1
ASYM_LOSSLESS = (-3 + 0j, -0.5 + 0j)      # n = -sqrt(1.5)
HIGH = (-6 + 0.3j, -1.5 + 0.05j)
MAGNETIC_METAL = (-2 + 0.01j, 0.1 + 0.5j)  # passive, not double negative, s = -1
TIE = (-1 + 1j, -1 + 1j)                  # kz**2 = -2i at normal incidence: Maystre's own cut
ACTIVE = (-2 - 0.1j, -1 - 0.1j)
D_LAYER = 37.0                            # a glass spacer, so every path has one layer
G1 = np.array([4.0, 4.0, 1.0, 1.0, 2.25, 2.25], dtype=complex)
G2 = np.ones((6, 6), dtype=complex)
G2[1:4, 2:5] = 4.0
G2[4, 0:2] = 2.25
H_G, P_G = 180.0, 700.0
PATHS = ('te', 'tm', 'conical', '2d', 'first_order')


def e(x):
    """An error or a small number, two significant figures."""
    return f'{float(x):.1e}'


def c(z, digits=4):
    """A complex number, its zeros unsigned: ``-0.0`` prints as ``+0.0000``."""
    z = complex(z)
    return f'{round(z.real, digits) + 0.0:+.{digits}f}{round(z.imag, digits) + 0.0:+.{digits}f}i'


def header(number, title):
    print(f'\n{number}. {title}')


def _host(v):
    return np.asarray(v.detach().cpu().numpy() if hasattr(v, 'detach') else v)


@contextlib.contextmanager
def _quiet():
    """Warnings off for the solves inside; the ones that matter are counted where they matter."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# ------------------------------------------------------------------ roots, without the solver

def maystre(z):
    """Maystre's root of ``kz**2 = z``: ``Re(kz) + Im(kz) > 0``, the root ``Re(kz) > 0`` on its
    cut (the negative imaginary axis) -- RETICOLO's default cut, which its source (``retsqrt.m``)
    calls the "coupure de Maystre" (RETICOLO: J. P. Hugonin and P. Lalanne, RETICOLO software
    for grating analysis, arXiv:2101.00901). What ``conventions.boundary_kz`` takes for an
    ordinary medium, written out again here."""
    v = np.sqrt(np.asarray(z, dtype=complex))
    return np.where(v.real + v.imag < 0, -v, v)


def mirror(z):
    """The mirror: ``Im(kz) - Re(kz) > 0``, the root ``Re(kz) < 0`` on its cut (the positive
    imaginary axis): ``-conj(maystre(conj(z)))``."""
    v = np.sqrt(np.asarray(z, dtype=complex))
    return np.where(v.imag - v.real <= 0, -v, v)


def physical(eps, mu, k_par):
    """The root physics fixes for a passive half-space, with no cut at all: the one that decays
    away from the interface, ``Im(kz) > 0``; on a lossless order, where both are real, the one
    that carries its power away, ``Re(kz/mu) > 0`` (``Re(kz/eps) > 0`` is the same root)."""
    kz = np.sqrt(complex(eps) * complex(mu) - complex(k_par) ** 2 + 0j)
    if kz.imag < 0 or (kz.imag == 0 and (kz / mu).real < 0):
        kz = -kz
    return kz


def fresnel(kz1, eps1, mu1, kz2, eps2, mu2):
    """One interface, medium 1 (lit) onto medium 2: ``{pol: (r, t, R, T)}``.

    s on the electric field, admittance ``kz/mu``; p on the magnetic field, ``kz/eps``. ``T`` is
    the transmitted flux over the incident flux, ``|t|**2 Re(Y2)/Re(Y1)``; for a lossless medium
    1 flux conservation reads ``R + T = 1`` whatever ``kz2`` is (``|Y1+Y2|**2 - |Y1-Y2|**2 =
    4 Y1 Re(Y2)`` for a real ``Y1``)."""
    out = {}
    for pol, (a, b) in (('s', (kz1 / mu1, kz2 / mu2)), ('p', (kz1 / eps1, kz2 / eps2))):
        r, t = (a - b) / (a + b), 2 * a / (a + b)
        out[pol] = (r, t, abs(r) ** 2, abs(t) ** 2 * b.real / a.real)
    return out


def glass_onto(eps, mu, theta, root):
    """Glass (n = 1.5) onto ``(eps, mu)`` at ``theta``, the medium taking ``root``."""
    k_par = N_GLASS * np.sin(theta)
    kz1 = np.sqrt(N_GLASS ** 2 - k_par ** 2 + 0j)
    if root == 'physical':
        kz2 = physical(eps, mu, k_par)
    else:
        kz2 = complex({'maystre': maystre, 'mirror': mirror}[root](eps * mu - k_par ** 2))
    return fresnel(kz1, N_GLASS ** 2, 1.0, kz2, eps, mu)


def phase_index(eps, mu, theta):
    """``N = |Re(k)|/k0`` of a plane wave with real angle ``theta`` and real ``k_par``:
    ``N**2 = (a + hypot(a, b/cos(theta)))/2`` for ``eps*mu = a + ib`` (``docs/reference/common/conventions.md``)."""
    z = complex(eps) * complex(mu)
    return np.sqrt((z.real + np.hypot(z.real, z.imag / np.cos(theta))) / 2)


# ------------------------------------------------------------------------- solver scaffolding

def formulation(value):
    """``value`` spelled as the imported ``wavebit`` takes it: ``'second_order'`` /
    ``'first_order'`` since 2026-10-10, ``'2ff'`` / ``'4ff'`` before. Archived trees run this
    script too (section 10's ``--develop``), and they know only the old spelling."""
    Formulation = _study_module('wavebit.rcwa._engine.enums', 'wavebit._core.enums').Formulation
    if value is None or hasattr(Formulation, 'FIRST_ORDER'):
        return value
    return {'second_order': '2ff', 'first_order': '4ff'}[value]


def backend_arg(name):
    """``name`` spelled as the imported ``wavebit`` takes it: ``'numpy'`` / ``'jax'`` /
    ``'torch'`` since 2026-10-11, the integers ``0`` / ``1`` / ``2`` before, when ``Backend``
    was an ``IntEnum``. Archived trees run this script too (section 10's ``--develop``), and
    they know only the old spelling."""
    Backend = _study_module('wavebit._common.enums', 'wavebit._core.enums').Backend
    if issubclass(Backend, int):
        return {'numpy': 0, 'jax': 1, 'torch': 2}[name]
    return name


def solver(top=N_GLASS ** 2, bot=N_GLASS ** 2, theta=0.0, algo='SMM', path='2d', side='top',
           phi=None, thickness=(D_LAYER,), answers='amplitude', **kw):
    """One uniform glass layer between the given half-spaces; the zeroth order only."""
    common = dict(backend=backend_arg(kw.pop('backend', 'numpy')), top=top, bot=bot,
                  theta=theta, wavelength=WL, thickness=list(thickness), layer_algorithm=algo,
                  incidence_side=side, answers=answers, **kw)
    if path in ('te', 'tm'):
        return _legacy_wavebit.rcwa(**common, tetm=path, truncation_order=0, period=[1000.0],
                            structure=np.full((1, 1, 2), N_GLASS ** 2, dtype=complex))
    if path == 'conical':
        return _legacy_wavebit.rcwa(**common, phi=0.4 if phi is None else phi, truncation_order=0,
                            period=[1000.0],
                            structure=np.full((1, 1, 2), N_GLASS ** 2, dtype=complex))
    s = _legacy_wavebit.rcwa(**common, phi=(0.4 if path == 'first_order' else 0.0) if phi is None else phi,
                     truncation_order=(0, 0), period=(1000.0, 1000.0),
                     structure=np.full((1, 2, 2), N_GLASS ** 2, dtype=complex))
    if path == 'first_order':
        s.formulation = formulation('first_order')
    return s


def grating(path, algo='SMM', far=PASSIVE, d=None, lit=None, side='top', theta=0.3, phi=0.4,
            answers='amplitude', backend='numpy', psi=0.3):
    """glass (lit) / grating / double-negative half-space; or, with ``d``, / a slab of the
    double negative ``d`` thick / glass; or, with ``lit``, that medium lit on ``side`` and glass
    the far side."""
    eps, mu = far
    common = dict(backend=backend_arg(backend), top=N_GLASS ** 2, wavelength=WL,
                  layer_algorithm=algo, theta=theta, answers=answers)
    if path in ('te', 'tm', 'conical'):
        layer, kw = G1[None, None, :], dict(period=[P_G], truncation_order=3)
    else:
        layer, kw = G2[None], dict(period=(P_G, 0.9 * P_G), truncation_order=(1, 1))
    kw.update(tetm=path) if path in ('te', 'tm') else kw.update(phi=phi, psi=psi)
    if lit is not None:
        half = wavebit.Material(*lit)
        common.update(top=half, bot=N_GLASS ** 2) if side == 'top' else common.update(bot=half)
        s = _legacy_wavebit.rcwa(**common, **kw, thickness=[H_G], structure=layer, incidence_side=side)
    elif d is None:
        s = _legacy_wavebit.rcwa(**common, **kw, bot=wavebit.Material(eps, mu=mu), thickness=[H_G],
                         structure=layer)
    else:
        s = _legacy_wavebit.rcwa(**common, **kw, bot=N_GLASS ** 2, thickness=[d, H_G],
                         structure=np.concatenate([np.full_like(layer, eps), layer]))
        s.mu = np.concatenate([np.full_like(layer, mu), np.ones_like(layer)])
    if path == 'first_order':
        s.formulation = formulation('first_order')
    return s


def variants(result, path):
    return [result.incident] if path in ('te', 'tm') else [result.incident, result.s_inc,
                                                           result.p_inc]


def amplitudes(result, path, fields=('R', 'T', 'r_s', 'r_p', 't_s', 't_p')):
    return np.concatenate([np.ravel(_host(getattr(v, f))) for v in variants(result, path)
                           for f in fields])


def poynting(field):
    """``S = Re(E x H*)/2`` of a ``(..., 6)`` field sample."""
    return 0.5 * np.real(np.cross(field[..., :3], np.conj(field[..., 3:])))


def check_interface(eps, mu, path, algo, side, thetas, lit):
    """Max deviation of ``r, t, R, T`` (s and p) from the closed form, glass on the far or lit
    side of ``(eps, mu)``; the glass spacer's phase is carried in the closed form."""
    half = wavebit.Material(eps, mu=mu)
    worst = 0.0
    for theta_deg in thetas:
        theta = np.deg2rad(theta_deg)
        if lit:
            scene = dict(top=half) if side == 'top' else dict(bot=half)
            n1 = np.sqrt(complex(eps)) * np.sqrt(complex(mu))
            k_par = np.sign(n1.real) * phase_index(eps, mu, theta) * np.sin(theta)
            k1, k2 = physical(eps, mu, k_par), np.sqrt(N_GLASS ** 2 - k_par ** 2 + 0j)
            phase = np.exp(2j * np.pi / WL * k2 * D_LAYER)
            want = fresnel(k1, eps, mu, k2, N_GLASS ** 2, 1.0)
            want = {'s': (want['s'][0], want['s'][1] * phase, *want['s'][2:]),
                    'p': (want['p'][0], want['p'][1] * (n1 / mu) / N_GLASS * phase,
                          *want['p'][2:])}
        else:
            scene = dict(bot=half) if side == 'top' else dict(top=half, bot=N_GLASS ** 2)
            k_par = N_GLASS * np.sin(theta)
            k1 = np.sqrt(N_GLASS ** 2 - k_par ** 2 + 0j)
            n2 = np.sqrt(complex(eps)) * np.sqrt(complex(mu))
            phase = np.exp(2j * np.pi / WL * k1 * D_LAYER)
            want = fresnel(k1, N_GLASS ** 2, 1.0, physical(eps, mu, k_par), eps, mu)
            want = {'s': (want['s'][0] * phase ** 2, want['s'][1] * phase, *want['s'][2:]),
                    'p': (want['p'][0] * phase ** 2, want['p'][1] * N_GLASS / (n2 / mu) * phase,
                          *want['p'][2:])}
        result = solver(theta=theta, algo=algo, path=path, side=side,
                        phi=0.0 if lit else None, **scene).solve()
        subs = {'te': {'s': result.incident}, 'tm': {'p': result.incident}}.get(
            path, {'s': result.s_inc, 'p': result.p_inc})
        for pol, sub in subs.items():
            r, t = (sub.r_s, sub.t_s) if pol == 's' else (sub.r_p, sub.t_p)
            got = np.array([complex(np.sum(_host(r))), complex(np.sum(_host(t))),
                            float(np.sum(_host(sub.R))), float(np.sum(_host(sub.T)))])
            worst = max(worst, np.max(np.abs(got - np.asarray(want[pol]))))
    return worst


# ------------------------------------------------------------------------------- 1. roots

def section_roots():
    header(1, 'roots -- glass (n 1.5) onto eps = -2+0.1i, mu = -1+0.1i at normal incidence')
    eps, mu = PASSIVE
    z = eps * mu
    print(f'  eps*mu = {z.real:.2f}{z.imag:+.2f}i: below the real axis, as a gain medium '
          '(Im(eps*mu) = Re(eps) Im(mu) + Im(eps) Re(mu) < 0)')
    roots = {'maystre': complex(maystre(z)), 'mirror': complex(mirror(z))}
    for name, kz in roots.items():
        print(f'  {name:7s} root kz = {kz.real:+.4f}{kz.imag:+.4f}i: Im(kz) {kz.imag:+.4f} '
              f'({"decays" if kz.imag > 0 else "grows"} away from the interface), Re(kz) '
              f'{kz.real:+.4f} (phase {"away from" if kz.real > 0 else "toward"} it), '
              f'Re(kz/mu) {(kz / mu).real:+.4f}, Re(kz/eps) {(kz / eps).real:+.4f} '
              f'(power {"away from" if (kz / mu).real > 0 else "toward"} it)')
    print(f'  physical root (decay; no cut): equals the mirror root to '
          f'{e(abs(physical(eps, mu, 0) - roots["mirror"]))}')
    true, wrong = glass_onto(eps, mu, 0.0, 'physical'), glass_onto(eps, mu, 0.0, 'maystre')
    for name, out in (('physical root', true), ("Maystre's root", wrong)):
        r, t, R, T = out['s']
        print(f'  {name:15s}: r = {r.real:+.6f}{r.imag:+.6f}i, R = {R:.10g}, T = {T:.10g}, '
              f'R + T - 1 = {e(R + T - 1)}')
    print(f'  r(Maystre) * r(physical) - 1 = {e(abs(wrong["s"][0] * true["s"][0] - 1))}: '
          'the wrong root negates the admittance, so r -> 1/r')
    with _quiet():
        for algo in ('ETM', 'SMM'):
            result = solver(bot=wavebit.Material(eps, mu=mu), algo=algo).solve()
            print(f'  the solver now, {algo}: R = {float(np.sum(result.s_inc.R)):.10f}, '
                  f'T = {float(np.sum(result.s_inc.T)):.10f}')
    # Which root an active medium takes depends on its dispersion, not on eps*mu at one
    # frequency: J. Skaar, Phys. Rev. E 73, 026605 (2006), whose construction gives the active
    # mu = 1 medium below the double negative's index; B. Nistad and J. Skaar, Phys. Rev. E 78,
    # 036603 (2008).
    print('\n  the same eps*mu, three media (glass onto each, normal incidence):')
    print(f'    the passive double negative, its decaying root: R = {true["s"][2]:.10g}')
    gain = glass_onto(z, 1.0, 0.0, 'maystre')['s'][2]
    other = fresnel(1.5, 2.25, 1.0, complex(mirror(z)), z, 1.0)['s'][2]
    print(f'    eps = 1.99-0.3i, mu = 1 (gain), Maystre\'s root (gain switched on from zero): '
          f'R = {gain:.10g}')
    print(f'    eps = 1.99-0.3i, mu = 1, the decaying root (Skaar 2006\'s construction):      '
          f'R = {other:.10g}')
    print('\n  the standard table, glass onto the double negative, R_s / R_p:')
    print('                      0 deg                      30 deg                     60 deg')
    for label, pair in (('lossy (-2+0.1i, -1+0.1i)', PASSIVE), ('lossless (-2, -1)', LOSSLESS)):
        for root in ('physical', 'maystre'):
            cells = []
            for theta in (0.0, 30.0, 60.0):
                out = glass_onto(*pair, np.deg2rad(theta), root)
                cells.append(f'{out["s"][2]:.10g} / {out["p"][2]:.10g}')
            print(f'    {label}, {root:8s}: ' + ';  '.join(cells))
    print('\n  0 deg, R_s: a lossless double negative reflects exactly as its double-positive '
          'twin, '
          'so lossless |mu| = 1 rows cannot tell them apart; an impedance-blind solver is caught '
          'only where |mu| != 1:')
    rows = [('(-2, -1), physical', glass_onto(*LOSSLESS, 0.0, 'physical')['s'][2]),
            ('(2, 1), its double-positive twin', glass_onto(2.0, 1.0, 0.0, 'physical')['s'][2]),
            ('(-3, -0.5), physical', glass_onto(*ASYM_LOSSLESS, 0.0, 'physical')['s'][2]),
            ('(3, 0.5), its twin', glass_onto(3.0, 0.5, 0.0, 'physical')['s'][2]),
            ('(-3, -0.5), Maystre\'s root', glass_onto(*ASYM_LOSSLESS, 0.0, 'maystre')['s'][2]),
            ('eps = 1.5, mu = 1 (eps*mu as eps)', glass_onto(1.5, 1.0, 0.0, 'physical')['s'][2])]
    for label, R in rows:
        print(f'    {label:36s} R = {R:.10g}')
    print('  the mirror medium (-conj eps, -conj mu) of the lossy pair, its own (Maystre) root:')
    mir = (-np.conj(PASSIVE[0]), -np.conj(PASSIVE[1]))
    for theta in (0.0, 30.0, 60.0):
        a = glass_onto(*PASSIVE, np.deg2rad(theta), 'physical')
        b = glass_onto(*mir, np.deg2rad(theta), 'maystre')
        print(f'    {theta:4.0f} deg: R_s - R_s(mirror) = {e(abs(a["s"][2] - b["s"][2]))}, '
              f'|r_s - conj(r_s(mirror))| = {e(abs(a["s"][0] - np.conj(b["s"][0])))}, '
              f'|r_s - r_s(mirror)| = {abs(a["s"][0] - b["s"][0]):.3f}')
    # The gain twin (eps*mu as a permittivity, mu = 1, Maystre's root) with a p admittance that
    # conjugates kz but not eps: the R_p column section 14 of the document quotes for a tool
    # that does that.
    twin = complex(PASSIVE[0] * PASSIVE[1])
    cells = []
    for theta in (0.0, 30.0, 60.0):
        k_par = N_GLASS * np.sin(np.deg2rad(theta))
        k1 = np.sqrt(N_GLASS ** 2 - k_par ** 2 + 0j)
        kz = complex(maystre(twin - k_par ** 2))
        y1, y2 = k1 / N_GLASS ** 2, np.conj(kz) / twin
        cells.append(f'{abs((y1 - y2) / (y1 + y2)) ** 2:.4g}')
    print('  the gain twin with the p admittance conj(kz)/eps (kz conjugated, eps not), R_p at '
          '0 / 30 / 60 deg: ' + ' / '.join(cells))
    # A negative-phase-velocity medium that is not a double negative: Maystre's cut already
    # takes its decaying root, which a rule that negates the principal root only for double
    # negatives (PyMoosh's) misses -- a root failure, not a sign of the index.
    metal = (-10 + 0.1j, 1 + 0.5j)
    principal = np.sqrt(complex(metal[0] * metal[1]))
    kz1 = complex(N_GLASS)
    r = (kz1 - principal / metal[1]) / (kz1 + principal / metal[1])
    print(f'  a passive magnetic metal (-10+0.1i, 1+0.5i), glass onto it, 0 deg, R_s: the decaying '
          f'root {glass_onto(*metal, 0.0, "physical")["s"][2]:.4g} (Maystre\'s cut takes it: '
          f'{glass_onto(*metal, 0.0, "maystre")["s"][2]:.4g}); the principal root of eps*mu, '
          f'{c(principal)}, which grows: {abs(r) ** 2:.4g}')


# ------------------------------------------------------------------------------- 2. mirror

def section_mirror():
    boundary_kz = _study_module('wavebit._common.conventions', 'wavebit._core.conventions').boundary_kz
    header(2, 'mirror -- the double negative\'s root, Im(kz) - Re(kz) > 0')
    rng = np.random.default_rng(3)
    z = rng.normal(size=20000) + 1j * rng.normal(size=20000)
    axis = rng.uniform(0.01, 9, 1000)
    z = np.concatenate([z, 1j * axis, -1j * axis, axis + 0j, -axis + 0j, axis * (1 + 1j),
                        axis * (1 - 1j), [0j, complex(0, -0.0), complex(-0.0, 0), 2j, -2j]])
    zero = np.zeros_like(z)
    with _quiet():
        rule = boundary_kz(np, z, zero, PASSIVE)
        ordinary = boundary_kz(np, z, zero)
        flipped = -np.conj(boundary_kz(np, np.conj(z), zero))
    same = np.all(rule.view(np.uint64).reshape(-1, 2) == flipped.view(np.uint64).reshape(-1, 2),
                  axis=1)
    print(f'  boundary_kz for a double negative against -conj(maystre(conj z)), bit for bit: '
          f'{same.sum()} of {z.size} equal (random z, both axes, the diagonals, signed zeros)')
    print(f'  this script\'s mirror() against boundary_kz: '
          f'{np.sum(mirror(z) == rule)} of {z.size} equal')
    with _quiet():
        on_mirror = complex(boundary_kz(np, np.asarray([2j]), np.asarray([0j]), PASSIVE)[0])
        on_maystre = complex(boundary_kz(np, np.asarray([-2j]), np.asarray([0j]))[0])
    print(f'  on each cut the tie: Maystre\'s, kz**2 = -2i -> {c(on_maystre, 0)} '
          '(Re(kz) > 0 kept); '
          f'the mirror\'s, kz**2 = +2i -> {c(on_mirror, 0)} (Re(kz) < 0 kept)')
    agree = rule == ordinary
    left, right = z.real < 0, z.real > 0
    print(f'  where Re(kz**2) < 0 the two cuts take the same root: {agree[left].sum()} of '
          f'{left.sum()}; where Re(kz**2) > 0, never: {agree[right].sum()} of {right.sum()}')
    # Passive double negatives at real k_par: how far the decision is from zero.
    eps = rng.uniform(-5, -0.01, 200000) + 1j * rng.uniform(0, 2, 200000)
    mu = rng.uniform(-5, -0.01, 200000) + 1j * rng.uniform(0, 2, 200000)
    k_par = rng.uniform(0, 4, 200000)
    v = np.sqrt(eps * mu - k_par ** 2)
    kz = mirror(eps * mu - k_par ** 2)
    ratio = np.abs(v.imag - v.real) / np.abs(v)
    decaying = np.where((v.imag < 0) | ((v.imag == 0) & ((v / mu).real < 0)), -v, v)
    print(f'  2e5 random passive double negatives at real k_par in [0, 4): the decision '
          f'|Im v - Re v| is at least {ratio.min():.6f} |v|; the mirror root is the physical one '
          f'(decaying; power outgoing if lossless) on {np.sum(kz == decaying)}; min Im(kz) '
          f'{kz.imag.min():.1e}')
    # Negative-phase-velocity media that are not double negatives: Maystre's root already decays.
    eps = rng.uniform(-5, 5, 100000) + 1j * rng.uniform(0.001, 2, 100000)
    mu = rng.uniform(-5, 5, 100000) + 1j * rng.uniform(0.001, 2, 100000)
    single = ((eps.real < 0) != (mu.real < 0)) & ((eps * mu).imag < 0)
    eps, mu = eps[single], mu[single]
    k_par_sq = np.linspace(0, 36, 61)[:, None]
    kz = maystre(eps * mu - k_par_sq)
    print(f'  {eps.size} passive negative-phase-velocity media with one real part negative '
          '(magnetic '
          f'metals), k_par 0 .. 6 in 61 steps: Maystre\'s root decays (Im(kz) > 0) on '
          f'{np.sum(kz.imag > 0)} of {kz.size}')
    print('\n  limiting absorption on the lossless double negative (-2, -1): the decaying root of '
          'the lossy medium, distance to each cut\'s lossless root')
    print('    k_par  loss on          delta 1e-2   1e-4     1e-6     -> mirror root   '
          'Maystre root  (distance at 1e-6)')
    for kp in (0.0, 0.8, 1.3, 1.7):
        target = complex(mirror(2.0 - kp ** 2))
        other = complex(maystre(2.0 - kp ** 2))
        for label, medium in (('eps and mu', lambda d: (-2 + 1j * d, -1 + 1j * d)),
                              ('eps only', lambda d: (-2 + 1j * d, -1 + 0j)),
                              ('mu only', lambda d: (-2 + 0j, -1 + 1j * d)),
                              ('the product', None)):
            cells, last = [], None
            for d in (1e-2, 1e-4, 1e-6):
                if medium is None:
                    kz = np.sqrt(2.0 - kp ** 2 + 1j * d)          # decaying: principal here
                else:
                    kz = physical(*medium(d), kp)
                cells.append(abs(kz - (other if medium is None else target)))
                last = kz
            print(f'    {kp:4.1f}   {label:12s}  ' + '  '.join(f'{e(c):>7s}' for c in cells)
                  + f'   {c(target)}  {c(other)}  ({e(abs(last - target))} / '
                  f'{e(abs(last - other))})')
    print('    (loss on the product, eps*mu + i*delta, is measured against Maystre\'s root, '
          'which it reaches)')
    # Continuity through grazing: k_par swept across sqrt(Re(eps*mu)).
    k = np.linspace(1.30, 1.52, 2201)
    z = PASSIVE[0] * PASSIVE[1] - k ** 2
    print(f'\n  k_par swept 1.30 .. 1.52 in steps of 1e-4 through grazing ({np.sqrt(1.99):.4f}):'
          f' max |step in kz| mirror {e(np.max(np.abs(np.diff(mirror(z)))))}, Maystre '
          f'{np.max(np.abs(np.diff(maystre(z)))):.2f}')
    print('\n  the tie: eps = mu = -1+1i, normal incidence, kz**2 = -2i (on Maystre\'s cut)')
    kz2 = TIE[0] * TIE[1]
    for name, kz in (('maystre', maystre(kz2)), ('mirror', mirror(kz2))):
        kz = complex(kz)
        R = fresnel(1.5, 2.25, 1.0, kz, *TIE)['s'][2]
        print(f'    {name:7s}: kz = {kz.real:+.0f}{kz.imag:+.0f}i, glass onto it R = {R:.6g}')
    with _quiet():
        for algo in ('ETM', 'SMM'):
            result = solver(bot=wavebit.Material(*TIE), algo=algo).solve()
            print(f'    the solver, {algo}: R = {float(np.sum(result.s_inc.R)):.12f}')
    # Every other medium: `medium=` given must be Maystre's arithmetic, to the bit.
    k_par = np.sqrt(np.linspace(0, 9, 401)) + 1j * rng.normal(scale=1.5, size=401)
    k_par_sq = np.concatenate([np.linspace(0, 9, 401) + 0j, k_par ** 2])
    media = [(2.25, 1.0), (-2 + 0.1j, 1.0), (MAGNETIC_METAL), (2.25 - 0.05j, 1.2),
             (2 + 0.1j, -1 + 0.1j), (1 - 0.05j, 1 - 0.05j), (-2 + 0.1j, 1.3), (4.0, 2.0)]
    equal = total = 0
    with _quiet():
        for eps, mu in media:
            plain = boundary_kz(np, eps * mu, k_par_sq)
            told = boundary_kz(np, eps * mu, k_par_sq, (eps, mu))
            equal += int(np.sum(np.all(plain.view(np.uint64).reshape(-1, 2)
                                       == told.view(np.uint64).reshape(-1, 2), axis=1)))
            total += plain.size
    print(f'\n  {len(media)} media with one real part or none negative (gain, metals, a magnetic '
          f'metal, mu-negative), real and complex k_par: {equal} of {total} roots bit-identical '
          'with and without the material')


# ------------------------------------------------------------------------------- 3. index

def section_index():
    signed_index = _study_module('wavebit._common.conventions', 'wavebit._core.conventions').signed_index
    header(3, "index -- n = sqrt(eps)*sqrt(mu), each root on Maystre's cut")
    print('    medium (eps, mu)            principal sqrt(eps*mu)   signed n              s')
    for label, eps, mu in (('2.25, no mu', 2.25, None), ('2.25, 1.3', 2.25, 1.3),
                           ('-2+0.1i, 1 (a metal)', -2 + 0.1j, 1.0),
                           ('-2+0.1i, -1+0.1i', *PASSIVE), ('-2, -1', *LOSSLESS),
                           ('-3+0.1i, -0.5+0.05i', *ASYM),
                           ('-2+0.01i, 0.1+0.5i', *MAGNETIC_METAL),
                           ('-1+1.2i, -1+1.2i', -1 + 1.2j, -1 + 1.2j),
                           ('2+0.1i, -1+0.1i (mu-negative)', 2 + 0.1j, -1 + 0.1j)):
        if mu is None:
            principal = n = np.sqrt(complex(eps))
        else:
            principal = np.sqrt(complex(eps) * complex(mu))
            n = signed_index(complex(eps), complex(mu))
        same = 'the same bits' if n == principal else c(n)
        print(f'    {label:30s}  {c(principal)}       '
              f'{same:20s}  {int(np.sign(n.real)):+d}')
    rng = np.random.default_rng(11)
    size = 1000000
    eps = rng.uniform(-5, 5, size) + 1j * rng.uniform(0, 2, size)
    mu = rng.uniform(-5, 5, size) + 1j * rng.uniform(0, 2, size)
    mu[::4] = mu[::4].real + 0j
    eps[1::4] = eps[1::4].real + 0j
    n = signed_index(eps, mu)
    principal = np.sqrt(eps * mu)
    s_minus = n.real < 0
    bits = n.view(np.uint64).reshape(-1, 2)
    same_bits = np.all(bits == principal.view(np.uint64).reshape(-1, 2), axis=1)
    negated = np.all(bits == (-principal).view(np.uint64).reshape(-1, 2), axis=1)
    print(f'\n  {size} random passive pairs (a quarter with lossless mu, a quarter lossless eps):')
    print(f'    s = +1: {np.sum(~s_minus)}, the principal root to the bit on '
          f'{np.sum(same_bits & ~s_minus)}; s = -1: {np.sum(s_minus)}, its exact negation on '
          f'{np.sum(negated & s_minus)}')
    print(f'    Im(n) >= 0 on {np.sum(n.imag >= 0)}; |n - sqrt(eps)*sqrt(mu)| max '
          f'{e(np.max(np.abs(n - np.sqrt(eps) * np.sqrt(mu))))}')
    er, ei, mr, mi = eps.real, eps.imag, mu.real, mu.imag
    tie = (eps * mu).imag == 0
    # The negative-phase-velocity conditions: M. W. McCall, A. Lakhtakia and W. S. Weiglhofer,
    # Eur. J. Phys. 23, 353 (2002); R. A. Depine and A. Lakhtakia, Microwave Opt. Technol. Lett.
    # 41, 315 (2004), who also give R. Ruppin's Im(eps*mu) < 0 as the same condition for a
    # passive medium -- off the axis Im(eps*mu) = 0, where a lossless double negative sits.
    conditions = {
        'Im(eps*mu) < 0': (eps * mu).imag < 0,
        "Depine-Lakhtakia: eps'|mu| + mu'|eps| < 0": er * np.abs(mu) + mr * np.abs(eps) < 0,
        "McCall-Lakhtakia-Weiglhofer: (|eps|-eps')(|mu|-mu') > eps''mu''":
            (np.abs(eps) - er) * (np.abs(mu) - mr) > ei * mi,
    }
    print(f'    against s = -1, on the {np.sum(~tie)} pairs off the axis Im(eps*mu) = 0 '
          f'({np.sum(tie)} on it):')
    for label, test in conditions.items():
        print(f'      {label:62s} agrees on {np.sum((test == s_minus)[~tie])}')
    on_axis = []
    for pair in (LOSSLESS, ASYM_LOSSLESS):
        x, y = complex(pair[0]), complex(pair[1])
        on_axis.append(f'{c_or_r(x.real)}, {c_or_r(y.real)}: s = '
                       f'{int(np.sign(signed_index(x, y).real)):+d}, ' + ', '.join(
                           f'{name} {bool(test)}' for name, test in (
                               ('Im(eps*mu) < 0', (x * y).imag < 0),
                               ('Depine-Lakhtakia', x.real * abs(y) + y.real * abs(x) < 0),
                               ('McCall-Lakhtakia-Weiglhofer',
                                (abs(x) - x.real) * (abs(y) - y.real) > x.imag * y.imag))))
    print('    on the axis, the lossless double negatives (' + '; '.join(on_axis) + ')')
    dng = (er < 0) & (mr < 0)
    propagating = (eps * mu).real > 0
    print(f'    s = -1 and double negative: {np.sum(s_minus & dng)}; s = -1 with one real part '
          f'non-negative (magnetic metals): {np.sum(s_minus & ~dng)}; double negatives with '
          f's = +1: {np.sum(dng & ~s_minus)}')
    print(f'    among media that propagate (Re(eps*mu) > 0): s = -1 exactly for the double '
          f'negatives on {np.sum((s_minus == dng)[propagating])} of {np.sum(propagating)}')
    # The class boundary: eps*mu within a few ulp of the negative real axis.
    e_b = rng.uniform(-5, 5, 400000) + 1j * rng.uniform(0.01, 2, 400000)
    m_re = rng.uniform(-5, 5, 400000)
    m_im = -e_b.imag * m_re / e_b.real
    keep = m_im > 0
    e_b, m_re, m_im = e_b[keep], m_re[keep], m_im[keep]
    m_b = m_re + 1j * (m_im + rng.integers(-3, 4, m_im.size) * np.spacing(m_im))
    r = np.sqrt(e_b * m_b)
    q = np.sqrt(e_b) * np.sqrt(m_b)
    by_re_q = np.where(q.real < 0, -r, r)
    projected = signed_index(e_b, m_b)
    print(f'\n  {e_b.size} passive pairs with eps*mu within 3 ulp of the negative real axis:')
    print(f'    Im(n) < 0 (a passive medium read as gain): sign of Re(q) alone '
          f'{np.sum(by_re_q.imag < 0)}, the projection Re(conj(r) q) {np.sum(projected.imag < 0)}')
    zeros = (signed_index(complex(-2.0, -0.0), complex(-1.0, 0.0)),
             signed_index(-(4 + 0j), 1.0 + 0j), np.sqrt(-(4 + 0j)))
    print(f'    signed zeros: (-2-0i, -1+0i) -> {c(zeros[0])}, lossless; on the axis, '
          f'(-(4+0i), 1) -> {c(zeros[1])}, the bits of sqrt(-(4+0i)) = {c(zeros[2])}: '
          f'{np.asarray(zeros[1]).tobytes() == np.asarray(zeros[2]).tobytes()}')
    # Continuity across the seams, through the solver.
    print('\n  a passive far side crossing Re(eps) = 0 (eps = +-1e-9+0.3i, mu = -1+0.1i) or '
          'Re(mu) = 0 (eps = -2+0.1i, mu = +-1e-9+0.4i), conical grating, SMM:')
    for which, pair in (('Re(eps)', lambda x: (x + 0.3j, -1 + 0.1j)),
                        ('Re(mu)', lambda x: (-2 + 0.1j, x + 0.4j))):
        rows, n_bot = [], []
        with _quiet():
            for x in (-1e-9, 1e-9):
                s = grating('conical', 'SMM', far=pair(x))
                result = s.solve(auto_detune=False)
                n_bot.append(complex(s.n_bot))
                rows.append(amplitudes(result, 'conical'))
        both_negative = [np.sqrt(complex(pair(x)[0] * pair(x)[1])) *
                         (-1 if pair(x)[0].real < 0 and pair(x)[1].real < 0 else 1)
                         for x in (-1e-9, 1e-9)]
        print(f'    {which} -1e-9 -> +1e-9: n_bot {n_bot[0].real:+.4f}{n_bot[0].imag:+.4f}i, '
              f'step {e(abs(n_bot[1] - n_bot[0]))}; every R, T, r, t: step '
              f'{e(np.max(np.abs(rows[1] - rows[0])))}; a sign keyed on both real parts '
              f'negative would step by {abs(both_negative[1] - both_negative[0]):.4f}')
    with _quiet():
        result = solver(bot=wavebit.Material(*MAGNETIC_METAL)).solve()
    t_s, t_p = complex(np.sum(result.s_inc.t_s)), complex(np.sum(result.p_inc.t_p))
    kz2 = physical(*MAGNETIC_METAL, 0.0)
    phase = np.exp(2j * np.pi / WL * N_GLASS * D_LAYER)
    principal = np.sqrt(complex(MAGNETIC_METAL[0] * MAGNETIC_METAL[1]))
    t_H = fresnel(1.5, 2.25, 1.0, kz2, *MAGNETIC_METAL)['p'][1]
    t_p_principal = t_H * N_GLASS / (principal / MAGNETIC_METAL[1]) * phase
    n2 = signed_index(*MAGNETIC_METAL)
    bare = fresnel(1.5, 2.25, 1.0, kz2, *MAGNETIC_METAL)
    print(f'\n  glass onto the magnetic metal (-2+0.01i, 0.1+0.5i), normal incidence, closed form: '
          f't_s = {c(bare["s"][1])}, t_p = t_H (n1/mu1)/(n2/mu2) = '
          f'{c(bare["p"][1] * N_GLASS / (n2 / MAGNETIC_METAL[1]))} with the signed n2; and the '
          f'solver, through {D_LAYER:.0f} nm of glass:')
    print(f'    t_s = {t_s.real:.4f}{t_s.imag:+.4f}i, t_p = {t_p.real:.4f}{t_p.imag:+.4f}i '
          f'(|t_p - t_s| {e(abs(t_p - t_s))}); with the principal index t_p would be '
          f'{t_p_principal.real:+.4f}{t_p_principal.imag:+.4f}i = -t_s')
    # Each factor's root is Maystre's (Re + Im >= 0): A. A. Govyadinov, V. A. Podolskiy and
    # M. A. Noginov, Appl. Phys. Lett. 91, 191103 (2007), take n = sqrt(eps)*sqrt(mu) with each
    # factor cut along the negative imaginary axis. It differs from the principal root only for
    # an active factor with Re < 0. At normal incidence the half-space takes kz = +-n, so the
    # question is whether n points along the kz it is solved with (t_p = t_s there iff it does).
    boundary_kz = _study_module('wavebit._common.conventions', 'wavebit._core.conventions').boundary_kz
    size = 400000
    eps = rng.uniform(-3, 3, size) + 1j * rng.uniform(-1, 1, size)
    mu = rng.uniform(-3, 3, size) + 1j * rng.uniform(-1, 1, size)
    r = np.sqrt(eps * mu)
    kz = boundary_kz(np, eps * mu, np.zeros(size), medium=(eps, mu))
    principal_factors = np.where(
        ((np.conj(r) * np.sqrt(eps + 0j) * np.sqrt(mu + 0j)).real < 0) & (r.real != 0), -r, r)
    n = signed_index(eps, mu)
    gain_metal = ((eps.real < 0) & (eps.imag < 0)) | ((mu.real < 0) & (mu.imag < 0))
    active = (eps.imag < 0) | (mu.imag < 0)
    dng = (eps.real < 0) & (mu.real < 0)
    propagating = (eps * mu).real > 0
    print(f'\n  {size} random media, Re in (-3, 3) and Im in (-1, 1) for eps and mu: does n point '
          'along the normal-incidence kz the half-space takes (Re(conj(n) kz) > 0)?')
    print('    class                                                   count   principal '
          'factors   Maystre factors (now)')
    for label, mask in (
            ('passive', ~active),
            ('active, no gain-metal factor', active & ~gain_metal),
            ('active, a gain-metal factor, not DNG, evanescent', gain_metal & ~dng & ~propagating),
            ('active, a gain-metal factor, not DNG, propagating', gain_metal & ~dng & propagating),
            ('active DNG, evanescent (accepted)', active & dng & ~propagating),
            ('active DNG, propagating (refused)', active & dng & propagating)):
        agree = [int(np.sum(((np.conj(v) * kz).real > 0)[mask])) for v in (principal_factors, n)]
        share = [100 * a / mask.sum() for a in agree]
        print(f'    {label:55s} {int(mask.sum()):7d}   {agree[0]:7d} ({share[0]:5.1f}%)'
              f'   {agree[1]:7d} ({share[1]:5.1f}%)')
    same = np.all(n.view(np.uint64).reshape(-1, 2) == principal_factors.view(np.uint64).reshape(
        -1, 2), axis=1)
    one = ((eps.real < 0) & (eps.imag < 0)) ^ ((mu.real < 0) & (mu.imag < 0))
    print(f'    the two rules agree to the bit on {int(same.sum())} of {size}; they differ on '
          f'{int((~same).sum())}, by an exact negation on '
          f'{int(np.sum(n[~same] == -principal_factors[~same]))}, and those are the media with '
          f'exactly one gain-metal factor (Re < 0, Im < 0): {bool(np.array_equal(~same, one))}')
    # A factor exactly on the negative imaginary axis is Maystre's tie: both its roots have
    # Re + Im = 0 and the principal one is kept, decided on the factor's real part. Read off the
    # computed Re + Im of the root, as the rule was first written, the tie was round-off.
    axis = np.asarray([complex(0.0, -x) for x in np.round(np.arange(1, 301) * 0.01, 2)])
    minus3 = np.full_like(axis, -3 + 0j)
    kz_axis = boundary_kz(np, axis * minus3, np.zeros(axis.size), medium=(axis, minus3))

    def by_sum(root):
        return np.where(root.real + root.imag < 0, -root, root)

    read_sum = by_sum(np.sqrt(axis + 0j)) * by_sum(np.sqrt(minus3 + 0j))
    read_sum_python = np.asarray([
        (-a if a.real + a.imag < 0 else a) * (-b if b.real + b.imag < 0 else b)
        for a, b in ((cmath.sqrt(x + 0j), cmath.sqrt(y + 0j)) for x, y in zip(axis, minus3))])
    now = signed_index(axis, minus3)
    now_python = np.asarray([signed_index(complex(x), complex(y)) for x, y in zip(axis, minus3)])

    def against(v):
        return int(np.sum((np.conj(v) * kz_axis).real < 0))

    print(f'  a factor on Maystre\'s cut itself, eps = -0.01i .. -3.00i ({axis.size}) with mu = -3 '
          f'(kz**2 = 3bi, no tie in kz): n against kz for {against(now)} on arrays and '
          f'{against(now_python)} on Python numbers; with each factor\'s sign read off the '
          f'computed Re + Im of its root, {against(read_sum)} and {against(read_sum_python)}')


# ------------------------------------------------------------------------------- 4. theta

def section_theta():
    header(4, 'theta -- a real angle names the direction the light travels')
    # Snell's law with the signed index, n1 sin(theta1) = n2 sin(theta2), angles of the light:
    # V. G. Veselago, Sov. Phys. Usp. 10, 509 (1968); J. B. Pendry, Phys. Rev. Lett. 85, 3966
    # (2000).
    theta = np.deg2rad(20.0)
    with _quiet():
        s = _legacy_wavebit.rcwa(top=wavebit.Material(*LOSSLESS), bot=2.25, theta=theta, phi=0.0,
                         wavelength=WL, period=[500.0], truncation_order=0, thickness=[0.0],
                         structure=np.full((1, 1, 1), 2.25), polarization=(1.0, 0.0))
        result = s.solve()
    kx = s.order_kxy()[0][0].real
    print(f'  lit from (-2, -1), n = {complex(s.n_top).real:.6f}, at 20 deg onto glass: kx = '
          f'{kx:.6f} = -sqrt(2) sin(20 deg); transmitted angle arcsin(kx/1.5) = '
          f'{np.degrees(np.arcsin(kx / 1.5)):.4f} deg; R = {float(np.sum(result.R)):.7f}, '
          f'T = {float(np.sum(result.T)):.7f}')
    expected = np.degrees(np.arcsin(-np.sqrt(2) * np.sin(theta) / N_GLASS))
    worst = 0.0
    with _quiet():
        for side in ('top', 'bottom'):
            far = 'bottom' if side == 'top' else 'top'
            scene = {('top' if side == 'top' else 'bot'): wavebit.Material(*LOSSLESS)}
            field = _host(solver(theta=theta, phi=0.6, side=side, answers='field', **scene)
                          .solve().boundary_fields(far, [0.0], [0.0], part='outgoing'))[0, 0]
            S = poynting(field)
            along = S[0] * np.cos(0.6) + S[1] * np.sin(0.6)
            worst = max(worst, abs(np.degrees(np.arctan2(along, abs(S[2]))) - expected))
    print(f'  the transmitted field\'s Poynting vector, 2D path at phi = 0.6, lit from either side:'
          f' {expected:.4f} deg to {e(worst)} deg')
    # Lossless lit media: theta and phi are the incident power's direction.
    print('\n  lossless lit media: the incident Poynting vector against (theta, phi = 0.6), '
          'theta 0.1 .. 1.3 rad, psi 0 / pi/2 / 0.7, both sides, 2D / conical / first-order:')
    for label, pair in (('(-3, -0.5), n = -sqrt(1.5)', ASYM_LOSSLESS),
                        ('(-2, -1), n = -sqrt(2)', LOSSLESS), ('(2.25, 1.3), n = +1.71', (2.25,
                                                                                         1.3))):
        polar_err = azimuth_err = kx_err = 0.0
        n = np.sqrt(complex(pair[0])) * np.sqrt(complex(pair[1]))
        with _quiet():
            for path in ('2d', 'conical', 'first_order'):
                for side in ('top', 'bottom'):
                    inward = np.array([0.0, 0.0, -1.0 if side == 'top' else 1.0])
                    scene = dict(top=wavebit.Material(*pair)) if side == 'top' else dict(
                        bot=wavebit.Material(*pair))
                    for th in (0.1, 0.4, 0.7, 1.0, 1.3):
                        for psi in (0.0, np.pi / 2, 0.7):
                            s = solver(theta=th, phi=0.6, side=side, path=path, answers='field',
                                       psi=psi, **scene)
                            result = s.solve()
                            kxy = np.array([complex(np.ravel(_host(k))[0])
                                            for k in s.order_kxy()])
                            kx_err = max(kx_err, np.max(np.abs(
                                kxy - n.real * np.sin(th) * np.array([np.cos(0.6),
                                                                      np.sin(0.6)]))))
                            field = _host(result.boundary_fields(side, [0.0], [0.0],
                                                                 part='incoming'))[0, 0]
                            S = poynting(field)
                            polar = np.arccos(S @ inward / np.linalg.norm(S))
                            polar_err = max(polar_err, abs(np.degrees(polar - th)))
                            azimuth_err = max(azimuth_err, abs(np.arctan2(S[1], S[0]) - 0.6))
        print(f'    {label:28s}: polar angle to {e(polar_err)} deg, azimuth to '
              f'{e(azimuth_err)} rad; (kx, ky) = n sin(theta) (cos phi, sin phi), signed n, '
              f'to {e(kx_err)}')
    # Absorbing lit media: the angle is that of s*Re(k), and the power departs from it.
    print('\n  absorbing lit media at theta = 45.8 deg, phi = 0.6: the incident Poynting '
          'vector\'s polar angle, signed in the plane of incidence (negative: across the normal)')
    for label, eps, mu in (('2+0.5i (lossy dielectric)', 2 + 0.5j, None),
                           ('-10+1i (metal)', -10 + 1j, None),
                           ('(-3+0.1i, -0.5+0.05i) (lossy DNG)', *ASYM),
                           ('(-1+1i, -1+0.5i) (absorbing DNG)', -1 + 1j, -1 + 0.5j),
                           ('(-1+0.2i, 0.5+2i) (magnetic metal)', -1 + 0.2j, 0.5 + 2j)):
        top = eps if mu is None else wavebit.Material(eps, mu=mu)
        cells = []
        with _quiet():
            for psi in (0.0, np.pi / 2):
                s = solver(top=top, theta=np.deg2rad(45.8), phi=0.6, answers='field', psi=psi)
                field = _host(s.solve().boundary_fields('top', [0.0], [0.0],
                                                        part='incoming'))[0, 0]
                S = poynting(field)
                along = S[0] * np.cos(0.6) + S[1] * np.sin(0.6)
                cells.append(np.degrees(np.arctan2(along, -S[2])))
        n = np.sqrt(complex(eps)) * (1 if mu is None else np.sqrt(complex(mu)))
        print(f'    {label:36s} s = {int(np.sign(n.real)):+d}: p {cells[0]:7.2f}, '
              f's {cells[1]:7.2f}')
    # A worked absorbing lit example: the lossy double negative itself, one interface.
    theta = np.deg2rad(20.0)
    N = phase_index(*PASSIVE, theta)
    n = np.sqrt(PASSIVE[0]) * np.sqrt(PASSIVE[1])
    rows = []
    with _quiet():
        for label, pol in (('s', (1.0, 0.0)), ('p', (0.0, 1.0))):
            s = _legacy_wavebit.rcwa(top=wavebit.Material(*PASSIVE), bot=2.25, theta=theta, phi=0.0,
                             wavelength=WL, period=[500.0], truncation_order=0, thickness=[0.0],
                             structure=np.full((1, 1, 1), 2.25), polarization=pol,
                             answers='field')
            result = s.solve()
            field = _host(result.boundary_fields('top', [0.0], [0.0], part='incoming'))[0, 0]
            S = poynting(field)
            rows.append((label, float(np.sum(result.R)), float(np.sum(result.T)),
                         float(np.sum(result.power_entering)),
                         np.degrees(np.arctan2(S[0], -S[2]))))
    kx = complex(np.ravel(_host(s.order_kxy()[0]))[0])
    # Lit from the top the incident wave travels toward -z: its public k_z is -kz, kz the
    # half-space's own (mirror) root at that k_par.
    k_z = -complex(mirror(PASSIVE[0] * PASSIVE[1] - kx * kx))
    print(f'\n  lit from the lossy (-2+0.1i, -1+0.1i) at 20 deg, phi = 0, onto glass (one '
          f'interface): N = {N:.6f} (|n| = {abs(n):.6f}), kx = -N sin(20 deg) = {kx.real:.6f}, '
          f'the incident wave\'s public k_z = {c(k_z, 6)} (phase up, away from the structure; '
          f'decaying toward it); transmitted angle arcsin(kx/1.5) = '
          f'{np.degrees(np.arcsin(kx.real / N_GLASS)):.3f} deg')
    for label, R, T, entering, angle in rows:
        print(f'    {label}: R = {R:.10f}, T = {T:.10f}, R + T = {R + T:.10f}, power_entering = '
              f'{entering:.10f}; the incident power travels at {angle:.3f} deg')
    # The s/p basis follows k_par, which a negative-index side turns to phi + pi.
    print('\n  the s/p basis from a negative-index side: lit at 20 deg, phi = 0, the same '
          'polarization from (-2, -1) and from its double-positive twin (2, 1):')
    with _quiet():
        for pol in ((1.0, 0.0), (0.0, 1.0)):
            got = []
            for pair in (LOSSLESS, (2.0 + 0j, 1.0 + 0j)):
                s = _legacy_wavebit.rcwa(top=wavebit.Material(*pair), bot=2.25, theta=theta, phi=0.0,
                                 wavelength=WL, period=[500.0], truncation_order=0,
                                 thickness=[0.0], structure=np.full((1, 1, 1), 2.25),
                                 polarization=pol, answers='field')
                field = _host(s.solve().boundary_fields('top', [0.0], [0.0],
                                                        part='incoming'))[0, 0]
                got.append((field[:3].real, poynting(field)))
            print(f'    polarization ({pol[0]:g}, {pol[1]:g}): E_inc = '
                  f'({", ".join(f"{round(v, 4) + 0.0:+.4f}" for v in got[0][0])}) from (-2, -1), '
                  f'({", ".join(f"{round(v, 4) + 0.0:+.4f}" for v in got[1][0])}) from (2, 1); '
                  f'their Poynting vectors agree to {e(np.max(np.abs(got[0][1] - got[1][1])))}')
    # Normal incidence: the frame at theta = 0 is the limit theta -> 0+.
    print('\n  normal incidence, a grating lit from (-3+0.1i, -0.5+0.05i) at phi = 0.6: '
          'theta = 0 against 1e-9, every amplitude, r_xyz / t_xyz, the field and the incident '
          'field, 5 paths x 2 sides x ETM / SMM:')
    worst, unturned = 0.0, np.inf
    with _quiet():
        for path in PATHS:
            for side in ('top', 'bottom'):
                for algo in ('ETM', 'SMM'):
                    rows = []
                    for th, ph in ((0.0, 0.6), (1e-9, 0.6), (0.0, 0.6 + np.pi)):
                        result = grating(path, algo, lit=ASYM, side=side, theta=th, phi=ph,
                                         answers='field').solve(auto_detune=False)
                        ys = [0.0, 77.0] if path in ('2d', 'first_order') else [0.0]
                        rows.append(np.concatenate(
                            [amplitudes(result, path, ('r_s', 'r_p', 't_s', 't_p', 'r_xyz',
                                                       't_xyz')),
                             np.ravel(_host(result.reconstruct_field([0.0, 123.0], ys,
                                                                     [10.0, 90.0]))),
                             np.ravel(_host(result.boundary_fields(side, [0.0, 50.0], [0.0],
                                                                   part='incoming')))]))
                    worst = max(worst, np.max(np.abs(rows[1] - rows[0])))
                    if path not in ('te', 'tm'):
                        unturned = min(unturned, np.max(np.abs(rows[1] - rows[2])))
    print(f'    max change {e(worst)}; with the frame left at phi (solved as theta = 0 at '
          f'phi + pi, whose frame is then phi), conical / 2D / first-order: at least '
          f'{unturned:.3f}')
    # Complex angles.
    print('\n  a complex angle, theta = 0.35 + 1e-9i against 0.35, lit from the double negative '
          '(te, tm, conical; grating; SMM):')
    lossless_worst = lossy_kx = 0.0
    signs = []
    with _quiet():
        for path in ('te', 'tm', 'conical'):
            for pair in (ASYM_LOSSLESS, ASYM):
                out = []
                for th in (0.35, 0.35 + 1e-9j):
                    s = grating(path, lit=pair, theta=th)
                    result = s.solve(auto_detune=False)
                    out.append((np.ravel(_host(s.order_kxy()[0])),
                                amplitudes(result, path, ('R', 'r_s', 'r_p', 't_s', 't_p'))))
                if pair is ASYM_LOSSLESS:
                    lossless_worst = max(lossless_worst, np.max(np.abs(out[1][0] - out[0][0])),
                                         np.max(np.abs(out[1][1] - out[0][1])))
                else:
                    lossy_kx = max(lossy_kx, np.max(np.abs(out[1][0] - out[0][0])))
                    signs.append((out[0][0][3].real, out[1][0][3].real))
    print(f'    lossless: kx and every amplitude move by {e(lossless_worst)}; lossy: kx moves by '
          f'{lossy_kx:.3f} (N against the complex n), zeroth-order kx {signs[0][0]:+.4f} real '
          f'angle, {signs[0][1]:+.4f} complex: the same side of the normal')
    # The s = +-1 seam of a lit absorbing medium, and theta through zero in glass.
    print('\n  the s = +-1 seam: lit from eps = -2+0.1i, mu = 1 + (0.05 +- 1e-9)i (Im(eps*mu) '
          'changes sign), theta = 0.3, phi = 0.6, psi = 0.3, conical grating:')
    rows = []
    with _quiet():
        for d in (-1e-9, 1e-9):
            pair = (-2 + 0.1j, 1 + (0.05 + d) * 1j)
            s = grating('conical', lit=pair, theta=0.3, phi=0.6, answers='field')
            result = s.solve(auto_detune=False)
            field = _host(result.boundary_fields('top', [0.0], [0.0], part='incoming'))[0, 0]
            rows.append((complex(s.n_top), complex(np.ravel(_host(s.order_kxy()[0]))[3]),
                         amplitudes(result, 'conical', ('R', 'T')), field[:3]))
    for (n, kx, _, E), d in zip(rows, ('-', '+')):
        print(f'    Im(mu) = 0.05 {d} 1e-9: n_top = {n.real:+.2e}{n.imag:+.4f}i, s = '
              f'{int(np.sign(n.real)):+d}, kx = {kx.real:+.2e}, E_inc = '
              f'({c(E[0], 3)}, {c(E[1], 3)}, {c(E[2], 3)})')
    step = np.abs(rows[1][2] - rows[0][2]).reshape(6, -1)
    print(f'    every R: step {e(step[0::2].max())}; every T: step {e(step[1::2].max())}, '
          f'relative {e(step[1::2].max() / np.abs(rows[0][2]).reshape(6, -1)[1::2].max())} (T '
          f'is a ratio to an incident power that vanishes on the seam: here up to '
          f'{np.abs(rows[0][2]).reshape(6, -1)[1::2].max():.1f})')
    rows = []
    with _quiet():
        for th in (-1e-9, 1e-9):
            result = grating('conical', lit=(2.25, 1.0), theta=th, phi=0.6,
                             answers='field').solve(auto_detune=False)
            field = _host(result.boundary_fields('top', [0.0], [0.0], part='incoming'))[0, 0]
            rows.append((amplitudes(result, 'conical', ('R', 'T')), field[:3].real))
    print(f'    glass at theta = -1e-9 and +1e-9 (phi = 0.6): E_inc = '
          f'({", ".join(f"{round(v, 3) + 0.0:+.3f}" for v in rows[0][1])}) and '
          f'({", ".join(f"{round(v, 3) + 0.0:+.3f}" for v in rows[1][1])}); every R and T: '
          f'step {e(np.max(np.abs(rows[1][0] - rows[0][0])))}')
    # Direct kx / ky are taken as given.
    with _quiet():
        s = grating('2d', lit=ASYM, theta=0.35, phi=0.6)
        by_angle = amplitudes(s.solve(auto_detune=False), '2d')
        kx, ky = (complex(_host(k)[len(k) // 2]) for k in s.order_kxy())
        k0 = 2 * np.pi / WL
        d = _legacy_wavebit.rcwa(top=wavebit.Material(*ASYM), bot=N_GLASS ** 2, wavelength=WL,
                         layer_algorithm='SMM', kx=k0 * kx.real, ky=k0 * ky.real, psi=0.3,
                         period=(P_G, 0.9 * P_G), truncation_order=(1, 1), thickness=[H_G],
                         structure=G2[None])
        by_k = amplitudes(d.solve(auto_detune=False), '2d')
    print(f'\n  direct kx, ky set to the theta door\'s (kx, ky) = ({kx.real:+.6f}, {ky.real:+.6f})'
          f': every amplitude to {e(np.max(np.abs(by_k - by_angle)))}')
    # At normal incidence the two doors take different frames from a negative-index side.
    print('  normal incidence from (-2, -1), polarization (1, 0): the theta door (theta = 0, '
          'phi = 0) against the direct-k door (kx = 0):')
    cells = []
    with _quiet():
        for kw in (dict(theta=0.0, phi=0.0), dict(kx=0.0)):
            s = _legacy_wavebit.rcwa(top=wavebit.Material(*LOSSLESS), bot=2.25, wavelength=WL,
                             period=[500.0], truncation_order=0, thickness=[0.0],
                             structure=np.full((1, 1, 1), 2.25), polarization=(1.0, 0.0),
                             answers='field', **kw)
            result = s.solve()
            field = _host(result.boundary_fields('top', [0.0], [0.0], part='incoming'))[0, 0]
            cells.append((field[:3].real, complex(np.ravel(_host(result.r_s))[0])))
    for label, (E, r_s) in zip(('theta door', 'kx door   '), cells):
        print(f'    {label}: E_inc = ({", ".join(f"{round(v, 4) + 0.0:+.4f}" for v in E)}), '
              f'r_s = {c(r_s, 6)}')


# --------------------------------------------------------------------------- 5. amplitudes

def section_amplitudes():
    order_grid_kxy = _study_module('wavebit.rcwa._engine.solve_2d', 'wavebit._core.solve_2d').order_grid_kxy
    header(5, 'amplitudes -- polarization, the incident wave, the scattering matrix')
    worst_r = worst_t = 0.0
    with _quiet():
        for pair in (PASSIVE, LOSSLESS, ASYM, MAGNETIC_METAL):
            for path in ('conical', '2d', 'first_order'):
                for algo in ('ETM', 'SMM'):
                    for lit in (False, True):
                        scene = dict(top=wavebit.Material(*pair)) if lit else dict(
                            bot=wavebit.Material(*pair))
                        if lit and pair is MAGNETIC_METAL:
                            continue
                        result = solver(path=path, algo=algo, **scene).solve()
                        r_s, t_s = (complex(np.sum(_host(result.s_inc.r_s))),
                                    complex(np.sum(_host(result.s_inc.t_s))))
                        r_p, t_p = (complex(np.sum(_host(result.p_inc.r_p))),
                                    complex(np.sum(_host(result.p_inc.t_p))))
                        worst_r = max(worst_r, abs(r_p + r_s))
                        worst_t = max(worst_t, abs(t_p - t_s))
    print(f'  normal incidence, one interface, lit and far double negatives and the magnetic '
          f'metal (far), conical / 2D / first-order, ETM / SMM: |r_p + r_s| {e(worst_r)}, '
          f'|t_p - t_s| {e(worst_t)}')
    with _quiet():
        for label, pair in (('(-3, -0.5)', ASYM_LOSSLESS), ('(-3+0.1i, -0.5+0.05i)', ASYM)):
            s = solver(top=wavebit.Material(*pair), theta=0.4, phi=0.6, psi=0.7,
                       answers='field')
            field = _host(s.solve().boundary_fields('top', [0.0], [0.0], part='incoming'))[0, 0]
            kx, ky = (complex(np.ravel(_host(k))[0]) for k in s.order_kxy())
            kz = complex(mirror(pair[0] * pair[1] - kx * kx - ky * ky))
            # Lit from the top the wave travels toward -z; its wavevector's z component is
            # -kz, kz the half-space's own root. Phase and power then point opposite ways.
            k = np.array([kx, ky, -kz])
            H = np.cross(k, field[:3]) / pair[1]
            S = poynting(field)
            print(f'  lit from {label} at theta 0.4, phi 0.6, psi 0.7: the incident wave has '
                  f'Re(k_z) = {k[2].real:+.4f} (phase up, away from the structure), '
                  f'Im(k_z) = {k[2].imag:+.4f}, S_z = {S[2]:+.4f} (power down); '
                  f'|H - k x E / mu| = {e(np.max(np.abs(field[3:] - H)))}, '
                  f'|k . E| = {e(abs(k @ field[:3]))}')
    # The beam of each diffraction order in a negative-index half-space: its power, read from the
    # field the solve returns, one Fourier component per order over a period.
    print('  the beam of each order in the lossless (-2, -1), n = -sqrt(2): a TE grating '
          f'({P_G:.0f} nm, glass on the other side, theta = 10 deg), the outgoing field sampled '
          'over one period, each order\'s Poynting vector, its angle from the outward normal '
          '(positive toward +x) against arcsin(kx_m / n) with the signed n:')
    n = -np.sqrt(2.0)
    samples = 64
    xs = np.arange(samples) * P_G / samples
    k0 = 2 * np.pi / WL
    for role, face in (('far side, transmitted', 'bottom'), ('lit side, reflected', 'top')):
        scene = (dict(top=N_GLASS ** 2, bot=wavebit.Material(*LOSSLESS)) if face == 'bottom'
                 else dict(top=wavebit.Material(*LOSSLESS), bot=N_GLASS ** 2))
        with _quiet():
            s = _legacy_wavebit.rcwa(**scene, wavelength=WL, period=[P_G], tetm='te',
                             truncation_order=3, structure=G1[None, None, :], thickness=[H_G],
                             theta=np.deg2rad(10.0), answers='field')
            field = _host(s.solve(auto_detune=False).boundary_fields(face, xs, [0.0],
                                                                      part='outgoing'))[0]
        kx = np.ravel(_host(s.order_kxy()[0])).real
        centre = len(kx) // 2
        coefficients = np.fft.fft(field * np.exp(-1j * k0 * kx[centre] * xs)[:, None],
                                  axis=0) / samples
        outward = -1.0 if face == 'bottom' else 1.0
        cells, worst, leaving = [], 0.0, 0
        for index, k in enumerate(kx):
            if abs(k) >= abs(n):
                continue
            component = coefficients[(index - centre) % samples]
            S = 0.5 * np.real(np.cross(component[:3], np.conj(component[3:])))
            angle = np.degrees(np.arctan2(S[0], outward * S[2]))
            want = np.degrees(np.arcsin(k / n))
            worst = max(worst, abs(angle - want))
            leaving += int(outward * S[2] > 0)
            cells.append(f'm = {index - centre:+d}: kx {k:+.4f}, beam {angle:+.2f} deg')
        print(f'    {role}: ' + '; '.join(cells) + f' (each to {e(worst)} deg of '
              f'arcsin(kx_m / n); {leaving} of {len(cells)} carry their power away from the '
              'interface)')
    # Unitarity: lossless ports and a lossless grating.
    print('  unitarity, glass grating lit from the lossless (-3, -0.5) onto glass, SMM, the '
          'public scattering matrix in flux units on the propagating orders:')
    for path in ('conical', '2d'):
        with _quiet():
            s = grating(path, lit=ASYM_LOSSLESS, theta=0.35, phi=0.6)
            s.solve(auto_detune=False)
            S = s.scattering
        gx, gy = order_grid_kxy(np, *[_host(a) for a in s.order_kxy()])
        k_par = np.sqrt((gx * gx + gy * gy).ravel() + 0j)

        def weights(eps, mu):
            kz = np.array([physical(eps, mu, k) for k in k_par])
            return np.concatenate([(kz / mu).real, (kz / eps).real])

        w_in, w_out = weights(*ASYM_LOSSLESS), weights(N_GLASS ** 2, 1.0)
        open_in, open_out = w_in > 1e-12, w_out > 1e-12
        full = np.block([[_host(S.S11), _host(S.S12)], [_host(S.S21), _host(S.S22)]])
        w = np.concatenate([w_in, w_out])
        keep = np.concatenate([open_in, open_out])
        root = np.sqrt(np.abs(w[keep]))
        U = root[:, None] * full[np.ix_(keep, keep)] / root[None, :]
        print(f'    {path}: {int(open_in.sum())} + {int(open_out.sum())} open channels, '
              f'|U^H U - I| = {e(np.max(np.abs(U.conj().T @ U - np.eye(U.shape[0]))))}')


# ------------------------------------------------------------------------------- 6. layers

def section_layers():
    engine = _study_module('wavebit.rcwa._engine.engine', 'wavebit._core.engine')
    DoubleNegativeLayer = _study_module('wavebit._common.conventions', 'wavebit._core.conventions').DoubleNegativeLayer
    header(6, 'layers -- which of a mode\'s +-q is "down"')
    original = engine.CoreFactorizationMixin._layer_double_negative

    def switched_off(self, like=None):
        said = original(self, like)
        return None if said is None else tuple(DoubleNegativeLayer(False, False) for _ in said)

    pairs = {'lossless': LOSSLESS, 'lossy': PASSIVE, 'asym': (-6.0 + 0j, -1.5 + 0j),
             'asym_lossy': ASYM}

    def raster(path, value):
        shape = (1, 1, 8) if path in ('te', 'tm', 'conical') else (1, 4, 4)
        return np.full(shape, value, dtype=complex)

    def build(path, algo, eps_r, mu_r, theta, bot, top=1.0, thickness=150.0, period=1000.0,
              wavelength=WL, order=5, phi=0.6):
        common = dict(top=top, bot=bot, theta=theta, wavelength=wavelength,
                      thickness=[thickness], layer_algorithm=algo, answers='amplitude',
                      structure=eps_r, mu=mu_r)
        if path in ('te', 'tm'):
            return _legacy_wavebit.rcwa(**common, tetm=path, period=[period], truncation_order=order)
        if path == 'conical':
            return _legacy_wavebit.rcwa(**common, phi=phi, psi=0.3, period=[period],
                                truncation_order=order)
        s = _legacy_wavebit.rcwa(**common, phi=phi, psi=0.3, period=(period, 0.9 * period),
                         truncation_order=(1, 1))
        if path == 'first_order':
            s.formulation = 'first_order'
        return s

    def compare(scenes, fields=('r_s', 'r_p', 't_s', 't_p')):
        out = {}
        for label, patch in (('on', None), ('off', switched_off)):
            if patch is not None:
                engine.CoreFactorizationMixin._layer_double_negative = patch
            failures = {'ETM': 0, 'SMM': 0}
            worst = 0.0
            try:
                for scene in scenes:
                    got = {}
                    for algo in ('ETM', 'SMM'):
                        try:
                            with _quiet():
                                got[algo] = amplitudes(build(*scene(algo)).solve(
                                    auto_detune=False), scene(algo)[0], fields)
                        except np.linalg.LinAlgError:
                            failures[algo] += 1
                    if len(got) == 2:
                        worst = max(worst, np.max(np.abs(got['ETM'] - got['SMM'])))
            finally:
                engine.CoreFactorizationMixin._layer_double_negative = original
            out[label] = (failures, worst)
        return out

    def report(out):
        for label, (failures, worst) in out.items():
            print(f'    label rules {label:3s}: LinAlgError ETM {failures["ETM"]}, SMM '
                  f'{failures["SMM"]}; |ETM - SMM| where both solved {e(worst)}')

    uniform = []
    for name, (eps, mu) in pairs.items():
        for path, theta in (('te', 0.0), ('te', 0.3), ('tm', 0.0), ('tm', 0.3),
                            ('conical', 0.3), ('2d', np.deg2rad(17)), ('first_order', 0.3)):
            for top in (1.0, 2.25):
                uniform.append(lambda algo, p=path, t=theta, e_=eps, m=mu, tp=top: (
                    p, algo, raster(p, e_), raster(p, m), t, wavebit.Material(e_, mu=m), tp))
    print(f'  {len(uniform)} scenes: a uniform double-negative layer, 150 nm, on a half-space of '
          f'its own medium -- no interface at all (4 media: (-2, -1), (-2+0.1i, -1+0.1i), '
          f'(-6, -1.5), (-3+0.1i, -0.5+0.05i); 7 path/angle pairs; vacuum or glass above):')
    report(compare(uniform))
    block = (-25 + 0j, -1 + 0j)
    pattern = (1, block, 1, block, block, block)
    eps_b = np.ones((1, 1, 60), complex)
    mu_b = np.ones((1, 1, 60), complex)
    for i, value in enumerate(pattern):
        if value != 1:
            eps_b[..., 10 * i:10 * (i + 1)], mu_b[..., 10 * i:10 * (i + 1)] = value
    print('  connecting-algorithms.md section 10\'s grating with its block made (-25, -1): '
          '[1, X, 1, X, X, X] over 2000 nm, 100 nm, on a half-space of X, truncation 30, 900 nm:')
    for path in ('te', 'tm', 'conical'):
        for theta in (0.5, 0.5 + 0.3j):
            scene = [lambda algo, p=path, t=theta: (p, algo, eps_b, mu_b, t,
                                                    wavebit.Material(*block), 1.0, 100.0, 2000.0,
                                                    900.0, 30, 0.3)]
            out = compare(scene)
            print(f'    {path:7s} theta {str(theta):10s} |ETM - SMM|: rules on '
                  f'{e(out["on"][1])}, off {e(out["off"][1])}')
    rng = np.random.default_rng(17)
    patterned = []
    backgrounds = [(-6.0 + 0j, -1.5 + 0j), PASSIVE, ASYM, (-2.5 + 0j, -0.8 + 0j)]
    for i in range(96):
        path = ('te', 'tm', 'conical', '2d')[i % 4]
        eps_g, mu_g = backgrounds[(i // 4) % 4]
        shape = (1, 1, 8) if path in ('te', 'tm', 'conical') else (1, 4, 4)
        mask = rng.uniform(size=shape) < rng.uniform(0.1, 0.9)
        inc = (2.25, 1.0) if i % 3 else (-10 + 1j, 1.0)
        eps_r = np.where(mask, eps_g, inc[0]).astype(complex)
        mu_r = np.where(mask, mu_g, inc[1]).astype(complex)
        if i % 8 >= 4:                              # the reverse: ordinary background
            eps_r, mu_r = (np.where(mask, inc[0], eps_g).astype(complex),
                           np.where(mask, inc[1], mu_g).astype(complex))
        bot = wavebit.Material(eps_g, mu=mu_g) if i % 2 else 2.25
        theta = (0.3, 0.5 + 0.2j, 0.7)[(i // 16) % 3]
        patterned.append(lambda algo, p=path, er=eps_r, mr=mu_r, t=theta, b=bot: (
            p, algo, er, mr, t, b, 1.0, 200.0))
    print(f'  {len(patterned)} random patterned layers, 200 nm (double-negative regions in '
          'glass or '
          f'a metal, and the reverse; on the double negative or glass; theta 0.3, 0.5+0.2i, 0.7):')
    report(compare(patterned))
    mask = np.array([[1, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0]], dtype=bool)[None]
    eps_r = np.where(mask, LOSSLESS[0], 2.25).astype(complex)
    mu_r = np.where(mask, LOSSLESS[1], 1.0).astype(complex)
    cells = []
    for factorization in ('li', 'laurent', 'nvf'):
        try:
            with _quiet():
                s = build('2d', 'SMM', eps_r, mu_r, 0.3, wavebit.Material(*LOSSLESS))
                s.factorization = factorization
                s.solve(auto_detune=False)
            cells.append(f'{factorization} solves it')
        except np.linalg.LinAlgError:
            cells.append(f'{factorization} LinAlgError')
    print('  a lossless (-2, -1) layer patterned with glass, 2D, a column of mu half +1 and '
          'half -1:'
          ' ' + ', '.join(cells) + ' (Li inverts the Toeplitz matrix of 1/mu, here exactly '
          'singular)')


# ----------------------------------------------------------------------------- 7. refusals

def section_refusals():
    gain_cutoff_band = _study_module('wavebit._common.conventions', 'wavebit._core.conventions').gain_cutoff_band
    header(7, 'refusals -- what is refused, what is only flagged')

    def message(call):
        try:
            with _quiet():
                call()
        except ValueError as error:
            text = str(error)
            return 'ValueError: ' + (text if len(text) <= 150 else text[:147] + '...')
        return 'accepted'

    s = solver()
    print('  an active propagating double negative (-2-0.1i, -1-0.1i) as bot:\n    ' +
          message(lambda: setattr(s, 'bot', wavebit.Material(*ACTIVE))))
    print('  gain in mu alone, (-2+0.1i, -1-0.01i), as top:\n    ' +
          message(lambda: setattr(s, 'top', wavebit.Material(-2 + 0.1j, -1 - 0.01j))))
    still = (-1 - 1.2j, -1 - 1.2j)
    with _quiet():
        R = float(np.sum(solver(bot=wavebit.Material(*still)).solve().s_inc.R))
    print(f'  an active double negative that does not propagate, (-1-1.2i, -1-1.2i), '
          f'Re(eps*mu) = {(still[0] * still[1]).real:.2f}: accepted, R = {R:.6f} (Maystre\'s '
          f'root, which the mirror cut shares there)')
    print('  active media that are not refused, and the sign of their index, lit at theta = 0.3, '
          'phi = 0:')
    for label, pair in (('(-2-0.01i, 0.1-0.5i), a magnetic metal with gain', (-2 - 0.01j,
                                                                             0.1 - 0.5j)),
                        ('(-1-1.2i, -1-1.2i), the double negative above', still)):
        s = solver(top=wavebit.Material(*pair), theta=0.3, phi=0.0)
        n = complex(s.n_top)
        kx = complex(np.ravel(_host(s.order_kxy()[0]))[0])
        print(f'    {label:50s} n_top = {c(n)}, s = {int(np.sign(n.real)):+d}, '
              f'kx = {kx.real:+.4f}: ' + message(lambda: s.solve()))
    print('  a GaussianBeam launched from (-3, -0.5):\n    ' + message(lambda: _legacy_wavebit.rcwa(
        wavelength=WL, period=[P_G], structure=G1[None, None, :], thickness=[H_G],
        truncation_order=(2, 0), top=wavebit.Material(*ASYM_LOSSLESS), bot=N_GLASS ** 2,
        answers='field', power_region=((-1000.0, 1000.0), (-1000.0, 1000.0)),
        excitation=[wavebit.GaussianBeam(waist=2000.0, theta=0.2)]).solve()))
    print('  a real angle from a lossless medium that carries no wave, (-4, 1):\n    ' +
          message(lambda: solver(top=wavebit.Material(-4.0, mu=1.0), theta=0.3).solve()))
    rng = np.random.default_rng(5)
    k_par_sq = np.linspace(0.0, 9.0, 901)
    equal = 0
    for _ in range(300):
        eps = rng.uniform(-5, -0.1) + 1j * rng.uniform(-0.5, 0.5)
        mu = rng.uniform(-3, -0.1) + 1j * rng.uniform(-0.5, 0.5)
        band = gain_cutoff_band(np, eps, mu, k_par_sq)
        twin = gain_cutoff_band(np, -np.conj(eps), -np.conj(mu), k_par_sq)
        equal += int(np.array_equal(band, twin))
    flagged = gain_cutoff_band(np, *ACTIVE, k_par_sq)
    below = ((ACTIVE[0] * ACTIVE[1] - k_par_sq) .imag < 0)
    print(f'\n  gain band near grazing, k_par**2 0 .. 9 in 901 steps: 300 random double negatives '
          f'(either sign of loss in either factor) against their double-positive mirrors: '
          f'{equal} of 300 identical')
    print(f'    passive (-2+0.1i, -1+0.1i), (-2, -1), (-3+0.1i, -0.5+0.05i): flagged '
          f'{sum(int(gain_cutoff_band(np, *p, k_par_sq).sum()) for p in (PASSIVE, LOSSLESS, ASYM))}'
          f' orders; active (-2-0.1i, -1-0.1i): {int(flagged.sum())} of 901, all with '
          f'Im(kz**2) > 0; testing Im(kz**2) < 0 as for an ordinary medium flags '
          f'{int(np.sum(below & flagged))}')


# ------------------------------------------------------------------------------- 8. oracles

def section_oracles():
    header(8, 'oracles')
    print('  closed-form Fresnel r, t, R, T (s and p), glass through a 37 nm glass spacer; '
          '5 paths x'
          ' 2 sides x ETM / SMM:')
    with _quiet():
        for label, pair, thetas, lit in (
                ('far side (-2+0.1i, -1+0.1i), 0 .. 80 deg', PASSIVE, (0, 30, 60, 80), False),
                ('far side (-3+0.1i, -0.5+0.05i)', ASYM, (0, 30, 60, 80), False),
                ('far side lossless (-2, -1)', LOSSLESS, (0, 30, 60, 80), False),
                ('far side (-6+0.3i, -1.5+0.05i)', HIGH, (0, 30, 60, 80), False),
                ('far side magnetic metal, 0/30/60 deg', MAGNETIC_METAL, (0, 30, 60), False),
                ('far side eps = mu = -1+1i, 0 deg', TIE, (0,), False),
                ('lit (-3+0.1i, -0.5+0.05i), 0/20/35/60 deg', ASYM, (0, 20, 35, 60), True),
                ('lit lossless (-3, -0.5)', ASYM_LOSSLESS, (0, 20, 35, 60), True),
                ('lit (-2+0.1i, -1+0.1i)', PASSIVE, (0, 20, 35, 60), True)):
            worst = max(check_interface(*pair, path, algo, side, thetas, lit)
                        for path in PATHS for side in ('top', 'bottom') for algo in ('ETM', 'SMM'))
            print(f'    {label:44s} {e(worst)}')
    print('  a grating on the half-space against the same grating on a 16 um slab of the medium '
          '(a layer keeps both roots), every order:')
    worst_R = worst_T = 0.0
    with _quiet():
        for path in PATHS:
            for algo in ('ETM', 'SMM'):
                half = grating(path, algo).solve(auto_detune=False)
                slab = grating(path, algo, d=16000.0).solve(auto_detune=False)
                for h, s in zip(variants(half, path), variants(slab, path)):
                    worst_R = max(worst_R, np.max(np.abs(_host(h.R) - _host(s.R))))
                    worst_T = max(worst_T, abs(float(np.sum(_host(h.T)))
                                               - float(_host(s.power_entering))))
    print(f'    R per order {e(worst_R)}; half-space T against the power the slab takes in '
          f'{e(worst_T)} (5 paths x ETM / SMM)')

    def total(delta):
        nim = (-2 + 1j * delta, -1 + 1j * delta)
        d = 34.0 / (2 * 2 * np.pi / WL * abs((-np.sqrt(nim[0] * nim[1])).imag))
        result = grating('conical', far=nim, d=d).solve(auto_detune=False)
        return np.concatenate([_host(v.R).ravel() for v in variants(result, 'conical')]), d

    with _quiet():
        lossless = grating('conical', far=LOSSLESS).solve(auto_detune=False)
        lossless = np.concatenate([_host(v.R).ravel() for v in variants(lossless, 'conical')])
        (a, _), (b, d) = total(1e-4), total(1e-5)
    extrapolated = b + (b - a) * 1e-5 / (1e-4 - 1e-5)
    print(f'  the lossless half-space as the limit of lossy slabs (-2+i delta, -1+i delta), '
          f'conical: delta = 1e-5 ({d / 1e6:.0f} mm) is {e(np.max(np.abs(b - lossless)))} away, '
          f'Richardson on 1e-4 and 1e-5 {e(np.max(np.abs(extrapolated - lossless)))}')
    print('  reciprocity: a grating lit from the lossless (-3, -0.5) at 0.35 rad, order m into '
          'glass, against glass lit along that order\'s reverse, read on the incident order:')
    worst = 0.0
    with _quiet():
        for tetm in ('te', 'tm'):
            for algo in ('ETM', 'SMM'):
                for m in (1, -1):
                    def solve(side, excitation, **incidence):
                        s = _legacy_wavebit.rcwa(wavelength=WL, period=[P_G], structure=G1[None, None, :],
                                         thickness=[H_G], truncation_order=3, tetm=tetm,
                                         layer_algorithm=algo,
                                         top=wavebit.Material(*ASYM_LOSSLESS), bot=N_GLASS ** 2,
                                         incidence_side=side, excitation=excitation,
                                         **incidence)
                        T = np.ravel(_host(s.solve(auto_detune=False).T))
                        (centre, half), _ = s.order_window
                        return (list(range(centre - half, centre + half + 1)),
                                np.ravel(_host(s.order_kxy()[0])).real, T)
                    orders, kx, forward = solve('top', {0: 1.0, m: 0.0, -m: 0.0}, theta=0.35)
                    kx_inc = kx[orders.index(0)]
                    orders_b, _, backward = solve('bottom', {-m: 1.0, 0: 0.0, m: 0.0},
                                                  kx=-kx_inc * 2 * np.pi / WL)
                    worst = max(worst, abs(forward[orders.index(m)] - backward[orders_b.index(0)]))
    print(f'    orders +-1, TE / TM, ETM / SMM: {e(worst)} (incident kx {kx_inc:+.4f} < 0)')
    print('  the coherent door (PlaneWave) against the ordinary SMM solve, lit from the double '
          'negative, (theta, phi) = (0.35, 0.6), (0, 0.6), (0, 2.5), both sides:')
    worst_power = worst_field = 0.0
    with _quiet():
        for pair in (ASYM, ASYM_LOSSLESS):
            for side in ('top', 'bottom'):
                half = wavebit.Material(*pair)
                scene = (dict(top=half, bot=N_GLASS ** 2) if side == 'top'
                         else dict(top=N_GLASS ** 2, bot=half))
                opts = dict(wavelength=WL, period=[P_G], structure=G1[None, None, :],
                            thickness=[H_G], truncation_order=(3, 0), answers='field', **scene)
                for theta, phi in ((0.35, 0.6), (0.0, 0.6), (0.0, 2.5)):
                    pol = (0.6, 0.8j)
                    coherent = _legacy_wavebit.rcwa(**opts, excitation={side: wavebit.PlaneWave(
                        theta=theta, phi=phi, polarization=pol)}).solve()
                    ordinary = _legacy_wavebit.rcwa(**opts, theta=theta, phi=phi, polarization=pol,
                                            layer_algorithm='SMM',
                                            incidence_side=side).solve(auto_detune=False)
                    back, through = ((coherent.output_top, coherent.output_bottom)
                                     if side == 'top' else
                                     (coherent.output_bottom, coherent.output_top))
                    worst_power = max(worst_power,
                                      abs(float(back) - float(np.sum(ordinary.R))),
                                      abs(float(through) - float(np.sum(ordinary.T))))
                    points = dict(sample_x=[-123.0, 211.0], sample_y=[0.0],
                                  sample_z=[10.0, 90.0])
                    worst_field = max(worst_field, np.max(np.abs(
                        _host(coherent.reconstruct_field(**points))
                        - _host(ordinary.reconstruct_field(**points)))))
    print(f'    power {e(worst_power)}, field {e(worst_field)}')
    section_fmmax()
    section_mirror_medium()
    section_pymoosh()


def _reference_path(name):
    """Resolve current grouped references and older test-root snapshots."""
    candidates = (_SELECTED_ROOT / 'tests' / 'references' / name,
                  _SELECTED_ROOT / 'tests' / name)
    return next((path for path in candidates if path.is_file()), candidates[0])


def _reference_module(name):
    """Load this checkout's oracle helper, independent of other tests packages."""
    path = _reference_path(name + '.py')
    spec = importlib.util.spec_from_file_location('_wavebit_study_reference_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def section_fmmax():
    path = _reference_path('golden_fmmax_dng.npz')
    if not os.path.exists(path):
        print('  fmmax golden: not found, skipped')
        return
    cases = _reference_module('_fmmax_dng_cases')
    CASES, LEVELS, THICK = cases.CASES, cases.LEVELS, cases.THICK
    incident_kxy, orders = cases.incident_kxy, cases.orders
    richardson = _reference_module('_fmmax_laurent_cases').richardson
    golden = np.load(path)
    print('  fmmax 1.7.1 FFT formulation, frozen (tests/references/golden_fmmax_dng.npz): '
          'unpolarized R and T of every order, Richardson over 16/32/64/128 samples per pixel;')
    print('    worst order over ETM / SMM and the theta / direct-k doors, against the bound of '
          'the extrapolation:')

    def unpolarized(result, name):
        mx, my = CASES[name]['order']
        out = np.zeros((2, len(orders(name))))
        for sub in (result.s_inc, result.p_inc):
            for k, grid in enumerate((np.asarray(sub.R), np.asarray(sub.T))):
                grid = grid.reshape(2 * my + 1, 2 * mx + 1)
                for i, (m, n) in enumerate(orders(name)):
                    out[k, i] += grid[n + my, m + mx] / 2
        return out

    for name, case in CASES.items():
        values = [golden[f'{name}__phase64__R{samples}__per_order'] for samples in LEVELS]
        fine = richardson(values[1:], LEVELS[1:])
        bound = np.abs(fine - richardson(values[:3], LEVELS[:3])).max() + 1e-12
        stock = richardson([golden[f'{name}__stock__R{samples}__per_order']
                            for samples in LEVELS][1:], LEVELS[1:])
        one_d = len(case['period']) == 1
        worst = 0.0
        for algo in ('ETM', 'SMM'):
            for door in ('k', 'theta'):
                kw = dict(top=wavebit.Material(*case['top']), bot=wavebit.Material(*case['bot']),
                          wavelength=WL, thickness=[THICK], structure=case['eps'][None],
                          factorization='laurent', layer_algorithm=algo,
                          truncation_order=case['order'][0] if one_d else case['order'],
                          period=[case['period'][0]] if one_d else case['period'])
                if case['mu'] is not None:
                    kw['mu'] = case['mu'][None]
                if door == 'k':
                    kx, ky = incident_kxy(name)
                    kw.update(kx=2 * np.pi / WL * kx, ky=2 * np.pi / WL * ky)
                else:
                    kw.update(theta=case['theta'], phi=case['phi'])
                with _quiet():
                    got = unpolarized(_legacy_wavebit.rcwa(**kw).solve(auto_detune=False), name)
                worst = max(worst, np.abs(got - fine).max())
        print(f'    {name:20s} {e(worst)} (bound {e(bound)}; fmmax as shipped, float32 centring '
              f'phase, {e(np.abs(stock - fine).max())} from the corrected run)')


def _mirrored(value):
    """``-conj(value)``; a number stays a Python complex, so both scenes take one square root
    (``cmath`` and NumPy differ by an ulp or two on some arguments; section 13)."""
    return complex(-np.conj(value)) if np.ndim(value) == 0 else -np.conj(value)


def _every_amplitude(result):
    """``{'s_inc.r_s': array, ...}``: every amplitude and power the result carries."""
    out = {}
    for name in ('incident', 's_inc', 'p_inc'):
        sub = getattr(result, name, None)
        for field in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
            value = None if sub is None else getattr(sub, field, None)
            if value is not None:
                out[f'{name}.{field}'] = np.ravel(_host(value))
    return out


def _mirror_scene(path, algo, side, theta, far, eps=None, mu=None, thickness=(10.0,),
                  truncation_order=None):
    """Glass lit at ``theta`` onto the half-space ``far``, through the raster layers ``eps`` /
    ``mu``, or through none (``solve_conv`` with no layer) when they are ``None``."""
    one_d = path in ('te', 'tm', 'conical')
    layers = eps is not None
    if not layers:
        shape = (1, 1, 4) if one_d else (1, 2, 2)
        eps, mu = np.full(shape, 2.25 + 0j), np.full(shape, 1.0 + 0j)
    glass, medium = wavebit.Material(2.25 + 0j, mu=1.0 + 0j), wavebit.Material(*far)
    top, bot = (glass, medium) if side == 'top' else (medium, glass)
    common = dict(top=top, bot=bot, theta=theta, wavelength=WL, layer_algorithm=algo,
                  thickness=list(thickness), answers='amplitude', incidence_side=side,
                  structure=eps, mu=mu)
    if path in ('te', 'tm'):
        s = _legacy_wavebit.rcwa(**common, tetm=path, period=[P_G], truncation_order=truncation_order or 3)
    elif path == 'conical':
        s = _legacy_wavebit.rcwa(**common, phi=0.6, psi=0.3, period=[P_G],
                         truncation_order=truncation_order or 3)
    else:
        s = _legacy_wavebit.rcwa(**common, phi=0.6, psi=0.3, period=(P_G, 0.9 * P_G),
                         truncation_order=truncation_order or (1, 1))
        if path == 'first_order':
            s.formulation = 'first_order'
    with _quiet():
        if layers:
            return _every_amplitude(s.solve(auto_detune=False))
        e_conv, m_conv = s.eps_conv_matrices(), s.mu_conv_matrices()
        s.thickness = []
        return _every_amplitude(s.solve_conv(
            *(None if a is None else a[:0] for a in e_conv),
            mu=tuple(None if a is None else a[:0] for a in m_conv), eps_top=s.top.epsilon,
            mu_top=s.top.mu, eps_bot=s.bot.epsilon, mu_bot=s.bot.mu))


#: Far half-spaces for the mirror identity: lossy, lossless, and lossless but evanescent from
#: glass at 60 degrees.
MIRROR_FAR = ((2 + 0.1j, 1.5 + 0.05j), (2.25 + 0j, 1.3 + 0j), (1.2 + 0j, 1.0 + 0j))


def section_mirror_medium():
    """A. Lakhtakia's conjugation symmetry, Microwave Opt. Technol. Lett. 40, 160 (2004):
    replacing (eps, mu) by (-conj(eps), -conj(mu)) conjugates E and H. V. Grunhut and
    R. A. Depine, Eur. Phys. J. D 62, 227 (2011), apply it to a flat interface lit at a real
    angle: the Fresnel coefficients change phase, not magnitude. Lit from a lossless medium at a
    real angle, with the far side (and every layer) mirrored, every amplitude is conjugated;
    that is what is measured here."""
    print('  the mirror medium (Lakhtakia 2004; Grunhut and Depine 2011): glass lit at 0, 0.3 rad '
          'and 60 deg onto (eps, mu) and onto (-conj(eps), -conj(mu)) -- lossy, lossless, and '
          'evanescent at 60 deg; 5 paths x ETM / SMM x either side lit:')
    exact = total = 0
    for path in PATHS:
        for algo in ('ETM', 'SMM'):
            for side in ('top', 'bottom'):
                for far in MIRROR_FAR:
                    for theta in (0.0, 0.3, np.deg2rad(60.0)):
                        plain = _mirror_scene(path, algo, side, theta, far)
                        mirror = _mirror_scene(path, algo, side, theta,
                                               tuple(_mirrored(v) for v in far))
                        for key, value in plain.items():
                            want = value if key.endswith(('.R', '.T')) else np.conj(value)
                            total += 1
                            exact += int(np.array_equal(mirror[key], want))
    print(f'    no layer between (solve_conv): {exact} of {total} arrays are exactly the '
          f'conjugate (R and T: equal), bit for bit')

    def uniform(path):
        shape = (1, 9) if path in ('te', 'tm', 'conical') else (5, 5)
        return (np.stack([np.full(shape, 2 + 0.1j), np.full(shape, -3 + 0.2j)]),
                np.stack([np.full(shape, 1.2 + 0.05j), np.full(shape, -1.1 + 0.1j)]))

    def symmetric_grating(path):
        shape = (1, 1, 9) if path in ('te', 'tm', 'conical') else (1, 5, 5)
        eps, mu = np.full(shape, 2.0 + 0.1j), np.full(shape, 1.0 + 0j)
        middle = (0, 0, slice(3, 6)) if shape[1] == 1 else (0, slice(1, 4), 2)
        eps[middle], mu[middle] = -3 + 0.3j, -1.2 + 0.1j
        return eps, mu

    worst = 0.0
    for path in PATHS:
        for build, thickness in ((uniform, (60.0, 90.0)), (symmetric_grating, (60.0,))):
            eps, mu = build(path)
            for far in MIRROR_FAR:
                for algo in ('ETM', 'SMM'):
                    for side in ('top', 'bottom'):
                        for theta in (0.0, 0.2):
                            kw = dict(thickness=thickness, truncation_order=1 if path in (
                                'te', 'tm', 'conical') else (1, 1))
                            plain = _mirror_scene(path, algo, side, theta, far, eps, mu, **kw)
                            mirror = _mirror_scene(path, algo, side, theta,
                                                   tuple(_mirrored(v) for v in far),
                                                   _mirrored(eps), _mirrored(mu), **kw)
                            for key, value in plain.items():
                                want = value if key.endswith(('.R', '.T')) else np.conj(value)
                                worst = max(worst, np.abs(mirror[key] - want).max()
                                            / max(1.0, np.abs(value).max()))
    print(f'    every layer mirrored too (two lossy uniform layers, a positive one and a double '
          f'negative; a grating symmetric about the cell centre), every order propagating in '
          f'glass, theta 0 and 0.2: {e(worst)} relative')


def section_pymoosh():
    path = _reference_path('golden_pymoosh_dng.npz')
    if not os.path.exists(path):
        print('  PyMoosh golden: not found, skipped')
        return
    cases = _reference_module('_pymoosh_dng_cases')
    CASES, media, pymoosh_kx = cases.CASES, cases.media, cases.pymoosh_kx
    golden = np.load(path)
    print(f'  PyMoosh {golden["__pymoosh_version"]}, frozen (tests/references/'
          'golden_pymoosh_dng.npz): flat stacks with a lossless double negative as substrate, lit '
          'superstrate, inner layer and all three; (r, t, R, T), TE and TM, at PyMoosh\'s kx:')

    def solve(name, index, path, algo, pol, stack):
        case = CASES[name]
        inner = stack[1:-1][::-1]
        one_d = path == 'tetm'
        shape = (1, 2) if one_d else (2, 2)
        kw = dict(top=wavebit.Material(*stack[0]), bot=wavebit.Material(*stack[-1]),
                  wavelength=WL, thickness=[d for _, d in case['layers']][::-1],
                  layer_algorithm=algo, answers='amplitude',
                  structure=np.stack([np.full(shape, m[0]) for m in inner]),
                  mu=np.stack([np.full(shape, m[1]) for m in inner]),
                  kx=2 * np.pi / WL * pymoosh_kx(name, case['thetas'][index]))
        if one_d:
            s = _legacy_wavebit.rcwa(**kw, period=[1000.0], truncation_order=0,
                             tetm='te' if pol == 0 else 'tm')
        else:
            s = _legacy_wavebit.rcwa(**kw, ky=0.0, period=(1000.0, 1000.0), truncation_order=(0, 0),
                             polarization=(1.0, 0.0) if pol == 0 else (0.0, 1.0))
            if path == 'first_order':
                s.formulation = 'first_order'
        with _quiet():
            result = s.solve(auto_detune=False)
        sub = result.incident if one_d else (result.s_inc if pol == 0 else result.p_inc)
        r, t = (sub.r_s, sub.t_s) if pol == 0 else (sub.r_p, sub.t_p)
        r, t = (complex(np.sum(np.asarray(v))) for v in (r, t))
        if pol == 1:                     # PyMoosh's TM t is the H ratio: undo the E ratio's index
            (e1, m1), (e2, m2) = stack[0], stack[-1]
            index_1 = np.sqrt(complex(e1)) * np.sqrt(complex(m1))
            index_2 = np.sqrt(complex(e2)) * np.sqrt(complex(m2))
            t /= (index_1 / m1) / (index_2 / m2)
        return np.asarray([r, t, float(np.sum(sub.R)), float(np.sum(sub.T))])

    worst = 0.0
    rows = 0
    for name in CASES:
        for path in ('tetm', '2d', 'first_order'):
            for algo in ('ETM', 'SMM'):
                for i in range(len(CASES[name]['thetas'])):
                    for pol in (0, 1):
                        got = solve(name, i, path, algo, pol, media(name))
                        worst = max(worst, np.abs(got - golden[f'{name}__pol{pol}'][i]).max())
                        rows += path == 'tetm' and algo == 'ETM'
    print(f'    {rows} rows x TE/TM, 2D and first-order paths x ETM / SMM: {e(worst)} at worst')
    misses = []
    for name in CASES:
        if not any(e_.real < 0 and m.real < 0 for e_, m in media(name)[1:]):
            continue
        twin = [(-e_, -m) if e_.real < 0 and m.real < 0 else (e_, m) for e_, m in media(name)]
        miss = max(np.abs(solve(name, i, 'tetm', 'SMM', pol, twin)
                          - golden[f'{name}__pol{pol}'][i]).max()
                   for i in range(len(CASES[name]['thetas'])) for pol in (0, 1))
        misses.append(f'{name} {miss:.1f}')
    print('    every double negative replaced by its double-positive twin (-eps, -mu), where one '
          'lies beyond the lit side: misses by ' + ', '.join(misses))


# ------------------------------------------------------------------------------ 9. backends

def section_backends():
    header(9, 'backends -- JAX, PyTorch, jit, gradients')
    try:
        import jax
        import jax.numpy as jnp
        import torch
    except ImportError as error:                    # pragma: no cover - environment
        print(f'  skipped: {error}')
        return
    wavebit.enable_x64()
    with _quiet():
        reference = {
            'far': amplitudes(grating('conical').solve(auto_detune=False), 'conical'),
            'lit': amplitudes(grating('conical', lit=ASYM, theta=0.35, phi=0.6).solve(
                auto_detune=False), 'conical'),
            'lit, theta = 0': amplitudes(grating('conical', lit=ASYM, theta=0.0, phi=0.6).solve(
                auto_detune=False), 'conical')}
        for backend, label in (('jax', 'JAX eager'), ('torch', 'PyTorch')):
            got = {
                'far': amplitudes(grating('conical', backend=backend).solve(auto_detune=False),
                                  'conical'),
                'lit': amplitudes(grating('conical', lit=ASYM, theta=0.35, phi=0.6,
                                          backend=backend).solve(auto_detune=False), 'conical'),
                'lit, theta = 0': amplitudes(grating('conical', lit=ASYM, theta=0.0, phi=0.6,
                                                     backend=backend).solve(auto_detune=False),
                                             'conical')}
            print(f'  {label:9s} against NumPy, conical grating, far / lit / lit at theta = 0: '
                  + ', '.join(e(np.max(np.abs(got[k] - reference[k]))) for k in reference))

        def lit(re_eps, theta=0.35):
            pair = (re_eps + 1j * ASYM[0].imag, ASYM[1])
            result = grating('conical', lit=pair, theta=theta, phi=0.6, backend='jax').solve(
                auto_detune=False)
            return [result.incident.R, result.incident.r_s, result.incident.r_p,
                    result.incident.t_s, result.incident.t_p]

        jitted = jax.jit(lit)
        traced = np.concatenate([np.ravel(np.asarray(a)) for a in jitted(jnp.asarray(-3.0))])
        positive = np.concatenate([np.ravel(np.asarray(a)) for a in jitted(jnp.asarray(3.0))])
        eager_positive = amplitudes(grating('conical', lit=(3 + 0.1j, ASYM[1]), theta=0.35,
                                            phi=0.6).solve(auto_detune=False), 'conical',
                                    ('R', 'r_s', 'r_p', 't_s', 't_p'))
        eager_negative = amplitudes(grating('conical', lit=ASYM, theta=0.35, phi=0.6).solve(
            auto_detune=False), 'conical', ('R', 'r_s', 'r_p', 't_s', 't_p'))
        size = len(eager_negative) // 3
        print(f'  jax.jit over the lit medium\'s Re(eps), conical grating at 0.35 rad: Re(eps) = -3'
              f' (a double negative) {e(np.max(np.abs(traced - eager_negative[:size])))}, '
              f'Re(eps) = +3 (eps = 3+0.1i, mu = -0.5+0.05i: not a double negative) '
              f'{e(np.max(np.abs(positive - eager_positive[:size])))}; compiles: '
              f'{jitted._cache_size()}')

        def total_R(re_eps):
            return grating('conical', far=(re_eps + 1j * PASSIVE[0].imag, PASSIVE[1]),
                           backend='jax').solve(auto_detune=False).R.sum().real

        def total_R_theta(theta):
            return grating('conical', lit=ASYM, theta=theta, phi=0.6, backend='jax').solve(
                auto_detune=False).R.sum().real

        h = 1e-6
        rows = []
        for label, f, x in (('far side, d R / d Re(eps)', total_R, -2.0),
                            ('lit side, d R / d theta', total_R_theta, 0.35)):
            auto = float(jax.grad(f)(jnp.asarray(x)))
            fd = (float(f(x + h)) - float(f(x - h))) / (2 * h)
            rows.append(f'{label}: {auto:+.6f} against {fd:+.6f} ({e(abs(auto - fd))})')
        print('  jax.grad against a central difference (h = 1e-6):\n    ' + '\n    '.join(rows))
        eps = torch.tensor(PASSIVE[0], dtype=torch.complex128, requires_grad=True)
        R = grating('conical', far=(eps, PASSIVE[1]), backend='torch').solve(
            auto_detune=False).R.sum()
        R.real.backward()
        auto = float(eps.grad.real)
        fd = (float(total_R(-2.0 + h)) - float(total_R(-2.0 - h))) / (2 * h)
        print(f'  PyTorch backward, far side, d R / d Re(eps): {auto:+.6f} against {fd:+.6f} '
              f'({e(abs(auto - fd))})')

        def band(re_eps):
            s = grating('conical', far=(re_eps + 1j * ACTIVE[0].imag, ACTIVE[1]), backend='jax')
            s.theta = np.deg2rad(60.0)
            return s.gain_cutoff_band()['bottom']

        flagged = np.asarray(jax.jit(band)(jnp.asarray(ACTIVE[0].real)))
        # Every order is solved under jit, flagged or not, with the mirror root; the active
        # medium's propagating orders have no single-frequency root (Skaar 2006), and nothing
        # but an eager construction refuses them.
        refused = 'accepted'
        try:
            band(ACTIVE[0].real)
        except ValueError:
            refused = 'refused'
    print(f'  an active double negative (-2-0.1i, -1-0.1i) under jax.jit, glass at 60 deg: '
          f'gain_cutoff_band flags {int(flagged.sum())} of {flagged.size} orders; eagerly it is '
          f'{refused}')


# ------------------------------------------------------------------------------- 10. develop

HARNESS_MEDIA = {
    'glass': (2.25, None), 'lossy': (2 + 0.5j, None), 'metal': (-10 + 1j, None),
    'lossless_metal': (-4.0, None), 'gain': (1 - 0.05j, None), 'magnetic': (2.25, 1.3),
    'magnetic_lossy': (2.25 + 0.1j, 1.2 + 0.05j), 'magnetic_metal_s+': (-2 + 0.1j, 1.3),
    'mu_negative': (2 + 0.1j, -1 + 0.1j), 'mu_negative_lossless': (2.0, -1.0),
    'magnetic_gain': (2.25 - 0.05j, 1.2), 'mu_one': (-4 + 0j, 1.0)}


def _record(out, key, call):
    """``out[key + '/...']`` from ``call()``, or the class name of what it raised."""
    try:
        with _quiet():
            values = call()
    except Exception as error:                      # recorded, not hidden
        out[key + '/error'] = np.frombuffer(type(error).__name__.encode(), dtype=np.uint8)
        return
    for name, value in values.items():
        out[f'{key}/{name}'] = np.asarray(_host(value))


def _solved(s, path):
    result = s.solve(auto_detune=False)
    values = {f'{v}/{f}': getattr(sub, f) for v, sub in zip('isp', variants(result, path))
              for f in ('R', 'T', 'r_s', 'r_p', 't_s', 't_p', 'r_xyz', 't_xyz',
                        'power_entering') if getattr(sub, f, None) is not None}
    ys = [0.0, 77.0] if path in ('2d', 'first_order') else [0.0]
    values.update(n_top=np.asarray(complex(s.n_top)), n_bot=np.asarray(complex(s.n_bot)),
                  kx=s.order_kxy()[0],
                  field=result.reconstruct_field([0.0, 123.0], ys, [10.0, 90.0]))
    return values


def harness():
    """``{name: array}`` over media with ``s = +1``. NumPy: every path, both algorithms, the
    medium lit or not, theta 0 / 0.3 / 0.3+0.05i / 1.4 (grazing); amplitudes, R / T, r_xyz /
    t_xyz, the indices, the order grid and a sampled field; and the coherent door. Eager JAX and
    PyTorch: the conical and 2D paths at 0.3. ``jax.jit``: a magnetic half-space's Re(eps)
    traced. Run here and in REV's tree, by the same code."""
    out = {}
    for name, (eps, mu) in HARNESS_MEDIA.items():
        half = eps if mu is None else wavebit.Material(eps, mu=mu)
        lit = (eps,) if mu is None else (eps, mu)
        for path in PATHS:
            for algo in ('ETM', 'SMM'):
                for theta in (0.0, 0.3, 0.3 + 0.05j, 1.4):
                    _record(out, f'np/{name}/{path}/{algo}/far/{theta}', lambda: _solved(
                        _far_grating(path, algo, half, theta), path))
                    _record(out, f'np/{name}/{path}/{algo}/lit/{theta}', lambda: _solved(
                        grating(path, algo, theta=theta, phi=0.6, lit=lit, answers='field'),
                        path))
        for theta in (0.3, 0.0):
            def coherent():
                scene = dict(wavelength=WL, period=[P_G], structure=G1[None, None, :],
                             thickness=[H_G], truncation_order=(3, 0), answers='field',
                             top=half, bot=N_GLASS ** 2)
                result = _legacy_wavebit.rcwa(**scene, excitation={'top': wavebit.PlaneWave(
                    theta=theta, phi=0.6, polarization=(0.6, 0.8j))}).solve()
                return dict(top=result.output_top, bottom=result.output_bottom,
                            field=result.reconstruct_field(sample_x=[-123.0, 211.0],
                                                           sample_y=[0.0], sample_z=[10.0, 90.0]))
            _record(out, f'coherent/{name}/{theta}', coherent)
    try:
        import jax
        import jax.numpy as jnp
        import torch                                # noqa: F401
    except ImportError:                             # pragma: no cover - environment
        return out
    wavebit.enable_x64()
    for backend in ('jax', 'torch'):
        for name, (eps, mu) in HARNESS_MEDIA.items():
            half = eps if mu is None else wavebit.Material(eps, mu=mu)
            lit = (eps,) if mu is None else (eps, mu)
            for path in ('conical', '2d'):
                for algo in ('ETM', 'SMM'):
                    _record(out, f'{backend}/{name}/{path}/{algo}/far', lambda: _solved(
                        _far_grating(path, algo, half, 0.3, backend=backend), path))
                    _record(out, f'{backend}/{name}/{path}/{algo}/lit', lambda: _solved(
                        grating(path, algo, theta=0.3, phi=0.6, lit=lit, answers='field',
                                backend=backend), path))
    for name in ('magnetic', 'magnetic_lossy', 'mu_negative'):
        eps, mu = HARNESS_MEDIA[name]
        for role in ('far', 'lit'):
            def traced(re_eps):
                pair = (re_eps + 1j * complex(eps).imag, mu)
                s = (grating('conical', theta=0.3, phi=0.6, lit=pair, backend='jax')
                     if role == 'lit'
                     else _far_grating('conical', 'SMM', wavebit.Material(*pair), 0.3,
                                       backend='jax'))
                result = s.solve(auto_detune=False)
                return {f: getattr(result.incident, f) for f in ('R', 'T', 'r_s', 'r_p',
                                                                 't_s', 't_p')}
            _record(out, f'jit/{name}/{role}', lambda: jax.jit(traced)(
                jnp.asarray(complex(eps).real)))
    return out


def _far_grating(path, algo, half, theta, backend='numpy'):
    common = dict(top=N_GLASS ** 2, bot=half, wavelength=WL, layer_algorithm=algo, theta=theta,
                  answers='field', backend=backend_arg(backend))
    if path in ('te', 'tm', 'conical'):
        layer, kw = G1[None, None, :], dict(period=[P_G], truncation_order=3)
    else:
        layer, kw = G2[None], dict(period=(P_G, 0.9 * P_G), truncation_order=(1, 1))
    kw.update(tetm=path) if path in ('te', 'tm') else kw.update(phi=0.6, psi=0.3)
    s = _legacy_wavebit.rcwa(**common, **kw, thickness=[H_G], structure=layer)
    if path == 'first_order':
        s.formulation = formulation('first_order')
    return s


def double_negative_outcomes():
    """What each tree does with the double negatives: a refusal's class name, or ``R``."""
    out = {}
    with _quiet():
        for label, pair in (('far (-2+0.1i, -1+0.1i)', PASSIVE), ('far (-2, -1)', LOSSLESS),
                            ('far eps = mu = -1+1i', TIE)):
            for algo in ('ETM', 'SMM'):
                try:
                    R = float(np.sum(solver(bot=wavebit.Material(*pair), algo=algo).solve(
                        auto_detune=False).s_inc.R))
                    out[f'{label}, {algo}'] = np.asarray(R)
                except Exception as error:
                    out[f'{label}, {algo}'] = np.frombuffer(type(error).__name__.encode(),
                                                            dtype=np.uint8)
        for algo in ('ETM', 'SMM'):
            try:
                R = float(np.sum(solver(top=wavebit.Material(*ASYM), theta=0.35, phi=0.6,
                                        algo=algo).solve(auto_detune=False).s_inc.R))
                out[f'lit (-3+0.1i, -0.5+0.05i), {algo}'] = np.asarray(R)
            except Exception as error:
                out[f'lit (-3+0.1i, -0.5+0.05i), {algo}'] = np.frombuffer(
                    type(error).__name__.encode(), dtype=np.uint8)
    return out


def section_develop(rev):
    header(10, f'develop -- this tree against {rev}' if rev
           else 'develop -- skipped (no --develop)')
    if not rev:
        return
    with tempfile.TemporaryDirectory() as tmp:
        tree = os.path.join(tmp, 'tree')
        os.mkdir(tree)
        archive = subprocess.run(['git', '-C', ROOT, 'archive', rev], check=True,
                                 capture_output=True).stdout
        subprocess.run(['tar', '-x', '-C', tree], input=archive, check=True)
        dump = os.path.join(tmp, 'develop.npz')
        env = dict(os.environ, PYTHONWARNINGS='ignore')
        subprocess.run([sys.executable, os.path.abspath(__file__), '--dump', dump,
                        '--root', tree], check=True, env=env, capture_output=True)
        theirs = dict(np.load(dump))
    ours = harness()
    ours.update({f'dng/{k}': v for k, v in double_negative_outcomes().items()})
    print(f'  {len(HARNESS_MEDIA)} media with s = +1: ' + ', '.join(
        f'{name} {c_or_r(eps)}' + ('' if mu is None else f' / {c_or_r(mu)}')
        for name, (eps, mu) in HARNESS_MEDIA.items()))
    groups = (('np', 'NumPy: 5 paths x ETM / SMM x lit / far x theta 0, 0.3, 0.3+0.05i, 1.4'),
              ('jax', 'JAX eager: conical and 2D x ETM / SMM x lit / far, theta 0.3'),
              ('torch', 'PyTorch eager: the same'),
              ('coherent', 'the coherent door, PlaneWave from the medium, theta 0.3 and 0'),
              ('jit', 'jax.jit, Re(eps) of a magnetic half-space traced, lit and far'))
    total = identical = 0
    for prefix, label in groups:
        keys = sorted(k for k in ours if k.startswith(prefix + '/'))
        differ = [k for k in keys if k not in theirs or ours[k].dtype != theirs[k].dtype
                  or ours[k].shape != theirs[k].shape
                  or ours[k].tobytes() != theirs[k].tobytes()]
        total, identical = total + len(keys), identical + len(keys) - len(differ)
        refusals = sum(1 for k in keys if k.endswith('/error'))
        print(f'  {label}:')
        print(f'    {len(keys)} arrays ({refusals} refusal records), '
              f'{len(keys) - len(differ)} byte-identical to {rev}, {len(differ)} not')
        if differ:
            worst = max(np.max(np.abs(ours[k] - theirs[k])) / np.max(np.abs(theirs[k]))
                        for k in differ)
            where = sorted({'/'.join(k.split('/')[1:-1]) for k in differ})
            print(f'    largest relative difference {e(worst)}, in ' + ', '.join(where))
    print(f'  in all: {total} arrays, {identical} byte-identical to {rev}, '
          f'{total - identical} not')
    print('  double negatives, glass onto each at normal incidence (lit: theta 0.35, phi 0.6), '
          'R_s or what was raised:')
    for k in sorted(k for k in ours if k.startswith('dng/')):
        def show(v):
            return f'{float(v):.10g}' if v.dtype != np.uint8 else v.tobytes().decode()
        print(f'    {k[4:]:38s} {rev}: {show(theirs[k]):>14s}   now: {show(ours[k]):>14s}')


def c_or_r(value):
    """A material value as written: a real number plainly, a complex one as ``a+bi``."""
    value = complex(value)
    if value.imag == 0:
        return f'{value.real:g}'
    return f'{value.real:g}{value.imag:+g}i'


# --------------------------------------------------------------------------------- 11. usage

def section_usage():
    """The document's section 16, statement for statement, printing what its comments quote."""
    header(11, 'usage -- the examples of the document\'s section 16, as written there')
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:                    # pragma: no cover - environment
        print(f'  skipped: {error}')
        return

    def show(label, *values):
        print(f'  {label}: ' + ', '.join(values))

    with _quiet():
        # As the substrate.
        solver = _legacy_wavebit.rcwa(
            top=2.25, bot=wavebit.Material(-2 + 0.1j, mu=-1 + 0.1j),
            wavelength=500., period=(1000., 1000.), truncation_order=(0, 0),
            structure=np.full((1, 2, 2), 2.25), thickness=[37.],
        )
        result = solver.solve()
        show('substrate: n_bot, R, T', c(solver.n_bot), f'{float(np.sum(result.R)):.10f}',
             f'{float(np.sum(result.T)):.10f}')
        # Lit from it: negative refraction.
        solver = _legacy_wavebit.rcwa(
            top=wavebit.Material(-2.0, mu=-1.0), bot=2.25,
            theta=np.deg2rad(20.), phi=0., polarization=(1., 0.),
            wavelength=500., period=[500.], truncation_order=0,
            structure=np.full((1, 1, 1), 2.25), thickness=[0.],
        )
        result = solver.solve()
        kx = solver.order_kxy()[0][0].real
        show('lit: kx, transmitted angle, R, T', f'{kx:.6f}',
             f'{np.degrees(np.arcsin(kx / 1.5)):.4f}', f'{float(np.sum(result.R)):.7f}',
             f'{float(np.sum(result.T)):.7f}')
        # An absorbing lit medium.
        solver.top = wavebit.Material(-2 + 0.1j, mu=-1 + 0.1j)
        result = solver.solve()
        show('absorbing lit: R + T, R, T, power_entering',
             f'{float(np.sum(result.R)) + float(np.sum(result.T)):.7f}',
             f'{float(np.sum(result.R)):.7f}', f'{float(np.sum(result.T)):.7f}',
             f'{float(np.sum(result.power_entering)):.7f}')
        # Direct kx.
        k0 = 2 * np.pi / 500.
        direct = _legacy_wavebit.rcwa(
            top=wavebit.Material(-2.0, mu=-1.0), bot=2.25, kx=-0.48369 * k0,
            polarization=(1., 0.), wavelength=500., period=[500.], truncation_order=0,
            structure=np.full((1, 1, 1), 2.25), thickness=[0.],
        )
        show('direct kx: kx / k0, R', f'{direct.order_kxy()[0][0].real:.5f}',
             f'{float(np.sum(direct.solve().R)):.7f}')
        # A double-negative layer on the raster path.
        eps = np.full((1, 4, 4), 2.25, dtype=complex)
        mu = np.ones((1, 4, 4), dtype=complex)
        eps[0, 1:3, 1:3], mu[0, 1:3, 1:3] = -3 + 0.1j, -0.5 + 0.05j
        solver = _legacy_wavebit.rcwa(
            top=1., bot=2.25, structure=eps, mu=mu, thickness=[200.],
            wavelength=500., period=(1000., 900.), truncation_order=(2, 2),
            theta=0.3, phi=0.6,
        )
        result = solver.solve()
        show('raster layer: R, T', f'{float(np.sum(result.R)):.5f}',
             f'{float(np.sum(result.T)):.5f}')
        # On a shape path.
        layer = {'background': 2.25,
                 'shapes': [{'type': 'circle', 'cx': 0., 'cy': 0., 'lx': 400.,
                             'material': wavebit.Material(-3 + 0.1j, mu=-0.5 + 0.05j)}]}
        solver = _legacy_wavebit.rcwa(
            top=1., bot=2.25, structure=[layer], thickness=[200.],
            wavelength=500., period=(1000., 900.), truncation_order=(2, 2),
            theta=0.3, phi=0.6,
        )
        result = solver.solve()
        show('shape layer: R, T', f'{float(np.sum(result.R)):.5f}',
             f'{float(np.sum(result.T)):.5f}')
        # The coherent door.
        grating = np.full((1, 4, 4), 2.25, dtype=complex)
        grating[0, :2, :3] = 4.0
        scene = dict(top=wavebit.Material(-3 + 0.1j, mu=-0.5 + 0.05j), bot=2.25,
                     wavelength=500., period=(1000., 900.), truncation_order=(2, 2),
                     structure=grating, thickness=[120.])
        coherent = _legacy_wavebit.rcwa(**scene, excitation={'top': wavebit.PlaneWave(
            theta=np.deg2rad(35.), phi=0.6, psi=0.3)}).solve()
        plain = _legacy_wavebit.rcwa(**scene, theta=np.deg2rad(35.), phi=0.6, psi=0.3).solve()
        show('coherent door: output_top, output_bottom, their sum, absorption',
             f'{float(coherent.output_top):.8f}', f'{float(coherent.output_bottom):.8f}',
             f'{float(coherent.output_top) + float(coherent.output_bottom):.8f}',
             e(abs(float(coherent.absorption))))
        show('the same light through theta: R, T',
             f'{float(np.sum(plain.R)):.8f}', f'{float(np.sum(plain.T)):.8f}')
        # A solve_conv half-space pair.
        nim = (-3 + 0.1j, -0.5 + 0.05j)
        common = dict(top=2.25, wavelength=500., period=(1000., 900.), truncation_order=(2, 2),
                      structure=grating, thickness=[120.], theta=0.3, phi=0.6)
        scene = _legacy_wavebit.rcwa(**common, bot=nim[0] * nim[1])
        paired = scene.solve_conv(*scene.eps_conv_matrices(), eps_bot=nim[0], mu_bot=nim[1])
        magnetic = _legacy_wavebit.rcwa(**common, bot=wavebit.Material(*nim)).solve()
        show('solve_conv pair: t_p equal to the magnetic scene\'s bit for bit, scene.n_bot after',
             str(np.array_equal(paired.t_p, magnetic.t_p)), c(scene.n_bot))
        # Differentiating through it.
        wavebit.enable_x64()

        def reflectance(re_eps):
            medium = wavebit.Material(re_eps + 0.1j, mu=-1 + 0.1j)
            s = _legacy_wavebit.rcwa(backend='jax', top=2.25, bot=medium, wavelength=500.,
                             period=(1000., 1000.), truncation_order=(0, 0),
                             structure=np.full((1, 2, 2), 2.25), thickness=[37.])
            return s.solve().R.sum().real

        h = 1e-6
        show('jax.grad of R in Re(eps), a central difference (h = 1e-6); jax.jit of R',
             f'{float(jax.grad(reflectance)(jnp.asarray(-2.0))):.7f}',
             f'{(float(reflectance(-2.0 + h)) - float(reflectance(-2.0 - h))) / (2 * h):.7f}',
             f'{float(jax.jit(reflectance)(jnp.asarray(-2.0))):.10f}')

        # Under jax.jit: the gain band is what is left to read.
        def band(re_eps):
            s = _legacy_wavebit.rcwa(backend='jax', top=2.25,
                             bot=wavebit.Material(re_eps - 0.1j, mu=-1 - 0.1j),
                             theta=np.deg2rad(60.), wavelength=500., period=[700.],
                             truncation_order=3, thickness=[180.],
                             structure=np.array([[[4., 4., 1., 1., 2.25, 2.25]]]))
            return s.gain_cutoff_band()['bottom']

        flagged = np.asarray(jax.jit(band)(jnp.asarray(-2.0)))
        try:
            band(-2.0)
            eager = 'accepted'
        except ValueError as error:
            eager = 'ValueError: ' + str(error)[:62] + '...'
        show('jit gain band; eager', str(flagged.ravel().astype(int).tolist()), eager)


# ---------------------------------------------------------------------- 12. the negation pair

PAIR = (2.25 + 0j, 1.0 + 0j)             # glass; below it, its exact negation (-2.25, -1)


def _negated(pair, delta=0.0):
    """``(-eps, -mu)``, with a loss ``delta`` added to both."""
    return (-pair[0] + 1j * delta, -pair[1] + 1j * delta)


@contextlib.contextmanager
def _negation_refusal_off():
    """The engine's refusal of the pair switched off, to show what a solve does without it."""
    CoreFactorizationMixin = _study_module('wavebit.rcwa._engine.engine', 'wavebit._core.engine').CoreFactorizationMixin
    original = CoreFactorizationMixin._refuse_half_space_negation_pair
    CoreFactorizationMixin._refuse_half_space_negation_pair = lambda self, *a, **k: None
    try:
        yield
    finally:
        CoreFactorizationMixin._refuse_half_space_negation_pair = original


def _pair_scene(path, algo, bot, layers, side='top', theta=0.2, backend='numpy', period=300.0,
                truncation=None, solver_only=False):
    """Glass over ``bot``, 500 nm, period 300 nm: at ``theta = 0.2`` the retained orders include
    evanescent ones in both media (printed by section 12), and one first order propagates.
    ``layers`` is a list of ``((eps, mu), thickness)``, top down; ``None`` solves the bare
    interface (``solve_conv`` with no layer). Returns the call that solves it, or with
    ``solver_only`` the solver."""
    one_d = path in ('te', 'tm', 'conical')
    shape = (1, 8) if one_d else (4, 4)
    stack = layers or [(PAIR, 10.0)]
    common = dict(backend=backend_arg(backend), top=wavebit.Material(*PAIR),
                  bot=wavebit.Material(*bot), theta=theta, wavelength=WL,
                  thickness=[d for _, d in stack][::-1],
                  layer_algorithm=algo, answers='amplitude', incidence_side=side,
                  structure=np.stack([np.full(shape, m[0]) for m, _ in stack][::-1]),
                  mu=np.stack([np.full(shape, m[1]) for m, _ in stack][::-1]))
    one_d = path in ('te', 'tm', 'conical')
    truncation = (3 if one_d else (1, 1)) if truncation is None else truncation
    if path in ('te', 'tm'):
        s = _legacy_wavebit.rcwa(**common, tetm=path, period=[period], truncation_order=truncation)
    elif path == 'conical':
        s = _legacy_wavebit.rcwa(**common, phi=0.6, psi=0.3, period=[period],
                         truncation_order=truncation)
    else:
        s = _legacy_wavebit.rcwa(**common, phi=0.6, psi=0.3, period=(period, 0.9 * period),
                         truncation_order=truncation)
        if path == 'first_order':
            s.formulation = 'first_order'
    if solver_only:
        return s
    if layers:
        return lambda: s.solve(auto_detune=False)

    def bare():
        e_conv, m_conv = s.eps_conv_matrices(), s.mu_conv_matrices()
        s.thickness = []
        return s.solve_conv(*(None if a is None else a[:0] for a in e_conv),
                            mu=tuple(None if a is None else a[:0] for a in m_conv),
                            eps_top=s.top.epsilon, mu_top=s.top.mu, eps_bot=s.bot.epsilon,
                            mu_bot=s.bot.mu)
    return bare


def _incident(result):
    return [sub for sub in (getattr(result, n, None) for n in ('incident', 's_inc', 'p_inc'))
            if sub is not None]


def _pair_outcome(call):
    """``(what happened, largest amplitude off the specular order, R, T)``. A uniform stack
    excites the specular order only, so the second is round-off wherever it is not zero."""
    try:
        with _quiet():
            result = call()
    except Exception as error:                      # noqa: BLE001 - its class is the answer
        text = str(error)
        if "each other's exact negation" in text:
            return 'refused', None, None, None
        if 'perfect-lens' in text:
            return 'the lens message', None, None, None
        return type(error).__name__.lstrip('_'), None, None, None
    amplitudes = [np.ravel(_host(getattr(sub, f))) for sub in _incident(result)
                  for f in ('r_s', 'r_p', 't_s', 't_p') if getattr(sub, f, None) is not None]
    if not all(np.all(np.isfinite(a)) for a in amplitudes):
        return 'non-finite', None, None, None
    off = max(float(np.max(np.abs(np.delete(a, a.size // 2)))) if a.size > 1 else 0.0
              for a in amplitudes)
    sub = _incident(result)[0]
    return 'finite', off, float(np.sum(_host(sub.R)).real), float(np.sum(_host(sub.T)).real)


PAIR_STACKS = {'nothing': None, 'a glass layer': [(PAIR, 100.0)],
               'a (-2.25, -1) layer': [(_negated(PAIR), 100.0)],
               'both': [(_negated(PAIR), 100.0), (PAIR, 60.0)]}


def section_negation_pair():
    header(12, "the negation pair -- two lossless half-spaces that are each other's negation")
    try:
        import jax                                  # noqa: F401
        import torch                                # noqa: F401
        backends = (('numpy', 'NumPy'), ('jax', 'JAX'), ('torch', 'PyTorch'))
        wavebit.enable_x64()
    except ImportError as error:                    # pragma: no cover - environment
        print(f'  JAX and PyTorch skipped: {error}')
        backends = (('numpy', 'NumPy'),)
    cases = [(name, stack, path, algo, side) for name, stack in PAIR_STACKS.items()
             for side in ('top', 'bottom') for path in PATHS for algo in ('ETM', 'SMM')]
    order_grid_kxy = _study_module('wavebit.rcwa._engine.solve_2d', 'wavebit._core.solve_2d').order_grid_kxy
    # Which retained orders are evanescent: kz**2 = eps*mu - k_par**2 is the same in both media.
    tally, first = collections.defaultdict(list), set()
    for path in PATHS:
        for side in ('top', 'bottom'):
            s = _pair_scene(path, 'SMM', _negated(PAIR), None, side, solver_only=True)
            gx, gy = (np.real(np.asarray(g)) for g in order_grid_kxy(np, *s.order_kxy()))
            evanescent = gx * gx + gy * gy > (PAIR[0] * PAIR[1]).real
            tally[(int(evanescent.sum()), evanescent.size)].append(path)
            # The order label m along x of each propagating order off the specular one.
            centre = gx[gx.shape[0] // 2, gx.shape[1] // 2]
            labels = np.rint((gx - centre) / (WL / 300.0)).astype(int)
            first.update((side, int(m)) for m in labels[~evanescent] if m != 0)
    print("  glass (2.25, 1) over (-2.25, -1), 500 nm, period 300 nm, theta = 0.2: of the retained "
          'orders ' + ' and '.join(f'{n} of {total} ({", ".join(sorted(set(p), key=PATHS.index))})'
                                   for (n, total), p in sorted(tally.items()))
          + ' are evanescent in both media, and one first order propagates in both ('
          + ', '.join(f'{m:+d} lit from {"glass" if side == "top" else "the double negative"}'
                      for side, m in sorted(first, reverse=True))
          + '); between them ' + ', '.join(PAIR_STACKS)
          + f'; 5 paths x ETM / SMM x either side lit: {len(cases)} scenes')
    for backend, label in backends:
        refused = sum(_pair_outcome(_pair_scene(path, algo, _negated(PAIR), stack, side,
                                                backend=backend))[0] == 'refused'
                      for _, stack, path, algo, side in cases)
        print(f'    {label:7s} refused: {refused} of {len(cases)}')
    print('  the same scenes with no evanescent order retained -- the specular order alone, or a '
          'period of 3000 nm at the same truncation: solved, not refused')
    for backend, label in backends:
        kinds, R_max, T_gap, gap = collections.Counter(), 0.0, 0.0, 0.0
        for _, stack, path, algo, side in cases:
            for period, truncation in ((300.0, 0 if path in ('te', 'tm', 'conical') else (0, 0)),
                                       (3000.0, None)):
                kind, _, R, T = _pair_outcome(_pair_scene(
                    path, algo, _negated(PAIR), stack, side, backend=backend, period=period,
                    truncation=truncation))
                kinds[kind] += 1
                if kind != 'finite':
                    continue
                R_max, T_gap = max(R_max, R), max(T_gap, abs(T - 1))
                if backend == 'numpy':
                    lossy = _negated(PAIR, 1e-12)
                    stack_d = None if stack is None else [(lossy if m[0].real < 0 else m, d)
                                                          for m, d in stack]
                    with _quiet():
                        exact = _pair_scene(path, algo, _negated(PAIR), stack, side,
                                            period=period, truncation=truncation)()
                        near = _pair_scene(path, algo, lossy, stack_d, side, period=period,
                                           truncation=truncation)()
                    gap = max(gap, max(np.abs(_host(getattr(a, f)) - _host(getattr(b, f))).max()
                                       for a, b in zip(_incident(exact), _incident(near))
                                       for f in ('r_s', 'r_p', 't_s', 't_p')
                                       if getattr(a, f, None) is not None))
        neighbour = (f'; every amplitude within {e(gap)} of the same scene with a loss of 1e-12'
                     if backend == 'numpy' else '')
        print(f'    {label:7s} ' + ', '.join(f'{kind} {count}' for kind, count in kinds.items())
              + f' of {sum(kinds.values())}; R <= {R_max:.0e}, |T - 1| <= {T_gap:.0e}{neighbour}')
    print('  the same scenes with the refusal switched off -- what a solve does without it:')
    with _negation_refusal_off():
        for backend, label in backends:
            for group in ('nothing', 'layers'):
                tally = collections.Counter()
                where = collections.defaultdict(set)
                off, powers = [], []
                for name, stack, path, algo, side in cases:
                    if (name == 'nothing') != (group == 'nothing'):
                        continue
                    kind, amplitude, R, T = _pair_outcome(_pair_scene(
                        path, algo, _negated(PAIR), stack, side, backend=backend))
                    tally[(algo, kind)] += 1
                    where[(algo, kind)].add(path)
                    if kind == 'finite':
                        powers.append((R, abs(T - 1)))
                    if kind == 'finite' and path == 'first_order' and algo == 'SMM':
                        off.append(amplitude)
                # Name the paths where only some of them answered one way.
                text = '; '.join(
                    f'{algo} {kind} {count}' + (
                        f' ({", ".join(p for p in PATHS if p in where[(algo, kind)])})'
                        if len(where[(algo, kind)]) < len(PATHS) else '')
                    for (algo, kind), count in sorted(tally.items()))
                wrong = [a for a in off if a > 1e-6]
                extra = (f'; every finite answer R <= {max(R for R, _ in powers):.0e}, |T - 1| <= '
                         f'{max(t for _, t in powers):.0e}' if powers else '')
                extra += (f'; of SMM\'s {len(off)} finite answers on the first-order path, '
                          f'{len(wrong)} carry {min(wrong):.2g} .. {max(wrong):.2g} in orders a '
                          f'uniform stack does not excite' if wrong else '')
                print(f'    {label:7s} {"nothing between" if group == "nothing" else "layers":16s}'
                      f'{text}{extra}')
    print('  a loss delta in the double negative and its layer, (-2.25+i delta, -1+i delta), '
          'NumPy, every scene above:')
    for delta in (1e-12, 1e-9, 1e-6, 1e-3):
        kinds = collections.Counter()
        R_bare, T_bare, R_all, gaps = [], [], [], {}
        for name, stack, path, algo, side in cases:
            if algo == 'SMM':
                continue
            pair = _negated(PAIR, delta)
            stack_d = None if stack is None else [(pair if m[0].real < 0 else m, d)
                                                  for m, d in stack]
            got = []
            for a in ('ETM', 'SMM'):
                kind, _, R, T = _pair_outcome(_pair_scene(path, a, pair, stack_d, side))
                kinds[kind] += 1
                if kind == 'finite':
                    got.append((R, T))
                    R_all.append(R)
                    if stack is None:
                        R_bare.append(R)
                        T_bare.append(T)
            if len(got) == 2:
                gap = max(abs(got[0][0] - got[1][0]), abs(got[0][1] - got[1][1]))
                gaps[(path, name)] = max(gaps.get((path, name), 0.0), gap)
        (path, name), worst = max(gaps.items(), key=lambda item: item[1])
        rest = max(v for k, v in gaps.items() if k != (path, name))
        T_gap = max(abs(1 - t) for t in T_bare)
        print(f'    delta = {delta:.0e}: solved {kinds["finite"]} of {sum(kinds.values())}; '
              f'the bare interface R {min(R_bare):.1e} .. {max(R_bare):.1e}, T = 1 to '
              f'{e(T_gap)}; with layers R up to {max(R_all):.1e}; ETM against SMM '
              f'{e(worst)} ({path}, {name}), elsewhere {e(rest)}')
    print('  lit by an evanescent wave from glass (theta = pi/2 - 0.5i), a glass layer between, '
          'SMM: delta * max|r_s|, the incident order on the pole')
    for path in ('te', '2d', 'first_order'):
        row = []
        for delta in (1e-3, 1e-6, 1e-9, 1e-12):
            with _quiet():
                result = _pair_scene(path, 'SMM', _negated(PAIR, delta), [(PAIR, 100.0)],
                                     theta=np.pi / 2 - 0.5j)()
            row.append(delta * np.max(np.abs(_host(_incident(result)[0].r_s))))
        with _negation_refusal_off(), _quiet():
            result = _pair_scene(path, 'SMM', _negated(PAIR), [(PAIR, 100.0)],
                                 theta=np.pi / 2 - 0.5j)()
        at_zero = np.max(np.abs(_host(_incident(result)[0].r_s)))
        print(f'    {path:4s} delta = 1e-3 / 1e-6 / 1e-9 / 1e-12: '
              + ' / '.join(f'{v:.4f}' for v in row)
              + f'; delta = 0 with the refusal off: max|r_s| = {at_zero:.0e}')


# ------------------------------------------------------------------------------ 13. literature

def section_literature():
    """The claims of the document's section 18 that a number can show, each computed here."""
    boundary_kz = _study_module('wavebit._common.conventions', 'wavebit._core.conventions').boundary_kz
    header(13, 'literature -- what the document\'s section 18 quotes, computed')

    def maystre_root(z):
        return maystre(complex(z))

    def describe(kz, mu):
        kz = complex(kz)
        return (f'{c(kz)} ({"decays" if kz.imag > 0 else "grows"} away from the interface; '
                f'Re(kz/mu) {(kz / mu).real:+.4f}: power '
                f'{"away from" if (kz / mu).real > 0 else "toward"} it)')

    # Lalanne and Hugonin (2026) state Re + Im > 0 for complex frequency; a lossless Drude-Drude
    # double negative (Cassier, Hazard and Joly's model) at a complex frequency, normal incidence.
    omega = 0.5 + 0.01j
    eps = mu = 1 - 1 / omega ** 2                  # Omega_e = Omega_m = 1, units with c = 1
    kz2 = omega ** 2 * eps * mu
    mirror_kz = complex(boundary_kz(np, np.asarray(kz2), 0.0, medium=(eps, mu)))
    decaying = np.sqrt(kz2) if np.sqrt(kz2).imag > 0 else -np.sqrt(kz2)
    print(f'  lossless Drude-Drude double negative (Omega_e = Omega_m = 1) at omega = '
          f'{c(omega, 2)}: eps = mu = {c(eps)}, kz**2 = {c(kz2, 3)}')
    print(f'    Re + Im > 0 (Maystre\'s cut on kz**2): {c(maystre_root(kz2), 3)}; the mirror '
          f'(boundary_kz): {c(mirror_kz, 3)}; the decaying root (Im omega > 0): '
          f'{c(decaying, 3)}')
    # Gralak (2020): a passive homogenized medium can have Im(mu) < 0.
    eps, mu = -2 + 0.1j, -1 - 0.5j
    kz2 = eps * mu
    print(f'  Im(eps) and Im(mu) of opposite signs, eps = {c(eps, 1)}, mu = {c(mu, 1)}: '
          f'kz**2 = {c(kz2, 2)} at normal incidence')
    mirror_kz = boundary_kz(np, np.asarray(kz2), 0.0, (eps, mu))
    print(f'    the mirror root     {describe(mirror_kz, mu)}')
    print(f'    Maystre\'s root      {describe(maystre_root(kz2), mu)}')
    try:
        solver(bot=wavebit.Material(eps, mu=mu))
        verdict = 'accepted'
    except ValueError as error:
        verdict = 'ValueError: ' + str(error)[:62] + '...'
    print(f'    as bot: {verdict}')
    # A uniaxial pair no per-medium switch serves (the optic axis along z).
    ex, ey, ez = 1.0, 1.0, -1.0
    mx, my, mz = -1.0, -1.0, 1.0
    kx = 1.5
    te = mx * (ey - kx ** 2 / mz)                  # kz**2 of the s wave, E along y
    tm = ex * (my - kx ** 2 / ez)                  # kz**2 of the p wave, H along y
    out_te = np.sqrt(te) * np.sign(mx)             # power along +z: Re(kz/mu_x) > 0
    out_tm = np.sqrt(tm) * np.sign(ex)             # power along +z: Re(kz/eps_x) > 0
    print(f'  eps = diag({ex:g}, {ey:g}, {ez:g}), mu = diag({mx:g}, {my:g}, {mz:g}), kx = {kx}: '
          f'kz**2 = {te:g} (s) and {tm:g} (p); the root carrying power away is {out_te:+.4f} '
          f'(s) and {out_tm:+.4f} (p) -- one kz**2, two roots')
    # Signed zero: the principal root, backend by backend.
    rows = [f'cmath {c(cmath.sqrt(complex(-1.0, -0.0)), 0)}',
            f'NumPy {c(np.sqrt(np.complex128(complex(-1.0, -0.0))), 0)}']
    try:
        import jax
        import jax.numpy as jnp
        import torch
        rows += [f'JAX {jax.__version__} {c(jnp.sqrt(jnp.asarray(complex(-1.0, -0.0))), 0)}',
                 f'PyTorch {torch.__version__.split("+")[0]} '
                 f'{c(torch.sqrt(torch.tensor(complex(-1.0, -0.0))).item(), 0)}']
    except ImportError:                              # pragma: no cover - environment
        pass
    print('  the principal sqrt of complex(-1, -0.0): ' + ', '.join(rows))
    # Wavebit's own index of it, each backend's array (`backend.principal_sqrt`).
    refractive_index_from_eps = wavebit.refractive_index_from_eps
    rows = [f'Python number {c(refractive_index_from_eps(complex(-1.0, -0.0)), 0)}',
            f'NumPy {c(complex(refractive_index_from_eps(np.asarray(complex(-1.0, -0.0)))), 0)}']
    try:
        import jax.numpy as jnp
        import torch
        rows += [f'JAX {c(complex(refractive_index_from_eps(jnp.asarray(complex(-1.0, -0.0)))), 0)}',
                 f'PyTorch {c(refractive_index_from_eps(torch.tensor(complex(-1.0, -0.0))).item(), 0)}']
    except ImportError:                              # pragma: no cover - environment
        pass
    print('  refractive_index_from_eps of complex(-1, -0.0): ' + ', '.join(rows))
    rng = np.random.default_rng(13)
    values = (rng.uniform(-3, 3, 2000) + 1j * rng.uniform(-1, 1, 2000)) * (
        rng.uniform(-3, 3, 2000) + 1j * rng.uniform(-1, 1, 2000))

    def ulps(a, b):
        """The larger component gap between two complex128 values, in ulp."""
        return max(abs(int(np.float64(x).view(np.int64)) - int(np.float64(y).view(np.int64)))
                   for x, y in ((a.real, b.real), (a.imag, b.imag)))

    gaps = collections.Counter(ulps(cmath.sqrt(complex(v)), complex(np.sqrt(v))) for v in values)
    differ = sum(count for gap, count in gaps.items() if gap)
    print(f'  cmath.sqrt against np.sqrt on 2000 random products eps*mu: {differ} differ; the '
          'larger gap of the two components is '
          + ' and '.join(f'{gap} ulp on {count}' for gap, count in sorted(gaps.items()) if gap))


# ------------------------------------------------------------------------------------- main

SECTIONS = {1: section_roots, 2: section_mirror, 3: section_index, 4: section_theta,
            5: section_amplitudes, 6: section_layers, 7: section_refusals, 8: section_oracles,
            9: section_backends}
#: Printed after sections 10 and 11, so that their numbers keep the places they had.
LATER_SECTIONS = {12: section_negation_pair, 13: section_literature}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--develop', metavar='REV', help='section 10: compare with this commit')
    parser.add_argument('--only', type=int, nargs='*', help='run these sections only')
    parser.add_argument('--dump', help=argparse.SUPPRESS)
    parser.add_argument('--root', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dump:
        out = harness()
        out.update({f'dng/{k}': v for k, v in double_negative_outcomes().items()})
        np.savez(args.dump, **out)
        return
    np.set_printoptions(precision=6)
    print('Negative-index media -- studies/rcwa/negative_index_media/negative_index_media.py')
    print(f'numpy {np.__version__}; every solve NumPy complex128 except sections 9 and 11')
    for number, section in SECTIONS.items():
        if not args.only or number in args.only:
            section()
    if not args.only or 10 in args.only:
        section_develop(args.develop)
    if not args.only or 11 in args.only:
        section_usage()
    for number, section in LATER_SECTIONS.items():
        if not args.only or number in args.only:
            section()


if __name__ == '__main__':
    main()
