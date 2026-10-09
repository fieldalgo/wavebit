"""What does SMM's field state get right, and what did each of its versions get wrong?

SMM's cascade answers at the outer ports only; field reconstruction needs each layer's own mode
amplitudes, which ``scattering._field_state`` recovers from the cascade's split-plane amplitudes.
That state has had three versions on develop:

* ``b9957c8`` (before 2026-10-09): the ETM's walk over each layer's ``+`` amplitude, advanced
  through the inverse of ``M``, the map from the gap below a layer onto those amplitudes, on the
  cascade's own decaying labels. ``M`` is singular where a ``+`` mode is not a wave the region
  below receives -- at a complex angle, in a gain layer, at single precision.
* ``429f4ac`` (2026-10-09, on develop through ``07ae0f6``): the same walk, with the layers where
  the two label rules may part rebuilt on the ETM's labels -- a second ``eig`` on an identical
  operator. Still an inverse of ``M``, now cancelling where a root grows and a rounding residue
  where a capped root meets an interface that reflects nothing.
* this tree: each layer's ``+`` and ``-`` amplitudes read off the gap amplitude between layers
  by forward maps, with no inverse of anything that maps into a layer, on the cascade's labels;
  a mode with ``|k0 q d| < 1`` has both read off the layer's top face.

  1 closed form  reflectionless uniform layers, one plane wave each: thin at a complex angle,
                 passive and gain at a real one, magnetic, lit from below, and three thick enough
                 that the ETM's cap turns a root. Worst residual over s and p, all six
                 components, relative to the wave where it is sampled (past the cap the largest
                 one is ``G`` times larger), on the SMM 2ff paths (conical, 2D, TE, TM) and on
                 every other path.
  2 gratings     ``[1, 25, 1, 25, 25, 25]`` on ``eps = 25`` (2D: a 6x6 raster), two layers, at
                 ``theta = 0.5+0.3j``: SMM's field against the ETM's, 160 nm and 4.8 um.
  3 grows below  8.5 um of vacuum under 500 nm of ``eps = 4`` on 2.25, ``theta = 0.5+0.3j``: a root
                 growing by ``e**8.66`` below another layer. TE field against a 60-digit closed
                 form, ETM and SMM.
  4 own medium   4869 nm of ``eps = 2.25`` inside ``eps = 2.25``, lit from below at a complex angle,
                 ``truncation_order = 7``: SMM against ETM, and under a 1e-14 nudge of ``theta``.
  5 real angle   what the relabel touched at a real angle: a lossy metal raster (oblique and
                 normal), an active patterned double negative, single precision.
  6 below axis   a uniform double negative on its own medium at ``theta = 0.5-0.3j``.
  7 rungs        amplitudes between ``answers='amplitude'`` and ``'field'``, at a complex angle and
                 for 10 um of vacuum on vacuum, where ``M`` rounds to zero.
  8 graze        orders +-1 exactly on a spacer's light line, ``q = 0``, which ``auto_detune``
                 lands 1e-12 off: SMM's field against the ETM's, worst of TE, TM, conical, 2D.
  9 ports        a label-free oracle: tangential ``E`` and ``H`` continuous at every interface,
                 and equal at both outer faces to the half-space fields ``R`` and ``T`` imply.
                 Three-layer gratings, worst of TE, TM, conical, 2D, lit from either side.
 10 cost         builder calls and retained blocks of an SMM field solve, at a real and a complex
                 angle; with ``--time``, also its time against the ETM's (left out of the log,
                 which is byte-for-byte reproducible).

Units: lengths in nm, wavevectors in units of k0. Time convention exp(-i w t). NumPy complex128
unless a row says otherwise, ``auto_detune=False``. Run from the repository root:

    python studies/smm_field_state.py --compare b9957c8 07ae0f6

It prints the log committed as ``studies/results/smm_field_state.log`` in about ten seconds.
``--compare`` extracts each revision with ``git archive`` into a temporary directory and runs
sections 1 to 9 there in a subprocess, by this same script, so each row carries one column per
tree; section 10 is this tree's only. ``docs/connecting-algorithms.md`` section 10 ("SMM's field
state is read off the gap") has the reading.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import mpmath as mp
import numpy as np

import wavebit

warnings.simplefilter('ignore')
mp.mp.dps = 60

WL, PHI = 900.0, 0.3
THETA = 0.5 + 0.3j
CHIP_THETA = 0.611565803730562 + 0.34349515827119764j


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


def maystre(x):
    v = np.sqrt(complex(x))
    return v if v.real + v.imag > 0 else -v


def guarded(fn):
    """A measurement, or the name of what it raised."""
    try:
        return fn()
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return type(exc).__name__


# --------------------------------------------------------------------------- 1 closed form

SCENES = [  # label, theta, eps, mu, side, incidence eps, thickness
    ('vacuum, 100 nm, 0.5+0.3j', THETA, 1.0, 1.0, 'top', 1.0, 100.0),
    ('eps 4 on itself, 100 nm, real 0.5', 0.5, 4.0, 1.0, 'top', 1.0, 100.0),
    ('eps 25 on itself, 100 nm, 0.5+0.3j', THETA, 25.0, 1.0, 'top', 1.0, 100.0),
    ('gain 25-0.5i on itself, 100 nm, real 0.5', 0.5, 25.0 - 0.5j, 1.0, 'top', 1.0, 100.0),
    ('vacuum lit from below, 100 nm, 0.5+0.3j', THETA, 1.0, 1.0, 'bottom', 1.0, 100.0),
    ('eps 4 mu 2.25 on itself, 100 nm, 0.5+0.3j', THETA, 4.0, 2.25, 'top', 1.0, 100.0),
    ('vacuum, 10 um, 0.5+0.3j (G 2.7e4)', THETA, 1.0, 1.0, 'top', 1.0, 1e4),
    ('eps 2.13 on itself, 20 um, 0.5+0.3j (G 6.3e5)', THETA, 2.13, 1.0, 'top', 1.0, 2e4),
    ('eps 2.25 in itself, 4869 nm, from below (G 2.8e4)', CHIP_THETA, 2.25, 1.0, 'bottom', 2.25,
     4869.0),
]
PATHS = [('conical', '2ff'), ('2d', '2ff'), ('te', None), ('tm', None), ('conical', '4ff'),
         ('2d', '4ff')]


def plane_wave_worst(scene, kind, form, algo):
    _, theta, eps, mu, side, eps_inc, thick = scene
    layer = eps if mu == 1.0 else wavebit.Material(eps, mu=mu)
    top, bot = (eps_inc, layer) if side == 'top' else (layer, layer)
    kw = dict(backend=0, top=top, bot=bot, theta=theta, wavelength=WL, thickness=[thick],
              layer_algorithm=algo, answers='field', incidence_side=side)
    grid = (1, 2, 2) if kind == '2d' else (1, 1, 4)
    if kind == '2d':
        kw.update(psi=np.pi / 2, phi=PHI, truncation_order=[0, 0], period=[2000., 2000.])
    elif kind == 'conical':
        kw.update(psi=np.pi / 2, phi=PHI, truncation_order=[0], period=[2000.])
    else:
        kw.update(tetm=kind, truncation_order=[0], period=[2000.])
    solver = wavebit.rcwa(**kw)
    solver.structure = np.full(grid, eps, dtype=complex)
    if mu != 1.0:
        solver.mu = np.full(grid, mu, dtype=complex)
    if form is not None:
        solver.formulation = form
    solver.solve(auto_detune=False)
    z = np.linspace(0.05, 0.95, 5) * thick
    if kind in ('te', 'tm'):
        c = np.asarray(solver.reconstruct_field(sample_x=[0.], sample_y=[0.], sample_z=z))[:, 0, 0]
        zero = np.zeros_like(c[:, 0])
        cells = ({'s': (zero, c[:, 0], zero, c[:, 1], zero, c[:, 2])} if kind == 'te' else
                 {'p': (c[:, 1], zero, c[:, 2], zero, c[:, 0], zero)})
    else:
        c = np.asarray(solver.reconstruct_field(sample_x=[0.], sample_y=[0.], sample_z=z,
                                                incidences=('s', 'p')))[:, :, 0, 0]
        cells = {pol: tuple(c[i, :, j] for j in range(6)) for i, pol in enumerate('sp')}
    n_inc = np.sqrt(complex(eps_inc))
    kpar = n_inc * np.sin(theta)
    kx, ky = (kpar * np.cos(PHI), kpar * np.sin(PHI)) if kind in ('conical', '2d') else (kpar, 0.0)
    kz_i, kz = maystre(eps_inc - kpar ** 2), maystre(eps * mu - kpar ** 2)
    k0 = 2 * np.pi / WL
    if side == 'top':
        travel, kzu = np.exp(1j * k0 * kz * (thick - z)), -kz
        t = {'s': 2 * kz_i / (kz_i + kz / mu),
             'p': n_inc * 2 * (kz_i / eps_inc) / (kz_i / eps_inc + kz / eps)}
    else:
        travel, kzu, t = np.exp(1j * k0 * kz * z), kz, {'s': 1.0, 'p': n_inc}
    worst = 0.0
    for pol, (Ex, Ey, Ez, Hx, Hy, Hz) in cells.items():
        lead = t[pol] * travel
        sc = np.abs(lead)
        if pol == 's':
            plain = [Ez, kx * Ex + ky * Ey, Hx - (ky * Ez - kzu * Ey) / mu,
                     Hy - (kzu * Ex - kx * Ez) / mu, Hz - (kx * Ey - ky * Ex) / mu]
            square = Ex ** 2 + Ey ** 2 - lead ** 2
        else:
            plain = [Hz, kx * Hx + ky * Hy, Ex + (ky * Hz - kzu * Hy) / eps,
                     Ey + (kzu * Hx - kx * Hz) / eps, Ez + (kx * Hy - ky * Hx) / eps]
            square = Hx ** 2 + Hy ** 2 - lead ** 2
        worst = max([worst, float(np.max(np.abs(square) / sc ** 2))]
                    + [float(np.max(np.abs(r) / sc)) for r in plain])
    return worst


def section_closed_form():
    rows = {}
    for scene in SCENES:
        smm = [guarded(lambda: plane_wave_worst(scene, k, f, 'SMM'))
               for k, f in PATHS if f != '4ff']
        others = ([guarded(lambda: plane_wave_worst(scene, k, f, 'ETM')) for k, f in PATHS]
                  + [guarded(lambda: plane_wave_worst(scene, k, f, 'SMM'))
                     for k, f in PATHS if f == '4ff'])
        rows[f'{scene[0]}: SMM 2ff/TE/TM'] = worst_of(smm)
        rows[f'{scene[0]}: every other path'] = worst_of(others)
    return rows


def worst_of(values):
    raised = [v for v in values if isinstance(v, str)]
    return raised[0] if raised else max(values)


# ------------------------------------------------------------------------------ 2 gratings

GRATING = np.array([[[0, 1, 0, 1, 1, 1]], [[1, 0, 0, 1, 1, 1]]]) * 24 + 1.
RASTER = np.ones((2, 6, 6)) * 25.0
RASTER[:, 1:3, 2:5] = 1.0
RASTER[1, 3:5, 0:2] = 1.0
CASES = {
    'conical': dict(psi=np.pi / 2, phi=PHI, structure=GRATING, truncation_order=[15],
                    period=[2000.]),
    'te': dict(tetm='te', structure=GRATING, truncation_order=[15], period=[2000.]),
    'tm': dict(tetm='tm', structure=GRATING, truncation_order=[15], period=[2000.]),
    '2d': dict(psi=np.pi / 2, phi=PHI, structure=RASTER, truncation_order=[3, 3],
               period=[1200., 1200.]),
}


def grating_field(algo, case, thickness, theta=THETA, answers='field', width=0):
    solver = wavebit.rcwa(backend=0, top=1.0, bot=25.0, theta=theta, wavelength=WL,
                          thickness=list(thickness), layer_algorithm=algo, answers=answers,
                          complex_dtype=width, **CASES[case])
    res = solver.solve(auto_detune=False)
    if answers != 'field':
        return res
    inc = ('incident',) if case in ('te', 'tm') else ('incident', 's', 'p')
    return np.asarray(solver.reconstruct_field(sample_x=6, sample_y=6, sample_z=4, incidences=inc))


def section_gratings():
    rows = {}
    for thickness in ((100., 60.), (3000., 1800.)):
        rows[f'{sum(thickness):g} nm, SMM vs ETM, worst of conical/te/tm/2d'] = worst_of([
            guarded(lambda: rel(grating_field('SMM', c, thickness),
                                grating_field('ETM', c, thickness)))
            for c in CASES])
    return rows


# --------------------------------------------------------------------------- 3 grows below

def closed_te(layers, bot, zs, top=1.0, theta=THETA):
    """TE ``E_y`` at public heights ``zs`` (bottom 0), unit incident amplitude at the top face.

    ``layers`` bottom-first, as ``thickness`` and ``structure`` take them. Propagates
    ``(E_y, Y)`` up from a substrate holding only its downward wave, in 60 digits.
    """
    k0 = 2 * mp.pi / WL
    kp2 = (mp.sin(mp.mpc(theta)) * mp.sqrt(top)) ** 2

    def kz(eps):
        v = mp.sqrt(mp.mpc(eps) - kp2)
        return v if mp.re(v) + mp.im(v) > 0 else -v
    F = mp.mpc(1)
    Y = -kz(bot) * F
    z0, segments = mp.mpf(0), []
    for eps, d in layers:
        k = kz(eps)
        U, D = (F + Y / k) / 2, (F - Y / k) / 2
        segments.append((z0, z0 + d, k, U, D))
        F = D * mp.exp(-1j * k0 * k * d) + U * mp.exp(1j * k0 * k * d)
        Y = k * (U * mp.exp(1j * k0 * k * d) - D * mp.exp(-1j * k0 * k * d))
        z0 = z0 + d
    k = kz(top)
    D0 = (F - Y / k) / 2
    out = []
    for z in zs:
        for lo, hi, k, U, D in segments:
            if lo <= z <= hi:
                dz = mp.mpf(z) - lo
                wave = D * mp.exp(-1j * k0 * k * dz) + U * mp.exp(1j * k0 * k * dz)
                out.append(complex(wave / D0))
                break
    return np.array(out)


def section_grows_below():
    layers = [(1.0, 8500.), (4.0, 500.)]
    zs = np.linspace(1.0, 8999.0, 41)
    ref = closed_te(layers, 2.25, zs)
    rows = {}
    for algo in ('ETM', 'SMM'):
        def te():
            s = wavebit.rcwa(backend=0, top=1.0, bot=2.25, theta=THETA, wavelength=WL, tetm='te',
                             thickness=[d for _, d in layers], truncation_order=[0], period=[2000.],
                             structure=np.array([[[e] * 4] for e, _ in layers], dtype=complex),
                             layer_algorithm=algo, answers='field')
            s.solve(auto_detune=False)
            got = np.asarray(s.reconstruct_field(sample_x=[0.], sample_y=[0.],
                                                 sample_z=zs))[:, 0, 0, 0]
            return rel(got, ref)
        rows[f'{algo} TE vs 60-digit closed form'] = guarded(te)
    return rows


# --------------------------------------------------------------------------- 4 own medium

def chip(algo, form='2ff', nudge=1.0):
    s = wavebit.rcwa(backend=0, top=2.25, bot=2.25, theta=CHIP_THETA * nudge, phi=0.41, psi=0.27,
                     wavelength=1000., period=[1654.489947568981], truncation_order=[7],
                     thickness=[4869.], structure=np.full((1, 1, 8), 2.25 + 0j),
                     layer_algorithm=algo, answers='field', incidence_side='bottom')
    s.formulation = form
    s.solve(auto_detune=False)
    return np.asarray(s.reconstruct_field(sample_x=5, sample_y=1, sample_z=6,
                                          incidences=('s', 'p')))


def section_own_medium():
    return {'SMM vs ETM': guarded(lambda: rel(chip('SMM'), chip('ETM'))),
            'SMM, 1e-14 nudge of theta': guarded(
                lambda: rel(chip('SMM', nudge=1 + 1e-14), chip('SMM'))),
            'ETM vs 4ff (4ff floor)': guarded(lambda: rel(chip('ETM'), chip('ETM', '4ff')))}


# ---------------------------------------------------------------------------- 5 real angle

def raster(eps_hi, eps_lo=1.0, n=16):
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    g = np.full((2, n, n), eps_lo, dtype=complex)
    g[0][np.hypot(xx - n / 2, yy - n / 2) < n * 0.35] = eps_hi
    g[1][(np.abs(xx - n / 2) < n * 0.3) & (np.abs(yy - n / 2) < n * 0.15)] = eps_hi
    return g


def metal_field(algo, form, theta, thickness):
    s = wavebit.rcwa(backend=0, top=1.0, bot=2.25, theta=theta, wavelength=600., psi=np.pi / 3,
                     phi=0.4, structure=raster(-15 + 0.8j), truncation_order=[5, 5],
                     period=[400., 400.], thickness=thickness, layer_algorithm=algo,
                     answers='field', factorization='li')
    s.formulation = form
    s.solve(auto_detune=False)
    return np.asarray(s.reconstruct_field(sample_x=8, sample_y=8, sample_z=8,
                                          incidences=('s', 'p')))


def dng_field(algo, form):
    eps, mu = RASTER.astype(complex), np.ones(RASTER.shape, dtype=complex)
    eps[RASTER == 25.0], mu[RASTER == 25.0] = -4 - 0.1j, -1 - 0.05j
    s = wavebit.rcwa(backend=0, top=1.0, bot=2.0, theta=0.5, phi=PHI, psi=np.pi / 2,
                     wavelength=WL, thickness=[100., 60.], truncation_order=[3, 3],
                     period=[1200., 1200.], layer_algorithm=algo, answers='field')
    s.structure, s.mu = eps, mu
    s.formulation = form
    s.solve(auto_detune=False)
    return np.asarray(s.reconstruct_field(sample_x=5, sample_y=5, sample_z=4,
                                          incidences=('s', 'p')))


def single_precision(kind, theta, algo):
    kw = dict(tetm='tm') if kind == 'tm' else dict(psi=np.pi / 2, phi=PHI)
    out = {}
    for width in (0, 1):
        s = wavebit.rcwa(backend=0, top=1.0, bot=25.0, theta=theta, wavelength=WL,
                         thickness=[100., 60.], structure=GRATING, truncation_order=[15],
                         period=[2000.], layer_algorithm=algo if width else 'ETM',
                         answers='field', complex_dtype=width, **kw)
        s.solve(auto_detune=False)
        out[width] = np.asarray(s.reconstruct_field(sample_x=6, sample_y=1, sample_z=4))
    return rel(out[1], out[0])


def section_real_angle():
    rows = {}
    for label, theta, thick in (('oblique 0.35', 0.35, [120., 99.]), ('normal', 0.0, [120., 95.])):
        rows[f'metal raster, {label}: SMM vs ETM'] = guarded(
            lambda: rel(metal_field('SMM', '2ff', theta, thick),
                        metal_field('ETM', '2ff', theta, thick)))
    rows['active patterned double negative, 0.5: SMM 2ff vs ETM 4ff'] = guarded(
        lambda: rel(dng_field('SMM', '2ff'), dng_field('ETM', '4ff')))
    for algo in ('SMM', 'ETM'):
        rows[f'complex64, {algo} vs complex128 ETM, worst of tm/conical at 0 and 0.5'] = worst_of([
            guarded(lambda: single_precision(k, t, algo))
            for k in ('tm', 'conical') for t in (0.0, 0.5)])
    return rows


# ---------------------------------------------------------------------------- 6 below axis

def uniform_dng(algo, form='2ff'):
    s = wavebit.rcwa(backend=0, top=1.0, bot=wavebit.Material(-4.0, mu=-1.0), theta=0.5 - 0.3j,
                     wavelength=WL, thickness=[100.], layer_algorithm=algo, answers='field',
                     psi=np.pi / 2, phi=PHI, truncation_order=[0, 0], period=[2000., 2000.])
    s.structure = np.full((1, 2, 2), -4.0 + 0j)
    s.mu = np.full((1, 2, 2), -1.0 + 0j)
    s.formulation = form
    s.solve(auto_detune=False)
    return np.asarray(s.reconstruct_field(sample_x=3, sample_y=3, sample_z=4,
                                          incidences=('s', 'p')))


def section_below_axis():
    return {'SMM vs ETM': guarded(lambda: rel(uniform_dng('SMM'), uniform_dng('ETM'))),
            'ETM vs 4ff': guarded(lambda: rel(uniform_dng('ETM'), uniform_dng('ETM', '4ff')))}


# --------------------------------------------------------------------------------- 7 rungs

def section_rungs():
    rows = {}
    moved = []
    for case in CASES:
        def amplitudes(answers):
            solver = wavebit.rcwa(backend=0, top=1.0, bot=25.0, theta=THETA, wavelength=WL,
                                  thickness=[100., 60.], layer_algorithm='SMM', answers=answers,
                                  **CASES[case])
            return solver.solve(auto_detune=False).incident
        a, b = amplitudes('amplitude'), amplitudes('field')
        moved.append(max(float(np.max(np.abs(np.asarray(getattr(a, k))
                                             - np.asarray(getattr(b, k)))))
                         for k in ('r_s', 'r_p', 't_s', 't_p')))
    rows['gratings at 0.5+0.3j: largest amplitude change, field vs amplitude rung'] = max(moved)

    def vacuum():
        s = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=THETA, wavelength=WL, thickness=[1e4],
                         truncation_order=[0], period=[2000.], structure=np.ones((1, 1, 4)),
                         layer_algorithm='SMM', answers='field', tetm='te')
        s.solve(auto_detune=False)
        return 'solves'
    rows['10 um vacuum on vacuum, TE, field rung'] = guarded(vacuum)
    return rows


# --------------------------------------------------------------------------------- 8 graze

GRAZE_GRATING = np.ones((2, 1, 8), complex)
GRAZE_GRATING[0, 0, 2:5] = 6.0
GRAZE_GRATING[1] = 2.25
GRAZE_RASTER = np.ones((2, 6, 6), complex)
GRAZE_RASTER[0, 1:4, 1:4] = 6.0
GRAZE_RASTER[1] = 2.25


def graze_field(algo, kind):
    kw = dict(tetm=kind) if kind in ('te', 'tm') else dict(psi=0.7, phi=0.0)
    two = kind == '2d'
    s = wavebit.rcwa(backend=0, top=1.0, bot=4.0, theta=0.0, wavelength=WL, thickness=[150., 400.],
                     period=[600.] * (2 if two else 1),
                     structure=GRAZE_RASTER if two else GRAZE_GRATING,
                     truncation_order=[2, 2] if two else [4], layer_algorithm=algo, answers='field',
                     **kw)
    s.solve()
    inc = ('incident',) if kind in ('te', 'tm') else ('incident', 's', 'p')
    return np.asarray(s.reconstruct_field(sample_x=5, sample_y=4 if two else 1,
                                          sample_z=[30., 200., 400., 520.], incidences=inc))


def section_graze():
    return {'SMM vs ETM, auto_detune, worst of te/tm/conical/2d': worst_of([
        guarded(lambda: rel(graze_field('SMM', k), graze_field('ETM', k)))
        for k in ('te', 'tm', 'conical', '2d')])}


# --------------------------------------------------------------------------------- 9 ports

PORT_SCENES = {  # theta, thickness, (background, block) per layer, substrate
    'three layers, 0.5+0.3j': (THETA, [200., 300., 150.], [(4., 1.), (2.25, 6.), (9., 1.)],
                               2.25),
    'two eps 25 gratings on eps 25, 0.5+0.3j': (THETA, [100., 60.], [(25., 1.), (25., 1.)], 25.),
    '10 um of vacuum between gratings, 0.5+0.3j': (THETA, [200., 1e4, 150.],
                                                    [(4., 1.), (1., 1.), (9., 1.)], 2.25),
    '10 um of vacuum on vacuum, 0.5+0.3j': (THETA, [1e4], [(1., 1.)], 1.),
    'metal grating -20+1j, real 0.3': (0.3, [100., 300., 100.],
                                       [(4., 1.), (-20 + 1j, 1.), (2.25, 1.)], 2.25),
}


def port_gap(algo, scene, kind, side):
    theta, thick, layers, bot = PORT_SCENES[scene]
    two = kind == '2d'
    kw = dict(psi=0.7, phi=0.3) if kind in ('conical', '2d') else dict(tetm=kind)
    grid = (len(thick), 4, 6) if two else (len(thick), 1, 14)
    structure = np.ones(grid, dtype=complex)
    for i, (background, block) in enumerate(layers):
        structure[i] = background
        structure[i, ..., :grid[-1] // 2] = block
    s = wavebit.rcwa(backend=0, top=1.0, bot=bot, theta=theta, wavelength=WL, thickness=thick,
                     layer_algorithm=algo, answers='field', incidence_side=side,
                     structure=structure, truncation_order=[2, 2] if two else [4],
                     period=[700., 600.] if two else [700.], **kw)
    res = s.solve(auto_detune=False)
    xs, ys = list(np.linspace(0., 690., 23)), [0., 211.]

    def tangential(c):
        return np.asarray(c)[..., [0, 1, 3, 4]]
    faces = tangential(res.reconstruct_field(sample_x=xs, sample_y=ys, sample_z=[0., sum(thick)]))
    ports = [tangential(res.boundary_fields(face, sample_x=xs, sample_y=ys, part='total'))
             for face in ('bottom', 'top')]
    scale = max(float(np.max(np.abs(c))) for c in [faces] + ports)
    worst = max(float(np.max(np.abs(faces[i] - ports[i]))) / scale for i in range(2))
    counts = tangential(res.reconstruct_field(sample_x=xs, sample_y=ys, sample_z=2))
    for k in range(len(thick) - 1):
        below, above = counts[2 * k + 1], counts[2 * k + 2]
        local = max(float(np.max(np.abs(below))), float(np.max(np.abs(above))))
        worst = max(worst, float(np.max(np.abs(below - above))) / local)
    return worst


def section_ports():
    return {f'{scene}: SMM': worst_of([guarded(lambda: port_gap('SMM', scene, k, side))
                                       for k in ('te', 'tm', 'conical', '2d')
                                       for side in ('top', 'bottom')])
            for scene in PORT_SCENES}


# --------------------------------------------------------------------------------- 10 cost

def retained_blocks(solver):
    """Bytes of the field state, in blocks of side ``2*n_orders`` per layer."""
    seen, total = set(), 0
    stack = [solver.layer_info_list, solver.T1]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif hasattr(item, 'nbytes') and id(item) not in seen:
            seen.add(id(item))
            total += item.nbytes
    dim = 2 * solver.layer_info_list[0][0].shape[0]       # `epz_conv_i` is one side per order
    return total / (dim * dim * 16) / len(solver.layer_info_list)


def section_cost(timed):
    from wavebit._core import scattering
    calls = []
    for name in ('layer_modes_1d', 'layer_modes_1d_conical', 'layer_modes_2d'):
        build = getattr(scattering, name)

        def counted(*a, _build=build, **k):
            calls.append(1)
            return _build(*a, **k)
        setattr(scattering, name, counted)
    big = np.ones((2, 16, 16)) * 6.0
    big[:, 4:12, 3:10] = 1.0
    rows = {}
    for label, theta in (('real 0.5', 0.5), ('0.5+0.3j', THETA)):
        times = {}
        for algo in ('ETM', 'SMM'):
            s = wavebit.rcwa(backend=0, top=1.0, bot=2.25, theta=theta, phi=0.3, psi=0.4,
                             wavelength=WL, thickness=[100., 60.], truncation_order=[6, 6],
                             period=[1200., 1200.], structure=big, layer_algorithm=algo,
                             answers='field')
            best = np.inf
            for _ in range(5):
                del calls[:]
                t0 = time.perf_counter()
                s.solve(auto_detune=False)
                best = min(best, time.perf_counter() - t0)
            times[algo] = best
        row = (f'{len(calls)} builds for 2 layers, '
               f'{retained_blocks(s):.2f} blocks retained per layer')
        if timed:
            row += f', {times["SMM"] / times["ETM"]:.2f}x the ETM field solve'
        rows[f'2D raster (6, 6), 2 layers, {label}: SMM field solve'] = row
    return rows


SECTIONS = [('1 closed form: worst residual against the plane wave', section_closed_form),
            ('2 gratings', section_gratings),
            ('3 a root growing below another layer', section_grows_below),
            ('4 a capped layer on its own medium', section_own_medium),
            ('5 real angle', section_real_angle),
            ('6 uniform double negative at 0.5-0.3j', section_below_axis),
            ('7 rungs', section_rungs),
            ('8 an order on a layer light line', section_graze),
            ('9 the field meets itself and its ports, label-free', section_ports)]


def measure():
    out = {}
    for title, fn in SECTIONS:
        rows = fn()
        out[title] = {k: (v if isinstance(v, str) else float(v)) for k, v in rows.items()}
    return out


def other_tree(rev):
    """Sections 1-7 on ``git archive`` of ``rev``, by this script, in a subprocess."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(['git', '-C', ROOT, 'archive', rev, 'wavebit'],
                                 check=True, capture_output=True).stdout
        subprocess.run(['tar', '-x', '-C', tmp], input=archive, check=True)
        env = dict(os.environ, PYTHONPATH=tmp)
        run = subprocess.run([sys.executable, os.path.abspath(__file__), '--emit-json'],
                             cwd=tmp, env=env, check=True, capture_output=True, text=True)
        return json.loads(run.stdout.splitlines()[-1])


def show(value):
    return value if isinstance(value, str) else f'{value:.1e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', nargs='*', default=[], metavar='REV')
    parser.add_argument('--emit-json', action='store_true')
    parser.add_argument('--time', action='store_true')
    args = parser.parse_args()
    if args.emit_json:
        print(json.dumps(measure()))
        return
    trees = [('this tree', measure())] + [(rev, other_tree(rev)) for rev in args.compare]
    names = [name for name, _ in trees]
    print('columns: ' + ' | '.join(names))
    for title, _ in SECTIONS:
        print(f'\n{title}')
        for row in trees[0][1][title]:
            cells = [show(tree[title].get(row, '-')) for _, tree in trees]
            print(f'  {row:72s} ' + ' | '.join(f'{c:>12s}' for c in cells))
    print('\n10 cost (this tree only)')
    for row, value in section_cost(args.time).items():
        print(f'  {row:72s} {value}')


if __name__ == '__main__':
    main()
