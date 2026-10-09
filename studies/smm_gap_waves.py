"""Why did SMM have a rounding floor ETM did not, on patterned double-negative TM gratings?

A seeded sweep found SMM losing digits ETM kept on gratings holding a double negative (Re eps < 0
and Re mu < 0 in part of the cell), at a real angle, so not through a complex k_par: one TM
grating with eps = -1.68, mu = -0.54+0.05j in part of a 432 nm cell at theta = 0.0963 had an SMM
floor of 4.3e-10 against ETM's 2.0e-13. The floor of a solve is the nonlinear part of its
response to a 1e-14 relative nudge of theta, max |d14 - d12/100| over every amplitude, with dN
the change under a 10**-N nudge. Three mechanisms were candidates: (a) the gap SMM joins layers
through, (b) the flux tie ``power_down_labels`` breaks for a patterned double negative's lossless
modes (``_FLUX_FLAT``), (c) Li's inverse rule where 1/eps changes sign. The answer is (a), though
not as a near-mirror: develop's gap had p impedance ``kz/eps = 1/(1 + k_par**2)``, which falls
like 1/k_par**2 on high orders, while a layer with negative permittivity beside positive carries
propagating high-order modes whose ``kz/eps`` grows like k_par. Each layer then sat between two
nearly totally reflecting interfaces in the gap basis, and the cascade's recovery of its
transmission cost the mismatch in rounding. ``scattering._gap_waves`` now gives both
polarizations ``sqrt(1 + |k_par|**2)``.

  1 mechanism  four patterned double-negative TM gratings (A to C the sweep's material, A at
               the size the sweep reported; D lossless and high-contrast), the floor under
               develop's gap, with _FLUX_FLAT moved, with develop's p impedance scaled, under
               Laurent's rule, and under the shipped gap; then each layer's largest propagating
               mode against the gap. (b) moves nothing, to the bit; (c) is a floor both
               algorithms share on D, not the difference between them; the gap's p impedance
               moves SMM's floor by about the factor it is scaled by.
  2 sweep      40 TM gratings of the sweep's materials (eps -1.68, mu -0.54+0.05j, 432 nm,
               theta 0.0963), background, fill, wavelength, thickness, truncation and substrate
               drawn at random: floors and |SMM - ETM|, develop's gap and the shipped one.
  3 families   12 random scenes each: double negatives on TE/TM/conical, on 2D, lossless and
               lossy metal TM gratings (mu = 1), and the families the gap should not move --
               high-index and lossy dielectrics, 2 to 20 um layers, truncation 25 to 40.
  4 cutoff     the cost: near a layer mode's cutoff (q -> 0) a larger p impedance is a larger
               mismatch. A high-index TM and TE grating swept to within 1e-8 of a crossing.
  5 imaginary  theta = i*asinh(1) puts the specular order at k_par = i, where 1 + k_par**2 = 0.
  6 mixture    a limit section 11 lists: lossless (-2, -1) beside (2, 1.5) at exactly half fill,
               where the cell average of 1/eps (and of eps) is zero. Neither algorithm, nor the
               gap; perturbing the fill or the contrast removes it.
  7 gauge      the gap cancels: scale both waves by 2.7 and see what moves.
  8 develop    with ``--develop REV``: the control this study calls develop's gap, against a
               ``git archive`` of REV, byte for byte.

Units: lengths in nm, wavevectors in units of k0. Time convention exp(-i w t). Every solve is
NumPy complex128, ``auto_detune=False``. Random draws are seeded. Run from the repository root:

    python studies/smm_gap_waves.py --develop b9957c8

It prints the log committed as ``studies/results/smm_gap_waves.log`` in about fifteen seconds.
``--develop`` extracts REV with ``git archive`` into a temporary directory and runs section 8's
scenes there in a subprocess, by this same script; without it section 8 says it was skipped.
``docs/connecting-algorithms.md`` section 11 ("SMM's gap") has the reading.
"""
import argparse
import contextlib
import os
import subprocess
import sys
import tempfile
import warnings
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _root_from_argv():
    """The tree to import ``wavebit`` from: this one, or section 8's ``--root``."""
    argv = sys.argv
    return argv[argv.index('--root') + 1] if '--root' in argv else ROOT


sys.path.insert(0, _root_from_argv())

import numpy as np                                          # noqa: E402

import wavebit                                              # noqa: E402

warnings.simplefilter('ignore')

DNG = (-1.68 + 0j, -0.54 + 0.05j)       # the sweep's double negative


def header(number, title):
    print(f'\n{number}. {title}')


def amplitudes(result):
    out = []
    for variant in ('incident', 's_inc', 'p_inc'):
        sub = getattr(result, variant, None)
        if sub is not None:
            out += [np.ravel(np.asarray(getattr(sub, f))) for f in ('r_s', 'r_p', 't_s', 't_p')]
    return np.concatenate(out)


# ------------------------------------------------------------------------------------- the gap

def develops_gap(xp, C, kx, ky=None):
    """The gap before 2026-10-09: ``eps = 1 + k_par**2``, ``mu = 1``, ``kz = 1`` (section 8)."""
    k_par_sq = kx ** 2 if ky is None else kx ** 2 + ky ** 2
    z_p = xp.astype(1 / (1 + k_par_sq), C)
    return xp.ones_like(z_p), z_p


def scaled(waves, s_factor, p_factor):
    def gap_waves(xp, C, kx, ky=None):
        y_s, z_p = waves(xp, C, kx, ky)
        return s_factor * y_s, p_factor * z_p
    return gap_waves


@contextlib.contextmanager
def gap(waves):
    """Solve with ``waves`` as the gap; ``None`` touches nothing (section 8 runs it on develop)."""
    if waves is None:
        yield
        return
    from wavebit._core import scattering
    shipped = scattering._gap_waves
    scattering._gap_waves = waves
    try:
        yield
    finally:
        scattering._gap_waves = shipped


@contextlib.contextmanager
def flux_flat(value):
    from wavebit._core import conventions
    shipped = conventions._FLUX_FLAT
    conventions._FLUX_FLAT = value
    try:
        yield
    finally:
        conventions._FLUX_FLAT = shipped


# ---------------------------------------------------------------------------------- the scenes

def grating(path, layers, *, wavelength, period, thickness, truncation, theta, top=1.0,
            bot=1.0, ny=1):
    """A stack of 1D (or y-extruded 2D) layers; ``layers`` are ``(eps, mu)`` rows, bottom first."""
    eps = np.array([[row[0]] * ny for row in layers], dtype=complex)
    mu = np.array([[row[1]] * ny for row in layers], dtype=complex)
    if ny > 1:                                  # 2D: the last rows ordinary, so not y-invariant
        for layer in range(len(layers)):
            ordinary = eps[layer, 0][eps[layer, 0].real > 0]
            eps[layer, ny // 2 + 1:] = ordinary[0] if len(ordinary) else 1.0
            mu[layer, ny // 2 + 1:] = 1.0
    return dict(path=path, structure=eps, mu=None if np.all(mu == 1) else mu,
                wavelength=wavelength, period=period, thickness=list(thickness),
                truncation_order=truncation, theta=theta, top=top, bot=bot)


def solve(scene, algo, theta=None, **extra):
    kw = {k: v for k, v in scene.items() if k not in ('path', 'theta')}
    kw.update(backend=0, layer_algorithm=algo, answers='amplitude',
              theta=scene['theta'] if theta is None else theta, **extra)
    path = scene['path']
    if path in ('te', 'tm'):
        kw.update(tetm=path, period=[scene['period']])
    elif path == 'conical':
        kw.update(phi=0.3, psi=0.4, period=[scene['period']])
    else:
        kw.update(phi=0.3, psi=0.4, period=(scene['period'], 0.8 * scene['period']))
    if kw['mu'] is None:
        del kw['mu']
    return amplitudes(wavebit.rcwa(**kw).solve(auto_detune=False))


def floor(scene, algo, **extra):
    """``(floor, amplitudes)``: the nonlinear response to a 1e-14 nudge of ``theta``."""
    theta = scene['theta']
    r0 = solve(scene, algo, **extra)
    d14 = solve(scene, algo, theta * (1 + 1e-14), **extra) - r0
    d12 = solve(scene, algo, theta * (1 + 1e-12), **extra) - r0
    return np.abs(d14 - d12 / 100).max() / max(1.0, np.abs(r0).max()), r0


def row(pixels, dng, other):
    """One layer's ``(eps, mu)`` from a mask: ``N`` the double negative, ``.`` the other."""
    eps = [dng[0] if c == 'N' else other for c in pixels]
    mu = [dng[1] if c == 'N' else 1.0 for c in pixels]
    return eps, mu


def sweep_layer(width, background):
    """20 pixels: the sweep's double negative on the first ``width``, ``background`` after."""
    return row('N' * width + '.' * (20 - width), DNG, background)


MECHANISM = {
    # The sweep's material, at the size the sweep reported (4.3e-10 against 2.0e-13).
    'A': grating('tm', [sweep_layer(5, 1.69)], wavelength=1089.0, period=432.0,
                 thickness=[299.0], truncation=22, theta=0.0963, bot=1.5),
    # tests/test_scattering_core.py's 'dng_tm'.
    'B': grating('tm', [sweep_layer(5, 1.69)], wavelength=1000.0, period=432.0,
                 thickness=[100.0], truncation=20, theta=0.0963),
    'C': grating('tm', [sweep_layer(14, 1.69)], wavelength=787.0, period=432.0,
                 thickness=[188.0], truncation=19, theta=0.0963),
    # Lossless and high contrast: a floor both algorithms share, which Laurent's rule removes.
    'D': grating('tm', [row('N....NNNNNNNNNNN', (-3.55, -1.41), 3.64)], wavelength=1171.0,
                 period=938.0, thickness=[281.0], truncation=6, theta=0.184, bot=1.5),
}


def section_mechanism():
    header(1, 'mechanism -- four patterned double-negative TM gratings, floors')
    for name, scene in MECHANISM.items():
        layers = []
        for eps, mu in zip(scene['structure'][:, 0], scene['mu'][:, 0]):
            k, j = int(np.argmin(eps.real)), int(np.argmax(eps.real))
            layers.append(f'{int((eps.real < 0).sum())}/{len(eps)} of ({eps[k].real:g}, {mu[k]:.4g}) '
                          f'beside {eps[j].real:g}')
        print(f"  {name}: {scene['wavelength']:g} nm, period {scene['period']:g}, layers "
              f"{scene['thickness']} of " + '; '.join(layers)
              + f", truncation {scene['truncation_order']}, theta {scene['theta']}")
    cols = ('ETM', 'SMM develop', 'develop, _FLUX_FLAT 1e-6/1e-9/1e-15', "develop's z_p x10",
            'x0.1', 'Laurent ETM / SMM develop', 'SMM shipped')
    print('  ' + ' | '.join(cols))
    identical = True
    for name, scene in MECHANISM.items():
        etm, _ = floor(scene, 'ETM')
        with gap(develops_gap):
            dev, _ = floor(scene, 'SMM')
            flux = []
            for value in (1e-6, 1e-9, 1e-15):
                with flux_flat(value):
                    flux.append(floor(scene, 'SMM')[0])
            identical &= all(f == dev for f in flux)
            lau_smm, _ = floor(scene, 'SMM', factorization='laurent')
        with gap(scaled(develops_gap, 1, 10)):
            up, _ = floor(scene, 'SMM')
        with gap(scaled(develops_gap, 1, 0.1)):
            down, _ = floor(scene, 'SMM')
        lau_etm, _ = floor(scene, 'ETM', factorization='laurent')
        shipped, _ = floor(scene, 'SMM')
        print(f'  {name}: {etm:.1e} | {dev:.1e} | ' + '/'.join(f'{f:.1e}' for f in flux)
              + f' | {up:.1e} | {down:.1e} | {lau_etm:.1e}/{lau_smm:.1e} | {shipped:.1e}')
    print(f'  every _FLUX_FLAT floor equal to develop\'s to the bit: {identical}')
    # The modes behind it: each layer's largest propagating |q| against the gap at the top order.
    from wavebit._core import engine, solve_1d
    kernel = engine._SOLVES['SMM']['1d']
    for name, scene in MECHANISM.items():
        captured = {}

        def capture(*args, **kw):
            captured['args'], captured['kw'] = args, kw
            return kernel(*args, **kw)
        engine._SOLVES['SMM']['1d'] = capture
        try:
            solve(scene, 'SMM')
        finally:
            engine._SOLVES['SMM']['1d'] = kernel
        args, kw = captured['args'], captured['kw']
        kx = np.asarray(args[2], complex)
        eye = np.eye(len(kx), dtype=complex)
        k0, top = 2 * np.pi / args[1], np.abs(kx).max()
        modes = []
        for layer, d in enumerate(args[6]):
            layer_mu = solve_1d._layer_mu(eye, tuple(kw['mu']), layer)
            _, _, q = solve_1d.layer_modes_1d(np, 1, kx, eye, args[7][layer], args[8][layer],
                                              args[9][layer], complex, mu=layer_mu,
                                              dng=kw['dng'][layer])
            propagating = q[np.abs(q.real) * k0 * d < 1]
            modes.append(f'{propagating[np.argmax(np.abs(propagating))]:.4g}')
        print(f"  {name}: largest propagating q {', '.join(modes)}; |kx| <= {top:.1f}, where "
              f"develop's z_p was 1/{1 + top ** 2:.0f} and the shipped is {np.sqrt(1 + top ** 2):.1f}")


def sweep_scene(rng):
    width = int(rng.integers(2, 19))
    background = float(rng.choice([1.0, 1.69, 2.25, 4.0]))
    return grating('tm', [sweep_layer(width, background)], wavelength=float(rng.uniform(300, 1200)),
                   period=432.0, thickness=[float(rng.uniform(30, 600))],
                   truncation=int(rng.integers(5, 25)), theta=0.0963,
                   bot=float(rng.choice([1.0, 1.5, 2.25])))


def compare(scenes):
    """Per scene: ETM's floor, then (floor, |SMM - ETM|) under develop's gap and the shipped."""
    out = []
    for scene in scenes:
        etm, r_etm = floor(scene, 'ETM')
        with gap(develops_gap):
            dev, r_dev = floor(scene, 'SMM')
        shipped, r_new = floor(scene, 'SMM')
        out.append((etm, dev, np.abs(r_dev - r_etm).max() / max(1, np.abs(r_etm).max()),
                    shipped, np.abs(r_new - r_etm).max() / max(1, np.abs(r_etm).max())))
    return np.array(out)


def summary(label, table):
    worst, median = table.max(axis=0), np.median(table, axis=0)
    print(f'  {label:24s} ETM floor {median[0]:.1e} / {worst[0]:.1e} | develop: floor '
          f'{median[1]:.1e} / {worst[1]:.1e}, |SMM-ETM| {median[2]:.1e} / {worst[2]:.1e} | '
          f'shipped: floor {median[3]:.1e} / {worst[3]:.1e}, |SMM-ETM| {median[4]:.1e} / '
          f'{worst[4]:.1e}')


def section_sweep():
    header(2, "sweep -- 40 TM gratings of the sweep's double negative, median / worst")
    rng = np.random.default_rng(1)
    table = compare([sweep_scene(rng) for _ in range(40)])
    summary('eps -1.68, mu -0.54+0.05i', table)
    print(f'  scenes with an SMM floor above 1e-11: develop {int((table[:, 1] > 1e-11).sum())}, '
          f'shipped {int((table[:, 3] > 1e-11).sum())}; above 10x ETM\'s: develop '
          f'{int((table[:, 1] > 10 * table[:, 0]).sum())}, shipped '
          f'{int((table[:, 3] > 10 * table[:, 0]).sum())}')


def family_scene(kind, rng):
    def mask():
        width, start = int(rng.integers(1, 16)), int(rng.integers(0, 16))
        on = np.zeros(16, bool)
        on[(start + np.arange(width)) % 16] = True
        return ''.join('N' if x else '.' for x in on)
    n_layers = int(rng.integers(1, 3))
    path = {'dng': str(rng.choice(['te', 'tm', 'conical'])), 'dng_2d': '2d'}.get(kind, 'tm')
    layers, thickness, truncation = [], list(rng.uniform(30, 400, n_layers)), int(rng.integers(3, 12))
    for _ in range(n_layers):
        if kind.startswith('dng'):
            region = (-rng.uniform(0.5, 4) + 1j * rng.choice([0, rng.uniform(0, 0.1)]),
                      -rng.uniform(0.3, 2) + 1j * rng.choice([0, rng.uniform(0, 0.1)]))
            layers.append(row(mask(), region, rng.uniform(1, 4)))
        elif kind.startswith('metal'):
            loss = rng.uniform(0.1, 3) if kind == 'metal_lossy' else 0.0
            layers.append(row(mask(), (-rng.uniform(1, 40) + 1j * loss, 1.0), rng.uniform(1, 4)))
        else:
            high = {'high_index': rng.uniform(9, 16), 'lossy': rng.uniform(2, 12)
                    + 1j * rng.uniform(0, 1)}.get(kind, rng.uniform(4, 12))
            layers.append(row(mask(), (high, 1.0), 1.0))
    if kind == 'thick':
        thickness = list(rng.uniform(2000, 20000, n_layers))
    if kind == 'truncation':
        truncation = int(rng.integers(25, 41))
    if path == '2d':
        truncation = (min(truncation, 5), 2)
    return grating(path, layers, wavelength=float(rng.uniform(400, 1200)),
                   period=float(rng.uniform(300, 1000)), thickness=thickness,
                   truncation=truncation, theta=float(rng.uniform(0.05, 0.8)),
                   bot=float(rng.choice([1.0, 1.5, 2.25])), ny=8 if path == '2d' else 1)


def section_families():
    header(3, 'families -- 12 random scenes each, median / worst')
    for kind, label in (('dng', 'double negative, 1D'), ('dng_2d', 'double negative, 2D'),
                        ('metal', 'lossless metal, TM'), ('metal_lossy', 'lossy metal, TM'),
                        ('high_index', 'eps 9..16, TM'), ('lossy', 'lossy dielectric, TM'),
                        ('thick', '2..20 um layers, TM'), ('truncation', 'truncation 25..40, TM')):
        rng = np.random.default_rng(zlib.crc32(kind.encode()))
        summary(label, compare([family_scene(kind, rng) for _ in range(12)]))


def section_cutoff():
    header(4, "cutoff -- the shipped gap's cost near a layer mode's cutoff")
    from scipy.optimize import brentq
    from wavebit._core import engine
    for path, eps_high in (('tm', 12.0), ('tm', 4.0), ('te', 12.0)):
        def scene(wavelength):
            return grating(path, [([eps_high] * 7 + [1.0] * 9, [1.0] * 16)],
                           wavelength=wavelength, period=700.0, thickness=[120.0], truncation=6,
                           theta=0.3, bot=1.5)
        captured = {}
        kernel = engine._SOLVES['ETM']['1d']

        def capture(*args, **kw):
            captured['args'] = args
            return kernel(*args, **kw)

        def eigenvalue(wavelength, which):
            engine._SOLVES['ETM']['1d'] = capture
            try:
                solve(scene(wavelength), 'ETM')
            finally:
                engine._SOLVES['ETM']['1d'] = kernel
            args = captured['args']
            kx = np.asarray(args[2], complex)
            eye = np.eye(len(kx), dtype=complex)
            operator = (np.diag(kx ** 2) - args[8][0] if path == 'te'
                        else args[7][0] @ (kx[:, None] * args[9][0] * kx[None, :] - eye))
            return np.sort(np.linalg.eigvals(operator).real)[which]
        grid = np.linspace(500, 1500, 101)
        for which in range(13):
            values = [eigenvalue(w, which) for w in grid]
            crossing = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
            if len(crossing):
                break
        i = crossing[0]
        root = brentq(lambda w: eigenvalue(w, which), grid[i], grid[i + 1], xtol=1e-13)
        print(f'  {path.upper()}, eps {eps_high:g} beside air: a mode crosses q = 0 at {root:.6f} nm')
        for offset in (1e-2, 1e-4, 1e-6, 1e-8):
            s = scene(root * (1 + offset))
            etm, r_etm = floor(s, 'ETM')
            with gap(develops_gap):
                dev, r_dev = floor(s, 'SMM')
            new, r_new = floor(s, 'SMM')
            print(f'    offset {offset:.0e}: q**2 {eigenvalue(root * (1 + offset), which):+.1e}; '
                  f'floor ETM {etm:.1e}, SMM develop {dev:.1e}, shipped {new:.1e}; |SMM-ETM| '
                  f'develop {np.abs(r_dev - r_etm).max():.1e}, shipped '
                  f'{np.abs(r_new - r_etm).max():.1e}')


def section_imaginary():
    header(5, 'imaginary -- theta = i*asinh(1), the specular order at k_par = i')
    for path in ('te', 'tm', 'conical'):
        scene = grating(path, [([4.0, 4.0, 1.0, 1.0, 1.0, 1.0], [1.0] * 6)], wavelength=900.0,
                        period=700.0, thickness=[200.0], truncation=4,
                        theta=1j * np.arcsinh(1.0), bot=2.25)
        etm = solve(scene, 'ETM')
        outcome = []
        for label, waves in (('develop', develops_gap), ('shipped', None)):
            with gap(waves):
                try:
                    outcome.append(f'{label} {np.abs(solve(scene, "SMM") - etm).max():.1e}')
                except FloatingPointError:
                    outcome.append(f'{label} FloatingPointError (non-finite efficiencies)')
        print(f'  {path}: |SMM - ETM| ' + ', '.join(outcome))


def section_mixture():
    header(6, 'mixture -- lossless (-2, -1) beside (2, 1.5) at exactly half fill, conical')
    def scene(to, other=2.0, width=10, loss=0.0, **kw):
        layer = ([(-2 + 1j * loss, -1 + 1j * loss)] * width
                 + [(other, 1.5)] * (20 - width))
        return grating('conical', [([e for e, _ in layer], [m for _, m in layer])],
                       truncation=to, **kw)
    grid = [dict(period=p, wavelength=w, thickness=[d], theta=t, bot=b)
            for p in (700.0, 1000.0) for w in (500.0, 600.0) for d in (40.0, 150.0)
            for t in (0.25, 0.7) for b in (1.0, 2.25)]
    for to in (5, 10):
        for label, extra in (('half fill', {}), ('fill 8/20', dict(width=8)),
                             ('beside 2.2', dict(other=2.2)), ('1e-6 loss', dict(loss=1e-6))):
            worst = {}
            for name, waves in (('develop', develops_gap), ('shipped', None)):
                values = []
                for g in grid:
                    s = scene(to, **extra, **g)
                    etm = solve(s, 'ETM')
                    with gap(waves):
                        values.append(np.abs(solve(s, 'SMM') - etm).max())
                worst[name] = max(values)
            print(f'  truncation {to:2d}, {label:11s}: worst |SMM - ETM| over {len(grid)} scenes '
                  f"develop {worst['develop']:.1e}, shipped {worst['shipped']:.1e}")


def section_gauge():
    header(7, 'gauge -- both gap waves scaled by 2.7, each against its own magnitude')
    from wavebit._core import scattering
    captured = {}
    field_state = scattering._field_state

    def capture(xp, I, S_ref, S_trn, S_layers, AB, c_inc):
        captured.update(S_ref=S_ref[0], S_trn=S_trn[0], S_layer=S_layers[0][0],
                        gamma=scattering._below_reflections(xp, I, S_layers, S_trn)[0])
        return field_state(xp, I, S_ref, S_trn, S_layers, AB, c_inc)
    stack = np.array([[[4., 4., 4., 1., 1., 1.]], [[9., 9., 1., 1., 9., 1.]]])
    scenes = {
        'TM': dict(structure=stack, tetm='tm', top=4.0, bot=2.25, theta=0.3,
                   wavelength=777.0, period=[770.0], thickness=[220.0, 150.0], truncation_order=5),
        'conical': dict(structure=stack, phi=0.4, psi=0.3, top=4.0, bot=2.25, theta=0.3,
                        wavelength=777.0, period=[770.0], thickness=[220.0, 150.0],
                        truncation_order=5),
        '2D': dict(structure=np.array([[[9., 9., 1., 1.], [1., 9., 9., 1.], [1., 1., 9., 9.],
                                        [9., 1., 1., 9.]]]), phi=0.3, psi=0.4, top=1.0, bot=2.25,
                   theta=0.2, wavelength=500.0, period=(600.0, 600.0), thickness=[300.0],
                   truncation_order=(2, 2)),
    }

    def run(waves, kw):
        with gap(waves):
            scattering._field_state = capture
            try:
                solver = wavebit.rcwa(backend=0, layer_algorithm='SMM', answers='field', **kw)
                result = solver.solve(auto_detune=False)
            finally:
                scattering._field_state = field_state
        sub = result.incident if result.s_inc is None else result.s_inc
        return dict(T=np.asarray(result.T), r_s=np.asarray(sub.r_s),
                    blocks=[np.asarray(b) for b in solver.scattering.blocks],
                    **{k: np.asarray(v) for k, v in captured.items()})

    def rel(a, b):
        return np.abs(a - b).max() / np.abs(b).max()
    for name, kw in scenes.items():
        a = run(None, kw)
        b = run(scaled(scattering._gap_waves, 2.7, 2.7), kw)
        print(f'  {name}: T {np.abs(a["T"] - b["T"]).max():.1e}, r_s '
              f'{np.abs(a["r_s"] - b["r_s"]).max():.1e}; whole-stack blocks '
              + ' '.join(f'{rel(x, y):.1e}' for x, y in zip(b['blocks'], a['blocks']))
              + f'; S_layers[0].S11 {rel(b["S_layer"], a["S_layer"]):.2f}, gamma[0] '
              f'{rel(b["gamma"], a["gamma"]):.2f}, S_ref.S11 {rel(b["S_ref"], a["S_ref"]):.2f}, '
              f'S_trn.S11 {rel(b["S_trn"], a["S_trn"]):.2f}')


# --------------------------------------------------------------------------------- 8. develop

def develop_arrays(waves):
    """SMM amplitudes and ``S11`` of section 1's and 2's scenes, every path, both sides."""
    out = {}
    scenes = dict(MECHANISM)
    scenes.update({f'sweep{i}': s for i, s in enumerate(
        sweep_scene(np.random.default_rng(1)) for _ in range(5))})
    for path in ('te', 'conical', '2d'):
        scenes[path] = grating(path, [sweep_layer(7, 1.69)],
                               wavelength=1000.0, period=432.0, thickness=[150.0],
                               truncation=12 if path != '2d' else (5, 2), theta=0.0963,
                               bot=2.25, ny=8 if path == '2d' else 1)
    with gap(waves):
        for name, scene in scenes.items():
            for side in ('top', 'bottom'):
                out[f'{name}/{side}'] = solve(scene, 'SMM', incidence_side=side)
    return out


def section_develop(rev):
    header(8, f"develop -- this study's develop gap against {rev}" if rev
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
        subprocess.run([sys.executable, os.path.abspath(__file__), '--dump', dump,
                        '--root', tree], check=True, capture_output=True,
                       env=dict(os.environ, PYTHONWARNINGS='ignore'))
        theirs = dict(np.load(dump))
    ours = develop_arrays(develops_gap)
    same = sum(ours[k].tobytes() == theirs[k].tobytes() for k in theirs)
    print(f'  {same} of {len(theirs)} arrays byte-identical (SMM amplitudes of {len(theirs) // 2} '
          f'scenes on the TE, TM, conical and 2D paths, each lit from either side)')
    shipped = develop_arrays(None)
    worst = max(np.abs(shipped[k] - theirs[k]).max() for k in theirs)
    print(f'  the shipped gap against {rev}: worst amplitude difference {worst:.1e}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--develop', metavar='REV', help='section 8: compare with this commit')
    parser.add_argument('--only', type=int, nargs='*', help='run these sections only')
    parser.add_argument('--dump', help=argparse.SUPPRESS)
    parser.add_argument('--root', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dump:
        np.savez(args.dump, **develop_arrays(None))
        return
    print("SMM's gap waves -- studies/smm_gap_waves.py")
    print(f'numpy {np.__version__}; every solve NumPy complex128')
    sections = {1: section_mechanism, 2: section_sweep, 3: section_families, 4: section_cutoff,
                5: section_imaginary, 6: section_mixture, 7: section_gauge,
                8: lambda: section_develop(args.develop)}
    for number, section in sections.items():
        if not args.only or number in args.only:
            section()


if __name__ == '__main__':
    main()
