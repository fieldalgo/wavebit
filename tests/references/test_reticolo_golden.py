"""Diffraction orders against RETICOLO, from a checked-in golden.

Every other oracle in this suite is blind to diffraction. ``test_lossy_media``'s Airy
recursion is exact but describes uniform layers, where only the zeroth order exists; the
legacy goldens are the previous implementation's own output, so they pin continuity rather
than correctness. RETICOLO is the one reference here that says a grating's *orders* are
right, and it is independent: a different group's solver, in a different language, validated
over two decades of published work.

RETICOLO builds a binary grating's Fourier coefficients from the *exact* edge positions, and
so do both raster paths here — the integrated one by compressing the raster into blocks
first, the sampled one straight
off the pixel grid. These gratings have their edges on pixel boundaries, so the raster *is*
the structure and all three land on each other at round-off, 1e-15 to 1e-13.

That is true of sampled only since 2026-08. It used to resample the raster onto a grid fine
enough to resolve ``4*truncation_order+1`` harmonics and transform *that*, which approaches the same
expression as 1/R — leaving it ~1e-5 from RETICOLO where it now sits at ~1e-14. The test that
holds sampled to this tolerance used to assert the opposite.

The values are frozen in ``golden_reticolo.npz`` by ``gen_goldens_reticolo.py``, which needs
Octave and RETICOLO. This file needs neither, which is the point: the evidence used to live
in a notebook under a gitignored ``dev/`` and in ``studies/``, which has no assertions and
does not run in CI, so the strongest result the project had was also the one nothing checked.

RETICOLO returns only the *propagating* orders, so the golden carries the order indices it
reported and the comparison selects those; that the two agree on **which** orders propagate
is itself part of the assertion.
"""
import pathlib

import numpy as np
import pytest

import wavebit

GOLDEN_PATH = pathlib.Path(__file__).parent / 'golden_reticolo.npz'

# 1D binary grating, exact edges on both sides. `structure` is a raster only in spelling: the
# integrated path reads the block edges off it, and RETICOLO is handed those same edges.
#
# **These are permittivities, and the physical scene is unchanged.** Every value below is the
# square of the refractive index this file carried before 2026-08-20 -- 9 was 3, 12.25 was 3.5,
# 4.84 was 2.2 -- so the frozen RETICOLO numbers still describe it and are not regenerated.
# `4.840000000000001` is `2.2 ** 2` as a double and not a typo for `4.84`; the tidy decimal is a
# different number by 8.9e-16, and it is the bridge's `index_from_eps` that would carry the
# difference back into the texture Octave is handed.
_HALF = np.array([[[9.0] * 5 + [1.0] * 5]])
_QUARTER = np.array([[[12.25] * 3 + [1.0] * 9]])
_TWO_LAYER = np.array([[[9.0] * 6 + [1.0] * 6], [[4.840000000000001] * 8 + [1.0] * 4]])

# name -> solver kwargs. `grating_type` picks RETICOLO's branch: 0 for pure TE/TM, 1 conical.
CASES = {
    'te_normal': dict(tetm='te', theta=0.0, phi=None, top=1.0, bot=2.25, truncation_order=10,
                      period=[0.7], wavelength=0.9, structure=_HALF, thickness=[0.15]),
    'te_oblique': dict(tetm='te', theta=0.2, phi=None, top=1.0, bot=2.25, truncation_order=10,
                       period=[0.7], wavelength=0.9, structure=_HALF, thickness=[0.15]),
    'tm_normal': dict(tetm='tm', theta=0.0, phi=None, top=1.0, bot=2.25, truncation_order=10,
                      period=[0.7], wavelength=0.9, structure=_HALF, thickness=[0.15]),
    'tm_oblique': dict(tetm='tm', theta=0.2, phi=None, top=1.0, bot=2.25, truncation_order=10,
                       period=[0.7], wavelength=0.9, structure=_HALF, thickness=[0.15]),
    # A duty cycle that is not 1/2 and a higher contrast: the +-1 orders carry real power.
    'te_quarter': dict(tetm='te', theta=0.15, phi=None, top=1.0, bot=2.25, truncation_order=12,
                       period=[0.8], wavelength=0.7, structure=_QUARTER, thickness=[0.25]),
    'tm_quarter': dict(tetm='tm', theta=0.15, phi=None, top=1.0, bot=2.25, truncation_order=12,
                       period=[0.8], wavelength=0.7, structure=_QUARTER, thickness=[0.25]),
    # Two patterned layers: the interface between them is under test, not just the boundary.
    'te_two_layer': dict(tetm='te', theta=0.25, phi=None, top=1.0, bot=4.0, truncation_order=10,
                         period=[0.7], wavelength=0.85, structure=_TWO_LAYER[::-1],
                         thickness=[0.22, 0.15]),
    'tm_two_layer': dict(tetm='tm', theta=0.25, phi=None, top=1.0, bot=4.0, truncation_order=10,
                         period=[0.7], wavelength=0.85, structure=_TWO_LAYER[::-1],
                         thickness=[0.22, 0.15]),
    # Denser truncation, where the two solvers' round-off has the most room to diverge.
    'te_high_order': dict(tetm='te', theta=0.2, phi=None, top=1.0, bot=2.25, truncation_order=30,
                          period=[0.7], wavelength=0.9, structure=_HALF, thickness=[0.15]),
    'tm_high_order': dict(tetm='tm', theta=0.2, phi=None, top=1.0, bot=2.25, truncation_order=30,
                          period=[0.7], wavelength=0.9, structure=_HALF, thickness=[0.15]),
    # Conical (azimuth != 0) — RETICOLO's other branch, and the solver's `solve_1d_conical`.
    'conical_te': dict(psi=np.pi / 2, theta=0.2, phi=0.3, top=1.0, bot=2.25, truncation_order=10,
                       period=[0.7], wavelength=0.9, structure=_HALF, thickness=[0.15]),
    'conical_tm': dict(psi=0.0, theta=0.2, phi=0.3, top=1.0, bot=2.25, truncation_order=10,
                       period=[0.7], wavelength=0.9, structure=_HALF, thickness=[0.15]),
}


def _solve(name, backend='numpy'):
    kw = dict(CASES[name])
    return wavebit.rcwa(backend=backend, fourier_coefficients='integrated',
                          **kw).solve()


def _golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} is missing — regenerate with gen_goldens_reticolo.py')
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize('name', list(CASES))
def test_orders_match_reticolo(name):
    """Per-order reflected and transmitted efficiency, against RETICOLO's own numbers.

    ``atol=1e-11`` is round-off room, not slack: the measured disagreement across these cases
    is 1e-15 to 1e-13, and it is bounded by how the two implementations order their
    arithmetic rather than by any modelling choice. A wrong Fourier order-index sign — the
    failure this oracle exists to catch, which mirrors +1 against -1 while conserving energy
    perfectly — moves an asymmetric case by O(0.1).
    """
    g = _golden()
    res = _solve(name)

    for side, arr in (('ri', np.asarray(res.R)), ('ti', np.asarray(res.T))):
        idx = g[f'{name}__idx_{side}']
        want = g[f'{name}__de_{side}']
        got = np.real(arr).ravel()[idx]
        assert got.shape == want.shape, (name, side, got.shape, want.shape)
        assert np.abs(got - want).max() < 1e-11, (name, side, np.abs(got - want).max())


@pytest.mark.parametrize('name', list(CASES))
def test_the_same_orders_propagate(name):
    """Every order RETICOLO did *not* report must be evanescent here, and carry no power.

    Without this the comparison above could pass while the solver put power into orders
    RETICOLO says cannot propagate — the selection would simply not look at them.
    """
    g = _golden()
    res = _solve(name)

    for side, arr in (('ri', np.asarray(res.R)), ('ti', np.asarray(res.T))):
        idx = set(g[f'{name}__idx_{side}'].tolist())
        rest = [v for i, v in enumerate(np.real(arr).ravel()) if i not in idx]
        assert max(abs(v) for v in rest) < 1e-12, (name, side, max(abs(v) for v in rest))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_reticolo_agreement_holds_on_every_backend(backend):
    """The oracle is about the physics, so it cannot be a NumPy-only result."""
    pytest.importorskip(backend)
    if backend == 'jax':
        wavebit.enable_x64()

    g = _golden()
    name = 'te_oblique'
    res = _solve(name, backend=backend)
    for side, arr in (('ri', np.asarray(res.R)), ('ti', np.asarray(res.T))):
        got = np.real(arr).ravel()[g[f'{name}__idx_{side}']]
        assert np.abs(got - g[f'{name}__de_{side}']).max() < 1e-9


ASYMMETRIC = ['te_quarter', 'tm_quarter', 'te_two_layer', 'tm_two_layer',
              'te_oblique', 'tm_oblique']


@pytest.mark.parametrize('name', ASYMMETRIC)
def test_the_order_axis_is_not_symmetric_in_these_cases(name):
    """Which cases can actually see a mirrored order axis — and it is not all of them.

    The failure this oracle is best placed to catch is a wrong Fourier order-index sign: it
    swaps +1 with -1, conserves energy perfectly, and is invisible to every other check here
    (``_core.conventions`` records that RETICOLO is what found it last time). But a grating
    that is mirror-symmetric, at normal incidence, has ``de(+m) == de(-m)`` *physically*, so
    reversing the order axis changes nothing and the comparison passes either way.

    Reversing ``T`` on the four symmetric cases above moves them by ~1e-15 — nothing. On
    these it moves them by 1e-3 to 3e-1. So the guard lives entirely in the asymmetric cases,
    and this test says so out loud rather than leaving it to be rediscovered by someone
    trimming the case list.
    """
    res = _solve(name)
    ti = np.real(np.asarray(res.T)).ravel()
    idx = _golden()[f'{name}__idx_ti']
    assert np.abs(ti[idx] - ti[::-1][idx]).max() > 1e-4, name


def _solve_bottom(name, backend='numpy'):
    """The same scene lit from below, at RETICOLO's own bottom-incidence k_parallel.

    RETICOLO answers both sides from one ``res2`` call at one ``k_parallel =
    n_top*sin(theta)``; Wavebit states the bottom-side angle in the incidence medium
    (``docs/conventions.md``), so the matching solve is ``incidence_side='bottom'`` at
    ``theta_bottom = asin(n_top*sin(theta)/n_bot)`` — same in-plane wavevector, said in the
    other medium's terms. Real for every case here since ``n_bot > n_top`` throughout.

    Snell is a statement about **indices**, so the two half-space permittivities are converted
    here rather than used raw. Skipping that conversion is not a type error and does not raise:
    ``asin(1.0 * sin(0.25) / 4.0)`` is a perfectly good angle, just the wrong one, and the
    comparison would then fail as a solver disagreement.
    """
    kw = dict(CASES[name])
    n_top = wavebit.index_from_eps(kw['top']).real
    n_bot = wavebit.index_from_eps(kw['bot']).real
    kw['theta'] = float(np.arcsin(n_top * np.sin(kw['theta']) / n_bot))
    return wavebit.rcwa(backend=backend, fourier_coefficients='integrated',
                          incidence_side='bottom', **kw).solve()


@pytest.mark.parametrize('name', list(CASES))
def test_bottom_incidence_orders_match_reticolo(name):
    """Backside per-order efficiencies against RETICOLO's own ``inc_bottom`` numbers.

    This is the one external solver pinning ``incidence_side='bottom'`` on actual
    diffraction orders (the analytic oracles in ``tests/test_backside_incidence.py`` stop at
    uniform stacks). The index sets swap roles relative to the top-side test: bottom-side
    reflection propagates in ``n_bot``, transmission in ``n_top``. Same 1e-11 as the top
    side — measured disagreement is 1e-16..1e-13.
    """
    g = _golden()
    if f'{name}__de_ri_bot' not in g:
        pytest.skip('golden predates bottom incidence — regenerate with '
                    'gen_goldens_reticolo.py')
    res = _solve_bottom(name)

    for side, idx_key, arr in (('ri_bot', 'idx_ti', np.asarray(res.R)),
                               ('ti_bot', 'idx_ri', np.asarray(res.T))):
        idx = g[f'{name}__{idx_key}']
        want = g[f'{name}__de_{side}']
        got = np.real(arr).ravel()[idx]
        assert got.shape == want.shape, (name, side, got.shape, want.shape)
        assert np.abs(got - want).max() < 1e-11, (name, side, np.abs(got - want).max())


@pytest.mark.parametrize('name', list(CASES))
def test_the_same_orders_propagate_from_below(name):
    """Orders evanescent on the bottom-incidence side must carry no power, same as on top."""
    g = _golden()
    if f'{name}__de_ri_bot' not in g:
        pytest.skip('golden predates bottom incidence — regenerate with '
                    'gen_goldens_reticolo.py')
    res = _solve_bottom(name)

    for idx_key, arr in (('idx_ti', np.asarray(res.R)), ('idx_ri', np.asarray(res.T))):
        idx = set(g[f'{name}__{idx_key}'].tolist())
        rest = [v for i, v in enumerate(np.real(arr).ravel()) if i not in idx]
        assert max(abs(v) for v in rest) < 1e-12, (name, idx_key)


@pytest.mark.parametrize('name', list(CASES))
def test_the_raster_path_holds_the_same_tolerance(name):
    """The sampled path is held to this too, and that is a recent thing.

    RETICOLO builds a binary grating's coefficients from the exact edge positions. So does
    the integrated path, and so does the sampled one now: a raster is a piecewise-constant
function, and the exact
    coefficients of one come out of an N-point transform without resampling it. Since these
    gratings have their edges on pixel boundaries, the raster *is* the structure, and all
    three agree to round-off.

    Until 2026-08 the sampled path upsampled the raster instead and took the transform of that,
    which converges to the same expression as 1/R — so this check would have failed at ~1e-5,
    and did: it used to assert the opposite, that the raster path could not be held here.
    """
    g = _golden()
    kw = dict(CASES[name])
    res = wavebit.rcwa(backend='numpy', fourier_coefficients='sampled', **kw).solve()
    for side, arr in (('ri', res.R), ('ti', res.T)):
        got = np.real(np.asarray(arr)).ravel()[g[f'{name}__idx_{side}']]
        assert np.abs(got - g[f'{name}__de_{side}']).max() < 1e-11, (name, side)
