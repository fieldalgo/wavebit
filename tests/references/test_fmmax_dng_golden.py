"""Gratings on, containing and lit from a **lossy double negative**, against fmmax's ``FFT``.

The outside check on the negative-index work (``docs/branch-cuts.md`` §9): the
double-negative half-space root (the mirror of Maystre's cut), the incidence rule for a lit
negative-index side (``k_par = s*N*sin(theta)``, through the ``theta`` door), and patterned
``eps`` *and* ``mu`` with double-negative regions, on ETM and on SMM. fmmax made none of those
choices: it keeps every layer's modes with ``Im(kz) > 0``, which is the physical rule for any
lossy medium whatever the sign of its index, and every negative-index medium here is lossy with
``|mu| != 1`` (``_fmmax_dng_cases.py``).

Measured with each rule reverted: the mirror cut switched off fails every case, an unsigned
``k_par`` or an unsigned index fails the ``theta`` door of ``lit_from_dng_1d``, and develop
(which refused a double-negative half-space) fails 32 of 35. What this golden cannot see is
ETM's *labelling* of a double-negative layer's modes: on these lossy scenes the labels move
only conditioning (ETM against SMM to 1e-15 with either label rule), and the scenes where they
decide a ``LinAlgError`` are lossless ones, where fmmax's own mode choice has no loss to read.
``tests/test_negative_index_layers.py`` holds those, against SMM and closed forms.

**Identity, not convergence.** ``factorization='laurent'`` assembles the operator fmmax's
``Formulation.FFT`` does, for ``eps`` and ``mu`` alike, so what separates the two codes is
fmmax's midpoint rule for each pixel's Fourier integral, extrapolated away over the samples per
pixel (``_fmmax_laurent_cases.py`` derives the ``1/R^2`` series), and fmmax 1.7.1's
single-precision centring phase, which the generator's second run removes. The compared quantity
is each order's **unpolarized** efficiency, ``trace(H_in^-1 H_out)/2``, which needs no s/p basis
matched between the codes.

Measured when the golden was frozen (2026-10-04, fmmax 1.7.1, numpy 2.4.6), worst order of
``R`` and ``T`` against the double-precision-phase run, ETM and SMM, through both incidence
doors:

======================  ==========  ===============  ===========  ========  ===============
case                    Wavebit     tolerance used   fmmax stock  Li rule   mirror medium
======================  ==========  ===============  ===========  ========  ===============
``grating1d_on_dng``    5.9e-14     4.7e-12          2.3e-9       9.3e-3    see the test
``grating2d_on_dng``    7.6e-14     5.6e-12          2.5e-9       8.1e-3    see the test
``dng_pattern_on_dng``  6.2e-12     3.9e-10          3.9e-9       4.1e-2    see the test
``lit_from_dng_1d``     2.7e-14     2.7e-12          2.5e-9       1.1e-2    see the test
======================  ==========  ===============  ===========  ========  ===============

The tolerance is the extrapolation's own -- the distance between the estimates from the coarser
and the finer three samplings, which bounds the finer one's error from above -- plus 1e-12 of
round-off (``ROUND_OFF``); the column includes both. Under numpy 2.5.1 the Wavebit column reads
5.9e-14, 7.5e-14, 6.1e-12 and 2.7e-14 (``studies/negative_index_media.py``, section 8 of its
log): another BLAS, the last digit. The raw gap falls by 4.00 per doubling of the samples at every level, the premise the
extrapolation rests on, and the test checks it before using it.

Values are frozen by ``gen_goldens_fmmax_dng.py``, which needs the ``fwref`` env. Nothing here
imports fmmax. Never regenerate to make this pass: the numbers did not come from this codebase,
which is the whole of their worth.
"""
import pathlib
import sys
import warnings

import numpy as np
import pytest

import wavebit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _fmmax_dng_cases import CASES, LEVELS, THICK, WL, incident_kxy, orders   # noqa: E402
from _fmmax_laurent_cases import richardson                                    # noqa: E402

GOLDEN_PATH = HERE / 'golden_fmmax_dng.npz'

#: Below this two float64 RCWA codes cannot be asked to agree.
ROUND_OFF = 1e-12


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with gen_goldens_fmmax_dng.py')
    return np.load(GOLDEN_PATH)


def _solve(name, algo='SMM', door='k', factorization='laurent', top=None, bot=None, kxy=None):
    """The case on NumPy. ``door='k'`` hands Wavebit the frozen in-plane wavevector;
    ``door='theta'`` hands it the case's angles and lets the solver's own rule make one."""
    case = CASES[name]
    one_d = len(case['period']) == 1
    kw = dict(backend=0, top=wavebit.Material(*(case['top'] if top is None else top)),
              bot=wavebit.Material(*(case['bot'] if bot is None else bot)), wavelength=WL,
              thickness=[THICK], structure=case['eps'][None], factorization=factorization,
              layer_algorithm=algo,
              truncation_order=case['order'][0] if one_d else case['order'],
              period=[case['period'][0]] if one_d else case['period'])
    if case['mu'] is not None:
        kw['mu'] = case['mu'][None]
    if door == 'k':
        k0 = 2 * np.pi / WL
        kx, ky = incident_kxy(name) if kxy is None else kxy
        kw.update(kx=k0 * kx, ky=k0 * ky)
    else:
        kw.update(theta=case['theta'], phi=case['phi'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return wavebit.rcwa(**kw).solve(auto_detune=False)


def _unpolarized(result, name, mirrored=False):
    """``[R/T, order]`` in the golden's order sequence: the mean of the s and p inputs, which
    are flux-orthogonal in an isotropic incident medium of any loss. ``mirrored`` reads every
    order at ``(-m, -n)`` instead -- the answer a flipped order index would give."""
    mx, my = CASES[name]['order']
    out = np.zeros((2, len(orders(name))))
    for sub in (result.s_inc, result.p_inc):
        for k, grid in enumerate((np.asarray(sub.R), np.asarray(sub.T))):
            grid = grid.reshape(2 * my + 1, 2 * mx + 1)
            for i, (m, n) in enumerate(orders(name)):
                if mirrored:
                    m, n = -m, -n
                out[k, i] += grid[n + my, m + mx] / 2
    return out


def _levels(golden, name, flavour):
    return [golden[f'{name}__{flavour}__R{samples}__per_order'] for samples in LEVELS]


def _extrapolated(golden, name, flavour):
    """The estimate from the three finest samplings, and a bound on its error."""
    values = _levels(golden, name, flavour)
    fine = richardson(values[1:], LEVELS[1:])
    coarse = richardson(values[:3], LEVELS[:3])
    return fine, np.abs(fine - coarse).max() + ROUND_OFF


@pytest.fixture(scope='module')
def solved():
    return {(name, algo, door): _unpolarized(_solve(name, algo, door), name)
            for name in CASES for algo in ('ETM', 'SMM') for door in ('k', 'theta')}


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_only_gap_is_the_quadrature(golden, solved, name):
    """The premise, checked before it is used: the raw gap falls fourfold per doubling of the
    samples per pixel -- the ``1/R^2`` of the midpoint rule -- at every level. A different root,
    index sign, order set or orientation would not fall at all."""
    got = solved[name, 'SMM', 'k']
    gaps = [np.abs(level - got).max() for level in _levels(golden, name, 'phase64')]
    ratios = np.asarray(gaps[:-1]) / np.asarray(gaps[1:])
    assert np.all((ratios > 3.9) & (ratios < 4.1)), (name, gaps, ratios)


@pytest.mark.parametrize('door', ['k', 'theta'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_every_order_is_fmmax_fft(golden, solved, name, algo, door):
    """Every order's unpolarized ``R`` and ``T``, placed by the order indices themselves, with
    ETM and SMM and through both incidence doors. Through ``theta`` the solver makes its own
    in-plane wavevector, so on ``lit_from_dng_1d`` this is the incidence rule for a
    negative-index lit side held to an outside solver: the light's direction, ``k_par =
    s*N*sin(theta)`` with ``s = -1``. The mirror of the order index has to miss, or the test
    could not have told an order flip from agreement."""
    want, tolerance = _extrapolated(golden, name, 'phase64')
    got = solved[name, algo, door]
    gap = np.abs(got - want).max()
    assert gap < tolerance, (f'{name} [{algo}, {door}]: {gap:.2e} from fmmax FFT, past '
                             f'{tolerance:.2e}\n  got  {np.array2string(got, precision=13)}'
                             f'\n  want {np.array2string(want, precision=13)}')
    mirrored = np.abs(_unpolarized(_solve(name, algo, door), name, mirrored=True) - want).max()
    assert mirrored > 1e3 * tolerance, (name, mirrored, tolerance)
    assert np.count_nonzero(want.max(axis=0) > 1e-6) >= 3, 'too few propagating orders'


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_fmmax_as_shipped_differs_by_its_single_precision_phase_only(golden, solved, name):
    """The stock run is still an oracle, with its disagreement named: Wavebit sits within twice
    the distance between fmmax's two runs, which differ only in the dtype of that phase."""
    stock, _ = _extrapolated(golden, name, 'stock')
    phase64, tolerance = _extrapolated(golden, name, 'phase64')
    spread = np.abs(stock - phase64).max()
    assert 1e-11 < spread < 1e-6, (name, spread)
    assert np.abs(solved[name, 'SMM', 'k'] - stock).max() < 2 * spread + tolerance


#: The cases whose double negatives are all half-spaces, which a mirror can swap whole.
UNIFORM = [name for name in CASES if CASES[name]['mu'] is None]


@pytest.mark.parametrize('name', UNIFORM, ids=UNIFORM)
def test_the_golden_tells_a_double_negative_from_its_mirror(golden, name):
    """Agreeing with this golden is a statement about the negative-index medium, not about the
    grating alone. Each double negative ``(eps, mu)`` replaced by its double-positive mirror
    ``(-conj(eps), -conj(mu))`` -- the same ``|n|``, the same loss, the other sign of index --
    misses by far more than the tolerance. So does lighting the negative-index side with the
    unsigned ``k_par = +N*sin(theta)`` that a positive-index rule would give."""
    want, tolerance = _extrapolated(golden, name, 'phase64')
    case = CASES[name]

    def mirror(pair):
        eps, mu = pair
        return (-np.conj(eps), -np.conj(mu)) if eps.real < 0 and mu.real < 0 else pair

    swapped = _unpolarized(_solve(name, top=mirror(case['top']), bot=mirror(case['bot'])), name)
    assert np.abs(swapped - want).max() > 1e6 * tolerance
    if case['top'][0].real < 0:
        kx, ky = incident_kxy(name)
        unsigned = _unpolarized(_solve(name, kxy=(-kx, -ky)), name)
        assert np.abs(unsigned - want).max() > 1e6 * tolerance


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_golden_tells_the_rules_apart(golden, name):
    """Li's rules answer these scenes differently by far more than the tolerance, so agreeing
    with the golden is also a statement about the factorization rule it was solved with."""
    want, tolerance = _extrapolated(golden, name, 'phase64')
    li = _unpolarized(_solve(name, factorization='li'), name)
    assert np.abs(li - want).max() > 1e6 * tolerance


#: Lit from glass, so the incident and reflected fluxes add without a cross term.
FROM_GLASS = [name for name in CASES if CASES[name]['top'][0].real > 0]


@pytest.mark.parametrize('name', FROM_GLASS, ids=FROM_GLASS)
def test_energy_is_accounted_for(solved, name):
    """Lit from lossless glass, everything not reflected enters the stack: with a lossless
    grating all of it reaches the lossy half-space, ``R + T = 1``; with double-negative pixels
    in the grating some is absorbed on the way, ``R + T < 1``. A theorem rather than a
    comparison, so it holds whatever the oracle says. (Lit from an absorbing medium, the
    incident and reflected waves interfere in the flux and ``R + T`` is not that balance.)"""
    got = solved[name, 'SMM', 'k'].sum(axis=1)
    if CASES[name]['mu'] is None:
        assert abs(got.sum() - 1.0) < 1e-12, (name, got)
    else:
        assert 0.0 < 1.0 - got.sum() < 1.0, (name, got)


def test_the_golden_holds_the_scenes_the_cases_state(golden):
    for name in CASES:
        assert [tuple(int(v) for v in row) for row in golden[f'{name}__orders']] == orders(name)
        np.testing.assert_array_equal(golden[f'{name}__kxy'], np.asarray(incident_kxy(name)))
    assert tuple(golden['__levels']) == LEVELS
