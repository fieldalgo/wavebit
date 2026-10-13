"""``factorization='laurent'`` against fmmax's ``Formulation.FFT``, from a checked-in golden.

The two assemble the same operator -- ``[[eps]]`` in both in-plane slots, ``[[eps]]^-1`` in the
z slot -- so this is a comparison of **identity**, not of convergence, once the one difference
left is taken out: fmmax's DFT is the midpoint rule for the Fourier integral Wavebit evaluates
exactly on each pixel. ``_fmmax_laurent_cases.py`` derives the factor, ``y / sin(y)`` per axis
and even in ``1/R``, and the golden holds fmmax's answer at four samplings ``R`` so the test can
extrapolate it away and check the premise first.

Measured when the golden was frozen (2026-09-27), totals and every order of both
polarizations, against the double-precision-phase run:

====================  ===========  ===============  ===================  ============
case                  Wavebit gap  tolerance used   mirrored order index  Li / NVF
====================  ===========  ===============  ===================  ============
``rod_eps12_rect``    6.1e-13      3.1e-11          4.8e-2               2.0e-1 / 1.3e-1
``hole_metal_rect``   3.2e-11      2.1e-9           1.3e-2               8.3e-2 / 1.7e-1
``cross_eps6_hex``    4.8e-13      8.8e-12          7.4e-2               refused / 4.2e-2
====================  ===========  ===============  ===================  ============

The tolerance is the extrapolation's own: the distance between the estimates from the coarser
and the finer three samplings, which bounds the finer one's error from above (its residual is
``1/R^6``, 64 times smaller), with 1e-12 as the round-off floor.

**fmmax as shipped sits 1e-8 to 1e-7 away, and that is fmmax's.** Its pixel-centring phase is
computed from a single-precision frequency grid (``fmmax.fft._fft2`` in fmmax 1.7.1), which
turns an exact translation into an order-dependent one at the 1e-8 level. The generator freezes
that run too; the test holds Wavebit to it within twice the distance between the two runs --
so the stock oracle is still checked, and its only disagreement is named.

Values are frozen by ``gen_goldens_fmmax_laurent.py``, which needs the ``fwref`` env. Nothing
here imports fmmax. Never regenerate to make this pass: the numbers did not come from this
codebase, which is the whole of their worth.
"""

from wavebit.compat import legacy as _legacy_wavebit

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import sys

import numpy as np
import pytest


HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE))
from tests.references._fmmax_laurent_cases import CASES, LEVELS, ORDER, THETA, THICK, WL, lossless, richardson

GOLDEN_PATH = HERE / 'golden_fmmax_laurent.npz'

#: Below this two float64 RCWA codes cannot be asked to agree.
ROUND_OFF = 1e-12


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with gen_goldens_fmmax_laurent.py')
    return np.load(GOLDEN_PATH)


def _solve(name, factorization='laurent', fourier_coefficients='sampled'):
    basis, raster, top, bot = CASES[name]
    kw = dict(backend='numpy', top=top, bot=bot, theta=np.deg2rad(THETA), phi=0.0, wavelength=WL,
              thickness=[THICK], truncation_order=(ORDER, ORDER), structure=raster[None],
              factorization=factorization, fourier_coefficients=fourier_coefficients)
    if np.allclose(basis, np.diag(np.diag(basis))):
        kw['period'] = (basis[0, 0], basis[1, 1])
    else:
        kw['lattice'] = basis
    return _legacy_wavebit.rcwa(**kw).solve()


def _totals(result):
    return np.array([float(np.sum(result.s_inc.R)), float(np.sum(result.s_inc.T)),
                     float(np.sum(result.p_inc.R)), float(np.sum(result.p_inc.T))])


def _levels(golden, name, flavour, what='totals'):
    return [golden[f'{name}__{flavour}__R{samples}__{what}'] for samples in LEVELS]


def _extrapolated(golden, name, flavour, what='totals'):
    """The estimate from the three finest samplings, and a bound on its error."""
    values = _levels(golden, name, flavour, what)
    fine = richardson(values[1:], LEVELS[1:])
    coarse = richardson(values[:3], LEVELS[:3])
    return fine, np.abs(fine - coarse).max() + ROUND_OFF


@pytest.fixture(scope='module')
def solved():
    return {name: _solve(name) for name in CASES}


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_only_gap_is_the_quadrature(golden, solved, name):
    """The premise, checked before it is used: the raw gap falls fourfold per doubling of the
    samples per pixel -- the ``1/R^2`` of ``y / sin(y)`` -- at every level. A difference of rule,
    of order set or of orientation would not fall at all."""
    got = _totals(solved[name])
    gaps = [np.abs(level - got).max() for level in _levels(golden, name, 'phase64')]
    ratios = np.asarray(gaps[:-1]) / np.asarray(gaps[1:])
    assert np.all((ratios > 3.9) & (ratios < 4.1)), (name, gaps, ratios)


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_laurent_is_fmmax_fft_in_total(golden, solved, name):
    want, tolerance = _extrapolated(golden, name, 'phase64')
    got = _totals(solved[name])
    assert np.abs(got - want).max() < tolerance, (
        f'{name}: {np.abs(got - want).max():.2e} from fmmax FFT, past {tolerance:.2e}\n'
        f'  got  {np.array2string(got, precision=13)}\n  want {np.array2string(want, precision=13)}')


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_laurent_is_fmmax_fft_order_by_order(golden, solved, name):
    """Every order of both polarizations, placed by the order indices themselves. Five orders
    propagate at this wavelength, asymmetrically, so the mirror of the order index is measured
    as well and has to miss -- otherwise this could not have told the two apart."""
    want, tolerance = _extrapolated(golden, name, 'phase64', 'per_order')   # [pol, R/T, order]
    orders = golden[f'{name}__orders']
    result = solved[name]
    worst = mirrored = 0.0
    for pol, sub in enumerate((result.s_inc, result.p_inc)):
        for kind, grid in enumerate((np.asarray(sub.R), np.asarray(sub.T))):
            for i, (m, n) in enumerate(orders):
                worst = max(worst, abs(grid[n + ORDER, m + ORDER] - want[pol, kind, i]))
                mirrored = max(mirrored, abs(grid[-n + ORDER, -m + ORDER] - want[pol, kind, i]))
    assert worst < tolerance, (name, worst, tolerance)
    assert mirrored > 1e3 * tolerance, (name, mirrored, tolerance)
    lit = (want[0, 0] > 1e-6) | (want[0, 1] > 1e-6)
    assert np.count_nonzero(lit) >= 4, 'too few propagating orders to see a mirror'


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_both_coefficient_series_are_the_same_operator(solved, name):
    """A raster is a staircase and both series integrate it exactly, so under Laurent's rule --
    which asks nothing else of the coefficients -- the two builds agree to round-off."""
    integrated = _totals(_solve(name, fourier_coefficients='integrated'))
    assert np.abs(integrated - _totals(solved[name])).max() < 1e-11


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_fmmax_as_shipped_differs_by_its_single_precision_phase_only(golden, solved, name):
    """The stock run is still an oracle, with its disagreement named: Wavebit sits within twice
    the distance between fmmax's two runs, which differ only in the dtype of that phase."""
    stock, _ = _extrapolated(golden, name, 'stock')
    phase64, tolerance = _extrapolated(golden, name, 'phase64')
    spread = np.abs(stock - phase64).max()
    assert 1e-10 < spread < 1e-6, (name, spread)
    assert np.abs(_totals(solved[name]) - stock).max() < 2 * spread + tolerance


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_golden_tells_the_rules_apart(golden, name):
    """Li's rules and NVF answer the same scenes differently by far more than the tolerance, so
    agreeing with this golden is a statement about the rule. On the hexagonal cell Li is refused
    -- its inverse rule runs along lattice axes that are not x and y -- which is why Laurent's
    rule, naming no axis, is the one a comparison there can use."""
    want, tolerance = _extrapolated(golden, name, 'phase64')
    basis = CASES[name][0]
    skewed = not np.allclose(basis, np.diag(np.diag(basis)))
    for rule in ('li', 'nvf'):
        if rule == 'li' and skewed:
            with pytest.raises(NotImplementedError, match="factorization='li'"):
                _solve(name, factorization=rule)
            continue
        assert np.abs(_totals(_solve(name, factorization=rule)) - want).max() > 1e6 * tolerance


@pytest.mark.parametrize('name', [n for n in CASES if lossless(n)])
def test_energy_closes_on_the_lossless_cases(solved, name):
    """A theorem rather than a comparison, so it holds whatever the oracle says: ``[[eps]]`` of a
    real permittivity is Hermitian, and the layer conserves energy at any truncation."""
    got = _totals(solved[name])
    for r, t in (got[:2], got[2:]):
        assert abs(r + t - 1.0) < 1e-12


def test_the_golden_holds_the_order_set_the_solver_uses(golden):
    want = {(m, n) for m in range(-ORDER, ORDER + 1) for n in range(-ORDER, ORDER + 1)}
    for name in CASES:
        orders = {tuple(int(v) for v in row) for row in golden[f'{name}__orders']}
        assert orders == want, name
    assert tuple(golden['__levels']) == LEVELS
