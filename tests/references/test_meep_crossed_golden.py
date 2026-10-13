"""A crossed grating against Meep, from a checked-in golden.

The fourth oracle in this directory and the first that is not RCWA. RETICOLO, S4 and
fmmax all truncate a Fourier series and inherit whatever factorization that forces, which
is the same formulation ``src/wavebit/rcwa/_engine/engine.py`` uses; none of them can arbitrate the
**time-domain** path, whose errors are numerical dispersion, a staircased boundary and an
imperfect absorber instead. Meep is an FDTD. Nothing is shared but Maxwell's equations.

What it reaches that nothing here did:

* a **crossed grating** — periodic on both axes — solved in the time domain. Until this
  file, ``src/wavebit/fdtd/_periodic/legacy.py::_extract_crossed`` had no external reference of any kind: it is
  a line-for-line copy of ``_extract_conical`` and every per-order sign in it, the three
  ``sigma`` legs, the ``s_hat``/``p_hat`` frame and the ``n / kz_m`` secant, is invisible
  to ``R``, ``T`` and ``R + T``;
* per-order **complex amplitudes on a two-dimensional order grid**, for either path.
  ``docs/theory/rcwa/amplitude-sp-frame.md`` §2 lists what the other goldens compare — efficiencies,
  or fields, or 1D amplitudes — and concludes that the per-order s/p half-turn "could not
  have been reached" by any of them. This is the case that reaches it: the sign flips on
  every order whose ``Re(kx) < 0``, and both cells here have such orders.

The numbers are frozen by ``gen_goldens_meep_crossed.py``, which needs an environment
with Meep in it. That file carries the convention derivation and the five harness faults
it took to get there. **Meep is GPL-2.0-or-later and wavebit is Apache-2.0**: no file
under the meep package was opened, read, grepped or copied, nothing in the shipped
package imports it, and every statement about it is a black-box measurement — the same
footing this directory's README puts inkstone on. This module needs only numpy.

**What the tolerances are, and why they are not tighter.** Both sides discretise space,
so this comparison is bounded by two grids rather than one, and on a corner-rich crossed
raster **neither code converges at second order** — measured 0.69 to 1.63 here, against
the 1.83 / 1.50 ``docs/guide/fdtd/fdtd.md`` §8g records for wavebit's own crossed path. That was
measured before it was read, and Meep's *published documentation* — not its source — states
the same thing as a property of the method rather than of this harness: subpixel smoothing
gives "roughly second-order convergence", but "objects with sharp corners or edges are
associated with field singularities which introduce an unavoidable error **intermediate
between first and second order**"
(https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/, read 2026-09-07). A crossed
raster is corners everywhere; the 1D grating below has one vertical interface per period and
no corners, and both codes keep second order there. **Reading the documentation is not
reading the source**, and the distinction is deliberate: the licence footing above is about
implementation, and every published word about behaviour is fair to cite. So the golden
is frozen at two resolutions and ``test_the_golden_states_its_own_error_bar`` asserts the
claim rather than an estimate: the finer tier has to be nearer a converged RCWA answer
than the coarser one, by an order in a band a crossed grating can be in. That leaves
1.2e-2 to 3.7e-2 on the frozen numbers.

Against that, a defect of the kind this file exists to catch moves an amplitude by 1e-1 to
1.2 — a single flipped leg sign is ``2|t_p|``, which is 1.28 and 1.19 on the two cells —
so the resolving power is 13x to 28x rather than the ~1e13 an RCWA-against-RCWA golden
gets, and ``test_a_leg_sign_would_still_be_visible`` asserts that rather than assuming it.
It is an oracle for *structure*, exactly as the fmmax field row is; slow drift is a job for
the in-process comparisons in ``tests/fdtd/test_fdtd.py``.
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

from wavebit.fdtd._periodic.legacy import solve_fdtd

HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE))
from tests.references._meep_crossed_cases import CASES, SIGMA, eps_of, in_plane_k, margin   # noqa: E402

GOLDEN_PATH = HERE / 'golden_meep_crossed.npz'

#: An order this close to a Rayleigh point is excluded from the amplitude comparison.
#: ``kz = sqrt(n**2 - k_par**2)`` has sensitivity ``1/kz`` and the p channel multiplies
#: by ``n / kz`` again, so both codes are slowest exactly there — see
#: ``tests/fdtd/test_fdtd.py::test_the_crossed_amplitudes_carry_the_per_leg_signs``. The scenes
#: were chosen so that nothing is anywhere near it: the smallest margin over both cells
#: and both half-spaces is 0.33.
MARGIN = 0.05

#: Truncation order for the RCWA reference. Measured against (13, 13): the worst
#: amplitude moves by 2.2e-4 on the square cell and 9.2e-4 on the rectangular one, which
#: is 25x to 100x below the gaps asserted here.
RCWA_ORDERS = dict(x=9, y=9)

#: The grid the time-domain comparison runs on, and it is a **measurement rather than a
#: budget**: refining wavebit's side past the golden's own error buys nothing.
#:
#: wavebit's crossed run converges at second order against the converged RCWA answer —
#: 2.6e-2 / 6.9e-3 at resolution 25 / 50 on the square cell, 6.9e-2 / 2.5e-2 / 1.4e-2 at
#: 20 / 30 / 40 on the rectangular one. Meep does **not**, on either cell: 3.9e-2 ->
#: 1.6e-2 over the square's 25 -> 50 is order 1.26, and 6.6e-2 -> 5.0e-2 -> 4.3e-2 ->
#: 3.7e-2 over the rectangular one's 30 / 40 / 50 / 60 is order 0.99, 0.65, 0.80. So the
#: golden is the binding constraint and it stays there: taking wavebit's rectangular run
#: from 30 to 40 costs 2.7x the wall time and moves the gap 6.0e-2 -> 4.8e-2, against a
#: floor of 3.7e-2 that is Meep's alone.
#:
#: Both entries divide the raster exactly (7 columns into 21 cells, 8 x 12 into 24 x 36),
#: so the two codes see the same staircase and no part of the gap is a geometry difference.
FDTD_RESOLUTION = {'square_18deg': 25, 'rect_28deg': 30}

#: Bounds on the time-domain comparison: ~1.5x the measured gap (2.6e-2 and 6.0e-2), and
#: ``test_a_leg_sign_would_still_be_visible`` asserts the other side of the same
#: question — that a flipped leg sign is more than 10x this.
FDTD_BOUND = {'square_18deg': 4.5e-2, 'rect_28deg': 9.0e-2}


from tests.references._legacy_conventions import AMPLITUDES, centre_phase, centred_meep


def _leg_grid(sub, case, result, *, legacy=False):
    """``(len(orders_v), len(orders_u))`` slices of one leg, on the case's own window.

    Both solvers report on a window they choose, and the case file declares the one the
    golden holds; this aligns on the **labels** rather than on positions, which is the
    whole point of ``Result.orders_u`` / ``orders_v`` existing. Indexing by position
    instead is right while there is one ``v`` order and silently wrong once there are
    several, because the two axes are not the same length on a rectangular cell.
    """
    u = np.asarray(result.orders_u).ravel()
    v = np.asarray(result.orders_v).ravel()
    iu = [int(np.flatnonzero(u == m)[0]) for m in case['orders_u']]
    iv = [int(np.flatnonzero(v == l)[0]) for l in case['orders_v']]
    return {name: np.asarray(getattr(sub, name))[np.ix_(iv, iu)]
                  * (centre_phase(case) if legacy and name in AMPLITUDES else 1.)
            for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T')}


def _propagating(case, name):
    """Mask over the case's order grid: True where that channel carries power."""
    n = case['n_top'] if name in ('r_s', 'r_p', 'R') else case['n_bot']
    return np.array([[margin(case, int(m), int(l), n) > MARGIN
                      for m in case['orders_u']] for l in case['orders_v']])


def _rcwa(case):
    return _legacy_wavebit.rcwa(
        backend='numpy', wavelength=case['wavelength'], period=list(case['period']),
        truncation_order=RCWA_ORDERS, top=eps_of(case['n_top']), bot=eps_of(case['n_bot']),
        theta=np.deg2rad(case['theta_deg']), phi=np.deg2rad(case['phi_deg']),
        psi=np.deg2rad(30.0), structure=eps_of(case['index']),
        thickness=list(case['thickness'])).solve()


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with '
                    f'gen_goldens_meep_crossed.py in an environment that has Meep')
    with np.load(GOLDEN_PATH) as data:
        return centred_meep(data, CASES)


@pytest.fixture(scope='module')
def rcwa_solutions():
    return {name: _rcwa(case) for name, case in CASES.items()}


# ---------------------------------------------------------------------------------------
# 1. the frequency-domain path -- the tight comparison, bounded by Meep's grid alone
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('leg', ['s', 'p'])
@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_crossed_amplitudes_match_meep(golden, rcwa_solutions, name, leg):
    """Per-order complex ``r_s``/``r_p``/``t_s``/``t_p`` on the 2D order grid.

    The RCWA path is converged here — the worst amplitude moves 3.4e-4 (square) and
    9.2e-4 (rectangular) between ``truncation_order`` 17 and 21, and the gap below is flat
    to three digits across 7, 9, 13, 17 and 21 — so what this measures is Meep's own
    discretisation. Measured worst gap 1.6e-2 on the square cell and 3.7e-2 on the
    rectangular one; the bound is ~1.6x the larger.
    """
    case = CASES[name]
    got = _leg_grid(getattr(rcwa_solutions[name], f'{leg}_inc'), case,
                    rcwa_solutions[name])
    checked, worst, where = 0, 0.0, ''
    for key in ('r_s', 'r_p', 't_s', 't_p'):
        want = golden[f'{name}__{leg}__fine__{key}']
        mask = _propagating(case, key)
        for j, l in enumerate(case['orders_v']):
            for i, m in enumerate(case['orders_u']):
                if not mask[j, i]:
                    continue
                checked += 1
                gap = abs(complex(got[key][j, i]) - complex(want[j, i]))
                if gap > worst:
                    worst, where = gap, f'{leg}_inc.{key}({m},{l})'
    assert checked >= 20, f'only {checked} channels propagate; the comparison is thinning'
    assert worst < 6.0e-2, f'{name} {leg}: {worst:.2e} from Meep, worst at {where}'


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_crossed_efficiencies_match_meep(golden, rcwa_solutions, name):
    """``R`` and ``T`` per order, both legs.

    An efficiency cannot see any of the signs the test above exists for — that is the
    whole reason the amplitude comparison is the primary one here. It is kept because it
    is the one quantity a reader of ``docs/guide/fdtd/fdtd.md`` §2 can put beside the published
    Meep column, and because a *shape* error shows up here first.
    """
    case = CASES[name]
    worst = 0.0
    for leg in ('s', 'p'):
        got = _leg_grid(getattr(rcwa_solutions[name], f'{leg}_inc'), case,
                        rcwa_solutions[name])
        for key in ('R', 'T'):
            want = golden[f'{name}__{leg}__fine__{key}']
            assert want.shape == (len(case['orders_v']), len(case['orders_u']))
            worst = max(worst, float(np.abs(np.asarray(got[key]) - want).max()))
    assert worst < 4.5e-2, f'{name}: per-order efficiency gap {worst:.2e} from Meep'


# ---------------------------------------------------------------------------------------
# 2. the time-domain path -- what this oracle was built for
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
@pytest.mark.slow
def test_the_crossed_fdtd_matches_meep(golden, name):
    """``_extract_crossed`` against a solver that is not ours.

    This is the row the file exists for. The crossed extraction had no external oracle at
    all: ``tests/fdtd/test_fdtd.py`` compares it against **wavebit's own** RCWA path, which
    shares every convention with it by construction, so a convention wrong in both moves
    nothing. Here the frame, the order index, the y ladder's period and the three leg
    signs are all read off an FDTD that was written by somebody else.

    Both sides discretise, so this is the loosest comparison in the file and it is
    deliberately run on a grid CI can afford rather than on the finest one available.
    ``test_a_leg_sign_would_still_be_visible`` is what says the bound is doing work.
    """
    case = CASES[name]
    got = solve_fdtd(
        structure=eps_of(case['index']), thickness=list(case['thickness']),
        top=eps_of(case['n_top']), bot=eps_of(case['n_bot']),
        period=list(case['period']), wavelength=case['wavelength'],
        theta=np.deg2rad(case['theta_deg']), phi=np.deg2rad(case['phi_deg']),
        psi=np.deg2rad(30.0), resolution=FDTD_RESOLUTION[name], n_periods=90)
    worst, where, checked = 0.0, '', 0
    for leg in ('s', 'p'):
        grid = _leg_grid(getattr(got, f'{leg}_inc'), case, got, legacy=True)
        for key in ('r_s', 'r_p', 't_s', 't_p'):
            want = golden[f'{name}__{leg}__fine__{key}']
            mask = _propagating(case, key)
            for j, l in enumerate(case['orders_v']):
                for i, m in enumerate(case['orders_u']):
                    if not mask[j, i]:
                        continue
                    checked += 1
                    gap = abs(complex(grid[key][j, i]) - complex(want[j, i]))
                    if gap > worst:
                        worst, where = gap, f'{leg}_inc.{key}({m},{l})'
    assert checked >= 40, f'only {checked} channels compared across both legs'
    assert worst < FDTD_BOUND[name], (
        f'{name}: the time-domain path is {worst:.2e} from Meep, worst at {where}')


# ---------------------------------------------------------------------------------------
# 3. guards on the golden itself, and on the scenes
# ---------------------------------------------------------------------------------------

def test_the_golden_states_its_own_error_bar(golden, rcwa_solutions):
    """Every case is frozen at TWO resolutions, and the pair is the error bar.

    A golden without one is a number with no claim attached. This asserts the claim
    rather than the estimate: with the converged RCWA answer in hand, the finer tier must
    be **nearer** it than the coarser one, and the implied order must be in a band a
    crossed grating can actually be in.

    **The order is not 2, and that is a result rather than a disappointment.** Measured
    here: 1.26 and 1.63 on the square cell, 0.83 and 0.69 on the rectangular one — so the
    honest bar is the first-order reading ``|fine - coarse|`` (2.7e-2 to 3.3e-2) rather
    than the second-order ``|fine - coarse| / 3``, and the measured gaps against RCWA
    (1.2e-2 to 3.7e-2) sit between the two. ``docs/guide/fdtd/fdtd.md`` §9 lists as unmeasured
    whether the corner cells where a vertical and a horizontal interface meet cost first
    or second order; Meep's subpixel scheme is a different implementation of the same
    idea and degrades the same way on the same geometry, which is as close to an
    independent answer as this directory can give.

    An earlier version of this test asserted only ``|fine - coarse| / 3 < 3e-2`` and
    passed while the rectangular case's frozen numbers were 4.6e-2 wrong: both tiers
    rounded a 14.4-pixel layer to 14, so they carried the identical offset and agreed
    with each other. **A convergence study cannot see an error that does not depend on the
    grid**, which is why this compares against a reference instead.
    """
    for name, case in CASES.items():
        ref = rcwa_solutions[name]
        for leg in ('s', 'p'):
            want = _leg_grid(getattr(ref, f'{leg}_inc'), case, ref)
            gaps, spread = {}, 0.0
            for tier in ('fine', 'coarse'):
                worst = 0.0
                for key in ('r_s', 'r_p', 't_s', 't_p'):
                    mask = _propagating(case, key)
                    a = golden[f'{name}__{leg}__{tier}__{key}'][mask]
                    worst = max(worst, float(np.abs(a - want[key][mask]).max()))
                    if tier == 'fine':
                        b = golden[f'{name}__{leg}__coarse__{key}'][mask]
                        spread = max(spread, float(np.abs(a - b).max()))
                gaps[tier] = worst
            assert gaps['fine'] < gaps['coarse'], (
                f'{name} {leg}: halving the cell moved the golden AWAY from the '
                f'converged answer, {gaps["coarse"]:.2e} -> {gaps["fine"]:.2e}')
            order = np.log(gaps['coarse'] / gaps['fine']) / np.log(2.0)
            assert 0.5 < order < 2.5, f'{name} {leg}: implied order {order:.2f}'
            # The spread between the tiers has to BE the error, to within a factor: at
            # order q it is `(2**q - 1)` times it, which is 0.41 at q = 0.5 and 3 at
            # q = 2. Measured 0.79 to 2.46 here. A pair of tiers carrying an identical
            # grid-independent offset drives this toward zero -- the rounded-thickness
            # fault sat at 0.45 -- though what actually caught that one was the amplitude
            # comparison above, not this.
            assert 0.5 <= spread / gaps['fine'] <= 3.0, (
                f'{name} {leg}: the two tiers differ by {spread:.2e} while the fine one '
                f'is {gaps["fine"]:.2e} from the reference; the pair is not measuring it')


def test_the_golden_conserves_energy(golden):
    """Nothing in these scenes absorbs, so ``R + T`` is 1 up to the Yee grid's O(dz**2).

    A guard on the *frozen numbers* rather than on the solver, and a weak one on purpose:
    three of the five faults the generator records left ``R + T`` at 1 while the answer
    was wrong — the geometry wrapping through the PML axis returned ``T = 0.412`` against
    a true 0.684 with ``R + T`` **exactly** 1. So this catches only a gross failure, which
    is still worth refusing before a number is committed, and the two tests above are what
    actually arbitrate. The 1.25e-2 the square cell sits at is the grid, not a leak: it
    falls 5.0e-2 -> 1.25e-2 from the coarse tier to the fine one.
    """
    for name in CASES:
        for leg in ('s', 'p'):
            total = float(golden[f'{name}__{leg}__fine__R'].sum()
                          + golden[f'{name}__{leg}__fine__T'].sum())
            assert abs(total - 1.0) < 2e-2, f'{name} {leg}: R + T = {total:.6f}'


def test_the_incident_only_run_was_a_single_plane_wave(golden):
    """The harness's own instrument, frozen beside the answer.

    Decompose the incident-only run into orders: a plane wave has one, so whatever lands
    in the others is the harness talking. It is what caught two of the five faults the
    generator records — an unweighted monitor quadrature (9.9e-3 of the incident into
    orders a plane wave does not have) and an oblique source sheet not carrying its own
    Bloch phase (2.99). Cross-polarisation purity is the same idea one step on: the s leg
    is driven by an electric sheet along ``s_hat`` and must contain no p.
    """
    for name in CASES:
        for leg in ('s', 'p'):
            leak = float(golden[f'{name}__{leg}__fine__leak'])
            purity = float(golden[f'{name}__{leg}__fine__purity'])
            assert leak < 1e-12, f'{name} {leg}: incident-order leak {leak:.2e}'
            assert purity < 1e-3, f'{name} {leg}: cross-polarisation {purity:.2e}'


def test_the_golden_holds_the_orders_the_case_file_declares(golden):
    """The frozen numbers and the scene they describe have to stay in step.

    ``_meep_crossed_cases.py`` is read by both halves, so editing a period, an angle or an
    order window there silently re-labels a golden nobody regenerated — the file would
    still load, the shapes would still match, and every comparison would be against the
    wrong scene. The order labels are what the golden can check on its own; the rest is
    the generator's docstring rule that these numbers are never regenerated to make a
    failing test pass.
    """
    for name, case in CASES.items():
        assert list(golden[f'{name}__orders_u']) == list(case['orders_u'])
        assert list(golden[f'{name}__orders_v']) == list(case['orders_v'])
        for leg in ('s', 'p'):
            assert golden[f'{name}__{leg}__fine__r_s'].shape == (
                len(case['orders_v']), len(case['orders_u']))


def test_a_leg_sign_would_still_be_visible(golden):
    """The bound in the comparisons above has to be doing work, and this is the check.

    ``_extract_crossed`` carries three leg signs and each is invisible to ``R``, ``T`` and
    ``R + T``: flipping the transmitted p sign moves every efficiency by exactly zero and
    every ``t_p`` by ``2 |t_p|``. So the question that decides whether this file is an
    oracle is how ``2 |t_p|`` compares with the tolerance, and it is asserted rather than
    assumed.

    **Over the legs together, not each one separately**, and the difference is the point.
    A sign lives in the extraction, so it moves both legs at once and one witness is
    enough — but the s leg's ``t_p`` is *cross-polarised* scattering, an order of
    magnitude smaller (0.256 against 1.279 on the square cell), so a per-leg version of
    this assertion would demand a tolerance the physics of that channel cannot supply.
    """
    for name, case in CASES.items():
        mask = _propagating(case, 't_p')
        biggest = max(float(np.abs(golden[f'{name}__{leg}__fine__t_p'][mask]).max())
                      for leg in ('s', 'p'))
        assert 2.0 * biggest > 10.0 * FDTD_BOUND[name], (
            f'{name}: a flipped transmitted p sign would move the worst amplitude by '
            f'{2 * biggest:.3f}, against a {FDTD_BOUND[name]:g} bound')


def test_the_scenes_are_genuinely_crossed_and_see_a_transpose():
    """A raster that varies in only one axis is the 2D case again, and a y-symmetric one
    cannot see a row-order defect.

    The second half is not hypothetical here. ``tests/references/README.md`` records that
    every external oracle this project used was once blind to the ucell row order, because
    every 2D raster it had been handed was y-symmetric.
    """
    for name, case in CASES.items():
        a = np.asarray(case['index'])[0]
        assert np.ptp(a.mean(axis=0)) > 0.05, f'{name} does not vary along x'
        assert np.ptp(a.mean(axis=1)) > 0.05, f'{name} does not vary along y'
        assert not np.allclose(a, a[::-1, :]), f'{name} is mirror-symmetric in y'
        assert not np.allclose(a, a[:, ::-1]), f'{name} is mirror-symmetric in x'
        if a.shape[0] == a.shape[1]:
            assert not np.allclose(a, a.T), f'{name} is its own transpose'


def test_the_rectangular_cell_radiates_a_y_order():
    """``Px != Py`` *and* a propagating ``l != 0`` order, which is the pair that matters.

    A y ladder written on the x period — ``l * wl / Px`` where ``l * wl / Py`` belongs —
    survives a square cell untouched, and on a rectangular one with every ``l != 0`` order
    evanescent it reaches nothing that carries power. Both conditions have to hold at
    once, in the same scene, or the case is measuring nothing new.
    """
    case = CASES['rect_28deg']
    px, py = case['period']
    assert px != py, 'the rectangular case has become square'
    radiating = [int(l) for l in case['orders_v']
                 if l != 0 and any(margin(case, int(m), int(l), case['n_top']) > MARGIN
                                   for m in case['orders_u'])]
    assert radiating, 'no y order radiates above, so a y ladder built on Px is invisible'


def test_the_half_turn_half_plane_is_covered():
    """The per-order s/p frame reverses on ``Re(kx) < 0``, so the set must reach there.

    ``docs/theory/rcwa/amplitude-sp-frame.md`` records a year of that reversal being wrong while every
    external golden agreed, because none of them compared amplitudes across orders on a 2D
    grid. This one does — but only if orders on both sides of ``kx = 0`` propagate.
    """
    for name, case in CASES.items():
        signs = {np.sign(in_plane_k(case, int(m), int(l))[0])
                 for m in case['orders_u'] for l in case['orders_v']
                 if margin(case, int(m), int(l), case['n_bot']) > MARGIN}
        assert signs == {-1.0, 1.0}, (
            f'{name} has propagating orders only on the {signs} side of kx = 0, so the '
            f'frame half-turn is not exercised')


def test_the_leg_signs_are_the_ones_the_generator_derived():
    """The generator and ``src/wavebit/fdtd/_periodic/legacy.py`` reach ``(-1, +1, -1)`` independently.

    Not a convention either of them chose: ``p_hat = k_hat x s_hat`` plus ``E . k_hat = 0``
    gives ``E . p_hat = -+ (n / kz)(c Ex + s Ey)`` with the upper sign downgoing, which is
    incident ``-1``, reflected ``+1``, transmitted ``-1``. This pins the two spellings
    together so that a change to either is a failing test rather than a silent
    disagreement absorbed by the tolerance.
    """
    import inspect
    from wavebit.fdtd._periodic import legacy as fdtd
    sig = inspect.signature(fdtd._extract_crossed).parameters['sigma'].default
    assert tuple(float(s) for s in sig) == SIGMA


def test_the_meep_set_is_not_another_oracle_s_configurations():
    """Agreement between two oracles handed identical inputs is worth less than it looks.

    Everything here is deliberately off the other sets' grid — micrometres rather than
    nanometres, crossed rather than 1D, and a conical mount both of the others' 2D cases
    lack.
    """
    from tests.references._fmmax_cases import WL as FMMAX_WL, PERIOD as FMMAX_PERIOD

    for case in CASES.values():
        assert case['wavelength'] != FMMAX_WL
        assert case['period'][0] != FMMAX_PERIOD
        assert np.asarray(case['index']).shape[1] > 1, 'this set must be crossed'
