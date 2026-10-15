"""The 1D benchmark grating against Meep, from a checked-in golden.

The fifth oracle in this directory and the second that is not RCWA. It shares a bridge
with ``test_meep_crossed_golden.py`` and answers a different question.

**What it is for: the published table.** ``docs/guide/fdtd/fdtd.md`` §2 prints a three-way agreement
-- the RCWA path, this project's FDTD, and Meep -- on one grating, and ``README.md``
quotes its worst per-order gaps as the FDTD path's headline validation. Those Meep
numbers were computed in a *sibling lab repository* on 2026-08-25 and **quoted, not
checked**: until this file, nothing in this repository could reproduce them. Since
2026-09-06 that column *is* ``golden_meep_1d.npz``. **Where §2's printed digits are
checked is ``test_fdtd_table_golden.py``**, which since 2026-09-07 parses the two tables
out of the markdown and holds every cell against ``golden_fdtd_table.npz`` -- all three
columns, with the conditions they were taken at beside them. Four hand-typed copies of
those digits lived in this module until then; a transcription of a document is one more
place for the document to drift from, so they are gone and what stays here is the part
that file cannot do -- the physics comparisons between this golden and the two solvers.
The crossed golden reaches further into the solver -- a 2D order grid, a conical mount,
per-order amplitudes -- but it does not touch the one table a reader is actually shown.

**Re-deriving the table moved two of its three columns, and the second one is the finding.**
The Meep column came back 5.3e-4 from the quoted one, which is inside what §2 records about
the quoted run and is held at that in ``test_the_2026_08_25_lab_column_is_reproduced``. The
*wavebit* column came back 8e-5 out -- because the printed values were a ``pad_scale=2.0``
run and the shipped clearance had since been halved, with the paragraph beneath the table
re-derived and the table itself not. That is what a column no test re-solves does, and it is
why all three are re-solved here rather than only the outside one.

**The scene is not a new one, deliberately.** It is ``tests/fdtd/test_fdtd.py::_grating()`` at
that module's own constants, and ``test_the_scene_is_the_one_tests_test_fdtd_uses``
asserts that rather than trusting the comment in ``_meep_1d_cases.py`` that says so. An
oracle stated on a scene *near* the one under test is a different oracle.

**The calibration is the first three tests and it involves no solver at all.**
``slab_uniform`` is the same 0.30 layer with the patterning removed, and a planar stack has
an exact closed form. ``_meep_1d_cases.py::airy_rt`` writes that form out from the boundary
conditions; the frozen Meep amplitudes are held to it directly. That is what says the
bridge -- flux normalisation, reference planes, the frame -- can answer at all, before any
grating answer taken through the same bridge is believed. It reaches three things no
efficiency comparison anywhere in this directory can: the **absolute** normalisation, the
**phase**, and the **p leg's sign**.

**The polarisation map is the trap this scene is most exposed to**, because TE and TM are
both legal answers to the same geometry and differ by no theorem. ``_meep_1d_cases.py``'s
``LEGS`` states the mapping and why the generator drives a named *component* rather than
asking Meep for a polarisation by name. ``test_the_two_legs_are_not_interchangeable`` is
the assertion that the mapping is doing work: the two legs' efficiencies differ by 8e-2 to
1.9e-1 against tolerances of ~1e-3, so a swapped label is a 60x failure rather than a
tolerance question -- while ``R + T`` stays at 1 for both.

The numbers are frozen by ``gen_goldens_meep_1d.py``, which needs an environment with Meep
in it. That file carries the calibration gate and the sixth harness fault -- a block face
meep smooths across even when the material is the same on both sides of it. **Meep is
GPL-2.0-or-later and wavebit is Apache-2.0**: no file under the meep package was opened,
read, grepped or copied, nothing in the shipped package imports it, and every statement
about it is a black-box measurement -- the same footing this directory's README puts
inkstone on. This module needs only numpy and wavebit.

**What the tolerances are, and why they are not tighter.** Both sides discretise space, so
this comparison is bounded by two grids rather than one. Unlike the crossed cells, a 1D
grating has one vertical interface per period and no corners, and **both codes keep second
order on it** -- so the golden is frozen at two resolutions and
``test_the_golden_states_its_own_error_bar`` asserts that claim rather than an estimate.
Slow drift is still the job of the in-process comparisons in ``tests/fdtd/test_fdtd.py``.
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

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
from tests.references._meep_1d_cases import CASES, LEGS, SIGMA, airy_rt, eps_of, in_plane_k, margin

GOLDEN_PATH = HERE / 'golden_meep_1d.npz'

#: An order this close to a Rayleigh point is excluded from the amplitude comparison.
#: ``kz = sqrt(n**2 - kx**2)`` has sensitivity ``1/kz`` and the p channel multiplies by
#: ``n / kz`` again, so both codes are slowest exactly there. This scene is nowhere near
#: one: the smallest margin over both half-spaces is 0.60.
MARGIN = 0.05

#: Truncation order for the RCWA reference, and ``docs/guide/fdtd/fdtd.md`` §2's own choice, read
#: from the record so that one place names it. Measured against ``truncation_order = 400``
#: on this scene: TE has converged (worst per-order move 1.2e-6), TM has not quite (1.3e-5),
#: so the reference itself carries ~1e-5 and every bound below is 50x clear of it.
#:
#: ``FDTD_PUBLISHED`` is the 5 nm cell §2's "this module" column is taken on, likewise.
from tests.references._meep_1d_cases import FDTD_PUBLISHED, RCWA_ORDERS                    # noqa: E402

#: The column that table printed **before** 2026-09-06, computed in a sibling lab repository
#: on 2026-08-25 and quoted here with nothing in this repository able to reproduce it. It is
#: kept because deleting it would delete the only reproduction claim this pair can make:
#: with the document now printing the golden, everything ``test_fdtd_table_golden.py`` does
#: is a transcription check, and this is the one that says an independently built harness
#: came back to the same answer. Held at
#: what §2 records about the old run rather than at a tolerance chosen here -- see
#: ``LAB_BOUND``.
LAB_MEEP_2026_08_25 = {
    ('te', 'R', 0): (0.204162,), ('te', 'T', 0): (0.352583,),
    ('te', 'T', 1): (0.221396, 0.221396),
    ('tm', 'R', 0): (0.282607,), ('tm', 'T', 0): (0.164495,),
    ('tm', 'T', 1): (0.276318, 0.276231),
}

#: How close the frozen planar run has to come to the closed form. It is the same bound
#: ``gen_goldens_meep_1d.py::calibration_bound`` gates the golden with at this resolution --
#: ``120 / resolution**2``, a *grid* bound rather than a fitted one -- restated here because
#: this module cannot import that one (it would pull in meep). Measured 1.72e-3 at the fine
#: tier and 6.88e-3 at the coarse one, which is order 2.00 and a coefficient of 68.8 at
#: both; a convention wrong instead misses by 1e-1 to 2 at every resolution, because it does
#: not depend on the grid at all.
CALIBRATION_BOUND = 3.0e-3

#: How closely the two legs have to reproduce ``r_p = -r_s`` and ``t_p = +t_s``. Not
#: machine zero, and the reason is physical rather than sloppy: the two legs are separate
#: meep runs driven by different current sheets, and ``Ey`` and ``Hy`` do not live at the
#: same place on a Yee cell -- so the degeneracy is a property of the continuum that each
#: discretisation approaches, not an identity the arithmetic enforces. Measured 9.0e-8 and
#: 1.3e-7, against a grid error of 1.7e-3 on the same numbers: four orders of magnitude
#: tighter than anything else here, which is what makes it worth asserting.
LEG_SIGN_BOUND = 1.0e-6

#: How far the frozen Meep column may sit from the 2026-08-25 lab column it replaced.
#: Measured **5.28e-4**, worst at TM ``R0``; TE's two first orders land within 5e-6. The
#: bound is ~1.9x that, and it is set by what §2 records about the old run rather than by
#: anything chosen here -- that section states its resolution and nothing else about it: not
#: the run length, the pad, the PML depth, the monitor distance or the source.
#:
#: **The justification for the size changed on 2026-09-07 and the bound did not.** It used to
#: rest on §2's claim that Meep's two readings of a transmitted order differ by 2.5e-4 (TE)
#: and 3.4e-4 (TM); re-measured, that ambiguity is <= 7.2e-5 per order (2.1e-4 on the total)
#: and cannot carry a 5.28e-4 gap. What can, and what the bound now rests on, is the
#: *configuration* sensitivity of an unrecorded run: at this resolution, moving ``dpml``
#: 0.6 -> 0.9 or ``pad`` 0.7 -> 1.4 moves this column by **1.6e-3** -- larger than this bound,
#: and neither was recorded of the old run. Two FDTD runs of one scene agreeing to 5e-4 with
#: that much unrecorded between them is the claim; anything tighter would be a claim the
#: record cannot support, and the honest reading of a pass is "no structural disagreement".
LAB_BOUND = 1.0e-3

#: How far the frozen Meep column may sit from a converged RCWA answer. This is Meep's own
#: discretisation and nothing else: measured **4.88e-4** (TE) and **7.42e-4** (TM), both
#: worst at ``R0``, against an RCWA reference converged to ~1e-5.
MEEP_RCWA_BOUND = 1.2e-3

#: How far wavebit's time-domain path may sit from the frozen Meep column, on the same 5 nm
#: cell. Bounded by two grids rather than one, and measured **1.14e-3** (TE, at ``R0``) and
#: **2.52e-4** (TM, at ``T(-1)``) -- which is roughly the two codes' own gaps against RCWA
#: added, as it should be when neither is converged and their errors do not cancel.
FDTD_MEEP_BOUND = 2.0e-3

#: The two first orders are equal by mirror symmetry. TE's are equal to round-off; TM's are
#: not, and that is the grid rather than a broken symmetry -- ``Ex`` is the component normal
#: to the tooth's vertical faces and therefore the discontinuous one. Measured 1.13e-4 (fine)
#: and 1.92e-4 (coarse) on TM, against 8.3e-17 and 0 on TE.
FIRST_ORDER_SPLIT = {'fine': 3.0e-4, 'coarse': 5.0e-4}

#: How closely the specular amplitudes have to agree with the RCWA path's. Looser than the
#: efficiencies because an amplitude carries a phase and an efficiency does not: measured
#: 2.4e-3 / 4.6e-3 (TE ``r``/``t``) and 1.8e-3 / 3.1e-4 (TM).
AMPLITUDE_BOUND = 8.0e-3

#: The grid wavebit's own time-domain run uses here, in cells per micrometre. 200 is the
#: 5 nm cell §2's "this module" column was taken at, and running the comparison on the same
#: cell as the golden is what makes the row a like-for-like reading of that table rather
#: than a new measurement. It divides the raster exactly (140 cells, the tooth edge on a
#: cell boundary at 70), so both codes see the same geometry with no staircase between them.
#:
#: **Everything else about the run is left at the shipped default**, deliberately: the
#: ``solve_fdtd`` call below passes a resolution and nothing more. A test that names its own
#: ``n_periods`` or ``pad_scale`` measures a configuration no user has, and it is exactly a
#: forgotten ``pad_scale`` that let §2's table go stale -- so the settings under test here
#: are the ones a reader of that table gets by typing the call. The value comes from
#: ``_meep_1d_cases.py::FDTD_PUBLISHED``, so the record names it once.
FDTD_RESOLUTION = FDTD_PUBLISHED


from tests.references._legacy_conventions import centred_meep


def _order_grid(result, case):
    """``(1, len(orders_u))`` slices of one solve, on the case's own order window.

    Aligned on the **labels** rather than on positions, which is the whole point of
    ``Result.orders_u`` existing. Both solvers report on a window they choose and the case
    file declares the one the golden holds.

    Read off ``result.incident`` rather than off the ``Result``: a ``tetm=`` solve puts the one
    incidence it was asked for there, and ``Result`` itself publishes only ``R`` and ``T``
    -- the amplitudes are on the sub-result, which is the distinction ``result.py`` draws
    between "the incidence actually asked for" and the pure-s / pure-p pair.
    """
    sub = result.incident
    u = np.asarray(result.orders_u).ravel()
    iu = [int(np.flatnonzero(u == m)[0]) for m in case['orders_u']]
    return {name: np.asarray(getattr(sub, name)).reshape(1, -1)[:, iu]
            for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T')}


def _propagating(case, name):
    """Mask over the case's order window: True where that channel carries power."""
    n = case['n_top'] if name in ('r_s', 'r_p', 'R') else case['n_bot']
    return np.array([[margin(case, int(m), n) > MARGIN for m in case['orders_u']]])


def _solve_rcwa(case, tetm, orders=RCWA_ORDERS):
    return _legacy_wavebit_rcwa(
        backend='numpy', wavelength=case['wavelength'], period=[case['period'][0]],
        truncation_order=dict(x=orders, y=0), top=eps_of(case['n_top']),
        bot=eps_of(case['n_bot']), theta=0.0, phi=0.0, tetm=tetm,
        structure=eps_of(case['index']), thickness=list(case['thickness'])).solve()


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with gen_goldens_meep_1d.py '
                    f'in an environment that has Meep')
    with np.load(GOLDEN_PATH) as data:
        return centred_meep(data, CASES)


@pytest.fixture(scope='module')
def rcwa_solutions():
    return {(name, leg): _solve_rcwa(case, leg)
            for name, case in CASES.items() for leg in LEGS}


@pytest.fixture(scope='module')
def fdtd_solutions():
    """wavebit's own time-domain run on the grating, at ``docs/guide/fdtd/fdtd.md`` §2's settings.

    Module-scoped because two tests read it and it is the expensive half of this file
    (~7 s a leg): one asks whether the printed column is still what the solver returns, the
    other whether that column agrees with Meep's. Those are the same run, and running it
    once is also what keeps them from drifting apart into two different configurations.
    """
    case = CASES['binary_grating']
    return {leg: solve_fdtd(
        structure=eps_of(case['index']), thickness=list(case['thickness']),
        top=eps_of(case['n_top']), bot=eps_of(case['n_bot']), period=[case['period'][0]],
        wavelength=case['wavelength'], tetm=leg, resolution=FDTD_RESOLUTION)
        for leg in LEGS}


# ---------------------------------------------------------------------------------------
# 1. the calibration -- no solver of any kind is involved
# ---------------------------------------------------------------------------------------

def test_the_planar_run_reproduces_the_closed_form(golden):
    """The frozen planar amplitudes against ``airy_rt``, which is written from the
    boundary conditions and calls nothing.

    This is the row that says the bridge can answer. Everything the grating comparisons
    below depend on -- the flux normalisation, both reference planes, the incident
    amplitude read at ``z = 0``, the s/p frame -- is pinned here against a number that is
    known exactly, and a convention wrong in any of them misses by 1e-1 to 2 rather than by
    the grid's own 1e-3. ``README.md``'s note that a reference must be shown able to answer
    before it is believed is the rule this implements.

    Measured, worst over both legs and both amplitudes: **1.72e-3** (fine, 5 nm cells) and
    **6.88e-3** (coarse, 10 nm), which is order **2.00** -- the second order a planar stack
    is entitled to, and the same coefficient (68.8) at both tiers.
    """
    case = CASES['slab_uniform']
    r_exact, t_exact = airy_rt(case['wavelength'], case['n_top'],
                               [float(np.asarray(case['index']).ravel()[0])],
                               list(case['thickness']), case['n_bot'])
    i0 = int(np.flatnonzero(np.asarray(case['orders_u']) == 0)[0])
    want = {'te': dict(r_s=r_exact, t_s=t_exact),
            # `r_p = -r_s` and `t_p = +t_s` at normal incidence: the two waves' tangential
            # fields obey identical equations, and the sign is the frame's `p_hat`
            # reversing between the incident and the reflected leg. No efficiency here or
            # anywhere else in this directory can tell that from `r_p = +r_s`.
            'tm': dict(r_p=-r_exact, t_p=t_exact)}
    worst, where = 0.0, ''
    for leg in LEGS:
        for key, value in want[leg].items():
            gap = abs(complex(golden[f'slab_uniform__{leg}__fine__{key}'][0, i0])
                      - complex(value))
            if gap > worst:
                worst, where = gap, f'{leg}.{key}'
    assert worst < CALIBRATION_BOUND, (
        f'the frozen planar run is {worst:.2e} from the closed form, worst at {where}')


def test_a_planar_stack_radiates_nothing_off_specular(golden):
    """A layer uniform along x has no grating harmonic, so ``m = +-1`` is identically zero.

    Both first orders **propagate in the substrate** here, so this is a channel that could
    carry power and does not -- which is what makes it an instrument rather than a
    tautology. It is also what caught a harness fault: with the planar layer emitted as one
    block per raster column, meep smoothed the block's own face at the cell edge and put
    4.4e-5 into each first order. The generator emits a run of identical columns as a single
    block with an infinite x extent, and the residue is round-off.
    """
    case = CASES['slab_uniform']
    for leg in LEGS:
        for i, m in enumerate(case['orders_u']):
            if m == 0:
                continue
            for key in ('r_s', 'r_p', 't_s', 't_p'):
                got = abs(complex(golden[f'slab_uniform__{leg}__fine__{key}'][0, i]))
                assert got < 1e-12, f'slab {leg} {key}({m}) = {got:.2e}'


def test_the_p_leg_is_the_s_leg_with_its_frame_reversed(golden):
    """``r_p = -r_s`` and ``t_p = +t_s`` on the planar scene, to round-off.

    The two legs are separate meep runs driven by different current sheets -- an electric
    one along ``y_hat`` and a magnetic one along ``y_hat`` -- so this is not an identity the
    harness can arrange. What it pins is that the two runs are the *same physics* read
    through the two halves of one frame, and it is the sharpest statement in this file:
    exact to 1e-7 rather than to the 1.7e-3 that bounds everything else in this file --
    four orders tighter, which is what makes it worth asserting. It is not machine zero
    because ``Ey`` and ``Hy`` sit at different places on a Yee cell, so the degeneracy is a
    property of the continuum each run approaches rather than an identity the arithmetic
    enforces.
    """
    case = CASES['slab_uniform']
    i0 = int(np.flatnonzero(np.asarray(case['orders_u']) == 0)[0])
    r_s = complex(golden['slab_uniform__te__fine__r_s'][0, i0])
    t_s = complex(golden['slab_uniform__te__fine__t_s'][0, i0])
    r_p = complex(golden['slab_uniform__tm__fine__r_p'][0, i0])
    t_p = complex(golden['slab_uniform__tm__fine__t_p'][0, i0])
    assert abs(r_p + r_s) < LEG_SIGN_BOUND, f'r_p = {r_p:.9f} against -r_s = {-r_s:.9f}'
    assert abs(t_p - t_s) < LEG_SIGN_BOUND, f't_p = {t_p:.9f} against t_s = {t_s:.9f}'


# ---------------------------------------------------------------------------------------
# 2. the reproduction claim -- the one thing here that is about Meep rather than bookkeeping
# ---------------------------------------------------------------------------------------

def _pairs(golden, leg, quantity, absm, orders):
    """The one or two frozen values a ``|m|`` row of §2's table stands for, sorted."""
    frozen = golden[f'binary_grating__{leg}__fine__{quantity}'][0]
    return sorted(float(frozen[orders.index(m)])
                  for m in ({0} if absm == 0 else {-absm, absm}))


def test_the_2026_08_25_lab_column_is_reproduced(golden):
    """The column §2 printed *before* the golden existed, against the golden.

    This is the reproduction claim, and it is the only test in either of this pair's files
    that says anything about *Meep* rather than about bookkeeping.
    ``test_fdtd_table_golden.py`` asks whether the document still matches the repository,
    which is now a transcription question with a deterministic answer. This one asks
    whether a harness built here from scratch came back to the answer an independently
    built harness got on another machine on 2026-08-25.

    **What "reproduce" can mean is bounded by what the record says about the old run, and
    the answer is: not the grid.** §2 recorded its resolution -- which this golden's fine
    tier matches -- and nothing else: not the run length, the pad, the PML depth, the
    monitor distance, the source, or which of Meep's two readings of a transmitted order it
    held. It *does* record that the two readings differ by 2.5e-4 (TE) and 3.4e-4 (TM) at
    that resolution. Measured 5.28e-4, worst at TM ``R0``; TE's two first orders land within
    5e-6. So the bound is the stated ambiguity rather than a tolerance chosen here, and the
    honest reading of a pass is "no structural disagreement", not "the same number".
    """
    orders = list(CASES['binary_grating']['orders_u'])
    worst, where = 0.0, ''
    for (leg, quantity, absm), lab in LAB_MEEP_2026_08_25.items():
        for a, b in zip(_pairs(golden, leg, quantity, absm, orders), sorted(lab)):
            if abs(a - b) > worst:
                worst, where = abs(a - b), f'{leg} {quantity}(|m|={absm})'
    assert worst < LAB_BOUND, (
        f'the frozen Meep column is {worst:.2e} from the 2026-08-25 lab column it '
        f'replaced, worst at {where}')


# ---------------------------------------------------------------------------------------
# 3. the two solvers against the frozen numbers
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('leg', list(LEGS), ids=list(LEGS))
def test_the_grating_efficiencies_match_the_rcwa_path(golden, rcwa_solutions, leg):
    """Meep's frozen per-order ``R`` and ``T`` against a converged RCWA answer.

    This is §2's "Meep, res 200" column read against its "RCWA (ref)" one, and what it
    measures is Meep's own discretisation: the RCWA side is converged to ~1e-5 on TE and
    ~1e-5 on TM at ``truncation_order = 80``.
    """
    case = CASES['binary_grating']
    got = _order_grid(rcwa_solutions[('binary_grating', leg)], case)
    worst, where = 0.0, ''
    for key in ('R', 'T'):
        want = golden[f'binary_grating__{leg}__fine__{key}']
        mask = _propagating(case, key)
        for i, m in enumerate(case['orders_u']):
            if not mask[0, i]:
                continue
            gap = abs(float(got[key][0, i]) - float(want[0, i]))
            if gap > worst:
                worst, where = gap, f'{key}({m})'
    assert worst < MEEP_RCWA_BOUND, f'{leg}: {worst:.2e} from Meep, worst at {where}'


@pytest.mark.parametrize('leg', list(LEGS), ids=list(LEGS))
@pytest.mark.slow
def test_the_time_domain_path_matches_meep(golden, fdtd_solutions, leg):
    """``solve_fdtd`` against a time-domain solver that is not ours.

    This is the row the oracle was built for, and it is §2's other two columns read against
    each other on the *same* 5 nm cell. ``tests/fdtd/test_fdtd.py`` compares this path against
    **wavebit's own** RCWA path, which shares every convention with it by construction, so
    a convention wrong in both moves nothing; here the frame, the order index, the ladder's
    period and the reference planes are all read off an FDTD written by somebody else.

    It shares its solve with the column check above, which is not only a saving: the two
    would otherwise be free to drift into different run lengths, and then the number this
    test measures would not be the number the document prints.
    """
    case = CASES['binary_grating']
    grid = _order_grid(fdtd_solutions[leg], case)
    worst, where = 0.0, ''
    for key in ('R', 'T'):
        want = golden[f'binary_grating__{leg}__fine__{key}']
        mask = _propagating(case, key)
        for i, m in enumerate(case['orders_u']):
            if not mask[0, i]:
                continue
            gap = abs(float(grid[key][0, i]) - float(want[0, i]))
            if gap > worst:
                worst, where = gap, f'{key}({m})'
    assert worst < FDTD_MEEP_BOUND, (
        f'{leg}: the time-domain path is {worst:.2e} from Meep, worst at {where}')


def test_pure_and_conical_amplitudes_share_the_meep_frame(golden):
    """Pure TE/TM and conical amplitudes use the same per-order frame and centre.

    Before 2026-09-27 the pure path used a Cartesian primary and differed by a
    sign on negative-kx orders. The frozen Meep data already uses the physical
    per-order frame; its corner reference is converted by the fixture without
    changing any stored golden bytes.
    """
    case = CASES['binary_grating']
    orders = list(case['orders_u'])
    i0, im, ip = orders.index(0), orders.index(-1), orders.index(1)
    for leg, tag in (('te', 's'), ('tm', 'p')):
        pure = _order_grid(_solve_rcwa(case, leg), case)
        # The conical route at the same scene. `theta` cannot be exactly zero on that path,
        # and 1e-9 rad is nine orders below every tolerance in this file.
        conical = _legacy_wavebit_rcwa(
            backend='numpy', wavelength=case['wavelength'], period=[case['period'][0]],
            truncation_order=dict(x=RCWA_ORDERS, y=0), top=eps_of(case['n_top']),
            bot=eps_of(case['n_bot']), theta=1e-9, phi=0.0,
            psi=np.deg2rad(90.0 if leg == 'te' else 0.0), structure=eps_of(case['index']),
            thickness=list(case['thickness'])).solve()
        u = list(np.asarray(conical.orders_u).ravel())
        sub = conical.s_inc if leg == 'te' else conical.p_inc

        for key in (f'r_{tag}', f't_{tag}'):
            want = complex(pure[key][0, i0])
            got = complex(golden[f'binary_grating__{leg}__fine__{key}'][0, i0])
            assert abs(got - want) < AMPLITUDE_BOUND, (
                f'{leg} {key}(0): Meep {got:.6f} against RCWA {want:.6f}')

            # Both RCWA spellings now agree on the specular and negative-kx orders.
            # 1e-7, not machine zero: the conical route runs at `theta = 1e-9` rather than
            # at 0, and that perturbation shows up here at 1.1e-9. It is four orders below
            # every other tolerance in this file.
            con0 = complex(np.asarray(getattr(sub, key)).ravel()[u.index(0)])
            assert abs(con0 - want) < 1e-7, f'{leg} {key}(0): the two paths already differ'
            con_m = complex(np.asarray(getattr(sub, key)).ravel()[u.index(-1)])
            assert abs(con_m - complex(pure[key][0, im])) < 1e-7, (
                f'{leg} {key}(-1): pure and conical paths must use the same s/p frame')

            # Meep independently checks that shared frame on both first orders.
            for i, m in ((im, -1), (ip, 1)):
                got = complex(golden[f'binary_grating__{leg}__fine__{key}'][0, i])
                con = complex(np.asarray(getattr(sub, key)).ravel()[u.index(m)])
                assert abs(got - con) < AMPLITUDE_BOUND, (
                    f'{leg} {key}({m}): Meep {got:.6f} against the conical path {con:.6f}')


# ---------------------------------------------------------------------------------------
# 4. guards on the golden itself, and on the scene
# ---------------------------------------------------------------------------------------

def test_the_golden_states_its_own_error_bar(golden, rcwa_solutions):
    """Every case is frozen at TWO resolutions, and the pair is the error bar.

    A golden without one is a number with no claim attached, and a reader of §2's table
    cannot otherwise tell a converged column from a lucky one. This asserts the claim
    rather than the estimate: with a converged RCWA answer in hand, the finer tier must be
    **nearer** it than the coarser one, and the implied order must be the second order this
    geometry is entitled to.

    **The order is 2 here and was not on the crossed cells** (0.69 to 1.63 there): measured
    1.93 on the planar scene in both legs, 2.17 (TE) and 1.73 (TM) on the grating, with the
    fine tier 4.7e-4 to 7.4e-4 from the reference. That is
    the difference between a raster whose interfaces are all corners and one whose single
    interface per period is a plane the grid can hold exactly -- which is the same reading
    ``docs/guide/fdtd/fdtd.md`` §8g gives of wavebit's own crossed convergence, from the other side.

    An earlier version of the crossed test asserted only ``|fine - coarse| / 3 < 3e-2`` and
    passed while the frozen numbers were 4.6e-2 wrong, because both tiers rounded a layer
    thickness the same way and carried the identical offset. **A convergence study cannot
    see an error that does not depend on the grid**, which is why this compares against a
    reference instead.
    """
    for name, case in CASES.items():
        for leg in LEGS:
            want = _order_grid(rcwa_solutions[(name, leg)], case)
            gaps, spread = {}, 0.0
            for tier in ('fine', 'coarse'):
                worst = 0.0
                for key in ('R', 'T'):
                    mask = _propagating(case, key)
                    a = golden[f'{name}__{leg}__{tier}__{key}'][mask]
                    worst = max(worst, float(np.abs(a - want[key][mask]).max()))
                    if tier == 'fine':
                        b = golden[f'{name}__{leg}__coarse__{key}'][mask]
                        spread = max(spread, float(np.abs(a - b).max()))
                gaps[tier] = worst
            assert gaps['fine'] < gaps['coarse'], (
                f'{name} {leg}: halving the cell moved the golden AWAY from the converged '
                f'answer, {gaps["coarse"]:.2e} -> {gaps["fine"]:.2e}')
            order = np.log(gaps['coarse'] / gaps['fine']) / np.log(2.0)
            assert 1.5 < order < 2.5, f'{name} {leg}: implied order {order:.2f}'
            # The spread between the tiers has to BE the error, to within a factor: at
            # order q it is `(2**q - 1)` times it, which is 3 at q = 2. Measured 2.31 to
            # 3.50 over the four rows. A pair of tiers carrying an identical
            # grid-independent offset drives this toward zero -- the crossed generator's
            # rounded-thickness fault sat at 0.45.
            assert 1.8 <= spread / gaps['fine'] <= 4.2, (
                f'{name} {leg}: the two tiers differ by {spread:.2e} while the fine one is '
                f'{gaps["fine"]:.2e} from the reference; the pair is not measuring it')


def test_the_golden_conserves_energy(golden):
    """Nothing in these scenes absorbs, so ``R + T`` is 1 up to the Yee grid's O(dz**2).

    A guard on the *frozen numbers* rather than on the solver, and a weak one on purpose:
    three of the five faults ``gen_goldens_meep_crossed.py`` records left ``R + T`` at 1
    while the answer was wrong -- geometry wrapping through the PML axis returned
    ``T = 0.412`` against a true 0.684 with ``R + T`` **exactly** 1. So this catches only a
    gross failure, which is still worth refusing before a number is committed, and the
    comparisons above are what actually arbitrate.
    """
    for name in CASES:
        for leg in LEGS:
            total = float(golden[f'{name}__{leg}__fine__R'].sum()
                          + golden[f'{name}__{leg}__fine__T'].sum())
            assert abs(total - 1.0) < 2e-3, f'{name} {leg}: R + T = {total:.6f}'


def test_the_incident_only_run_was_a_single_plane_wave(golden):
    """The harness's own instrument, frozen beside the answer.

    Decompose the incident-only run into orders: a plane wave has one, so whatever lands in
    the others is the harness talking. It is what caught two of the five faults the crossed
    generator records -- an unweighted monitor quadrature (9.9e-3 of the incident into
    orders a plane wave does not have) and an oblique source sheet not carrying its own
    Bloch phase (2.99).

    Cross-polarisation purity is **exactly zero** here rather than merely small, and that is
    a fact about the mount rather than about the harness: at ``phi = 0`` the two legs
    decouple exactly, so an electric sheet along ``y_hat`` puts nothing at all in the p
    channel. The crossed cases, at a conical mount, sit at 2.4e-5 to 3.3e-4.
    """
    for name in CASES:
        for leg in LEGS:
            leak = float(golden[f'{name}__{leg}__fine__leak'])
            purity = float(golden[f'{name}__{leg}__fine__purity'])
            assert leak < 1e-12, f'{name} {leg}: incident-order leak {leak:.2e}'
            assert purity == 0.0, f'{name} {leg}: cross-polarisation {purity:.2e}'


def test_the_two_first_orders_are_equal(golden):
    """A 50 % duty profile is mirror-symmetric, so ``T(+1) == T(-1)`` exactly.

    RCWA returns them equal to round-off and ``tests/fdtd/test_fdtd.py`` asserts it of wavebit's
    own run. It is asserted of the *frozen Meep numbers* here because it is a symmetry the
    answer must have and the harness cannot arrange -- and it is what found the sixth
    harness fault, which ``gen_goldens_meep_1d.py::_layer_blocks`` records: an explicit
    block for the air half of the grating put a face on the cell edge with the same
    material on both sides of it, and meep smoothed it anyway. That split TE's two first
    orders by 1.7e-3 at the frozen resolution -- larger than the 1.3e-3 gap this file
    exists to measure -- with ``R + T`` at 0.9996 throughout.

    The two legs land in different places and the difference is physical: TE's split is
    **exactly zero**, TM's is not, because TM's ``Ex`` is the component normal to the
    tooth's vertical faces and therefore the discontinuous one. TM's residue falls at order
    1.7 (2.8e-3 / 8.8e-4 at resolutions 20 / 40), so it is the grid rather than a broken
    symmetry, and the frozen value is well under the gaps above.
    """
    case = CASES['binary_grating']
    orders = list(case['orders_u'])
    im, ip = orders.index(-1), orders.index(1)
    for tier in ('fine', 'coarse'):
        te = golden[f'binary_grating__te__{tier}__T'][0]
        tm = golden[f'binary_grating__tm__{tier}__T'][0]
        assert abs(float(te[ip]) - float(te[im])) < 1e-12, (
            f'TE {tier}: T(+1) = {te[ip]:.8f} against T(-1) = {te[im]:.8f}')
        assert abs(float(tm[ip]) - float(tm[im])) < FIRST_ORDER_SPLIT[tier], (
            f'TM {tier}: T(+1) = {tm[ip]:.8f} against T(-1) = {tm[im]:.8f}')


def test_the_two_legs_are_not_interchangeable(golden):
    """A swapped TE/TM label is the trap this scene is most exposed to, so it is asserted.

    Nothing in the physics of a lossless grating separates the two labels: both conserve
    energy, both are legal answers to the same geometry, and the totals are equal to the
    digit. What separates them is that they are *different numbers* -- and the gap has to be
    large against the tolerances above, or the mapping in ``_meep_1d_cases.py::LEGS`` is
    doing no work and this file could have them exchanged without noticing.
    """
    frozen = {leg: np.concatenate([golden[f'binary_grating__{leg}__fine__R'].ravel(),
                                   golden[f'binary_grating__{leg}__fine__T'].ravel()])
              for leg in LEGS}
    separation = float(np.abs(frozen['te'] - frozen['tm']).max())
    totals = {leg: float(v.sum()) for leg, v in frozen.items()}
    assert abs(totals['te'] - totals['tm']) < 5e-3, (
        'the two legs no longer agree on R + T, so this scene is not the lossless one '
        'that makes a swapped label invisible to an energy check')
    assert separation > 20.0 * MEEP_RCWA_BOUND, (
        f'the two legs differ by only {separation:.2e} against a {MEEP_RCWA_BOUND:g} '
        f'bound, so a swapped polarisation label would pass every comparison here')


def test_the_scene_is_the_one_tests_test_fdtd_uses():
    """``_meep_1d_cases.py`` says where its numbers came from; this is what checks it.

    An oracle stated on a scene *near* the one under test is a different oracle, and the
    two files cannot share a constant -- ``tests/fdtd/test_fdtd.py`` imports wavebit and the
    case file has to be readable in an environment that has only numpy and meep. So the
    provenance is a comment there and an assertion here, in the one environment that can
    import both. The unit differs by 1000 on purpose (§2's table is in nanometres, this
    oracle is stated in micrometres), which is why these compare *ratios* rather than values.
    """
    sys.path.insert(0, str(HERE.parent))
    import tests.fdtd.test_fdtd as bench

    case = CASES['binary_grating']
    scale = bench.WL / case['wavelength']
    assert scale == pytest.approx(1000.0, rel=1e-12)
    assert bench.PERIOD / case['period'][0] == pytest.approx(scale, rel=1e-12)
    assert bench.THICK / sum(case['thickness']) == pytest.approx(scale, rel=1e-12)
    assert bench.N_TOP == case['n_top'] and bench.N_BOT == case['n_bot']

    # The raster itself, against `_grating()`'s own output rather than against its
    # docstring: two columns, the tooth's index first.
    want = np.asarray(bench._grating()).ravel()
    got = eps_of(np.asarray(case['index'])).ravel()
    assert np.allclose(got, want), f'{got} against tests/fdtd/test_fdtd.py::_grating() {want}'
    assert float(np.asarray(case['index']).ravel()[0]) == bench.N_TOOTH

    # And the planar scene is that grating with exactly one thing removed.
    slab = CASES['slab_uniform']
    for key in ('wavelength', 'period', 'thickness', 'n_top', 'n_bot', 'theta_deg',
                'phi_deg'):
        assert slab[key] == case[key], f'the calibration scene differs in {key}'
    assert np.ptp(np.asarray(slab['index'])) == 0.0


def test_the_closed_form_agrees_with_the_one_the_benchmark_carries():
    """Two independently written planar solvers, pinned to each other.

    ``_meep_1d_cases.py::airy_rt`` is a transfer-matrix chain and
    ``tests/fdtd/test_fdtd.py::_stack_rt`` is the benchmark's own; the calibration above is only
    as good as the first, and this is the one place in the repository that can compare them
    because it is the one environment that imports both. They agree to round-off, which
    also settles the ``t_p`` convention -- the published p transmission is the **E** ratio
    rather than the magnetic one, and the two differ by ``n_top / n_bot``.
    """
    sys.path.insert(0, str(HERE.parent))
    import tests.fdtd.test_fdtd as bench

    case = CASES['slab_uniform']
    n_tooth = float(np.asarray(case['index']).ravel()[0])
    for tetm, sign in (('te', 1.0), ('tm', -1.0)):
        r_ref, t_ref = bench._stack_rt(bench.WL, bench.N_TOP, [n_tooth], [bench.THICK],
                                       bench.N_BOT, tetm)[:2]
        r, t = airy_rt(case['wavelength'], case['n_top'], [n_tooth],
                       list(case['thickness']), case['n_bot'])
        assert abs(r_ref - sign * r) < 1e-13, f'{tetm}: r {r_ref} against {sign * r}'
        assert abs(t_ref - t) < 1e-13, f'{tetm}: t {t_ref} against {t}'


def test_the_golden_holds_the_orders_the_case_file_declares(golden):
    """The frozen numbers and the scene they describe have to stay in step.

    ``_meep_1d_cases.py`` is read by both halves, so editing a period, an angle or an order
    window there silently re-labels a golden nobody regenerated -- the file would still
    load, the shapes would still match, and every comparison would be against the wrong
    scene.
    """
    for name, case in CASES.items():
        assert list(golden[f'{name}__orders_u']) == list(case['orders_u'])
        assert list(golden[f'{name}__orders_v']) == [0]
        assert int(golden[f'{name}__fine__resolution']) == case['meep_resolution']
        for leg in LEGS:
            assert golden[f'{name}__{leg}__fine__r_s'].shape == (1, len(case['orders_u']))


def test_the_leg_signs_are_the_ones_the_generator_derived():
    """The generator and ``src/wavebit/fdtd/_periodic/legacy.py`` reach ``(-1, +1, -1)`` independently.

    Not a convention either of them chose: ``p_hat = k_hat x s_hat`` plus ``E . k_hat = 0``
    gives ``E . p_hat = -+ (n / kz)(c Ex + s Ey)`` with the upper sign downgoing, which is
    incident ``-1``, reflected ``+1``, transmitted ``-1``. This pins the two spellings
    together so that a change to either is a failing test rather than a silent disagreement
    absorbed by a tolerance.
    """
    import inspect
    from wavebit.fdtd._periodic import legacy as fdtd
    for name in ('_extract_crossed', '_extract_conical'):
        sig = inspect.signature(getattr(fdtd, name)).parameters['sigma'].default
        assert tuple(float(s) for s in sig) == SIGMA, name


def test_the_order_window_is_the_one_the_physics_gives():
    """Three orders above, three below, and only one of them radiates into air.

    Written as an assertion rather than left to the case file's tuple because it is what
    makes ``test_a_planar_stack_radiates_nothing_off_specular`` an instrument: the first
    orders have to be channels that *could* carry power in the substrate. In air they
    cannot, which is the other half of the same statement and is why ``R[+-1]`` is
    identically zero throughout §2's table.
    """
    for name, case in CASES.items():
        for m in case['orders_u']:
            above = margin(case, int(m), case['n_top'])
            below = margin(case, int(m), case['n_bot'])
            if m == 0:
                assert above > MARGIN and below > MARGIN
            else:
                assert above < -MARGIN, f'{name}: order {m} radiates into air'
                assert below > MARGIN, f'{name}: order {m} is evanescent in the substrate'
        assert abs(in_plane_k(case, 1)) > 1.0
