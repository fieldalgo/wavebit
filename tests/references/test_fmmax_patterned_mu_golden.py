"""A **patterned permeability** against fmmax, from a golden frozen before ``mu`` had a builder.

The magnetic work landed with ``mu`` reaching the layer operator as convolution matrices the
caller supplies; nothing built those from a patterned raster. These rows were generated in that
state, so nothing in ``pn_mu_conv_matrices`` / ``to_conv_mat_raster_pn_mu`` could be fitted to
them.

**The claim under test is that no new factorization was needed.** ``pn_conv_matrices`` serves
``mu`` verbatim, because the Popov-Neviere rearrangement is driven entirely by which quantities
are continuous at an interface — tangential ``E`` and normal ``D`` for the permittivity,
tangential ``H`` and normal ``B`` for the permeability — and the second line is the first with
every symbol replaced. A separate derivation would be the same derivation. If that were wrong,
these rows are where it would show.

Judged by **convergence**, like the patterned ``eps`` golden: the Popov-Neviere form, fmmax's
``FFT`` (the plain direct rule) and fmmax's ``JONES_DIRECT`` (a vector formulation of another
construction) meet only in the limit, so the golden holds both fmmax answers and the tolerance
is the spread between them.

fmmax was gated first on four questions — a patterned ``mu`` moves its eigenvalues (8.8e-2), the
*pattern* matters rather than the mean (7.3e-2), a patterned ``mu_xy`` is used (7.2e-2), and
exchanging ``eps`` with ``mu`` trades the s and p channels (3e-14). The last is the one that
says its permeability is not merely present but dual-correct.
"""
import pathlib
import sys

import numpy as np
import pytest

import wavebit
from wavebit._core.factorization import to_conv_mat_raster_pn, to_conv_mat_raster_pn_mu

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _fmmax_patterned_mu_cases import (                                        # noqa: E402
    CASES, FTO, GRID, PERIOD, THETA, THICK, WL, tensors)

GOLDEN_PATH = HERE / 'golden_fmmax_patterned_mu.npz'

# Cases where the solver sits nearer the vector formulation. `different_shapes` is excluded,
# and only that one: its reflectances are ~1.5e-5, where fmmax's own two answers (2.2e-5 and
# 1.2e-5) are a factor of two apart, so "which is nearer" stops being a statement about
# formulations. It sits between them there, stably across fto 8..14.
#
# An earlier version of this file asserted **bracketing** instead — componentwise between the
# two fmmax answers — on the reasoning that it is the sharper claim. Measured, it is false:
# three of six cases sit just outside, near the vector answer but past it (the gyromagnetic one
# by 9e-6 against a 2.4e-5 bracket). Three formulations converging to one limit approach it
# from whatever side their own truncation puts them on; nothing makes them enclose each other.
# Recorded because the claim is plausible enough to be re-added.
VECTOR_SIDE = [n for n in CASES() if n != 'different_shapes']


def _as33(diag, off):
    t = np.zeros((GRID, GRID, 3, 3), dtype=complex)
    t[..., 0, 0] = t[..., 1, 1] = t[..., 2, 2] = diag
    t[..., 0, 1], t[..., 1, 0] = off, -off
    return t


def solve(case, fto=FTO, algo='ETM'):
    """``[R_s, T_s, R_p, T_p]`` for one layer with both tensors patterned."""
    eps, eps_xy, mu, mu_xy = tensors(case)

    epx, epy, epz_i, epxy, epyx = to_conv_mat_raster_pn(_as33(eps, eps_xy)[None], fto, fto)
    mu_slots = to_conv_mat_raster_pn_mu(_as33(mu, mu_xy)[None], fto, fto)

    mee = wavebit.rcwa(backend=0, n_top=1.0, n_bot=1.0, theta=np.deg2rad(THETA), phi=0.0,
                       psi=0.0, fto=(fto, fto), period=(PERIOD, PERIOD), wavelength=WL,
                       thickness=[THICK], ucell=np.expand_dims(np.sqrt(eps), axis=0),
                       connecting_algo=algo)
    res = mee.solve_for_conv(WL, epx, epy, epz_i, epxy, epyx, mu=mu_slots)
    te, tm = res.res_te_inc, res.res_tm_inc
    return np.array([float(te.R.sum()), float(te.T.sum()),
                     float(tm.R.sum()), float(tm.T.sum())])


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with '
                    'gen_goldens_fmmax_patterned_mu.py')
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(CASES()), ids=list(CASES()))
def test_lands_inside_the_spread_between_fmmax_own_formulations(golden, name, algo):
    fft, jones = golden[f'{name}__fft'], golden[f'{name}__jones']
    spread = np.abs(fft - jones).max()
    got = solve(CASES()[name], algo=algo)
    gap = min(np.abs(got - fft).max(), np.abs(got - jones).max())
    assert gap < spread, (f'{name} [{algo}]: {gap:.2e} from the nearer fmmax formulation, past '
                          f'the {spread:.2e} the two of them differ by\n'
                          f'  got   {np.array2string(got, precision=9)}\n'
                          f'  fft   {np.array2string(fft, precision=9)}\n'
                          f'  jones {np.array2string(jones, precision=9)}')


@pytest.mark.parametrize('name', VECTOR_SIDE, ids=VECTOR_SIDE)
def test_it_is_the_vector_formulation_that_it_agrees_with(golden, name):
    """The factorization must be visible in the answer, not only present in the code.

    A build that fed ``mu`` through unfactorized — the plain direct rule on every component —
    would still conserve energy, still be inside the band, and would sit on the ``FFT`` side.
    """
    fft, jones = golden[f'{name}__fft'], golden[f'{name}__jones']
    got = solve(CASES()[name])
    assert np.abs(got - jones).max() < np.abs(got - fft).max()


@pytest.mark.parametrize('name', list(CASES()), ids=list(CASES()))
def test_energy_closes(name):
    """Every tensor is Hermitian at every pixel, so both media are lossless however patterned.

    A theorem, so it survives the oracle being wrong — and, as the magnetic work established,
    it is *not* the check that catches a misplaced permeability.
    """
    got = solve(CASES()[name])
    for pol, (r, t) in (('s', got[:2]), ('p', got[2:])):
        assert abs(r + t - 1.0) < 1e-6, f'{name} {pol}: R+T = {r + t:.12f}'


def test_a_patterned_mu_is_not_solved_as_its_average(golden):
    """The pattern in ``mu`` has to reach the answer, not just its mean.

    ``uniform_eps_patterned_mu`` is the case for it: ``eps`` is constant, so if the permeability
    were averaged the layer would be uniform and the whole answer would collapse to a slab.
    """
    case = CASES()['uniform_eps_patterned_mu']
    eps, eps_xy, mu, mu_xy = tensors(case)
    averaged = dict(eps=eps, mu=np.full_like(mu, mu.mean()))
    assert np.abs(solve(case) - solve(averaged)).max() > 1e-4


def test_the_two_tensors_get_their_own_normal_fields(golden):
    """``different_shapes`` patterns ``eps`` and ``mu`` on different rectangles.

    The rearrangement is written around one interface normal, so each tensor needs the normal of
    its own boundary. Building ``mu``'s slots against the *permittivity's* field is the natural
    shortcut and is wrong; it passes every case where the two shapes coincide, which is why this
    one exists.
    """
    from wavebit._core.factorization import normal_vector_field

    case = CASES()['different_shapes']
    eps, eps_xy, mu, mu_xy = tensors(case)
    correct = solve(case)

    fto = FTO
    epx, epy, epz_i, epxy, epyx = to_conv_mat_raster_pn(_as33(eps, eps_xy)[None], fto, fto)
    wrong_field = [normal_vector_field(eps)]        # eps's boundary, applied to mu
    wrong_mu = to_conv_mat_raster_pn_mu(_as33(mu, mu_xy)[None], fto, fto,
                                        normal_field=wrong_field)
    mee = wavebit.rcwa(backend=0, n_top=1.0, n_bot=1.0, theta=np.deg2rad(THETA), phi=0.0,
                       psi=0.0, fto=(fto, fto), period=(PERIOD, PERIOD), wavelength=WL,
                       thickness=[THICK], ucell=np.expand_dims(np.sqrt(eps), axis=0))
    res = mee.solve_for_conv(WL, epx, epy, epz_i, epxy, epyx, mu=wrong_mu)
    wrong = np.array([float(res.res_te_inc.R.sum()), float(res.res_te_inc.T.sum()),
                      float(res.res_tm_inc.R.sum()), float(res.res_tm_inc.T.sum())])

    assert np.abs(correct - wrong).max() > 1e-7, (
        'sharing one normal field between two differently-shaped tensors changed nothing; '
        'this case is not exercising what it was written for')


def test_the_golden_is_not_two_copies_of_one_answer(golden):
    """Guards the oracle: the tolerance is the spread, so a collapsed spread is a silent break."""
    for name in CASES():
        spread = np.abs(golden[f'{name}__fft'] - golden[f'{name}__jones']).max()
        assert spread > 1e-9, f'{name}: the two frozen formulations differ by {spread:.2e}'
