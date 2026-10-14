"""A finite gain film between passive half-spaces, against the closed-form Airy film.

``Im(eps) < 0`` is gain under this package's ``exp(-i omega t)`` convention: the small-signal
response of a pumped medium, so ``R + T > 1`` is the pump's power, not a broken energy balance.
In a **finite** film it needs no root choice at all. The film's field is ``a exp(i k0 q s) +
b exp(-i k0 q s)``, and ``q -> -q`` with ``a <-> b`` leaves it unchanged, so the Airy formula
gives one answer whatever branch the solver labels its modes with. A gain **half-space** is a
different problem -- removing one of the two waves there does fix a branch, and which one is
causal is not a property of ``eps`` at one frequency (S. Nistad and J. Skaar, Phys. Rev. E 78,
036603 (2008)); that case is ``tests/rcwa/test_branch_selection.py``'s.

So this file pins the case that is well posed: air / film / air, film ``eps = 2.25 + i*eta``
with ``eta`` in ``+0.03, +0.003, 0, -0.003, -0.03``, at ``beta = 0`` and ``0.5`` in air, for
s and p, on both formulations, ETM and SMM, lit from either side. The solver answered all of
them to 2e-15 when this was written (2026-10-15), with no warning: the result guard that
used to flag a gain film as "not physical" was removed with ``check_physical``.

The scene and its numbers come from a 2026-09-23 study that ran the same film through
Wavebit, fmmax 1.7.1 and S4; the three agreed with the Airy film to 1.5e-15, 8.9e-16 and
4.3e-15 at the normal-incidence s case pinned below.
"""
import warnings

import numpy as np
import pytest

from wavebit.compat import legacy as _legacy_wavebit
from wavebit import refractive_index_from_eps
from tests.rcwa.test_lossy_media import analytic_r, analytic_t

WL = 1.0
D = 0.2
EPS_FILM = 2.25
ETAS = (0.03, 0.003, 0.0, -0.003, -0.03)        # loss, lossless, gain
BETAS = (0.0, 0.5)                              # kx / k0 in the air: 0 and 30 degrees


def _solve(eps, beta, tetm, formulation, algo='ETM', side='top', answers='amplitude'):
    """One air / film / air solve, s or p at ``phi = 0``, with the formulation forced."""
    solver = _legacy_wavebit.rcwa(
        backend='numpy', top=1.0, bot=1.0, theta=np.arcsin(beta), phi=0.0,
        polarization=(1, 0) if tetm == 'te' else (0, 1), truncation_order=[0, 0],
        period=[2 * WL, 2 * WL], wavelength=WL, thickness=[D],
        structure=np.full((1, 2, 2), eps), layer_algorithm=algo, incidence_side=side,
        answers=answers)
    solver.formulation = formulation
    result = solver.solve()
    assert solver.formulation_assigned == formulation
    return solver, result


def _airy(eps, beta, tetm):
    """``(r, t)`` of the film from the closed form; the air is symmetric, so either side."""
    ns = [1.0, complex(refractive_index_from_eps(eps)), 1.0]
    theta = np.arcsin(beta)
    return analytic_r(ns, [D], WL, theta, tetm), analytic_t(ns, [D], WL, theta, tetm)


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('formulation', ['second_order', 'first_order'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_gain_film_is_the_airy_film(tetm, formulation, algo, side):
    """Amplitudes, efficiencies and the film's absorption, gain and loss alike, to 1e-12."""
    for eta in ETAS:
        for beta in BETAS:
            eps = EPS_FILM + 1j * eta
            _, res = _solve(eps, beta, tetm, formulation, algo, side)
            r, t = _airy(eps, beta, tetm)
            inc = res.incident
            got_r = complex(np.asarray(inc.r_s if tetm == 'te' else inc.r_p).ravel()[0])
            got_t = complex(np.asarray(inc.t_s if tetm == 'te' else inc.t_p).ravel()[0])
            where = f'eta={eta} beta={beta}'
            assert abs(got_r - r) < 1e-12, (where, got_r, r)
            assert abs(got_t - t) < 1e-12, (where, got_t, t)
            R, T = float(np.sum(np.asarray(inc.R))), float(np.sum(np.asarray(inc.T)))
            assert R == pytest.approx(abs(r) ** 2, abs=1e-12), where
            assert T == pytest.approx(abs(t) ** 2, abs=1e-12), where   # air both sides
            absorption = float(np.real(res.absorption))
            assert absorption == pytest.approx(1.0 - abs(r) ** 2 - abs(t) ** 2, abs=1e-12), where
            # The sign is the physics: gain supplies power, loss takes it, lossless does neither.
            if eta < 0:
                assert absorption < 0 and R + T > 1, where
            elif eta > 0:
                assert absorption > 0 and R + T < 1, where
            else:
                assert abs(absorption) < 1e-12, where


def test_the_normal_incidence_gain_film_keeps_its_three_solver_numbers():
    """The case three codes agreed on, as literal numbers: s at normal incidence, eta = -0.03."""
    _, res = _solve(EPS_FILM - 0.03j, 0.0, 'te', 'second_order')
    assert float(np.sum(np.asarray(res.R))) == pytest.approx(0.138837529106086, abs=1e-13)
    assert float(np.sum(np.asarray(res.T))) == pytest.approx(0.883570549029302, abs=1e-13)
    assert float(np.real(res.absorption)) == pytest.approx(-0.022408078135388, abs=1e-13)


@pytest.mark.parametrize('formulation', ['second_order', 'first_order'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_gain_film_solves_without_a_warning(tetm, formulation):
    """Nothing about a finite gain film is suspect, so nothing may say it is."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _solve(EPS_FILM - 0.03j, 0.5, tetm, formulation)


@pytest.mark.parametrize('formulation', ['second_order', 'first_order'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_negative_absorption_is_the_work_the_gain_does_on_the_field(tetm, formulation):
    """An independent reading of ``absorption < 0``: the medium's own work on the field.

    The time-averaged power a non-magnetic medium takes from the field is
    ``(omega/2) eps0 Im(eps) |E|^2`` per volume. Integrated over the film and divided by the
    incident flux ``(1/2) n0 cos(theta) |E0|^2 / Z0`` it is ``k0 Im(eps) int |E|^2 dz /
    (n0 cos(theta))`` for a unit incident field -- negative for gain. That has to equal the
    boundary bookkeeping ``power_entering - sum(T)``, which never looks inside the film.
    Measured 2.6e-15 apart with Simpson's rule on 801 heights.
    """
    eps, beta = EPS_FILM - 0.03j, 0.5
    solver, res = _solve(eps, beta, tetm, formulation, answers='field')
    n = 801
    cell = np.asarray(solver.reconstruct_field(sample_x=1, sample_y=1, sample_z=n))
    cell = cell[0] if cell.ndim == 5 else cell
    e2 = np.sum(np.abs(cell[:, 0, 0, :3]) ** 2, axis=-1)
    h = D / (n - 1)
    integral = h / 3 * (e2[0] + e2[-1] + 4 * e2[1:-1:2].sum() + 2 * e2[2:-1:2].sum())
    work = 2 * np.pi / WL * eps.imag * integral / np.cos(np.arcsin(beta))
    assert work < 0
    assert work == pytest.approx(float(np.real(res.absorption)), abs=1e-12)
