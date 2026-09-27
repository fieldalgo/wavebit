"""Magnetic **half-spaces** against fmmax, from a golden frozen before the boundary could hold one.

``docs/anisotropic-materials.md`` §5 said to change the boundary conventions "with a golden in
hand", because ``conventions.py`` is the module that exists after a sign seam there once turned
absorbing media into gain media. This is that golden. It was generated while ``boundary_kz``
took one material argument and the p-channel admittance was spelled ``/ eps_bot`` — there was
nowhere to put a ``mu``, so nothing in it could be fitted to.

**This file carries a closed form, and that is the point.** The magnetic *layer* golden could
lean on its ``__mu1`` half: every case generated twice, the non-magnetic copy held to the solver
as a convention gate while the magnetic one waited. Here there is no such twin — the ambient
*is* the thing under test, so a ``mu = 1`` companion tests nothing about it. What replaces it is
a bare interface, which has an exact answer::

    s:  Y = kz/mu        p:  Y = kz/eps        kz = sqrt(eps*mu - sin^2(theta))
    R = |(Y_inc - Y_sub) / (Y_inc + Y_sub)|^2

``_fmmax_mu_ambient_cases.fresnel`` states it, the generator enforced it on fmmax before
freezing, and :func:`test_a_bare_interface_matches_the_closed_form` holds the solver to it
directly. That check needs no oracle at all and would survive fmmax being wrong.

The case worth naming is ``impedance_only_mismatch_30deg``: ``eps = 0.5``, ``mu = 2.0``, so
``n = 1`` and the substrate is **index-matched to vacuum**. A solver that carries the refractive
index and drops the permeability sees no interface and returns ``R = 0``; the answer is 0.111.
Nothing else in the suite distinguishes "the boundary reads mu" from "the boundary happens to
agree".

**The case module's ambients now need no conversion at all, and that is the payoff of a
decision it made before the answer was known.** It states each half-space as a physical
``(eps, mu)`` pair on the explicit grounds that "the solver's spelling of a magnetic half-space
is an API decision that came after this file, and a golden that encoded it would have to be
regenerated if it changed". The decision came on 2026-08-20 and it is
:class:`wavebit.Material`, which is that same pair — so ``top=Material(eps, mu)`` is now a
transcription rather than a conversion, and the ``sqrt(eps*mu)`` this file used to compute has
moved inside the solver, where the branch is argued once.
"""
import pathlib
import sys

import numpy as np
import pytest

import wavebit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _fmmax_mu_ambient_cases import (                                           # noqa: E402
    CASES, PERIOD, THICK, WL, ZERO_THICKNESS_CASES, fresnel)

GOLDEN_PATH = HERE / 'golden_fmmax_mu_ambient.npz'
I1 = np.eye(1, dtype=np.complex128)


def _slots(tensor):
    t = np.asarray(tensor, dtype=complex) if np.ndim(tensor) else np.eye(3) * tensor
    return [np.stack([t[0, 0] * I1]), np.stack([t[1, 1] * I1]),
            np.stack([np.linalg.inv(t[2, 2] * I1)]),
            np.stack([-t[0, 1] * I1]), np.stack([-t[1, 0] * I1])]


def _is_unit(t):
    return np.abs(np.asarray(t, dtype=complex) - np.eye(3)).max() <= 1e-12


def solve(top, bot, layer_eps, layer_mu, theta_deg, thick=THICK, algo='ETM'):
    """``[R_s, T_s, R_p, T_p]``. ``top`` / ``bot`` are physical ``(eps, mu)`` pairs.

    Each becomes a :class:`wavebit.Material` verbatim. The solver derives the refractive index
    ``sqrt(eps*mu)`` from it once, at the setter, and reads *that* everywhere ``kz``, Snell and
    the order grid are written — which is why none of those changed when permeability arrived,
    and why this file no longer takes that root itself.

    The two half-space quantities are then passed to :meth:`solve_conv` **explicitly**, and
    they have to be: this is the low-level door, below the point where ``solve`` reads them
    off ``self.top`` / ``self.bot``. They ride together rather than singly — ``mu`` without its
    ``eps`` would leave the solve recovering ``eps = n**2/mu`` from an index built out of the
    pair, which agrees to round-off but says the same thing twice.
    """
    (eps_t, mu_t), (eps_b, mu_b) = top, bot
    mee = wavebit.rcwa(backend=0,
                       top=wavebit.Material(eps_t, mu_t),
                       bot=wavebit.Material(eps_b, mu_b),
                       theta=np.deg2rad(theta_deg), phi=0.0, psi=0.0, truncation_order=(0, 0),
                       period=(PERIOD, PERIOD), wavelength=WL, thickness=[thick],
                       layer_algorithm=algo)
    res = mee.solve_conv(*_slots(layer_eps),
                             mu=None if _is_unit(layer_mu) else tuple(_slots(layer_mu)),
                             mu_top=mu_t, mu_bot=mu_b, eps_top=eps_t, eps_bot=eps_b)
    te, tm = res.s_inc, res.p_inc
    return np.array([float(te.R.sum()), float(te.T.sum()),
                     float(tm.R.sum()), float(tm.T.sum())])


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with '
                    'gen_goldens_fmmax_mu_ambient.py')
    return np.load(GOLDEN_PATH)


# ------------------------------------------------------------------ the oracle-free half

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(ZERO_THICKNESS_CASES), ids=list(ZERO_THICKNESS_CASES))
def test_a_bare_interface_matches_the_closed_form(name, algo):
    """One interface, vacuum to a magnetic half-space. No solver on the other side.

    This is what a magnetic ambient's convention gate looks like when there is no ``mu = 1``
    twin to lean on. It pins both admittances separately — ``kz/mu`` on s and ``kz/eps`` on p —
    because a formula that got them backwards would still conserve energy and still be
    symmetric under reciprocity.
    """
    (eps_s, mu_s), theta = ZERO_THICKNESS_CASES[name]
    got = solve((1.0, 1.0), (eps_s, mu_s), np.eye(3), np.eye(3), theta, thick=0.0, algo=algo)
    want = fresnel(eps_s, mu_s, theta)
    assert np.abs(got - want).max() < 1e-12, (
        f'{name} [{algo}]\n  got  {np.array2string(got, precision=9)}'
        f'\n  want {np.array2string(want, precision=9)}')


def test_an_index_matched_magnetic_substrate_still_reflects():
    """``eps = 0.5, mu = 2.0`` has ``n = 1``: index-matched to vacuum, impedance-mismatched.

    Stated on its own because it is the single case that cannot be passed by accident. Every
    other row here has an index step, so a boundary that reads only ``n`` gets *something*
    right; this one it must report as no interface at all.
    """
    got = solve((1.0, 1.0), (0.5, 2.0), np.eye(3), np.eye(3), 30.0, thick=0.0)
    assert got[0] > 0.1, f'an impedance-only mismatch was solved as vacuum: R_s = {got[0]:.3e}'
    assert np.abs(got - fresnel(0.5, 2.0, 30.0)).max() < 1e-12


# ------------------------------------------------------------------------- against fmmax

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_stacks_match_fmmax(golden, name, algo):
    """Both connecting algorithms. ``smm_2d`` builds its own half-space modes, so one of them
    could have been left non-magnetic with nothing else failing."""
    top, bot, layer_eps, layer_mu, theta = CASES[name]
    gap = np.abs(solve(top, bot, layer_eps, layer_mu, theta, algo=algo) - golden[name]).max()
    assert gap < 1e-12, f'{name} [{algo}]: {gap:.2e} from fmmax'


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_energy_closes(name):
    """Every medium in every case is real and lossless, so this is a theorem.

    It is *not* the check that would catch a wrong admittance, though — a boundary that used
    ``kz/eps`` for both polarizations conserves energy perfectly. That is what the closed-form
    test above is for, and why this one is not relied on alone.
    """
    top, bot, layer_eps, layer_mu, theta = CASES[name]
    r_s, t_s, r_p, t_p = solve(top, bot, layer_eps, layer_mu, theta)
    assert r_s + t_s == pytest.approx(1.0, abs=1e-10)
    assert r_p + t_p == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_ambient_permeability_is_not_silently_dropped(name):
    """Solving the same scene with ``mu_top = mu_bot = 1`` must give a different answer.

    Keeping ``n`` fixed while dropping ``mu`` is exactly what a boundary that never learned
    about permeability would compute, and it returns a finite, energy-conserving, plausible
    number. Only a comparison against what the answer must *not* be catches it.
    """
    top, bot, layer_eps, layer_mu, theta = CASES[name]
    with_mu = solve(top, bot, layer_eps, layer_mu, theta)
    without = solve((top[0] * top[1], 1.0), (bot[0] * bot[1], 1.0), layer_eps, layer_mu, theta)
    assert np.abs(with_mu - without).max() > 1e-6, (
        f'{name}: dropping the ambient mu changed the answer by only '
        f'{np.abs(with_mu - without).max():.2e}')


# ------------------------------------- the two paths a mutation survey found uncovered

def _solve_psi(top, bot, layer_eps, theta_deg, psi_deg, algo='ETM', side='top'):
    (eps_t, mu_t), (eps_b, mu_b) = top, bot
    mee = wavebit.rcwa(backend=0,
                       top=wavebit.Material(eps_t, mu_t),
                       bot=wavebit.Material(eps_b, mu_b),
                       theta=np.deg2rad(theta_deg), phi=0.0, psi=np.deg2rad(psi_deg),
                       truncation_order=(0, 0), period=(PERIOD, PERIOD), wavelength=WL, thickness=[THICK],
                       incidence_side=side, layer_algorithm=algo)
    res = mee.solve_conv(*_slots(layer_eps), mu_top=mu_t, mu_bot=mu_b,
                             eps_top=eps_t, eps_bot=eps_b)
    return res


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('psi_deg', [0.0, 30.0, 90.0])
def test_the_psi_incidence_path_carries_the_ambient_permeability(psi_deg, algo):
    """``incident`` — the psi-specified incidence — as opposed to the pure-s/pure-p variants.

    Added because a mutation survey said it was uncovered, and the way it said so is worth
    recording: two mutants of the incident column *survived*, and measuring them showed they
    changed nothing at all. They were not redundant code — they were in the copy of that column
    the tests never reached. Every magnetic assertion up to that point read ``s_inc`` /
    ``p_inc``, which are built from a different block.

    The gap was not small. With the incident s-flux left non-magnetic, this path returns
    ``R + T = 1.575``.
    """
    top, bot = (1.9, 1.3), (2.6, 1.7)
    res = _solve_psi(top, bot, np.eye(3) * 2.2, 28.0, psi_deg, algo=algo)
    r, t = float(res.incident.R.sum()), float(res.incident.T.sum())
    assert r + t == pytest.approx(1.0, abs=1e-10), f'psi={psi_deg} [{algo}]: R+T = {r + t}'

    # psi = 90 is pure s and psi = 0 pure p, by the definition of (a_s, a_p) = (sin, cos), so
    # each has to reproduce the corresponding variant exactly. That is what ties the two
    # incident columns together rather than leaving them free to drift apart.
    twin = {90.0: res.s_inc, 0.0: res.p_inc}.get(psi_deg)
    if twin is not None:
        assert abs(r - float(twin.R.sum())) < 1e-12
        assert abs(t - float(twin.T.sum())) < 1e-12


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('theta_top', [0.0, 33.0, 50.0])
def test_backside_incidence_is_reciprocal_with_magnetic_ambients(theta_top, algo):
    """Transmittance is direction-independent for a lossless reciprocal stack.

    This was the only check that reached the ``mu_top`` / ``mu_bot`` swap in the backside
    mirror, and a mutation survey found nothing else did — deleting the swap survived
    everything else in this file. That was while the backside, the propagation kernels' port,
    served public bottom incidence. Since 2026-09-21 it serves public *top* incidence, so every
    top-lit case here reaches the swap as well: with the 2D ETM swap deleted, 16 tests in this
    file fail — the three ETM reciprocity cases below and 13 top-lit ETM comparisons. This check
    solves both sides, so it reaches the swap either way.

    **The angle has to be converted.** ``theta`` is stated against the *incidence medium's*
    normal, so top-at-33-degrees and bottom-at-33-degrees are different in-plane wavevectors
    and therefore different problems; comparing them directly puts T out by 9e-3 and looks
    exactly like a broken mirror. What reciprocity equates is one ``k_par`` seen from both
    sides: ``n_top sin(th_top) = n_bot sin(th_bot)``.

    Only ``T`` is the theorem. ``R`` follows here because the stack is lossless, not because
    reflection is direction-independent in general.
    """
    top, bot = (1.9, 1.3), (2.6, 1.7)
    # Snell is written in indices, and a magnetic medium's is `sqrt(eps*mu)` -- so the product
    # goes into `index_from_eps` rather than the permittivity alone. Dropping the `mu` here
    # would not raise; it would just state the wrong angle and read as a broken mirror.
    n_t = float(wavebit.index_from_eps(top[0] * top[1]).real)
    n_b = float(wavebit.index_from_eps(bot[0] * bot[1]).real)
    theta_bot = np.rad2deg(np.arcsin(n_t * np.sin(np.deg2rad(theta_top)) / n_b))

    down = solve(top, bot, np.eye(3) * 2.2, np.eye(3), theta_top, algo=algo)
    mee_up = _solve_psi(top, bot, np.eye(3) * 2.2, theta_bot, 0.0, algo=algo, side='bottom')
    up_s = float(mee_up.s_inc.T.sum())
    up_p = float(mee_up.p_inc.T.sum())

    assert abs(down[1] - up_s) < 1e-10, f'T_s {down[1]} vs {up_s}'
    assert abs(down[3] - up_p) < 1e-10, f'T_p {down[3]} vs {up_p}'


def test_the_reciprocity_check_can_tell_the_two_ambients_apart():
    """Otherwise the test above would pass on a solver that used one permeability for both."""
    a = solve((1.9, 1.3), (2.6, 1.7), np.eye(3) * 2.2, np.eye(3), 33.0)
    b = solve((1.9, 1.7), (2.6, 1.3), np.eye(3) * 2.2, np.eye(3), 33.0)
    assert np.abs(a - b).max() > 1e-3


# --------------------------------------------------------------- the boundary, off the 2D path

def test_a_magnetic_ambient_on_a_one_dimensional_structure_now_solves():
    """The 1D and conical boundaries carry the permeability too, since 2026-09-01.

    This was the refusal until then. What replaces it is the check the refusal stood in for:
    that ``mu_bot`` reaches the substrate admittance rather than being accepted and dropped.
    Asserted against the **closed form** for the bare interface, the same one the 2D rows above
    are judged by -- ``R = |(Y_inc - Y_sub)/(Y_inc + Y_sub)|**2`` with ``Y = kz/mu`` on s.

    A magnetic substrate under a *vacuum* stack is exactly a bare interface, so no grating is
    needed to state it; the index stays 2.1 whichever ``mu_bot`` is named, which is what makes
    this reach the admittance and nothing else.
    """
    # 4.41 is the index-2.1 substrate this case always had, said as a permittivity.
    n_bot, mu_bot = 2.1, 1.7
    mee = wavebit.rcwa(backend=0, top=1.0, bot=n_bot ** 2, theta=0.0, phi=0.0, tetm='te',
                       truncation_order=(3, 0), period=(PERIOD, PERIOD), wavelength=WL,
                       thickness=[THICK], structure=np.ones((1, 1, 8)))
    e = [np.stack([np.eye(7, dtype=complex) * v]) for v in (1.0, 1.0, 1.0)]
    got = float(np.sum(mee.solve_conv(*e, mu_bot=mu_bot,
                                      eps_bot=n_bot ** 2 / mu_bot).incident.R))
    y_inc, y_sub = 1.0, n_bot / mu_bot          # kz/mu at normal incidence, kz = n
    assert got == pytest.approx(abs((y_inc - y_sub) / (y_inc + y_sub)) ** 2, abs=1e-13)
    # And it is not the answer an index-only boundary gives, which is the whole point of the
    # slot: that one reads |(1 - 2.1)/(1 + 2.1)|**2 = 0.126.
    assert abs(got - abs((1.0 - n_bot) / (1.0 + n_bot)) ** 2) > 1e-2
