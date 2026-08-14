"""Permeability on the 2D path — what holds without an oracle.

The external check is ``tests/references/test_fmmax_magnetic_golden.py``, whose magnetic rows
were frozen before this code existed. This file is the other half: theorems and identities that
would hold even if that golden were wrong.

The one that carries the most weight is the duality. Maxwell under
``(E, H, eps, mu) -> (H, -E, mu, eps)`` maps onto itself, so a slab with its two tensors
exchanged is the same problem with the polarizations traded. That reaches **only** the ``mu``
path — nothing about it can be satisfied by a solver that quietly ignores permeability, and
nothing about it depends on fmmax.
"""
import numpy as np
import pytest

import wavebit

WL, PERIOD, THICK = 700.0, 400.0, 210.0
I1 = np.eye(1, dtype=np.complex128)


def tensor(xx, yy, zz, xy=0.0, yx=0.0):
    return np.array([[xx, xy, 0.0], [yx, yy, 0.0], [0.0, 0.0, zz]], dtype=complex)


ONE = tensor(1.0, 1.0, 1.0)

# (label, eps, mu). Chosen so each one reaches a different part of the operator: scalar touches
# only the diagonal, uniaxial only `muz_i`, biaxial breaks the x/y symmetry, and gyromagnetic is
# the only one that reaches the off-diagonal slots.
CASES = [
    ('scalar_mu', tensor(2.6, 2.6, 2.6), tensor(1.7, 1.7, 1.7)),
    ('uniaxial_mu', tensor(2.6, 2.6, 2.6), tensor(1.4, 1.4, 2.2)),
    ('biaxial_both', tensor(2.3, 3.0, 2.6), tensor(1.3, 1.8, 1.5)),
    ('gyromagnetic', tensor(2.6, 2.6, 2.6), tensor(1.6, 1.6, 1.6, 0.40j, -0.40j)),
]
IDS = [c[0] for c in CASES]


def slots(t):
    """A uniform tensor as the five stacks — the same convention for ``eps`` and for ``mu``."""
    t = np.asarray(t, dtype=complex)
    return [np.stack([t[0, 0] * I1]), np.stack([t[1, 1] * I1]),
            np.stack([np.linalg.inv(t[2, 2] * I1)]),
            np.stack([-t[0, 1] * I1]), np.stack([-t[1, 0] * I1])]


def solve(eps, mu=None, theta=0.0, algo='ETM', backend=0, fto=(0, 0), period=None):
    mee = wavebit.rcwa(backend=backend, n_top=1.0, n_bot=1.0, theta=np.deg2rad(theta),
                       phi=0.0, psi=0.0, fto=fto, period=period or (PERIOD, PERIOD),
                       wavelength=WL, thickness=[THICK], connecting_algo=algo)
    res = mee.solve_for_conv(WL, *slots(eps), mu=None if mu is None else tuple(slots(mu)))
    te, tm = res.res_te_inc, res.res_tm_inc
    f = lambda v: float(np.asarray(v.detach() if hasattr(v, 'detach') else v).sum())
    return np.array([f(te.R), f(te.T), f(tm.R), f(tm.T)])


# ---------------------------------------------------------------------------- the duality

@pytest.mark.parametrize('theta', [0.0, 39.0, 62.0])
@pytest.mark.parametrize('name, eps, mu', CASES, ids=IDS)
def test_exchanging_eps_and_mu_trades_s_and_p(name, eps, mu, theta):
    """``R_s(eps, mu) == R_p(mu, eps)``, exactly, at any incidence.

    A theorem, so it needs no reference solver, and it is the sharpest statement available
    about *where* ``mu`` sits: the duality only closes if the transverse permeability lands in
    ``Omega_L`` and ``mu_zz`` in ``Omega_R``, cross-paired (``B`` takes ``mu_yy``, ``D`` takes
    ``mu_xx``). Put ``mu`` anywhere else and this fails while energy conservation, reciprocity
    and a plausible-looking R/T all survive.

    Stated as a *trade* rather than as invariance on purpose. Invariance is what one expects at
    normal incidence and it is wrong there too: a biaxial in-plane tensor still distinguishes x
    from y, so the swap moves the answer by 4.7e-2 while trading it exactly.
    """
    straight = solve(eps, mu, theta=theta)
    swapped = solve(mu, eps, theta=theta)
    traded = np.array([swapped[2], swapped[3], swapped[0], swapped[1]])
    assert np.abs(straight - traded).max() < 1e-12


def test_the_duality_test_is_not_vacuous():
    """It would pass trivially if the swap changed nothing, so check that it does.

    ``scalar_mu`` at normal incidence genuinely is invariant — that case cannot tell the two
    apart and is excluded rather than weakened.
    """
    moved = 0
    for name, eps, mu in CASES:
        if np.abs(solve(eps, mu, theta=39.0) - solve(mu, eps, theta=39.0)).max() > 1e-3:
            moved += 1
    assert moved == len(CASES), f'only {moved} of {len(CASES)} cases respond to the swap'


# --------------------------------------------------------------- the non-magnetic path

@pytest.mark.parametrize('name, eps, mu', CASES, ids=IDS)
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_unit_permeability_is_the_non_magnetic_answer(name, eps, mu, algo):
    """Handing over an explicit ``mu = 1`` must agree with handing over nothing.

    Close rather than bit-identical, deliberately: ``mu=None`` keeps ``Omega2_LR``'s
    hand-expanded form and an explicit ``mu`` multiplies ``Omega_L @ Omega_R`` out. They are the
    same operator by different arithmetic, so they agree to rounding and not below it. The
    branch exists because the expanded form is *only* valid at ``mu = 1`` — see
    :func:`~wavebit._core.solve_2d.layer_modes_2d` — and keeping it means a non-magnetic solve
    is untouched by this feature, which the ``__mu1`` golden rows check against numbers that
    predate it.
    """
    assert np.abs(solve(eps, ONE, theta=39.0, algo=algo)
                  - solve(eps, None, theta=39.0, algo=algo)).max() < 1e-12


@pytest.mark.parametrize('name, eps, mu', CASES, ids=IDS)
def test_the_two_connecting_algorithms_agree_with_mu(name, eps, mu):
    """``mu`` is a property of a layer; ETM and SMM only differ in how layers are joined.

    ``smm_2d`` reaches ``layer_modes_2d`` by its own route, so it could have been left behind
    without anything failing — a stack would simply have come back non-magnetic under one
    algorithm and magnetic under the other.
    """
    assert np.abs(solve(eps, mu, theta=39.0, algo='ETM')
                  - solve(eps, mu, theta=39.0, algo='SMM')).max() < 1e-10


@pytest.mark.parametrize('backend, name', [(1, 'jax'), (2, 'torch')])
def test_the_other_backends_carry_mu_too(backend, name):
    """Same answer under jax and torch as under numpy."""
    pytest.importorskip(name)
    lift = {1: lambda a: __import__('jax.numpy', fromlist=['x']).asarray(a),
            2: lambda a: __import__('torch').as_tensor(a)}[backend]

    eps, mu = CASES[3][1], CASES[3][2]                # the gyromagnetic one
    want = solve(eps, mu, theta=39.0)

    mee = wavebit.rcwa(backend=backend, n_top=1.0, n_bot=1.0, theta=np.deg2rad(39.0), phi=0.0,
                       psi=0.0, fto=(0, 0), period=(PERIOD, PERIOD), wavelength=WL,
                       thickness=[THICK])
    res = mee.solve_for_conv(WL, *[lift(s) for s in slots(eps)],
                             mu=tuple(lift(s) for s in slots(mu)))
    f = lambda v: float(np.asarray(v.detach() if hasattr(v, 'detach') else v).sum())
    got = np.array([f(res.res_te_inc.R), f(res.res_te_inc.T),
                    f(res.res_tm_inc.R), f(res.res_tm_inc.T)])
    assert np.abs(got - want).max() < 1e-12


# ------------------------------------------------------------------ physics that must hold

@pytest.mark.parametrize('name, eps, mu', CASES, ids=IDS)
@pytest.mark.parametrize('theta', [0.0, 55.0])
def test_a_hermitian_pair_is_lossless(name, eps, mu, theta):
    """Hermitian in **both** tensors is what makes a magnetic medium lossless.

    Worth stating separately from the permittivity case: a gyromagnetic ``mu_xy = +i g`` with
    ``mu_yx = -i g`` is Hermitian and lossless exactly as its electric twin is, and the first
    version of this operator broke it — ``R + T`` came out 1.035, energy created out of the
    permeability.
    """
    for t in (eps, mu):
        assert np.abs(t - t.conj().T).max() < 1e-12, 'this case was meant to be lossless'
    r_s, t_s, r_p, t_p = solve(eps, mu, theta=theta)
    assert r_s + t_s == pytest.approx(1.0, abs=1e-10)
    assert r_p + t_p == pytest.approx(1.0, abs=1e-10)


# ------------------------------------------------------------------------------- the refusals

def test_mu_on_a_one_dimensional_structure_is_refused():
    """The 1D and conical operators carry no permeability slot, so ``mu`` there is a lie.

    Refusing beats ignoring: a dropped ``mu`` returns a finite, energy-conserving,
    entirely plausible non-magnetic answer.
    """
    mee = wavebit.rcwa(backend=0, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                       fto=(3, 0), period=(PERIOD, PERIOD), wavelength=WL, thickness=[THICK],
                       ucell=np.ones((1, 1, 8)) * 1.5)
    e = [np.stack([np.eye(7, dtype=complex) * v]) for v in (2.6, 2.6, 1 / 2.6)]
    m = tuple(np.stack([np.eye(7, dtype=complex) * v]) for v in (1.7, 1.7, 1 / 1.7, 0.0, 0.0))
    with pytest.raises(NotImplementedError, match='2D path only'):
        mee.solve_for_conv(WL, *e, mu=m)


def test_mu_with_a_z_coupled_permittivity_is_refused():
    """A z row routes to the first-order (4ff) solve, which has no ``mu`` in it.

    The two features are independent and both implemented, which is exactly why the
    *combination* has to say so rather than silently taking whichever path it lands on.
    """
    mee = wavebit.rcwa(backend=0, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                       fto=(0, 0), period=(PERIOD, PERIOD), wavelength=WL, thickness=[THICK])
    e = slots(tensor(2.6, 2.6, 2.6))
    z_row = [np.stack([0.2j * I1])] + [np.stack([0.0 * I1])] * 3
    m = tuple(slots(tensor(1.7, 1.7, 1.7)))
    with pytest.raises(NotImplementedError, match='first-order'):
        mee.solve_for_conv(WL, *e, *z_row, mu=m)
