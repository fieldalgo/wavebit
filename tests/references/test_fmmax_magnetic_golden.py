"""Magnetic (``mu != 1``) slabs against fmmax — the golden that was frozen before its consumer.

Permeability landed on the 2D path on 2026-08-14 (``layer_modes_2d``, ``solve_2d``,
``smm_2d``), and **this file is why that landing can be believed**. The magnetic rows here were
generated while the solver had nowhere to put a ``mu`` at all, deliberately: an oracle chosen
after the work is an oracle chosen to agree. Nothing in the implementation could be fitted to
them, because they existed first and were not readable as a target — the derivation was settled
against Maxwell and against the shipped ``mu = 1`` operator, and only then run against these.

It caught a real error on the first run, which is the point of the arrangement. A scalar ``mu``
came out exact at both normal and oblique incidence while a *uniaxial* one was wrong by 1e-2 in
s only, and a gyromagnetic one broke energy conservation outright (``R + T = 1.035``). The cause
was ``Omega2_LR``'s hand-expanded form: its four bare ``K @ K`` terms are not geometry that
passed through untouched, they are what other terms collapse to when the K factors commute past
a ``muz_conv_i`` of one. See :func:`~wavebit.rcwa._engine.solve_2d.layer_modes_2d`.

What the file holds, and what each half is for:

* the ``__mu1`` rows — every case is generated twice, once with its permeability and once with
  ``mu = 1``. These were the convention gate while the magnetic half was unreachable, and they
  keep that job: they are what says the harness, units, incidence convention and flux
  normalisation are right, independently of anything ``mu`` does. They are also now a
  **regression gate on the non-magnetic path**, which must be untouched.
* the magnetic rows — held to the solver directly. Their own guard is still physical rather
  than comparative: every case is Hermitian in both tensors, hence lossless, so ``R + T = 1``
  is enforced on the frozen numbers as well as on ours.

Scope. ``eigensolve_general_anisotropic_media`` takes five ``mu`` components with no z row, so
this is the same second-order tensor class extended to ``mu``; a magnetic Voigt geometry is as far
out of fmmax's reach as ours. And both half-spaces are **vacuum** in every case, which is what lets
these rows validate the layer operator alone — a magnetic ``n_top`` / ``n_bot`` needs the
boundary admittance, which nothing here checks and which is not implemented.
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


HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE))
from tests.references.gen_goldens_fmmax_magnetic import CASES, WL, PERIOD, THICK, ONE     # noqa: E402

GOLDEN_PATH = HERE / 'golden_fmmax_magnetic.npz'


I1 = np.eye(1, dtype=np.complex128)


def _slots(tensor):
    """A uniform tensor as the five stacks, in the solver's slot convention.

    Diagonal entries as they are, ``zz`` inverted, off-diagonals negated — identical for ``eps``
    and ``mu``, which is the whole design: ``mu`` is not a second kind of input, it is the same
    five slots on the dual side.
    """
    t = np.asarray(tensor, dtype=complex)
    return [np.stack([t[0, 0] * I1]), np.stack([t[1, 1] * I1]),
            np.stack([np.linalg.inv(t[2, 2] * I1)]),
            np.stack([-t[0, 1] * I1]), np.stack([-t[1, 0] * I1])]


def solve(eps, theta_deg, mu=None, algo='ETM'):
    """``[R_s, T_s, R_p, T_p]`` for one uniform slab between vacuum."""
    mee = _legacy_wavebit_rcwa(backend='numpy', top=1.0, bot=1.0, theta=np.deg2rad(theta_deg),
                         phi=0.0, psi=0.0, truncation_order=(0, 0), period=(PERIOD, PERIOD),
                         wavelength=WL, thickness=[THICK], layer_algorithm=algo)
    res = mee.solve_conv(*_slots(eps),
                             mu=None if mu is None else tuple(_slots(mu)))
    te, tm = res.s_inc, res.p_inc
    return np.array([float(te.R.sum()), float(te.T.sum()),
                     float(tm.R.sum()), float(tm.T.sum())])


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with '
                    'gen_goldens_fmmax_magnetic.py')
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_mu1_half_matches(golden, name):
    """The convention gate, and now also the non-magnetic regression gate.

    Passing ``mu=None`` has to leave the old arithmetic alone — not merely close to it. These
    rows say so against numbers that predate the ``mu`` work entirely.
    """
    eps, _, theta_deg = CASES[name]
    gap = np.abs(solve(eps, theta_deg) - golden[name + '__mu1']).max()
    assert gap < 1e-12, f'{name}__mu1: {gap:.2e} from fmmax'


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_magnetic_rows_match(golden, name, algo):
    """The half that had no consumer until now.

    Run under **both** connecting algorithms. The connecting algorithm is a choice about how
    layers are joined, not about what a layer is made of, so one accepting ``mu`` while the
    other quietly ignored it would return a plausible non-magnetic answer for a magnetic stack
    — and ``smm_2d`` reaches ``layer_modes_2d`` by its own route, so nothing but this makes the
    two agree.
    """
    eps, mu, theta_deg = CASES[name]
    gap = np.abs(solve(eps, theta_deg, mu=mu, algo=algo) - golden[name]).max()
    assert gap < 1e-12, f'{name} [{algo}]: {gap:.2e} from fmmax'


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_a_magnetic_layer_is_not_silently_solved_as_a_non_magnetic_one(golden, name):
    """The failure this file exists to prevent, stated as its own assertion.

    Dropping ``mu`` on the floor produces a finite, energy-conserving, entirely plausible
    number. Only a comparison against what the answer *should not* be catches it, so the
    magnetic result must differ from the ``mu = 1`` result by more than the tolerance above —
    except for the one case whose permeability is genuinely unity.
    """
    eps, mu, theta_deg = CASES[name]
    if np.abs(np.asarray(mu, dtype=complex) - ONE).max() <= 1e-12:
        pytest.skip('this case has mu = 1 by construction; see the twin test below')
    moved = np.abs(solve(eps, theta_deg, mu=mu) - solve(eps, theta_deg)).max()
    assert moved > 1e-9, f'{name}: mu changed the answer by only {moved:.2e}'


def test_the_magnetic_rows_conserve_energy(golden):
    """Guards the frozen magnetic rows themselves, independently of the solver.

    Every case is Hermitian in both tensors and therefore lossless, so this is a real
    constraint rather than a restatement. Kept now that the rows have a consumer: it is what
    would tell a future regeneration apart from a corruption.
    """
    def hermitian(a):
        a = np.asarray(a, dtype=complex)
        return np.abs(a - a.conj().T).max() <= 1e-12

    checked = 0
    for name, (eps, mu, _) in CASES.items():
        if not (hermitian(eps) and hermitian(mu)):
            continue
        r_s, t_s, r_p, t_p = golden[name]
        assert r_s + t_s == pytest.approx(1.0, abs=1e-9), f'{name} s'
        assert r_p + t_p == pytest.approx(1.0, abs=1e-9), f'{name} p'
        checked += 1
    assert checked >= 6, 'expected the magnetic cases to be lossless by construction'


def test_the_magnetic_rows_actually_differ_from_their_mu1_twins(golden):
    """Otherwise the golden records nothing about ``mu`` at all.

    ``gyroelectric_46deg`` is the deliberate exception — its permeability *is* unity, so its two
    rows coincide, and that coincidence is itself a check that the ``__mu1`` generation path is
    doing what it claims.
    """
    moved, unmoved = [], []
    for name, (_, mu, _) in CASES.items():
        same = np.abs(golden[name] - golden[name + '__mu1']).max() < 1e-12
        (unmoved if same else moved).append(name)
        is_unity = np.abs(np.asarray(mu, dtype=complex) - ONE).max() <= 1e-12
        assert same == is_unity, (
            f'{name}: mu {"is" if is_unity else "is not"} unity but the rows '
            f'{"coincide" if same else "differ"}')
    assert len(moved) >= 5, f'only {len(moved)} cases respond to mu; the golden is thin'
    assert unmoved == ['gyroelectric_46deg'], f'unexpected unity-mu cases: {unmoved}'
