"""Anisotropic uniform slabs against S4, from a checked-in golden.

The fourth oracle over this ground, and the least independent of them: fmmax generalises the
S4 paper, so those two share a formulation lineage and only RETICOLO and the Berreman 4x4 sit
outside it. What this adds is a second *implementation* of that lineage — C++ against LAPACK,
a decade older — which catches implementation errors and not formulation ones.

Its real value is that it makes two conventions executable rather than prose. Both were
misdiagnosed once, and both would be rediscovered by the next person to reach for S4:

* S4 indexes the permittivity tensor with **x and y exchanged** relative to this repo.
  ``test_the_axis_convention_is_load_bearing`` below fails by 4e-1 without the swap, which is
  the size that first read as a physics disagreement.
* Its ``IncidenceAngles`` is ``(polar, azimuth)`` as written — the angle convention was the
  first suspect and was innocent.

**Real tensors only.** This build rejects a complex ``Epsilon``, so the antisymmetric cases
here are active media rather than gyrotropic ones; ``R + T`` above 1 is their correct answer.
They discriminate the same algebra the imaginary off-diagonal does — which is the whole reason
the ordering defect in ``_core/solve_2d.py`` was reachable from a real tensor at all.

Values are frozen in ``golden_s4.npz`` by ``gen_goldens_s4.py``, which needs the S4 extension.
This file needs only numpy.
"""
import pathlib
import sys

import numpy as np
import pytest

import wavebit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_goldens_s4 import CASES, WL, PERIOD, THICK, SWAP      # noqa: E402

GOLDEN_PATH = HERE / 'golden_s4.npz'


def solve(eps, theta_deg):
    """``[R_s, T_s, R_p, T_p]`` for one uniform anisotropic layer between vacuum."""
    I = np.eye(1, dtype=np.complex128)
    e = np.asarray(eps, dtype=complex)
    mee = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=np.deg2rad(theta_deg),
                         phi=0.0, psi=0.0, fto=(0, 0), period=(PERIOD, PERIOD),
                         wavelength=WL, thickness=[THICK])
    res = mee.solve_conv(np.stack([e[0, 0] * I]), np.stack([e[1, 1] * I]),
                             np.stack([np.linalg.inv(e[2, 2] * I)]),
                             np.stack([-e[0, 1] * I]), np.stack([-e[1, 0] * I]))
    te, tm = res.res_te_inc, res.res_tm_inc
    return np.array([float(te.R.sum()), float(te.T.sum()),
                     float(tm.R.sum()), float(tm.T.sum())])


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with gen_goldens_s4.py')
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_matches_s4(golden, name):
    eps, theta_deg = CASES[name]
    gap = np.abs(solve(eps, theta_deg) - golden[name]).max()
    assert gap < 1e-12, f'{name}: {gap:.2e} from S4'


def test_the_axis_convention_is_load_bearing(golden):
    """Feeding S4's tensor without the x/y swap must be visibly wrong, not subtly so.

    This is the executable form of the note in ``gen_goldens_s4.py``. If someone removes the
    conjugation there and regenerates, the golden moves and every case above fails — but only
    if the swap actually matters on these tensors, which is what this pins. A diagonal-only
    case set would make the conjugation a no-op and the guard vacuous.
    """
    name = 'symmetric_offdiag_53deg'
    eps, theta_deg = CASES[name]
    swapped = SWAP @ np.asarray(eps, dtype=float) @ SWAP
    assert np.abs(swapped - np.asarray(eps, dtype=float)).max() > 0.1, (
        'the case set no longer distinguishes x from y; the swap guard is vacuous')
    gap = np.abs(solve(swapped, theta_deg) - golden[name]).max()
    assert gap > 1e-3, (
        f'swapping x and y changed the answer by only {gap:.2e}; either the tensor is too '
        'nearly symmetric to test this, or the convention has changed')


def test_golden_conserves_energy_on_the_symmetric_cases(golden):
    """Guards the golden. A real tensor is lossless exactly when it is symmetric.

    The antisymmetric cases are excluded on purpose: they are active, and ``R + T`` above 1 is
    correct for them rather than a symptom — the same distinction the RETICOLO and fmmax
    generators each got wrong once.
    """
    checked = 0
    for name, (eps, _) in CASES.items():
        e = np.asarray(eps, dtype=float)
        if np.abs(e - e.T).max() > 1e-12:
            continue
        r_s, t_s, r_p, t_p = golden[name]
        assert r_s + t_s == pytest.approx(1.0, abs=1e-9), f'{name} s'
        assert r_p + t_p == pytest.approx(1.0, abs=1e-9), f'{name} p'
        checked += 1
    assert checked >= 5, 'expected several lossless cases to guard'
