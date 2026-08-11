"""Anisotropic uniform slabs against fmmax, from a checked-in golden.

The third oracle over these cases, after the Berreman 4x4 in
``tests/test_anisotropic_berreman.py`` and the RETICOLO golden beside this file. Three is not
excessive here: the first is an independent *formulation* but our own *implementation*, and the
second needed Octave and a convention gate before it could be trusted. fmmax is MIT, installs
with pip, and reproduced the analytic slab at 1e-17 and Berreman at ~1e-16 with **no convention
fitting at all** — the only one of the three that did.

Its anisotropic entry point takes the same five slots this solver does
(``permittivity_xx, xy, yx, yy, zz``, no z row), so it is a 2ff formulation too and cannot
arbitrate the Voigt geometry. RETICOLO stays the oracle there. What fmmax reaches that neither
other does is a **patterned** complex off-diagonal tensor — nothing here uses that yet, because
the solver cannot express one, but it is why this file exists rather than leaning on the
RETICOLO golden alone.

The cases deliberately share nothing with the RETICOLO set: different wavelength, period,
thickness, tensors and angles. Two oracles that agree only where they were handed identical
numbers would not be telling us much.

Values are frozen in ``golden_fmmax.npz`` by ``gen_goldens_fmmax.py``, which needs an
environment with fmmax in it (``fwref``, see that file). This one needs only numpy.
"""
import pathlib
import sys

import numpy as np
import pytest

import fieldwave

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _fmmax_cases import CASES, WL, PERIOD, THICK          # noqa: E402

GOLDEN_PATH = HERE / 'golden_fmmax.npz'


def solve(eps, theta_deg):
    """``[R_s, T_s, R_p, T_p]`` for one uniform anisotropic layer between vacuum.

    ``_fmmax_cases`` states the physical tensor; the slots take its negation off the diagonal
    — see ``docs/off-diagonal-slot-convention.md`` §2.
    """
    I = np.eye(1, dtype=np.complex128)
    e = np.asarray(eps, dtype=complex)
    mee = fieldwave.rcwa(backend=0, n_top=1.0, n_bot=1.0, theta=np.deg2rad(theta_deg),
                         phi=0.0, psi=0.0, fto=(0, 0), period=(PERIOD, PERIOD),
                         wavelength=WL, thickness=[THICK])
    res = mee.solve_for_conv(WL, np.stack([e[0, 0] * I]), np.stack([e[1, 1] * I]),
                             np.stack([np.linalg.inv(e[2, 2] * I)]),
                             np.stack([-e[0, 1] * I]), np.stack([-e[1, 0] * I]))
    te, tm = res.res_te_inc, res.res_tm_inc
    return np.array([float(te.R.sum()), float(te.T.sum()),
                     float(tm.R.sum()), float(tm.T.sum())])


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with gen_goldens_fmmax.py')
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_matches_fmmax(golden, name):
    eps, theta_deg = CASES[name]
    gap = np.abs(solve(eps, theta_deg) - golden[name]).max()
    assert gap < 1e-12, f'{name}: {gap:.2e} from fmmax'


def test_golden_conserves_energy_on_the_lossless_cases(golden):
    """Guards the golden itself.

    A Hermitian tensor is lossless whatever its off-diagonal looks like — which includes every
    gyrotropic case here, since ``eps_xy = +i g`` with ``eps_yx = -i g`` is Hermitian. An
    earlier version of the generator tested for a *real* tensor instead and so skipped exactly
    those, leaving the cases most worth guarding unguarded.
    """
    checked = 0
    for name, (eps, _) in CASES.items():
        e = np.asarray(eps, dtype=complex)
        if np.abs(e - e.conj().T).max() > 1e-12:
            continue
        r_s, t_s, r_p, t_p = golden[name]
        assert r_s + t_s == pytest.approx(1.0, abs=1e-9), f'{name} s'
        assert r_p + t_p == pytest.approx(1.0, abs=1e-9), f'{name} p'
        checked += 1
    assert checked >= 10, 'expected most cases to be lossless'


def test_the_two_external_oracles_are_not_the_same_configurations():
    """The RETICOLO and fmmax goldens must not be measuring the same physical setups.

    Agreement between two oracles handed identical inputs is worth less than it looks; this
    keeps the sets from drifting together as cases get added to either.

    Names are allowed to collide — ``isotropic_normal`` is the obvious thing to call the
    obvious case in both files, and does. What must differ is the configuration, which the
    stack geometry already guarantees.
    """
    from test_reticolo_anisotropic_golden import (WL as RETI_WL, PERIOD as RETI_PERIOD,
                                                  THICK as RETI_THICK)

    assert (WL, PERIOD, THICK) != (RETI_WL, RETI_PERIOD, RETI_THICK), (
        'the two oracle sets share a stack geometry, so every case in one has a twin in the '
        'other; change one of them')
    assert WL != RETI_WL, 'differing only in period or thickness is too close for comfort'
