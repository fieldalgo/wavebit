"""Anisotropic uniform stacks against RETICOLO, from a checked-in golden.

``tests/rcwa/test_anisotropic_berreman.py`` judges the same cases with a Berreman 4x4 written in
this repo. That is an independent *formulation*, but not an independent *implementation* — a
misconception shared by both would survive it. This file is the outside check: a different
group's solver, in a different language, validated over two decades of published work, and
already this suite's strongest oracle for isotropic gratings.

What made it possible is that RETICOLO has carried a toolbox for arbitrarily anisotropic
multilayer stacks since V9 (Hugonin and Lalanne, arXiv:2101.00901). A texture may be given as
a cell holding a full complex 3x3 permittivity tensor; ``exemple_V9_0D_anisotrope.m`` ships
with it and is what ``interface/reti_aniso.m`` follows.

**No convention fitting was needed.** The isotropic and symmetric cases below agree with both
Berreman and this solver to ~1e-15 as they stand, which is what licenses reading the
asymmetric ones as physics rather than as a bookkeeping difference. S4 was tried for this role
first and failed exactly that gate — see ``docs/theory/rcwa/off-diagonal-slot-convention.md`` §8.

The asymmetric cases are asserted at 1e-12 like the symmetric ones. They were
``xfail(strict)`` until 60801164 (2026-08-10): ``layer_modes_2d`` carried the off-diagonal pair
in one order inside ``Omega2_LR`` and the other inside ``Omega_R``, so a tensor with
``eps_xy != eps_yx`` took its modes from one medium and its H-field from that medium's
transpose, and RETICOLO put the operator 4.2e-2 to 5.5e-2 out; exchanging the pair inside
``Omega_R`` alone landed every case at ~1e-15. No shipped path can produce an asymmetric tensor
— every producer emits ``-eps_xy`` twice — so the defect was latent, not a wrong released
number, and these cases are what keeps it fixed for a hand-filled tensor.

The golden also holds a ``Z_COUPLED_CASES`` block that **this file deliberately does not
consume**: those tensors have a populated z row, which the second-order path has no slot for. They
are generated here because the bridge and the guards already exist, and are asserted in
``test_first_order_external_goldens.py``.

Values are frozen in ``golden_reticolo_anisotropic.npz`` by
``gen_goldens_reticolo_anisotropic.py``, which needs Octave and RETICOLO V10. This file needs
neither.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(repository_path('tests')))
from tests.rcwa.test_anisotropic_berreman import solve_tensor_stack          # noqa: E402

GOLDEN_PATH = repository_path('tests/references/golden_reticolo_anisotropic.npz')

WL = 600.0
PERIOD = 300.0
THICK = 220.0


def _tensor(exx, eyy, ezz, exy=0.0, eyx=0.0):
    return np.array([[exx, exy, 0], [eyx, eyy, 0], [0, 0, ezz]], dtype=complex)


def is_hermitian(eps, tol=1e-12):
    """Hermitian permittivity means lossless. Also imported by the generator.

    Worth stating because the two asymmetric families here differ on it: a gyrotropic tensor
    (``eps_xy = +i g``, ``eps_yx = -i g``) *is* Hermitian and conserves energy, while a real
    antisymmetric one is not — it is an active medium, and R + T above one is its correct
    answer rather than a symptom of anything.
    """
    e = np.asarray(eps, dtype=complex)
    return bool(np.abs(e - e.conj().T).max() <= tol)


# name -> (eps 3x3, theta in degrees). Thickness and period are shared.
SYMMETRIC_CASES = {
    'isotropic_normal': (_tensor(2.25, 2.25, 2.25), 0.0),
    'isotropic_35deg': (_tensor(2.25, 2.25, 2.25), 35.0),
    'diagonal_biaxial_35deg': (_tensor(2.1, 3.4, 2.8), 35.0),
    'symmetric_offdiag_normal': (_tensor(2.25, 4.0, 2.25, 0.3, 0.3), 0.0),
    'symmetric_offdiag_35deg': (_tensor(2.25, 4.0, 2.25, 0.3, 0.3), 35.0),
}

ASYMMETRIC_CASES = {
    'real_antisymmetric_normal': (_tensor(2.25, 4.0, 2.25, 0.3, -0.3), 0.0),
    'real_antisymmetric_35deg': (_tensor(2.25, 4.0, 2.25, 0.3, -0.3), 35.0),
    'gyrotropic_normal': (_tensor(2.25, 2.25, 2.25, 0.45j, -0.45j), 0.0),
    'gyrotropic_45deg': (_tensor(2.25, 2.25, 2.25, 0.45j, -0.45j), 45.0),
}

def _gyro(direction, g=0.45, ed=2.25):
    """``eps_ij = ed d_ij + i e_ijk g_k`` — the magnetisation direction picks which cells fill.

    Along ``z`` this reproduces the Faraday tensor above; along ``x`` or ``y`` it populates the
    z row, which is the Voigt geometry and the whole reason a first-order solver exists.
    Hermitian either way, so lossless, so the generator's energy guard still applies.
    """
    e = np.eye(3, dtype=complex) * ed
    gx, gy, gz = np.asarray(direction, dtype=float) * g
    e[0, 1] += 1j * gz
    e[1, 0] -= 1j * gz
    e[1, 2] += 1j * gx
    e[2, 1] -= 1j * gx
    e[2, 0] += 1j * gy
    e[0, 2] -= 1j * gy
    return e


# A populated z row. **The second-order path cannot express these at all** — it has no slot for
# eps_xz or eps_yz — so none of the tests in this file may consume them; they exist for
# ``test_first_order_external_goldens.py``, and they are the first external check this repo has of
# the geometry the first-order solver was written for.
#
# Exactly one off-diagonal pair each, and delta0 = 0, because that is as far as RETICOLO can be
# trusted here — see the measured boundary in ``gen_goldens_reticolo_anisotropic.py``. A tilted
# magnetisation (xy and yz together) is deliberately absent: RETICOLO returns R + T = 1.0044 for
# it, on a tensor that is Hermitian and therefore lossless.
#
# Normal incidence is a control: with kx = 0 the z row has nothing to couple to, so a solver
# that silently dropped it would still be right there and wrong at 35 deg.
Z_COUPLED_CASES = {
    'voigt_x_normal': (_gyro((1, 0, 0)), 0.0),
    'voigt_x_35deg': (_gyro((1, 0, 0)), 35.0),
    'voigt_y_35deg': (_gyro((0, 1, 0)), 35.0),
}

ALL_CASES = {**SYMMETRIC_CASES, **ASYMMETRIC_CASES, **Z_COUPLED_CASES}


def solver_efficiencies(eps, theta_deg):
    """``[R_TE, T_TE, R_TM, T_TM]`` — the four numbers RETICOLO reports."""
    d = solve_tensor_stack([(eps, THICK)], np.deg2rad(theta_deg), wl=WL)
    return np.array([d['s']['R_s'] + d['s']['R_p'], d['s']['T_s'] + d['s']['T_p'],
                     d['p']['R_s'] + d['p']['R_p'], d['p']['T_s'] + d['p']['T_p']])


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with '
                    'gen_goldens_reticolo_anisotropic.py')
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize('name', list(SYMMETRIC_CASES), ids=list(SYMMETRIC_CASES))
def test_symmetric_tensors_match_reticolo(golden, name):
    """The convention gate. These are the tensors a shipped path can actually produce."""
    eps, theta_deg = SYMMETRIC_CASES[name]
    gap = np.abs(solver_efficiencies(eps, theta_deg) - golden[name]).max()
    assert gap < 1e-12, f'{name}: {gap:.2e} from RETICOLO'


@pytest.mark.parametrize('name', list(ASYMMETRIC_CASES), ids=list(ASYMMETRIC_CASES))
def test_asymmetric_tensors_match_reticolo(golden, name):
    """``eps_xy != eps_yx``, which nothing in the library builds and only a hand-filled tensor
    reaches.

    These were ``xfail`` until the two blocks in ``layer_modes_2d`` were given the off-diagonal
    pair in opposite orders; before that the shipped operator was 4.2e-2 to 5.5e-2 out here.
    """
    eps, theta_deg = ASYMMETRIC_CASES[name]
    gap = np.abs(solver_efficiencies(eps, theta_deg) - golden[name]).max()
    assert gap < 1e-12, f'{name}: {gap:.2e} from RETICOLO'


def test_golden_conserves_energy_on_the_hermitian_cases(golden):
    """Guards the golden itself.

    A Hermitian tensor is lossless, so RETICOLO's own numbers must sum to one. A golden that
    failed this would mean the generator wired something wrong, and the comparisons above would
    be measuring that instead of the solver.

    The real antisymmetric cases are deliberately excluded: they are *not* Hermitian, so they
    are active media and R + T above one is correct for them. Asserting otherwise is a mistake
    this file's generator caught the first time it ran.
    """
    checked = 0
    for name, (eps, _) in ALL_CASES.items():
        if not is_hermitian(eps):
            continue
        r_te, t_te, r_tm, t_tm = golden[name]
        assert r_te + t_te == pytest.approx(1.0, abs=1e-9), f'{name} TE'
        assert r_tm + t_tm == pytest.approx(1.0, abs=1e-9), f'{name} TM'
        checked += 1
    assert checked >= 5, 'expected several lossless cases to guard'
