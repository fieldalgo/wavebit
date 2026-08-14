"""Magnetic (``mu != 1``) slabs against fmmax — a golden laid down before its consumer exists.

Permeability has no slot anywhere in this solver: ``layer_modes_2d`` writes ``I`` where a
``mu_conv`` would go, and ``boundary_kz`` takes one material argument
(``docs/anisotropic-materials.md`` §5). So none of the magnetic rows in
``golden_fmmax_magnetic.npz`` can be compared against anything here yet.

Freezing them anyway is the point. **An oracle chosen after the work is an oracle chosen to
agree** — if the mu rows were generated once ``mu`` support existed, every convention question
they might have caught would already have been settled by whoever wrote the support, in
whatever way made their code pass. Recorded first, they are evidence; recorded after, they are
a mirror.

What keeps that from being a leap of faith is the ``__mu1`` half. Every case is generated twice,
once with its permeability and once with ``mu = 1``, and the ``mu = 1`` rows **are** held to the
solver below. If the harness, the units, the incidence convention or the flux normalisation
were wrong, that subset would fail today, while the answer is independently knowable. It is the
same convention gate the other three oracles passed, applied to the half of an oracle that can
still be gated.

Two things this does not pin, and should not be read as pinning:

* The magnetic rows are *unverified against anything but fmmax itself*. Their guard is
  physical, not comparative: every one is Hermitian in both tensors, hence lossless, so
  ``R + T = 1`` is enforced on the frozen numbers.
* ``eigensolve_general_anisotropic_media`` takes five ``mu`` components with no z row, so this
  is the same 2ff tensor class extended to ``mu``. A magnetic Voigt geometry is as far out of
  fmmax's reach as ours.
"""
import pathlib
import sys

import numpy as np
import pytest

import wavebit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_goldens_fmmax_magnetic import CASES, WL, PERIOD, THICK, ONE     # noqa: E402

GOLDEN_PATH = HERE / 'golden_fmmax_magnetic.npz'


def solve(eps, theta_deg):
    """``[R_s, T_s, R_p, T_p]``. Only ``eps`` — there is nowhere to put ``mu``."""
    I = np.eye(1, dtype=np.complex128)
    e = np.asarray(eps, dtype=complex)
    mee = wavebit.rcwa(backend=0, n_top=1.0, n_bot=1.0, theta=np.deg2rad(theta_deg),
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
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with '
                    'gen_goldens_fmmax_magnetic.py')
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_the_mu1_half_matches(golden, name):
    """The convention gate. These rows have ``mu = 1``, so the solver can be held to them."""
    eps, _, theta_deg = CASES[name]
    gap = np.abs(solve(eps, theta_deg) - golden[name + '__mu1']).max()
    assert gap < 1e-12, f'{name}__mu1: {gap:.2e} from fmmax'


def test_the_magnetic_rows_conserve_energy(golden):
    """Guards the frozen magnetic rows, which nothing else can check yet.

    Every case is Hermitian in both tensors and therefore lossless, so this is a real
    constraint rather than a restatement.
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
