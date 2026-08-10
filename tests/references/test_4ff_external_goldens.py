"""The 4ff solve against two outside solvers, from goldens already checked in.

``tests/test_solve_2d_4ff.py`` pins the first-order formulation against the 2ff path and
against the Berreman 4x4 in ``tests/test_anisotropic_berreman.py``. Both of those are ours. An
error in how this repo reads the problem — a convention, a factor, a transverse-field
definition — is shared by all three and survives every one of those tests.

This file is the outside check, and it costs nothing new: ``golden_reticolo_anisotropic.npz``
and ``golden_fmmax.npz`` were frozen for the 2ff path and happen to cover exactly the 4ff
domain, uniform layers between vacuum. Nine RETICOLO cases and twelve fmmax ones, sharing no
wavelength, thickness, tensor or angle between the two sets.

Three of the RETICOLO cases have a **populated z row** — the Voigt geometry, which is the whole
reason the 4ff module exists and which the 2ff path cannot express at all. Those are the first
external confirmation of it this repo has had; everything else about that geometry rested on a
Berreman 4x4 we wrote ourselves.

How far that goes is bounded, and the bound was measured rather than assumed. RETICOLO's
uniform anisotropic layer is exact for a diagonal tensor at any mounting and for a single
off-diagonal pair at ``delta0 = 0``; past that it returns ``R + T`` up to 1.2e-2 away from one
on a tensor that is provably lossless. The table and the reasoning are in
``gen_goldens_reticolo_anisotropic.py``. So a tilted magnetisation — xy and yz populated at
once, the ordinary case for a real magnetised film — still has no outside judge, and
``test_solve_2d_4ff.py`` remains the only thing standing behind it.
"""
import pathlib
import sys

import numpy as np
import pytest

from fieldwave._core.solve_2d_4ff import solve_uniform_stack, vacuum_sp_states

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _fmmax_cases import CASES as FMMAX_CASES, THICK as FMMAX_THICK, WL as FMMAX_WL  # noqa: E402
from test_reticolo_anisotropic_golden import (ALL_CASES as RETICOLO_CASES,  # noqa: E402
                                              THICK as RETICOLO_THICK, WL as RETICOLO_WL,
                                              is_hermitian)


def efficiencies(eps, theta_deg, wl, thick):
    """``[R_TE, T_TE, R_TM, T_TM]`` — the four numbers both goldens store, in that order.

    TE is s and TM is p, named through :func:`vacuum_sp_states` rather than read off the vacuum
    eigenbasis. That basis is degenerate and its ordering is not something ``eig`` promises.
    """
    kx, ky = np.array([np.sin(np.deg2rad(theta_deg))]), np.array([0.0])
    e = np.asarray(eps, dtype=complex)
    I = np.eye(1, dtype=complex)
    layers = [(dict(exx=e[0, 0] * I, exy=e[0, 1] * I, exz=e[0, 2] * I,
                    eyx=e[1, 0] * I, eyy=e[1, 1] * I, eyz=e[1, 2] * I,
                    ezx=e[2, 0] * I, ezy=e[2, 1] * I, ezz_i=np.linalg.inv(e[2, 2] * I)),
               thick)]
    psi_s, psi_p = vacuum_sp_states(np, kx, ky, 1)
    r_te, t_te = solve_uniform_stack(wl, kx, ky, layers, 1, psi_s)
    r_tm, t_tm = solve_uniform_stack(wl, kx, ky, layers, 1, psi_p)
    return np.array([r_te, t_te, r_tm, t_tm])


def _golden(name):
    path = HERE / name
    if not path.exists():
        pytest.skip(f'{name} missing; regenerate with its gen_goldens_*.py')
    return np.load(path)


@pytest.mark.parametrize('name', list(RETICOLO_CASES), ids=list(RETICOLO_CASES))
def test_matches_reticolo(name):
    """RETICOLO is the strongest oracle here — a different group's solver, different language.

    Four of these tensors are asymmetric (``eps_xy != eps_yx``), which is what caught the
    ordering defect in ``layer_modes_2d``. The 4ff operator takes the pair as given and never
    forms ``Omega_R``, so it cannot make that mistake; passing here says it does not make a
    different one.
    """
    eps, theta_deg = RETICOLO_CASES[name]
    got = efficiencies(eps, theta_deg, RETICOLO_WL, RETICOLO_THICK)
    gap = np.abs(got - _golden('golden_reticolo_anisotropic.npz')[name]).max()
    assert gap < 1e-12, f'{name}: {gap:.2e} from RETICOLO'


@pytest.mark.parametrize('name', list(FMMAX_CASES), ids=list(FMMAX_CASES))
def test_matches_fmmax(name):
    """A second outside solver, on cases sharing nothing with the RETICOLO set.

    Includes ``absorbing_gyrotropic_25deg``, the only lossy tensor either golden holds — which
    is the case the direction split has to get right without help from energy conservation.
    """
    eps, theta_deg = FMMAX_CASES[name]
    got = efficiencies(eps, theta_deg, FMMAX_WL, FMMAX_THICK)
    gap = np.abs(got - _golden('golden_fmmax.npz')[name]).max()
    assert gap < 1e-12, f'{name}: {gap:.2e} from fmmax'


def test_the_external_coverage_of_z_coupling_is_what_the_docs_claim():
    """Pins how far outside validation actually reaches, so the prose cannot quietly rot.

    Two separate claims, both load-bearing and both easy to overstate later:

    * At least one RETICOLO case has a populated z row. Without this the module's "externally
      validated" line would be true only of geometries 2ff already handles.
    * Every such case has exactly **one** off-diagonal pair. Beyond that RETICOLO stops
      conserving energy on lossless tensors, so anything richer must not be frozen — a tilted
      magnetisation is measured at ``R + T = 1.0044``.

    Adding a Voigt case to the fmmax generator would also trip this; fmmax's anisotropic entry
    point takes the five 2ff slots and has no z row to give.
    """
    def pairs(eps):
        e = np.asarray(eps, dtype=complex)
        return sum(1 for a, b in ((0, 1), (0, 2), (1, 2)) if max(abs(e[a, b]), abs(e[b, a])) > 0)

    def z_coupled(eps):
        e = np.asarray(eps, dtype=complex)
        return np.abs([e[0, 2], e[1, 2], e[2, 0], e[2, 1]]).max() > 0

    reti_z = [n for n, (eps, _) in RETICOLO_CASES.items() if z_coupled(eps)]
    assert reti_z, 'no RETICOLO case reaches the geometry 4ff exists for'
    for name in reti_z:
        n = pairs(RETICOLO_CASES[name][0])
        assert n == 1, (f'{name} has {n} off-diagonal pairs; RETICOLO is only exact for one — '
                        'see gen_goldens_reticolo_anisotropic.py before freezing this')

    for name, (eps, _) in FMMAX_CASES.items():
        assert not z_coupled(eps), f'{name}: fmmax has no z row to give'


@pytest.mark.parametrize('name', list(RETICOLO_CASES), ids=list(RETICOLO_CASES))
def test_lossless_cases_conserve_energy(name):
    """Physical guard on our own numbers, independent of the goldens.

    Deliberately skips the real-antisymmetric tensors: those are not Hermitian, so they are
    active media and ``R + T > 1`` is their correct answer — a distinction
    ``test_reticolo_anisotropic_golden.py`` records having gotten wrong once.
    """
    eps, theta_deg = RETICOLO_CASES[name]
    if not is_hermitian(eps):
        pytest.skip('not Hermitian, so not lossless — R + T > 1 is correct here')
    r_te, t_te, r_tm, t_tm = efficiencies(eps, theta_deg, RETICOLO_WL, RETICOLO_THICK)
    assert r_te + t_te == pytest.approx(1.0, abs=1e-12), f'{name} TE'
    assert r_tm + t_tm == pytest.approx(1.0, abs=1e-12), f'{name} TM'
