"""The three formulations must describe the same structure.

Wavebit solves a 1D grating three ways, and which one runs is decided by inputs rather
than by a mode switch: ``phi=None`` takes the 1D TE/TM formulation, ``phi=0`` takes the 1D
conical one, and a ``structure`` with more than one row takes the full 2D solver. At normal
azimuth these are three routes to one answer, so any disagreement is a bug in whichever
route drifted -- there is no discretization difference to hide behind, because all three are
handed the same raster.

This is the check that used to live in ``QA/1d_pattern_in_1dc_and_2d.py``, where it printed
efficiencies for a human to compare by eye. Nothing recorded what "close enough" meant, and
nothing ran it. The tolerances below are what the three routes actually achieve, measured:
TE agrees to ~1e-14 and TM to ~1e-13, so 1e-11 fails on a real regression and not on
round-off.

Asserted per diffraction order, not on the totals. Totals can agree while individual orders
are wrong -- that is exactly the defect ``test_reticolo_golden`` exists to catch on the
absolute scale, and the same trap applies to a self-consistency check.
"""
import numpy as np
import pytest

import wavebit

# A binary grating, deliberately y-invariant so all three formulations are describing the
# identical structure. Two layers, so the stacking is exercised rather than a single slab.
_ROW = [1, 1, 1, 12.1104, 12.1104, 12.1104, 1, 1, 1, 1]   # eps: the index 3.48 squared
UCELL_1D = np.array([[_ROW], [_ROW]], dtype=float)
UCELL_2D = np.repeat(UCELL_1D, 2, axis=1)          # same structure, two identical y rows

CFG = dict(backend=0, top=1.0, bot=1.0, theta=1e-10, truncation_order=[20, 0],
           wavelength=300.0, period=[700.0, 700.0], thickness=[460.0, 22.0])

TOL = 1e-11


#: The linear angle each declaration is the pure case of. The two spellings are what this
#: file compares: ``tetm`` selects the TE/TM formulation, ``psi`` the conical one, and the
#: whole point is that they answer identically on a structure both can describe.
_PSI_OF = {'te': np.pi / 2, 'tm': 0.0}


def _solve(tetm, phi, structure):
    """``phi is None`` takes the declaration (the pure path); any azimuth takes the angle."""
    incidence = {'tetm': tetm} if phi is None else {'psi': _PSI_OF[tetm]}
    solver = wavebit.rcwa(**incidence, **CFG)
    solver.phi = phi
    solver.structure = structure
    res = solver.solve().given
    return np.squeeze(np.asarray(res.R)), np.squeeze(np.asarray(res.T))


@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['TE', 'TM'])
def test_1d_matches_1d_conical(tetm):
    """``phi=None`` and ``phi=0`` are the same physics through different algebra.

    The conical formulation carries both polarizations and the azimuthal coupling; at
    ``phi=0`` that coupling is empty, so it has to reduce to the TE/TM formulation exactly.
    A failure here means one of the two lost a term that the other kept.
    """
    ri_1d, ti_1d = _solve(tetm, None, UCELL_1D)
    ri_1c, ti_1c = _solve(tetm, 0.0, UCELL_1D)

    assert ri_1d.shape == ri_1c.shape
    assert np.abs(ri_1d - ri_1c).max() < TOL
    assert np.abs(ti_1d - ti_1c).max() < TOL


@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['TE', 'TM'])
def test_1d_conical_matches_2d(tetm):
    """A y-invariant structure solved by the 2D route must give the 1D conical answer.

    The 2D solver couples both in-plane directions and both polarizations. Feeding it a
    structure with no y dependence removes the coupling but not the code path, so this is
    the check that the general solver degrades to the special case rather than merely
    resembling it.
    """
    ri_1c, ti_1c = _solve(tetm, 0.0, UCELL_1D)
    ri_2d, ti_2d = _solve(tetm, 0.0, UCELL_2D)

    assert ri_1c.shape == ri_2d.shape
    assert np.abs(ri_1c - ri_2d).max() < TOL
    assert np.abs(ti_1c - ti_2d).max() < TOL


@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['TE', 'TM'])
def test_all_three_conserve_energy(tetm):
    """The structure is lossless, so each route must return R + T == 1 independently.

    Without this, the two tests above would still pass if all three routes were wrong in the
    same way -- agreement is not correctness.
    """
    for phi, structure in ((None, UCELL_1D), (0.0, UCELL_1D), (0.0, UCELL_2D)):
        ri, ti = _solve(tetm, phi, structure)
        assert abs(ri.sum() + ti.sum() - 1.0) < 1e-9, (phi, structure.shape)
