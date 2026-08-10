"""The three formulations must describe the same structure.

FieldWave solves a 1D grating three ways, and which one runs is decided by inputs rather
than by a mode switch: ``phi=None`` takes the 1D TE/TM formulation, ``phi=0`` takes the 1D
conical one, and a ``ucell`` with more than one row takes the full 2D solver. At normal
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

import fieldwave

# A binary grating, deliberately y-invariant so all three formulations are describing the
# identical structure. Two layers, so the stacking is exercised rather than a single slab.
_ROW = [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1]
UCELL_1D = np.array([[_ROW], [_ROW]], dtype=float)
UCELL_2D = np.repeat(UCELL_1D, 2, axis=1)          # same structure, two identical y rows

CFG = dict(backend=0, n_top=1.0, n_bot=1.0, theta=1e-10, fto=[20, 0],
           wavelength=300.0, period=[700.0, 700.0], thickness=[460.0, 22.0])

TOL = 1e-11


def _solve(pol, phi, ucell):
    solver = fieldwave.rcwa(pol=pol, **CFG)
    solver.phi = phi
    solver.ucell = ucell
    res = solver.conv_solve().res
    return np.squeeze(np.asarray(res.de_ri)), np.squeeze(np.asarray(res.de_ti))


@pytest.mark.parametrize('pol', [0, 1], ids=['TE', 'TM'])
def test_1d_matches_1d_conical(pol):
    """``phi=None`` and ``phi=0`` are the same physics through different algebra.

    The conical formulation carries both polarizations and the azimuthal coupling; at
    ``phi=0`` that coupling is empty, so it has to reduce to the TE/TM formulation exactly.
    A failure here means one of the two lost a term that the other kept.
    """
    ri_1d, ti_1d = _solve(pol, None, UCELL_1D)
    ri_1c, ti_1c = _solve(pol, 0.0, UCELL_1D)

    assert ri_1d.shape == ri_1c.shape
    assert np.abs(ri_1d - ri_1c).max() < TOL
    assert np.abs(ti_1d - ti_1c).max() < TOL


@pytest.mark.parametrize('pol', [0, 1], ids=['TE', 'TM'])
def test_1d_conical_matches_2d(pol):
    """A y-invariant structure solved by the 2D route must give the 1D conical answer.

    The 2D solver couples both in-plane directions and both polarizations. Feeding it a
    structure with no y dependence removes the coupling but not the code path, so this is
    the check that the general solver degrades to the special case rather than merely
    resembling it.
    """
    ri_1c, ti_1c = _solve(pol, 0.0, UCELL_1D)
    ri_2d, ti_2d = _solve(pol, 0.0, UCELL_2D)

    assert ri_1c.shape == ri_2d.shape
    assert np.abs(ri_1c - ri_2d).max() < TOL
    assert np.abs(ti_1c - ti_2d).max() < TOL


@pytest.mark.parametrize('pol', [0, 1], ids=['TE', 'TM'])
def test_all_three_conserve_energy(pol):
    """The structure is lossless, so each route must return R + T == 1 independently.

    Without this, the two tests above would still pass if all three routes were wrong in the
    same way -- agreement is not correctness.
    """
    for phi, ucell in ((None, UCELL_1D), (0.0, UCELL_1D), (0.0, UCELL_2D)):
        ri, ti = _solve(pol, phi, ucell)
        assert abs(ri.sum() + ti.sum() - 1.0) < 1e-9, (phi, ucell.shape)


# --------------------------------------------------------- block path vs raster, per backend

# A single rectangle, given twice: once as a drawn shape for the block path, once as the raster
# of that same shape. Edges sit on pixel boundaries, so the raster *is* the rectangle and the two
# descriptions are the same structure rather than an approximation of one another.
# The raster is 12 cells over a 1200 nm period, so cells 4..7 are x = 400..800 -- a 400 nm
# rectangle centred at 600. `ly` spans the full period because the structure is y-invariant.
# These are absolute lengths in the units of `period`, not fractions of it.
# n = 3.48 rather than a round number on purpose: 3.5 is exactly representable in float32, so
# a single-precision raster would round-trip through it unharmed and this test would pass while
# blind to the very defect it exists for. 3.48 is not, and it separates the two by ~3e-8.
_N = 3.48
_RASTER = np.array([[[1.0] * 4 + [_N] * 4 + [1.0] * 4]])
_BLOCK = [[1.0, [['rectangle', 600.0, 600.0, 400.0, 1200.0, _N, 0, 0, 0]]]]

_CFG = dict(n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, pol=0.0,
            fto=[5, 0], wavelength=900.0, period=[1200.0, 1200.0], thickness=[300.0])


@pytest.mark.parametrize('backend,name', [(0, 'numpy'), (1, 'jax'), (2, 'torch')])
def test_block_and_raster_describe_the_same_structure(backend, name):
    """The two modeling paths must agree, on every backend, to round-off.

    This is a regression test for a defect in the *example* that used to demonstrate it:
    ``examples/vector_*_verification.py`` built the raster with ``torch.as_tensor`` on a Python
    list, which takes torch's **default** dtype -- float32 -- and no later cast recovers digits
    the constructor already dropped. It printed a block-vs-raster disagreement of 5.6e-5 on
    torch while numpy and jax agreed to 1e-13, which reads as a solver defect and was not one.

    The tolerance is what the paths actually achieve once the dtype is right: ~1e-13. Anything
    near 1e-5 again means someone has reintroduced a single-precision array on one backend.
    """
    if name != 'numpy':
        pytest.importorskip(name)

    block = fieldwave.rcwa(backend=backend, ucell=_BLOCK, **_CFG).conv_solve().res
    raster = fieldwave.rcwa(backend=backend, ucell=_RASTER, **_CFG).conv_solve().res

    for a, b, which in ((block.de_ri, raster.de_ri, 'de_ri'),
                        (block.de_ti, raster.de_ti, 'de_ti')):
        diff = np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
        assert diff < 1e-11, f'{name} {which}: {diff:.3e}'
