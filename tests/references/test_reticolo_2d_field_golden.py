"""The in-cell field of a **crossed grating**, against RETICOLO, under both in-plane rules.

The 1D and conical field goldens beside this one reach the grating paths. This one reaches
``field_2d`` on two patterned layers (``_reticolo_2d_field_cases.py``), with the raster drawn so
that it *is* the structure RETICOLO is given as rectangles, and with
``fourier_coefficients='integrated'`` and ``factorization='li'``, so both codes build the same
matrices and every difference left is implementation and convention.

**Two rules, two references.** RETICOLO's ``parm.res1.champ`` picks how ``res3`` rebuilds the
in-plane ``E`` inside a patterned layer, and the golden stores both:

* ``champ = 1`` is the line-wise continuity rule of P. Lalanne and M. P. Jurek, J. Mod. Opt. 45,
  1357 (1998): ``E_x`` summed over the y orders, turned into ``D_x`` by Li's inverse rule along
  the raster row under the sample, summed along x, divided by ``eps``; ``E_y`` the same with the
  axes swapped. That is ``discontinuous='continuity'``, and it agrees on **all six** components.
* ``champ = 0`` is the plain Fourier sum, this project's default ``'fourier'``. RETICOLO leaves
  ``E_z`` and ``H_z`` unset inside a layer under ``champ = 0`` (exactly zero, asserted below),
  so that comparison covers the four transverse components; the default's ``E_z`` and ``H_z``
  are the continuity rule's, which ``champ = 1`` pins.

The two rules differ by about 0.3 in ``E_x`` and ``E_y`` on these layers, so each reference
tells them apart by ten orders of magnitude over the tolerance.

**The frame needs no transform.** ``res3``'s 2D cell is ``(z, x, y, 6)``, z ascending from the
bottom of the stack, which is this project's public cell at the same centre-origin points once
``x`` and ``y`` are swapped. RETICOLO's incident wave is unit at its coordinate origin, the cell
centre here, which is where this project puts it too, so there is no phase to apply either. The
one adjustment is ``z``: ``res3`` repeats each interface once per layer, and the row a layer
owns at its *top* face is the one this project assigns to the layer above, so that row moves
inward by 1e-9 nm. ``einc = [a_p, a_s]`` is ``polarization=(a_s, a_p)``.
"""
import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path

import numpy as np
import pytest

from tests.references._reticolo_2d_field_cases import (
    ANGLES, EINC, LAYERS, MAX_ORDER, NAMES, NPTS, N_BOT, N_TOP, OFF_GRID_ANGLE, OFF_GRID_LAYERS, PERIOD,
    WL, blocks, key, off_grid_blocks, raster, samples, shapes, thickness)
from wavebit import Material, Units
from wavebit import rcwa as rcwa_api
from wavebit.rcwa._factory import rcwa
from wavebit.units import Z0

GOLDEN = np.load(repository_path('tests/references/golden_reticolo_2d_field.npz'))

# Measured over all eight comparisons: the worst gap is 3.9e-11, on every component alike. This
# is round-off room above that, not slack.
TOL = 1e-10

# res3 lays out NPTS rows per profile entry, ambients included: bottom ambient, the layers bottom
# up, top ambient. The ambients have zero thickness, so only the layers' rows are kept.
_LAYER_ROWS = np.arange(NPTS, NPTS * (len(LAYERS) + 1))
_TOP_FACES = np.arange(NPTS - 1, NPTS * len(LAYERS), NPTS)      # within the kept rows

CASES = [(angle, pol) for angle in ANGLES for pol in EINC]


def _z():
    z = GOLDEN['z'][_LAYER_ROWS].copy()
    z[_TOP_FACES] -= 1e-9
    return z


def _reticolo(champ, angle, pol):
    """RETICOLO's cell in this project's layout, ``(z, y, x, 6)``."""
    return np.transpose(GOLDEN[f'champ{champ}__{key(angle, pol)}'][_LAYER_ROWS], (0, 2, 1, 3))


_SOLVED = {}


def _solver(angle, pol):
    if (angle, pol) not in _SOLVED:
        a_p, a_s = EINC[pol]
        s = rcwa(backend='numpy', top=N_TOP ** 2, bot=N_BOT ** 2, period=list(PERIOD),
                 truncation_order=[MAX_ORDER] * 2, wavelength=WL, thickness=thickness(),
                 structure=raster(), theta=np.deg2rad(angle[0]), phi=np.deg2rad(angle[1]),
                 polarization=(a_s, a_p), fourier_coefficients='integrated', factorization='li',
                 answers='field')
        s.solve()
        _SOLVED[angle, pol] = s
    return _SOLVED[angle, pol]


def _ours(angle, pol, rule):
    x, y = samples()
    return np.asarray(_solver(angle, pol).reconstruct_field(
        sample_x=x, sample_y=y, sample_z=_z(), discontinuous=rule))[0]


def _gap(ours, theirs, slot):
    return np.abs(ours[..., slot] - theirs[..., slot]).max() / np.abs(theirs[..., slot]).max()


def test_the_layout_is_the_one_assumed():
    """The rows kept are the layers', bottom up, and the raster is the texture res3 saw."""
    assert np.allclose(GOLDEN['z'][_LAYER_ROWS[[0, -1]]], [0.0, sum(thickness())])
    index = np.transpose(GOLDEN['o'][_LAYER_ROWS], (0, 2, 1))                  # (z, y, x)
    x, y = samples()
    cols = np.floor((x / PERIOD[0] + 0.5) * raster().shape[2]).astype(int)
    rows = np.floor((y / PERIOD[1] + 0.5) * raster().shape[1]).astype(int)
    for z_row, layer in zip(index, np.repeat(np.arange(len(LAYERS)), NPTS)):
        assert np.allclose(z_row ** 2, raster()[layer][np.ix_(rows, cols)])


@pytest.mark.parametrize('angle,pol', CASES)
def test_the_continuity_rule_is_reticolo_champ_1_on_every_slot(angle, pol):
    ours, theirs = _ours(angle, pol, 'continuity'), _reticolo(1, angle, pol)
    gaps = {name: _gap(ours, theirs, slot) for slot, name in enumerate(NAMES)}
    assert max(gaps.values()) < TOL, gaps


@pytest.mark.parametrize('angle,pol', CASES)
def test_the_fourier_sum_is_reticolo_champ_0_on_the_transverse_slots(angle, pol):
    ours, theirs = _ours(angle, pol, 'fourier'), _reticolo(0, angle, pol)
    gaps = {NAMES[slot]: _gap(ours, theirs, slot) for slot in (0, 1, 3, 4)}
    assert max(gaps.values()) < TOL, gaps
    # What the comparison leaves out, and why: RETICOLO does not fill these under champ = 0.
    assert not np.any(theirs[..., [2, 5]])


@pytest.mark.parametrize('angle,pol', CASES)
def test_each_reference_tells_the_two_rules_apart(angle, pol):
    """The rules differ in ``E_x`` and ``E_y`` only, and by far more than the tolerance."""
    plain, continuity = _ours(angle, pol, 'fourier'), _ours(angle, pol, 'continuity')
    np.testing.assert_array_equal(plain[..., 2:], continuity[..., 2:])
    for slot in (0, 1):
        assert _gap(plain, continuity, slot) > 1e-2
        assert _gap(plain, _reticolo(1, angle, pol), slot) > 1e-2


# ------------------------------------------------------------------------- the same on shapes

GOLDEN_OFF_GRID = np.load(repository_path('tests/references/golden_reticolo_2d_field_off_grid.npz'))


def _on_shapes(layers, rows, angle, pol):
    a_p, a_s = EINC[pol]
    s = rcwa(backend='numpy', top=N_TOP ** 2, bot=N_BOT ** 2, period=list(PERIOD),
             truncation_order=[MAX_ORDER] * 2, wavelength=WL,
             thickness=[layer['thickness'] for layer in layers[::-1]],
             structure=shapes(layers, rows), theta=np.deg2rad(angle[0]), phi=np.deg2rad(angle[1]),
             polarization=(a_s, a_p), factorization='li', answers='field')
    s.solve()
    return s


@pytest.mark.parametrize('angle,pol', CASES)
def test_the_rule_on_shapes_is_champ_1_where_a_raster_draws_the_same_walls(angle, pol):
    x, y = samples()
    ours = np.asarray(_on_shapes(LAYERS, blocks, angle, pol).reconstruct_field(
        sample_x=x, sample_y=y, sample_z=_z(), discontinuous='continuity'))[0]
    gaps = {name: _gap(ours, _reticolo(1, angle, pol), slot) for slot, name in enumerate(NAMES)}
    assert max(gaps.values()) < TOL, gaps


@pytest.mark.parametrize('pol', list(EINC))
def test_the_rule_on_shapes_is_champ_1_on_walls_no_raster_has(pol):
    """No pixel line holds these walls, so this reads the rule off the shapes alone."""
    x, y = samples()
    z = GOLDEN_OFF_GRID['z'][_LAYER_ROWS].copy()
    z[_TOP_FACES] -= 1e-9
    solver = _on_shapes(OFF_GRID_LAYERS, off_grid_blocks, OFF_GRID_ANGLE, pol)
    rule = np.asarray(solver.reconstruct_field(x, y, z, discontinuous='continuity'))[0]
    plain = np.asarray(solver.reconstruct_field(x, y, z))[0]
    case = key(OFF_GRID_ANGLE, pol)
    champ1 = np.transpose(GOLDEN_OFF_GRID[f'champ1__{case}'][_LAYER_ROWS], (0, 2, 1, 3))
    champ0 = np.transpose(GOLDEN_OFF_GRID[f'champ0__{case}'][_LAYER_ROWS], (0, 2, 1, 3))
    gaps = {name: _gap(rule, champ1, slot) for slot, name in enumerate(NAMES)}
    assert max(gaps.values()) < TOL, gaps
    gaps = {NAMES[slot]: _gap(plain, champ0, slot) for slot in (0, 1, 3, 4)}
    assert max(gaps.values()) < TOL, gaps
    assert _gap(plain, champ1, 0) > 0.1
