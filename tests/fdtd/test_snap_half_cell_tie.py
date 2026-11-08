"""A cell-snapped face exactly half way between two cell boundaries: one rule, every spelling.

Before this rule a face on a half cell was snapped with Python's ``round`` (half to even),
measured from the grid padded with the absorbing cells. Which neighbour a tie went to then
followed the parity of the PML cell count and the floating-point spelling of the coordinate:
``SnappedSlab(z=(-0.1125, 0.1125))`` at ``dx = 0.025`` in a 1.6-long domain ran as 8 cells with
``PML(cells=12)`` and as 10 cells with ``PML(cells=13)`` (the probe's Ey differed by 19 % of its
peak), and the one-cell ``SnappedSlab(z=(-0.0875, -0.0625))`` was one cell in um, two cells in nm,
and refused as empty with 13 PML cells.

The references here do not come from the snapping code. The physical scene is the same in um
and nm and with any number of PML cells, so the snapped cells must be too; the documented rule
(nearest boundary, a tie goes to the higher one) gives the expected boundaries by hand; and a
tie scene must run bit for bit like a scene whose faces are not at a tie but lie in the same
cells.
"""
import math
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit._common.conventions import snap_to_cell_boundary

DX = .025            # um; the 1.6 um domain's boundaries fall on multiples of DX
SCALE = {'um': 1.0, 'nm': 1000.0}


def _problem(unit, shape_um):
    s = SCALE[unit]
    if isinstance(shape_um, dict):
        shape = fdtd.SnappedBox(**{k: tuple(v * s for v in shape_um[k]) for k in ('size', 'center')})
    else:
        shape = fdtd.SnappedSlab(z=(shape_um[0] * s, shape_um[1] * s))
    return fdtd.Problem(units=Units(length=unit), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1 * s, .1 * s, 1.6 * s)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(fdtd.Structure('film', shape=shape, material=Material(epsilon=2.25)),)),
        sources=(fdtd.PlaneWave('src', plane=fdtd.Plane(z=.5 * s),
                 pulse=fdtd.GaussianPulse(wavelength=.8 * s, frequency_sigma_hz=2e13)),),
        outputs=fdtd.Outputs(monitors=(
            fdtd.TimeSeriesMonitor('probe', at=(0., 0., -.3 * s), component='Ey'),)))


def _numerics(unit, cells, steps=400):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX * SCALE[unit]),
                         duration=fdtd.Duration(steps=steps), absorber=fdtd.PML(cells=cells))


def _snapped(unit, cells, shape_um):
    """The z span ``check`` reports the film to run at, in um."""
    report = fdtd.check(_problem(unit, shape_um), numerics=_numerics(unit, cells))
    assert report.ok, [i.message for i in report.issues]
    rows = [issue.actual for issue in report.issues if issue.code == 'geometry.cell_snapped']
    assert len(rows) == 1
    (z,) = [row for row in rows[0] if row[0] in ('z', 'box.z')]
    return tuple(round(v / SCALE[unit] / DX) for v in z[1:])     # boundaries in cells from 0


SPELLINGS = [(unit, cells) for unit in ('um', 'nm') for cells in (12, 13)]


@pytest.mark.parametrize('unit,cells', SPELLINGS)
def test_a_slab_with_both_faces_on_half_cells_snaps_up_in_every_spelling(unit, cells):
    # -0.1125 is -4.5 cells, 0.1125 is 4.5: both go up, to -4 and 5 (main ran -4..4 or -5..5).
    assert _snapped(unit, cells, (-.1125, .1125)) == (-4, 5)


@pytest.mark.parametrize('unit,cells', SPELLINGS)
def test_a_one_cell_slab_on_half_cells_is_one_cell_in_every_spelling(unit, cells):
    # -3.5 .. -2.5 cells; main ran it as one cell, two cells, or refused it as empty.
    assert _snapped(unit, cells, (-.0875, -.0625)) == (-3, -2)


@pytest.mark.parametrize('unit,cells', SPELLINGS)
def test_a_snapped_box_face_on_a_half_cell_snaps_up_in_every_spelling(unit, cells):
    # A box clipped by the periodic x/y faces, its z faces at -2.5 and 1.5 cells.
    assert _snapped(unit, cells, dict(size=(.2, .2, .1), center=(0., 0., -.0125))) == (-2, 2)


def test_a_tie_runs_bit_for_bit_like_a_non_tie_in_the_same_cells_and_unit_and_pml_do_not_matter():
    """The tie film against faces well inside the same cells (-0.105 -> -0.1, 0.12 -> 0.125)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        runs = {}
        for unit in ('um', 'nm'):
            for cells in (12, 13):
                n = _numerics(unit, cells)
                tie = fdtd.run(_problem(unit, (-.1125, .1125)), numerics=n)['probe'].values
                inside = fdtd.run(_problem(unit, (-.105, .12)), numerics=n)['probe'].values
                np.testing.assert_array_equal(np.asarray(tie), np.asarray(inside))
                runs[unit, cells] = np.asarray(tie)
    peak = np.max(np.abs(runs['um', 12]))
    assert peak > 0
    for cells in (12, 13):      # the same scene in two units: the same run, to rounding
        assert np.max(np.abs(runs['nm', cells] - runs['um', cells])) <= 1e-12 * peak


def test_the_rule_ignores_a_whole_cell_shift_of_the_origin_and_representation_error():
    for k in range(-40, 41):
        for shift in (0, 1, 12, 13):
            origin = -0.8 - shift * DX
            # A half cell, nudged by a few ulp either way, always goes to the higher boundary.
            x = -0.8 + (k + .5) * DX
            for nudged in (x, math.nextafter(x, -math.inf), math.nextafter(x, math.inf),
                           math.nextafter(math.nextafter(x, -math.inf), -math.inf)):
                assert snap_to_cell_boundary(nudged, origin, DX) == k + 1 + shift
            # Away from a tie it is the nearest boundary.
            assert snap_to_cell_boundary(-0.8 + (k + .3) * DX, origin, DX) == k + shift
            assert snap_to_cell_boundary(-0.8 + (k + .7) * DX, origin, DX) == k + 1 + shift
    # The same half cell written in nm.
    assert snap_to_cell_boundary(-112.5, -800.0, 25.0) == snap_to_cell_boundary(-.1125, -.8, .025) == 28
