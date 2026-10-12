"""Gate 2 -- the PMC box eigenmode: the dual of gate 1, with the walls half a cell outside.

Same 16^3 box, same (1, 1, 1) mode, but the pattern is on H and the walls are where PMC
puts them on this layout: ``x_{-1/2}`` on the low face (the zero H pad is the wall) and
``x_{N-1/2}`` on the high face (the masked tangential H layer is the wall). The box is
still ``N dx`` long, so the frequency is gate 1's; what moves is the origin the mode is
written from, by half a cell -- and writing it from the PEC walls instead is the mutant here,
alongside the one-cell shift. Held on every backend to round-off, at an E probe and an H
probe, plus the closed form (``H(n+1/2) = H0 cos(w (n+1/2) dt)``, ``E(n) = e curl H0 sin(w n
dt)``).
"""
import numpy as np
import pytest

import wavebit.fd as fd
from fd_helpers import (backends, cavity_box, closed_form, discrete_omega, recurrence_residual,
                        to_numpy)

TOL = 1e-11
MUTANT_FLOOR = 1e-4


@pytest.fixture(scope='module')
def box():
    return cavity_box('pmc', N=16, steps=300)


def test_the_plan_records_the_pmc_walls_half_a_cell_outside(box):
    scene, mode, omega = box
    faces = fd.plan(scene).faces
    dx = scene.resolution.dx
    for axis in 'xyz':
        assert faces[axis + '-'].kind == 'pmc'
        assert faces[axis + '-'].position == pytest.approx(-0.5 - dx / 2)
        assert faces[axis + '+'].position == pytest.approx(0.5 - dx / 2)
    assert mode.w == pytest.approx((-0.5 - dx / 2,) * 3)


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('probe', ['Ey', 'Hz'])
def test_the_recurrence_holds_to_roundoff(box, backend, probe):
    scene, mode, omega = box
    res = fd.run(scene, backend=backend)
    values = to_numpy(res[probe].values)
    assert np.max(np.abs(values)) > 1e-3, 'the probe sits on a node of the mode'
    assert recurrence_residual(values, omega, res.plan.dt) < TOL


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('probe', ['Ey', 'Hz'])
def test_the_closed_form_holds_amplitude_and_phase(box, backend, probe):
    scene, mode, omega = box
    res = fd.run(scene, backend=backend)
    expected = closed_form('pmc', mode, omega, res.plan, res[probe])
    values = to_numpy(res[probe].values)
    assert np.max(np.abs(values - expected)) < 1e-10 * np.max(np.abs(expected))


def test_mutant_a_wall_one_cell_off_fails_the_recurrence(box):
    scene, mode, omega = box
    res = fd.run(scene)
    dx, dt = res.plan.dx, res.plan.dt
    shifted = discrete_omega((np.pi / (1.0 + dx), np.pi, np.pi), dx, dt)
    values = to_numpy(res['Hz'].values)
    assert recurrence_residual(values, omega, dt) < TOL
    assert recurrence_residual(values, shifted, dt) > MUTANT_FLOOR


def test_mutant_the_mode_written_from_the_pec_walls_is_not_a_pmc_mode():
    """Half a cell is the whole difference between the two boxes, and the gate sees it."""
    scene, mode, omega = cavity_box('pmc', N=16, steps=200)
    from fd_helpers import CavityMode
    dx = scene.resolution.dx
    wrong = CavityMode((-0.5,) * 3, (1.0,) * 3, (1, 1, 1), dx)      # PEC origin, PMC box
    cos_half = np.cos(omega * scene.run.courant * dx / np.sqrt(3) / 2)
    bad = scene.replace(initial=fd.InitialField(
        H=tuple((lambda c: (lambda x, y, z: cos_half * wrong.component(c)(x, y, z)))(c)
                for c in range(3))))
    res = fd.run(bad)
    assert recurrence_residual(to_numpy(res['Hz'].values), omega, res.plan.dt) > MUTANT_FLOOR
