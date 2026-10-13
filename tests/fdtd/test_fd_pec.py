"""Gate 1 -- the PEC box eigenmode: the kernel advances an exact Yee mode at the discrete frequency.

A 16^3 PEC box carries the (1, 1, 1) cavity mode written at the Yee points with the discrete
wavenumbers (``tests/fixtures/fdtd/fd_helpers.py`` derives it). Three things are held, on every backend:

* the three-term recurrence ``f(n+1) + f(n-1) = 2 cos(w dt) f(n)`` at an E probe and an H
  probe, with ``w`` from the discrete dispersion relation and the wall positions the plan
  records (PEC: on the E nodes, ``x_0`` and ``x_N``) -- to round-off;
* the closed form ``E(n) = E0 cos(w n dt)``, ``H(n+1/2) = h curl E0 sin(w (n+1/2) dt)``, which
  pins the phase and the amplitude and not only the frequency;
* the mutant: the same recurrence with a wall shifted by one cell -- ``k = pi/(L + dx)`` --
  fails by orders of magnitude, so the tolerance is doing work.

The tolerance is 1e-11 on the recurrence residual relative to the series' peak: a
round-off budget for 300 leapfrog steps in float64. A wall one cell off costs ~1e-2.
"""
import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from tests.fixtures.fdtd.fd_helpers import backends, cavity_box, closed_form, discrete_omega, recurrence_residual, to_numpy

TOL = 1e-11
MUTANT_FLOOR = 1e-4


@pytest.fixture(scope='module')
def box():
    return cavity_box('pec', N=16, steps=300)


def test_the_plan_records_the_pec_walls_on_the_nodes(box):
    scene, mode, omega = box
    faces = fd.plan(scene).faces
    dx = scene.resolution.dx
    for axis in 'xyz':
        assert faces[axis + '-'].kind == 'pec'
        assert faces[axis + '-'].position == pytest.approx(-0.5)
        assert faces[axis + '+'].position == pytest.approx(0.5)
    # and the mode was written from those walls
    assert mode.w == (-0.5, -0.5, -0.5)
    assert omega == pytest.approx(discrete_omega(mode.k, dx, fd.plan(scene).dt))


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
    expected = closed_form('pec', mode, omega, res.plan, res[probe])
    values = to_numpy(res[probe].values)
    assert np.max(np.abs(values - expected)) < 1e-10 * np.max(np.abs(expected))


def test_mutant_a_wall_one_cell_off_fails_the_recurrence(box):
    """The gate's sensitivity: ``k_x = pi/(L + dx)`` instead of ``pi/L`` must be rejected."""
    scene, mode, omega = box
    res = fd.run(scene)
    dx, dt = res.plan.dx, res.plan.dt
    L = 1.0
    shifted = discrete_omega((np.pi / (L + dx), np.pi / L, np.pi / L), dx, dt)
    values = to_numpy(res['Ey'].values)
    assert recurrence_residual(values, omega, dt) < TOL
    assert recurrence_residual(values, shifted, dt) > MUTANT_FLOOR


def test_the_final_fields_are_the_mode_at_the_last_step(box):
    scene, mode, omega = box
    res = fd.run(scene)
    plan = res.plan
    t_E = plan.steps * plan.dt
    E = to_numpy(res.final.E)
    for c in range(3):
        X, Y, Z = plan.grid.meshgrid('E', c)
        expected = mode.component(c)(X, Y, Z) * np.cos(omega * t_E)
        assert np.max(np.abs(E[c] - expected)) < 1e-10
