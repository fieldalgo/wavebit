"""The near field of a true cylinder against a 3D finite-element field.

``golden_gyptis_cylinder_field.npz`` holds gyptis's ``E`` on 2304 samples of the mid-height plane
of a dielectric cylinder (radius 100 nm, ``eps`` 12.25, in a 150 nm layer of a 400 nm cell, normal
incidence at 900 nm with ``E`` along x); its record names the run. The two sides discretize
differently, a mesh against a Fourier truncation, so this is an oracle at the percent level: it
separates a reconstruction that follows the wall from one that rings at it. The finite-element
field's own next mesh level differs from it by 0.11% of the plane RMS.

Measured here at order 8 with the NVF solve: the plain sum is 13.6% of the plane RMS away, and
``reconstruction='splitw'`` 2.4%. At order 16 the same two are 9.6% and 1.3%, which a regular
run does not afford.
"""
from pathlib import Path

import numpy as np
import pytest

from wavebit import Material, Units, rcwa

GOLDEN = Path(__file__).with_name('golden_gyptis_cylinder_field.npz')
PERIOD, RADIUS, THICKNESS, WAVELENGTH, EPSILON = 400.0, 100.0, 150.0, 900.0, 12.25


@pytest.fixture(scope='module')
def reference():
    """The samples and the finite-element field in Wavebit's convention, ``(n_y, n_x, 3)``."""
    with np.load(GOLDEN) as golden:
        x, y, z, field = golden['x'], golden['y'], float(golden['z']), golden['E']
    # gyptis's phasors are the conjugate of Wavebit's, and its incident wave is referenced to the
    # other face of the layer.
    return x, y, z, -np.exp(2j * np.pi * THICKNESS / WAVELENGTH) * np.conj(field)


@pytest.fixture(scope='module')
def result():
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(PERIOD, PERIOD), layers=[rcwa.Layer(thickness=THICKNESS, regions=[
            rcwa.Region(geometry=rcwa.Circle(radius=RADIUS), material=Material(epsilon=EPSILON))])]),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH,
                                   sources=[rcwa.PlaneWave(side='top', polarization='p')]),
        outputs=rcwa.Outputs(fields=True))
    return rcwa.run(problem, numerics=rcwa.Numerics(orders=(8, 8), factorization='nvf'))


def _plane_error(result, reference, rule):
    """RMS of the in-plane vector error over the plane, over the reference's own plane RMS."""
    x, y, z, field = reference
    values = np.asarray(result.field(x=x, y=y, z=[z], reconstruction=rule).values)[0, 0]
    scale = np.sqrt(np.mean(np.sum(np.abs(field[..., :2]) ** 2, axis=-1)))
    return np.sqrt(np.mean(np.sum(np.abs(values[..., :2] - field[..., :2]) ** 2, axis=-1))) / scale


def test_the_solve_agrees_with_the_reference_on_reflection(result):
    # The record's settings give the finite-element R; the field comparison means nothing
    # where the two solves disagree on the scattering itself.
    assert float(np.sum(np.asarray(result.R))) == pytest.approx(0.024309, abs=2e-4)


def test_the_plain_sum_rings_at_the_curved_wall(result, reference):
    assert _plane_error(result, reference, 'plain') == pytest.approx(0.1364, abs=1e-3)


def test_splitw_follows_the_curved_wall(result, reference):
    assert _plane_error(result, reference, 'splitw') == pytest.approx(0.0235, abs=1e-3)


def test_splitw_is_closer_than_the_sum_near_the_wall_and_away_from_it(result, reference):
    x, y, z, field = reference
    distance = np.abs(np.hypot(*np.meshgrid(x, y)) - RADIUS)

    def error(rule, where):
        values = np.asarray(result.field(x=x, y=y, z=[z], reconstruction=rule).values)[0, 0]
        return np.sqrt(np.mean(np.sum(np.abs(values[..., :2] - field[..., :2])[where] ** 2, axis=-1)))

    for where in (distance < 0.1 * PERIOD, distance > 0.1 * PERIOD):
        assert error('splitw', where) < 0.5 * error('plain', where)
