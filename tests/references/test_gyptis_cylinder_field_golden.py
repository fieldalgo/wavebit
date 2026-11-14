"""The near field of a true cylinder against a 3D finite-element field.

``golden_gyptis_cylinder_field.npz`` holds gyptis's ``E`` on 2304 samples of the mid-height plane
of a cylinder (radius 100 nm, in a 150 nm layer of a 400 nm cell, normal incidence at 900 nm with
``E`` along x), once as a dielectric, ``eps`` 12.25 (``E``), and once as a metal, ``eps``
-20+1.5j (``E_metal``); its record names the runs. The two sides discretize differently, a mesh
against a Fourier truncation, so this is an oracle at the percent level: it separates a
reconstruction that follows the wall from one that rings at it. The finite-element fields' own
next mesh level differs from them by 0.11% and 0.15% of the plane RMS.

Measured here at order 8 with the NVF solve: the plain sum is 13.6% of the plane RMS away,
``reconstruction='split'`` 3.1% and ``'splitw'`` 2.4%. At order 16 the same three are 9.6%, 1.4%
and 1.3%, which a regular run does not afford.

On the metal the solve itself is the limit: at order 8 its ``R`` is 0.1840 where the reference has
0.1880, and the sum is 27.4% away. The split rules answer inside the metal only and leave the
sum outside it; that is 21.7% over the plane, and inside the metal 11.6% against the sum's 39.4%.
"""
from pathlib import Path

import numpy as np
import pytest

from wavebit import Material, Units, rcwa

GOLDEN = Path(__file__).with_name('golden_gyptis_cylinder_field.npz')
PERIOD, RADIUS, THICKNESS, WAVELENGTH, EPSILON, METAL = 400.0, 100.0, 150.0, 900.0, 12.25, -20 + 1.5j


@pytest.fixture(scope='module')
def reference():
    """The samples and the finite-element field in Wavebit's convention, ``(n_y, n_x, 3)``."""
    return _reference('E')


def _reference(key):
    with np.load(GOLDEN) as golden:
        x, y, z, field = golden['x'], golden['y'], float(golden['z']), golden[key]
    # gyptis's phasors are the conjugate of Wavebit's, and its incident wave is referenced to the
    # other face of the layer.
    return x, y, z, -np.exp(2j * np.pi * THICKNESS / WAVELENGTH) * np.conj(field)


def _solve(epsilon):
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(PERIOD, PERIOD), layers=[rcwa.Layer(thickness=THICKNESS, regions=[
            rcwa.Region(geometry=rcwa.Circle(radius=RADIUS), material=Material(epsilon=epsilon))])]),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH,
                                   sources=[rcwa.PlaneWave(side='top', polarization='p')]),
        outputs=rcwa.Outputs(fields=True))
    return rcwa.run(problem, numerics=rcwa.Numerics(orders=(8, 8), factorization='nvf'))


@pytest.fixture(scope='module')
def metal_reference():
    return _reference('E_metal')


@pytest.fixture(scope='module')
def metal_result():
    return _solve(METAL)


@pytest.fixture(scope='module')
def result():
    return _solve(EPSILON)


def _plane_error(result, reference, rule, where=None):
    """RMS of the in-plane vector error over the plane (or ``where``), over the reference's own plane RMS."""
    x, y, z, field = reference
    values = np.asarray(result.field(x=x, y=y, z=[z], reconstruction=rule).values)[0, 0]
    scale = np.sqrt(np.mean(np.sum(np.abs(field[..., :2]) ** 2, axis=-1)))
    error = np.sum(np.abs(values[..., :2] - field[..., :2]) ** 2, axis=-1)
    return np.sqrt(np.mean(error if where is None else error[where])) / scale


def test_the_solve_agrees_with_the_reference_on_reflection(result):
    # The record's settings give the finite-element R; the field comparison means nothing
    # where the two solves disagree on the scattering itself.
    assert float(np.sum(np.asarray(result.R))) == pytest.approx(0.024309, abs=2e-4)


def test_the_plain_sum_rings_at_the_curved_wall(result, reference):
    assert _plane_error(result, reference, 'plain') == pytest.approx(0.1364, abs=1e-3)


def test_splitw_follows_the_curved_wall(result, reference):
    assert _plane_error(result, reference, 'splitw') == pytest.approx(0.0235, abs=1e-3)


def test_split_follows_the_curved_wall(result, reference):
    assert _plane_error(result, reference, 'split') == pytest.approx(0.0311, abs=1e-3)


def test_both_split_rules_are_closer_than_the_sum_near_the_wall_and_away_from_it(result, reference):
    """Measured, as RMS over each zone: within a tenth of the period of the wall the sum is 0.244 off,
    ``'split'`` 0.041 and ``'splitw'`` 0.032; further away 0.041, 0.027 and 0.020. Away from the wall
    is where the unwindowed normal costs ``'split'`` and the window pays."""
    x, y, z, field = reference
    distance = np.abs(np.hypot(*np.meshgrid(x, y)) - RADIUS)

    def error(rule, where):
        values = np.asarray(result.field(x=x, y=y, z=[z], reconstruction=rule).values)[0, 0]
        return np.sqrt(np.mean(np.sum(np.abs(values[..., :2] - field[..., :2])[where] ** 2, axis=-1)))

    near, far = distance < 0.1 * PERIOD, distance > 0.1 * PERIOD
    assert error('split', near) < 0.2 * error('plain', near)
    assert error('splitw', near) < 0.2 * error('plain', near)
    assert error('split', far) < 0.7 * error('plain', far)
    assert error('splitw', far) < 0.5 * error('plain', far)
    assert error('splitw', near) < error('split', near) and error('splitw', far) < error('split', far)


def test_the_metal_solve_is_what_limits_the_field(metal_result):
    """Its R is 0.1840 at this order against the reference's 0.1880: 2% off, where the dielectric's is 0.3%."""
    assert float(np.sum(np.asarray(metal_result.R))) == pytest.approx(0.1840, abs=5e-4)


@pytest.mark.parametrize('rule, plane, inside', [('split', 0.2157, 0.1122), ('splitw', 0.2169, 0.1160)])
def test_on_a_metal_a_split_rule_answers_inside_it_and_is_closer_there(metal_result, metal_reference, rule, plane, inside):
    x, y, _, _ = metal_reference
    within = np.hypot(*np.meshgrid(x, y)) < RADIUS
    assert _plane_error(metal_result, metal_reference, 'plain') == pytest.approx(0.2740, abs=1e-3)
    assert _plane_error(metal_result, metal_reference, 'plain', within) == pytest.approx(0.3942, abs=1e-3)
    assert _plane_error(metal_result, metal_reference, rule) == pytest.approx(plane, abs=1e-3)
    assert _plane_error(metal_result, metal_reference, rule, within) == pytest.approx(inside, abs=1e-3)
    # Outside the metal the rule hands back the sum, so the two errors there are one number.
    assert _plane_error(metal_result, metal_reference, rule, ~within) == pytest.approx(
        _plane_error(metal_result, metal_reference, 'plain', ~within), rel=1e-9)
