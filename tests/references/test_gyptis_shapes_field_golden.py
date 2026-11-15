"""The near field of an ellipse and of two circles against 3D finite-element fields.

``golden_gyptis_shapes_field.npz`` holds gyptis's ``E`` on 2304 samples of the mid-height plane of
a 150 nm layer in a 400 nm cell, normal incidence at 900 nm with ``E`` along x, for two layers of
``eps`` 12.25 in air: one ellipse of semi-axes 130 and 80 nm (``E_ellipse``) and two circles of
radius 60 nm, 180 nm apart (``E_two_circles``). Its record names the runs. The two sides
discretize differently, a mesh against a Fourier truncation, so this is an oracle at the percent
level, as the cylinder's is. The finite-element fields' own coarser mesh level differs from them
by 0.38% of the plane RMS on both.

Measured here at order 8 with the NVF solve, wall samples included: on the ellipse the plain sum
is 15.5% of the plane RMS away, ``reconstruction='split'`` 5.0% and ``'splitw'`` 3.2%; on the two
circles 17.2%, 4.8% and 4.2%. At order 16, which a regular run does not afford, the ellipse reads
11.3%, 2.1% and 1.6% and the two circles 11.4%, 2.7% and 2.3%.
"""
from pathlib import Path

import numpy as np
import pytest

from wavebit import Material, Units, rcwa

GOLDEN = Path(__file__).with_name('golden_gyptis_shapes_field.npz')
PERIOD, THICKNESS, WAVELENGTH, EPSILON = 400.0, 150.0, 900.0, 12.25
SHAPES = {
    'E_ellipse': lambda material: [rcwa.Region(geometry=rcwa.Ellipse(radii=(130.0, 80.0)), material=material)],
    'E_two_circles': lambda material: [rcwa.Region(geometry=rcwa.Circle(radius=60.0, center=(centre, 0.0)),
                                                   material=material) for centre in (-90.0, 90.0)],
}
#: Signed distance to the nearest wall at the plane's samples, negative inside a shape; the ellipse's
#: is the first-order one, which is enough to name the samples near its wall.
WALLS = {
    'E_ellipse': lambda X, Y: (np.hypot(X / 130.0, Y / 80.0) - 1.0) / np.hypot(X / 130.0 ** 2, Y / 80.0 ** 2)
    * np.hypot(X / 130.0, Y / 80.0),
    'E_two_circles': lambda X, Y: np.minimum(np.hypot(X + 90.0, Y), np.hypot(X - 90.0, Y)) - 60.0,
}


def _reference(key):
    with np.load(GOLDEN) as golden:
        x, y, z, field = golden['x'], golden['y'], float(golden['z']), golden[key]
    # gyptis's phasors are the conjugate of Wavebit's, and its incident wave is referenced to the
    # other face of the layer.
    return x, y, z, -np.exp(2j * np.pi * THICKNESS / WAVELENGTH) * np.conj(field)


@pytest.fixture(scope='module', params=sorted(SHAPES))
def case(request):
    key = request.param
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(PERIOD, PERIOD), layers=[rcwa.Layer(
            thickness=THICKNESS, regions=SHAPES[key](Material(epsilon=EPSILON)))]),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH,
                                   sources=[rcwa.PlaneWave(side='top', polarization='p')]),
        outputs=rcwa.Outputs(fields=True))
    return key, rcwa.run(problem, numerics=rcwa.Numerics(orders=(8, 8), factorization='nvf')), _reference(key)


def _plane_error(result, reference, rule, where=None):
    """RMS of the in-plane vector error over the plane (or ``where``), over the reference's own plane RMS."""
    x, y, z, field = reference
    values = np.asarray(result.field(x=x, y=y, z=[z], reconstruction=rule).values)[0, 0]
    scale = np.sqrt(np.mean(np.sum(np.abs(field[..., :2]) ** 2, axis=-1)))
    error = np.sum(np.abs(values[..., :2] - field[..., :2]) ** 2, axis=-1)
    return np.sqrt(np.mean(error if where is None else error[where])) / scale


MEASURED = {   # R of the solve and the reference's; plane errors of 'plain', 'split', 'splitw'
    'E_ellipse': (0.05284, 0.052820, 0.1546, 0.0496, 0.0320),
    'E_two_circles': (0.02228, 0.022449, 0.1720, 0.0480, 0.0423),
}


def test_the_solve_agrees_with_the_reference_on_reflection(case):
    key, result, _ = case
    solved, reference = MEASURED[key][:2]
    assert float(np.sum(np.asarray(result.R))) == pytest.approx(solved, abs=2e-5)
    # 0.04% on the ellipse and 0.8% on the two circles at this order (0.2% at order 16).
    assert solved == pytest.approx(reference, rel=1e-2)


@pytest.mark.parametrize('rule, column', [('plain', 2), ('split', 3), ('splitw', 4)])
def test_each_reconstruction_is_where_it_was_measured(case, rule, column):
    key, result, reference = case
    assert _plane_error(result, reference, rule) == pytest.approx(MEASURED[key][column], abs=1e-3)


def test_both_split_rules_are_closer_than_the_sum_at_the_walls_and_away_from_them(case):
    """Within 10 nm of a wall the sum is 48 to 51% of the plane RMS off and the rules 5 to 8%; beyond
    0.1 period outside, where the sum is 4.5 to 4.8% off, they are 2.9 to 4.5%."""
    key, result, reference = case
    x, y, _, _ = reference
    distance = WALLS[key](*np.meshgrid(x, y))
    at_wall, far = np.abs(distance) < 10.0, distance > 0.1 * PERIOD
    for rule in ('split', 'splitw'):
        assert _plane_error(result, reference, rule, at_wall) < 0.2 * _plane_error(result, reference, 'plain', at_wall)
        assert _plane_error(result, reference, rule, far) < _plane_error(result, reference, 'plain', far)
    assert _plane_error(result, reference, 'splitw') < _plane_error(result, reference, 'split')
