"""The near field of a circle and an ellipse in matched coordinates against 3D finite-element fields.

``Numerics(wall_resolution=0.02)`` on one circle or axis-aligned ellipse solves in coordinates matched to
it (``tests/rcwa/test_wall_resolution_matched.py`` pins the construction). The references are the ones
the split reconstructions are measured against, read here a second time: ``golden_gyptis_cylinder_field``
(a cylinder of radius 100 nm in a 150 nm layer of a 400 nm cell, normal incidence at 900 nm with ``E``
along x, as a metal of ``eps`` -20+1.5j and as a dielectric of ``eps`` 12.25, mesh level T4) and the
ellipse of ``golden_gyptis_shapes_field`` (semi-axes 130 and 80 nm, ``eps`` 12.25, mesh level T2). Their
records name the runs; their own next mesh levels are 0.15%, 0.11% and 0.38% of the plane RMS away.

Measured here, as the RMS of the in-plane vector error over 2304 samples of the mid-height plane, wall
samples included, over the reference's plane RMS: the product's default solve (NVF, ``'plain'``) and
then ``factorization='li', wall_resolution=0.02``. See ``MEASURED`` below for the numbers and
docs/validation/rcwa/wall-resolution.md for the other nine cells, whose references are not stored.
"""
from pathlib import Path

import numpy as np
import pytest

from wavebit import Material, Units, rcwa

HERE = Path(__file__).parent
# wall_resolution is experimental and warns at each solve; tests/rcwa/test_wall_resolution.py pins that.
pytestmark = pytest.mark.filterwarnings('ignore::wavebit.rcwa.diagnostics.ExperimentalWarning')
PERIOD, THICKNESS, WAVELENGTH, METAL = 400.0, 150.0, 900.0, -20 + 1.5j
#: file, array, shape, permittivity
CELLS = {
    'metal_cylinder': ('golden_gyptis_cylinder_field.npz', 'E_metal', rcwa.Circle(radius=100.0), METAL),
    'dielectric_cylinder': ('golden_gyptis_cylinder_field.npz', 'E', rcwa.Circle(radius=100.0), 12.25),
    'dielectric_ellipse': ('golden_gyptis_shapes_field.npz', 'E_ellipse', rcwa.Ellipse(radii=(130.0, 80.0)), 12.25),
}
#: the reference's R; then per order (8, 12): R of the matched solve, plane errors of NVF and matched
MEASURED = {
    'metal_cylinder': (0.18799, {8: (0.18804, 0.2740, 0.0212), 12: (0.18792, 0.2147, 0.0042)}),
    'dielectric_cylinder': (0.024309, {8: (0.02429, 0.1364, 0.0024), 12: (0.02428, 0.1116, 0.0014)}),
    'dielectric_ellipse': (0.052820, {8: (0.05278, 0.1546, 0.0037), 12: (0.05277, 0.1280, 0.0025)}),
}


def _reference(key):
    name, array = CELLS[key][:2]
    with np.load(HERE / name) as golden:
        x, y, z, field = golden['x'], golden['y'], float(golden['z']), golden[array]
    # gyptis's phasors are the conjugate of Wavebit's, and its incident wave is referenced to the
    # other face of the layer.
    return x, y, z, -np.exp(2j * np.pi * THICKNESS / WAVELENGTH) * np.conj(field)


def _solve(key, order, matched):
    _, _, geometry, epsilon = CELLS[key]
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(PERIOD, PERIOD), layers=[rcwa.Layer(thickness=THICKNESS, regions=[
            rcwa.Region(geometry=geometry, material=Material(epsilon=epsilon))])]),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH, sources=[rcwa.PlaneWave(side='top', polarization='p')]),
        outputs=rcwa.Outputs(fields=True))
    numerics = (rcwa.Numerics(orders=(order, order), factorization='li', wall_resolution=0.02) if matched
                else rcwa.Numerics(orders=(order, order), factorization='nvf'))
    return rcwa.run(problem, numerics=numerics)


def _plane_error(result, reference):
    """RMS of the in-plane vector error over the plane, over the reference's own plane RMS."""
    x, y, z, field = reference
    values = np.asarray(result.field(x=x, y=y, z=[z]).values)[0, 0]
    scale = np.sqrt(np.mean(np.sum(np.abs(field[..., :2]) ** 2, axis=-1)))
    return np.sqrt(np.mean(np.sum(np.abs(values[..., :2] - field[..., :2]) ** 2, axis=-1))) / scale


def _check(key, order):
    reference = _reference(key)
    default, matched = _solve(key, order, False), _solve(key, order, True)
    expected, by_order = MEASURED[key]
    solved, default_error, matched_error = by_order[order]
    R = float(np.sum(np.asarray(matched.R)))
    errors = _plane_error(default, reference), _plane_error(matched, reference)
    assert R == pytest.approx(solved, abs=2e-5)
    assert errors[0] == pytest.approx(default_error, abs=1e-3)
    assert errors[1] == pytest.approx(matched_error, abs=5e-4)
    return R, expected, errors


@pytest.mark.parametrize('key', sorted(CELLS))
def test_at_order_8_the_matched_solve_is_where_it_was_measured(key):
    R, expected, errors = _check(key, 8)
    assert R == pytest.approx(expected, rel=3e-3)
    assert errors[1] < 0.1 * errors[0]


#: The criteria written down before the implementation (design section 8.3), at order 12: the plane
#: error of the matched solve as a fraction of the default's on a metal, its absolute value on a
#: dielectric, and R against the reference's.
@pytest.mark.slow
@pytest.mark.parametrize('key', sorted(CELLS))
def test_at_order_12_the_criteria_set_before_the_implementation_hold(key):
    R, expected, errors = _check(key, 12)
    assert R == pytest.approx(expected, rel=3e-3)
    if key == 'metal_cylinder':
        assert errors[1] < 0.1 * errors[0]
    else:
        assert errors[1] < 5e-3
