"""The near field beside axis-aligned walls against 3D finite-element fields, with and without the stretch.

``golden_gyptis_rectangles_field.npz`` holds gyptis's ``E`` on 2304 samples of the mid-height plane of a
150 nm layer in a 400 nm cell, normal incidence at 900 nm with ``E`` along x, for four layers: a 200 nm
metal block (``eps`` -20+1.5j) in air, a 200 nm air hole in a film of that metal, a 200 nm dielectric
square (``eps`` 12.25) and a 250 by 150 nm metal rectangle. Its record names the runs. The two sides
discretize differently, a mesh against a Fourier truncation, so this is an oracle at the percent level:
the block's and the hole's coarser mesh levels are 0.37% and 0.20% of the plane RMS from the stored ones,
and the square and the rectangle have one level.

Measured here with ``factorization='li'``, wall samples included, as the RMS of the in-plane vector error
over the reference's plane RMS, unstretched and then with ``Numerics(wall_resolution=0.02)``:

=================  =======================  =========================
layer              order 8                  order 12
=================  =======================  =========================
metal block        32.1% -> 4.0%            25.8% -> 0.63%
hole in a metal    36.8% -> 6.5%            30.9% -> 3.3%
dielectric square  14.8% -> 0.71%           11.5% -> 0.41%
metal rectangle    37.9% -> 4.5%            36.4% -> 3.6%
=================  =======================  =========================

At order 16, which a regular run does not afford, the stretched solve reads 0.52%, 1.8%, 0.39% and 3.4%.
The square and the rectangle stop at 0.4% and 3.4% with the solve's ``R`` settled to five digits, which is
taken as those two references' own accuracy.
"""
from pathlib import Path

import numpy as np
import pytest

from wavebit import Material, Units, rcwa

GOLDEN = Path(__file__).with_name('golden_gyptis_rectangles_field.npz')
# wall_resolution is experimental and warns at each solve; tests/rcwa/test_wall_resolution.py pins that.
pytestmark = pytest.mark.filterwarnings('ignore::wavebit.rcwa.diagnostics.ExperimentalWarning')
PERIOD, THICKNESS, WAVELENGTH, METAL = 400.0, 150.0, 900.0, -20 + 1.5j
#: size of the rectangle (nm), its permittivity, the layer's around it
LAYERS = {
    'E_metal_block': ((200.0, 200.0), METAL, 1.0),
    'E_hole': ((200.0, 200.0), 1.0, METAL),
    'E_dielectric_square': ((200.0, 200.0), 12.25, 1.0),
    'E_metal_rectangle': ((250.0, 150.0), METAL, 1.0),
}
#: R of the stretched solve and the reference's; plane errors unstretched and stretched -- order 8, then 12
MEASURED = {
    'E_metal_block': ((0.35555, 0.356150, 0.3209, 0.0400), (0.35553, 0.356150, 0.2576, 0.0063)),
    'E_hole': ((0.89270, 0.892808, 0.3679, 0.0647), (0.89331, 0.892808, 0.3086, 0.0331)),
    'E_dielectric_square': ((0.03650, 0.036622, 0.1483, 0.0071), (0.03650, 0.036622, 0.1149, 0.0041)),
    'E_metal_rectangle': ((0.63826, 0.640434, 0.3786, 0.0450), (0.63888, 0.640434, 0.3641, 0.0363)),
}


def _reference(key):
    with np.load(GOLDEN) as golden:
        x, y, z, field = golden['x'], golden['y'], float(golden['z']), golden[key]
    # gyptis's phasors are the conjugate of Wavebit's, and its incident wave is referenced to the
    # other face of the layer.
    return x, y, z, -np.exp(2j * np.pi * THICKNESS / WAVELENGTH) * np.conj(field)


def _solve(key, order, slope):
    size, inside, around = LAYERS[key]
    layer = rcwa.Layer(thickness=THICKNESS, background=Material(epsilon=around), regions=[
        rcwa.Region(geometry=rcwa.Rectangle(size=size), material=Material(epsilon=inside))])
    problem = rcwa.Problem(
        units=Units(length='nm'), stack=rcwa.Stack(period=(PERIOD, PERIOD), layers=[layer]),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH, sources=[rcwa.PlaneWave(side='top', polarization='p')]),
        outputs=rcwa.Outputs(fields=True))
    return rcwa.run(problem, numerics=rcwa.Numerics(
        orders=(order, order), factorization='li', **({} if slope is None else {'wall_resolution': slope})))


def _plane_error(result, reference):
    """RMS of the in-plane vector error over the plane, over the reference's own plane RMS."""
    x, y, z, field = reference
    values = np.asarray(result.field(x=x, y=y, z=[z]).values)[0, 0]
    scale = np.sqrt(np.mean(np.sum(np.abs(field[..., :2]) ** 2, axis=-1)))
    return np.sqrt(np.mean(np.sum(np.abs(values[..., :2] - field[..., :2]) ** 2, axis=-1))) / scale


@pytest.fixture(scope='module', params=sorted(LAYERS))
def case(request):
    key = request.param
    return key, _solve(key, 8, None), _solve(key, 8, 0.02), _reference(key)


def test_the_stretched_solve_agrees_with_the_reference_on_reflection(case):
    key, _, stretched, _ = case
    solved, reference = MEASURED[key][0][:2]
    assert float(np.sum(np.asarray(stretched.R))) == pytest.approx(solved, abs=2e-5)
    # 0.2% on the block, 0.01% on the hole, 0.3% on the square and the rectangle, at this order already.
    assert solved == pytest.approx(reference, rel=4e-3)


def test_each_solve_is_where_it_was_measured(case):
    key, plain, stretched, reference = case
    assert _plane_error(plain, reference) == pytest.approx(MEASURED[key][0][2], abs=1e-3)
    assert _plane_error(stretched, reference) == pytest.approx(MEASURED[key][0][3], abs=1e-3)


def test_the_stretch_brings_the_field_several_times_closer_at_order_8(case):
    key, plain, stretched, reference = case
    assert _plane_error(stretched, reference) < 0.2 * _plane_error(plain, reference)


#: The criteria written down before the implementation, at order 12: the stretched plane error as a
#: fraction of the unstretched one, and R against the reference's where one was set.
CRITERIA = {'E_metal_block': (0.06, 3e-3), 'E_dielectric_square': (0.1, 5e-3),
            'E_metal_rectangle': (0.25, None), 'E_hole': (0.2, None)}


@pytest.mark.slow
@pytest.mark.parametrize('key', sorted(LAYERS))
def test_at_order_12_the_criteria_set_before_the_implementation_hold(key):
    reference = _reference(key)
    plain, stretched = _solve(key, 12, None), _solve(key, 12, 0.02)
    solved, expected, plain_error, stretched_error = MEASURED[key][1]
    R = float(np.sum(np.asarray(stretched.R)))
    errors = _plane_error(plain, reference), _plane_error(stretched, reference)
    assert R == pytest.approx(solved, abs=2e-5)
    assert errors[0] == pytest.approx(plain_error, abs=1e-3) and errors[1] == pytest.approx(stretched_error, abs=1e-3)
    fraction, reflection = CRITERIA[key]
    assert errors[1] < fraction * errors[0]
    if reflection is not None:
        assert R == pytest.approx(expected, rel=reflection)
