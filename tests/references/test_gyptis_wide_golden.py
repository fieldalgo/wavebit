"""``wall_resolution`` beyond normal incidence, against stored 3D finite-element references.

``golden_gyptis_wide.npz`` holds sixteen gyptis cases (its record names the runs): thirteen metal
ones, ``eps`` -20+1.5j -- the 200 nm block at theta 20 and 40 deg (p and s), over glass, two layers
with a 100 nm gap and at 350 nm (P1); the radius 100 nm circle at theta 20 deg, over glass, in a
400 x 300 cell and in two layers (C1) -- and the dielectric block, ``eps`` 12.25, at normal
incidence and at theta 20 deg. Each case has ``R`` and ``T`` on the finer of two mesh levels and
on the coarser, and the in-plane ``E`` on the mid-plane of each patterned layer.

The criteria are the ones written before the table was measured (the validation page,
``wall-resolution-wide``): ``u_F``, the relative change of the reference ``R`` between the two
mesh levels, sets the tolerance for both ``R`` and ``T``, ``max(0.3%, 2 u_F)``, at order 12. On a
metal the in-plane plane error (RMS over the reference's plane RMS, all patterned planes together)
is at most a quarter of the default solve's at the same order -- uncompressed Li on a box, NVF on
a circle. On the dielectric it is at most 1%.

All sixteen run at order 12 in the slow tier, about 1 to 3 minutes each. The default run takes
the cheapest metal and dielectric cases, the 400 x 300 circle (p) and the block at normal
incidence, at order 8 under the same criteria: measured when the table was made, order 8 already
met them there (circle: ``R`` +0.06%, ``T`` -0.12%, field ratio 0.064; block: ``R`` -0.07%,
``T`` 0.00%, field 0.73%). Other cases do not all meet them at order 8 (the substrate ones least),
so order 8 is not asserted for them.
"""
from pathlib import Path

import numpy as np
import pytest

from wavebit import Material, Units, rcwa

GOLDEN = Path(__file__).with_name('golden_gyptis_wide.npz')
METAL, GLASS = -20 + 1.5j, 1.45 ** 2
BOX, CIRCLE = ('box', 200.0, 200.0), ('cyl', 100.0)
pytestmark = pytest.mark.filterwarnings('ignore::wavebit.rcwa.diagnostics.ExperimentalWarning')

#: per case: period (nm), wavelength (nm), theta (deg), polarization, substrate epsilon, gap (nm)
#: and layers bottom to top as (thickness, shape, shape epsilon, background epsilon)
CASES = {
    'R1p': ((400.0, 400.0), 900.0, 20, 'p', 1.0, None, [(150.0, BOX, METAL, 1.0)]),
    'R1s': ((400.0, 400.0), 900.0, 20, 's', 1.0, None, [(150.0, BOX, METAL, 1.0)]),
    'R2p': ((400.0, 400.0), 900.0, 40, 'p', 1.0, None, [(150.0, BOX, METAL, 1.0)]),
    'R2s': ((400.0, 400.0), 900.0, 40, 's', 1.0, None, [(150.0, BOX, METAL, 1.0)]),
    'R4p': ((400.0, 400.0), 900.0, 0, 'p', GLASS, None, [(150.0, BOX, METAL, 1.0)]),
    'R5p': ((400.0, 400.0), 900.0, 0, 'p', 1.0, 100.0, [(150.0, BOX, METAL, 1.0)] * 2),
    'R6p': ((400.0, 400.0), 350.0, 0, 'p', 1.0, None, [(150.0, BOX, METAL, 1.0)]),
    'C1ap': ((400.0, 400.0), 900.0, 20, 'p', 1.0, None, [(150.0, CIRCLE, METAL, 1.0)]),
    'C1as': ((400.0, 400.0), 900.0, 20, 's', 1.0, None, [(150.0, CIRCLE, METAL, 1.0)]),
    'C1bp': ((400.0, 400.0), 900.0, 0, 'p', GLASS, None, [(150.0, CIRCLE, METAL, 1.0)]),
    'C1cp': ((400.0, 300.0), 900.0, 0, 'p', 1.0, None, [(150.0, CIRCLE, METAL, 1.0)]),
    'C1cs': ((400.0, 300.0), 900.0, 0, 's', 1.0, None, [(150.0, CIRCLE, METAL, 1.0)]),
    'C1dp': ((400.0, 400.0), 900.0, 0, 'p', 1.0, 100.0, [(150.0, CIRCLE, METAL, 1.0)] * 2),
    'blk_d_n': ((400.0, 400.0), 900.0, 0, 'p', 1.0, None, [(150.0, BOX, 12.25, 1.0)]),
    'blk_d_o20p': ((400.0, 400.0), 900.0, 20, 'p', 1.0, None, [(150.0, BOX, 12.25, 1.0)]),
    'blk_d_o20s': ((400.0, 400.0), 900.0, 20, 's', 1.0, None, [(150.0, BOX, 12.25, 1.0)]),
}


def _reference(name):
    """R, T, u_F and the in-plane field in Wavebit's convention, ``(plane, n_y, n_x, 2)``."""
    with np.load(GOLDEN) as golden:
        k = list(golden['case']).index(name)
        R, T, u_F = float(golden['R'][k]), float(golden['T'][k]), float(golden['u_F'][k])
        x, y, depth = golden[f'x_{name}'], golden[f'y_{name}'], golden[f'depth_{name}']
        field = golden[f'E_{name}'].astype(np.complex128)
    # gyptis's phasors are the conjugate of Wavebit's; its incident wave has phase 0 at the stack's
    # top, as Wavebit's, and the sign of Wavebit's p unit vector is the opposite of its own.
    sign = -1.0 if CASES[name][3] == 'p' else 1.0
    return R, T, u_F, x, y, depth, sign * np.conj(field)


def _problem(name):
    period, wavelength, theta, polarization, substrate, gap, stack = CASES[name]
    layers = []
    for k, (thickness, shape, epsilon, background) in enumerate(stack):
        if k:
            layers.append(rcwa.Layer(thickness=gap, background=Material(epsilon=1.0)))
        geometry = (rcwa.Rectangle(size=shape[1:]) if shape[0] == 'box' else rcwa.Circle(radius=shape[1]))
        layers.append(rcwa.Layer(thickness=thickness, background=Material(epsilon=background),
                                 regions=[rcwa.Region(geometry=geometry, material=Material(epsilon=epsilon))]))
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=period, layers=layers, bottom=Material(epsilon=substrate)),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=[rcwa.PlaneWave(
            side='top', polarization=polarization, theta=np.radians(theta), phi=0.0)]),
        outputs=rcwa.Outputs(fields=True))


def _solve(name, order, stretched):
    """R, T and the in-plane field on the reference's planes; ``stretched``: P1 or C1, else the default solve."""
    if stretched:
        numerics = rcwa.Numerics(orders=(order, order), factorization='li', wall_resolution=0.02)
    else:
        numerics = rcwa.Numerics(orders=(order, order),
                                 factorization='li' if CASES[name][6][0][1][0] == 'box' else 'nvf')
    result = rcwa.run(_problem(name), numerics=numerics)
    _, _, _, x, y, depth, _ = _reference(name)
    stack = CASES[name][6]
    height = sum(layer[0] for layer in stack) + (CASES[name][5] or 0.0) * (len(stack) - 1)
    # Wavebit's z is 0 at the stack's bottom; the reference's depth is measured down from its top.
    field = np.stack([np.asarray(result.field(x=x, y=y, z=[height - d]).values)[0, 0][..., :2] for d in depth])
    return float(np.sum(np.asarray(result.R))), float(np.sum(np.asarray(result.T))), field


def _plane_error(field, reference):
    error = np.sum(np.abs(field - reference) ** 2, axis=-1)
    return float(np.sqrt(error.mean() / np.mean(np.sum(np.abs(reference) ** 2, axis=-1))))


def _check(name, order):
    R_ref, T_ref, u_F, *_, reference = _reference(name)
    tolerance = max(3e-3, 2 * u_F)
    R, T, field = _solve(name, order, stretched=True)
    assert abs(R - R_ref) / R_ref <= tolerance, (R, R_ref, tolerance)
    assert abs(T - T_ref) / T_ref <= tolerance, (T, T_ref, tolerance)
    error = _plane_error(field, reference)
    if CASES[name][6][0][2] == METAL:
        default = _plane_error(_solve(name, order, stretched=False)[2], reference)
        assert error <= 0.25 * default, (error, default)
    else:
        assert error <= 1e-2, error


def test_the_golden_holds_the_sixteen_cases():
    with np.load(GOLDEN) as golden:
        assert list(golden['case']) == list(CASES)
        # u_F is the stored levels' own R difference, which the tolerances read.
        np.testing.assert_allclose(golden['u_F'], np.abs(golden['R'] - golden['R_coarse']) / golden['R'], rtol=1e-12)
        assert np.all(golden['R'] + golden['T'] <= 1.0)


@pytest.mark.parametrize('name', ['C1cp', 'blk_d_n'])
def test_order_8_already_meets_the_criteria_on_the_cheapest_cases(name):
    _check(name, 8)


@pytest.mark.slow
@pytest.mark.shard_unit
@pytest.mark.parametrize('name', list(CASES))
def test_order_12_meets_the_preregistered_criteria(name):
    _check(name, 12)
