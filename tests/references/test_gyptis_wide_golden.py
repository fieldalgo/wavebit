"""``wall_resolution`` beyond normal incidence, against stored 3D finite-element references.

``golden_gyptis_wide.npz`` holds sixteen gyptis cases (its record names the runs): thirteen metal
ones, ``eps`` -20+1.5j -- the 200 nm block at theta 20 and 40 deg (p and s), over glass, two layers
with a 100 nm gap and at 350 nm (P1); the radius 100 nm circle at theta 20 deg, over glass, in a
400 x 300 cell and in two layers (C1) -- and the dielectric block, ``eps`` 12.25, at normal
incidence and at theta 20 deg. ``golden_gyptis_wide_c1.npz`` holds five more metal C1 cases, run
after the first table under the same criteria: the circle at theta 40 deg (p and s) and at 350 nm,
and a 130 x 80 nm ellipse at theta 20 deg (p and s). ``golden_gyptis_wide_c1e.npz`` holds six more,
the same ellipse at theta 40 deg, over glass and in two layers with a 100 nm gap (p and s each).
``golden_gyptis_wide_d.npz`` holds six with shapes away from the cell's centre: the circle moved across
the cell's edge and its corner and the ellipse moved across the corner at theta 20 deg, the circle moved
across the edge of the 400 x 300 cell, the block moved across the corner (written as the four
rectangles the cell's edges cut it into, as a rectangle may not cross the cell's edge), and a T of two
rectangles with no x mirror at theta 20 deg, p. A moved shape is the centred structure moved: its
reference ``R`` and ``T`` are the centred structure's, and its field is the centred one read at the
moved samples.
Each case has ``R`` and ``T`` on the finer of two mesh levels and on the coarser, and the in-plane
``E`` on the mid-plane of each patterned layer.

The criteria are the ones written before the table was measured (the validation page,
``wall-resolution-wide``): ``u_F``, the relative change of the reference ``R`` between the two
mesh levels, sets the tolerance for both ``R`` and ``T``, ``max(0.3%, 2 u_F)``, at order 12. On a
metal the in-plane plane error (RMS over the reference's plane RMS, all patterned planes together)
is at most a quarter of the default solve's at the same order -- uncompressed Li on a block or
rectangles, NVF on a circle or an ellipse. On the dielectric it is at most 1%.

All thirty-three run at order 12 in the slow tier, about 1 to 3 minutes each. The default run takes
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

GOLDENS = [Path(__file__).with_name(name)
           for name in ('golden_gyptis_wide.npz', 'golden_gyptis_wide_c1.npz', 'golden_gyptis_wide_c1e.npz',
                        'golden_gyptis_wide_d.npz')]
METAL, GLASS = -20 + 1.5j, 1.45 ** 2
BOX, CIRCLE, ELLIPSE = ('box', 200.0, 200.0), ('cyl', 100.0), ('ell', 130.0, 80.0)
#: a shape away from the cell's centre: ('at', shape, (x, y)); several rectangles: ('rects', ((x0, y0, x1, y1), ..))
BLOCK_MOVED = ('rects', ((46.0, 30.0, 200.0, 200.0), (-200.0, 30.0, -154.0, 200.0), (46.0, -200.0, 200.0, -170.0),
                         (-200.0, -200.0, -154.0, -170.0)))
T_SHAPE = ('rects', ((-120.0, -100.0, 0.0, 100.0), (0.0, -50.0, 80.0, 50.0)))
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
    # golden_gyptis_wide_c1.npz
    'C1r6p': ((400.0, 400.0), 350.0, 0, 'p', 1.0, None, [(150.0, CIRCLE, METAL, 1.0)]),
    'C1a40p': ((400.0, 400.0), 900.0, 40, 'p', 1.0, None, [(150.0, CIRCLE, METAL, 1.0)]),
    'C1a40s': ((400.0, 400.0), 900.0, 40, 's', 1.0, None, [(150.0, CIRCLE, METAL, 1.0)]),
    'C1ep': ((400.0, 400.0), 900.0, 20, 'p', 1.0, None, [(150.0, ELLIPSE, METAL, 1.0)]),
    'C1es': ((400.0, 400.0), 900.0, 20, 's', 1.0, None, [(150.0, ELLIPSE, METAL, 1.0)]),
    # golden_gyptis_wide_c1e.npz
    'C1e40p': ((400.0, 400.0), 900.0, 40, 'p', 1.0, None, [(150.0, ELLIPSE, METAL, 1.0)]),
    'C1e40s': ((400.0, 400.0), 900.0, 40, 's', 1.0, None, [(150.0, ELLIPSE, METAL, 1.0)]),
    'C1ebp': ((400.0, 400.0), 900.0, 0, 'p', GLASS, None, [(150.0, ELLIPSE, METAL, 1.0)]),
    'C1ebs': ((400.0, 400.0), 900.0, 0, 's', GLASS, None, [(150.0, ELLIPSE, METAL, 1.0)]),
    'C1edp': ((400.0, 400.0), 900.0, 0, 'p', 1.0, 100.0, [(150.0, ELLIPSE, METAL, 1.0)] * 2),
    'C1eds': ((400.0, 400.0), 900.0, 0, 's', 1.0, 100.0, [(150.0, ELLIPSE, METAL, 1.0)] * 2),
    # golden_gyptis_wide_d.npz
    'C1ap_m1': ((400.0, 400.0), 900.0, 20, 'p', 1.0, None, [(150.0, ('at', CIRCLE, (196.0, 0.0)), METAL, 1.0)]),
    'C1ap_m2': ((400.0, 400.0), 900.0, 20, 'p', 1.0, None, [(150.0, ('at', CIRCLE, (196.0, 187.0)), METAL, 1.0)]),
    'C1ep_m': ((400.0, 400.0), 900.0, 20, 'p', 1.0, None, [(150.0, ('at', ELLIPSE, (-146.0, 130.0)), METAL, 1.0)]),
    'C1cp_m': ((400.0, 300.0), 900.0, 0, 'p', 1.0, None, [(150.0, ('at', CIRCLE, (60.0, 110.0)), METAL, 1.0)]),
    'R1p_m': ((400.0, 400.0), 900.0, 20, 'p', 1.0, None, [(150.0, BLOCK_MOVED, METAL, 1.0)]),
    'A1T_o20p': ((400.0, 400.0), 900.0, 20, 'p', 1.0, None, [(150.0, T_SHAPE, METAL, 1.0)]),
}


def _cases(path):
    with np.load(path) as golden:
        return list(golden['case'])


def _reference(name):
    """R, T, u_F and the in-plane field in Wavebit's convention, ``(plane, n_y, n_x, 2)``."""
    golden_file = next(path for path in GOLDENS if name in _cases(path))
    with np.load(golden_file) as golden:
        k = list(golden['case']).index(name)
        R, T, u_F = float(golden['R'][k]), float(golden['T'][k]), float(golden['u_F'][k])
        x, y, depth = golden[f'x_{name}'], golden[f'y_{name}'], golden[f'depth_{name}']
        field = golden[f'E_{name}'].astype(np.complex128)
    # gyptis's phasors are the conjugate of Wavebit's; its incident wave has phase 0 at the stack's
    # top, as Wavebit's, and the sign of Wavebit's p unit vector is the opposite of its own.
    sign = -1.0 if CASES[name][3] == 'p' else 1.0
    return R, T, u_F, x, y, depth, sign * np.conj(field)


def _geometries(shape, centre=(0.0, 0.0)):
    if shape[0] == 'at':
        return _geometries(shape[1], shape[2])
    if shape[0] == 'rects':
        return [rcwa.Rectangle(size=(x1 - x0, y1 - y0), center=((x0 + x1) / 2, (y0 + y1) / 2))
                for x0, y0, x1, y1 in shape[1]]
    if shape[0] == 'box':
        return [rcwa.Rectangle(size=shape[1:], center=centre)]
    if shape[0] == 'cyl':
        return [rcwa.Circle(radius=shape[1], center=centre)]
    return [rcwa.Ellipse(radii=shape[1:], center=centre)]


def _rectangles(name):
    shape = CASES[name][6][0][1]
    return (shape[1] if shape[0] == 'at' else shape)[0] in ('box', 'rects')


def _problem(name):
    period, wavelength, theta, polarization, substrate, gap, stack = CASES[name]
    layers = []
    for k, (thickness, shape, epsilon, background) in enumerate(stack):
        if k:
            layers.append(rcwa.Layer(thickness=gap, background=Material(epsilon=1.0)))
        layers.append(rcwa.Layer(thickness=thickness, background=Material(epsilon=background),
                                 regions=[rcwa.Region(geometry=geometry, material=Material(epsilon=epsilon))
                                          for geometry in _geometries(shape)]))
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
                                 factorization='li' if _rectangles(name) else 'nvf')
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


def test_the_goldens_hold_the_thirty_three_cases():
    assert sum((_cases(path) for path in GOLDENS), []) == list(CASES)
    for path in GOLDENS:
        with np.load(path) as golden:
            # u_F is the stored levels' own R difference, which the tolerances read.
            np.testing.assert_allclose(golden['u_F'], np.abs(golden['R'] - golden['R_coarse']) / golden['R'],
                                       rtol=1e-12)
            assert np.all(golden['R'] + golden['T'] <= 1.0)


@pytest.mark.parametrize('name', ['C1cp', 'blk_d_n'])
def test_order_8_already_meets_the_criteria_on_the_cheapest_cases(name):
    _check(name, 8)


@pytest.mark.slow
@pytest.mark.shard_unit
@pytest.mark.parametrize('name', list(CASES))
def test_order_12_meets_the_preregistered_criteria(name):
    _check(name, 12)
