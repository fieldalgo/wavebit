"""A FarFieldMonitor box radiates only what it encloses, so a dipole outside it is refused.

The far field is the transform of the equivalent currents on the box's six faces. A dipole
outside the box contributes no net current there: the box reads its field passing through
and the transform returns almost nothing. Before the refusal, an ``Ez`` dipole beside a
0.3 box (the scene below, 20 cells per wavelength) was admitted and its sphere integral was
2.2e-20 against 8.5e-15 for a 0.5 box around it, about 3e-6 of the radiated power, with no
warning. A box that encloses the dipole is unchanged.
"""
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit import Units, fdtd
from wavebit.errors import ValidationError
from tests.fixtures.fdtd.fd_helpers import WAVELENGTHS, farfield_dipole, sphere_integral, to_numpy

pytestmark = pytest.mark.filterwarnings('ignore::wavebit.fdtd._legacy.CoarseMeshWarning')

CENTER = (0.013, -0.02, 0.007)
DX = 0.05
#: Held by tests/fdtd/test_fd_farfield.py: the sphere against the box flux, the flux against
#: the dipole's power, and the absolute field against the closed form (so 1-(1-tol)^2 in power).
SPHERE_TOL = 1.5e-2
FLUX_POWER_TOL = 1e-6
FIELD_TOL = 3e-2


def _problem(box, dipoles=(CENTER,)):
    pulse = fdtd.GaussianPulse(wavelength=1.0, frequency_sigma_hz=6e13)
    sources = tuple(fdtd.CurrentDipole(f'd{i}', center=c, component='Ez', pulse=pulse)
                    for i, c in enumerate(dipoles))
    return fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(1.4,) * 3),
                         boundaries=fdtd.Boundaries(x=fdtd.Open(), y=fdtd.Open(), z=fdtd.Open())),
        sources=sources,
        outputs=fdtd.Outputs(monitors=(fdtd.FarFieldMonitor('ff', wavelengths=(1.0,), box=box),)))


NUMERICS = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
                         duration=fdtd.Duration(periods=14, source='d0'))


def _check(problem):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fdtd.check(problem, numerics=NUMERICS)


@pytest.mark.parametrize('box, dipoles', [
    (fdtd.SnappedBox(size=(0.3,) * 3, center=(0.35, 0., 0.)), (CENTER,)),
    (fdtd.SnappedBox(size=(0.3,) * 3, center=(0., -0.4, 0.)), (CENTER,)),
    # One dipole inside a generous box, a second well outside it.
    (fdtd.SnappedBox(size=(0.5,) * 3), (CENTER, (0.013, -0.02, 0.5))),
], ids=['beside-x', 'beside-y', 'second-dipole-outside'])
def test_a_box_that_does_not_enclose_a_dipole_is_refused(box, dipoles):
    problem = _problem(box, dipoles)
    report = _check(problem)
    assert not report.ok
    codes = [issue.code for issue in report.issues]
    assert codes == ['monitor.source_outside'], report.issues
    assert report.issues[0].path == 'outputs.ff.box'
    with pytest.raises(ValidationError, match='does not enclose') as raised:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fdtd.prepare(problem, numerics=NUMERICS)
    assert raised.value.code == 'monitor.source_outside'


def test_every_admitted_box_encloses_the_dipole():
    """Slide a 0.5 box along x through and past the dipole, a half cell at a time: a box is
    admitted only while the dipole's centre is inside it. The stamp covers its centre and the
    snapped face keeps one cell beyond the stamp, so the declared face, which snapping moves
    by at most half a cell, is at least half a cell past the centre."""
    admitted = []
    for shift in np.arange(-0.6, 0.6 + 1e-9, DX / 2):
        box = fdtd.SnappedBox(size=(0.5,) * 3, center=(shift, 0., 0.))
        report = _check(_problem(box))
        if report.ok:
            admitted.append(shift)
            inside = abs(CENTER[0] - shift) <= 0.25 - DX / 2
            assert inside, (shift, report)
        else:
            assert {i.code for i in report.issues} <= {'monitor.source_outside', 'fdtd.invalid'}
    # The centred box and its near neighbours stay admitted.
    assert any(abs(s) < 1e-9 for s in admitted) and len(admitted) >= 3, admitted


def test_the_legacy_declaration_is_refused_the_same_way():
    scene = farfield_dipole(box=0.3, periods=1).replace(
        monitors=(fd.FarField('ff', box=fd.Region(size=(0.3,) * 3, center=(0.35, 0., 0.)),
                              wavelengths=WAVELENGTHS),))
    with pytest.raises(ValueError, match='does not enclose'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fd.plan(scene)


def test_an_enclosing_box_still_radiates_the_dipole_power():
    """The scene of tests/fdtd/test_fd_farfield.py: the box's flux is the dipole's ``power()``,
    the sphere integral is that flux, and both are the closed form ``k^2 |Il|^2 / (12 pi)``
    to the grid's deficit that test bounds."""
    res = fd.run(farfield_dipole(component='Ez'))
    ff, src = res['ff'], res.sources['d']
    power = to_numpy(src.power())
    flux = to_numpy(ff.flux)
    sphere = sphere_integral(ff)
    assert np.all(np.abs(flux / power - 1.0) < FLUX_POWER_TOL), flux / power - 1.0
    assert np.all(np.abs(sphere / flux - 1.0) < SPHERE_TOL), sphere / flux - 1.0
    f = 1.0 / np.asarray(WAVELENGTHS)
    phasor = (np.exp(2j * math.pi * f[:, None] * to_numpy(src.t)[None, :])
              @ to_numpy(src.drive)) * res.plan.dt
    k = 2 * math.pi * f
    closed_form = k ** 2 * np.abs(phasor) ** 2 / (12 * math.pi)
    assert np.all(np.abs(sphere / closed_form - 1.0) < 1.0 - (1.0 - FIELD_TOL) ** 2), \
        sphere / closed_form - 1.0
