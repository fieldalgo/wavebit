"""``check().resolved['boundary_planes']`` reports where each wall or mirror actually acts.

On the Yee grid tangential E sits on node planes and tangential H on half planes, so a
PMC wall (zero tangential H) and a high-side MirrorSymmetry (whose padded E is the image of
the last E layer) act on a half plane next to the declared face: a PMC wall half a cell
outside the low face and half a cell inside the high face, a high-side mirror half a cell
inside. PEC walls and a low-side mirror act on the face itself
(``src/wavebit/fdtd/_kernel/faces.py``).

**The measurement.** A one-dimensional cavity along x (two Periodic cells along y and z)
starts from ``Ey(x)``, a standing wave written from the *reported* planes ``a, b``: ``sin``
from a plane where tangential E vanishes (PEC, ``MirrorSymmetry(-1)``) and ``cos`` from one
where tangential H does (PMC, ``MirrorSymmetry(+1)``), with the lowest ``k`` both planes
allow. The continuum sinusoid is exactly a discrete eigenvector of the Yee lattice with
those walls, so every probe obeys ``f(n+1) + f(n-1) = 2 cos(w dt) f(n)`` with the closed
form ``cos(w dt) = 1 - 2 (c dt/dx)^2 sin^2(k dx/2)``; measured (2026-11-08, NumPy, 16 cells,
400 steps) to 8e-16..3.6e-15 for all seven wall pairs below. The same wave written from the
*declared* faces misses by 1.2e-3 (PEC/PMC) to 7.6e-2 (PEC/MirrorSymmetry(-1)) wherever the
two differ, and matches to round-off where they do not -- which is why both kinds of pair
are here.
"""
import math
import warnings

import numpy as np
import pytest

from wavebit import Units, fdtd

DX = 0.05
N = 16
C0 = 299792458.0
TOL = 1e-12
#: The smallest declared-plane miss measured above is 1.16e-3.
MUTANT_FLOOR = 1e-4
PROBES = (-0.31, -0.02, 0.21, 0.36)

PAIRS = {
    'pmc-pmc': (fdtd.PMC(), fdtd.PMC(), -0.5, -0.5),
    'pec-pmc': (fdtd.PEC(), fdtd.PMC(), 0.0, -0.5),
    'pec-mirror+': (fdtd.PEC(), fdtd.MirrorSymmetry(parity=1), 0.0, -0.5),
    'pec-mirror-': (fdtd.PEC(), fdtd.MirrorSymmetry(parity=-1), 0.0, -0.5),
    'mirror--pmc': (fdtd.MirrorSymmetry(parity=-1), fdtd.PMC(), 0.0, -0.5),
    'mirror+-pec': (fdtd.MirrorSymmetry(parity=1), fdtd.PEC(), 0.0, 0.0),
    'pec-pec': (fdtd.PEC(), fdtd.PEC(), 0.0, 0.0),
}


def _problem(low, high):
    return fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(N * DX, 2 * DX, 2 * DX)),
                         boundaries=fdtd.Boundaries(x=(low, high), y=fdtd.Periodic(),
                                                    z=fdtd.Periodic())))


NUMERICS = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX), duration=fdtd.Duration(steps=400))


def _check(problem):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fdtd.check(problem, numerics=NUMERICS)


def _tangential_e_vanishes(face):
    return isinstance(face, fdtd.PEC) or (isinstance(face, fdtd.MirrorSymmetry)
                                          and face.parity == -1)


def recurrence_miss(low, high, a, b):
    """``max |f(n+1) + f(n-1) - 2 cos(w dt) f(n)| / max |f|`` over the probes, for the
    standing wave written from planes ``a`` (low) and ``b`` (high)."""
    za, zb = _tangential_e_vanishes(low), _tangential_e_vanishes(high)
    k = math.pi / (b - a) if za == zb else math.pi / (2 * (b - a))
    shape = (lambda x: np.sin(k * (x - a))) if za else (lambda x: np.cos(k * (x - a)))
    problem = _problem(low, high).replace(
        initial=fdtd.InitialField(E=(None, lambda x, y, z: shape(x) + 0 * y + 0 * z, None)),
        outputs=fdtd.Outputs(monitors=tuple(
            fdtd.TimeSeriesMonitor(f'p{i}', at=(x, 0., 0.), component='Ey')
            for i, x in enumerate(PROBES))))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fdtd.run(problem, numerics=NUMERICS)
    rho = C0 * res.provenance['time_step_seconds'] / (DX * 1e-6)
    cos_wdt = 1 - 2 * rho ** 2 * math.sin(k * DX / 2) ** 2
    worst = 0.0
    for i in range(len(PROBES)):
        f = np.asarray(res[f'p{i}'].values).ravel()
        assert np.max(np.abs(f)) > 1e-3, 'the probe sits on a node of the mode'
        r = f[2:] + f[:-2] - 2 * cos_wdt * f[1:-1]
        worst = max(worst, float(np.max(np.abs(r)) / np.max(np.abs(f))))
    return worst


@pytest.mark.parametrize('pair', PAIRS)
def test_the_report_names_each_face_and_its_offset(pair):
    low, high, off_lo, off_hi = PAIRS[pair]
    planes = _check(_problem(low, high)).resolved['boundary_planes']
    half = N * DX / 2
    for face, declared, offset, kind in (('x-', -half, off_lo, low), ('x+', half, off_hi, high)):
        row = planes[face]
        assert row['boundary'] == type(kind).__name__
        assert row['declared'] == pytest.approx(declared, abs=1e-12)
        assert row['offset_cells'] == offset
        assert row['effective'] == pytest.approx(declared + offset * DX, abs=1e-12)
    for face in ('y-', 'y+', 'z-', 'z+'):
        assert planes[face]['boundary'] == 'Periodic'
        assert planes[face]['effective'] is None and planes[face]['offset_cells'] is None


def test_open_faces_report_no_plane():
    problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(0.6, 0.6, 0.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Open(), y=fdtd.Periodic(), z=(fdtd.PEC(), fdtd.PMC()))))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX), duration=fdtd.Duration(steps=10),
                             absorber=fdtd.PML(cells=12, wavelength=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        planes = fdtd.check(problem, numerics=numerics).resolved['boundary_planes']
    assert planes['x-']['boundary'] == 'Open' and planes['x-']['effective'] is None
    # The declared face of an Open axis is the physical one, not the padded grid's edge.
    assert planes['x-']['declared'] == pytest.approx(-0.3) and planes['x+']['declared'] == pytest.approx(0.3)
    assert planes['z-']['offset_cells'] == 0.0 and planes['z+']['offset_cells'] == -0.5


@pytest.mark.parametrize('pair', PAIRS)
def test_a_wave_written_from_the_reported_planes_is_a_mode_of_the_run(pair):
    low, high, off_lo, off_hi = PAIRS[pair]
    planes = _check(_problem(low, high)).resolved['boundary_planes']
    miss = recurrence_miss(low, high, planes['x-']['effective'], planes['x+']['effective'])
    assert miss < TOL, miss
    declared = recurrence_miss(low, high, planes['x-']['declared'], planes['x+']['declared'])
    if off_lo or off_hi:
        # Mutant: the declared faces, where the report says the condition does not act.
        assert declared > MUTANT_FLOOR, declared
    else:
        assert declared < TOL, declared
