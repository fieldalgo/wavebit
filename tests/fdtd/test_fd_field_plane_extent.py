"""A FieldMonitor plane without spans covers the physical domain, never an absorbing layer.

An ``Open()`` face's PML (or ConductiveAbsorber) is padding outside the declared domain, and
inside it the stretched field is not the physical one. A FluxMonitor plane across such an axis
is refused for that reason. A FieldMonitor plane with no transverse span used to read the
whole allocated grid instead: on the dipole scene below (1.4 um cube, 0.05 um cells, a
12-cell PML all round) the ``z = 0.2`` plane returned 52 x 52 samples, of which 2.6% (Ex),
22% (Ez) and 27% (Hy) of ``sum |F|^2`` lay in the layers, with nothing marking them. A plane
without a span along an axis now reads exactly what the explicit span of the physical domain
along that axis reads; axes without an absorbing face keep every point.
"""
import warnings

import numpy as np
import pytest

from wavebit import Units, fdtd

DX = 0.05
SIDE = 1.4
W = (1.0,)
COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def _problem(boundaries, monitors, absorber=None):
    problem = fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(SIDE,) * 3), boundaries=boundaries),
        sources=(fdtd.CurrentDipole('d', center=(0.013, -0.02, 0.007), component='Ez',
                                    pulse=fdtd.GaussianPulse(wavelength=1.0,
                                                             frequency_sigma_hz=6e13)),),
        outputs=fdtd.Outputs(monitors=monitors))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
                             duration=fdtd.Duration(steps=60),
                             **({} if absorber is None else {'absorber': absorber}))
    return problem, numerics


def _run(problem, numerics):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fdtd.run(problem, numerics=numerics)


def _pair(normal, spans):
    plane = fdtd.Plane(**{normal: 0.2})
    full = fdtd.Plane(**{normal: 0.2}, **spans)
    return (fdtd.FieldMonitor('default', wavelengths=W, plane=plane),
            fdtd.FieldMonitor('explicit', wavelengths=W, plane=full))


HALF = (-SIDE / 2, SIDE / 2)


@pytest.mark.parametrize('absorber', [None, fdtd.ConductiveAbsorber(cells=12)],
                         ids=['pml', 'conductive'])
def test_a_plane_without_spans_reads_the_explicit_physical_span(absorber):
    problem, numerics = _problem(fdtd.Boundaries.open(), _pair('z', dict(x=HALF, y=HALF)),
                                 absorber)
    res = _run(problem, numerics)
    default, explicit = res['default'], res['explicit']
    for c in COMPONENTS:
        xs, ys, _ = (np.asarray(v) for v in default.coordinates[c])
        assert xs.min() >= HALF[0] - 1e-12 and xs.max() <= HALF[1] + 1e-12, (c, xs)
        assert ys.min() >= HALF[0] - 1e-12 and ys.max() <= HALF[1] + 1e-12, (c, ys)
        for got, want in zip(default.coordinates[c], explicit.coordinates[c]):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        np.testing.assert_array_equal(np.asarray(default.values[c]),
                                      np.asarray(explicit.values[c]))
    # 28 cells across: 29 node points and 28 half-cell points per transverse axis.
    assert np.asarray(default.values['Ez']).shape[-2:] == (29, 29)
    assert np.asarray(default.values['Ex']).shape[-2:] == (29, 28)


def test_only_absorbing_axes_are_clipped():
    """x Periodic, y PEC, z Open: a plane across x keeps every point along y (no layer) and
    stops at the physical domain along z."""
    boundaries = fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.PEC(), z=fdtd.Open())
    problem, numerics = _problem(boundaries, _pair('x', dict(z=HALF)))
    res = _run(problem, numerics)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        report = fdtd.check(problem, numerics=numerics)
    ny = report.resolved['grid_shape'][1]
    default, explicit = res['default'], res['explicit']
    for c in COMPONENTS:
        _, ys, zs = (np.asarray(v) for v in default.coordinates[c])
        assert len(ys) == ny, (c, len(ys), ny)
        assert zs.min() >= HALF[0] - 1e-12 and zs.max() <= HALF[1] + 1e-12, (c, zs)
        np.testing.assert_array_equal(np.asarray(default.values[c]),
                                      np.asarray(explicit.values[c]))


def test_a_plane_with_no_absorbing_axis_keeps_the_whole_axis():
    """Periodic transverse faces have no padding: the default is every point, as before."""
    boundaries = fdtd.Boundaries.periodic(z=fdtd.Open())
    problem, numerics = _problem(boundaries, (fdtd.FieldMonitor(
        'default', wavelengths=W, plane=fdtd.Plane(z=0.2)),))
    res = _run(problem, numerics)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nx, ny, _ = fdtd.check(problem, numerics=numerics).resolved['grid_shape']
    for c in COMPONENTS:
        assert np.asarray(res['default'].values[c]).shape[-2:] == (ny, nx)


def test_a_plane_without_spans_prepares_the_plan_of_the_explicit_span():
    """The static plan (its skeleton) of a span-less plane is the explicit physical span's, so
    the pinned skeletons in tests/fdtd/test_public_names.py follow from this rule."""
    def skeleton(plane):
        problem, numerics = _problem(fdtd.Boundaries.open(), (fdtd.FieldMonitor(
            'f', wavelengths=W, plane=plane, components=('Ex', 'Hy')),))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return fdtd.prepare(problem, numerics=numerics)._plan.skeleton
    assert skeleton(fdtd.Plane(z=-0.3)) == skeleton(fdtd.Plane(z=-0.3, x=HALF, y=HALF))
    assert skeleton(fdtd.Plane(z=-0.3)) != skeleton(fdtd.Plane(z=-0.3, x=(-0.5, 0.5), y=HALF))
