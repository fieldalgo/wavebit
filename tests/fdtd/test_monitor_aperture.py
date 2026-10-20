"""Public finite apertures: requested physical bounds and actual Yee samples."""
import numpy as np
import pytest

from wavebit import Units, fdtd as fd
from wavebit.errors import ValidationError
from wavebit.units import Z0

NUMERICS = fd.Numerics(grid=fd.UniformGrid(spacing=.025), duration=fd.Duration(steps=2))


def problem(monitors, axis='z', reverse=False):
    normal = 'xyz'.index(axis)
    electric = (normal + (2 if reverse else 1)) % 3
    magnetic = (normal + (1 if reverse else 2)) % 3
    return fd.Problem(units=Units(length='um'), scene=fd.Scene(
        domain=fd.Domain(size=(.4,)*3),
        boundaries=fd.Boundaries(**{a: fd.Periodic() for a in 'xyz'})),
        initial=fd.InitialField(E=tuple(float(a == electric) for a in range(3)),
                              H=tuple(float(a == magnetic)/Z0 for a in range(3))),
        outputs=fd.Outputs(monitors=tuple(monitors)))


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('kind', [fd.FieldMonitor, fd.FluxMonitor])
@pytest.mark.parametrize('span', [(-.201, .05), (-.25, .05), (-.05, .201)])
def test_requested_transverse_bounds_cannot_be_clipped_or_rounded_into_domain(axis, kind, span):
    transverse = 'xyz'[('xyz'.index(axis)+1) % 3]
    plane = fd.Plane(**{axis: 0., transverse: span})
    p = problem([kind('aperture', plane=plane, wavelengths=(.8,))])
    (issue,) = fd.check(p, numerics=NUMERICS).issues
    assert (issue.code, issue.path) == ('monitor_extent', 'outputs.aperture.plane.'+transverse)
    assert issue.expected == (-.2, .2)
    assert issue.actual == span
    for door in (fd.prepare, fd.run):
        with pytest.raises(ValidationError) as caught:
            door(p, numerics=NUMERICS)
        assert caught.value.issue == issue


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('spans', [((-.15, .05), (-.025, .1)),
                                  ((-.173, .067), (-.061, .146)),
                                  ((-.2, -.129), (.119, .174))])
def test_asymmetric_aperture_samples_and_flux_use_the_selected_surface(axis, reverse, spans):
    normal = 'xyz'.index(axis)
    transverse = [a for a in range(3) if a != normal]
    extents = dict(zip(transverse, spans))
    finite = fd.Plane(**{axis: 0., **{'xyz'[a]: extents[a] for a in transverse}})
    p = problem([fd.FieldMonitor('samples', plane=finite, wavelengths=(.8,)),
                 fd.FluxMonitor('finite', plane=finite, wavelengths=(.8,)),
                 fd.FluxMonitor('whole', plane=fd.Plane(**{axis: 0.}), wavelengths=(.8,))], axis, reverse)
    r = fd.run(p, numerics=NUMERICS)
    dx = .025
    for component, coords in r['samples'].coordinates.items():
        component_axis = 'xyz'.index(component[1])
        for a in transverse:
            # Independent Yee coordinates, then selection by physical endpoints.
            half = (a == component_axis) if component[0] == 'E' else (a != component_axis)
            all_points = -.2 + (np.arange(16) + .5*half)*dx
            low, high = extents[a]
            selected = all_points[(all_points >= low-1e-14) & (all_points <= high+1e-14)]
            np.testing.assert_allclose(coords[a], selected, rtol=0., atol=1e-14)
        assert r['samples'].values[component].shape == (1, *(len(c) for c in reversed(coords)))
    # Flux snaps endpoints to nearest nodes, unlike Field's point-in-interval selection.
    widths = [(round((high+.2)/dx)-round((low+.2)/dx))*dx for low, high in spans]
    area = widths[0]*widths[1]
    full = np.asarray(r['whole'].spectral_energy)
    finite_flux = np.asarray(r['finite'].spectral_energy)
    assert np.all(np.abs(full) > 0)
    assert np.all(full < 0) if reverse else np.all(full > 0)
    np.testing.assert_allclose(finite_flux/full, area/(.4*.4), rtol=2e-14, atol=0.)


@pytest.mark.parametrize('kind', [fd.FieldMonitor, fd.FluxMonitor])
def test_an_aperture_with_no_selected_points_or_cells_is_diagnosed(kind):
    p = problem([kind('empty', plane=fd.Plane(z=0., x=(.001, .002)), wavelengths=(.8,))])
    report = fd.check(p, numerics=NUMERICS)
    assert not report.ok
    assert 'empty' in report.issues[0].message
    for door in (fd.prepare, fd.run):
        with pytest.raises(ValueError, match='empty'):
            door(p, numerics=NUMERICS)


def test_reversed_or_zero_width_spans_are_rejected_when_declared():
    for span in ((.1, -.1), (.1, .1)):
        with pytest.raises(ValueError, match='lo < hi'):
            fd.Plane(z=0., x=span)


@pytest.mark.parametrize('kind', [fd.FieldMonitor, fd.FluxMonitor])
@pytest.mark.parametrize('span', [(-.201, .05), (-.05, .201)])
def test_aperture_bounds_use_physical_domain_before_open_padding(kind, span):
    p = problem([kind('padded', plane=fd.Plane(z=0., x=span), wavelengths=(.8,))])
    p = p.replace(scene=p.scene.replace(boundaries=fd.Boundaries(
        x=fd.Open(), y=fd.Periodic(), z=fd.Periodic())))
    (issue,) = fd.check(p, numerics=NUMERICS).issues
    assert (issue.code, issue.path) == ('monitor_extent', 'outputs.padded.plane.x')
    assert issue.expected == (-.2, .2)
    assert issue.actual == span
