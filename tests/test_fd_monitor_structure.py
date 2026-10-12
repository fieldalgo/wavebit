"""C3: closed power surfaces cannot silently report a fragment of a structure."""
import warnings

import pytest
import wavebit.fd as fd


def scene(shape, monitor):
    return fd.Scene(domain=fd.Domain(size=(3.2,)*3),
                    boundaries=fd.Boundaries.open(fd.PML(cells=12, wavelength=1.)),
                    resolution=fd.Resolution(dx=.05), run=fd.Run(steps=2),
                    structures=(fd.Structure(shape, 2.25),), monitors=(monitor,))


def planned(s):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        return fd.plan(s)


@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('side', (-1, 1))
def test_closed_flux_refuses_each_face_cutting_a_sphere(axis, side):
    center = [0., 0., 0.]
    center[axis] = side*.35
    shape = fd.Sphere(radius=.2, center=tuple(center))
    monitor = fd.Flux('abs', box=fd.Region(size=(.8,)*3), wavelengths=(1.,))
    with pytest.raises(ValueError, match=r"Flux\('abs'\).*support of structures\[0\].*Sphere"):
        planned(scene(shape, monitor))


@pytest.mark.parametrize('center', ((0., 0., 0.), (.65, 0., 0.)))
def test_closed_flux_accepts_whole_inside_or_whole_outside(center):
    shape = fd.Sphere(radius=.1, center=center)
    monitor = fd.Flux('power', box=fd.Region(size=(.6,)*3), wavelengths=(1.,))
    assert planned(scene(shape, monitor))


def test_closed_flux_inside_a_structure_does_not_claim_whole_power():
    shape = fd.Region(size=(1.,)*3)
    monitor = fd.Flux('part', box=fd.Region(size=(.4,)*3), wavelengths=(1.,))
    with pytest.raises(ValueError, match='whole absorption or scattering'):
        planned(scene(shape, monitor))


def test_field_volume_may_sample_the_material_interior():
    shape = fd.Sphere(radius=.25)
    monitor = fd.Field('inside', box=fd.Region(size=(.2,)*3), wavelengths=(1.,))
    assert planned(scene(shape, monitor))


def test_farfield_already_requires_the_whole_structure_inside():
    shape = fd.Sphere(radius=.2, center=(.3, 0., 0.))
    monitor = fd.FarField('ff', box=fd.Region(size=(.6,)*3), wavelengths=(1.,))
    with pytest.raises(ValueError, match=r'FarField.*structures\[0\].*not strictly inside'):
        planned(scene(shape, monitor))
