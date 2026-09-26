"""Stage 4's C5: how few cells across a smoothed body may be.

Study S2b (``studies/fdtd_subpixel_cells_across.py``, ``docs/fd-subpixel-occupancy.md``) set
the numbers: under 4 cells across its smallest extent a smoothed body is refused, 4 to 8 warns.
The fill cannot see it -- every public body's fill is exact at any size -- so the door reads
``Shape.min_extent`` over the axes a body is bounded along, and a layer (uniform along two
axes) is exempt, because the rule is exact for a plane at any thickness. It runs ahead of the
constitutive energy certificate, whose message now says where refining the grid helps.
"""
import warnings

import pytest

import wavebit.fd as fd

DX = 0.05


def _box(shape, eps=2.25, size=0.8):
    return fd.Scene(domain=fd.Domain(size=(size,) * 3),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
                    structures=(fd.Structure(shape, eps),))


def _across_warnings(scene):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fd.plan(scene)
    return [str(w.message) for w in caught if issubclass(w.category, fd.CoarseMeshWarning)
            and 'cells across' in str(w.message)]


def _refusal(scene):
    with pytest.raises(ValueError) as caught:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fd.plan(scene)
    return str(caught.value)


def test_a_sphere_under_four_cells_across_is_refused_with_the_study_numbers():
    """Radius 0.0975 at dx 0.05 is 3.9 cells across."""
    message = _refusal(_box(fd.Sphere(radius=0.0975)))
    assert '3.90 cells across' in message and 'S2b' in message
    assert 'dx <= 0.04875' in message and 'dx <= 0.024375' in message


def test_four_to_eight_cells_across_warns_and_eight_is_quiet():
    message, = _across_warnings(_box(fd.Sphere(radius=0.1)))
    assert '4.0 cells across' in message and 'dx <= 0.025' in message
    assert _across_warnings(_box(fd.Sphere(radius=0.2))) == []


def test_a_rod_is_judged_by_its_section():
    """A cylinder running through z is a two-dimensional particle: 2.8 cells across."""
    rod = fd.Cylinder(radius=0.07, height=1.0, axis='z')
    assert '2.80 cells across' in _refusal(_box(rod))


def test_a_turned_box_is_judged_by_its_thinnest_side():
    """Bounded along all three axes, so not a layer: 0.15 thick is 3 cells."""
    slab = fd.Box(size=(0.4, 0.4, 0.15), angle=30)
    assert '3.00 cells across' in _refusal(_box(slab))


def test_a_layer_is_exempt_at_any_thickness():
    """Two cells thick, spanning x and y with a cell to spare: the rule is exact for a plane."""
    film = fd.Box(size=(1.0, 1.0, 0.1))
    assert _across_warnings(_box(film)) == []


def test_the_door_speaks_before_the_certificate():
    """At eps 20 the certificate fails every placement at every resolution S2b tried; a sphere
    3 cells across hears this door's reason, not the certificate's."""
    message = _refusal(_box(fd.Sphere(radius=0.075), eps=20.0))
    assert '3.00 cells across' in message and 'positive-energy' not in message


def test_the_certificate_no_longer_advises_refining_above_its_ceiling():
    """At eps 20 and 8 cells across the certificate refuses, and says refining does not help."""
    message = _refusal(_box(fd.Sphere(radius=0.2), eps=20.0))
    assert 'positive-energy certificate' in message and 'Gershgorin' in message
    assert 'refining the grid helps up to eps about 12 and not above it' in message
