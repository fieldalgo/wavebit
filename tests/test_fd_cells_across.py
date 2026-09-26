"""Stage 4's C5: how few cells across a smoothed body may be.

Study S2b (``studies/fdtd_subpixel_cells_across.py``, ``docs/fd-subpixel-occupancy.md``) set
the numbers: under 4 cells across its smallest extent a smoothed body is refused, 4 to 8 warns.
The fill cannot see it -- every public body's fill is exact at any size -- so the door reads
``Shape.min_extent`` over the axes a body is bounded along, and a layer (uniform along two
axes) is exempt, because the rule is exact for a plane at any thickness. It runs ahead of the
constitutive energy certificate, whose message now says where refining the grid helps.
"""
import math
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


def test_the_door_refuses_a_high_contrast_sphere_three_cells_across():
    """The door is about accuracy and holds at any contrast: at eps 20 a sphere 3 cells across
    is refused by name, though the lift would plan it stably."""
    message = _refusal(_box(fd.Sphere(radius=0.075), eps=20.0))
    assert '3.00 cells across' in message


def test_an_eps_20_sphere_eight_cells_across_plans():
    """S2b's Gershgorin certificate refused eps 20 at almost every placement and resolution;
    the per-block lift plans it, moving some of its blocks."""
    from wavebit.fd._lower import constitutive
    from wavebit.fd._lower.coefficients import build
    from wavebit.fd._runtime import runtime
    scene = _box(fd.Sphere(radius=0.2), eps=20.0)
    plan = fd.plan(scene)
    rt, nodes, lifted = runtime('numpy'), {}, {}
    params = dict(build(scene, plan, rt))
    params['constitutive'] = constitutive.lower(scene, plan, rt, nodes=nodes)
    assert constitutive.lift(plan, params, nodes, rt, lifted)
    assert sum(out.moved for out in lifted.values()) > 0


def _hexagon(flat):
    """A regular hexagon ``flat`` across its flats, counter-clockwise."""
    r = flat / 3 ** 0.5
    return tuple((r * math.cos(math.pi / 3 * i), r * math.sin(math.pi / 3 * i)) for i in range(6))


def _long_box(shape):
    return fd.Scene(domain=fd.Domain(size=(1.2, 0.6, 0.6)),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
                    structures=(fd.Structure(shape, 2.25),))


@pytest.mark.parametrize('cx', (0.05, 0.15, 0.2, 0.25, 0.3, 0.35, -0.25))
def test_a_body_exactly_four_cells_across_plans_wherever_it_sits(cx):
    """0.2 at dx 0.05 read 3.9999999999999996 cells at some centres and was refused there
    (the C5 review); it is 4 everywhere, and warns."""
    message, = _across_warnings(_long_box(fd.Box(size=(0.2, 0.2, 0.3), center=(cx, 0, 0))))
    assert '4.0 cells across' in message


def test_a_prism_is_measured_across_its_hull():
    """A regular hexagon by its flats: 4 cells warns (its hull width read 3.9999999999999996),
    8 is quiet, 3.8 is refused; and the width is the smallest altitude of a triangle."""
    grid = fd.plan(_box(fd.Sphere(radius=0.2))).grid
    assert fd.Prism(vertices=_hexagon(0.2), height=0.3).min_extent(grid) == pytest.approx(0.2)
    sliver = ((-0.25, -0.1), (0.25, -0.1), (0.2, 0.12))
    area = 0.5 * 0.5 * 0.22
    longest = math.hypot(0.45, 0.22)
    assert fd.Prism(vertices=sliver, height=0.3).min_extent(grid) == pytest.approx(
        2 * area / longest)
    message, = _across_warnings(_box(fd.Prism(vertices=_hexagon(0.2), height=0.3)))
    assert '4.0 cells across' in message
    assert _across_warnings(_box(fd.Prism(vertices=_hexagon(0.4), height=0.5))) == []
    assert '3.80 cells across' in _refusal(_box(fd.Prism(vertices=_hexagon(0.19), height=0.3)))


def _mirrored(shape, side='-', parity=-1, other=None):
    """A 0.6 box with a Symmetry face on x: ``x in [0, 0.6]`` for ``-``, ``[-0.6, 0]`` for ``+``."""
    mirror = fd.Symmetry(parity=parity)
    other = other or fd.PEC()
    x = (mirror, other) if side == '-' else (other, mirror)
    return fd.Scene(domain=fd.Domain(size=(0.6, 0.6, 0.6),
                                     center=(0.3 if side == '-' else -0.3, 0.0, 0.0)),
                    boundaries=fd.Boundaries(x=x, y=fd.PEC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
                    structures=(fd.Structure(shape, 2.25),))


def test_a_body_at_a_mirror_is_measured_as_the_scene_holds_it():
    """Across the plane, as declared; flush against it at the zeroing parity, with its image.

    A bar 7.4 cells long symmetric about a ``-`` plane read 3.7 on the box and was refused; a
    box 2 cells long flush against a ``parity=-1`` plane is 4 with its image and plans (0.09
    long, 3.6 whole, is refused, and told to declare it whole). A film symmetric about a ``+``
    plane, running past the image of the far wall, is a layer (the C5 review refused it as
    2 cells, from an extension a cell too long).
    """
    _quiet_plan(_mirrored(fd.Box(size=(0.37, 0.2, 0.2))))
    message, = _across_warnings(_mirrored(fd.Box(size=(0.1, 0.3, 0.3), center=(0.05, 0, 0))))
    assert '4.0 cells across' in message
    refused = _refusal(_mirrored(fd.Box(size=(0.09, 0.3, 0.3), center=(0.045, 0, 0))))
    assert '3.60 cells across' in refused and 'declared whole, symmetric about the plane' in refused
    film = fd.Box(size=(1.7, 1.0, 0.1), center=(-0.025, 0.0, 0.0))
    assert _across_warnings(_mirrored(film, side='+')) == []


def test_a_film_flush_against_a_wall_is_a_layer():
    """Flush against the x- wall, out through x+: still a film, still exempt.

    ``uniform_along`` asks a cell of margin past the x- face, so this read as bounded along x
    and was refused as 2 cells across (the C5 review). (Flush against a ``+`` wall is door 3's:
    the nodes' last voxel stops half a cell short of it.)
    """
    assert _across_warnings(_box(fd.Box(size=(0.9, 1.0, 0.1), center=(0.05, 0.0, 0.0)))) == []


def _quiet_plan(scene):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fd.plan(scene)
