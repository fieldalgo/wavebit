"""Stage 4's C4: the extruded bodies ``fd.Box``, ``fd.Cylinder`` and ``fd.Prism``.

What is new is not the geometry -- the kernel bodies under them (``src/wavebit/fd/_shapes``) carry
their own gates in ``tests/test_fd_shapes.py`` -- but the public door onto them: what each
answers of the ``fd.Shape`` protocol, above all :meth:`uniform_along`, which the periodic,
``Axis1D`` and absorbing-layer rules read; the bounding box its support is cut from, which
has to hold a turned section; the turn itself, in degrees and counter-clockwise about
``+axis``; and the refusals a static, finite, straight body makes at construction.

The oracles are independent of the code under test: an analytic volume (``a b c``,
``pi r^2 h``, the shoelace area times the height) for the lowered fill, and a prism whose
vertices the test turns by hand for the turn.
"""
import math
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._lower.occupancy import occupancy
from wavebit.fd._runtime import runtime

N, DX = 24, 0.05
CENTRE = (0.013, -0.007, 0.021)
HEXAGON = tuple((0.2 * math.cos(math.pi / 3 * i), 0.2 * math.sin(math.pi / 3 * i))
                for i in range(6))
#: A non-convex outline, counter-clockwise.
ELL = ((-0.15, -0.15), (0.15, -0.15), (0.15, -0.05), (-0.05, -0.05), (-0.05, 0.15),
       (-0.15, 0.15))
#: Long, thin and lopsided: turned 50 degrees, its section is where degrees read as radians
#: would put it nowhere near. (A clockwise turn moves its box by less than the support's cell
#: of margin, so that mutant is the L-shaped row's to catch, not this one's.) Its smallest
#: altitude is 0.22, 4.4 cells: thinner, C5's cells-across door refuses it (it was 0.11).
SLIVER = ((-0.25, -0.1), (0.25, -0.1), (0.2, 0.12))


def _area(vertices):
    return 0.5 * sum(u0 * v1 - u1 * v0 for (u0, v0), (u1, v1)
                     in zip(vertices, vertices[1:] + vertices[:1]))


def _scene(structures, n=N, dx=DX, boundaries=None):
    """A closed PEC box: no absorbing layer, no source, so nothing but geometry is under test."""
    return fd.Scene(
        domain=fd.Domain(size=(n * dx,) * 3),
        boundaries=boundaries or fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=4), structures=structures)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def _grid():
    return _quiet(fd.plan, _scene((fd.Structure(fd.Region(size=(0.1,) * 3), 2.0),))).grid


VOLUMES = {
    'box': (fd.Box(size=(0.3, 0.2, 0.25), center=CENTRE), 0.3 * 0.2 * 0.25),
    'box with its faces on node planes': (fd.Box(size=(0.3, 0.2, 0.3)), 0.3 * 0.2 * 0.3),
    'box turned 30 degrees about y': (fd.Box(size=(0.3, 0.2, 0.25), center=CENTRE, angle=30,
                                             axis='y'), 0.3 * 0.2 * 0.25),
    'cylinder along z': (fd.Cylinder(radius=0.2, height=0.3, center=CENTRE),
                         math.pi * 0.04 * 0.3),
    'cylinder along x': (fd.Cylinder(radius=0.15, height=0.4, axis='x', center=CENTRE),
                         math.pi * 0.0225 * 0.4),
    'hexagonal prism along x, turned 10': (fd.Prism(vertices=HEXAGON, height=0.3, axis='x',
                                                    center=CENTRE, angle=10),
                                           _area(HEXAGON) * 0.3),
    'L-shaped prism along y, turned -25': (fd.Prism(vertices=ELL, height=0.3, axis='y',
                                                    center=CENTRE, angle=-25),
                                           _area(ELL) * 0.3),
    'sliver turned 50': (fd.Prism(vertices=SLIVER, height=0.25, center=CENTRE, angle=50),
                         _area(SLIVER) * 0.25),
}


@pytest.mark.parametrize('name', VOLUMES)
def test_the_lowered_fill_is_the_bodys_volume(name):
    """Every component's fill sums to the analytic volume, so the support holds the whole body.

    The fill is the kernel's exact clip, so the only way to lose volume is a support box that
    does not reach every voxel the body touches: a bounding box cut from the section the wrong
    way (the turned rows), or one without the cell of margin a node-centred component needs
    (the box whose faces sit on node planes -- the placement :meth:`fd.Sphere.support`
    measured as the worst).

    Mutants, each alone, measured 2026-09-26 over this file and the five door and occupancy
    files beside it: the support without its cell of margin fails all eight rows; the bounding
    box turned clockwise fails the L-shaped row (a rectangle's box is the same either way, so
    the turned-box row cannot see it); ``fd.Prism`` handing the kernel degrees fails the
    L-shaped and sliver rows, and the turn test.
    """
    body, truth = VOLUMES[name]
    scene = _scene((fd.Structure(body, fd.Material(eps=2.25)),))
    plan = _quiet(fd.plan, scene)
    occ = occupancy(scene, plan, runtime('numpy'), fields=('E', 'H'))
    for field in ('E', 'H'):
        for c in range(3):
            volume = float(np.sum(np.asarray(occ[0][field][c][0]))) * plan.dx ** 3
            assert volume == pytest.approx(truth, rel=1e-12), (field, c, volume, truth)


def test_a_turn_is_counter_clockwise_about_its_axis_and_in_degrees():
    """A turned body is the prism whose vertices the test turned by hand, voxel for voxel.

    The volume cannot see a turn -- it is the same at every angle -- so this compares fills.
    ``fd.Box(angle=30)`` is the rectangle's corners turned 30 degrees counter-clockwise in
    ``(u, v)`` and handed to an unturned ``fd.Prism``, and so is ``fd.Prism(angle=30)`` of the
    unturned corners; on every axis, since ``(u, v)`` is the right-handed pair across it. And
    a quarter turn is the box with its sides swapped, up to the rounding of ``cos(pi/2)``.

    Mutants, each alone, measured 2026-09-26: ``fd.Box`` handing the kernel degrees fails this
    case alone; ``fd.Prism`` doing the same fails it and two volume rows.
    """
    grid = _grid()
    for axis in ('x', 'y', 'z'):
        a = 'xyz'.index(axis)
        u, v = ((1, 2), (2, 0), (0, 1))[a]
        size = [0.3, 0.3, 0.3]
        size[u], size[v] = 0.32, 0.18
        box = fd.Box(size=tuple(size), center=CENTRE, angle=30, axis=axis)
        t = math.radians(30)
        corners = ((-0.16, -0.09), (0.16, -0.09), (0.16, 0.09), (-0.16, 0.09))
        turned = tuple((p * math.cos(t) - q * math.sin(t), p * math.sin(t) + q * math.cos(t))
                       for p, q in corners)
        prism = fd.Prism(vertices=turned, height=0.3, axis=axis, center=CENTRE)
        turned_prism = fd.Prism(vertices=corners, height=0.3, axis=axis, center=CENTRE, angle=30)
        quarter = fd.Box(size=tuple(size), center=CENTRE, angle=90, axis=axis)
        swapped = list(size)
        swapped[u], swapped[v] = size[v], size[u]
        swapped = fd.Box(size=tuple(swapped), center=CENTRE, axis=axis)
        for one, other, tol in ((box, prism, 1e-14), (turned_prism, prism, 1e-14),
                                (quarter, swapped, 1e-13)):
            for field in ('E', 'H'):
                got = _fills(one, grid, field)
                want = _fills(other, grid, field)
                for g, w in zip(got, want):
                    assert np.max(np.abs(g - w)) <= tol, (axis, one.describe(), field)
        assert np.max(np.abs(_fills(box, grid, 'E')[0] - _fills(swapped, grid, 'E')[0])) > .1


def _fills(shape, grid, field):
    """The fill of ``shape`` at each component's Yee points over the whole grid."""
    from wavebit.fd._lower.occupancy import component_occupancy
    box = tuple((0, n) for n in grid.n)
    return [np.asarray(component_occupancy(shape.body(), grid, box, field, c,
                                           runtime('numpy'))[0]) for c in range(3)]


def test_a_body_is_uniform_along_its_own_axis_where_it_spans_it_and_nowhere_else():
    """``uniform_along`` -- the answer the periodic, ``Axis1D`` and absorbing-layer rules read.

    Along its own axis a body spans a range of layers when its extent covers every voxel of
    them, a cell beyond each end; across the axis only an unturned box does, because only its
    faces are planes of the grid's own axes. Where the body cannot reach the layers at all it
    is uniform there trivially, as a ball is.

    Mutants, each alone, measured 2026-09-26: answering True along a straight axis without
    asking whether the body spans it fails this case and the wrapping-face one; a turned box
    counted uniform across its axis fails this case alone, on the strip; spanning with no
    margin at all fails it alone, on the flush rod. Spanning with half a cell of margin
    instead of a whole one fails :func:`test_a_rod_that_spans_a_period_lowers_every_layer_alike`
    (an earlier writing of this docstring called it equivalent, which it is for the fill and
    not for the normal -- see that test).
    """
    grid = _grid()
    full = [(0, n) for n in grid.n]
    side = grid.n[0] * DX
    rod = fd.Cylinder(radius=0.1, height=side + 3 * DX, axis='x')
    assert rod.uniform_along(0, full[0], grid)
    assert not rod.uniform_along(1, full[1], grid) and not rod.uniform_along(2, full[2], grid)
    # Flush with both faces is not spanning them: the node-centred voxels at the faces reach
    # half a cell beyond, and the test asks a whole cell.
    flush = fd.Cylinder(radius=0.1, height=side, axis='x')
    assert not flush.uniform_along(0, full[0], grid)
    short = fd.Cylinder(radius=0.1, height=0.4, axis='x')
    assert not short.uniform_along(0, full[0], grid)
    assert short.uniform_along(0, (0, 2), grid)          # layers it cannot reach
    slab = fd.Box(size=(side + 3 * DX, side + 3 * DX, 0.2))
    assert slab.uniform_along(0, full[0], grid) and slab.uniform_along(1, full[1], grid)
    assert not slab.uniform_along(2, full[2], grid)
    turned = fd.Box(size=(0.3, 0.2, side + 3 * DX), angle=20)
    assert turned.uniform_along(2, full[2], grid)
    assert not turned.uniform_along(0, full[0], grid) and not turned.uniform_along(1, full[1], grid)
    # A turned strip whose bounding box spans x is still a strip that climbs in y along x.
    strip = fd.Box(size=(3.0, 0.2, 0.2), angle=20)
    assert strip.low[0] < -side and strip.high[0] > side
    assert not strip.uniform_along(0, full[0], grid)
    prism = fd.Prism(vertices=HEXAGON, height=side + 3 * DX, axis='y')
    assert prism.uniform_along(1, full[1], grid) and not prism.uniform_along(0, full[0], grid)


def _periodic(structure, boundaries=None):
    return _scene((structure,), boundaries=boundaries or fd.Boundaries(
        x=fd.Periodic(), y=fd.Periodic(), z=fd.PEC()))


def test_a_body_crosses_a_wrapping_face_only_where_it_is_uniform_along_it():
    """A pillar running the whole period is carried; a body that changes across the face is not.

    Before C4 every smoothed shape that reached a ``Periodic``, ``Bloch`` or ``Axis1D`` face
    was refused, because resolving the part that wraps needs the image ring and that fold is
    not written. A body uniform along the axis over the whole domain needs no fold -- every
    layer already holds the section -- so the door now reads ``uniform_along`` first, and a
    rod along a periodic ``x`` plans. The same rod stopping short of the faces, a rod along
    ``z`` that crosses the ``x`` face, and a ball on an ``Axis1D`` axis are still refused, each
    by its own sentence. A cylinder along the ``Axis1D`` axis, spanning it, is the invariant
    cross-section that axis stands for, and it plans.

    Mutants, each alone, measured 2026-09-26: the periodic door ignoring ``uniform_along``, and
    the ``Axis1D`` door ignoring it, each fail this case alone.
    """
    side = N * DX
    glass = fd.Material(eps=2.25)
    rod = fd.Structure(fd.Cylinder(radius=0.15, height=side + 3 * DX, axis='x'), glass)
    _quiet(fd.plan, _periodic(rod))
    crossing = fd.Structure(fd.Cylinder(radius=0.15, height=0.3, center=(0.55, 0.0, 0.0)),
                            glass)
    with pytest.raises(ValueError, match='reaches the Periodic x face'):
        _quiet(fd.plan, _periodic(crossing))
    short = fd.Structure(fd.Cylinder(radius=0.15, height=side - DX, axis='x'), glass)
    with pytest.raises(ValueError, match='reaches the Periodic x face'):
        _quiet(fd.plan, _periodic(short))

    def invariant(structure):
        return fd.Scene(
            domain=fd.Domain(size=(side, DX, side)),
            boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Axis1D(k=0.3, k_max=1.0),
                                     z=fd.PEC()),
            resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4), structures=(structure,))
    _quiet(fd.plan, invariant(fd.Structure(
        fd.Cylinder(radius=0.15, height=1.0, axis='y'), glass)))
    with pytest.raises(ValueError, match='fd.Axis1D face'):
        _quiet(fd.plan, invariant(fd.Structure(fd.Sphere(radius=0.2), glass)))
    with pytest.raises(ValueError, match='fd.Axis1D face'):
        _quiet(fd.plan, invariant(fd.Structure(
            fd.Cylinder(radius=0.15, height=0.4, axis='z'), glass)))


def test_a_region_door_names_the_body_box():
    """``fd.Box`` is the body now; a monitor's or a TFSF's box is ``fd.Region``, and says so.

    Every one of these was ``fd.Box`` until C4, so the likeliest mistake after the rename is
    the old spelling, and it has to be refused with the name that works rather than with the
    type-mismatch sentence a stranger's object gets.

    Mutant, measured 2026-09-26: the monitor door's body-box sentence removed fails this case
    alone (the TFSF and FarField doors are the same three lines).
    """
    body = fd.Box(size=(0.2,) * 3)
    for make in (lambda: fd.Flux('f', box=body, wavelengths=(1.0,)),
                 lambda: fd.FarField('ff', box=body, wavelengths=(1.0,)),
                 lambda: fd.TFSF('t', box=body, direction='-z', polarization='x',
                                 band=fd.Band(wavelength=1.0, df=0.1))):
        with pytest.raises(TypeError, match='fd.Box is the smoothed body') as e:
            make()
        assert 'fd.Region(size=' in str(e.value)


def test_a_body_is_static_finite_and_straight_or_refused_at_construction():
    """Refused where it is spelled, by name: the constructors run the checks, not ``fd.plan``.

    A tilted axis names stage 5; a NaN, an infinity, a length that is not positive, a triple
    that is not three, a traced value, a cylinder given an ``angle`` (a circle turned about its
    own axis is the same circle, and a key whose gradient can only be zero is refused), and an
    outline that is clockwise, crosses itself or has two vertices. The last three are the
    kernel's own ``polygon_shape``, asked at construction so that it is not asked again on
    every ``body()`` call.

    Mutants, each alone, measured 2026-09-26: ``fd.Prism`` without the outline check at
    construction, and a NaN passing as finite, each fail this case alone. The traced value is
    its own test below, so that these run where Torch is not installed.
    """
    for make, match in (
            (lambda: fd.Box(size=(1, 1, 1), axis=(0, 0, 1)), 'stage 5'),
            (lambda: fd.Cylinder(radius=0.1, height=1, axis='w'), 'stage 5'),
            (lambda: fd.Box(size=(1, 1, 1), center=(0, math.nan, 0)), 'finite'),
            (lambda: fd.Box(size=(1, 1, 1), angle=math.inf), 'finite'),
            (lambda: fd.Box(size=(1, -1, 1)), 'positive length'),
            (lambda: fd.Box(size=(1, 1)), 'three lengths'),
            (lambda: fd.Cylinder(radius=0.0, height=1), 'positive length'),
            (lambda: fd.Prism(vertices=HEXAGON, height=1, center=(0, 0)), 'triple'),
            (lambda: fd.Prism(vertices=((0, 0), (1, 0)), height=1), 'three vertices'),
            (lambda: fd.Prism(vertices=((0, 0), (0, 1), (1, 0)), height=1), 'counter-clockwise'),
            (lambda: fd.Prism(vertices=((0, 0), (1, 1), (1, 0), (0, 1)), height=1), 'simple'),
            (lambda: fd.Prism(vertices=((0, 0), (1, 0), (2, 0)), height=1), 'no area')):
        with pytest.raises(ValueError, match=match):
            make()
    with pytest.raises(TypeError, match="does not take 'angle'"):
        fd.Cylinder(radius=0.1, height=1, angle=10)
    # An array is a spelling of the same values, not a refusal.
    assert fd.Box(size=np.array([0.3, 0.2, 0.1])).size == (0.3, 0.2, 0.1)
    assert fd.Prism(vertices=np.array(((0, 0), (1, 0), (0, 1.0))), height=1).vertices[2] == (0.0, 1.0)
    with pytest.raises(ValueError, match=r'size\[1\]=0.0'):
        fd.Box(size=(1, 0, 1))


def test_a_traced_value_is_refused_at_construction():
    """``range=`` is C6; until then a traced radius is static and refused, by name."""
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError, match='is static'):
        fd.Cylinder(radius=torch.tensor(0.1, requires_grad=True), height=1)


def test_each_body_describes_itself_and_labels_its_rows():
    """``describe`` is the body's own spelling; the report rows carry a label per body.

    ``box.x``/``box.y``/``box.z`` are ``fd.Region``'s, frozen in the stage-3 archive, so the
    body box's rows are ``cuboid.*``: a row a reader cannot tell apart from a snapped one would
    undo the point of printing asked beside got.
    """
    grid = _grid()
    box = fd.Box(size=(0.3, 0.2, 0.25), center=CENTRE, angle=30)
    assert box.describe() == ("Box(size=(0.3, 0.2, 0.25), center=(0.013, -0.007, 0.021), "
                              "angle=30.0, axis='z')")
    assert fd.Box(size=(0.3, 0.2, 0.25)).describe() == 'Box(size=(0.3, 0.2, 0.25), center=(0.0, 0.0, 0.0))'
    rod = fd.Cylinder(radius=0.1, height=0.4, axis='x')
    assert rod.describe() == "Cylinder(radius=0.1, height=0.4, axis='x', center=(0.0, 0.0, 0.0))"
    for shape, label in ((box, 'cuboid'), (rod, 'cylinder'),
                         (fd.Prism(vertices=HEXAGON, height=0.3), 'prism')):
        rows = shape.span_rows(grid)
        assert [r[0] for r in rows] == [f'{label}.{a}' for a in 'xyz']
        assert all(r[1:3] == r[3:5] for r in rows)
        assert [r[1] for r in rows] == list(shape.low)
    assert rod.low == (-0.2, -0.1, -0.1) and rod.high == (0.2, 0.1, 0.1)


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_three_backends_lower_a_turned_body_alike(backend):
    """The coefficient arrays of a turned prism and a cylinder, on each backend, against NumPy.

    One lowering body serves all three; what differs is only each library's own ``sqrt`` and
    trigonometry, so the arrays agree to rounding: ``be`` (measured 1.1e-16 and 3.9e-15), and
    the constitutive cross array, where the body's own surface normals enter (measured 2.7e-15
    on JAX and 2.2e-17 on Torch for the prism) -- the one array no other test compares across
    backends for these bodies.
    """
    rt = runtime(backend)
    for shape in (fd.Prism(vertices=HEXAGON, height=0.3, axis='x', center=CENTRE, angle=10),
                  fd.Cylinder(radius=0.15, height=0.3, axis='y', center=CENTRE)):
        scene = _scene((fd.Structure(shape, fd.Material(eps=2.25)),))
        plan = _quiet(fd.plan, scene)
        want = _quiet(fd.lower, scene, plan, 'numpy')
        got = _quiet(fd.lower, scene, plan, backend)
        pairs = [(np.asarray(rt.numpy(got['be'])), np.asarray(want['be']))]
        assert set(got['constitutive']) == set(want['constitutive']) != set()
        pairs += [(np.asarray(rt.numpy(got['constitutive'][k])), np.asarray(want['constitutive'][k]))
                  for k in want['constitutive']]
        for g, w in pairs:
            assert np.max(np.abs(w)) > 0
            assert np.max(np.abs(g - w)) <= 1e-13 * np.max(np.abs(w)), shape.describe()


def test_the_stencil_is_the_spheres_and_refuses_a_body_by_name():
    """``discretisation='stencil'`` needs a ball's exact edge and dual-face fills; a body asks
    the voxel rule, and is refused by the stencil's own name rather than given it quietly."""
    for shape in (fd.Box(size=(0.3, 0.2, 0.25)), fd.Cylinder(radius=0.15, height=0.3),
                  fd.Prism(vertices=HEXAGON, height=0.3)):
        with pytest.raises(ValueError, match='StencilShapeRefused'):
            _quiet(fd.plan, _scene((fd.Structure(shape, 2.25, discretisation='stencil'),)))


def test_a_rod_that_spans_a_period_lowers_every_layer_alike():
    """Along a periodic axis a spanning rod's face layer is its every other layer, to round-off.

    What ``uniform_along`` promises the periodic door is that no layer along the axis differs,
    so that no image ring is needed. The fill alone would allow half a cell of margin beyond
    each face -- the voxels reach half a cell out -- but the moment and the cross terms read the
    **nearest** surface, and a voxel on the rim can sit up to about 0.71 of a cell inside the
    side: with a cap nearer than that, the face layer takes the cap's normal. Found in review:
    with half a cell of margin a rod whose caps sit 0.52 of a cell past each face was carried,
    and its first layer differed from its middle one by 1.6e-3 in ``be`` and 7.9e-4 in the
    cross array. So the margin is a whole cell, a rod 0.55 of a cell past each face is refused,
    and one 1.05 cells past lowers every layer alike.

    Mutant, measured 2026-09-26: the spanning test asking half a cell instead of a whole one
    fails this case.
    """
    side = N * DX
    periodic = fd.Boundaries(x=fd.Periodic(), y=fd.PEC(), z=fd.PEC())

    def rod(past):
        return fd.Structure(fd.Cylinder(radius=0.17, height=side + 2 * past * DX, axis='x',
                                        center=(0.0, 0.013, -0.021)), fd.Material(eps=2.25))
    with pytest.raises(ValueError, match='reaches the Periodic x face'):
        _quiet(fd.plan, _scene((rod(0.55),), boundaries=periodic))
    scene = _scene((rod(1.05),), boundaries=periodic)
    plan = _quiet(fd.plan, scene)
    params = _quiet(fd.lower, scene, plan, 'numpy')
    arrays = [np.asarray(params['be'])] + [np.asarray(v) for v in params['constitutive'].values()]
    assert len(arrays) == 2
    for a in arrays:
        middle = a[..., a.shape[-1] // 2]
        assert np.max(np.abs(middle - a[..., 0])) < 1e-14 and np.max(np.abs(middle - a[..., -1])) < 1e-14
        assert np.ptp(middle) > 0.1 * np.max(np.abs(middle))


def test_a_plane_beside_a_smoothed_film_reads_the_medium_its_voxels_hold():
    """A diffraction plane under a smoothed ``fd.Box`` film: air where its layers are air.

    ``medium_at`` read "this structure fills the layer" off the shape's support and its
    uniformity across the plane -- exact for the snapped shapes, whose support is their filled
    cells, and wrong for a smoothed one, whose support carries a cell of margin and the
    fractional face cell. Found in review: a plane 0.6 of a cell under a smoothed film planned
    with the film's ``eps`` 4 -- its orders de-embedded in glass that is not there -- and a
    plane 1.6 cells under it was refused as "an interface on the plane". Now a smoothed shape
    is asked by its own fill (``_post.layer_share``): the plane whose layer holds part of the
    film's face voxels is refused, the one a cell and a half clear reads air, and a plane inside
    reads the film.

    Mutant, measured 2026-09-26: a smoothed shape answering "full" wherever its support
    reaches the layer (the old reading) fails this case.
    """
    from wavebit.fd._post import medium_at
    dx, period = 0.025, 0.8

    def scene(film, z):
        return fd.Scene(
            domain=fd.Domain(size=(period, period, 2.0)),
            boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                     z=fd.PML(cells=12, wavelength=1.0)),
            resolution=fd.Resolution(dx=dx), run=fd.Run(steps=4),
            structures=(fd.Structure(film, 4.0),),
            sources=(fd.PlaneWave('pw', z=0.5, direction='-z', polarization='s',
                                  band=fd.Band(wavelength=1.0, df=0.1)),),
            monitors=(fd.Diffraction('t', z=z, wavelengths=(1.0,)),))

    def medium(film, z):
        s = scene(film, z)
        plan = _quiet(fd.plan, s)
        k = round((z - plan.grid.low[2]) / dx)
        return medium_at(s, plan.grid, 2, (k - 1, k), 'probe')
    size = (period + 3 * dx, period + 3 * dx, 0.2)
    box = fd.Box(size=size, center=(0.0, 0.0, -0.11))
    region = fd.Region(size=size, center=(0.0, 0.0, -0.11))
    with pytest.raises(ValueError, match='is not one medium'):
        medium(box, -0.225)
    assert medium(box, -0.25) == medium(region, -0.25) == medium(region, -0.225) == (1.0, 1.0)
    assert medium(box, -0.1) == medium(region, -0.1) == (4.0, 1.0)


def test_an_unturned_box_plans_no_cross_term_and_takes_an_initial_field():
    """Its cross coefficients are zero everywhere, so the plan gives it no constitutive term.

    A ``ConstitutiveTerm`` adds only the off-diagonal increment; over an axis-aligned box it
    would add exactly nothing and hold a region of flux memory to do it. Without one, an
    initial field is not refused over it -- the refusal is about a coupled region's flux memory
    -- which is what lets the Airy gate (G5) spell its slab as a box. A turned box has one.

    Mutant, measured 2026-09-26: ``constitutive.cross_free`` answering False fails this case.
    """
    from wavebit.fd._kernel.constitutive import ConstitutiveTerm

    def coupled(shape, initial=None):
        scene = _scene((fd.Structure(shape, fd.Material(eps=2.25)),))
        if initial is not None:
            scene = scene.replace(initial=initial)
        return [t for t in _quiet(fd.plan, scene).terms if isinstance(t, ConstitutiveTerm)]
    assert coupled(fd.Box(size=(0.3, 0.2, 0.25), center=CENTRE)) == []
    assert len(coupled(fd.Box(size=(0.3, 0.2, 0.25), center=CENTRE, angle=30))) == 1
    grid = _grid()
    field = np.zeros((3,) + tuple(reversed(grid.n)))
    field[1, N // 2, N // 2, N // 2] = 1.0
    initial = fd.InitialField(E=field)
    assert coupled(fd.Box(size=(0.3, 0.2, 0.25), center=CENTRE), initial) == []
    with pytest.raises(ValueError, match='InitialField'):
        coupled(fd.Box(size=(0.3, 0.2, 0.25), center=CENTRE, angle=30), initial)


def test_min_extent_measures_what_reaches_into_the_box():
    """A body leaving through one face is measured inside, as ``fd.Region`` is (the contract)."""
    grid = _grid()
    side = N * DX
    through = fd.Cylinder(radius=0.1, height=side + 3 * DX, axis='x')
    assert through.min_extent(grid) == pytest.approx(0.2)
    leaving = fd.Cylinder(radius=0.3, height=0.4, axis='x', center=(side / 2, 0.0, 0.0))
    assert leaving.min_extent(grid) == pytest.approx(0.2)            # 0.2 of its 0.4 is inside
    film = fd.Box(size=(side + 3 * DX, side + 3 * DX, 0.1))
    assert film.min_extent(grid) == pytest.approx(0.1)
    assert fd.Box(size=(0.3, 0.2, 0.25), angle=30).min_extent(grid) == pytest.approx(0.2)
