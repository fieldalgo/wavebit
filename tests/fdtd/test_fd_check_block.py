"""Stage 4's C5: what ``fd.check`` says about each smoothed structure.

One block per smoothed body: its support, how many cells across it is (the number the
cells-across door reads), its volume asked, in the box and held per lowered component, the
share past the box's faces, the voxels its surface cuts, what its cross coefficients cost and
how good their normal is, and the mirrors it is reflected in. The numbers come from the fills
the plan's notes take anyway. The oracles are closed forms: ``4 pi r^3/3``, the box's side
lengths, and a count of cut voxels from the voxel boundaries by hand.
"""
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._prepare.report import report as describe
from wavebit.fdtd._kernel.grid import OFFSETS

DX = 0.05


def _box(structures, size=1.2, dx=DX, boundaries=None, eps=2.25):
    return fd.Scene(domain=fd.Domain(size=(size,) * 3),
                    boundaries=boundaries or fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=4),
                    structures=tuple(fd.Structure(s, eps) for s in structures))


def _report(scene):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return describe(scene)


@pytest.mark.parametrize('centre', ((0.013, -0.007, 0.021), (0.0, 0.0, 0.0)))
def test_a_sphere_is_held_whole_on_every_component(centre):
    """Held against ``4 pi r^3/3`` on each of Ex, Ey, Ez, off a node and on one.

    Dropping the support's cell of margin loses 3.79e-3 of it on some components and not
    others at the off-node centre, 1.45e-2 on a node (``geometry.Sphere.support``), which is
    why the block reports the worst component and not one.
    """
    report = _report(_box((fd.Sphere(radius=0.25, center=centre),)))
    row, = report.bodies
    assert row.asked == pytest.approx(4 * math.pi * 0.25 ** 3 / 3, rel=1e-15)
    assert set(row.held) == {'Ex', 'Ey', 'Ez'}
    for value in row.held.values():
        assert abs(value / row.inside - 1) < 1e-12
    assert row.across == pytest.approx((10.0, 0.5), rel=1e-12)
    assert row.outside == () and row.normals == 'exact'
    text = str(report)
    assert 'body       structures[0] Sphere(radius=0.25' in text
    assert '  across   10.00 cells at its narrowest (0.5)' in text
    assert '  normals  exact moment (the sphere)' in text


def test_a_substrate_reports_what_the_box_holds_and_what_lies_outside():
    """``fd.Box(size=(3, 3, 0.3))`` in a 2.4 box: 2.7 asked, ``2.4 * 2.4 * 0.3`` held, 36.0% out."""
    report = _report(_box((fd.Box(size=(3.0, 3.0, 0.3)),), size=2.4, dx=0.1))
    row, = report.bodies
    assert row.asked == pytest.approx(2.7) and row.inside == pytest.approx(1.728)
    for value in row.held.values():
        assert value == pytest.approx(1.728, rel=1e-12)
    assert row.outside == ('x-', 'x+', 'y-', 'y+') and row.across is None
    text = str(report)
    assert '  outside  36.0% past x-, x+, y-, y+' in text
    assert '  across   a layer' in text
    assert '  cross    none' in text


def _full_voxels(lo, hi, low, n, offset):
    """Voxels of one component along one axis wholly inside ``[lo, hi]``, and those cut."""
    centres = low + (np.arange(-1, n + 1) + 0.5 * offset) * DX
    a, b = centres - DX / 2, centres + DX / 2
    inside = np.count_nonzero((a >= lo) & (b <= hi))
    touched = np.count_nonzero((b > lo) & (a < hi))
    return inside, touched


def test_the_cut_count_is_the_shell_of_voxels_the_faces_pass_through():
    """An unturned box with no face on a voxel boundary: touched minus whole, per component.

    Per axis the voxels the box touches and those it holds whole are counted from the voxel
    boundaries by hand; the shell is ``prod(touched) - prod(whole)``.
    """
    lo, hi = (-0.231, -0.172, -0.143), (0.219, 0.181, 0.207)
    shape = fd.Box(size=tuple(h - l for l, h in zip(lo, hi)),
                   center=tuple((h + l) / 2 for l, h in zip(lo, hi)))
    report = _report(_box((shape,)))
    row, = report.bodies
    for c, name in enumerate(('Ex', 'Ey', 'Ez')):
        whole, touched = 1, 1
        for a in range(3):
            w, t = _full_voxels(lo[a], hi[a], -0.6, 24, OFFSETS['E'][c][a])
            whole, touched = whole * w, touched * t
        assert row.cut[name] == touched - whole, name


def test_the_cross_terms_are_named_with_their_cost_and_their_normal():
    turned = _report(_box((fd.Box(size=(0.5, 0.4, 0.3), angle=30),)))
    row, = turned.bodies
    assert row.terms == ('constitutive[E0]',) and 0 < row.normals < 1
    text = str(turned)
    assert '  cross    constitutive[E0]: ' in text and 'MB params +' in text
    assert '  normals  nearest-surface proxy, max |n_c^2 - w_c| ' in text


def test_a_mirror_is_named_in_the_block():
    boundaries = fd.Boundaries(x=(fd.Symmetry(parity=1), fd.PEC()), y=fd.PEC(), z=fd.PEC())
    scene = fd.Scene(domain=fd.Domain(size=(0.6, 1.2, 1.2), center=(0.3, 0.0, 0.0)),
                     boundaries=boundaries, resolution=fd.Resolution(dx=DX),
                     run=fd.Run(steps=4),
                     structures=(fd.Structure(fd.Sphere(radius=0.15, center=(0.3, 0, 0)), 2.25),))
    assert '  mirror   x = 0 (x-): the scene is this and its image' in str(_report(scene))


def test_a_scene_without_a_smoothed_structure_prints_no_block():
    text = str(_report(_box((fd.Region(size=(0.4, 0.4, 0.4)),))))
    assert 'body ' not in text and '  support' not in text


def test_params_that_could_not_be_built_read_unknown_not_zero():
    """A traced material cannot be lowered on NumPy; the count is unknown, not 0.00 MB."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    out = {}

    def probe(eps):
        scene = _box((fd.Region(size=(0.4, 0.4, 0.4)),)).replace(
            structures=(fd.Structure(fd.Region(size=(0.4, 0.4, 0.4)), fd.Material(eps=eps)),))
        out['text'] = str(_report(scene))
        return eps
    jax.jit(probe)(jnp.asarray(2.25))
    assert 'params     unknown' in out['text'] and 'params     0.00 MB' not in out['text']


def test_every_body_reports_its_analytic_volume_and_holds_it():
    """``volume()`` against the hand formulas, and the held fill against it, per component."""
    hexagon = tuple((0.2 * math.cos(math.pi / 3 * i), 0.2 * math.sin(math.pi / 3 * i))
                    for i in range(6))
    for shape, truth in (
            (fd.Cylinder(radius=0.2, height=0.3, center=(0.013, -0.007, 0.021)),
             math.pi * 0.04 * 0.3),
            (fd.Prism(vertices=hexagon, height=0.3, axis='x', angle=10),
             1.5 * math.sqrt(3) * 0.2 ** 2 * 0.3),
            (fd.Box(size=(0.5, 0.4, 0.3), angle=30), 0.5 * 0.4 * 0.3)):
        row, = _report(_box((shape,))).bodies
        assert row.asked == pytest.approx(truth, rel=1e-14), shape.describe()
        for value in row.held.values():
            assert abs(value / truth - 1) < 1e-12, shape.describe()


def test_a_jitted_run_of_a_body_with_cross_terms_builds_its_report():
    """``fd.run`` builds its report from the params it lowered, which are tracers under jit.

    The block's byte count once read them with ``np.asarray`` and every smoothed body with a
    constitutive term -- a sphere, a turned box -- crashed under ``jax.jit(fd.run)`` (C5 review);
    it counts from shape and dtype now, as ``params_bytes`` does.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from tests.fixtures.fdtd.fd_helpers import band

    def energy(amplitude):
        scene = _box((fd.Sphere(radius=0.15),), size=0.8, dx=0.04).replace(
            run=fd.Run(steps=20),
            sources=(fd.Dipole('d', center=(0.24, 0.0, 0.0), component='Ey', band=band(),
                               amplitude=amplitude),))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return jnp.sum(jnp.asarray(fd.run(scene, 'jax').final.E) ** 2)
    eager = float(energy(1.0))
    assert float(jax.jit(energy)(1.0)) == pytest.approx(eager, rel=1e-12)


def _cover_holds(row):
    """Every lowered component holds the body's volume inside its own voxels, to round-off."""
    for name, value in row.held.items():
        assert abs(value / row.cover[name] - 1) < 1e-12, (name, value, row.cover[name])


@pytest.mark.parametrize('cx', (-1.0, 1.0))
def test_a_body_leaving_through_one_face_is_held_as_its_voxels_cover_it(cx):
    """A node component's voxels reach half a cell past a minus face and stop half a cell short
    of a plus one; against the box the block read +-2.9% on a correct lowering (the C5
    review), which looked like a missing support margin."""
    report = _report(_box((fd.Box(size=(3.0, 1.0, 0.5), center=(cx, 0.0, 0.0)),),
                          size=2.4, dx=0.1))
    row, = report.bodies
    _cover_holds(row)
    assert row.outside == ('x-',) if cx < 0 else ('x+',)
    line = next(l for l in str(report).splitlines() if l.startswith('  volume'))
    assert abs(float(line.split(' to ')[1].split(' of what')[0])) < 1e-12, line


@pytest.mark.parametrize('side', ('-', '+'))
def test_a_box_across_a_mirror_is_held_whole_and_nothing_is_outside(side):
    """At a mirror the scene is the body and its image: no end is passed, and every component
    holds what its voxels cover (the block read 12% off and '50.0% past x-')."""
    mirror = fd.Symmetry(parity=1)
    x = (mirror, fd.PEC()) if side == '-' else (fd.PEC(), mirror)
    plane = 0.0 if side == '-' else -0.025
    scene = fd.Scene(domain=fd.Domain(size=(0.6, 1.2, 1.2),
                                      center=(0.3 if side == '-' else -0.3, 0.0, 0.0)),
                     boundaries=fd.Boundaries(x=x, y=fd.PEC(), z=fd.PEC()),
                     resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
                     structures=(fd.Structure(fd.Box(size=(0.4, 0.3, 0.3),
                                                     center=(plane, 0.0, 0.0)), 2.25),))
    row, = _report(scene).bodies
    _cover_holds(row)
    assert row.outside == ()


def test_a_stencil_body_and_a_shared_term_are_named_as_they_are():
    """A stencil sphere's cross terms are the prepared operator, not 'none (an unturned box)';
    two bodies in one merged term each say it is shared."""
    ball = fd.Structure(fd.Sphere(radius=0.15), 2.25, discretisation='stencil')
    text = str(_report(_box(()).replace(structures=(ball,))))
    assert 'the prepared stencil operator' in text and 'unturned box' not in text
    pair = _report(_box((fd.Box(size=(0.3,) * 3, angle=30, center=(-0.27, 0, 0)),
                         fd.Box(size=(0.3,) * 3, angle=30, center=(0.27, 0, 0)))))
    first, second = pair.bodies
    assert first.terms == second.terms and first.shared[first.terms[0]] == (1,)
    assert 'shared with structures[1]' in str(pair) and 'shared with structures[0]' in str(pair)


class _NoVolume(fd.Box):
    """A body that declares its extent but answers no ``volume()``."""

    volume = None


def test_a_body_without_a_volume_names_the_faces_it_passes_without_a_share():
    text = str(_report(_box((_NoVolume(size=(3.0, 3.0, 0.3)),), size=2.4, dx=0.1)))
    assert '  outside  past x-, x+, y-, y+' in text and 'nan' not in text
