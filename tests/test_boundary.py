"""
Verification tests for the BOUNDARY path -- painting resolved into polygons, transformed exactly.

Covers: the polygon transform against an independent area/DFT check, the clipper's resolved
regions against the structure they are supposed to describe, the aliasing floor on the normal
grid, cross-backend agreement, and differentiability of the whole chain from shape parameters
to convolution matrices.

Two facts these tests exist to pin down, both learned the hard way:

* The **normal field is discontinuous on the medial axis**, so a *symmetric* shape sits exactly
  on its own discontinuity -- a square's 45-degree medial axis passes through the pixel centres
  of a square grid, those pixels are exact ``argmin`` ties, and finite differences there diverge
  like 1/h while the analytic gradient is the correct one-sided value. Every gradient test here
  therefore uses asymmetric shapes on purpose.
* The sampled normal field needs **4*truncation_order+1 points per axis** or distinct orders alias onto one
  bin, and the damage is structural rather than gradual (a constant normal field stops
  transforming to the identity, so NVF no longer collapses onto Li's rules).
"""
import os

import numpy as np
import pytest

os.environ.setdefault('JAX_ENABLE_X64', '1')

import array_api_compat.numpy as xnp
import wavebit
from wavebit._core import boundary as _b
from wavebit._core import clipper as _c
from wavebit._core.polygon import polygon_ft, polygon_area, rect_verts, ellipse_verts

PERIOD = (1.0, 1.0)


# --------------------------------------------------------------------------- helpers
def resolve(xp, params, plan=None):
    """Plan with NumPy on concrete values, then replay against (possibly traced) vertices."""
    verts = [rect_verts(xp, *p[:5]) for p in params]
    if plan is None:
        concrete = [np.asarray(rect_verts(np, *p[:5]), float) for p in params]
        plan = _c.resolve_painting(concrete)
    recs, regions = plan
    vals = _c.eval_table(xp, recs, verts)
    regs = [(p[5], [_c.loop_verts(xp, lp, vals) for lp in region])
            for p, region in zip(params, regions)]
    return regs, plan


#: Slot 5 is the material -- a permittivity since the flip, so 4 and 9 where the pre-flip
#: file wrote the indices 2 and 3. Nothing here reads it as physics; it labels the region.
TWO_RODS = [(0.40, 0.43, 0.36, 0.29, 0.17, 4.0),
            (0.57, 0.55, 0.31, 0.38, 0.41, 9.0)]


# --------------------------------------------------------------------------- polygon FT
def test_polygon_ft_dc_is_the_area():
    """The k=0 coefficient is the area over the cell -- the branch the 1/|k|^2 form cannot take."""
    V = rect_verts(xnp, 0.47, 0.53, 0.42, 0.31, 0.4)
    k0 = xnp.zeros((1, 1))
    got = complex(polygon_ft(xnp, V, k0, k0, PERIOD)[0, 0])
    assert abs(got.real - 0.42 * 0.31) < 1e-15
    assert abs(got.imag) < 1e-15


def test_polygon_ft_exact_for_a_rotated_rectangle():
    """A rectangle is a polygon, so the transform is exact at any angle, not merely convergent."""
    ky = xnp.reshape(xnp.asarray([float(k) for k in range(-4, 5)]), (-1, 1)) * xnp.ones((1, 9))
    kx = xnp.ones((9, 1)) * xnp.reshape(xnp.asarray([float(k) for k in range(-4, 5)]), (1, -1))
    lx, ly = 0.42, 0.31
    for angle in (0.0, 0.1, 0.4, 0.9, 1.3):
        V = rect_verts(xnp, 0.5, 0.5, lx, ly, angle)
        got = polygon_ft(xnp, V, ky, kx, PERIOD)
        # closed form for a centred rotated rectangle: sinc x sinc in the shape's own frame
        c, s = np.cos(angle), np.sin(angle)
        u = np.asarray(kx) * c + np.asarray(ky) * s
        v = -np.asarray(kx) * s + np.asarray(ky) * c
        want = (lx * ly * np.sinc(u * lx) * np.sinc(v * ly)
                * np.exp(-2j * np.pi * 0.5 * (np.asarray(kx) + np.asarray(ky))))
        assert np.abs(np.asarray(got) - want).max() < 1e-14, angle


def test_polygon_ft_hole_subtracts_itself():
    """A reversed loop is a hole; no branch anywhere has to know about it."""
    outer = rect_verts(xnp, 0.5, 0.5, 0.6, 0.6, 0.0)
    inner = rect_verts(xnp, 0.5, 0.5, 0.2, 0.2, 0.0)
    k0 = xnp.zeros((1, 1))
    ring = (polygon_ft(xnp, outer, k0, k0, PERIOD)
            + polygon_ft(xnp, inner[::-1], k0, k0, PERIOD))
    assert abs(complex(ring[0, 0]).real - (0.36 - 0.04)) < 1e-15


# --------------------------------------------------------------------------- the clipper
def test_resolved_regions_reproduce_the_painted_area():
    """Painting order is the whole point: the later rod wins, and the areas have to say so."""
    regs, _ = resolve(xnp, TWO_RODS)
    areas = {n: sum(float(polygon_area(xnp, V)) for V in loops) for n, loops in regs}
    total = sum(areas.values())
    a0, a1 = 0.36 * 0.29, 0.31 * 0.38
    assert abs(areas[9.0] - a1) < 1e-12          # the later rod is untouched
    assert areas[4.0] < a0 - 1e-3                 # the earlier one lost the overlap
    assert total < a0 + a1                        # and the union is less than the sum
    assert abs(total - (a0 + a1 - (a0 + a1 - total))) < 1e-12


def test_resolved_regions_are_disjoint_on_a_fine_raster():
    """No pixel may belong to two regions -- that is what 'resolved' means."""
    regs, _ = resolve(xnp, TWO_RODS)
    n = 400
    g = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(g, g)
    hits = np.zeros((n, n), dtype=int)
    for _, loops in regs:
        inside = np.zeros((n, n), dtype=bool)
        for V in loops:
            V = np.asarray(V)
            wind = np.zeros((n, n))
            for i in range(len(V)):
                x0, y0 = V[i]
                x1, y1 = V[(i + 1) % len(V)]
                cond = ((y0 <= Y) != (y1 <= Y))
                t = np.where(cond, (Y - y0) / np.where(y1 == y0, 1.0, y1 - y0), 0.0)
                xint = x0 + t * (x1 - x0)
                wind += np.where(cond & (xint > X), np.sign(y1 - y0), 0.0)
            inside ^= (wind != 0)
        hits += inside.astype(int)
    assert hits.max() <= 1, f'{int((hits > 1).sum())} pixels claimed twice'


def test_plan_replays_against_moved_vertices():
    """The plan holds indices, not coordinates, so it survives the shapes moving."""
    regs, plan = resolve(xnp, TWO_RODS)
    moved = [(p[0] + 0.01, p[1] - 0.005) + p[2:] for p in TWO_RODS]
    regs2, _ = resolve(xnp, moved, plan)
    a1 = sum(float(polygon_area(xnp, V)) for _, loops in regs for V in loops)
    a2 = sum(float(polygon_area(xnp, V)) for _, loops in regs2 for V in loops)
    assert a1 != a2                       # it really did move
    assert abs(a2 - a1) < 0.02            # and only by about the nudge


# --------------------------------------------------------------------------- normal field
@pytest.mark.parametrize('truncation_order', [3, 6, 12])
def test_normal_grid_floor_is_enforced(truncation_order):
    """Below 4*truncation_order+1 the sampled field aliases; that must raise, not silently mis-transform."""
    need = 4 * truncation_order + 1
    with pytest.raises(ValueError, match='aliases'):
        _b.check_normal_grid((need - 1, need - 1), truncation_order, truncation_order)
    _b.check_normal_grid((need, need), truncation_order, truncation_order)


def test_normal_field_is_the_edge_normal_beside_an_edge():
    """Far from any corner the field must be exactly that edge's own normal."""
    V = rect_verts(xnp, 0.5, 0.5, 0.6, 0.4, 0.0)
    nx, ny = _b.distance_gradient_normal_field(xnp, [V], (256, 256), PERIOD, xnp.float64)
    # a pixel just outside the right edge (x=0.8), at mid-height
    j = int(0.85 * 256)
    i = int(0.50 * 256)
    assert abs(abs(float(nx[i, j])) - 1.0) < 1e-12
    assert abs(float(ny[i, j])) < 1e-12


def test_normal_field_is_a_unit_vector_everywhere():
    V = ellipse_verts(xnp, 0.5, 0.5, 0.34, 0.19, 0.7, 64)
    nx, ny = _b.distance_gradient_normal_field(xnp, [V], (128, 128), PERIOD, xnp.float64)
    mag = np.sqrt(np.asarray(nx) ** 2 + np.asarray(ny) ** 2)
    assert np.abs(mag - 1.0).max() < 1e-12


def test_the_normal_field_gets_the_two_axes_the_right_way_round():
    """A disc's normal is radial in REAL space -- true only if the sample cell is not transposed.

    ``period[0]`` is Px everywhere in this package (``polygon.polygon_ft``,
    ``boundary.normal_projector_coeffs``, ``analytic.shape_ft``), so the sample grid has to
    stretch ``u`` -- which indexes Nx -- by ``period[0]``. Reading the pair the other way round
    samples the field over a transposed cell, and **a square cell hides it completely**: every
    other normal-field test here runs on 1x1. So this one is 2:1, and a disc is the shape that
    says which axis is which, because its correct normal is the radial direction and nothing
    else.

    The annulus skips the centre (the disc's medial axis is the single point there, where the
    direction is undefined) and stops well inside the cell, so no pixel's nearest boundary point
    is decided by the cell edge. The tolerance is set by the polygonisation: a 256-gon's edge
    normal is off-radial by at most pi/256, i.e. 7.5e-5 in the cosine. A swapped pair is not a
    small effect -- it puts this at 1.0.
    """
    Px, Py, r = 2.0, 1.0, 0.3
    V = ellipse_verts(xnp, Px / 2.0, Py / 2.0, 2 * r, 2 * r, 0.0, 256)
    Ny, Nx = 129, 257
    nx, ny = _b.distance_gradient_normal_field(xnp, [V], (Ny, Nx), (Px, Py), xnp.float64)

    X, Y = np.meshgrid((np.arange(Nx) + 0.5) / Nx * Px, (np.arange(Ny) + 0.5) / Ny * Py)
    dx, dy = X - Px / 2.0, Y - Py / 2.0
    rad = np.hypot(dx, dy)
    align = (np.asarray(nx) * dx + np.asarray(ny) * dy) / np.where(rad > 0, rad, 1.0)
    near = (rad > 0.5 * r) & (rad < 1.5 * r)
    assert np.abs(np.abs(align[near]) - 1.0).max() < 1e-3

    # And the same statement read off one pixel, exactly, where a swap is plainest. The grid is
    # odd so row 64 sits *on* the centre line, and the polygonisation puts a vertex on the +x
    # axis, so due east of the centre the nearest boundary point is that vertex and the normal
    # is x_hat with no polygon tilt left in it. On the transposed cell this pixel is not even
    # outside the disc.
    i, j = Ny // 2, int((Px / 2.0 + 1.3 * r) / Px * Nx)
    assert abs(abs(float(nx[i, j])) - 1.0) < 1e-14
    assert abs(float(ny[i, j])) < 1e-14


def test_normal_field_chunking_changes_nothing():
    """Chunking is a static split for memory; it must not be visible in the answer."""
    V = ellipse_verts(xnp, 0.5, 0.5, 0.34, 0.19, 0.7, 64)
    a = _b.distance_gradient_normal_field(xnp, [V], (128, 128), PERIOD, xnp.float64, chunk=10 ** 9)
    for chunk in (1000, 4001, None):
        b = _b.distance_gradient_normal_field(xnp, [V], (128, 128), PERIOD, xnp.float64,
                                              chunk=chunk)
        assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
        assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


# --------------------------------------------------------------------------- conv matrices
def _mats(xp, params, plan=None, truncation_order=4, ng=64, dtype=None, cdtype=None):
    regs, plan = resolve(xp, params, plan)
    return _b.conv_mats(xp, regs, 1.0, truncation_order, truncation_order, PERIOD, ng,
                        dtype or xp.float64, cdtype or xp.complex128), plan


def test_conv_mats_are_hermitian_for_a_lossless_layer():
    (m, _) = _mats(xnp, TWO_RODS)
    for k in (0, 1):
        A = np.asarray(m[k])
        assert np.abs(A - A.conj().T).max() < 1e-11


@pytest.mark.parametrize('truncation_order', [4, 8])
def test_permittivity_matches_the_analytic_path_for_one_shape(truncation_order):
    """One rectangle overlaps nothing, so both paths hold the *same* permittivity -- exactly.

    Only ``epz_i`` is compared, because it is ``inv(ep_dir)`` and never touches the normal field.
    The other four do, and the two paths extend the normal differently -- closed form off the
    shape here, gradient-of-distance off the polygon there. Both are valid NVF choices, and the
    matrices they build differ by a *fixed* amount that does not shrink with truncation
    (measured 4.8e-3 flat over truncation_order 4..12), so requiring them to agree would be requiring one
    extension to be the other.
    """
    from wavebit._core.analytic import to_conv_mat_smooth_nvf
    shape = dict(type='rectangle', cx=0.5, cy=0.5, lx=0.42, ly=0.31, angle=0.4, material=4.0)
    ng = 1024
    (mine, _) = _mats(xnp, [(0.5, 0.5, 0.42, 0.31, 0.4, 4.0)], truncation_order=truncation_order, ng=ng)
    theirs = to_conv_mat_smooth_nvf(xnp, [shape], 1.0, truncation_order, truncation_order, PERIOD, xnp.complex128,
                                      normal_grid=ng)
    rel = (np.abs(np.asarray(mine.epz_i) - np.asarray(theirs.epz_i)).max()
           / np.abs(np.asarray(theirs.epz_i)).max())
    assert rel < 1e-13, rel


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_conv_mats_agree_across_backends(backend):
    (ref, plan) = _mats(xnp, TWO_RODS)
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        (got, _) = _mats(jnp, TWO_RODS, plan, dtype=jnp.float64, cdtype=jnp.complex128)
        back = np.asarray
    else:
        torch = pytest.importorskip('torch')
        import array_api_compat.torch as xt
        torch.set_default_dtype(torch.float64)
        (got, _) = _mats(xt, TWO_RODS, plan, dtype=torch.float64, cdtype=torch.complex128)
        back = lambda t: t.detach().cpu().numpy()          # noqa: E731
    for k in range(5):
        assert np.abs(np.asarray(ref[k]) - back(got[k])).max() < 1e-12, k


def test_gradient_reaches_the_shape_parameters():
    """cx, cy, lx, ly, angle -> clipper replay -> polygon FT -> normal field -> matrices.

    Asymmetric shapes on purpose: see the module docstring on the medial axis.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _, plan = resolve(xnp, TWO_RODS)
    p0 = jnp.asarray([v for s in TWO_RODS for v in s[:5]])

    def loss(flat):
        params = [tuple(flat[i] for i in range(0, 5)) + (TWO_RODS[0][5],),
                  tuple(flat[i] for i in range(5, 10)) + (TWO_RODS[1][5],)]
        (m, _) = _mats(jnp, params, plan, dtype=jnp.float64, cdtype=jnp.complex128)
        return jnp.real(jnp.sum(m[0] * jnp.conj(m[0])) + jnp.sum(m[2] * jnp.conj(m[2])))

    g = jax.grad(loss)(p0)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.abs(np.asarray(g)).max() > 1.0            # not a dead gradient

    h = 1e-6
    for i in (0, 2, 4, 9):
        fd = (float(loss(p0.at[i].add(h))) - float(loss(p0.at[i].add(-h)))) / (2 * h)
        an = float(g[i])
        assert abs(fd - an) <= 1e-5 * max(abs(fd), abs(an), 1.0), (i, fd, an)


# --------------------------------------------------------------- one dict, one reading
#: A circle written the natural way: the diameter given once. Summing this used to work and
#: painting it used to raise ``KeyError: 'ly'``, because the two paths read the dict apart.
CIRCLE_NO_LY = dict(type='circle', cx=0.42, cy=0.38, lx=0.34, material=5.76)


@pytest.mark.parametrize('shape, spelt_out', [
    # a circle's second length has no spelling of its own -- `ellipse` is where two go, which
    # is exactly what the resolved circle has to equal
    (CIRCLE_NO_LY,
     dict(type='ellipse', cx=0.42, cy=0.38, lx=0.34, ly=0.34, angle=0.0)),
    # no centre either -- the transform centres a shape in the cell, so this path must too
    (dict(type='circle', lx=0.34, material=5.76),
     dict(type='ellipse', cx=0.5, cy=0.5, lx=0.34, ly=0.34, angle=0.0)),
    # a square is square, and `rectangle` is where the second side goes
    (dict(type='square', cx=0.4, cy=0.6, lx=0.3, angle=0.2, material=5.76),
     dict(type='rectangle', cx=0.4, cy=0.6, lx=0.3, ly=0.3, angle=0.2)),
    # `l` is the both-axes spelling of a length
    (dict(type='rectangle', cx=0.4, cy=0.6, l=0.3, material=5.76),
     dict(type='rectangle', cx=0.4, cy=0.6, lx=0.3, ly=0.3, angle=0.0)),
    (dict(type='ellipse', cx=0.4, cy=0.6, lx=0.3, ly=0.2, angle=-0.7, material=5.76),
     dict(type='ellipse', cx=0.4, cy=0.6, lx=0.3, ly=0.2, angle=-0.7)),
])
def test_shape_vertices_reads_the_dict_like_the_transform(shape, spelt_out):
    """One shape dict, one reading -- `analytic.shape_geometry` -- for polygons too.

    Every default this path used to carry on its own is checked against a dict that states the
    same geometry outright, so the two have to draw the *identical* polygon: omitted ``ly``,
    omitted ``cx`` / ``cy``, ``l``, and a round type's second length. Vertex-for-vertex, not to
    a tolerance -- there is one reader, so there is nothing to round.

    The pairs cross types on purpose (``circle`` against ``ellipse``, ``square`` against
    ``rectangle``): a round type has one length, so the form that states two *is* the other
    type. Only the geometry is compared, not the name.
    """
    from wavebit._core.analytic import shape_geometry
    assert shape_geometry(shape, *PERIOD)[1:] == shape_geometry(spelt_out, *PERIOD)[1:]
    a = np.asarray(_b.shape_vertices(xnp, shape, PERIOD), float)
    b = np.asarray(_b.shape_vertices(xnp, spelt_out, PERIOD), float)
    assert np.array_equal(a, b)


@pytest.mark.parametrize('truncation_order', [4, 8])
def test_a_circle_without_ly_is_the_same_circle_on_both_paths(truncation_order):
    """Painted and summed hold the same permittivity for a lone circle -- so it must be round.

    ``epz_i`` again, for the reason given in the one-rectangle test above: it is ``inv(ep_dir)``
    and never touches the normal field, which the two paths extend differently. Unlike a
    rectangle this cannot be exact -- the polygon is a 256-gon and the transform is a Bessel
    form -- but the polygon is **area-matched**, which drops that gap from 1.6e-5 to the 1e-9
    band. The tolerance is set from there rather than from a round number, so it would notice
    the scaling being dropped. A misread ``ly`` is not a 1e-9 effect either way: falling back to
    zero empties the circle and falling back to the old 500.0 fills the cell.

    ``skew`` drops ``cx`` / ``cy`` as well, on a cell where the default for them is *arguable*:
    ``shape_geometry`` fills an omitted centre from the period, and on a parallelogram that is
    not the cell centre (measured 0.5, 0.522 against a centre of 0.65, 0.5 on this shear). This
    asserts the two paths **agree**, not that the location is right -- there is one default, so
    correcting it later moves both paths together, which is the property worth pinning.
    """
    from wavebit._core.analytic import to_conv_mat_smooth_nvf
    from wavebit._core.lattice import as_lattice, periods_of
    ng = 1024
    for name, lat in (('square', None), ('skew', as_lattice(lattice=[[1.0, 0.3], [0.0, 1.0]]))):
        shape = CIRCLE_NO_LY if lat is None else dict(type='circle', lx=0.34, material=5.76)
        period = PERIOD if lat is None else periods_of(lat)
        mine = _b.to_conv_mat_boundary_layer(xnp, [shape], 1.0, truncation_order, truncation_order, period,
                                             xnp.complex128, normal_grid=ng, lattice=lat)
        theirs = to_conv_mat_smooth_nvf(xnp, [shape], 1.0, truncation_order, truncation_order, period,
                                          xnp.complex128, normal_grid=ng, lattice=lat)
        rel = (np.abs(np.asarray(mine.epz_i) - np.asarray(theirs.epz_i)).max()
               / np.abs(np.asarray(theirs.epz_i)).max())
        # Measured 3.2e-10 (truncation_order 4) and 6.2e-10 (truncation_order 8), square and skew alike. Set just
        # above that, not at a round number: dropping the area matching puts it back at
        # 1.6e-5 and a 1e-4 tolerance would have absorbed that silently.
        assert rel < 5e-9, (name, rel)


# --------------------------------------------------------------------------- wiring
def _solve(structure, truncation_order=6, **over):
    import wavebit
    cfg = dict(backend=0, truncation_order=(truncation_order, truncation_order), period=PERIOD, thickness=(0.6,), wavelength=1.6,
               top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=np.pi / 2)
    cfg.update(over)
    return wavebit.rcwa(structure=structure, **cfg)


PAINT_SHAPES = [dict(type='rectangle', cx=0.40, cy=0.40, lx=0.34, ly=0.34, angle=0.0,
                     material=4.0),
                dict(type='rectangle', cx=0.56, cy=0.56, lx=0.34, ly=0.34, angle=0.0,
                     material=9.0)]


def _total(res):
    return float(np.asarray(res.T).real.sum())


def test_paint_is_accepted_and_is_not_add():
    """The two rules are two structures; asking for one must not quietly give the other."""
    painted = _total(_solve([dict(background=1.0, combine='paint', shapes=PAINT_SHAPES)]).solve())
    added = _total(_solve([dict(background=1.0, combine='add', shapes=PAINT_SHAPES)]).solve())
    assert abs(painted - added) > 0.1, (painted, added)


def test_paint_agrees_with_the_vector_path():
    """The block path paints too, so on axis-aligned shapes both describe the same structure."""
    painted = _total(_solve([dict(background=1.0, combine='paint', shapes=PAINT_SHAPES)]).solve())
    vec = [[1.0, [['rectangle', s['cx'], s['cy'], s['lx'], s['ly'], s['material'], s['angle'],
                   2, 2] for s in PAINT_SHAPES]]]
    assert abs(painted - _total(_solve(vec).solve())) < 5e-3


def test_paint_without_overlap_equals_add():
    """While nothing intersects, the two rules mean the same thing and must agree exactly."""
    apart = [dict(type='rectangle', cx=0.22, cy=0.22, lx=0.26, ly=0.26, angle=0.3, material=4.0),
             dict(type='rectangle', cx=0.74, cy=0.74, lx=0.26, ly=0.26, angle=-0.4, material=9.0)]
    a = _total(_solve([dict(background=1.0, combine='paint', shapes=apart)]).solve())
    b = _total(_solve([dict(background=1.0, combine='add', shapes=apart)]).solve())
    assert abs(a - b) < 2e-3, (a, b)


def test_freeze_geometry_records_a_plan_only_for_painted_layers():
    solver = _solve([dict(background=1.0, combine='paint', shapes=PAINT_SHAPES),
                     dict(background=1.0, combine='add', shapes=PAINT_SHAPES)],
                    thickness=(0.3, 0.3))
    g = solver.freeze_geometry()
    assert g.boundary_plans is not None
    assert g.boundary_plans[0] is not None      # painted
    assert g.boundary_plans[1] is None          # summed -- nothing discrete to freeze


def test_frozen_plan_reproduces_the_eager_answer():
    solver = _solve([dict(background=1.0, combine='paint', shapes=PAINT_SHAPES)])
    eager = _total(solver.solve())
    replayed = _total(solver.solve(geometry=solver.freeze_geometry()))
    assert abs(eager - replayed) < 1e-12


def test_a_circle_without_ly_paints_and_sums_to_the_same_answer():
    """The dict that used to sum and not paint, through both rules, on one structure.

    Nothing here overlaps, so the two rules describe the same thing and the two answers are
    the *same number* rather than merely two numbers. Note what this does and does not reach:
    while nothing intersects, ``paint`` stays on the summed closed form (see
    ``layer_needs_resolving``), so this pins the contract but never calls ``shape_vertices``.
    The test below is the one that paints.
    """
    apart = [CIRCLE_NO_LY,
             dict(type='rectangle', cx=0.80, cy=0.82, lx=0.22, ly=0.18, angle=-0.4,
                  material=9.0)]
    a = _total(_solve([dict(background=1.0, combine='paint', shapes=apart)]).solve())
    b = _total(_solve([dict(background=1.0, combine='add', shapes=apart)]).solve())
    assert abs(a - b) < 2e-3, (a, b)


def test_a_circle_without_ly_survives_the_polygon_path():
    """The regression: an overlap sends the same dict through ``shape_vertices``.

    ``freeze_geometry`` is where this reached first and raised ``KeyError: 'ly'`` -- it plans
    the painting, so it reads the shapes before the solve does. Spelling ``ly`` out on the
    circle is the workaround that hid this, so the comparison is against the shape that
    genuinely takes two lengths instead: an equal-axis ``ellipse`` is the same structure, and
    has to give the same number.
    """
    over = dict(type='rectangle', cx=0.60, cy=0.58, lx=0.34, ly=0.26, angle=0.4, material=9.0)
    circle = dict(CIRCLE_NO_LY, cx=0.44, cy=0.44, lx=0.36)
    spelt = dict(circle, type='ellipse', ly=circle['lx'], angle=0.0)

    got = []
    for shapes in ([circle, over], [spelt, over]):
        solver = _solve([dict(background=1.0, combine='paint', shapes=shapes)])
        assert solver.freeze_geometry().boundary_plans[0] is not None    # it really painted
        got.append(_total(solver.solve()))
    assert got[0] == got[1], got


def test_a_painted_ellipse_is_polygonised():
    """Curved primitives are polygonised for this path; the answer must still be sane."""
    shapes = [dict(type='ellipse', cx=0.42, cy=0.46, lx=0.44, ly=0.28, angle=0.5, material=4.0),
              dict(type='circle', cx=0.60, cy=0.58, lx=0.30, angle=0.0, material=9.0)]
    r = _solve([dict(background=1.0, combine='paint', shapes=shapes)]).solve()
    t = _total(r)
    assert 0.0 < t < 1.0
    assert abs(t + float(np.asarray(r.R).real.sum()) - 1.0) < 1e-9    # lossless


# ------------------------------------------------------- a shape painted entirely inside another
#
# The case the rest of this file never reaches. Every overlap above is a *partial* one, where
# the clipper's difference walk finds crossings and returns disjoint simple pieces -- all
# ``sign = +1``, so a caller that drops the sign is indistinguishable from one that applies it.
# Full enclosure is the branch that produces a ``sign = -1`` loop (`clipper.plan_difference`,
# the ``n_x == 0`` containment case), and it was answered wrongly for as long as it existed.

#: A disc of permittivity 9 wholly inside a square of permittivity 4. Concentric on purpose:
#: it is
#: the arrangement whose exact cell-average permittivity is a one-line closed form.
ENCLOSED_SHAPES = [dict(type='rectangle', cx=0.5, cy=0.5, lx=0.6, ly=0.6, angle=0.0,
                        material=4.0),
                   dict(type='circle', cx=0.5, cy=0.5, lx=0.2, material=9.0)]

#: The same two shapes moved so they overlap without either containing the other -- the control
#: that says the boundary path as a whole is sound and only enclosure was broken.
CLIPPED_SHAPES = [dict(type='rectangle', cx=0.43, cy=0.50, lx=0.30, ly=0.30, angle=0.0,
                       material=4.0),
                  dict(type='circle', cx=0.60, cy=0.52, lx=0.32, material=9.0)]


def _raster_of(shapes, n):
    """The same instruction drawn as an `n x n` permittivity raster, painting order kept."""
    axis = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(axis, axis)             # X indexes x (axis 1), Y indexes y (axis 0)
    cell = np.ones((1, n, n))
    for s in shapes:
        if s['type'] == 'rectangle':
            hit = ((np.abs(X - s['cx']) <= s['lx'] / 2)
                   & (np.abs(Y - s['cy']) <= s['ly'] / 2))
        else:
            hit = (X - s['cx']) ** 2 + (Y - s['cy']) ** 2 <= (s['lx'] / 2) ** 2
        cell[0][hit] = s['material']
    return cell


@pytest.mark.parametrize('truncation_order', [6, 12])
def test_an_enclosed_shape_is_cut_out_of_its_host_not_added_to_it(truncation_order):
    """The cell-average permittivity, against the closed form -- the truncation-free instrument.

    ``epz_i`` is ``inv(ep_dir)``, so inverting it back recovers the matrix the solve holds, and
    that matrix's centre element is the ``k = 0`` coefficient: the cell average of ``eps``. It
    is an *exact* quantity, identical at every ``truncation_order``, which is what makes it the right probe
    here -- a wrong answer that does not move with truncation cannot be argued away as a series
    that has not converged, and this one does not move at all.

    Exactly: background 1, a 0.6 x 0.6 square of eps 4 with a disc of eps 9 cut out of it,
    and the disc laid back in. ``polygon.ellipse_verts`` is **area-matched**, so the 256-gon's
    area is ``pi * 0.1**2`` to about 1e-13 and the closed form below is not a polygonisation
    approximation.

    The number this returned before the fix was 2.4256 -- the disc counted *twice*, once as
    part of the square and once as itself, because ``clipper`` returns the hole loop
    counter-clockwise with ``sign = -1`` and ``boundary.conv_mats`` was reading only the loop.
    Note what that failure is not: the matrices stayed Hermitian and ``R + T`` stayed exactly
    1.000000000000 (see the raster test below), so no conservation check could have seen it.
    """
    area = float(polygon_area(xnp, _b.shape_vertices(xnp, ENCLOSED_SHAPES[1], PERIOD)))
    assert abs(area - np.pi * 0.1 ** 2) < 1e-12, area
    exact = 1.0 + (4.0 - 1.0) * (0.36 - area) + (9.0 - 1.0) * area
    wrong = 1.0 + (4.0 - 1.0) * (0.36 + area) + (9.0 - 1.0) * area              # sign dropped

    solver = _solve([dict(background=1.0, combine='paint', shapes=ENCLOSED_SHAPES)], truncation_order=truncation_order,
                    normal_grid=256)
    assert solver.freeze_geometry().boundary_plans[0] is not None      # it really painted
    direct = np.linalg.inv(np.asarray(solver.eps_conv_matrices()[2])[0])
    n_orders = (2 * truncation_order + 1) ** 2
    got = complex(direct[n_orders // 2, n_orders // 2])

    assert abs(got.imag) < 1e-12, got
    assert abs(got.real - exact) < 1e-12, (got.real, exact)
    assert abs(got.real - wrong) > 0.18                                 # the old answer


def test_paint_with_an_enclosed_shape_agrees_with_a_fine_raster():
    """The end-to-end statement, against the one path that draws enclosure without a clipper.

    A raster simply overwrites pixels in painting order, so it has no sign convention to get
    wrong and is the independent reading of "a disc of index 3 inside a square of index 2".
    Both resolutions are here rather than one so the convergence is *visible*: 800 and 1600
    sit 3.6e-5 apart, which is a fifty-fold smaller number than the gap being asserted, so the
    raster is a fixed reference and not the thing under test.

    The remaining 1.8e-3 is not error in either path -- it is the normal field. The boundary
    path takes it from the distance gradient off the exact polygons, the raster path from the
    pixel grid, and the two are different valid NVF choices whose gap does not shrink with the
    raster (it is the same at N=800 and N=1600). The control below fixes the size of that gap
    independently, on geometry that never produces a hole: 5.1e-4 there, same configuration.

    Before the fix this stood at 9.9e-2 -- 54x the tolerance. **A test that only checked
    ``R + T`` would have passed throughout**: energy came out 1.000000000000 with the sign
    dropped, because summing a hole instead of subtracting it still leaves a lossless,
    Hermitian permittivity -- just of a different structure. That is why the assertion here is
    against an outside reading of the geometry rather than an internal consistency check.
    """
    painted = _total(_solve([dict(background=1.0, combine='paint', shapes=ENCLOSED_SHAPES)],
                            normal_grid=256).solve())
    coarse = _total(_solve(_raster_of(ENCLOSED_SHAPES, 800), normal_grid=256).solve())
    fine = _total(_solve(_raster_of(ENCLOSED_SHAPES, 1600), normal_grid=256).solve())

    assert abs(coarse - fine) < 2e-4, (coarse, fine)          # the raster has converged
    assert abs(painted - fine) < 4e-3, (painted, fine)


def test_a_partial_overlap_was_never_affected():
    """The control: the same two shapes moved apart until neither contains the other.

    This is what says the fix is about the enclosure branch and not about the boundary path in
    general. With crossings present the difference walk returns disjoint simple pieces, every
    one of them counter-clockwise -- asserted below rather than assumed, because a region with
    no reversed loop in it is exactly a region no orientation convention can get wrong, which
    is why this number is bit-identical across both the sign fix and the move to windings.
    """
    recs, regions = _b.plan_layer(CLIPPED_SHAPES, PERIOD)
    verts = [_b.shape_vertices(xnp, s, PERIOD) for s in CLIPPED_SHAPES]
    vals = _c.eval_table(xnp, recs, verts)
    areas = [float(polygon_area(xnp, _c.loop_verts(xnp, lp, vals)))
             for region in regions for lp in region]
    assert len(areas) == 2 and all(a > 0 for a in areas), areas

    painted = _total(_solve([dict(background=1.0, combine='paint', shapes=CLIPPED_SHAPES)],
                            normal_grid=256).solve())
    fine = _total(_solve(_raster_of(CLIPPED_SHAPES, 1600), normal_grid=256).solve())
    assert abs(painted - fine) < 2e-3, (painted, fine)


def test_an_enclosed_shape_comes_back_wound_clockwise():
    """The convention itself, stated where a reader will look for it.

    This is the whole of the hole bookkeeping: the orientation *is* the data. ``plan_difference``
    emits a contained subtractor walked backwards, and ``subtract_loop`` keeps it that way, so
    the signed shoelace and the signed contour integral both subtract it with no sign to carry
    alongside them and therefore none for a consumer to forget. One consumer did forget, back
    when the orientation was normalised away into a separate ``sign`` field, and the enclosure
    tests above are what that cost.
    """
    recs, regions = _b.plan_layer(ENCLOSED_SHAPES, PERIOD)
    verts = [_b.shape_vertices(xnp, s, PERIOD) for s in ENCLOSED_SHAPES]
    vals = _c.eval_table(xnp, recs, verts)
    areas = [[float(polygon_area(xnp, _c.loop_verts(xnp, lp, vals))) for lp in region]
             for region in regions]

    assert len(areas[0]) == 2 and len(areas[1]) == 1, areas
    disc = np.pi * 0.1 ** 2
    assert areas[0][0] == pytest.approx(0.36, abs=1e-12)          # the square, material
    assert areas[0][1] == pytest.approx(-disc, abs=1e-12)         # the disc, a hole: CLOCKWISE
    assert areas[1][0] == pytest.approx(+disc, abs=1e-12)         # the disc itself, material

    # and the region sums to the visible area with no sign arithmetic at all
    assert float(_c.region_area(xnp, regions[0], vals)) == pytest.approx(0.36 - disc, abs=1e-12)


def test_a_hole_inside_a_hole_is_material_again():
    """Nested enclosure has to compose by geometry, because there is no longer any sign to
    multiply.

    Three concentric squares, each painted over the last. The middle one cuts a hole in the
    outer one; the inner one then cuts a hole in *both*, and cutting it out of the middle
    square's hole has to put the material back. Under the old representation that was the
    ``sign * rel`` product; here it is a loop being reversed twice and arriving counter-clockwise.

    Both readings are asserted, because the loop list is the thing that has to be right and the
    total is what a consumer sees: the outer region comes back as four loops whose areas are
    ``+0.64, -0.04, -0.25, +0.04`` -- note the small square appearing with *both* signs, which
    is the composition working -- and they sum to 0.39, the visible area. The cell average is
    then checked against the closed form at two ``truncation_order``, since it is exact and truncation-free.
    """
    nested = [dict(type='rectangle', cx=0.5, cy=0.5, lx=0.80, ly=0.80, angle=0.0, material=4.0),
              dict(type='rectangle', cx=0.5, cy=0.5, lx=0.50, ly=0.50, angle=0.0, material=9.0),
              dict(type='rectangle', cx=0.5, cy=0.5, lx=0.20, ly=0.20, angle=0.0, material=16.0)]
    recs, regions = _b.plan_layer(nested, PERIOD)
    verts = [_b.shape_vertices(xnp, s, PERIOD) for s in nested]
    vals = _c.eval_table(xnp, recs, verts)

    got = [sorted(float(polygon_area(xnp, _c.loop_verts(xnp, lp, vals))) for lp in region)
           for region in regions]
    assert got[0] == pytest.approx([-0.25, -0.04, 0.04, 0.64], abs=1e-12), got[0]
    assert got[1] == pytest.approx([-0.04, 0.25], abs=1e-12), got[1]
    assert got[2] == pytest.approx([0.04], abs=1e-12), got[2]

    visible = [0.64 - 0.25, 0.25 - 0.04, 0.04]
    for region, want in zip(regions, visible):
        assert float(_c.region_area(xnp, region, vals)) == pytest.approx(want, abs=1e-12)

    exact = 1.0 + (4.0 - 1) * visible[0] + (9.0 - 1) * visible[1] \
        + (16.0 - 1) * visible[2]
    for truncation_order in (6, 12):
        solver = _solve([dict(background=1.0, combine='paint', shapes=nested)], truncation_order=truncation_order,
                        normal_grid=256)
        direct = np.linalg.inv(np.asarray(solver.eps_conv_matrices()[2])[0])
        n_orders = (2 * truncation_order + 1) ** 2
        assert complex(direct[n_orders // 2, n_orders // 2]).real == pytest.approx(exact, abs=1e-12), truncation_order


def test_the_normal_field_does_not_care_which_way_a_hole_is_wound():
    """The one output the winding *does* touch, and the measurement that says to accept it.

    ``conv_mats`` hands the loops to the normal field exactly as the clipper produced them,
    holes wound clockwise included. That is safe by construction: the field is the direction to
    the nearest boundary *point*, so it depends on which segments exist and not on how they are
    wound, and only ``nx**2``, ``nx*ny``, ``ny**2`` are consumed, so the vector's own sign
    cancels as well.

    Safe, but not bit-invariant, and the difference is pinned here precisely rather than hidden
    under a loose tolerance. Reversing a loop renumbers its edges, so ``argmin`` breaks its
    *exact* ties the other way. Measured: 20 pixels of 65536 move, each by 0.024541 in
    ``nx**2`` -- ``sin(2*pi/256)``, one facet of the polygonised disc, so the tie is between two
    adjacent facets. That is 6.0e-6 in ``c_xx`` and ``c_yy``, 7.0e-18 in ``c_xy``.

    The last block is the part that decides it: against the **ideal** circle's radial normal,
    both windings are wrong by 0.7031 degrees on every one of those pixels, and the two errors
    sum to 1.4062 = 360/256, exactly one facet step. An exact tie -- so there is no accuracy
    argument for normalising the loops before the field, and normalising would put back the
    orientation conversion the clipper was rewritten to remove.
    """
    recs, regions = _b.plan_layer(ENCLOSED_SHAPES, PERIOD)
    verts = [_b.shape_vertices(xnp, s, PERIOD) for s in ENCLOSED_SHAPES]
    vals = _c.eval_table(xnp, recs, verts)
    asis = [_c.loop_verts(xnp, lp, vals) for region in regions for lp in region]
    ccw = [xnp.flip(V, axis=0) if float(polygon_area(xnp, V)) < 0 else V for V in asis]
    assert any(float(polygon_area(xnp, V)) < 0 for V in asis)      # there IS a hole to reverse

    a = _b.normal_projector_coeffs(xnp, asis, (256, 256), PERIOD, 6, 6,
                                   xnp.float64, xnp.complex128)
    b = _b.normal_projector_coeffs(xnp, ccw, (256, 256), PERIOD, 6, 6,
                                   xnp.float64, xnp.complex128)
    for got, want in zip(a, b):
        assert np.abs(np.asarray(got) - np.asarray(want)).max() < 1e-5

    fa = _b.distance_gradient_normal_field(xnp, asis, (256, 256), PERIOD, xnp.float64)
    fb = _b.distance_gradient_normal_field(xnp, ccw, (256, 256), PERIOD, xnp.float64)
    moved = None
    for u, v in zip(fa, fb):
        # squared, because the field's own sign is meaningless -- see the docstring
        d = np.abs(np.asarray(u) ** 2 - np.asarray(v) ** 2)
        m = d > 1e-12
        # counted, not bounded: a genuine dependence on the winding would move the *field*,
        # not twenty pixels of it, and a plain max() would read the same either way.
        assert int(m.sum()) < 100, int(m.sum())
        assert float(d[m].max()) == pytest.approx(np.sin(2 * np.pi / 256), abs=1e-6)
        moved = m if moved is None else (moved | m)

    # neither winding is closer to the true normal of the circle the polygon approximates
    axis = (np.arange(256) + 0.5) / 256
    X, Y = np.meshgrid(axis, axis)
    dx, dy = X - 0.5, Y - 0.5
    rad = np.hypot(dx, dy)
    ux, uy = dx / rad, dy / rad
    errs = []
    for f in (fa, fb):
        cos = np.abs(np.asarray(f[0]) * ux + np.asarray(f[1]) * uy)
        errs.append(np.degrees(np.arccos(np.clip(cos[moved], -1.0, 1.0))))
    assert errs[0].mean() == pytest.approx(errs[1].mean(), abs=1e-9)      # an exact tie
    assert errs[0].mean() + errs[1].mean() == pytest.approx(360.0 / 256, abs=1e-3)


# ------------------------------------------------- the painting plan across backends

def _two_painted(w, curved=False):
    """Two overlapping shapes, painted, with every geometry value wrapped by `w`."""
    first = (dict(type='circle', cx=w(0.55), cy=w(0.5), lx=w(0.30), material=12.1104) if curved
             else dict(type='rectangle', cx=w(0.55), cy=w(0.5), lx=w(0.26), ly=w(0.24),
                       angle=w(0.0), material=12.1104))
    return [dict(background=1.0, combine='paint', shapes=[
        first,
        dict(type='rectangle', cx=0.68, cy=0.58, lx=0.26, ly=0.30, angle=0.0, material=4.41)])]


_PAINT_CFG = dict(period=(1.0, 1.0), wavelength=0.94, thickness=(0.32,), truncation_order=(4, 4),
                  top=1.0, bot=2.1025, theta=0.0, phi=None, psi=np.pi / 2, normal_grid=256)


def _painted_total(backend, w, curved=False, geometry=None):
    solver = wavebit.rcwa(backend=backend, structure=_two_painted(w, curved), **_PAINT_CFG)
    res = solver.solve() if geometry is None else solver.solve(geometry=geometry)
    t = res.T
    return float(np.sum(np.asarray(t.detach() if hasattr(t, 'detach') else t)).real)


@pytest.mark.parametrize('curved', [False, True], ids=['rectangles', 'circle'])
def test_a_painted_overlap_agrees_across_backends(curved):
    """Every backend, same painted overlap, same answer.

    Both halves of this used to raise on PyTorch and on PyTorch only, for reasons that had
    nothing to do with the physics:

    * ``plan_layer`` runs on NumPy by design but read the caller's shape dict, so a torch
      tensor reached ``rect_verts`` and met ``numpy_array * Tensor``. JAX survived by
      accident -- its arrays satisfy ``__array__`` and convert on contact -- so testing on
      JAX proved nothing about torch here.
    * ``ellipse_verts`` called ``xp.cos`` on a scalar, which torch refuses. That one fired
      for *every* angle including 0.0, so the boundary path could not draw a circle or an
      ellipse on torch at all.

    Routing is on the overlap, so both are reached only when the shapes actually intersect --
    which is why a non-overlapping smoke test would have stayed green through both.
    """
    reference = _painted_total(0, float, curved)
    for backend, lib in ((1, 'jax'), (2, 'torch')):
        pytest.importorskip(lib)
        assert _painted_total(backend, float, curved) == pytest.approx(reference, abs=1e-12)


def test_a_painted_overlap_takes_traced_geometry_on_torch():
    """The same, with the shape parameters as tensors rather than floats.

    Separate from the case above because the float path and the tensor path reach
    ``plan_layer`` differently, and only the second one raised.
    """
    torch = pytest.importorskip('torch')
    reference = _painted_total(0, float)
    traced = _painted_total(2, lambda v: torch.tensor(v, dtype=torch.float64))
    assert traced == pytest.approx(reference, abs=1e-12)


def test_painted_geometry_follows_the_dtype_it_was_given():
    """float32 parameters give a float32 answer, and that is the caller's choice, not a bug.

    Worth pinning because ``torch.tensor(0.55)`` is float32 under torch's stock default,
    regardless of ``complex_dtype`` -- so the natural way to write a traced parameter quietly
    halves the precision of the geometry. The library follows the input rather than overriding
    it; this records that it does, and how much it costs.

    Both dtypes are named explicitly rather than leaning on ``torch.get_default_dtype()``.
    That default is process-global and another test in the suite sets it, so a version of this
    test that relied on it passed alone and failed in the file -- which is a statement about
    test isolation, not about the geometry.
    """
    torch = pytest.importorskip('torch')
    reference = _painted_total(0, float)
    single = _painted_total(2, lambda v: torch.tensor(v, dtype=torch.float32))
    double = _painted_total(2, lambda v: torch.tensor(v, dtype=torch.float64))
    assert double == pytest.approx(reference, abs=1e-12)
    assert single != pytest.approx(reference, abs=1e-12)
    assert single == pytest.approx(reference, abs=1e-6)


def test_freezing_a_painted_layer_works_on_torch():
    """``freeze_geometry()`` is what the docstring points at, and it raised in the same frame.

    It is the call that runs ``plan_layer``, so taking the freeze was itself the failure.
    """
    torch = pytest.importorskip('torch')
    structure = _two_painted(lambda v: torch.tensor(v, dtype=torch.float64))
    solver = wavebit.rcwa(backend=2, structure=structure, **_PAINT_CFG)
    frozen = solver.freeze_geometry()
    total = float(np.sum(np.asarray(solver.solve(geometry=frozen).T.detach())).real)
    assert total == pytest.approx(_painted_total(0, float), abs=1e-12)


def test_a_traced_painted_layer_is_refused_without_a_frozen_geometry():
    """The silent one: `combine='paint'` under `jax.grad` used to solve a *summed* layer.

    Whether two shapes overlap decides which structure a painted layer is, and that question
    needs concrete numbers. A JAX tracer refuses to give them, and the check used to read that
    refusal as "not overlapping" and fall through to the summing branch -- on the reasoning
    that a traced call has already had its eager pass through ``freeze_geometry``. Nothing
    enforced that, and the cost of the assumption failing is not a slower path: it is a
    different structure, solved and differentiated without a word.

    How it was identified is worth keeping. The gradient looked merely wrong -- -3.48 where
    finite differences give -0.6461 -- until it was compared against the same layer spelled
    ``combine='add'``, which returns **the same number to the last digit**. Not a broken
    derivative; the derivative of something else.

    PyTorch never reached this, because its tensors convert to floats and the check ran. So a
    torch-only test would have passed throughout.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    cfg = dict(period=(1.0, 1.0), wavelength=0.94, truncation_order=(5, 5), top=1.0, bot=2.1025,
               theta=0.0, phi=None, psi=np.pi / 2, thickness=(0.3,), normal_grid=256)
    fixed = dict(type='rectangle', cx=0.70, cy=0.62, lx=0.26, ly=0.30, angle=0.0, material=4.41)

    def structure(v):
        return [dict(background=1.0, combine='paint', shapes=[
            dict(type='rectangle', cx=0.46, cy=0.5, lx=v, ly=0.26, angle=0.3, material=12.1104),
            dict(fixed)])]

    def total(v, geometry=None):
        solver = wavebit.rcwa(backend=1, structure=structure(v), **cfg)
        res = solver.solve() if geometry is None else solver.solve(geometry=geometry)
        return jnp.sum(res.T).real

    with pytest.raises(ValueError, match='freeze_geometry'):
        jax.grad(total)(jnp.asarray(0.34))

    # and the documented route works, and agrees with central differences on the *painted*
    # structure rather than on the summed one
    frozen = wavebit.rcwa(backend=0, structure=structure(0.34), **cfg).freeze_geometry()
    analytic = float(jax.grad(lambda v: total(v, frozen))(jnp.asarray(0.34)))
    h = 1e-4
    plus = float(np.sum(np.asarray(
        wavebit.rcwa(backend=0, structure=structure(0.34 + h), **cfg).solve().T)).real)
    minus = float(np.sum(np.asarray(
        wavebit.rcwa(backend=0, structure=structure(0.34 - h), **cfg).solve().T)).real)
    # 2e-2, not the 2e-3 this carried until 2026-08-16. The two sides do not compute the same
    # thing by construction: the AD side holds the *frozen* normal field while the finite
    # difference rebuilds it at each `v`, so the gap is the frozen-geometry approximation, not a
    # convergent error -- and it grows as the normal field is refined, for the same reason a
    # finer field has more to change. Measured `rel` at normal_grid = 256 / 512 / 1024:
    #
    #     before the half-pixel phase   1.09e-03   4.88e-03   1.54e-02
    #     after it                      5.44e-03   9.33e-03   1.34e-02
    #
    # Both climb, and at the finest grid the corrected transform is the closer of the two. The
    # old 2e-3 fitted the one column where the discretization error happened to sit on top of
    # the frozen-geometry gap and mask it; it was pinning a coincidence at one `normal_grid`.
    # Widened to what this comparison can actually support, which is a sanity band on the
    # gradient rather than a convergence claim.
    assert analytic == pytest.approx((plus - minus) / (2 * h), rel=2e-2)


def test_a_traced_summed_layer_is_not_refused():
    """The refusal is for `paint` only -- a summed layer has no overlap question to ask."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    cfg = dict(period=(1.0, 1.0), wavelength=0.94, truncation_order=(5, 5), top=1.0, bot=2.1025,
               theta=0.0, phi=None, psi=np.pi / 2, thickness=(0.3,), normal_grid=256)

    def total(v):
        u = [dict(background=1.0, combine='add', shapes=[
            dict(type='rectangle', cx=0.46, cy=0.5, lx=v, ly=0.26, angle=0.3, material=12.1104),
            dict(type='rectangle', cx=0.70, cy=0.62, lx=0.26, ly=0.30, angle=0.0,
                 material=4.41)])]
        return jnp.sum(wavebit.rcwa(backend=1, structure=u, **cfg).solve().T).real

    assert np.isfinite(float(jax.grad(total)(jnp.asarray(0.34))))


def test_a_frozen_geometry_answers_the_overlap_question_for_every_layer():
    """A freeze records the *decision*, not only the plans it produced.

    ``boundary_plans`` used to be left unset when nothing overlapped, which made two cases
    indistinguishable downstream: "looked at while the numbers were concrete, and it does not
    overlap" and "no freeze was taken". The first is an answer; the second is not. Under a
    trace the difference decides whether the overlap test has to be re-taken on values that
    cannot answer it -- so a *disjoint* painted layer, re-frozen every step exactly as the
    documentation advises, raised.

    Both sides of the crossing therefore belong in one test: the entry is ``None`` where the
    shapes are apart and a plan where they meet, and both differentiate.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    cfg = dict(period=(1.0, 1.0), wavelength=0.94, truncation_order=(5, 5), top=1.0, bot=2.1025,
               theta=0.0, phi=None, psi=np.pi / 2, thickness=(0.3,), normal_grid=256)
    fixed = dict(type='rectangle', cx=0.70, cy=0.5, lx=0.26, ly=0.30, angle=0.0, material=4.41)

    def structure(cx):
        return [dict(background=1.0, combine='paint', shapes=[
            dict(type='rectangle', cx=cx, cy=0.5, lx=0.26, ly=0.30, angle=0.0, material=12.1104),
            dict(fixed)])]

    # the rods meet at cx = 0.44: 0.44 + 0.13 == 0.70 - 0.13
    for cx, overlapping in ((0.42, False), (0.46, True)):
        frozen = wavebit.rcwa(backend=0, structure=structure(cx), **cfg).freeze_geometry()
        assert frozen.boundary_plans is not None, cx
        assert (frozen.boundary_plans[0] is not None) is overlapping, cx

        def total(v):
            return jnp.sum(wavebit.rcwa(backend=1, structure=structure(v), **cfg)
                           .solve(geometry=frozen).T).real

        analytic = float(jax.grad(total)(jnp.asarray(cx)))
        h = 1e-4
        step = [float(np.sum(np.asarray(
            wavebit.rcwa(backend=0, structure=structure(cx + s * h), **cfg).solve().T)).real)
            for s in (1, -1)]
        assert analytic == pytest.approx((step[0] - step[1]) / (2 * h), rel=3e-2), cx
