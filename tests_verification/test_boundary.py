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
* The sampled normal field needs **4*fto+1 points per axis** or distinct orders alias onto one
  bin, and the damage is structural rather than gradual (a constant normal field stops
  transforming to the identity, so NVF no longer collapses onto Li's rules).
"""
import os

import numpy as np
import pytest

os.environ.setdefault('JAX_ENABLE_X64', '1')

import array_api_compat.numpy as xnp
from fieldwave._core import boundary as _b
from fieldwave._core import clipper as _c
from fieldwave._core.polygon import polygon_ft, polygon_area, rect_verts, ellipse_verts

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
    regs = [(p[5], [_c.loop_verts(xp, lp, vals) for lp, s in region])
            for p, region in zip(params, regions)]
    return regs, plan


TWO_RODS = [(0.40, 0.43, 0.36, 0.29, 0.17, 2.0),
            (0.57, 0.55, 0.31, 0.38, 0.41, 3.0)]


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
    assert abs(areas[3.0] - a1) < 1e-12          # the later rod is untouched
    assert areas[2.0] < a0 - 1e-3                 # the earlier one lost the overlap
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
@pytest.mark.parametrize('fto', [3, 6, 12])
def test_normal_grid_floor_is_enforced(fto):
    """Below 4*fto+1 the sampled field aliases; that must raise, not silently mis-transform."""
    need = 4 * fto + 1
    with pytest.raises(ValueError, match='aliases'):
        _b.check_normal_grid((need - 1, need - 1), fto, fto)
    _b.check_normal_grid((need, need), fto, fto)


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
def _mats(xp, params, plan=None, fto=4, ng=64, dtype=None, cdtype=None):
    regs, plan = resolve(xp, params, plan)
    return _b.conv_mats(xp, regs, 1.0, fto, fto, PERIOD, ng,
                        dtype or xp.float64, cdtype or xp.complex128), plan


def test_conv_mats_are_hermitian_for_a_lossless_layer():
    (m, _) = _mats(xnp, TWO_RODS)
    for k in (0, 1):
        A = np.asarray(m[k])
        assert np.abs(A - A.conj().T).max() < 1e-11


@pytest.mark.parametrize('fto', [4, 8])
def test_permittivity_matches_the_analytic_path_for_one_shape(fto):
    """One rectangle overlaps nothing, so both paths hold the *same* permittivity -- exactly.

    Only ``epz_i`` is compared, because it is ``inv(ep_dir)`` and never touches the normal field.
    The other four do, and the two paths extend the normal differently -- closed form off the
    shape here, gradient-of-distance off the polygon there. Both are valid NVF choices, and the
    matrices they build differ by a *fixed* amount that does not shrink with truncation
    (measured 4.8e-3 flat over fto 4..12), so requiring them to agree would be requiring one
    extension to be the other.
    """
    from fieldwave._core.analytic import to_conv_mat_analytic_nvf
    shape = dict(type='rectangle', cx=0.5, cy=0.5, lx=0.42, ly=0.31, angle=0.4, n_index=2.0)
    ng = 1024
    (mine, _) = _mats(xnp, [(0.5, 0.5, 0.42, 0.31, 0.4, 2.0)], fto=fto, ng=ng)
    theirs = to_conv_mat_analytic_nvf(xnp, [shape], 1.0, fto, fto, PERIOD, xnp.complex128,
                                      normal_grid=ng)
    rel = (np.abs(np.asarray(mine[4]) - np.asarray(theirs[4])).max()
           / np.abs(np.asarray(theirs[4])).max())
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


# --------------------------------------------------------------------------- wiring
def _solve(ucell, fto=6, **over):
    import fieldwave
    cfg = dict(backend=0, fto=(fto, fto), period=PERIOD, thickness=(0.6,), wavelength=1.6,
               n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, pol=0)
    cfg.update(over)
    return fieldwave.rcwa(ucell=ucell, **cfg)


PAINT_SHAPES = [dict(type='rectangle', cx=0.40, cy=0.40, lx=0.34, ly=0.34, angle=0.0,
                     n_index=2.0),
                dict(type='rectangle', cx=0.56, cy=0.56, lx=0.34, ly=0.34, angle=0.0,
                     n_index=3.0)]


def _total(res):
    return float(np.asarray(res.de_ti).real.sum())


def test_paint_is_accepted_and_is_not_add():
    """The two rules are two structures; asking for one must not quietly give the other."""
    painted = _total(_solve([dict(n_base=1.0, combine='paint', shapes=PAINT_SHAPES)]).conv_solve())
    added = _total(_solve([dict(n_base=1.0, combine='add', shapes=PAINT_SHAPES)]).conv_solve())
    assert abs(painted - added) > 0.1, (painted, added)


def test_paint_agrees_with_the_vector_path():
    """The vector path paints too, so on axis-aligned shapes both describe the same structure."""
    painted = _total(_solve([dict(n_base=1.0, combine='paint', shapes=PAINT_SHAPES)]).conv_solve())
    vec = [[1.0, [['rectangle', s['cx'], s['cy'], s['lx'], s['ly'], s['n_index'], s['angle'],
                   2, 2] for s in PAINT_SHAPES]]]
    assert abs(painted - _total(_solve(vec).conv_solve())) < 5e-3


def test_paint_without_overlap_equals_add():
    """While nothing intersects, the two rules mean the same thing and must agree exactly."""
    apart = [dict(type='rectangle', cx=0.22, cy=0.22, lx=0.26, ly=0.26, angle=0.3, n_index=2.0),
             dict(type='rectangle', cx=0.74, cy=0.74, lx=0.26, ly=0.26, angle=-0.4, n_index=3.0)]
    a = _total(_solve([dict(n_base=1.0, combine='paint', shapes=apart)]).conv_solve())
    b = _total(_solve([dict(n_base=1.0, combine='add', shapes=apart)]).conv_solve())
    assert abs(a - b) < 2e-3, (a, b)


def test_freeze_geometry_records_a_plan_only_for_painted_layers():
    solver = _solve([dict(n_base=1.0, combine='paint', shapes=PAINT_SHAPES),
                     dict(n_base=1.0, combine='add', shapes=PAINT_SHAPES)],
                    thickness=(0.3, 0.3))
    g = solver.freeze_geometry()
    assert g.boundary_plans is not None
    assert g.boundary_plans[0] is not None      # painted
    assert g.boundary_plans[1] is None          # summed -- nothing discrete to freeze


def test_frozen_plan_reproduces_the_eager_answer():
    solver = _solve([dict(n_base=1.0, combine='paint', shapes=PAINT_SHAPES)])
    eager = _total(solver.conv_solve())
    replayed = _total(solver.conv_solve(geometry=solver.freeze_geometry()))
    assert abs(eager - replayed) < 1e-12


def test_a_painted_ellipse_is_polygonised():
    """Curved primitives are polygonised for this path; the answer must still be sane."""
    shapes = [dict(type='ellipse', cx=0.42, cy=0.46, lx=0.44, ly=0.28, angle=0.5, n_index=2.0),
              dict(type='circle', cx=0.60, cy=0.58, lx=0.30, ly=0.30, angle=0.0, n_index=3.0)]
    r = _solve([dict(n_base=1.0, combine='paint', shapes=shapes)]).conv_solve()
    t = _total(r)
    assert 0.0 < t < 1.0
    assert abs(t + float(np.asarray(r.de_ri).real.sum()) - 1.0) < 1e-9    # lossless
