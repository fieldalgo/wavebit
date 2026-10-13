"""Regression instruments for actual interfaces of a resolved polygon painting."""

from wavebit.compat import legacy as _legacy_wavebit
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np
import array_api_compat.numpy as xp
import pytest
from wavebit.rcwa._engine import boundary as b; from wavebit._common.geometry import clipper
from wavebit._common.geometry.polygon import rect_verts

CELL = ((700., 0.), (0., 700.))


def resolved(vertices, materials):
    recs, regions = clipper.resolve_painting(vertices)
    vals = clipper.eval_table(xp, recs, vertices)
    return [(m, [clipper.loop_verts(xp, lp, vals) for lp in reg])
            for m, reg in zip(materials, regions)]


def mats(regions):
    return b.conv_mats(xp, regions, 1., 2, 2, CELL, 64, xp.float64, xp.complex128)


def test_nested_same_material_paint_order_does_not_change_matrices():
    outer = rect_verts(xp, 350., 350., 400., 400., 0.)
    inner = rect_verts(xp, 350., 350., 200., 200., np.pi / 4)
    a = mats(resolved([outer, inner], [4., 4.]))
    z = mats(resolved([inner, outer], [4., 4.]))
    for x, y in zip(a, z):
        np.testing.assert_allclose(x, y, rtol=0, atol=2e-12)


def test_overlapping_spanning_bars_have_exact_one_dimensional_factorization():
    # Tilt the inner cutter so the clipper has no collinear degeneracy.
    outer = rect_verts(xp, 350., 350., 400., 700., 0.)
    inner = rect_verts(xp, 350., 350., 200., 200., .3)
    regs = resolved([outer, inner], [4., 4.])
    epx, epy, _, exy, eyx = mats(regs)
    np.testing.assert_allclose(exy, 0, atol=2e-12, rtol=0)
    np.testing.assert_allclose(eyx, 0, atol=2e-12, rtol=0)
    # No Fourier coupling along y in any matrix.
    yy = np.repeat(np.arange(5), 5)
    for mat in (epx, epy):
        np.testing.assert_allclose(mat[yy[:, None] != yy[None, :]], 0, atol=2e-12, rtol=0)


def field(regs, lattice=CELL, plan=None, backend=xp, grid=(96, 96)):
    transform = b.reduction_transform(lattice)
    if plan is None:
        plan = b.plan_interfaces(regs, lattice, transform)
    data = b.replay_interfaces(backend, regs, 1., lattice, transform, plan)
    loops = [v for _, vv in regs for v in vv]
    return b.distance_gradient_normal_field(backend, loops, grid, lattice, backend.float64,
                                           transform=transform, segment_data=data,
                                           images=plan.images)


def nvf_projector(normals):
    x, y = map(np.asarray, normals)
    return np.stack((x*x, x*y, y*y))


def partial_regions(right=4.):
    return [(4., [rect_verts(xp, 200., 350., 300., 400., 0.)]),
            (right, [rect_verts(xp, 450., 300., 200., 200., 0.)])]


UNION = np.array([(50., 150.), (350., 150.), (350., 200.), (550., 200.),
                  (550., 400.), (350., 400.), (350., 550.), (50., 550.)])


@pytest.mark.parametrize('angle,swap,shear', [(0., False, 0.), (0., True, 0.),
                                             (.37, False, 0.), (.23, False, 180.)])
def test_only_the_shared_interval_disappears_in_the_entire_cell(angle, swap, shear):
    """An independently written union catches whole-edge removal and local-only masks."""
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    if swap:
        rotation = rotation @ np.array([[0., 1.], [1., 0.]])
    lattice = rotation @ np.array([[700., shear], [0., 700.]])
    # Reflection reverses orientation: restore material-on-left convention.
    def moved(v):
        out = v @ rotation.T
        return out[::-1] if np.linalg.det(rotation) < 0 else out
    regs = [(m, [moved(v) for v in vv]) for m, vv in partial_regions()]
    a, e, l2 = b.segments_of(xp, [moved(UNION)])
    oracle = (a, e, l2, np.ones(len(a), dtype=bool))
    want = b.distance_gradient_normal_field(xp, [moved(UNION)], (96, 96), lattice, xp.float64,
                                            segment_data=oracle)
    np.testing.assert_allclose(nvf_projector(field(regs, lattice)), nvf_projector(want), atol=2e-12, rtol=0)


def test_different_material_shared_interval_is_a_real_interface():
    same, different = field(partial_regions()), field(partial_regions(9.+.2j))
    # A row inside the short box and a column just right of the seam.
    row, col = 40, 48
    assert abs(different[0][row, col]) > .999999
    assert abs(same[0][row, col]) < 1e-12
    assert np.max(np.abs(nvf_projector(same) - nvf_projector(different))) > .9


def test_nested_invisible_seam_has_no_interior_field_effect():
    outer = rect_verts(xp, 350., 350., 400., 400., 0.)
    inner = rect_verts(xp, 350., 350., 200., 200., np.pi/4)
    regs = resolved([outer, inner], [4., 4.])
    want = b.distance_gradient_normal_field(xp, [outer], (128, 128), CELL, xp.float64)
    got = field(regs, grid=(128, 128))
    np.testing.assert_allclose(nvf_projector(got), nvf_projector(want), atol=2e-12, rtol=0)
    legacy = b.distance_gradient_normal_field(xp, [v for _, vv in regs for v in vv],
                                             (128, 128), CELL, xp.float64)
    assert np.max(np.abs(nvf_projector(legacy) - nvf_projector(want))) > .49
    assert np.mean(np.max(np.abs(nvf_projector(legacy) - nvf_projector(want)), axis=0) > 1e-8) > .17


@pytest.mark.parametrize('shear', [200., -200.])
def test_periodic_partial_abutment_keeps_the_uncovered_ends(shear):
    lattice = ((700., shear), (0., 700.))
    box = rect_verts(xp, 351.3, 347.1, 500., 700., 0.)
    plan = b.plan_interfaces([(4., [box])], lattice, b.reduction_transform(lattice))
    a, e, _, active = b.replay_interfaces(xp, [(4., [box])], 1., lattice,
                                         b.reduction_transform(lattice), plan)
    horizontal = np.abs(e[:, 1]) < 1e-10
    # Top and bottom each lose 300 of 500, leaving 200 apiece.
    assert np.sum(np.linalg.norm(e[horizontal & active], axis=1)) == pytest.approx(400.)
    assert np.sum(np.linalg.norm(e[horizontal & ~active], axis=1)) == pytest.approx(600.)
    # Independent segment oracle: only the exposed ends of the two horizontal faces.
    if shear < 0:
        lines = [(box[0], box[0]+[200., 0.]), (box[2], box[2]-[200., 0.])]
    else:
        lines = [(box[1]-[200., 0.], box[1]), (box[3], box[3]+[200., 0.])]
    lines += [(box[1], box[2]), (box[3], box[0])]
    oa = np.array([p for p, q in lines]); oe = np.array([q-p for p, q in lines])
    oracle = (oa, oe, np.sum(oe*oe, axis=1), np.ones(len(oa), dtype=bool))
    want = b.distance_gradient_normal_field(xp, [box], (96, 96), lattice, xp.float64,
                                            segment_data=oracle, images=plan.images)
    np.testing.assert_allclose(nvf_projector(field([(4., [box])], lattice)),
                               nvf_projector(want), atol=2e-12, rtol=0)


def test_empty_interface_set_is_a_finite_constant():
    box = rect_verts(xp, 350., 350., 700., 700., 0.)
    for material in (1., 4.):
        nx, ny = field([(material, [box])])
        np.testing.assert_array_equal(nx, 1.)
        np.testing.assert_array_equal(ny, 0.)


def test_long_segments_use_a_bounded_image_set_not_a_fixed_ring():
    # An arbitrary resolved polygon edge can span many reduced cells.
    lattice = ((700., 0.), (0., 70.))
    box = rect_verts(xp, 350., 35., 30., 900., .23)
    regs = [(4., [box])]
    transform = b.reduction_transform(lattice)
    plan = b.plan_interfaces(regs, lattice, transform)
    assert max(max(abs(m), abs(n)) for m, n in plan.images) > 3
    data = b.replay_interfaces(xp, regs, 1., lattice, transform, plan)
    dense = tuple((i, j) for i in range(-20, 21) for j in range(-20, 21))
    got = field(regs, lattice, plan=plan, grid=(12, 12))
    want = b.distance_gradient_normal_field(xp, [box], (12, 12), lattice, xp.float64,
                                            transform=transform, segment_data=data, images=dense)
    np.testing.assert_allclose(nvf_projector(got), nvf_projector(want), atol=2e-12, rtol=0)


def _shapes(width=500.):
    return [dict(type='rectangle', lx=width, ly=700., cx=0., cy=0., angle=0., material=4.),
            dict(type='rectangle', lx=100., ly=80., cx=13., cy=-17., angle=.31, material=4.)]


def test_frozen_interface_replay_jit_and_geometry_gradient():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    lattice = ((700., 183.2), (0., 700.))
    plan = b.plan_layer(_shapes(493.7), lattice)
    p = jnp.array([493.7, 183.2])
    def loss(params):
        cell = jnp.array([[700., params[1]], [0., 700.]])
        mats = b.to_conv_mat_boundary_layer(jnp, _shapes(params[0]), 1., 1, 1,
                                            cell, jnp.complex128, shape_nvf_grid_size=32, plan=plan)
        return jnp.real(jnp.sum(mats[0] * jnp.conj(mats[0])))
    eager = loss(p)
    np.testing.assert_allclose(jax.jit(loss)(p), eager, atol=2e-11, rtol=0)
    grad = jax.jit(jax.grad(loss))(p)
    assert np.all(np.isfinite(grad))
    for i in range(2):
        h = 1e-4
        fd = (loss(p.at[i].add(h)) - loss(p.at[i].add(-h))) / (2*h)
        np.testing.assert_allclose(grad[i], fd, atol=2e-7, rtol=2e-5)
        assert abs(float(grad[i])) > 1e-7


def test_frozen_material_equality_is_evaluated_live_under_jit():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    regs = partial_regions()
    transform = b.reduction_transform(CELL)
    plan = b.plan_interfaces(regs, CELL, transform)
    def make(material):
        traced = [(jnp.asarray(m), [jnp.asarray(v) for v in vv]) for m, vv in regs]
        traced[1] = (material, traced[1][1])
        data = b.replay_interfaces(jnp, traced, 1., CELL, transform, plan)
        return b.distance_gradient_normal_field(jnp, [v for _, vv in traced for v in vv],
                                                (24, 24), CELL, jnp.float64,
                                                segment_data=data, images=plan.images)
    compiled = jax.jit(make)
    for material in (4.+0j, 4.+.2j, 9.+0j):
        got = compiled(jnp.asarray(material))
        want = field(partial_regions(material), grid=(24, 24))
        np.testing.assert_allclose(nvf_projector(got), nvf_projector(want), atol=2e-11, rtol=0)


def test_frozen_plan_rejects_changed_periodic_contact_topology():
    from wavebit.rcwa._engine.geometry import StaleGeometryError
    lattice = ((700., 200.), (0., 700.))
    plan = b.plan_layer(_shapes(), lattice)
    moved = _shapes()
    moved[0]['ly'] = 699.
    with pytest.raises(StaleGeometryError, match='interface'):
        b.check_plan_still_fits(moved, lattice, plan)
    # Material changes are not topology changes.
    changed = _shapes()
    changed[1]['material'] = 9.
    b.check_plan_still_fits(changed, lattice, plan)


def test_epsilon_and_mu_build_their_own_actual_interface_sets(monkeypatch):
    from wavebit.materials.response import Material
    saved = b.normal_projector_coeffs
    active_counts = []
    def capture(*args, **kwargs):
        active_counts.append(int(np.count_nonzero(kwargs['segment_data'][3])))
        return saved(*args, **kwargs)
    monkeypatch.setattr(b, 'normal_projector_coeffs', capture)
    shapes = [dict(type='rectangle', lx=400., ly=400., angle=0., material=Material(4.)),
              dict(type='rectangle', lx=200., ly=200., angle=.3, material=Material(4., mu=2.))]
    solver = _legacy_wavebit.rcwa(backend='numpy', lattice=CELL, structure=[dict(background=1., shapes=shapes)],
                         thickness=[100.], wavelength=900., truncation_order=(1, 1),
                         shape_nvf_grid_size=32)
    result = solver.solve()
    assert np.isfinite(np.asarray(result.T)).all()
    assert active_counts == [4, 8]  # dielectric outer contour, magnetic inner contour


@pytest.mark.parametrize('order', [1, 3, 5])
def test_polygon_grating_matches_li_and_has_no_transverse_order_dependence(order):
    from wavebit.rcwa._engine import parametric
    shapes = _shapes(350.)
    def transmission(orders, structure, **kwargs):
        parametric._PROJECTOR_CACHE.clear()
        solver = _legacy_wavebit.rcwa(backend='numpy', period=(700., 700.), structure=structure,
                             thickness=[300.], wavelength=550., psi=.31,
                             truncation_order=orders,
                             **dict(dict(shape_nvf_grid_size=64) if isinstance(structure, list) else {}, **kwargs))
        return float(np.sum(np.asarray(solver.solve().T)))
    layer = [dict(background=1., shapes=shapes)]
    one = transmission((order, 0), layer)
    two = transmission((order, order), layer)
    raster = np.ones((1, 4, 4)); raster[:, :, 1:3] = 4.
    li = transmission((order, order), raster, factorization='li')
    np.testing.assert_allclose(two, one, atol=2e-11, rtol=0)
    np.testing.assert_allclose(two, li, atol=2e-11, rtol=0)


@pytest.mark.parametrize('center', [(350., 350.), (313.7, 381.2)])
def test_nested_paint_order_is_invariant_away_from_the_cell_center(center):
    outer = rect_verts(xp, *center, 417., 361., .13)
    inner = rect_verts(xp, *center, 170., 133., .43)
    for x, y in zip(mats(resolved([outer, inner], [4., 4.])),
                    mats(resolved([inner, outer], [4., 4.]))):
        np.testing.assert_allclose(x, y, atol=2e-12, rtol=0)


def test_same_material_partial_overlap_is_independent_of_paint_order():
    left = rect_verts(xp, 299., 313., 331., 297., .17)
    right = rect_verts(xp, 429., 391., 273., 331., .43)
    a, z = resolved([left, right], [4., 4.]), resolved([right, left], [4., 4.])
    np.testing.assert_allclose(nvf_projector(field(a)), nvf_projector(field(z)), atol=2e-12, rtol=0)
    for x, y in zip(mats(a), mats(z)):
        np.testing.assert_allclose(x, y, atol=2e-12, rtol=0)


def test_grouped_image_bound_matches_the_full_bound():
    from wavebit._common.geometry.polygon import ellipse_verts
    lattice = ((700., 180.), (0., 430.))
    regions = [(4., [rect_verts(xp, 321., 221., 310., 90., .27)]),
               (9., [ellipse_verts(xp, 510., 307., 70., 53., .19, 64)])]
    transform = b.reduction_transform(lattice)
    plan = b.plan_interfaces(regions, lattice, transform)
    data = b.replay_interfaces(xp, regions, 1., lattice, transform, plan)
    loops = [v for _, vv in regions for v in vv]
    assert sum(map(len, plan.groups)) < len(plan.pieces) * len(plan.images)
    broad = b.distance_gradient_normal_field(xp, loops, (31, 33), lattice, xp.float64,
                                             segment_data=data, images=plan.images,
                                             transform=transform)
    grouped = b.distance_gradient_normal_field(xp, loops, (31, 33), lattice, xp.float64,
                                               segment_data=data, images=plan.images,
                                               image_groups=plan.groups, transform=transform)
    np.testing.assert_array_equal(nvf_projector(grouped), nvf_projector(broad))


def test_a_polygon_grating_has_an_exact_normal_everywhere():
    outer = rect_verts(xp, 350., 350., 350., 700., 0.)
    inner = rect_verts(xp, 350., 350., 100., 80., .31)
    nx, ny = field(resolved([outer, inner], [4., 4.]))
    np.testing.assert_array_equal(nx * nx, 1.)
    np.testing.assert_array_equal(ny, 0.)


def test_material_gradient_with_a_frozen_interface_plan():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    regs = partial_regions(9.)
    transform = b.reduction_transform(CELL)
    plan = b.plan_interfaces(regs, CELL, transform)
    def loss(material):
        traced = [(jnp.asarray(m), [jnp.asarray(v) for v in vv]) for m, vv in regs]
        traced[1] = (material, traced[1][1])
        mats = b.conv_mats(jnp, traced, 1., 1, 1, CELL, 32, jnp.float64, jnp.complex128,
                           transform=transform, interface_plan=plan)
        return jnp.real(jnp.sum(mats[0] * jnp.conj(mats[0])))
    got = jax.jit(jax.grad(loss))(jnp.asarray(9.))
    h = 1e-5
    want = (loss(9.+h) - loss(9.-h)) / (2*h)
    np.testing.assert_allclose(got, want, atol=2e-8, rtol=2e-7)


def test_disabling_the_material_contrast_is_detected(monkeypatch):
    # Production now removes an invisible interval through its zero contrast. Forcing
    # only the old boolean mask on is neutral; force the live contrast on instead.
    original = b.interface_contrasts
    def all_edges(xp, *args):
        return xp.ones_like(original(xp, *args))
    monkeypatch.setattr(b, 'interface_contrasts', all_edges)
    with pytest.raises(AssertionError):
        test_nested_same_material_paint_order_does_not_change_matrices()


def test_zero_area_backtracking_spur_cannot_hide_a_real_exterior_interval():
    left = rect_verts(xp, 250., 350., 300., 300., 0.)
    right = rect_verts(xp, 450., 350., 300., 300., 0.)
    union = rect_verts(xp, 350., 350., 500., 300., 0.)
    regs = resolved([left, right], [4., 4.])
    # The clipper's first contour has an out-and-back run on the second box's
    # bottom edge. Pairwise opposed-edge deletion wrongly removes all three.
    np.testing.assert_allclose(nvf_projector(field(regs)), nvf_projector(field([(4., [union])])),
                               atol=2e-12, rtol=0)
    for got, want in zip(mats(regs), mats([(4., [union])])):
        np.testing.assert_allclose(got, want, atol=2e-12, rtol=0)


def test_two_periodic_images_can_jointly_cover_a_face():
    lattice = ((1.25, 1.25), (1.25, -1.25))
    box = rect_verts(xp, 1.25, 0., 2.5, 1.25, 0.)
    nx, ny = field([(4., [box])], lattice, grid=(32, 32))
    np.testing.assert_array_equal(nx, 1.)
    np.testing.assert_array_equal(ny, 0.)


def test_one_edge_can_have_both_equal_and_unequal_material_neighbors():
    regs = [(4., [rect_verts(xp, 200., 350., 300., 400., 0.)]),
            (4., [rect_verts(xp, 450., 450., 200., 200., 0.)]),
            (9., [rect_verts(xp, 450., 250., 200., 200., 0.)])]
    transform = b.reduction_transform(CELL)
    plan = b.plan_interfaces(regs, CELL, transform)
    a, e, l2, live = b.replay_interfaces(xp, regs, 1., CELL, transform, plan)
    a, e, l2 = b.periodic_segments(xp, a, e, l2, CELL, transform, images=plan.images)
    live = np.tile(live, len(plan.images))
    distance, _ = b._seg_dist(xp, np.array([[350., 450.], [350., 250.]]), a, e, l2)
    got = np.min(np.where(live[None, :], distance, np.inf), axis=1)
    np.testing.assert_allclose(got, [10000., 0.], atol=2e-12, rtol=0)
