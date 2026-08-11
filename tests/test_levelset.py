"""
Verification tests for the LEVEL-SET path -- the zero contour of a field, transformed exactly.

Covers: the marching-triangle decomposition against the closed-form circle it approximates, the
periodic boundary (a design that percolates across the cell, which is the case a polygon
representation has to special-case and this one does not), the aliasing floor, staleness, and
the shape derivative.

Three facts these tests exist to pin down:

* **The area and the contour must be one decomposition, not two.** The area is the ``k = 0``
  Fourier coefficient and the contour supplies every other order; read off two different
  resolutions of the same cell they would disagree, and the disagreement would be invisible --
  the answer stays smooth and plausible. Triangles have no ambiguous case, which is why they are
  what this path cuts the grid into.
* **No polygon is ever assembled**, so a boundary that wraps the period is not a special case.
  ``test_a_stripe_that_wraps_the_period_is_exact`` is the statement of that: the transform of a
  cell-spanning stripe comes out at 1e-15 of the closed-form rectangle, with the boundary
  crossing the cell edge twice and closing only on the torus.
* **A gradient can be exactly right and mean nothing.** Autodiff and a finite difference agree
  by construction on one smooth piece of the loss, on every path in this repository -- so the
  shape derivative here is checked against the *closed-form circle path's* own derivative in its
  radius, which is the same physical quantity reached by an unrelated discretisation. See
  ``test_the_shape_derivative_converges_to_the_closed_form_one``.
"""
import os

import numpy as np
import pytest

os.environ.setdefault('JAX_ENABLE_X64', '1')

import array_api_compat.numpy as xnp

import fieldwave
from fieldwave._core import levelset as _ls
from fieldwave._core.analytic import assemble_eps_coeffs
from fieldwave._core.polygon import edge_ft

P = 1.0
PERIOD = (P, P)
R = 0.30


# --------------------------------------------------------------------------- helpers
def grid(N):
    """The (X, Y) the level set is sampled on -- pixel centres, the convention of this repo."""
    j = (np.arange(N) + 0.5) / N * P
    return np.meshgrid(j, j, indexing='xy')


def phi_circle(N, r=R, cx=0.5, cy=0.5):
    """A signed distance function, which is what a level-set scheme reinitialises to."""
    X, Y = grid(N)
    return r - np.hypot(X - cx, Y - cy)


def transform(phi, fto, plan=None):
    """``(eps-indicator FT, area)`` for one level set, on the doubled reciprocal grid."""
    plan = _ls.plan_layer(phi, PERIOD) if plan is None else plan
    A, E, area = _ls.boundary_and_area(xnp, phi, plan, np.float64)
    k = np.arange(-2 * fto, 2 * fto + 1, dtype=float)
    KY = k[:, None] + 0.0 * k[None, :]
    KX = 0.0 * k[:, None] + k[None, :]
    return edge_ft(xnp, A, E, area, KY, KX, PERIOD), float(area)


def circle_coeffs(fto, r=R, n_index=3.0, n_base=1.0):
    """The same thing in closed form, from the analytic shape path."""
    shapes = [dict(type='circle', cx=0.5, cy=0.5, lx=2 * r, n_index=n_index)]
    eps_c, _ = assemble_eps_coeffs(xnp, shapes, n_base, fto, fto, PERIOD, np.complex128, None)
    return (eps_c - n_base ** 2 * ((np.arange(-2 * fto, 2 * fto + 1)[:, None] == 0) &
                                   (np.arange(-2 * fto, 2 * fto + 1)[None, :] == 0))) \
        / (n_index ** 2 - n_base ** 2)


SOLVE = dict(period=[P, P], thickness=[0.4], wavelength=0.8, theta=0.0, phi=0.0,
             n_top=1.0, n_bot=1.5, fto=[6, 6])


# --------------------------------------------------------------------------- geometry
def test_the_area_converges_on_the_circle_at_second_order():
    """The enclosed area is the DC coefficient, so its error is an error on every low order."""
    errs = [abs(transform(phi_circle(N), 4)[1] - np.pi * R ** 2) for N in (64, 128, 256, 512)]
    for coarse, fine in zip(errs, errs[1:]):
        assert 3.2 < coarse / fine < 5.0, errs        # midpoint sampling: 4x per doubling


def test_the_area_is_exactly_the_dc_fourier_coefficient():
    """Two formulas, one decomposition -- this is what marching *triangles* buys."""
    ft, area = transform(phi_circle(128), 5)
    assert ft[10, 10].imag == pytest.approx(0.0, abs=1e-15)
    assert ft[10, 10].real == pytest.approx(area / (P * P), rel=1e-14)


def test_the_permittivity_coefficients_converge_on_the_closed_form_circle():
    ref = circle_coeffs(6)
    errs = [np.abs(transform(phi_circle(N), 6)[0] - ref).max() for N in (64, 128, 256, 512)]
    for coarse, fine in zip(errs, errs[1:]):
        assert 3.2 < coarse / fine < 5.0, errs
    assert errs[-1] < 2e-5


def test_the_boundary_closes():
    """Segment vectors summing to zero says the segments form closed loops, nothing else does.

    Nothing in the build ever links them, so this is the only statement that the decomposition
    produced a boundary at all rather than a bag of unrelated edges.
    """
    plan = _ls.plan_layer(phi_circle(128), PERIOD)
    _, E, _ = _ls.boundary_and_area(xnp, phi_circle(128), plan, np.float64)
    assert np.abs(np.asarray(E).sum(0)).max() < 1e-14


def test_a_stripe_that_wraps_the_period_is_exact():
    """The one case a polygon representation cannot state, and this one needs nothing for.

    The stripe's boundary is two straight lines that leave the cell and re-enter it; they close
    only on the torus. Because the transform is a sum over *segments* rather than an integral
    around an assembled loop, that costs nothing and the answer is the exact rectangle's -- at
    every grid resolution, because a straight axis-aligned edge is resolved exactly.
    """
    shapes = [dict(type='rectangle', cx=0.5, cy=0.5, lx=0.4, ly=P, n_index=3.0)]
    eps_c, _ = assemble_eps_coeffs(xnp, shapes, 1.0, 6, 6, PERIOD, np.complex128, None)
    ref = (eps_c - np.eye(1) * 0) / 1.0
    dc = (np.arange(-12, 13)[:, None] == 0) & (np.arange(-12, 13)[None, :] == 0)
    ref = (np.asarray(eps_c) - 1.0 * dc) / (3.0 ** 2 - 1.0 ** 2)

    for N in (64, 128, 256):
        X, _ = grid(N)
        ft, area = transform(0.2 - np.abs(X - 0.5), 6)
        assert area == pytest.approx(0.4, abs=1e-14)
        assert np.abs(np.asarray(ft) - ref).max() < 1e-14


def test_a_design_that_percolates_in_both_axes_converges():
    """A cross touches every cell edge and is one connected region of genus 1 on the torus."""
    from fieldwave._core import clipper as _c
    from fieldwave._core.boundary import plan_layer as bplan, shape_vertices
    from fieldwave._core.polygon import polygon_ft

    shapes = [dict(type='rectangle', cx=0.5, cy=0.5, lx=0.4, ly=P, n_index=3.0),
              dict(type='rectangle', cx=0.5, cy=0.5, lx=P, ly=0.4, n_index=3.0)]
    recs, regions = bplan(shapes, PERIOD)               # the rods overlap: paint, do not sum
    vals = _c.eval_table(xnp, recs, [shape_vertices(xnp, s, PERIOD) for s in shapes])
    k = np.arange(-12, 13, dtype=float)
    KY, KX = k[:, None] + 0.0 * k[None, :], 0.0 * k[:, None] + k[None, :]
    ref = sum(polygon_ft(xnp, _c.loop_verts(xnp, lp, vals), KY, KX, PERIOD)
              for region in regions for lp, _s in region)

    errs = []
    for N in (64, 128, 256):
        X, Y = grid(N)
        phi = np.maximum(0.2 - np.abs(X - 0.5), 0.2 - np.abs(Y - 0.5))
        ft, area = transform(phi, 6)
        assert area == pytest.approx(0.4 + 0.4 - 0.16, abs=2e-4)
        errs.append(np.abs(np.asarray(ft) - np.asarray(ref)).max())
    assert errs[-1] < errs[0] / 20, errs


def test_the_normal_field_is_unit_everywhere():
    """``C_xx + C_yy = I`` in the factorization holds only if it is, interface or bulk."""
    nx, ny = _ls.normal_field(xnp, phi_circle(96), PERIOD)
    assert np.abs(np.asarray(nx) ** 2 + np.asarray(ny) ** 2 - 1.0).max() < 1e-13


def test_the_normal_field_gets_the_two_axes_the_right_way_round():
    """A square cell hides an x/y swap completely; only a non-square one can see it.

    ``period[0]`` is Px and axis 1 of ``phi`` is x, here as everywhere else. On a disc in a 2:1
    cell the normal must still point radially *in real space*, which is what a swap breaks.
    """
    Px, Py, N = 2.0, 1.0, 128
    x = (np.arange(N) + 0.5) / N * Px
    y = (np.arange(N) + 0.5) / N * Py
    X, Y = np.meshgrid(x, y, indexing='xy')
    phi = 0.3 - np.hypot(X - Px / 2, Y - Py / 2)

    nx, ny = (np.asarray(v) for v in _ls.normal_field(xnp, phi, (Px, Py)))
    band = np.abs(phi) < 0.05                                   # read it near the interface
    rx, ry = X - Px / 2, Y - Py / 2
    r = np.hypot(rx, ry)
    # the field points *away* from the disc (phi decreases outward), so compare against -r_hat
    cos = -(nx[band] * rx[band] + ny[band] * ry[band]) / r[band]
    # 1.7e-07 as written. A swapped pair puts this at ~1.0 -- the normal comes out near
    # perpendicular to the correct one, so any threshold catches it; 1e-05 is tight enough to
    # also catch a partial regression, and still 60x above what the sampling achieves.
    assert 1.0 - cos.min() < 1e-5, 1.0 - cos.min()


# --------------------------------------------------------------------------- the plan
def test_the_plan_hashes_by_content_so_an_unchanged_topology_does_not_recompile():
    a = _ls.plan_layer(phi_circle(64), PERIOD)
    b = _ls.plan_layer(phi_circle(64) * 1.7, PERIOD)          # same signs, different values
    c = _ls.plan_layer(phi_circle(64, r=0.20), PERIOD)
    assert a == b and hash(a) == hash(b)
    assert a != c


def test_a_plan_knows_when_its_sign_pattern_has_moved():
    plan = _ls.plan_layer(phi_circle(64), PERIOD)
    assert not plan.is_stale_for(phi_circle(64) * 0.5)
    assert plan.is_stale_for(phi_circle(64, r=0.31))


def test_a_misspelled_layer_key_is_refused_rather_than_ignored():
    """The failure this guards is silent: no shapes and no phi is a legal *uniform* layer.

    Found the hard way — a level-set layer read by a build that had no level-set path came back
    as a slab of ``n_base``, solved, and conserved energy to 1e-14.
    """
    with pytest.raises(ValueError, match='not read by any path'):
        fieldwave.rcwa(ucell=[dict(n_base=1.0, n_index=3.0, pih=phi_circle(64))],
                       **SOLVE).conv_solve()
    with pytest.raises(ValueError, match='needs n_index'):
        fieldwave.rcwa(ucell=[dict(n_base=1.0, phi=phi_circle(64))], **SOLVE).conv_solve()
    with pytest.raises(ValueError, match='not both'):
        fieldwave.rcwa(ucell=[dict(n_base=1.0, n_index=3.0, phi=phi_circle(64),
                                   shapes=[dict(type='circle', lx=0.2, n_index=2.0)])],
                       **SOLVE).conv_solve()


def test_a_grid_below_the_aliasing_floor_is_refused():
    """`4*fto+1` per axis. Structural, not gradual -- see boundary.check_normal_grid."""
    solver = fieldwave.rcwa(ucell=[dict(n_base=1.0, n_index=3.0, phi=phi_circle(16))], **SOLVE)
    with pytest.raises(ValueError, match='aliases'):
        solver.conv_solve()


def test_a_topology_change_is_a_new_plan_and_the_area_stays_continuous():
    """Two blobs merging: the plan must change, the geometry must not jump."""
    N = 256
    X, Y = grid(N)

    def two(d):
        return np.maximum(0.15 - np.hypot(X - (0.5 - d / 2), Y - 0.5),
                          0.15 - np.hypot(X - (0.5 + d / 2), Y - 0.5))

    def union_area(d, r=0.15):
        """Two discs, exactly -- 2*pi*r^2 while apart, minus the lens once they overlap."""
        if d >= 2 * r:
            return 2 * np.pi * r * r
        lens = 2 * r * r * np.arccos(d / (2 * r)) - (d / 2) * np.sqrt(4 * r * r - d * d)
        return 2 * np.pi * r * r - lens

    plans = []
    for d in (0.32, 0.30, 0.28, 0.26):
        plans.append(_ls.plan_layer(two(d), PERIOD))
        area = transform(two(d), 4, plans[-1])[1]
        # The merged region is one loop where the pair was two, and nothing in the build had to
        # be told: the same case table cuts both, and the transform never links loops at all.
        assert area == pytest.approx(union_area(d), abs=3e-4), d
    assert len(set(plans)) == 4                                # every step is its own topology


# --------------------------------------------------------------------------- through the solve
def test_the_solve_converges_on_the_analytic_circle_path():
    ref = fieldwave.rcwa(ucell=[dict(n_base=1.0, shapes=[
        dict(type='circle', cx=0.5, cy=0.5, lx=2 * R, n_index=3.0)])], **SOLVE).conv_solve()
    target = float(ref.res.de_ti[6, 6])

    errs = []
    for N in (64, 128, 256):
        r = fieldwave.rcwa(ucell=[dict(n_base=1.0, n_index=3.0, phi=phi_circle(N))],
                           **SOLVE).conv_solve()
        assert float(np.sum(r.res.de_ti) + np.sum(r.res.de_ri)) == pytest.approx(1.0, abs=1e-9)
        errs.append(abs(float(r.res.de_ti[6, 6]) - target))
    assert errs[-1] < errs[0] / 20, errs


def _specular(ucell, fto, **kw):
    opt = dict(period=[P, P], thickness=[0.4], wavelength=0.8, theta=0., phi=0.,
               n_top=1.0, n_bot=1.5, fto=[fto, fto])
    return float(fieldwave.rcwa(ucell=ucell, **opt, **kw).conv_solve().res.de_ti[fto, fto])


def test_it_beats_the_raster_it_replaces_on_the_same_field_at_every_truncation():
    """One phi, two ways to hand it over. The **magnitude** is the claim, not a trend.

    An earlier version of this test asserted the ratio grew with `fto`, quoting 26x at 4 and
    38x at 10. It passes, and it is picking two points out of 26 / 90 / 33 / 38 / 53 — the
    sequence is not monotone and the growth claim was imported from `docs/boundary-modeling.md`,
    where it was measured against S4, without being measured here. What is robust is that the
    level set is more than an order of magnitude closer at every truncation.

    The reference is taken at the **same fto** as the paths under test, which is the whole
    reason the number means anything: it is the same solve with exact geometry, so truncation
    is common-mode and the difference is geometry. Against a converged reference at a high fixed
    fto the level set reads *worse* than the staircase at fto 4, because there the difference is
    truncation rather than shape.
    """
    phi = phi_circle(128)
    raster = np.where(phi > 0, 3.0, 1.0)[None, :, :]
    circle = [dict(n_base=1.0, shapes=[dict(type='circle', cx=0.5, cy=0.5, lx=2 * R,
                                            n_index=3.0)])]

    ratios = []
    for fto in (4, 8, 12):
        truth = _specular(circle, fto)
        ras = abs(_specular(raster, fto, factorization='nvf') - truth)
        lvl = abs(_specular([dict(n_base=1.0, n_index=3.0, phi=phi)], fto) - truth)
        ratios.append(ras / lvl)

    assert min(ratios) > 10, ratios


def test_the_rasters_error_is_bought_with_grid_not_with_truncation():
    """The asymmetry that decides which knob an optimiser has to pay: at fixed N the staircase
    barely improves with fto, while doubling N buys several times more."""
    circle = [dict(n_base=1.0, shapes=[dict(type='circle', cx=0.5, cy=0.5, lx=2 * R,
                                            n_index=3.0)])]

    def raster_err(N, fto):
        r = np.where(phi_circle(N) > 0, 3.0, 1.0)[None, :, :]
        return abs(_specular(r, fto, factorization='nvf') - _specular(circle, fto))

    by_fto = raster_err(128, 4) / raster_err(128, 12)        # fto 4 -> 12 at fixed N
    by_grid = raster_err(128, 12) / raster_err(256, 12)      # N 128 -> 256 at fixed fto
    assert by_fto < 2.0, by_fto
    assert by_grid > 2.0 * by_fto, (by_grid, by_fto)


def test_the_replay_gives_the_same_five_matrices_on_torch():
    """One plan, two backends. The replay is arithmetic, so this is the statement that it is."""
    torch = pytest.importorskip('torch')
    from fieldwave._core.backend import array_namespace

    phi = phi_circle(96)
    plan = _ls.plan_layer(phi, PERIOD)
    t = torch.tensor(phi, dtype=torch.float64, requires_grad=True)
    ref = _ls.to_conv_mat_levelset_layer(xnp, dict(n_base=1.0, n_index=3.0, phi=phi),
                                         6, 6, PERIOD, np.complex128, plan=plan)
    got = _ls.to_conv_mat_levelset_layer(array_namespace(t),
                                         dict(n_base=1.0, n_index=3.0, phi=t),
                                         6, 6, PERIOD, torch.complex128, plan=plan)
    for a, b in zip(ref, got):
        assert float(torch.abs(torch.as_tensor(a) - b).max().detach()) < 1e-13

    # and the plan can be taken from a grad-tracking tensor: planning reads only the sign
    torch.abs(got[0]).sum().backward()
    assert bool(torch.isfinite(t.grad).all())


# --------------------------------------------------------------------------- the gradient
jax = pytest.importorskip('jax')
import jax.numpy as jnp                                        # noqa: E402
import fieldwave.on_jax                                        # noqa: E402,F401  (x64)

GRAD = dict(SOLVE, fto=[8, 8], backend=1)


def _merit(res):
    return jnp.abs(res.res.de_ti[8, 8])


def test_the_gradient_of_every_node_is_finite_and_sums_to_the_uniform_offset_one():
    base = jnp.asarray(phi_circle(128))
    solver = fieldwave.rcwa(ucell=[dict(n_base=1.0, n_index=3.0, phi=base)], **GRAD)
    geom = solver.freeze_geometry()

    def J(p):
        return _merit(solver.conv_solve(ucell=[dict(n_base=1.0, n_index=3.0, phi=p)],
                                        geometry=geom))

    g = np.asarray(jax.grad(J)(base))
    assert np.isfinite(g).all()
    # Adding a constant to phi moves the whole boundary out by it, so the two must agree.
    g_off = float(jax.grad(lambda c: J(base + c))(jnp.asarray(0.0)))
    assert g.sum() == pytest.approx(g_off, rel=1e-11)


def test_the_shape_derivative_converges_to_the_closed_form_one():
    """The check autodiff-versus-finite-difference cannot be.

    Both agree on one smooth piece of the loss by construction, on every path here, so agreeing
    says nothing about whether the slope is the *structure's*. For ``phi = R - r`` a uniform
    offset ``c`` is exactly ``R -> R + c``, which makes the closed-form circle path's ``dJ/dR``
    the same physical derivative obtained from an unrelated discretisation. Converging onto it
    is what makes this a shape derivative rather than a discretisation's own.
    """
    def circle_J(r):
        s = fieldwave.rcwa(ucell=[dict(n_base=1.0, shapes=[
            dict(type='circle', cx=0.5, cy=0.5, lx=2 * r, n_index=3.0)])], **GRAD)
        return _merit(s.conv_solve())

    ref = float(jax.grad(circle_J)(jnp.asarray(R)))

    errs = []
    for N in (64, 128, 256):
        base = jnp.asarray(phi_circle(N))
        solver = fieldwave.rcwa(ucell=[dict(n_base=1.0, n_index=3.0, phi=base)], **GRAD)
        geom = solver.freeze_geometry()
        g = float(jax.grad(lambda c: _merit(solver.conv_solve(
            ucell=[dict(n_base=1.0, n_index=3.0, phi=base + c)], geometry=geom)))(
                jnp.asarray(0.0)))
        errs.append(abs(g - ref) / abs(ref))

    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < 3e-3, errs                     # 0.12% at N=256, from 1.6% at N=64
