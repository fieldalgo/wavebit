"""
Verification tests for `structure_slice` / `plot_structure` -- the picture of the *input*.

The reason this file exists is one hazard, and most of it is spent on that hazard: a
``Modeling.draw()`` edge list is not guaranteed increasing, so the obvious render of a
block-modelled layer is silently wrong for any shape that crosses the cell boundary. The
naive render is therefore checked in *alongside* the real one and asserted to be wrong, so
the test says what the hazard is rather than only that the fix holds. Delete
``naive_render`` and this file stops making its argument.

The other three claims pinned here:

* the analytic sampler agrees with the layer's own DC coefficient, on a skewed cell as well
  as an axis-aligned one -- which is the only thing that catches a wrong periodic-image
  shift, since a shape sitting comfortably inside the cell cannot tell;
* a painted, overlapping layer goes through the resolved loops and not through the summed
  rule, which agrees to the eye and differs by 5.4% in ``<eps>``;
* the band-limited reconstruction *converges* on the geometry as ``max_order`` rises. That is the
  check that catches a flipped order-index sign, which would mirror the picture: a mirrored
  reconstruction of an asymmetric layer does not converge on anything.
"""
import os

import numpy as np
import pytest

os.environ.setdefault('JAX_ENABLE_X64', '1')

import array_api_compat.numpy as xnp
import wavebit
from wavebit._core import boundary as _b
from wavebit._core import clipper as _c
from wavebit._core.analytic import assemble_eps_coeffs
from wavebit._core.draw import _render_block, structure_slice

P = 1000.0
COMMON = dict(backend=0, period=(P, P), thickness=(300.,), wavelength=900.,
              top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0, max_order=(9, 9))

#: A 200x160 rod of permittivity 12.25 in a 1000x1000 cell of permittivity 1: `<eps>` is
#: `1 + 0.032*(12.25 - 1) = 1.36` wherever it is placed, since a translation is not a
#: different structure. Every straddle case below is that same rod.
ROD = dict(lx=200.0, ly=160.0, eps=12.25)
ROD_EPS = 1.36


def rod_instruction(cx, cy):
    return [[1.0, [['rectangle', cx, cy, ROD['lx'], ROD['ly'], ROD['eps'], 0.0, 0, 0]]]]


def rod_truth(cx, cy, n):
    """`eps` of the rod sampled at cell centres, wrapped -- the geometry, independently built."""
    s = (np.arange(n) + 0.5) * P / n
    X, Y = np.meshgrid(s, s)
    dx = (X - cx + P / 2) % P - P / 2
    dy = (Y - cy + P / 2) % P - P / 2
    return np.where((np.abs(dx) <= ROD['lx'] / 2) & (np.abs(dy) <= ROD['ly'] / 2),
                    ROD['eps'], 1.0)


def naive_render(cell, x_list, y_list, n):
    """The wrong render, kept on purpose: edges as right-hand edges, bands painted in order.

    This is what a reader writes from ``draw()``'s return value without knowing that an edge
    list may leave ``[0, period]``. It is right whenever every shape sits inside the cell,
    which is why nothing catches it.
    """
    xe = np.real(np.asarray(x_list)).reshape(-1)
    ye = np.real(np.asarray(y_list)).reshape(-1)
    s = (np.arange(n) + 0.5) * P / n
    ci = np.clip(np.searchsorted(xe, s, 'left'), 0, len(xe) - 1)
    ri = np.clip(np.searchsorted(ye, s, 'left'), 0, len(ye) - 1)
    # `cell` is permittivity, as `_render_block` reads it -- no squaring on either side.
    return np.real(np.asarray(cell))[np.ix_(ri, ci)]


def block_layer(instruction):
    solver = wavebit.rcwa(structure=instruction, **COMMON)
    return solver, solver.modeling_block_instruction(instruction)[0]


# ------------------------------------------------------- the straddle, which is the whole point

#: `(cx, cy, how much of the rod a naive render keeps, its <eps>)`. The four positions are the
#: ones `test_a_solver_answer_does_not_move_with_the_rod` shows the solver cannot distinguish.
STRADDLES = [
    (400.0, 500.0, 1.00, 1.3600),      # inside the cell: both renders are right
    (550.0, 500.0, 1.00, 1.3600),      # inside the cell
    (50.0, 500.0, 0.75, 1.2700),       # crosses x = 0     -> edges [-50, 150, 1000]
    (960.0, 500.0, 0.70, 1.2520),      # crosses x = Px    -> edges [860, 1060, 1000], decreasing
    (50.0, 40.0, 0.5625, 1.2025),      # crosses both axes
]


@pytest.mark.parametrize('cx,cy,kept,naive_eps', STRADDLES)
def test_the_block_render_is_exact_at_every_straddle(cx, cy, kept, naive_eps):
    """max|render - truth| == 0, not "close": the render is a sum of exact segment values.

    Nothing here is interpolated. Each sample takes the modeler's own segment value through a
    signed periodic-image count, so on a rod whose edges avoid the sample centres the two
    arrays are equal bit-for-bit, and any tolerance would be hiding something.
    """
    n = 400
    _, (cell, x_list, y_list) = block_layer(rod_instruction(cx, cy))
    s = (np.arange(n) + 0.5) * P / n
    got = _render_block(cell, x_list, y_list, (P, P), s, s).real

    assert np.abs(got - rod_truth(cx, cy, n)).max() == 0.0
    assert got.mean() == pytest.approx(ROD_EPS, abs=1e-12)


@pytest.mark.parametrize('cx,cy,kept,naive_eps', STRADDLES)
def test_the_naive_render_loses_the_wrapped_part(cx, cy, kept, naive_eps):
    """The hazard, asserted rather than described.

    A straddling rod loses 25% / 30% / 43.8% of its area to the in-order render and lands
    ``<eps>`` at 1.2700 / 1.2520 / 1.2025 against the 1.3600 the solver integrates. The two
    interior positions are in the list to show what makes this so easy to ship: there the
    naive render is exactly right.
    """
    n = 400
    _, (cell, x_list, y_list) = block_layer(rod_instruction(cx, cy))
    truth = rod_truth(cx, cy, n)
    naive = naive_render(cell, x_list, y_list, n)

    area_kept = (naive > 1.5).sum() / (truth > 1.5).sum()
    assert area_kept == pytest.approx(kept, abs=5e-3)
    assert naive.mean() == pytest.approx(naive_eps, abs=1e-3)
    if kept == 1.0:
        assert np.abs(naive - truth).max() == 0.0
    else:
        assert np.abs(naive - truth).max() > 11.0


def test_the_edge_list_really_does_leave_the_cell():
    """Pins the premise, so a change to the modeler that removes the hazard is visible here.

    ``draw()`` places the edges where the shape is, not where the cell is: at ``cx = 960`` the
    array *decreases* over its last step. If this ever starts coming back sorted and inside
    ``[0, Px]``, the tests above go from load-bearing to redundant, and this failure is what
    says so.
    """
    _, (_, x_list, _) = block_layer(rod_instruction(960.0, 500.0))
    x = np.real(np.asarray(x_list)).reshape(-1)
    np.testing.assert_allclose(x, [860.0, 1060.0, 1000.0])
    assert x[-1] < x[-2]

    _, (_, x_low, _) = block_layer(rod_instruction(50.0, 500.0))
    assert np.real(np.asarray(x_low)).reshape(-1)[0] < 0.0


def test_a_solver_answer_does_not_move_with_the_rod():
    """The reason the render has to wrap: the *solver* already does, so the picture owes it.

    Same rod named in four periods is one structure, and the integrated series carries the
    wrapped part on the Fourier phase. The spread measured across the four is 1.4e-13 on
    ``sum(R)`` and 5.4e-14 on ``sum(T)``; the bound below is loose by ~70x on purpose, since
    the failure this guards against is the *picture*'s failure -- dropping the wrapped part
    moves ``<eps>`` by 0.09, twelve orders above any BLAS-dependent noise here.
    """
    totals = [float(np.sum(wavebit.rcwa(structure=rod_instruction(cx, 500.0),
                                        **COMMON).solve().R).real)
              for cx, _, _, _ in STRADDLES[:4]]
    assert np.ptp(totals) < 1e-11, totals


# --------------------------------------------------------------------------- the other paths

def test_the_raster_path_is_the_raster():
    """`structure[layer]`, untouched, at its own resolution -- `n` is deliberately ignored.

    Also pins the way up: row 0 is ``y = 0`` (``docs/conventions.md``), which is what makes
    ``origin='lower'`` line up with ``plot_field``'s top view.
    """
    N = 64
    cell = np.ones((2, N, N))
    cell[0, :N // 4, :] = 9.0             # a stripe at small y, in the top layer only
    cell[1] = 4.0
    solver = wavebit.rcwa(structure=cell, **COMMON)

    got = solver.structure_slice(layer=0, n=999)
    assert got.shape == (N, N)
    np.testing.assert_array_equal(got.real, cell[0])
    assert got[0, 0].real == 9.0 and got[-1, 0].real == 1.0

    assert solver.structure_slice(layer=1).real.mean() == pytest.approx(4.0)


def test_layer_is_indexed_the_way_the_structure_is():
    """0 is the layer light meets first, on every path -- and out of range says so."""
    shapes = [dict(type='circle', cx=500.0, cy=500.0, lx=200.0, material=9.0)]
    layers = [dict(background=1.0, shapes=[]), dict(background=1.0, shapes=shapes)]
    solver = wavebit.rcwa(structure=layers, **COMMON)

    assert solver.structure_slice(layer=0, n=128).real.mean() == pytest.approx(1.0)
    assert solver.structure_slice(layer=1, n=128).real.mean() > 1.2
    assert solver.structure_slice(layer=-1, n=128).real.mean() > 1.2
    with pytest.raises(IndexError, match='light meets first'):
        solver.structure_slice(layer=2)


@pytest.mark.parametrize('lattice', [None, [[P, P * 0.5], [0.0, P * 0.866]]])
def test_the_analytic_sampler_agrees_with_its_own_dc_coefficient(lattice):
    """Sampled area against the closed-form transform's DC term -- including a skewed cell.

    The skewed case is the one that carries weight. A periodic image is ``r + m*a1 + n*a2``,
    not ``r + (m*Px, n*Py)``, and a shape sitting inside the cell cannot tell the two apart;
    this layer has a rectangle at ``cy = 650`` with a 300 length across a sheared cell, so it
    does.
    """
    shapes = [dict(type='circle', cx=300.0, cy=300.0, lx=260.0, material=12.25),
              dict(type='rectangle', cx=740.0, cy=650.0, lx=300.0, ly=180.0,
                   angle=0.45, material=4.840000000000001)]
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], lattice=lattice, **COMMON)
    fx, fy = int(solver.max_order[0]), int(solver.max_order[1])
    dc = assemble_eps_coeffs(xnp, shapes, 1.0, fx, fy, solver.period, xnp.complex128,
                             lattice=solver.lattice)[0][2 * fy, 2 * fx].real

    assert solver.structure_slice(n=1024).real.mean() == pytest.approx(dc, rel=1e-4)


def test_a_circle_is_read_through_shape_geometry():
    """A circle names its diameter once. Reading the dict directly is how that stops being true.

    ``ly`` is absent here, and ``shape_geometry`` is the one place that knows it falls back to
    ``lx`` for a circle rather than to 0 or to 500. A second reader gets that wrong quietly --
    which is the defect that function was written to end.
    """
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=[
        dict(type='circle', cx=500.0, cy=500.0, lx=400.0, material=9.0)])], **COMMON)
    area = np.pi * 200.0 ** 2 / (P * P)
    assert solver.structure_slice(n=1024).real.mean() == pytest.approx(
        1.0 + area * (9.0 - 1.0), rel=1e-3)


# ------------------------------------------------------------------- painted, overlapping

OVERLAP = [dict(type='rectangle', cx=430.0, cy=500.0, lx=300.0, ly=300.0, material=4.0),
           dict(type='circle', cx=600.0, cy=520.0, lx=320.0, material=9.0)]


def loop_mean_eps(shapes, background=1.0):
    """`<eps>` straight off the boundary path's resolved loops -- areas, no sampling at all."""
    recs, regions = _b.plan_layer(shapes, (P, P))
    verts = [_b.shape_vertices(xnp, s, (P, P)) for s in shapes]
    vals = _c.eval_table(xnp, recs, verts)
    total = background * P * P
    for shape, region in zip(shapes, regions):
        eps_s = complex(shape['material']).real
        for loop in region:
            V = np.asarray(_c.loop_verts(xnp, loop, vals), float)
            # No sign factor: a hole arrives wound clockwise, so its shoelace is negative.
            total += (eps_s - background) * _c.signed_area(V)
    return total / (P * P)


def test_the_paint_renderer_matches_the_boundary_path_loops():
    """The band-limited DC reproduces the loop areas to ~1e-13; the sampler converges on them.

    Two different statements, and both are wanted. The coefficient path integrates the same
    polygons the solver integrates, so it is an identity up to float noise. The sampled render
    is a picture on a finite grid, so it converges instead -- 1.8137 / 1.8161 / 1.8159 at
    n = 128 / 1024 / 2048 -- and that is what says the scanline fill lands on the loops rather
    than merely averaging to them.
    """
    solver = wavebit.rcwa(structure=[dict(background=1.0, combine='paint', shapes=OVERLAP)], **COMMON)
    want = loop_mean_eps(OVERLAP)

    assert solver.structure_slice(n=64, band_limited=True).real.mean() == pytest.approx(
        want, abs=1e-13)
    assert solver.structure_slice(n=2048).real.mean() == pytest.approx(want, abs=2e-4)


def test_painting_and_adding_are_drawn_as_the_different_structures_they_are():
    """Same shape list, two ``combine`` values, two pictures -- 5.4% apart in ``<eps>``.

    Falling back to the summed sampler for a painted layer is the plausible wrong answer here:
    the overlap is drawn at ``eps_a + eps_b - eps_base`` instead of ``eps_b``, which looks like
    a structure someone might have meant.
    """
    paint = wavebit.rcwa(structure=[dict(background=1.0, combine='paint', shapes=OVERLAP)], **COMMON)
    add = wavebit.rcwa(structure=[dict(background=1.0, combine='add', shapes=OVERLAP)], **COMMON)

    e_paint = paint.structure_slice(n=2048).real.mean()
    e_add = add.structure_slice(n=2048).real.mean()
    assert e_paint == pytest.approx(1.8158, abs=1e-3)
    assert e_add == pytest.approx(1.9134, abs=1e-3)
    assert (e_add - e_paint) / e_paint == pytest.approx(0.054, abs=2e-3)


def test_a_hole_is_drawn_as_a_hole():
    """A shape painted *inside* another leaves a two-loop region, and the render cuts it out.

    The orientation carries it: the hole comes back **clockwise**, so its winding number is
    ``-1`` and the scanline fill subtracts it with no sign of its own -- the same way
    ``polygon_ft`` does. A renderer that took the absolute area instead would put ``<eps>`` at
    2.4256 where the geometry is 2.2371, which is what the solver itself did until the
    orientation stopped being carried in a separate field.
    """
    shapes = [dict(type='rectangle', cx=500.0, cy=500.0, lx=600.0, ly=600.0, material=4.0),
              dict(type='circle', cx=500.0, cy=500.0, lx=200.0, material=9.0)]
    solver = wavebit.rcwa(structure=[dict(background=1.0, combine='paint', shapes=shapes)], **COMMON)

    hole_area = np.pi * 100.0 ** 2
    geometry = 1.0 + 3.0 * (360000.0 - hole_area) / (P * P) + 8.0 * hole_area / (P * P)
    assert solver.structure_slice(n=2048).real.mean() == pytest.approx(geometry, abs=2e-3)
    assert loop_mean_eps(shapes) == pytest.approx(geometry, rel=1e-4)
    assert 2.4256 - geometry > 0.18                    # what ignoring the winding would give

    # The premise, pinned: the two loops of the host region are wound opposite ways.
    recs, regions = _b.plan_layer(shapes, (P, P))
    vals = _c.eval_table(xnp, recs, [_b.shape_vertices(xnp, s, (P, P)) for s in shapes])
    areas = [_c.signed_area(np.asarray(_c.loop_verts(xnp, lp, vals), float))
             for lp in regions[0]]
    assert len(areas) == 2 and areas[0] > 0 > areas[1]


# ----------------------------------------------------------------- geometry against the series

def test_band_limited_converges_on_the_geometry():
    """Rising ``max_order`` closes on the sampled geometry -- which is also the mirror test.

    Relative L2 measured at 28.4% / 16.8% / 10.2% for ``max_order`` 3 / 9 / 25 on an asymmetric
    layer. A flipped order-index sign reconstructs the mirrored structure, whose distance to
    the geometry does not fall with ``max_order`` at all, so monotone convergence is what pins the
    convention here without restating it.
    """
    shapes = [dict(type='circle', cx=300.0, cy=300.0, lx=260.0, material=12.25),
              dict(type='rectangle', cx=740.0, cy=650.0, lx=300.0, ly=180.0,
                   angle=0.45, material=4.840000000000001)]
    cfg = {k: v for k, v in COMMON.items() if k != 'max_order'}
    geometry = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], max_order=(3, 3),
                            **cfg).structure_slice(n=512).real

    errors = []
    for max_order in (3, 9, 25):
        solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], max_order=(max_order, max_order), **cfg)
        band = solver.structure_slice(n=512, band_limited=True).real
        errors.append(np.linalg.norm(band - geometry) / np.linalg.norm(geometry))
        # `<eps>` is converged from the start; the picture is not. That gap is the warning.
        assert band.mean() == pytest.approx(geometry.mean(), abs=1e-3)

    assert errors[0] > errors[1] > errors[2]
    np.testing.assert_allclose(errors, [0.284, 0.168, 0.102], atol=5e-3)


def test_the_truncated_series_leaves_the_material_range():
    """Gibbs, stated as a number: negative permittivity and 13% overshoot at ``max_order = 9``.

    Nothing is wrong with the solve; this is what a truncated transform of a discontinuous
    function looks like, and it is the reason ``band_limited`` is not the default.
    """
    shapes = [dict(type='circle', cx=300.0, cy=300.0, lx=260.0, material=12.25),
              dict(type='rectangle', cx=740.0, cy=650.0, lx=300.0, ly=180.0,
                   angle=0.45, material=4.840000000000001)]
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], **COMMON)
    band = solver.structure_slice(n=512, band_limited=True).real

    assert band.min() == pytest.approx(-0.186, abs=5e-3)
    assert band.max() == pytest.approx(13.85, abs=5e-2)
    assert 1.0 <= solver.structure_slice(n=512).real.min()
    assert solver.structure_slice(n=512).real.max() <= 12.25


def test_the_block_path_draws_the_blocks_not_the_rectangle():
    """The picture is honest about the staircase, which is 1.9% of the rod away from the shape.

    Not a defect in the render -- the blocks *are* what the solver has. It is worth a test
    because the difference shrinks with ``n_split`` and a reader may conclude that is the knob
    for the answer too; it is not (``Modeling.rectangle`` measures the kink against harmonics).
    """
    n = 1024
    cx, cy, lx, ly, angle = 500.0, 500.0, 240.0, 180.0, 0.4
    s = (np.arange(n) + 0.5) * P / n
    X, Y = np.meshgrid(s, s)
    u = np.cos(angle) * (X - cx) + np.sin(angle) * (Y - cy)
    v = -np.sin(angle) * (X - cx) + np.cos(angle) * (Y - cy)
    exact = (np.abs(u) <= lx / 2) & (np.abs(v) <= ly / 2)

    diffs = []
    for n_split in (4, 16, 64):
        instruction = [[1.0, [['rectangle', cx, cy, lx, ly, 12.25, angle, n_split, n_split]]]]
        drawn = structure_slice(wavebit.rcwa(structure=instruction, **COMMON), n=n).real > 1.5
        diffs.append(np.logical_xor(drawn, exact).sum() / exact.sum())

    np.testing.assert_allclose(diffs, [0.077, 0.019, 0.005], atol=3e-3)
    assert diffs[0] > diffs[1] > diffs[2]


def test_an_absorbing_layer_keeps_its_imaginary_part():
    """`structure_slice` returns complex; taking the real part is the *plot*'s decision."""
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=[
        # (3.5 + 0.4j)**2 -- the same absorbing medium, as permittivity.
        dict(type='circle', cx=500.0, cy=500.0, lx=300.0,
             material=12.09 + 2.8000000000000003j)])], **COMMON)
    values = solver.structure_slice(n=128)

    assert np.abs(values.imag).max() == pytest.approx(2.0 * 3.5 * 0.4, rel=1e-12)
    np.testing.assert_allclose(solver.structure_slice(n=128, show='n').max(),
                               3.5 + 0.4j, rtol=1e-12)


def test_show_takes_one_of_two_words():
    solver = wavebit.rcwa(structure=np.ones((1, 8, 8)) * 4.0, **COMMON)   # eps 4 is n 2
    np.testing.assert_allclose(solver.structure_slice(show='eps').real, 4.0)
    np.testing.assert_allclose(solver.structure_slice(show='n').real, 2.0)
    with pytest.raises(ValueError, match="'eps' or 'n'"):
        solver.structure_slice(show='epsilon')


def test_it_draws():
    """The drawing half, once per coordinate system. Presentation, so it only has to run."""
    plt = pytest.importorskip('matplotlib.pyplot')
    plt.switch_backend('Agg')
    shapes = [dict(type='circle', cx=300.0, cy=300.0, lx=260.0, material=12.25)]

    axis_aligned = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], **COMMON)
    ax = axis_aligned.plot_structure(n=64)
    assert ax.images and not ax.collections          # imshow on an axis-aligned cell

    skewed = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)],
                          lattice=[[P, P * 0.5], [0.0, P * 0.866]], **COMMON)
    ax = skewed.plot_structure(n=64)
    assert ax.collections and not ax.images          # pcolormesh on the parallelogram
    plt.close('all')
