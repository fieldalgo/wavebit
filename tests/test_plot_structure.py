"""
Verification tests for `slice_structure` / `plot_structure` -- the picture of the *input*.

What this file pins:

* the analytic sampler agrees with the layer's own DC coefficient, on a skewed cell as well
  as an axis-aligned one -- which is the only thing that catches a wrong periodic-image
  shift, since a shape sitting comfortably inside the cell cannot tell;
* a painted, overlapping layer goes through the resolved loops and not through the summed
  rule, which agrees to the eye and differs by 5.4% in ``<eps>``;
* the band-limited reconstruction *converges* on the geometry as ``truncation_order`` rises. That is the
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
from wavebit._core.parametric import assemble_eps_coeffs
from wavebit._core.lattice import as_lattice

P = 1000.0
#: The axis-aligned cell as a *basis*. The boundary helpers below take one, not a period pair:
#: a shape's `cx` / `cy` are measured from `0.5*(a1 + a2)`, which a pair of lengths cannot say.
CELL = as_lattice(period=(P, P))
COMMON = dict(backend=0, period=(P, P), thickness=(300.,), wavelength=900.,
              top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0, truncation_order=(9, 9))

#: `COMMON` without its cell, for the two calls that give a basis instead: naming `period` and
#: `lattice` together is refused, they are two spellings of one cell.
COMMON_NO_CELL = {k: v for k, v in COMMON.items() if k != 'period'}

#: A 200x160 rod of permittivity 12.25 in a 1000x1000 cell of permittivity 1: `<eps>` is
#: `1 + 0.032*(12.25 - 1) = 1.36` wherever it is placed, since a translation is not a
#: different structure. Every straddle case below is that same rod.
ROD = dict(lx=200.0, ly=160.0, eps=12.25)
# ------------------------------------------------------- the straddle, which is the whole point

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

    got = solver.slice_structure(layer=0, n=999)
    assert got.shape == (N, N)
    np.testing.assert_array_equal(got.real, cell[0])
    assert got[0, 0].real == 9.0 and got[-1, 0].real == 1.0

    assert solver.slice_structure(layer=1).real.mean() == pytest.approx(4.0)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_layer_is_indexed_the_way_the_structure_is(side):
    """0 is the layer nearest ``top``, on every path -- and out of range says so.

    Parametrised over ``incidence_side`` because that is the whole claim: the scene labels do
    not move when the illumination does (``docs/conventions.md``). Lit from below the light
    meets ``structure[-1]`` first, and ``layer=0`` still has to mean the top one -- "the layer
    light meets first" was the wording here until 2026-08-22, and it is only true from above.
    """
    shapes = [dict(type='circle', cx=0.0, cy=0.0, lx=200.0, material=9.0)]
    layers = [dict(background=1.0, shapes=[]), dict(background=1.0, shapes=shapes)]
    solver = wavebit.rcwa(structure=layers, incidence_side=side, **COMMON)

    assert solver.slice_structure(layer=0, n=128).real.mean() == pytest.approx(1.0)
    assert solver.slice_structure(layer=1, n=128).real.mean() > 1.2
    assert solver.slice_structure(layer=-1, n=128).real.mean() > 1.2
    with pytest.raises(IndexError, match='nearest'):
        solver.slice_structure(layer=2)


@pytest.mark.parametrize('lattice', [None, [[P, P * 0.5], [0.0, P * 0.866]]])
def test_the_analytic_sampler_agrees_with_its_own_dc_coefficient(lattice):
    """Sampled area against the closed-form transform's DC term -- including a skewed cell.

    The skewed case is the one that carries weight. A periodic image is ``r + m*a1 + n*a2``,
    not ``r + (m*Px, n*Py)``, and a shape sitting inside the cell cannot tell the two apart;
    this layer has a rectangle 150 above the cell centre with a 300 length across a sheared
    cell, so it does.
    """
    shapes = [dict(type='circle', cx=-200.0, cy=-200.0, lx=260.0, material=12.25),
              dict(type='rectangle', cx=240.0, cy=150.0, lx=300.0, ly=180.0,
                   angle=0.45, material=4.840000000000001)]
    # this case is parametrized over `lattice=None` too, so the cell is named whichever way it
    # was given -- naming both is refused.
    cell = {'lattice': lattice} if lattice is not None else {'period': COMMON['period']}
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], **cell,
                          **COMMON_NO_CELL)
    fx, fy = int(solver.truncation_order[0]), int(solver.truncation_order[1])
    dc = assemble_eps_coeffs(xnp, shapes, 1.0, fx, fy, solver.lattice,
                             xnp.complex128)[0][2 * fy, 2 * fx].real

    assert solver.slice_structure(n=1024).real.mean() == pytest.approx(dc, rel=1e-4)


def test_a_circle_is_read_through_shape_geometry():
    """A circle names its diameter once. Reading the dict directly is how that stops being true.

    ``ly`` is absent here, and ``shape_geometry`` is the one place that knows it falls back to
    ``lx`` for a circle rather than to 0 or to 500. A second reader gets that wrong quietly --
    which is the defect that function was written to end.
    """
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=[
        dict(type='circle', cx=0.0, cy=0.0, lx=400.0, material=9.0)])], **COMMON)
    area = np.pi * 200.0 ** 2 / (P * P)
    assert solver.slice_structure(n=1024).real.mean() == pytest.approx(
        1.0 + area * (9.0 - 1.0), rel=1e-3)


# ------------------------------------------------------------------- painted, overlapping

OVERLAP = [dict(type='rectangle', cx=-70.0, cy=0.0, lx=300.0, ly=300.0, material=4.0),
           dict(type='circle', cx=100.0, cy=20.0, lx=320.0, material=9.0)]


def loop_mean_eps(shapes, background=1.0):
    """`<eps>` straight off the boundary route's resolved loops -- areas, no sampling at all."""
    recs, regions = _b.plan_layer(shapes, CELL)
    verts = [_b.shape_vertices(xnp, s, CELL) for s in shapes]
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
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=OVERLAP)], **COMMON)
    want = loop_mean_eps(OVERLAP)

    assert solver.slice_structure(n=64, band_limited=True).real.mean() == pytest.approx(
        want, abs=1e-13)
    assert solver.slice_structure(n=2048).real.mean() == pytest.approx(want, abs=2e-4)


def test_an_overlap_is_drawn_painted_and_not_summed():
    """The plausible wrong picture, 5.4% away in ``<eps>``, pinned by both numbers.

    Falling back to the summed sampler is the mistake this catches: the overlap would be drawn
    at ``eps_a + eps_b - eps_base`` instead of ``eps_b``, which looks like a structure someone
    might have meant. It was reachable by name until 2026-08-27 (``combine='add'``); the sum is
    now what the closed-form builder computes for shapes that do not overlap, and this picture
    is the one the solver builds.

    The summed number stays as a literal because it is the failure's signature, not a second
    structure: a renderer that lost the painting lands on 1.9134 exactly.
    """
    paint = wavebit.rcwa(structure=[dict(background=1.0, shapes=OVERLAP)], **COMMON)

    e_paint = paint.slice_structure(n=2048).real.mean()
    e_summed = 1.9134                                  # eps_a + eps_b - eps_base in the overlap
    assert e_paint == pytest.approx(1.8158, abs=1e-3)
    assert (e_summed - e_paint) / e_paint == pytest.approx(0.054, abs=2e-3)


def test_a_hole_is_drawn_as_a_hole():
    """A shape painted *inside* another leaves a two-loop region, and the render cuts it out.

    The orientation carries it: the hole comes back **clockwise**, so its winding number is
    ``-1`` and the scanline fill subtracts it with no sign of its own -- the same way
    ``polygon_ft`` does. A renderer that took the absolute area instead would put ``<eps>`` at
    2.4256 where the geometry is 2.2371, which is what the solver itself did until the
    orientation stopped being carried in a separate field.
    """
    shapes = [dict(type='rectangle', cx=0.0, cy=0.0, lx=600.0, ly=600.0, material=4.0),
              dict(type='circle', cx=0.0, cy=0.0, lx=200.0, material=9.0)]
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], **COMMON)

    hole_area = np.pi * 100.0 ** 2
    geometry = 1.0 + 3.0 * (360000.0 - hole_area) / (P * P) + 8.0 * hole_area / (P * P)
    assert solver.slice_structure(n=2048).real.mean() == pytest.approx(geometry, abs=2e-3)
    assert loop_mean_eps(shapes) == pytest.approx(geometry, rel=1e-4)
    assert 2.4256 - geometry > 0.18                    # what ignoring the winding would give

    # The premise, pinned: the two loops of the host region are wound opposite ways.
    recs, regions = _b.plan_layer(shapes, CELL)
    vals = _c.eval_table(xnp, recs, [_b.shape_vertices(xnp, s, CELL) for s in shapes])
    areas = [_c.signed_area(np.asarray(_c.loop_verts(xnp, lp, vals), float))
             for lp in regions[0]]
    assert len(areas) == 2 and areas[0] > 0 > areas[1]


# ----------------------------------------------------------------- geometry against the series

def test_band_limited_converges_on_the_geometry():
    """Rising ``truncation_order`` closes on the sampled geometry -- which is also the mirror test.

    Relative L2 measured at 28.4% / 16.8% / 10.2% for ``truncation_order`` 3 / 9 / 25 on an asymmetric
    layer. A flipped order-index sign reconstructs the mirrored structure, whose distance to
    the geometry does not fall with ``truncation_order`` at all, so monotone convergence is what pins the
    convention here without restating it.
    """
    shapes = [dict(type='circle', cx=-200.0, cy=-200.0, lx=260.0, material=12.25),
              dict(type='rectangle', cx=240.0, cy=150.0, lx=300.0, ly=180.0,
                   angle=0.45, material=4.840000000000001)]
    cfg = {k: v for k, v in COMMON.items() if k != 'truncation_order'}
    geometry = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], truncation_order=(3, 3),
                            **cfg).slice_structure(n=512).real

    errors = []
    for truncation_order in (3, 9, 25):
        solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], truncation_order=(truncation_order, truncation_order), **cfg)
        band = solver.slice_structure(n=512, band_limited=True).real
        errors.append(np.linalg.norm(band - geometry) / np.linalg.norm(geometry))
        # `<eps>` is converged from the start; the picture is not. That gap is the warning.
        assert band.mean() == pytest.approx(geometry.mean(), abs=1e-3)

    assert errors[0] > errors[1] > errors[2]
    np.testing.assert_allclose(errors, [0.284, 0.168, 0.102], atol=5e-3)


def test_the_truncated_series_leaves_the_material_range():
    """Gibbs, stated as a number: negative permittivity and 13% overshoot at ``truncation_order = 9``.

    Nothing is wrong with the solve; this is what a truncated transform of a discontinuous
    function looks like, and it is the reason ``band_limited`` is not the default.
    """
    shapes = [dict(type='circle', cx=-200.0, cy=-200.0, lx=260.0, material=12.25),
              dict(type='rectangle', cx=240.0, cy=150.0, lx=300.0, ly=180.0,
                   angle=0.45, material=4.840000000000001)]
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], **COMMON)
    band = solver.slice_structure(n=512, band_limited=True).real

    assert band.min() == pytest.approx(-0.186, abs=5e-3)
    assert band.max() == pytest.approx(13.85, abs=5e-2)
    assert 1.0 <= solver.slice_structure(n=512).real.min()
    assert solver.slice_structure(n=512).real.max() <= 12.25


def test_an_absorbing_layer_keeps_its_imaginary_part():
    """`slice_structure` returns complex; taking the real part is the *plot*'s decision."""
    solver = wavebit.rcwa(structure=[dict(background=1.0, shapes=[
        # (3.5 + 0.4j)**2 -- the same absorbing medium, as permittivity.
        dict(type='circle', cx=0.0, cy=0.0, lx=300.0,
             material=12.09 + 2.8000000000000003j)])], **COMMON)
    values = solver.slice_structure(n=128)

    assert np.abs(values.imag).max() == pytest.approx(2.0 * 3.5 * 0.4, rel=1e-12)
    np.testing.assert_allclose(solver.slice_structure(n=128, show='n').max(),
                               3.5 + 0.4j, rtol=1e-12)


def test_show_takes_one_of_two_words():
    solver = wavebit.rcwa(structure=np.ones((1, 8, 8)) * 4.0, **COMMON)   # eps 4 is n 2
    np.testing.assert_allclose(solver.slice_structure(show='eps').real, 4.0)
    np.testing.assert_allclose(solver.slice_structure(show='n').real, 2.0)
    with pytest.raises(ValueError, match="'eps' or 'n'"):
        solver.slice_structure(show='epsilon')


def test_it_draws():
    """The drawing half, once per coordinate system. Presentation, so it only has to run."""
    plt = pytest.importorskip('matplotlib.pyplot')
    plt.switch_backend('Agg')
    shapes = [dict(type='circle', cx=-200.0, cy=-200.0, lx=260.0, material=12.25)]

    axis_aligned = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)], **COMMON)
    ax = axis_aligned.plot_structure(n=64)
    assert ax.images and not ax.collections          # imshow on an axis-aligned cell

    skewed = wavebit.rcwa(structure=[dict(background=1.0, shapes=shapes)],
                          lattice=[[P, P * 0.5], [0.0, P * 0.866]], **COMMON_NO_CELL)
    ax = skewed.plot_structure(n=64)
    assert ax.collections and not ax.images          # pcolormesh on the parallelogram
    plt.close('all')


def test_a_traced_structure_is_refused_by_name_rather_than_by_numpy():
    """A picture needs values, and saying so is the renderer's job.

    ``slice_structure`` samples the geometry on a NumPy grid, so a traced shape parameter has
    nowhere to go. It used to be covered by accident: ``layer_needs_resolving`` raised for any
    traced multi-shape layer and the renderer never got that far. Since 2026-08-28 that routing
    question answers under ``jax.grad``, and without this the renderer fell out of
    ``numpy.stack`` with a ``TracerArrayConversionError`` naming nothing the caller wrote.

    The predicate is an *array* conversion, not ``checked_float``: reading one scalar at a time
    reaches a grad tracer's primal -- which is what lets the geometry checks run under a
    gradient -- while a whole-array conversion is refused under any JAX trace. A torch leaf
    under autograd detaches and still draws, and is asserted here so the two do not drift.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def layer(lx):
        return [dict(background=1.0, shapes=[
            dict(type='rectangle', cx=-250.0, cy=0.0, lx=lx, ly=200.0, angle=0.0,
                 material=4.0),
            dict(type='rectangle', cx=250.0, cy=0.0, lx=200.0, ly=200.0, angle=0.0,
                 material=9.0)])]

    def drawn(lx):
        return jnp.sum(jnp.asarray(wavebit.rcwa(
            structure=layer(lx), **dict(COMMON, backend=1)).slice_structure(n=32)).real)

    with pytest.raises(ValueError, match='traced structure cannot be drawn'):
        jax.grad(drawn)(jnp.asarray(200.0))

    # eager still draws, on the same solver type
    assert float(drawn(jnp.asarray(200.0))) > 0.0
