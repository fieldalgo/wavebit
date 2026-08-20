"""The ``normal_method`` option — plumbing, and the properties each construction must keep.

Three constructions were added next to the original ``'distance'`` field because its
directions quantize: central differences of a distance-to-a-digital-set land on small
integers, so the boundary band emits ~{0, 45} degrees with a mean error of ~14 degrees that
does not shrink with N (``docs/nvf-upsampling.md`` has the measurements; ``'contour'`` is
the one that lost no resolved solver cell on either measured scene).

What this file pins, and why each pin is the one that matters:

- **The untouched default is bit-identical to the explicit spelling** (it was also bit-identical
  to the code before the option existed, until `339df21` made the `'distance'` construction
  periodic; that is a change to what the default selects, not to the selection) — the
  option must be opt-in, not a quiet renumbering of every NVF user's results.
- **An explicit request nothing would read is refused, not ignored** — the shape paths
  build normals from geometry and Li builds none, so a ``normal_method`` there would be the
  ``enhanced_dfs`` pattern again: a knob without a reader.
- **The frozen-geometry record carries the method** — same raster, same mask, different
  construction is a *different field*, and a replay across methods must be refused by the
  staleness tell, not silently answered with the wrong field.
- **Angular accuracy against an exact normal, on a y-asymmetric case** — every external
  oracle this suite ever ran was blind to a y-flip (see the raster row-order history), so
  the accuracy pin uses an off-center circle, where a transposed or flipped field cannot
  score well.
- **The floor lift keeps the base grid's smoothing scale** — a lifted raster is the same
  staircase spelled with more pixels, so a construction that read its pixel policies off
  the lifted shape would smooth 3x too narrowly exactly when the floor kicks in.
"""
from importlib.util import find_spec

import numpy as np
import pytest

import wavebit
from wavebit._core.enums import NormalMethod
from wavebit._core.factorization import (StaleGeometryError, _interface_mask,
                                           normal_vector_field)

#: 'contour' is the one construction with an optional dependency; a minimal install must
#: skip its cases, not fail them (the package itself declares contourpy optional).
needs_contourpy = pytest.mark.skipif(
    find_spec('contourpy') is None,
    reason="normal_method='contour' needs the optional contourpy dependency")

def _param(name):
    return pytest.param(name, marks=needs_contourpy) if name == 'contour' else name

METHODS = [_param(m.value) for m in NormalMethod]
NEW_METHODS = [_param(m.value) for m in NormalMethod if m.value != 'distance']


# ---------------------------------------------------------------- rasters and helpers

def circle_raster(N, cx=0.5, cy=0.5, r=0.3, eps_in=5.76):
    """A centred disc as a PERMITTIVITY raster; 5.76 is the index 2.4 it used to hold."""
    ax = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(ax, ax)
    eps = np.ones((N, N))
    eps[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = eps_in
    return eps


def angular_error_deg(nx, ny, tx, ty):
    """Pointwise angle between two unit fields, modulo 180 (n and -n are one field)."""
    dot = np.abs(nx * tx + ny * ty)
    return np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))


def band_error(eps, nx, ny, cx=0.5, cy=0.5, r=0.3, width_px=2):
    """Mean angular error vs the exact radial normal, within ``width_px`` of the circle."""
    N = eps.shape[0]
    ax = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(ax, ax)
    dx, dy = X - cx, Y - cy
    rr = np.hypot(dx, dy)
    band = np.abs(rr - r) * N <= width_px
    tx, ty = dx / np.where(rr == 0, 1, rr), dy / np.where(rr == 0, 1, rr)
    return float(angular_error_deg(nx, ny, tx, ty)[band].mean())


SOLVE_KW = dict(backend=0, period=[1., 1.], wavelength=.55, thickness=[.3],
                theta=np.deg2rad(12), phi=np.deg2rad(30), psi=np.deg2rad(40),
                top=1., bot=1.)


# ------------------------------------------------------------------------ plumbing

def test_explicit_distance_is_bit_identical_to_the_untouched_default():
    """The option must be opt-in: naming the default changes nothing, to the bit."""
    structure = circle_raster(32)[None]
    base = wavebit.rcwa(structure=structure, fto=[4, 4], factorization='nvf', **SOLVE_KW).conv_solve()
    named = wavebit.rcwa(structure=structure, fto=[4, 4], factorization='nvf',
                         normal_method='distance', **SOLVE_KW).conv_solve()
    assert np.array_equal(np.asarray(base.R), np.asarray(named.R))
    assert np.array_equal(np.asarray(base.T), np.asarray(named.T))


def test_an_invalid_method_is_refused_by_name():
    with pytest.raises(ValueError, match='normal_method'):
        wavebit.rcwa(structure=circle_raster(16)[None], fto=[2, 2],
                     factorization='nvf', normal_method='sobel', **SOLVE_KW)


@pytest.mark.parametrize('cfg', [
    dict(factorization='li'),   # explicit Li: no normal field at all
    dict(),                     # untouched default resolves to Li on a raster
])
def test_a_method_nothing_reads_is_refused_not_ignored(cfg):
    s = wavebit.rcwa(structure=circle_raster(16)[None], fto=[2, 2],
                     normal_method='contour', **SOLVE_KW, **cfg)
    with pytest.raises(NotImplementedError, match='no reader'):
        s.conv_solve()


def test_a_method_on_the_smooth_shape_path_is_refused():
    """The shape paths build their normal fields from the geometry itself."""
    smooth = [{'background': 1.0, 'shapes': [{'type': 'circle', 'cx': 0.5, 'cy': 0.5,
                                              'lx': 0.6, 'material': 5.76}]}]
    s = wavebit.rcwa(structure=smooth, fto=[2, 2], normal_method='contour', **SOLVE_KW)
    with pytest.raises(NotImplementedError, match='no reader'):
        s.conv_solve()


@pytest.mark.parametrize('method', NEW_METHODS)
def test_freeze_replay_is_bit_identical_and_method_mismatch_is_stale(method):
    """The record carries the construction; replaying it under another is refused."""
    structure = circle_raster(32)[None]
    s = wavebit.rcwa(structure=structure, fto=[3, 3], factorization='nvf',
                     normal_method=method, **SOLVE_KW)
    geo = s.freeze_geometry()
    replayed = s.conv_solve(geometry=geo)
    rebuilt = wavebit.rcwa(structure=structure, fto=[3, 3], factorization='nvf',
                           normal_method=method, **SOLVE_KW).conv_solve()
    assert np.array_equal(np.asarray(replayed.R), np.asarray(rebuilt.R))

    other = wavebit.rcwa(structure=structure, fto=[3, 3], factorization='nvf', **SOLVE_KW)
    with pytest.raises(StaleGeometryError, match='different construction'):
        other.conv_solve(geometry=geo)


@pytest.mark.parametrize('method', NEW_METHODS)
def test_energy_is_conserved_by_every_construction(method):
    res = wavebit.rcwa(structure=circle_raster(48)[None], fto=[5, 5], factorization='nvf',
                       normal_method=method, **SOLVE_KW).conv_solve()
    total = float(np.sum(res.R) + np.sum(res.T))
    assert abs(total - 1.0) < 1e-10


# --------------------------------------------------------------- field invariants

@pytest.mark.parametrize('method', NEW_METHODS)
def test_unit_magnitude_everywhere_with_zeros_only_where_deliberate(method):
    """The factorization needs C_xx + C_yy = I, which is |n| = 1 across the cell."""
    eps = circle_raster(64)
    nx, ny = normal_vector_field(eps, method=method)
    mag = np.hypot(nx, ny)
    nonzero = mag > 0
    assert np.abs(mag[nonzero] - 1.0).max() < 1e-12
    # contour and spectral emit no zeros at all; smooth may only on the skeleton,
    # which for this circle is empty (no corners for the cancellation to be exact at).
    assert nonzero.all()


@pytest.mark.parametrize('method', NEW_METHODS)
def test_a_flat_interface_gets_the_axis_normal_exactly(method):
    """Nothing here is a bias in where the boundary sits: a strip's normal is the axis."""
    eps = np.ones((32, 32))
    eps[:, 10:20] = 2.4            # vertical strip: interfaces normal to x
    nx, ny = normal_vector_field(eps, method=method)
    assert float(np.abs(ny).max()) == 0.0


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('shape', [(1, 8), (8, 1)])
def test_a_one_cell_axis_keeps_the_constant_normal_rule(method, shape):
    """The 1D collapse onto Li must not depend on the construction."""
    eps = np.linspace(1.0, 2.0, 8).reshape(shape)
    nx, ny = normal_vector_field(eps, method=method)
    expect_x = 1.0 if shape[0] == 1 else 0.0
    assert np.array_equal(nx, np.full(shape, expect_x))
    assert np.array_equal(ny, np.full(shape, 1.0 - expect_x))


@pytest.mark.parametrize('method', NEW_METHODS)
def test_a_uniform_layer_gets_a_constant_field(method):
    nx, ny = normal_vector_field(np.full((16, 16), 1.7), method=method)
    assert np.array_equal(nx, np.ones((16, 16)))
    assert np.array_equal(ny, np.zeros((16, 16)))


@pytest.mark.parametrize('flip', ['y', 'x'])
def test_smooth_is_bit_exactly_covariant_under_both_flips(flip):
    """'smooth' mirrors exactly, like 'distance' — in BOTH directions. The x direction is
    the one every y-symmetric golden was historically blind to, so it is pinned by name."""
    eps = circle_raster(48, cx=0.4, cy=0.55)           # asymmetric on purpose
    nx, ny = normal_vector_field(eps, method='smooth')
    if flip == 'y':
        fx, fy = normal_vector_field(eps[::-1].copy(), method='smooth')
        assert np.array_equal((nx ** 2)[::-1], fx ** 2)
        assert np.array_equal((nx * ny)[::-1], -(fx * fy))
    else:
        fx, fy = normal_vector_field(eps[:, ::-1].copy(), method='smooth')
        assert np.array_equal((nx ** 2)[:, ::-1], fx ** 2)
        assert np.array_equal((nx * ny)[:, ::-1], -(fx * fy))


@pytest.mark.parametrize('method,limit', [
    ('distance', 1e-12), pytest.param('contour', 5e-4, marks=needs_contourpy),
    ('spectral', 5e-4), ('smooth', 1e-12)])
def test_mirror_asymmetry_of_the_solve_is_bounded(method, limit):
    """The user-visible covariance fact, pinned where it is felt: solve a structure and
    its x-mirror and compare mirrored efficiencies. 'distance' and 'smooth' agree to
    machine precision; 'contour' and 'spectral' carry tie-break and resampling-phase
    asymmetries worth ~1.5e-4 here — an order below their accuracy gains at the same
    settings, but real, which is why the bound is stated rather than pretended away."""
    eps = circle_raster(48, cx=0.4, cy=0.55)
    cfg = dict(backend=0, period=[1., 1.], wavelength=.55, thickness=[.3], theta=0.,
               phi=0., psi=0., top=1., bot=1., fto=[5, 5], factorization='nvf')
    r1 = wavebit.rcwa(structure=eps[None], normal_method=method, **cfg).conv_solve()
    r2 = wavebit.rcwa(structure=eps[:, ::-1].copy()[None], normal_method=method,
                      **cfg).conv_solve()
    gap = float(np.abs(np.asarray(r1.R) - np.asarray(r2.R)[:, ::-1]).max())
    assert gap < limit


@pytest.mark.parametrize('method', NEW_METHODS)
def test_the_field_ignores_which_value_is_which(method):
    """Swap the two material values (identical geometry, identical mask): the field is a
    function of the REGIONS, not the values — the corner-canonical indicator's job, and
    what lets the frozen-record staleness tell stay honest."""
    A = circle_raster(48, cx=0.4, cy=0.55, eps_in=4.0)
    B = np.where(A == 4.0, 1.0, 4.0)                   # same partition, values exchanged
    na = normal_vector_field(A, method=method)
    nb = normal_vector_field(B, method=method)
    dot = np.abs(na[0] * nb[0] + na[1] * nb[1])
    assert float(np.degrees(np.arccos(np.clip(dot, 0, 1))).max()) < 1e-4


@pytest.mark.parametrize('method,limit_deg', [('contour', 3.0), ('spectral', 5.0),
                                              ('smooth', 2.0)])
def test_angular_accuracy_beats_distance_on_an_off_center_circle(method, limit_deg):
    """The point of the option, pinned where a flip or transpose cannot pass.

    The shipped 'distance' field measures ~14 deg in this band (flat in N); each
    construction must land under its measured bound, on a circle deliberately off
    center — the case every y-symmetric golden was blind to.
    """
    cx, cy = 0.42, 0.61
    eps = circle_raster(64, cx=cx, cy=cy, r=0.22)
    nx, ny = normal_vector_field(eps, method=method)
    err = band_error(eps, nx, ny, cx=cx, cy=cy, r=0.22)
    assert err < limit_deg
    dx_, dy_ = normal_vector_field(eps, method='distance')
    assert err < band_error(eps, dx_, dy_, cx=cx, cy=cy, r=0.22) / 3.0


@pytest.mark.parametrize('method', NEW_METHODS)
def test_a_wrap_crossing_shape_is_periodic(method):
    """A circle at the cell corner is four arcs through both edges; the periodic
    constructions must score like the centered one ('distance' is one-sided at the
    edge and cannot take this test)."""
    N = 64
    ax = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(ax, ax)
    # The periodic circle: distance to the nearest periodic copy of the corner center —
    # building it non-periodically would draw a quarter disc, a different shape.
    dx = (X + 0.5) % 1.0 - 0.5
    dy = (Y + 0.5) % 1.0 - 0.5
    rr = np.hypot(dx, dy)
    eps = np.ones((N, N))
    eps[rr <= 0.3] = 2.4
    nx, ny = normal_vector_field(eps, method=method)
    band = np.abs(rr - 0.3) * N <= 2
    tx, ty = dx / np.where(rr == 0, 1, rr), dy / np.where(rr == 0, 1, rr)
    err = float(angular_error_deg(nx, ny, tx, ty)[band].mean())
    assert err < 6.0


@pytest.mark.parametrize('method', NEW_METHODS)
def test_a_non_square_cell_measures_in_physical_space(method):
    """A circle in a 2:1 cell rasterized on a square grid has non-square pixels; a
    construction that stayed in index space would see an ellipse and tilt every
    normal (the period-pair trap). The exact radial normal is the referee."""
    from wavebit._core.lattice import as_lattice
    N = 64
    ax = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(ax * 2.0, ax)          # physical x in [0, 2), y in [0, 1)
    eps = np.ones((N, N))
    cx, cy, r = 1.0, 0.5, 0.3
    eps[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = 2.4
    lattice = as_lattice(period=[2.0, 1.0])
    nx, ny = normal_vector_field(eps, method=method, lattice=lattice)
    dx, dy = X - cx, Y - cy
    rr = np.hypot(dx, dy)
    band = (np.abs(rr - r) / (2.0 / N)) <= 2      # within ~2 of the coarser pixel
    tx, ty = dx / np.where(rr == 0, 1, rr), dy / np.where(rr == 0, 1, rr)
    err = float(angular_error_deg(nx, ny, tx, ty)[band].mean())
    assert err < 6.0


@pytest.mark.parametrize('method', NEW_METHODS)
def test_a_skewed_basis_is_refused(method):
    lattice = np.array([[1.0, 0.3], [0.0, 1.0]])
    with pytest.raises(NotImplementedError, match='axis-aligned'):
        normal_vector_field(circle_raster(16), method=method, lattice=lattice)


# ------------------------------------------------------------------- the floor lift

@pytest.mark.parametrize('method,base_n', [('contour', 16), ('smooth', 8)])
def test_the_floor_lift_keeps_the_base_grids_smoothing_scale(method, base_n):
    """A small layer at fto=10 block-repeats to reach the 41-point floor. The pixel-unit
    policies must be read off the base grid and carried up — the same construction run
    on the lifted raster *as if it were the base* smooths too narrowly and produces a
    measurably different field. Equality with the base-aware call plus inequality with
    the base-blind one pins the threading, not just the plumbing.

    The base sizes are chosen where base-aware and base-blind genuinely differ:
    contour's sigma is nonlinear in N (0.25*sqrt(N)), so any lifted base differs;
    smooth's is linear (N/16), where base-aware and base-blind coincide *except* below
    the one-pixel floor — base 8 puts it there (1 base px = 6 lifted px against the
    blind reading's 3)."""
    from wavebit._core.factorization import _normal_field_floor, normal_field_for_raster
    base = circle_raster(base_n)
    repeat = _normal_field_floor(base.shape, 10, 10)[0]
    lifted = np.repeat(np.repeat(base, repeat, axis=0), repeat, axis=1)

    [(nx, ny)] = normal_field_for_raster(base[None], 10, 10, method=method)
    ex, ey = normal_vector_field(lifted, method=method, base_shape=(base_n, base_n))
    assert np.array_equal(nx, ex) and np.array_equal(ny, ey)

    bx, by = normal_vector_field(lifted, method=method)     # base-blind on purpose
    assert not np.allclose(nx, bx)


@pytest.mark.parametrize('method', NEW_METHODS)
def test_the_lifted_solve_runs_and_conserves_energy(method):
    res = wavebit.rcwa(structure=circle_raster(16)[None], fto=[10, 10],
                       factorization='nvf', normal_method=method, **SOLVE_KW).conv_solve()
    assert abs(float(np.sum(res.R) + np.sum(res.T)) - 1.0) < 1e-10


# ----------------------------------------------------------- multi-phase and tracers

@pytest.mark.parametrize('method', ['contour', 'smooth'])
def test_three_phases_delegate_to_distance(method):
    """The two-phase indicator cannot say which of three materials is inside; the honest
    fallback is the construction whose mask sees every interface."""
    eps = np.ones((32, 32))
    eps[4:14, 4:14] = 4.0
    eps[18:28, 18:28] = 9.0
    got = normal_vector_field(eps, method=method)
    want = normal_vector_field(eps, method='distance')
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_a_tracer_is_told_to_freeze_instead_of_failing_inside_scipy():
    jax = pytest.importorskip('jax')
    structure = circle_raster(24)[None]

    def loss(u):
        res = wavebit.rcwa(backend=1, structure=u, fto=[2, 2], factorization='nvf',
                           normal_method='contour', **{k: v for k, v in SOLVE_KW.items()
                                                       if k != 'backend'}).conv_solve()
        return jax.numpy.sum(res.R)

    with pytest.raises(TypeError, match='freeze_geometry'):
        jax.grad(loss)(jax.numpy.asarray(structure))


# ------------------------------------------------- the record's value-derived inputs

def test_merging_two_non_adjacent_phases_is_stale_not_silently_replayed():
    """The one value-derived input the mask cannot see: the phase COUNT. Merging two
    phases that never touch moves no interface — the mask is bit-identical — yet it
    changes which construction runs (3+ phases delegate to 'distance', 2 do not), so the
    frozen record carries the count and the replay refuses across the merge."""
    eps = np.ones((32, 32))
    eps[4:14, 4:14] = 2.0
    eps[20:30, 20:30] = 3.0            # never adjacent to the 2.0 rod
    merged = np.where(eps == 3.0, 2.0, eps)
    assert np.array_equal(_interface_mask(eps), _interface_mask(merged))

    s = wavebit.rcwa(structure=eps[None], fto=[3, 3], factorization='nvf',
                     normal_method='spectral', **SOLVE_KW)
    geo = s.freeze_geometry()
    other = wavebit.rcwa(structure=merged[None], fto=[3, 3], factorization='nvf',
                         normal_method='spectral', **SOLVE_KW)
    with pytest.raises(StaleGeometryError, match='distinct phases'):
        other.conv_solve(geometry=geo)


# ------------------------------------------------------------- the public surfaces

def test_the_enum_is_exported_at_the_top_level():
    """Every other option enum is; the fourth axis must not need a private import."""
    assert wavebit.NormalMethod.CONTOUR.value == 'contour'


def test_the_jobspec_surface_can_spell_the_option():
    """The declarative contract the manifest advertises must round-trip the option, and
    an unset field must stay unset (omitted from kwargs) so the spec never asks for
    anything the caller did not."""
    from wavebit._core.spec import Numerics
    n = Numerics.from_dict({'normal_method': 'contour', 'factorization': 'nvf'})
    assert n.normal_method == NormalMethod.CONTOUR
    assert n.to_dict()['normal_method'] == 'contour'
    assert Numerics.from_dict(n.to_dict()).normal_method == NormalMethod.CONTOUR
    assert n.as_kwargs()['normal_method'] == 'contour'
    assert 'normal_method' not in Numerics.from_dict({}).as_kwargs()


def test_spectral_bulk_copy_is_periodic_at_the_wrap():
    """Bulk pixels of a wrap-crossing shape must copy their PERIODICALLY nearest band
    donor. The unwrapped transform handed wrapped-edge pixels a farther in-cell donor
    (up to ~72 deg off the radial truth); the tiled copy keeps the shell sane."""
    N = 64
    ax = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(ax, ax)
    dx = (X + 0.48) % 1.0 - 0.5
    dy = Y - 0.5
    rr = np.hypot(dx, dy)
    eps = np.ones((N, N))
    eps[rr <= 0.3] = 2.4
    nx, ny = normal_vector_field(eps, method='spectral')
    tx, ty = dx / np.where(rr == 0, 1, rr), dy / np.where(rr == 0, 1, rr)
    err = angular_error_deg(nx, ny, tx, ty)
    shell = (np.abs(rr - 0.3) * N > 3) & (np.abs(rr - 0.3) * N <= 8)
    assert float(err[shell].max()) < 25.0
