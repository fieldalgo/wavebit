"""The ``raster_normal_method`` option — plumbing, and the properties each construction must keep.

Three constructions were added next to the original ``'distance'`` field because its
directions quantize: central differences of a distance-to-a-digital-set land on small
integers, so the boundary band emits ~{0, 45} degrees with a mean error of ~14 degrees that
does not shrink with N (``docs/validation/rcwa/nvf-upsampling.md`` has the measurements; ``'contour'`` is
the one that lost no resolved solver cell on either measured scene).

What this file pins, and why each pin is the one that matters:

- **The untouched default is bit-identical to the explicit spelling** (it was also bit-identical
  to the code before the option existed, until `339df21` made the `'distance'` construction
  periodic; that is a change to what the default selects, not to the selection) — the
  option must be opt-in, not a quiet renumbering of every NVF user's results.
- **An explicit request nothing would read is refused, not ignored** — the shape paths
  build normals from geometry and Li builds none, so a ``raster_normal_method`` there would be the
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
- **The cell metric, oblique cells included** — accuracy against a skewed disc's exact
  normal, and, because an accuracy bound cannot say a metric is *right*, the covariance
  that can: a rotated basis is one structure seen from a turned frame, so the field must
  turn with it and nothing else may move. Until 2026-08-21 this file pinned the opposite,
  that a skewed basis raises.
"""

from wavebit.rcwa._engine.enums import NormalMethod as _legacy_wavebit_NormalMethod

from wavebit.lattice import meshgrid_xy as _legacy_wavebit_meshgrid_xy

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
from importlib.util import find_spec

import numpy as np
import pytest

from wavebit.rcwa._engine.enums import NormalMethod
from wavebit.rcwa._engine.factorization import StaleGeometryError, _interface_mask, normal_vector_field

#: 'contour' is the one construction with an optional dependency; a minimal install must
#: skip its cases, not fail them (the package itself declares contourpy optional).
needs_contourpy = pytest.mark.skipif(
    find_spec('contourpy') is None,
    reason="raster_normal_method='contour' needs the optional contourpy dependency")

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


def angular_error_deg(normal_x, normal_y, tx, ty):
    """Pointwise angle between two unit fields, modulo 180 (n and -n are one field)."""
    dot = np.abs(normal_x * tx + normal_y * ty)
    return np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))


def band_error(eps, normal_x, normal_y, cx=0.5, cy=0.5, r=0.3, width_px=2):
    """Mean angular error vs the exact radial normal, within ``width_px`` of the circle."""
    N = eps.shape[0]
    ax = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(ax, ax)
    dx, dy = X - cx, Y - cy
    rr = np.hypot(dx, dy)
    band = np.abs(rr - r) * N <= width_px
    tx, ty = dx / np.where(rr == 0, 1, rr), dy / np.where(rr == 0, 1, rr)
    return float(angular_error_deg(normal_x, normal_y, tx, ty)[band].mean())


SOLVE_KW = dict(backend='numpy', period=[1., 1.], wavelength=.55, thickness=[.3],
                theta=np.deg2rad(12), phi=np.deg2rad(30), psi=np.deg2rad(40),
                top=1., bot=1.)


# ------------------------------------------------------------------------ plumbing

def test_explicit_distance_is_bit_identical_to_the_untouched_default():
    """The option must be opt-in: naming the default changes nothing, to the bit."""
    structure = circle_raster(32)[None]
    base = _legacy_wavebit_rcwa(structure=structure, truncation_order=[4, 4], factorization='nvf', **SOLVE_KW).solve()
    named = _legacy_wavebit_rcwa(structure=structure, truncation_order=[4, 4], factorization='nvf',
                         raster_normal_method='distance', **SOLVE_KW).solve()
    assert np.array_equal(np.asarray(base.R), np.asarray(named.R))
    assert np.array_equal(np.asarray(base.T), np.asarray(named.T))


def test_an_invalid_method_is_refused_by_name():
    with pytest.raises(ValueError, match='raster_normal_method'):
        _legacy_wavebit_rcwa(structure=circle_raster(16)[None], truncation_order=[2, 2],
                     factorization='nvf', raster_normal_method='sobel', **SOLVE_KW)


@pytest.mark.parametrize('cfg', [
    dict(factorization='li'),   # explicit Li: no normal field at all
    dict(),                     # untouched default resolves to Li on a raster
])
def test_a_method_nothing_reads_is_refused_not_ignored(cfg):
    s = _legacy_wavebit_rcwa(structure=circle_raster(16)[None], truncation_order=[2, 2],
                     raster_normal_method='contour', **SOLVE_KW, **cfg)
    with pytest.raises(NotImplementedError, match='no reader'):
        s.solve()


def test_a_method_on_the_smooth_shape_path_is_refused():
    """The shape paths build their normal fields from the geometry itself."""
    smooth = [{'background': 1.0, 'shapes': [{'type': 'circle', 'cx': 0.0, 'cy': 0.0,
                                              'lx': 0.6, 'material': 5.76}]}]
    s = _legacy_wavebit_rcwa(structure=smooth, truncation_order=[2, 2], raster_normal_method='contour', **SOLVE_KW)
    with pytest.raises(NotImplementedError, match='no reader'):
        s.solve()


@pytest.mark.parametrize('method', NEW_METHODS)
def test_freeze_replay_is_bit_identical_and_method_mismatch_is_stale(method):
    """The record carries the construction; replaying it under another is refused."""
    structure = circle_raster(32)[None]
    s = _legacy_wavebit_rcwa(structure=structure, truncation_order=[3, 3], factorization='nvf',
                     raster_normal_method=method, **SOLVE_KW)
    geo = s.freeze_geometry()
    replayed = s.solve(geometry=geo)
    rebuilt = _legacy_wavebit_rcwa(structure=structure, truncation_order=[3, 3], factorization='nvf',
                           raster_normal_method=method, **SOLVE_KW).solve()
    assert np.array_equal(np.asarray(replayed.R), np.asarray(rebuilt.R))

    other = _legacy_wavebit_rcwa(structure=structure, truncation_order=[3, 3], factorization='nvf', **SOLVE_KW)
    with pytest.raises(StaleGeometryError, match='different construction'):
        other.solve(geometry=geo)


@pytest.mark.parametrize('method', NEW_METHODS)
def test_energy_is_conserved_by_every_construction(method):
    res = _legacy_wavebit_rcwa(structure=circle_raster(48)[None], truncation_order=[5, 5], factorization='nvf',
                       raster_normal_method=method, **SOLVE_KW).solve()
    total = float(np.sum(res.R) + np.sum(res.T))
    assert abs(total - 1.0) < 1e-10


# --------------------------------------------------------------- field invariants

@pytest.mark.parametrize('method', NEW_METHODS)
def test_unit_magnitude_everywhere_with_zeros_only_where_deliberate(method):
    """The factorization needs C_xx + C_yy = I, which is |n| = 1 across the cell."""
    eps = circle_raster(64)
    normal_x, normal_y = normal_vector_field(eps, method=method)
    mag = np.hypot(normal_x, normal_y)
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
    normal_x, normal_y = normal_vector_field(eps, method=method)
    assert float(np.abs(normal_y).max()) == 0.0


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('shape', [(1, 8), (8, 1)])
def test_a_one_cell_axis_keeps_the_constant_normal_rule(method, shape):
    """The 1D collapse onto Li must not depend on the construction."""
    eps = np.linspace(1.0, 2.0, 8).reshape(shape)
    normal_x, normal_y = normal_vector_field(eps, method=method)
    expect_x = 1.0 if shape[0] == 1 else 0.0
    assert np.array_equal(normal_x, np.full(shape, expect_x))
    assert np.array_equal(normal_y, np.full(shape, 1.0 - expect_x))


@pytest.mark.parametrize('method', NEW_METHODS)
def test_a_uniform_layer_gets_a_constant_field(method):
    normal_x, normal_y = normal_vector_field(np.full((16, 16), 1.7), method=method)
    assert np.array_equal(normal_x, np.ones((16, 16)))
    assert np.array_equal(normal_y, np.zeros((16, 16)))


@pytest.mark.parametrize('flip', ['y', 'x'])
def test_smooth_is_bit_exactly_covariant_under_both_flips(flip):
    """'smooth' mirrors exactly, like 'distance' — in BOTH directions. The x direction is
    the one every y-symmetric golden was historically blind to, so it is pinned by name."""
    eps = circle_raster(48, cx=0.4, cy=0.55)           # asymmetric on purpose
    normal_x, normal_y = normal_vector_field(eps, method='smooth')
    if flip == 'y':
        fx, fy = normal_vector_field(eps[::-1].copy(), method='smooth')
        assert np.array_equal((normal_x ** 2)[::-1], fx ** 2)
        assert np.array_equal((normal_x * normal_y)[::-1], -(fx * fy))
    else:
        fx, fy = normal_vector_field(eps[:, ::-1].copy(), method='smooth')
        assert np.array_equal((normal_x ** 2)[:, ::-1], fx ** 2)
        assert np.array_equal((normal_x * normal_y)[:, ::-1], -(fx * fy))


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
    cfg = dict(backend='numpy', period=[1., 1.], wavelength=.55, thickness=[.3], theta=0.,
               phi=0., psi=0., top=1., bot=1., truncation_order=[5, 5], factorization='nvf')
    r1 = _legacy_wavebit_rcwa(structure=eps[None], raster_normal_method=method, **cfg).solve()
    r2 = _legacy_wavebit_rcwa(structure=eps[:, ::-1].copy()[None], raster_normal_method=method,
                      **cfg).solve()
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
    normal_x, normal_y = normal_vector_field(eps, method=method)
    err = band_error(eps, normal_x, normal_y, cx=cx, cy=cy, r=0.22)
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
    normal_x, normal_y = normal_vector_field(eps, method=method)
    band = np.abs(rr - 0.3) * N <= 2
    tx, ty = dx / np.where(rr == 0, 1, rr), dy / np.where(rr == 0, 1, rr)
    err = float(angular_error_deg(normal_x, normal_y, tx, ty)[band].mean())
    assert err < 6.0


@pytest.mark.parametrize('method', NEW_METHODS)
def test_a_non_square_cell_measures_in_physical_space(method):
    """A circle in a 2:1 cell rasterized on a square grid has non-square pixels; a
    construction that stayed in index space would see an ellipse and tilt every
    normal (the period-pair trap). The exact radial normal is the referee."""
    from wavebit.lattice import as_lattice
    N = 64
    ax = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(ax * 2.0, ax)          # physical x in [0, 2), y in [0, 1)
    eps = np.ones((N, N))
    cx, cy, r = 1.0, 0.5, 0.3
    eps[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = 2.4
    lattice = as_lattice(period=[2.0, 1.0])
    normal_x, normal_y = normal_vector_field(eps, method=method, lattice=lattice)
    dx, dy = X - cx, Y - cy
    rr = np.hypot(dx, dy)
    band = (np.abs(rr - r) / (2.0 / N)) <= 2      # within ~2 of the coarser pixel
    tx, ty = dx / np.where(rr == 0, 1, rr), dy / np.where(rr == 0, 1, rr)
    err = float(angular_error_deg(normal_x, normal_y, tx, ty)[band].mean())
    assert err < 6.0


# --------------------------------------------------- the cell metric, oblique included

SKEWS = [90.0, 60.0, 30.0]


def skewed_disc(N, gamma_deg, P=400.0, r=60.0, eps_in=5.76):
    """``(A, eps, tx, ty)``: a disc on the lattice grid of a skewed cell, and its exact normal.

    ``eps[j, i]`` is the permittivity at ``(i+.5)/N * a1 + (j+.5)/N * a2`` — the convention an
    oblique raster is read in (``docs/theory/rcwa/oblique-lattice.md``), and what makes this a disc rather
    than the ellipse a Cartesian-drawn array would describe. Membership uses the minimum image
    over the lattice, so the shape is genuinely periodic; drawn without it, it is a different
    structure at the wrap and every construction here is periodic.

    ``r = 60`` of a 400-unit cell, rather than the 90 the oblique tests use elsewhere, for a
    reason worth stating: at ``gamma = 30`` the cell is only 200 units thick across ``a1``, so
    an ``r = 90`` disc passes within 20 units of its own periodic copy — comparable to
    ``'smooth'``'s period/16 kernel, which then reads two shapes as one and scores 1.8 degrees
    for a reason that has nothing to do with the metric. That is a property of the scene, and
    this file is measuring the metric.
    """
    a1 = np.array([P, 0.0])
    a2 = P * np.array([np.cos(np.deg2rad(gamma_deg)), np.sin(np.deg2rad(gamma_deg))])
    A = np.column_stack((a1, a2))
    # `wavebit.meshgrid_xy` centres the cell on the origin, so these displacements are its output
    X, Y = (np.asarray(c) for c in _legacy_wavebit_meshgrid_xy((N, N), lattice=A))
    d = np.stack([X, Y], axis=-1)
    rho, disp = None, None
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            s = d - i * a1 - j * a2
            rr = np.hypot(s[..., 0], s[..., 1])
            if rho is None:
                rho, disp = rr, s
            else:
                nearer = rr < rho
                rho, disp = np.where(nearer, rr, rho), np.where(nearer[..., None], s, disp)
    safe = np.where(rho == 0.0, 1.0, rho)
    return (A, np.where(rho <= r, eps_in, 1.0), disp[..., 0] / safe, disp[..., 1] / safe)


def skew_error(method, gamma, N=128):
    """Mean angular error against the disc's exact normal, over the interface mask."""
    A, eps, tx, ty = skewed_disc(N, gamma)
    normal_x, normal_y = normal_vector_field(eps, lattice=A, method=method)
    return float(angular_error_deg(normal_x, normal_y, tx, ty)[_interface_mask(eps)].mean())


@pytest.mark.parametrize('method', NEW_METHODS)
def test_a_skewed_cell_costs_these_constructions_nothing(method):
    """Every construction here carries the cell metric, and a skew does not degrade it.

    Until 2026-08-21 all three refused a skewed basis, on the grounds that only
    ``'distance'`` carried the ``J^-T`` pullback that makes an index-space gradient a
    Cartesian normal. That was a fact about the code and not about the constructions: two of
    these never take an index-space difference at all — ``'contour'`` measures between
    Cartesian points, ``'spectral'`` multiplies by a Cartesian wavevector — so what they
    needed was the *coordinates*, and ``'smooth'`` needed the two per-axis scipy routines it
    leans on swapped for ones that can express a metric which is not diagonal.

    Two assertions, and the second is the one that catches a half-correction: an absolute
    bound at every shear, and no growth relative to the unskewed cell. Handing scipy the
    per-axis step lengths and calling that the metric — the most a diagonal consumer can be
    told — passes neither, leaving ``'smooth'`` between 2 and 12 degrees.
    """
    errs = {gamma: skew_error(method, gamma) for gamma in SKEWS}
    for gamma, err in errs.items():
        assert err < 2.0, f'{method} at gamma={gamma}: {err:.3f} deg'
    assert max(errs.values()) < 2.5 * errs[90.0], errs


def test_distance_is_the_field_a_skewed_cell_used_to_be_stuck_with():
    """The control that makes the bounds above a result rather than a loose tolerance.

    ``'distance'`` is not *wrong* on a skewed cell — it has carried the pullback since the
    lattice feature landed — it is quantized, and the skew makes the quantization worse: the
    handful of directions a pair of integer central differences can spell stays put while the
    cell they are read in shears away from them.
    """
    errs = {gamma: skew_error('distance', gamma) for gamma in SKEWS}
    assert errs[90.0] > 10.0, errs
    assert errs[30.0] > 1.3 * errs[90.0], errs


@pytest.mark.parametrize('method', METHODS)
def test_turning_the_cell_turns_the_field(method):
    """A rotated basis is one structure seen from a turned frame, so the field turns with it.

    The property no accuracy bound can state, and the one that separates a metric from a
    number that merely happens to fit: a rotation is an isometry, so nothing in any of these
    constructions may notice it beyond the direction of the answer. It also crosses the one
    branch in :mod:`wavebit.rcwa._engine.normal_methods` — rotating an axis-aligned cell makes the
    metric non-diagonal — so the separable implementations of the Gaussian and of the distance
    transform are compared against the frequency-domain and kd-tree ones here, rather than
    each only ever being measured within its own regime.

    Compared as a **mean** over the interface mask, not a maximum. ``'contour'`` and
    ``'spectral'`` carry tie-breaks (the nearest polyline sample, the nearest valid band
    donor) that a rotation can settle the other way on the medial ridge, where the direction
    is genuinely ambiguous — the same asymmetry the mirror note in ``normal_methods`` measures,
    and it puts single pixels tens of degrees out. The means sit between 1e-7 and 5e-3
    degrees; a construction that ignored the rotation would be out by tens of degrees on
    average, so the bound below separates the two by three orders.
    """
    turn = np.deg2rad(30.0)
    c, s = np.cos(turn), np.sin(turn)
    R = np.array([[c, -s], [s, c]])
    for gamma in (90.0, 60.0):
        A, eps, _, _ = skewed_disc(96, gamma)
        normal_x, normal_y = normal_vector_field(eps, lattice=A, method=method)
        gx, gy = normal_vector_field(eps, lattice=R @ A, method=method)
        err = angular_error_deg(gx, gy, R[0, 0] * normal_x + R[0, 1] * normal_y,
                                R[1, 0] * normal_x + R[1, 1] * normal_y)
        moved = float(err[_interface_mask(eps)].mean())
        assert moved < 0.05, f'{method} at gamma={gamma}: field moved {moved:.3e} deg'


def test_the_two_signed_distance_branches_measure_the_same_distance():
    """``'smooth'`` measures its signed distance two ways, and they have to be one quantity.

    A diagonal metric is a per-axis rescaling, which is exactly what
    ``distance_transform_edt``'s ``sampling`` expresses; a skewed one is not expressible that
    way at all, so it goes to a kd-tree over the tiled phase sets instead. Rotating the cell
    is what lets the two be compared: it moves no distance at all (an isometry) while moving
    the metric off the diagonal, so the two implementations must return the *same array*.

    Without this the kd-tree branch is only ever exercised where nothing independent knows
    the answer, and a construction that quietly measured to the wrong tiling — or dropped the
    sign — would still produce a plausible smooth field.
    """
    from wavebit.rcwa._engine.normal_methods import _metric, _periodic_signed_distance, _separable, _two_phase_indicator
    turn = 0.4
    c, s = np.cos(turn), np.sin(turn)
    R = np.array([[c, -s], [s, c]])
    for gamma in (90.0, 60.0):
        A, eps, _, _ = skewed_disc(64, gamma)
        inside = _two_phase_indicator(eps).astype(bool)
        M, M_turned = _metric(A, eps.shape), _metric(R @ A, eps.shape)
        assert _separable(M) == (gamma == 90.0) and not _separable(M_turned)
        moved = float(np.abs(_periodic_signed_distance(inside, M)
                             - _periodic_signed_distance(inside, M_turned)).max())
        assert moved < 1e-12, f'gamma={gamma}: the two branches differ by {moved:.2e}'


def test_smooths_kernel_is_a_length_rather_than_a_pixel_count_per_axis():
    """``'smooth'`` smooths by ``sqrt(cell area)/16``, one isotropic length (changed 2026-08-21).

    It used to be ``base_N/16`` *pixels along each axis*, which on a square cell is the same
    number and on any other cell is an ellipse stretched toward whichever axis spans more of
    the cell — a shape the sweep that froze the 1/16 never chose, since it only ever saw
    square cells. It is also not a property of the *lattice*, which is the decisive objection:
    ``a2' = a2 + a1`` generates the identical point lattice and would have moved the kernel.

    Measured on the 2:1 cell of :func:`test_a_non_square_cell_measures_in_physical_space`, the
    per-axis reading scores 2.22 degrees and the length 0.39 — so the bound below is what
    separates them, and it is why this is a test of its own rather than a tighter number on
    that one (``'contour'`` and ``'spectral'`` sit near 2 degrees on the same scene for
    reasons of their own, and would fail a shared bound that had teeth here).
    """
    from wavebit.lattice import as_lattice
    N = 64
    ax = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(ax * 2.0, ax)               # physical x in [0, 2), y in [0, 1)
    eps = np.ones((N, N))
    cx, cy, r = 1.0, 0.5, 0.3
    eps[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = 5.76
    normal_x, normal_y = normal_vector_field(eps, method='smooth', lattice=as_lattice(period=[2.0, 1.0]))
    dx, dy = X - cx, Y - cy
    rr = np.hypot(dx, dy)
    band = (np.abs(rr - r) / (2.0 / N)) <= 2
    tx, ty = dx / np.where(rr == 0, 1, rr), dy / np.where(rr == 0, 1, rr)
    err = float(angular_error_deg(normal_x, normal_y, tx, ty)[band].mean())
    assert err < 1.0, f'{err:.3f} deg — the per-axis kernel scored 2.22 here'


# ------------------------------------------------------------------- the floor lift

@pytest.mark.parametrize('method,base_n', [('contour', 16), ('smooth', 8)])
def test_the_floor_lift_keeps_the_base_grids_smoothing_scale(method, base_n):
    """A small layer at truncation_order=10 block-repeats to reach the 41-point floor. The pixel-unit
    policies must be read off the base grid and carried up — the same construction run
    on the lifted raster *as if it were the base* smooths too narrowly and produces a
    measurably different field. Equality with the base-aware call plus inequality with
    the base-blind one pins the threading, not just the plumbing.

    The base sizes are chosen where base-aware and base-blind genuinely differ:
    contour's sigma is nonlinear in N (0.25*sqrt(N)), so any lifted base differs;
    smooth's is linear (N/16), where base-aware and base-blind coincide *except* below
    the one-pixel floor — base 8 puts it there (1 base px = 6 lifted px against the
    blind reading's 3)."""
    from wavebit.rcwa._engine.factorization import _normal_field_floor, normal_field_for_raster
    base = circle_raster(base_n)
    repeat = _normal_field_floor(base.shape, 10, 10)[0]
    lifted = np.repeat(np.repeat(base, repeat, axis=0), repeat, axis=1)

    [(normal_x, normal_y)] = normal_field_for_raster(base[None], 10, 10, method=method)
    ex, ey = normal_vector_field(lifted, method=method, base_shape=(base_n, base_n))
    assert np.array_equal(normal_x, ex) and np.array_equal(normal_y, ey)

    bx, by = normal_vector_field(lifted, method=method)     # base-blind on purpose
    assert not np.allclose(normal_x, bx)


@pytest.mark.parametrize('method', NEW_METHODS)
def test_the_lifted_solve_runs_and_conserves_energy(method):
    res = _legacy_wavebit_rcwa(structure=circle_raster(16)[None], truncation_order=[10, 10],
                       factorization='nvf', raster_normal_method=method, **SOLVE_KW).solve()
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
        res = _legacy_wavebit_rcwa(backend='jax', structure=u, truncation_order=[2, 2], factorization='nvf',
                           raster_normal_method='contour', **{k: v for k, v in SOLVE_KW.items()
                                                       if k != 'backend'}).solve()
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

    s = _legacy_wavebit_rcwa(structure=eps[None], truncation_order=[3, 3], factorization='nvf',
                     raster_normal_method='spectral', **SOLVE_KW)
    geo = s.freeze_geometry()
    other = _legacy_wavebit_rcwa(structure=merged[None], truncation_order=[3, 3], factorization='nvf',
                         raster_normal_method='spectral', **SOLVE_KW)
    with pytest.raises(StaleGeometryError, match='distinct phases'):
        other.solve(geometry=geo)


# ------------------------------------------------------------- the public surfaces

def test_the_enum_is_exported_at_the_top_level():
    """Every other option enum is; the fourth axis must not need a private import."""
    assert _legacy_wavebit_NormalMethod.CONTOUR.value == 'contour'


def test_the_jobspec_surface_can_spell_the_option():
    """The declarative contract the manifest advertises must round-trip the option, and
    an unset field must stay unset (omitted from kwargs) so the spec never asks for
    anything the caller did not."""
    from wavebit._common._jobs_v1 import Numerics
    n = Numerics.from_dict({'raster_normal_method': 'contour', 'factorization': 'nvf'})
    assert n.raster_normal_method == NormalMethod.CONTOUR
    assert n.to_dict()['raster_normal_method'] == 'contour'
    assert Numerics.from_dict(n.to_dict()).raster_normal_method == NormalMethod.CONTOUR
    assert n.as_kwargs()['raster_normal_method'] == 'contour'
    assert 'raster_normal_method' not in Numerics.from_dict({}).as_kwargs()


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
    normal_x, normal_y = normal_vector_field(eps, method='spectral')
    tx, ty = dx / np.where(rr == 0, 1, rr), dy / np.where(rr == 0, 1, rr)
    err = angular_error_deg(normal_x, normal_y, tx, ty)
    shell = (np.abs(rr - 0.3) * N > 3) & (np.abs(rr - 0.3) * N <= 8)
    assert float(err[shell].max()) < 25.0
