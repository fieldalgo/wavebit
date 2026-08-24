"""Gradients with respect to a shape's *position* on the smooth path.

``cx`` and ``cy`` were the two shape parameters ``test_gradients`` never covered. The reason
was recorded as a gap — a finite-difference comparison that failed and got removed — and the
suspicion was sampling noise: the normal field is evaluated on a grid, so the objective
carries a sawtooth and a difference step that straddles a grid line reports the jump.

Translation settles it without finite differences. Sliding a single layer sideways changes
only the phase of each diffracted order, never its efficiency, so ``d/dcx`` and ``d/dcy`` are
**exactly zero** — a reference no discretization can argue with. Measured against it, the gap
turned out not to be sampling at all. Both halves of it came from one thing: the normal field
was *sampled at the shape's actual place*.

* **The bias.** A rectangle's normals end in ``sign()`` and a comparison mask, whose
  derivatives are zero everywhere, so the projectors contributed nothing to ``d/dcy`` while the
  permittivity term contributed its exact share — and the cancellation those two owe each other
  left the permittivity term standing alone at -1.7e-1.

* **The spread.** Not the grid: it survived at 4096 samples unchanged. The normal field is
  defined over the whole plane, and a fixed cell clips a different part of it depending on
  where the shape sits, so the same structure translated was a *different* structure to the
  factorization — 1.6e-3 on ``sum(R)`` for a circle.

:func:`wavebit._core.analytic.normal_projector_coeffs` now builds a lone shape's field at the
centre of the cell and translates the coefficients by a phase, which removes both at once. The
numbers below are what that is worth, and they are absolute now rather than relative to a
spread, because there is no longer a spread to measure against.
"""
import numpy as np
from wavebit._core.lattice import as_lattice
import pytest

import wavebit
from wavebit._core import analytic as _an

pytest.importorskip('jax')
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
import wavebit.on_jax                                        # noqa: E402,F401  (enables x64)

P = 0.77
BASE = dict(period=[P, P], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
            wavelength=0.777, top=1.0, bot=2.25, truncation_order=[4, 4])

# Nine positions, spaced so no two land on the same normal-field grid line — when this file was
# written that mattered, because the answer moved between grid lines. It no longer does, and
# the spacing now serves as a check that it does not. Confined to 0.30-0.70 of the period so
# that the widest shape here (a circle of radius 0.25*P) stays clear of the cell edge: a shape
# wrapping the boundary is a separate question from this one, and it does move the objective
# (1.5e-2 at 0.17*P, where the circle overhangs).
POSITIONS = [(0.30 + 0.05 * i) * P for i in range(9)]

CASES = [(s, p) for s in ('rectangle', 'ellipse', 'circle') for p in ('cx', 'cy')]
IDS = [f'{s}_{p}' for s, p in CASES]


def _shape(kind, param, value):
    s = dict(type=kind, cx=0.385, cy=0.385, lx=0.385, ly=0.231, angle=0.4, material=12.25)
    if kind == 'circle':
        # Same geometry as the `ly = lx` this used to write -- `lx` is the diameter and
        # `shape_geometry` resolves the second axis to it -- but a circle has no second length
        # to name, and `check_shape_spec` now says so rather than dropping it.
        del s['ly']
    s[param] = value
    return s


def _position_gradients(kind, param):
    """``d(sum R)/d(param)`` at each position. Every entry should be zero."""
    def objective(v):
        res = wavebit.rcwa(backend=1, structure=[dict(background=1.0, shapes=[_shape(kind, param, v)])],
                             **BASE).solve()
        return jnp.real(jnp.sum(res.R))

    return np.array([float(jax.grad(objective)(jnp.asarray(x))) for x in POSITIONS])


@pytest.mark.parametrize('kind,param', CASES, ids=IDS)
def test_translation_invariance_holds_in_the_objective(kind, param):
    """The efficiency really is flat under translation — and now it is flat to the last digit.

    This began as the premise: without it the gradient tests below would be asserting zero
    against an objective that genuinely varies, and would be wrong rather than revealing. It
    passed at 5e-3 while the true spread was 1.6e-3, which was the clipped normal field showing
    through. With the field carried by a phase the spread is 1.5e-14 — float noise — so the
    bound is where a real regression, not a rounding difference, would land.
    """
    def total(v):
        res = wavebit.rcwa(backend=0, structure=[dict(background=1.0, shapes=[_shape(kind, param, v)])],
                             **BASE).solve()
        return float(np.sum(np.asarray(res.R)).real)

    values = [total(x) for x in POSITIONS]
    assert np.ptp(values) < 1e-12, (kind, param, np.ptp(values))


@pytest.mark.parametrize('kind,param', CASES, ids=IDS)
def test_position_gradient_is_zero(kind, param):
    """Every entry is zero, to an absolute tolerance rather than against its own spread.

    The earlier form asked whether the mean sat inside one standard deviation, because the
    spread was 5e-3 to 8e-2 and a bound tight enough to catch the -1.7e-1 bias would have failed
    the honest cases too. That comparison has outlived its purpose in both directions: the
    spread is now 8e-14, so the criterion would flag the surviving 5e-9 as a bias when it is the
    solve's own float floor — the same 5e-9 appears at every position, which is precisely what a
    discretization artefact does not do.

    Measured: mean 4e-10 to 5e-9, spread 2e-14 to 8e-14, largest single value 4.6e-9.
    """
    g = _position_gradients(kind, param)
    assert np.abs(g).max() < 1e-7, (kind, param, g.mean(), g.std())


@pytest.mark.parametrize('kind,param', CASES, ids=IDS)
def test_position_gradient_carries_no_sampling_noise(kind, param):
    """And the spread on its own, which is the half that used to make this unusable.

    A position optimizer sees one point at a time, not a mean, so the 5e-3 to 8e-2 spread was
    the more practical obstacle of the two even where the mean was honest. It is 8e-14 now.
    Kept separate from the bound above because the two fail for different reasons: this one
    catches the normal field being sampled in place again, that one catches a term losing its
    phase.
    """
    g = _position_gradients(kind, param)
    assert g.std() < 1e-11, (kind, param, g.std())


def test_the_projectors_move_by_a_phase_and_by_nothing_else():
    """Straight at the mechanism, with no solver in the way.

    The claim the whole file rests on: the three projector coefficient grids at ``cy`` are the
    grids at ``cy0`` times ``exp(-2i*pi*ky*(cy-cy0)/Py)``, exactly. If that holds then every
    convolution matrix is ``D X D^-1`` for one diagonal ``D``, the solve is conjugated by it,
    and each order's efficiency cannot move — which is why the assertions above can be absolute.
    """
    from wavebit._core.analytic import normal_projector_coeffs

    truncation_order, grid = 4, 256
    cy0, cy = 0.385, 0.31 * P

    def coeffs(v):
        return normal_projector_coeffs(jnp, [_shape('rectangle', 'cy', v)], (grid, grid),
                                       as_lattice(period=(P, P)), truncation_order,
                                       truncation_order, jnp.complex128)

    ky = np.arange(-2 * truncation_order, 2 * truncation_order + 1)[:, None]
    phase = np.exp(-2j * np.pi * ky * (cy - cy0) / P)

    for at_ref, at_cy in zip(coeffs(cy0), coeffs(cy)):
        moved = np.asarray(at_ref) * phase
        assert np.abs(np.asarray(at_cy) - moved).max() < 1e-14


def test_the_position_reaches_the_projectors_through_that_phase():
    """...and autograd sees it, which is the part ``sign()`` used to swallow.

    On the DC coefficient the phase is 1 and there is nothing to see, so this asks a coefficient
    that moves: ``ky = 1``, whose derivative is ``-2i*pi/Py`` times itself. A difference quotient
    is the independent witness; before the fix autograd returned exactly 0.0 here.
    """
    from wavebit._core.analytic import normal_projector_coeffs

    truncation_order, grid = 4, 256

    def c_ky1(v):
        c_xx = normal_projector_coeffs(jnp, [_shape('rectangle', 'cy', v)], (grid, grid),
                                       as_lattice(period=(P, P)), truncation_order,
                                       truncation_order, jnp.complex128)[0]
        return jnp.real(c_xx[2 * truncation_order + 1, 2 * truncation_order])

    cy, h = 0.31 * P, 1e-6
    auto = float(jax.grad(c_ky1)(jnp.asarray(cy)))
    fd = float((c_ky1(jnp.asarray(cy + h)) - c_ky1(jnp.asarray(cy - h))) / (2 * h))

    assert abs(fd) > 1e-3, fd                      # the coefficient has to move to be a test
    assert abs(auto - fd) < 1e-6 * abs(fd), (auto, fd)


def test_the_sampled_field_no_longer_depends_on_where_in_the_cell_a_shape_SITS():
    """What ``sign()`` used to hide, and what removing it exposed instead.

    This replaced ``test_the_rectangle_normal_field_still_cannot_carry_a_position`` on
    2026-08-17. That test asserted ``autograd == 0.0`` for a rectangle -- the signature of
    ``sign()``, whose derivative vanishes -- against a finite central difference, and called
    the difference real. It was not. Its own docstring said so: "That ``<normal_x**2>`` moves at all
    is the clipped window, not physics: a circle's mean ``normal_x**2`` is 1/2 by symmetry and reads
    1/2 only when it sits at the centre of the cell."

    Folding the displacement to the nearest image removed the window, so the quantity that
    test differentiated is now flat. A circle reads 1/2 wherever it sits (2.7e-7 across a whole
    period, against a construction that used to swing 0.44 to 1.00), and a rectangle is flat to
    7e-5. What is left inside one pixel is a sawtooth of that amplitude which returns exactly
    at the next sample -- resampling, not geometry, and it flips sign with the grid
    (-4.4e-2 / +6.7e-2 / -3.7e-2 at N = 128 / 256 / 512). Differentiating it measures the
    resampling, which is why the replacement asserts flatness rather than a derivative.

    The position gradient that *is* physical -- one shape moving relative to another -- is
    measured at the solve in the two tests above, where it converges under grid refinement
    rather than changing sign.
    """
    from wavebit._core.analytic import normal_vector_field_analytical

    def mean_nx2(kind, cy):
        normal_x, _ = normal_vector_field_analytical(
            jnp, (256, 256), [_shape(kind, 'cy', cy)], jnp.float64,
            as_lattice(period=(P, P)))
        return float(jnp.mean(normal_x ** 2))

    for kind, tol in (('circle', 1e-5), ('ellipse', 1e-4), ('rectangle', 1e-3)):
        seen = [mean_nx2(kind, c * P) for c in (0.10, 0.31, 0.50, 0.77, 1.31)]
        assert np.ptp(seen) < tol, (kind, seen)

    # A circle's mean normal_x**2 is 1/2 by symmetry, and now reads it from anywhere in the cell.
    assert abs(mean_nx2('circle', 0.13 * P) - 0.5) < 1e-4


# ------------------------------------------------------------------ more than one layer
#
# Everything above is one layer, where the phase cancels against itself. A stack asks a
# different question, and it has two halves that have to be tested together: translating the
# whole stack must change nothing, and translating *one* layer of it must change something.
# Only the first would pass just as well if the phases cancelled globally for a reason that had
# nothing to do with each layer being right.

STACK = dict(period=[P, P], thickness=[0.25, 0.30], theta=0.35, phi=0.4, psi=0.0,
             wavelength=0.777, top=1.0, bot=2.25, truncation_order=[4, 4])
CX0, CX1 = 0.35 * P, 0.55 * P


def _stack(cx0, cx1):
    return [dict(background=1.0, shapes=[dict(type='rectangle', cx=cx0, cy=0.385, lx=0.385,
                                          ly=0.231, angle=0.4, material=12.25)]),
            dict(background=1.0, shapes=[dict(type='circle', cx=cx1, cy=0.462, lx=0.385,
                                          material=6.25)])]


def _stack_total(cx0, cx1):
    res = wavebit.rcwa(backend=0, structure=_stack(cx0, cx1), **STACK).solve()
    return float(np.real(np.sum(np.asarray(res.R))))


def _stack_jtotal(cx0, cx1):
    res = wavebit.rcwa(backend=1, structure=_stack(cx0, cx1), **STACK).solve()
    return jnp.real(jnp.sum(res.R))


def test_a_stack_translated_bodily_does_not_move():
    """Sliding every layer by the same amount is still one translation, so still exactly zero.

    Each layer carries its own phase ``D_i``, and a common shift multiplies all of them by the
    same ``D``, so the whole solve is conjugated by it. Measured: 1.9e-14 across the sweep, and
    the derivative with respect to the common shift is 3.8e-9 — the same float floor the
    single-layer cases sit on.
    """
    shifts = np.linspace(0.0, 0.15, 5)

    values = [_stack_total(CX0 + d, CX1 + d) for d in shifts]
    assert np.ptp(values) < 1e-12, np.ptp(values)

    g = np.array([float(jax.grad(lambda d: _stack_jtotal(CX0 + d, CX1 + d))(jnp.asarray(d)))
                  for d in shifts[:3]])
    assert np.abs(g).max() < 1e-7, g


def test_moving_one_layer_of_two_moves_the_answer_and_the_gradient_says_by_how_much():
    """And the half that the invariance alone cannot see.

    Moving one layer changes the two layers' *relative* position, which is a different structure
    — ``sum(R)`` runs over 1.6e-2 here — so the right answer is a large non-zero gradient,
    not a small one. A construction that cancelled the phases globally would pass the test above
    and fail this one; each layer's phase has to be its own.

    Against central differences, at three configurations and for both layers: agreement to
    2.6e-9 - 2.6e-8 relative, on gradients of 0.12 to 0.81.
    """
    h = 1e-6
    for d in (0.0, 0.05, 0.10):
        auto0 = float(jax.grad(lambda c: _stack_jtotal(c, CX1))(jnp.asarray(CX0 + d)))
        fd0 = (_stack_total(CX0 + d + h, CX1) - _stack_total(CX0 + d - h, CX1)) / (2 * h)
        assert abs(fd0) > 1e-2, (d, fd0)          # the layers really do see each other
        assert abs(auto0 - fd0) < 1e-5 * abs(fd0), (d, auto0, fd0)

        auto1 = float(jax.grad(lambda c: _stack_jtotal(CX0, c))(jnp.asarray(CX1 + d)))
        fd1 = (_stack_total(CX0, CX1 + d + h) - _stack_total(CX0, CX1 + d - h)) / (2 * h)
        assert abs(fd1) > 1e-2, (d, fd1)
        assert abs(auto1 - fd1) < 1e-5 * abs(fd1), (d, auto1, fd1)


# ------------------------------------------------------- more than one shape in a layer
#
# The limitation, and it is two things rather than one. A layer's field is ``sum(g/|g|)``
# divided by its own norm, so:
#
#   (a) a *common* translation is not carried by a phase, and the clipped window comes back;
#   (b) moving one shape *relative* to the others genuinely reshapes the field, and a
#       rectangle's share of that reshaping is invisible to a grid — its level-set gradient is
#       piecewise constant, so the derivative lives on the wedge boundaries as a line measure
#       and no sample sees it.
#
# (a) alone is fixable and was tried: place the shapes at their offsets from the *mean* of the
# centres, which reduces to the present rule when there is one shape, and carry that mean as
# the phase. It works on paper and in the generic case — a rectangle pair goes from 1.2e-4 to
# 6.6e-15. It was rejected because it manufactures the degeneracy it needs to avoid: two shapes
# always land exactly opposed about the centre of the cell, so their unit fields cancel exactly
# along the segment between them. For two identical circles that puts 16 grid points at
# ``|n_total| == 0`` (410 below 1e-12), where ``_hypot_safe`` returns a finite value but its
# gradient is ``0 * inf`` — the common-translation gradient comes back NaN, and the forward
# jitters by 2.4e-6 as float rounding moves which points land on the segment. It also made (b)
# worse for rectangles, 1.6e-1 to 2.9e-1 at the same point. One defect for two.
#
# (b) is not reachable by any phase, and not by the closed form that would work for a lone
# rectangle either: the *combined* normalized field is not piecewise constant even when both
# shapes are rectangles, because ``|sum|`` varies continuously between them. It needs a normal
# field whose rectangle branch is differentiable in position to begin with — which is the
# ``tanh`` mollification already rejected, on this path, for breaking
# ``test_analytic_angle_is_smooth_through_normal``.
#
# So: smooth shapes are already right in (b) and only (a) is missing; rectangles are wrong in
# both. Both halves are pinned below so that neither can be assumed from the other.

TWO_CFG = dict(period=[P, P], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
               wavelength=0.777, top=1.0, bot=2.25, truncation_order=[4, 4])
CA, CB = 0.28 * P, 0.68 * P

TWO_CIRCLES = (dict(type='circle', cy=0.30 * P, lx=0.20 * P, material=9.0),
               dict(type='circle', cy=0.70 * P, lx=0.20 * P, material=4.840000000000001))
TWO_RECTANGLES = (dict(type='rectangle', cy=0.30 * P, lx=0.20 * P, ly=0.15 * P, angle=0.4,
                       material=9.0),
                  dict(type='rectangle', cy=0.70 * P, lx=0.20 * P, ly=0.15 * P, angle=0.9,
                       material=4.840000000000001))


def _two(pair, ca, cb):
    return [dict(background=1.0, shapes=[dict(pair[0], cx=ca), dict(pair[1], cx=cb)])]


def _two_total(pair, ca, cb):
    res = wavebit.rcwa(backend=0, structure=_two(pair, ca, cb), **TWO_CFG).solve()
    return float(np.real(np.sum(np.asarray(res.R))))


def _two_jtotal(pair, ca, cb):
    res = wavebit.rcwa(backend=1, structure=_two(pair, ca, cb), **TWO_CFG).solve()
    return jnp.real(jnp.sum(res.R))


def test_two_smooth_shapes_already_carry_a_correct_relative_gradient():
    """Half of the multi-shape case is not broken, and saying so is the point of this test.

    A circle's normals are ``2*xr/a**2``, linear in position, so moving one circle past another
    reaches the field through arithmetic that differentiates. Autograd matches central
    differences to 4.7e-6 - 3.1e-5 here. Whatever is wrong with a multi-shape layer, it is not
    that the *sum* blocks the gradient — it is the rectangle, which the next test pins.
    """
    h = 1e-5
    for d in (0.02, 0.05, 0.08):
        auto = float(jax.grad(lambda c: _two_jtotal(TWO_CIRCLES, c, CB))(jnp.asarray(CA + d)))
        fd = (_two_total(TWO_CIRCLES, CA + d + h, CB)
              - _two_total(TWO_CIRCLES, CA + d - h, CB)) / (2 * h)
        assert abs(fd) > 1e-4, (d, fd)
        assert abs(auto - fd) < 1e-3 * abs(fd), (d, auto, fd)


def test_two_rectangles_carry_a_correct_relative_gradient_too():
    """The half that used to be impossible, and what made it possible.

    Until 2026-08-17 a rectangle's normals ended in ``sign()`` and the nearest face was picked
    with a hard ``where``, so the field was piecewise constant: all of its position dependence
    sat on seams that translating the shape slid across, and a grid sample cannot see a seam
    move. That is not a resolution problem -- the error was flat at 1.8e-2 from normal_grid
    1024 to 4096 while the circle pair converged -- and the only escape on offer was a ``tanh``
    mollification, rejected because no width served both this and
    ``test_analytic_angle_is_smooth_through_normal``.

    The seams are gone rather than smoothed. Treating +x and -x (and +y, -y) as four separate
    faces blended by distance products makes them cancel exactly at the centre line and
    collapse exactly at each face, with no width anywhere: autograd goes from 212-274% out to
    1e-5. ``h`` is 1e-6 rather than 1e-5 because the difference, not the derivative, is the
    limiting party here -- at 1e-5 the d = 0.08 point sits at 1.00e-2 on truncation alone, and
    every point converges as h shrinks.
    """
    h = 1e-6
    for d in (0.02, 0.05, 0.08):
        auto = float(jax.grad(lambda c: _two_jtotal(TWO_RECTANGLES, c, CB))(jnp.asarray(CA + d)))
        fd = (_two_total(TWO_RECTANGLES, CA + d + h, CB)
              - _two_total(TWO_RECTANGLES, CA + d - h, CB)) / (2 * h)
        assert abs(fd) > 1e-4, (d, fd)
        assert abs(auto - fd) < 1e-2 * abs(fd), (d, auto, fd)


def test_a_layer_of_two_shapes_is_translation_invariant_too():
    """Sliding every shape together is one structure named twice, and no phase carries it.

    A lone shape is translated by a phase on its coefficients, which is exact. Several shapes
    combine nonlinearly, so a common shift is not a common factor and the field has to be
    right about the shift itself. It now is: fold the displacement to the nearest image and the
    combination of periodic fields is periodic.

    Stated at two tolerances because two different things are being measured. A shift that
    lands on the sampling grid is exact -- the same field read at the same points. A
    fractional-pixel shift resamples it elsewhere and leaves a discretization residual that
    converges with the grid (7.3e-6, 5.2e-6, 1.8e-6, 7.6e-7 at normal_grid 256 to 2048), which
    is what separates it from the clipped window this used to measure at 1e-2.
    """
    cfg = dict(backend=0, period=(1.0, 1.0), wavelength=0.94, thickness=(0.32,), truncation_order=(4, 4),
               top=1.0, bot=2.1025, theta=0.0, phi=None, psi=np.pi / 2)

    def total(d, grid=256):
        shapes = [dict(type='rectangle', cx=0.30 + d, cy=0.5, lx=0.30, ly=0.24, angle=0.0,
                       material=12.1104),
                  dict(type='circle', cx=0.70 + d, cy=0.30, lx=0.22, angle=0.0, material=5.76)]
        structure = [dict(background=1.0, shapes=shapes)]
        return float(np.sum(np.asarray(
            wavebit.rcwa(structure=structure, normal_grid=grid, **cfg).solve().T)).real)

    on_grid = [total(k / 256) for k in (0, 8, 16, 32, 64)]
    assert np.ptp(on_grid) < 1e-11, np.ptp(on_grid)

    coarse = np.ptp([total(d, 256) for d in np.linspace(0.0, 0.2, 5)])
    fine = np.ptp([total(d, 1024) for d in np.linspace(0.0, 0.2, 5)])
    assert coarse < 1e-4, coarse
    assert fine < coarse, (coarse, fine)


def test_a_multishape_layer_is_periodic_across_the_cell_edge():
    """The invariant the normalized-sum field could not hold, and this one does.

    A shape straddling the cell boundary may be named in any period -- ``cx`` and ``cx + Px``
    are one structure -- so every efficiency must be unchanged. A single-shape layer always
    agreed, because the phase in ``normal_projector_coeffs`` carries it. A multi-shape layer
    did not, and could not: the field was a normalized sum of each shape's field sampled at
    ``X - cx``, which is the displacement to the shape *as named*, so naming it a period over
    slid a different part of an infinite wedge into the window. Spread was 1.9e-2 on this case,
    and the caller's only defence was the docstring telling them to clamp the centres.

    Both halves of the fix show here. The displacement is folded to the nearest image, so one
    shape's field is periodic; and the shapes are combined by distance-product weights rather
    than as a sum of unit vectors, so the combination of periodic fields stays periodic. This
    test replaced ``test_a_multishape_layer_is_not_periodic_across_the_cell_edge`` on
    2026-08-17, which existed to fail loudly on the day the path was repaired.
    """
    cfg = dict(backend=0, period=(1.0, 1.0), wavelength=0.94, thickness=(0.32,), truncation_order=(4, 4),
               top=1.0, bot=2.1025, theta=0.0, phi=None, psi=np.pi / 2)
    fixed = dict(type='rectangle', cx=0.72, cy=0.62, lx=0.26, ly=0.30, angle=0.0, material=4.41)

    def total(cx, shapes_after):
        moving = dict(type='rectangle', cx=cx, cy=0.5, lx=0.30, ly=0.24, angle=0.0,
                      material=12.1104)
        structure = [dict(background=1.0, shapes=[moving] + shapes_after)]
        return float(np.sum(np.asarray(
            wavebit.rcwa(structure=structure, normal_grid=256, **cfg).solve().T)).real)

    alone = [total(cx, []) for cx in (0.10, 1.10, 2.10)]
    assert np.ptp(alone) < 1e-12, alone

    pair = [total(cx, [fixed]) for cx in (0.10, 1.10, 2.10)]
    assert np.ptp(pair) < 1e-9, pair

    # The projector coefficients are what used to move; the permittivity never did. Comparing
    # both keeps this a statement about the normal field rather than about the solve.
    shapes = lambda cx: [dict(type='rectangle', cx=cx, cy=0.5, lx=0.30, ly=0.24, angle=0.0,
                              material=12.1104), fixed]
    eps = [np.asarray(_an.assemble_eps_coeffs(np, shapes(cx), 1.0, 4, 4,
                                              as_lattice(period=(1.0, 1.0)), np.complex128)) for cx in (0.10, 1.10)]
    assert np.max(np.abs(eps[0] - eps[1])) < 1e-13
    nxx = [np.asarray(_an.normal_projector_coeffs(np, shapes(cx), (256, 256),
                                                  as_lattice(period=(1.0, 1.0)), 4, 4,
                                                  np.complex128)[0])
           for cx in (0.10, 1.10)]
    assert np.max(np.abs(nxx[0] - nxx[1])) < 1e-12


def test_a_common_translation_of_every_shape_leaves_the_answer_alone():
    """The other half of the same invariant, and the one a phase never reached.

    Sliding *all* the shapes together is again one structure named twice, but no phase carries
    it: the field is a nonlinear combination, so a common shift is not a common factor. It is
    exact here only for a shift that lands on the sampling grid -- a fractional-pixel shift
    resamples a periodic field at different points and leaves ~6e-5, which is discretization
    and not the clipped window this used to be confused with.
    """
    cfg = dict(backend=0, period=(1.0, 1.0), wavelength=0.94, thickness=(0.32,), truncation_order=(4, 4),
               top=1.0, bot=2.1025, theta=0.0, phi=None, psi=np.pi / 2)

    def total(shift):
        shapes = [dict(type='rectangle', cx=0.30 + shift, cy=0.5, lx=0.30, ly=0.24,
                       angle=0.0, material=12.1104),
                  dict(type='rectangle', cx=0.72 + shift, cy=0.62, lx=0.26, ly=0.30,
                       angle=0.0, material=4.41)]
        return float(np.sum(np.asarray(wavebit.rcwa(
            structure=[dict(background=1.0, shapes=shapes)], normal_grid=256, **cfg)
            .solve().T)).real)

    on_grid = [total(k / 256) for k in (0, 16, 32, 64)]
    assert np.ptp(on_grid) < 1e-11, on_grid
