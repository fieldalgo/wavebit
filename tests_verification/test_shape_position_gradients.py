"""Gradients with respect to a shape's *position* on the analytic path.

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
  factorization — 1.6e-3 on ``sum(de_ri)`` for a circle.

:func:`fieldwave._core.analytic.normal_projector_coeffs` now builds a lone shape's field at the
centre of the cell and translates the coefficients by a phase, which removes both at once. The
numbers below are what that is worth, and they are absolute now rather than relative to a
spread, because there is no longer a spread to measure against.
"""
import numpy as np
import pytest

import fieldwave

pytest.importorskip('jax')
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
import fieldwave.on_jax                                        # noqa: E402,F401  (enables x64)

P = 0.77
BASE = dict(period=[P, P], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
            wavelength=0.777, n_top=1.0, n_bot=1.5, fto=[4, 4])

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
    s = dict(type=kind, cx=0.385, cy=0.385, lx=0.385, ly=0.231, angle=0.4, n_index=3.5)
    if kind == 'circle':
        s['ly'] = s['lx']
    s[param] = value
    return s


def _position_gradients(kind, param):
    """``d(sum de_ri)/d(param)`` at each position. Every entry should be zero."""
    def objective(v):
        res = fieldwave.rcwa(backend=1, ucell=[dict(n_base=1.0, shapes=[_shape(kind, param, v)])],
                             **BASE).conv_solve()
        return jnp.real(jnp.sum(res.de_ri))

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
        res = fieldwave.rcwa(backend=0, ucell=[dict(n_base=1.0, shapes=[_shape(kind, param, v)])],
                             **BASE).conv_solve()
        return float(np.sum(np.asarray(res.de_ri)).real)

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
    from fieldwave._core.analytic import normal_projector_coeffs

    fto, grid = 4, 256
    cy0, cy = 0.385, 0.31 * P

    def coeffs(v):
        return normal_projector_coeffs(jnp, [_shape('rectangle', 'cy', v)], (grid, grid),
                                       (P, P), fto, fto, jnp.complex128)

    ky = np.arange(-2 * fto, 2 * fto + 1)[:, None]
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
    from fieldwave._core.analytic import normal_projector_coeffs

    fto, grid = 4, 256

    def c_ky1(v):
        c_xx = normal_projector_coeffs(jnp, [_shape('rectangle', 'cy', v)], (grid, grid),
                                       (P, P), fto, fto, jnp.complex128)[0]
        return jnp.real(c_xx[2 * fto + 1, 2 * fto])

    cy, h = 0.31 * P, 1e-6
    auto = float(jax.grad(c_ky1)(jnp.asarray(cy)))
    fd = float((c_ky1(jnp.asarray(cy + h)) - c_ky1(jnp.asarray(cy - h))) / (2 * h))

    assert abs(fd) > 1e-3, fd                      # the coefficient has to move to be a test
    assert abs(auto - fd) < 1e-6 * abs(fd), (auto, fd)


def test_the_rectangle_normal_field_still_cannot_carry_a_position():
    """Why the phase exists, pinned where it happens.

    ``<nx**2>`` over the cell moves when the shape moves — finite differences say so for all
    three shapes. Autograd agrees for the ellipse and the circle and returns *exactly* zero for
    the rectangle, which is what ``sign()`` does: it is silent, it raises nothing, and zero is
    indistinguishable from "this term does not contribute". Nothing here is fixed and nothing
    here needs to be; ``normal_projector_coeffs`` simply stopped sending a position through it.

    Its other half is the finite difference itself. That ``<nx**2>`` moves at all is the clipped
    window, not physics: a circle's mean ``nx**2`` is 1/2 by symmetry and reads 1/2 only when it
    sits at the centre of the cell.
    """
    from fieldwave._core.analytic import normal_vector_field_analytical

    def mean_nx2(kind, cy):
        nx, _ = normal_vector_field_analytical(
            jnp, (256, 256), [_shape(kind, 'cy', cy)], (P, P), jnp.float64)
        return jnp.mean(nx ** 2)

    cy, h = 0.31 * P, 1e-4
    for kind in ('rectangle', 'ellipse', 'circle'):
        auto = float(jax.grad(lambda v: mean_nx2(kind, v))(jnp.asarray(cy)))
        fd = float((mean_nx2(kind, jnp.asarray(cy + h))
                    - mean_nx2(kind, jnp.asarray(cy - h))) / (2 * h))
        assert abs(fd) > 1e-3, (kind, fd)          # the field really does depend on cy
        if kind == 'rectangle':
            assert auto == 0.0, (kind, auto)       # ...and autograd says it does not
        else:
            assert abs(auto - fd) < 0.05 * abs(fd), (kind, auto, fd)


# ------------------------------------------------------------------ more than one layer
#
# Everything above is one layer, where the phase cancels against itself. A stack asks a
# different question, and it has two halves that have to be tested together: translating the
# whole stack must change nothing, and translating *one* layer of it must change something.
# Only the first would pass just as well if the phases cancelled globally for a reason that had
# nothing to do with each layer being right.

STACK = dict(period=[P, P], thickness=[0.25, 0.30], theta=0.35, phi=0.4, psi=0.0,
             wavelength=0.777, n_top=1.0, n_bot=1.5, fto=[4, 4])
CX0, CX1 = 0.35 * P, 0.55 * P


def _stack(cx0, cx1):
    return [dict(n_base=1.0, shapes=[dict(type='rectangle', cx=cx0, cy=0.385, lx=0.385,
                                          ly=0.231, angle=0.4, n_index=3.5)]),
            dict(n_base=1.0, shapes=[dict(type='circle', cx=cx1, cy=0.462, lx=0.385,
                                          n_index=2.5)])]


def _stack_total(cx0, cx1):
    res = fieldwave.rcwa(backend=0, ucell=_stack(cx0, cx1), **STACK).conv_solve()
    return float(np.real(np.sum(np.asarray(res.de_ri))))


def _stack_jtotal(cx0, cx1):
    res = fieldwave.rcwa(backend=1, ucell=_stack(cx0, cx1), **STACK).conv_solve()
    return jnp.real(jnp.sum(res.de_ri))


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
    — ``sum(de_ri)`` runs over 1.6e-2 here — so the right answer is a large non-zero gradient,
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
               wavelength=0.777, n_top=1.0, n_bot=1.5, fto=[4, 4])
CA, CB = 0.28 * P, 0.68 * P

TWO_CIRCLES = (dict(type='circle', cy=0.30 * P, lx=0.20 * P, n_index=3.0),
               dict(type='circle', cy=0.70 * P, lx=0.20 * P, n_index=2.2))
TWO_RECTANGLES = (dict(type='rectangle', cy=0.30 * P, lx=0.20 * P, ly=0.15 * P, angle=0.4,
                       n_index=3.0),
                  dict(type='rectangle', cy=0.70 * P, lx=0.20 * P, ly=0.15 * P, angle=0.9,
                       n_index=2.2))


def _two(pair, ca, cb):
    return [dict(n_base=1.0, shapes=[dict(pair[0], cx=ca), dict(pair[1], cx=cb)])]


def _two_total(pair, ca, cb):
    res = fieldwave.rcwa(backend=0, ucell=_two(pair, ca, cb), **TWO_CFG).conv_solve()
    return float(np.real(np.sum(np.asarray(res.de_ri))))


def _two_jtotal(pair, ca, cb):
    res = fieldwave.rcwa(backend=1, ucell=_two(pair, ca, cb), **TWO_CFG).conv_solve()
    return jnp.real(jnp.sum(res.de_ri))


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


@pytest.mark.xfail(
    strict=True,
    reason="a rectangle's normals end in sign(), and with a second shape in the layer there is "
           'no phase to route its position around that: moving it relative to the other really '
           'does reshape the normalized sum, and a grid cannot see a piecewise-constant '
           "field's derivative")
def test_two_rectangles_carry_a_correct_relative_gradient_too():
    """The half that is broken, at the point where it breaks.

    Same question as above with rectangles instead of circles, and autograd is off by 11% to
    105% against central differences. This is ``sign()`` again — the same zero derivative the
    single-shape case had — but here the phase cannot be used to route around it, because the
    field genuinely changes shape rather than translating.
    """
    h = 1e-5
    for d in (0.02, 0.05, 0.08):
        auto = float(jax.grad(lambda c: _two_jtotal(TWO_RECTANGLES, c, CB))(jnp.asarray(CA + d)))
        fd = (_two_total(TWO_RECTANGLES, CA + d + h, CB)
              - _two_total(TWO_RECTANGLES, CA + d - h, CB)) / (2 * h)
        assert abs(fd) > 1e-4, (d, fd)
        assert abs(auto - fd) < 1e-2 * abs(fd), (d, auto, fd)


@pytest.mark.xfail(
    strict=True,
    reason='a layer of several shapes takes a normalized sum of their fields, which no single '
           'phase translates, so its normal field is still sampled in place; centring on the '
           'mean of the shape centres fixes this but manufactures exact cancellations (see the '
           'note above this test)')
def test_a_layer_of_two_shapes_is_translation_invariant_too():
    """The other half: a common translation, which is the same zero a lone shape gets exactly.

    Moving *both* shapes by the same amount is one physical translation, so this is the same
    statement ``test_translation_invariance_holds_in_the_objective`` makes. A multi-shape layer
    keeps the sampled field and the clipped window: 7.1e-4 on ``sum(de_ri)`` here, 1.2e-4 for
    the tighter pairs above, against the 1.5e-14 a lone shape gets.

    Interpolating each shape's field onto the common grid was tried and rejected — it brings the
    forward spread down to 7e-5 but the residue oscillates over one grid cell, so the *gradient*
    lands at 2e-1, worse than the bias it would remove.
    """
    cfg = dict(period=[1.0, 1.2], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
               wavelength=0.777, n_top=1.0, n_bot=1.5, fto=[4, 4])

    def total(d):
        ucell = [dict(n_base=1.0, shapes=[
            dict(type='rectangle', cx=0.30 + d, cy=0.40, lx=0.25, ly=0.20, angle=0.4,
                 n_index=3.5),
            dict(type='circle', cx=0.70 + d, cy=0.80, lx=0.25, n_index=2.5)])]
        return float(np.sum(np.asarray(
            fieldwave.rcwa(backend=0, ucell=ucell, **cfg).conv_solve().de_ri)).real)

    values = [total(d) for d in np.linspace(0.0, 0.2, 9)]
    assert np.ptp(values) < 1e-12, np.ptp(values)
