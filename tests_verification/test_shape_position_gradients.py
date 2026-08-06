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


@pytest.mark.xfail(
    strict=True,
    reason='a layer of several shapes takes a normalized sum of their fields, which no single '
           'phase translates, so its normal field is still sampled in place')
def test_a_layer_of_two_shapes_is_translation_invariant_too():
    """The limitation, stated as the test that would notice it going away.

    Moving *both* shapes by the same amount is the same physical translation as moving one, so
    this is the same zero — but the field of a multi-shape layer is ``sum(g/|g|)`` divided by
    its own norm, and that sum is not a translate of anything. It stays on the sampled path and
    keeps the clipped window: 7.1e-4 on ``sum(de_ri)``, against the 1.5e-14 a lone shape gets.

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
