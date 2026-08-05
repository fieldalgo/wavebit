"""Gradients with respect to a shape's *position* on the analytic path.

``cx`` and ``cy`` were the two shape parameters ``test_gradients`` never covered. The reason
was recorded as a gap — a finite-difference comparison that failed and got removed — and the
suspicion was sampling noise: the normal field is evaluated on a grid, so the objective
carries a sawtooth and a difference step that straddles a grid line reports the jump.

Translation settles it without finite differences. Sliding a single layer sideways changes
only the phase of each diffracted order, never its efficiency, so ``d/dcx`` and ``d/dcy`` are
**exactly zero** — a reference no discretization can argue with. Measured against it, the gap
turns out to be two separate things:

* **A bias, on rectangles, in ``cy``.** The mean gradient over thirteen positions is -1.7e-1
  where zero is the answer, and it is fourteen times the spread, so it is not noise. Its cause
  is in :func:`fieldwave._core.analytic.normal_vector_field_analytical`: the rectangle branch
  ends in ``sign(xr)`` and a comparison mask, and the derivative of ``sign`` is zero
  everywhere. So the normal field's contribution to the position gradient is not merely
  approximated, it is *absent* — and translation invariance holds by that contribution
  cancelling the permittivity one, which leaves the permittivity term standing alone.
  Ellipses and circles take ``2*xr/a**2`` instead, which is linear in ``xr``, and they do not
  show it.

* **Noise, on every shape.** Spread of 5e-3 to 8e-2 across positions, with single points as
  far out as 2.3e-1. This is the sampling effect, and it is four orders larger than the 4.2e-5
  the earlier note quotes — that figure is the *objective*'s variation, and a gradient divides
  it by a step. It is the more practical obstacle of the two: a position optimizer sees one
  point at a time, not a mean.

Both are pinned below, the first as an expected failure so that fixing it is noticed.
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

# Nine positions, spaced so no two land on the same normal-field grid line — the sample is of
# the sawtooth, not of one phase of it. Confined to 0.30-0.70 of the period so that the widest
# shape here (a circle of radius 0.25*P) stays clear of the cell edge: a shape wrapping the
# boundary is a separate question from this one, and it does move the objective (1.5e-2 at
# 0.17*P, where the circle overhangs).
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
    """First: the premise. The efficiency really is flat under translation.

    Without this the gradient tests below would be asserting zero against an objective that
    genuinely varies, and would be wrong rather than revealing.
    """
    def total(v):
        res = fieldwave.rcwa(backend=0, ucell=[dict(n_base=1.0, shapes=[_shape(kind, param, v)])],
                             **BASE).conv_solve()
        return float(np.sum(np.asarray(res.de_ri)).real)

    values = [total(x) for x in POSITIONS]
    assert np.ptp(values) < 5e-3, (kind, param, np.ptp(values))


_KNOWN_BIAS = pytest.mark.xfail(
    strict=True,
    reason='rectangle normals end in sign(), whose derivative is zero, so the normal field '
           'contributes nothing to d/dcy and the permittivity term is left uncancelled')

UNBIASED_CASES = [
    pytest.param(s, p, marks=[_KNOWN_BIAS] if (s, p) == ('rectangle', 'cy') else [], id=i)
    for (s, p), i in zip(CASES, IDS)
]


@pytest.mark.parametrize('kind,param', UNBIASED_CASES)
def test_position_gradient_is_unbiased(kind, param):
    """The mean over positions must be indistinguishable from zero.

    Compared against the spread rather than against a fixed tolerance: the noise is real and
    large, so a bound tight enough to catch the bias would fail the honest cases too. What
    separates them is that noise averages down and a bias does not — measured, the five sound
    combinations sit at 0.08 to 0.46 of one standard deviation, and ``rectangle``/``cy`` at 14.
    """
    g = _position_gradients(kind, param)
    assert abs(g.mean()) <= g.std(), (kind, param, g.mean(), g.std())


@pytest.mark.parametrize('kind,param', CASES, ids=IDS)
def test_position_gradient_noise_stays_where_it_is(kind, param):
    """And the noise itself, so that it cannot quietly grow.

    Not a statement that this level is acceptable — 8e-2 on a quantity whose true value is 0
    is not usable for optimization. It is a floor to measure any fix against, and the number
    to beat if the normal projectors ever get a closed form.
    """
    g = _position_gradients(kind, param)
    assert g.std() < 0.15, (kind, param, g.std())


def test_the_rectangle_normal_field_is_where_the_gradient_stops():
    """Straight at the mechanism, with no solver in the way.

    ``<nx**2>`` over the cell moves when the shape moves — finite differences say so for all
    three shapes. Autograd agrees for the ellipse and the circle and returns *exactly* zero for
    the rectangle, which is what ``sign()`` does: it is silent, it raises nothing, and zero is
    indistinguishable from "this term does not contribute".
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
