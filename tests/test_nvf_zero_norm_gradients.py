"""Gradients where the analytic normal field has no direction to report.

:func:`wavebit._core.parametric.normal_vector_field_analytical` normalizes twice — once per
shape, once over the combination — and the norm reaches exactly zero on perfectly ordinary
geometry: **a shape's own centre**, where its level-set gradient ``g`` vanishes.

A second zero used to sit beside it — the locus between two symmetrically placed shapes, where
their unit normals opposed and cancelled. That one belonged to the old *sum of unit vectors*;
the distance-product combination that replaced it on 2026-08-17 cannot cancel, and
``test_a_zero_norm_configuration_is_actually_reached`` records that it is no longer reachable.
The guard below is unchanged and still needed: the per-shape zero is the one the NaN came
through, and it is reachable with a single shape.

Zero was already guarded, but on the *output* of the root: ``n = (a*a + b*b)**0.5`` followed by
``where(n == 0, 1, n)``. That fixes the forward value and nothing else. Autograd still
differentiates the branch ``where`` discarded, ``d/da`` of that root at ``a == b == 0`` is
infinite, and ``where`` contributes ``0 * inf`` — NaN, which then spreads through the whole
gradient. The forward pass stays finite throughout, so nothing raises and nothing warns; the
only symptom is a NaN arriving at the caller. The guard now replaces the *radicand*, so the
root is only ever taken at 1, where the derivative is finite.

The same shape appeared a second time in :func:`wavebit._core.parametric.shape_ft`: the circle
branch took a bare ``**0.5`` of ``u**2 + v**2`` and handed it to ``jinc``, whose own ``where``
supplied the ``0``. The ellipse branch beside it was already guarded, which is why
``d/d(period)`` used to come back NaN on a circle and finite on an ellipse — the same layer,
the same DC point, two different answers.

Every assertion below is only that the gradient is a *number*. What that number is worth is a
separate question, and :mod:`test_shape_position_gradients` is where it is asked: the
cancellation locus is a genuine singularity of the normalized field (the unit vector spins
arbitrarily fast as the shapes slide through it), so the surviving gradient is finite but
large — 9e9 on the two-circle case below, against 1e-3 half a grid cell away. That is the
sampling problem those tests already measure, not this one.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.lattice import as_lattice

pytest.importorskip('jax')
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
wavebit.enable_x64()

from wavebit._core.parametric import (                         # noqa: E402
    assemble_eps_coeffs, normal_vector_field_analytical)

P = 0.77
BASE = dict(period=[P, P], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
            wavelength=0.777, top=1.0, bot=2.25, truncation_order=[4, 4])

NORMAL_GRID = 1024                                             # the solver's default


def _twin_circles(d):
    """Two equal circles either side of the cell centre, translated together by ``d``.

    At ``d == 0`` they sit symmetrically about the centre, so on the diagonal between them the
    two unit normals are exactly opposed and their sum is exactly zero.
    """
    return [dict(type='circle', cx=-0.1 + d, cy=-0.1, lx=0.20 * P, material=9.0),
            dict(type='circle', cx=0.1 + d, cy=0.1, lx=0.20 * P, material=9.0)]


def _grid_point(i, n=NORMAL_GRID):
    """The ``i``-th normal-field sample, which is where a centre has to land to give ``g == 0``.

    Returned as a shape centre is stated -- from the middle of the cell -- so that it can be
    handed straight to ``cx`` / ``cy``. The samples themselves are laid out from the corner.
    """
    return (i + 0.5) * P / n - 0.5 * P


def test_a_zero_norm_configuration_is_actually_reached():
    """First: that the case exists at all, so the tests below cannot pass by missing it.

    **One of the two zeros this file was written for is gone**, and the other is what this now
    guards. The header describes a field that normalized twice -- once per shape, once over the
    sum of unit vectors -- and the second normalization reached zero wherever two shapes' unit
    normals opposed exactly. Since 2026-08-17 the shapes are combined by distance-product
    weights instead of summed as unit vectors, so that cancellation has no way to occur:
    ``_twin_circles(0.0)``, built for exactly this, now yields no zeros at all.

    The remaining zero is the one inside the round branch: a round shape's own centre, where
    its level-set gradient ``g`` vanishes and no weighting can put a direction back. It is
    reachable with a single circle or ellipse whose centre lands on a sample, which is what this
    asserts -- and it keeps the guard live, because that is the zero the NaN came through. Note
    it takes a *lone* shape: with a second shape in the layer the blend contributes a direction
    at the first shape's centre, so the zero closes.

    **A rectangle closed its own zero on 2026-09-06.** Its centre used to be the one point where
    the four face weights balanced to ``(0, 0)``; the piece field's tie rule reads an axis there
    instead (the pair that did not cancel, or ``(1, 0)`` when both did), so a lone box has no
    null sample at any size or position. Asserted as ``== 0`` so that a return of the null --
    a null projector at that pixel, not a harmless one -- is noticed.
    """
    def zeros_for(shapes, n=NORMAL_GRID):
        normal_x, normal_y = normal_vector_field_analytical(jnp, (n, n), shapes, jnp.float64,
                                                            as_lattice(period=(P, P)))
        return int(np.count_nonzero(np.asarray(normal_x) ** 2 + np.asarray(normal_y) ** 2 == 0))

    n = 256
    c = _grid_point(100, n)
    for kind in ('circle', 'ellipse'):
        centred = dict(type=kind, cx=c, cy=c, lx=0.20 * P, ly=0.10 * P, material=9.0)
        assert zeros_for([centred], n) > 0, kind
    box = dict(type='rectangle', cx=c, cy=c, lx=0.20 * P, ly=0.10 * P, material=9.0)
    assert zeros_for([box], n) == 0

    # Half a grid cell away there are none, which is why the NaN read as intermittent.
    off = dict(type='circle', cx=c + 0.5 * P / n, cy=c, lx=0.20 * P, material=9.0)
    assert zeros_for([off], n) == 0

    # And the cancellation the sum used to produce is simply not reachable any more.
    assert zeros_for(_twin_circles(0.0)) == 0


def test_symmetric_two_shape_layer_has_a_finite_position_gradient():
    """The reported case, through the public API: two shapes, one solve, one ``jax.grad``.

    The freeze is not incidental to the reproduction. A traced layer of more than one shape
    cannot be routed -- whether the shapes overlap decides which builder it takes, and a tracer
    cannot answer that -- so ``freeze_geometry`` is how any two-shape gradient is taken. These
    circles stay disjoint across the window, so one freeze covers it.
    """
    frozen = wavebit.rcwa(backend='numpy', structure=[dict(background=1.0, shapes=_twin_circles(0.0))],
                          **BASE).freeze_geometry()

    def objective(d):
        res = wavebit.rcwa(backend='jax', structure=[dict(background=1.0, shapes=_twin_circles(d))],
                             **BASE).solve(geometry=frozen)
        return jnp.real(jnp.sum(res.R))

    at_zero = float(jax.grad(objective)(jnp.asarray(0.0)))
    assert np.isfinite(at_zero), at_zero

    # And the forward value was finite all along, which is what made this silent.
    assert np.isfinite(float(objective(jnp.asarray(0.0))))


@pytest.mark.parametrize('kind', ['circle', 'ellipse', 'rectangle'])
def test_shape_centred_on_a_grid_point_has_a_finite_gradient(kind):
    """The other zero, inside the loop: one shape, its own centre, no cancellation involved.

    Reachable with a single shape, so it does not need the two-shape geometry above — it needs
    only a centre that lands on a sample, which ``_grid_point`` arranges. ``rectangle`` has no
    zero to reach any more (its centre reads an axis since 2026-09-06, see the test above) and
    is kept as the control that a finite gradient is what a field without the zero gives too.
    """
    n = 256
    c = _grid_point(100, n)

    def mean_nx2(dy):
        shape = dict(type=kind, cx=c, cy=c + dy, lx=0.20 * P, ly=0.10 * P, material=9.0)
        normal_x, _ = normal_vector_field_analytical(jnp, (n, n), [shape], jnp.float64,
                                                      as_lattice(period=(P, P)))
        return jnp.mean(normal_x ** 2)

    g = float(jax.grad(mean_nx2)(jnp.asarray(0.0)))
    assert np.isfinite(g), (kind, g)


@pytest.mark.parametrize('kind,extra', [('circle', {}), ('ellipse', dict(ly=0.10 * P))],
                         ids=['circle', 'ellipse'])
def test_dc_coefficient_gradient_in_period_is_finite(kind, extra):
    """And the third zero: ``shape_ft``'s DC point, where the radial frequency is zero.

    ``period`` is the parameter that reaches it — it scales ``u`` and ``v``, so the radicand is
    differentiated rather than constant. The ellipse was already guarded and the circle was
    not, so this pins them to the same answer.
    """
    def total(px):
        shape = dict(type=kind, cx=0.0, cy=0.0, lx=0.20 * P, material=9.0, **extra)
        eps_c, _ = assemble_eps_coeffs(jnp, [shape], 1.0, 2, 2, as_lattice(period=(px, px)), jnp.complex128)
        return jnp.real(jnp.sum(eps_c))

    g = float(jax.grad(total)(jnp.asarray(P)))
    assert np.isfinite(g), (kind, g)
