"""PyTorch derivatives through a ``Rectangle.angle`` on a layer that holds other regions.

On the default NVF route a layer of several shapes builds its normal field from complex
material slots (the seam contrast carries a complex phase), while the tie that orients a box's
own centre picks between *real* axes built from ``cos(angle)`` and ``sin(angle)``. ``where``
promoted the forward value exactly, but PyTorch's backward handed the complex cotangent to the
real branch and raised ``RuntimeError: Expected isFloatingType(grad.scalar_type()) ||
(input_is_complex == grad_is_complex)`` -- for any angle under autograd once a second region
shared the layer. A rectangle alone (real field) and eager ``jax.grad`` (which promotes and
takes the real part) were never affected.

References are independent of the code under test: central differences of the NumPy forward
at ``h = 1e-5``, where the difference has settled (across ``h = 1e-4, 1e-5, 1e-6`` it moves by
under 1e-8 relative on the box-only layers and by up to 1.0e-6 beside the circle, whose
sampled arc adds the most ripple), and eager ``jax.grad``, a second autodiff of the same
forward.
"""

import numpy as np
import pytest

from wavebit import Material, rcwa

torch = pytest.importorskip('torch')

NUMERICS = rcwa.Numerics(orders=(3, 3), shape_nvf_grid_size=64)
#: (cx, cy, lx, ly, angle) of the rotated rectangle.
POINT = (.05, -.02, .4, .25, .3)
OTHERS = {
    'alone': (),
    'two_rectangles': (('rectangle', (.15, .2), (.33, .3)),),
    'rectangle_circle': (('circle', .1, (-.3, .3)),),
    'three_regions': (('rectangle', (.15, .2), (.33, .3)), ('circle', .1, (-.3, .3))),
}
#: Cases painted into disjoint polygons first (``layer_needs_resolving``): a different builder.
PAINTED = {
    'near': (('rectangle', (.15, .2), (.33, .02)),),
    'overlapping': (('rectangle', (.2, .2), (.2, .05)),),
}
STEP = 1e-5


def problem(values, others):
    cx, cy, lx, ly, angle = values
    regions = [rcwa.Region(geometry=rcwa.Rectangle(size=(lx, ly), center=(cx, cy), angle=angle),
                           material=Material(6.25))]
    for kind, size, center in others:
        geometry = (rcwa.Rectangle(size=size, center=center) if kind == 'rectangle'
                    else rcwa.Circle(radius=size, center=center))
        regions.append(rcwa.Region(geometry=geometry, material=Material(3.)))
    layer = rcwa.Layer(thickness=.4, background=Material(1.), regions=tuple(regions))
    return rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=(layer,), bottom=Material(2.25)),
                        excitation=rcwa.Excitation(wavelength=1.3, sources=(
                            rcwa.PlaneWave(theta=.35, phi=.6, polarization='s'),)))


def total_r(values, others):
    return float(np.sum(np.asarray(rcwa.run(problem(values, others), numerics=NUMERICS).R)))


def central_difference(others, index):
    plus, minus = list(POINT), list(POINT)
    plus[index] += STEP
    minus[index] -= STEP
    return (total_r(plus, others) - total_r(minus, others)) / (2 * STEP)


def torch_gradient(others, indices):
    leaves = [torch.tensor(v, dtype=torch.float64, requires_grad=i in indices)
              for i, v in enumerate(POINT)]
    total = rcwa.run(problem(leaves, others), numerics=NUMERICS, backend='torch').R.sum()
    total.backward()
    return float(total.detach()), np.array([float(leaves[i].grad) for i in indices])


@pytest.mark.parametrize('case', list(OTHERS))
def test_torch_angle_derivative_matches_central_differences(case):
    value, gradient = torch_gradient(OTHERS[case], [4])
    assert value == pytest.approx(total_r(POINT, OTHERS[case]), rel=1e-12, abs=1e-14)
    reference = central_difference(OTHERS[case], 4)
    assert abs(reference) > 1e-3
    np.testing.assert_allclose(gradient, [reference], rtol=2e-6, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize('case', list(PAINTED))
def test_torch_angle_derivative_on_a_painted_layer_matches_central_differences(case):
    value, gradient = torch_gradient(PAINTED[case], [4])
    assert value == pytest.approx(total_r(POINT, PAINTED[case]), rel=1e-12, abs=1e-14)
    np.testing.assert_allclose(gradient, [central_difference(PAINTED[case], 4)],
                               rtol=2e-6, atol=0)


def test_torch_angle_with_centre_and_size_matches_central_differences():
    """Every rectangle field under autograd at once, beside a second rectangle."""
    others = OTHERS['two_rectangles']
    _, gradient = torch_gradient(others, [0, 1, 2, 3, 4])
    reference = [central_difference(others, i) for i in range(5)]
    # The centre and size differences carry more sampling ripple than the angle's (they move
    # by up to 7e-6 between h = 1e-4 and 1e-5), hence the absolute floor.
    np.testing.assert_allclose(gradient, reference, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize('case', ['two_rectangles', 'rectangle_circle'])
def test_torch_and_eager_jax_angle_derivatives_agree(case):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    with jax.enable_x64(True):
        def loss(q):
            values = list(POINT[:2]) + [q[0], q[1], q[2]]
            return jnp.sum(rcwa.run(problem(values, OTHERS[case]), numerics=NUMERICS,
                                    backend='jax').R)
        expected = np.asarray(jax.grad(loss)(jnp.asarray(POINT[2:])))
    _, gradient = torch_gradient(OTHERS[case], [2, 3, 4])
    np.testing.assert_allclose(gradient, expected, rtol=1e-9, atol=1e-13)
