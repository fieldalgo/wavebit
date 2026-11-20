"""Explicit heights at a stack's faces: a face written as a decimal sum is that face.

A face is a sum of thicknesses, and the caller's sum need not round as the stack's does:
``0.1 + 0.2 + 0.3`` is ``0.6000000000000001`` while the stack ``(0.3, 0.2, 0.1)`` sums to
``0.6``. The reference for each check is the field at the face as the stack computes it (its
exact float), and the limit just inside the documented layer: the upper one at an internal
face, the finite layer at an outer one.
"""
import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import ValidationError

EPSILONS = (4.2, 6.0, 2.9)


def run(thickness, side):
    layers = tuple(rcwa.Layer(thickness=d, background=Material(e))
                   for d, e in zip(thickness, EPSILONS))
    problem = rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(period=(.7, .8), top=Material(2.25), bottom=Material(3.24), layers=layers),
        # p light at an oblique angle, so Ez jumps at every material face.
        excitation=rcwa.Excitation(wavelength=.8, sources=(rcwa.PlaneWave(
            side=side, polarization='p', theta=.29, phi=.4),)),
        outputs=rcwa.Outputs(fields=True))
    return rcwa.run(problem, numerics=rcwa.Numerics())


def field(result, z):
    return result.field(x=[0.], y=[0.], z=list(z)).values[0, :, 0, 0, :]


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_top_face_written_as_a_decimal_sum_is_the_top_face(side):
    result = run((.3, .2, .1), side)
    face = .3 + .2 + .1                      # the stack's own sum
    written = .1 + .2 + .3
    assert written != face                   # the case: one rounding apart
    at_face, = field(result, [face])
    np.testing.assert_array_equal(field(result, [written])[0], at_face)
    # Continuity just inside the top layer.
    inside, = field(result, [face - 1e-9])
    np.testing.assert_allclose(at_face, inside, rtol=1e-6, atol=1e-9 * np.abs(at_face).max())


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_an_internal_face_off_by_rounding_selects_the_upper_layer(side):
    result = run((.1, .2, .4), side)
    face = .1 + .2                           # 0.30000000000000004, the second face
    written = .3
    assert written < face
    at_face, above, below = field(result, [face, face + 1e-9, face - 1e-9])
    got, = field(result, [written])
    np.testing.assert_array_equal(got, at_face)
    np.testing.assert_allclose(got, above, rtol=1e-6, atol=1e-9 * np.abs(above).max())
    # The lower layer's limit is a different Ez: the side is a real choice here.
    assert abs(got[2] - below[2]) > 1e-2 * abs(got[2])


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_the_bottom_face_takes_a_rounding_below_zero(side):
    result = run((.3, .2, .1), side)
    np.testing.assert_array_equal(field(result, [-1e-17])[0], field(result, [0.])[0])


def test_heights_off_the_faces_are_returned_bit_for_bit_in_the_requested_order():
    result = run((.3, .2, .1), 'top')
    z = [.55, .05, .31, .5, .3, .6]
    together = field(result, z)
    for k, height in enumerate(z):
        np.testing.assert_array_equal(together[k], field(result, [height])[0])


def test_a_height_really_outside_names_the_range_and_the_faces():
    result = run((.3, .2, .1), 'top')
    for z in ([.6 + 1e-6], [-1e-6], [.2, .7]):
        with pytest.raises(ValidationError) as caught:
            result.field(x=[0.], y=[0.], z=z)
        message = str(caught.value)
        assert caught.value.code == 'field.coordinates'
        assert caught.value.path == 'z'
        assert 'from 0 (its bottom face) to 0.6 (its top face' in message
        assert "faces are at 0.0, 0.3, 0.5, 0.6." in message
        assert 'boundary_fields' in message


def test_a_snapped_height_keeps_its_own_z_derivative():
    torch = pytest.importorskip('torch')
    layers = tuple(rcwa.Layer(thickness=d, background=Material(e))
                   for d, e in zip((.3, .2, .1), EPSILONS))
    problem = rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(period=(.7, .8), top=Material(2.25), bottom=Material(3.24), layers=layers),
        excitation=rcwa.Excitation(wavelength=.8, sources=(rcwa.PlaneWave(
            side='top', polarization='p', theta=.29, phi=.4),)),
        outputs=rcwa.Outputs(fields=True))
    result = rcwa.run(problem, numerics=rcwa.Numerics(), backend='torch')

    def value_and_slope(height):
        z = torch.tensor([height], dtype=torch.float64, requires_grad=True)
        value = result.field(x=[0.], y=[0.], z=z).values[0, 0, 0, 0, 1].real
        value.backward()
        return value.item(), z.grad.item()

    assert value_and_slope(.1 + .2 + .3) == value_and_slope(.6)


# ------------------------------------------------------------------ under a trace

def _jax_problem(thickness, epsilons, side='top'):
    layers = tuple(rcwa.Layer(thickness=d, background=Material(e)) for d, e in zip(thickness, epsilons))
    return rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(period=(.7, .8), top=Material(2.25), bottom=Material(3.24), layers=layers),
        excitation=rcwa.Excitation(wavelength=.8, sources=(rcwa.PlaneWave(
            side=side, polarization='p', theta=.29, phi=.4),)),
        outputs=rcwa.Outputs(fields=True))


def _heights_by_rounding(thickness):
    """Each internal face as the stack sums it, two ulps below it, and the literal one rounding below."""
    faces = np.cumsum(thickness)[:-1]
    return np.concatenate([faces, faces * (1 - 2 * np.finfo(float).eps), [.3]])


@pytest.mark.parametrize('traced', ['permittivity', 'heights'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_face_off_by_rounding_selects_the_upper_layer_under_jit(side, traced):
    """Under ``jax.jit`` a height a rounding below a face reads the upper layer, as eagerly.

    The stack's thicknesses are tracers inside the trace, constants or not, so the faces have no
    value to read on the host, and the snap onto a face used to be skipped there: ``z = 0.3``, one
    rounding below the face ``0.1 + 0.2``, and every height two ulps below a face read the lower
    layer under ``jit`` and the upper one on NumPy, PyTorch and eager JAX -- ``E_z`` up to 0.077
    apart on an ``E_z`` of 0.17 (2026-11-20). The snap now runs on the arrays.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    thickness, epsilons = (.1, .2, .4), EPSILONS
    z = _heights_by_rounding(thickness)
    expected = np.asarray(rcwa.run(_jax_problem(thickness, epsilons, side), numerics=rcwa.Numerics())
                          .field(x=[0.], y=[0.], z=list(z)).values)
    with jax.enable_x64(True):
        prepared = rcwa.prepare(_jax_problem(thickness, epsilons, side), backend='jax',
                                numerics=rcwa.Numerics())
        if traced == 'permittivity':
            def values(e):
                problem = _jax_problem(thickness, (EPSILONS[0], e, EPSILONS[2]), side)
                return prepared.run(problem).field(x=[0.], y=[0.], z=z).values
            got = np.asarray(jax.jit(values)(jnp.asarray(EPSILONS[1])))
        else:
            result = prepared.run(_jax_problem(thickness, epsilons, side))
            got = np.asarray(jax.jit(lambda h: result.field(x=[0.], y=[0.], z=h).values)(
                jnp.asarray(z)))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)        # measured 8.1e-16


def test_a_snapped_height_keeps_its_own_thickness_derivative_under_jit():
    """``jit(grad)`` by a thickness, at heights on and a rounding below the faces, is eager
    ``jax.grad``'s: the snap's shift is a constant to differentiation under the trace as the
    host read makes it eagerly, so a height does not follow the face it was moved onto."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    thickness = (.1, .2, .4)
    z = _heights_by_rounding(thickness)
    with jax.enable_x64(True):
        prepared = rcwa.prepare(_jax_problem(thickness, EPSILONS), backend='jax', numerics=rcwa.Numerics())

        def loss(d):
            problem = _jax_problem((thickness[0], d, thickness[2]), EPSILONS)
            return jnp.sum(jnp.abs(prepared.run(problem).field(x=[0.], y=[0.], z=z).values) ** 2)
        eager = float(jax.grad(loss)(jnp.asarray(thickness[1])))
        traced = float(jax.jit(jax.grad(loss))(jnp.asarray(thickness[1])))
    assert abs(eager) > 1e-3
    assert traced == pytest.approx(eager, rel=1e-12)                      # measured 3.0e-16
