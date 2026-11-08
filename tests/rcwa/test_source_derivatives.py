"""Derivatives through coherent-source assembly, against central finite differences.

Each derivative is taken in JAX and in PyTorch and compared with central differences of the
same scalar computed by the NumPy backend. Two steps estimate the finite-difference error,
which is asserted to sit far below the tolerance used for the comparison.
"""

from wavebit.rcwa._engine.sources import GaussianBeam as _legacy_wavebit_GaussianBeam

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest


BOX = ((-3., 3.), (-3., 3.))
#: Two layers between unequal half-spaces, so both faces reflect.
T0 = .43


def stack(backend, thickness):
    return dict(backend=backend, wavelength=1., period=(2.3, 2.7), top=1., bot=2.25,
                structure=[dict(background=4., shapes=[]), dict(background=1.7, shapes=[])],
                thickness=[thickness, .35], truncation_order=(1, 1), answers='amplitude')


def linear_and_power(result):
    """One real field component on the exit face plus the collected power: both carry the
    beam's phase at its entrance face, which is what a thickness derivative must see."""
    field = result.boundary_fields('bottom', [.13], [-.07], part='outgoing')
    return field[0, 0, 1].real + result.power.outgoing_bottom


def central_difference(scalar, x0, step):
    coarse = (scalar(x0+2*step) - scalar(x0-2*step))/(4*step)
    fine = (scalar(x0+step) - scalar(x0-step))/(2*step)
    return fine, abs(fine-coarse)


def derivative(backend, scalar, x0):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        _legacy_wavebit_enable_x64()
        return float(jax.grad(lambda x: scalar('jax', x))(jnp.asarray(x0)))
    torch = pytest.importorskip('torch')
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(scalar('torch', x), x)
    return float(gradient)


TOP_BEAM = _legacy_wavebit_GaussianBeam(side='top', waist=1.5, focus=(.1, -.2, .9),
                                        sampling=(3, 3), theta=.2, phi=.3,
                                        polarization=(.6, .8j))


def top_beam(backend, thickness):
    solver = _legacy_wavebit_rcwa(**stack(backend, thickness), excitation=TOP_BEAM)
    return linear_and_power(solver.solve(power_region=BOX))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_top_gaussian_beam_thickness_derivative_matches_finite_differences(backend):
    """The focus is fixed in global z, so the beam's phase on the top face moves with every
    layer thickness. That face height was read as a host float, the phase lost its
    dependence, and both backends agreed on a derivative of -4.18 where it is -2.07."""
    reference, error = central_difference(lambda t: top_beam('numpy', t), T0, 1e-4)
    assert error < 1e-6*abs(reference)
    np.testing.assert_allclose(derivative(backend, top_beam, T0), reference, rtol=1e-6)


def test_top_gaussian_beam_under_jit_is_refused_not_answered():
    """The coherent route groups sources eagerly. Under jit the public API refuses with a
    code, in the value and in the derivative alike, rather than return a constant phase."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit import Material, Units, rcwa
    from wavebit.errors import UnsupportedFeatureError

    def power(thickness):
        problem = rcwa.Problem(units=Units(length='um'),
            stack=rcwa.Stack(period=(2.3, 2.7), bottom=Material(2.25), layers=(
                rcwa.Layer(thickness=thickness, background=Material(4.)),
                rcwa.Layer(thickness=.35, background=Material(1.7)))),
            excitation=rcwa.Excitation(wavelength=1., sources=(rcwa.GaussianBeam(
                side='top', waist=1.5, focus=(.1, -.2, .9), sampling=(3, 3), theta=.2, phi=.3,
                polarization=(.6, .8j)),)),
            outputs=rcwa.Outputs(diffraction=False, power_region=BOX))
        result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 1)), backend='jax')
        return result.outputs['bottom'].outgoing[0]

    with jax.enable_x64(True):
        for function in (power, jax.grad(power)):
            with pytest.raises(UnsupportedFeatureError) as caught:
                jax.jit(function)(jnp.asarray(T0))
            assert caught.value.code == 'prepare.concrete'
