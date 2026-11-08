"""Derivatives through coherent-source assembly, against central finite differences.

Each derivative is taken in JAX and in PyTorch and compared with central differences of the
same scalar computed by the NumPy backend. Two steps estimate the finite-difference error,
which is asserted to sit far below the tolerance used for the comparison.
"""

from wavebit.rcwa._engine.sources import ElectricDipole as _legacy_wavebit_ElectricDipole

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


DIPOLE = _legacy_wavebit_ElectricDipole(position=(.1, -.07, .2), moment=(1e-29, .2e-29j, .3e-29),
                                        bloch_k=(.2, .1))


def dipole(layer):
    """The dipole sits in layer 0 at z = 0.2; vary that host layer or the one above it."""
    def scalar(backend, thickness):
        options = stack(backend, T0)
        options['thickness'] = [thickness, .35] if layer == 0 else [T0, thickness]
        result = _legacy_wavebit_rcwa(**options, excitation=DIPOLE, length_unit=1e-6).solve(
            power_region='cell')
        field = result.boundary_fields('top', [.13], [-.07], part='outgoing')
        return field[0, 0, 0].real + result.power.outgoing_top + result.power.outgoing_bottom
    return scalar


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('layer,thickness', [(0, T0), (1, .35)])
def test_dipole_run_thickness_derivative_matches_finite_differences(backend, layer, thickness):
    """Splitting a host layer at the source plane rebuilt every segment thickness from host
    edges. Under jax.grad each derivative came back exactly 0 (FD: -10.92 for the host layer,
    -12.76 for the one above); PyTorch had no graph at all."""
    scalar = dipole(layer)
    reference, error = central_difference(lambda t: scalar('numpy', t), thickness, 1e-5)
    assert error < 1e-7*abs(reference)
    np.testing.assert_allclose(derivative(backend, scalar, thickness), reference, rtol=1e-6)


def jones_beam(backend, a_p):
    beam = _legacy_wavebit_GaussianBeam(side='top', waist=1.5, focus=(.1, -.2, .9),
                                        sampling=(3, 3), theta=.2, phi=.3, polarization=(.6, a_p))
    solver = _legacy_wavebit_rcwa(**stack(backend, T0), excitation=beam)
    return linear_and_power(solver.solve(power_region=BOX))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_gaussian_beam_polarization_derivative_matches_finite_differences(backend):
    """The Jones pair was read to host complex numbers before the angular spectrum was built:
    0 under jax.grad (FD: 0.368), no graph in PyTorch."""
    reference, error = central_difference(lambda a: jones_beam('numpy', a), .8, 1e-4)
    assert error < 1e-7*abs(reference)
    np.testing.assert_allclose(derivative(backend, jones_beam, .8), reference, rtol=1e-6)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('name', ['theta', 'wavelength', 'top'])
def test_gaussian_beam_refuses_derivatives_its_host_quadrature_cannot_carry(backend, name):
    """Direction, wavelength and lit-medium derivatives would be partial: the quadrature
    directions are built on the host. jax.grad returned 0 for d/dtheta (FD: 0.094) and both
    backends 2.79 for d/dwavelength (FD: 3.05). They are refused, by code, instead."""
    from wavebit import Material, Units, rcwa
    from wavebit.errors import UnsupportedFeatureError

    def power(backend, value):
        beam = dict(side='top', waist=1.5, sampling=(3, 3), theta=.2, polarization=(.6, .8j))
        wavelength, top = 1., Material(1.)
        if name == 'theta':
            beam['theta'] = value
        elif name == 'wavelength':
            wavelength = value
        else:
            top = Material(value)
        problem = rcwa.Problem(units=Units(length='um'),
            stack=rcwa.Stack(period=(2.3, 2.7), top=top, bottom=Material(2.25),
                             layers=(rcwa.Layer(thickness=T0, background=Material(4.)),)),
            excitation=rcwa.Excitation(wavelength=wavelength,
                                       sources=(rcwa.GaussianBeam(**beam),)),
            outputs=rcwa.Outputs(diffraction=False, power_region=BOX))
        result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 1)), backend=backend)
        return result.outputs['bottom'].outgoing[0]

    start = dict(theta=.2, wavelength=1., top=1.)[name]
    with pytest.raises(UnsupportedFeatureError, match='GaussianBeam builds its angular') as caught:
        derivative(backend, power, start)
    assert caught.value.code == 'rcwa.unsupported'


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('internal', [False, True])
def test_interior_field_thickness_derivative_matches_finite_differences(backend, internal):
    """A field sampled in layer 1 lies at a fixed global z; its depth below that layer's lower
    face moves with the thickness of layer 0. Reconstruction measured it from host edges, so
    both backends returned 1.62 for FD -6.05 (two plane waves) and -4.61 for FD -0.82 (dipole)."""
    from wavebit.rcwa._engine.sources import PlaneWave
    excitation = DIPOLE if internal else [PlaneWave(theta=.2), PlaneWave(side='bottom', theta=-.1)]

    def scalar(backend, thickness):
        options = stack(backend, thickness) | dict(answers='field')
        extra = dict(length_unit=1e-6) if internal else {}
        result = _legacy_wavebit_rcwa(**options, excitation=excitation, **extra).solve(
            power_region='infinite')
        return result.reconstruct_field([.13], [-.07], [.6])[0, 0, 0, 1].real

    reference, error = central_difference(lambda t: scalar('numpy', t), T0, 1e-5)
    assert error < 1e-7*abs(reference)
    np.testing.assert_allclose(derivative(backend, scalar, T0), reference, rtol=1e-6)
