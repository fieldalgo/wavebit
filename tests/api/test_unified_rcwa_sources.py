"""Physical SI power, including cross-sector interference on a finite aperture."""
import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.units import Z0


@pytest.mark.parametrize('unit,scale', [('m', 1e-6), ('um', 1.), ('nm', 1e3)])
def test_two_bloch_sectors_match_the_analytic_finite_aperture_power(unit, scale):
    theta = .23
    # The s frames of the +kx and -kx waves point along +y and -y.
    # Thus Ey = exp(i*kx*x) - i*exp(-i*kx*x), with Hx = cos(theta)*Ey/Z0.
    # Integrating their interference over this off-centre aperture is not the
    # sum of the two separately averaged powers.
    problem = rcwa.Problem(
        units=Units(length=unit),
        stack=rcwa.Stack(period=(scale, scale), layers=(
            rcwa.Layer(thickness=.2*scale, background=Material(epsilon=1.)),)),
        excitation=rcwa.Excitation(wavelength=.8*scale, sources=(
            rcwa.PlaneWave(theta=theta, amplitude=1.),
            rcwa.PlaneWave(theta=-theta, amplitude=1j))),
        outputs=rcwa.Outputs(diffraction=False, fields=True,
            power_region=((-0.2*scale, .35*scale), (-.4*scale, .4*scale))))
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(1, 0)))
    kx_per_m = 2*np.pi/(.8e-6)*np.sin(theta)
    width_m, centre_m, height_m = .55e-6, .075e-6, .8e-6
    mean_electric_squared = 2 - 2*np.sin(2*kx_per_m*centre_m)*np.sinc(kx_per_m*width_m/np.pi)
    watts = .5*np.cos(theta)/Z0*mean_electric_squared*width_m*height_m
    np.testing.assert_allclose(result.outputs['top'].incoming, [watts], rtol=1e-12, atol=0)
    np.testing.assert_allclose(result.outputs['bottom'].outgoing, [watts], rtol=1e-12, atol=0)
    assert result.outputs['top'].units == 'W'
    # Independently evaluate the physical field at the aperture centre.
    fields = result.boundary_fields(side='top', x=[.075*scale], y=[0.], part='incoming').values
    ey = np.exp(1j*kx_per_m*centre_m)-1j*np.exp(-1j*kx_per_m*centre_m)
    np.testing.assert_allclose(fields[0, 0, 0, 1], ey, rtol=1e-12, atol=0)
    np.testing.assert_allclose(fields[0, 0, 0, 3], np.cos(theta)*ey/Z0, rtol=1e-12, atol=0)


def test_problem_owns_python_containers_but_keeps_backend_array_leaves():
    center = [.1, -.1]
    polarization = [1., 1j]
    wavelengths = [.8, .9]
    raster = [[[2., 3.], [4., 5.]]]
    basis = [[1., 0.], [0., 1.]]
    rectangle = rcwa.Rectangle(size=(.2, .1), center=center)
    beam = rcwa.GaussianBeam(waist=1., polarization=polarization)
    orders = rcwa.OrderExcitation(orders=(((0, 0), 1.),), polarization=polarization)
    excitation = rcwa.Excitation(wavelength=wavelengths, sources=(beam, orders))
    layer = rcwa.Layer(thickness=.2, epsilon=raster)
    stack = rcwa.Stack(lattice=basis)
    center[0] = 10.
    polarization[0] = 0.
    wavelengths[0] = 100.
    raster[0][0][0] = 9.
    basis[0][0] = 7.
    assert rectangle.center == (.1, -.1)
    assert beam.polarization == orders.polarization == (1., 1j)
    assert excitation.wavelength == (.8, .9)
    assert layer.epsilon[0][0][0] == 2.
    assert stack.lattice[0][0] == 1.
    array = np.ones((2, 2))
    assert rcwa.Layer(thickness=.2, epsilon=array).epsilon is array
