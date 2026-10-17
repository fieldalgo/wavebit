"""Pure CPU preflight of independent Gaussian jets, stamps and Yee sensitivities."""
import numpy as np
import pytest

from wavebit.units import C0, EPSILON0
from tests.fdtd._pulse_yee_reference import (
    BASE_DELAY, CELLS, CENTER, CUTOFF, DT, DX, EPSILON, KAPPA, MOMENT, MU, NONZERO,
    POINTS, SECANT_STEPS, SECANT_TOL, SIGMA, T, objectives, pulse_jet, reference,
    stamp_array,
)


@pytest.mark.parametrize('stamp', ['bspline', 'tent'])
def test_independent_stamp_moment_centroid_and_one_step(stamp):
    weight = stamp_array(stamp)
    assert np.min(weight) >= 0.
    np.testing.assert_allclose(np.sum(weight), 1., rtol=1e-13, atol=0)
    coordinates = [-CELLS*DX/2+(np.arange(CELLS)+(.5 if a == 1 else 0.))*DX for a in range(3)]
    z, y, x = np.meshgrid(coordinates[2], coordinates[1], coordinates[0], indexing='ij')
    for axis, grid in enumerate((x, y, z)):
        np.testing.assert_allclose(np.sum(weight*grid), CENTER[axis], rtol=0, atol=1e-13*DX)
    expected = reference(POINTS[0], stamp, 1)
    factor = -(EPSILON0*EPSILON+SIGMA*DT/2)*DX**3/DT
    for field, drive in ((expected['fields']['E'][1], expected['drive'][0]),
                         (expected['first_fields']['E'][:, 1], expected['first_drive'][:, 0]),
                         (expected['second_fields']['E'][:, :, 1], expected['second_drive'][:, :, 0])):
        np.testing.assert_allclose(np.sum(field, axis=(-3, -2, -1))*factor, drive,
                                   rtol=1e-13, atol=0)
    assert np.max(abs(expected['fields']['H'])) == 0.
    assert np.max(abs(expected['fields']['E'][[0, 2]])) == 0.


@pytest.mark.parametrize('stamp', ['bspline', 'tent'])
@pytest.mark.parametrize('steps', [17, 41])
@pytest.mark.parametrize('point', POINTS, ids=['point0', 'point1'])
def test_pulse_objectives_full_hessian_two_secants(stamp, steps, point, record_property):
    expected = reference(point, stamp, steps)
    exact = objectives(expected)
    # Independent static resolution and positive lower Gaussian-band margin.
    l, bandwidth, _, delay = point
    carrier, width = 1/(2*l*T), .08*bandwidth/T
    assert carrier-KAPPA*width > 0
    shortest_band_wavelength = C0/(carrier+KAPPA*width)
    assert shortest_band_wavelength/(np.sqrt(EPSILON*MU)*DX) > 10
    peak_time = BASE_DELAY+T*delay+KAPPA/(2*np.pi*width)
    assert 0 < peak_time < steps*DT
    assert 0 < CUTOFF < 1
    for name, (loss, gradient, hessian) in exact.items():
        assert loss > 0 and np.isfinite(gradient).all() and np.isfinite(hessian).all()
        assert np.min(abs(gradient))/loss > NONZERO, (name, 'gradient', gradient/loss)
        assert np.min(abs(hessian))/loss > NONZERO, (name, 'hessian', hessian/loss)
        np.testing.assert_array_equal(hessian, hessian.T)
        record_property(name+'_loss', loss)
        record_property(name+'_gradient', gradient.tolist())
        record_property(name+'_hessian', hessian.tolist())
        record_property(name+'_minimum_normalized_hessian', float(np.min(abs(hessian))/loss))
    maxima = []
    for h in SECANT_STEPS:
        for j in range(4):
            plus, minus = np.array(point), np.array(point)
            plus[j] += h
            minus[j] -= h
            upper, lower = objectives(reference(plus, stamp, steps)), objectives(reference(minus, stamp, steps))
            for name, (loss, gradient, hessian) in exact.items():
                secant_gradient = (upper[name][0]-lower[name][0])/(2*h)
                secant_hessian = (upper[name][1]-lower[name][1])/(2*h)
                gradient_error = abs(secant_gradient/gradient[j]-1)
                hessian_error = np.max(abs(secant_hessian/hessian[:, j]-1))
                assert gradient_error < SECANT_TOL, (name, j, h, gradient_error)
                assert hessian_error < SECANT_TOL, (name, j, h, hessian_error)
                maxima.extend((float(gradient_error), float(hessian_error)))
    record_property('maximum_secant_relative_error', max(maxima))
    record_property('physical_peak_time_seconds', peak_time)


@pytest.mark.parametrize('stamp', ['bspline', 'tent'])
def test_pulse_forcing_and_hessian_mutants_are_detectable(stamp, record_property):
    point, steps = POINTS[0], 41
    expected = reference(point, stamp, steps)
    exact = objectives(expected)
    distances = {}
    for mutant in ('frozen_peak', 'omit_carrier', 'wrong_delay_units', 'wrong_clock',
                   'omit_mixed', 'omit_second', 'stamp_squared_normalization'):
        altered = objectives(reference(point, stamp, steps, mutant))
        distances[mutant] = max(float(np.max(abs((got-want)/want)))
            for name in exact for got, want in zip(altered[name], exact[name]))
        assert distances[mutant] > 1e-4, (mutant, distances[mutant])
    record_property('mutant_relative_separations', distances)


@pytest.mark.parametrize('point', POINTS, ids=['point0', 'point1'])
def test_gaussian_sample_uses_physical_delay_and_amplitude_spectrum_width(point):
    steps = 17
    drive, _, _ = pulse_jet(point, steps)
    l, bandwidth, phase, delay = point
    tau = T/(2*np.pi*.08*bandwidth)
    shifted = (np.arange(steps)+.5)*DT-(BASE_DELAY+T*delay)-KAPPA*tau
    separate = MOMENT*np.exp(-.5*(shifted/tau)**2)*np.sin(np.pi*shifted/(T*l)+phase)
    peak = np.max(abs(separate))
    assert peak > .01*MOMENT
    assert np.max(abs(drive-separate))/peak < 2e-12
