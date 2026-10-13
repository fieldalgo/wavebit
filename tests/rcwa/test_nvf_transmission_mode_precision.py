"""Independent complete characteristic, sector transfer and reflection checks."""
from decimal import Decimal as Z, localcontext
import numpy as np
import pytest
from wavebit.rcwa._engine.conforming.pair_transmission_modes import TransmissionMode
from wavebit.rcwa._engine.conforming.pair_transmission_common_precision import local_mode
from wavebit.rcwa._engine.conforming.pair_corner_precision import D
from tests.rcwa.test_nvf_contact_fourier_precision import add, sub, mul, div, times, scalar, exponential, decimal_pi

pytestmark = pytest.mark.research


REFERENCES = ((12.25, -10+1j, 1), (4, 9, 1), (1, 1+1e-12, 1),
              (1-1e-12+5e-13j, 1, 1), (-10+1j, 1, 1), (12.25, 1, 1),
              (4+.1j, 9+.2j, 1+.02j), (12.25j, -1-10j, 1j))


def exact(value):
    return add(scalar(value.hi), scalar(value.lo))


def norm(value):
    return max(abs(value[0]), abs(value[1]))


@pytest.mark.parametrize('coefficients', REFERENCES)
@pytest.mark.parametrize('branch', range(4))
def test_complete_parameters_satisfy_independent_decimal_transmission(coefficients, branch):
    mode = TransmissionMode(coefficients, branch=branch)
    with localcontext() as context:
        context.prec = 100
        pi, one, ii = decimal_pi(), (Z(1), Z(0)), (Z(0), Z(1))
        lam = exact(mode.complete_exponent)
        cs = [div(scalar(v), scalar(coefficients[-1])) for v in coefficients]
        for actual, target in zip(mode.complete_coefficients, cs):
            assert norm(sub(exact(actual), target)) < Z('3e-28')*max(norm(target), 1)
        a, b, c = cs
        first = add(div(a, b), div(b, a))
        second = add(div(add(a, b), c), mul(c, add(div(one, a), div(one, b))))
        characteristic = div(sub((Z(-4), Z(0)), second), add((Z(2), Z(0)), add(first, second)))
        z = mul(ii, times(lam, pi))
        cosine = times(add(exponential(z), exponential(times(z, -1))), Z('.5'))
        assert norm(sub(cosine, characteristic)) < Z('3e-28')*max(norm(characteristic), 1)
        state = list(map(exact, mode.complete_states[0]))
        for index, (coefficient, angle) in enumerate(zip(cs, (pi/2, pi/2, pi))):
            z = mul(ii, times(lam, angle))
            positive, negative = exponential(z), exponential(times(z, -1))
            cosine = times(add(positive, negative), Z('.5'))
            sine = div(sub(positive, negative), (Z(0), Z(2)))
            p, q = state
            state = [add(mul(cosine, p), div(mul(sine, q), coefficient)),
                     sub(mul(cosine, q), mul(mul(coefficient, sine), p))]
            for actual, target in zip(mode.complete_states[index+1], state):
                assert norm(sub(exact(actual), target)) < Z('3e-28')*max(norm(target), 1)
        for actual, target in zip(mode.complete_states[0], state):
            assert norm(sub(exact(actual), target)) < Z('3e-28')*max(norm(target), 1)


@pytest.mark.parametrize('coefficients', REFERENCES)
@pytest.mark.parametrize('branch', range(4))
def test_complete_mode_and_gradient_span_is_closed_under_reflection(coefficients, branch):
    a, b, c = coefficients
    bottom = TransmissionMode((b, a, c), branch=branch)
    top = TransmissionMode((a, b, c), branch=branch, start=np.pi)
    factor = top.complete_states[0][0]/bottom.complete_states[2][0]
    # This scalar aligns two independently generated bases of the same
    # one-dimensional complex solution space. It never changes a field.
    intervals = ((0, np.pi/2), (np.pi/2, np.pi), (np.pi, 2*np.pi))
    for sector, (left, right) in enumerate(intervals):
        angle = np.linspace(left+.13*(right-left), right-.13*(right-left), 7)
        x, y = D(.17*np.cos(angle)), D(.17*np.sin(angle))
        lower = local_mode(bottom, x, y, sector, float(angle.mean()))
        upper_sector = 1-sector if sector < 2 else 2
        reflected = (np.arctan2(-y.hi, x.hi)-np.pi) % (2*np.pi)
        upper = local_mode(top, x, -y, upper_sector, float(reflected.mean()))
        for actual, expected, sign in zip(upper, lower, (1, 1, -1)):
            residual = actual-factor*expected*sign
            scale = max(np.max(abs(actual.hi+actual.lo)),
                        np.max(abs((factor*expected).hi+(factor*expected).lo)))
            assert np.max(abs(residual.hi+residual.lo)) < 3e-28*scale
