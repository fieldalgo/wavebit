"""Independent Decimal complex exponentials across radial and angular scales."""
from decimal import Decimal as Z, localcontext

import numpy as np
import pytest

from wavebit._core.conforming.pair_complex_precision import C
from wavebit._core.conforming.pair_complex_exponential import exponential
from test_nvf_transmission_common_precision import exponential as decimal_exponential


def test_complete_transmission_exponential_matches_decimal_cartesian_series():
    real = np.array([-500., -100., -4., -.125, 0., .125, 4., 100., 500.])[:, None]
    angle = np.array([-700., -32*np.pi, -np.pi/2, -.25, 0., np.pi/2,
                      np.nextafter(np.pi/2, np.inf), 12*np.pi, 700.])[None, :]
    hi = real+1j*angle
    low = np.random.default_rng(11).normal(size=hi.shape+(2,))*2.**-60
    lo = low[..., 0]+1j*low[..., 1]
    actual = exponential(C(hi, lo))
    with localcontext() as context:
        context.prec = 100
        for index in np.ndindex(hi.shape):
            argument = tuple(Z.from_float(float(component(hi[index])))
                             +Z.from_float(float(component(lo[index])))
                             for component in (np.real, np.imag))
            expected = decimal_exponential(argument)
            observed = tuple(Z.from_float(float(component(actual.hi[index])))
                             +Z.from_float(float(component(actual.lo[index])))
                             for component in (np.real, np.imag))
            error = max(abs(a-b) for a, b in zip(observed, expected))
            scale = max(abs(expected[0]), abs(expected[1]), Z('1e-300'))
            assert error/scale < Z('3e-28'), (index, error/scale)


@pytest.mark.parametrize('argument', (0., 1j, -3j, 2., -2.))
def test_scalar_transmission_exponential_retains_its_shape(argument):
    actual = exponential(C(argument))
    assert actual.hi.shape == ()
    assert actual.lo.shape == ()
    np.testing.assert_allclose(actual.hi+actual.lo, np.exp(argument), atol=3e-15, rtol=0)


@pytest.mark.parametrize('argument', (np.nan, np.inf, complex(0., np.nan)))
def test_transmission_exponential_rejects_nonfinite_coordinates(argument):
    with pytest.raises(ArithmeticError, match='finite coordinates'):
        exponential(C(argument))
