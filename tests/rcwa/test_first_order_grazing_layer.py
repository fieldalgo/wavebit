"""A first-order layer at exact grazing: a followable refusal, and the detuned solve.

An order on a uniform layer's light line has modal constant 0. The second-order formulation
refuses that as a defective pair (``rcwa.singular``) and ``Numerics(detuning=...)`` retries it.
The first-order formulation's mode split miscounted there and raised ``rcwa.invalid``, advising
a higher Fourier order -- which keeps the condition -- and a detuned run refused the same way,
since the retry takes only a numerical refusal. Reference: the second-order formulation on the
same layer, detuned the same way.
"""
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import NumericalError

WAVELENGTH, PERIOD = .8, .5


def problem(polarization, theta, layer_epsilon):
    return rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(period=(PERIOD, PERIOD), top=Material(1.), bottom=Material(2.25),
                         layers=(rcwa.Layer(thickness=.3, background=Material(layer_epsilon)),)),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH, sources=(rcwa.PlaneWave(
            side='top', polarization=polarization, theta=theta, phi=0.),)))


# Order (1, 0) on the layer's light line: kx/k0 = sin(theta) + wavelength/period = n_layer.
CASES = [('s', 0., (WAVELENGTH / PERIOD) ** 2),
         ('p', 0., (WAVELENGTH / PERIOD) ** 2),
         ('s', .3, (np.sin(.3) + WAVELENGTH / PERIOD) ** 2)]


@pytest.mark.parametrize('orders', [(1, 1), (2, 2)])
@pytest.mark.parametrize('polarization,theta,layer_epsilon', CASES)
def test_the_first_order_refusal_matches_the_second_orders_and_names_detuning(
        polarization, theta, layer_epsilon, orders):
    for formulation in ('second_order', 'first_order'):
        with pytest.raises(NumericalError) as caught:
            rcwa.run(problem(polarization, theta, layer_epsilon),
                     numerics=rcwa.Numerics(orders=orders, formulation=formulation))
        assert caught.value.code == 'rcwa.singular', formulation
        assert 'grazing' in str(caught.value)
        assert 'rcwa.Numerics(detuning=...)' in str(caught.value)
        assert 'Fourier order and' not in str(caught.value)


@pytest.mark.parametrize('orders', [(1, 1), (2, 2)])
@pytest.mark.parametrize('polarization,theta,layer_epsilon', CASES)
def test_the_advised_detuning_solves_the_first_order_layer_as_the_second_order_does(
        polarization, theta, layer_epsilon, orders):
    results = {}
    for formulation in ('second_order', 'first_order'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results[formulation] = rcwa.run(
                problem(polarization, theta, layer_epsilon),
                numerics=rcwa.Numerics(orders=orders, formulation=formulation, detuning=1e-12))
        assert any('auto_detune' in str(w.message) for w in caught), formulation
    first, second = results['first_order'], results['second_order']
    np.testing.assert_allclose(first.R, second.R, rtol=0, atol=1e-13)
    np.testing.assert_allclose(first.T, second.T, rtol=0, atol=1e-13)
    np.testing.assert_allclose(first.R.sum() + first.T.sum(), 1., atol=1e-12)


def test_an_ambiguous_split_away_from_grazing_is_still_a_value_error():
    """``M = i*I`` has four fluxless modes and no root near zero: the old refusal stands."""
    from wavebit.rcwa._engine.solve_2d_first_order import split_forward_backward
    with pytest.raises(ValueError, match='mode split found 4 forward modes'):
        split_forward_backward(np, np.eye(4, dtype=complex) * 1j, 1)
