"""The order window of a periodic cell invariant along y is ``(M, 0)``, in both algorithms.

``periodic.Numerics(orders=)`` documents ``(M, 0)`` for a one-dimensional period. Asked for
``(M, 1)`` on such a cell -- a one-dimensional period, or a two-dimensional one with a one-row
raster -- ``algorithm='shared'`` refused it as a spatial-Nyquist violation with "increase
resolution", which no resolution fixes, and ``algorithm='reference'`` dropped the ``v``
half-width and returned the ``(M, 0)`` answer without a word. Both now refuse it at every door
with the same message, which names ``orders=(M, 0)``; the refusal keeps the engine's
``periodic.invalid`` code. The references are ``rcwa.run`` on the same grating, which refuses
the same window and gives the efficiencies the ``(M, 0)`` runs must reproduce, and the two
algorithms against each other.
"""
import warnings

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import ValidationError
from wavebit.fdtd import periodic as p
from wavebit.units import C0

EPSILON = np.array([[2.25, 1., 1., 2.25]])
PERIOD, THICKNESS, WAVELENGTH = .6, .1, .5


def problem(period=(PERIOD,)):
    return p.Problem(
        units=Units(length='um'),
        stack=p.Stack(period=period, top=Material(1.), bottom=Material(1.),
                      layers=(p.Layer(thickness=THICKNESS, epsilon=EPSILON),)),
        excitation=p.Excitation(wavelength=(WAVELENGTH,), sources=(p.PlaneWave(side='top'),)))


def numerics(algorithm, orders):
    return p.Numerics(algorithm=algorithm, spacing=1/30, duration_seconds=80*WAVELENGTH*1e-6/C0,
                      orders=orders)


def rcwa_problem():
    return rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(period=(PERIOD, PERIOD), top=Material(1.), bottom=Material(1.),
                         layers=(rcwa.Layer(thickness=THICKNESS, epsilon=EPSILON),)),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH,
                                   sources=(rcwa.PlaneWave(polarization=(1., 0.)),)))


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


@pytest.mark.parametrize('period', [(PERIOD,), (PERIOD, PERIOD)], ids=['1d-period', 'one-row'])
@pytest.mark.parametrize('algorithm', ['shared', 'reference'])
@pytest.mark.parametrize('orders', [(1, 1), (2, 1), (0, 3)])
def test_v_orders_on_an_invariant_cell_are_refused_with_the_window_to_use(period, algorithm, orders):
    q, n = problem(period), numerics(algorithm, orders)
    advice = f'Use Numerics(orders=({orders[0]}, 0))'
    (issue,) = quiet(p.check, q, numerics=n).issues
    assert issue.code == 'periodic.invalid' and advice in issue.message
    assert 'Nyquist' not in issue.message
    for door in (p.prepare, p.run):
        with pytest.raises(ValidationError) as caught:
            quiet(door, q, numerics=n)
        assert (caught.value.code, caught.value.message) == (issue.code, issue.message)


@pytest.mark.parametrize('orders', [(1, 1), (2, 1)])
def test_rcwa_refuses_the_same_window_on_the_same_grating(orders):
    with pytest.raises(ValidationError, match=rf'Use Numerics\(orders=\({orders[0]}, 0\)\)'):
        rcwa.run(rcwa_problem(), numerics=rcwa.Numerics(orders=orders))


def test_the_advised_window_agrees_across_algorithms_and_with_rcwa():
    converged = rcwa.run(rcwa_problem(), numerics=rcwa.Numerics(orders=(30, 0)))
    centre = slice(29, 32)    # orders -1, 0, +1 of 61
    reference = {name: np.asarray(converged.outputs[port].efficiency).reshape(-1)[centre]
                 for name, port in (('R', 'reflected'), ('T', 'transmitted'))}
    runs = {a: quiet(p.run, problem(), numerics=numerics(a, (1, 0))) for a in ('shared', 'reference')}
    for result in runs.values():
        assert result.reflection.orders_u == (-1, 0, 1) and result.reflection.orders_v == (0,)
        for name in ('R', 'T'):
            # 18 cells per period: the FDTD discretization, not the window, sets this gap.
            np.testing.assert_allclose(getattr(result, name)[0, 0], reference[name], atol=5e-3)
    for name in ('R', 'T'):
        np.testing.assert_allclose(getattr(runs['shared'], name), getattr(runs['reference'], name),
                                   atol=1e-3)
