"""``v`` orders on a raster with one row: refused where they cannot be returned, kept where they are.

Every ``n != 0`` order of a cell invariant along y is identically zero, so its window is
``(M, 0)`` (``periodic.Numerics(orders=)``). Asked for ``(M, 1)`` on a one-row raster:

* ``algorithm='shared'`` keeps two y cells under a one-dimensional period or at in-plane
  incidence (phi None or 0) and refused it as a spatial-Nyquist violation, "increase
  resolution", which no resolution fixes. That refusal now names ``orders=(M, 0)``.
* ``algorithm='reference'`` has no ``v`` axis for such a cell and returned the ``(M, 0)``
  answer without a word, for every period and azimuth. It now refuses with the same advice.
* ``algorithm='shared'`` under a two-dimensional period at a nonzero azimuth resolves y and
  computes the ``v`` orders; that is unchanged, and checked here: its ``v = 0`` row is the
  ``(M, 0)`` run's, bit for bit, and the other rows are zero to round-off.

Both refusals keep the engine's ``periodic.invalid`` code. The references are ``rcwa.run`` on
the same grating, which refuses ``(M, 1)`` with the same advice and gives the efficiencies the
advised ``(M, 0)`` runs reproduce, and the two algorithms against each other.
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


def problem(period=(PERIOD,), phi=None, theta=None, polarization=None):
    source = p.PlaneWave(side='top', theta=theta, phi=phi,
                         **({} if polarization is None else dict(polarization=polarization)))
    return p.Problem(
        units=Units(length='um'),
        stack=p.Stack(period=period, top=Material(1.), bottom=Material(1.),
                      layers=(p.Layer(thickness=THICKNESS, epsilon=EPSILON),)),
        excitation=p.Excitation(wavelength=(WAVELENGTH,), sources=(source,)))


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


def assert_refused_at_every_door(q, n, orders, why):
    advice = f'use Numerics(orders=({orders[0]}, 0))'
    (issue,) = quiet(p.check, q, numerics=n).issues
    assert issue.code == 'periodic.invalid' and advice in issue.message and why in issue.message
    assert 'Nyquist' not in issue.message
    for door in (p.prepare, p.run):
        with pytest.raises(ValidationError) as caught:
            quiet(door, q, numerics=n)
        assert (caught.value.code, caught.value.message) == (issue.code, issue.message)


SHARED_PINNED = [((PERIOD,), None), ((PERIOD,), .1), ((PERIOD, PERIOD), None), ((PERIOD, PERIOD), 0.)]


@pytest.mark.parametrize('period, phi', SHARED_PINNED,
                         ids=['1d-normal', '1d-azimuth', '2d-phi-none', '2d-phi-zero'])
@pytest.mark.parametrize('orders', [(1, 1), (2, 1), (0, 3)])
def test_shared_refuses_v_orders_where_y_holds_two_cells(period, phi, orders):
    why = ('one-dimensional period' if len(period) == 1 else 'in-plane incidence')
    assert_refused_at_every_door(problem(period, phi, theta=None if phi is None else .2),
                                 numerics('shared', orders), orders, why)


@pytest.mark.parametrize('period, phi', [((PERIOD,), None), ((PERIOD,), .1),
                                         ((PERIOD, PERIOD), None), ((PERIOD, PERIOD), .1)],
                         ids=['1d-normal', '1d-azimuth', '2d-normal', '2d-azimuth'])
@pytest.mark.parametrize('orders', [(1, 1), (2, 1)])
def test_reference_refuses_v_orders_instead_of_dropping_them(period, phi, orders):
    q = problem(period, phi, theta=None if phi is None else .2)
    assert_refused_at_every_door(q, numerics('reference', orders), orders, 'has no v axis')


def test_shared_still_returns_v_orders_where_it_resolves_y():
    # A two-dimensional period at a nonzero azimuth: y is resolved (6 cells), and the v orders
    # it computes are those of a y-invariant cell -- zero beside the (M, 0) answer.
    q = problem((PERIOD, .2), phi=.1, theta=.2, polarization=(1., .6j))
    crossed = quiet(p.run, q, numerics=numerics('shared', (1, 1)))
    plain = quiet(p.run, q, numerics=numerics('shared', (1, 0)))
    assert crossed.reflection.orders_v == (-1, 0, 1) and plain.reflection.orders_v == (0,)
    for name in ('R', 'T'):
        np.testing.assert_array_equal(getattr(crossed, name)[:, 1:2], getattr(plain, name))
        assert np.max(getattr(crossed, name)[:, [0, 2]]) < 1e-24
    np.testing.assert_array_equal(crossed.reflection.amplitude_ratio_s[:, 1:2],
                                  plain.reflection.amplitude_ratio_s)


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
