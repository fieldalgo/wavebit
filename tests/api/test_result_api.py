"""The result surface: names, layout, and the ``order()`` accessor.

``docs/reference/common/conventions.md`` states the contract; this file pins it. Uppercase is power,
lowercase is amplitude; the Fresnel amplitudes are validated against closed forms in
``test_lossy_media.py`` — here the concern is the *shape* of the API: delegation,
indexing, and that the arrays and the accessor agree with each other.
"""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

import wavebit

WL = 500.0
# Permittivity: 12.25 is the index 3.5 these cells were written with before the flip.
CELL_1D = np.array([[[1.0, 1.0, 12.25, 12.25]]])
CELL_2D = np.array([[[1.0, 12.25], [12.25, 1.0]]])


def _solve_1d(**kw):
    args = dict(backend='numpy', tetm='te', top=1.0, bot=2.25, theta=0.3, truncation_order=2, wavelength=WL,
                period=[900.0], structure=CELL_1D, thickness=[300.0])
    args.update(kw)
    return _legacy_wavebit.rcwa(**args).solve()


def _solve_2d(**kw):
    args = dict(backend='numpy', top=1.0, bot=2.25, theta=0.3, phi=0.0, psi=0.0, truncation_order=[2, 1],
                wavelength=WL, period=[900.0, 900.0], structure=CELL_2D, thickness=[300.0])
    args.update(kw)
    return _legacy_wavebit.rcwa(**args).solve()


def test_efficiency_arrays_are_the_power_surface():
    res = _solve_1d()
    R, T = np.asarray(res.R), np.asarray(res.T)
    assert R.shape == (1, 5) and T.shape == (1, 5)        # truncation_order=2 -> 2*2+1 orders
    assert float(R.sum() + T.sum()) == pytest.approx(1.0, abs=1e-12)


def test_polarization_split_sums_to_the_total():
    sub = _solve_2d().incident
    assert np.allclose(np.asarray(sub.R), np.asarray(sub.R_s) + np.asarray(sub.R_p))
    assert np.allclose(np.asarray(sub.T), np.asarray(sub.T_s) + np.asarray(sub.T_p))


def test_order_accessor_agrees_with_the_arrays():
    """``order(m, n)`` is the same data as the arrays, addressed by order index.

    The arrays are ``(n_orders_v, n_orders_u)`` — note the transpose of the ``truncation_order=[x, y]`` input
    order — with the incident order at the centre. That layout is exactly what the
    accessor exists to spare users from, so this test IS the layout's documentation.
    """
    res = _solve_2d()
    R = np.asarray(res.R)
    (cx, fx), (cy, fy) = res.order_window
    for m, n in ((0, 0), (1, 0), (-2, 1), (2, -1)):
        assert float(res.order(m, n).R) == R[n + fy, m + fx]


def test_order_accessor_rejects_out_of_range_orders():
    res = _solve_1d()
    with pytest.raises(IndexError):
        res.order(3)              # truncation_order=2 retains |m| <= 2
    with pytest.raises(IndexError):
        res.order(0, 1)           # the 1D solve has a single row of orders


def test_unpolarized_delegation_averages_the_two_incidences():
    """A manually constructed power-average result has no incident-state response."""
    res = _solve_2d()
    if res.incident is not None:
        want = np.asarray(res.incident.R)
        assert np.allclose(np.asarray(res.R), want)
    te, tm = np.asarray(res.s_inc.R), np.asarray(res.p_inc.R)
    avg = wavebit.observations.diffraction.Result(None, res.s_inc, res.p_inc, order_window=res.order_window)
    assert np.allclose(np.asarray(avg.R), 0.5 * (te + tm))
    for name in ('power_entering', 'absorption'):
        assert getattr(avg, name) == pytest.approx(
            .5 * (getattr(res.s_inc, name) + getattr(res.p_inc, name)), abs=1e-12)
    for name in ('r_s', 'r_p', 't_s', 't_p'):
        with pytest.raises(ValueError, match='complex amplitudes require an incident result'):
            getattr(avg, name)


@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
@pytest.mark.parametrize('answers', ['efficiency', 'amplitude', 'field'])
def test_amplitude_shortcuts_read_the_incident_result(algorithm, answers):
    result = _solve_2d(layer_algorithm=algorithm, answers=answers)
    for name in ('r_s', 'r_p', 't_s', 't_p'):
        assert getattr(result, name) is getattr(result.incident, name)
    zero = result.order(0, 0)
    for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
        values = getattr(result, name)
        if values is None:
            assert getattr(zero, name) is None
        else:
            assert getattr(zero, name) == values[1, 2]


@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_omitted_polarization_returns_the_default_s_incidence(algorithm):
    result = _solve_2d(psi=None, layer_algorithm=algorithm)
    assert result.incident is not None
    np.testing.assert_allclose(result.T, result.s_inc.T, atol=1e-12, rtol=0)
    assert np.max(np.abs(result.T - result.p_inc.T)) > 1e-4
    assert result.T is result.incident.T
    assert result.order(0, 0).T == result.T[1, 2]


def test_amplitudes_are_lowercase_and_complex():
    sub = _solve_2d().incident
    for name in ('r_s', 'r_p', 't_s', 't_p'):
        assert np.iscomplexobj(np.asarray(getattr(sub, name))), name


def test_old_names_are_gone():
    """The meent-era names were replaced outright before the first release — no aliases.

    If either ever comes back as an alias, this test is the place that decision must
    consciously delete.
    """
    res = _solve_1d()
    assert not hasattr(res, 'given')
    for stale in ('de_ri', 'de_ti'):
        assert not hasattr(res, stale)
        assert not hasattr(res.incident, stale)
