"""Independent residual jets and a fixed, dimensionless pulse fitting protocol.

The target and all iterates use the independent 15-lane SI Yee reference. The
step sizes depend only on the target Gauss--Newton matrix, never fitted outcomes.
This bounded trajectory makes no parameter-identifiability or convergence claim.
"""
from functools import lru_cache

import numpy as np

from tests.fdtd import _pulse_yee_reference as pulse
from wavebit.units import Z0

TARGET = (.985, 1.005, .1, .14)
SEEDS = pulse.POINTS
STEPS = 41
UPDATES = 12
DIRECTION = np.array([1., -.5, .25, .75])
BOUNDS = np.array([[.94, 1.04], [.97, 1.04], [-.5, .7], [.08, .2]])
RTOL = 2e-12


def reference(point):
    return pulse.reference(tuple(point), stamp='bspline', steps=STEPS)


def observations(expected):
    """Six component values and jets, with H expressed as Z0 H."""
    fields = np.concatenate((expected['fields']['E'], Z0*expected['fields']['H']))
    first = np.concatenate((expected['first_fields']['E'],
                            Z0*expected['first_fields']['H']), axis=1)
    second = np.concatenate((expected['second_fields']['E'],
                             Z0*expected['second_fields']['H']), axis=2)
    records = np.stack([expected['records'][k]*(Z0 if k[0] == 'H' else 1.)
                        for k in pulse.LABELS])
    dr = np.stack([expected['first_records'][k]*(Z0 if k[0] == 'H' else 1.)
                   for k in pulse.LABELS], axis=1)
    ddr = np.stack([expected['second_records'][k]*(Z0 if k[0] == 'H' else 1.)
                    for k in pulse.LABELS], axis=2)
    return {'fields': (fields, first, second), 'records': (records, dr, ddr)}


@lru_cache(maxsize=1)
def target_data():
    expected = reference(TARGET)
    observed = observations(expected)
    weights = {'fields': pulse.FIELD_WEIGHTS[:, None, None, None],
               'records': pulse.FIELD_WEIGHTS[:, None]*pulse.time_weights(STEPS)}
    normalization = (np.mean(weights['records']*observed['records'][0]**2)
                     + .25*np.mean(weights['fields']*observed['fields'][0]**2))
    assert np.isfinite(normalization) and normalization > 0
    return expected, observed, weights, float(normalization)


def residual_objective(expected, *, omit_curvature=False):
    """L, four gradients, full Hessian of the fixed weighted residual."""
    _, target, weights, normalization = target_data()
    observed = observations(expected)
    answer = (0., np.zeros(4), np.zeros((4, 4)))
    for group, factor in (('records', 1.), ('fields', .25)):
        value, first, second = observed[group]
        residual = value-target[group][0]
        if omit_curvature:
            second = np.zeros_like(second)
        jet = pulse.weighted_jet(residual, first, second, weights[group])
        answer = tuple(a+factor*b/normalization for a, b in zip(answer, jet))
    return answer


@lru_cache(maxsize=1)
def plan():
    target, observed, weights, normalization = target_data()
    _, _, gn = residual_objective(target)
    diagonal = np.diag(gn)
    assert np.isfinite(gn).all() and np.all(diagonal > 0)
    d = 1/diagonal
    scaled = np.sqrt(d[:, None])*gn*np.sqrt(d[None, :])
    maximum = float(np.linalg.eigvalsh(scaled)[-1])
    assert np.isfinite(maximum) and maximum > 0
    eta = .1/maximum
    return {'target': target, 'target_fields': observed['fields'][0],
            'target_records': observed['records'][0], 'weights': weights,
            'normalization': normalization, 'gauss_newton': gn,
            'diagonal': d, 'eta': eta, 'rates': eta*d}


def oracle(point):
    expected = reference(point)
    value, gradient, hessian = residual_objective(expected)
    return {'value': value, 'gradient': gradient, 'hessian': hessian,
            'hvp': hessian@DIRECTION, 'expected': expected}


def in_bounds(point):
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and np.all(point >= BOUNDS[:, 0])
                and np.all(point <= BOUNDS[:, 1]))


def trajectory(seed):
    point = np.array(SEEDS[seed], dtype=float)
    rows = []
    rates = plan()['rates']
    for index in range(UPDATES):
        assert in_bounds(point), (index, point)
        exact = oracle(point)
        value, gradient, hvp = (exact[k] for k in ('value', 'gradient', 'hvp'))
        raw = point-rates*gradient
        after = np.clip(raw, BOUNDS[:, 0], BOUNDS[:, 1])
        assert np.isfinite(value) and value > 0
        assert np.isfinite(gradient).all() and np.isfinite(hvp).all()
        assert np.all(abs(gradient)/value > pulse.NONZERO)
        assert np.all(abs(hvp)/value > pulse.NONZERO)
        assert in_bounds(after), (index, after)
        rows.append({'iteration': index, 'before': point.copy(), 'value': value,
                     'gradient': gradient.copy(), 'hvp': hvp.copy(), 'proposal': raw.copy(),
                     'projected': raw != after, 'after': after.copy()})
        point = after
    final = oracle(point)
    assert final['value'] < rows[0]['value']
    return {'rows': rows, 'final_point': point, 'final_value': final['value']}
