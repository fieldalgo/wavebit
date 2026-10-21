"""Fixed accuracy limits with separately bounded measurement uncertainty.

These rules qualify finite runs, not asymptotic stability. Missing or non-finite
evidence never passes, and a large time/grid difference never enlarges a limit.
"""
from itertools import combinations

import numpy as np


TIME_FACTORS = (1., 1.25, 1.5, 1.75, 2.)
TIME_FRACTION = .1
BOUNDARY_FRACTION = .1
REFERENCE_FRACTION = .05


def maximum_gap(values):
    """Largest pairwise difference, per observable (including complex phase)."""
    return np.maximum.reduce([np.abs(a - b) for a, b in combinations(values, 2)])


def assess(time_rows, boundary_rows, reference, reference_check, limits, *, qualification_only=False):
    """Grade one geometry/resolution with fixed per-observable absolute limits.

    Rows contain ``values`` and a bool ``valid`` established by the runner.
    All five times and both boundary controls are mandatory. Boundary rows are
    compared with the longest baseline time. All rows must be valid before an
    accuracy verdict can be made; failures of qualification are BLOCKED.
    ``qualification_only`` reserves accuracy for a separate convergence gate;
    it does not change any uncertainty cap.
    """
    expected = np.asarray(reference, complex)
    checked = np.asarray(reference_check, complex)
    budget = np.asarray(limits, float)
    rows = list(time_rows) + list(boundary_rows)
    result = dict(status='BLOCKED', issues=[])
    if len(time_rows) != len(TIME_FACTORS) or len(boundary_rows) != 2:
        result['issues'].append('missing_runs')
        return result
    if (expected.size == 0 or checked.shape != expected.shape or budget.shape != expected.shape
            or not np.isfinite(expected).all() or not np.isfinite(checked).all()
            or not np.isfinite(budget).all() or np.any(budget <= 0)):
        result['issues'].append('invalid_reference_or_limits')
        return result
    values = [np.asarray(row.get('values'), complex) for row in rows]
    if any(v.shape != expected.shape or not np.isfinite(v).all() for v in values):
        result['issues'].append('invalid_values')
        return result
    if any(row.get('valid') is not True for row in rows):
        result['issues'].append('invalid_run_diagnostics')
    time_gap = maximum_gap(values[:len(time_rows)])
    boundary_gap = maximum_gap([values[len(time_rows) - 1]] + values[len(time_rows):])
    reference_gap = np.abs(expected - checked)
    errors = np.maximum.reduce([np.abs(v - expected) for v in values])
    for name, gap, fraction in (('time', time_gap, TIME_FRACTION),
                                 ('boundary', boundary_gap, BOUNDARY_FRACTION),
                                 ('reference', reference_gap, REFERENCE_FRACTION)):
        result[name + '_ratio'] = (gap / (fraction * budget)).tolist()
        if np.any(gap > fraction * budget):
            result['issues'].append(name + '_uncertainty')
    result['accuracy_ratio'] = (errors / budget).tolist()
    if result['issues']:
        return result
    result['qualification_only'] = qualification_only
    result['status'] = 'FAIL' if not qualification_only and np.any(errors >= budget) else 'PASS'
    if result['status'] == 'FAIL':
        result['issues'].append('accuracy')
    return result


def overall(statuses):
    """An observed failure outranks a missing/invalid required measurement."""
    states = list(statuses)
    if 'FAIL' in states:
        return 'FAIL'
    return 'PASS' if states and all(s == 'PASS' for s in states) else 'BLOCKED'
