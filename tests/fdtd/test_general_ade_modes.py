"""Homogeneous ADE: independent SI recurrences, modes and startup diagnostics."""
import math
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from wavebit.units import Z0

from tests.fdtd._homogeneous_ade_reference import (
    EPSILON,
    FIELD_TOL,
    RECURRENCE_TOL,
    UNITS,
    coefficients,
    continuum_history,
    fourier_mode,
    host,
    make_problem,
    matrix_history,
    mesh,
    numerics,
    pole_parameters,
    probe_at,
    run_problem,
    scaled_error,
    transition,
    uniform_history,
)


@pytest.fixture(params=['cpu', 'cuda:0'], ids=['cpu', 'cuda'])
def execution_device(request):
    torch = pytest.importorskip('torch')


    if request.param != 'cpu' and not torch.cuda.is_available():
        pytest.skip('requires an actual CUDA device')
    return request.param


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('damped', [False, True], ids=['lossless', 'damped'])
@pytest.mark.parametrize('component', range(3), ids=list('xyz'))
def test_uniform_closed_roots_and_recurrence(execution_device, kind, damped, component):
    cells, steps = 8, 97
    electric = np.zeros((3, cells, cells, cells))
    electric[component] = 1.
    magnetic = np.zeros_like(electric)
    saved_electric, saved_magnetic = electric.copy(), magnetic.copy()
    problem = make_problem(kind, damped, electric, magnetic, cells)
    result = run_problem(problem, numerics(cells, steps), device=execution_device)
    dt = result.provenance['time_step_seconds']
    assert dt == pytest.approx(.5/cells/math.sqrt(3)*UNITS.time_unit_seconds, rel=2e-15)
    parameters = pole_parameters(kind, damped)
    expected = uniform_history(steps, dt, *parameters)
    metrics = {'device': execution_device, 'kind': kind, 'damped': damped,
               'axis': component, 'field': {}, 'record': {}}
    for field in ('E', 'H'):
        wanted = electric*expected[-1] if field == 'E' else magnetic
        values = host(result.final_fields.values[field])*(Z0 if field == 'H' else 1.)
        error = scaled_error(values, wanted)
        metrics['field'][field] = error
        assert error < FIELD_TOL
        for c in range(3):
            key = field+'xyz'[c]
            wanted = expected[1:] if field == 'E' and c == component else np.zeros(steps)
            values = host(result[key].values)*(Z0 if field == 'H' else 1.)
            error = scaled_error(values, wanted)
            metrics['record'][key] = error
            assert error < FIELD_TOL
            times = (np.arange(steps)+(1. if field == 'E' else .5))*dt
            np.testing.assert_allclose(host(result[key].times_seconds), times, rtol=2e-14, atol=0.)
    values = np.r_[1., host(result['E'+'xyz'[component]].values)]
    g, h, d = coefficients(dt, *parameters)
    residual = values[2:]-(g-d/EPSILON)*values[1:-1]+h*values[:-2]-(1-g+h)
    metrics['recurrence'] = float(np.max(abs(residual)))
    assert metrics['recurrence'] < RECURRENCE_TOL
    assert values[1] == pytest.approx(1., rel=0., abs=1e-14)
    assert np.max(abs(expected-1)) > 1e-4  # Omitting the pole cannot pass.
    wrong_startup = uniform_history(steps+1, dt, *parameters)[1:]
    assert np.max(abs(expected-wrong_startup)) > 1e-4
    np.testing.assert_array_equal(problem.initial.E, saved_electric)
    np.testing.assert_array_equal(problem.initial.H, saved_magnetic)
    print(metrics, flush=True)


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('damped', [False, True], ids=['lossless', 'damped'])
@pytest.mark.parametrize('rotation', range(3), ids=list('xyz'))
def test_nonzero_fourier_mode_against_independent_si_matrix(execution_device, kind, damped, rotation):
    cells, steps = 12, 83
    k, discrete_k, initial, electric, magnetic, spatial = fourier_mode(cells, rotation)
    initial_e, initial_h = electric.copy(), magnetic.copy()
    problem = make_problem(kind, damped, electric, magnetic, cells)
    result = run_problem(problem, numerics(cells, steps), device=execution_device)
    dt = result.provenance['time_step_seconds']
    assert dt == pytest.approx(.5/cells/math.sqrt(3)*UNITS.time_unit_seconds, rel=2e-15)
    parameters = pole_parameters(kind, damped)
    matrix = transition(dt, discrete_k, *parameters)
    expected = matrix_history(matrix, initial, steps)
    assert np.max(abs(expected[:, 3:6])) > .05
    assert np.max(abs(expected[:, 6:9])) > .05
    metrics = {'device': execution_device, 'kind': kind, 'damped': damped, 'rotation': rotation,
               'k': k.tolist(), 'field': {}, 'record': {}}
    for field, offset in (('E', 0), ('H', 3)):
        wanted = np.stack([(expected[-1, offset+c]*spatial(field, c, mesh(cells, field, c))).real
                           for c in range(3)])
        values = host(result.final_fields.values[field])*(Z0 if field == 'H' else 1.)
        error = scaled_error(values, wanted)
        metrics['field'][field] = error
        assert error < FIELD_TOL
        for c in range(3):
            key = field+'xyz'[c]
            wanted = (expected[1:, offset+c]*spatial(field, c, probe_at(cells, field, c))).real
            assert np.max(abs(wanted)) > .01, 'TimeSeriesMonitor must not sample a nodal plane'
            values = host(result[key].values)*(Z0 if field == 'H' else 1.)
            error = scaled_error(values, wanted)
            metrics['record'][key] = error
            assert error < FIELD_TOL
    omitted = matrix_history(transition(dt, discrete_k, parameters[0], parameters[1], 0.),
                             initial, steps)
    wrong_initial = initial.copy()
    _, _, drive = coefficients(dt, *parameters)
    wrong_initial[6:9] = drive*initial[:3]
    wrong_initial[12:15] = drive*initial[:3]
    wrong_startup = matrix_history(matrix, wrong_initial, steps)
    metrics['omitted_pole_difference'] = float(np.max(abs(omitted[:, :6]-expected[:, :6])))
    metrics['wrong_startup_difference'] = float(np.max(abs(wrong_startup[:, :6]-expected[:, :6])))
    assert metrics['omitted_pole_difference'] > 1e-4
    assert metrics['wrong_startup_difference'] > 1e-4
    np.testing.assert_array_equal(problem.initial.E, initial_e)
    np.testing.assert_array_equal(problem.initial.H, initial_h)
    print(metrics, flush=True)


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
def test_raw_continuum_refinement_diagnostic_without_phase_alignment(kind):
    cells, end_time = 8, 2*UNITS.time_unit_seconds
    electric = np.zeros((3, cells, cells, cells))
    electric[1] = 1.
    problem = make_problem(kind, False, electric, np.zeros_like(electric), cells)
    parameters = pole_parameters(kind, False)
    rows = []
    for steps in (64, 128, 256):
        dt = end_time/steps
        courant = dt/UNITS.time_unit_seconds*cells*math.sqrt(3)
        result = run_problem(problem, numerics(cells, steps, courant), device='cpu')
        assert result.provenance['time_step_seconds'] == pytest.approx(dt, rel=2e-15)
        actual = np.r_[1., host(result['Ey'].values)]
        discrete = uniform_history(steps, dt, *parameters)
        assert scaled_error(actual, discrete) < FIELD_TOL
        times = np.arange(steps+1)*dt
        raw = continuum_history(times, *parameters)
        shifted = continuum_history(times-dt/2, *parameters)
        rows.append({'steps': steps, 'dt_seconds': dt,
                     'raw_window': float(np.max(abs(actual-raw))),
                     'raw_endpoint': float(abs(actual[-1]-raw[-1])),
                     'startup_shift_window': float(np.max(abs(actual-shifted))),
                     'startup_shift_endpoint': float(abs(actual[-1]-shifted[-1]))})
    # These are diagnostics, not acceptance orders. Keep raw and shifted
    # quantities distinct even when one converges faster.
    for previous, current in pairwise(rows):
        current['raw_window_order'] = math.log2(previous['raw_window']/current['raw_window'])
        current['startup_shift_window_order'] = math.log2(
            previous['startup_shift_window']/current['startup_shift_window'])
    print({'continuum_diagnostic': kind, 'fixed_end_seconds': end_time, 'rows': rows}, flush=True)


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
def test_magnetic_only_initial_fourier_mode(execution_device, kind, record_property):
    """H alone drives E and the ADE with the default nonmagnetic material."""
    cells, steps = 12, 23
    _, discrete_k, initial, _, _, spatial = fourier_mode(cells, 0)
    direction = discrete_k/np.linalg.norm(discrete_k)
    h = np.cross(direction, [.37, -.8, .53])
    h /= np.linalg.norm(h)
    initial[:] = 0
    initial[3:6] = h  # The matrix magnetic variable is Z0*H.
    magnetic = np.stack([(h[c]*spatial('H', c, mesh(cells, 'H', c))).real/Z0
                         for c in range(3)])
    electric = np.zeros_like(magnetic)
    saved_magnetic = magnetic.copy()
    problem = make_problem(kind, True, electric, magnetic, cells)
    structure = problem.scene.structures[0]
    problem = replace(problem,
                      scene=replace(problem.scene, structures=(replace(
                          structure, material=replace(structure.material, mu=None)),)),
                      initial=replace(problem.initial, E=None))
    result = run_problem(problem, numerics(cells, steps), device=execution_device)
    dt = .5/cells/math.sqrt(3)*UNITS.time_unit_seconds
    np.testing.assert_allclose(result.provenance['time_step_seconds'], dt, rtol=2e-15, atol=0.)
    expected = matrix_history(transition(dt, discrete_k, *pole_parameters(kind, True), mu=1.),
                              initial, steps)
    assert np.max(abs(expected[1:, :3])) > .05
    assert np.max(abs(expected[1:, 6:9])) > .01
    errors = {}
    for field, offset in (('E', 0), ('H', 3)):
        wanted = np.stack([(expected[-1, offset+c]*spatial(field, c, mesh(cells, field, c))).real
                           for c in range(3)])
        values = host(result.final_fields.values[field])*(Z0 if field == 'H' else 1.)
        errors[field] = scaled_error(values, wanted)
        assert errors[field] < FIELD_TOL
        for c in range(3):
            label = field+'xyz'[c]
            wanted = (expected[1:, offset+c]*spatial(field, c, probe_at(cells, field, c))).real
            assert np.max(abs(wanted)) > .01
            values = host(result[label].values)*(Z0 if field == 'H' else 1.)
            errors[label] = scaled_error(values, wanted)
            assert errors[label] < FIELD_TOL
    assert problem.initial.E is None
    assert problem.scene.structures[0].material.mu is None
    np.testing.assert_array_equal(problem.initial.H, saved_magnetic)
    record_property('magnetic_initial_max_field_record_error', max(errors.values()))
