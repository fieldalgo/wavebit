"""Acceptance controls for the automatic grid/refit example, with exact models."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / 'examples/rcwa/spectral_fit_grid.py'
_SPEC = importlib.util.spec_from_file_location('spectral_fit_grid_example', _PATH)
example = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = example
_SPEC.loader.exec_module(example)


def fit(factory, **overrides):
    kw = dict(grids=(1, 2, 4, 8, 16, 32), noise_sigma=.01,
              parameter_tolerance=.03, spectrum_tolerance=.03)
    kw.update(overrides)
    return example.fit_grid_ladder(factory, [0.], [1., 2.], **kw)


def biased(grid):
    return lambda p: (np.array([p[0] + grid**-2, 2*p[0] + 2*grid**-2]),
                      np.array([[1.], [2.]]))


def test_refitting_removes_bias_and_requires_two_passes():
    report = fit(biased, spectrum_tolerance=.06)
    assert report.converged
    assert [s.grid for s in report.stages] == [1, 2, 4, 8, 16, 32]
    np.testing.assert_allclose(report.parameters, [1 - 32**-2], atol=1e-12)
    # Known noise propagates through a known, full-rank linear inverse problem.
    np.testing.assert_allclose(report.stages[-1].noise_std, [.01/np.sqrt(5)])
    assert abs(report.stages[0].parameters[0] - 1) > 100*report.stages[0].noise_std[0]


def test_optimizer_success_does_not_mean_grid_convergence():
    report = fit(biased, grids=(1, 2, 4))
    assert all(s.optimizer_success for s in report.stages)
    assert not report.converged
    assert 'budget exhausted' in report.reason


def test_an_accidental_pairwise_plateau_does_not_pass():
    biases = {1: 0., 2: 0., 4: .2, 8: .2}
    def factory(grid):
        return lambda p: (np.array([p[0] + biases[grid], 2*p[0]]), np.array([[1.], [2.]]))
    report = fit(factory, grids=(1, 2, 4, 8))
    assert not report.converged
    assert report.stages[1].parameter_change < 1
    assert report.stages[2].spectrum_change > 1


def test_stable_parameters_cannot_hide_forward_model_drift():
    def factory(grid):
        return lambda p: (np.array([p[0], float(grid)]), np.array([[1.], [0.]]))
    report = fit(factory, grids=(1, 2, 4))
    assert not report.converged
    assert report.stages[-1].parameter_change < 1e-10
    assert report.stages[-1].spectrum_change > 1


def test_rank_deficient_problem_is_not_reported_precise():
    report = fit(lambda grid: lambda p: (np.ones(2), np.zeros((2, 1))))
    assert not report.converged
    assert 'rank deficient' in report.reason
    assert np.isinf(report.stages[-1].noise_std).all()


def test_optimizer_budget_is_not_accepted():
    report = fit(biased, max_nfev=1)
    assert not report.converged
    assert not report.stages[-1].optimizer_success


@pytest.mark.parametrize('which', ['value', 'jacobian'])
def test_nonfinite_evaluation_is_rejected(which):
    def factory(grid):
        return lambda p: (np.array([np.nan if which == 'value' else p[0], 2*p[0]]),
                          np.array([[np.nan if which == 'jacobian' else 1.], [2.]]))
    with pytest.raises(ValueError, match='invalid spectrum or Jacobian'):
        fit(factory)


@pytest.mark.parametrize('kw', [dict(grids=(1, 2)), dict(grids=(1, 2, 2)),
                               dict(grids=(1, 2, 3.5)), dict(noise_sigma=0),
                               dict(parameter_tolerance=np.nan), dict(spectrum_tolerance=-1)])
def test_invalid_controls_are_rejected(kw):
    with pytest.raises(ValueError):
        fit(biased, **kw)


def test_a_bad_coarse_fit_is_retried_but_does_not_count_as_a_pass():
    def factory(grid):
        if grid == 1:
            return lambda p: (np.ones(2), np.zeros((2, 1)))
        return lambda p: (np.array([p[0], 2*p[0]]), np.array([[1.], [2.]]))
    report = fit(factory, grids=(1, 2, 4, 8))
    assert report.converged
    assert [s.grid for s in report.stages] == [1, 2, 4, 8]
    assert np.isinf(report.stages[0].noise_std).all()
    np.testing.assert_allclose(report.parameters, [1.], atol=1e-12)


def test_progress_callback_receives_each_completed_fit_once():
    seen = []
    report = fit(biased, spectrum_tolerance=.06, on_stage=seen.append)
    assert [s.grid for s in seen] == [s.grid for s in report.stages]
    assert len(seen) == 6


@pytest.mark.parametrize('field', ['initial', 'data', 'noise_sigma'])
def test_complex_inputs_are_not_silently_projected_to_real(field):
    kwargs = dict(initial=[0.], data=[1., 2.], grids=(1, 2, 4),
                  noise_sigma=.01, parameter_tolerance=.03, spectrum_tolerance=.03)
    kwargs[field] = np.asarray(kwargs[field], dtype=complex) + 1j
    with pytest.raises(ValueError, match='must be real'):
        example.fit_grid_ladder(biased, **kwargs)
