"""Independent adjoint provenance and controls without assuming continuum equivalence."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path

import numpy as np
import pytest

from studies.fdtd.adjoint_comparison.fdtd_adjoint_comparison import reference_controls, measure


@pytest.mark.parametrize('resolution,suffix', [(20, '_coarse'), (40, '')])
def test_meep_adjoint_and_finite_difference_reference_controls(resolution, suffix, record_property):
    base = repository_path('tests/references')
    data = dict(np.load(base/f'golden_meep_adjoint_rotation{suffix}.npz'))
    previous = dict(np.load(base/f'golden_meep_rotation_smoothing{suffix}.npz'))
    assert data['amplitudes'].shape == (2, 2)
    assert data['adjoint_jacobians'].shape == (2, 2, 2)
    assert data['finite_jacobians'].shape == data['mapping_jacobians'].shape == (2, 2, 2, 2)
    for name in ('amplitudes', 'adjoint_jacobians', 'finite_jacobians', 'mapping_jacobians', 'run_end_times'):
        assert np.isfinite(data[name]).all()
    assert data['resolution'] == resolution and data['dimensions'] == 3
    assert data['version'] == '1.34.0'
    assert data['profile_grid'] == 256 and data['shape_smoothing'] == .05
    assert data['profile_sampling'] == 'cell_centres' and data['profile_probe_error'] < 2e-12
    assert not data['material_grid_averaging'] and not data['yee_grid']
    assert data['decimation_factor'] == 1 and data['source_kind'] == 'GaussianSource'
    np.testing.assert_array_equal(data['parameters'], previous['parameters'])
    for name in ('period', 'thickness', 'top', 'bottom', 'epsilon', 'ly', 'centre'):
        np.testing.assert_array_equal(data[name], previous[name])
    np.testing.assert_array_equal(data['parameter_names'], ['angle', 'width'])
    np.testing.assert_array_equal(data['channels'], ['t_ss', 't_pp'])
    np.testing.assert_array_equal(data['original_channel_indices'], [2, 7])
    np.testing.assert_array_equal(data['minimum_run_times'], [120., 140.])
    np.testing.assert_array_equal(data['maximum_run_times'], [120., 140.])
    assert data['termination'] == 'fixed_minimum_maximum'
    np.testing.assert_array_equal(data['deltas'], [1e-4, 5e-5])
    np.testing.assert_array_equal(data['mapping_deltas'], [1e-5, 5e-6])
    for name, expected in dict(wavelength=1., decay_by=1e-11, source_width=2.5,
                               source_cutoff=5., calibration_duration=120., cell_length=8.,
                               pml_thickness=1., courant=.4, source_position=-2.).items():
        assert data[name] == expected
    np.testing.assert_array_equal(data['monitor_positions'], [-1., 1.2])
    assert data['run_end_times'].shape == (2, 2, 9)
    assert np.all(data['run_end_times'] >= data['minimum_run_times'][:, None, None])
    assert np.all(data['run_end_times'] <= data['maximum_run_times'][:, None, None]+.8/resolution)
    controls = reference_controls(data, previous)
    assert controls['amplitude_time_change'] < 2e-8
    assert controls['adjoint_time_change'] < 3e-7
    assert controls['finite_time_change'] < 3e-7
    assert max(controls['finite_step_change']) < 5e-5
    assert max(controls['mapping_change']) < 2e-6
    assert controls['previous_amplitude_change'] < 3e-9
    assert controls['previous_derivative_change'] < 1e-7
    # Adjoint-versus-difference agreement is a separate measurement: neither
    # tight runtime controls nor the existence of an adjoint API establish it.
    record_property('measured_adjoint_finite_gaps', controls['adjoint_finite_gap'])


@pytest.mark.slow
def test_common_fdtd_accuracy_is_checked_against_independent_fourier_reference(record_property):
    pytest.importorskip('jax')
    result = measure()
    coarse, fine = result['rows']
    assert fine['wavebit_amplitude_error'] < coarse['wavebit_amplitude_error']
    assert np.all(np.array(fine['wavebit_derivative_error']) < coarse['wavebit_derivative_error'])
    assert fine['wavebit_amplitude_error'] < .003
    assert fine['wavebit_derivative_error'][0] < .001
    assert fine['wavebit_derivative_error'][1] < .005
    record_property('separate_gradient_comparisons', result)
