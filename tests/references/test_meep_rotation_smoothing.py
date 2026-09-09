"""Graded-profile reference controls, independent scattering and shape derivatives."""
from pathlib import Path

import numpy as np
import pytest

from studies.fdtd_rotation_gradients import finite_difference, response
from studies.fdtd_rotation_smoothing import graded_reference


@pytest.fixture(scope='module')
def references():
    base = Path(__file__).parent
    data = {key: dict(np.load(base/name)) for key, name in (
        (20, 'golden_meep_rotation_smoothing_coarse.npz'),
        (40, 'golden_meep_rotation_smoothing.npz'),
        ('profile', 'golden_meep_rotation_smoothing_profile.npz'))}
    p = data[40]['parameters']
    return data, graded_reference(p), finite_difference(graded_reference, p, 1e-5)


def test_graded_fourier_reference_grid_order_and_difference_controls(references):
    data, target, derivative = references
    p = data[40]['parameters']
    for f in (lambda p: graded_reference(p, grid=512), lambda p: graded_reference(p, order=10)):
        assert np.max(abs(f(p)-target)) < 1e-5
        assert np.max(abs(finite_difference(f, p, 1e-5)-derivative)) < 3e-4
    assert np.max(abs(finite_difference(graded_reference, p, 5e-6)-derivative)) < 1e-6


def test_independent_graded_meep_controls_and_cell_refinement(references, record_property):
    data, target, derivative = references
    errors = {}
    for key, archive in data.items():
        a, j = archive['amplitudes'], archive['jacobians']
        assert a.shape == (2, 8) and j.shape == (2, 2, 8, 2)
        assert np.isfinite(a).all() and np.isfinite(j).all()
        assert archive['dimensions'] == 3
        assert archive['shape_smoothing'] == .05
        assert archive['profile_sampling'] == 'cell_centres'
        assert not archive['material_grid_averaging']
        assert archive['profile_probe_error'] < 2e-12
        assert archive['profile_grid'] == (512 if key == 'profile' else 256)
        assert archive['resolution'] == (20 if key == 'profile' else key)
        np.testing.assert_array_equal(archive['durations'], [80., 100.])
        np.testing.assert_array_equal(archive['deltas'], [1e-4, 5e-5])
        assert np.max(abs(a[1]-a[0])) < 2e-8
        assert np.max(abs(j[1]-j[0])) < 2e-7
        assert np.max(abs(j[:, 1]-j[:, 0])) < 1e-4
        errors[key] = np.r_[np.max(abs(a[-1]-target)), np.max(abs(j[-1, -1]-derivative), axis=0)]
        record_property(f'meep_{key}_amplitude_angle_width_errors', errors[key].tolist())
    assert np.max(abs(data['profile']['amplitudes']-data[20]['amplitudes'])) < 3e-5
    assert np.max(abs(data['profile']['jacobians']-data[20]['jacobians'])) < 1e-3
    assert np.all(errors[40] < errors[20])
    # The measured width error is still 0.052 at resolution 40. These Meep
    # data establish controlled cell refinement, not a precise continuum
    # width-gradient oracle. Do not assign them the Fourier accuracy budget
    # used for Wavebit below; retain the discrepancy in the recorded results.
    assert errors[40][0] < .008 and errors[40][1] < .006


def test_smoothed_area_scattering_refines_against_graded_rcwa_and_meep(references, record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    data, target, derivative = references
    p = data[40]['parameters']
    errors = {}
    for resolution in (20, 30):
        f = lambda p: response(p, resolution, smoothing=.05)
        value, jac = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        errors[resolution] = np.r_[np.max(abs(value-target)), np.max(abs(jac-derivative), axis=0)]
        record_property(f'wavebit_{resolution}_amplitude_angle_width_errors', errors[resolution].tolist())
    assert np.all(errors[30] < errors[20])
    assert errors[30][0] < .005 and errors[30][1] < .0015 and errors[30][2] < .009
    gap = np.r_[np.max(abs(value-data[40]['amplitudes'][-1])),
                np.max(abs(jac-data[40]['jacobians'][-1, -1]), axis=0)]
    record_property('wavebit_30_vs_meep_40_amplitude_angle_width_gaps', gap.tolist())
    assert gap[0] < .012 and gap[1] < .008 and gap[2] < .06
