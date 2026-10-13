"""Independent graded ellipse controls and three Meep shape sensitivities."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path

import numpy as np
import pytest

from studies.cross_solver.ellipse_gradients.fdtd_ellipse_gradients import reference
from studies.cross_solver.rotation_gradients.fdtd_rotation_gradients import finite_difference


@pytest.fixture(scope='module')
def references():
    base = repository_path('tests/references')
    data = {key: dict(np.load(base/name)) for key, name in (
        (20, 'golden_meep_ellipse_coarse.npz'), (40, 'golden_meep_ellipse.npz'),
        ('profile', 'golden_meep_ellipse_profile.npz'))}
    p = data[40]['parameters']
    return data, reference(p), finite_difference(reference, p, 1e-5)


def test_meep_ellipse_profile_runtime_step_and_refinement_controls(references, record_property):
    data, value, jac = references
    errors = {}
    for key, archive in data.items():
        a, j = archive['amplitudes'], archive['jacobians']
        assert a.shape == (2, 8) and j.shape == (2, 2, 8, 3)
        assert np.isfinite(a).all() and np.isfinite(j).all()
        assert archive['shape'] == 'ellipse' and archive['dimensions'] == 3
        np.testing.assert_array_equal(archive['parameter_names'], ['angle', 'width', 'height'])
        np.testing.assert_array_equal(archive['parameters'], [.37, .34, .2])
        assert archive['ly'] == .2 and archive['shape_smoothing'] == .05
        assert archive['profile_sampling'] == 'cell_centres'
        assert not archive['material_grid_averaging']
        assert archive['profile_probe_error'] < 2e-12
        assert archive['profile_grid'] == (512 if key == 'profile' else 256)
        assert archive['resolution'] == (20 if key == 'profile' else key)
        np.testing.assert_array_equal(archive['durations'], [80., 100.])
        np.testing.assert_array_equal(archive['deltas'], [1e-4, 5e-5])
        assert np.max(abs(a[1]-a[0])) < 2e-8
        assert np.max(abs(j[1]-j[0])) < 2e-7
        assert np.max(abs(j[:, 1]-j[:, 0])) < .002
        errors[key] = np.r_[np.max(abs(a[-1]-value)), np.max(abs(j[-1, -1]-jac), axis=0)]
        record_property(f'meep_{key}_amplitude_angle_width_height_errors', errors[key].tolist())
    assert np.max(abs(data['profile']['amplitudes']-data[20]['amplitudes'])) < 5e-5
    assert np.max(abs(data['profile']['jacobians']-data[20]['jacobians'])) < .01
    assert np.all(errors[40] < errors[20])
    assert np.all(errors[40] < [.02, .02, .2, .2])
