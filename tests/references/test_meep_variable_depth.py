"""Public Meep moving-exit differences and independently controlled film optics."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path

import numpy as np
import pytest

from studies.fdtd.variable_depth.fdtd_variable_depth import reference, response, complex_amplitudes


def test_meep_moving_exit_reference_controls_and_refinement(record_property):
    base = repository_path('tests/references')
    archives = {k: dict(np.load(base/f'golden_meep_variable_depth{suffix}.npz'))
                for k, suffix in ((40, '_coarse'), (80, ''), ('refined', '_refined'))}
    target, derivative = map(complex_amplitudes, reference(.173))
    errors = {}
    for key, d in archives.items():
        a, j = d['amplitudes'], d['jacobians']
        assert a.shape == (2, 2, 2) and j.shape == (2, 2, 2, 2, 1)
        assert np.isfinite(a).all() and np.isfinite(j).all()
        assert d['version'] == '1.34.0' and d['dimensions'] == 3
        assert d['reference_plane'] == 'moving_exit' and d['reference_depth'] == .173
        assert d['resolution'] == (40 if key == 40 else 80)
        assert d['layer_smoothing'] == .025
        assert d['subpixel_tol'] == (1e-10 if key == 'refined' else 1e-8)
        assert d['transverse_cells'] == (2 if key == 'refined' else 1)
        np.testing.assert_array_equal(d['parameters'], [.173])
        np.testing.assert_array_equal(d['background'], [1.2, 1.1, 1.6, 1.15])
        np.testing.assert_array_equal(d['wavelengths'], [.98, 1.02])
        np.testing.assert_array_equal(d['durations'], [50., 70.])
        np.testing.assert_array_equal(d['deltas'], [1e-4, 5e-5])
        assert np.max(abs(a[1]-a[0])) < 2e-8
        assert np.max(abs(j[1]-j[0])) < 2e-7
        assert np.max(abs(j[:, 1]-j[:, 0])) < 2e-5
        errors[key] = [float(np.max(abs(a[-1]-target))), float(np.max(abs(j[-1, -1, ..., 0]-derivative)))]
    for name, tolerance in (('amplitudes', 2e-7), ('jacobians', 2e-5)):
        assert np.max(abs(archives['refined'][name]-archives[80][name])) < tolerance
    assert np.all(np.array(errors[80]) < errors[40])
    assert errors[80][0] < .004 and errors[80][1] < .03
    record_property('meep_moving_exit_errors', errors)


def test_total_depth_autodiff_matches_the_public_meep_perturbations(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    data = np.load(repository_path('tests/references/golden_meep_variable_depth.npz'))
    f = lambda d: response(d, 80)
    value, derivative = map(complex_amplitudes, jax.jit(lambda d: (f(d), jax.jacfwd(f)(d)))(np.array(.173)))
    amplitude_gap = float(np.max(abs(value-data['amplitudes'][-1])))
    derivative_gap = float(np.max(abs(derivative-data['jacobians'][-1, -1, ..., 0])))
    assert amplitude_gap < .003
    assert derivative_gap < .03
    record_property('wavebit_meep_amplitude_and_derivative_gaps', [amplitude_gap, derivative_gap])
