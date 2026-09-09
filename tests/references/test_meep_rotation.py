"""Rotated-block reference controls and independently checked area-FDTD gradients."""
from pathlib import Path

import numpy as np
import pytest

from studies.fdtd_rotation_gradients import rcwa_reference, response, finite_difference


@pytest.fixture(scope='module')
def references():
    base = Path(__file__).parent
    data = {r: dict(np.load(base/name)) for r, name in
            ((20, 'golden_meep_rotation_coarse.npz'), (40, 'golden_meep_rotation.npz'))}
    p = data[40]['parameters']
    return data, rcwa_reference(p), finite_difference(rcwa_reference, p, 1e-5)


def test_independent_rotated_block_reference_uncertainty_and_refinement(references, record_property):
    data, target, derivative = references
    errors = {}
    for resolution, archive in data.items():
        a, j = archive['amplitudes'], archive['jacobians']
        assert a.shape == (2, 8) and j.shape == (2, 2, 8, 2)
        assert np.isfinite(a).all() and np.isfinite(j).all()
        assert archive['dimensions'] == 3 and archive['resolution'] == resolution
        np.testing.assert_array_equal(archive['durations'], [80., 100.])
        np.testing.assert_array_equal(archive['deltas'], [1e-4, 5e-5])
        assert np.max(abs(a[1]-a[0])) < 2e-8
        assert np.max(abs(j[1]-j[0])) < 1e-7
        assert np.max(abs(j[:, 1]-j[:, 0])) < 1e-4
        errors[resolution] = np.r_[np.max(abs(a[-1]-target)), np.max(abs(j[-1, -1]-derivative), axis=0)]
        record_property(f'meep_{resolution}_amplitude_angle_width_errors', errors[resolution].tolist())
    assert np.all(errors[40] < errors[20])
    assert errors[40][0] < .004
    # Local derivatives are repeatable but finite-grid shape errors remain
    # substantial, especially for width. This is an explicit measured limit.
    assert errors[40][1] < .004 and errors[40][2] < .05


def test_area_scattering_derivatives_against_rcwa_and_independent_meep(references, record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    data, target, derivative = references
    p = data[40]['parameters']
    errors = {}
    for resolution in (16, 32):
        f = lambda p: response(p, resolution)
        value, jac = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        errors[resolution] = np.r_[np.max(abs(value-target)), np.max(abs(jac-derivative), axis=0)]
    assert np.all(errors[32] < errors[16])
    assert errors[32][0] < .008 and errors[32][1] < .003 and errors[32][2] < .02
    # These two regression grids bound cost. The documented 24/40 study also
    # records nonmonotonic width-error changes, so no every-grid claim follows.
    external = data[40]
    gap = np.r_[np.max(abs(value-external['amplitudes'][-1])),
                np.max(abs(jac-external['jacobians'][-1, -1]), axis=0)]
    record_property('wavebit_32_amplitude_angle_width_errors', errors[32].tolist())
    record_property('wavebit_32_vs_meep_40_amplitude_angle_width_gaps', gap.tolist())
    assert gap[0] < .012 and gap[1] < .006 and gap[2] < .08
