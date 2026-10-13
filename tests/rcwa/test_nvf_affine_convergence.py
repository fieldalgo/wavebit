"""Independent slab optics and provenance guards for affine comparisons."""
import json

import numpy as np
import pytest

from studies.rcwa.nvf_affine_convergence.nvf_affine_convergence import convergence, directional_changes, load_projection, material_coefficients, power_record, real_coefficients

pytestmark = pytest.mark.research


def uniform_coefficients(epsilon=4., cap=4):
    c = np.zeros((8, 2*cap+1, 2*cap+1), complex)
    c[[0, 3], cap, cap] = 1
    c[[4, 6, 7], cap, cap] = epsilon
    return c


@pytest.mark.parametrize('order', [0, 1, (2, 1)])
@pytest.mark.parametrize('transform', [np.array([[1., .45], [0., .7]]), np.array([[.2, -.7], [1., .45]])])
def test_affine_optical_readout_matches_independent_uniform_slab(order, transform):
    c = uniform_coefficients()
    result = power_record(c, order, transform)
    epsilon = np.array([1., 4., 2.25])
    kz = np.sqrt(epsilon-np.sin(.2)**2)
    phase = np.exp(2j*np.pi/900*300*kz[1])
    for pol, admittance in (('s', kz), ('p', epsilon/kz)):
        y0, y1, y2 = admittance
        r01, r12 = (y0-y1)/(y0+y1), (y1-y2)/(y1+y2)
        t01, t12 = 2*y0/(y0+y1), 2*y1/(y1+y2)
        r = (r01+r12*phase**2)/(1+r01*r12*phase**2)
        t = t01*t12*phase/(1+r01*r12*phase**2)
        assert result[f'{pol}_T'] == pytest.approx(y2/y0*abs(t)**2, abs=2e-12, rel=0)
        assert result[f'{pol}_R'] == pytest.approx(abs(r)**2, abs=2e-12, rel=0)


def test_complex_material_and_wrong_mean_cannot_enter_real_enclosure():
    c = uniform_coefficients()
    np.testing.assert_array_equal(real_coefficients(np.concatenate([c, c[5:6]])), c)
    invalid = c.copy()
    invalid[4, 4, 4] += .1j
    with pytest.raises(ValueError, match='real-field'):
        real_coefficients(invalid)
    invalid = np.concatenate([c, c[5:6]+.01])
    with pytest.raises(ValueError, match='symmetric'):
        real_coefficients(invalid)
    invalid = c.copy()
    invalid[0, 4, 4] = 1.1
    with pytest.raises(ValueError, match='identity mean'):
        real_coefficients(invalid)


@pytest.mark.parametrize('key,changed', [('transform', [[1., 0.], [0., 1.]]), ('gap', 0.),
                                        ('fixture', 'partitioned_gap'), ('cap', 5)])
def test_reference_export_must_describe_the_same_physical_profile(tmp_path, key, changed):
    context = dict(transform=[[1., .45], [0., .7]], gap=17., width=350., seams=[31.], values=[4., 9.])
    metadata = dict(fixture='affine_partitioned_gap', cap=4, **context)
    path = tmp_path/'projection.npz'
    c = uniform_coefficients()
    c[7] = material_coefficients(4, context['gap'], context['width'], context['seams'], context['values'])
    np.savez(path, coefficients=c, metadata=json.dumps(metadata))
    np.testing.assert_array_equal(load_projection(path, **context)[0], c)
    metadata[key] = changed
    np.savez(path, coefficients=c, metadata=json.dumps(metadata))
    with pytest.raises(ValueError, match='fixture|mismatch|cap'):
        load_projection(path, **context)


def test_reference_material_and_missing_metadata_are_checked(tmp_path):
    context = dict(transform=[[1., .45], [0., .7]], gap=17., width=350., seams=[31.], values=[4., 9.])
    metadata = dict(fixture='affine_partitioned_gap', cap=4, **context)
    path = tmp_path/'projection.npz'
    c = uniform_coefficients()
    np.savez(path, coefficients=c, metadata=json.dumps(metadata))
    with pytest.raises(ValueError, match='material'):
        load_projection(path, **context)
    metadata.pop('gap')
    np.savez(path, coefficients=c, metadata=json.dumps(metadata))
    with pytest.raises(ValueError, match='mismatch: gap'):
        load_projection(path, **context)


def test_invalid_study_parameters_and_missing_projection_modes_raise():
    with pytest.raises(ValueError, match='nonempty'):
        list(convergence([], []))
    with pytest.raises(ValueError, match='positive real'):
        list(convergence([1], [], values=[4.+.1j, 9.]))
    with pytest.raises(ValueError, match='transform'):
        list(convergence([1], [], transform=[[1., 1.], [0., 0.]]))
    with pytest.raises(ValueError, match='each cell'):
        list(convergence([1], [(1, 4)]))
    with pytest.raises(ValueError, match='cap'):
        power_record(uniform_coefficients(cap=1), (1, 0), np.eye(2))


def test_contact_candidate_has_exact_anisotropic_static_enclosure():
    transform = np.array([[1., .45], [0., .7]])
    result = list(convergence([0], [(1, 2, 2, 2)], transform=transform, gap=0.))
    bounds, optics = result
    normal = np.linalg.solve(transform.T, [1., 0.])
    normal /= np.linalg.norm(normal)
    nn = np.outer(normal, normal)
    harmonic = 1/(.5+206/700/4+144/700/9)
    arithmetic = .5+206/700*4+144/700*9
    exact = harmonic*nn+arithmetic*(np.eye(2)-nn)
    for key in ('upper', 'lower'):
        np.testing.assert_allclose(bounds[key], exact, atol=3e-11, rtol=0)
    assert optics['energy_error'] < 2e-12


def optical_record(order, p, s, **settings):
    return dict(kind='affine_candidate', order=order, p_T=p, p_R=1-p, s_T=s, s_R=1-s,
                transform=[[1., .45], [0., .7]], gap=17., width=350., seams=[31.], values=[4., 9.],
                **settings)


def test_directional_changes_expose_cancellation_and_both_parallel_edges():
    records = [optical_record(order, p, s) for order, p, s in [
        ((4, 8), .40, .70), ((6, 8), .42, .71), ((4, 12), .37, .68), ((6, 12), .40, .685),
    ]]
    result, = directional_changes(records)
    assert result['x_orders'] == [4, 6]
    assert result['y_orders'] == [8, 12]
    assert result['source_kind'] == 'affine_candidate'
    assert result['p'] == pytest.approx(dict(x_at_lower_y=.02, x_at_upper_y=.03,
                                            y_at_lower_x=-.03, y_at_upper_x=-.02,
                                            mixed=.01, diagonal=0.), abs=2e-16)
    assert result['s'] == pytest.approx(dict(x_at_lower_y=.01, x_at_upper_y=.005,
                                            y_at_lower_x=-.02, y_at_upper_x=-.025,
                                            mixed=-.005, diagonal=-.015), abs=2e-16)
    assert 'converged' not in result


@pytest.mark.parametrize('setting', ['body_degree', 'projection'])
def test_directional_changes_keep_auxiliary_resolutions_separate(setting):
    points = [(2, 2), (4, 2), (2, 4), (4, 4)]
    records = [optical_record(order, .4, .7, **{setting: value}) for value in (2, 4) for order in points]
    results = directional_changes(records)
    assert len(results) == 2
    assert {record[setting] for record in results} == {2, 4}
    assert directional_changes(records[:3]+records[7:]) == []


def test_directional_changes_skip_missing_corners_and_accept_scalar_orders():
    records = [optical_record(order, .4, .7) for order in [2, (4, 2), (2, 4), 4, (6, 6)]]
    result, = directional_changes([dict(kind='affine_cell_bounds'), *records])
    assert result['x_orders'] == [2, 4]
    assert result['y_orders'] == [2, 4]
    assert directional_changes(records[:3]) == []
    with pytest.raises(ValueError, match='duplicate'):
        directional_changes([*records, optical_record((2, 2), .5, .6)])


@pytest.mark.parametrize('changes', [dict(p_T=np.nan), dict(p_R=.1), dict(order=(-1, 2)), dict(kind='unknown')])
def test_directional_changes_refuse_invalid_optical_records(changes):
    with pytest.raises(ValueError):
        directional_changes([dict(optical_record(2, .4, .7), **changes)])
