"""Layer fitting preserves geometry and converges to independent planar optics."""
import numpy as np
import pytest
import wavebit

from studies.fdtd_layer_alignment import THICKNESS, multilayer, run
from studies.fdtd_oblique_pulse import BAND, airy, epsilon


def options(**changes):
    return dict(dict(structure=np.array([[[2.25]], [[1.5]]]), thickness=THICKNESS,
                     top=1., bot=1., period=.1, wavelength=1., backend='numpy',
                     resolution=10, n_periods=30, n_pml=4, truncation_order=0,
                     tetm='te', align_layers=True, return_diagnostics=True), **changes)


def test_characteristic_matrix_matches_airy_and_rcwa():
    single = multilayer(BAND, [epsilon(BAND)], [.213])
    for pol in ('te', 'tm'):
        np.testing.assert_allclose(single[pol], airy(.213)[pol], atol=1e-14)
    expected = multilayer(BAND, [np.full_like(BAND, 2.25), epsilon(BAND)], THICKNESS)
    for i, wavelength in enumerate(BAND):
        for pol, component in [('te', 's'), ('tm', 'p')]:
            result = wavebit.rcwa(structure=np.array([[[epsilon(wavelength)]], [[2.25]]]),
                                  thickness=THICKNESS[::-1], top=1., bot=1., period=.05,
                                  wavelength=wavelength, theta=np.arcsin(np.sin(.45)*wavelength),
                                  tetm=pol, truncation_order=0, backend='numpy').solve()
            actual = [getattr(result.incident, key+'_'+component).item() for key in ('r', 't')]
            np.testing.assert_allclose(actual, expected[pol][:, i], atol=1e-13)


def test_typed_conical_magnetic_layers_against_rcwa():
    stack = wavebit.Stack.from_structure(
        np.array([[[2.1]], [[3.2]]]), THICKNESS,
        top=wavebit.Material(1.2, 1.1), bot=wavebit.Material(2., 1.3), period=.1)
    incidence = wavebit.Incidence(wavelength=1., theta=.27, phi=.41, polarization=(1., .6j))
    common = dict(mu=np.array([[[1.1]], [[1.3]]]), truncation_order=0, backend='numpy')
    result = wavebit.fdtd(stack, incidence, **common, resolution=50, n_periods=30,
                          grading=True, align_layers=True)
    expected = wavebit.rcwa(stack=stack, incidence=incidence, **common).solve()
    for key in ('r_s', 'r_p', 't_s', 't_p'):
        np.testing.assert_allclose(getattr(result.incident, key), getattr(expected.incident, key), atol=.012)


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('grading', [None, True])
@pytest.mark.parametrize('subpixel', [False, True])
def test_all_layer_faces_and_staggered_record_bounds(side, grading, subpixel):
    kw = options(incidence_side=side, grading=grading, subpixel=subpixel, record_field=37,
                 record_components=['ey', 'ez', 'hx', 'hz'])
    result, full = wavebit.fdtd(**kw)
    bounds = {'z': (-.07, .27), 'x': (-.05, .05)}
    cropped_result, cropped = wavebit.fdtd(**kw, record_bounds=bounds)
    assert full['align_layers']
    np.testing.assert_array_equal(cropped_result.incident.t_s, result.incident.t_s)
    nodes = full['z_nodes']
    assert np.all(np.diff(nodes) > 0)
    np.testing.assert_allclose(np.diff(nodes), full['z_spacing'][:-1], atol=1e-14)
    if grading is None:
        np.testing.assert_allclose(np.diff(np.array(full['pml_bounds']).reshape(2, 2)),
                                   kw['n_pml']/kw['resolution'], atol=1e-14)
    for lower, upper in zip(full['layer_planes'][:-1], full['layer_planes'][1:]):
        start, stop = [np.argmin(abs(nodes-face)) for face in (lower, upper)]
        np.testing.assert_allclose(nodes[[start, stop]], [lower, upper], atol=1e-14)
        assert stop-start == np.ceil((upper-lower)*kw['resolution'])
        np.testing.assert_allclose(np.diff(nodes[start:stop+1]), (upper-lower)/(stop-start), atol=1e-14)
    for name, field in cropped['fields'].items():
        original = full['fields'][name]
        values = original['values']
        for dimension, axis in enumerate(('z', 'y', 'x'), 1):
            lower, upper = bounds.get(axis, (-np.inf, np.inf))
            indices = np.flatnonzero((original[axis] >= lower-1e-14) & (original[axis] <= upper+1e-14))
            values = np.take(values, indices, axis=dimension)
            np.testing.assert_allclose(field[axis], original[axis][indices], atol=1e-14)
        np.testing.assert_array_equal(field['values'], values)


def test_default_mesh_is_unchanged_and_thin_layers_reduce_time_step():
    kw = options()
    kw.pop('align_layers')
    implicit, a = wavebit.fdtd(**kw)
    explicit, b = wavebit.fdtd(**kw, align_layers=False)
    np.testing.assert_array_equal(implicit.incident.t_s, explicit.incident.t_s)
    np.testing.assert_array_equal(a['z_nodes'], b['z_nodes'])
    _, thin = wavebit.fdtd(**options(thickness=[.013, .213]))
    assert thin['dt'] < a['dt']/3
    assert np.min(abs(thin['z_nodes']-.013)) < 1e-14


@pytest.mark.parametrize('value', [1, None, 'yes', [True]])
def test_alignment_requires_static_boolean(value):
    with pytest.raises(ValueError, match='align_layers'):
        wavebit.fdtd(**options(align_layers=value))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('grading', [None, True])
def test_aligned_bottom_pole_gradient_with_records_and_monitors(backend, grading):
    def objective(value, which):
        result, d = wavebit.fdtd(**options(
            backend=which, grading=grading, incidence_side='bottom', tetm='tm', theta=.25,
            dispersion=np.array([[[0]], [[1]]]), poles={1: [(0., .3, value)]},
            pulse_width=1.5, n_periods=35, dft_subsample=2,
            record_field=41, record_components=['ex', 'hy'], record_bounds={'z': (-.07, .27)},
            field_monitors={'out': -.3}))
        return (result.incident.t_p.real.sum()+d['monitors']['out']['power_z'].sum()
                +.01*(abs(d['fields']['ex']['values'])**2).sum())
    expected = objective(30., 'numpy')
    delta = 1e-3
    finite = (objective(30.+delta, 'numpy')-objective(30.-delta, 'numpy'))/(2*delta)
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        value, derivative = jax.jit(jax.value_and_grad(lambda v: objective(v, 'jax')))(30.)
        convert = np.asarray
    else:
        torch = pytest.importorskip('torch')
        parameter = torch.tensor(30., dtype=torch.float64, requires_grad=True)
        value = objective(parameter, 'torch')
        derivative, = torch.autograd.grad(value, parameter)
        convert = lambda v: v.detach().numpy()
    np.testing.assert_allclose(convert(value), expected, atol=1e-11)
    np.testing.assert_allclose(convert(derivative), finite, rtol=2e-5, atol=1e-8)
    assert abs(finite) > 1e-4


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_oblique_metal_stack_refines_to_characteristic_matrix(side):
    errors = []
    for resolution in (40, 80):
        _, row = run(resolution, backend='numpy', side=side, periods=80)
        assert row['state_decayed']
        assert row['amplitudes_settled']
        assert max(row['boundary_distance']) < 1e-13
        errors.append(np.array(list(row['error'].values())))
    assert np.all(errors[1] < .006)
    assert np.all(errors[1] < .4*errors[0])
