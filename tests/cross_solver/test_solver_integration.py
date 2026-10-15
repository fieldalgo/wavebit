"""One public Stack and phase/coordinate convention across RCWA and periodic FDTD."""

from wavebit._common.periodic import Incidence as _legacy_wavebit_Incidence

from wavebit._common.periodic import Stack as _legacy_wavebit_Stack

from wavebit._common.enums import Tetm as _legacy_wavebit_Tetm

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

from wavebit.fdtd._periodic.probes import fdtd_poynting as _legacy_wavebit_fdtd_poynting

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

import wavebit


@pytest.mark.parametrize('backend', [None, 'numpy'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_asymmetric_typed_stack_has_the_same_complex_response(backend, side, tetm):
    # Unequal layers and bounding media make a reversed stack a different problem.
    stack = _legacy_wavebit_Stack.from_structure(np.array([[[2.1]], [[3.2]]]), [.13, .21],
                                         top=1., bot=2.25, period=.4)
    incidence = _legacy_wavebit_Incidence(wavelength=1., theta=.27, tetm=tetm, incidence_side=side)
    reference = _legacy_wavebit_rcwa(stack=stack, incidence=incidence, truncation_order=0).solve()
    actual = _legacy_wavebit_fdtd(stack=stack, incidence=incidence, backend=backend,
                          resolution=40, n_periods=80 if backend is None else 35,
                          truncation_order=0)
    for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
        np.testing.assert_allclose(getattr(actual, name), getattr(reference, name), atol=.015)
    assert actual.incident is not None
    np.testing.assert_allclose(actual.power_entering, 1-actual.R.sum())
    assert abs(actual.absorption) < .01


@pytest.mark.parametrize('backend', [None, 'numpy'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_nonzero_orders_have_the_same_complex_phase_and_sp_frame(backend, tetm):
    structure = np.ones((1, 1, 64))
    structure[..., 15:43] = 2.25
    scene = dict(structure=structure, thickness=[.23], top=1., bot=1., period=1.6,
                 wavelength=1., theta=.13, tetm=tetm)
    reference = _legacy_wavebit_rcwa(**scene, truncation_order=15).solve()
    actual = _legacy_wavebit_fdtd(**scene, backend=backend, resolution=40,
                          n_periods=80 if backend is None else 40, truncation_order=1)
    channel = 's' if tetm == 'te' else 'p'
    assert abs(getattr(reference.order(-1), 't_'+channel)) > .05
    for order in (-1, 0, 1):
        for name in ('r_'+channel, 't_'+channel):
            np.testing.assert_allclose(getattr(actual.order(order), name),
                                       getattr(reference.order(order), name), atol=.025)


@pytest.mark.parametrize('backend', [None, 'numpy'])
@pytest.mark.parametrize('side, direction', [('top', -1), ('bottom', 1)])
def test_recorded_fields_and_monitors_use_upward_z(backend, side, direction):
    _, data = _legacy_wavebit_fdtd(structure=np.ones((1, 1, 1)), thickness=[.3], top=1., bot=1.,
                           period=.4, wavelength=1., tetm='te', backend=backend,
                           incidence_side=side, resolution=12,
                           n_periods=80 if backend is None else 30, truncation_order=0,
                           record_field=3, record_components='all', return_diagnostics=True,
                           field_monitors={'inside': .15})
    sample = _legacy_wavebit_fdtd_poynting(data, z=.15)
    assert direction*sample['S'][:, 2].sum() > 0
    assert direction*data['monitors']['inside']['power_z'].item() > 0
    assert sample['region'] == 'total'
    assert data['monitors']['inside']['z'] == pytest.approx(.15)
    np.testing.assert_allclose(data['layer_planes'], [0., .3])
    assert data['reflection_z'] == pytest.approx(.3 if side == 'top' else 0.)
    for field in data['fields'].values():
        assert np.all(np.diff(field['z']) > 0)
        assert field['values'].shape[1] == len(field['z'])


@pytest.mark.parametrize('name', ['kx', 'ky'])
def test_unsupported_cartesian_incidence_has_an_explicit_error(name):
    with pytest.raises(ValueError, match='Cartesian kx/ky.*theta/phi'):
        _legacy_wavebit_fdtd(incidence=_legacy_wavebit_Incidence(wavelength=1., **{name: .2}))


def test_jax_typed_stack_preserves_live_material_derivative():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    def run(value, backend):
        xp = jnp if backend == 'jax' else np
        stack = _legacy_wavebit_Stack.from_structure(value*xp.array([[[1.]], [[1.3]]]), [.13, .21],
                                             top=1., bot=2.25, period=.4)
        incidence = _legacy_wavebit_Incidence(wavelength=1., tetm='tm', incidence_side='top')
        return _legacy_wavebit_fdtd(stack, incidence, backend=backend, resolution=8,
                             n_periods=30, truncation_order=0).T.sum()

    value, derivative = jax.jit(jax.value_and_grad(lambda v: run(v, 'jax')))(2.)
    step = 1e-4
    finite = (run(2+step, 'numpy')-run(2-step, 'numpy'))/(2*step)
    np.testing.assert_allclose(value, run(2., 'numpy'), atol=1e-10)
    np.testing.assert_allclose(derivative, finite, atol=1e-7, rtol=3e-5)
    assert abs(finite) > 1e-3


def test_missing_required_scene_is_a_named_argument_error():
    with pytest.raises(TypeError, match='missing required scene arguments.*structure.*top.*bot'):
        _legacy_wavebit_fdtd()
    with pytest.raises(TypeError, match='missing required scene arguments.*top.*bot'):
        _legacy_wavebit_fdtd(structure=np.ones((1, 1, 1)), thickness=[.1], wavelength=1., period=.3)


@pytest.mark.parametrize('backend', [None, 'numpy'])
def test_default_incidence_is_the_same_pure_s_state_as_rcwa(backend):
    scene = dict(structure=np.array([[[2.25]]]), thickness=[.2], top=1., bot=1.,
                 period=.4, wavelength=1.)
    reference = _legacy_wavebit_rcwa(**scene, truncation_order=0).solve()
    actual = _legacy_wavebit_fdtd(**scene, backend=backend, resolution=30,
                          n_periods=80 if backend is None else 35, truncation_order=0)
    np.testing.assert_allclose(actual.r_s, reference.r_s, atol=.02)
    np.testing.assert_allclose(actual.t_s, reference.t_s, atol=.02)
    np.testing.assert_allclose(actual.r_p, 0., atol=1e-14)
    np.testing.assert_allclose(actual.t_p, 0., atol=1e-14)


@pytest.mark.parametrize('backend', [None, 'numpy'])
def test_invariant_y_fields_do_not_pick_up_an_unused_period_phase(backend):
    # Conical incidence has nonzero ky, even though the published y=0 slice
    # of this structure does not depend on the arbitrary second cell length.
    records = []
    for period_y in (.8, 1.2):
        _, data = _legacy_wavebit_fdtd(
            structure=np.ones((1, 1, 1)), thickness=[.3], top=1., bot=1.,
            period=[.4, period_y], wavelength=1., theta=.3, phi=.5,
            polarization=(1., .4j), backend=backend, resolution=10,
            n_periods=80 if backend is None else 35, truncation_order=0,
            field_monitors={'inside': .15}, record_field=7,
            record_components='all', return_diagnostics=True)
        assert data['invariant_y']
        np.testing.assert_allclose(data['angles'], [.3], atol=1e-14)
        if backend is None:
            np.testing.assert_allclose(data['angles_deg'], np.rad2deg(data['angles']), atol=1e-14)
        assert data['azimuth'] == pytest.approx(.5)
        np.testing.assert_array_equal(data['monitors']['inside']['y'], [0.])
        records.append(data)
    for component in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        np.testing.assert_allclose(records[0]['monitors']['inside']['fields'][component],
                                   records[1]['monitors']['inside']['fields'][component], atol=1e-12)
        np.testing.assert_allclose(records[0]['fields'][component]['values'],
                                   records[1]['fields'][component]['values'], atol=1e-12)
    np.testing.assert_allclose(records[0]['field_e'], records[1]['field_e'], atol=1e-12)


@pytest.mark.parametrize('side, direction', [('top', -1), ('bottom', 1)])
def test_crossed_mixed_jones_fields_have_the_analytic_centre_phase(side, direction):
    theta, phi, depth, height = .3, .5, .3, .15
    _, data = _legacy_wavebit_fdtd(
        structure=np.ones((1, 2, 2)), thickness=[depth], top=1., bot=1.,
        period=[.4, .8], wavelength=1., theta=theta, phi=phi,
        polarization=(1., .4j), backend='numpy', incidence_side=side,
        resolution=20, n_periods=35, truncation_order=0,
        field_monitors={'inside': height}, return_diagnostics=True)
    assert not data['invariant_y']
    plane = data['monitors']['inside']
    x, y = np.meshgrid(plane['x'], plane['y'])
    k = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),
                  direction*np.cos(theta)])
    es = np.array([-np.sin(phi), np.cos(phi), 0.])
    ep = np.array([-np.cos(theta)*np.cos(phi), -np.cos(theta)*np.sin(phi),
                   direction*np.sin(theta)])
    electric = (es+.4j*ep)/np.sqrt(1+.4**2)
    magnetic = np.cross(k, electric)
    entrance = depth if side == 'top' else 0.
    phase = np.exp(2j*np.pi*(k[0]*x+k[1]*y+k[2]*(height-entrance)))
    for family, vector in (('e', electric), ('h', magnetic)):
        for index, axis in enumerate('xyz'):
            np.testing.assert_allclose(plane['fields'][family+axis][0],
                                       vector[index]*phase, atol=.02)


@pytest.mark.parametrize('phi', [0., .3])
def test_legacy_negative_tilt_jones_amplitudes_use_the_public_incident_frame(phi):
    structure = np.ones((1, 1, 32))
    structure[..., 7:22] = 2.25
    scene = dict(structure=structure, thickness=[.23], top=1., bot=1., period=1.6,
                 wavelength=1., theta=-.13, phi=phi, polarization=(1., .4j))
    reference = _legacy_wavebit_rcwa(**scene, truncation_order=10).solve()
    actual = _legacy_wavebit_fdtd(**scene, resolution=30, n_periods=80, truncation_order=1)
    for leg in ('incident', 's_inc', 'p_inc'):
        for name in ('r_s', 'r_p', 't_s', 't_p'):
            np.testing.assert_allclose(getattr(getattr(actual, leg), name),
                                       getattr(getattr(reference, leg), name)[:, 9:12], atol=.025)


def test_legacy_pure_tm_and_jones_p_describe_the_same_fields():
    records = []
    medium = wavebit.Material(1., mu=1.21)
    for polarization in ({'tetm': 'tm'}, {'polarization': (0., 1.)}):
        _, data = _legacy_wavebit_fdtd(
            structure=np.ones((1, 1, 1)), mu=np.full((1, 1, 1), 1.21),
            thickness=[.3], top=medium, bot=medium, period=.4,
            wavelength=1., theta=-.3, resolution=12, n_periods=80,
            truncation_order=0, field_monitors={'inside': .15},
            record_field=7, record_components='all', return_diagnostics=True, **polarization)
        np.testing.assert_allclose(data['angles'], [-.3], atol=1e-14)
        np.testing.assert_allclose(data['angles_deg'], np.rad2deg(data['angles']), atol=1e-14)
        records.append(data)
    for component in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        np.testing.assert_allclose(records[0]['monitors']['inside']['fields'][component],
                                   records[1]['monitors']['inside']['fields'][component], atol=1e-12)
        np.testing.assert_allclose(records[0]['fields'][component]['values'],
                                   records[1]['fields'][component]['values'], atol=1e-12)
    for name in ('flux_density', 'power_z'):
        np.testing.assert_allclose(records[0]['monitors']['inside'][name],
                                   records[1]['monitors']['inside'][name], atol=1e-12)


@pytest.mark.parametrize('backend', [None, 'numpy'])
@pytest.mark.parametrize('tetm', ['TM', _legacy_wavebit_Tetm.TM])
def test_periodic_fdtd_accepts_shared_typed_polarization_declarations(backend, tetm):
    stack = _legacy_wavebit_Stack.from_structure(np.array([[[2.25]]]), [.2], top=1., bot=1., period=.4)
    incidence = _legacy_wavebit_Incidence(wavelength=1., tetm=tetm)
    reference = _legacy_wavebit_rcwa(stack=stack, incidence=incidence, truncation_order=0).solve()
    actual = _legacy_wavebit_fdtd(stack, incidence, backend=backend, resolution=30,
                          n_periods=80 if backend is None else 35, truncation_order=0)
    np.testing.assert_allclose(actual.r_p, reference.r_p, atol=.02)
    np.testing.assert_allclose(actual.t_p, reference.t_p, atol=.02)
    np.testing.assert_allclose(actual.r_s, 0., atol=1e-14)


@pytest.mark.parametrize('shape', [(1, 1, 1), (1, 2, 2)], ids=['conical', 'crossed'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_legacy_coupled_magnetic_transmission_matches_fresnel(shape, side):
    # A layer equal to the bottom medium makes a single magnetic interface.
    # Check the independent Fresnel powers and the electric-amplitude admittance
    # for both incident ports; equal exterior mu would hide a missing mu factor.
    theta = .2
    eps_i, mu_i, eps_t, mu_t = ((1., 1., 2.25, 1.6) if side == 'top'
                               else (2.25, 1.6, 1., 1.))
    n_i, n_t = np.sqrt(eps_i*mu_i), np.sqrt(eps_t*mu_t)
    q_i = n_i*np.cos(theta)
    q_t = np.sqrt(n_t**2-(n_i*np.sin(theta))**2)
    r_s = (q_i/mu_i-q_t/mu_t)/(q_i/mu_i+q_t/mu_t)
    r_p = (eps_t*q_i-eps_i*q_t)/(eps_t*q_i+eps_i*q_t)
    t_s = 1+r_s
    t_p = (q_i/n_i)/(q_t/n_t)*(1-r_p)
    admittance = (q_t/mu_t)/(q_i/mu_i)
    actual = _legacy_wavebit_fdtd(
        structure=np.full(shape, 2.25), mu=np.full(shape, 1.6),
        thickness=[.2], top=1., bot=wavebit.Material(2.25, mu=1.6),
        period=[.2, .2], wavelength=1., theta=theta, phi=.3,
        polarization=(1., .4j), incidence_side=side, resolution=40,
        n_periods=80, truncation_order=0)
    for leg, s_weight, p_weight in (('s_inc', 1., 0.), ('p_inc', 0., 1.),
                                    ('incident', 1/1.16, .16/1.16)):
        result = getattr(actual, leg)
        expected_r = s_weight*r_s**2+p_weight*r_p**2
        expected_t = admittance*(s_weight*t_s**2+p_weight*t_p**2)
        np.testing.assert_allclose(result.R.sum(), expected_r, rtol=0., atol=.015)
        np.testing.assert_allclose(result.T.sum(), expected_t, rtol=0., atol=.02)
        for channel in ('s', 'p'):
            np.testing.assert_allclose(getattr(result, 'T_'+channel),
                                       admittance*abs(getattr(result, 't_'+channel))**2,
                                       rtol=1e-12, atol=1e-14)
