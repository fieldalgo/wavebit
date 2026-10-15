"""Live total depth: fixed laboratory coordinates and moving reference phases."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd
import numpy as np
import pytest

import wavebit
from wavebit.fdtd._periodic.runtime import time_loop_backend
from wavebit.fdtd._periodic.scene import _layer_geometry, _layer_weights, _phase_between_planes
from tests.fdtd.test_fdtd_scene_angles import value_and_jacobian


def options(**changes):
    return dict(dict(structure=np.array([[[2.25]], [[1.6]]]), thickness=[.173, .227],
                     mesh_thickness=[.2, .2], variable_depth=True, top=1.2, bot=1.3,
                     period=.2, wavelength=1., backend='numpy', resolution=10, n_pml=6,
                     n_periods=40, truncation_order=0, theta=.3, phi=.4,
                     polarization=(1., .6j), layer_smoothing=.05), **changes)


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('value', [0, 1, None, 'yes', [], np.array(True)])
def test_variable_depth_is_an_explicit_static_boolean(monkeypatch, side, value):
    import wavebit.fdtd._periodic.scene as scene
    monkeypatch.setattr(scene, 'fdtd_grid', lambda *a, **kw: pytest.fail('reached time stepping'))
    with pytest.raises(ValueError, match='variable_depth'):
        _legacy_wavebit_fdtd(**options(incidence_side=side, variable_depth=value))


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('changes', [dict(mesh_thickness=None), dict(subpixel=False),
                                    dict(thickness=[.173, 2.5]), dict(thickness=[.01, .1])])
def test_invalid_geometry_and_exterior_clearance_are_checked_before_stepping(monkeypatch, side, changes):
    import wavebit.fdtd._periodic.scene as scene
    monkeypatch.setattr(scene, 'fdtd_grid', lambda *a, **kw: pytest.fail('reached time stepping'))
    with pytest.raises(ValueError, match='variable_depth|layer_smoothing'):
        _legacy_wavebit_fdtd(**options(incidence_side=side, **changes))


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_live_planes_and_nominal_reflection_preserve_the_length_jacobian(backend):
    if backend != 'numpy':
        pytest.importorskip('jax' if backend == 'jax' else 'torch')
    xp, array = (getattr(time_loop_backend(backend), name) for name in ('xp', 'array'))
    def planes(p):
        normal = _layer_geometry(p, [.2, .2], 2, xp=xp, array=array, variable_depth=True)[1]
        mirror = _layer_geometry(xp.flip(p, axis=0), [.2, .2], 2, xp=xp, array=array,
                                 variable_depth=True, anchor_bottom=True)[1]
        return xp.concat((normal, .4-xp.flip(mirror, axis=0)))
    p = np.array([.173, .317])
    expected = np.tile([0., .173, .49], 2)
    if backend != 'numpy':
        value, jac = value_and_jacobian(planes, p, backend)
        np.testing.assert_array_equal(jac, np.tile([[0, 0], [1, 0], [1, 1]], (2, 1)))
    else:
        value = planes(p)
    np.testing.assert_allclose(value, expected, atol=1e-16)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_fractional_cell_phase_has_independent_endpoint_derivatives(backend):
    if backend != 'numpy':
        pytest.importorskip('jax' if backend == 'jax' else 'torch')
    runtime = time_loop_backend(backend); xp, array = runtime.xp, runtime.array
    nodes, widths = np.array([0., .2, .5]), np.array([.2, .3, .4])
    rates = np.array([[2.+.3j, 3.-.2j], [4.+.1j, 5.-.4j], [6.+.7j, 7.-.8j]])
    def phase(p):
        value = _phase_between_planes(nodes, widths, array(widths[:, None]*rates), p[0], p[1], xp=xp, array=array)
        return xp.concat((xp.real(value), xp.imag(value)))
    expected = .03*rates[0]+.3*rates[1]+.13*rates[2]
    if backend != 'numpy':
        value, jac = value_and_jacobian(phase, np.array([.17, .63]), backend)
        target = np.stack((-rates[0], rates[2]), axis=-1)
        np.testing.assert_allclose(jac, np.r_[target.real, target.imag], atol=2e-15)
        _, at_node = value_and_jacobian(phase, np.array([.2, .5]), backend)
        target = np.stack((-rates[1], rates[2]), axis=-1)
        np.testing.assert_allclose(at_node, np.r_[target.real, target.imag], atol=2e-15)
    else:
        value = phase(np.array([.17, .63]))
    np.testing.assert_allclose(value, np.r_[expected.real, expected.imag], atol=1e-15)


def test_moving_exterior_smoothing_matches_independent_quadrature_and_reflection():
    from numpy.polynomial.legendre import leggauss
    centres, widths = np.array([-.2, .01, .17, .29, .34, .6]), np.array([.1, .04, .05, .08, .1, .2])
    edges, smoothing = np.array([-2., 0., .317, 2.]), .08
    actual = _layer_weights(centres, widths, edges, smoothing=smoothing,
                            moving_face='bottom', xp=np, array=np.asarray)
    expected = []
    points, weights = leggauss(4)
    for centre, width in zip(centres, widths):
        left, right = centre-width/2, centre+width/2
        cuts = sorted({left, right, *(v for v in (0., .317-.04, .317+.04) if left < v < right)})
        total = np.zeros(3)
        for a, b in zip(cuts, cuts[1:]):
            z = (a+b)/2+(b-a)/2*points
            upper = (z < 0).astype(float)
            lower = np.clip((z-.317)/smoothing+.5, 0., 1.)
            total += np.stack((upper, 1-upper-lower, lower))@weights*(b-a)/(2*width)
        expected.append(total)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    reflected = _layer_weights(.4-centres[::-1], widths[::-1], .4-edges[::-1], smoothing=smoothing,
                               moving_face='top', xp=np, array=np.asarray)
    np.testing.assert_allclose(actual, reflected[::-1, ::-1], atol=2e-14)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_roundoff_at_shared_nodes_never_doubles_the_phase_derivative(backend):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    runtime = time_loop_backend(backend); xp, array = runtime.xp, runtime.array
    nodes, widths = np.arange(-4, 9)*.1, np.full(13, .1)
    def phases(p):
        values = xp.stack([_phase_between_planes(nodes, widths, array(widths*(3+2j)),
                                                 array(nodes[0]), p[i], xp=xp, array=array) for i in range(len(p))])
        return xp.concat((xp.real(values), xp.imag(values)))
    p = nodes[1:-1]
    value, derivative = value_and_jacobian(phases, p, backend)
    np.testing.assert_allclose(value, np.r_[3*(p-nodes[0]), 2*(p-nodes[0])], atol=3e-14, rtol=0)
    np.testing.assert_allclose(derivative, np.r_[3*np.eye(len(p)), 2*np.eye(len(p))], atol=3e-14, rtol=0)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('side,grading,method', [('top', None, 'electric'), ('bottom', 1.06, 'directional')])
def test_joint_depth_material_and_angle_graph_matches_two_finite_differences(backend, side, grading, method):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    def evaluate(p, engine):
        runtime = time_loop_backend(engine); xp, array = runtime.xp, runtime.array
        result = _legacy_wavebit_fdtd(**options(backend=engine, thickness=p[:2],
            structure=xp.reshape(xp.stack((p[2], array(1.6))), (2, 1, 1)), theta=p[3],
            mesh_thickness=[.17, .23], incidence_side=side, grading=grading,
            align_layers=True, diffraction_method=method))
        v = xp.stack([getattr(getattr(result, column), field).reshape(())
                      for column in ('s_inc', 'p_inc') for field in ('r_s', 'r_p', 't_s', 't_p')])
        return xp.concat((xp.real(v), xp.imag(v)))
    p = np.array([.173, .287, 2.25, .3])
    value, derivative = value_and_jacobian(lambda p: evaluate(p, backend), p, backend)
    np.testing.assert_allclose(value, evaluate(p, 'numpy'), atol=3e-11, rtol=2e-10)
    for h in (1e-5, 5e-6):
        target = np.stack([(evaluate(p+d, 'numpy')-evaluate(p-d, 'numpy'))/(2*h) for d in np.eye(len(p))*h], axis=-1)
        np.testing.assert_allclose(derivative, target, atol=8e-8, rtol=3e-5)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_jit_keeps_laboratory_mesh_sources_monitors_and_records_fixed(side):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traces = []
    def solve(p):
        traces.append(None)
        r, out = _legacy_wavebit_fdtd(**options(backend='jax', thickness=p, incidence_side=side, align_layers=True,
            return_diagnostics=True, field_monitors={'probe': .11}, record_field=173,
            record_components=['ey', 'hz'], record_bounds={'z': (.05, .15)}))
        return dict(T=r.incident.T,
            **{key: out[key] for key in ('stable', 'z_nodes', 'source_z', 'reflection_z',
                                         'transmission_z', 'layer_planes', 'field_z', 'field_e')},
            probe_ey=out['monitors']['probe']['fields']['ey'],
            probe_z=out['monitors']['probe']['z'],
            **{key: out['diffraction'][key]
               for key in ('reflection_monitor_z', 'transmission_monitor_z')})
    compiled = jax.jit(solve)
    a, b = compiled(np.array([.173, .227])), compiled(np.array([.213, .287]))
    # Actual extraction planes, current sheets and field samples stay fixed in the lab.
    for key in ('z_nodes', 'source_z', 'field_z', 'probe_z',
                'reflection_monitor_z', 'transmission_monitor_z'):
        np.testing.assert_array_equal(a[key], b[key])
    assert a['stable'] and b['stable']
    np.testing.assert_allclose(a['probe_z'], .11, atol=1e-15, rtol=0)
    np.testing.assert_allclose(b['layer_planes'], [0., .213, .5], atol=1e-15)
    # Public +z points up: the bottom reference remains at zero and the top follows D.
    # Reflection is at the incident interface; transmission is at the opposite one.
    for out, depth in ((a, .4), (b, .5)):
        expected = (depth, 0.) if side == 'top' else (0., depth)
        np.testing.assert_array_equal((out['reflection_z'], out['transmission_z']), expected)
    invalid = compiled(np.array([.173, 3.]))
    assert not invalid['stable']
    for key in ('T', 'field_e', 'probe_ey'):
        assert np.isnan(invalid[key]).all()
    assert len(traces) == 1


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('depth', [.473, .5])
def test_uniform_medium_depth_gradient_is_the_independent_yee_phase(backend, side, depth):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    runtime = time_loop_backend(backend); xp = runtime.xp
    band = np.array([.98, 1.02])
    kw = options(structure=np.array([[[1.2]]]), mu=np.array([[[1.1]]]), mesh_thickness=[.4],
                  top=wavebit.Material(1.2, 1.1), bot=wavebit.Material(1.2, 1.1), wavelength=band,
                  backend=backend, incidence_side=side, resolution=12, n_pml=10,
                  n_periods=55, pulse_width=2., diffraction_method='directional')
    def transmission(p):
        results = _legacy_wavebit_fdtd(**dict(kw, thickness=p))
        v = xp.stack([getattr(getattr(r, column), field).reshape(()) for r in results
                      for column, field in (('s_inc', 't_s'), ('p_inc', 't_p'))])
        return xp.concat((xp.real(v), xp.imag(v)))
    value, derivative = value_and_jacobian(transmission, np.array([depth]), backend)
    _, diagnostic = _legacy_wavebit_fdtd(**dict(kw, backend='numpy', thickness=[depth], return_diagnostics=True))
    dz, dy, dx = diagnostic['spacing']; dt = diagnostic['dt']
    omega = 2*np.sin(np.pi*dt/band)/dt
    kx, ky = 2*np.pi*np.sqrt(1.32)*np.sin(.3)*np.array([np.cos(.4), np.sin(.4)])
    kz = 2/dz*np.arcsin(dz/2*np.sqrt(1.32*omega**2-(2*np.sin(kx*dx/2)/dx)**2-(2*np.sin(ky*dy/2)/dy)**2))
    expected = np.repeat(np.exp(1j*kz*depth), 2)
    gradient = 1j*np.repeat(kz, 2)*expected
    np.testing.assert_allclose(value, np.r_[expected.real, expected.imag], atol=3e-9, rtol=0)
    np.testing.assert_allclose(derivative[:, 0], np.r_[gradient.real, gradient.imag], atol=3e-8, rtol=0)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_rotated_recipe_keeps_shape_and_total_depth_gradients_together(backend):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    def evaluate(p, engine):
        xp = time_loop_backend(engine).xp
        structure = [dict(background=1.4, shapes=[dict(type='rectangle', lx=p[2], ly=.13,
            cx=.021, cy=-.017, angle=p[3], material=2.25)]), dict(background=1.6, shapes=[])]
        result = _legacy_wavebit_fdtd(**options(backend=engine, structure=structure, thickness=p[:2],
            period=[.2, .2], incidence_side='bottom', shape_integration='area', shape_smoothing=.04))
        v = xp.stack([getattr(getattr(result, c), k).reshape(())
                      for c in ('s_inc', 'p_inc') for k in ('r_s', 'r_p', 't_s', 't_p')])
        return xp.concat((xp.real(v), xp.imag(v)))
    p = np.array([.173, .287, .09, .21])
    value, derivative = value_and_jacobian(lambda p: evaluate(p, backend), p, backend)
    np.testing.assert_allclose(value, evaluate(p, 'numpy'), atol=3e-11, rtol=2e-10)
    for h in (1e-5, 5e-6):
        finite = np.stack([(evaluate(p+d, 'numpy')-evaluate(p-d, 'numpy'))/(2*h) for d in np.eye(4)*h], axis=-1)
        np.testing.assert_allclose(derivative, finite, atol=1e-7, rtol=3e-5)


def test_exterior_smoothing_makes_uniform_mesh_crossing_derivatives_continuous():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    from studies.fdtd.variable_depth.fdtd_variable_depth import response
    gaps = []
    for smoothing in (0., .025):
        derivative = jax.jit(jax.jacfwd(lambda d: response(d, 20, smoothing=smoothing)))
        gaps.append([np.max(abs(derivative(.175+h)-derivative(.175-h))) for h in (1e-4, 1e-6)])
    assert gaps[0][1] > .01 and gaps[0][1] > .9*gaps[0][0]
    assert gaps[1][1] < .03*gaps[1][0] and gaps[1][1] < .001
