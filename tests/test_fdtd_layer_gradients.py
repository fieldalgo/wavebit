"""Moving layer interfaces: physical overlaps, fixed meshes and full AD graphs."""
import numpy as np
import pytest

import wavebit
from wavebit.backends.time_domain import time_loop_backend

from test_fdtd_scene_angles import value_and_jacobian


def options(**changes):
    return dict(dict(structure=np.array([[[2.25]], [[1.3]]]), thickness=[.173, .227],
                     mesh_thickness=[.2, .2], top=1., bot=1., period=.2, wavelength=1.,
                     backend=0, resolution=10, n_pml=6, n_periods=30,
                     truncation_order=0, theta=.3, phi=.4, polarization=(1., .6j)), **changes)


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_overlap_partition_values_and_interface_derivative(backend):
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    from wavebit._core.scene_fdtd import _layer_weights
    runtime = time_loop_backend(backend)
    xp, array = runtime.xp, runtime.array
    def overlap(position):
        edges = xp.concat((array([-2., 0.]), position, array([1., 2.])))
        return _layer_weights(np.array([-.5, .1, .35, .7, 1.5]), np.array([.2, .4, .2, .4, .2]),
                              edges, xp=xp, array=array)
    expected = np.array([[1., 0., 0., 0.], [.25, .75, 0., 0.],
                         [0., .4, .6, 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    if backend:
        value, derivative = value_and_jacobian(overlap, np.array([.33]), backend)
        target = np.zeros((5, 4, 1)); target[2, 1, 0] = 5; target[2, 2, 0] = -5
        np.testing.assert_allclose(derivative, target, atol=1e-14)
    else:
        value = overlap(array([.33]))
    np.testing.assert_allclose(value, expected, atol=1e-14)
    np.testing.assert_allclose(np.sum(value, axis=1), 1., atol=1e-15)
    assert np.min(value) >= 0


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('name,value', [
    ('mesh_thickness', [.4]), ('mesh_thickness', [.2, 0.]),
    ('mesh_thickness', [.2, np.inf]), ('mesh_thickness', [.2, .2j]),
    ('mesh_thickness', [True, False]), ('thickness', [.2]),
    ('thickness', [.5, -.1]), ('thickness', [0., .4]),
    ('thickness', [np.nan, .2]), ('thickness', [.2, .3]), ('thickness', [.4, 1e-16]),
    ('thickness', [.2, .2j]), ('thickness', [True, False]), ('subpixel', False)])
def test_invalid_layer_geometry_never_reaches_time_steps(monkeypatch, side, name, value):
    import wavebit._core.scene_fdtd as scene
    monkeypatch.setattr(scene, 'fdtd_grid', lambda *a, **kw: pytest.fail('reached time stepping'))
    with pytest.raises(ValueError, match='thickness|mesh_thickness'):
        wavebit.fdtd(**options(incidence_side=side, **{name: value}))


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_eager_backend_thickness_is_never_silently_detached(backend, side):
    pytest.importorskip('jax' if backend == 1 else 'torch')
    array = time_loop_backend(backend).array
    for thickness in (array([.173, .227]), [array(.173), array(.227)]):
        with pytest.raises(ValueError, match='explicit mesh_thickness'):
            wavebit.fdtd(**options(backend=backend, thickness=thickness, mesh_thickness=None,
                                  incidence_side=side))


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_bottom_integer_lengths_with_explicit_mesh(backend):
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    result, d = wavebit.fdtd(**options(backend=backend, incidence_side='bottom',
                                       thickness=[1, 1], mesh_thickness=[1, 1],
                                       period=1., wavelength=10., resolution=2,
                                       return_diagnostics=True))
    assert d['stable']
    np.testing.assert_array_equal(d['layer_planes'], [0., 1., 2.])
    xp = time_loop_backend(backend).xp
    assert xp.all(xp.isfinite(result.s_inc.T))


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('fitted', [False, True])
def test_fixed_mesh_crossings_and_nominal_forward_compatibility(side, fitted):
    kw = options(incidence_side=side, grading=True if fitted else None,
                 align_layers=fitted, mesh_thickness=[.13, .27], return_diagnostics=True)
    ds = []
    for depth in (.173, .2, .281):
        _, d = wavebit.fdtd(**dict(kw, thickness=[depth, .4-depth]))
        assert d['stable']
        np.testing.assert_allclose(d['layer_planes'], [0., depth, .4], atol=1e-15)
        ds.append(d)
    for key in ('shape', 'spacing', 'dt', 'steps', 'z_nodes', 'z_spacing',
                'source_z', 'pml_bounds', 'reflection_z', 'transmission_z', 'mesh_thickness'):
        for d in ds[1:]:
            np.testing.assert_array_equal(ds[0][key], d[key])
    for key in ('reflection_monitor_z', 'transmission_monitor_z'):
        for d in ds[1:]:
            np.testing.assert_array_equal(ds[0]['diffraction'][key], d['diffraction'][key])
    legacy, _ = wavebit.fdtd(**dict(kw, thickness=[.13, .27], mesh_thickness=None))
    live, _ = wavebit.fdtd(**dict(kw, thickness=[.13, .27]))
    for basis in ('given', 's_inc', 'p_inc'):
        for key in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
            np.testing.assert_allclose(getattr(getattr(legacy, basis), key),
                                       getattr(getattr(live, basis), key), atol=1e-13)


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('side,method', [('top', 'electric'), ('bottom', 'directional')])
def test_joint_layer_material_pole_gradients_in_typed_stack(backend, side, method):
    pattern = np.array([[[1.2, 1.8], [1.5, 1.1]], [[1.6, 1.3], [1.1, 1.7]]])
    def objective(p, which):
        runtime = time_loop_backend(which)
        xp, array = runtime.xp, runtime.array
        stack = wavebit.Stack.from_structure(array(pattern)*p[1], [p[0], .4-p[0]],
                                             mu=array(pattern)*.2+.8,
                                             top=wavebit.Material(p[3], 1.1), bot=1.3,
                                             period=[.3, .4])
        light = wavebit.Incidence(wavelength=[.98, 1.02], theta=.3, phi=.4,
                                  polarization=(1., .6j), incidence_side=side)
        results, d = wavebit.fdtd(stack, light, backend=which, resolution=10,
                                 mesh_indices=(1.1, 1.2), mesh_thickness=[.13, .27],
                                 transverse_resolution=(12., 13.), n_pml=6, n_periods=35,
                                 grading=True, align_layers=True, pulse_width=1.5,
                                 pulse_notches=.7, source_distance=1., diffraction_distance=.3,
                                 dispersion=np.array([[[1, 0], [0, 1]], [[0, 0], [0, 0]]]),
                                 poles={1: [(8., .3, p[2])]}, diffraction_method=method,
                                 truncation_order=[1, 1], return_diagnostics=True,
                                 convergence_window=100, field_monitors={'inside': .1},
                                 record_field=113, record_components=['ex'],
                                 record_bounds={'z': (.05, .15)})
        amplitude = sum(xp.real(.2*r.given.r_s+.3j*r.given.r_p+.4*r.given.t_s+.5j*r.given.t_p).sum()
                        +.1*r.given.T.sum() for r in results)
        return xp.stack((amplitude, .1*xp.mean(xp.abs(d['field_e'])**2)
                         +.2*xp.mean(xp.abs(d['monitors']['inside']['fields']['ey'])**2),
                         .3*xp.sum(d['convergence']['scattering_change_bound']['r_p']**2),
                         d['layer_planes'][1]))
    p = np.array([.173, 1., .8, 1.2])
    value, derivative = value_and_jacobian(lambda p: objective(p, backend), p, backend)
    np.testing.assert_allclose(value, objective(p, 0), atol=2e-10)
    for step in (2e-5, 1e-5):
        finite = np.stack([(objective(p+v, 0)-objective(p-v, 0))/(2*step)
                           for v in np.eye(4)*step], axis=-1)
        np.testing.assert_allclose(derivative, finite, rtol=5e-5, atol=4e-8)
    assert np.min(np.max(abs(finite[:3]), axis=0)) > 1e-6
    np.testing.assert_array_equal(derivative[3], [1., 0., 0., 0.])


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_jit_moves_interfaces_without_retracing_and_masks_invalid_depth(side):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traces = []
    def solve(thickness):
        traces.append(None)
        r, d = wavebit.fdtd(**options(backend=1, thickness=thickness, incidence_side=side,
                                     align_layers=True, return_diagnostics=True,
                                     field_monitors={'inside': .1}, record_field=151,
                                     record_components=['ey'], record_bounds={'z': (.05, .15)}))
        return (r.given.T, d['stable'], d['z_nodes'], d['layer_planes'], d['field_e'],
                d['monitors']['inside']['fields']['ey'])
    compiled = jax.jit(solve)
    a, b = compiled(np.array([.173, .227])), compiled(np.array([.281, .119]))
    assert a[1] and b[1]
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_allclose(b[3], [0., .281, .4])
    assert not np.allclose(a[0], b[0])
    for thickness in ([.2, .3], [.5, -.1], [np.nan, .2], [.4, 1e-16]):
        result = compiled(np.array(thickness))
        assert not result[1]
        for key in (0, 4, 5):
            assert np.isnan(result[key]).all()
    assert len(traces) == 1


def test_thickness_sensitivity_error_envelope_refines_to_airy(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    from studies.fdtd_layer_gradients import airy, scene_vector
    errors = []
    for resolution in (20, 80):
        f = lambda d: scene_vector(d, resolution)
        evaluate = jax.jit(lambda d: (f(d), jax.jacfwd(f)(d)))
        samples = []
        for depth in (.163, .173, .183):
            value, derivative = evaluate(np.array(depth))
            step = 1e-5
            target = (airy(depth+step)-airy(depth-step))/(2*step)
            finer = (airy(depth+step/2)-airy(depth-step/2))/step
            np.testing.assert_allclose(target, finer, atol=1e-8, rtol=0)
            samples.append([np.max(abs(value-airy(depth))), np.max(abs(derivative-target))])
        errors.append(np.max(samples, axis=0))
        for name, error in zip(('value', 'derivative'), errors[-1]):
            record_property(f'{name}_error_{resolution}', float(error))
    assert errors[1][0] < .002 and errors[1][1] < .12
    assert np.all(errors[1] < .25*errors[0])
