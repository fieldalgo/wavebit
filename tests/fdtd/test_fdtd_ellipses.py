"""Rotated ellipse cells against independent implicit-conic chord quadrature."""

from wavebit._common.periodic import Incidence as _legacy_wavebit_Incidence

from wavebit._common.periodic import Stack as _legacy_wavebit_Stack

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd
import numpy as np
import pytest
from scipy.integrate import quad

import wavebit
from wavebit.fdtd._periodic.scene import _shape_partition, _shape_cell_fractions, _require_scene
from wavebit.fdtd._periodic.runtime import time_loop_backend
from tests.fdtd.test_fdtd_scene_angles import value_and_jacobian
from tests.fdtd.test_fdtd_circles import setup, fractions as circle_fractions


def chord_integral(p, x, y, smoothing):
    """Independent quadratic-form roots and Cartesian adaptive quadrature."""
    co, si = np.cos(p[4]), np.sin(p[4])
    rotation = np.array([[co, -si], [si, co]])
    q = rotation @ np.diag(4/np.array(p[2:4])**2) @ rotation.T
    qxx, qxy, qyy = q[0, 0], q[0, 1], q[1, 1]
    determinant = np.linalg.det(q)
    rx, ry = np.sqrt(qyy/determinant), np.sqrt(qxx/determinant)
    def knots(c, smooth):
        return np.unique(c+np.array([-(.1+smooth)/2, -(.1-smooth)/2, (.1-smooth)/2, (.1+smooth)/2]))
    xs, ys = knots(x, smoothing[0]), knots(y, smoothing[1])
    def weight(v, c, smooth):
        if not smooth:
            return float(abs(v-c) <= .05)
        return max(0., min(c+.05, v+smooth/2)-max(c-.05, v-smooth/2))/smooth
    if ys[-1] <= p[1]-ry or ys[0] >= p[1]+ry:
        return 0.
    left, right = max(xs[0], p[0]-rx), min(xs[-1], p[0]+rx)
    if right <= left:
        return 0.
    cuts = [left, right, *xs]
    for v in ys-p[1]:
        disc = qxx-determinant*v*v
        if disc > 0:
            cuts.extend(p[0]+(-qxy*v+np.array([-1, 1])*np.sqrt(disc))/qxx)
    cuts = np.unique(np.clip(cuts, left, right))
    def integrand(u):
        v = u-p[0]
        roots = p[1]+(-qxy*v+np.array([-1, 1])*np.sqrt(max(0., qyy-determinant*v*v)))/qyy
        inner = np.unique(np.clip(ys, *roots))
        mass = sum((b-a)*weight((a+b)/2, y, smoothing[1]) for a, b in zip(inner[:-1], inner[1:]))
        return weight(u, x, smoothing[0])*mass
    # Sub-ulp Cartesian strips have area far below the absolute quadrature
    # budget; adaptive subdivision there can only revisit rounded endpoints.
    return sum(quad(integrand, a, b, epsabs=2e-14, epsrel=2e-12)[0] for a, b in zip(cuts[:-1], cuts[1:])
               if b-a > 8*np.finfo(float).eps*max(1., abs(a), abs(b)))


def oracle(p, smoothing=(0., 0.), offset=(0., 0.)):
    result = np.zeros((8, 10))
    for j, i in np.ndindex(result.shape):
        for a in (-1., 0., 1.):
            for b in (-.8, 0., .8):
                translated = np.asarray(p)+[.5+a, .4+b, 0., 0., 0.]
                result[j, i] += chord_integral(translated, (i+offset[0])*.1, (j+offset[1])*.1, smoothing)/.01
    return result


def fractions(p, backend='numpy', smoothing=(0., 0.), offset=(0., 0.)):
    rt = time_loop_backend(backend)
    _, _, layers = _shape_partition([dict(shapes=[dict(type='ellipse', cx=p[0], cy=p[1],
        lx=p[2], ly=p[3], angle=p[4])])], [1., .8], xp=rt.xp, array=rt.array, check=_require_scene, integration='area')
    return _shape_cell_fractions(layers[0][1][0], (np.arange(10)+offset[0])*.1,
        (np.arange(8)+offset[1])*.1, .1, .1, [1., .8], xp=rt.xp, array=rt.array, smoothing=smoothing)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('angle', [0., .37, np.pi/2, -.63])
@pytest.mark.parametrize('smoothing', [(0., 0.), (.07, .09)])
def test_cell_areas_match_independent_implicit_conic_integrals(backend, angle, smoothing):
    setup(backend)
    p = np.array([.431, -.369, .413, .267, angle])
    for offset in ((0., 0.), (.5, .5)):
        value = np.asarray(fractions(p, backend, smoothing, offset))
        np.testing.assert_allclose(value, oracle(p, smoothing, offset), atol=3e-12, rtol=0.)
        np.testing.assert_allclose(value.sum()*.01, np.pi*p[2]*p[3]/4, atol=3e-13)
        assert value.min() >= 0. and value.max() <= 1.
        np.testing.assert_allclose(fractions(p+[2., -1.6, 0., 0., 0.], backend, smoothing, offset), value, atol=3e-12)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('smoothing', [(0., 0.), (.07, .09)])
def test_five_geometry_derivatives_match_independent_chords(backend, smoothing):
    setup(backend)
    p = np.array([.431, -.369, .413, .267, .37])
    _, jac = value_and_jacobian(lambda p: fractions(p, backend, smoothing).reshape(-1), p, backend)
    for h in (2e-6, 1e-6):
        finite = np.stack([(oracle(p+d, smoothing)-oracle(p-d, smoothing))/(2*h) for d in np.eye(5)*h], axis=-1)
        np.testing.assert_allclose(jac, finite.reshape(80, 5), atol=3e-7, rtol=3e-6)
    np.testing.assert_allclose(jac.sum(axis=0)*.01, [0., 0., np.pi*p[3]/4, np.pi*p[2]/4, 0.], atol=3e-12)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('smoothing', [(0., 0.), (1e-9, 1e-9), (.1, .1), (1., .8)])
def test_circle_limit_and_narrow_full_period_kernels_preserve_gradients(backend, smoothing):
    setup(backend)
    p = np.array([.431, -.369, .4])
    def f(p):
        return fractions([p[0], p[1], p[2], p[2], .53], backend, smoothing).reshape(-1)
    value, jac = value_and_jacobian(f, p, backend)
    expected, target = value_and_jacobian(lambda p: circle_fractions(p, backend, smoothing).reshape(-1), p, backend)
    np.testing.assert_allclose(value, expected, atol=3e-11)
    np.testing.assert_allclose(jac, target, atol=3e-10)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('p', [[-.05, -.05, .4, .2, 0.], [0., 0., .4, .8, 0.], [0., 0., .8, .4, np.pi/2]])
def test_axes_tangencies_and_quarter_turn_keep_finite_total_derivatives(backend, p):
    setup(backend)
    value, jac = value_and_jacobian(lambda p: fractions(p, backend).reshape(-1), np.array(p), backend)
    assert np.isfinite(jac).all()
    np.testing.assert_allclose(value, oracle(p).reshape(-1), atol=3e-12)
    np.testing.assert_allclose(jac.sum(axis=0)*.01, [0., 0., np.pi*p[3]/4, np.pi*p[2]/4, 0.], atol=3e-11)


def response(p, backend='numpy', diagnostics=False, side='bottom', smoothing=0.):
    stack = _legacy_wavebit_Stack.from_structure([
        dict(background=wavebit.Material(1.2, 1.03), shapes=[dict(type='ellipse', cx=p[0], cy=p[1],
            lx=p[2], ly=p[3], angle=p[4], material=wavebit.Material(p[5], p[6]))]),
        dict(background=1.4)], [p[7], .3-p[7]], top=1., bot=1.4, period=[.4, .3])
    result, out = _legacy_wavebit_fdtd(stack, _legacy_wavebit_Incidence(wavelength=.9, theta=.17, phi=.23,
        polarization=(1., .4j), incidence_side=side), backend=backend, resolution=9,
        transverse_resolution=[12., 15.], n_pml=6, n_periods=38, pulse_width=1.5,
        truncation_order=0, mesh_thickness=[.15, .15], layer_smoothing=.025, align_layers=True,
        shape_integration='area', shape_smoothing=smoothing, diffraction_method='directional', return_diagnostics=True)
    xp = time_loop_backend(backend).xp
    value = xp.concat([xp.real(result.incident.r_s).reshape(-1), xp.imag(result.incident.t_p).reshape(-1)])
    return (value, out) if diagnostics else value


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('smoothing', [0., (.04, .03)])
@pytest.mark.slow
@pytest.mark.shard_unit
def test_bottom_conical_joint_eight_parameter_shape_material_and_depth_graph(backend, smoothing):
    setup(backend)
    p = np.array([.13, -.09, .18, .11, .37, 2.4, 1.13, .14])
    value, jac = value_and_jacobian(lambda p: response(p, backend, smoothing=smoothing), p, backend)
    expected, mesh = response(p, diagnostics=True, smoothing=smoothing)
    np.testing.assert_allclose(value, expected, atol=3e-11)
    for h in (2e-5, 1e-5):
        finite = np.stack([(response(p+d, smoothing=smoothing)-response(p-d, smoothing=smoothing))/(2*h)
                            for d in np.eye(8)*h], axis=-1)
        np.testing.assert_allclose(jac, finite, atol=5e-8, rtol=6e-5)
    assert np.min(np.max(abs(jac), axis=0)) > 1e-5
    _, moved = response(p+[.01, -.01, .01, .01, .03, .1, .01, .01], diagnostics=True, smoothing=smoothing)
    for key in ('z_nodes', 'z_spacing', 'dt', 'steps', 'source_z', 'pml_bounds', 'mesh_thickness'):
        np.testing.assert_array_equal(mesh[key], moved[key])


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_single_trace_masks_invalid_projected_extent_and_accepts_rotation(side):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traces = []
    def f(p):
        traces.append(None)
        value, d = response(p, 'jax', diagnostics=True, side=side)
        return value, d['stable']
    run = jax.jit(f)
    p = np.array([.13, -.09, .18, .11, .37, 2.4, 1.13, .14])
    value, stable = run(p)
    assert stable
    translated, valid = run(p+[.8, -.6, 0., 0., np.pi, 0., 0., 0.])
    assert valid
    np.testing.assert_allclose(value, translated, atol=3e-13)
    moved, valid = run(p+[0., 0., 0., 0., .23, 0., 0., 0.])
    assert valid and np.max(abs(moved-value)) > 1e-6
    for index, bad in [(2, .6), (3, 0.), (4, np.nan), (5, -1.)]:
        invalid = p.copy(); invalid[index] = bad
        result, valid = run(invalid)
        assert not valid and np.isnan(result).all()
    assert len(traces) == 1


@pytest.mark.parametrize('shape,integration,message', [
    (dict(type='ellipse', lx=.2, ly=.1), 'separable', 'require'),
    (dict(type='ellipse', lx=.4, ly=.1, angle=np.pi/2), 'area', 'extents'),
    (dict(type='ellipse', lx=.2, ly=0.), 'area', 'positive'),
    (dict(type='ellipse', lx=.2, ly=.1, angle=np.nan), 'area', 'finite')])
def test_ellipse_recipe_contract(shape, integration, message):
    rt = time_loop_backend('numpy')
    with pytest.raises(ValueError, match=message):
        _shape_partition([dict(shapes=[shape])], [.4, .3], xp=rt.xp, array=rt.array,
                         check=_require_scene, integration=integration)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_swapping_principal_axes_and_quarter_turn_preserves_same_physical_shape(backend):
    setup(backend)
    p = np.array([.41, -.36, .47, .23, .37])
    moved = p[[0, 1, 3, 2, 4]]+[0., 0., 0., 0., np.pi/2]
    for smoothing in [(0., 0.), (.07, .09)]:
        np.testing.assert_allclose(fractions(p, backend, smoothing), fractions(moved, backend, smoothing), atol=3e-12)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_projection_bound_accepts_axis_longer_than_its_unrotated_period(backend):
    setup(backend)
    rt = time_loop_backend(backend)
    _, _, layers = _shape_partition([dict(shapes=[dict(type='ellipse', lx=.9, ly=.4, angle=np.pi/2)])],
        [.6, 1.], xp=rt.xp, array=rt.array, check=_require_scene, integration='area')
    value = _shape_cell_fractions(layers[0][1][0], np.arange(6)*.1, np.arange(10)*.1, .1, .1,
        [.6, 1.], xp=rt.xp, array=rt.array)
    np.testing.assert_allclose(np.asarray(value).sum()*.01, np.pi*.9*.4/4, atol=3e-13)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('smoothing', [(0., 0.), (1e-9, 1e-9)])
def test_thin_rotated_ellipse_keeps_area_and_axis_sensitivities(backend, smoothing):
    setup(backend)
    p = np.array([.431, -.369, .6, .006, .63])
    value, jac = value_and_jacobian(lambda p: fractions(p, backend, smoothing).reshape(-1), p, backend)
    np.testing.assert_allclose(value, oracle(p).reshape(-1), atol=3e-11)
    assert np.isfinite(jac).all()
    np.testing.assert_allclose(value.sum()*.01, np.pi*p[2]*p[3]/4, atol=3e-13)
    np.testing.assert_allclose(jac.sum(axis=0)*.01, [0., 0., np.pi*p[3]/4, np.pi*p[2]/4, 0.], atol=3e-10)
