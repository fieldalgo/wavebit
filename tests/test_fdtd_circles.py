"""Disk cells against independent chord quadrature; full-graph shape derivatives."""
import numpy as np
import pytest
from scipy.integrate import quad

import wavebit
from wavebit._core.scene_fdtd import _shape_partition, _shape_cell_fractions, _require_scene
from wavebit.backends.time_domain import time_loop_backend
from test_fdtd_scene_angles import value_and_jacobian


def chord_integral(cx, cy, radius, x, y, dx, dy, smoothing):
    """Integrate disk chords with independently constructed trapezoid kernels."""
    def knots(c, width, smooth):
        return np.unique(c+np.array([-(width+smooth)/2, -(width-smooth)/2,
                                     (width-smooth)/2, (width+smooth)/2]))
    xknots, yknots = knots(x, dx, smoothing[0]), knots(y, dy, smoothing[1])
    def weight(t, c, width, smooth):
        if not smooth:
            return float(abs(t-c) <= width/2)
        return max(0., min(c+width/2, t+smooth/2)-max(c-width/2, t-smooth/2))/smooth
    def inner(lo, hi):
        cuts = np.unique(np.clip(yknots, lo, hi))
        return sum((b-a)*weight((a+b)/2, y, dy, smoothing[1]) for a, b in zip(cuts[:-1], cuts[1:]))
    left, right = max(cx-radius, xknots[0]), min(cx+radius, xknots[-1])
    if right <= left:
        return 0.
    cuts = [left, right, *xknots]
    for v in yknots-cy:
        if abs(v) < radius:
            d = np.sqrt(radius**2-v**2)
            cuts.extend([cx-d, cx+d])
    cuts = np.unique(np.clip(cuts, left, right))
    def integrand(u):
        half = np.sqrt(max(0., radius**2-(u-cx)**2))
        return weight(u, x, dx, smoothing[0])*inner(cy-half, cy+half)
    return sum(quad(integrand, a, b, epsabs=2e-14, epsrel=2e-12)[0] for a, b in zip(cuts[:-1], cuts[1:]))


def oracle(p, smoothing=(0., 0.), offset=(0., 0.)):
    result = np.zeros((8, 10))
    for j, i in np.ndindex(result.shape):
        for a in (-1., 0., 1.):
            for b in (-.8, 0., .8):
                result[j, i] += chord_integral(p[0]+.5+a, p[1]+.4+b, p[2]/2,
                    (i+offset[0])*.1, (j+offset[1])*.1, .1, .1, smoothing)/.01
    return result


def fractions(p, backend='numpy', smoothing=(0., 0.), offset=(0., 0.)):
    runtime = time_loop_backend(backend)
    _, _, layers = _shape_partition([dict(shapes=[dict(type='circle', cx=p[0], cy=p[1], lx=p[2])])],
        [1., .8], xp=runtime.xp, array=runtime.array, check=_require_scene, integration='area')
    return _shape_cell_fractions(layers[0][1][0], (np.arange(10)+offset[0])*.1,
        (np.arange(8)+offset[1])*.1, .1, .1, [1., .8], xp=runtime.xp, array=runtime.array, smoothing=smoothing)


def setup(backend):
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('smoothing', [(0., 0.), (.07, .09), (0., .08)])
def test_disk_cells_match_independent_chords_and_periodic_area(backend, smoothing):
    setup(backend)
    p = np.array([.431, -.369, .413])
    for offset in ((0., 0.), (.5, .5)):
        value = np.asarray(fractions(p, backend, smoothing, offset))
        np.testing.assert_allclose(value, oracle(p, smoothing, offset), atol=3e-12, rtol=0)
        np.testing.assert_allclose(value.sum()*.01, np.pi*p[2]**2/4, atol=2e-13)
        assert value.min() >= 0. and value.max() <= 1.
        np.testing.assert_allclose(fractions(p+[2., -1.6, 0.], backend, smoothing, offset), value, atol=3e-12)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('smoothing', [(0., 0.), (.07, .09)])
def test_geometry_jacobian_matches_chord_finite_differences(backend, smoothing):
    setup(backend)
    p = np.array([.431, -.369, .413])
    value, jac = value_and_jacobian(lambda p: fractions(p, backend, smoothing).reshape(-1), p, backend)
    np.testing.assert_allclose(value, oracle(p, smoothing).reshape(-1), atol=3e-12)
    for h in (2e-6, 1e-6):
        finite = np.stack([(oracle(p+d, smoothing)-oracle(p-d, smoothing))/(2*h) for d in np.eye(3)*h], axis=-1)
        np.testing.assert_allclose(jac, finite.reshape(80, 3), atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(jac.sum(axis=0)*.01, [0., 0., np.pi*p[2]/2], atol=3e-12)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('p', [[-.05, -.05, .4], [0., 0., .8], [.5, -.4, .4]])
def test_axes_tangencies_and_maximum_diameter_keep_finite_correct_first_derivatives(backend, p):
    setup(backend)
    p = np.array(p)
    value, jac = value_and_jacobian(lambda p: fractions(p, backend).reshape(-1), p, backend)
    np.testing.assert_allclose(value, oracle(p).reshape(-1), atol=2e-13)
    assert np.isfinite(jac).all()
    np.testing.assert_allclose(jac.sum(axis=0)*.01, [0., 0., np.pi*p[2]/2], atol=3e-12)
    # Independent centred differences cross quadrant axes and tangent contacts.
    h = 1e-7
    finite = np.stack([(oracle(p+d)-oracle(p-d))/(2*h) for d in np.eye(3)*h], axis=-1)
    np.testing.assert_allclose(jac, finite.reshape(80, 3), atol=8e-3, rtol=2e-3)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_full_period_translation_kernel_gives_exact_constant_and_area_gradient(backend):
    setup(backend)
    p = np.array([.49, -.391, .8])
    if backend != 'numpy':
        value, jac = value_and_jacobian(lambda p: fractions(p, backend, (1., .8)).reshape(-1), p, backend)
        np.testing.assert_allclose(jac, np.broadcast_to([0., 0., np.pi*p[2]/(2*.8)], jac.shape), atol=2e-11)
    else:
        value = fractions(p, backend, (1., .8))
    np.testing.assert_allclose(value, np.pi*p[2]**2/(4*.8), atol=3e-12)


def response(p, backend='numpy', diagnostics=False, side='bottom', smoothing=0.):
    stack = wavebit.Stack.from_structure([
        dict(background=wavebit.Material(1.2, 1.03), shapes=[dict(type='disk',
            cx=p[0], cy=p[1], l=p[2], material=wavebit.Material(p[3], p[4]))]),
        dict(background=1.4)], [p[5], .3-p[5]], top=1., bot=1.4, period=[.4, .3])
    result, out = wavebit.fdtd(stack, wavebit.Incidence(wavelength=.9, theta=.17, phi=.23,
        polarization=(1., .4j), incidence_side=side), backend=backend, resolution=9,
        transverse_resolution=[12., 15.], n_pml=6, n_periods=38, pulse_width=1.5,
        truncation_order=0, mesh_thickness=[.15, .15], layer_smoothing=.025, align_layers=True,
        shape_integration='area', shape_smoothing=smoothing, diffraction_method='directional', return_diagnostics=True)
    xp = time_loop_backend(backend).xp
    value = xp.concat([xp.real(result.incident.r_s).reshape(-1), xp.imag(result.incident.t_p).reshape(-1)])
    return (value, out) if diagnostics else value


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('smoothing', [0., (.04, .03)])
def test_bottom_conical_joint_shape_material_and_thickness_derivatives(backend, smoothing):
    setup(backend)
    p = np.array([.13, -.09, .16, 2.4, 1.13, .14])
    value, jac = value_and_jacobian(lambda p: response(p, backend, smoothing=smoothing), p, backend)
    expected, mesh = response(p, diagnostics=True, smoothing=smoothing)
    np.testing.assert_allclose(value, expected, atol=3e-11)
    for h in (2e-5, 1e-5):
        finite = np.stack([(response(p+d, smoothing=smoothing)-response(p-d, smoothing=smoothing))/(2*h)
                            for d in np.eye(6)*h], axis=-1)
        np.testing.assert_allclose(jac, finite, atol=4e-8, rtol=5e-5)
    assert np.min(np.max(abs(jac), axis=0)) > 1e-5
    _, moved = response(p+[.01, -.01, .01, .1, .01, .01], diagnostics=True, smoothing=smoothing)
    for key in ('z_nodes', 'z_spacing', 'dt', 'steps', 'source_z', 'pml_bounds', 'mesh_thickness'):
        np.testing.assert_array_equal(mesh[key], moved[key])


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_single_trace_accepts_periodic_motion_and_masks_invalid_circle(side):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traces = []
    def run(p):
        traces.append(None)
        value, out = response(p, 'jax', diagnostics=True, side=side)
        return value, out['stable']
    run = jax.jit(run)
    p = np.array([.13, -.09, .16, 2.4, 1.13, .14])
    value, stable = run(p)
    assert stable
    translated, valid = run(p+[.8, -.6, 0., 0., 0., 0.])
    assert valid
    np.testing.assert_allclose(value, translated, atol=2e-13)
    for index, bad in ((0, np.nan), (2, .31), (2, 0.), (3, -1.)):
        invalid = p.copy(); invalid[index] = bad
        value, valid = run(invalid)
        assert not valid and np.isnan(value).all()
    assert len(traces) == 1


@pytest.mark.parametrize('shape, integration, message', [
    (dict(type='circle', lx=.2), 'separable', 'require'),
    (dict(type='ellipse', lx=.2, ly=.1), 'separable', 'require'),
    (dict(type='circle', lx=.2, ly=.2), 'area', 'one length'),
    (dict(type='circle', lx=0.), 'area', 'positive'),
    (dict(type='disk', lx=.31), 'area', 'diameter'),
    (dict(type='circle', lx=.2, angle=np.nan), 'area', 'finite'),
])
def test_circle_recipe_contract(shape, integration, message):
    runtime = time_loop_backend('numpy')
    with pytest.raises(ValueError, match=message):
        _shape_partition([dict(shapes=[shape])], [.4, .3], xp=runtime.xp, array=runtime.array,
                         check=_require_scene, integration=integration)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('width', [1e-9, .1-1e-10, .1, .1+1e-10])
def test_narrow_translation_windows_remain_well_conditioned(backend, width):
    setup(backend)
    p = np.array([.431, -.369, .413])
    smooth = (width, width)
    value = np.asarray(fractions(p, backend, smooth))
    target = oracle(p, (0., 0.) if width < 1e-8 else (.1, .1))
    np.testing.assert_allclose(value, target, atol=2e-9, rtol=0.)
    np.testing.assert_allclose(value.sum()*.01, np.pi*p[2]**2/4, atol=2e-13)
    if backend != 'numpy':
        _, jac = value_and_jacobian(lambda p: fractions(p, backend, smooth).reshape(-1), p, backend)
        assert np.isfinite(jac).all()
        np.testing.assert_allclose(jac.sum(axis=0)*.01, [0., 0., np.pi*p[2]/2], atol=2e-11)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_translation_width_equal_to_cell_with_rounded_zero_middle_piece(backend):
    setup(backend)
    rt = time_loop_backend(backend)
    dx, p = .6/12, np.array([.371, .253, .15])
    def f(p):
        return _shape_cell_fractions(rt.array(p), np.arange(12)*dx, np.arange(12)*dx, dx, dx,
            [.6, .6], xp=rt.xp, array=rt.array, smoothing=(.05, .05)).reshape(-1)
    if backend != 'numpy':
        value, jac = value_and_jacobian(f, p, backend)
        assert np.isfinite(jac).all()
        np.testing.assert_allclose(jac.sum(axis=0)*dx**2, [0., 0., 2*np.pi*p[2]], atol=2e-11)
    else:
        value = f(p)
    assert np.isfinite(value).all()
    np.testing.assert_allclose(value.sum()*dx**2, np.pi*p[2]**2, atol=2e-13)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_circle_angle_is_finite_validated_and_geometrically_inert(backend):
    setup(backend)
    rt = time_loop_backend(backend)
    def area(angle):
        _, _, layers = _shape_partition([dict(shapes=[dict(type='disk', l=.4, angle=angle[0])])],
            [1., .8], xp=rt.xp, array=rt.array, check=_require_scene, integration='area')
        return _shape_cell_fractions(layers[0][1][0], np.arange(10)*.1, np.arange(8)*.1, .1, .1,
            [1., .8], xp=rt.xp, array=rt.array).reshape(-1)
    value, jac = value_and_jacobian(area, np.array([.73]), backend)
    np.testing.assert_allclose(value, oracle([0., 0., .4]).reshape(-1), atol=2e-13)
    np.testing.assert_array_equal(jac, 0.)
