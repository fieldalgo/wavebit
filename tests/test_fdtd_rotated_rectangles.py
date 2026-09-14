"""Independent area integration and full-graph rotation derivatives on fixed meshes."""
import numpy as np
import pytest

import wavebit
from wavebit._core.scene_fdtd import _shape_partition, _shape_cell_fractions, _require_scene
from wavebit.backends.time_domain import time_loop_backend
from test_fdtd_scene_angles import value_and_jacobian


def vertices_for(p):
    """Independent physical vertices: centre-relative recipe to a 1 by .8 cell."""
    c, s = np.cos(p[4]), np.sin(p[4])
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*p[2:4]/2
    return local @ np.array([[c, s], [-s, c]])+p[:2]+[.5, .4]


def vertical_integral(vertices, left, right, bottom, top):
    """Exact midpoint integration between all vertical-slice slope changes.

    Independent of polygon clipping and shoelace area. Include intersections
    with the horizontal cell faces before integrating each linear slice width.
    """
    edges = list(zip(vertices, np.roll(vertices, -1, axis=0)))
    cuts = [left, right, *vertices[:, 0]]
    for a, b in edges:
        if a[1] != b[1]:
            for y in (bottom, top):
                t = (y-a[1])/(b[1]-a[1])
                if 0 < t < 1:
                    cuts.append(a[0]+t*(b[0]-a[0]))
    cuts = np.unique(np.clip(cuts, left, right))
    area = 0.
    for l, r in zip(cuts[:-1], cuts[1:]):
        x = (l+r)/2
        y = [a[1]+(x-a[0])*(b[1]-a[1])/(b[0]-a[0]) for a, b in edges
             if min(a[0], b[0]) < x < max(a[0], b[0])]
        if y:
            area += (r-l)*max(0., min(top, max(y))-max(bottom, min(y)))
    return area


def oracle(p, offset=(0., 0.)):
    vertices = vertices_for(p)
    result = np.zeros((8, 10))
    for j, i in np.ndindex(result.shape):
        x, y = (i+offset[0])*.1, (j+offset[1])*.1
        for a in (-1., 0., 1.):
            for b in (-.8, 0., .8):
                result[j, i] += vertical_integral(vertices+[a, b], x-.05, x+.05, y-.05, y+.05)/.01
    return result


def fractions(p, backend=0, offset=(0., 0.)):
    runtime = time_loop_backend(backend)
    _, _, layers = _shape_partition([dict(background=1., shapes=[
        dict(cx=p[0], cy=p[1], lx=p[2], ly=p[3], angle=p[4], material=2.)])],
        [1., .8], xp=runtime.xp, array=runtime.array, check=_require_scene, integration='area')
    return _shape_cell_fractions(layers[0][1][0], (np.arange(10)+offset[0])*.1,
                                     (np.arange(8)+offset[1])*.1, .1, .1, [1., .8],
                                     xp=runtime.xp, array=runtime.array)


@pytest.mark.parametrize('backend', [0, 1, 2])
@pytest.mark.parametrize('angle', [0., .31, np.pi/2, -.63])
def test_exact_cell_area_matches_independent_vertical_integration(backend, angle):
    if backend == 1:
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    elif backend == 2:
        pytest.importorskip('torch')
    p = np.array([.47, -.371, .4, .3, angle])
    for offset in ((0., 0.), (.5, .5)):
        result = np.asarray(fractions(p, backend, offset))
        np.testing.assert_allclose(result, oracle(p, offset), atol=2e-14, rtol=0)
        np.testing.assert_allclose(result.sum()*.01, .12, atol=4e-15)
        assert result.min() >= -1e-14 and result.max() <= 1+1e-14
        translated = np.asarray(fractions(p+[2., -1.6, 0., 0., 0.], backend, offset))
        np.testing.assert_allclose(result, translated, atol=2e-14)


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('angle', [0., .31, np.pi/2])
def test_cell_shape_jacobian_matches_independent_area_and_exact_total(backend, angle):
    pytest.importorskip('jax' if backend == 1 else 'torch')
    p = np.array([.473, -.369, .413, .287, angle])
    value, jac = value_and_jacobian(lambda p: fractions(p, backend).reshape(-1), p, backend)
    np.testing.assert_allclose(value, oracle(p).reshape(-1), atol=2e-14)
    for step in (2e-6, 1e-6):
        finite = np.stack([(oracle(p+d)-oracle(p-d))/(2*step) for d in np.eye(5)*step], axis=-1).reshape(80, 5)
        np.testing.assert_allclose(jac, finite, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(jac.sum(axis=0)*.01, [0., 0., p[3], p[2], 0.], atol=2e-14)


def scene(p):
    return wavebit.Stack.from_structure([
        dict(background=wavebit.Material(1.2, 1.03), shapes=[
            dict(cx=p[0], cy=p[1], lx=p[2], ly=p[3], angle=p[4], material=wavebit.Material(p[5], p[6]))]),
        dict(background=1.4, shapes=[])], [p[7], .3-p[7]], top=1., bot=1.4, period=[.4, .3])


def response(p, backend=0, diagnostics=False, side='bottom', smoothing=0.):
    result, out = wavebit.fdtd(scene(p), wavebit.Incidence(wavelength=.9, theta=.17, phi=.23,
                             polarization=(1., .4j), incidence_side=side), backend=backend,
                             resolution=9, transverse_resolution=[12., 15.], n_pml=6, n_periods=38,
                             pulse_width=1.5, truncation_order=0, mesh_thickness=[.15, .15],
                             layer_smoothing=.025, align_layers=True, shape_integration='area', shape_smoothing=smoothing,
                             diffraction_method='directional', return_diagnostics=True)
    xp = time_loop_backend(backend).xp
    value = xp.concat([xp.real(result.given.r_s).reshape(-1), xp.imag(result.given.t_p).reshape(-1)])
    return (value, out) if diagnostics else value


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.slow
def test_bottom_conical_jones_joint_eight_parameter_derivatives(backend):
    pytest.importorskip('jax' if backend == 1 else 'torch')
    p = np.array([.13, -.09, .16, .12, .37, 2.4, 1.13, .14])
    value, jac = value_and_jacobian(lambda p: response(p, backend), p, backend)
    expected, mesh = response(p, diagnostics=True)
    np.testing.assert_allclose(value, expected, atol=2e-11)
    for step in (2e-5, 1e-5):
        finite = np.stack([(response(p+d)-response(p-d))/(2*step) for d in np.eye(8)*step], axis=-1)
        np.testing.assert_allclose(jac, finite, atol=3e-8, rtol=5e-5)
    assert np.min(np.max(abs(jac), axis=0)) > 1e-5
    _, moved = response(p+[.01, -.01, .01, .01, .05, .1, .01, .01], diagnostics=True)
    for key in ('z_nodes', 'z_spacing', 'dt', 'steps', 'source_z', 'pml_bounds', 'mesh_thickness'):
        np.testing.assert_array_equal(mesh[key], moved[key])
    assert mesh['shape_integration'] == 'area'


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('smoothing', [0., (.04, .03)])
def test_one_jit_trace_handles_rotation_translation_and_invalid_extent(side, smoothing):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traces = []
    def run(p):
        traces.append(None)
        value, out = response(p, 1, diagnostics=True, side=side, smoothing=smoothing)
        return value, out['stable']
    run = jax.jit(run)
    p = np.array([.13, -.09, .16, .12, 0., 2.4, 1.13, .14])
    value, stable = run(p)
    assert stable
    translated, valid = run(p+[.8, -.6, 0., 0., 2*np.pi, 0., 0., 0.])
    assert valid
    np.testing.assert_allclose(value, translated, atol=2e-13)
    moved, valid = run(p+[0., 0., 0., 0., .37, 0., 0., 0.])
    assert valid and np.max(abs(moved-value)) > 1e-5
    for index, bad in ((4, np.nan), (2, .5), (5, -1.)):
        invalid = p.copy(); invalid[index] = bad
        value, valid = run(invalid)
        assert not valid and np.isnan(value).all()
    assert len(traces) == 1


@pytest.mark.parametrize('change', [dict(shape_integration='unknown'), dict(shape_smoothing=.5),
                                   dict(subpixel=False), dict(period=.4), dict(mu=np.ones((1, 1, 1)))])
def test_area_option_contract_rejects_unsupported_combinations(change):
    options = dict(structure=[dict(background=1., shapes=[dict(lx=.2, ly=.1, angle=.3, material=2.)])],
                   thickness=[.2], top=1., bot=1., period=[.4, .3], wavelength=1., backend=0,
                   resolution=10, n_pml=5, n_periods=30, truncation_order=0, shape_integration='area')
    with pytest.raises(ValueError):
        wavebit.fdtd(**dict(options, **change))


def test_area_rejects_projected_self_overlap():
    runtime = time_loop_backend(0)
    with pytest.raises(ValueError, match='extents'):
        _shape_partition([dict(shapes=[dict(lx=.9, ly=.7, angle=.7)])], [1., .8],
                         xp=runtime.xp, array=runtime.array, check=_require_scene, integration='area')
