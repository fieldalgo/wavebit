"""Exact periodic geometry and derivatives for shared rectangle recipes."""
import numpy as np
import pytest

import wavebit
from wavebit.backends.time_domain import time_loop_backend
from wavebit._core.scene_fdtd import _rectangle_partition, _periodic_shape_weights, _require_scene
from test_fdtd_scene_angles import value_and_jacobian


def partition(scene, backend=0, periods=(1., .8)):
    if backend == 1:
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    runtime = time_loop_backend(backend)
    return _rectangle_partition(scene, periods, xp=runtime.xp, array=runtime.array, check=_require_scene)


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_periodic_painting_integrates_known_areas_and_preserves_materials(backend):
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    # Both boxes wrap x. Their widths are .4/.3, common height .4, overlap .15*.4.
    scene = [dict(background=wavebit.Material(1., 1.1), shapes=[
        dict(lx=.4, ly=.4, cx=.4, material=wavebit.Material(4., 1.3)),
        dict(lx=.3, ly=.4, cx=-.4, material=wavebit.Material(7., 1.7))])]
    eps, mu, (x, y) = partition(scene, backend)
    eps, mu, x, y = map(np.asarray, (eps, mu, x, y))
    areas = np.diff(y)[:, None]*np.diff(x)[None, :]
    expected = 1+(3*.4*.4+6*.3*.4-3*.15*.4)/.8
    expected_mu = 1.1+(.2*.4*.4+.6*.3*.4-.2*.15*.4)/.8
    np.testing.assert_allclose(np.sum(eps[0]*areas)/.8, expected, atol=2e-15)
    np.testing.assert_allclose(np.sum(mu[0]*areas)/.8, expected_mu, atol=2e-15)
    assert set(np.unique(eps)) <= {1., 4., 7.}
    moved = [dict(scene[0], shapes=[dict(s, cx=s['cx']+2, cy=-1.6) for s in scene[0]['shapes']])]
    a, b, edges = partition(moved, backend)
    # Zero-area partition cells have no physical material; their midpoint can
    # pick either side of a coincident face after roundoff in periodic folding.
    for actual, expected in zip((np.asarray(a)*areas, np.asarray(b)*areas, *edges), (eps*areas, mu*areas, x, y)):
        np.testing.assert_allclose(actual, expected, atol=2e-15)


def independent_interval_average(centres, width, edges, smoothing):
    """Integrate shifted sharp intervals by exact midpoint quadrature on each kink-free piece."""
    result = np.zeros((len(centres), len(edges)-1))
    for i, centre in enumerate(centres):
        left, right = centre-width/2, centre+width/2
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            for image in (-1., 0., 1.):
                a1, b1 = a+image, b+image
                if smoothing:
                    cuts = np.unique(np.clip([-.5*smoothing, .5*smoothing, left-a1, right-a1,
                                               left-b1, right-b1], -.5*smoothing, .5*smoothing))
                    for start, end in zip(cuts[:-1], cuts[1:]):
                        shift = (start+end)/2
                        result[i, j] += max(0., min(right, b1+shift)-max(left, a1+shift))*(end-start)/(width*smoothing)
                else:
                    result[i, j] += max(0., min(right, b1)-max(left, a1))/width
    return result


@pytest.mark.parametrize('backend', [0, 1, 2])
@pytest.mark.parametrize('smoothing', [0., .08, 1.])
def test_periodic_cell_integrals_match_independent_displacement_integration(backend, smoothing):
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    if backend == 1:
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    runtime = time_loop_backend(backend)
    centres = np.arange(10)/10
    edges = np.array([0., .013, .27, .27, .891, 1.])
    expected = independent_interval_average(centres, .1, edges, smoothing)
    actual = _periodic_shape_weights(centres, .1, runtime.array(edges), 1., smoothing,
                                     xp=runtime.xp, array=runtime.array)
    np.testing.assert_allclose(actual, expected, atol=4e-15, rtol=0)
    np.testing.assert_allclose(np.asarray(actual).sum(axis=1), 1., atol=3e-15)
    np.testing.assert_allclose(np.asarray(actual).sum(axis=0)*.1, np.diff(edges), atol=3e-15)


def scene_for(p):
    return wavebit.Stack.from_structure([
        dict(background=wavebit.Material(1.3, 1.02), shapes=[
            dict(lx=p[2], ly=p[3], cx=p[0], cy=p[1], material=wavebit.Material(p[4], p[5])),
            dict(lx=.1, ly=.09, cx=.07, cy=.015, material=wavebit.Material(1.8, 1.06))]),
        dict(background=wavebit.Material(1.5, 1.04), shapes=[])],
        [p[6], .3-p[6]], top=1.2, bot=wavebit.Material(1.5, 1.04), period=[.4, .3])


def solve_joint(p, backend, *, diagnostics=False):
    xp = time_loop_backend(backend).xp
    result, out = wavebit.fdtd(scene_for(p), wavebit.Incidence(wavelength=.9, theta=.16, phi=.2,
                                polarization=(1., .5j), incidence_side='bottom'),
                               backend=backend, resolution=10, transverse_resolution=[12., 16.],
                               n_pml=6, n_periods=40, pulse_width=1.5, truncation_order=0,
                               mesh_thickness=[.15]*2, layer_smoothing=.03, shape_smoothing=[.03, .025],
                               align_layers=True, diffraction_method='directional', return_diagnostics=True)
    value = (result.given.R+.1*xp.real(result.given.r_s)-.03*xp.real(result.given.t_p)).reshape(1)
    return (value, out) if diagnostics else value


@pytest.mark.parametrize('backend', [1, 2])
def test_bottom_joint_shape_material_and_thickness_gradients(backend):
    pytest.importorskip('jax' if backend == 1 else 'torch')
    p = np.array([.16, -.093, .15, .14, 2.4, 1.2, .14])
    actual, gradient = value_and_jacobian(lambda p: solve_joint(p, backend), p, backend)
    expected, mesh = solve_joint(p, 0, diagnostics=True)
    np.testing.assert_allclose(actual, expected, atol=2e-11)
    for step in (2e-5, 1e-5):
        finite = np.stack([(solve_joint(p+v, 0)-solve_joint(p-v, 0))/(2*step) for v in np.eye(7)*step], axis=-1)
        np.testing.assert_allclose(gradient, finite, atol=3e-8, rtol=4e-5)
    assert np.min(abs(gradient)) > 1e-5
    _, moved = solve_joint(p+np.array([.03, -.02, .02, -.01, .1, -.03, .01]), 0, diagnostics=True)
    for key in ('z_nodes', 'z_spacing', 'dt', 'steps', 'source_z', 'pml_bounds', 'mesh_thickness'):
        np.testing.assert_array_equal(mesh[key], moved[key])


@pytest.mark.parametrize('change', [dict(type='circle'), dict(angle=.1), dict(lx=0.), dict(ly=-.1),
                                   dict(cx=np.nan), dict(lx=1.1), dict(material=0.), dict(unknown=1)])
def test_invalid_or_unsupported_recipes_raise(change):
    with pytest.raises(ValueError):
        partition([dict(background=1., shapes=[dict(dict(lx=.2, ly=.3, material=2.), **change)])])


def test_paint_contact_has_known_one_sided_area_derivatives():
    def mean(offset):
        eps, _, (x, y) = partition([dict(background=1., shapes=[
            dict(lx=.4, ly=.2, cy=offset, material=4.),
            dict(lx=.2, ly=.4, cx=.1, cy=.3, material=7.)])])
        return np.sum(eps[0]*np.diff(y)[:, None]*np.diff(x)[None, :])/.8
    # A later rectangle starts covering the first at contact. The physical
    # painting rule has a kink, even if cell sampling is averaged afterwards.
    for step in (1e-4, 1e-5):
        np.testing.assert_allclose((mean(step)-mean(0.))/step, -.75, atol=5e-11)
        np.testing.assert_allclose((mean(0.)-mean(-step))/step, 0., atol=5e-11)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_aligned_recipe_equals_exact_raster_on_both_incidence_sides(side):
    raster, mu = np.full((1, 3, 4), 1.4), np.full((1, 3, 4), 1.03)
    raster[0, 1, 1:3], mu[0, 1, 1:3] = 2.3, 1.15
    recipe = [dict(background=wavebit.Material(1.4, 1.03), shapes=[
        dict(lx=.2, ly=.1, material=wavebit.Material(2.3, 1.15))])]
    options = dict(thickness=[.2], period=[.4, .3], top=1., bot=1.21, wavelength=.9,
                   theta=.2, phi=.3, polarization=(1., .4j), backend=0, resolution=12,
                   n_pml=6, pulse_width=1.5, n_periods=35, truncation_order=0, incidence_side=side)
    actual = wavebit.fdtd(structure=recipe, **options)
    expected = wavebit.fdtd(structure=raster, mu=mu, **options)
    for column in ('given', 's_inc', 'p_inc'):
        for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
            np.testing.assert_allclose(getattr(getattr(actual, column), name),
                                       getattr(getattr(expected, column), name), atol=2e-13, rtol=0)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_one_jit_trace_preserves_periodic_motion_and_masks_invalid_shapes(side):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traces = []
    def run(p):
        traces.append(None)
        scene = [dict(background=1.2, shapes=[dict(lx=p[1], ly=.14, cx=p[0], material=p[2])])]
        result, out = wavebit.fdtd(structure=scene, thickness=[.2], period=[.4, .3], top=1., bot=1.21,
                                   wavelength=1., backend=1, resolution=8, n_pml=5, n_periods=30,
                                   shape_smoothing=.03, truncation_order=0, return_diagnostics=True,
                                   incidence_side=side)
        return result.s_inc.R, out['stable']
    run = jax.jit(run)
    p = np.array([.16, .15, 2.4])
    actual, stable = run(p)
    assert stable and np.isfinite(actual).all()
    translated, valid = run(p+np.array([1.2, 0., 0.]))
    assert valid
    np.testing.assert_allclose(translated, actual, atol=2e-13, rtol=0)
    for bad in ([.16, -.1, 2.4], [.16, .41, 2.4], [np.nan, .15, 2.4], [.16, .15, -1.]):
        value, valid = run(np.array(bad))
        assert not valid and np.isnan(value).all()
    assert len(traces) == 1


@pytest.mark.parametrize('backend', [1, 2])
def test_positive_averaging_removes_cell_crossing_derivative_jump(backend):
    pytest.importorskip('jax' if backend == 1 else 'torch')
    runtime = time_loop_backend(backend)
    def derivative(position, smoothing):
        def f(p):
            edges = runtime.xp.concat((runtime.array([0.]), p, runtime.array([.83, 1.])))
            return _periodic_shape_weights(np.array([.3]), .1, edges, 1., smoothing,
                                           xp=runtime.xp, array=runtime.array)[:, 0]
        return value_and_jacobian(f, np.array([position]), backend)[1].item()
    assert abs(derivative(.35-1e-6, 0.)-derivative(.35+1e-6, 0.)) > 9.
    for offset in (1e-4, 1e-6):
        gap = abs(derivative(.35-offset, .08)-derivative(.35+offset, .08))
        np.testing.assert_allclose(gap, 2*offset/(.08*.1), atol=2e-13)


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('same_layer', [True, False])
def test_coincident_projected_edges_of_disjoint_shapes_keep_correct_gradients(backend, same_layer):
    pytest.importorskip('jax' if backend == 1 else 'torch')
    def average(p, which):
        runtime = time_loop_backend(which)
        xp, array = runtime.xp, runtime.array
        shapes = [
            dict(lx=.2, ly=p[1], cx=-.22, cy=p[0], material=3.),
            dict(lx=.22, ly=.243, cx=.22 if same_layer else -.22, cy=.0317, material=4.)]
        scene = [dict(background=1., shapes=shapes)] if same_layer else [dict(background=1., shapes=[s]) for s in shapes]
        eps, _, (x, y) = partition(scene, which)
        wx = _periodic_shape_weights(np.arange(10)/10, .1, x, 1., .08, xp=xp, array=array)
        wy = _periodic_shape_weights(np.arange(8)/10, .1, y, .8, .08, xp=xp, array=array)
        return xp.stack([wy @ row @ xp.matrix_transpose(wx) for row in eps]).reshape(-1)
    p = np.array([.0317, .243])
    _, derivative = value_and_jacobian(lambda p: average(p, backend), p, backend)
    for step in (2e-5, 1e-5):
        finite = np.stack([(average(p+v, 0)-average(p-v, 0))/(2*step) for v in np.eye(2)*step], axis=-1)
        np.testing.assert_allclose(derivative, finite, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(derivative.mean(axis=0), [0., 2*.2/.8/(1 if same_layer else 2)], atol=2e-14)


@pytest.mark.parametrize('change', [dict(subpixel=False), dict(mu=np.ones((1, 1, 1))),
                                   dict(dispersion=np.ones((1, 1, 1), int), poles={1: [(1., .1, .2)]}),
                                   dict(period=.4), dict(shape_smoothing=-.1), dict(shape_smoothing=.5),
                                   dict(shape_smoothing=[.1]), dict(shape_smoothing=np.nan)])
def test_recipe_option_constraints_are_refused_before_time_stepping(change):
    options = dict(structure=[dict(background=1., shapes=[dict(lx=.2, ly=.1, material=2.)])],
                   thickness=[.2], top=1., bot=1., period=[.4, .3], wavelength=1., backend=0,
                   resolution=10, n_pml=5, n_periods=30, truncation_order=0)
    with pytest.raises(ValueError):
        wavebit.fdtd(**dict(options, **change))


def test_only_statically_full_height_recipes_use_one_row_partition():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    for height, rows in ((.8, 1), (jnp.asarray(.8), 3)):
        eps, _, _ = partition([dict(background=1., shapes=[dict(lx=.2, ly=height, material=2.)])], 1)
        assert eps.shape[1] == rows
