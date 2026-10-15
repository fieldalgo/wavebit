"""C1 layer fractions: independent integration, derivatives and fixed boundaries."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd
import numpy as np
import pytest

from wavebit.fdtd._periodic.runtime import time_loop_backend
from wavebit.fdtd._periodic.scene import _layer_weights

from tests.fdtd.test_fdtd_layer_gradients import options
from tests.fdtd.test_fdtd_scene_angles import value_and_jacobian


def box_average(centres, widths, edges, smoothing):
    """Integrate piecewise-linear geometric overlaps exactly by midpoints.

    Split the independent integration at every shifted face/cell intersection;
    each resulting integrand is linear, so no smoothing primitive is reused.
    """
    rows = []
    for centre, width in zip(centres, widths):
        left, right = centre-width/2, centre+width/2
        points = [-smoothing/2, smoothing/2]
        points += [v for edge in edges[2:-2] for v in (left-edge, right-edge)
                   if -smoothing/2 < v < smoothing/2]
        points = sorted(set(points))
        mean = np.zeros(len(edges)-1)
        for a, b in zip(points, points[1:]):
            shifted = edges.copy(); shifted[2:-2] += (a+b)/2
            overlap = np.maximum(0, np.minimum(right, shifted[1:])-np.maximum(left, shifted[:-1]))/width
            mean += (b-a)/smoothing*overlap
        rows.append(mean)
    return np.asarray(rows)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_smoothed_fractions_match_integrated_shifted_geometry(backend):
    if backend != 'numpy':
        pytest.importorskip('jax' if backend == 'jax' else 'torch')
    runtime = time_loop_backend(backend)
    xp, array = runtime.xp, runtime.array
    centres = np.array([-.3, 0., .19, .37, .4, .43, .72, 1., 1.3])
    widths = np.array([.1, .12, .09, .13, .24, .1, .3, .08, .11])
    positions, smoothing = np.array([.37, .43]), .2
    def objective(p):
        edges = xp.concat((array([-2., 0.]), p, array([1., 3.])))
        return _layer_weights(centres, widths, edges, xp=xp, array=array, smoothing=smoothing)
    expected = box_average(centres, widths, np.r_[-2., 0., positions, 1., 3.], smoothing)
    if backend != 'numpy':
        value, gradient = value_and_jacobian(objective, positions, backend)
        target = np.zeros((len(centres), 5, 2))
        for k, boundary in enumerate(positions):
            left, right = centres-widths/2-boundary, centres+widths/2-boundary
            speed = (np.clip(right/smoothing+.5, 0, 1)-np.clip(left/smoothing+.5, 0, 1))/widths
            target[:, k+1, k] = speed
            target[:, k+2, k] = -speed
        np.testing.assert_allclose(gradient, target, atol=2e-14)
    else:
        value = objective(positions)
    np.testing.assert_allclose(value, expected, atol=2e-14)
    np.testing.assert_allclose(np.sum(value, axis=1), 1., atol=1e-15)
    assert np.min(value) >= 0


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('position', [.375, .475, .525, .625])
def test_primitive_join_derivatives_match_both_sides(backend, position):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    runtime = time_loop_backend(backend)
    xp, array = runtime.xp, runtime.array
    def objective(p):
        edges = xp.concat((array([-2., 0.]), p, array([1., 3.])))
        return _layer_weights(np.array([.5]), .15, edges, xp=xp, array=array, smoothing=.1)
    # Each position makes one cell face meet a join at +/- smoothing/2.
    _, derivative = value_and_jacobian(objective, np.array([position]), backend)
    speed = (np.clip((.575-position)/.1+.5, 0, 1)-np.clip((.425-position)/.1+.5, 0, 1))/.15
    np.testing.assert_allclose(derivative.reshape(-1), [0., speed, -speed, 0.], atol=2e-14)
    for step in (1e-5, 1e-6):
        a = objective(array([position-step]))
        b = objective(array([position]))
        c = objective(array([position+step]))
        if backend == 'torch':
            a, b, c = (v.detach().numpy() for v in (a, b, c))
        for finite in ((b-a)/step, (c-b)/step):
            np.testing.assert_allclose(np.asarray(finite).ravel(), [0., speed, -speed, 0.], atol=40*step, rtol=0)


def test_volume_partition_reflection_and_exact_exterior_values():
    width = .01
    centres = np.arange(-100, 200)*width+width/2
    edges = np.array([-2., 0., .37, .43, 1., 3.])
    value = _layer_weights(centres, width, edges, smoothing=.2, xp=np, array=np.asarray)
    np.testing.assert_allclose(value.sum(axis=0)*width, [1., .37, .06, .57, 1.], atol=3e-15)
    assert np.min(value) >= 0
    np.testing.assert_array_equal(value[centres < 0], np.tile([1., 0., 0., 0., 0.], (100, 1)))
    # The adjacent cell can straddle z=1 by floating-point roundoff.
    np.testing.assert_allclose(value[centres > 1], np.tile([0., 0., 0., 0., 1.], (100, 1)), atol=3e-14)
    np.testing.assert_array_equal(value[centres > 1+width], np.tile([0., 0., 0., 0., 1.], (99, 1)))
    reflected = _layer_weights(1-centres[::-1], width, 1-edges[::-1], smoothing=.2, xp=np, array=np.asarray)
    np.testing.assert_allclose(value, reflected[::-1, ::-1], atol=3e-14)


@pytest.mark.parametrize('scale', [1e-6, 1e6])
def test_interface_averaging_is_independent_of_length_unit(scale):
    centres, widths = np.array([0., .2, .4, .6, 1.]), np.array([.07, .13, .1, .17, .08])
    edges = np.array([-2., 0., .37, .43, 1., 3.])
    expected = box_average(centres, widths, edges, .2)
    actual = _layer_weights(centres*scale, widths*scale, edges*scale,
                            smoothing=.2*scale, xp=np, array=np.asarray)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_single_layer_has_no_internal_interface_to_smooth(side):
    kw = options(structure=np.array([[[2.25]]]), thickness=[.4], mesh_thickness=[.4],
                 incidence_side=side)
    sharp = _legacy_wavebit_fdtd(**kw)
    smoothed = _legacy_wavebit_fdtd(**kw, layer_smoothing=.1)
    for column in ('s_inc', 'p_inc'):
        for component in ('r_s', 'r_p', 't_s', 't_p'):
            np.testing.assert_allclose(getattr(getattr(smoothed, column), component),
                                       getattr(getattr(sharp, column), component), atol=2e-13, rtol=0)


@pytest.mark.parametrize('value', [-.1, np.nan, np.inf, .1j, True, [.1], '0.1', np.nextafter(0., 1.)])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_invalid_smoothing_is_rejected_before_stepping(monkeypatch, value, side):
    import wavebit.fdtd._periodic.scene as scene
    monkeypatch.setattr(scene, 'fdtd_grid', lambda *a, **kw: pytest.fail('reached time stepping'))
    with pytest.raises(ValueError, match='layer_smoothing'):
        _legacy_wavebit_fdtd(**options(layer_smoothing=value, incidence_side=side))


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('changes', [dict(mesh_thickness=None), dict(subpixel=False),
                                    dict(thickness=[.04, .36]), dict(thickness=[.36, .04])])
def test_positive_smoothing_requires_fixed_mesh_and_exterior_clearance(monkeypatch, side, changes):
    import wavebit.fdtd._periodic.scene as scene
    monkeypatch.setattr(scene, 'fdtd_grid', lambda *a, **kw: pytest.fail('reached time stepping'))
    with pytest.raises(ValueError, match='layer_smoothing'):
        _legacy_wavebit_fdtd(**options(layer_smoothing=.1, incidence_side=side, **changes))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_smoothing_width_is_a_host_mesh_parameter(backend):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    width = time_loop_backend(backend).array(.1)
    with pytest.raises(ValueError, match='static real scalar'):
        _legacy_wavebit_fdtd(**options(backend=backend, layer_smoothing=width))


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_jit_keeps_mesh_and_masks_smoothing_outside_stack(side):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traces = []
    def solve(d):
        traces.append(None)
        r, out = _legacy_wavebit_fdtd(**options(backend='jax', thickness=[d, .4-d], layer_smoothing=.1,
                                       incidence_side=side, align_layers=True, return_diagnostics=True,
                                       field_monitors={'inside': .1}, record_field=151,
                                       record_components=['ey'], record_bounds={'z': (.05, .15)}))
        return r.incident.T, out['stable'], out['z_nodes'], out['layer_smoothing'], out['field_e'], out['monitors']['inside']['fields']['ey']
    compiled = jax.jit(solve)
    a, b = compiled(np.array(.173)), compiled(np.array(.281))
    assert a[1] and b[1]
    np.testing.assert_array_equal(a[2], b[2])
    assert a[3] == b[3] == .1
    for d in (.1/2, .4-.1/2):
        boundary = compiled(np.array(d))
        assert boundary[1]
        assert np.isfinite(boundary[0]).all()
    for d in (.04, .36):
        out = compiled(np.array(d))
        assert not out[1]
        for index in (0, 4, 5):
            assert np.isnan(out[index]).all()
    assert len(traces) == 1


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_support_can_touch_exterior_faces_despite_reflection_roundoff(side):
    for d in (.1/2, .4-.1/2):
        result = _legacy_wavebit_fdtd(**options(thickness=[d, .4-d], layer_smoothing=.1, incidence_side=side))
        assert np.isfinite(result.s_inc.T).all()


@pytest.mark.parametrize('smoothing', [0., .025, .1])
def test_independent_graded_reference_translation_derivative(smoothing):
    from studies.fdtd.layer_gradients.fdtd_layer_gradients import airy
    from studies.fdtd.layer_smoothing.fdtd_layer_smoothing import graded_film
    d = .173
    value, derivative = graded_film(d, smoothing)
    if smoothing == 0:
        np.testing.assert_allclose(value, airy(d), atol=2e-14, rtol=0)
    np.testing.assert_allclose(value[-4:-2]+value[-2:], 1., atol=2e-13, rtol=0)
    for step in (1e-5, 5e-6):
        finite = (graded_film(d+step, smoothing)[0]-graded_film(d-step, smoothing)[0])/(2*step)
        np.testing.assert_allclose(derivative, finite, atol=2e-8, rtol=0)
    refined = graded_film(d, smoothing, slices=1024)
    np.testing.assert_allclose(value, refined[0], atol=1e-7, rtol=0)
    np.testing.assert_allclose(derivative, refined[1], atol=2e-6, rtol=0)
    same = graded_film(d, smoothing, film=(1.6, 1.15))
    np.testing.assert_allclose(same[1], 0., atol=2e-13, rtol=0)


def test_complete_scattering_derivative_is_continuous_across_cell_face():
    pytest.importorskip('jax')
    from studies.fdtd.layer_smoothing.fdtd_layer_smoothing import crossing
    sharp_far, sharp_near, smooth_far, smooth_near = crossing()
    # At a genuine crossing, the sharp derivative jump persists while a C1
    # response's two-sided difference vanishes linearly with the offset.
    assert sharp_near['derivative_gap'] > .1
    assert sharp_near['derivative_gap'] > .9*sharp_far['derivative_gap']
    assert smooth_near['derivative_gap'] < .02*smooth_far['derivative_gap']
    assert smooth_near['derivative_gap'] < 1e-3


def test_smoothed_film_scattering_and_derivative_match_continuum():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    from studies.fdtd.layer_gradients.fdtd_layer_gradients import scene_vector
    from studies.fdtd.layer_smoothing.fdtd_layer_smoothing import graded_film
    def objective(d):
        return scene_vector(d, 80, smoothing=1/80)
    evaluate = jax.jit(lambda d: (objective(d), jax.jacfwd(objective)(d)))
    # The documented three-position error envelope, including complex r/t,
    # guards physical accuracy separately from AD agreeing with its own model.
    for d in (.163, .173, .183):
        value, derivative = evaluate(np.array(d))
        sharp, sharp_derivative = graded_film(d, 0.)
        graded, graded_derivative = graded_film(d, 1/80)
        np.testing.assert_allclose(value, sharp, atol=.002, rtol=0)
        np.testing.assert_allclose(derivative, sharp_derivative, atol=.01, rtol=0)
        np.testing.assert_allclose(value, graded, atol=.002, rtol=0)
        np.testing.assert_allclose(derivative, graded_derivative, atol=.006, rtol=0)
