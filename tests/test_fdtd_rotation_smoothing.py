"""Exact translated-polygon moments against independent weighted slice integration."""
import numpy as np
import pytest

from wavebit._core.scene_fdtd import _shape_cell_fractions, _shape_partition, _require_scene
from wavebit.backends.time_domain import time_loop_backend
from test_fdtd_scene_angles import value_and_jacobian
from test_fdtd_rotated_rectangles import vertices_for, response as sharp_response


def kernel_pieces(centre, width, smooth):
    if not smooth:
        return [(centre-width/2, centre+width/2, 1., 0.)]
    knots = sorted([centre-(width+smooth)/2, centre+(smooth-width)/2,
                    centre+(width-smooth)/2, centre+(width+smooth)/2])
    height = min(width, smooth)/smooth
    return [(a, b, low, (high-low)/(b-a)) for a, b, low, high in
            zip(knots[:-1], knots[1:], [0., height, height], [height, height, 0.]) if b > a]


def weighted_slices(vertices, x, y, smoothing):
    """Independent cubic slice integrals, with exact two-point Gaussian integration.

    Integrate the y weight analytically between the rectangle's upper/lower
    edges. Split x at all edges, kernel knots and horizontal intersections;
    each remaining integrand is cubic. No polygon clipping or area moments.
    """
    px, py = kernel_pieces(x, .1, smoothing[0]), kernel_pieces(y, .1, smoothing[1])
    edges = list(zip(vertices, np.roll(vertices, -1, axis=0)))
    cuts = [*vertices[:, 0], *[v for piece in px for v in piece[:2]]]
    for a, b in edges:
        if b[1] != a[1]:
            for yy in [v for piece in py for v in piece[:2]]:
                t = (yy-a[1])/(b[1]-a[1])
                if 0 < t < 1:
                    cuts.append(a[0]+t*(b[0]-a[0]))
    cuts = np.unique(np.clip(cuts, px[0][0], px[-1][1]))
    def primitive(yy):
        return sum(low*(t := np.clip(yy-a, 0., b-a))+.5*slope*t*t for a, b, low, slope in py)
    result = 0.
    for left, right in zip(cuts[:-1], cuts[1:]):
        for sign in (-1, 1):
            xx = (left+right)/2+sign*(right-left)/(2*np.sqrt(3))
            yy = [a[1]+(xx-a[0])*(b[1]-a[1])/(b[0]-a[0]) for a, b in edges
                  if min(a[0], b[0]) < xx < max(a[0], b[0])]
            if yy:
                wx = sum(low+slope*(xx-a) for a, b, low, slope in px if a < xx < b)
                result += (right-left)/2*wx*(primitive(max(yy))-primitive(min(yy)))
    return result/.01


def oracle(p, smoothing, indices=((0, 0), (0, 9), (7, 0), (7, 9), (1, 1), (4, 5))):
    vertices = vertices_for(p)
    return np.array([sum(weighted_slices(vertices+[a, b], i*.1, j*.1, smoothing)
                         for a in (-2., -1., 0., 1., 2.) for b in (-1.6, -.8, 0., .8, 1.6))
                     for j, i in indices])


def fractions(p, smoothing, backend='numpy'):
    rt = time_loop_backend(backend)
    _, _, layers = _shape_partition([dict(background=1., shapes=[
        dict(cx=p[0], cy=p[1], lx=p[2], ly=p[3], angle=p[4], material=2.)])],
        [1., .8], xp=rt.xp, array=rt.array, check=_require_scene, integration='area')
    return _shape_cell_fractions(layers[0][1][0], np.arange(10)*.1, np.arange(8)*.1,
                                     .1, .1, [1., .8], xp=rt.xp, array=rt.array, smoothing=smoothing)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('smoothing', [(.08, .06), (.1, .1), (0., .1), (1., .8)])
def test_weighted_polygon_moments_match_independent_cubic_slice_integrals(backend, smoothing):
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')
    p = np.array([.473, -.369, .413, .287, .31])
    actual = np.asarray(fractions(p, smoothing, backend))
    indices = ((0, 0), (0, 9), (7, 0), (7, 9), (1, 1), (4, 5))
    np.testing.assert_allclose([actual[j, i] for j, i in indices], oracle(p, smoothing), atol=3e-13, rtol=0)
    np.testing.assert_allclose(actual.sum()*.01, p[2]*p[3], atol=3e-15)
    np.testing.assert_allclose(fractions(p+[2., -1.6, 0., 0., 0.], smoothing, backend), actual, atol=5e-14)
    assert actual.min() >= 0 and actual.max() <= 1


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('angle', [0., .31])
def test_five_shape_derivatives_match_independent_weighted_geometry(backend, angle):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    p = np.array([.473, -.369, .413, .287, angle])
    indices = ((0, 0), (0, 9), (7, 0), (7, 9), (1, 1), (4, 5))
    def f(p):
        rt = time_loop_backend(backend)
        full = fractions(p, (.08, .06), backend)
        return rt.xp.stack([full[j, i] for j, i in indices])
    _, jac = value_and_jacobian(f, p, backend)
    for step in (2e-6, 1e-6):
        expected = np.stack([(oracle(p+d, (.08, .06))-oracle(p-d, (.08, .06)))/(2*step)
                             for d in np.eye(5)*step], axis=-1)
        np.testing.assert_allclose(jac, expected, atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_full_period_averaging_requires_outer_images_and_has_exact_mean_gradient(backend):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    # Near-full shapes at the upper periodic seam reach the second image ring.
    p = np.array([.499, .399, .99, .79, 0.])
    value, jac = value_and_jacobian(lambda p: fractions(p, (1., .8), backend).reshape(-1), p, backend)
    np.testing.assert_allclose(value, p[2]*p[3]/.8, atol=5e-15)
    np.testing.assert_allclose(jac, np.broadcast_to([0., 0., p[3]/.8, p[2]/.8, 0.], jac.shape), atol=4e-14)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_averaging_has_the_known_continuous_cell_crossing_derivative(backend):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    def derivative(offset):
        return value_and_jacobian(lambda p: fractions(p, (.08, .06), backend)[4, 3].reshape(1),
                                  np.array([-.05+offset, 0., .2, .3, 0.]), backend)[1][0, 0]
    for delta in (1e-4, 1e-6):
        np.testing.assert_allclose(abs(derivative(delta)-derivative(-delta)), 2*delta/(.08*.1), atol=3e-12)


def test_equal_and_nearly_equal_kernel_cell_widths_remain_finite():
    p = np.array([.473, -.369, .413, .287, .31])
    exact = fractions(p, (.1, .1))
    for width in (np.nextafter(.1, 0.), np.nextafter(.1, 1.)):
        value = fractions(p, (width, width))
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, exact, atol=3e-14)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.slow
def test_smoothed_bottom_conical_eight_parameter_graph(backend):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    p = np.array([.13, -.09, .16, .12, .37, 2.4, 1.13, .14])
    f = lambda p, b=backend: sharp_response(p, b, smoothing=(.04, .03))
    value, jac = value_and_jacobian(f, p, backend)
    np.testing.assert_allclose(value, f(p, 0), atol=2e-11)
    for step in (2e-5, 1e-5):
        finite = np.stack([(f(p+d, 0)-f(p-d, 0))/(2*step) for d in np.eye(8)*step], axis=-1)
        np.testing.assert_allclose(jac, finite, atol=3e-8, rtol=5e-5)
    assert np.min(np.max(abs(jac), axis=0)) > 1e-5


def test_full_period_average_reduces_to_the_same_uniform_magnetic_slab():
    import wavebit
    options = dict(thickness=[.2], period=[.4, .3], top=1., bot=1.21, wavelength=.9,
                   theta=.17, phi=.23, polarization=(1., .4j), incidence_side='bottom',
                   backend='numpy', resolution=9, n_pml=6, n_periods=38, pulse_width=1.5,
                   truncation_order=0, align_layers=True, diffraction_method='directional')
    recipe = [dict(background=wavebit.Material(1.2, 1.03), shapes=[
        dict(cx=.13, cy=-.09, lx=.16, ly=.12, angle=.37, material=wavebit.Material(2.4, 1.13))])]
    actual = wavebit.fdtd(structure=recipe, shape_integration='area', shape_smoothing=[.4, .3], **options)
    fill = .16*.12/(.4*.3)
    expected = wavebit.fdtd(structure=np.full((1, 2, 1), 1.2+fill*(2.4-1.2)),
                            mu=np.full((1, 2, 1), 1.03+fill*(1.13-1.03)), **options)
    for column in ('s_inc', 'p_inc', 'incident'):
        for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
            np.testing.assert_allclose(getattr(getattr(actual, column), name),
                                       getattr(getattr(expected, column), name), atol=2e-13)


def test_fourier_reference_profile_matches_independent_box_intersections():
    from studies.fdtd_rotation_smoothing import density
    from test_fdtd_rotated_rectangles import vertical_integral
    angle, width, height, smooth = .37, .3, .18, .05
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*[width/2, height/2] @ rotation+[.371, .253]
    points = np.concatenate([vertices+offset for offset in ([0., 0.], [.02, .01], [-.02, -.01])])
    for grid, tolerance in ((256, 2e-4), (512, 6e-5)):
        profile = density([angle, width], smooth, grid)
        indices = np.floor(points/.6*grid).astype(int)
        for i, j in indices:
            x, y = (np.array([i, j])+.5)*.6/grid
            expected = sum(vertical_integral(vertices+[a, b], x-smooth/2, x+smooth/2, y-smooth/2, y+smooth/2)
                           for a in (-.6, 0., .6) for b in (-.6, 0., .6))/smooth**2
            np.testing.assert_allclose(profile[j, i], expected, atol=tolerance, rtol=0)


@pytest.mark.parametrize('angle', [0., .37, np.pi/2])
@pytest.mark.parametrize('clip', [False, True])
def test_independent_sinc_profile_analytic_shape_jacobian(angle, clip):
    from studies.fdtd_rotation_smoothing import density
    p = np.array([angle, .3])
    raw = density(p, grid=128, clip=False)
    _, jac = density(p, grid=128, clip=clip, return_jacobian=True)
    assert np.isfinite(jac).all()
    if not clip:
        np.testing.assert_allclose(jac.mean(axis=(0, 1)), [0., .18/.6**2], atol=3e-15)
    else:
        np.testing.assert_array_equal(jac[(raw < 0) | (raw > 1)], 0.)
    for step in (2e-7, 1e-7):
        for k, delta in enumerate(np.eye(2)*step):
            plus, minus = [density(p+d, grid=128, clip=False) for d in (delta, -delta)]
            # Central differences crossing a clipping knot average two local
            # slopes. Check the analytic derivative where the branch is fixed.
            if clip:
                region = lambda a: np.where(a <= 0, -1, np.where(a >= 1, 1, 0))
                smooth = (region(raw) == region(plus)) & (region(raw) == region(minus))
                plus, minus = np.clip(plus, 0., 1.), np.clip(minus, 0., 1.)
            else:
                smooth = np.ones(raw.shape, dtype=bool)
            assert smooth.mean() > .98
            np.testing.assert_allclose(jac[..., k][smooth], ((plus-minus)/(2*step))[smooth], atol=8e-9, rtol=2e-5)


def test_smoothed_complex_transmission_forward_reverse_and_finite_gradients():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from studies.fdtd_rotation_gradients import response, finite_difference
    p = np.array([.37, .3])
    def f(p):
        value = response(p, resolution=10, smoothing=.05)[jnp.array([2, 7])]
        return jnp.concatenate((jnp.real(value), jnp.imag(value)))
    forward = np.asarray(jax.jit(jax.jacfwd(f))(p))
    reverse = np.asarray(jax.jit(jax.jacrev(f))(p))
    np.testing.assert_allclose(reverse, forward, atol=3e-11, rtol=2e-9)
    def independent(p):
        value = np.asarray(response(p, resolution=10, smoothing=.05, backend='numpy'))[[2, 7]]
        return np.r_[value.real, value.imag]
    for step in (2e-5, 1e-5):
        np.testing.assert_allclose(reverse, finite_difference(independent, p, step), atol=3e-8, rtol=2e-5)
