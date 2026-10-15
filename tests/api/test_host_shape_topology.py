"""Explicit host topology checks retain live JAX derivatives and reject stale plans."""
import numpy as np
import pytest

from wavebit import Job, Material, Units, rcwa
from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError


NUMERICS = rcwa.Numerics(orders=(1, 1), factorization='nvf',
                        nvf_projector='circular', shape_nvf_grid_size=32,
                        topology_check='host')


def scene(shapes, *, period=(700., 700.), lattice=None, epsilon=4., fields=False):
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=period if lattice is None else None, lattice=lattice,
            bottom=Material(2.25), layers=(rcwa.Layer(thickness=180., regions=tuple(
                rcwa.Region(geometry=s, material=Material(epsilon+i)) for i, s in enumerate(shapes))),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(
            rcwa.PlaneWave(theta=.2, phi=.13, polarization='p'),)),
        outputs=rcwa.Outputs(fields=fields, scattering=fields))


CASES = {
    'radius': (lambda v: scene((rcwa.Circle(radius=v),)), 120., .001),
    'width': (lambda v: scene((rcwa.Rectangle(size=(v, 200.), angle=.17),)), 280., .001),
    'angle': (lambda v: scene((rcwa.Rectangle(size=(280., 200.), angle=v),)), .17, 1e-6),
    'period': (lambda v: scene((rcwa.Circle(radius=120.),), period=(v, 700.)), 700., .001),
    'painting': (lambda v: scene((rcwa.Rectangle(size=(250., 200.), center=(-50., 0.)),
                                  rcwa.Rectangle(size=(200., 240.), center=(v, 30.)))), 80., .001),
}


@pytest.fixture
def jax():
    module = pytest.importorskip('jax')
    with module.enable_x64(True):
        yield module


@pytest.mark.parametrize('case', CASES)
def test_host_guard_jit_gradient_and_jvp_match_fresh_differences(jax, case):
    build, at, step = CASES[case]
    prepared = rcwa.prepare(build(at), numerics=NUMERICS, backend='jax')
    loss = lambda value: prepared.run(build(value)).T.sum()
    value, gradient = jax.jit(jax.value_and_grad(loss))(at)
    tangent = jax.jit(lambda x: jax.jvp(loss, (x,), (1.,))[1])(at)
    fresh = lambda x: rcwa.run(build(x), numerics=NUMERICS).T.sum()
    finite = (fresh(at+step)-fresh(at-step))/(2*step)
    np.testing.assert_allclose(value, fresh(at), rtol=2e-12)
    assert abs(finite) > 1e-7
    np.testing.assert_allclose(gradient, finite, rtol=2e-5, atol=2e-9)
    np.testing.assert_allclose(tangent, gradient, rtol=2e-11, atol=1e-12)


def test_vmap_runs_a_check_for_each_live_geometry_and_masks_all_observations(jax):
    import jax.numpy as jnp
    def pair(gap):
        return scene((rcwa.Rectangle(size=(200., 200.), center=(-100.-gap/2, 0.)),
                      rcwa.Rectangle(size=(200., 200.), center=(100.+gap/2, 0.))), fields=True)
    prepared = rcwa.prepare(pair(0.), numerics=NUMERICS, backend='jax')
    def run(gap):
        result = prepared.run(pair(gap))
        return (result.validity.checks['shape_topology'], result.R, result.T,
                result.transmission.amplitude_s, result.transmission.amplitude_p,
                result.field(x=[0.], y=[0.], z=[90.]).values,
                result.boundary_fields(side='top', x=[0.], y=[0.]).values,
                tuple(getattr(result.scattering[0], k) for k in ('S11', 'S12', 'S21', 'S22')))
    compiled = jax.jit(jax.vmap(run))
    for gaps in ([10., 0., -10.], [0., 20., -20.]):
        flag, *values, scattering = compiled(jnp.asarray(gaps))
        np.testing.assert_array_equal(flag, [True, True, False])
        for value in (*values, *scattering):
            assert np.isfinite(value[:2]).all()
            assert np.isnan(value[2]).all()
        np.testing.assert_allclose(values[1][0], rcwa.run(pair(gaps[0]), numerics=NUMERICS).T, rtol=2e-12)
    with pytest.raises(IncompatibleProblemError):
        prepared.run(pair(-10.))


@pytest.mark.parametrize('bad', [-1., 0., 500., np.nan, np.inf])
def test_invalid_size_or_image_bound_never_returns_a_finite_answer(jax, bad):
    build = CASES['radius'][0]
    prepared = rcwa.prepare(build(120.), numerics=NUMERICS, backend='jax')
    def run(radius):
        result = prepared.run(build(radius))
        return result.validity.checks['shape_topology'], result.T
    valid, power = jax.jit(run)(bad)
    assert not bool(valid) and np.isnan(power).all()


def test_polygon_self_crossing_is_checked_inside_the_trace(jax):
    import jax.numpy as jnp
    vertices = jnp.asarray([[-100., -100.], [100., -100.], [100., 100.], [-100., 100.]])
    build = lambda v: scene((rcwa.Polygon(vertices=v),))
    prepared = rcwa.prepare(build(vertices), numerics=NUMERICS.replace(nvf_projector='linear'), backend='jax')
    def run(v):
        r = prepared.run(build(v))
        return r.validity.checks['shape_topology'], r.R
    evaluate = jax.jit(run)
    assert bool(evaluate(vertices)[0])
    crossed = vertices[jnp.asarray([0, 2, 1, 3])]
    valid, power = evaluate(crossed)
    assert not bool(valid) and np.isnan(power).all()


def test_polygon_painting_plan_change_is_not_hidden_by_a_finite_solve(jax):
    def build(offset):
        return scene((rcwa.Rectangle(size=(250., 200.), center=(-50., 0.)),
                      rcwa.Rectangle(size=(200., 240.), center=(offset, 30.))))
    prepared = rcwa.prepare(build(80.), numerics=NUMERICS, backend='jax')
    def run(offset):
        result = prepared.run(build(offset))
        return result.validity.checks['shape_topology'], result.T
    evaluate = jax.jit(run)
    assert bool(evaluate(81.)[0])
    valid, power = evaluate(200.)
    assert not bool(valid) and np.isnan(power).all()


def test_host_policy_does_not_claim_tensor_or_raster_topology_support():
    base = scene((rcwa.Circle(radius=120.),))
    layer = base.stack.layers[0]
    raster = layer.replace(regions=(), background=None, epsilon=np.ones((4, 4)))
    tensor = layer.replace(regions=(layer.regions[0].replace(material=Material(np.diag([4., 5., 6.]))),))
    for changed in (raster, tensor):
        with pytest.raises(UnsupportedFeatureError) as error:
            rcwa.prepare(base.replace(stack=base.stack.replace(layers=(changed,))), numerics=NUMERICS)
        assert error.value.code == 'prepared.host_topology_scope'


def test_host_policy_is_a_portable_static_numerical_choice():
    from wavebit import run
    job = Job(solver='rcwa', problem=scene((rcwa.Circle(radius=120.),)), numerics=NUMERICS)
    restored = Job.from_json(job.to_json())
    assert restored.key == job.key and restored.numerics.topology_check == 'host'
    np.testing.assert_array_equal(run(restored).T, run(job).T)
