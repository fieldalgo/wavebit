"""Concrete sampling containers share the compiled field API's static meaning."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest


jax = pytest.importorskip('jax')
import jax.numpy as jnp

POINTS = (-211., 11., 129.)


def solver(backend='jax', basis=False):
    cell = dict(lattice=np.diag([600., 750.])) if basis else dict(period=(600., 750.))
    return _legacy_wavebit.rcwa(backend=backend, **cell, structure=np.full((1, 2, 2), 4.),
                        wavelength=900., thickness=(170.,), truncation_order=(0, 0),
                        answers='field', theta=.13, phi=.21)


def samples(spelling):
    return dict(tuple=POINTS, list=list(POINTS), numpy=np.asarray(POINTS),
                jax=jnp.asarray(POINTS))[spelling]


@pytest.mark.parametrize('spelling', ['tuple', 'list', 'numpy', 'jax'])
@pytest.mark.parametrize('basis', [False, True])
@pytest.mark.parametrize('axis', ['sample_x', 'sample_y'])
def test_coordinate_containers_reconstruct_the_same_numpy_field(spelling, basis, axis):
    s = solver(basis=basis)
    kw = dict(sample_x=2, sample_y=2, sample_z=2, incidences=['p', 's'])
    kw[axis] = samples(spelling)
    result, field = s.solve_field(geometry=s.freeze_geometry(), **kw)
    reference, expected = solver('numpy', basis=basis).solve_field(**kw)
    np.testing.assert_allclose(field, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result.T, reference.T, atol=1e-10)


def test_equivalent_containers_reuse_the_static_cache_entry():
    s = solver()
    geometry = s.freeze_geometry()
    s.solve_field(POINTS, 2, 2, geometry=geometry)
    cache_size = s._solve_field_compiled._cache_size()
    for spelling in ['list', 'numpy', 'jax']:
        s.solve_field(samples(spelling), np.int64(2), 2, incidences=['incident'], geometry=geometry)
        assert s._solve_field_compiled._cache_size() == cache_size


def test_mutating_a_coordinate_array_updates_the_sampled_field():
    s = solver()
    geometry = s.freeze_geometry()
    x = np.asarray(POINTS)
    _, before = s.solve_field(x, 2, 2, geometry=geometry)
    x[1] += 20.
    _, after = s.solve_field(x, 2, 2, geometry=geometry)
    _, expected = solver('numpy').solve_field(x, 2, 2)
    assert np.max(np.abs(np.asarray(after)-np.asarray(before))) > 1e-4
    np.testing.assert_allclose(after, expected, atol=1e-10)


@pytest.mark.parametrize('incidences', ['incident', [], ['s', 's'], [True], ['unknown']])
def test_invalid_incidence_requests_are_refused_before_compilation(incidences, monkeypatch):
    s = solver()
    def unexpected(*args, **kwargs):
        pytest.fail('an invalid request must not enter the compiled solve')
    monkeypatch.setattr(s, '_solve_field_compiled', unexpected)
    with pytest.raises(ValueError, match='incidences'):
        s.solve_field(2, 2, 2, incidences=incidences, geometry=s.freeze_geometry())


@pytest.mark.parametrize('bad', [3., True, np.asarray(3), [[0., 1.]]])
def test_invalid_coordinate_specs_keep_the_named_error(bad):
    s = solver()
    with pytest.raises(TypeError, match='sample_x'):
        s.solve_field(bad, 2, 2, geometry=s.freeze_geometry())


def test_traced_coordinates_have_an_explicit_static_sampling_refusal():
    s = solver()
    geometry = s.freeze_geometry()
    def loss(x):
        _, field = s.solve_field(x, 2, 2, geometry=geometry)
        return jnp.mean(jnp.abs(field)**2)
    with pytest.raises(ValueError, match='sample_x.*static sampling'):
        jax.grad(loss)(jnp.asarray(POINTS))


def test_array_coordinates_allow_a_wavelength_derivative_under_outer_jit():
    s = solver()
    geometry = s.freeze_geometry()
    x = np.asarray(POINTS)
    def loss(wavelength):
        _, field = s.solve_field(x, 2, 2, geometry=geometry, wavelength=wavelength)
        return jnp.mean(jnp.abs(field)**2)
    got = float(jax.jit(jax.grad(loss))(900.))
    def reference(wavelength):
        _, field = solver('numpy').solve_field(x, 2, 2, wavelength=wavelength)
        return np.mean(np.abs(field)**2)
    h = 1e-3
    expected = (reference(900.+h)-reference(900.-h))/(2*h)
    assert abs(expected) > 1e-7
    assert got == pytest.approx(expected, abs=1e-9, rel=2e-6)


@pytest.mark.parametrize('door', ['eager', 'result'])
def test_recommended_xy_derivative_doors_match_numpy_differences(door):
    s = solver()
    result = s.solve()
    def loss(x):
        field = (s.solve_field_eager(x, 2, 2)[1] if door == 'eager'
                 else result.reconstruct_field(x, 2, 2))
        return jnp.mean(jnp.real(field))
    got = np.asarray(jax.grad(loss)(jnp.asarray(POINTS)))
    ref = solver('numpy')
    saved = ref.solve()
    def reference(x):
        field = (ref.solve_field(x, 2, 2)[1] if door == 'eager'
                 else saved.reconstruct_field(x, 2, 2))
        return np.mean(np.real(field))
    expected = []
    for i in range(len(POINTS)):
        plus, minus = np.asarray(POINTS), np.asarray(POINTS)
        plus[i] += 1e-3
        minus[i] -= 1e-3
        expected.append((reference(plus)-reference(minus))/2e-3)
    assert np.linalg.norm(expected) > 1e-7
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=2e-6)


def test_explicit_z_keeps_its_thickness_refusal_and_works_eagerly():
    s = solver()
    z = [0., 70., 170.]
    with pytest.raises(TypeError, match='explicit z.*thickness'):
        s.solve_field(2, 2, z, geometry=s.freeze_geometry())
    _, field = s.solve_field_eager(2, 2, z)
    _, expected = solver('numpy').solve_field(2, 2, z)
    np.testing.assert_allclose(field, expected, atol=1e-10)
