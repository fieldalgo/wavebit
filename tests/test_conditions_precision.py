"""Restored traced values follow concrete precision changes kept by the call."""
import numpy as np
import pytest

import wavebit

jax = pytest.importorskip('jax')
import jax.numpy as jnp


def solver(backend=1, **kwargs):
    cfg = dict(period=(600., 750.), thickness=(170.,), truncation_order=(1, 0),
               wavelength=900., answers='field', tetm='tm',
               structure=np.array([[[1., 4., 4., 1.]]]))
    return wavebit.rcwa(backend=backend, **(cfg | kwargs))


@pytest.mark.parametrize('parameter', ['thickness', 'structure'])
@pytest.mark.parametrize('precision_first', [False, True])
@pytest.mark.parametrize('door', ['freeze_geometry', 'solve', 'solve_field'])
def test_restored_values_match_the_kept_precision(parameter, precision_first, door):
    from wavebit._core.engine import _holds_tracer
    s = solver()
    original = getattr(s, parameter)
    geometry = s.freeze_geometry()
    expected = solver(complex_dtype=1)
    def loss(value):
        conditions = (dict(complex_dtype=1) | {parameter: value} if precision_first
                      else {parameter: value} | dict(complex_dtype=1))
        if door == 'freeze_geometry':
            s.freeze_geometry(**conditions)
            return jnp.sum(value.real**2)
        if door == 'solve_field':
            result, _ = s.solve_field(2, 1, 2, geometry=geometry, **conditions)
        else:
            result = s.solve(geometry=geometry, **conditions)
        return result.T.sum().real
    derivative = jax.grad(loss)(original)
    assert np.isfinite(np.asarray(derivative)).all()
    assert s.complex_dtype == jnp.complex64
    for name in ['thickness', 'structure', 'period', 'theta', 'phi']:
        actual, want = getattr(s, name), getattr(expected, name)
        if want is None:
            assert actual is None, name
            continue
        assert actual.dtype == want.dtype, name
        np.testing.assert_array_equal(actual, want, err_msg=name)
    assert not any(_holds_tracer(value) for key, value in vars(s).items()
                   if key.startswith(('_thickness', '_structure', '_period')))
    if door != 'freeze_geometry':
        # This is the next ordinary solve after the gradient, compared with a solver born
        # at the retained precision and an independent NumPy solve of the same scene.
        got = np.asarray(s.solve().T)
        np.testing.assert_array_equal(got, expected.solve().T)
        np.testing.assert_allclose(got, solver(0, complex_dtype=1).solve().T,
                                   atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_precision_recasting_preserves_the_declared_cell(backend):
    if backend == 2:
        pytest.importorskip('torch')
    cell = wavebit.Lattice.rectangular(600., 750.)
    s = solver(backend, structure=cell.drawn(np.array([[[1., 4., 4., 1.]]])))
    s.complex_dtype = 1
    np.testing.assert_array_equal(s._structure_drawn_on, np.asarray(cell))
    with pytest.raises(ValueError, match='drawn|lattice|cell'):
        s.solve(period=(610., 750.))


@pytest.mark.parametrize('parameter', ['thickness', 'structure'])
@pytest.mark.parametrize('precision_first', [False, True])
def test_restoring_traced_values_keeps_the_cell_declaration(parameter, precision_first):
    cell = wavebit.Lattice.rectangular(600., 750.)
    s = solver(structure=cell.drawn(np.array([[[1., 4., 4., 1.]]])))
    original = getattr(s, parameter)
    def loss(value):
        conditions = (dict(complex_dtype=1) | {parameter: value} if precision_first
                      else {parameter: value} | dict(complex_dtype=1))
        s.freeze_geometry(**conditions)
        return jnp.sum(value.real**2)
    jax.grad(loss)(original)
    np.testing.assert_array_equal(s._structure_drawn_on, np.asarray(cell))
    assert s.complex_dtype == jnp.complex64
    with pytest.raises(ValueError, match='drawn|lattice|cell'):
        s.solve(period=(610., 750.))
