"""Rejected periods preserve the chosen cell, including its declaration."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

import wavebit


@pytest.fixture(params=['numpy', 'jax', 'torch'])
def backend(request):
    pytest.importorskip(request.param)
    return request.param


def solver(backend, **kwargs):
    return _legacy_wavebit.rcwa(backend=backend, wavelength=900., thickness=(170.,),
                       truncation_order=(0, 0), **kwargs)


def test_rejected_period_does_not_declare_the_implicit_unit_cell(backend):
    s = solver(backend)
    s.structure = np.full((1, 2, 2), 4.)
    with pytest.raises(ValueError, match='no cell'):
        s.solve()
    with pytest.raises(ValueError):
        s.period = 'invalid'
    with pytest.raises(ValueError, match='no cell'):
        s.solve()


@pytest.mark.parametrize('basis', [False, True])
@pytest.mark.parametrize('door', ['assignment', 'keyword'])
@pytest.mark.parametrize('invalid', [[], [600., 600., 600.], [[600.], [750.]]])
def test_rejected_period_preserves_both_cell_spellings(backend, basis, door, invalid):
    cell = dict(lattice=np.diag([600., 750.])) if basis else dict(period=(600., 750.))
    s = solver(backend, structure=np.full((1, 2, 2), 4.), **cell)
    with pytest.raises(ValueError, match='period'):
        if door == 'assignment':
            s.period = invalid
        else:
            s.freeze_geometry(period=invalid)
    np.testing.assert_array_equal(s.period, [600., 750.])
    np.testing.assert_array_equal(s.lattice, np.diag([600., 750.]))


@pytest.mark.parametrize('period', [650., [650.], (650., 850.), {'x': 650., 'y': 850.}])
def test_valid_period_still_updates_a_named_basis(backend, period):
    s = solver(backend, lattice=np.diag([600., 750.]))
    s.period = period
    expected = [650., 650.] if isinstance(period, (float, list)) else [650., 850.]
    np.testing.assert_array_equal(s.period, expected)
    np.testing.assert_array_equal(s.lattice, np.diag(expected))


@pytest.mark.parametrize('size', [0, 3])
@pytest.mark.parametrize('compile', [False, True])
def test_traced_invalid_period_is_refused_before_mutation(size, compile):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from wavebit.rcwa._engine.engine import _holds_tracer

    s = solver('jax', lattice=np.diag([600., 750.]))
    def loss(p):
        with s._conditions(dict(period=jnp.repeat(p, size))):
            return jnp.sum(s.period)
    derivative = jax.grad(loss)
    if compile:
        derivative = jax.jit(derivative)
    with pytest.raises(ValueError, match='period'):
        derivative(600.)
    assert not any(_holds_tracer(value) for value in vars(s).values())
    np.testing.assert_array_equal(s.period, [600., 750.])
    np.testing.assert_array_equal(s.lattice, np.diag([600., 750.]))


@pytest.mark.parametrize('named', [False, True])
def test_rejected_lattice_keeps_the_previous_cell_and_declaration(backend, named):
    s = solver(backend, **(dict(lattice=np.diag([600., 750.])) if named else {}))
    before = np.asarray(s.period).copy()
    basis = s._lattice_basis
    # Direct Lattice records with unsupported list vectors are refused by the existing
    # conversion; that error must not publish the unusable record on the solver.
    invalid = wavebit.Lattice([600., 0.], [0., 750.], True)
    with pytest.raises((TypeError, ValueError)):
        s.lattice = invalid
    assert s._lattice_basis is basis
    np.testing.assert_array_equal(s.period, before)
    if not named:
        s.structure = np.full((1, 2, 2), 4.)
        with pytest.raises(ValueError, match='no cell'):
            s.solve()
