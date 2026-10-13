"""One call cannot choose two cell spellings and silently discard one."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

import wavebit

CFG = dict(structure=np.full((1, 2, 2), 4.), thickness=(100.,),
           wavelength=900., truncation_order=(0, 0), answers='field')


def solver(backend='numpy', **kwargs):
    return _legacy_wavebit.rcwa(backend=backend, **CFG, **(dict(period=(600., 750.)) | kwargs))


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('door', ['freeze_geometry', 'solve', 'solve_field',
                                 'eps_conv_matrices', 'mu_conv_matrices'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('equal', [False, True])
def test_every_keyword_door_refuses_two_cell_spellings_before_assignment(backend, door, reverse, equal):
    s = solver(backend)
    geometry = s.freeze_geometry()
    lengths = (620., 780.) if equal else (630., 790.)
    cell = dict(period=(620., 780.), lattice=wavebit.Lattice.rectangular(*lengths))
    if reverse:
        cell = dict(reversed(list(cell.items())))
    with pytest.raises(ValueError, match='period and lattice both given'):
        _legacy_wavebit.rcwa(backend=backend, **CFG, **cell)
    before = s.period
    kwargs = dict(thickness=(200.,), **cell)
    with pytest.raises(ValueError, match='period and lattice both given'):
        if door == 'freeze_geometry':
            s.freeze_geometry(**kwargs)
        elif door == 'solve_field':
            s.solve_field(2, 2, 2, geometry=geometry, **kwargs)
        else:
            getattr(s, door)(geometry=geometry, **kwargs)
    assert s.period is before
    np.testing.assert_array_equal(s.thickness, [100.])
    assert s._lattice_basis is None


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('transform', ['grad', 'jit_grad', 'grad_jit'])
def test_conflicting_traced_spellings_are_refused_without_changing_the_solver(reverse, transform):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    s = solver('jax')
    before = s.period
    def loss(p):
        cell = dict(period=[p, 750.], lattice=wavebit.Lattice.rectangular(p, 780.))
        if reverse:
            cell = dict(reversed(list(cell.items())))
        s.freeze_geometry(thickness=[p], **cell)
        return jnp.sum(s.period)
    derivative = jax.grad(jax.jit(loss) if transform == 'grad_jit' else loss)
    if transform == 'jit_grad':
        derivative = jax.jit(derivative)
    with pytest.raises(ValueError, match='period and lattice both given'):
        derivative(620.)
    assert s.period is before
    np.testing.assert_array_equal(s.thickness, [100.])
    assert s._lattice_basis is None


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_withdrawing_the_basis_then_naming_a_period_remains_valid(backend):
    s = solver(backend)
    s.lattice = wavebit.Lattice.oblique(600., 750., 1.2)
    s.freeze_geometry(lattice=None, period=(620., 780.))
    assert s._lattice_basis is None
    np.testing.assert_array_equal(s.period, [620., 780.])
