"""Nested basis containers preserve derivatives and their declared matrix shape."""
import numpy as np
import pytest

import wavebit

SPELLINGS = ['list', 'tuple', 'rows', 'array', 'numpy_scalar', 'numpy_array', 'literal']
CONSTANT = 180.12345678912345


def basis_value(p, spelling, stack):
    rows = [[p, .3*p], [.1*p, 2*p]]
    if spelling in ('numpy_scalar', 'numpy_array', 'literal'):
        rows[0][1] = (np.float64(CONSTANT) if spelling == 'numpy_scalar' else
                      np.asarray(CONSTANT) if spelling == 'numpy_array' else CONSTANT)
    if spelling == 'tuple':
        return tuple(map(tuple, rows))
    if spelling in ('rows', 'array'):
        rows = [stack(row) for row in rows]
        return stack(rows) if spelling == 'array' else rows
    return rows


def expected_derivative(spelling):
    u, v = np.meshgrid((np.arange(5)+.5)/5-.5, (np.arange(3)+.5)/3-.5)
    y = 600*(.1*u+2*v)
    if spelling in ('numpy_scalar', 'numpy_array', 'literal'):
        x, dx = 600*u+CONSTANT*v, u
    else:
        x, dx = 600*(u+.3*v), u+.3*v
    return 2*np.sum(x*dx + y*(.1*u+2*v))


@pytest.mark.parametrize('spelling', SPELLINGS)
@pytest.mark.parametrize('transform', ['grad', 'jit_grad', 'grad_jit'])
def test_jax_nested_lattice_derivative(spelling, transform):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()
    def loss(p):
        x, y = wavebit.meshgrid_xy((3, 5), lattice=basis_value(p, spelling, jnp.stack), backend=1)
        return jnp.sum(x*x+y*y)
    derivative = jax.grad(jax.jit(loss) if transform == 'grad_jit' else loss)
    if transform == 'jit_grad':
        derivative = jax.jit(derivative)
    assert float(derivative(600.)) == pytest.approx(expected_derivative(spelling), rel=1e-12)


@pytest.mark.parametrize('spelling', SPELLINGS)
def test_torch_nested_lattice_derivative(spelling):
    torch = pytest.importorskip('torch')
    p = torch.tensor(600., dtype=torch.float64, requires_grad=True)
    x, y = wavebit.meshgrid_xy((3, 5), lattice=basis_value(p, spelling, torch.stack), backend=2)
    (x*x+y*y).sum().backward()
    assert float(p.grad) == pytest.approx(expected_derivative(spelling), rel=1e-12)


@pytest.mark.parametrize('shape', [(2, 3), (3, 2), (2,), (1, 2), (2, 1)])
@pytest.mark.parametrize('nested', [False, True])
def test_tracing_cannot_hide_a_malformed_basis(shape, nested):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()
    @jax.jit
    def grid(p):
        basis = jnp.ones(shape)*p
        if nested:
            basis = [list(row) if row.ndim else row for row in basis]
        return wavebit.meshgrid_xy((3, 5), lattice=basis, backend=1)
    with pytest.raises(ValueError, match='lattice.*2x2'):
        grid(600.)
