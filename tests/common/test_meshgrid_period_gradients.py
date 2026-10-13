"""Period spellings have the same coordinates and analytical derivatives."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest



SPELLINGS = ['scalar', 'singleton', 'list', 'tuple', 'array', 'mixed_x', 'mixed_y']


def period_value(spelling, p, stack):
    if spelling == 'scalar':
        return p
    if spelling == 'singleton':
        return [p]
    if spelling == 'list':
        return [p, 2*p]
    if spelling == 'tuple':
        return p, 2*p
    if spelling == 'array':
        return stack([p, 2*p])
    if spelling == 'mixed_x':
        return [p, 1200.]
    return [600., p]


def expected_derivative(spelling):
    x = (np.arange(5) + .5)/5 - .5
    y = (np.arange(3) + .5)/3 - .5
    dx = 0 if spelling == 'mixed_y' else 1
    dy = 0 if spelling == 'mixed_x' else 1 if spelling in ('scalar', 'singleton', 'mixed_y') else 4
    return 2*600*(3*np.sum(x*x)*dx + 5*np.sum(y*y)*dy)


@pytest.mark.parametrize('spelling', SPELLINGS)
@pytest.mark.parametrize('transform', ['grad', 'jit_grad', 'grad_jit'])
def test_jax_period_coordinate_derivative(spelling, transform):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()

    def loss(p):
        period = period_value(spelling, p, jnp.stack)
        x, y = _legacy_wavebit.meshgrid_xy((3, 5), period=period, backend='jax')
        return jnp.sum(x*x + y*y)
    derivative = jax.grad(jax.jit(loss) if transform == 'grad_jit' else loss)
    if transform == 'jit_grad':
        derivative = jax.jit(derivative)
    assert float(derivative(600.)) == pytest.approx(expected_derivative(spelling), rel=1e-12)


@pytest.mark.parametrize('spelling', SPELLINGS)
def test_torch_period_coordinate_derivative(spelling):
    torch = pytest.importorskip('torch')
    p = torch.tensor(600., dtype=torch.float64, requires_grad=True)
    period = period_value(spelling, p, torch.stack)
    x, y = _legacy_wavebit.meshgrid_xy((3, 5), period=period, backend='torch')
    (x*x + y*y).sum().backward()
    assert float(p.grad) == pytest.approx(expected_derivative(spelling), rel=1e-12)


@pytest.mark.parametrize('size', [0, 3])
def test_a_traced_period_cannot_drop_extra_lengths(size):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()
    @jax.jit
    def grid(p):
        return _legacy_wavebit.meshgrid_xy((3, 5), period=jnp.repeat(p, size), backend='jax')
    with pytest.raises(ValueError, match='period'):
        grid(600.)
