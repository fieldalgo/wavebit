"""Native scalar design variables have the same length meaning as Python scalars."""

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest



def native(backend, value):
    if backend == 'numpy':
        return np.asarray(value)
    if backend == 'jax':
        jnp = pytest.importorskip('jax.numpy')
        _legacy_wavebit_enable_x64()
        return jnp.asarray(value, dtype=jnp.float64)
    torch = pytest.importorskip('torch')
    return torch.tensor(value, dtype=torch.float64)


def solver(backend='numpy', **kwargs):
    cfg = dict(period=(600., 750.), thickness=(170.,),
               structure=np.array([[[1., 4., 4., 1.]]]),
               wavelength=900., truncation_order=(1, 0), theta=.13, tetm='tm')
    return _legacy_wavebit_rcwa(backend=backend, **(cfg | kwargs))


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('quantity,value,expected', [('period', 600., [600., 600.]),
                                                  ('thickness', 170., [170.])])
def test_native_scalar_lengths_follow_python_scalars(backend, quantity, value, expected):
    s = solver(backend, **{quantity: native(backend, value)})
    np.testing.assert_array_equal(getattr(s, quantity), expected)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('door', ['constructor', 'assignment', 'keyword'])
@pytest.mark.parametrize('value', [[[170.]], [[170., 230.]]])
def test_multidimensional_thickness_is_refused_before_it_can_broadcast(backend, door, value):
    value = native(backend, value)
    s = solver(backend)
    before = np.asarray(s.thickness).copy()
    with pytest.raises(ValueError, match='thickness.*one-dimensional'):
        if door == 'constructor':
            solver(backend, thickness=value)
        elif door == 'assignment':
            s.thickness = value
        else:
            s.freeze_geometry(thickness=value)
    np.testing.assert_array_equal(s.thickness, before)


@pytest.mark.parametrize('quantity,value,count', [('period', 600., 2), ('thickness', 170., 1)])
@pytest.mark.parametrize('transform', ['grad', 'jit_grad', 'grad_jit'])
def test_scalar_jax_length_derivatives_keep_every_stored_component(quantity, value, count, transform):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    s = solver('jax')
    original = np.asarray(getattr(s, quantity)).copy()
    def loss(v):
        with s._conditions({quantity: v}):
            return jnp.sum(getattr(s, quantity)**2)
    derivative = jax.grad(jax.jit(loss) if transform == 'grad_jit' else loss)
    if transform == 'jit_grad':
        derivative = jax.jit(derivative)
    assert float(derivative(value)) == pytest.approx(2*count*value)
    np.testing.assert_array_equal(getattr(s, quantity), original)


@pytest.mark.parametrize('quantity,value,count', [('period', 600., 2), ('thickness', 170., 1)])
def test_scalar_torch_lengths_keep_the_callers_leaf(quantity, value, count):
    torch = pytest.importorskip('torch')
    leaf = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    s = solver('torch', **{quantity: leaf})
    getattr(s, quantity).square().sum().backward()
    assert float(leaf.grad) == pytest.approx(2*count*value)


@pytest.mark.parametrize('quantity,value', [('period', 600.), ('thickness', 170.)])
@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_scalar_length_solve_derivatives_match_fresh_numpy(quantity, value, backend):
    s = solver(backend)
    geometry = s.freeze_geometry()
    if backend == 'jax':
        import jax
        def loss(v):
            return s.solve(geometry=geometry, **{quantity: v}).T.sum().real
        got = float(jax.jit(jax.grad(loss))(value))
    else:
        torch = pytest.importorskip('torch')
        leaf = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        s.solve(geometry=geometry, **{quantity: leaf}).T.sum().real.backward()
        got = float(leaf.grad)
    h = 1e-3
    def reference(v):
        return solver('numpy', **{quantity: v}).solve().T.sum().real
    expected = (reference(value+h)-reference(value-h))/(2*h)
    assert abs(expected) > 1e-7
    assert got == pytest.approx(expected, abs=1e-9, rel=2e-6)
