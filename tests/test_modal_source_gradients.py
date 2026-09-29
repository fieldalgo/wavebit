"""Modal tangents survive public solve keywords and source field snapshots."""
import numpy as np
import pytest
import wavebit

MIX = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
BASE = np.diag([4., 4., 4.])[None, None, None, ...]
CFG = dict(period=(600., 750.), wavelength=900., thickness=(170.,),
           truncation_order=(0, 0), answers='field', factorization='li',
           fourier_coefficients='integrated')


def solver(backend, source):
    cfg = CFG.copy()
    if source:
        cfg['excitation'] = [wavebit.PlaneWave(), wavebit.PlaneWave(side='bottom', amplitude=.3j)]
    else:
        cfg.update(theta=0., phi=0.)
    s = wavebit.rcwa(backend=backend, **cfg, structure=BASE.copy())
    s.formulation = '4ff'
    return s


def loss(s, xp, t, geometry=None):
    eps = xp.asarray(BASE) + t*xp.asarray(MIX)[None, None, None, ...]
    conditions = {'structure': eps}
    if geometry is not None:
        conditions['geometry'] = geometry
    result = s.solve(**conditions)
    field = result.reconstruct_field([23.], [-19.], [83.])[0, 0, 0]
    return xp.real(field[0])+.37*xp.imag(field[0])+.11*xp.sum(xp.abs(field)**2)


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
@pytest.mark.parametrize('source', [False, True])
def test_retained_tensor_field_sensitivity(backend, source):
    s = solver(2 if backend == 'torch' else 1, source)
    if backend in ('jax', 'jit'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        wavebit.enable_x64()
        geometry = s.freeze_geometry() if backend == 'jit' and not source else None
        calculation = jax.grad(lambda t: loss(s, jnp, t, geometry))
        if backend == 'jit':
            calculation = jax.jit(calculation)
        if source and backend == 'jit':
            with pytest.raises(ValueError, match='Source grouping needs concrete'):
                calculation(jnp.asarray(0.))
            return
        with jax.checking_leaks():
            got = float(calculation(jnp.asarray(0.)))
    else:
        torch = pytest.importorskip('torch')
        t = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        loss(s, torch, t).backward()
        got = float(t.grad)
    h=1e-4
    expected = (float(loss(solver(0, source), np, h))
                -float(loss(solver(0, source), np, -h)))/(2*h)
    assert abs(expected) > 1e-5
    assert got == pytest.approx(expected, rel=2e-5, abs=2e-8)


def test_compiled_modal_gradient_recomputes_the_basis_on_each_call_and_under_vmap():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()
    s = solver(1, False)
    geometry = s.freeze_geometry()
    calculate = jax.jit(jax.value_and_grad(lambda t: loss(s, jnp, t, geometry)))
    points = np.array([0., .02, -.2, .13, 0.])
    with jax.checking_leaks():
        sequential = np.array([calculate(jnp.asarray(t)) for t in points])
        batched = np.array(jax.jit(jax.vmap(calculate))(jnp.asarray(points))).T
    np.testing.assert_allclose(batched, sequential, rtol=2e-12, atol=2e-12)
    for t, (value, gradient) in zip(points, sequential):
        expected_value = loss(solver(0, False), np, t)
        h = 1e-4
        expected_gradient = (loss(solver(0, False), np, t+h)
                             -loss(solver(0, False), np, t-h))/(2*h)
        assert value == pytest.approx(expected_value, abs=2e-12)
        assert gradient == pytest.approx(expected_gradient, rel=2e-5, abs=2e-8)


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
def test_single_precision_degenerate_field_gradient_has_an_independent_reference(backend):
    s = solver(2 if backend == 'torch' else 1, False)
    s.complex_dtype = 1
    if backend in ('jax', 'jit'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        geometry = s.freeze_geometry() if backend == 'jit' else None
        calculate = jax.grad(lambda t: loss(s, jnp, t, geometry))
        if backend == 'jit':
            calculate = jax.jit(calculate)
        got = float(calculate(jnp.asarray(0., dtype=jnp.float32)))
    else:
        torch = pytest.importorskip('torch')
        t = torch.tensor(0., dtype=torch.float32, requires_grad=True)
        loss(s, torch, t).backward()
        got = float(t.grad)
    h = 1e-4
    expected = (float(loss(solver(0, False), np, h))
                -float(loss(solver(0, False), np, -h)))/(2*h)
    assert got == pytest.approx(expected, rel=2e-4, abs=2e-6)
