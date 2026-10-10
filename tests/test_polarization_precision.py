"""Changing precision recasts all stored polarization coordinates."""
import numpy as np
import pytest
import wavebit


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('initial', [0, 1])
@pytest.mark.parametrize('chart', ['chi', 'polarization'])
def test_polarization_precision_follows_solver_precision(backend, initial, chart):
    if backend == 'jax':
        pytest.importorskip('jax')
    if backend == 'torch':
        pytest.importorskip('torch')
    options = {'chi': .213456789} if chart == 'chi' else {'polarization': (1+.245678912j, .321234567-.412345678j)}
    cfg = dict(period=(600., 750.), wavelength=900., thickness=(170.,),
               truncation_order=(0, 0), theta=.13, phi=.21, answers='field',
               structure=np.full((1, 2, 2), 4.), **options)
    s = wavebit.rcwa(backend=backend, complex_dtype=initial, **cfg)
    # Widen the values actually stored: lost single-precision bits cannot be recovered.
    accepted = getattr(s, chart)
    s.complex_dtype = 1-initial
    resolved = cfg | {chart: accepted, 'theta': s.theta, 'phi': s.phi}
    if chart == 'chi':
        resolved['psi'] = s.psi
    expected = wavebit.rcwa(backend=backend, complex_dtype=1-initial, **resolved)
    values = [s.chi] if chart == 'chi' else s.polarization
    for value in values:
        assert value.dtype == (s.float_dtype if chart == 'chi' else s.complex_dtype)
    np.testing.assert_allclose(np.asarray(s.solve().T), np.asarray(expected.solve().T), rtol=0, atol=0)


@pytest.mark.parametrize('chart', ['chi', 'polarization'])
def test_a_kept_precision_keyword_recasts_the_restored_gradient_parameter(chart):
    from functools import partial
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    options = {'chi': .21} if chart == 'chi' else {'polarization': (1.+.24j, .32-.41j)}
    cfg = dict(period=(600., 750.), wavelength=900., thickness=(170.,),
               truncation_order=(0, 0), theta=.13, phi=.21, answers='field',
               structure=np.full((1, 2, 2), 4.), **options)
    s = wavebit.rcwa(backend='jax', **cfg)
    def loss(result):
        regularization = (s.chi**2 if chart == 'chi' else
                          sum(jnp.abs(v)**2 for v in s.polarization))
        return result.T.sum().real + .1*regularization
    with jax.checking_leaks():
        got = s.grad([chart], partial(s.solve, complex_dtype=1), loss)[chart]
    values = [s.chi] if chart == 'chi' else s.polarization
    for v in values:
        assert v.dtype == (s.float_dtype if chart == 'chi' else s.complex_dtype)
    def reference(coords):
        chosen = (float(coords[0]) if chart == 'chi' else
                  (coords[0]+1j*coords[1], coords[2]+1j*coords[3]))
        r = wavebit.rcwa(backend='numpy', **(cfg | {chart: chosen})).solve()
        return float(r.T.sum().real + .1*np.sum(coords**2))
    point = np.array([.21] if chart == 'chi' else [1., .24, .32, -.41])
    expected = []
    for index in range(len(point)):
        a, b = point.copy(), point.copy()
        a[index] += 1e-4
        b[index] -= 1e-4
        expected.append((reference(a)-reference(b))/2e-4)
    actual = ([float(got)] if chart == 'chi' else
              [float(jnp.real(got[0])), float(jnp.imag(got[0])),
               float(jnp.real(got[1])), float(jnp.imag(got[1]))])
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
