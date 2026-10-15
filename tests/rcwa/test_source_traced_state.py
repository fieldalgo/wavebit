"""Source keyword differentiation must not poison the reusable facade."""

from wavebit.rcwa._engine.sources import PlaneWave as _legacy_wavebit_PlaneWave

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

CFG = dict(period=(600., 750.), wavelength=900., thickness=(170.,),
           truncation_order=(0, 0), answers='field', structure=np.full((1, 1, 1), 4.))


def wrap(amplitude, container):
    source = _legacy_wavebit_PlaneWave(amplitude=amplitude)
    return {'single': source, 'list': [source], 'dict': {'top': source}}[container]


def intensity(s, xp, excitation):
    r = s.solve(excitation=excitation)
    return xp.sum(xp.abs(r.reconstruct_field([23.], [-19.], [83.]))**2)


@pytest.mark.parametrize('container', ['single', 'list', 'dict'])
@pytest.mark.parametrize('fails', [False, True])
def test_traced_source_keywords_restore_before_an_eager_reuse(container, fails):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit_enable_x64()
    s = _legacy_wavebit_rcwa(backend='jax', **CFG, excitation=_legacy_wavebit_PlaneWave(amplitude=.8j))
    original = s.excitation
    saved = s.solve()
    saved_field = np.asarray(saved.reconstruct_field([23.], [-19.], [83.]))
    def loss(a):
        source = wrap(a, container)
        if fails:
            return s.solve(excitation=source, kx=.2).output_bottom
        return intensity(s, jnp, source)
    with jax.checking_leaks():
        if fails:
            with pytest.raises(ValueError, match='Put kx'):
                jax.grad(loss)(jnp.asarray(1.2))
        else:
            got = float(jax.grad(loss)(jnp.asarray(1.2)))
            reference = _legacy_wavebit_rcwa(backend='numpy', **CFG, excitation=wrap(1.2, container))
            expected = 2*float(intensity(reference, np, wrap(1.2, container)))/1.2
            assert got == pytest.approx(expected, rel=2e-10, abs=2e-10)
    assert s.excitation is original
    np.testing.assert_allclose(s.solve().reconstruct_field([23.], [-19.], [83.]), saved_field, atol=1e-12)
    np.testing.assert_array_equal(saved.reconstruct_field([23.], [-19.], [83.]), saved_field)


def test_a_trace_local_source_solver_keeps_legacy_field_access():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit_enable_x64()
    def loss(a):
        s = _legacy_wavebit_rcwa(backend='jax', **CFG, excitation=_legacy_wavebit_PlaneWave(amplitude=a))
        s.solve()
        return jnp.sum(jnp.abs(s.reconstruct_field([23.], [-19.], [83.]))**2)
    with jax.checking_leaks():
        got = float(jax.grad(loss)(jnp.asarray(1.2)))
    s = _legacy_wavebit_rcwa(backend='numpy', **CFG, excitation=_legacy_wavebit_PlaneWave(amplitude=1.2))
    expected = 2*float(intensity(s, np, _legacy_wavebit_PlaneWave(amplitude=1.2)))/1.2
    assert got == pytest.approx(expected, rel=2e-10, abs=2e-10)
