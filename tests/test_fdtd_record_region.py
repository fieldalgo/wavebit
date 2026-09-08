"""Region histories preserve exact modes, coordinates and differentiability."""
import numpy as np
import pytest
import wavebit
from test_fdtd_grid import initial_mode


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_exact_mode_region_preserves_staggering_and_field_state(backend):
    if backend == 1:
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    if backend == 2:
        pytest.importorskip('torch')
    host = (lambda v: v.detach().numpy()) if backend == 2 else np.asarray
    e, h, omega = initial_mode()
    kw = dict(spacing=(1.,)*3, dt=.2, steps=20, electric=e, magnetic=h,
              record_field=3, record_components=('ey', 'hx'), backend=backend)
    region = (slice(2, 14, 3), -1, slice(None, None, 2))
    result = wavebit.fdtd_grid(np.full((16, 2, 2), 2.), **kw, record_region=region)
    full = wavebit.fdtd_grid(np.full((16, 2, 2), 2.), **kw)
    for name, initial in [('ey', e[1]), ('hx', h[0])]:
        field = result['fields'][name]
        assert field['values'].shape == (7, 4, 1, 1)
        expected = initial[None, 2:14:3, -1:, ::2]*np.exp(-1j*omega*(np.arange(0, 20, 3)+1)*.2)[:, None, None, None]
        np.testing.assert_allclose(host(field['values']), expected, atol=2e-14)
        np.testing.assert_array_equal(host(field['values']), host(full['fields'][name]['values'])[:, 2:14:3, -1:, ::2])
        for axis, selection in zip(('z', 'y', 'x'), (slice(2, 14, 3), slice(-1, None), slice(None, None, 2))):
            np.testing.assert_array_equal(host(field[axis]), host(full['fields'][name][axis])[selection])
        np.testing.assert_array_equal(host(field['t']), host(full['fields'][name]['t']))
    np.testing.assert_array_equal(host(result['E']), host(full['E']))


def test_nonuniform_coordinates_and_storage_count():
    widths = np.linspace(.7, 1.3, 12)
    out = wavebit.fdtd_grid(np.ones((12, 4, 6)), spacing=(1., .5, .25), dt=.1, steps=10,
                            z_spacing=widths, record_field=2, record_components=('ex', 'hz'),
                            record_region=(slice(-5, None, 2), 2, slice(1, 5, 2)))
    for name, field in out['fields'].items():
        np.testing.assert_allclose(field['z'], np.r_[0., np.cumsum(widths[:-1])][-5::2])
        assert field['values'].shape == (5, 3, 1, 2)
    assert sum(v['values'].nbytes for v in out['fields'].values()) == 5*3*1*2*2*16


@pytest.mark.parametrize('backend', [1, 2])
def test_region_material_gradient_matches_full_field_objective(backend):
    e, h, _ = initial_mode()
    def objective(value, which, cropped):
        epsilon = value*np.ones((16, 2, 2)) if which == 0 else value*ones
        out = wavebit.fdtd_grid(epsilon, spacing=(1.,)*3, dt=.2, steps=15,
                                electric=e, magnetic=h, backend=which, checkpoint_steps=3,
                                record_field=4, record_components='ey',
                                record_region=(slice(3, 11, 2), 0, 1) if cropped else None)
        values = out['fields']['ey']['values']
        if not cropped:
            values = values[:, 3:11:2, :1, 1:2]
        return (abs(values)**2).sum()
    if backend == 1:
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        ones = jnp.ones((16, 2, 2))
        actual = jax.jit(jax.grad(lambda v: objective(v, 1, True)))(2.)
    else:
        torch = pytest.importorskip('torch')
        ones = torch.ones((16, 2, 2), dtype=torch.float64)
        value = torch.tensor(2., dtype=torch.float64, requires_grad=True)
        actual, = torch.autograd.grad(objective(value, 2, True), value)
    delta = 1e-5
    expected = (objective(2.+delta, 0, False)-objective(2.-delta, 0, False))/(2*delta)
    assert float(actual) == pytest.approx(expected, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize('region', [(1, 2), (True, 0, 0), (16, 0, 0), (-17, 0, 0),
                                    (slice(3, 3), 0, 0), (slice(None, None, -1), 0, 0),
                                    (slice(None, None, 0), 0, 0), (slice(.5, 3), 0, 0)])
def test_invalid_regions(region):
    with pytest.raises(ValueError, match='record_region'):
        wavebit.fdtd_grid(np.ones((16, 2, 2)), spacing=(1.,)*3, dt=.2, steps=2,
                          record_field=1, record_region=region)


def test_region_requires_recording():
    with pytest.raises(ValueError, match='record_region requires'):
        wavebit.fdtd_grid(np.ones((16, 2, 2)), spacing=(1.,)*3, dt=.2, steps=2,
                          record_region=(0, 0, 0))
