"""Pulse diagnostics against an exactly solvable homogeneous conductive field."""
import numpy as np
import pytest
import wavebit


def run(backend=0, **kwargs):
    e = np.zeros((3, 4, 2, 2), complex)
    e[0] = 1
    return wavebit.fdtd_grid(np.ones((4, 2, 2)), spacing=(1., 1., 1.),
                            dt=.2, steps=60, electric=e, conductivity=2.,
                            backend=backend, convergence_window=10,
                            wavelengths=(1.3,), monitor_z=(1,), **kwargs)


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_conductive_uniform_field_has_exact_decay_and_dft(backend):
    if backend == 1:
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    if backend == 2:
        pytest.importorskip('torch')
    result = run(backend, record_field=7, record_components='ex')
    c = result['convergence']
    gain = (1-.2)/(1+.2)
    # Spatially uniform E has zero curl and decays geometrically. Initial H=0.
    assert float(c['field_decay']) == pytest.approx(gain**51, rel=1e-10)
    n = np.arange(1, 61)
    dft = gain**n * np.exp(2j*np.pi*n*.2/1.3)
    expected = abs(dft[-10:].sum()/dft.sum())
    assert float(c['spectral_tail'][0, 0]) == pytest.approx(expected, rel=1e-10)
    assert bool(c['field_decayed'])
    assert bool(c['source_inactive'])
    assert float(abs(result['E'][0, 0, 0, 0])) == pytest.approx(gain**60, rel=1e-10)


def test_zero_field_is_not_evidence_of_decay():
    result = wavebit.fdtd_grid(np.ones((4, 2, 2)), spacing=(1., 1., 1.),
                              dt=.2, steps=10, convergence_window=3,
                              wavelengths=(1.,), monitor_z=(1,))
    c = result['convergence']
    assert not c['signal_present']
    assert not c['field_decayed']
    assert np.isnan(c['field_decay'])
    assert np.isnan(c['spectral_tail']).all()


def test_lossless_persistent_field_and_active_source_do_not_pass():
    e = np.ones((3, 4, 2, 2), complex)
    for sigma, source, pulse in [(0., None, None), (2., e, np.ones(60))]:
        c = wavebit.fdtd_grid(np.ones((4, 2, 2)), spacing=(1., 1., 1.),
                             dt=.2, steps=60, electric=e, conductivity=sigma,
                             source=source, pulse=pulse, convergence_window=10)['convergence']
        assert not c['field_decayed']
        if source is None:
            assert c['field_decay'] == pytest.approx(1.)
        else:
            assert not c['source_inactive']


@pytest.mark.parametrize('window', [True, 0, -1, 10, 1.5])
def test_invalid_windows(window):
    with pytest.raises(ValueError, match='convergence_window'):
        wavebit.fdtd_grid(np.ones((4, 2, 2)), spacing=(1., 1., 1.), dt=.2,
                          steps=10, convergence_window=window)


@pytest.mark.parametrize('backend', [1, 2])
def test_diagnostic_gradient_matches_exact_geometric_decay(backend):
    e = np.zeros((3, 4, 2, 2), complex)
    e[0] = 1
    def objective(sigma):
        return wavebit.fdtd_grid(np.ones((4, 2, 2)), spacing=(1., 1., 1.),
                                 dt=.2, steps=12, electric=e, conductivity=sigma,
                                 backend=backend, convergence_window=4)['convergence']['field_decay']
    gain = (1-.1*2)/(1+.1*2)
    expected = 9*gain**8*(-.2/(1+.1*2)**2)
    if backend == 1:
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        value = jax.jit(jax.grad(objective))(2.)
    else:
        torch = pytest.importorskip('torch')
        sigma = torch.tensor(2., dtype=torch.float64, requires_grad=True)
        value, = torch.autograd.grad(objective(sigma), sigma)
    assert float(value) == pytest.approx(expected, rel=1e-10)


def test_scene_reports_both_actual_and_reference_bases_without_changing_result():
    settings = dict(structure=np.array([[[2.]]]), thickness=[.2], top=1., bot=1.,
                    period=.4, wavelength=1., backend=0, resolution=10,
                    n_periods=30, truncation_order=0, tetm='te', incidence_side='bottom',
                    grading=True, return_diagnostics=True)
    plain, _ = wavebit.fdtd(**settings)
    checked, diagnostic = wavebit.fdtd(**settings, convergence_window=25)
    np.testing.assert_array_equal(plain.given.t_s, checked.given.t_s)
    for basis in diagnostic['convergence']['bases']:
        for kind in ('reference', 'actual'):
            c = basis[kind]
            assert bool(c['source_inactive'])
            assert bool(c['signal_present'])
            assert c['spectral_tail'].shape == (1, 2)
    assert diagnostic['incidence_side'] == 'bottom'


def test_nonuniform_field_norm_uses_native_control_volumes():
    widths = np.array([.5, 1., 1.5, 2.])
    e = np.zeros((3, 4, 2, 2), complex)
    e[0, 0] = 1
    result = wavebit.fdtd_grid(np.ones((4, 2, 2)), spacing=(1., 1., 1.),
                              z_spacing=widths, dt=.1, steps=8, electric=e,
                              conductivity=1., record_field=1, convergence_window=3)
    dual = (widths+np.roll(widths, 1))/2
    weights = dict(ex=dual, ey=dual, ez=widths, hx=widths, hy=widths, hz=dual)
    norm = sum(np.sum(abs(record['values'])**2*weights[name][None, :, None, None],
                      axis=(1, 2, 3)) for name, record in result['fields'].items())
    initial = 4*dual[0]
    expected = np.sqrt(norm[-3:].max()/max(initial, norm.max()))
    assert result['convergence']['field_decay'] == pytest.approx(expected)
