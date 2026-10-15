"""ADE diagnostics distinguish residual oscillator motion from field decay."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

from wavebit.fdtd._periodic.grid import fdtd_grid as _legacy_wavebit_fdtd_grid
import numpy as np
import pytest


def exact_activity(strength, steps=43, dt=.1):
    # Uniform lossless D=E+P=1: P oscillates about s/(1+s), with P_0=P_-1=0.
    angle = np.arccos(1-.5*(1+strength)*dt**2)
    n = np.arange(steps+1)
    p = strength/(1+strength)*(1-np.cos((n+.5)*angle)/np.cos(.5*angle))
    velocity = np.diff(p)/dt
    return (p[1:]**2+velocity**2)/strength


def run(strength=.5, backend='numpy'):
    e = np.zeros((3, 4, 2, 2));e[0] = 1
    return _legacy_wavebit_fdtd_grid(np.ones((4, 2, 2)), spacing=(1.,)*3, dt=.1, steps=43,
                            electric=e, poles=[(1., 0., strength)], backend=backend,
                            convergence_window=6, dft_subsample=2,
                            wavelengths=(2.,), monitor_z=(1,), record_field=7)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_material_activity_matches_closed_form_discrete_oscillator(backend):
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    if backend == 'torch':
        pytest.importorskip('torch')
    c = run(backend=backend)['convergence']
    activity = exact_activity(.5)
    assert float(c['material_activity_peak'][0]) == pytest.approx(16*activity.max(), rel=1e-12)
    assert float(c['material_activity_tail'][0]) == pytest.approx(16*activity[-6:].max(), rel=1e-12)
    assert float(c['material_decay'][0]) == pytest.approx(np.sqrt(activity[-6:].max()/activity.max()), rel=1e-12)
    assert not bool(c['material_decayed'])
    assert not bool(c['state_decayed'])


def test_weakly_coupled_lorentz_motion_survives_small_electric_field():
    e = np.zeros((3, 4, 2, 2));e[0] = 1
    c = _legacy_wavebit_fdtd_grid(np.ones((4, 2, 2)), spacing=(1.,)*3, dt=.02, steps=2000,
                         electric=e, conductivity=10., poles=[(1., .01, .01)],
                         convergence_window=100)['convergence']
    assert c['field_decayed']
    assert c['material_signal_present'][0]
    assert c['material_decay'][0] > .5
    assert not c['material_decayed']
    assert not c['state_decayed']


def test_static_drude_polarization_is_not_residual_oscillator_motion():
    e = np.zeros((3, 4, 2, 2));e[0] = 1
    out = _legacy_wavebit_fdtd_grid(np.ones((4, 2, 2)), spacing=(1.,)*3, dt=.1, steps=400,
                           electric=e, poles=[(0., 2., 1.)], convergence_window=30)
    assert np.max(abs(out['polarization'][0])) > .9
    assert out['convergence']['material_decayed']
    assert out['convergence']['state_decayed']


@pytest.mark.parametrize('poles', [(), [(1., 0., 0.)]])
def test_absent_or_unexcited_poles_do_not_block_material_flag(poles):
    out = _legacy_wavebit_fdtd_grid(np.ones((4, 2, 2)), spacing=(1.,)*3, dt=.1, steps=10,
                           poles=poles, convergence_window=3)
    c = out['convergence']
    assert not c['material_signal_present'].any()
    assert c['material_decayed']
    assert not c['state_decayed']  # A completely absent field is not convergence evidence.
    assert np.isnan(c['material_decay']).all()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_material_decay_gradient_matches_closed_form(backend):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        derivative = jax.jit(jax.grad(lambda s: run(s, 'jax')['convergence']['material_decay'][0]))(.5)
    else:
        torch = pytest.importorskip('torch')
        strength = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
        derivative, = torch.autograd.grad(run(strength, 'torch')['convergence']['material_decay'][0], strength)
    def expected(strength):
        activity = exact_activity(strength)
        return np.sqrt(activity[-6:].max()/activity.max())
    delta = 1e-5
    finite = (expected(.5+delta)-expected(.5-delta))/(2*delta)
    assert abs(finite) > .01
    assert float(derivative) == pytest.approx(finite, rel=1e-6, abs=1e-9)


def test_scene_reports_material_and_combined_decay_for_all_bases():
    _, d = _legacy_wavebit_fdtd(structure=np.ones((1, 1, 1))*1.5, thickness=[.2], top=1., bot=1.,
                        period=.2, wavelength=1., backend='numpy', resolution=10, n_periods=25,
                        tetm='te', truncation_order=0, dispersion=np.ones((1, 1, 1), int),
                        poles={1: [(6., .2, 4.)]}, return_diagnostics=True, convergence_window=40)
    for key in ('material_decayed', 'state_decayed'):
        assert d['convergence'][key] == all(b[k][key] for b in d['convergence']['bases']
                                           for k in ('actual', 'reference'))
    for basis in d['convergence']['bases']:
        assert basis['reference']['material_decay'].shape == (0,)
        assert basis['actual']['material_decay'].shape == (1,)


def test_material_activity_uses_nonuniform_native_volumes_and_zero_strength_mask():
    widths = np.array([.5, .7, 1.1, 1.3])
    strengths = np.array([0., .2, .5, 1.])
    e = np.zeros((3, 4, 2, 2));e[2] = 1
    out = _legacy_wavebit_fdtd_grid(np.ones((4, 2, 2)), spacing=(1.,)*3, z_spacing=widths,
                           dt=.1, steps=43, electric=e,
                           poles=[(1., 0., strengths[:, None, None])], convergence_window=6)
    # Longitudinal fields vary only along z, so the four oscillators have zero curl.
    angle = np.arccos(1-.5*(1+strengths)*.1**2)
    n = np.arange(44)[:, None]
    p = strengths/(1+strengths)*(1-np.cos((n+.5)*angle)/np.cos(.5*angle))
    density = (p[1:]**2+(np.diff(p, axis=0)/.1)**2)/np.where(strengths > 0, strengths, 1)
    activity = 4*np.sum(widths*density, axis=1)
    c = out['convergence']
    assert float(c['material_activity_peak'][0]) == pytest.approx(activity.max(), rel=1e-12)
    assert float(c['material_activity_tail'][0]) == pytest.approx(activity[-6:].max(), rel=1e-12)
