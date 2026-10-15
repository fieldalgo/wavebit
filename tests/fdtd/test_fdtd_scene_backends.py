"""Typed scenes, physical amplitudes and material graphs through the common core."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import numpy as np
import pytest

from wavebit._common.periodic import Stack

from wavebit._common.periodic import Incidence
from wavebit import Material


def options(**changes):
    return dict(dict(thickness=[.3], top=1., bot=1., period=.4, wavelength=1.,
                     resolution=20, n_periods=30, truncation_order=0, tetm='te'), **changes)


@pytest.mark.slow
def test_slab_complex_amplitudes_converge_to_airy():
    phase = 2*np.pi*1.5*.3
    interface = (1-1.5)/(1+1.5)
    r = interface*(1-np.exp(2j*phase))/(1-interface**2*np.exp(2j*phase))
    t = (1-interface**2)*np.exp(1j*phase)/(1-interface**2*np.exp(2j*phase))
    errors = []
    for resolution in (40, 80):
        result = _legacy_wavebit_fdtd(structure=np.full((1, 1, 1), 2.25), backend='numpy',
                              **options(resolution=resolution))
        errors.append(max(abs(result.incident.r_s.item()-r), abs(result.incident.t_s.item()-t)))
        np.testing.assert_allclose(result.incident.R+result.incident.T, 1, atol=.002)
        np.testing.assert_allclose(result.s_inc.r_s, -result.p_inc.r_p, atol=1e-12)
    assert errors[1] < errors[0]*.4
    assert errors[1] < .003


def test_magnetic_half_spaces_and_conical_jones_against_rcwa():
    kw = options(top=Material(1.2, 1.1), bot=Material(2., 1.3), tetm=None,
                 theta=.28, phi=.41, polarization=(1., .6j), resolution=40)
    material = np.full((1, 1, 1), 2.25)
    permeability = np.full_like(material, 1.15)
    result = _legacy_wavebit_fdtd(structure=material, mu=permeability, backend='numpy', **kw).incident
    reference_kw = {k: v for k, v in kw.items() if k not in ('resolution', 'n_periods')}
    reference = _legacy_wavebit_rcwa(structure=material, mu=permeability, backend='numpy', **reference_kw).solve().incident
    for key in ('r_s', 'r_p', 't_s', 't_p'):
        np.testing.assert_allclose(getattr(result, key), getattr(reference, key), atol=.009)
    np.testing.assert_allclose(result.R+result.T, 1, atol=.006)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_typed_material_gradients_match_finite_difference_and_numpy(backend):
    kw = options(resolution=10, n_periods=25)
    incidence = Incidence(wavelength=kw.pop('wavelength'), tetm=kw.pop('tetm'))
    thickness = kw.pop('thickness')
    scene_kw = {k: kw.pop(k) for k in ('top', 'bot', 'period')}
    def solve(value, which):
        stack = Stack.from_structure(value.reshape(1, 1, 1), thickness, **scene_kw)
        return _legacy_wavebit_fdtd(stack, incidence, backend=which, **kw)
    expected = solve(np.array(2.25), 'numpy')
    delta = 1e-4
    finite = (solve(np.array(2.25+delta), 'numpy').incident.T.sum()
              - solve(np.array(2.25-delta), 'numpy').incident.T.sum())/(2*delta)
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        compiled = jax.jit(lambda v: solve(v, 'jax'))
        actual = compiled(jnp.array(2.25))
        assert np.isnan(np.asarray(compiled(jnp.array(.01)).incident.T)).all()
        derivative = jax.grad(lambda v: compiled(v).incident.T.sum())(jnp.array(2.25))
        convert = np.asarray
    else:
        torch = pytest.importorskip('torch')
        value = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
        actual = solve(value, 'torch')
        derivative = torch.autograd.grad(actual.incident.T.sum(), value)[0].detach().numpy()
        convert = lambda v: v.detach().numpy()
    for key in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
        np.testing.assert_allclose(convert(getattr(actual.incident, key)), getattr(expected.incident, key), atol=1e-11)
    np.testing.assert_allclose(derivative, finite, rtol=2e-5, atol=1e-7)
    assert abs(finite) > .01


def test_asymmetric_diffraction_order_phase_against_rcwa():
    kw = options(period=1.6, thickness=[.2], wavelength=.9, resolution=40,
                 truncation_order=1, theta=.12)
    structure = np.array([[[3., 2., 1., 1.]]])
    actual = _legacy_wavebit_fdtd(structure=structure, backend='numpy', **kw).incident
    reference_kw = {k: v for k, v in kw.items() if k not in ('resolution', 'n_periods')}
    reference_kw['truncation_order'] = 8
    ref = _legacy_wavebit_rcwa(structure=structure, backend='numpy', **reference_kw).solve().incident
    for key in ('r_s', 't_s'):
        np.testing.assert_allclose(getattr(actual, key), getattr(ref, key)[:, 7:10], atol=.025)


def test_invalid_scene_features_are_not_silently_routed():
    structure = np.ones((1, 1, 1))
    with pytest.raises(ValueError, match='explicit resolution'):
        _legacy_wavebit_fdtd(structure=structure, backend='jax', **options(resolution=None))
    with pytest.raises(ValueError, match='too short'):
        _legacy_wavebit_fdtd(structure=structure, backend='numpy', **options(n_periods=3))
    with pytest.raises(ValueError, match='must name'):
        _legacy_wavebit_fdtd(structure=structure, backend='numpy', dispersion=np.ones((1, 1, 1), int),
                      poles={2: [(1., .1, .2)]}, **options())
    with pytest.raises(ValueError, match='grading'):
        _legacy_wavebit_fdtd(structure=structure, backend='numpy', grading=.2, **options())


def test_crossed_jones_band_and_dispersive_torch_gradient():
    torch = pytest.importorskip('torch')
    structure = np.array([[[1.7, 2.1], [1.2, 1.8]]])
    labels = np.array([[[1, 0], [0, 1]]])
    kw = options(period=[.4, .4], thickness=[.2], wavelength=[.9, 1., 1.1],
                 tetm=None, polarization=(1., 1j), theta=.2, phi=.6,
                 resolution=10, n_periods=30, dispersion=labels)
    def run(strength, backend):
        return _legacy_wavebit_fdtd(structure=structure, backend=backend,
                            poles={1: [(8., .3, strength)]}, **kw)
    strength = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    actual = run(strength, 'torch')
    reference = run(2., 'numpy')
    for a, b in zip(actual, reference):
        np.testing.assert_allclose(a.incident.t_s.detach(), b.incident.t_s, atol=1e-10)
        assert .8 < b.incident.R.sum()+b.incident.T.sum() < 1.05
    gradient = torch.autograd.grad(sum(v.incident.T.sum() for v in actual), strength)[0]
    delta = 1e-4
    finite = (sum(v.incident.T.sum() for v in run(2.+delta, 'numpy'))
              - sum(v.incident.T.sum() for v in run(2.-delta, 'numpy')))/(2*delta)
    np.testing.assert_allclose(gradient.detach(), finite, rtol=2e-5, atol=1e-7)
    assert abs(finite) > 1e-4


def test_scene_transmission_against_frozen_meep_public_api_run():
    with np.load(repository_path('tests/references/golden_native_grid_external.npz')) as ref:
        wavelengths = ref['wavelengths']
        expected = ref['meep-grid-80_power']
    results = _legacy_wavebit_fdtd(structure=np.full((1, 1, 1), 2.25), backend='numpy',
                           **options(thickness=[.5], period=.1, wavelength=wavelengths,
                                     resolution=80, n_periods=40))
    np.testing.assert_allclose([r.incident.T.item() for r in results], expected, atol=.004)


def test_default_subwavelength_window_and_integer_raster():
    kw = options(period=.1, thickness=[.1])
    kw.pop('truncation_order')
    result = _legacy_wavebit_fdtd(structure=np.ones((1, 1, 1), int), backend='numpy', **kw)
    assert result.incident.T.shape == (1, 1)
    np.testing.assert_allclose(result.incident.R, 0, atol=1e-15)
    np.testing.assert_allclose(result.incident.T, 1, atol=.001)
