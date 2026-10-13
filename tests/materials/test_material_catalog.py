"""The material catalog selects sources, preserves units and returns models."""

from wavebit.compat import legacy as _legacy_wavebit
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import wavebit
from wavebit.materials import catalog as materials; from wavebit import TabulatedMaterial


def test_catalog_discovery_is_io_free_and_loading_is_selective(monkeypatch):
    materials._load.cache_clear()
    opened = []
    original = Path.open

    def track(path, *args, **kwargs):
        opened.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', track)
    assert materials.available() == ('filmetrics/Al2O3', 'filmetrics/Si',
                                      'filmetrics/Si3N4', 'filmetrics/SiO2')
    assert opened == []
    m = materials.load('si')
    assert opened == ['Si.txt']
    assert materials.load('FILMETRICS/SI') is m
    assert opened == ['Si.txt']
    with pytest.raises(FrozenInstanceError):
        m.values = ()


@pytest.mark.parametrize('name', materials.available())
def test_bundled_data_preserves_interpolated_index_and_loss_sign(name):
    m = materials.load(name)
    # Independent interpolation of the shipped source data checks column order,
    # source selection and the conversion boundary, not a fitted optical model.
    source = Path(wavebit.__file__).parent / 'data' / (name + '.txt')
    data = np.loadtxt(source, skiprows=1)
    wavelengths = np.array([m.wavelength_range[0], 900., m.wavelength_range[1]])
    expected = (np.interp(wavelengths, data[:, 0], data[:, 1]) +
                1j*np.interp(wavelengths, data[:, 0], data[:, 2]))**2
    np.testing.assert_allclose(m.at(wavelengths).epsilon, expected, rtol=1e-14)
    assert np.all(m.at(wavelengths).epsilon.imag >= 0)
    assert m.mu is None


@pytest.mark.parametrize('unit,scale', [('nm', 1.), ('um', .001), ('mm', 1e-6), ('m', 1e-9)])
def test_converted_wavelengths_preserve_optical_values_and_endpoints(unit, scale):
    m = materials.load('Si', wavelength_unit=unit)
    assert m.wavelength_range == pytest.approx(np.array([190.7, 2500.])*scale)
    assert m.at(900.*scale).epsilon == pytest.approx(materials.load('Si').at(900.).epsilon)
    assert np.isfinite(m.at(np.array(m.wavelength_range)).epsilon).all()
    with pytest.raises(ValueError, match='range'):
        m.at(3000.*scale)


def test_length_unit_change_preserves_film_optics():
    def solve(unit, scale):
        return _legacy_wavebit.rcwa(wavelength=900.*scale, period=600.*scale,
                            thickness=[120.*scale], truncation_order=0,
                            structure=[dict(background=materials.load('Si', wavelength_unit=unit), shapes=[])],
                            polarization=(1., 0.), factorization='li').solve()
    a, b = solve('nm', 1.), solve('um', .001)
    np.testing.assert_allclose([a.R.sum(), a.T.sum()], [b.R.sum(), b.T.sum()], atol=1e-12)


@pytest.mark.parametrize('name', ['', None, 12, '../Si', 'Si__real', 'p_Si', 'unknown'])
def test_unknown_or_implicit_modified_material_is_rejected(name):
    with pytest.raises(ValueError, match='material'):
        materials.load(name)


@pytest.mark.parametrize('unit', ['angstrom', 'NM', None, 1, []])
def test_unknown_unit_is_rejected(unit):
    with pytest.raises(ValueError, match='wavelength_unit'):
        materials.load('Si', wavelength_unit=unit)


def test_ambiguous_short_name_never_selects_by_insertion_order(monkeypatch):
    monkeypatch.setitem(materials._CATALOG, 'another/Si', 'unused.txt')
    with pytest.raises(ValueError, match='ambiguous'):
        materials.load('Si')
    assert isinstance(materials.load('filmetrics/Si'), TabulatedMaterial)


def test_only_explicit_evaluation_returns_a_snapshot():
    assert all(not hasattr(wavebit, name) for name in
               ('find_material', 'find_nk_index', 'read_material_table'))
    with pytest.raises(TypeError):
        materials.load('Si', 900.)
    model = materials.load('Si')
    snapshot = model.at(900.)
    assert isinstance(snapshot, wavebit.MaterialResponse)
    assert not hasattr(snapshot, 'at'), 'an evaluated response cannot stand in for a broadband model'
    original = snapshot.epsilon
    assert model.at(1000.).epsilon != original
    assert snapshot.epsilon == original


def test_catalog_model_round_trip_is_self_contained_in_scene_units(monkeypatch):
    model = materials.load('Si', wavelength_unit='um')
    spec = wavebit.Job(solver='rcwa',
        problem=wavebit.rcwa.Problem(units=wavebit.Units(length='um'),
            stack=wavebit.rcwa.Stack(layers=(wavebit.rcwa.Layer(thickness=.12, background=model),),
                                     period=(.6, .6)),
            excitation=wavebit.rcwa.Excitation(wavelength=.9,
                sources=(wavebit.rcwa.PlaneWave(polarization=(1., 0.)),))),
        numerics=wavebit.rcwa.Numerics())
    encoded = spec.to_json()
    monkeypatch.setattr(materials, 'load', lambda *a, **k: pytest.fail('JSON must embed the table'))
    restored = wavebit.Job.from_json(encoded)
    assert restored.key == spec.key
    restored_model = restored.problem.stack.layers[0].background
    assert restored_model == model
    assert restored_model.wavelength_unit == 'um'
    assert restored_model.at(.9, units=restored.problem.units).epsilon == model.at(.9).epsilon



@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_vector_evaluation_preserves_shape_backend_and_wavelength_derivatives(backend):
    m = TabulatedMaterial([800., 900., 1000.], [4., 6., 9.], mu=[1., 2., 3.], wavelength_unit='nm')
    values = [[800., 850.], [900., 1000.]]
    if backend == 'numpy':
        w = np.array(values)
        actual = m.at(w)
    elif backend == 'jax':
        jax = pytest.importorskip('jax')
        _legacy_wavebit.enable_x64()
        w = jax.numpy.array(values)
        actual = jax.jit(m.at)(w)
        gradient = jax.grad(lambda x: m.at(x).epsilon.real.sum())(w)
        np.testing.assert_allclose(gradient, [[.02, .02], [.03, .03]])
        bad = jax.jit(lambda x: m.at(x).epsilon)(jax.numpy.array([850., 1100.]))
        assert np.isfinite(bad[0]) and np.isnan(bad[1])
    else:
        torch = pytest.importorskip('torch')
        w = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        actual = m.at(w)
        actual.epsilon.real.sum().backward()
        np.testing.assert_allclose(w.grad, [[.02, .02], [.03, .03]])
        actual = wavebit.Material(actual.epsilon.detach().numpy(), actual.mu.detach().numpy())
    np.testing.assert_allclose(actual.epsilon, [[4., 5.], [6., 9.]])
    np.testing.assert_allclose(actual.mu, [[1., 1.5], [2., 3.]])
    with pytest.raises(ValueError, match='range'):
        m.at(np.array([850., 1100.]))
