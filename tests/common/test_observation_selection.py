"""Spectral selection must preserve calibration and reference-plane alignment."""
import numpy as np
import pytest

from wavebit.observations import DiffractionOrders, RawDiffractionSpectrum
from wavebit.errors import ValidationError


def observation(cls=DiffractionOrders, **extra):
    shape = (3, 1, 2)
    data = np.arange(6.).reshape(shape)
    return cls(wavelengths=np.array([.5, .7, .9]), orders_u=(0, 1), orders_v=(0,),
               amplitude_s=data + 1j, k_normal=data, k_parallel=np.stack((data, 2*data), -1),
               reference={'epsilon': (2., 3., 4.), 'mu': 1., 'plane': (1., 2., 3.),
                          'incident_power': np.array([4., 5., 6.]), 'cell_measure': (7., 8., 9.),
                          'outward_normal': (0., 0., 1.), 'lateral_origin': (0., 0.)},
               metadata={'power_defined': np.array([True, False, True]), 'theta_radians': np.array([.1, .2, .3]), 'axis_order': ('wavelength', 'v', 'u')},
               **extra)


def test_spectral_selection_slices_declared_calibration_but_not_static_vectors():
    original = observation()
    selected = original.at_wavelength_index(1)
    np.testing.assert_array_equal(selected.wavelengths, [.7])
    np.testing.assert_array_equal(selected.k_parallel, original.k_parallel[1:2])
    np.testing.assert_array_equal(selected.k_normal, original.k_normal[1:2])
    assert selected.reference['epsilon'] == (3.,)
    assert selected.reference['plane'] == (2.,)
    assert selected.reference['cell_measure'] == (8.,)
    assert selected.reference['mu'] == 1.
    np.testing.assert_array_equal(selected.reference['incident_power'], [5.])
    assert selected.reference['outward_normal'] == (0., 0., 1.)
    assert selected.metadata['axis_order'] == ('wavelength', 'v', 'u')
    np.testing.assert_array_equal(selected.metadata['theta_radians'], [.2])
    np.testing.assert_array_equal(selected.metadata['power_defined'], [False])
    assert original.reference['epsilon'] == (2., 3., 4.)
    np.testing.assert_array_equal(selected.order(1).k_parallel, original.k_parallel[1:2, 0, 1])


def test_raw_spectral_energy_and_custom_reference_fields_follow_selection():
    original = observation(RawDiffractionSpectrum, spectral_energy=np.arange(6.).reshape(3, 1, 2))
    original = original.replace(reference={'calibration': np.array([3., 4., 5.])},
                                reference_wavelength_fields=('calibration',))
    selected = original.at_wavelength_index(2)
    np.testing.assert_array_equal(selected.reference['calibration'], [5.])
    np.testing.assert_array_equal(selected.spectral_energy, original.spectral_energy[2:3])
    assert selected.transform == original.transform


def test_parallel_wavevectors_require_the_same_spectral_order_axes():
    with pytest.raises(ValidationError, match='Parallel wave vectors'):
        observation().replace(k_parallel=np.zeros((1, 1, 2, 2)))


def test_spectral_selection_survives_jit_without_detaching_reference_arrays():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    def select(power):
        return DiffractionOrders(wavelengths=jnp.array([.5, .7, .9]), orders_u=(0,), orders_v=(0,),
                                 power=power[:, None, None], reference={'incident_power': power,
                                 'outward_normal': (0., 0., 1.)}).at_wavelength_index(1)
    values = jnp.array([1., 2., 3.])
    selected = jax.jit(select)(values)
    assert selected.power.shape == (1, 1, 1)
    np.testing.assert_array_equal(selected.reference['incident_power'], [2.])
    np.testing.assert_array_equal(jax.grad(lambda v: select(v).reference['incident_power'].sum())(values), [0., 1., 0.])
