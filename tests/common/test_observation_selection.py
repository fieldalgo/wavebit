"""Spectral selection must preserve calibration and reference-plane alignment."""
import dataclasses

import numpy as np
import pytest

from wavebit.observations import (DiffractionOrder, DiffractionOrders, RawDiffractionOrder,
                                  RawDiffractionSpectrum)
from wavebit.errors import ValidationError


def observation(cls=DiffractionOrders, reference=None, **extra):
    shape = (3, 1, 2)
    data = np.arange(6.).reshape(shape)
    # A calibrated record holds amplitude ratios; a raw one holds pulse spectra (#54).
    first = 'spectrum_s' if cls is RawDiffractionSpectrum else 'amplitude_ratio_s'
    return cls(wavelengths=np.array([.5, .7, .9]), orders_u=(0, 1), orders_v=(0,),
               **{first: data + 1j}, k_normal=data, k_parallel=np.stack((data, 2*data), -1),
               reference={'epsilon': (2., 3., 4.), 'mu': 1., 'plane': (1., 2., 3.),
                          'incident_power': np.array([4., 5., 6.]), 'cell_measure': (7., 8., 9.),
                          'outward_normal': (0., 0., 1.), 'lateral_origin': (0., 0.),
                          **(reference or {})},
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


# --- One name, one physical quantity (fieldalgo/wavebit#54) -----------------------------------

#: Each many-order record with its one-order record, the one amplitude label they hold, and
#: that label's unit. A one-order record runs the reference check of the record it comes from.
OWN_LABEL = pytest.mark.parametrize('cls, one, key, unit', [
    (DiffractionOrders, DiffractionOrder, 'amplitude_ratio_units', '1'),
    (RawDiffractionSpectrum, RawDiffractionOrder, 'spectrum_units', 'V*s/m')],
    ids=['ratios', 'raw'])


def builds(cls, one):
    """Build the many-order record, and one order of its kind, from a reference alone."""
    return (lambda reference: observation(cls, reference=reference),
            lambda reference: one(wavelengths=(.5,), order_u=0, order_v=0, reference=reference))


@OWN_LABEL
@pytest.mark.parametrize('given', ['1', 'V*s/m', 'V/m', 1, ''])
def test_an_amplitude_labelled_with_any_unit_but_its_own_is_refused(cls, one, key, unit, given):
    for build in builds(cls, one):
        if given == unit:
            assert build({key: given}).reference[key] == unit
            continue
        with pytest.raises(ValidationError) as caught:
            build({key: given})
        assert caught.value.code == 'observation.amplitude_units'
        assert caught.value.path == f'reference.{key}'
        # Refused for its unit: the label itself is the record's own.
        assert caught.value.expected == unit and caught.value.actual == given


@pytest.mark.parametrize('cls, one, other, unit', [
    (DiffractionOrders, DiffractionOrder, 'spectrum_units', 'V*s/m'),
    (RawDiffractionSpectrum, RawDiffractionOrder, 'amplitude_ratio_units', '1')],
    ids=['ratios-labelled-as-a-spectrum', 'spectrum-labelled-as-ratios'])
def test_a_record_carrying_the_other_kinds_amplitude_label_is_refused(cls, one, other, unit):
    """Calibrated ratios labelled as a pulse spectrum, and a raw spectrum labelled as ratios,
    are refused even with that label's own unit: the label names a quantity the record does
    not hold."""
    for build in builds(cls, one):
        with pytest.raises(ValidationError) as caught:
            build({other: unit})
        assert caught.value.code == 'observation.amplitude_units'
        assert caught.value.path == f'reference.{other}'
        # Refused for the label, whatever its unit: there is no unit it would be admitted with.
        assert caught.value.expected is None and caught.value.actual == unit


@pytest.mark.parametrize('key', ['incident_amplitude', 'incident_amplitude_units', 'amplitude_units'])
@pytest.mark.parametrize('cls, one', [(DiffractionOrders, DiffractionOrder),
                                      (RawDiffractionSpectrum, RawDiffractionOrder)],
                         ids=lambda c: c.__name__)
def test_the_reference_keys_54_replaced_are_refused_by_name(key, cls, one):
    value = {'incident_amplitude': 2j, 'incident_amplitude_units': 'V/m', 'amplitude_units': '1'}[key]
    with pytest.raises(ValidationError) as caught:
        observation(cls, reference={key: value})
    assert caught.value.code == 'observation.reference_key'
    assert caught.value.path == f'reference.{key}'
    replacement = {'incident_amplitude': 'incident_field', 'incident_amplitude_units': 'incident_field_units',
                   'amplitude_units': 'amplitude_ratio_units'}[key]
    assert replacement in str(caught.value)
    with pytest.raises(ValidationError) as caught:
        one(wavelengths=(.5,), order_u=0, order_v=0, reference={key: value})
    assert caught.value.code == 'observation.reference_key'
    assert caught.value.path == f'reference.{key}'


@pytest.mark.parametrize('label', [None, 'V*s/m', 'V'])
def test_a_cw_drive_without_its_v_per_m_label_is_refused(label):
    reference = {'incident_field': 2j}
    if label is not None:
        reference['incident_field_units'] = label
    with pytest.raises(ValidationError) as caught:
        observation(reference=reference)
    assert caught.value.code == 'observation.amplitude_units'
    assert caught.value.path == 'reference.incident_field_units'
    with pytest.raises(ValidationError) as caught:
        observation(reference={'incident_spectrum': np.ones(3), 'incident_spectrum_units': 'V/m'})
    assert caught.value.code == 'observation.amplitude_units'
    assert caught.value.path == 'reference.incident_spectrum_units'


@pytest.mark.parametrize('key, label, unit', [('incident_field', 'incident_field_units', 'V/m'),
                                              ('incident_spectrum', 'incident_spectrum_units', 'V*s/m')],
                         ids=['field', 'spectrum'])
@pytest.mark.parametrize('cls, one', [(DiffractionOrders, DiffractionOrder),
                                      (RawDiffractionSpectrum, RawDiffractionOrder)],
                         ids=['ratios', 'raw'])
@pytest.mark.parametrize('value', ['absent', None], ids=['absent', 'none'])
def test_a_drive_label_without_its_value_is_refused(key, label, unit, cls, one, value):
    """A drive is published with its label or not at all: a label with no value beside it, or
    with None, is refused at construction (a CW label would otherwise send every ``field_*``
    read to a value that is not there)."""
    reference = {label: unit} if value == 'absent' else {label: unit, key: None}
    for build in builds(cls, one):
        with pytest.raises(ValidationError) as caught:
            build(reference)
        assert caught.value.code == 'observation.amplitude_units'
        assert caught.value.path == f'reference.{key}'


def test_fields_are_the_ratios_times_the_cw_drive_and_a_pulse_has_none():
    drive = 2j
    orders = observation(reference={'incident_field': drive, 'incident_field_units': 'V/m',
                                    'amplitude_ratio_units': '1'},
                         amplitude_ratio_p=np.full((3, 1, 2), .5 - .25j),
                         amplitude_ratio_xyz=np.ones((3, 1, 2, 3)) * (1 + 2j))
    np.testing.assert_array_equal(orders.field_s, drive*orders.amplitude_ratio_s)
    np.testing.assert_array_equal(orders.field_p, drive*orders.amplitude_ratio_p)
    np.testing.assert_array_equal(orders.field_xyz, drive*orders.amplitude_ratio_xyz)
    # The ratios themselves are not the drive's: a stronger drive changes fields only.
    brighter = orders.replace(reference={**orders.reference, 'incident_field': 2*drive})
    np.testing.assert_array_equal(brighter.amplitude_ratio_s, orders.amplitude_ratio_s)
    np.testing.assert_array_equal(brighter.field_s, 2*orders.field_s)
    # A missing ratio has no field.
    assert orders.replace(amplitude_ratio_p=None).field_p is None
    # A pulse drive has a spectrum in V*s/m and no single field in V/m.
    pulsed = observation(reference={'incident_spectrum': np.array([1., 2., 3.]) + 0j,
                                    'incident_spectrum_units': 'V*s/m'})
    assert pulsed.field_s is None and pulsed.field_p is None and pulsed.field_xyz is None
    assert pulsed.order(0, 0).field_s is None
    # No drive at all: no field either.
    assert observation().field_s is None


def test_one_order_field_is_the_drive_times_that_order_ratio():
    drive = .3 - 1.5j
    orders = observation(reference={'incident_field': drive, 'incident_field_units': 'V/m'})
    for m in orders.orders_u:
        i, j = orders.orders_u.index(m), orders.orders_v.index(0)
        one = orders.order(m, 0)
        assert isinstance(one, DiffractionOrder)
        np.testing.assert_array_equal(one.amplitude_ratio_s, orders.amplitude_ratio_s[:, j, i])
        np.testing.assert_array_equal(one.field_s, drive*orders.amplitude_ratio_s[:, j, i])
        np.testing.assert_array_equal(one.field_s, orders.field_s[:, j, i])
        assert one.field_p is None


@pytest.mark.parametrize('container', [np.array, tuple, list], ids=['array', 'tuple', 'list'])
def test_a_per_wavelength_drive_meets_the_wavelength_axis_not_the_order_axis(container):
    # Three wavelengths and three orders along u: a trailing-axis broadcast would also run,
    # silently pairing wavelength k with order k. A tuple or list drive is per-wavelength too,
    # as at_wavelength_index reads it, and is not left to broadcast as a trailing axis.
    wavelengths = np.array([.5, .7, .9])
    ratio = (np.arange(9.) + 1j).reshape(3, 1, 3)
    values = np.array([1., 2j, -3.])
    drive = container((1., 2j, -3.))
    orders = DiffractionOrders(wavelengths=wavelengths, orders_u=(-1, 0, 1), orders_v=(0,),
                               amplitude_ratio_s=ratio,
                               amplitude_ratio_xyz=np.stack((ratio, 2*ratio, 3*ratio), -1),
                               reference={'incident_field': drive, 'incident_field_units': 'V/m'})
    assert isinstance(orders.field_s, np.ndarray)
    np.testing.assert_array_equal(orders.field_s, ratio*values[:, None, None])
    np.testing.assert_array_equal(orders.field_xyz, orders.amplitude_ratio_xyz*values[:, None, None, None])
    assert not np.array_equal(orders.field_s, ratio*values)
    for m in orders.orders_u:
        np.testing.assert_array_equal(orders.order(m).field_s, orders.field_s[:, 0, m + 1])
    # Selecting a wavelength slices the drive with the ratios, so field_* keeps the selected
    # shape: an unsliced drive would broadcast (1, 1, 3) against (3, 1, 1) without an error.
    selected = orders.at_wavelength_index(1)
    assert isinstance(selected.reference['incident_field'], type(drive))
    np.testing.assert_array_equal(selected.reference['incident_field'], values[1:2])
    assert selected.field_s.shape == (1, 1, 3)
    np.testing.assert_array_equal(selected.field_s, ratio[1:2]*values[1])
    np.testing.assert_array_equal(selected.field_xyz, orders.field_xyz[1:2])
    np.testing.assert_array_equal(selected.order(0).field_s, orders.field_s[1:2, 0, 1])


def test_a_raw_spectrum_is_a_sibling_of_the_calibrated_orders():
    raw = observation(RawDiffractionSpectrum, spectral_energy=np.arange(6.).reshape(3, 1, 2),
                      reference={'spectrum_units': 'V*s/m'})
    assert not isinstance(raw, DiffractionOrders)
    assert not isinstance(observation(), RawDiffractionSpectrum)
    one = raw.order(0, 0)
    assert isinstance(one, RawDiffractionOrder) and not isinstance(one, DiffractionOrder)
    np.testing.assert_array_equal(one.spectrum_s, raw.spectrum_s[:, 0, 0])
    np.testing.assert_array_equal(one.spectral_energy, raw.spectral_energy[:, 0, 0])
    assert one.reference['spectrum_units'] == 'V*s/m'
    # A raw spectrum holds no ratio and no field: they are not there to be read. The class is
    # checked as well as the instance: an inherited field_* property would raise AttributeError
    # on reading the missing ratio, so on the instance alone hasattr is False either way.
    for record in (raw, one):
        for name in ('amplitude_ratio_s', 'field_s', 'field_p', 'field_xyz'):
            assert not hasattr(record, name)
            assert not hasattr(type(record), name), (type(record).__name__, name)


def test_a_raw_spectrum_unit_label_moved_into_its_reference():
    """``RawDiffractionSpectrum.amplitude_units`` is gone (fieldalgo/wavebit#54): the label is
    ``reference['spectrum_units']``. The old attribute and keyword are refused and say where."""
    assert 'amplitude_units' not in {f.name for f in dataclasses.fields(RawDiffractionSpectrum)}
    raw = observation(RawDiffractionSpectrum, reference={'spectrum_units': 'V*s/m'})
    assert not hasattr(raw, 'amplitude_units')
    where = r"its unit label is reference\['spectrum_units'\]"
    with pytest.raises(AttributeError, match=r'RawDiffractionSpectrum\.amplitude_units is now '
                                             r'RawDiffractionSpectrum\.reference; ' + where):
        raw.amplitude_units
    with pytest.raises(TypeError, match=r'RawDiffractionSpectrum\(amplitude_units=\.\.\.\) is now '
                                        r'RawDiffractionSpectrum\(reference=\.\.\.\); ' + where):
        observation(RawDiffractionSpectrum, amplitude_units='V*s/m')


@pytest.mark.parametrize('orders, one', [(DiffractionOrders, DiffractionOrder),
                                         (RawDiffractionSpectrum, RawDiffractionOrder)])
def test_one_order_has_a_field_for_every_array_of_the_table(orders, one):
    assert orders._ORDER is one
    assert orders._ARRAYS
    names = {f.name for f in dataclasses.fields(one)}
    assert set(orders._ARRAYS) <= names
    # The table also names every per-order array of the many-order record.
    assert set(orders._ARRAYS) <= {f.name for f in dataclasses.fields(orders)}
    # field_* is derived, never stored: it is no field, so no pytree leaf and no saved key.
    assert not any(name.startswith('field_') for name in names)


def test_spectral_selection_slices_the_incident_spectrum_and_power():
    spectrum = np.array([1., 2., 3.]) * (1 + 1j)
    raw = observation(RawDiffractionSpectrum,
                      reference={'incident_spectrum': spectrum, 'incident_spectrum_units': 'V*s/m'})
    calibrated = observation(reference={'incident_spectrum': spectrum,
                                        'incident_spectrum_units': 'V*s/m'})
    for record in (raw, calibrated):
        selected = record.at_wavelength_index(2)
        np.testing.assert_array_equal(selected.reference['incident_spectrum'], spectrum[2:3])
        np.testing.assert_array_equal(selected.reference['incident_power'], [6.])
        assert selected.reference['incident_spectrum_units'] == 'V*s/m'
