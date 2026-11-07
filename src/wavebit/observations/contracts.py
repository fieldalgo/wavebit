"""Shared physical observation records, independent of either solver."""
from wavebit._common.backend import array_namespace, eager_under_torch_compile
from dataclasses import dataclass, field
from operator import index as integer_index
from wavebit._common.declarations import Declaration, renamed_attributes, spec
from wavebit._common.pytrees import register_dataclass
from wavebit._common.execution import frozen_mapping
from wavebit.errors import ValidationError


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Validity(Declaration):
    """Admission/numerical checks are distinct from physical convergence evidence."""
    checks: object = field(default_factory=dict)
    convergence: object = field(default_factory=dict)
    issues: tuple = ()

    def __post_init__(self):
        register_dataclass(type(self))
        for name in ('checks', 'convergence'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))
        object.__setattr__(self, 'issues', tuple(self.issues))


#: Renamed members of DiffractionOrder(s) and their replacements: the #57 names
#: (fieldalgo/wavebit#57) and the amplitude ratios of fieldalgo/wavebit#54.
_ORDER_RENAMED = {'cartesian': 'amplitude_ratio_xyz', 'kz': 'k_normal',
                  'amplitude_s': 'amplitude_ratio_s', 'amplitude_p': 'amplitude_ratio_p',
                  'amplitude_xyz': 'amplitude_ratio_xyz'}
#: DiffractionOrders adds a renamed method to those of DiffractionOrder.
_ORDERS_RENAMED = {**_ORDER_RENAMED, 'at_wavelength': 'at_wavelength_index'}
#: A raw spectrum's arrays are pulse spectra in V*s/m, not ratios (#54); 'spectrum' was #57's.
_RAW_ORDER_RENAMED = {'cartesian': 'spectrum_xyz', 'kz': 'k_normal', 'amplitude_s': 'spectrum_s',
                      'amplitude_p': 'spectrum_p', 'amplitude_xyz': 'spectrum_xyz'}
#: The raw spectrum's own field, which moved into its reference (#54), and #57's 'spectrum'.
_RAW_FIELDS_RENAMED = {'spectrum': 'sidedness',
                       'amplitude_units': ('reference', "its unit label is reference['spectrum_units']")}
_RAW_RENAMED = {**_RAW_ORDER_RENAMED, **_RAW_FIELDS_RENAMED, 'at_wavelength': 'at_wavelength_index'}

#: Every per-order array of a calibrated result: ``{name: (trailing shape, masked)}``. ``masked``
#: says a producer's admission guard turns the array to NaN where a value fails admission; the
#: wavevectors are left as computed. ``order``, ``at_wavelength_index`` and the shape check all
#: read this table, and :class:`DiffractionOrder` has a field for each entry.
RATIO_ARRAYS = {'amplitude_ratio_s': ((), True), 'amplitude_ratio_p': ((), True),
                'amplitude_ratio_xyz': ((3,), True), 'power': ((), True),
                'efficiency': ((), True), 'k_parallel': ((2,), False), 'k_normal': ((), False)}
#: The same for a raw pulse spectrum, and :class:`RawDiffractionOrder`. No producer masks a raw
#: spectrum: it stays finite where the calibrated ratios are NaN (outside a pulse's usable band).
SPECTRUM_ARRAYS = {'spectrum_s': ((), False), 'spectrum_p': ((), False),
                   'spectrum_xyz': ((3,), False), 'spectral_energy': ((), False),
                   'k_parallel': ((2,), False), 'k_normal': ((), False)}

#: The one unit each kind of amplitude is labelled with, in ``reference``. A class's own label
#: may be present (with that unit) or absent; the other kind's label is refused.
_UNITS = {'amplitude_ratio_units': '1', 'spectrum_units': 'V*s/m'}
#: A drive is published with its unit label beside it, or not at all: ``field_*`` reads both,
#: so a value without its label, or a label without its value, is refused.
_DRIVES = {'incident_field': ('incident_field_units', 'V/m'),
           'incident_spectrum': ('incident_spectrum_units', 'V*s/m')}


def _field(ratio, reference):
    """``ratio`` times the CW incident field, in V/m; None where there is no CW field.

    ``reference['incident_field']`` is the drive of a CW producer (``rcwa.run``,
    ``fdtd.periodic.run``). A pulse drive (``fdtd.run``) has no single field in V/m, only a
    spectrum (``reference['incident_spectrum']``, V*s/m), so its orders have no ``field_*``.
    """
    if ratio is None or reference.get('incident_field_units') != 'V/m':
        return None
    incident = reference['incident_field']
    if isinstance(incident, (tuple, list)):
        # A per-wavelength drive given as a sequence: stack it in the ratio's own namespace.
        xp = array_namespace(ratio)
        incident = xp.stack([xp.asarray(v) for v in incident])
    ndim = getattr(incident, 'ndim', 0)
    if ndim:
        # A per-wavelength drive meets the leading wavelength axis.
        incident = incident.reshape(tuple(incident.shape) + (1,) * (ratio.ndim - ndim))
    return ratio * incident


#: Reference keys #54 replaced, refused by name: a producer that still writes one would publish
#: an amplitude or a drive under a name that no longer says which it is.
_RENAMED_REFERENCE_KEYS = {
    'amplitude_units': "'amplitude_ratio_units' (or 'spectrum_units' on a raw spectrum)",
    'incident_amplitude': "'incident_field' (a CW drive, V/m) or 'incident_spectrum' (a pulse, V*s/m)",
    'incident_amplitude_units': "'incident_field_units' or 'incident_spectrum_units'"}


def _unit_error(key, unit, given, why):
    return ValidationError(f"reference['{key}'] {why}: one name holds one physical quantity "
                           f"(fieldalgo/wavebit#54)", code='observation.amplitude_units',
                           path=f'reference.{key}', expected=unit, actual=given)


def _check_units(obj):
    """Refuse a reference whose unit labels or drive keys would mix two quantities.

    ``type(obj)._LABEL`` names the class's own amplitude label (``amplitude_ratio_units`` on the
    calibrated classes, ``spectrum_units`` on the raw ones).
    """
    reference = obj.reference
    for key, new in _RENAMED_REFERENCE_KEYS.items():
        if key in reference:
            raise ValidationError(f"reference['{key}'] is now {new} (fieldalgo/wavebit#54)",
                                  code='observation.reference_key', path=f'reference.{key}')
    own = type(obj)._LABEL
    for key, unit in _UNITS.items():
        if key not in reference:
            continue
        if key != own:
            raise _unit_error(key, None, reference[key],
                              f"labels {'pulse spectra' if key == 'spectrum_units' else 'amplitude ratios'}, "
                              f"which a {type(obj).__name__} does not hold")
        if reference[key] != unit:
            raise _unit_error(key, unit, reference[key], f'must be {unit!r}, not {reference[key]!r}')
    for key, (label, unit) in _DRIVES.items():
        has_value = reference.get(key) is not None
        if has_value and reference.get(label) != unit:
            raise _unit_error(label, unit, reference.get(label),
                              f"must be {unit!r} beside reference['{key}'], not {reference.get(label)!r}")
        if label in reference and not has_value:
            raise _unit_error(key, unit, reference.get(key),
                              f"is missing beside its unit label reference['{label}']")


@eager_under_torch_compile
class _FieldsOfOneOrder:
    """``field_*`` for one order or many: the amplitude ratios times the CW incident field."""

    @property
    def field_s(self):
        """The outgoing CW field of the s component in V/m; None for a pulse drive."""
        return _field(self.amplitude_ratio_s, self.reference)

    @property
    def field_p(self):
        """The outgoing CW field of the p component in V/m; None for a pulse drive."""
        return _field(self.amplitude_ratio_p, self.reference)

    @property
    def field_xyz(self):
        """The outgoing CW electric field, Cartesian (trailing xyz axis), in V/m; or None."""
        return _field(self.amplitude_ratio_xyz, self.reference)


@spec(renamed=_ORDER_RENAMED)
class DiffractionOrder(_FieldsOfOneOrder, Declaration):
    """One diffraction order of :class:`DiffractionOrders`, with the same per-order arrays.

    Each array has the wavelength axis only, ``k_parallel`` with a trailing two-component tangent
    axis and ``amplitude_ratio_xyz`` and ``field_xyz`` with a trailing xyz axis too. See
    :class:`DiffractionOrders` for what each one means.
    """
    wavelengths: object
    order_u: int
    order_v: int
    amplitude_ratio_s: object = None
    amplitude_ratio_p: object = None
    power: object = None
    efficiency: object = None
    amplitude_ratio_xyz: object = None
    k_parallel: object = None
    k_normal: object = None
    reference: object = field(default_factory=dict)
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        register_dataclass(type(self))
        for name in ('reference', 'metadata'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))
        _check_units(self)

    _LABEL = 'amplitude_ratio_units'

    __getattr__ = renamed_attributes(_ORDER_RENAMED)


@spec(renamed=_RAW_ORDER_RENAMED)
class RawDiffractionOrder(Declaration):
    """One diffraction order of :class:`RawDiffractionSpectrum`, with the same per-order arrays."""
    wavelengths: object
    order_u: int
    order_v: int
    spectrum_s: object = None
    spectrum_p: object = None
    spectrum_xyz: object = None
    spectral_energy: object = None
    k_parallel: object = None
    k_normal: object = None
    reference: object = field(default_factory=dict)
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        register_dataclass(type(self))
        for name in ('reference', 'metadata'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))
        _check_units(self)

    _LABEL = 'spectrum_units'

    __getattr__ = renamed_attributes(_RAW_ORDER_RENAMED)


@spec
class _OrderArrays(Declaration):
    """What :class:`DiffractionOrders` and :class:`RawDiffractionSpectrum` share: the axes,
    the port's ``reference`` and the run's ``metadata``. Each subclass names its own arrays in
    ``_ARRAYS`` and its one-order type in ``_ORDER``."""
    wavelengths: object
    orders_u: tuple
    orders_v: tuple
    reference: object = field(default_factory=dict)
    metadata: object = field(default_factory=dict)
    reference_wavelength_fields: tuple = ('plane', 'epsilon', 'mu', 'cell_measure', 'incident_power',
                                          'incident_spectrum', 'incident_field')
    metadata_wavelength_fields: tuple = ('theta_radians', 'source_usable', 'power_defined', 'efficiency_defined')

    _ARRAYS = {}
    _ORDER = None

    def __post_init__(self):
        register_dataclass(type(self), static_fields=('orders_u', 'orders_v', 'reference_wavelength_fields', 'metadata_wavelength_fields'))
        for name in ('reference_wavelength_fields', 'metadata_wavelength_fields'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ('orders_u', 'orders_v'):
            values = tuple(integer_index(v) for v in getattr(self, name))
            if len(set(values)) != len(values):
                raise ValidationError('order labels must be unique', code='observation.order_labels', path=name)
            object.__setattr__(self, name, values)
        for name in ('reference', 'metadata'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))
        shape = getattr(self.wavelengths, 'shape', None)
        if shape is None:
            object.__setattr__(self, 'wavelengths', tuple(self.wavelengths))
            nw = len(self.wavelengths)
        else:
            if len(shape) != 1:
                raise ValidationError('wavelengths must have one explicit axis', code='observation.wavelength_axis', path='wavelengths')
            nw = shape[0]
        expected = (nw, len(self.orders_v), len(self.orders_u))
        if nw < 1 or not self.orders_u or not self.orders_v:
            raise ValidationError('spectral and order axes must be nonempty', code='observation.empty', path='outputs')
        for name, (trailing, _) in self._ARRAYS.items():
            value = getattr(self, name)
            if value is None or tuple(getattr(value, 'shape', ())) == expected + trailing:
                continue
            what = {(): f'expected shape {expected}',
                    (2,): 'Parallel wave vectors require a trailing two-dimensional tangent axis',
                    (3,): f'{name} requires a trailing xyz axis'}[trailing]
            raise ValidationError(what, code='observation.shape', path=name,
                                  expected=expected + trailing, actual=getattr(value, 'shape', None))
        _check_units(self)

    def order(self, m, n=0):
        """Order ``(m, n)`` as one record: every array of the table at that order."""
        try:
            i, j = self.orders_u.index(integer_index(m)), self.orders_v.index(integer_index(n))
        except ValueError as exc:
            raise KeyError((m,n)) from exc
        data = {name: None if getattr(self, name) is None else getattr(self, name)[:,j,i] for name in self._ARRAYS}
        return self._ORDER(wavelengths=self.wavelengths, order_u=m, order_v=n, reference=self.reference, metadata=self.metadata, **data)

    def at_wavelength_index(self, index):
        """These orders at the wavelength in position ``index`` of ``wavelengths``, as a length-1 axis."""
        i = integer_index(index)
        if not 0 <= i < len(self.wavelengths): raise IndexError(i)
        data = {name: None if getattr(self, name) is None else getattr(self, name)[i:i+1] for name in self._ARRAYS}
        # Reference coordinates and media can vary spectrally. The explicit
        # declaration prevents a static xyz vector being sliced merely because
        # its length happens to equal the number of wavelengths.
        for mapping_name, fields_name in (('reference', 'reference_wavelength_fields'),
                                          ('metadata', 'metadata_wavelength_fields')):
            selected = dict(getattr(self, mapping_name))
            for name in getattr(self, fields_name):
                value = selected.get(name)
                shape = getattr(value, 'shape', None)
                has_axis = bool(shape) if shape is not None else isinstance(value, (tuple, list))
                if has_axis and len(value) == len(self.wavelengths):
                    selected[name] = value[i:i+1]
            data[mapping_name] = selected
        return self.replace(wavelengths=self.wavelengths[i:i+1], **data)


@spec(renamed=_ORDER_RENAMED)
class DiffractionOrders(_FieldsOfOneOrder, _OrderArrays):
    """Outgoing orders at one port and one Bloch group, always with a wavelength axis.

    **Amplitude ratios, fields and powers** (fieldalgo/wavebit#54). ``amplitude_ratio_s`` and
    ``amplitude_ratio_p`` are each order's outgoing electric field, in its own outgoing s/p frame,
    at ``reference['plane']`` and ``reference['lateral_origin']``, divided by the incident wave's
    complex amplitude along its unit polarization at its own reference: the lit face, or
    ``reference['incident_plane']`` where a producer names one (for a transmitted port that is
    not the port's own plane, where no incident wave runs). Dimensionless
    (``reference['amplitude_ratio_units']`` is ``'1'``) and, like a Fresnel coefficient, a property of the structure that a stronger source leaves
    unchanged. ``amplitude_ratio_xyz`` is the same ratio of the Cartesian field, with a trailing
    xyz axis. ``field_s``, ``field_p`` and ``field_xyz`` are those ratios times the CW incident
    field ``reference['incident_field']``: the outgoing field in V/m, which scales with the source
    amplitude, phase included. They are None where the drive is a pulse (``fdtd.run``), which has
    a spectrum (``reference['incident_spectrum']``, V*s/m) and no single field. ``efficiency`` is
    each order's outgoing power over the incident power and ``power`` its outgoing power, in
    ``reference['power_units']``; ``power`` scales with the square of the source amplitude.
    Compare ratios, efficiencies, powers and fields across solvers; never divide one producer's
    incident field by another's incident spectrum.

    k_parallel has a trailing two-dimensional tangent axis, named by the port
    reference (xy for z-normal ports). ``k_normal`` is each order's wavevector component along
    the port's normal axis, signed along that axis as a Cartesian component: it holds kz for a
    z-normal port and kx for an x-normal one, so a propagating order leaving through a port whose
    outward normal is -z has Re(k_normal) < 0.
    Scalar modal arrays have shape (wavelength, order_v, order_u). Missing
    calibrated power or incident normalization is None, never fabricated zero,
    where its absence is structural. Where a value decides it, per wavelength
    and under a trace too, the entry is NaN and a metadata flag is false:
    ``power_defined`` for ``power`` (a complex transverse wavevector has no
    cell power integral; on ``fdtd.periodic``, whose power is efficiency times incident power,
    also no propagating incident wave) and ``efficiency_defined`` for ``efficiency`` and
    ``reference['incident_power']`` (there is no incident power to divide
    by). ``efficiency_defined`` implies ``power_defined``.
    ``reference`` records the port and its plane, the medium, lateral origin, polarization
    frame, propagation convention and incident normalization, with every unit label beside
    the value it labels. Every producer gives the same core keys: ``port`` (the face, in the
    solver's own face vocabulary: 'top'/'bottom' or 'z+'/'x-'), ``wave`` ('reflected' or
    'transmitted'), ``plane`` (the port plane's coordinate along its normal, in
    ``length_units``), ``lateral_origin`` (coordinates), ``amplitude_ratio_units`` ('1', and
    refused otherwise), ``incident_field`` with ``incident_field_units`` ('V/m') for a CW drive
    or ``incident_spectrum`` with ``incident_spectrum_units`` ('V*s/m') for a pulse, and
    ``cell_measure`` with ``cell_measure_units`` (the periodic cell's measure: 'm^2', or 'm' or
    '1' across invariant axes) and ``invariant_axes``. ``incident_power`` with
    ``incident_power_units``, and ``power_units``, are present where a power is defined.
    ``reference_wavelength_fields`` and ``metadata_wavelength_fields`` identify entries with a leading
    wavelength axis; unlisted entries are static, even if their lengths match.
    """
    amplitude_ratio_s: object = None
    amplitude_ratio_p: object = None
    power: object = None
    efficiency: object = None
    amplitude_ratio_xyz: object = None
    k_parallel: object = None
    k_normal: object = None

    _ARRAYS = RATIO_ARRAYS
    _ORDER = DiffractionOrder
    _LABEL = 'amplitude_ratio_units'

    __getattr__ = renamed_attributes(_ORDERS_RENAMED)


@spec(renamed={**_RAW_ORDER_RENAMED, **_RAW_FIELDS_RENAMED})
class RawDiffractionSpectrum(_OrderArrays):
    """Raw pulse Fourier amplitudes, not calibrated monochromatic fields or power.

    ``spectrum_s``, ``spectrum_p`` and ``spectrum_xyz`` are each order's outgoing field spectrum
    in V*s/m (``reference['spectrum_units']``): the transform of the pulse response, which scales
    with the drive. They are not ratios, and this is not a :class:`DiffractionOrders`. Divided by
    the ``reference['incident_spectrum']`` of a calibrated monitor with the same plane,
    ``reference_plane``, wave and wavelengths, they give its amplitude ratios inside the pulse's
    usable band (``metadata['source_usable']``); outside it the ratios are NaN and these stay
    finite.
    ``spectral_energy`` is each order's energy spectrum. ``sidedness`` says which spectrum
    convention the values follow ('two-sided').
    """
    spectrum_s: object = None
    spectrum_p: object = None
    spectrum_xyz: object = None
    spectral_energy: object = None
    k_parallel: object = None
    k_normal: object = None
    transform: str = 'integral exp(+i 2 pi f t) dt'
    window: object = None
    sidedness: str = 'two-sided'

    _ARRAYS = SPECTRUM_ARRAYS
    _ORDER = RawDiffractionOrder
    _LABEL = 'spectrum_units'

    __getattr__ = renamed_attributes(_RAW_RENAMED)
