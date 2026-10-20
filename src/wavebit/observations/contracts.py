"""Shared physical observation records, independent of either solver."""
from wavebit._common.backend import eager_under_torch_compile
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


#: DiffractionOrder(s) fields that were renamed (fieldalgo/wavebit#57), and their replacements.
_ORDER_RENAMED = {'cartesian': 'amplitude_xyz', 'kz': 'k_normal'}
#: DiffractionOrders attributes that were renamed: its fields and a method.
_ORDERS_RENAMED = {**_ORDER_RENAMED, 'at_wavelength': 'at_wavelength_index'}
#: RawDiffractionSpectrum adds its own renamed field to those of DiffractionOrders.
_RAW_RENAMED = {**_ORDER_RENAMED, 'spectrum': 'sidedness'}

#: The per-order arrays that ``order`` and ``at_wavelength_index`` select from.
_ORDER_ARRAYS = ('amplitude_s', 'amplitude_p', 'power', 'efficiency', 'amplitude_xyz', 'k_parallel',
                 'k_normal')


@spec(renamed=_ORDER_RENAMED)
class DiffractionOrder(Declaration):
    """One diffraction order of :class:`DiffractionOrders`, with the same per-order fields.

    ``amplitude_xyz`` is the Cartesian field amplitude (trailing xyz axis); ``k_normal`` is the
    wavevector component along the port's normal axis (see :class:`DiffractionOrders`).
    """
    wavelengths: object
    order_u: int
    order_v: int
    amplitude_s: object = None
    amplitude_p: object = None
    power: object = None
    efficiency: object = None
    amplitude_xyz: object = None
    k_parallel: object = None
    k_normal: object = None
    reference: object = field(default_factory=dict)
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        register_dataclass(type(self))
        for name in ('reference', 'metadata'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))

    __getattr__ = renamed_attributes(_ORDER_RENAMED)


@spec(renamed=_ORDER_RENAMED)
class DiffractionOrders(Declaration):
    """Outgoing orders at one port and one Bloch group, always with a wavelength axis.

    k_parallel has a trailing two-dimensional tangent axis, named by the port
    reference (xy for z-normal ports). ``k_normal`` is each order's wavevector component along
    the port's normal axis, signed along that axis as a Cartesian component: it holds kz for a
    z-normal port and kx for an x-normal one, so a propagating order leaving through a port whose
    outward normal is -z has Re(k_normal) < 0. ``amplitude_xyz`` is the Cartesian field
    amplitude, with a trailing xyz axis.
    Scalar modal arrays have shape (wavelength, order_v, order_u). Missing
    calibrated power or incident normalization is None, never fabricated zero.
    ``reference`` records the port and its plane, the medium, lateral origin, polarization
    frame, propagation convention and incident normalization, with every unit label beside
    the value it labels. Every producer gives the same core keys: ``port`` (the face, in the
    solver's own face vocabulary: 'top'/'bottom' or 'z+'/'x-'), ``wave`` ('reflected' or
    'transmitted'), ``plane`` (the port plane's coordinate along its normal, in
    ``length_units``), ``lateral_origin`` (coordinates), ``incident_amplitude`` with
    ``incident_amplitude_units``, and ``cell_measure`` with ``cell_measure_units`` (the
    periodic cell's measure: 'm^2', or 'm' or '1' across invariant axes) and
    ``invariant_axes``. ``incident_power`` with ``incident_power_units`` is present where an
    incident power is defined.
    ``reference_wavelength_fields`` and ``metadata_wavelength_fields`` identify entries with a leading
    wavelength axis; unlisted entries are static, even if their lengths match.
    """
    wavelengths: object
    orders_u: tuple
    orders_v: tuple
    amplitude_s: object = None
    amplitude_p: object = None
    power: object = None
    efficiency: object = None
    reference: object = field(default_factory=dict)
    metadata: object = field(default_factory=dict)
    amplitude_xyz: object = None
    k_parallel: object = None
    k_normal: object = None
    reference_wavelength_fields: tuple = ('plane', 'epsilon', 'mu', 'cell_measure', 'incident_power')
    metadata_wavelength_fields: tuple = ('theta_radians', 'source_usable', 'power_defined')

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
        for name in ('amplitude_s', 'amplitude_p', 'power', 'efficiency', 'k_normal'):
            value = getattr(self, name)
            if value is not None and tuple(getattr(value, 'shape', ())) != expected:
                raise ValidationError(f'expected shape {expected}', code='observation.shape', path=name, expected=expected, actual=getattr(value, 'shape', None))
        if self.k_parallel is not None and tuple(self.k_parallel.shape) != expected + (2,):
            raise ValidationError('Parallel wave vectors require a trailing two-dimensional tangent axis', code='observation.shape', path='k_parallel', expected=expected+(2,), actual=self.k_parallel.shape)
        if self.amplitude_xyz is not None and tuple(self.amplitude_xyz.shape) != expected + (3,):
            raise ValidationError('amplitude_xyz requires a trailing xyz axis', code='observation.shape', path='amplitude_xyz', expected=expected+(3,), actual=self.amplitude_xyz.shape)

    __getattr__ = renamed_attributes(_ORDERS_RENAMED)

    def order(self, m, n=0):
        try:
            i, j = self.orders_u.index(integer_index(m)), self.orders_v.index(integer_index(n))
        except ValueError as exc:
            raise KeyError((m,n)) from exc
        data = {name: None if getattr(self, name) is None else getattr(self, name)[:,j,i] for name in _ORDER_ARRAYS}
        return DiffractionOrder(wavelengths=self.wavelengths, order_u=m, order_v=n, reference=self.reference, metadata=self.metadata, **data)

    def at_wavelength_index(self, index):
        """These orders at the wavelength in position ``index`` of ``wavelengths``, as a length-1 axis."""
        i = integer_index(index)
        if not 0 <= i < len(self.wavelengths): raise IndexError(i)
        data = {name: None if getattr(self, name) is None else getattr(self, name)[i:i+1] for name in _ORDER_ARRAYS}
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
        if isinstance(self, RawDiffractionSpectrum) and self.spectral_energy is not None:
            data['spectral_energy'] = self.spectral_energy[i:i+1]
        return self.replace(wavelengths=self.wavelengths[i:i+1], **data)


@spec(renamed=_RAW_RENAMED)
class RawDiffractionSpectrum(DiffractionOrders):
    """Raw pulse Fourier amplitudes, not calibrated monochromatic fields or power.

    ``sidedness`` says which spectrum convention the values follow ('two-sided').
    """
    transform: str = 'integral exp(+i 2 pi f t) dt'
    window: object = None
    sidedness: str = 'two-sided'
    amplitude_units: object = 'V*s/m'
    spectral_energy: object = None

    __getattr__ = renamed_attributes({**_ORDERS_RENAMED, **_RAW_RENAMED})
