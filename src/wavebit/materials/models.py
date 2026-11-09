"""Physical broadband constitutive models in SI units.

Rates belong to the material, independently of a problem's geometry units. The
exp(-i omega t) convention makes passive electric loss a positive imaginary epsilon.
No time discretization or solver coefficients are constructed in this module.
"""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass
from decimal import Decimal
import math
import numbers
import sys
import numpy as np
from wavebit._common.backend import array_namespace, as_array, in_namespace, _concrete
from wavebit._common.declarations import Declaration
from wavebit.units import C0, EPSILON0, require_units
from wavebit.errors import ValidationError
from .response import Material, MaterialResponse
from ._tabulated import TabulatedMaterial as LegacyTabulatedMaterial


def _leaves(v):
    if isinstance(v, (tuple, list)):
        return [x for item in v for x in _leaves(item)]
    return [v]


def _namespace(*values):
    arrays = [v for v in _leaves(values) if hasattr(v, 'dtype') and hasattr(v, 'shape')]
    live = [v for v in arrays if not isinstance(v, (np.ndarray, np.generic))]
    selected = live or arrays or [np.asarray(0.)]
    return array_namespace(*selected), selected[0]


def _scalar(value, name, *, positive=False, nonnegative=False):
    def refuse(rule=None):
        field = name.rsplit('.', 1)[-1]
        detail = f'a finite {rule} number' if rule else f'one real number, not {type(value).__name__}'
        return ValidationError(f'expected a real scalar: {field} must be {detail}',
                               code='material.scalar', path=name)
    if (value is None or getattr(value, 'shape', ()) != ()
            or isinstance(value, (list, tuple, dict, str, bytes, bool))):
        raise refuse()
    xp, ref = _namespace(value)
    try:
        a = as_array(xp, value, device=getattr(ref, 'device', None))
    except (TypeError, ValueError) as exc:
        raise refuse() from exc
    # NumPy reads any object as a zero-dimensional object array; only a number is a rate.
    if isinstance(a, (np.ndarray, np.generic)) and a.dtype.kind not in 'iufc':
        raise refuse()
    if xp.isdtype(a.dtype, 'complex floating'):
        raise refuse()
    valid = xp.isfinite(a)
    if positive: valid = valid & (a > 0)
    if nonnegative: valid = valid & (a >= 0)
    if _concrete(valid) is False:
        raise refuse('positive' if positive else 'nonnegative' if nonnegative else 'real')
    return valid


def _constitutive_shape(value):
    """The shape of a permittivity or permeability spelling, or ``None`` if it holds a non-number.

    A nested list or tuple is read as the tensor it spells; a ragged one has no shape. Only
    structure is read, never a value, so a traced leaf passes through.
    """
    if isinstance(value, (list, tuple)):
        inner = [_constitutive_shape(v) for v in value]
        if not inner or any(s is None for s in inner) or len(set(inner)) > 1:
            return None
        return (len(value),) + inner[0]
    if value is None or isinstance(value, (str, bytes, dict)):
        return None
    if isinstance(value, numbers.Number):
        return ()
    shape, dtype = getattr(value, 'shape', None), getattr(value, 'dtype', None)
    if shape is None or dtype is None:
        return None
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind not in 'biufc':
        return None
    return tuple(shape)


def _constitutive(value, name, *, optional=False):
    """Refuse a permittivity or permeability that is not a number, a diagonal or a 3x3 tensor."""
    if optional and value is None:
        return
    if _constitutive_shape(value) not in ((), (3,), (3, 3)):
        raise ValidationError(
            f'{name.rsplit(".", 1)[-1]} must be a number, three diagonal values [xx, yy, zz] or a '
            f'3x3 tensor{" (or None when non-magnetic)" if optional else ""}; got '
            f'{type(value).__name__}.', code='material.shape', path=name)


def _guard_response(value, *checks):
    """A traced invalid pole is visible as NaN, as for invalid wavelengths."""
    xp, ref = _namespace(value, checks)
    device = getattr(ref, 'device', None)
    valid = as_array(xp, True, dtype=xp.bool, device=device)
    for check in checks:
        valid = valid & as_array(xp, check, dtype=xp.bool, device=device)
    return xp.where(valid, value, as_array(xp, float('nan'), dtype=getattr(value, 'dtype', xp.complex128), device=device))


def _register(cls, dynamic, static=()):
    from wavebit._common.pytrees import _STATIC_FIELDS
    _STATIC_FIELDS[cls] = tuple(static)
    jax = sys.modules.get('jax')
    tree = getattr(jax, 'tree_util', None)
    if tree is None: return
    if cls in _REGISTERED: return
    def flatten(obj):
        return tuple(getattr(obj, n) for n in dynamic), tuple(getattr(obj, n) for n in static)
    def unflatten(aux, children):
        # JAX may unflatten abstract sentinels while computing tree structure.
        obj = object.__new__(cls)
        for name, value in zip(dynamic, children): object.__setattr__(obj, name, value)
        for name, value in zip(static, aux): object.__setattr__(obj, name, value)
        return obj
    tree.register_pytree_node(cls, flatten, unflatten)
    _REGISTERED.add(cls)


_REGISTERED = set()


def _array(xp, value, *, dtype, device):
    if isinstance(value, (tuple, list)):
        return xp.stack([_array(xp, v, dtype=dtype, device=device) for v in value])
    return as_array(xp, value, dtype=dtype, device=device)


def _wide(value):
    """Whether any leaf of ``value`` is a double-precision real or complex array."""
    if isinstance(value, (tuple, list)):
        return any(_wide(v) for v in value)
    return str(getattr(value, 'dtype', '')) in (
        'float64', 'complex128', 'torch.float64', 'torch.complex128')


def _double_available(xp):
    """JAX without x64 has no float64; asking for it would warn and narrow anyway."""
    if 'jax' not in xp.__name__:
        return True
    jax = sys.modules.get('jax')
    return bool(jax is not None and jax.config.jax_enable_x64)


def _response(epsilon, mu, term, omega):
    """Add an isotropic response to scalar/diagonal/full constitutive data."""
    xp, ref = _namespace(epsilon, mu, term, omega)
    device = getattr(ref, 'device', None)
    dtype = xp.complex64 if str(getattr(omega, 'dtype', '')) in ('float32', 'torch.float32') else xp.complex128
    # A Python wavelength may select Torch's float32 default. It must not narrow
    # a double material (or its backward cotangent) through a complex64 response.
    if _wide(epsilon) or _wide(mu) or _wide(term):
        dtype = xp.complex128
    e = _array(xp, epsilon, dtype=dtype, device=device)
    spectral_shape = tuple(omega.shape)
    def broadcast(value):
        a = _array(xp, value, dtype=dtype, device=device)
        if a.shape not in ((), (3,), (3, 3)):
            raise ValidationError('material response must be scalar, diagonal, or a 3x3 tensor', code='material.shape', path='material')
        return xp.broadcast_to(a, spectral_shape + tuple(a.shape))
    e = broadcast(e)
    original_shape = tuple(getattr(epsilon, 'shape', ()))
    if isinstance(epsilon, (tuple, list)):
        from .tensors import tensor_shape
        original_shape = tensor_shape(epsilon)
    if original_shape == (3,): e = e + term[..., None]
    elif original_shape == (3, 3): e = e + term[..., None, None] * xp.eye(3, dtype=dtype, device=device)
    else: e = e + term
    return MaterialResponse(e, None if mu is None else broadcast(mu), _wavelength_shape=spectral_shape or None)


def constant_response(epsilon, mu, wavelength, *, units):
    omega = angular_frequency(wavelength, units=units, values=(epsilon, mu))
    return _response(epsilon, mu, omega * 0, omega)


def angular_frequency(wavelength, *, units, values=()):
    units = require_units(units)
    xp, ref = _namespace(wavelength, values)
    w = as_array(xp, wavelength, device=getattr(ref, 'device', None))
    if xp.isdtype(w.dtype, 'complex floating'):
        raise ValidationError('vacuum wavelength must be real', code='wavelength.value', path='wavelength')
    # Form omega at the widest precision among the inputs. A Python wavelength
    # takes Torch's float32 default; a double material must not see its
    # frequency rounded to single precision. Convert the original value, not
    # the float32 copy, so a Python float keeps all of its digits.
    # Only the backend's own arrays count: a host NumPy constant beside a Torch
    # wavelength tensor keeps the precision that wavelength asked for.
    if (w.dtype != xp.float64 and _double_available(xp)
            and _wide([v for v in _leaves(values) if in_namespace(xp, v)])):
        w = as_array(xp, wavelength, dtype=xp.float64, device=getattr(ref, 'device', None))
    valid = xp.isfinite(w) & (w > 0)
    if _concrete(xp.all(valid)) is False:
        raise ValidationError('vacuum wavelength must be finite and positive', code='wavelength.value', path='wavelength')
    safe = xp.where(valid, w, xp.ones_like(w))
    omega = 2 * math.pi * C0 / (safe * units.length_unit_meters)
    return xp.where(valid, omega, as_array(xp, float('nan'), device=getattr(ref, 'device', None)))


@eager_under_torch_compile
@dataclass(frozen=True)
class Drude(Declaration):
    omega_p: object
    gamma: object

    def __post_init__(self):
        _scalar(self.omega_p, 'material.poles.omega_p', nonnegative=True)
        _scalar(self.gamma, 'material.poles.gamma', nonnegative=True)
        _register(type(self), ('omega_p', 'gamma'))

    def epsilon_term(self, omega):
        from ._time_domain import _drude_response
        # The normalized evaluator is float64. Keep the public single-precision
        # arithmetic and dtype; the large finite damping repair is a float64 path.
        if any(str(getattr(v, 'dtype', '')) in ('float32', 'torch.float32')
               for v in (omega, self.omega_p, self.gamma)):
            value = -(self.omega_p ** 2) / (omega ** 2 + 1j * self.gamma * omega)
        else:
            value = _drude_response(omega, self.omega_p, self.gamma)
        return _guard_response(value, _scalar(self.omega_p, 'material.poles.omega_p', nonnegative=True),
                               _scalar(self.gamma, 'material.poles.gamma', nonnegative=True))


@eager_under_torch_compile
@dataclass(frozen=True)
class Lorentz(Declaration):
    omega_0: object
    gamma: object
    strength: object

    def __post_init__(self):
        _scalar(self.omega_0, 'material.poles.omega_0', nonnegative=True)
        _scalar(self.gamma, 'material.poles.gamma', nonnegative=True)
        _scalar(self.strength, 'material.poles.strength')
        _register(type(self), ('omega_0', 'gamma', 'strength'))

    def epsilon_term(self, omega):
        """SI susceptibility; a concrete zero strength contributes exactly zero.

        A differentiated strength at zero retains its susceptibility derivative.
        Active undamped resonances remain singular; no broadening is introduced.
        """
        from wavebit._common.declarations import is_jax_typed, torch_carried
        from ._time_domain import concrete
        if (not torch_carried(self.strength) and concrete(self.strength) == 0
                and (is_jax_typed(self.strength) or self.strength == 0)):
            # Avoid forming a singular denominator or overflowing finite rates.
            # Keep zero tangents in frequency, resonance, and damping attached.
            value = 0j * omega + 0j * self.omega_0 + 0j * self.gamma + 0j * self.strength
        else:
            value = self.strength / (self.omega_0 ** 2 - omega ** 2 - 1j * self.gamma * omega)
        return _guard_response(value, _scalar(self.omega_0, 'material.poles.omega_0', nonnegative=True),
                               _scalar(self.gamma, 'material.poles.gamma', nonnegative=True),
                               _scalar(self.strength, 'material.poles.strength'))


@eager_under_torch_compile
@dataclass(frozen=True)
class Debye(Declaration):
    tau: object
    delta_epsilon: object

    def __post_init__(self):
        _scalar(self.tau, 'material.poles.tau', positive=True)
        _scalar(self.delta_epsilon, 'material.poles.delta_epsilon')
        _register(type(self), ('tau', 'delta_epsilon'))

    def epsilon_term(self, omega):
        value = self.delta_epsilon / (1 - 1j * omega * self.tau)
        return _guard_response(value, _scalar(self.tau, 'material.poles.tau', positive=True),
                               _scalar(self.delta_epsilon, 'material.poles.delta_epsilon'))


@eager_under_torch_compile
@dataclass(frozen=True)
class ConductiveMaterial(Declaration):
    epsilon: object
    conductivity: object
    mu: object = None

    def __post_init__(self):
        _constitutive(self.epsilon, 'material.epsilon')
        _scalar(self.conductivity, 'material.conductivity')
        _constitutive(self.mu, 'material.mu', optional=True)
        _register(type(self), ('epsilon', 'conductivity', 'mu'))

    def at(self, wavelength, *, units):
        omega = angular_frequency(wavelength, units=units, values=(self.epsilon, self.conductivity, self.mu))
        return _response(self.epsilon, self.mu, 1j * self.conductivity / (EPSILON0 * omega), omega)


@eager_under_torch_compile
@dataclass(frozen=True)
class DispersiveMaterial(Declaration):
    epsilon_inf: object
    poles: tuple
    mu: object = None

    def __post_init__(self):
        _constitutive(self.epsilon_inf, 'material.epsilon_inf')
        _constitutive(self.mu, 'material.mu', optional=True)
        message = 'poles must be a nonempty sequence of Drude, Lorentz, or Debye models'
        if isinstance(self.poles, (Drude, Lorentz, Debye)):
            raise ValidationError(message + f'; wrap a single pole as poles=({type(self.poles).__name__}(...),).',
                                  code='material.poles', path='material.poles')
        try:
            object.__setattr__(self, 'poles', tuple(self.poles))
        except TypeError as exc:
            raise ValidationError(message + f'; got {type(self.poles).__name__}.',
                                  code='material.poles', path='material.poles') from exc
        if not self.poles or any(not isinstance(p, (Drude, Lorentz, Debye)) for p in self.poles):
            raise ValidationError(message, code='material.poles', path='material.poles')
        _register(type(self), ('epsilon_inf', 'poles', 'mu'))

    def at(self, wavelength, *, units):
        leaves = tuple(v for p in self.poles for v in vars(p).values())
        omega = angular_frequency(wavelength, units=units, values=(self.epsilon_inf, self.mu, leaves))
        term = omega * 0
        for pole in self.poles: term = term + pole.epsilon_term(omega)
        return _response(self.epsilon_inf, self.mu, term, omega)


@eager_under_torch_compile
class TabulatedMaterial(LegacyTabulatedMaterial):
    """A static spectrum with its own wavelength unit and interpolation convention."""
    # Its wavelengths and every query pass through a length-unit conversion; see _evaluate.
    _ENDPOINT_ULPS = 4
    def __init__(self, wavelengths, epsilon, *, wavelength_unit, mu=None):
        from wavebit.units import Units
        unit = Units(length=wavelength_unit)
        super().__init__(wavelengths, epsilon, mu)
        object.__setattr__(self, 'wavelength_unit', unit.length)
        self._register_pytree()

    @classmethod
    def from_refractive_index(cls, wavelengths, index, *, wavelength_unit, mu=None):
        from wavebit.units import Units
        unit = Units(length=wavelength_unit)
        obj = object.__new__(cls)
        obj._initialize(wavelengths, index, mu, 'index')
        object.__setattr__(obj, 'wavelength_unit', unit.length)
        obj._register_pytree()
        return obj

    def _register_pytree(self):
        if hasattr(self, 'wavelength_unit'):
            _register(type(self), (), ('wavelengths', 'values', 'mu', '_quantity', 'wavelength_unit'))

    def __hash__(self):
        return hash((self.wavelengths, self.values, self.mu, self._quantity, self.wavelength_unit))

    def __eq__(self, other):
        return type(self) is type(other) and (self.wavelengths, self.values, self.mu, self._quantity, self.wavelength_unit) == (other.wavelengths, other.values, other.mu, other._quantity, other.wavelength_unit)

    def at(self, wavelength, *, units=None):
        from wavebit.units import Units
        units = Units(length=self.wavelength_unit) if units is None else require_units(units)
        # Form the exact decimal SI-prefix ratio before conversion to binary.
        # 1e-6/1e-9 rounds below 1000 and wrongly excludes the 500 nm endpoint.
        scale = float(Decimal(str(units.length_unit_meters)) /
                      Decimal(str(Units(length=self.wavelength_unit).length_unit_meters)))
        try:
            xp, ref = _namespace(wavelength)
            w = as_array(xp, wavelength, device=getattr(ref, 'device', None))
            result = super().at(w * scale)
        except ValueError as exc:
            raise ValidationError(str(exc), code='material.wavelength_range', path='wavelength', expected=self.wavelength_range, actual=wavelength) from exc
        return MaterialResponse(result.epsilon, result.mu, _wavelength_shape=result._wavelength_shape)

    @property
    def interpolated_quantity(self):
        """What the table interpolates between its wavelengths: ``'epsilon'`` or ``'index'``."""
        return self._quantity

    def __repr__(self):
        return (f'TabulatedMaterial(samples={len(self.wavelengths)}, '
                f'wavelength_range={self.wavelength_range}, '
                f'interpolated_quantity={self._quantity!r}, magnetic={self.is_magnetic})')

    def __getattr__(self, name):
        # Reached only when normal lookup fails, so it never shadows an attribute.
        if name == 'interpolation':
            from wavebit._common.declarations import removed
            raise removed('TabulatedMaterial', name, 'TabulatedMaterial.interpolated_quantity')
        raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')

    def rotated(self, angle, axis='z'):
        if self._quantity == 'index':
            Material(1.).rotated(angle, axis)
            return self
        rotated = [Material(e, None if self.mu is None else self.mu[i]).rotated(angle, axis) for i,e in enumerate(self.values)]
        return type(self)(self.wavelengths, [m.epsilon for m in rotated], wavelength_unit=self.wavelength_unit, mu=None if self.mu is None else [m.mu for m in rotated])

    def replace(self, **changes):
        allowed = {'wavelengths', 'epsilon' if self._quantity == 'epsilon' else 'index', 'mu', 'wavelength_unit'}
        if set(changes) - allowed:
            raise ValidationError('unknown table field', code='material.unknown_field', path='material', actual=sorted(set(changes)-allowed))
        values = dict(wavelengths=self.wavelengths, **{'epsilon' if self._quantity=='epsilon' else 'index':self.values}, mu=self.mu, wavelength_unit=self.wavelength_unit)
        values.update(changes)
        return (type(self).from_refractive_index if self._quantity=='index' else type(self))(**values)
