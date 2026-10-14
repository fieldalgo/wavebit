"""Physical broadband constitutive models in SI units.

Rates belong to the material, independently of a problem's geometry units. The
exp(-i omega t) convention makes passive electric loss a positive imaginary epsilon.
No time discretization or solver coefficients are constructed in this module.
"""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass
from decimal import Decimal
import math
import sys
import numpy as np
from wavebit._common.backend import array_namespace, as_array, _concrete
from wavebit._common.declarations import Declaration
from wavebit.units import C0, EPSILON0, require_units
from wavebit.errors import ValidationError
from .response import Material, MaterialResponse
from .tabulated import TabulatedMaterial as LegacyTabulatedMaterial


def _namespace(*values):
    def leaves(v):
        if isinstance(v, (tuple, list)):
            return [x for item in v for x in leaves(item)]
        return [v]
    arrays = [v for v in leaves(values) if hasattr(v, 'dtype') and hasattr(v, 'shape')]
    live = [v for v in arrays if not isinstance(v, (np.ndarray, np.generic))]
    selected = live or arrays or [np.asarray(0.)]
    return array_namespace(*selected), selected[0]


def _scalar(value, name, *, positive=False, nonnegative=False):
    if getattr(value, 'shape', ()) != () or isinstance(value, (list, tuple, dict, str, bytes, bool)):
        raise ValidationError('expected a real scalar', code='material_scalar', path=name)
    xp, ref = _namespace(value)
    a = as_array(xp, value, device=getattr(ref, 'device', None))
    if xp.isdtype(a.dtype, 'complex floating'):
        raise ValidationError('expected a real scalar', code='material_scalar', path=name)
    valid = xp.isfinite(a)
    if positive: valid = valid & (a > 0)
    if nonnegative: valid = valid & (a >= 0)
    if _concrete(valid) is False:
        rule = 'positive' if positive else 'nonnegative' if nonnegative else 'finite'
        raise ValidationError(f'expected a finite {rule} scalar', code='material_scalar', path=name)
    return valid


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


def _response(epsilon, mu, term, omega):
    """Add an isotropic response to scalar/diagonal/full constitutive data."""
    xp, ref = _namespace(epsilon, mu, term, omega)
    device = getattr(ref, 'device', None)
    dtype = xp.complex64 if str(getattr(omega, 'dtype', '')) in ('float32', 'torch.float32') else xp.complex128
    e = _array(xp, epsilon, dtype=dtype, device=device)
    spectral_shape = tuple(omega.shape)
    def broadcast(value):
        a = _array(xp, value, dtype=dtype, device=device)
        if a.shape not in ((), (3,), (3, 3)):
            raise ValidationError('material response must be scalar, diagonal, or a 3x3 tensor', code='material_shape', path='material')
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
        raise ValidationError('vacuum wavelength must be real', code='wavelength', path='wavelength')
    valid = xp.isfinite(w) & (w > 0)
    if _concrete(xp.all(valid)) is False:
        raise ValidationError('vacuum wavelength must be finite and positive', code='wavelength', path='wavelength')
    safe = xp.where(valid, w, xp.ones_like(w))
    omega = 2 * math.pi * C0 / (safe * units.length_metres)
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
        value = -(self.omega_p ** 2) / (omega ** 2 + 1j * self.gamma * omega)
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
        """strength has SI units s^-2, not a dimensionless oscillator weight."""
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
        _scalar(self.conductivity, 'material.conductivity')
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
        object.__setattr__(self, 'poles', tuple(self.poles))
        if not self.poles or any(not isinstance(p, (Drude, Lorentz, Debye)) for p in self.poles):
            raise ValidationError('poles must be a nonempty sequence of Drude, Lorentz, or Debye models', code='material_poles', path='material.poles')
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
        scale = float(Decimal(str(units.length_metres)) /
                      Decimal(str(Units(length=self.wavelength_unit).length_metres)))
        try:
            xp, ref = _namespace(wavelength)
            w = as_array(xp, wavelength, device=getattr(ref, 'device', None))
            result = super().at(w * scale)
        except ValueError as exc:
            raise ValidationError(str(exc), code='material_wavelength', path='wavelength', expected=self.wavelength_range, actual=wavelength) from exc
        return MaterialResponse(result.epsilon, result.mu, _wavelength_shape=result._wavelength_shape)

    @property
    def interpolation(self):
        return self._quantity

    def rotated(self, angle, axis='z'):
        if self._quantity == 'index':
            Material(1.).rotated(angle, axis)
            return self
        rotated = [Material(e, None if self.mu is None else self.mu[i]).rotated(angle, axis) for i,e in enumerate(self.values)]
        return type(self)(self.wavelengths, [m.epsilon for m in rotated], wavelength_unit=self.wavelength_unit, mu=None if self.mu is None else [m.mu for m in rotated])

    def replace(self, **changes):
        allowed = {'wavelengths', 'epsilon' if self._quantity == 'epsilon' else 'index', 'mu', 'wavelength_unit'}
        if set(changes) - allowed:
            raise ValidationError('unknown table field', code='unknown_field', path='material', actual=sorted(set(changes)-allowed))
        values = dict(wavelengths=self.wavelengths, **{'epsilon' if self._quantity=='epsilon' else 'index':self.values}, mu=self.mu, wavelength_unit=self.wavelength_unit)
        values.update(changes)
        return (type(self).from_refractive_index if self._quantity=='index' else type(self))(**values)
