"""Explicit conversion of physical models to normalized time-domain declarations.

This adapter creates no mesh or ADE coefficients. Those remain FDTD responsibilities.
"""
import math

from wavebit.units import require_units
from wavebit.errors import UnsupportedFeatureError
from .response import Material, MaterialResponse
from .models import ConductiveMaterial, DispersiveMaterial, TabulatedMaterial, Drude, Lorentz, Debye
from . import _time_domain as td


#: The SI unit each converted public field is quoted in.
_SI = {'conductivity': ' in S/m', 'omega_p': ' in rad/s', 'omega_0': ' in rad/s',
       'gamma': ' in rad/s', 'strength': ' in rad^2/s^2', 'tau': ' in s'}


def _refuse(message):
    return UnsupportedFeatureError(message, code='material.fdtd', path='material')


def _components(where, name, value):
    """``value`` as its per-axis entries, refused as the engine refuses it, in public words."""
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise _refuse(f'{where}({name}={value!r}): a diagonal is (x, y, z), three entries; '
                          f'got {len(value)}.')
        return tuple(value)
    shape = tuple(getattr(value, 'shape', ()))
    if shape == (3,):
        return tuple(value[i] for i in range(3))
    if shape == (3, 3):
        raise _refuse(f'{where}({name}=...): a full 3x3 tensor is not a time-domain material; '
                      f'FDTD takes a scalar or a diagonal (x, y, z).')
    if shape != ():
        raise _refuse(f'{where}({name}=...): one scalar per component; got shape {shape}. A '
                      f'value per cell is not a material: say the geometry with shapes '
                      f'(SnappedSlab, SnappedBox, Sphere, Box, Cylinder, Prism), each with one '
                      f'material; a diagonal is {name}=(x, y, z).')
    return (value,) * 3


def _complex(where, name, value):
    if name in ('epsilon', 'epsilon_inf'):
        return _refuse(
            f'{where}({name}={value!r}): a complex permittivity is refused. One complex '
            f'constant is right at one frequency and silently wrong at every other, and a '
            f'time-domain run is broadband by construction. Say ConductiveMaterial(epsilon=..., '
            f'conductivity=...) for a loss that is flat in conductivity, or '
            f'DispersiveMaterial(epsilon_inf=..., poles=(Drude(...), Lorentz(...), Debye(...))) '
            f'for one with a frequency dependence; DispersiveMaterial.at(wavelength, '
            f'units=...) turns the latter back into the single complex number a '
            f'frequency-domain solver wants.')
    return _refuse(f'{where}({name}={value!r}): {name} is a real number.')


def _check(where, name, public, internal=None, *, strict=True):
    """Refuse ``public`` where the engine would refuse ``internal`` (its value converted to the
    engine's units, ``public`` itself when there is no conversion), quoting the public value.

    The rule is read off the converted value, so what is admitted is exactly what the engine
    admits -- a conversion that underflows a tiny positive rate to zero is refused here too --
    while the message names the field and the number the caller wrote."""
    publics = _components(where, name, public)
    internals = publics if internal is None else _components(where, name, internal)
    uniform = all(p is publics[0] for p in publics)
    for axis, p, v in zip('xyz', publics, internals):
        if td._is_complex(p):
            raise _complex(where, name, p)
        c = td.concrete(v)
        if c is None:
            continue                                  # traced: the runtime guards hold it
        if not math.isfinite(c) or (c <= 0 if strict else c < 0):
            what = 'positive' if strict else 'non-negative'
            entry = '' if uniform else f' ({axis} entry)'
            raise _refuse(f'{where}({name}=...){entry}: {name} is a finite {what} '
                          f'number{_SI.get(name, "")}; got {td.concrete(p)!r}.')


def _check_public(model, units):
    """The engine's material rules, asked of the public declaration in its own names and SI
    values before anything is converted (issue #56)."""
    where = type(model).__name__
    if isinstance(model, DispersiveMaterial):
        _check(where, 'epsilon_inf', model.epsilon_inf)
        scale = units.time_unit_seconds
        for p in model.poles:
            pole = type(p).__name__
            if isinstance(p, Drude):
                _check(pole, 'omega_p', p.omega_p, p.omega_p * scale)
                _check(pole, 'gamma', p.gamma, p.gamma * scale, strict=False)
            elif isinstance(p, Lorentz):
                _check(pole, 'omega_0', p.omega_0, p.omega_0 * scale)
                _check(pole, 'gamma', p.gamma, p.gamma * scale, strict=False)
                _check(pole, 'strength', p.strength, p.strength * scale ** 2, strict=False)
            elif isinstance(p, Debye):
                _check(pole, 'tau', p.tau, p.tau / scale)
                _check(pole, 'delta_epsilon', p.delta_epsilon, strict=False)
    else:
        _check(where, 'epsilon', model.epsilon)
    if isinstance(model, ConductiveMaterial):
        _check(where, 'conductivity', model.conductivity,
               units._conductivity_to_internal(model.conductivity), strict=False)
    if model.mu is not None:
        _check(where, 'mu', model.mu)


def lower_material(model, *, units):
    """Convert SI rates/conductivity exactly once for a declared geometry length unit."""
    units = require_units(units)
    if isinstance(model, td.MaterialBase):
        raise UnsupportedFeatureError('normalized engine materials are not physical declarations; use a common physical material model', code='material.engine_model', path='material')
    if isinstance(model, (TabulatedMaterial, MaterialResponse)):
        raise UnsupportedFeatureError('time-domain execution requires a supported causal material model, not a table or an evaluated response', code='material.causal_model', path='material')
    if not isinstance(model, (Material, ConductiveMaterial, DispersiveMaterial)):
        model = Material(model)
    def diagonal(value):
        # Extract scalar leaves without NumPy conversion or graph detachment.
        if tuple(getattr(value, 'shape', ())) == (3,):
            return tuple(value[i] for i in range(3))
        return value
    _check_public(model, units)
    try:
        if isinstance(model, Material):
            return td.Material(eps=diagonal(model.epsilon), mu=diagonal(model.mu))
        if isinstance(model, ConductiveMaterial):
            return td.Conductor(eps=diagonal(model.epsilon), sigma=units._conductivity_to_internal(model.conductivity), mu=diagonal(model.mu))
        poles = []
        scale = units.time_unit_seconds
        for p in model.poles:
            if isinstance(p, Drude): poles.append(td.Drude(omega_p=p.omega_p * scale, gamma=p.gamma * scale))
            elif isinstance(p, Lorentz): poles.append(td.Lorentz(omega_0=p.omega_0 * scale, gamma=p.gamma * scale, strength=p.strength * scale ** 2))
            elif isinstance(p, Debye): poles.append(td.Debye(tau=p.tau / scale, delta_eps=p.delta_epsilon))
            else: raise UnsupportedFeatureError('unsupported material pole', code='material.pole_type', path='material.poles')
        return td.Dispersive(eps_inf=diagonal(model.epsilon_inf), poles=tuple(poles), mu=diagonal(model.mu))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, UnsupportedFeatureError): raise
        raise UnsupportedFeatureError(str(exc), code='material.fdtd', path='material') from exc
