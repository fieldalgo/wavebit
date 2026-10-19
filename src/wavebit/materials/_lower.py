"""Explicit conversion of physical models to normalized time-domain declarations.

This adapter creates no mesh or ADE coefficients. Those remain FDTD responsibilities.
"""
from wavebit.units import require_units
from wavebit.errors import UnsupportedFeatureError
from .response import Material, MaterialResponse
from .models import ConductiveMaterial, DispersiveMaterial, TabulatedMaterial, Drude, Lorentz, Debye
from . import _time_domain as td


def lower_material(model, *, units):
    """Convert SI rates/conductivity exactly once for a declared geometry length unit."""
    units = require_units(units)
    if isinstance(model, td.MaterialBase):
        raise UnsupportedFeatureError('normalized engine materials are not physical declarations; use a common physical material model', code='legacy_material', path='material')
    if isinstance(model, (TabulatedMaterial, MaterialResponse)):
        raise UnsupportedFeatureError('time-domain execution requires a supported causal material model, not a table or an evaluated response', code='causal_model_required', path='material')
    if not isinstance(model, (Material, ConductiveMaterial, DispersiveMaterial)):
        model = Material(model)
    def diagonal(value):
        # Extract scalar leaves without NumPy conversion or graph detachment.
        if tuple(getattr(value, 'shape', ())) == (3,):
            return tuple(value[i] for i in range(3))
        return value
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
            else: raise UnsupportedFeatureError('unsupported material pole', code='material_pole', path='material.poles')
        return td.Dispersive(eps_inf=diagonal(model.epsilon_inf), poles=tuple(poles), mu=diagonal(model.mu))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, UnsupportedFeatureError): raise
        raise UnsupportedFeatureError(str(exc), code='fdtd_material', path='material') from exc
