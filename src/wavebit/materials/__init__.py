"""Shared physical constitutive models and optical catalog."""
from .response import Material, MaterialResponse, as_material as as_material, eps_from_index, index_from_eps
from .models import ConductiveMaterial, DispersiveMaterial, Drude, Lorentz, Debye, TabulatedMaterial
from .catalog import available, load
__all__ = ['Material', 'MaterialResponse', 'ConductiveMaterial', 'DispersiveMaterial', 'Drude', 'Lorentz', 'Debye', 'TabulatedMaterial', 'available', 'load', 'eps_from_index', 'index_from_eps']
