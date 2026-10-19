"""Shared physical constitutive models and optical catalog."""
from .response import Material, MaterialResponse, as_material as as_material, eps_from_refractive_index, refractive_index_from_eps
from .models import ConductiveMaterial, DispersiveMaterial, Drude, Lorentz, Debye, TabulatedMaterial
from .catalog import available, load
__all__ = ['Material', 'MaterialResponse', 'ConductiveMaterial', 'DispersiveMaterial', 'Drude', 'Lorentz', 'Debye', 'TabulatedMaterial', 'available', 'load', 'eps_from_refractive_index', 'refractive_index_from_eps']


#: Implementation modules that used to be importable as public paths. Their classes are the
#: solvers' normalized-unit models (a second Drude with other rate units), not the public ones.
_PRIVATE_MODULES = ('time_domain', 'tabulated')


def __getattr__(name):
    if name in _PRIVATE_MODULES:
        from wavebit._common.declarations import Answered
        raise AttributeError(
            f'wavebit.materials.{name} is no longer a public module; the public materials live in '
            f'wavebit.materials itself (for example wavebit.materials.Drude, in SI units). See the '
            f'public-name migration guide, docs/migration/public-names.md.', name=name, obj=Answered())
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
