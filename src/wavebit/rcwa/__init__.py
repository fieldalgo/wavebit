"""Rigorous coupled-wave analysis: immutable problems and explicit preparation."""
from .problem import Problem, Stack, Layer, Region, Rectangle, Circle, Ellipse, Polygon, Excitation, Outputs
from .sources import PlaneWave, IncidentOrders, GaussianBeam, ElectricDipole
from .numerics import Numerics
from .results import Result, BoundaryPower
from ._lifecycle import Prepared, check, prepare, run
from ._structure import sample_structure
from . import diagnostics  # noqa: F401 -- public submodule, listed by __dir__ as fdtd.diagnostics is

__all__ = ['Problem', 'Stack', 'Layer', 'Region', 'Rectangle', 'Circle', 'Ellipse', 'Polygon',
           'Excitation', 'Outputs', 'PlaneWave', 'IncidentOrders', 'GaussianBeam', 'ElectricDipole',
           'Numerics', 'Result', 'BoundaryPower', 'Prepared', 'check', 'prepare', 'run',
           'sample_structure']

#: Names this module no longer has (fieldalgo/wavebit#57), and their full replacements. Each is
#: a tombstone, not an alias: reading one raises, naming the replacement.
_RENAMED = {'OrderExcitation': 'wavebit.rcwa.IncidentOrders',
            'layer_thicknesses': 'wavebit.layer_thicknesses'}

#: Modules that used to be importable as public paths and are now private.
_PRIVATE_MODULES = ('plotting',)

#: Public submodules beside ``__all__``; ``operators`` and ``experimental`` load on first import.
_SUBMODULES = ('diagnostics', 'experimental', 'operators')


def __getattr__(name):
    from .._common.declarations import Answered, removed
    if name in _RENAMED:
        raise removed('wavebit.rcwa', name, _RENAMED[name])
    if name in _PRIVATE_MODULES:
        raise AttributeError(
            f'wavebit.rcwa.{name} is no longer public: it takes the engine solver, not a public '
            f'result, and has no public replacement. See the public-name migration guide, '
            f'docs/migration/public-names.md.', name=name, obj=Answered())
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    """The public surface only, so ``dir(rcwa)`` and Python's "Did you mean" name no private module."""
    return sorted(set(__all__) | set(_SUBMODULES)
                  | {n for n in globals() if n.startswith('__') and n.endswith('__')})
