"""FDTD physical problems, numerical settings, and immutable prepared execution."""
from ._problem import Problem, Scene
from ._geometry import (Domain, Shape, SnappedSlab, SnappedBox, Plane, Sphere, Box, Cylinder, Prism,
                        Structure)
from ._boundaries import Open, Periodic, Bloch, PEC, PMC, MirrorSymmetry, InvariantAxis, Boundaries
from ._sources import GaussianPulse, PlaneWave, CurrentDipole, TFSF, ModeSource, InitialField
from ._monitors import (TimeSeriesMonitor, FieldMonitor, FluxMonitor, DiffractionMonitor,
                        FarFieldMonitor, ModeMonitor, Outputs)
from ._numerics import UniformGrid, Duration, PML, ConductiveAbsorber, DecayStop, Numerics
from ._api import Prepared, check, prepare, run
from ._result import Result, SpectralFlux, FarFieldSpectrum
from ._modes import solve_modes, WaveguideMode, WaveguideModes, ModeSpectrum
from . import native, periodic

__all__ = ['Problem', 'Scene',
           'Domain', 'Shape', 'SnappedSlab', 'SnappedBox', 'Plane', 'Sphere', 'Box', 'Cylinder',
           'Prism', 'Structure',
           'Open', 'Periodic', 'Bloch', 'PEC', 'PMC', 'MirrorSymmetry', 'InvariantAxis', 'Boundaries',
           'GaussianPulse', 'PlaneWave', 'CurrentDipole', 'TFSF', 'ModeSource', 'InitialField',
           'TimeSeriesMonitor', 'FieldMonitor', 'FluxMonitor', 'DiffractionMonitor',
           'FarFieldMonitor', 'ModeMonitor', 'Outputs',
           'UniformGrid', 'Duration', 'PML', 'ConductiveAbsorber', 'DecayStop', 'Numerics',
           'Prepared', 'check', 'prepare', 'run',
           'Result', 'SpectralFlux', 'FarFieldSpectrum',
           'solve_modes', 'WaveguideMode', 'WaveguideModes', 'ModeSpectrum',
           'native', 'periodic']

#: Names this module no longer has, and what replaced them. Each is a tombstone, not an alias:
#: reading one raises, naming the replacement.
from ._public_names import PUBLIC_NAMES as _RENAMED  # noqa: E402

#: Implementation modules that used to be importable as public paths.
_PRIVATE_MODULES = ('problem', 'geometry', 'boundaries', 'sources', 'monitors', 'numerics',
                    'result', 'results')


from ._public_names import Answered as _Answered, removed as _removed  # noqa: E402


def __getattr__(name):
    if name in _RENAMED:
        raise _removed('wavebit.fdtd', name, f'wavebit.fdtd.{_RENAMED[name]}')
    if name in _PRIVATE_MODULES:
        raise AttributeError(
            f'wavebit.fdtd.{name} is no longer a public module; every public name lives in '
            f'wavebit.fdtd itself (for example wavebit.fdtd.FluxMonitor). See the public-name '
            f'migration guide, docs/migration/public-names.md.', name=name, obj=_Answered())
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    """The public surface only, so ``dir(fdtd)`` and Python's "Did you mean" name no private module."""
    return sorted(set(__all__) | {'diagnostics'}
                  | {n for n in globals() if n.startswith('__') and n.endswith('__')})
