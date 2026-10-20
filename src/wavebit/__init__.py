"""Differentiable electromagnetic simulation with shared physical materials.

Use rcwa.Problem or fdtd.Problem with the corresponding check/prepare/run
lifecycle. Historical inputs require explicit migration; see wavebit.migration.
Importing this package never enables a backend or changes global precision.
"""
__version__ = '0.0.4'

from . import materials, rcwa, fdtd
from .materials import (Material, MaterialResponse, ConductiveMaterial, DispersiveMaterial,
                        TabulatedMaterial, Drude, Lorentz, Debye, eps_from_refractive_index,
                        refractive_index_from_eps)
from .units import Units
from .lattice import Lattice
from ._common.execution import Execution, CheckReport
from ._common.enums import Backend
from .observations import (DiffractionOrders, FieldSamples, RawFieldSpectrum,
                           RawDiffractionSpectrum, TimeSeries, Validity)
from .errors import (WavebitError, ValidationError, UnsupportedFeatureError, IncompatibleProblemError,
                     NumericalError, Issue)
from .jobs import Job, run

__all__ = ['materials', 'rcwa', 'fdtd', 'Material', 'MaterialResponse', 'ConductiveMaterial',
           'DispersiveMaterial', 'TabulatedMaterial', 'Drude', 'Lorentz', 'Debye',
           'eps_from_refractive_index', 'refractive_index_from_eps', 'Units', 'Lattice',
           'Execution', 'CheckReport', 'Backend', 'DiffractionOrders', 'FieldSamples',
           'RawFieldSpectrum', 'RawDiffractionSpectrum', 'TimeSeries', 'Validity',
           'ValidationError', 'UnsupportedFeatureError', 'IncompatibleProblemError',
           'NumericalError', 'WavebitError', 'Issue', 'Job', 'run']

#: Names this package no longer has (fieldalgo/wavebit#57): each is a tombstone naming its
#: replacement, not an alias.
_RENAMED = {'SpectralFieldSamples': 'RawFieldSpectrum'}


def __getattr__(name):
    if name in _RENAMED:
        from ._common.declarations import removed
        raise removed('wavebit', name, f'wavebit.{_RENAMED[name]}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
