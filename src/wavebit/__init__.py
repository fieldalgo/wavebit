"""Differentiable electromagnetic simulation with shared physical materials.

Use rcwa.Problem or fdtd.Problem with the corresponding check/prepare/run
lifecycle. Transitional old entry points live explicitly in compat.legacy.
Importing this package never enables a backend or changes global precision.
"""
__version__ = '0.0.2'

from . import materials, rcwa, fdtd
from .materials import (Material, MaterialResponse, ConductiveMaterial, DispersiveMaterial,
                        TabulatedMaterial, Drude, Lorentz, Debye, eps_from_refractive_index,
                        refractive_index_from_eps)
from .units import Units
from .lattice import Lattice
from ._common.execution import Execution, CheckReport
from ._common.enums import Backend
from .observations import (DiffractionOrders, FieldSamples, SpectralFieldSamples,
                           RawDiffractionSpectrum, TimeSeries, Validity)
from .errors import ValidationError, UnsupportedFeatureError, IncompatibleProblemError, NumericalError, Issue
from .jobs import Job, run

__all__ = ['materials', 'rcwa', 'fdtd', 'Material', 'MaterialResponse', 'ConductiveMaterial',
           'DispersiveMaterial', 'TabulatedMaterial', 'Drude', 'Lorentz', 'Debye',
           'eps_from_refractive_index', 'refractive_index_from_eps', 'Units', 'Lattice',
           'Execution', 'CheckReport', 'Backend', 'DiffractionOrders', 'FieldSamples',
           'SpectralFieldSamples', 'RawDiffractionSpectrum', 'TimeSeries', 'Validity',
           'ValidationError', 'UnsupportedFeatureError', 'IncompatibleProblemError',
           'NumericalError', 'Issue', 'Job', 'run']
