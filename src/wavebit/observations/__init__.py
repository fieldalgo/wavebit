"""Solver-independent field and diffraction observation contracts."""
from .contracts import DiffractionOrder, DiffractionOrders, RawDiffractionSpectrum, Validity
from .fields import FieldSamples, RawFieldSpectrum, TimeSeries
__all__ = ['DiffractionOrder', 'DiffractionOrders', 'RawDiffractionSpectrum', 'Validity', 'FieldSamples', 'RawFieldSpectrum', 'TimeSeries']

from .modes import ModeFit, fit_modes
__all__ += ["ModeFit", "fit_modes"]

#: Names this module no longer has (fieldalgo/wavebit#57): each is a tombstone naming its
#: replacement, not an alias.
_RENAMED = {'SpectralFieldSamples': 'RawFieldSpectrum'}


def __getattr__(name):
    if name in _RENAMED:
        from wavebit._common.declarations import removed
        raise removed('wavebit.observations', name, f'wavebit.observations.{_RENAMED[name]}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
