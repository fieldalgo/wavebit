"""Typed enumerations for the public API.

These are ``IntEnum``s equal to the integer codes the solver already uses, so they are a
drop-in, non-breaking upgrade: ``Pol.TE == 0`` is true, and the existing ``if self.pol == 0``
/ ``if self.fourier_type in (2, 'nvf', 'NVF')`` comparisons keep working unchanged. Users may
pass either the enum (``pol=Pol.TE``) or the raw int (``pol=0``).
"""
from enum import IntEnum


class Backend(IntEnum):
    """Array backend for :func:`fieldwave.rcwa`."""
    NUMPY = 0
    JAX = 1
    TORCH = 2


class Pol(IntEnum):
    """Incident polarization (1D / conical)."""
    TE = 0
    TM = 1


class FourierType(IntEnum):
    """Fourier-factorization / permittivity-representation method."""
    DFS = 0         # discrete Fourier series (standard Li rules on a raster)
    CONTINUOUS = 1  # analytic Fourier of axis-aligned raster rectangles
    NVF = 2         # normal vector field (also accepts the strings 'nvf' / 'NVF')
