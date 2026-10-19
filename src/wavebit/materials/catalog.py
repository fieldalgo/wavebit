"""Select bundled optical materials without managing files or lookup tables.

``load('Si')`` returns a reusable dispersion model in nanometres. Assign it to a
region once; RCWA evaluates it at each solve wavelength. ``model.at(900.)``
explicitly evaluates a numerical ``Material``. No I/O occurs on import.
"""
from functools import lru_cache
from importlib.resources import files

import numpy as np

from wavebit.materials.models import TabulatedMaterial
from wavebit._common.backend import eager_under_torch_compile as _eager_under_torch_compile

__all__ = ['available', 'load']

# Explicit entries preserve source identity and avoid filesystem-order defaults.
# Each identifier names the publications its table comes from; the data are
# from the public-domain refractiveindex.info database, cited in
# data/refractiveindex/README.md and rebuilt by tools/build_material_tables.py.
_CATALOG = {
    'Malitson1962/Al2O3': 'refractiveindex/Al2O3.txt',
    'Green2008-Li1980/Si': 'refractiveindex/Si.txt',
    'Luke2015/Si3N4': 'refractiveindex/Si3N4.txt',
    'Malitson1965/SiO2': 'refractiveindex/SiO2.txt',
}
# Tables shipped until 2026-10-19 whose source terms could not be confirmed.
_REMOVED = {f'filmetrics/{name}' for name in ('Al2O3', 'Si', 'Si3N4', 'SiO2')}
_UNIT_METRES = {'nm': 1e-9, 'um': 1e-6, 'mm': 1e-3, 'm': 1.}


@_eager_under_torch_compile
def available():
    """Return the sorted, source-qualified material identifiers, without file I/O.

    Short names such as ``'Si'`` are accepted by :func:`load` when unambiguous.
    Listing the qualified identifier records which dataset is being selected;
    a chemical formula alone does not specify every form of a real material.
    """
    return tuple(sorted(_CATALOG))


@_eager_under_torch_compile
def load(name, *, wavelength_unit='nm'):
    """Load one bundled material as an immutable :class:`wavebit.TabulatedMaterial`.

    Example: ``silicon = wavebit.materials.catalog.load('Si')``. No wavelength argument
    is accepted: loading always returns a dispersion model, and ``silicon.at(w)``
    always returns numerical constitutive values. Names are case-insensitive.
    Use ``'Green2008-Li1980/Si'`` to identify the dataset explicitly. Ambiguous short
    names raise an error instead of choosing an arbitrary source.

    ``wavelength_unit`` is the unit of **all lengths in the receiving scene**:
    ``'nm'`` (default), ``'um'``, ``'mm'`` or ``'m'``. Only the table's wavelength
    axis is converted here; scene geometry and solver wavelengths are never
    rescaled implicitly. For a micrometre scene, use ``load('Si', wavelength_unit='um')``
    with e.g. ``wavelength=0.9``. Numeric scene inputs carry no unit metadata.

    Only the selected data file is read, once per cached material and unit.
    Its published n and k are interpolated before conversion to permittivity.
    Evaluation outside the table range is an error eagerly and NaN under JAX
    tracing; no extrapolation, endpoint clamping or implicit loss removal occurs.
    Tables are static data, not time-domain dispersion models for FDTD.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError('material name must be a nonempty string; see materials.available()')
    if not isinstance(wavelength_unit, str) or wavelength_unit not in _UNIT_METRES:
        raise ValueError("wavelength_unit must be 'nm', 'um', 'mm' or 'm'")
    query = name.strip().casefold()
    if query in {key.casefold() for key in _REMOVED}:
        raise ValueError(f'{name!r} was removed on 2026-10-19 because its source terms could not be confirmed; '
                         f'load the short name or one of: {", ".join(available())}')
    matches = [key for key in _CATALOG
               if key.casefold() == query or ('/' not in query and key.split('/')[-1].casefold() == query)]
    if not matches:
        raise ValueError(f'unknown material {name!r}; available: {", ".join(available())}')
    if len(matches) != 1:
        raise ValueError(f'ambiguous material {name!r}; use a qualified name: {", ".join(sorted(matches))}')
    return _load(matches[0], wavelength_unit)


@lru_cache(maxsize=64)
def _load(identifier, wavelength_unit):
    resource = files('wavebit').joinpath('data', _CATALOG[identifier])
    with resource.open('r', encoding='utf-8-sig') as stream:
        header = stream.readline().split()
        if header != ['Wavelength(nm)', 'n', 'k']:
            raise ValueError(f'invalid wavelength units or columns in bundled material {identifier!r}')
        data = np.loadtxt(stream)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f'bundled material {identifier!r} must have wavelength, n, k columns')
    wavelengths = data[:, 0] * (_UNIT_METRES['nm'] / _UNIT_METRES[wavelength_unit])
    return TabulatedMaterial.from_refractive_index(wavelengths, data[:, 1] + 1j*data[:, 2], wavelength_unit=wavelength_unit)
