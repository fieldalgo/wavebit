"""The bundled nk material tables, and the two ways to read one.

:func:`find_material` is the front door and hands back a :class:`~wavebit._core.domain.Material`
-- a *permittivity*, which is what every material slot in this library takes.
:func:`find_nk_index` is the low-level one and hands back the table's own quantity, ``n + 1j*k``,
for the questions that want the index itself: a dispersion curve, a datasheet check.

This module was the block modeller until 2026-08-26, and the tables were its other tenant. The
path is gone -- its only offering was Li's rule on exactly transformed shapes, which measured
bit-identical to a raster under the same rule on a commensurate 1D grating and behind NVF
everywhere else -- and the tables are what is left.

NumPy throughout, and that is the one genuinely backend-specific thing here: the table is read
off disk (``np.loadtxt`` / ``scipy.io.loadmat``) and interpolated with NumPy, because there is
no Array-API ``interp`` and static data is not a design variable.
"""
from os import walk
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ namespace / dtype helpers



# ------------------------------------------------------------------------------ material table

#: Down here with the nk readers, not up with the modeller helpers where it was written.
#: `read_material_table` is its only surviving caller -- the other one was the block
#: modeller's `complex_dtype` setter -- and it sat inside a contiguous run of block
#: machinery, where a region delete would take it out and break a public export.
def _precision_is_double(complex_dtype):
    """Map any spelling of the complex dtype onto the double / single precision flag.

    Accepts the legacy shorthands (``0`` = complex128, ``1`` = complex64) as well as any
    backend's dtype object — matching on the bit width keeps this free of ``if torch:``.
    """
    if complex_dtype is None:
        return True
    if isinstance(complex_dtype, int) and not isinstance(complex_dtype, bool):
        if complex_dtype == 0:
            return True
        if complex_dtype == 1:
            return False
        raise ValueError(f'unsupported complex_dtype: {complex_dtype}')
    s = str(complex_dtype).lower()
    if '128' in s:
        return True
    if '64' in s or '32' in s:
        return False
    raise ValueError(f'unsupported complex_dtype: {complex_dtype}')

#
# The nk tables are static data read off disk, not design variables, so they are held and
# interpolated in NumPy: there is no Array-API ``interp``, and nothing downstream
# differentiates through the table lookup. Both functions are public
# (``wavebit.read_material_table`` / ``wavebit.find_material`` / ``wavebit.find_nk_index``)
# -- see the module docstring for
# the three-line path from a wavelength to a Material, and for why the permittivity conversion
# is left to the caller.

def find_nk_index(material, mat_table, wl):
    """Complex refractive index of ``material`` at ``wl``, interpolated from the nk table.

    ``mat_table`` is what :func:`read_material_table` returns and ``material`` is one of its
    keys, matched case-insensitively (``'si'``, ``'Si'`` and ``'SI'`` are one material).

    Returns ``n + 1j*k``, an **index**, because that is the quantity an nk table publishes and
    this is the function that reads one. It is the *low-level* door:
    :func:`find_material` is the one that fills a material slot, and it hands back a
    :class:`~wavebit._core.domain.Material` so there is no conversion to forget.

    This one is for the questions that want the index itself -- plotting a dispersion curve,
    checking a datasheet, comparing against a table published as ``n`` and ``k``.

    Loss comes back as ``+1j*k``, which is already this project's convention
    (``Im > 0`` is loss under ``e^{-iwt}``, ``docs/conventions.md``) and survives
    ``eps_from_index`` because squaring maps the upper half plane into itself. Nothing has to
    be conjugated on the way in.

    A ``__real`` suffix on the name asks for ``n`` only (no absorption) -- ``'si__real'``.
    It exists for the convergence question "is this feature doing anything, or is the layer
    just absorbing?", which is otherwise a hand-edit of the table.
    """
    if material[-6:] == '__real':
        material = material[:-6]
        n_only = True
    else:
        n_only = False

    mat_data = mat_table[material.upper()]
    n_index = np.interp(wl, mat_data[:, 0], mat_data[:, 1])

    if n_only:
        return n_index

    k_index = np.interp(wl, mat_data[:, 0], mat_data[:, 2])

    return n_index + 1j * k_index


def find_material(material, mat_table, wl):
    """``material`` at ``wl`` as a :class:`~wavebit._core.domain.Material`, ready for a slot.

    The front door, and the reason it exists is that every material slot in wavebit takes a
    **relative permittivity** -- ``top``, ``bot``, the raster, a shape's ``material``, a block
    slot -- while an nk table publishes a refractive index. Two bare complex numbers, one
    convention apart, and nothing to tell them apart at the point of use.

    :func:`find_nk_index` returns the index and the caller squared it. The conversion was left
    visible on purpose, the argument being that a call you can see is a call you will not skip.
    That argument is wrong, and the size of it is measurable: silicon at 900 nm is ``n = 3.6388
    + 0.0075j``, so ``top=n`` builds a medium of permittivity 3.64 where 13.24 was meant. Both
    solve. Both conserve energy. `T` comes out 0.9026 against 0.6764 -- a fifth of the light,
    for a squaring. No refusal can fire, because a number carries no name.

    Handing back a `Material` is what gives it a name. There is nothing left to forget.

    ``mat_table`` is what :func:`read_material_table` returns and ``material`` is one of its
    keys, matched case-insensitively. A ``__real`` suffix asks for ``n`` only, no absorption --
    it survives here because "is this feature doing anything, or is the layer just absorbing?"
    is a question about a *scene*, so it belongs where scenes are built.

    Non-magnetic: the tables carry no permeability, so ``mu`` is left unset rather than
    asserted as 1. A magnetic medium is `Material(find_material(...).epsilon, mu=...)`.
    """
    from .domain import Material, eps_from_index

    return Material(eps_from_index(find_nk_index(material, mat_table, wl)))


def read_material_table(nk_path=None, complex_dtype=None):
    """Load every nk table under ``nk_path`` (defaults to the bundled ``wavebit/nk_data``).

    Returns ``{NAME: (n_rows, 3) array}`` with columns ``(wavelength, n, k)``, keyed by the
    file's stem upper-cased, walking the directory recursively and reading both ``.txt``
    (whitespace, one header line) and MATLAB ``.mat`` (``WL`` / ``n`` / ``k``). Directory
    layout carries no meaning: two files of the same stem in different subdirectories are one
    key, and the last one walked wins.

    The wavelength column is in **whatever unit the file uses**, and nothing here converts it.
    A solve has one length unit shared by ``wavelength``, ``period`` and ``thickness``, so the
    table has to be in that unit too -- and since a table's own unit is not recorded anywhere
    in the format, this cannot check it for you. The bundled ``filmetrics`` tables are in nm.

    ``complex_dtype`` only sets the stored float width (single or double); it takes any spelling
    the solver does. Reading the whole directory is deliberate over a per-material open: the
    tables are small, and a caller who has the dict can ask which materials exist rather than
    guessing at filenames.
    """
    dtype = np.float64 if _precision_is_double(complex_dtype) else np.float32

    mat_table = {}

    if nk_path is None:
        nk_path = str(Path(__file__).resolve().parent.parent) + '/nk_data'

    full_path_list, name_list = [], []
    for (dirpath, _dirnames, filenames) in walk(nk_path):
        full_path_list.extend([f'{dirpath}/{filename}' for filename in filenames])
        name_list.extend(filenames)

    for path, name in zip(full_path_list, name_list):
        if name[-3:] == 'txt':
            data = np.loadtxt(path, skiprows=1)
            mat_table[name[:-4].upper()] = data.astype(dtype)

        elif name[-3:] == 'mat':
            from scipy.io import loadmat
            data = loadmat(path)
            data = np.array([data['WL'], data['n'], data['k']], dtype=dtype)[:, :, 0].T
            mat_table[name[:-4].upper()] = data

    return mat_table
