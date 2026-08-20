"""The bundled nk tables and their two readers -- the package's only dispersion front door.

Wavebit ships dispersion data in ``wavebit/nk_data`` (five files, in the wheel via
``pyproject.toml``'s package-data), and exactly two functions can open it:
``wavebit.read_material_table`` and ``wavebit.find_nk_index``.

They were briefly neither called nor exported. ``put_refractive_index_in_ucell`` had been the
only caller, and when it went the readers stayed behind in ``_core/modeler.py`` -- private,
unreferenced, and still described by that module's docstring as the way to look a material up
and pass it through ``eps_from_index``. Shipped data whose only readers are private is the
same dead end as a keyword nothing reads: the wheel carries the files, the documentation
points at them, and nothing supported can open them.

So the tests here are deliberately about the *path* rather than about interpolation accuracy.
There is no oracle for a vendor's table -- the numbers are the reference -- and pinning them
would only assert that the file on disk is the file on disk. What can go wrong and be silent
is everything around them: an export that quietly disappears, a column order that swaps ``n``
and ``k``, a lossy medium that comes back as gain, and above all the missing squaring, since
an index and a permittivity are both bare floats and the wrong one solves happily.
"""
import numpy as np
import pytest

import wavebit
from wavebit import Material, index_from_eps

# Silicon at 900 nm, from the bundled filmetrics table: n ~ 3.64 with a little absorption.
# Chosen because it is the one material whose numbers a reader can sanity-check from memory,
# and because it is absorbing there -- a lossless entry would not exercise the sign of `k`.
SI_WL = 900.0


def test_the_nk_readers_are_exported_from_the_package_root():
    """Shipped data needs a public reader, for the same reason a keyword needs one.

    Importable only from ``wavebit._core.modeler``, the tables are effectively private: a user
    is not meant to reach into ``_core``, so the wheel would carry five files that nothing
    supported could open. This is the assertion that would have failed while that was true.
    """
    assert callable(wavebit.read_material_table)
    assert callable(wavebit.find_nk_index)


def test_the_bundled_directory_loads_and_is_keyed_by_upper_cased_stem():
    """The whole ``nk_data`` tree, read the way a caller with no filenames would read it.

    Loading the directory rather than one file is the documented behaviour and worth pinning:
    it is what lets a caller ask *which* materials exist. Both formats are represented --
    ``.txt`` and MATLAB ``.mat`` -- so a change that dropped one loader would leave a table
    that still looks fine.
    """
    table = wavebit.read_material_table()

    assert {'SI', 'SIO2', 'SI3N4', 'AL2O3'} <= set(table), sorted(table)
    assert 'P_SI' in table, 'the .mat loader dropped out; only the .txt files came back'
    assert all(k == k.upper() for k in table), sorted(table)

    for name, data in table.items():
        assert data.ndim == 2 and data.shape[1] == 3, (name, data.shape)
        # Wavelength ascending is what `np.interp` requires and what nothing checks at read
        # time; out of order it returns a plausible number from the wrong rows.
        assert np.all(np.diff(data[:, 0]) > 0), f'{name}: wavelengths are not ascending'


def test_a_bundled_table_reaches_a_material_through_eps_from_index():
    """The documented three lines, walked end to end on the data actually in the wheel.

    Checked for *physics* rather than against a golden: this asserts the file parsed, the
    interpolation landed between its rows, and the two quantities did not swap. Silicon near
    900 nm is index ~3.6 with a small positive absorption, so a table read as ``(wl, k, n)``,
    or an index handed over unsquared, is visible right here.

    The ``Im > 0`` assertion is the one with a convention behind it. An nk table's ``k`` is
    already loss in this project's ``e^{-iwt}`` reading, and squaring maps the upper half
    plane into itself, so the path needs no conjugation anywhere along it. If that stopped
    holding, every lossy material read off a table would quietly become a gain medium -- which
    a solve reports as more power out than in, but only once it is large enough to notice.
    """
    table = wavebit.read_material_table()
    n = wavebit.find_nk_index('Si', table, SI_WL)

    assert n.real == pytest.approx(3.64, abs=0.05)
    assert 0 < n.imag < 0.1

    material = Material(wavebit.eps_from_index(n))
    assert complex(material.epsilon) == pytest.approx(complex(n) ** 2)
    assert complex(material.epsilon).imag > 0, 'loss must survive the squaring'
    assert not material.is_magnetic, 'a table names one quantity; mu stays unnamed'


def test_the_lookup_understands_case_and_the_real_suffix():
    """Two spellings the signature cannot show, so they are pinned instead of documented only.

    ``__real`` exists for the convergence question "is this feature doing anything, or is the
    layer just absorbing?", which is otherwise a hand-edit of the table -- and it returns a
    real float rather than a complex with a zero imaginary part, which is the detail a caller
    building an array of results will notice.
    """
    table = wavebit.read_material_table()
    n = wavebit.find_nk_index('Si', table, SI_WL)

    assert wavebit.find_nk_index('si', table, SI_WL) == n
    assert wavebit.find_nk_index('SI', table, SI_WL) == n

    real_only = wavebit.find_nk_index('si__real', table, SI_WL)
    assert real_only == pytest.approx(n.real)
    assert not isinstance(real_only, complex)


def test_forgetting_the_conversion_is_legal_and_wrong():
    """Why ``eps_from_index`` is written out at the call site instead of folded into the lookup.

    A table's number is an index, every material slot takes a permittivity, and both are bare
    floats -- so the one migration mistake with no possible refusal is skipping the squaring.
    Nothing raises. The scene is simply made of a thinner, still perfectly passive material,
    and it conserves energy while answering a different question.

    Pinned so the size of that gap sits next to the function that opens it: this is the reason
    the conversion is a visible call and not a keyword on ``find_nk_index``.
    """
    table = wavebit.read_material_table()
    n = wavebit.find_nk_index('Si', table, SI_WL)

    converted = Material(wavebit.eps_from_index(n))
    forgotten = Material(n)                        # legal, and a completely different medium

    assert abs(complex(converted.epsilon) - complex(forgotten.epsilon)) > 9.0
    assert complex(forgotten.epsilon).imag > 0, 'and still passive, so no check downstream fires'
    assert complex(index_from_eps(forgotten.epsilon)).real == pytest.approx(1.9, abs=0.1)


def test_the_stored_width_follows_type_complex():
    """``complex_dtype`` reaches the table, in every spelling the solver accepts.

    The only knob these functions have. Worth a line because it is the sort of argument that
    gets declared, threaded through, and then not applied -- which is the defect this whole
    round of work is about.
    """
    assert wavebit.read_material_table()['SI'].dtype == np.float64
    assert wavebit.read_material_table(complex_dtype=0)['SI'].dtype == np.float64
    assert wavebit.read_material_table(complex_dtype=1)['SI'].dtype == np.float32
    assert wavebit.read_material_table(complex_dtype=np.complex64)['SI'].dtype == np.float32
