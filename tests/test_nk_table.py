"""The bundled nk tables and their two readers -- the package's only dispersion front door.

Wavebit ships dispersion data in ``wavebit/nk_data`` (five files, in the wheel via
``pyproject.toml``'s package-data), and exactly two functions can open it:
``wavebit.read_material_table``, ``wavebit.find_material`` and
``wavebit.find_nk_index``.

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
import inspect

import numpy as np
import pytest

import wavebit
from wavebit import Material

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


def test_the_front_door_hands_back_a_material_so_there_is_nothing_to_forget():
    """``find_material`` exists because the visible conversion did not survive being visible.

    ``find_nk_index`` returns an index, every material slot takes a permittivity, and both are
    bare complex numbers. The conversion was written out at the call site on purpose -- the
    argument being that a call you can see is a call you will not skip -- and this test used to
    pin how wrong skipping it is, under the name ``test_forgetting_the_conversion_is_legal_and_
    wrong``. It is the same measurement, read the other way round: nothing raises, the medium is
    simply thinner, and the answer is a fifth of the light off.

    The fix is not a louder call. It is handing back a value that carries its own name.
    """
    table = wavebit.read_material_table()
    material = wavebit.find_material('Si', table, SI_WL)
    index = wavebit.find_nk_index('Si', table, SI_WL)

    assert isinstance(material, Material)
    assert complex(material.epsilon) == pytest.approx(complex(index) ** 2)
    assert material.mu is None, 'an nk table carries no permeability, so none is asserted'

    # The size of the mistake the front door removes, kept as a number rather than a claim.
    forgotten = Material(index)
    assert abs(complex(material.epsilon) - complex(forgotten.epsilon)) > 9.0
    assert complex(forgotten.epsilon).imag > 0, 'and still passive, so nothing downstream fires'


def test_the_two_doors_answer_the_same_medium_in_their_own_quantities():
    """One reads the table, the other names what the table means. They cannot disagree.

    `find_material` is `Material(eps_from_index(find_nk_index(...)))` and this is the test that
    says so -- through the solver, because the quantity that matters is the one the solve reads.
    """
    table = wavebit.read_material_table()
    solver = wavebit.rcwa(backend=0, structure=np.ones((1, 1, 8)), period=[1000.0, 1000.0],
                          truncation_order=[2, 0], thickness=[150.0], wavelength=900.0,
                          top=wavebit.find_material('Si', table, SI_WL))

    assert complex(solver.n_top) == pytest.approx(complex(wavebit.find_nk_index('Si', table, SI_WL)))
    assert np.isfinite(np.asarray(solver.solve().T).sum())


def test_the_real_only_suffix_reaches_the_front_door_too():
    """``__real`` asks for ``n`` with no absorption, and that is a question about a *scene*.

    It exists for "is this feature doing anything, or is the layer just absorbing?", so it has
    to be answerable where scenes are built, not only at the low-level reader.
    """
    table = wavebit.read_material_table()
    lossless = wavebit.find_material('si__real', table, SI_WL)
    lossy = wavebit.find_material('Si', table, SI_WL)

    assert complex(lossless.epsilon).imag == pytest.approx(0.0, abs=1e-15)
    assert complex(lossy.epsilon).imag > 0
    assert complex(lossless.epsilon).real == pytest.approx(complex(lossy.epsilon).real, rel=1e-4)


def test_a_swept_wavelength_is_refused_by_the_slot_rather_than_by_the_lookup():
    """An array ``wl`` gives a `Material` holding several permittivities, which is not one medium.

    `find_material` does not police that, because the half-space setter already does and says
    the shape out loud. A sweep is `[find_material(name, table, w) for w in wls]`.
    """
    table = wavebit.read_material_table()
    swept = wavebit.find_material('Si', table, np.array([700.0, 900.0, 1100.0]))
    assert np.shape(np.asarray(swept.epsilon)) == (3,)

    with pytest.raises(ValueError, match=r'half-space.*single value'):
        wavebit.rcwa(backend=0, structure=np.ones((1, 1, 8)), period=[1000.0, 1000.0],
                     truncation_order=[2, 0], thickness=[150.0], wavelength=900.0, top=swept)


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


def test_the_readers_depend_on_nothing_above_the_material_table_section():
    """The nk readers are one tenant of `modeler.py`; the block modeller is the other.

    `_precision_is_double` was written among the modeller's helpers and is called by
    `read_material_table` -- so a region delete of the block machinery took it out and broke a
    public export with `NameError: name '_precision_is_double' is not defined`, eight tests in
    this file with it. It has been moved down beside the readers.

    Asserted from the source rather than by calling, because calling passes right up until the
    delete: every name the three readers reference must be defined *below* the material-table
    banner, be a module import, or be a builtin. A helper that drifts back up fails here.
    """
    import ast
    import builtins

    from wavebit._core import modeler

    source = inspect.getsource(modeler)
    tree = ast.parse(source)
    lines = source.splitlines()
    banner = next(i for i, line in enumerate(lines, 1)
                  if line.startswith('# -') and 'material table' in line)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update((a.asname or a.name.split('.')[0]) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update((a.asname or a.name) for a in node.names)

    below = {n.name: n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.lineno > banner}
    below_names = set(below) | {n.targets[0].id for n in tree.body
                                if isinstance(n, ast.Assign) and n.lineno > banner
                                and isinstance(n.targets[0], ast.Name)}

    stray = {}
    for reader in ('find_nk_index', 'find_material', 'read_material_table'):
        for node in ast.walk(below[reader]):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                if (name not in below_names and name not in imported
                        and not hasattr(builtins, name)):
                    stray.setdefault(reader, set()).add(name)
        # Every way a name is bound *inside* the function: parameters, assignment targets
        # (including tuple unpacking), `for` targets, comprehension targets, `with ... as`,
        # `except ... as`, and nested defs. Missing any of these reports a local as a stray,
        # which is how a first version of this flagged eight of `read_material_table`'s own
        # loop variables.
        local = {a.arg for a in below[reader].args.args}
        local.add('__file__')
        for node in ast.walk(below[reader]):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    local.update(t.id for t in ast.walk(target)
                                 if isinstance(t, ast.Name))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                local.update(t.id for t in ast.walk(node.target) if isinstance(t, ast.Name))
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                local.update(t.id for t in ast.walk(node.optional_vars)
                             if isinstance(t, ast.Name))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                local.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
                local.update(a.arg for a in node.args.args)
                if isinstance(node, ast.FunctionDef):
                    local.add(node.name)
        if reader in stray:
            stray[reader] -= local
            if not stray[reader]:
                del stray[reader]

    assert not stray, (
        f'these readers reach a name defined above the material-table banner: {stray}. '
        f'Everything above the banner was the block modeller and is gone; a reader that '
        f'reaches back up there is reaching for something that no longer exists.')
