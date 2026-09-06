"""``docs/fdtd.md`` §2's tables, read out of the markdown and held against a golden.

**The gate this repository did not have.** ``tests/test_docs_contract.py`` checks that the
documents' paths, symbols, URLs and toctree entries are real. Nothing anywhere read a
*number* out of a document -- so §2's three-way table drifted for as long as it took
somebody to notice, which was three weeks. What drifted was one word's worth of
configuration: the wavebit column was taken at ``pad_scale=2.0``, ``PAD_WAVELENGTHS`` was
then halved, the sentence "at the shipped defaults" stayed true-looking, and the printed
digits became a configuration the library no longer had. Measured 2026-09-07:
``pad_scale=2.0`` reproduces that old column to the last printed digit in all eight
entries, ``Σ`` included.

So this module does three things, and the order is the argument:

1. **reads the markdown** -- both of §2's tables, cell by cell, and the four headline gaps
   quoted in §2's TL;DR and in ``README.md`` -- and compares every value to
   ``golden_fdtd_table.npz``. A digit edited in a document is now a failing test;
2. **re-solves the two columns this environment can** and compares them to the same
   artefact, so the artefact cannot drift away from the code either;
3. **asserts the recorded conditions are still the shipped ones**
   (``test_the_recorded_conditions_are_the_ones_the_module_ships``). That is the one that
   would have caught the original defect on the day it happened, and it is the reason
   ``_meep_1d_cases.py`` writes every determinant out **by value** instead of saying "the
   defaults": a record that names the defaults is true by construction and can be checked
   against nothing.

**Bounds are transcription bounds, not physics ones.** Both solvers are deterministic --
measured bit-identical across processes -- so a printed value is either the value or it is
not, and the only honest tolerance is half a unit in the last printed place. Read that off
the printed string itself (:func:`_half_ulp`) rather than fixing a constant, so a cell
printed to two significant figures is held to two and one printed to six is held to six.
**None of these tolerances may be loosened to accommodate a solver change**: a change that
moves these digits is a table that needs re-typing, which is a two-minute job with
``gen_goldens_fdtd_table.py`` and is the whole point.

**What is deliberately not asserted**: the ``wall`` column. It is a property of the
machine rather than of the method, it is frozen in the artefact so a reader knows which
machine, and asserting it in CI would be asserting the hardware.

The Meep column is *copied* into the artefact from ``golden_meep_1d.npz`` rather than
recomputed -- meep is GPL and lives in another environment, and that file is its own
oracle with its own closed-form calibration gate. ``test_the_meep_column_is_the_committed
_meep_golden`` holds the copy against the original, digest included.
"""
import pathlib
import re
import sys

import numpy as np
import pytest

from wavebit._core import fdtd as fdtd_module

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _meep_1d_cases import (CASES, FDTD_CONVERGENCE, FDTD_PUBLISHED, LEGS,  # noqa: E402
                            RCWA_ORDERS, WAVEBIT_CALL, WAVEBIT_DERIVED, WAVEBIT_SHIPPED)
from gen_goldens_fdtd_table import (_fdtd, _grid, _rcwa, _shipped,          # noqa: E402
                                    CASE_NAME)

ROOT = HERE.parent.parent
DOC = ROOT / 'docs' / 'fdtd.md'
README = ROOT / 'README.md'
TABLE_PATH = HERE / 'golden_fdtd_table.npz'
MEEP_PATH = HERE / 'golden_meep_1d.npz'

#: A tie in the last printed place is a rounding coin-flip, not a disagreement; this makes
#: ``<=`` mean ``<=`` in floating point.
SLACK = 1.0 + 1e-9

#: How closely a re-solve has to reproduce the frozen column. Both paths are deterministic
#: on this machine, so in principle this could be zero -- it is 1e-9 because a compiler,
#: a BLAS or a numpy release may move the last bits of an eigen-decomposition, and a
#: disagreement that small could not change a printed digit anyway. Anything a reader
#: could see is nine orders larger.
RESOLVE_TOL = 1e-9


def _half_ulp(printed):
    """Half a unit in the last place the string actually prints.

    ``0.204898`` -> 5e-7, ``6.6e-4`` -> 5e-6, ``2.10`` -> 5e-3. Derived from the string
    rather than fixed, because §2's cells are printed to anything from two significant
    figures to six decimals and holding them all to one constant would either be a lie in
    one direction or unfalsifiable in the other.
    """
    s = printed.strip()
    mantissa, _, exponent = s.partition('e')
    decimals = len(mantissa.partition('.')[2])
    return 0.5 * 10.0 ** (-decimals + (int(exponent) if exponent else 0))


def _tables(text):
    """Every markdown pipe table in ``text``, as ``(header cells, [row cells])``."""
    out, rows = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            rows.append([c.strip() for c in stripped.strip('|').split('|')])
        elif rows:
            if len(rows) > 2:
                out.append((rows[0], rows[2:]))    # rows[1] is the --- separator
            rows = []
    if len(rows) > 2:
        out.append((rows[0], rows[2:]))
    return out


def _table_with(text, *needles):
    """The one table whose header contains all of ``needles``."""
    found = [t for t in _tables(text) if all(any(n in c for c in t[0]) for n in needles)]
    assert len(found) == 1, (
        f'expected exactly one table whose header mentions {needles}, found {len(found)}; '
        f'docs/fdtd.md has been restructured and this gate cannot find the table it holds')
    return found[0]


def _numbers_after(text, anchor, count):
    """The first ``count`` numbers of the form ``1.2e-3`` following ``anchor``."""
    flat = ' '.join(text.split())
    at = flat.find(anchor)
    assert at >= 0, f'{anchor!r} is no longer in the text; this claim has been reworded'
    found = re.findall(r'\d+\.\d+e-\d+', flat[at:at + 400])
    assert len(found) >= count, (
        f'expected {count} numbers after {anchor!r}, found {found}')
    return found[:count]


@pytest.fixture(scope='module')
def table():
    if not TABLE_PATH.exists():
        pytest.skip(f'{TABLE_PATH.name} missing; regenerate with gen_goldens_fdtd_table.py')
    return np.load(TABLE_PATH)


@pytest.fixture(scope='module')
def doc():
    return DOC.read_text()


@pytest.fixture(scope='module')
def solutions():
    """The two columns this environment can re-derive, solved once.

    Module-scoped because three tests read them and the FDTD half is the expensive part
    (~8 s a leg). Sharing the solve is also what keeps those tests from drifting into
    different configurations, which is the failure mode one level up from the one this
    file exists for.
    """
    case = CASES[CASE_NAME]
    out = {}
    for leg in LEGS:
        out[('rcwa', leg)] = _grid(_rcwa(case, leg, RCWA_ORDERS), case)
        out[('fdtd', leg)] = _grid(_fdtd(case, leg, FDTD_PUBLISHED), case)
    return out


def _row_key(label):
    """``TE `R0``` -> ``('te', 'R', 0)``; ``TM `T±1``` -> ``('tm', 'T', 1)``."""
    leg = 'te' if label.upper().startswith('TE') else 'tm'
    quantity = 'R' if '`R' in label else 'T'
    absm = 1 if '1' in label.split('`')[1][1:] else 0
    return leg, quantity, absm


# ---------------------------------------------------------------------------------------
# 1. the document, read back
# ---------------------------------------------------------------------------------------

def test_the_document_prints_the_frozen_three_way_table(table, doc):
    """Every cell of §2's three-way table, against ``golden_fdtd_table.npz``.

    **This is the test whose absence is the whole finding.** The table had been printing a
    ``pad_scale=2.0`` wavebit column for three weeks; nothing could contradict it, because
    nothing read it. Six columns are held here: the RCWA reference, wavebit's answer and
    its gap, Meep's answer and its gap, and the ``Σ`` row -- each to half a unit in its own
    last printed place.

    The ``T±1`` rows carry a *pair* where the two first orders differ (Meep's TM row), and
    a single value where the claim is that they came out equal; both spellings are read
    and both are held against the frozen pair.

    **The rows are checked by name, not by count.** A cell-count threshold is the shape a
    gate takes when it cannot fail: this table's rows are worth 5 cells or 10, so a
    threshold low enough to admit the 46 it has admits the 41 left after a single-order row
    is deleted -- and deleting a row is the likeliest edit anyone makes to a table. It is
    the set of ``(leg, quantity, |m|)`` keys that is asserted, so a dropped row fails
    saying *which*, and the count is asserted exactly beside it to catch a ``T±1`` cell
    that loses its ``/`` pair without losing its row.
    """
    header, rows = _table_with(doc, 'RCWA (ref)', 'Meep')
    orders = list(np.asarray(table['orders_u']).ravel())
    column = {'RCWA (ref)': 'rcwa', 'this module': 'fdtd', 'Meep': 'meep'}
    #: Which rows have to be *there*. A count is not enough: this table's rows are worth 5
    #: cells (a single order) or 10 (a `T±1` pair), so a threshold that a dropped pair trips
    #: is one a dropped single row clears. Measured 2026-09-07 -- the full table is 46 cells
    #: and dropping `TE T0` leaves 41, so the `>= 40` this replaces passed the likeliest
    #: edit there is. Naming the rows makes a deletion say which row went.
    want_rows = {(leg, quantity, absm) for leg in LEGS
                 for quantity, absm in (('R', 0), ('T', 0), ('T', 1))}
    seen, seen_rows, seen_sigma = 0, set(), 0

    for row in rows:
        label = row[0]
        if label.startswith('Σ'):
            seen_sigma += 1
            for name, key in (('this module', 'fdtd'), ('Meep', 'meep')):
                cell = row[[i for i, h in enumerate(header) if name in h][0]]
                for leg, printed in zip(('te', 'tm'), cell.split('/')):
                    got = float(table[f'{key}__{leg}__{FDTD_PUBLISHED}__sigma']
                                if key == 'fdtd' else table[f'{key}__{leg}__sigma'])
                    assert abs(got - float(printed)) <= _half_ulp(printed) * SLACK, (
                        f'Sigma, {name}, {leg}: the document prints {printed.strip()} and '
                        f'the golden holds {got:.8f}')
                    seen += 1
            # The RCWA cell prints a bare "1", whose last printed place is a whole unit --
            # useless as a bound. It is a claim that the reference conserves energy to the
            # digit, so it is held at round-off instead of at half a ULP.
            i_rcwa = [i for i, h in enumerate(header) if 'RCWA' in h][0]
            assert row[i_rcwa].strip() == '1'
            for leg in LEGS:
                assert abs(float(table[f'rcwa__{leg}__sigma']) - 1.0) < 1e-9
                seen += 1
            continue

        leg, quantity, absm = _row_key(label)
        seen_rows.add((leg, quantity, absm))
        i = [orders.index(m) for m in ((0,) if absm == 0 else (-absm, absm))]
        for name, key in column.items():
            head = [j for j, h in enumerate(header) if name in h][0]
            cell = row[head]
            frozen = sorted(float(np.asarray(
                table[f'{key}__{leg}__{FDTD_PUBLISHED}__{quantity}'] if key == 'fdtd'
                else table[f'{key}__{leg}__{quantity}']).ravel()[j]) for j in i)
            printed = [p for p in cell.split('/')]
            if len(printed) == 1:
                printed = printed * len(frozen)
            for a, b in zip(frozen, sorted(printed, key=float)):
                assert abs(a - float(b)) <= _half_ulp(b) * SLACK, (
                    f'{label}, {name}: the document prints {b.strip()} and the golden '
                    f'holds {a:.8f}')
                seen += 1
            if key == 'rcwa':
                continue
            gap_cell = row[head + 1]
            gaps = sorted(float(np.asarray(table[f'gap__{key}__{leg}__{quantity}'])
                                .ravel()[j]) for j in i)
            printed_gaps = gap_cell.split('/')
            if len(printed_gaps) == 1:
                printed_gaps = printed_gaps * len(gaps)
            for a, b in zip(gaps, sorted(printed_gaps, key=float)):
                assert abs(a - float(b)) <= _half_ulp(b) * SLACK, (
                    f'{label}, {name} gap: the document prints {b.strip()} and the '
                    f'golden holds {a:.3e}')
                seen += 1
    assert seen_rows == want_rows, (
        'docs/fdtd.md §2\'s three-way table no longer has the rows this gate holds.\n'
        f'  missing: {sorted(want_rows - seen_rows)}\n'
        f'  unexpected: {sorted(seen_rows - want_rows)}\n'
        'A row deleted from the table is a claim deleted from the record. Restore it, or '
        're-run tests/references/gen_goldens_fdtd_table.py and update this expectation '
        'deliberately.')
    assert seen_sigma == 1, f'expected exactly one Sigma row, read {seen_sigma}'
    assert seen == 46, (
        f'{seen} cells were checked and this table is 46; its shape has changed without a '
        f'row going missing -- a `T±1` cell has probably lost or gained its `/` pair.')


def test_the_document_prints_the_frozen_convergence_table(table, doc):
    """§2's convergence table, which shares its 5 nm row with the table above.

    The two tables were taken on either side of the ``PAD_WAVELENGTHS`` change and only
    one of them was re-derived, so §2 disagreed with itself by 8e-5 in one section. They
    now come out of one artefact and one configuration, and the shared row is what makes
    that checkable rather than merely claimed.

    The ``wall`` column is deliberately not asserted: it is a property of the machine.
    """
    header, rows = _table_with(doc, 'cell', 'order', 'wall')
    i_cell = 0
    checked = set()
    for row in rows:
        res = int(round(1000.0 / float(row[i_cell].split()[0])))
        assert res in FDTD_CONVERGENCE, f'{row[i_cell]} is not a recorded resolution'
        checked.add(res)
        for leg, i_worst, i_order in (('te', 1, 2), ('tm', 3, 4)):
            printed = row[i_worst]
            got = float(table[f'fdtd__{leg}__{res}__worst'])
            assert abs(got - float(printed)) <= _half_ulp(printed) * SLACK, (
                f'{row[i_cell]} {leg}: the document prints {printed} and the golden '
                f'holds {got:.3e}')
            if row[i_order].strip():
                key = f'fdtd__{leg}__{res}__order'
                assert key in table.files, f'{row[i_cell]} {leg} prints an order and the '\
                                          f'golden has none (it is the first row)'
                assert abs(float(table[key]) - float(row[i_order])) \
                    <= _half_ulp(row[i_order]) * SLACK, (
                    f'{row[i_cell]} {leg}: the document prints order {row[i_order]} and '
                    f'the golden holds {float(table[key]):.4f}')
        for leg, printed in zip(('te', 'tm'), row[5].split('/')):
            got = float(table[f'fdtd__{leg}__{res}__sigma'])
            assert abs(got - float(printed)) <= _half_ulp(printed) * SLACK, (
                f'{row[i_cell]} {leg} Sigma: the document prints {printed.strip()} and '
                f'the golden holds {got:.8f}')
    assert checked == set(FDTD_CONVERGENCE), (
        f'the convergence table covers {sorted(checked)} and the record declares '
        f'{sorted(FDTD_CONVERGENCE)}')


def test_the_headline_gaps_quoted_in_prose_are_the_frozen_ones(table, doc):
    """§2's TL;DR and ``README.md`` both quote four numbers; here they are.

    A table can be right while the sentence above it is wrong, and the sentence is what
    most readers take away -- ``README.md``'s is the FDTD path's whole public claim. Both
    are transcriptions of the same four ``worst`` entries in the artefact and are held to
    the last figure they print.
    """
    want = [float(table[f'fdtd__{leg}__{FDTD_PUBLISHED}__worst']) for leg in ('te', 'tm')]
    want += [float(table[f'meep__{leg}__worst']) for leg in ('te', 'tm')]
    for name, text in (('docs/fdtd.md', doc), ('README.md', README.read_text())):
        printed = _numbers_after(text, 'worst per-order gap', 4)
        for a, b in zip(want, printed):
            assert abs(a - float(b)) <= _half_ulp(b) * SLACK, (
                f'{name} quotes {b} where the golden holds {a:.3e}')
    # "Refine once more and it reaches ..." -- the 2.5 nm row, quoted in the same breath.
    for a, b in zip([float(table[f'fdtd__{leg}__400__worst']) for leg in ('te', 'tm')],
                    _numbers_after(doc, 'Refine once', 2)):
        assert abs(a - float(b)) <= _half_ulp(b) * SLACK


# ---------------------------------------------------------------------------------------
# 2. the artefact, against the code
# ---------------------------------------------------------------------------------------

def test_the_recorded_conditions_are_the_ones_the_module_ships():
    """**The gate.** Every determinant in the record, against the live module.

    ``_meep_1d_cases.py::WAVEBIT_SHIPPED`` writes out, by value, everything §2's one-line
    ``solve_fdtd`` call inherits rather than passes -- the clearance, the absorber depth,
    the run length, the Courant number, the smoothing, and every PML constant. If any of
    them moves, this fails and names the table, on the commit that moved it. That is the
    difference between a record and the phrase "at the shipped defaults", which was what
    §2 had: the phrase stayed true when ``PAD_WAVELENGTHS`` was halved, and the table did
    not.

    Both spellings of the three doubled constants are checked, because they do not bind
    the same way: ``PML_CELLS`` and ``COURANT`` are bound into ``solve_fdtd``'s signature
    at import, so editing the module attribute is inert and only the signature default is
    what a call gets, while ``PAD_WAVELENGTHS`` and ``N_PERIODS`` are read at call time.
    A record naming one spelling of each would miss half the ways this can move.
    """
    wrong = [f'{name}: recorded {recorded!r}, module ships {_shipped(name)!r}'
             for name, recorded in WAVEBIT_SHIPPED.items()
             if _shipped(name) != recorded]
    assert not wrong, (
        'the conditions docs/fdtd.md §2 is quoted at are no longer the shipped ones:\n  '
        + '\n  '.join(wrong)
        + '\nRe-run tests/references/gen_goldens_fdtd_table.py and re-type §2, or put '
          'the default back. Do not edit the record to match -- the record is what the '
          'published digits mean.')


def test_the_recorded_grid_is_the_one_the_solver_builds():
    """The derived half of the record: lengths and counts, against the run's diagnostics.

    Inputs alone are not a record here, because two of them are indistinguishable:
    ``pad_scale=2.0`` and ``PAD_WAVELENGTHS=1.0`` are bit-identical and both give a 0.90
    clearance. So the record also states the clearance, the absorber and the run in
    **lengths and steps**, and this is what makes those statements checkable. It also
    catches a change that moves the grid without touching any constant the test above
    reads -- a rounding rule, say.
    """
    case = CASES[CASE_NAME]
    _, diagnostics = _fdtd(case, 'te', FDTD_PUBLISHED, diagnostics=True)
    for key in ('nx', 'nz', 'dt', 'n_steps', 'dz'):
        got = float(np.asarray(diagnostics[key]).ravel()[0])
        want = float(WAVEBIT_DERIVED[key])
        assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (
            f'the record says {key} = {want} and the solver built {got}')
    pml = np.asarray(diagnostics['pml_thickness'], dtype=float)
    assert np.abs(pml - WAVEBIT_DERIVED['pml_thickness']).max() < 1e-9
    # The clearances, which is the pair the two indistinguishable spellings collapse to.
    wl, n_top, n_bot = case['wavelength'], case['n_top'], case['n_bot']
    pad = _shipped('PAD_WAVELENGTHS') * _shipped('pad_scale')
    assert abs(pad * wl / n_top - WAVEBIT_DERIVED['clearance_top']) < 1e-12
    assert abs(pad * wl / n_bot - WAVEBIT_DERIVED['clearance_bot']) < 1e-12


def test_the_frozen_wavebit_column_is_still_what_the_solver_returns(table, solutions):
    """§2's "this module, 5 nm" column, re-solved through ``WAVEBIT_CALL``.

    The call passes a resolution and nothing else, on purpose: a table taken at settings
    no user types is a table about nothing, and it is exactly a *named* ``pad_scale`` in a
    private script that produced the stale column. What makes the short call reproducible
    is the record, not the shortness.
    """
    for leg in LEGS:
        r, t = solutions[('fdtd', leg)]
        for key, got in (('R', r), ('T', t)):
            want = np.asarray(table[f'fdtd__{leg}__{FDTD_PUBLISHED}__{key}'])
            assert np.abs(got - want).max() < RESOLVE_TOL, (
                f'{leg} {key}: solve_fdtd no longer returns the frozen column '
                f'({np.abs(got - want).max():.2e})')


def test_the_frozen_rcwa_column_is_still_what_the_solver_returns(table, solutions):
    """The reference column, which both published gaps are measured against.

    Free -- one 1D solve -- and it is what makes the *table* the object under test rather
    than two thirds of it. A drift here silently rewrites both gaps.
    """
    for leg in LEGS:
        r, t = solutions[('rcwa', leg)]
        for key, got in (('R', r), ('T', t)):
            want = np.asarray(table[f'rcwa__{leg}__{key}'])
            assert np.abs(got - want).max() < RESOLVE_TOL, f'{leg} {key}'


def test_the_reference_is_converged_enough_to_carry_the_gaps(table):
    """The reference's own uncertainty, frozen beside it, against the gaps it bounds.

    A gap quoted against a reference is only as good as the reference, and this is the
    one number in the table that has no column of its own. Measured against
    ``truncation_order = 400``: 1.21e-6 (TE) and 1.30e-5 (TM). The **smallest** published
    gap has to clear it by a real margin or the table is quoting noise: measured 64x on TE
    (7.7e-5) and 18x on TM (2.4e-4), which is the claim §2 makes in prose.

    The identically-zero entries are dropped rather than counted: ``R[+-1]`` is zero in
    both columns because those orders are evanescent in air, so their "gap" is an exact
    zero that carries no information and would drive this to nonsense.
    """
    for leg in LEGS:
        unc = float(table[f'rcwa__{leg}__uncertainty'])
        gaps = np.concatenate([np.abs(np.asarray(table[f'gap__{col}__{leg}__{key}']).ravel())
                               for col in ('fdtd', 'meep') for key in ('R', 'T')])
        smallest = float(gaps[gaps > 0.0].min())
        assert smallest > 10.0 * unc, (
            f'{leg}: the reference carries {unc:.2e} and the table publishes gaps down '
            f'to {smallest:.2e}')


def test_the_meep_column_is_the_committed_meep_golden(table):
    """The third column is a *copy*, and this is what says the copy is current.

    Meep is GPL and runs in another environment, so its column cannot be re-solved here;
    it is copied into this artefact from ``golden_meep_1d.npz``, which carries its own
    closed-form calibration gate. Two committed artefacts holding the same numbers can
    come apart, so the copy carries the source's digest and this checks both.
    """
    if not MEEP_PATH.exists():
        pytest.skip('golden_meep_1d.npz missing')
    import hashlib
    assert str(table['meep_golden_sha256']) == \
        hashlib.sha256(MEEP_PATH.read_bytes()).hexdigest(), (
        'golden_meep_1d.npz has been regenerated since golden_fdtd_table.npz copied its '
        'column; re-run gen_goldens_fdtd_table.py')
    meep = np.load(MEEP_PATH)
    orders = list(np.asarray(table['orders_u']).ravel())
    frozen = list(np.asarray(meep[f'{CASE_NAME}__orders_u']).ravel())
    i = [frozen.index(m) for m in orders]
    for leg in LEGS:
        for key in ('R', 'T'):
            got = np.asarray(meep[f'{CASE_NAME}__{leg}__fine__{key}']).ravel()[i]
            assert np.abs(got - np.asarray(table[f'meep__{leg}__{key}'])).max() == 0.0


def test_the_artefact_carries_the_conditions_beside_the_numbers(table):
    """A frozen number without the configuration it was taken at is not reproducible.

    That sentence is the whole finding, so it is asserted of the artefact rather than left
    to the generator: every recorded determinant, the derived grid, the call, the build
    and the unit travel inside the ``.npz`` with the digits they produced.
    """
    for name in WAVEBIT_SHIPPED:
        assert f'record__shipped__{name}' in table.files, name
    for name in WAVEBIT_DERIVED:
        assert f'record__derived__{name}' in table.files, name
    for name in WAVEBIT_CALL:
        assert f'record__call__{name}' in table.files, name
    assert str(table['units']) == 'micrometre'
    assert int(table['fdtd_resolution']) == FDTD_PUBLISHED
    assert int(table['rcwa_orders']) == RCWA_ORDERS
    assert str(table['wavebit_build__numpy'])


def test_the_shipped_lookup_reads_both_spellings():
    """``_shipped`` has to find a constant wherever it actually lives, and it is not
    one place.

    ``n_pml`` and ``courant`` are lowercase signature defaults; ``PML_CELLS`` and
    ``COURANT`` are the module attributes those defaults were bound from at import.
    Patching the attribute does not change what a call gets, which is a real trap for
    anyone re-deriving the sensitivities by monkeypatching -- they would report a live
    knob as inert. If this helper silently fell back to the module for the lowercase
    names, the gate above would be reading the wrong thing and would pass while the
    shipped call had changed.
    """
    import inspect
    params = inspect.signature(fdtd_module.solve_fdtd).parameters
    assert _shipped('n_pml') == params['n_pml'].default
    assert _shipped('PML_CELLS') == fdtd_module.PML_CELLS
    assert _shipped('PAD_WAVELENGTHS') == fdtd_module.PAD_WAVELENGTHS
    assert 'PAD_WAVELENGTHS' not in params
