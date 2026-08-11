"""The capability manifest must describe the solver that exists, not the one it described.

The manifest is read by programs outside this process, so a claim that drifts is worse than
one that was never made: the consumer keeps answering confidently with a fact the solver
stopped honouring, and nothing raises anywhere. These tests are the only thing standing
between that and a published document.

**They are deliberately not mirror tests.** :mod:`fieldwave.manifest` reads
``_core/engine.py``'s refusal tables at import, so asserting that the two agree would be
asserting that a variable equals itself. What is checked instead is the manifest against
*runtime behaviour*: build the ucell each spelling describes and confirm the solver assigns
the path claimed, then ask for each option the manifest says is unavailable and confirm the
build refuses it. That closes the loop the tables sit in the middle of.
"""
import json

import numpy as np
import pytest

import fieldwave
from fieldwave.manifest import LATTICE, MATERIALS, SPELLINGS, cited_sources, manifest
from fieldwave._core.enums import Backend, Factorization, FourierCoefficients, Pol

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

KW = dict(backend=0, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
          fto=[4, 4], period=[1.0, 1.0], wavelength=1.1, thickness=[0.3])

#: A hexagonal basis, for the two refusals a skewed cell carries.
HEX = np.column_stack((np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])))

_N = 32
_axis = (np.arange(_N) + 0.5) / _N - 0.5
_X, _Y = np.meshgrid(_axis, _axis)
_RASTER = np.ones((1, _N, _N))
_RASTER[0, _X ** 2 + _Y ** 2 <= 0.3 ** 2] = 2.5

#: One structure per spelling in the manifest, written the way the manifest says to write it.
#: The point is that these are transcribed *from* the manifest's `ucell` field — if that
#: description stops matching what the setter accepts, this is where it shows.
UCELLS = {
    'raster': _RASTER,
    'block': [[1.0, [['ellipse', 0.5, 0.5, 0.6, 0.6, 2.5, 0.0, 24, 24]]]],
    'smooth': [{'n_base': 1.0, 'combine': 'add',
                'shapes': [{'type': 'circle', 'cx': 0.5, 'cy': 0.5,
                            'lx': 0.6, 'n_index': 2.5}]}],
    'boundary': [{'n_base': 1.0, 'combine': 'paint', 'shapes': [
        {'type': 'rectangle', 'cx': 0.40, 'cy': 0.40, 'lx': 0.34, 'ly': 0.34,
         'angle': 0.0, 'n_index': 2.0},
        {'type': 'rectangle', 'cx': 0.56, 'cy': 0.56, 'lx': 0.34, 'ly': 0.34,
         'angle': 0.3, 'n_index': 3.0},
    ]}],
}

PATHS = {entry['path']: entry for entry in manifest()['paths']}


# ------------------------------------------------------- the spellings select the paths

@pytest.mark.parametrize('spelling', SPELLINGS, ids=[s['path'] for s in SPELLINGS])
def test_each_spelling_selects_the_modeling_type_it_claims(spelling):
    """`ucell`'s *type* is the path selector — there is no keyword to check it against.

    So the manifest's prose description of how to spell a structure is load-bearing in a way
    a keyword's would not be: get it wrong and a consumer sends users down a different path
    than the one whose trade-offs it just explained.
    """
    solver = fieldwave.rcwa(ucell=UCELLS[spelling['path']], **KW)
    assert solver.modeling_type_assigned == spelling['modeling_type']


def test_the_manifest_covers_every_spelling_the_setter_accepts():
    """A path the solver can assign but the manifest omits is a path nobody is told about."""
    assigned = {fieldwave.rcwa(ucell=u, **KW).modeling_type_assigned for u in UCELLS.values()}
    assert assigned == {s['modeling_type'] for s in SPELLINGS}


# ------------------------------------------------------- unavailable options are refused

def _unavailable(path, axis, enum_cls):
    """The values of one axis the manifest says this path does not implement."""
    return [m.value for m in enum_cls if m.value not in PATHS[path][axis]]


@pytest.mark.parametrize('path', sorted(UCELLS))
def test_a_factorization_the_manifest_withholds_is_refused(path):
    for value in _unavailable(path, 'factorization', Factorization):
        with pytest.raises(Exception) as excinfo:
            fieldwave.rcwa(ucell=UCELLS[path], factorization=value, **KW).conv_solve()
        assert value in str(excinfo.value), (
            f'{path}: refused factorization={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(UCELLS))
def test_a_series_the_manifest_withholds_is_refused(path):
    for value in _unavailable(path, 'fourier_coefficients', FourierCoefficients):
        with pytest.raises(Exception) as excinfo:
            fieldwave.rcwa(ucell=UCELLS[path], fourier_coefficients=value,
                           **KW).conv_solve()
        assert value in str(excinfo.value), (
            f'{path}: refused fourier_coefficients={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(UCELLS))
def test_an_option_the_manifest_offers_actually_builds(path):
    """The other half — a path must not be described as implementing what it refuses."""
    for value in PATHS[path]['factorization']:
        fieldwave.rcwa(ucell=UCELLS[path], factorization=value, **KW).conv_solve()
    for value in PATHS[path]['fourier_coefficients']:
        fieldwave.rcwa(ucell=UCELLS[path], fourier_coefficients=value, **KW).conv_solve()


# ------------------------------------------------------- the lattice refusals

def test_the_block_path_is_refused_on_a_skewed_cell():
    with pytest.raises(NotImplementedError, match='block modeler'):
        fieldwave.rcwa(ucell=UCELLS['block'], lattice=HEX, factorization='nvf',
                       **KW).conv_solve()


def test_li_is_refused_on_a_skewed_cell():
    with pytest.raises(NotImplementedError, match="factorization='nvf'"):
        fieldwave.rcwa(ucell=UCELLS['raster'], lattice=HEX, factorization='li',
                       **KW).conv_solve()


def test_the_manifest_lists_both_skewed_refusals():
    """Guards against one being dropped from the document while the solver still enforces it."""
    listed = ' '.join(r['what'] for r in LATTICE['refused_on_oblique'])
    assert 'block path' in listed
    assert "factorization='li'" in listed


def test_the_named_and_described_skewed_refusals_agree():
    """The constraint is stated twice — as prose to read and as names to test membership on.

    Two spellings of one fact drift, and the machine-readable one is the one a consumer acts
    on, so a mismatch would mean the document reads correctly and behaves wrongly.
    """
    named = (set(LATTICE['paths_refused_on_oblique'])
             | set(LATTICE['factorizations_refused_on_oblique']))
    described = ' '.join(r['what'] for r in LATTICE['refused_on_oblique'])
    for name in named:
        assert name in described, f'{name!r} is refused but not described'
    assert len(named) == len(LATTICE['refused_on_oblique'])


def test_every_named_skewed_refusal_is_a_thing_that_exists():
    """A refused path or rule that is not in the catalogue is a typo nobody would notice."""
    paths = {s['path'] for s in SPELLINGS}
    assert set(LATTICE['paths_refused_on_oblique']) <= paths
    assert set(LATTICE['factorizations_refused_on_oblique']) <= {
        m.value for m in Factorization}


def test_a_skewed_cell_still_builds_on_the_paths_the_manifest_leaves_open():
    for path in ('raster', 'smooth', 'boundary'):
        fieldwave.rcwa(ucell=UCELLS[path], lattice=HEX, factorization='nvf',
                       **KW).conv_solve()


# ------------------------------------------------------- the document itself

def test_manifest_is_json_round_trippable():
    """A consumer receives this over the wire; anything unserialisable is a runtime failure."""
    assert json.loads(json.dumps(manifest())) == json.loads(json.dumps(manifest()))


def test_enum_lists_match_the_enums():
    enums = manifest()['enums']
    assert enums['backend'] == [m.name.lower() for m in Backend]
    assert enums['pol'] == [m.name.lower() for m in Pol]
    assert enums['factorization'] == [m.value for m in Factorization]
    assert enums['fourier_coefficients'] == [m.value for m in FourierCoefficients]


def test_every_cited_source_exists():
    """Each unsupported case names the file that would have to change for it to become false.

    The citation is the manifest's only defence against confident staleness, so a renamed or
    deleted path has to fail here — not because a missing file is itself a defect, but
    because the claim it was anchoring has quietly become unanchored.
    """
    for cited in sorted(cited_sources()):
        assert (REPO / cited).exists(), f'cited source does not exist: {cited}'


def test_every_material_case_states_a_reason():
    for entry in MATERIALS:
        assert entry['why'], f"{entry['case']}: no reason given"
        assert isinstance(entry['supported'], bool)
