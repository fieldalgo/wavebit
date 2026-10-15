"""The capability manifest must describe the solver that exists, not the one it described.

The manifest is read by programs outside this process, so a claim that drifts is worse than
one that was never made: the consumer keeps answering confidently with a fact the solver
stopped honouring, and nothing raises anywhere. These tests are the only thing standing
between that and a published document.

**They are deliberately not mirror tests.** :mod:`wavebit.manifest` reads
``src/wavebit/rcwa/_engine/engine.py``'s refusal tables at import, so asserting that the two agree would be
asserting that a variable equals itself. What is checked instead is the manifest against
*runtime behaviour*: build the structure each spelling describes and confirm the solver assigns
the path claimed, then ask for each option the manifest says is unavailable and confirm the
build refuses it. That closes the loop the tables sit in the middle of.
"""

from wavebit.compat import legacy as _legacy_wavebit

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_root
import json

import numpy as np
import pytest

import wavebit
from wavebit.manifest import LATTICE, MATERIAL_QUANTITY, MATERIALS, SPELLINGS, cited_sources, manifest
from wavebit._common.enums import Backend, Tetm; from wavebit.rcwa._engine.enums import Factorization, FourierCoefficients, ModelingPath, Nesting, NormalMethod, Projector


REPO = repository_root()

KW = dict(backend='numpy', top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
          truncation_order=[4, 4], period=[1.0, 1.0], wavelength=1.1, thickness=[0.3])

#: `KW` without its cell. `period` and `lattice` are two spellings of one cell and naming
#: both is refused, so the calls below that give a basis take the scene without the period.
KW_NO_CELL = {k: v for k, v in KW.items() if k != 'period'}

#: A hexagonal basis, for the two refusals a skewed cell carries.
HEX = np.column_stack((np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])))

_N = 32
_X, _Y = wavebit.Lattice.rectangular(1.0).meshgrid_uv((_N, _N))
_RASTER = np.ones((1, _N, _N))
_RASTER[0, _X ** 2 + _Y ** 2 <= 0.3 ** 2] = 6.25          # permittivity: 2.5**2

#: One structure per spelling in the manifest, written the way the manifest says to write it.
#: The point is that these are transcribed *from* the manifest's `structure` field — if that
#: description stops matching what the setter accepts, this is where it shows. Every material
#: number here is a relative permittivity, which is the other half of that description and
#: what `material_quantity` publishes.
STRUCTURES = {
    'raster': _RASTER,
    'parametric': [{'background': 1.0,
                    'shapes': [{'type': 'circle', 'cx': 0.0, 'cy': 0.0,
                                'lx': 0.6, 'material': 6.25}]}],
}

#: The same parametric spelling with its shapes overlapping. It is not a third entry in
#: `STRUCTURES` because it is not a third path -- it selects the same one, and what differs is
#: what the layer does about the overlap. `test_an_overlap_is_a_property_of_one_path_not
#: _another_path` is what holds that.
_OVERLAPPING = [{'background': 1.0, 'shapes': [
    {'type': 'rectangle', 'cx': -0.10, 'cy': -0.10, 'lx': 0.34, 'ly': 0.34,
     'angle': 0.0, 'material': 4.0},
    {'type': 'rectangle', 'cx': 0.06, 'cy': 0.06, 'lx': 0.34, 'ly': 0.34,
     'angle': 0.3, 'material': 9.0},
]}]

PATHS = {entry['path']: entry for entry in manifest()['paths']}


# ------------------------------------------------------- the spellings select the paths

@pytest.mark.parametrize('spelling', SPELLINGS, ids=[s['path'] for s in SPELLINGS])
def test_each_spelling_selects_the_modeling_path_it_claims(spelling):
    """`structure`'s *type* is the path selector — there is no keyword to check it against.

    So the manifest's prose description of how to spell a structure is load-bearing in a way
    a keyword's would not be: get it wrong and a consumer sends users down a different path
    than the one whose trade-offs it just explained.
    """
    solver = _legacy_wavebit.rcwa(structure=STRUCTURES[spelling['path']], **KW)
    assert solver.modeling_path == ModelingPath(spelling['path'])


def test_the_manifest_covers_every_spelling_the_setter_accepts():
    """A path the solver can assign but the manifest omits is a path nobody is told about."""
    assigned = {_legacy_wavebit.rcwa(structure=u, **KW).modeling_path for u in STRUCTURES.values()}
    assert assigned == {ModelingPath(s['path']) for s in SPELLINGS}


# ------------------------------------------------------- an overlap is a property, not a path

def test_an_overlap_selects_the_same_path_rather_than_another_one():
    """The whole of the '9' reshape, as one assertion.

    `smooth` and `boundary` were published as sibling paths until 2026-09-03, which said that
    a `structure` picks between them. It does not: the spelling is the same and so is the
    path. What differs is what a layer does about an overlap, which is why `overlap` is a key
    on this entry rather than an entry of its own.
    """
    disjoint = _legacy_wavebit.rcwa(structure=STRUCTURES['parametric'], **KW)
    overlapping = _legacy_wavebit.rcwa(structure=_OVERLAPPING, **KW)
    assert disjoint.modeling_path == overlapping.modeling_path == ModelingPath.PARAMETRIC


def test_the_published_overlap_property_is_the_one_the_solver_takes():
    """`overlap` says the choice is per layer and read at `geometry.boundary_plans`. Both halves.

    A solver-wide answer cannot express a structure whose first layer resolves and whose
    second does not, so the per-layer claim is the one that has to be checked against a real
    mixed structure rather than against two single-layer ones.
    """
    published = PATHS['parametric']['overlap']
    assert published['allowed'] is True
    assert 'per layer' in published['decided']

    mixed = [_OVERLAPPING[0], STRUCTURES['parametric'][0]]
    solver = _legacy_wavebit.rcwa(structure=mixed, **dict(KW, thickness=[0.3, 0.3]))
    solver.solve()
    plans = solver.freeze_geometry().boundary_plans
    assert [plan is not None for plan in plans] == [True, False], (
        'the overlapping layer must resolve and the disjoint one must not; '
        f'got {plans!r}')


def test_the_raster_path_publishes_no_overlap_property():
    """`None` rather than a missing key: a consumer's membership test gets an answer.

    Same argument as `normal_methods_refused_on_oblique` being kept while empty -- the absence
    of the concept on this path is a fact about the path, not a gap in the document.
    """
    assert 'overlap' in PATHS['raster'] and PATHS['raster']['overlap'] is None


# ------------------------------------------------------- unavailable options are refused

def _unavailable(path, axis, enum_cls):
    """The values of one axis the manifest says this path does not implement."""
    return [m.value for m in enum_cls if m.value not in PATHS[path][axis]]


@pytest.mark.parametrize('path', sorted(STRUCTURES))
def test_a_factorization_the_manifest_withholds_is_refused(path):
    for value in _unavailable(path, 'factorization', Factorization):
        with pytest.raises(Exception) as excinfo:
            _legacy_wavebit.rcwa(structure=STRUCTURES[path], factorization=value, **KW).solve()
        assert value in str(excinfo.value), (
            f'{path}: refused factorization={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(STRUCTURES))
def test_a_series_the_manifest_withholds_is_refused(path):
    for value in _unavailable(path, 'fourier_coefficients', FourierCoefficients):
        with pytest.raises(Exception) as excinfo:
            _legacy_wavebit.rcwa(structure=STRUCTURES[path], fourier_coefficients=value,
                           **KW).solve()
        assert value in str(excinfo.value), (
            f'{path}: refused fourier_coefficients={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(STRUCTURES))
def test_a_normal_method_the_manifest_withholds_is_refused(path):
    """The shape paths list no raster construction; naming one there must refuse."""
    for value in _unavailable(path, 'raster_normal_method', NormalMethod):
        with pytest.raises(Exception) as excinfo:
            _legacy_wavebit.rcwa(structure=STRUCTURES[path], raster_normal_method=value, **KW).solve()
        assert value in str(excinfo.value), (
            f'{path}: refused raster_normal_method={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(STRUCTURES))
def test_a_projector_the_manifest_withholds_is_refused(path):
    """`raster_normal_method`'s twin, over the axis published beside it since 2026-09-01.

    Both raster and shape NVF now implement both published projectors.
    """
    for value in _unavailable(path, 'nvf_projector', Projector):
        with pytest.raises(Exception) as excinfo:
            _legacy_wavebit.rcwa(structure=STRUCTURES[path], nvf_projector=value, **KW).solve()
        assert value in str(excinfo.value), (
            f'{path}: refused nvf_projector={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(STRUCTURES))
def test_a_nesting_the_manifest_withholds_is_refused(path):
    """The Li-side twin of the two above. The parametric path lists no plain nesting: its
    default NVF nests no rules, and its Li build is itself a conditional option (next test), so
    naming one on the circle must refuse rather than be ignored."""
    for value in _unavailable(path, 'li_nesting', Nesting):
        with pytest.raises(Exception) as excinfo:
            _legacy_wavebit.rcwa(structure=STRUCTURES[path], li_nesting=value, **KW).solve()
        assert value in str(excinfo.value), (
            f'{path}: refused li_nesting={value!r} without naming it')


#: An axis-aligned rectangle inside the cell: what the parametric path's conditional
#: `factorization='li'` requires, so the conditional options can be built rather than read.
_RECTANGLE = [{'background': 1.0,
               'shapes': [{'type': 'rectangle', 'cx': 0.05, 'cy': -0.1,
                           'lx': 0.5, 'ly': 0.3, 'material': 6.25}]}]


def test_a_conditional_option_builds_where_its_condition_holds():
    """`conditional_options` is a promise too: each entry builds on a structure that meets its
    requirement. The nesting entries ride on the Li one, so both are built together."""
    conditional = PATHS['parametric']['conditional_options']
    assert {(c['option'], c['value']) for c in conditional} >= (
        {('factorization', 'li')} | {('li_nesting', m.value) for m in Nesting})
    for entry in conditional:
        kw = {entry['option']: entry['value']}
        if entry['option'] == 'li_nesting':
            kw['factorization'] = 'li'
        _legacy_wavebit.rcwa(structure=_RECTANGLE, **kw, **KW).solve()


def test_an_option_the_manifest_offers_builds_on_an_overlapping_layer_too():
    """The `parametric` entry's options are claimed for every layer that spelling accepts.

    An overlapping layer takes a different route inside (`_core.boundary` rather than the
    closed-form sum), and while it was published as its own path that route had its own row of
    the option battery. Collapsing the entries must not quietly drop the coverage with them.
    """
    entry = PATHS['parametric']
    for value in entry['factorization']:
        _legacy_wavebit.rcwa(structure=_OVERLAPPING, factorization=value, **KW).solve()
    for value in entry['fourier_coefficients']:
        _legacy_wavebit.rcwa(structure=_OVERLAPPING, fourier_coefficients=value, **KW).solve()
    for value in entry['nvf_projector']:
        _legacy_wavebit.rcwa(structure=_OVERLAPPING, nvf_projector=value, **KW).solve()
    for axis, enum_cls in (('factorization', Factorization),
                           ('fourier_coefficients', FourierCoefficients),
                           ('raster_normal_method', NormalMethod), ('nvf_projector', Projector),
                           ('li_nesting', Nesting)):
        for value in _unavailable('parametric', axis, enum_cls):
            with pytest.raises(Exception) as excinfo:
                _legacy_wavebit.rcwa(structure=_OVERLAPPING, **{axis: value}, **KW).solve()
            assert value in str(excinfo.value), (
                f'overlapping layer: refused {axis}={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(STRUCTURES))
def test_an_option_the_manifest_offers_actually_builds(path):
    """The other half — a path must not be described as implementing what it refuses."""
    for value in PATHS[path]['factorization']:
        _legacy_wavebit.rcwa(structure=STRUCTURES[path], factorization=value, **KW).solve()
    for value in PATHS[path]['fourier_coefficients']:
        _legacy_wavebit.rcwa(structure=STRUCTURES[path], fourier_coefficients=value, **KW).solve()
    for value in PATHS[path]['raster_normal_method']:
        # The manifest's own caveat: the axis is read only under factorization='nvf'.
        _legacy_wavebit.rcwa(structure=STRUCTURES[path], factorization='nvf',
                       raster_normal_method=value, **KW).solve()
    for value in PATHS[path]['nvf_projector']:
        # Same caveat, same reader -- the two axes are published side by side.
        _legacy_wavebit.rcwa(structure=STRUCTURES[path], factorization='nvf',
                       nvf_projector=value, **KW).solve()
    for value in PATHS[path]['li_nesting']:
        # The other rule's axis, and the other caveat: read only under 'li'.
        _legacy_wavebit.rcwa(structure=STRUCTURES[path], factorization='li',
                       li_nesting=value, **KW).solve()


#: A patterned tensor raster, for the refusals `tensor_materials` publishes: the per-path lists
#: say what a raster takes, and cannot say what a raster *of tensors* refuses.
_TENSOR_RASTER = np.ones((1, _N, _N, 3, 3)) * np.eye(3)
_TENSOR_RASTER[0, _X ** 2 + _Y ** 2 <= 0.3 ** 2] *= np.array([2.0, 3.0, 4.0])


def test_a_factorization_tensor_materials_refuse_is_refused_by_name():
    """Both halves, on a tensor raster: what `tensor_materials` withholds is refused naming the
    value, and every other factorization builds -- so the list is neither short nor long."""
    refused = manifest()['tensor_materials']['factorizations_refused']
    assert set(refused) <= {m.value for m in Factorization}
    for value in (m.value for m in Factorization):
        solver = _legacy_wavebit.rcwa(structure=_TENSOR_RASTER, factorization=value, **KW)
        if value in refused:
            with pytest.raises(NotImplementedError) as excinfo:
                solver.solve()
            assert value in str(excinfo.value)
        else:
            solver.solve()


# ------------------------------------------------------- the lattice refusals

def test_li_is_refused_on_a_skewed_cell():
    with pytest.raises(NotImplementedError, match="factorization='nvf'"):
        _legacy_wavebit.rcwa(structure=STRUCTURES['raster'], lattice=HEX, factorization='li',
                       **KW_NO_CELL).solve()


def test_the_manifest_lists_every_skewed_refusal():
    """Guards against one being dropped from the document while the solver still enforces it."""
    listed = ' '.join(r['what'] for r in LATTICE['refused_on_oblique'])
    # The block path was the other entry until it was removed; both spellings must
    # have lost it together, or the document tells a consumer to avoid a path that
    # no longer exists.
    assert 'block' not in listed
    assert LATTICE['paths_refused_on_oblique'] == ()
    assert "factorization='li'" in listed
    # `raster_normal_method` was the third entry here until 2026-08-21, when all four constructions
    # gained the cell metric. What is asserted now is that it is gone from *both* spellings --
    # a stale entry in either would tell a consumer to avoid the option that is in fact the
    # one to recommend on a skewed cell.
    assert 'raster_normal_method' not in listed
    assert LATTICE['normal_methods_refused_on_oblique'] == ()


def test_the_named_and_described_skewed_refusals_agree():
    """The constraint is stated twice — as prose to read and as names to test membership on.

    Two spellings of one fact drift, and the machine-readable one is the one a consumer acts
    on, so a mismatch would mean the document reads correctly and behaves wrongly.
    """
    named = (set(LATTICE['paths_refused_on_oblique'])
             | set(LATTICE['factorizations_refused_on_oblique'])
             | set(LATTICE['normal_methods_refused_on_oblique']))
    described = ' '.join(r['what'] for r in LATTICE['refused_on_oblique'])
    for name in named:
        assert name in described, f'{name!r} is refused but not described'
    # Every described refusal must name at least one machine-checkable member — an entry
    # naming none would be prose with no name to test membership on.
    for r in LATTICE['refused_on_oblique']:
        assert any(name in r['what'] for name in named), r['what']


def test_every_named_skewed_refusal_is_a_thing_that_exists():
    """A refused path or rule that is not in the catalogue is a typo nobody would notice."""
    paths = {s['path'] for s in SPELLINGS}
    assert set(LATTICE['paths_refused_on_oblique']) <= paths
    assert set(LATTICE['factorizations_refused_on_oblique']) <= {
        m.value for m in Factorization}
    assert set(LATTICE['normal_methods_refused_on_oblique']) <= {
        m.value for m in NormalMethod}


@pytest.mark.parametrize('method', ['contour', 'spectral', 'smooth'])
def test_a_smoothing_normal_method_builds_on_a_skewed_cell(method):
    """The refusal this used to assert is gone, and the manifest has to agree with the solver.

    Both directions are worth holding: the document says nothing is refused on this axis, and
    the solver must actually build. A manifest that opened an option the engine still refuses
    would send a consumer straight into a NotImplementedError it was told could not happen.

    ``'contour'`` needs the optional contourpy, and the manifest already probes for it -- the
    per-path ``raster_normal_method`` list omits it in a minimal install -- so this case follows the
    same probe rather than asserting an option that environment does not have.
    """
    if method == 'contour':
        pytest.importorskip('contourpy',
                            reason="raster_normal_method='contour' needs the optional dependency")
    assert method not in LATTICE['normal_methods_refused_on_oblique']
    result = _legacy_wavebit.rcwa(structure=STRUCTURES['raster'], lattice=HEX, factorization='nvf',
                          raster_normal_method=method, **KW_NO_CELL).solve()
    assert np.isfinite(np.asarray(result.T)).all()


def test_a_skewed_cell_still_builds_on_the_paths_the_manifest_leaves_open():
    for structure in (*STRUCTURES.values(), _OVERLAPPING):
        _legacy_wavebit.rcwa(structure=structure, lattice=HEX, factorization='nvf',
                       **KW_NO_CELL).solve()


# ------------------------------------------------------- the document itself

def test_manifest_is_json_round_trippable():
    """A consumer receives this over the wire; anything unserialisable is a runtime failure."""
    assert json.loads(json.dumps(manifest())) == json.loads(json.dumps(manifest()))


def test_enum_lists_match_the_enums():
    enums = manifest()['enums']
    assert enums['backend'] == [m.value for m in Backend] == ['numpy', 'jax', 'torch']
    assert enums['tetm'] == [m.name.lower() for m in Tetm]
    assert enums['factorization'] == [m.value for m in Factorization]
    assert enums['fourier_coefficients'] == [m.value for m in FourierCoefficients]
    assert enums['nvf_projector'] == [m.value for m in Projector]
    assert enums['li_nesting'] == [m.value for m in Nesting]


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


# --------------------------------------------- what a material number means, against runtime

def test_the_published_material_quantity_is_the_one_the_solver_takes():
    """``material_quantity`` is the claim a generating consumer bets its numbers on.

    Every other section of this document is about *shape*: which keyword, which enum value,
    which path. This one is about a *number*, and it is the only claim here whose failure is
    silent end to end -- an index handed to a permittivity solver builds a legal,
    energy-conserving, different scene. So it is checked against arithmetic rather than against
    a table: a uniform slab of the published quantity has to match the closed-form Fresnel
    reflectance computed from that same quantity, and the two readings of one literal are far
    enough apart that no tolerance could confuse them.

    Written with ``kz`` and ``eps`` and no refractive index anywhere, for the reason
    ``tests/materials/test_material_convention_anchor.py`` gives at length: an oracle that spelled the
    quantity the way the API does could be co-migrated into agreement with a wrong flip.
    """
    assert MATERIAL_QUANTITY['quantity'] == 'permittivity'
    assert MATERIAL_QUANTITY['relative'] is True

    eps_sub = 12.25
    kw = dict(backend='numpy', top=1.0, bot=eps_sub, theta=0.0, phi=0.0, tetm='te',
              truncation_order=[0, 0], period=[1.0], wavelength=0.5, thickness=[0.0],
              structure=np.ones((1, 1, 1)))
    got = float(np.sum(np.asarray(_legacy_wavebit.rcwa(**kw).solve().R)))

    kz_top, kz_bot = np.sqrt(1.0), np.sqrt(eps_sub)
    want = abs((kz_top - kz_bot) / (kz_top + kz_bot)) ** 2
    assert got == pytest.approx(want, rel=1e-12)
    # and the other reading of 12.25 is nowhere near, so this is a convention check and not a
    # tolerance check
    other = abs((1.0 - np.sqrt(eps_sub ** 2)) / (1.0 + np.sqrt(eps_sub ** 2))) ** 2
    assert abs(want - other) > 0.15


def test_the_published_conversions_exist_and_invert_each_other():
    """``conversions`` names the way out of the flip; a name that is not exported is no way out."""
    for name in MATERIAL_QUANTITY['conversions']:
        assert hasattr(wavebit, name), f'manifest names wavebit.{name}, which is not exported'
    assert wavebit.eps_from_refractive_index(3.5) == 12.25
    assert complex(wavebit.refractive_index_from_eps(12.25)) == pytest.approx(3.5)


# ------------------------------------------------ the results contract, against a real result

def test_results_section_matches_a_real_result():
    """Every claim in ``results`` against a solve's actual output — not the tables.

    Same non-mirror principle as the rest of this file: the manifest could only drift from
    the result surface silently, because nothing else reads both. So build a solver, solve,
    and hold each published name to the object: uppercase fields must exist and be real
    (power), lowercase must exist and be complex (amplitude), the incidence variants must be
    present, the accessor must work and agree with the arrays at the claimed layout, and the
    JobSpec facts must match ``_core.spec``. A future rename that misses the manifest fails
    here rather than in a consumer's UI.
    """
    from wavebit.compat._jobs_v1 import SPEC_VERSION, JobSpec

    m = manifest()['results']
    res = _legacy_wavebit.rcwa(structure=_RASTER, **KW).solve()
    sub = res.incident if res.incident is not None else res.s_inc

    for name in list(m['efficiencies']['total']) + list(m['efficiencies']['by_outgoing_polarization']):
        arr = np.asarray(getattr(sub, name))
        assert np.isrealobj(arr), f'{name} is published as power and must be real'
    for name in m['amplitudes']['fields']:
        arr = np.asarray(getattr(sub, name))
        assert np.iscomplexobj(arr), f'{name} is published as an amplitude and must be complex'
        assert getattr(res, name) is getattr(res.incident, name)
    # The Cartesian vectors sit under a key of their own because their layout is not the
    # others': the order grid plus a trailing (Ex, Ey, Ez) axis.
    for name in m['amplitudes']['cartesian']['fields']:
        arr = np.asarray(getattr(sub, name))
        assert np.iscomplexobj(arr), f'{name} is published as an amplitude and must be complex'
        assert arr.shape == np.asarray(res.R).shape + (3,), (name, arr.shape)
        assert getattr(res, name) is getattr(res.incident, name)
        assert np.asarray(getattr(res.order(1, -1), name)).shape == (3,)
    for name in m['incidences']['fields']:
        assert hasattr(res, name), name

    # the layout claim and the accessor claim, against each other
    (cx, fx), (cy, fy) = res.order_window
    R = np.asarray(res.R)
    assert R.shape == (2 * fy + 1, 2 * fx + 1)
    assert float(res.order(1, -1).R) == R[fy - 1, fx + 1]

    # `outputs_default` is deliberately absent: `JobSpec.outputs` was removed in spec
    # version 4 (never read, but hashed into `key()`), so publishing a default for it would
    # promise a field the schema no longer has.
    assert 'outputs_default' not in m['jobspec']
    assert not hasattr(JobSpec(), 'outputs')
    assert m['jobspec']['spec_version'] == SPEC_VERSION


def test_incidence_side_claims_match_the_solver():
    """The published sides against runtime behaviour, both directions.

    Every value the manifest lists must build and must *matter* — on a z-asymmetric scene
    the two sides must return different reflectances, otherwise the manifest is publishing a
    knob that changes nothing. A value it does not list must be refused by name, and the
    published default must be what an untouched solver actually does. Physics is pinned
    elsewhere (``tests/rcwa/test_backside_incidence.py``, the RETICOLO ``inc_bottom`` golden);
    this closes the document-vs-solver loop only.
    """
    m = manifest()['results']['incidence_side']
    asym = np.array([[[9.0] * 16 + [1.0] * 16]])
    kw = dict(KW, phi=None, psi=None, truncation_order=[4, 0], bot=4.0, structure=asym, thickness=[0.3])

    got = {}
    for side in m['values']:
        res = _legacy_wavebit.rcwa(incidence_side=side, **kw).solve()
        got[side] = float(np.sum(np.asarray(res.R)))
    assert abs(got['top'] - got['bottom']) > 1e-3,\
        'the published sides must select genuinely different solves'

    solver = _legacy_wavebit.rcwa(**kw)
    assert solver.incidence_side == m['default']

    with pytest.raises(ValueError) as excinfo:
        _legacy_wavebit.rcwa(incidence_side='sideways', **kw)
    for side in m['values']:
        assert side in str(excinfo.value), (
            f'the refusal must name {side!r} so a consumer can relay the legal values')


def test_polarization_claims_match_runtime():
    """The ``incidences`` section's polarization claims, against the solver.

    Same runtime-contrast principle: the manifest says the pair fills the specified-solve
    slot, supersedes ``psi`` (naming both raises), equals ``psi``'s ``(sin, cos)`` spelling,
    and is normalized to unit incident flux — so do each of those, on a real solve. The full
    physics (circular states, energy at complex pairs) lives in ``tests/rcwa/test_polarization.py``.
    """
    kw = {k: v for k, v in KW.items() if k != 'psi'}

    via_pair = _legacy_wavebit.rcwa(structure=_RASTER, polarization=(np.sin(0.4), np.cos(0.4)),
                              **kw).solve()
    assert via_pair.incident is not None, 'the pair must fill the specified-incidence slot'

    via_psi = _legacy_wavebit.rcwa(structure=_RASTER, psi=0.4, **kw).solve()
    np.testing.assert_allclose(via_pair.incident.R, via_psi.incident.R, atol=1e-14)

    scaled = _legacy_wavebit.rcwa(structure=_RASTER, polarization=(5 * np.sin(0.4), 5 * np.cos(0.4)),
                            **kw).solve()
    np.testing.assert_allclose(scaled.incident.R, via_pair.incident.R, atol=1e-14)

    with pytest.raises(ValueError, match='psi and polarization'):
        _legacy_wavebit.rcwa(structure=_RASTER, psi=0.4, polarization=(1, 0), **kw)

# ------------------------------------------------- the material flags, against the solver

#: ``supported`` was the only claim in this document that nothing exercised, and it is the one
#: that drifted: ``patterned_anisotropic`` and ``magnetic`` both stayed False for days after
#: they started working, and ``z_coupled`` before them.
#:
#: Each case is required to *run and conserve energy*. Reachability alone would be the weaker
#: claim and it was the first draft here, until it turned out that a slot order mistake sails
#: straight through "did it return something" while producing R+T = 1.9. Every layer below is
#: lossless and passive, so R + T = 1 is a real constraint rather than a smoke test, and it is
#: the cheapest statement that fails when the case is wired up wrong.
#:
#: Absolute accuracy is still the goldens' job — ``tests/references/test_fmmax_patterned_golden.py``
#: and ``test_fmmax_magnetic_golden.py``, against oracles frozen before the code existed.
_I1 = np.eye(1, dtype=complex)
_MAT_KW = dict(backend='numpy', top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
               period=(1.0, 1.0), wavelength=1.1, thickness=[0.3])


def _tensor(xx, yy, zz, xy=0.0, yx=0.0, xz=0.0, zx=0.0, yz=0.0, zy=0.0):
    return np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]], dtype=complex)


def _in_plane(t):
    """The five in-plane slots for a uniform tensor. Off-diagonals hold the negative."""
    return [np.stack([t[0, 0] * _I1]), np.stack([t[1, 1] * _I1]),
            np.stack([np.linalg.inv(t[2, 2] * _I1)]),
            np.stack([-t[0, 1] * _I1]), np.stack([-t[1, 0] * _I1])]


def _z_row(t):
    """The four z-row slots. Handing them over *at all* is what selects the first-order formulation,
    so they go only to the case that has a z row — a zero-filled set is a different request."""
    return [np.stack([-t[0, 2] * _I1]), np.stack([-t[1, 2] * _I1]),
            np.stack([-t[2, 0] * _I1]), np.stack([-t[2, 1] * _I1])]


def _run_uniform(eps, mu=None):
    mee = _legacy_wavebit.rcwa(truncation_order=[0, 0], **_MAT_KW)
    z = _z_row(eps) if np.any(eps[2, :2]) or np.any(eps[:2, 2]) else []
    return mee._solve_conv(*_in_plane(eps), *z,
                          mu=None if mu is None else tuple(_in_plane(mu)))


def _run_patterned():
    """A pillar of a real symmetric tensor in vacuum, through the Popov-Neviere builder.

    The unpacking is written out rather than splatted, which since 2026-08-18 is a style
    choice rather than a requirement: the builders and :meth:`solve_conv` now agree on
    ``(epx, epy, epz_i, epxy, epyx)``, so ``_solve_conv(*build(...))`` is correct too.
    It was not. The builders grouped the in-plane pair together and put ``epz_i`` last, so
    splatting landed the inverted zz matrix in the ``epxy`` slot and nothing objected — the
    solve returned a full set of plausible efficiencies summing to 1.9. That is how this test
    was first written, and the energy assertion below is what caught it. ``tests/rcwa/test_slot_order.py``
    now pins the agreement for every builder; this case is left spelled out because naming the
    slots is what makes the structure under test readable.
    """
    from wavebit.rcwa._engine.factorization import to_conv_mat_raster_pn_eps
    grid, truncation_order = 48, 4
    eps = np.zeros((1, grid, grid, 3, 3), dtype=complex)
    for i in range(3):
        eps[..., i, i] = 1.0
    eps[0, 14:34, 18:30] = _tensor(2.25, 2.25, 2.89)
    epx, epy, epz_i, epxy, epyx = to_conv_mat_raster_pn_eps(eps, truncation_order, truncation_order)
    mee = _legacy_wavebit.rcwa(truncation_order=[truncation_order, truncation_order], **_MAT_KW)
    return mee._solve_conv(epx, epy, epz_i, epxy, epyx)


MATERIAL_RUNS = {
    'isotropic': lambda: _run_uniform(_tensor(2.6, 2.6, 2.6)),
    'uniform_anisotropic': lambda: _run_uniform(_tensor(2.3, 3.0, 2.6, xy=0.2, yx=0.2)),
    'magnetic': lambda: _run_uniform(_tensor(2.6, 2.6, 2.6), mu=_tensor(1.4, 1.4, 2.2)),
    'z_coupled': lambda: _run_uniform(_tensor(2.6, 2.6, 2.6, xz=0.4, zx=0.4)),
    'patterned_anisotropic': _run_patterned,
    'patterned_z_coupled': lambda: _legacy_wavebit.rcwa(
        truncation_order=(2,1), factorization='li',
        structure=[dict(background=1.,shapes=[dict(type='rectangle',lx=.4,ly=.5,
            material=wavebit.Material(_tensor(2.6,3.,2.4,xz=.3,zx=.3)))])],
        **_MAT_KW).solve(),
}


def test_every_supported_material_is_one_this_file_performs():
    """No supported case may be added without something here that actually runs it."""
    named = {e['case'] for e in MATERIALS if e['supported']}
    assert named == set(MATERIAL_RUNS), (
        f'not exercised: {sorted(named - set(MATERIAL_RUNS))}; '
        f'exercised but not claimed: {sorted(set(MATERIAL_RUNS) - named)}')


@pytest.mark.parametrize('case', sorted(MATERIAL_RUNS), ids=sorted(MATERIAL_RUNS))
def test_a_supported_material_solves_and_conserves_energy(case):
    """A stale True sends a consumer's user into a traceback. This is what stops that.

    Every layer here is lossless and passive, so ``R + T == 1`` holds at any truncation and
    is not a convergence claim. It is asserted because "it returned" does not distinguish a
    working case from a misconfigured one.
    """
    res = MATERIAL_RUNS[case]()
    total = float(np.asarray(res.incident.R).sum() + np.asarray(res.incident.T).sum())
    assert abs(total - 1.0) < 1e-6, f'{case}: R + T = {total!r} on a lossless passive layer'


def _z_coupled_cell(**over):
    cell = np.zeros((1, 16, 16, 3, 3), dtype=complex)
    for i in range(3):
        cell[..., i, i] = 1.0
    cell[0, 4:12, 4:12] = _tensor(2.9, 2.6, 2.9, xy=0.3, yx=0.3, **over)
    return cell


def test_tensor_builders_retain_all_z_coupling_slots():
    from wavebit.rcwa._engine.factorization import to_conv_mat_raster_pn_eps, to_conv_mat_raster_pn_mu
    for build in (to_conv_mat_raster_pn_eps,to_conv_mat_raster_pn_mu):
        for entries,positions in ((dict(xz=.4,zx=.4),(5,7)),(dict(yz=.4,zy=.4),(6,8))):
            full=build(_z_coupled_cell(**entries),2,2)
            assert len(full)==9
            assert all(np.max(abs(full[i]))>1e-3 for i in positions)
        assert len(build(_z_coupled_cell(),2,2))==5


def test_in_plane_tensor_builders_keep_the_five_slot_fast_path():
    """A z-decoupled concrete tensor does not force the larger eigensystem."""
    from wavebit.rcwa._engine import factorization
    cell = np.zeros((1, 16, 16, 3, 3), dtype=complex)
    for i in range(3):
        cell[..., i, i] = 1.0
    cell[0, 4:12, 4:12] = _tensor(2.9, 2.6, 2.9, xy=0.3, yx=0.3)     # no z row: reaches them
    reached = []
    for name in (n for n in dir(factorization) if n.startswith('to_conv_mat')):
        try:
            out = getattr(factorization, name)(cell, 2, 2)
        except Exception:
            continue                      # not a tensor-raster builder; nothing to claim
        reached.append(name)
        assert len(out) == 5, (
            f'{name} returned {len(out)} slots — if it now emits the z row, patterned and '
            'z-coupled may be reachable and MATERIALS should say so')
    # The sweep has to actually reach the tensor builders, or it asserts nothing. This is the
    # line the version before 2026-08-27 was missing, and it is why a refusal made it vacuous.
    assert set(reached) >= {'to_conv_mat_raster_pn_eps', 'to_conv_mat_raster_pn_mu'}, reached


# ------------------------------------------------------------------- what each backend can do

def test_differentiability_claims_match_the_solver():
    """The published answer, held against a real solver of each backend.

    `backends[*].differentiable` is read off `_DIFFERENTIABLE`, so this checks the thing the
    declaration is *about*: whether `grad` and `fit` are there and answer. A backend that gains
    or loses autodiff fails here rather than going stale in a consumer -- which is the rule
    every other claim in this file is held to.
    """
    published = manifest()['backends']
    assert set(published) == {'numpy', 'jax', 'torch'}, sorted(published)

    for backend in ('numpy', 'jax', 'torch'):
        claim = published[backend]
        if not claim['available']:
            with pytest.raises(ImportError):
                __import__(backend)
            continue
        pytest.importorskip(backend)

        solver = _legacy_wavebit.rcwa(backend=backend, structure=np.ones((1, 4, 4)) * 2.0,
                              period=[400.0, 400.0], truncation_order=[1, 1],
                              thickness=[150.0], wavelength=900.0, top=1.0, bot=1.0)
        assert type(solver)._DIFFERENTIABLE is claim['differentiable'], backend

        # Both are present on every backend; on a non-differentiating one they exist to refuse.
        assert hasattr(solver, 'grad') and hasattr(solver, 'fit'), backend

        # And then ASK, because the line above is circular on its own: `_DIFFERENTIABLE` and
        # the published claim are both read off `DIFFERENTIABLE_BACKENDS`, so they agree
        # whatever the table says. A table declaring that NumPy differentiates passed every
        # test in this file until this call went in.
        if claim['differentiable']:
            gradient = _gradient_of_a_transmittance(solver, backend)
            assert np.any(np.asarray(gradient) != 0), (
                f'{backend} is published as differentiable and returned an all-zero gradient')
        else:
            with pytest.raises(NotImplementedError, match=r'automatic differentiation'):
                solver.grad(['structure'], solver.solve, lambda result: result.T.sum())


def _gradient_of_a_transmittance(solver, backend):
    """d(sum T)/d(structure) through the backend's own `grad`, detached for comparison."""
    import functools

    if backend == 'jax':
        import jax.numpy as jnp
        _legacy_wavebit.enable_x64()
        solver.structure = jnp.asarray(np.asarray(solver.structure))
        loss = (lambda result: jnp.sum(result.T).real)
    else:
        import torch
        solver.structure = torch.tensor(np.asarray(solver.structure), dtype=torch.complex128)
        loss = (lambda result: torch.sum(result.T).real)

    got = solver.grad(['structure'], functools.partial(solver.solve), loss)['structure']
    return got.detach() if hasattr(got, 'detach') else got


def test_a_backend_that_does_not_differentiate_says_so_rather_than_missing_the_method():
    """`AttributeError: 'SolverNumpy' object has no attribute 'grad'` is true and says nothing.

    Not that the absence is deliberate, not that two other backends have it, not that the scene
    needs no change to move. The refusal names all three, and names the declaration a caller can
    read *before* building a solver it cannot optimize.
    """
    solver = _legacy_wavebit.rcwa(backend='numpy', structure=np.ones((1, 4, 4)), period=[400.0, 400.0],
                          truncation_order=[1, 1], thickness=[150.0], wavelength=900.0,
                          top=1.0, bot=1.0)
    for name in ('grad', 'fit'):
        with pytest.raises(NotImplementedError) as caught:
            getattr(solver, name)(['structure'], None, None)
        message = str(caught.value)
        assert name in message, message
        assert "backend='jax'" in message and "backend='torch'" in message, message
        assert '_DIFFERENTIABLE' in message, message


def test_the_manifest_does_not_import_an_optional_backend_to_answer():
    """Building the document must not pull JAX or PyTorch in.

    `_backend_capabilities` reads a class attribute, which needs the class -- so the imports are
    lazy and one at a time. A module-scope import of all three would make `import
    wavebit.manifest` import both optional libraries, and the manifest is exactly the thing a
    caller reads to decide whether to install one.
    """
    import subprocess
    import sys

    code = ('import sys, json; import wavebit.manifest as m; '
            'd = m.manifest(); '
            'print(json.dumps({"jax": "jax" in sys.modules, "torch": "torch" in sys.modules, '
            '"backends": sorted(d["backends"])}))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen['backends'] == ['jax', 'numpy', 'torch'], seen
    assert not seen['jax'] and not seen['torch'], (
        f'building the manifest imported an optional backend: {seen}')


def test_the_published_fourier_origin_matches_a_centred_stripe():
    claim = manifest()['lattice']
    assert claim['fourier_origin'] == 'cell_centre'
    assert claim['fourier_origin_expression'] == '0.5*(a1 + a2)'
    row = np.ones((1, 1, 100));row[..., 30:70] = 3.
    solver = _legacy_wavebit.rcwa(structure=row, period=(1., 1.), thickness=[.2],
                         truncation_order=(1, 0), tetm='te')
    coefficient = solver.eps_conv_matrices().epy[0, 2, 1]
    np.testing.assert_allclose(coefficient, 2*.4*np.sinc(.4), atol=2e-14, rtol=0)

# ------------------------------------------------------- the second door, against runtime

def test_both_published_entry_points_exist_and_are_what_they_claim():
    """A door named in the document has to be callable under that exact name."""
    import importlib

    from wavebit.manifest import SOLVERS
    for door, entry in SOLVERS.items():
        assert entry['entry_point'] == f'wavebit.compat.legacy.{door}'
        module, _, name = entry['entry_point'].rpartition('.')
        assert callable(getattr(importlib.import_module(module), name)), entry['entry_point']
    assert SOLVERS['fdtd']['differentiable'] is True
    assert SOLVERS['fdtd']['backends'] == ('numpy', 'jax', 'torch')


def test_the_sections_a_door_claims_are_keys_of_this_document():
    """`sections` is what scopes `backends` to the door it is true of.

    Without it a consumer reads `backends.differentiable` as a promise about every solve here,
    whose default time-domain route does not differentiate -- so names resolve
    independently of each route's additional restrictions.
    """
    from wavebit.manifest import SOLVERS
    keys = set(manifest())
    for door, entry in SOLVERS.items():
        missing = set(entry['sections']) - keys
        assert not missing, (door, sorted(missing))


#: One override per published refusal, keyed by the name the manifest publishes. A dict rather
#: than a list so the two sets can be compared: a refusal published and not exercised, or
#: exercised and not published, is what fails here.
FDTD_REFUSALS = {
    'a complex theta': dict(theta=0.3 + 0.1j),
    'a complex permittivity': dict(structure=np.array([[[1.0 + 0.3j, 1.0]]])),
    'Re(eps) <= 0': dict(structure=np.array([[[-2.0, 1.0]]])),
    'lattice': dict(lattice=[[1.0, 0.0], [0.5, 1.0]]),
}

FDTD_KW = dict(structure=np.full((1, 1, 2), 12.25), thickness=[300.0], top=1.0, bot=2.25,
               period=[700.0], wavelength=900.0, tetm='te', resolution=0.02, n_periods=5)


def test_every_published_refusal_is_one_the_door_performs():
    from wavebit.manifest import SOLVERS
    published = set(SOLVERS['fdtd']['refused'])
    assert published == set(FDTD_REFUSALS), (
        f'published and not exercised: {sorted(published - set(FDTD_REFUSALS))}; '
        f'exercised and not published: {sorted(set(FDTD_REFUSALS) - published)}')


@pytest.mark.parametrize('backend', [None, 'numpy'])
@pytest.mark.parametrize('name', sorted(FDTD_REFUSALS), ids=sorted(FDTD_REFUSALS))
def test_a_refusal_the_manifest_publishes_is_refused(name, backend):
    """Each raises before any time stepping, so the whole battery costs milliseconds."""
    with pytest.raises((ValueError, TypeError)):
        _legacy_wavebit.fdtd(**dict(FDTD_KW, backend=backend, **FDTD_REFUSALS[name]))


FDTD_COMMON_KW = dict(structure=np.full((1, 1, 1), 2.), thickness=[.2], top=1., bot=1.,
                      period=.2, wavelength=1., resolution=10, n_pml=4, n_periods=25,
                      truncation_order=0)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('integration', ['separable', 'area'])
def test_published_fdtd_rectangles_use_common_route(backend, integration):
    from wavebit.manifest import SOLVERS
    if backend != 'numpy':
        pytest.importorskip(backend)
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    common = SOLVERS['fdtd']['routes']['common']
    assert common['parametric_shapes'] == ('rectangle', 'square', 'circle', 'disk', 'ellipse')
    assert 'rectangle centre' in common['differentiable_parameters']
    assert 'shape_smoothing' in common['static_parameters']
    assert common['shape_integration'] == ('separable', 'area')
    assert 'exact bilinear polygon moments' in common['area_smoothing']
    assert 'rectangle angle (area)' in common['differentiable_parameters']
    settings = dict(FDTD_COMMON_KW, period=[.2, .2], structure=[dict(background=2., shapes=[
        dict(type='square', l=.08, angle=.23 if integration == 'area' else 0., material=2.)])],
        backend=backend, shape_integration=integration, shape_smoothing=.02)
    result = _legacy_wavebit.fdtd(**settings)
    target = _legacy_wavebit.fdtd(**dict(settings, structure=np.full((1, 2, 2), 2.), shape_smoothing=0., shape_integration='separable'))
    np.testing.assert_allclose(result.s_inc.R, target.s_inc.R, atol=1e-13)
    with pytest.raises((TypeError, ValueError)):
        _legacy_wavebit.fdtd(**dict(settings, backend=None))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('smoothing', [0., .05])
def test_published_fdtd_thickness_capability_requires_fixed_mesh_and_total(backend, smoothing):
    from wavebit.manifest import SOLVERS
    from wavebit.fdtd._periodic.runtime import time_loop_backend
    from tests.fdtd.test_fdtd_scene_angles import value_and_jacobian
    pytest.importorskip(backend)
    common = SOLVERS['fdtd']['routes']['common']
    assert 'fixed-total layer thickness' in common['differentiable_parameters']
    assert common['backend_thickness_requires'] == ('mesh_thickness', 'subpixel=True')
    assert 'layer_smoothing' in common['static_parameters']
    assert common['positive_layer_smoothing_requires'] == ('mesh_thickness', 'subpixel=True')
    which = backend
    def objective(d, backend=backend):
        return _legacy_wavebit.fdtd(**dict(FDTD_COMMON_KW, structure=np.array([[[2.]], [[1.]]]),
                                   thickness=[d, .4-d], mesh_thickness=[.2, .2], layer_smoothing=smoothing,
                                   backend=backend)).s_inc.T.reshape(1)
    value, derivative = value_and_jacobian(objective, .173, which)
    step = 1e-5
    finite = (objective(.173+step, 'numpy')-objective(.173-step, 'numpy'))/(2*step)
    np.testing.assert_allclose(value, objective(.173, 'numpy'), atol=1e-10)
    np.testing.assert_allclose(derivative, finite, atol=2e-8, rtol=3e-5)
    assert abs(finite).item() > .01
    with pytest.raises(ValueError, match='fixed total depth'):
        _legacy_wavebit.fdtd(**dict(FDTD_COMMON_KW, backend=backend, mesh_thickness=[.3]))
    with pytest.raises(ValueError, match='explicit mesh_thickness'):
        _legacy_wavebit.fdtd(**dict(FDTD_COMMON_KW, backend=backend,
                            thickness=time_loop_backend(which).array([.2])))


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_published_variable_depth_uses_live_faces_on_a_fixed_mesh(backend):
    from wavebit.manifest import SOLVERS
    pytest.importorskip(backend)
    common = SOLVERS['fdtd']['routes']['common']
    assert 'independent layer thickness (variable_depth)' in common['differentiable_parameters']
    assert 'variable_depth' in common['static_parameters']
    assert common['variable_depth_requires'][:2] == ('mesh_thickness', 'subpixel=True')
    result, out = _legacy_wavebit.fdtd(**dict(FDTD_COMMON_KW, backend=backend, thickness=[.273],
                                     mesh_thickness=[.2], variable_depth=True, return_diagnostics=True))
    assert np.isfinite(np.asarray(result.s_inc.T)).all() and out['variable_depth']
    np.testing.assert_allclose(out['transmission_z'], 0., atol=1e-15)
    with pytest.raises(ValueError, match='fixed total depth'):
        _legacy_wavebit.fdtd(**dict(FDTD_COMMON_KW, backend=backend, thickness=[.273], mesh_thickness=[.2]))


def test_fdtd_manifest_scopes_backend_omission_to_legacy_numpy():
    from wavebit.manifest import SOLVERS
    entry = SOLVERS['fdtd']
    legacy = entry['routes'][entry['default_route']]
    assert legacy['backends'] == ('numpy',)
    assert legacy['differentiable'] is False
    assert legacy['incidence_sides'] == ('top', 'bottom')


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_default_fdtd_route_answers_the_bottom_incidence_it_lists(tetm):
    """``legacy_numpy`` lists ``'bottom'``, and its kernel refuses it: both are true.

    ``solve_fdtd`` illuminates its first layer only, and refuses ``incidence_side='bottom'``.
    The public route never hands it one: ``adapter._legacy_inputs`` swaps the half-spaces and
    keeps the layer order, so the kernel solves the same scene lit from the other face. So the
    listing is a claim about that mapping, and a finite ``T`` cannot check it -- on a
    symmetric scene, or with the layers in the wrong order, it is finite and wrong. Held here
    against RCWA, amplitude by amplitude, on a scene whose two faces differ: two layers and
    two half-spaces, at an angle. Measured 2026-10-15: the worst gap is 3.9e-2 / 9.6e-3 (TE)
    and 3.7e-2 / 1.0e-2 (TM) at 20 / 40 cells per unit, falling at second order, and the
    same scene lit from the top face gives 3.8e-2 / 9.1e-3 and 3.9e-2 / 9.0e-3. An adapter
    that also reverses the layers misses by 0.25 at both resolutions, one that ignores the
    side by 0.58.
    """
    scene = dict(structure=np.array([[[2.1]], [[3.2]]]), thickness=[.13, .21], top=1., bot=2.25,
                 period=.4, wavelength=1., theta=.27, tetm=tetm, incidence_side='bottom',
                 truncation_order=0)
    reference = _legacy_wavebit.rcwa(backend='numpy', **scene).solve()
    gaps = []
    for resolution in (20, 40):
        result = _legacy_wavebit.fdtd(**scene, resolution=resolution, n_periods=60)
        gaps.append(max(np.abs(np.asarray(getattr(result.incident, name))
                               - np.asarray(getattr(reference.incident, name))).max()
                        for name in ('r_s', 'r_p', 't_s', 't_p')))
    assert gaps[1] < .015, gaps
    assert gaps[1] < gaps[0] / 3, gaps


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_published_common_fdtd_backends_accept_bottom_jones_and_all_columns(backend):
    from wavebit.manifest import SOLVERS
    from wavebit.observations.diffraction import Result
    if backend != 'numpy':
        pytest.importorskip(backend)
    common = SOLVERS['fdtd']['routes']['common']
    assert backend in common['backends']
    assert 'bottom' in common['incidence_sides']
    result = _legacy_wavebit.fdtd(**dict(FDTD_COMMON_KW, backend=backend, incidence_side='bottom',
                                 polarization=(1., .6j)))
    assert isinstance(result, Result)
    for column in common['results_incidences']:
        values = getattr(result, column).T
        if hasattr(values, 'detach'):
            values = values.detach().numpy()
        assert np.isfinite(values).all()


@pytest.mark.parametrize('polarization', [dict(polarization=(1., .6j)), dict(psi=.6, chi=.2)])
def test_legacy_fdtd_also_accepts_the_published_polarization_family(polarization):
    # The legacy source has a longer envelope than the common engine's pulse.
    result = _legacy_wavebit.fdtd(**dict(FDTD_COMMON_KW, n_periods=60, **polarization))
    assert np.isfinite(result.incident.T).all()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_published_common_fdtd_gradient_and_halfspace_requirements(backend):
    from wavebit.manifest import SOLVERS
    from tests.fdtd.test_fdtd_scene_angles import value_and_jacobian
    pytest.importorskip(backend)
    common = SOLVERS['fdtd']['routes']['common']
    assert backend in common['differentiable_backends']
    which = backend
    def objective(epsilon):
        return _legacy_wavebit.fdtd(**dict(FDTD_COMMON_KW, backend=backend, top=epsilon,
                                   mesh_indices=(1., 1.))).s_inc.T.reshape(1)
    value, gradient = value_and_jacobian(objective, 1.2, which)
    step = 1e-4
    def numpy(epsilon):
        return _legacy_wavebit.fdtd(**dict(FDTD_COMMON_KW, backend='numpy', top=epsilon,
                                   mesh_indices=(1., 1.))).s_inc.T.reshape(1)
    finite = (numpy(1.2+step)-numpy(1.2-step))/(2*step)
    np.testing.assert_allclose(value, numpy(1.2), atol=1e-10)
    np.testing.assert_allclose(gradient, finite, rtol=3e-5, atol=1e-8)
    assert abs(finite).item() > .001
    from wavebit.fdtd._periodic.runtime import time_loop_backend
    epsilon = time_loop_backend(which).array(1.2)
    for missing in common['array_half_spaces_require']:
        options = dict(FDTD_COMMON_KW, backend=backend, top=epsilon, mesh_indices=(1., 1.))
        options[missing] = None
        with pytest.raises(ValueError, match='explicit mesh_indices and explicit truncation_order'):
            _legacy_wavebit.fdtd(**options)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('kind', ['circle', 'disk'])
def test_published_fdtd_disk_contract_uses_common_area_route(backend, kind):
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')
    common = manifest()['solvers']['fdtd']['routes']['common']
    assert 'circle diameter (area)' in common['differentiable_parameters']
    assert 'fixed 16-point Gaussian' in common['area_smoothing']
    assert 'no ly' in common['circle_convention']
    stack = _legacy_wavebit.Stack.from_structure([dict(shapes=[dict(type=kind, l=.08, material=2.)])],
        [.1], top=1., bot=1., period=[.2, .2])
    result, diagnostics = _legacy_wavebit.fdtd(stack, _legacy_wavebit.Incidence(wavelength=1.), backend=backend,
        resolution=10, n_pml=5, n_periods=30, truncation_order=0, shape_integration='area',
        return_diagnostics=True)
    assert np.isfinite(np.asarray(result.incident.r_s)).all()
    assert diagnostics['geometry_kind'] == 'parametric'


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_published_fdtd_ellipse_is_rotated_and_uses_full_axis_lengths(backend):
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')
    common = manifest()['solvers']['fdtd']['routes']['common']
    assert 'ellipse' in common['parametric_shapes']
    assert 'ellipse centre/axes/angle (area)' in common['differentiable_parameters']
    assert 'full principal-axis lengths' in common['ellipse_convention']
    stack = _legacy_wavebit.Stack.from_structure([dict(shapes=[
        dict(type='ellipse', lx=.15, ly=.07, angle=.41, material=2.)])],
        [.1], top=1., bot=1., period=[.2, .2])
    result, out = _legacy_wavebit.fdtd(stack, _legacy_wavebit.Incidence(wavelength=1.), backend=backend,
        resolution=10, n_pml=5, n_periods=30, truncation_order=0, shape_integration='area',
        shape_smoothing=.02, return_diagnostics=True)
    assert np.isfinite(np.asarray(result.incident.r_s)).all()
    assert out['geometry_kind'] == 'parametric' and out['stable']


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_published_fdtd_multiple_area_rectangles_use_last_shape_painting(backend):
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')
    common = manifest()['solvers']['fdtd']['routes']['common']
    assert 'multiple rectangle/square' in common['area_requires']
    assert 'later rectangles overwrite' in common['area_painting']
    stack = _legacy_wavebit.Stack.from_structure([dict(shapes=[
        dict(lx=.13, ly=.07, cx=-.02, angle=.31, material=1.8),
        dict(lx=.09, ly=.11, cx=.025, angle=-.2, material=2.3)])],
        [.1], top=1., bot=1., period=[.2, .2])
    result, out = _legacy_wavebit.fdtd(stack, _legacy_wavebit.Incidence(wavelength=1.), backend=backend,
        resolution=10, n_pml=5, n_periods=30, truncation_order=0, shape_integration='area',
        shape_smoothing=.02, return_diagnostics=True)
    assert np.isfinite(np.asarray(result.incident.r_s)).all()
    assert out['geometry_kind'] == 'parametric' and out['stable']


def test_current_lifecycle_names_real_physical_interfaces_and_serialization():
    """The current index resolves actual interfaces independently of legacy tables."""
    import dataclasses
    import importlib
    import inspect
    from types import ModuleType

    def resolve(path):
        module, _, name = path.rpartition('.')
        return getattr(importlib.import_module(module), name)

    current = manifest()['current_lifecycle']
    for name, entry in current['solvers'].items():
        namespace = resolve(entry['module'])
        assert isinstance(namespace, ModuleType)
        assert dataclasses.is_dataclass(resolve(entry['problem']))
        assert dataclasses.is_dataclass(resolve(entry['numerics']))
        for operation in ('check', 'prepare', 'run'):
            method = resolve(entry[operation])
            assert method is getattr(namespace, operation)
            parameters = inspect.signature(method).parameters
            assert parameters['problem'].default is inspect.Parameter.empty
            assert parameters['numerics'].kind is inspect.Parameter.KEYWORD_ONLY
            assert set(current['execution_keywords']) <= parameters.keys()
        prepared = resolve(entry['prepared'])
        assert inspect.signature(prepared.run).parameters['problem'].default is inspect.Parameter.empty
        assert callable(prepared.validate)
        assert manifest()['solvers'][name]['entry_point'] != entry['module']

    report = resolve(current['check_report']['type'])()
    assert report.ok is True
    for field in current['check_report']['fields']:
        assert hasattr(report, field)
    assert getattr(report, current['check_report']['raise_method'])() is report
    error = resolve(current['validation_error']['type'])('bad', code='example', path='problem')
    for field in current['validation_error']['fields']:
        assert hasattr(error, field)

    rcwa = importlib.import_module('wavebit.rcwa')
    problem = rcwa.Problem(stack=rcwa.Stack(), excitation=rcwa.Excitation(
        wavelength=1., sources=(rcwa.PlaneWave(),)))
    job_type = resolve(current['job']['type'])
    job = job_type(solver='rcwa', problem=problem, numerics=rcwa.Numerics())
    assert json.loads(job.to_json())['schema'] == current['job']['schema']
    assert job_type.from_json(job.to_json()).key == job.key
    for method in current['job']['serialization']:
        assert callable(getattr(job_type, method))
    assert resolve(current['job']['run']) is wavebit.run
