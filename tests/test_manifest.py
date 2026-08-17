"""The capability manifest must describe the solver that exists, not the one it described.

The manifest is read by programs outside this process, so a claim that drifts is worse than
one that was never made: the consumer keeps answering confidently with a fact the solver
stopped honouring, and nothing raises anywhere. These tests are the only thing standing
between that and a published document.

**They are deliberately not mirror tests.** :mod:`wavebit.manifest` reads
``_core/engine.py``'s refusal tables at import, so asserting that the two agree would be
asserting that a variable equals itself. What is checked instead is the manifest against
*runtime behaviour*: build the ucell each spelling describes and confirm the solver assigns
the path claimed, then ask for each option the manifest says is unavailable and confirm the
build refuses it. That closes the loop the tables sit in the middle of.
"""
import json
import warnings

import numpy as np
import pytest

import wavebit
from wavebit.manifest import LATTICE, MATERIALS, SPELLINGS, cited_sources, manifest
from wavebit._core.enums import (Backend, Factorization, FourierCoefficients, NormalMethod,
                                   Pol)

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
    solver = wavebit.rcwa(ucell=UCELLS[spelling['path']], **KW)
    assert solver.modeling_type_assigned == spelling['modeling_type']


def test_the_manifest_covers_every_spelling_the_setter_accepts():
    """A path the solver can assign but the manifest omits is a path nobody is told about."""
    assigned = {wavebit.rcwa(ucell=u, **KW).modeling_type_assigned for u in UCELLS.values()}
    assert assigned == {s['modeling_type'] for s in SPELLINGS}


# ------------------------------------------------------- unavailable options are refused

def _unavailable(path, axis, enum_cls):
    """The values of one axis the manifest says this path does not implement."""
    return [m.value for m in enum_cls if m.value not in PATHS[path][axis]]


@pytest.mark.parametrize('path', sorted(UCELLS))
def test_a_factorization_the_manifest_withholds_is_refused(path):
    for value in _unavailable(path, 'factorization', Factorization):
        with pytest.raises(Exception) as excinfo:
            wavebit.rcwa(ucell=UCELLS[path], factorization=value, **KW).conv_solve()
        assert value in str(excinfo.value), (
            f'{path}: refused factorization={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(UCELLS))
def test_a_series_the_manifest_withholds_is_refused(path):
    for value in _unavailable(path, 'fourier_coefficients', FourierCoefficients):
        with pytest.raises(Exception) as excinfo:
            wavebit.rcwa(ucell=UCELLS[path], fourier_coefficients=value,
                           **KW).conv_solve()
        assert value in str(excinfo.value), (
            f'{path}: refused fourier_coefficients={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(UCELLS))
def test_a_normal_method_the_manifest_withholds_is_refused(path):
    """The shape paths list no raster construction; naming one there must refuse."""
    for value in _unavailable(path, 'normal_method', NormalMethod):
        with pytest.raises(Exception) as excinfo:
            wavebit.rcwa(ucell=UCELLS[path], normal_method=value, **KW).conv_solve()
        assert value in str(excinfo.value), (
            f'{path}: refused normal_method={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(UCELLS))
def test_an_option_the_manifest_offers_actually_builds(path):
    """The other half — a path must not be described as implementing what it refuses."""
    for value in PATHS[path]['factorization']:
        wavebit.rcwa(ucell=UCELLS[path], factorization=value, **KW).conv_solve()
    for value in PATHS[path]['fourier_coefficients']:
        wavebit.rcwa(ucell=UCELLS[path], fourier_coefficients=value, **KW).conv_solve()
    for value in PATHS[path]['normal_method']:
        # The manifest's own caveat: the axis is read only under factorization='nvf'.
        wavebit.rcwa(ucell=UCELLS[path], factorization='nvf',
                       normal_method=value, **KW).conv_solve()


# ------------------------------------------------------- the lattice refusals

def test_the_block_path_is_refused_on_a_skewed_cell():
    with pytest.raises(NotImplementedError, match='block modeler'):
        wavebit.rcwa(ucell=UCELLS['block'], lattice=HEX, factorization='nvf',
                       **KW).conv_solve()


def test_li_is_refused_on_a_skewed_cell():
    with pytest.raises(NotImplementedError, match="factorization='nvf'"):
        wavebit.rcwa(ucell=UCELLS['raster'], lattice=HEX, factorization='li',
                       **KW).conv_solve()


def test_the_manifest_lists_every_skewed_refusal():
    """Guards against one being dropped from the document while the solver still enforces it."""
    listed = ' '.join(r['what'] for r in LATTICE['refused_on_oblique'])
    assert 'block path' in listed
    assert "factorization='li'" in listed
    assert 'normal_method' in listed


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
        assert any(name in r['what'] for name in named | {'block'}), r['what']


def test_every_named_skewed_refusal_is_a_thing_that_exists():
    """A refused path or rule that is not in the catalogue is a typo nobody would notice."""
    paths = {s['path'] for s in SPELLINGS}
    assert set(LATTICE['paths_refused_on_oblique']) <= paths
    assert set(LATTICE['factorizations_refused_on_oblique']) <= {
        m.value for m in Factorization}
    assert set(LATTICE['normal_methods_refused_on_oblique']) <= {
        m.value for m in NormalMethod}


def test_a_smoothing_normal_method_is_refused_on_a_skewed_cell():
    """Only 'distance' carries the Jacobian pullback; the others must refuse, by name."""
    with pytest.raises(NotImplementedError, match='normal_method'):
        wavebit.rcwa(ucell=UCELLS['raster'], lattice=HEX, factorization='nvf',
                       normal_method='contour', **KW).conv_solve()


def test_a_skewed_cell_still_builds_on_the_paths_the_manifest_leaves_open():
    for path in ('raster', 'smooth', 'boundary'):
        wavebit.rcwa(ucell=UCELLS[path], lattice=HEX, factorization='nvf',
                       **KW).conv_solve()


# ------------------------------------------------------- the document itself

def test_manifest_is_json_round_trippable():
    """A consumer receives this over the wire; anything unserialisable is a runtime failure."""
    assert json.loads(json.dumps(manifest())) == json.loads(json.dumps(manifest()))


def test_enum_lists_match_the_enums():
    enums = manifest()['enums']
    assert enums['backend'] == [m.name.lower() for m in Backend]
    assert enums['tetm'] == [m.name.lower() for m in Pol]
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
    from wavebit._core.spec import SPEC_VERSION, JobSpec

    m = manifest()['results']
    res = wavebit.rcwa(ucell=_RASTER, **KW).conv_solve()
    sub = res.res if res.res is not None else res.res_te_inc

    for name in list(m['efficiencies']['total']) + list(m['efficiencies']['by_outgoing_polarization']):
        arr = np.asarray(getattr(sub, name))
        assert np.isrealobj(arr), f'{name} is published as power and must be real'
    for name in m['amplitudes']['fields']:
        arr = np.asarray(getattr(sub, name))
        assert np.iscomplexobj(arr), f'{name} is published as an amplitude and must be complex'
    for name in m['incidences']['fields']:
        assert hasattr(res, name), name

    # the layout claim and the accessor claim, against each other
    fx, fy = res.fto
    R = np.asarray(res.R)
    assert R.shape == (2 * fy + 1, 2 * fx + 1)
    assert float(res.order(1, -1).R) == R[fy - 1, fx + 1]

    assert tuple(m['jobspec']['outputs_default']) == tuple(JobSpec().outputs)
    assert m['jobspec']['spec_version'] == SPEC_VERSION


def test_incidence_side_claims_match_the_solver():
    """The published sides against runtime behaviour, both directions.

    Every value the manifest lists must build and must *matter* — on a z-asymmetric scene
    the two sides must return different reflectances, otherwise the manifest is publishing a
    knob that changes nothing. A value it does not list must be refused by name, and the
    published default must be what an untouched solver actually does. Physics is pinned
    elsewhere (``tests/test_backside_incidence.py``, the RETICOLO ``inc_bottom`` golden);
    this closes the document-vs-solver loop only.
    """
    m = manifest()['results']['incidence_side']
    asym = np.array([[[3.0] * 16 + [1.0] * 16]])
    kw = dict(KW, phi=None, psi=None, fto=[4, 0], n_bot=2.0, ucell=asym, thickness=[0.3])

    got = {}
    for side in m['values']:
        res = wavebit.rcwa(incidence_side=side, **kw).conv_solve()
        got[side] = float(np.sum(np.asarray(res.R)))
    assert abs(got['top'] - got['bottom']) > 1e-3, \
        'the published sides must select genuinely different solves'

    solver = wavebit.rcwa(**kw)
    assert solver.incidence_side == m['default']

    with pytest.raises(ValueError) as excinfo:
        wavebit.rcwa(incidence_side='sideways', **kw)
    for side in m['values']:
        assert side in str(excinfo.value), (
            f'the refusal must name {side!r} so a consumer can relay the legal values')


def test_polarization_claims_match_runtime():
    """The ``incidences`` section's polarization claims, against the solver.

    Same runtime-contrast principle: the manifest says the pair fills the specified-solve
    slot, supersedes ``psi`` (naming both raises), equals ``psi``'s ``(sin, cos)`` spelling,
    and is normalized to unit incident flux — so do each of those, on a real solve. The full
    physics (circular states, energy at complex pairs) lives in ``tests/test_polarization.py``.
    """
    kw = {k: v for k, v in KW.items() if k != 'psi'}

    via_pair = wavebit.rcwa(ucell=_RASTER, polarization=(np.sin(0.4), np.cos(0.4)),
                              **kw).conv_solve()
    assert via_pair.res is not None, 'the pair must fill the specified-incidence slot'

    via_psi = wavebit.rcwa(ucell=_RASTER, psi=0.4, **kw).conv_solve()
    np.testing.assert_allclose(via_pair.res.R, via_psi.res.R, atol=1e-14)

    scaled = wavebit.rcwa(ucell=_RASTER, polarization=(5 * np.sin(0.4), 5 * np.cos(0.4)),
                            **kw).conv_solve()
    np.testing.assert_allclose(scaled.res.R, via_pair.res.R, atol=1e-14)

    with pytest.raises(ValueError, match='psi and polarization'):
        wavebit.rcwa(ucell=_RASTER, psi=0.4, polarization=(1, 0), **kw)

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
_MAT_KW = dict(backend=0, n_top=1.0, n_bot=1.0, theta=0.0, phi=0.0, psi=0.0,
               period=(1.0, 1.0), wavelength=1.1, thickness=[0.3])


def _tensor(xx, yy, zz, xy=0.0, yx=0.0, xz=0.0, zx=0.0):
    return np.array([[xx, xy, xz], [yx, yy, 0.0], [zx, 0.0, zz]], dtype=complex)


def _in_plane(t):
    """The five in-plane slots for a uniform tensor. Off-diagonals hold the negative."""
    return [np.stack([t[0, 0] * _I1]), np.stack([t[1, 1] * _I1]),
            np.stack([np.linalg.inv(t[2, 2] * _I1)]),
            np.stack([-t[0, 1] * _I1]), np.stack([-t[1, 0] * _I1])]


def _z_row(t):
    """The four z-row slots. Handing them over *at all* is what selects the 4ff formulation,
    so they go only to the case that has a z row — a zero-filled set is a different request."""
    return [np.stack([-t[0, 2] * _I1]), np.stack([-t[1, 2] * _I1]),
            np.stack([-t[2, 0] * _I1]), np.stack([-t[2, 1] * _I1])]


def _run_uniform(eps, mu=None):
    mee = wavebit.rcwa(fto=[0, 0], **_MAT_KW)
    z = _z_row(eps) if np.any(eps[2, :2]) or np.any(eps[:2, 2]) else []
    return mee.solve_for_conv(_MAT_KW['wavelength'], *_in_plane(eps), *z,
                              mu=None if mu is None else tuple(_in_plane(mu)))


def _run_patterned():
    """A pillar of a real symmetric tensor in vacuum, through the Popov-Neviere builder.

    The unpacking is written out rather than splatted on purpose. The builders return
    ``(epx, epy, epxy, epyx, epz_i)`` and :meth:`solve_for_conv` takes
    ``(epx, epy, epz_i, epxy, epyx)`` — both documented, and different. Splatting one into
    the other puts the inverted zz slot where epxy belongs, and nothing objects: the solve
    returns a full set of plausible efficiencies that happen to sum to 1.9. That is how this
    test was first written, and the energy assertion below is what caught it.
    """
    from wavebit._core.factorization import to_conv_mat_raster_pn
    grid, fto = 48, 4
    eps = np.zeros((1, grid, grid, 3, 3), dtype=complex)
    for i in range(3):
        eps[..., i, i] = 1.0
    eps[0, 14:34, 18:30] = _tensor(2.25, 2.25, 2.89)
    epx, epy, epxy, epyx, epz_i = to_conv_mat_raster_pn(eps, fto, fto)
    mee = wavebit.rcwa(fto=[fto, fto], **_MAT_KW)
    return mee.solve_for_conv(_MAT_KW['wavelength'], epx, epy, epz_i, epxy, epyx)


MATERIAL_RUNS = {
    'isotropic': lambda: _run_uniform(_tensor(2.6, 2.6, 2.6)),
    'uniform_anisotropic': lambda: _run_uniform(_tensor(2.3, 3.0, 2.6, xy=0.2, yx=0.2)),
    'magnetic': lambda: _run_uniform(_tensor(2.6, 2.6, 2.6), mu=_tensor(1.4, 1.4, 2.2)),
    'z_coupled': lambda: _run_uniform(_tensor(2.6, 2.6, 2.6, xz=0.4, zx=0.4)),
    'patterned_anisotropic': _run_patterned,
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
    with warnings.catch_warnings():
        warnings.simplefilter('error', wavebit.UnphysicalResultWarning)
        res = MATERIAL_RUNS[case]()
    total = float(np.asarray(res.res.R).sum() + np.asarray(res.res.T).sum())
    assert abs(total - 1.0) < 1e-6, f'{case}: R + T = {total!r} on a lossless passive layer'


def test_a_patterned_z_row_has_no_builder_to_come_from():
    """``patterned_z_coupled: False`` is structural, so it is checked as structure.

    Nothing raises if a z-coupled tensor raster goes to the Popov-Neviere builder: it returns
    the same five in-plane slots and the z row is simply dropped. The honest statement of the
    gap is that no builder emits the four z-row slots from a raster at all, and that is what
    this pins. The day one does, this fails — which is the day the flag should be revisited.

    Worth flagging for whoever gets there: silently dropping the row sits oddly beside the
    rest of this solver, which refuses rather than ignores.
    """
    from wavebit._core import factorization
    cell = np.zeros((1, 16, 16, 3, 3), dtype=complex)
    for i in range(3):
        cell[..., i, i] = 1.0
    cell[0, 4:12, 4:12] = _tensor(2.9, 2.6, 2.9, xy=0.3, yx=0.3, xz=0.4, zx=0.4)
    for name in (n for n in dir(factorization) if n.startswith('to_conv_mat')):
        try:
            out = getattr(factorization, name)(cell, 2, 2)
        except Exception:
            continue                      # not a tensor-raster builder; nothing to claim
        assert len(out) == 5, (
            f'{name} returned {len(out)} slots — if it now emits the z row, patterned and '
            'z-coupled may be reachable and MATERIALS should say so')
