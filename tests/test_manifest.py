"""The capability manifest must describe the solver that exists, not the one it described.

The manifest is read by programs outside this process, so a claim that drifts is worse than
one that was never made: the consumer keeps answering confidently with a fact the solver
stopped honouring, and nothing raises anywhere. These tests are the only thing standing
between that and a published document.

**They are deliberately not mirror tests.** :mod:`wavebit.manifest` reads
``_core/engine.py``'s refusal tables at import, so asserting that the two agree would be
asserting that a variable equals itself. What is checked instead is the manifest against
*runtime behaviour*: build the structure each spelling describes and confirm the solver assigns
the path claimed, then ask for each option the manifest says is unavailable and confirm the
build refuses it. That closes the loop the tables sit in the middle of.
"""
import json
import warnings

import numpy as np
import pytest

import wavebit
from wavebit.manifest import (LATTICE, MATERIAL_QUANTITY, MATERIALS, SPELLINGS,
                              cited_sources, manifest)
from wavebit._core.enums import (Backend, Factorization, FourierCoefficients, NormalMethod,
                                   Tetm)

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

KW = dict(backend=0, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
          truncation_order=[4, 4], period=[1.0, 1.0], wavelength=1.1, thickness=[0.3])

#: A hexagonal basis, for the two refusals a skewed cell carries.
HEX = np.column_stack((np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])))

_N = 32
_axis = (np.arange(_N) + 0.5) / _N - 0.5
_X, _Y = np.meshgrid(_axis, _axis)
_RASTER = np.ones((1, _N, _N))
_RASTER[0, _X ** 2 + _Y ** 2 <= 0.3 ** 2] = 6.25          # permittivity: 2.5**2

#: One structure per spelling in the manifest, written the way the manifest says to write it.
#: The point is that these are transcribed *from* the manifest's `structure` field — if that
#: description stops matching what the setter accepts, this is where it shows. Every material
#: number here is a relative permittivity, which is the other half of that description and
#: what `material_quantity` publishes.
STRUCTURES = {
    'raster': _RASTER,
    'smooth': [{'background': 1.0,
                'shapes': [{'type': 'circle', 'cx': 0.0, 'cy': 0.0,
                            'lx': 0.6, 'material': 6.25}]}],
    'boundary': [{'background': 1.0, 'shapes': [
        {'type': 'rectangle', 'cx': -0.10, 'cy': -0.10, 'lx': 0.34, 'ly': 0.34,
         'angle': 0.0, 'material': 4.0},
        {'type': 'rectangle', 'cx': 0.06, 'cy': 0.06, 'lx': 0.34, 'ly': 0.34,
         'angle': 0.3, 'material': 9.0},
    ]}],
}

PATHS = {entry['path']: entry for entry in manifest()['paths']}


# ------------------------------------------------------- the spellings select the paths

@pytest.mark.parametrize('spelling', SPELLINGS, ids=[s['path'] for s in SPELLINGS])
def test_each_spelling_selects_the_modeling_type_it_claims(spelling):
    """`structure`'s *type* is the path selector — there is no keyword to check it against.

    So the manifest's prose description of how to spell a structure is load-bearing in a way
    a keyword's would not be: get it wrong and a consumer sends users down a different path
    than the one whose trade-offs it just explained.
    """
    solver = wavebit.rcwa(structure=STRUCTURES[spelling['path']], **KW)
    assert solver.modeling_type == spelling['modeling_type']


def test_the_manifest_covers_every_spelling_the_setter_accepts():
    """A path the solver can assign but the manifest omits is a path nobody is told about."""
    assigned = {wavebit.rcwa(structure=u, **KW).modeling_type for u in STRUCTURES.values()}
    assert assigned == {s['modeling_type'] for s in SPELLINGS}


# ------------------------------------------------------- unavailable options are refused

def _unavailable(path, axis, enum_cls):
    """The values of one axis the manifest says this path does not implement."""
    return [m.value for m in enum_cls if m.value not in PATHS[path][axis]]


@pytest.mark.parametrize('path', sorted(STRUCTURES))
def test_a_factorization_the_manifest_withholds_is_refused(path):
    for value in _unavailable(path, 'factorization', Factorization):
        with pytest.raises(Exception) as excinfo:
            wavebit.rcwa(structure=STRUCTURES[path], factorization=value, **KW).solve()
        assert value in str(excinfo.value), (
            f'{path}: refused factorization={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(STRUCTURES))
def test_a_series_the_manifest_withholds_is_refused(path):
    for value in _unavailable(path, 'fourier_coefficients', FourierCoefficients):
        with pytest.raises(Exception) as excinfo:
            wavebit.rcwa(structure=STRUCTURES[path], fourier_coefficients=value,
                           **KW).solve()
        assert value in str(excinfo.value), (
            f'{path}: refused fourier_coefficients={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(STRUCTURES))
def test_a_normal_method_the_manifest_withholds_is_refused(path):
    """The shape paths list no raster construction; naming one there must refuse."""
    for value in _unavailable(path, 'normal_method', NormalMethod):
        with pytest.raises(Exception) as excinfo:
            wavebit.rcwa(structure=STRUCTURES[path], normal_method=value, **KW).solve()
        assert value in str(excinfo.value), (
            f'{path}: refused normal_method={value!r} without naming it')


@pytest.mark.parametrize('path', sorted(STRUCTURES))
def test_an_option_the_manifest_offers_actually_builds(path):
    """The other half — a path must not be described as implementing what it refuses."""
    for value in PATHS[path]['factorization']:
        wavebit.rcwa(structure=STRUCTURES[path], factorization=value, **KW).solve()
    for value in PATHS[path]['fourier_coefficients']:
        wavebit.rcwa(structure=STRUCTURES[path], fourier_coefficients=value, **KW).solve()
    for value in PATHS[path]['normal_method']:
        # The manifest's own caveat: the axis is read only under factorization='nvf'.
        wavebit.rcwa(structure=STRUCTURES[path], factorization='nvf',
                       normal_method=value, **KW).solve()


# ------------------------------------------------------- the lattice refusals

def test_li_is_refused_on_a_skewed_cell():
    with pytest.raises(NotImplementedError, match="factorization='nvf'"):
        wavebit.rcwa(structure=STRUCTURES['raster'], lattice=HEX, factorization='li',
                       **KW).solve()


def test_the_manifest_lists_every_skewed_refusal():
    """Guards against one being dropped from the document while the solver still enforces it."""
    listed = ' '.join(r['what'] for r in LATTICE['refused_on_oblique'])
    # The block path was the other entry until it was removed; both spellings must
    # have lost it together, or the document tells a consumer to avoid a path that
    # no longer exists.
    assert 'block' not in listed
    assert LATTICE['paths_refused_on_oblique'] == ()
    assert "factorization='li'" in listed
    # `normal_method` was the third entry here until 2026-08-21, when all four constructions
    # gained the cell metric. What is asserted now is that it is gone from *both* spellings --
    # a stale entry in either would tell a consumer to avoid the option that is in fact the
    # one to recommend on a skewed cell.
    assert 'normal_method' not in listed
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
    per-path ``normal_method`` list omits it in a minimal install -- so this case follows the
    same probe rather than asserting an option that environment does not have.
    """
    if method == 'contour':
        pytest.importorskip('contourpy',
                            reason="normal_method='contour' needs the optional dependency")
    assert method not in LATTICE['normal_methods_refused_on_oblique']
    result = wavebit.rcwa(structure=STRUCTURES['raster'], lattice=HEX, factorization='nvf',
                          normal_method=method, **KW).solve()
    assert np.isfinite(np.asarray(result.T)).all()


def test_a_skewed_cell_still_builds_on_the_paths_the_manifest_leaves_open():
    for path in ('raster', 'smooth', 'boundary'):
        wavebit.rcwa(structure=STRUCTURES[path], lattice=HEX, factorization='nvf',
                       **KW).solve()


# ------------------------------------------------------- the document itself

def test_manifest_is_json_round_trippable():
    """A consumer receives this over the wire; anything unserialisable is a runtime failure."""
    assert json.loads(json.dumps(manifest())) == json.loads(json.dumps(manifest()))


def test_enum_lists_match_the_enums():
    enums = manifest()['enums']
    assert enums['backend'] == [m.name.lower() for m in Backend]
    assert enums['tetm'] == [m.name.lower() for m in Tetm]
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
    ``tests/test_material_convention_anchor.py`` gives at length: an oracle that spelled the
    quantity the way the API does could be co-migrated into agreement with a wrong flip.
    """
    assert MATERIAL_QUANTITY['quantity'] == 'permittivity'
    assert MATERIAL_QUANTITY['relative'] is True

    eps_sub = 12.25
    kw = dict(backend=0, top=1.0, bot=eps_sub, theta=0.0, phi=0.0, tetm='te',
              truncation_order=[0, 0], period=[1.0], wavelength=0.5, thickness=[0.0],
              structure=np.ones((1, 1, 1)))
    got = float(np.sum(np.asarray(wavebit.rcwa(**kw).solve().R)))

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
    assert wavebit.eps_from_index(3.5) == 12.25
    assert complex(wavebit.index_from_eps(12.25)) == pytest.approx(3.5)


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
    res = wavebit.rcwa(structure=_RASTER, **KW).solve()
    sub = res.given if res.given is not None else res.s_inc

    for name in list(m['efficiencies']['total']) + list(m['efficiencies']['by_outgoing_polarization']):
        arr = np.asarray(getattr(sub, name))
        assert np.isrealobj(arr), f'{name} is published as power and must be real'
    for name in m['amplitudes']['fields']:
        arr = np.asarray(getattr(sub, name))
        assert np.iscomplexobj(arr), f'{name} is published as an amplitude and must be complex'
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
    elsewhere (``tests/test_backside_incidence.py``, the RETICOLO ``inc_bottom`` golden);
    this closes the document-vs-solver loop only.
    """
    m = manifest()['results']['incidence_side']
    asym = np.array([[[9.0] * 16 + [1.0] * 16]])
    kw = dict(KW, phi=None, psi=None, truncation_order=[4, 0], bot=4.0, structure=asym, thickness=[0.3])

    got = {}
    for side in m['values']:
        res = wavebit.rcwa(incidence_side=side, **kw).solve()
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

    via_pair = wavebit.rcwa(structure=_RASTER, polarization=(np.sin(0.4), np.cos(0.4)),
                              **kw).solve()
    assert via_pair.given is not None, 'the pair must fill the specified-incidence slot'

    via_psi = wavebit.rcwa(structure=_RASTER, psi=0.4, **kw).solve()
    np.testing.assert_allclose(via_pair.given.R, via_psi.given.R, atol=1e-14)

    scaled = wavebit.rcwa(structure=_RASTER, polarization=(5 * np.sin(0.4), 5 * np.cos(0.4)),
                            **kw).solve()
    np.testing.assert_allclose(scaled.given.R, via_pair.given.R, atol=1e-14)

    with pytest.raises(ValueError, match='psi and polarization'):
        wavebit.rcwa(structure=_RASTER, psi=0.4, polarization=(1, 0), **kw)

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
_MAT_KW = dict(backend=0, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
               period=(1.0, 1.0), wavelength=1.1, thickness=[0.3])


def _tensor(xx, yy, zz, xy=0.0, yx=0.0, xz=0.0, zx=0.0, yz=0.0, zy=0.0):
    return np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]], dtype=complex)


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
    mee = wavebit.rcwa(truncation_order=[0, 0], **_MAT_KW)
    z = _z_row(eps) if np.any(eps[2, :2]) or np.any(eps[:2, 2]) else []
    return mee.solve_conv(*_in_plane(eps), *z,
                          mu=None if mu is None else tuple(_in_plane(mu)))


def _run_patterned():
    """A pillar of a real symmetric tensor in vacuum, through the Popov-Neviere builder.

    The unpacking is written out rather than splatted, which since 2026-08-18 is a style
    choice rather than a requirement: the builders and :meth:`solve_conv` now agree on
    ``(epx, epy, epz_i, epxy, epyx)``, so ``solve_conv(*build(...))`` is correct too.
    It was not. The builders grouped the in-plane pair together and put ``epz_i`` last, so
    splatting landed the inverted zz matrix in the ``epxy`` slot and nothing objected — the
    solve returned a full set of plausible efficiencies summing to 1.9. That is how this test
    was first written, and the energy assertion below is what caught it. ``tests/test_slot_order.py``
    now pins the agreement for every builder; this case is left spelled out because naming the
    slots is what makes the structure under test readable.
    """
    from wavebit._core.factorization import to_conv_mat_raster_pn_eps
    grid, truncation_order = 48, 4
    eps = np.zeros((1, grid, grid, 3, 3), dtype=complex)
    for i in range(3):
        eps[..., i, i] = 1.0
    eps[0, 14:34, 18:30] = _tensor(2.25, 2.25, 2.89)
    epx, epy, epz_i, epxy, epyx = to_conv_mat_raster_pn_eps(eps, truncation_order, truncation_order)
    mee = wavebit.rcwa(truncation_order=[truncation_order, truncation_order], **_MAT_KW)
    return mee.solve_conv(epx, epy, epz_i, epxy, epyx)


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
    total = float(np.asarray(res.given.R).sum() + np.asarray(res.given.T).sum())
    assert abs(total - 1.0) < 1e-6, f'{case}: R + T = {total!r} on a lossless passive layer'


def _z_coupled_cell(**over):
    cell = np.zeros((1, 16, 16, 3, 3), dtype=complex)
    for i in range(3):
        cell[..., i, i] = 1.0
    cell[0, 4:12, 4:12] = _tensor(2.9, 2.6, 2.9, xy=0.3, yx=0.3, **over)
    return cell


def test_a_patterned_z_row_is_refused_by_the_builder_that_cannot_read_it():
    """``patterned_z_coupled: False`` is a refusal now, not an omission, so it is pinned as one.

    It used to be dropped. The Popov-Neviere rearrangement closes over the in-plane 2x2 block,
    so a z-coupled tensor came back **bit-identical** to the same tensor with the z row zeroed
    -- all five slots -- and conserved energy. Against the 4ff path, which holds a z row
    exactly on a uniform layer, the cost is quadratic in the coupling: ``T`` in error by 3.3e-9
    at ``eps_xz = 1e-6`` and 1.9e-2 at 0.8.

    Written by name rather than by sweeping ``dir(factorization)``. The sweep this replaced
    wrapped its call in ``except Exception: continue``, so the moment a builder started
    *refusing*, the sweep skipped it and went green while asserting nothing about the two
    builders it was written for.
    """
    from wavebit._core.factorization import (to_conv_mat_raster_pn_eps,
                                             to_conv_mat_raster_pn_mu)

    for build in (to_conv_mat_raster_pn_eps, to_conv_mat_raster_pn_mu):
        with pytest.raises(ValueError, match='z row or column'):
            build(_z_coupled_cell(xz=0.4, zx=0.4), 2, 2)
        with pytest.raises(ValueError, match='z row or column'):
            build(_z_coupled_cell(yz=0.4, zy=0.4), 2, 2)

        # `zz` is read, and is not what the refusal is about -- changing it moves `epz_i` by
        # 1.4e-1, so a guard that swept the whole z row would refuse every uniaxial layer.
        assert len(build(_z_coupled_cell(), 2, 2)) == 5


def test_no_builder_emits_the_four_z_row_slots():
    """The other half, and the one that still has teeth for a builder written later.

    The refusal above is about the builder that exists. This is about the flag: nothing in
    ``factorization`` returns more than the five in-plane slots from a raster, which is what
    makes ``patterned_z_coupled: False`` structural rather than a policy. The day something
    returns nine, this fails -- and that is the day the flag should be revisited.
    """
    from wavebit._core import factorization
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

    for name, backend in (('numpy', 0), ('jax', 1), ('torch', 2)):
        claim = published[name]
        if not claim['available']:
            with pytest.raises(ImportError):
                __import__({1: 'jax', 2: 'torch'}[backend])
            continue
        if backend:
            pytest.importorskip({1: 'jax', 2: 'torch'}[backend])

        solver = wavebit.rcwa(backend=backend, structure=np.ones((1, 4, 4)) * 2.0,
                              period=[400.0, 400.0], truncation_order=[1, 1],
                              thickness=[150.0], wavelength=900.0, top=1.0, bot=1.0)
        assert type(solver)._DIFFERENTIABLE is claim['differentiable'], name

        # Both are present on every backend; on a non-differentiating one they exist to refuse.
        assert hasattr(solver, 'grad') and hasattr(solver, 'fit'), name

        # And then ASK, because the line above is circular on its own: `_DIFFERENTIABLE` and
        # the published claim are both read off `DIFFERENTIABLE_BACKENDS`, so they agree
        # whatever the table says. A table declaring that NumPy differentiates passed every
        # test in this file until this call went in.
        if claim['differentiable']:
            gradient = _gradient_of_a_transmittance(solver, backend)
            assert np.any(np.asarray(gradient) != 0), (
                f'{name} is published as differentiable and returned an all-zero gradient')
        else:
            with pytest.raises(NotImplementedError, match=r'automatic differentiation'):
                solver.grad(['structure'], solver.solve, lambda result: result.T.sum())


def _gradient_of_a_transmittance(solver, backend):
    """d(sum T)/d(structure) through the backend's own `grad`, detached for comparison."""
    import functools

    if backend == 1:
        import jax.numpy as jnp
        wavebit.enable_x64()
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
    solver = wavebit.rcwa(backend=0, structure=np.ones((1, 4, 4)), period=[400.0, 400.0],
                          truncation_order=[1, 1], thickness=[150.0], wavelength=900.0,
                          top=1.0, bot=1.0)
    for name in ('grad', 'fit'):
        with pytest.raises(NotImplementedError) as caught:
            getattr(solver, name)(['structure'], None, None)
        message = str(caught.value)
        assert name in message, message
        assert 'backend=1' in message and 'backend=2' in message, message
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
