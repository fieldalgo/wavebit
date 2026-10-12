"""Gate 8 -- three backends on the stage-3 monitor scenes, and each pinned to its own golden.

Gates 5, 7 and the stage-2 gate did this for the walls, the absorbers and materials, and the
sources. Stage 3 added one kernel observer -- a running DFT of named components over a region,
carried in the scan (``DFTRegion``) -- and everything else is post-processing of its
accumulators after the run, in the backend's own arrays: flux, Field phasors, the far-field
transform, the Diffraction orders with their direction split, Yee kz and s/p frame, and the
sources' spectra. None of it is reached by a stage-2 scene. So this gate runs the five scenes
of ``tests/references/_fd_stage3_cases.py``:

    flux_dipole   two dipoles in a 26^3 PML box: a Flux box, a Flux plane, a Field plane
    bloch_stack   a 25-degree p PlaneWave in a Bloch cell over glass: two Diffraction planes
    lamellar      a cell-snapped Region grating at normal incidence: three orders
    farfield      a magnetic and an electric dipole in a FarField box
    conical       a Region grating under a conical PlaneWave, y an Axis1D axis (complex)

**Two halves, and the second is the one that can fail alone.** NumPy, JAX and Torch are held
to each other at 1e-12 of each array's peak on every DFT accumulator (whole), every result
built from them, every source spectrum, the stage-2 reference series and the whole final
fields (measured 2026-09-19, worst array of each scene: JAX and Torch 1e-15 to 3e-14 off
NumPy, the worst a dipole's ``power()``, where the end correction divides by ``Omega``). Each
backend is then held to what *it* produced when ``fd_golden_stage3.npz`` was frozen
(``tests/references/interface/freeze_fd_stage3.py``). The mutants at the bottom make the
argument executable: the continuum kz in the de-embedding, the far field's magnetic current
with its sign flipped and the DFT's time convention flipped each keep all three backends
agreeing and put every one of them far outside the pin -- and the first two leave every DFT
accumulator bit-identical to the same backend's unmutated run, so the pin on the *results*,
not only on the accumulators, is what sees them. Same-run invariants are checked exactly;
comparisons with the archived environment keep the gate's 1e-12 budget.

**The archive is read before it is trusted.** Its content digest is recomputed, the scene
behind each key is shown to be the scene built here, and the frozen numbers are held to what
they are known to satisfy exactly: the lowered DFT frequencies are ``2 pi / wavelength`` and
the Axis1D factor is ``i k dx`` with the ``k`` of ``Bloch.from_source``; a Flux box is the sum
of its six face shares; ``R = R_s + R_p``; fixed ``k_par`` makes ``sin(theta_at) / wavelength``
one number across the band; the far-field intensity is ``(|E_theta|^2 + |E_phi|^2) / 2`` in
vacuum; the lamellar grating's p arrays and ``Ex``/``Hy`` accumulators are exact zeros (TE by
symmetry) and its ``+1`` and ``-1`` orders equal (the tooth is centred); and no frozen array is
round-off, the stage-2 README's trap (a first draft of the case list had four). The stack's
cross-polarised arrays (``r_s``, ``t_s``, ``R_s``, ``T_s`` of its p wave) are round-off since the
PlaneWave launches the grid's own p wave (``src/wavebit/fd/_lower/sources.py::plane_wave_drive``),
so they are not frozen and are held under a line live instead
(:func:`test_a_layered_cell_carries_no_cross_polarisation_on_any_backend`).
"""
import json
import math
import pathlib
import sys
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from fd_helpers import backends, to_numpy
from wavebit.fd import _post
from wavebit.fd._lower import lower
from wavebit.fd._run import _results, execute
from wavebit.fd._runtime import runtime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'references'))
from _fd_stage3_cases import (BACKENDS, EXPECTED, EXPECTED_WARNINGS, FAR_ANGLES,   # noqa: E402
                              GOLDEN, GRATING_WAVELENGTHS, RUN_WARNING,
                              SERIES_FIELDS, TOL, WAVELENGTHS, capture, conical_source,
                              content_digest, describe, lowered, relative_gap,
                              sample_indices, scenes, solve, bloch_stack_source)

SCENES = scenes()
NAMES = tuple(SCENES)

#: A frozen array is either exactly 0.0 on every backend (a symmetry zero) or at least this
#: large in absolute terms -- every scene is driven at order one, and the smallest real array
#: frozen is 1.3e-2 (the bloch_stack's ``Flux`` plane). The four Hz accumulators of the first
#: draft sat at 1e-17; the bloch_stack's ``R_s`` sat at 1.8e-5 until the PlaneWave launched the
#: grid's own p wave (2026-09-19), and is round-off now and not frozen.
ROUND_OFF_LINE = 1e-10

_MARKS = {p.values[0]: list(p.marks) for p in backends()}
INSTALLED = tuple(b for b, marks in _MARKS.items() if not marks)
_RUNS = {}


def quiet(fn, *args, **kwargs):
    """``fn(*args)`` with the warnings these scenes raise on purpose silenced."""
    with warnings.catch_warnings():
        for category in EXPECTED:
            warnings.simplefilter('ignore', category)
        return fn(*args, **kwargs)


def plan_of(name):
    return quiet(fd.plan, SCENES[name])


def result(name, backend):
    """``(res, state)`` of scene ``name`` on ``backend``, once per process."""
    key = (name, backend)
    if key not in _RUNS:
        _RUNS[key] = quiet(solve, SCENES[name], backend)
    return _RUNS[key]


def pairs():
    """Every pair of backends, skipped when either half is not installed."""
    out = []
    for a, b in (('numpy', 'jax'), ('numpy', 'torch'), ('jax', 'torch')):
        out.append(pytest.param((a, b), id=f'{a}-{b}', marks=(_MARKS[a] + _MARKS[b])[:1]))
    return out


@pytest.fixture(scope='module')
def golden():
    with np.load(GOLDEN, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    metadata = json.loads(str(arrays.pop('metadata')))
    return arrays, metadata


def _per_backend(arrays, name, backend):
    prefix = f'{name}__{backend}__'
    return {k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}


def _frequency_monitors(scene):
    return [m.name for m in scene.monitors if not isinstance(m, fd.Record)]


# ---------------------------------------------------------------------------------------------
# The archive, and the scenes behind it
# ---------------------------------------------------------------------------------------------

def test_the_golden_is_intact_and_says_where_it_came_from(golden):
    arrays, meta = golden
    assert content_digest(arrays) == meta['content_sha256']
    assert meta['backends'] == list(BACKENDS)
    assert set(meta['scenes']) == set(NAMES)
    assert meta['tolerance'] == TOL
    for key in ('python', 'wavebit', 'numpy', 'jax', 'torch'):
        assert isinstance(meta['versions'][key], str) and meta['versions'][key]
    for key in ('generator_sha256', 'cases_sha256', 'shared_cases_sha256',
                'sampling_cases_sha256'):
        assert len(meta[key]) == 64
    for name in NAMES:
        assert arrays[f'{name}__samples'].shape[1] == 4
        for backend in BACKENDS:
            got = _per_backend(arrays, name, backend)
            for key in ('final__E', 'final__H', 'energy'):
                assert key in got
            for region, _, labels in meta['scenes'][name]['regions']:
                for label in labels + ['energy']:
                    assert f'dft__{region}__{label}' in got, (name, backend, region, label)
            for monitor, _ in meta['scenes'][name]['monitors']:
                assert any(k.startswith(f'result__{monitor}__') for k in got), (name, monitor)
            for source, _ in meta['scenes'][name]['sources']:
                assert any(k.startswith(f'source__{source}__') for k in got), (name, source)


def test_the_archive_covers_every_stage_three_path(golden):
    """The five scenes are not five copies of one: each carries the thing it is named for.

    Written as an assertion rather than left to the docstring, because a case list that quietly
    lost its box, its Axis1D face or its second order would still pass every comparison below
    -- against a golden frozen from the same emptied scene.
    """
    arrays, meta = golden
    s = meta['scenes']
    monitors = {name: dict(map(tuple, s[name]['monitors'])) for name in NAMES}
    assert monitors['flux_dipole'] == {'box': 'Flux', 'plane': 'Flux', 'field': 'Field'}
    faces = sorted(r for r, m, _ in s['flux_dipole']['regions'] if m == 'box')
    assert faces == sorted(f'box::{a}{side}' for a in 'xyz' for side in '+-'), faces
    for _, _, labels in s['flux_dipole']['regions']:
        if labels != ['Ex', 'Hy']:
            assert labels == ['Eu', 'Ev', 'Hu', 'Hv'], labels            # the 2-cell face
    assert s['flux_dipole']['dtype'] == 'float64'
    assert s['flux_dipole']['terms'] == ['DipoleTerm', 'DipoleTerm']
    assert monitors['farfield'] == {'ff': 'FarField'}
    assert len([r for r, m, _ in s['farfield']['regions'] if m == 'ff']) == 6
    for name in ('bloch_stack', 'lamellar', 'conical'):
        assert monitors[name]['r'] == monitors[name]['t'] == 'Diffraction'
        assert s[name]['terms'] == ['ColumnTerm']
        assert s[name]['spans'], 'a structure under the sheet'
    assert s['bloch_stack']['dtype'] == s['conical']['dtype'] == 'complex128'
    # the Region structures cut x (and the one-cell y), not only z
    for name in ('lamellar', 'conical'):
        assert {row[0] for row in s[name]['spans']} == {'box.x', 'box.y', 'box.z'}, s[name]['spans']
    # three orders at the lamellar grating, both couplings at the conical one
    for name in ('lamellar', 'conical'):
        R = arrays[f'{name}__numpy__result__r__R']
        assert R.shape == (len(GRATING_WAVELENGTHS), 1, 3), R.shape
        assert np.all(R > 1e-3), R
    for monitor, key in (('r', 'r_s'), ('r', 'r_p'), ('t', 't_s'), ('t', 't_p')):
        assert np.all(np.abs(arrays[f'conical__numpy__result__{monitor}__{key}']) > 1e-4), key
    assert 'conical__lowered__axis1d__y' in arrays
    assert 'bloch_stack__lowered__axis1d__y' not in arrays


@pytest.mark.parametrize('name', NAMES)
def test_the_scene_is_the_one_frozen(golden, name):
    """Static content by the skeleton and the regions; the leaves by the lowered rows below."""
    arrays, meta = golden
    p = plan_of(name)
    assert describe(SCENES[name], p) == meta['scenes'][name]
    assert np.array_equal(arrays[f'{name}__samples'], sample_indices(p.shape))


@pytest.mark.parametrize('name', NAMES)
def test_the_scene_warns_about_what_it_does_on_purpose(name):
    """The plan's warnings, and the run's: every frequency-domain monitor is named unsettled.

    Silenced everywhere else in this file, so asserted here: none can stop firing unnoticed, no
    scene raises one it was not built to, and a run that stopped early is always said to have.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        p = fd.plan(SCENES[name])
    got = sorted(type(w.message).__name__ for w in caught)
    assert got == sorted(EXPECTED_WARNINGS[name]), [str(w.message) for w in caught]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fd.run(SCENES[name], plan=p)
    unsettled = [str(w.message) for w in caught if type(w.message).__name__ == RUN_WARNING]
    assert unsettled, [str(w.message) for w in caught]
    for monitor in _frequency_monitors(SCENES[name]):
        assert any(repr(monitor) in m or f"'{monitor}'" in m for m in unsettled), \
            (monitor, unsettled)


@pytest.mark.parametrize('name', NAMES)
def test_the_lowered_frequencies_and_axis1d_factor_are_the_frozen_ones(golden, name):
    """Bit-level where the lowering is deterministic arithmetic: what a run is given."""
    arrays, _ = golden
    got = lowered(SCENES[name], plan_of(name))
    prefix = f'{name}__lowered__'
    want = {k: v for k, v in arrays.items() if k.startswith(prefix)}
    assert {f'{name}__{k}' for k in got} == set(want)
    for key, value in got.items():
        frozen = want[f'{name}__{key}']
        assert value.dtype == frozen.dtype and value.shape == frozen.shape, key
        assert relative_gap(value, frozen) < TOL, key


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('backend', backends())
def test_every_backend_lowers_the_same_frequencies(golden, name, backend):
    """``params['dft']`` and ``params['axis1d']`` are built per backend, outside the scan."""
    arrays, _ = golden
    params = quiet(lower, SCENES[name], plan_of(name), runtime(backend))
    for region, omega in params['dft'].items():
        want = arrays[f'{name}__lowered__dft__{region}']
        assert relative_gap(to_numpy(omega), want) < TOL, region
    for axis, value in (params.get('axis1d') or {}).items():
        assert relative_gap(to_numpy(value), arrays[f'{name}__lowered__axis1d__{axis}']) < TOL


def test_the_capture_is_what_fd_run_returns():
    """``solve`` is ``fd.run``'s own three calls: the results it freezes are ``fd.run``'s, bit
    for bit, so the archive cannot describe a path no user takes."""
    name = 'bloch_stack'
    res, state = result(name, 'numpy')
    p = plan_of(name)
    again = quiet(fd.run, SCENES[name], plan=p)
    samples = sample_indices(p.shape)
    mine = capture(res, state, samples)
    # ``Results`` carries no accumulator, so the DFT rows of ``fd.run``'s capture are read from
    # the same state; every other row is ``fd.run``'s own.
    theirs = capture(again, state, samples)
    assert set(theirs) == set(mine)
    for key, value in theirs.items():
        assert np.array_equal(value, mine[key]), key
    assert np.array_equal(to_numpy(again.final.E), to_numpy(state['E']))


# ---------------------------------------------------------------------------------------------
# What the frozen numbers are known to be right about
# ---------------------------------------------------------------------------------------------

def test_no_frozen_array_is_round_off(golden):
    """Every frozen array is an exact 0.0 on every backend or far above round-off.

    A frozen array of round-off is a pin on round-off: its own peak is the scale, so a
    harmless reordering moves it by O(1). The first draft of the case list froze four (the Hz
    accumulators of a lone Ez dipole's box faces: JAX and Torch 1.0 to 1.8 of their own peak
    off NumPy), which is why each dipole scene now carries a second dipole.
    """
    arrays, _ = golden
    for key, value in arrays.items():
        if key.endswith('__samples'):
            continue
        peak = float(np.max(np.abs(value)))
        assert peak == 0.0 or peak > ROUND_OFF_LINE, (key, peak)


@pytest.mark.parametrize('backend', backends())
def test_a_layered_cell_carries_no_cross_polarisation_on_any_backend(backend):
    """The stack is a slab in a Bloch cell under a conical (``phi = 30``) p wave: it couples no
    s, on the grid as in the continuum, once the PlaneWave launches the grid's own p wave. So
    its s arrays are round-off on every backend -- which is why ``capture`` leaves them out
    (``_fd_stage3_cases.cross_polarised``) -- and the co-polarised ones are order one. Before
    the drive carried the stagger phase they were ``|r_s| = 4.3e-3``, ``|t_s| = 8.8e-3``
    (the first freeze of this archive)."""
    res, _ = result('bloch_stack', backend)
    for monitor, amp, power in (('r', 'r', 'R'), ('t', 't', 'T')):
        orders = res[monitor].orders.incident
        co = np.max(np.abs(to_numpy(getattr(orders, f'{amp}_p'))))
        assert co > 0.1, (monitor, co)
        assert np.max(np.abs(to_numpy(getattr(orders, f'{amp}_s')))) < 1e-13 * co, monitor
        assert np.max(np.abs(to_numpy(getattr(orders, f'{power}_s')))) < 1e-24, monitor
    samples = sample_indices(res.plan.shape)
    keys = set(capture(res, result('bloch_stack', backend)[1], samples))
    assert not keys & {f'result__{m}__{k}' for m in 'rt'
                       for k in ('r_s', 't_s', 'R_s', 'T_s')}, keys


def test_the_lamellar_grating_is_te_and_symmetric_on_every_backend(golden):
    """Normal incidence, s, a tooth centred in its cell, one cell deep in y: the field is
    ``Ey``, ``Hx``, ``Hz`` only, so every p array and every ``Ex``/``Hy`` accumulator is an exact
    zero, and the ``+1`` and ``-1`` orders are mirror images."""
    arrays, _ = golden
    for backend in BACKENDS:
        got = _per_backend(arrays, 'lamellar', backend)
        for monitor in ('r', 't'):
            for key in ('r_p', 'R_p') if monitor == 'r' else ('t_p', 'T_p'):
                assert not np.any(got[f'result__{monitor}__{key}']), (backend, key)
            assert not np.any(got[f'dft__{monitor}__Eu']) and not np.any(got[f'dft__{monitor}__Hv'])
            assert not np.any(got[f'result__{monitor}__theta_at'])
            for key in ('R', 'T'):
                if f'result__{monitor}__{key}' in got:
                    R = got[f'result__{monitor}__{key}']
                    assert np.max(np.abs(R[..., 0] - R[..., 2])) < 1e-12 * np.max(R), (backend, key)


def test_a_flux_box_is_the_sum_of_its_faces_in_the_frozen_numbers(golden):
    arrays, _ = golden
    for name, monitor in (('flux_dipole', 'box'), ('farfield', 'ff')):
        for backend in BACKENDS:
            got = _per_backend(arrays, name, backend)
            faces = [v for k, v in got.items() if k.startswith(f'result__{monitor}__face__')]
            assert len(faces) == 6
            total = got[f'result__{monitor}__flux']
            assert np.max(np.abs(sum(faces) - total)) < 1e-14 * np.max(np.abs(total)), name


def test_the_frozen_orders_split_exactly_into_s_and_p(golden):
    arrays, _ = golden
    for name in ('bloch_stack', 'lamellar', 'conical'):
        for backend in BACKENDS:
            got = _per_backend(arrays, name, backend)
            for monitor, total in (('r', 'R'), ('t', 'T')):
                whole = got[f'result__{monitor}__{total}']
                # A layered cell's cross-polarised part is not frozen (round-off): it adds 0.
                parts = (got.get(f'result__{monitor}__{total}_s', 0.0)
                         + got.get(f'result__{monitor}__{total}_p', 0.0))
                assert np.max(np.abs(parts - whole)) < 1e-14 * np.max(whole), (name, total)


def test_the_frozen_angle_is_the_fixed_k_angle(golden):
    """``k_par`` is fixed across the band, so ``sin(theta_at) / wavelength`` is the band
    centre's ``sin(theta)`` at every wavelength (the source's own ``theta``, in degrees)."""
    arrays, _ = golden
    for name, source, ws in (('bloch_stack', bloch_stack_source(), WAVELENGTHS),
                             ('conical', conical_source(), GRATING_WAVELENGTHS)):
        want = math.sin(math.radians(source.theta)) / float(source.band.wavelength)
        for backend in BACKENDS:
            theta = arrays[f'{name}__{backend}__result__r__theta_at']
            got = np.sin(np.radians(theta)) / np.asarray(ws)
            assert np.max(np.abs(got - want)) < 1e-14, (name, backend, got, want)


def test_the_frozen_lowering_is_the_closed_form(golden):
    """``omega = 2 pi / wavelength`` (``c = 1``) for every region, and the Axis1D factor ``i k dx``
    with the ``k`` that ``Bloch.from_source(pw, axis1d='y')`` wrote for the conical wave."""
    arrays, meta = golden
    for name in NAMES:
        for region, monitor, _ in meta['scenes'][name]['regions']:
            m = next(m for m in SCENES[name].monitors if m.name == monitor)
            want = 2.0 * np.pi / np.asarray(m.wavelengths)
            got = arrays[f'{name}__lowered__dft__{region}']
            assert np.max(np.abs(got - want)) < 1e-14 * np.max(want), region
    k = fd.Bloch.from_source(conical_source(), axis1d='y')['y'].k
    dx = meta['scenes']['conical']['dx']
    factor = arrays['conical__lowered__axis1d__y']
    assert factor.shape == (1,) and abs(factor.item() - 1j * k * dx) < 1e-16


def test_the_frozen_far_field_intensity_is_its_fields(golden):
    """``U = r^2 |E|^2 / (2 eta)`` with ``eta = 1`` in vacuum: the intensity and the two field
    components were frozen from the same transform, and agree."""
    arrays, _ = golden
    for backend in BACKENDS:
        got = _per_backend(arrays, 'farfield', backend)
        U = got['result__ff__intensity']
        want = 0.5 * (np.abs(got['result__ff__E_theta']) ** 2 + np.abs(got['result__ff__E_phi']) ** 2)
        assert U.shape == (len(WAVELENGTHS), FAR_ANGLES[0].size)
        assert np.max(np.abs(U - want)) < 1e-14 * np.max(U), backend


# ---------------------------------------------------------------------------------------------
# The two halves: agreement, and each backend against its own golden
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('pair', pairs())
def test_the_backends_agree(name, pair):
    """Whole fields, every accumulator, every result, every source series, each pair, to TOL."""
    a, b = pair
    (ra, sa), (rb, sb) = result(name, a), result(name, b)
    assert ra.plan.skeleton == rb.plan.skeleton
    for field in 'EH':
        fa, fb = to_numpy(getattr(ra.final, field)), to_numpy(getattr(rb.final, field))
        assert fa.shape == fb.shape and fa.dtype == fb.dtype
        assert relative_gap(fa, fb) < TOL, field
    samples = sample_indices(ra.plan.shape)
    ca, cb = capture(ra, sa, samples), capture(rb, sb, samples)
    assert set(ca) == set(cb)
    for key in ca:
        assert ca[key].shape == cb[key].shape and ca[key].dtype == cb[key].dtype, key
        assert relative_gap(ca[key], cb[key]) < TOL, key
    for source, xa in ra.sources.items():
        xb = rb.sources[source]
        for field in SERIES_FIELDS:
            va, vb = getattr(xa, field), getattr(xb, field)
            assert (va is None) == (vb is None), (source, field)
            if va is not None:
                assert relative_gap(to_numpy(va), to_numpy(vb)) < TOL, (source, field)
    for monitor in _frequency_monitors(SCENES[name]):
        if isinstance(ra[monitor], fd.results.FieldResult):
            for c, coords in ra[monitor].coords.items():
                for u, w in zip(coords, rb[monitor].coords[c]):
                    assert np.array_equal(np.asarray(u), np.asarray(w)), (monitor, c)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('backend', backends())
def test_each_backend_reproduces_its_own_golden(golden, name, backend):
    arrays, _ = golden
    res, state = result(name, backend)
    got = capture(res, state, arrays[f'{name}__samples'])
    want = _per_backend(arrays, name, backend)
    assert set(want) == set(got)
    for key, value in got.items():
        assert value.dtype == want[key].dtype and value.shape == want[key].shape, key
        assert relative_gap(value, want[key]) < TOL, key


@pytest.mark.parametrize('name', ('flux_dipole', 'conical'))
@pytest.mark.parametrize('backend', backends())
def test_a_run_reproduces_itself_bit_for_bit(name, backend):
    """A pin needs one answer per backend: a second run is the first, byte for byte.

    The accumulators are carried state and the NumPy runtime keeps one DFT scratch buffer per
    accumulator under a prefixed key; a buffer reused without being zeroed, or shared between
    two regions, would stop the result being a function of its scene here.
    """
    res, state = result(name, backend)
    again, state2 = quiet(solve, SCENES[name], backend)
    samples = sample_indices(res.plan.shape)
    first, second = capture(res, state, samples), capture(again, state2, samples)
    for key, value in first.items():
        assert np.array_equal(value, second[key]), key


# ---------------------------------------------------------------------------------------------
# The mutants: what agreement cannot see and the pin can
# ---------------------------------------------------------------------------------------------

def _baseline_captures(golden, name):
    """Snapshot normal results before patching, including lazily evaluated transforms.

    Copy the arrays so a mutant cannot modify its baseline through shared state. Populate
    the run cache here as well: these tests must work without an earlier golden test.
    """
    arrays, _ = golden
    samples = arrays[f'{name}__samples']
    return {b: {k: v.copy() for k, v in capture(*result(name, b), samples).items()}
            for b in INSTALLED}


def _mutant_gaps(golden, name, runs, baseline, changed):
    """Backend agreement and archived gaps, with exact same-backend mutation invariants.

    ``changed(key)`` identifies the outputs affected by the mutant. All other arrays must
    equal the live baseline exactly, while archived comparisons retain the normal budget.
    """
    arrays, _ = golden
    samples = arrays[f'{name}__samples']
    caps = {b: capture(res, state, samples) for b, (res, state) in runs.items()}
    for a in caps:
        for b in caps:
            for key in caps[a]:
                assert relative_gap(caps[a][key], caps[b][key]) < TOL, (a, b, key)
    gaps = {}
    for backend, cap in caps.items():
        want = _per_backend(arrays, name, backend)
        assert set(cap) == set(baseline[backend]) == set(want), backend
        gaps[backend] = {}
        for key, value in cap.items():
            assert relative_gap(baseline[backend][key], want[key]) < TOL, (backend, key)
            gaps[backend][key] = relative_gap(value, want[key])
            if not changed(key):
                np.testing.assert_array_equal(value, baseline[backend][key],
                                              err_msg=f'{backend}: {key}')
                assert gaps[backend][key] < TOL, (backend, key, gaps[backend][key])
    return gaps


def _rebuild(name, backend):
    """The results of the cached run rebuilt from its final state: post-processing only."""
    res, state = result(name, backend)
    rt = runtime(backend)
    params = quiet(lower, SCENES[name], res.plan, rt)
    return quiet(_results, SCENES[name], res.plan, state, rt, params), state


@pytest.mark.parametrize('name', ('bloch_stack', 'conical'))
def test_mutant_the_continuum_kz_passes_agreement_and_fails_every_golden(golden, monkeypatch,
                                                                          name):
    """``_post.DIFFRACTION_KZ = 'continuum'``: the orders de-embedded with ``sqrt(n^2 w^2 -
    k_par^2)`` instead of the Yee kappa. A post-processing defect, so every DFT accumulator,
    every field and every source spectrum stays **bit-identical** to its live baseline, as do the
    efficiencies ``R`` and ``T`` (a wrong kz over a propagating distance is a phase) -- only the
    pin on the complex amplitudes sees it. Measured 2026-09-19, the same on every backend:
    ``r_p`` 2.1e-2 and ``t_p`` 1.4e-2 of their peaks on the stack (its s arrays are round-off
    and not frozen), 1.4e-2 to
    2.0e-2 on the conical grating (1.9e-2 / 1.7e-2 on the lamellar one), the ``(kz - kappa)``
    of 20 cells per wavelength over the half wavelength from each plane to its reference."""
    baseline = _baseline_captures(golden, name)
    amplitudes = ('result__r__r_s', 'result__r__r_p', 'result__t__t_s', 'result__t__t_p')
    monkeypatch.setattr(_post, 'DIFFRACTION_KZ', 'continuum')
    gaps = _mutant_gaps(golden, name, {b: _rebuild(name, b) for b in INSTALLED},
                        baseline, lambda key: key in amplitudes)
    for backend, g in gaps.items():
        amplitudes = tuple(k for k in amplitudes if k in g)     # the stack freezes p only
        assert len(amplitudes) == (2 if name == 'bloch_stack' else 4), amplitudes
        assert min(g[k] for k in amplitudes) > 1e-3, (backend, g)


def test_mutant_the_far_field_magnetic_current_sign_passes_agreement_and_fails_every_golden(
        golden, monkeypatch):
    """``_post.FARFIELD_M_SIGN = -1``: forward and backward lobes of each face trade places.

    The flux, the face shares, the power and every accumulator are untouched (none of them
    uses ``M``), bit for bit; the frozen intensity moves by 0.94 of its peak, ``E_theta`` by
    1.1 and ``E_phi`` by 1.4 (measured 2026-09-19, the same on every backend).
    """
    baseline = _baseline_captures(golden, 'farfield')
    transform = ('result__ff__intensity', 'result__ff__E_theta', 'result__ff__E_phi')
    monkeypatch.setattr(_post, 'FARFIELD_M_SIGN', -1.0)
    gaps = _mutant_gaps(golden, 'farfield', {b: _rebuild('farfield', b) for b in INSTALLED},
                        baseline, lambda key: key in transform)
    for backend, g in gaps.items():
        assert min(g[k] for k in transform) > 0.1, (backend, g)


def _mutated_solve(name, backend, mutate):
    """``solve`` with the lowered params passed through ``mutate`` -- the traced leaves only, so
    the plan, the skeleton and the jitted executor are the unmutated ones."""
    scene = SCENES[name]
    rt = runtime(backend)
    p = plan_of(name)
    params = mutate(quiet(lower, scene, p, rt))
    state = execute(p, params, rt)
    return quiet(_results, scene, p, state, rt, params), state


def test_mutant_the_time_convention_flipped_passes_agreement_and_fails_every_golden(golden):
    """Every region's ``omega`` negated in the params: the kernel accumulates the DFT of the
    ``e^{+iwt}`` convention, the complex conjugate of every phasor, while the post-processing
    still works in ``e^{-iwt}``. A lowering defect, so the fields and the dipoles' ``power()``
    (read from the source series, not from a region) stay bit-identical and only the pins on
    the accumulators and what is built from them see it. Measured 2026-09-19, the same on every
    backend: every accumulator 1.6 to 2.0 of its peak (twice its imaginary part), the Field
    phasors 1.6 to 1.8, the box flux 0.39 -- not invariant, although ``Re(E x conj H)`` of two
    conjugates is: the end correction is built from the static wavelengths in ``e^{-iwt}`` and
    no longer matches the conjugated sum."""
    def flip(params):
        return dict(params, dft={k: -v for k, v in params['dft'].items()})

    baseline = _baseline_captures(golden, 'flux_dipole')
    runs = {b: _mutated_solve('flux_dipole', b, flip) for b in INSTALLED}
    gaps = _mutant_gaps(
        golden, 'flux_dipole', runs, baseline,
        lambda key: (key.startswith('dft__') and not key.endswith('__energy'))
        or key.startswith('result__'))
    for backend, g in gaps.items():
        for key, gap in g.items():
            if key.startswith('dft__') and not key.endswith('__energy'):
                assert gap > 1.0, (backend, key, gap)
            elif key.startswith('result__'):
                assert gap > 0.1, (backend, key, gap)


def test_the_pin_resolves_a_part_in_a_billion_of_an_amplitude(golden):
    """The budget is doing work: both dipoles' amplitudes times ``1 + 1e-9`` are seen, at the
    size linear algebra says. The accumulators and fields are linear in the amplitudes, so they
    move by 1e-9 of their peak; a flux and a power are quadratic, so by 2e-9."""
    arrays, _ = golden

    def nudge(params):
        sources = {n: dict(e, amplitude=e['amplitude'] * (1.0 + 1e-9))
                   for n, e in params['sources'].items()}
        return dict(params, sources=sources)

    res, state = _mutated_solve('flux_dipole', 'numpy', nudge)
    got = capture(res, state, arrays['flux_dipole__samples'])
    want = _per_backend(arrays, 'flux_dipole', 'numpy')
    gaps = {k: relative_gap(v, want[k]) for k, v in got.items()}
    assert max(gaps.values()) < 1e-8, gaps
    assert math.isclose(gaps['dft__box::z+__Eu'], 1e-9, rel_tol=1e-3), gaps['dft__box::z+__Eu']
    assert math.isclose(gaps['result__box__flux'], 2e-9, rel_tol=1e-3), gaps['result__box__flux']
    assert math.isclose(gaps['source__d__power'], 2e-9, rel_tol=1e-3), gaps['source__d__power']
