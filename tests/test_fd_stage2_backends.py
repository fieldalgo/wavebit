"""Gate 6 -- three backends on the stage-2 source scenes, and each pinned to its own golden.

Gates 5 and 7 did this for the stage-0 walls and the stage-1 absorbers and materials. Stage 2
added the first things that *drive* the grid -- a waveform table built inside the traced
function, a dipole's stamp added into ``cH`` or ``cE``, and an incident column carried in a
term's memory that the TFSF box and the PlaneWave sheet read their corrections from -- and none
of it is exercised by a scene that starts from an initial field. So this gate runs the four
scenes of ``tests/references/_fd_stage2_cases.py``:

    dipole          an electric (B-spline) and a magnetic (tent) dipole in a 26^3 PML box
    plane_wave      a 20-degree p PlaneWave in a Bloch cell over a glass layer (complex)
    tfsf            a +x TFSF box in vacuum
    tfsf_substrate  a -z TFSF box on a glass half-space crossing the box

**Two halves, and the second is the one that can fail alone.** NumPy, JAX and Torch are held to
each other on every cell of the final E and H, on every ``Record`` series and on **every array
of every source's reference series** (``res.sources``), at 1e-12 of the array's peak (measured
2026-09-19: Torch bit-identical to NumPy on both TFSF scenes and 2.1e-15 / 3.3e-15 off on the
dipole and the PlaneWave, JAX 8.3e-16 to 4.0e-15 off). That agreement is necessary and nowhere near
sufficient -- a stamp normalised the wrong way or a TFSF face with the wrong sign moves all
three backends together -- so each backend is also held to what *it* produced when
``fd_golden_stage2.npz`` was frozen (``tests/references/interface/freeze_fd_stage2.py``). The two
mutants at the bottom make that argument executable: each keeps all three backends agreeing to
1e-12 and puts every one of them O(1) from its golden.

**The lowered numbers are pinned once, not per backend**: the stamp weights, the Bloch phase
pattern, the waveform scalars and the waveform table are what a run is *given*, the same on
every backend by construction, and every backend is shown to build them alike.

**The archive is read before it is trusted.** Its content digest is recomputed, the scene behind
each key is shown to be the scene built here, and the frozen numbers are held to what these
scenes have a closed-form answer for: the frozen drive of every backend is the band's closed
form ``w(t)`` at the slot's half step, the frozen stamp is a partition of unity whose centroid is
the dipole's centre, the frozen Bloch pattern is ``exp(i k_par . r)`` with the ``k_par`` that
``Bloch.from_source`` wrote, and the two TFSF scenes leave the scattered-field region at
round-off -- the one property of a TFSF box that is exact.
"""
import dataclasses
import json
import math
import pathlib
import sys
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from fd_helpers import backends, run_mutated, run_plan, scattered_region_peak, to_numpy
from wavebit.fd._kernel.grid import OFFSETS
from wavebit.fd._kernel.sources import ColumnTerm, DipoleTerm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'references'))
from _fd_stage2_cases import (BACKENDS, EXPECTED_WARNINGS, GOLDEN, SERIES_FIELDS,   # noqa: E402
                              TOL, capture, content_digest, lowered, plane_wave_source,
                              relative_gap, sample_indices, scenes)

SCENES = scenes()
NAMES = tuple(SCENES)

#: What the two mutants do to the pin, at least. Measured 2026-09-19, the worst array of the
#: scene on every backend: 1.9 of its peak for the sum-of-squares stamp, 1.2 for one TFSF face
#: with its sign flipped (which also leaks 0.62 into the scattered region).
MUTANT_FLOOR = 1e-3

#: A TFSF box leaves the scattered region at round-off: measured 0.0 on NumPy and Torch on both
#: scenes (the corrections cancel the incident field exactly, term for term) and 1.7e-16 /
#: 2.3e-16 on JAX. The line is set where floating-point reordering could still land and twelve
#: decades below a leaking face (0.62, the face-sign mutant).
LEAK_LINE = 1e-13

#: The warnings the scenes raise on purpose, silenced where a run is only a run.
EXPECTED = (fd.CoarseMeshWarning, fd.AbsorberProximityWarning)

_MARKS = {p.values[0]: list(p.marks) for p in backends()}
INSTALLED = tuple(b for b, marks in _MARKS.items() if not marks)
_RUNS = {}


def quiet_plan(scene):
    """``fd.plan`` with the warnings these scenes raise on purpose silenced."""
    with warnings.catch_warnings():
        for category in EXPECTED:
            warnings.simplefilter('ignore', category)
        return fd.plan(scene)


def result(name, backend):
    """The run of scene ``name`` on ``backend``, once per process."""
    key = (name, backend)
    if key not in _RUNS:
        with warnings.catch_warnings():
            for category in EXPECTED:
                warnings.simplefilter('ignore', category)
            _RUNS[key] = fd.run(SCENES[name], backend=backend)
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
    for key in ('generator_sha256', 'cases_sha256', 'shared_cases_sha256'):
        assert len(meta[key]) == 64
    for name in NAMES:
        assert arrays[f'{name}__samples'].shape[1] == 4
        for backend in BACKENDS:
            got = _per_backend(arrays, name, backend)
            for probe in meta['scenes'][name]['monitors']:
                assert f'series__{probe}' in got
            for key in ('final__E', 'final__H', 'energy'):
                assert key in got
            for source, _ in meta['scenes'][name]['sources']:
                assert any(k.startswith(f'source__{source}__') for k in got), (name, source)


def test_the_archive_covers_every_stage_two_path(golden):
    """The four scenes are not four copies of one: each carries the thing it is named for.

    Written as an assertion rather than left to the docstring, because a case list that quietly
    lost its magnetic dipole, its Bloch pattern or its substrate would still pass every
    comparison below -- against a golden frozen from the same emptied scene.
    """
    arrays, meta = golden
    s = meta['scenes']
    assert s['dipole']['terms'] == ['DipoleTerm', 'DipoleTerm']
    assert s['dipole']['dtype'] == 'float64'
    assert arrays['dipole__lowered__waveform'].shape == (2, s['dipole']['steps'])
    assert arrays['dipole__lowered__j__stamp'].shape == (4, 4, 4)       # off-node on all axes
    for key in ('drive', 'work', 'field'):
        for d in ('j', 'm'):
            assert f'dipole__numpy__source__{d}__{key}' in arrays
    assert s['plane_wave']['dtype'] == 'complex128'
    assert s['plane_wave']['terms'] == ['ColumnTerm']
    pattern = arrays['plane_wave__lowered__pw__pattern']
    assert pattern.shape[1] > 1 and pattern.shape[2] > 1               # Bloch on both axes
    assert s['plane_wave']['spans'], 'the glass layer under the sheet'
    for name in ('tfsf', 'tfsf_substrate'):
        assert s[name]['terms'] == ['ColumnTerm']
        assert len(s[name]['absorbing']) == 6
        for key in ('incident', 'incident_H'):
            assert arrays[f'{name}__numpy__source__t__{key}'].shape == (s[name]['steps'], 3)
    assert not s['tfsf']['spans'] and s['tfsf_substrate']['spans']


@pytest.mark.parametrize('name', NAMES)
def test_the_scene_is_the_one_frozen(golden, name):
    """Static content by the skeleton; the sources' leaves by the lowered rows below.

    The skeleton holds no leaf, so a helper that moved an amplitude, a phase or a delay would
    leave it unchanged and read as kernel drift; the frozen lowered scalars and the waveform
    table are what tell the two apart, and they are held in the next test.
    """
    arrays, meta = golden
    p = quiet_plan(SCENES[name])
    frozen = meta['scenes'][name]
    assert p.skeleton == frozen['skeleton']
    assert [p.nx, p.ny, p.nz] == frozen['cells']
    assert p.dtype == frozen['dtype'] and p.steps == frozen['steps']
    assert p.dt == frozen['dt'] and p.dx == frozen['dx']
    assert [type(t).__name__ for t in p.terms] == frozen['terms']
    assert [[s.name, type(s).__name__] for s in SCENES[name].sources] == frozen['sources']
    assert [list(f) for f in p.absorbing_faces] == frozen['absorbing']
    assert [list(s) for s in p.spans] == frozen['spans']
    assert np.array_equal(arrays[f'{name}__samples'], sample_indices(p.shape))


@pytest.mark.parametrize('name', NAMES)
def test_the_scene_warns_about_what_it_does_on_purpose(name):
    """The warnings silenced everywhere else in this file, asserted here, so none can stop
    firing unnoticed -- and no scene raises one it was not built to."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fd.plan(SCENES[name])
    got = sorted(type(w.message).__name__ for w in caught)
    assert got == sorted(EXPECTED_WARNINGS[name]), [str(w.message) for w in caught]


@pytest.mark.parametrize('name', NAMES)
def test_the_lowered_stamps_patterns_and_waveforms_are_the_frozen_ones(golden, name):
    """Bit-level where the lowering is deterministic arithmetic: what a run is given."""
    arrays, _ = golden
    got = lowered(SCENES[name], quiet_plan(SCENES[name]))
    prefix = f'{name}__lowered__'
    want = {k: v for k, v in arrays.items() if k.startswith(prefix)}
    assert {f'{name}__{k}' for k in got} == set(want)
    for key, value in got.items():
        frozen = want[f'{name}__{key}']
        assert value.dtype == frozen.dtype and value.shape == frozen.shape, key
        assert relative_gap(value, frozen) < TOL, key


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('backend', backends())
def test_every_backend_lowers_the_same_sources(golden, name, backend):
    """The stamp and the pattern are built per backend, outside the scan: on Torch through
    ``as_tensor``, on JAX with x64 lifted. An x64 that was not lifted, or a stamp written with
    a NumPy-only idiom, would land here and nowhere else."""
    arrays, _ = golden
    scene = SCENES[name]
    params = fd.lower(scene, quiet_plan(scene), backend)
    for source, entry in params['sources'].items():
        for key in ('f0', 'tau', 't0', 'amplitude', 'phase', 'delay', 'stamp', 'pattern'):
            if key not in entry:
                continue
            got = to_numpy(entry[key])
            want = arrays[f'{name}__lowered__{source}__{key}']
            assert got.shape == want.shape, (source, key)
            assert relative_gap(got, want) < TOL, (source, key)


# ---------------------------------------------------------------------------------------------
# What the frozen numbers are known to be right about
# ---------------------------------------------------------------------------------------------

def test_the_frozen_drive_of_every_backend_is_the_closed_form(golden):
    """``w(t)`` of ``Band.waveform`` at the slot's half step, for both dipoles.

    An electric dipole is a ``before_E`` term and acts across ``E(n) -> E(n+1)``, so its row is
    sampled at ``(n + 1/2) dt``; a magnetic one is ``before_H`` and acts across ``H(n-1/2) ->
    H(n+1/2)``, at ``n dt``. The frozen drive is what the run itself recorded, per backend, so
    this says the archive froze the right waveform, at the right times, on every backend.
    """
    arrays, meta = golden
    dt, steps = meta['scenes']['dipole']['dt'], meta['scenes']['dipole']['steps']
    n = np.arange(steps)
    for source in SCENES['dipole'].sources:
        shift = 0.5 if source.kind == 'electric' else 0.0
        want = source.band.waveform((n + shift) * dt, amplitude=source.amplitude,
                                    phase=source.phase, delay=source.delay)
        for backend in BACKENDS:
            got = arrays[f'dipole__{backend}__source__{source.name}__drive']
            assert relative_gap(got, want) < 1e-13, (source.name, backend)


def test_the_frozen_stamps_are_a_partition_of_unity_centred_on_the_dipole(golden):
    """``sum S = 1`` (the moment is ``w`` exactly) and ``sum S r = center`` (the centroid is exact).

    Both stamps reproduce linear functions on a lattice; a stamp that was only normalised, or
    that sat on the wrong Yee point of its component, would keep the first and lose the second.
    """
    arrays, _ = golden
    plan = quiet_plan(SCENES['dipole'])
    grid = plan.grid
    for source, term in zip(SCENES['dipole'].sources, plan.terms):
        assert isinstance(term, DipoleTerm) and term.name == source.name
        S = arrays[f'dipole__lowered__{source.name}__stamp']
        assert abs(S.sum() - 1.0) < 1e-14, source.name
        ranges = {0: term.slab.x, 1: term.slab.y, 2: term.slab.z}
        for a in range(3):
            lo, hi = ranges[a]
            r = grid.low[a] + (np.arange(lo, hi) + 0.5 * OFFSETS[term.field][term.slab.c][a]) \
                * grid.dx
            shape = [1, 1, 1]
            shape[2 - a] = hi - lo
            centroid = float(np.sum(S * r.reshape(shape)))
            assert abs(centroid - source.center[a]) < 1e-14, (source.name, a)


def test_the_frozen_bloch_pattern_is_the_transverse_phase_from_the_source(golden):
    """``exp(i (k_x x + k_y y))`` per cell, relative to cell ``(0, 0)``, with the ``k`` that
    ``Bloch.from_source`` wrote for this wave -- so the sheet's corrections carry the same phase
    the faces do."""
    arrays, meta = golden
    faces = fd.Bloch.from_source(plane_wave_source())
    dx = meta['scenes']['plane_wave']['dx']
    nx, ny, _ = meta['scenes']['plane_wave']['cells']
    i, j = np.arange(nx), np.arange(ny)
    want = np.exp(1j * (faces['x'].k * dx * i[None, :] + faces['y'].k * dx * j[:, None]))
    got = arrays['plane_wave__lowered__pw__pattern']
    assert got.shape == (1, ny, nx)
    assert np.max(np.abs(got[0] - want)) < 1e-14


@pytest.mark.parametrize('name', ('tfsf', 'tfsf_substrate'))
def test_the_tfsf_scenes_leave_the_scattered_region_at_round_off(golden, name):
    """The live run the pins compare against leaks nothing outside the box.

    Transitively, the golden froze a right answer: the per-backend pins below hold every
    backend to its frozen numbers at 1e-12, and this holds the run behind them to an empty
    scattered region, relative to the incident field the source itself recorded -- and holds
    the box to have injected it: the main grid's own probe inside the box (``inside`` in
    vacuum, ``above`` over the substrate) carries at least half the recorded incident peak. A
    box that injects nothing leaks nothing too, and the column's series alone is full either
    way, so without that floor this test passed with every correction zeroed.
    """
    res = result(name, 'numpy')
    source = SCENES[name].sources[0]
    incident = float(np.max(np.abs(to_numpy(res.sources['t'].incident))))
    assert incident > 0.5
    probe = {'tfsf': 'inside', 'tfsf_substrate': 'above'}[name]
    injected = float(np.max(np.abs(to_numpy(res[probe].values))))
    assert np.isfinite(injected) and injected > 0.5 * incident, (injected, incident)
    assert scattered_region_peak(res, source) <= LEAK_LINE * incident


# ---------------------------------------------------------------------------------------------
# The two halves: agreement, and each backend against its own golden
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('pair', pairs())
def test_the_backends_agree(name, pair):
    """Whole volumes, every series and every source series, each pair of backends, to TOL."""
    a, b = pair
    ra, rb = result(name, a), result(name, b)
    assert ra.plan.skeleton == rb.plan.skeleton
    for field in 'EH':
        fa, fb = to_numpy(getattr(ra.final, field)), to_numpy(getattr(rb.final, field))
        assert fa.shape == fb.shape and fa.dtype == fb.dtype
        assert relative_gap(fa, fb) < TOL, field
    assert sorted(ra) == sorted(rb)
    for probe in ra:
        assert np.array_equal(ra[probe].t, rb[probe].t)
        assert relative_gap(to_numpy(ra[probe].values), to_numpy(rb[probe].values)) < TOL, probe
    assert sorted(ra.sources) == sorted(rb.sources)
    for source, sa in ra.sources.items():
        sb = rb.sources[source]
        assert sa.kind == sb.kind and sa.plane == sb.plane and sa.plane_H == sb.plane_H
        for times in ('t', 't_H', 't_field'):
            ta, tb = getattr(sa, times), getattr(sb, times)
            assert (ta is None) == (tb is None)
            if ta is not None:
                assert np.array_equal(ta, tb), (source, times)
        for field in SERIES_FIELDS:
            va, vb = getattr(sa, field), getattr(sb, field)
            assert (va is None) == (vb is None), (source, field)
            if va is not None:
                va, vb = to_numpy(va), to_numpy(vb)
                assert va.shape == vb.shape and va.dtype == vb.dtype, (source, field)
                assert relative_gap(va, vb) < TOL, (source, field)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('backend', backends())
def test_each_backend_reproduces_its_own_golden(golden, name, backend):
    arrays, _ = golden
    got = capture(result(name, backend), arrays[f'{name}__samples'])
    want = _per_backend(arrays, name, backend)
    assert set(want) == set(got)
    for key, value in got.items():
        assert value.dtype == want[key].dtype and value.shape == want[key].shape, key
        assert relative_gap(value, want[key]) < TOL, key


@pytest.mark.parametrize('name', ('dipole', 'plane_wave', 'tfsf_substrate'))
@pytest.mark.parametrize('backend', backends())
def test_a_run_reproduces_itself_bit_for_bit(name, backend):
    """A pin needs one answer per backend: a second run is the first, byte for byte.

    The incident column is carried state (its fields, its psi, its pole memory) and the NumPy
    runtime keeps its scratch under prefixed keys; a column that reused a buffer it should have
    zeroed, or shared one with the main grid, would stop being a function of its scene here.
    """
    first = result(name, backend)
    with warnings.catch_warnings():
        for category in EXPECTED:
            warnings.simplefilter('ignore', category)
        again = fd.run(SCENES[name], backend=backend)
    for field in 'EH':
        assert np.array_equal(to_numpy(getattr(first.final, field)),
                              to_numpy(getattr(again.final, field)))
    for source, series in first.sources.items():
        for field in SERIES_FIELDS:
            value = getattr(series, field)
            if value is not None:
                assert np.array_equal(to_numpy(value),
                                      to_numpy(getattr(again.sources[source], field)))


# ---------------------------------------------------------------------------------------------
# The mutants: what agreement cannot see and the pin can
# ---------------------------------------------------------------------------------------------

def _mutants_agree_and_fail_the_pin(golden, name, runs):
    """Every backend ran the same wrong problem: they agree with each other and miss the pin."""
    arrays, _ = golden
    for a in INSTALLED:
        for b in INSTALLED:
            for field in 'EH':
                assert relative_gap(to_numpy(getattr(runs[a].final, field)),
                                    to_numpy(getattr(runs[b].final, field))) < TOL
    samples = arrays[f'{name}__samples']
    for backend, res in runs.items():
        got = capture(res, samples)
        want = _per_backend(arrays, name, backend)
        gaps = {k: relative_gap(v, want[k]) for k, v in got.items()}
        assert max(gaps.values()) > MUTANT_FLOOR, (name, backend, gaps)


def test_mutant_a_stamp_normalised_by_its_squares_passes_agreement_and_fails_every_golden(golden):
    """The stamp mutant the design names: ``S^2 / sum S^2`` in place of ``S``.

    It is the natural wrong normalisation -- the one that makes ``sum S^2 = 1``, as an energy
    argument might suggest -- and it keeps the stamp's support, shape and symmetry, so the plan,
    the skeleton and every array shape are unchanged. All three backends run it alike.
    """
    scene = SCENES['dipole']
    p = quiet_plan(scene)

    def squares(params, scene, plan, rt):
        sources = {}
        for name, entry in params['sources'].items():
            S = entry['stamp'] ** 2
            sources[name] = dict(entry, stamp=S / S.sum())
        return dict(params, sources=sources)

    _mutants_agree_and_fail_the_pin(golden, 'dipole',
                                    {b: run_mutated(scene, p, b, squares) for b in INSTALLED})


def test_mutant_one_tfsf_face_with_its_sign_flipped_passes_agreement_and_fails_every_golden(
        golden):
    """The face-sign mutant, through the plan: one correction rectangle with ``-sign``.

    A static change, so the mutated plan carries a different skeleton -- but every backend is
    handed the same one and agrees on it. The box then leaks at that face, and every backend
    is O(1) from its golden.
    """
    scene = SCENES['tfsf_substrate']
    p = quiet_plan(scene)
    (term,) = p.terms
    assert isinstance(term, ColumnTerm) and len(term.corrections) > 1
    first = term.corrections[0]
    flipped = (dataclasses.replace(first, sign=-first.sign),) + term.corrections[1:]
    mutated = p.with_static(terms=(dataclasses.replace(term, corrections=flipped),))
    runs = {b: run_plan(scene, mutated, b) for b in INSTALLED}
    source = scene.sources[0]
    assert scattered_region_peak(runs['numpy'], source) > 1e-3
    _mutants_agree_and_fail_the_pin(golden, 'tfsf_substrate', runs)


def test_the_pin_resolves_a_part_in_a_billion_of_an_amplitude(golden):
    """The budget is doing work: a ``1 + 1e-9`` amplitude is seen, at 1e-9 and no more.

    The amplitude is a traced leaf, so the plan, the skeleton and the frozen metadata are
    unchanged and only the pin can see it. That the gap comes out *at* 1e-9 (the source series
    and the fields are linear in it) says the pin measures a perturbation rather than merely
    noticing one.
    """
    arrays, _ = golden
    scene = SCENES['tfsf']

    def nudge(params, scene, plan, rt):
        sources = {n: dict(e, amplitude=e['amplitude'] * (1.0 + 1e-9))
                   for n, e in params['sources'].items()}
        return dict(params, sources=sources)

    res = run_mutated(scene, quiet_plan(scene), 'numpy', nudge)
    got = capture(res, arrays['tfsf__samples'])
    want = _per_backend(arrays, 'tfsf', 'numpy')
    gaps = {k: relative_gap(v, want[k]) for k, v in got.items()}
    assert max(gaps.values()) > TOL, gaps
    assert max(gaps.values()) < 1e-8, gaps
    assert math.isclose(gaps['source__t__incident'], 1e-9, rel_tol=1e-3)
