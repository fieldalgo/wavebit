"""Gate 5 -- three backends: NumPy, JAX and Torch agree on gates 1-4's scenes, and each is pinned to its own golden.

The kernel is one body of code dispatched through a runtime, so the three backends should
give the same numbers up to the order XLA fuses an update in. On the eight scenes of
``tests/references/_fd_stage0_cases.py`` -- the PEC, PMC, periodic and Bloch boxes of
gates 1, 2 and 4 and the four half boxes of gate 3 -- every component of every cell of the
final E and H and every recorded series agree between each pair of backends to 1e-12 of
the array's peak (measured 2026-09-18: Torch bit-identical to NumPy on the real scenes and
2e-15 off on the complex one, JAX 1e-15 to 4e-14 off both).

**Agreement is necessary and nowhere near sufficient**, which is why the second half of
this gate exists. A defect in the plan -- a pad sign, a wall a cell off, a mask that is not
applied -- moves NumPy, JAX and Torch *together*, and three backends that agree with each
other say nothing about it. So each backend is also held to ``fd_golden_stage0.npz``: the
probe series, the final fields at frozen sample cells and the whole-volume energy that
backend itself produced when the archive was frozen (``tests/references/interface/
freeze_fd_stage0.py``), to the same 1e-12. Per backend rather than one shared number, so
that a red test names the backend that moved. The mutant is the argument made
executable: a plan with its wall masks dropped passes the agreement half on all three
backends and fails the pin on all three by O(1).

The archive is read before it is trusted: its content digest is recomputed from the loaded
arrays, the scene each key names is shown to be the scene built here (same skeleton, same
sampled initial field), and the frozen eigenmode series are held to the recurrence gates
1, 2 and 4 hold their live series to -- so the archive is known to have frozen a *right*
answer, which a self-captured golden does not get for free.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import json
import sys

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from tests.fixtures.fdtd.fd_helpers import backends, recurrence_residual, run_plan, to_numpy

sys.path.insert(0, str(repository_path('tests/references')))
from tests.references._fd_stage0_cases import BACKENDS, GOLDEN, TOL, capture, content_digest, relative_gap, sample_indices, scenes

SCENES = scenes()
NAMES = tuple(SCENES)
#: The scenes that carry an exact discrete eigenmode, and therefore a recurrence to hold.
EIGENMODES = ('pec', 'pmc', 'periodic', 'bloch')
#: The gates' own round-off budget on the recurrence, for the frozen series.
RECURRENCE_TOL = 1e-11
#: What dropping the walls does to the pin, at least; measured on the PEC box: 4e-2 on the
#: energy, 5e-1 to 2 on the final fields, 3 to 13 on the series.
MUTANT_FLOOR = 1e-3

_MARKS = {p.values[0]: list(p.marks) for p in backends()}
INSTALLED = tuple(b for b, marks in _MARKS.items() if not marks)
_RUNS = {}


def result(name, backend):
    """The run of scene ``name`` on ``backend``, once per process."""
    key = (name, backend)
    if key not in _RUNS:
        _RUNS[key] = fd.run(SCENES[name][0], backend=backend)
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


def test_the_golden_is_intact_and_says_where_it_came_from(golden):
    arrays, meta = golden
    assert content_digest(arrays) == meta['content_sha256']
    assert meta['backends'] == list(BACKENDS)
    assert set(meta['scenes']) == set(NAMES)
    assert meta['tolerance'] == TOL
    for key in ('python', 'wavebit', 'numpy', 'jax', 'torch'):
        assert isinstance(meta['versions'][key], str) and meta['versions'][key]
    assert len(meta['generator_sha256']) == len(meta['cases_sha256']) == 64
    for name in NAMES:
        assert arrays[f'{name}__samples'].shape[1] == 4
        for backend in BACKENDS:
            for probe in meta['scenes'][name]['monitors']:
                assert f'{name}__{backend}__series__{probe}' in arrays
            for key in ('final__E', 'final__H', 'energy'):
                assert f'{name}__{backend}__{key}' in arrays


@pytest.mark.parametrize('name', NAMES)
def test_the_scene_is_the_one_frozen(golden, name):
    """Static content by skeleton, the excitation by its sampled values.

    The skeleton has no leaf in it, so a helper that moved a mode amplitude or a seed would
    leave it unchanged and read as kernel drift below; the sampled initial field is what
    tells the two apart. 1e-13 is round-off on a transcendental evaluated by another libm.
    """
    arrays, meta = golden
    scene, _ = SCENES[name]
    p = fd.plan(scene)
    assert p.skeleton == meta['scenes'][name]['skeleton']
    assert [p.nx, p.ny, p.nz] == meta['scenes'][name]['cells']
    assert p.dtype == meta['scenes'][name]['dtype']
    samples = arrays[f'{name}__samples']
    assert np.array_equal(samples, sample_indices(p.shape))
    params = fd.lower(scene, p, 'numpy')
    for field in 'EH':
        key = f'{name}__initial__{field}'
        F0 = params[f'{field}0']
        assert (key in arrays) == (F0 is not None)
        if F0 is not None:
            assert relative_gap(np.asarray(F0)[tuple(samples.T)], arrays[key]) < 1e-13


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('pair', pairs())
def test_the_backends_agree(name, pair):
    """Whole volumes and every series, each pair of backends, to TOL of the peak."""
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


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('backend', backends())
def test_each_backend_reproduces_its_own_golden(golden, name, backend):
    arrays, meta = golden
    got = capture(result(name, backend), arrays[f'{name}__samples'])
    prefix = f'{name}__{backend}__'
    assert {k[len(prefix):] for k in arrays if k.startswith(prefix)} == set(got)
    for key, value in got.items():
        want = arrays[prefix + key]
        assert value.dtype == want.dtype and value.shape == want.shape, key
        assert relative_gap(value, want) < TOL, key


@pytest.mark.parametrize('name', EIGENMODES)
@pytest.mark.parametrize('backend', BACKENDS)
def test_the_frozen_series_obey_the_gates_recurrence(golden, name, backend):
    """The archive froze a right answer: gates 1, 2 and 4's recurrence, on the frozen series.

    Reads the archive only, so it runs whether or not the backend is installed here.
    """
    arrays, meta = golden
    _, omega = SCENES[name]
    dt = meta['scenes'][name]['dt']
    assert omega is not None and dt == fd.plan(SCENES[name][0]).dt
    for probe in meta['scenes'][name]['monitors']:
        series = arrays[f'{name}__{backend}__series__{probe}']
        assert np.max(np.abs(series)) > 1e-3
        assert recurrence_residual(series, omega, dt) < RECURRENCE_TOL


@pytest.mark.parametrize('name', ['pec', 'bloch'])
@pytest.mark.parametrize('backend', backends())
def test_a_run_reproduces_itself_bit_for_bit(name, backend):
    """A pin needs one answer per backend: a second run is the first, byte for byte."""
    first = result(name, backend)
    again = fd.run(SCENES[name][0], backend=backend)
    for field in 'EH':
        assert np.array_equal(to_numpy(getattr(first.final, field)),
                              to_numpy(getattr(again.final, field)))
    for probe in first:
        assert np.array_equal(to_numpy(first[probe].values), to_numpy(again[probe].values))


def test_mutant_dropped_masks_pass_agreement_and_fail_every_golden(golden):
    """The case for the pin: a plan defect the three backends share is invisible to agreement.

    The PEC box's plan with its wall masks removed is the same wrong problem on every
    backend, so they still agree to TOL -- and every one of them is O(1) from its golden.
    """
    arrays, meta = golden
    scene, _ = SCENES['pec']
    p = fd.plan(scene)
    assert p.masks, 'the PEC box has walls to drop'
    mutated = p.with_static(masks=())
    runs = {b: run_plan(scene, mutated, b) for b in INSTALLED}
    for a in INSTALLED:
        for b in INSTALLED:
            assert relative_gap(to_numpy(runs[a].final.E), to_numpy(runs[b].final.E)) < TOL
    samples = arrays['pec__samples']
    for backend, res in runs.items():
        got = capture(res, samples)
        gaps = {k: relative_gap(v, arrays[f'pec__{backend}__{k}']) for k, v in got.items()}
        assert max(gaps.values()) > MUTANT_FLOOR, gaps


def test_mutant_the_pin_resolves_a_part_in_a_billion_of_the_excitation(golden):
    """The budget is doing work: a 1e-9 change of a traced leaf is seen, at 1e-9 and not more.

    ``amplitude`` is a leaf, so the skeleton and the sampled initial field are unchanged --
    the scene check above passes -- and only the pin can see it.
    """
    arrays, meta = golden
    scene, _ = SCENES['pec']
    nudged = scene.replace(initial=scene.initial.replace(amplitude=1.0 + 1e-9))
    assert fd.plan(nudged).skeleton == fd.plan(scene).skeleton
    got = capture(fd.run(nudged), arrays['pec__samples'])
    gaps = {k: relative_gap(v, arrays[f'pec__numpy__{k}']) for k, v in got.items()}
    assert all(TOL < g < 1e-8 for g in gaps.values()), gaps
