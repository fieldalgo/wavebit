"""Gate 7 -- three backends on the stage-1 scenes, and each pinned to its own golden.

Gate 5 did this for the stage-0 kernel, on boxes made of walls and one eigenmode. Stage 1
added everything that is not a wall -- a stretched CPML slab with four psi per face, graded
``Absorber`` damping, per-cell update coefficients built from a material, and an ADE pole
recurrence with its own memory -- and none of it is covered by an eigenmode in a PEC box. So
this gate runs the six scenes of ``tests/references/_fd_stage1_cases.py``, one per feature:

    pml_box       a 20^3 vacuum box with a PML on all six faces (24 psi arrays)
    pml_oblique   the same stretch under a Bloch phase, at 30 degrees, so the run is complex
    slab          a lossless dielectric layer, i.e. a ``be`` that is an array
    dispersive    a Drude + Lorentz half-space running out through an ``Absorber``
    conductor     a conductive slab, complex, i.e. ``ae`` and ``be`` both arrays
    absorber      an ``Absorber`` box with no PML at all

**Two halves, and the second is the one that can fail alone.** NumPy, JAX and Torch are held
to each other on every cell of the final E and H and on every recorded series, at 1e-12 of
the array's peak (measured 2026-09-18: Torch bit-identical to NumPy on the four real scenes
and 4.7e-15 off on the two complex ones, JAX 4.2e-15 to 1.2e-13 off both -- the large end
being the scenes whose field has been absorbed down by three decades, where the array's own
peak is that much smaller a scale). That agreement is necessary and nowhere near sufficient --
a wrong profile, a dropped pole term or a slab snapped a cell off moves all three together --
so each backend is also held to what *it* produced when ``fd_golden_stage1.npz`` was frozen
(``tests/references/interface/freeze_fd_stage1.py``). Per backend, so a red test names the
one that moved. Two mutants make the argument executable: a plan with its CPML faces dropped
and an ADE recurrence whose ``drive`` is zeroed both keep all three backends agreeing to
1e-12 and put every one of them O(1) from its golden.

**The lowered numbers are pinned once, not per backend.** The CPML profiles, the four update
coefficients and the pole coefficients are the same on every backend by construction, and
every backend is shown to build them alike. Freezing them beside the fields is what separates
"the layer was regraded" from "the step changed under an unchanged layer" in a failure.

**The archive is read before it is trusted**, as gate 5 reads its own: the content digest is
recomputed, the scene behind each key is shown to be the scene built here (skeleton, absorbing
faces, snapped spans, sampled initial field), and the frozen energies are held to the one
property these scenes have a right answer for -- an absorber absorbs. The same box with its
absorbing faces replaced by PEC keeps its energy; the frozen one is orders below it. For the
two scenes whose loss is in the medium, the control is the same scene with the material
replaced by a lossless one of the same ``eps``.
"""
import json
import pathlib
import sys
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from fd_helpers import backends, run_mutated, run_plan, to_numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'references'))
from _fd_stage1_cases import (BACKENDS, BOUNDARY_LOSS_ONLY, GOLDEN, MATERIAL_LOSS,   # noqa: E402
                              TOL, capture, content_digest, lossless, lowered,
                              psi_tag, relative_gap, sample_indices, scenes, sealed)

SCENES = scenes()
NAMES = tuple(SCENES)

#: What dropping the CPML faces or the pole term does to the pin, at least. Measured
#: 2026-09-18: 1e-2 to 3 on the slab's arrays with the stretch gone, 2e-3 to 2 on the
#: dispersive scene's with the ADE term gone.
MUTANT_FLOOR = 1e-3

#: A sealed box's energy wanders (the leapfrog conserves a nearby quadratic form, not this
#: one) but does not decay: measured 0.99 to 1.01 of the initial energy on these four scenes.
SEALED_BAND = (0.5, 2.0)

#: What the absorbing faces are worth, as a fraction of what the sealed box keeps. Measured
#: 2026-09-18: 2.2e-3 (pml_box), 5.5e-2 (pml_oblique), 1.6e-3 (slab), 6.1e-6 (absorber).
#: **The oblique number is the honest one**: a packet with a fixed transverse wavevector and a
#: band of ``kz`` reaches near-grazing at the edges of its band, and grazing is where gate 1's
#: own table says a 12-cell layer reflects 3e-2 -- so a few percent of the energy bounces
#: between the two z layers for the rest of the run (measured: still 3e-2 after twice as many
#: steps). The line is set above that and two decades below what a layer that stopped
#: absorbing does: the sealed control *is* that mutant, and it keeps 1.0.
ABSORBED_TO = 0.15

_MARKS = {p.values[0]: list(p.marks) for p in backends()}
INSTALLED = tuple(b for b, marks in _MARKS.items() if not marks)
_RUNS = {}


def result(name, backend):
    """The run of scene ``name`` on ``backend``, once per process."""
    key = (name, backend)
    if key not in _RUNS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
            _RUNS[key] = fd.run(SCENES[name], backend=backend)
    return _RUNS[key]


def quiet_plan(scene):
    """``fd.plan`` with the one warning these scenes raise on purpose silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        return fd.plan(scene)


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
    for key in ('generator_sha256', 'cases_sha256', 'shared_cases_sha256'):
        assert len(meta[key]) == 64
    for name in NAMES:
        assert arrays[f'{name}__samples'].shape[1] == 4
        for backend in BACKENDS:
            for probe in meta['scenes'][name]['monitors']:
                assert f'{name}__{backend}__series__{probe}' in arrays
            for key in ('final__E', 'final__H', 'energy'):
                assert f'{name}__{backend}__{key}' in arrays


def test_the_archive_covers_every_stage_one_feature(golden):
    """The six scenes are not six copies of one: each carries the thing it is named for.

    Written as an assertion rather than left to the docstring, because a case list that
    quietly lost its pole term or its Absorber would still pass every comparison below --
    against a golden frozen from the same emptied scene.
    """
    arrays, meta = golden
    assert meta['scenes']['pml_box']['absorbing'] == [[f, 6] for f in
                                                      ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')]
    assert meta['scenes']['pml_box']['cells'] == [20, 20, 20]
    assert sum(1 for k in arrays if k.startswith('pml_box__cpml__')) == 24 * 3
    assert meta['scenes']['pml_oblique']['dtype'] == 'complex128'
    assert meta['scenes']['conductor']['dtype'] == 'complex128'
    assert meta['scenes']['dispersive']['poles'] == 2
    assert [f[0] for f in meta['scenes']['dispersive']['absorbing']] == ['z-', 'z+']
    assert meta['scenes']['absorber']['absorbing'] == [['z-', 12], ['z+', 12]]
    assert not any(k.startswith('absorber__cpml__') for k in arrays)
    for name in ('slab', 'dispersive', 'conductor'):
        assert meta['scenes'][name]['spans'], name
    assert arrays['slab__coefficient__be'].shape == (3, 60, 1, 1)
    assert arrays['pml_box__coefficient__be'].shape == ()      # vacuum: the stage-0 scalar


@pytest.mark.parametrize('name', NAMES)
def test_the_scene_is_the_one_frozen(golden, name):
    """Static content by the skeleton and by what stage 1 added to it; the excitation by value.

    The skeleton holds no leaf, so a helper that moved the pulse would leave it unchanged and
    read as kernel drift below; the sampled initial field is what tells the two apart. The
    absorbing faces and the snapped spans are listed separately because they are the stage-1
    static content a reader of a failure wants named.
    """
    arrays, meta = golden
    scene = SCENES[name]
    p = quiet_plan(scene)
    frozen = meta['scenes'][name]
    assert p.skeleton == frozen['skeleton']
    assert [p.nx, p.ny, p.nz] == frozen['cells']
    assert p.dtype == frozen['dtype'] and p.steps == frozen['steps']
    assert [list(f) for f in p.absorbing_faces] == frozen['absorbing']
    assert [list(s) for s in p.spans] == frozen['spans']
    assert np.array_equal(arrays[f'{name}__samples'], sample_indices(p.shape))


@pytest.mark.parametrize('name', NAMES)
def test_the_lowered_coefficients_profiles_and_poles_are_the_frozen_ones(golden, name):
    """The CPML profiles, the four coefficients and the pole recurrence, from the lowering.

    Bit for bit where the lowering is deterministic arithmetic, which it is on NumPy: these
    are not a run's output, they are what a run is given, so a difference here is a regraded
    layer or a moved coefficient formula and never round-off in a scan.
    """
    arrays, _ = golden
    p = quiet_plan(SCENES[name])
    got = lowered(SCENES[name], p, arrays[f'{name}__samples'])
    prefix = f'{name}__'
    want = {k[len(prefix):]: v for k, v in arrays.items()
            if k.startswith(prefix) and k.split('__')[1] not in BACKENDS
            and not k.endswith('__samples')}
    assert set(got) == set(want)
    for key, value in got.items():
        assert value.dtype == want[key].dtype and value.shape == want[key].shape, key
        assert relative_gap(value, want[key]) < TOL, key


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('backend', backends())
def test_every_backend_lowers_the_same_profiles(golden, name, backend):
    """A profile built on one backend and not another is a defect agreement would not see.

    ``lower`` is where the CPML profiles, the coefficient arrays and the pole coefficients are
    built, and it is the one part of the stage-1 path that runs per backend *outside* the
    scan -- on Torch through ``as_tensor``, on JAX through ``jnp.asarray`` with x64 lifted.
    An x64 that was not lifted would land here as float32 profiles and nowhere else.
    """
    arrays, _ = golden
    scene = SCENES[name]
    p = quiet_plan(scene)
    params = fd.lower(scene, p, backend)
    for key in ('ae', 'be', 'ah', 'bh'):
        assert relative_gap(to_numpy(params[key]),
                            arrays[f'{name}__coefficient__{key}']) < TOL, key
    for psi, profile in params['cpml'].items():
        for which in ('b', 'a', 'ik'):
            got = to_numpy(profile[which]).reshape(-1)
            assert got.dtype == np.float64
            assert relative_gap(got, arrays[f'{name}__cpml__{psi_tag(psi)}__{which}']) < TOL
    for term, coefficients in params['poles'].items():
        tag = term.replace('[', '_').replace(']', '')
        for which in ('step_gain', 'hold', 'drive'):
            got = np.array([float(to_numpy(v)) for v in coefficients[which]])
            assert relative_gap(got, arrays[f'{name}__poles__{tag}__{which}']) < TOL


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
    arrays, _ = golden
    got = capture(result(name, backend), arrays[f'{name}__samples'])
    prefix = f'{name}__{backend}__'
    assert {k[len(prefix):] for k in arrays if k.startswith(prefix)} == set(got)
    for key, value in got.items():
        want = arrays[prefix + key]
        assert value.dtype == want.dtype and value.shape == want.shape, key
        assert relative_gap(value, want) < TOL, key


@pytest.mark.parametrize('name', ['pml_box', 'dispersive', 'conductor'])
@pytest.mark.parametrize('backend', backends())
def test_a_run_reproduces_itself_bit_for_bit(name, backend):
    """A pin needs one answer per backend: a second run is the first, byte for byte.

    The psi and the pole memory are carried state, and carried state is where a run that
    reuses a buffer it should have zeroed stops being a function of its scene.
    """
    first = result(name, backend)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        again = fd.run(SCENES[name], backend=backend)
    for field in 'EH':
        assert np.array_equal(to_numpy(getattr(first.final, field)),
                              to_numpy(getattr(again.final, field)))
    for probe in first:
        assert np.array_equal(to_numpy(first[probe].values), to_numpy(again[probe].values))


# ---------------------------------------------------------------------------------------------
# What the frozen numbers are known to be right about: an absorber absorbs
# ---------------------------------------------------------------------------------------------

def _energy(res):
    E, H = to_numpy(res.final.E), to_numpy(res.final.H)
    return float(np.sum(np.abs(E) ** 2) + np.sum(np.abs(H) ** 2))


def _initial_energy(scene, plan):
    params = fd.lower(scene, plan, 'numpy')
    total = 0.0
    for field in ('E0', 'H0'):
        if params[field] is not None:
            total += float(np.sum(np.abs(np.asarray(params[field])) ** 2))
    return total * abs(complex(params['amplitude'])) ** 2


@pytest.mark.parametrize('name', BOUNDARY_LOSS_ONLY)
def test_the_frozen_energy_is_what_the_absorbing_faces_left(golden, name):
    """The archive froze a *right* answer, for the one thing these scenes state exactly.

    A number that only ever falls proves nothing -- a pulse leaves a window too. The same box
    with PEC in place of every absorbing face keeps its energy (measured 0.988 to 1.002 of the
    initial energy; the leapfrog conserves a nearby quadratic form, not exactly this one), and
    the frozen energy is 1.3 to 5 decades below it -- see ``ABSORBED_TO`` for the four numbers
    and for why the oblique one is the smallest margin. The pair is the claim: the energy went
    into the layer and not out of the window.
    """
    arrays, _ = golden
    scene = SCENES[name]
    control = fd.run(sealed(scene))
    kept = _energy(control)
    start = _initial_energy(sealed(scene), fd.plan(sealed(scene)))
    assert SEALED_BAND[0] * start < kept < SEALED_BAND[1] * start, (kept, start)
    for backend in BACKENDS:
        frozen = float(arrays[f'{name}__{backend}__energy'])
        assert frozen < ABSORBED_TO * kept, (name, backend, frozen, kept)


@pytest.mark.parametrize('name', MATERIAL_LOSS)
def test_the_frozen_energy_of_a_lossy_medium_is_below_its_lossless_twin(golden, name):
    """The other two scenes lose energy in the *medium*, so PEC walls are not their control.

    The control is the same box with the conductivity and the poles taken out and the same
    ``eps`` left in -- sealed, so nothing leaves through a face either. The lossy run keeps
    less: measured 0.33 of its twin for the two-pole metal and 0.82 for the conductive slab,
    which is the weaker of the two because its packet crosses a half-wavelength of
    ``sigma = 0.5`` rather than filling a third of the box. That is what says ``sigma`` and the
    ADE term did something, which neither the cross-backend half nor the pin can say.
    """
    arrays, _ = golden
    scene = SCENES[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.AbsorberProximityWarning)
        lossy_sealed = _energy(fd.run(sealed(scene)))
        control = _energy(fd.run(sealed(lossless(scene))))
    assert lossy_sealed < 0.9 * control, (lossy_sealed, control)
    for backend in BACKENDS:
        assert float(arrays[f'{name}__{backend}__energy']) < control


# ---------------------------------------------------------------------------------------------
# The mutants: what agreement cannot see and the pin can
# ---------------------------------------------------------------------------------------------

def _mutants_agree_and_fail_the_pin(golden, name, runs):
    """Every backend ran the same wrong problem: they agree with each other and miss the pin."""
    arrays, _ = golden
    for a in INSTALLED:
        for b in INSTALLED:
            assert relative_gap(to_numpy(runs[a].final.E), to_numpy(runs[b].final.E)) < TOL
    samples = arrays[f'{name}__samples']
    for backend, res in runs.items():
        got = capture(res, samples)
        gaps = {k: relative_gap(v, arrays[f'{name}__{backend}__{k}']) for k, v in got.items()}
        assert max(gaps.values()) > MUTANT_FLOOR, (name, backend, gaps)


def test_mutant_the_stretch_removed_passes_agreement_and_fails_every_golden(golden):
    """The case for the pin: a plan defect all three backends share.

    ``cpml=()`` leaves the PML's PEC wall and its masks exactly where they were and takes only
    the stretch away, which is the closest thing to "the layer stopped absorbing" a plan can
    express. All three backends then run that same wrong problem and agree with each other to
    the same 1e-12 they agree on the right one, while every one of them is O(1) from its
    golden.
    """
    scene = SCENES['slab']
    p = quiet_plan(scene)
    assert p.cpml, 'the slab scene has PML faces to drop'
    mutated = p.with_static(cpml=())
    _mutants_agree_and_fail_the_pin(golden, 'slab',
                                    {b: run_plan(scene, mutated, b) for b in INSTALLED})


def test_mutant_the_poles_stop_driving_passes_agreement_and_fails_every_golden(golden):
    """The same argument for the ADE term, mutated through the params rather than the plan.

    The pole term cannot be dropped from the plan -- ``lower`` asserts that a dispersive
    structure has a term, so ``terms=()`` raises before it runs -- so the mutant zeroes the
    leaf the recurrence is driven by instead. ``drive = 0`` leaves the memory, the slab, the
    ``eps_inf`` coefficients and every array shape alone and makes the polarisation stay at
    rest: the metal becomes its own high-frequency limit. All three backends agree on it.
    """
    scene = SCENES['dispersive']
    p = quiet_plan(scene)
    assert p.terms, 'the dispersive scene has a pole term'

    def silence(params, scene, plan, rt):
        poles = {name: dict(term, drive=tuple(0.0 * d for d in term['drive']))
                 for name, term in params['poles'].items()}
        return dict(params, poles=poles)

    _mutants_agree_and_fail_the_pin(
        golden, 'dispersive',
        {b: run_mutated(scene, p, b, silence) for b in INSTALLED})


def test_the_pin_resolves_a_part_in_a_billion_of_a_profile(golden):
    """The budget is doing work: a 1e-9 change of one CPML profile is seen, at 1e-9 and no more.

    ``b`` is a traced leaf of the params, so the plan, the skeleton and the frozen scene
    metadata are all unchanged -- every check above this one passes -- and only the pin can
    see it. That the gap comes out *at* 1e-9 and not at 1 is what says the pin measures a
    perturbation rather than merely noticing one.
    """
    arrays, _ = golden
    scene = SCENES['pml_oblique']

    def nudge(params, scene, plan, rt):
        first = sorted(params['cpml'])[0]
        profile = dict(params['cpml'][first])
        profile['b'] = profile['b'] * (1.0 + 1e-9)
        params = dict(params, cpml=dict(params['cpml'], **{first: profile}))
        return params

    res = run_mutated(scene, quiet_plan(scene), 'numpy', nudge)
    got = capture(res, arrays['pml_oblique__samples'])
    gaps = {k: relative_gap(v, arrays[f'pml_oblique__numpy__{k}']) for k, v in got.items()}
    assert max(gaps.values()) > TOL, gaps
    assert max(gaps.values()) < 1e-6, gaps


def test_the_dispersive_half_space_warns_that_it_reaches_its_absorber():
    """The one warning these scenes raise on purpose, asserted rather than only silenced.

    Every other test here silences ``AbsorberProximityWarning`` for the dispersive scene. A
    silenced warning nobody asserts is a warning that can stop firing unnoticed, and this one
    is load-bearing: the scene is a *half-space*, so the medium is meant to run out through
    the absorber, and the layer is matched to the background rather than to it.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fd.plan(SCENES['dispersive'])
    messages = [str(w.message) for w in caught
                if issubclass(w.category, fd.AbsorberProximityWarning)]
    assert len(messages) == 1, [str(w.message) for w in caught]
    assert "'z+'" in messages[0] and 'structures[0]' in messages[0]
