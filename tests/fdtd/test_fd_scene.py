"""Gate 9 -- the skeleton hash, and the doors: what a scene refuses by name, and what it accepts.

Not a physics gate. Two things are pinned here.

**The skeleton** (``plan.skeleton``) is what ``fd.run(scene, plan=)`` compares before it lets
a prepared plan drive a scene, so it has to be a function of the scene's *static* content
and of nothing else: the same in another process (no ``id()``, no hash randomisation, no
``repr()`` of an array), different when a static value moves (``dx``, ``steps``, a face
type), and unchanged when a traced leaf moves (a Bloch phase, an initial amplitude, the
initial field's values) -- otherwise a gradient through one of those would need a new plan
per evaluation. The two subprocesses run under different ``PYTHONHASHSEED`` values on
purpose. The mutant recomputes the hash with one static field left out and shows two
different scenes then collide, which is what the real skeleton must never do.

**The doors** are the refusals the design lists, each with the accepted form beside it so
that a refusal is shown to be a decision and not a broken constructor: every axis of
``Boundaries`` is required; an unknown keyword is refused with the nearest valid name on
every public constructor; ``Periodic`` and ``Bloch`` must be one declaration for both faces
of an axis; ``Symmetry`` sits on at most one face, and its parity is a value (``1``, ``1.0``
and ``numpy.int64(1)`` are one mirror with one skeleton); the Courant number is in
``(0, 1)`` -- open at 1, where the corner Nyquist mode grows linearly; ``Bloch`` with a
concrete zero phase is refused naming ``Periodic`` and with anything that is not a real
scalar or a tracer is refused naming ``Bloch``; ``Boundaries.open`` and the keywords of
later stages are refused naming the stage.

The reason every one of these is an error and not a default or a warning is the same: a
keyword nobody reads, a face nobody chose or a plan built for another scene makes the run
answer a question the user did not ask, and it does so silently.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_root
import dataclasses
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._prepare import skeleton_of
from tests.fixtures.fdtd.fd_helpers import to_numpy

ROOT = repository_root()
PY = sys.executable
PROBE = (0.1, 0.0, -0.05)


def reference_scene(**changes):
    """The gate's scene: a PEC box of 8^3 cells, one Ey probe, 20 steps; ``changes`` replace fields."""
    scene = fd.Scene(
        domain=fd.Domain(size=(1.0, 1.0, 1.0)),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        resolution=fd.Resolution(dx=1 / 8),
        run=fd.Run(steps=20),
        initial=fd.InitialField(E=(None, lambda x, y, z: np.cos(np.pi * x) * np.cos(np.pi * z), None)),
        monitors=(fd.Record('probe', at=PROBE, component='Ey'),))
    return scene.replace(**changes) if changes else scene


def bloch_scene(phase=0.7):
    """The reference scene with ``Bloch(phase)`` on x and ``Periodic`` on y: a complex run."""
    return reference_scene(boundaries=fd.Boundaries(x=fd.Bloch(phase=phase), y=fd.Periodic(),
                                                    z=fd.PEC()))


def _kwargs(scene):
    """The four required keywords of ``Scene``, taken from ``scene``."""
    return dict(domain=scene.domain, boundaries=scene.boundaries, resolution=scene.resolution,
                run=scene.run)


def _static(plan):
    """The plan's static content, as ``skeleton_of`` receives it."""
    content = {f.name: getattr(plan, f.name) for f in dataclasses.fields(plan) if f.name != 'skeleton'}
    # Absent optional records preserve skeletons from before these fields existed.
    for name in ('junctions', 'shape_ranges', 'decay'):
        if not getattr(plan, name):
            content.pop(name)
    return content


#: A per-component initial field (E_y only) and the four required keywords of ``Scene``.
_INITIAL = (None, lambda x, y, z: x, None)
_SCENE_KW = _kwargs(reference_scene())


# ---------------------------------------------------------------------------------------------
# The skeleton
# ---------------------------------------------------------------------------------------------

def _skeleton_in_a_subprocess(hashseed):
    code = ('import sys; sys.path.insert(0, "tests")\n'
            'import wavebit.fdtd._legacy as fd\n'
            'from tests.fdtd.test_fd_scene import reference_scene\n'
            'print(fd.plan(reference_scene()).skeleton)')
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run([PY, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True, timeout=300)
    return out.stdout.strip()


def test_the_skeleton_is_the_same_in_two_subprocesses():
    """Gate 9: two fresh interpreters, two hash seeds, one skeleton -- and it is this one."""
    here = fd.plan(reference_scene()).skeleton
    first, second = _skeleton_in_a_subprocess(1), _skeleton_in_a_subprocess(2)
    assert first == second == here
    assert len(here) == 64 and int(here, 16) >= 0          # a sha256 hex digest


def test_the_skeleton_is_reproducible_and_is_what_the_report_says():
    """Gate 9: rebuilding the plan gives the same skeleton, and ``check`` reports that one."""
    plan = fd.plan(reference_scene())
    assert fd.plan(reference_scene()).skeleton == plan.skeleton
    assert fd.plan(reference_scene()) == plan and hash(fd.plan(reference_scene())) == hash(plan)
    assert fd.check(reference_scene()).skeleton == plan.skeleton


STATIC_CHANGES = [
    pytest.param(dict(resolution=fd.Resolution(dx=1 / 16)), id='dx'),
    pytest.param(dict(run=fd.Run(steps=21)), id='steps'),
    pytest.param(dict(run=fd.Run(steps=20, courant=0.4)), id='courant'),
    pytest.param(dict(boundaries=fd.Boundaries(x=fd.PMC(), y=fd.PEC(), z=fd.PEC())), id='face-type'),
    pytest.param(dict(boundaries=fd.Boundaries(x=(fd.PMC(), fd.PEC()), y=fd.PEC(), z=fd.PEC())),
                 id='one-face'),
    pytest.param(dict(boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.Periodic())),
                 id='periodic-face'),
    pytest.param(dict(boundaries=fd.Boundaries(x=fd.Bloch(phase=0.7), y=fd.PEC(), z=fd.PEC())),
                 id='bloch-face'),
    pytest.param(dict(boundaries=fd.Boundaries(x=(fd.Symmetry(parity=1), fd.PEC()), y=fd.PEC(),
                                               z=fd.PEC())), id='symmetry-face'),
    pytest.param(dict(domain=fd.Domain(size=(1.0, 1.0, 1.5))), id='size'),
    pytest.param(dict(domain=fd.Domain(size=(1.0, 1.0, 1.0), center=(0.5, 0.0, 0.0))), id='center'),
    pytest.param(dict(monitors=(fd.Record('probe', at=PROBE, component='Hz'),)), id='monitor-component'),
    pytest.param(dict(monitors=(fd.Record('probe', at=(0.3, 0.0, -0.05)),)), id='monitor-point'),
    pytest.param(dict(monitors=()), id='no-monitor'),
    pytest.param(dict(initial=fd.InitialField(H=np.ones((3, 8, 8, 8)))), id='which-field-is-given'),
]


@pytest.mark.parametrize('changes', STATIC_CHANGES)
def test_the_skeleton_moves_on_a_static_value(changes):
    """Gate 9: a static change is a different plan, and the hash says so."""
    assert fd.plan(reference_scene(**changes)).skeleton != fd.plan(reference_scene()).skeleton


def test_the_static_changes_are_pairwise_distinct():
    """Gate 9: no two of the static variants above hash alike either."""
    skeletons = [fd.plan(reference_scene(**p.values[0])).skeleton for p in STATIC_CHANGES]
    assert len(set(skeletons)) == len(skeletons)


def test_the_skeleton_does_not_move_on_an_initial_amplitude_or_the_field_values():
    """Gate 9: the excitation's size and pattern are leaves of the params, not of the plan."""
    scene = reference_scene()
    s = fd.plan(scene).skeleton
    assert fd.plan(scene.replace(initial=scene.initial.replace(amplitude=2.0))).skeleton == s
    assert fd.plan(scene.replace(initial=scene.initial.replace(amplitude=-0.25))).skeleton == s
    other = fd.InitialField(E=(None, lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * z), None))
    assert fd.plan(scene.replace(initial=other)).skeleton == s
    shape = fd.plan(scene).shape
    rng = np.random.default_rng(0)
    assert fd.plan(scene.replace(initial=fd.InitialField(E=rng.standard_normal(shape)))).skeleton == s
    assert fd.plan(scene.replace(initial=fd.InitialField(E=np.zeros(shape)))).skeleton == s


def test_the_skeleton_does_not_move_on_a_bloch_phase():
    """Gate 9: the phase is traced (a gradient may flow through it); only 'this face is Bloch' is static."""
    s = fd.plan(bloch_scene(0.7)).skeleton
    assert fd.plan(bloch_scene(0.9)).skeleton == s
    assert fd.plan(bloch_scene(-2.5)).skeleton == s
    assert fd.plan(bloch_scene(np.float64(0.7))).skeleton == s
    by_k = reference_scene(boundaries=fd.Boundaries(x=fd.Bloch(k=0.7), y=fd.Periodic(), z=fd.PEC()))
    assert fd.plan(by_k).skeleton == s


def test_the_skeleton_does_not_move_on_a_torch_tensor_phase():
    """Gate 9: a phase that is a live tensor (requires_grad) plans, hashes alike, and runs complex."""
    torch = pytest.importorskip('torch')
    phase = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    scene = bloch_scene(phase)
    assert fd.plan(scene).skeleton == fd.plan(bloch_scene(0.9)).skeleton
    res = fd.run(scene, backend='torch')
    assert res.final.E.dtype == torch.complex128
    assert np.allclose(to_numpy(res['probe'].values),
                       to_numpy(fd.run(bloch_scene(0.9), backend='torch')['probe'].values),
                       rtol=0, atol=1e-13)


def test_run_refuses_a_plan_of_another_skeleton_by_name():
    """Gate 9: the refusal names ``plan=`` and both skeletons, so the reader sees which scene the plan was for."""
    scene = reference_scene()
    mine = fd.plan(scene)
    for changes in (dict(run=fd.Run(steps=21)), dict(resolution=fd.Resolution(dx=1 / 16)),
                    dict(boundaries=fd.Boundaries(x=fd.PMC(), y=fd.PEC(), z=fd.PEC()))):
        other = fd.plan(reference_scene(**changes))
        with pytest.raises(ValueError, match='different scene') as e:
            fd.run(scene, plan=other)
        message = str(e.value)
        assert 'plan=' in message
        assert other.skeleton[:12] in message and mine.skeleton[:12] in message


def test_run_accepts_the_scenes_own_plan_and_uses_it():
    """Gate 9 companion: ``plan=fd.plan(scene)`` is accepted, is the plan the result carries, and changes nothing."""
    scene = reference_scene()
    plan = fd.plan(scene)
    with_plan = fd.run(scene, plan=plan)
    without = fd.run(scene)
    assert with_plan.plan is plan
    assert without.plan.skeleton == plan.skeleton
    assert np.array_equal(to_numpy(with_plan.final.E), to_numpy(without.final.E))
    assert np.array_equal(to_numpy(with_plan['probe'].values), to_numpy(without['probe'].values))
    # A plan built from an equal scene is the same plan, so it is accepted too.
    assert fd.run(scene, plan=fd.plan(reference_scene())).plan.skeleton == plan.skeleton


@pytest.mark.parametrize('field, changes', [
    pytest.param('steps', dict(run=fd.Run(steps=21)), id='steps'),
    pytest.param('observers', dict(monitors=(fd.Record('probe', at=PROBE, component='Hz'),)),
                 id='observers'),
    pytest.param('has_H0', dict(initial=fd.InitialField(E=_INITIAL, H=np.zeros((3, 8, 8, 8)))),
                 id='which-field-is-given'),
])
def test_mutant_a_skeleton_blind_to_one_static_field_lets_two_scenes_collide(field, changes):
    """Gate 9 mutant: drop one static field from the hash and two different scenes hash alike.

    The three fields are the ones a scene change moves on its own (a moved box moves the
    walls and the probe with it, so ``low`` is never alone).
    """
    a, b = fd.plan(reference_scene()), fd.plan(reference_scene(**changes))
    assert a.skeleton != b.skeleton
    sa, sb = _static(a), _static(b)
    assert sa[field] != sb[field]
    assert skeleton_of(**sa) == a.skeleton and skeleton_of(**sb) == b.skeleton
    del sa[field], sb[field]
    assert sa == sb                             # the only difference was that field
    assert skeleton_of(**sa) == skeleton_of(**sb)


def test_a_field_marked_for_a_digest_is_hashed_and_not_written_out():
    """Gate 9: ``metadata={'digest': True}`` encodes a field as ``['digest', sha256]``.

    The rule is the static half of a prepared material map (stage 4): arrays whose *content*
    has to move the skeleton but which have no business being written into the hash input.
    Nothing in the package carries the mark: no field under ``wavebit/`` declares
    ``metadata=`` at all, so no plan built today can take the branch, and every frozen
    skeleton in ``tests/references/fd_golden_stage*.npz`` is unchanged by it -- checked by
    rebuilding all 23 stage 0-3 reference skeletons against a ``git archive`` of this
    commit's parent, and by the four freezers, which reproduce their arrays bit for bit.

    Three claims, and the third is the one a mutant breaks first: the mark routes the field
    to its digest, two arrays one entry apart give two different digests, and *without* the
    mark the same field has no encoding at all. Drop the ``f.metadata`` guard and every
    metadata-free field -- ``nx``, ``face_rules``, all of them -- takes the branch instead.
    """
    from wavebit.fdtd._prepare.plan import _encode

    @dataclasses.dataclass(frozen=True)
    class Marked:
        label: str
        arrays: tuple = dataclasses.field(default=(), metadata={'digest': True})

    @dataclasses.dataclass(frozen=True)
    class Plain:
        label: str
        arrays: tuple = ()

    a = np.arange(6.0).reshape(2, 3)
    b = a.copy()
    b[1, 2] += 1.0
    name, fields = _encode(Marked('m', (a,)))
    assert name == 'Marked' and fields[0] == ['label', ['str', 'm']]
    # The contract stated in ``_digest``, restated here: dtype, shape, then the C-order bytes.
    want = hashlib.sha256(f'{a.dtype.str}{a.shape}'.encode() + a.tobytes()).hexdigest()
    assert fields[1] == ['arrays', ['digest', want]]
    assert _encode(Marked('m', (b,)))[1][1] != fields[1]
    assert skeleton_of(terms=(Marked('m', (a,)),)) != skeleton_of(terms=(Marked('m', (b,)),))
    # Unmarked, the same content is refused outright: the mark is what routes it.
    with pytest.raises(TypeError, match='no place in a static plan'):
        _encode(Plain('m', (a,)))


@dataclasses.dataclass(frozen=True)
class _Digested:
    """A term whose one field is marked for a digest -- the shape stage 4's material map has."""

    label: str
    arrays: tuple = dataclasses.field(default=(), metadata={'digest': True})


class _Addressed:
    """A value with no array behind it: ``np.ascontiguousarray`` makes it an ``object`` array."""


def test_a_value_that_carries_an_address_cannot_be_digested():
    """Gate 9: the digest's "no ``id()``, no ``repr()``, no address" is enforced, not asserted.

    ``_digest`` calls ``np.ascontiguousarray`` on whatever the field holds, and that takes
    *anything*: a list of Python objects, or one object, becomes an ``object`` array whose
    ``tobytes()`` is the array of **pointers**. Hashing those is hashing addresses -- the
    thing the docstring promises does not happen, the thing this repository has a memory
    about (``a-digest-you-did-not-validate``: a ``repr()`` carries an address, so run it
    twice), and the thing that would quietly make a skeleton unreproducible while every
    single-process test still passed.

    Measured here rather than argued: the refusal is checked first, and then what was behind
    it -- two objects built one after the other have the same (empty) content and different
    bytes, because those bytes are their addresses. A ``str`` and a ``datetime64`` array are
    refused by the same door. Both of those would in fact digest reproducibly; the door
    admits the kinds that **are** content (``biufc``) rather than the kinds that happen to be
    safe, because the second list has to be re-decided at every addition.

    The mutant is the guard deleted. Measured 2026-09-24: **this test alone** goes red, on
    the first ``DID NOT RAISE``, and the other 131 cases in this file pass -- including
    ``test_a_digest_is_the_same_number_in_two_interpreters``, which digests ``float64`` and
    ``int32`` arrays and cannot see an address. That is the shape of the defect and the
    reason the door is here rather than in a comparison downstream: an unguarded digest is
    wrong only for the values nothing else in the suite hands it, and it is wrong silently.
    """
    from wavebit.fdtd._prepare.plan import _digest, _encode

    for bad, kind in ((np.array([_Addressed(), _Addressed()], dtype=object), 'O'),
                      (_Addressed(), 'O'),
                      (np.array(['ab', 'cd']), 'U'),
                      (np.array(['2026-09-24'], dtype='datetime64[D]'), 'M')):
        assert np.ascontiguousarray(bad).dtype.kind == kind
        with pytest.raises(TypeError, match='addresses of its elements'):
            _digest(bad)
        with pytest.raises(TypeError, match='addresses of its elements'):
            _encode(_Digested('m', (bad,)))
        with pytest.raises(TypeError, match='addresses of its elements'):
            skeleton_of(terms=(_Digested('m', (bad,)),))
    # What was behind the door: the bytes of an object array are pointers, so two objects
    # with identical (empty) content do not digest alike.
    raw = [np.ascontiguousarray(np.array([obj], dtype=object)).tobytes()
           for obj in (_Addressed(), _Addressed())]
    assert raw[0] != raw[1], 'an object array would have to carry content for this to tie'
    # Every kind the door admits still goes through, including a bool and a complex one.
    for good in (np.arange(6.0), np.arange(4, dtype=np.int32), np.array([True, False]),
                 np.array([1 + 2j]), np.uint8(3)):
        assert _digest(good)[0] == 'digest'


def _digest_in_a_subprocess(hashseed):
    """``_digest`` of one fixed pair of arrays, in a fresh interpreter under ``hashseed``."""
    code = ('import numpy as np\n'
            'from wavebit.fdtd._prepare.plan import _digest\n'
            'a = np.arange(6.0).reshape(2, 3)\n'
            'b = np.arange(4, dtype=np.int32)\n'
            'print(_digest((a, b))[1])')
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run([PY, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True, timeout=300)
    return out.stdout.strip()


def test_a_digest_is_the_same_number_in_two_interpreters():
    """Gate 9: a digest is a function of the content, so two processes must agree on it.

    The twin of :func:`test_the_skeleton_is_the_same_in_two_subprocesses`, one level down and
    for the same reason: a skeleton that holds a digest is compared across processes by
    ``fd.run(scene, plan=)``, and a digest that read an address would differ on every run
    while agreeing with itself inside one. Two fresh interpreters under two ``PYTHONHASHSEED``
    values, digesting the same ``float64`` and ``int32`` arrays, against this process's answer.
    """
    from wavebit.fdtd._prepare.plan import _digest
    here = _digest((np.arange(6.0).reshape(2, 3), np.arange(4, dtype=np.int32)))[1]
    first, second = _digest_in_a_subprocess(1), _digest_in_a_subprocess(2)
    assert first == second == here
    assert len(here) == 64 and int(here, 16) >= 0          # a sha256 hex digest


# ---------------------------------------------------------------------------------------------
# The doors: Boundaries
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('missing', ['x', 'y', 'z'])
def test_boundaries_missing_an_axis_is_refused_naming_it(missing):
    """Gate 9 door: there is no default face; the refusal names the axis left out."""
    given = {axis: fd.PEC() for axis in 'xyz' if axis != missing}
    with pytest.raises(TypeError, match=f"'{missing}'"):
        fd.Boundaries(**given)
    with pytest.raises(TypeError, match='face type'):
        fd.Boundaries(**dict(given, **{missing: 'pec'}))
    with pytest.raises(TypeError, match='face type'):
        fd.Boundaries(**dict(given, **{missing: fd.PEC}))       # the class, not a face


def test_boundaries_with_every_axis_is_accepted_in_either_spelling():
    """Gate 9 companion: one face type for an axis, or a (low, high) pair, both name six faces."""
    both = fd.Boundaries(x=fd.PEC(), y=fd.PMC(), z=fd.Periodic())
    pairs = fd.Boundaries(x=(fd.PEC(), fd.PEC()), y=(fd.PMC(), fd.PMC()), z=fd.Periodic())
    assert both == pairs
    assert [both.face(n) for n in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')] == \
        [fd.PEC(), fd.PEC(), fd.PMC(), fd.PMC(), fd.Periodic(), fd.Periodic()]
    mixed = fd.Boundaries(x=(fd.PEC(), fd.PMC()), y=fd.PEC(), z=fd.PEC())
    assert mixed.face('x-') == fd.PEC() and mixed.face('x+') == fd.PMC()
    plan = fd.plan(reference_scene(boundaries=mixed))
    assert plan.faces['x-'].kind == 'pec' and plan.faces['x+'].kind == 'pmc'
    assert plan.faces['x-'].position == -0.5 and plan.faces['x+'].position == 0.5 - plan.dx / 2


# ---------------------------------------------------------------------------------------------
# The doors: unknown keywords, on every public constructor
# ---------------------------------------------------------------------------------------------

CONSTRUCTORS = [
    # (the misspelt call, the call it meant, the keyword the refusal must suggest)
    pytest.param(lambda: fd.Domain(sizes=(1, 1, 1)),
                 lambda: fd.Domain(size=(1, 1, 1)), 'size', id='Domain'),
    pytest.param(lambda: fd.Domain(size=(1, 1, 1), centre=(0, 0, 0)),
                 lambda: fd.Domain(size=(1, 1, 1), center=(0, 0, 0)), 'center', id='Domain-center'),
    pytest.param(lambda: fd.Resolution(d_x=0.1),
                 lambda: fd.Resolution(dx=0.1), 'dx', id='Resolution'),
    pytest.param(lambda: fd.Resolution(cells_per_wavelength=10, wavelenght=1.0),
                 lambda: fd.Resolution(cells_per_wavelength=10, wavelength=1.0), 'wavelength',
                 id='Resolution-wavelength'),
    pytest.param(lambda: fd.Run(step=10),
                 lambda: fd.Run(steps=10), 'steps', id='Run'),
    pytest.param(lambda: fd.Run(steps=10, courant_number=0.5),
                 lambda: fd.Run(steps=10, courant=0.5), 'courant', id='Run-courant'),
    pytest.param(lambda: fd.Boundaries(xs=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                 lambda: fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()), 'x', id='Boundaries'),
    pytest.param(lambda: fd.Bloch(phases=0.3),
                 lambda: fd.Bloch(phase=0.3), 'phase', id='Bloch'),
    pytest.param(lambda: fd.Bloch(kx=0.3),
                 lambda: fd.Bloch(k=0.3), 'k', id='Bloch-k'),
    pytest.param(lambda: fd.Symmetry(parities=1),
                 lambda: fd.Symmetry(parity=1), 'parity', id='Symmetry'),
    pytest.param(lambda: fd.InitialField(E=_INITIAL, amplitudes=2.0),
                 lambda: fd.InitialField(E=_INITIAL, amplitude=2.0), 'amplitude', id='InitialField'),
    pytest.param(lambda: fd.Record('probe', at=(0, 0, 0), components='Ey'),
                 lambda: fd.Record('probe', at=(0, 0, 0), component='Ey'), 'component', id='Record'),
    pytest.param(lambda: fd.Scene(monitor=(), **_SCENE_KW),
                 lambda: fd.Scene(monitors=(), **_SCENE_KW), 'monitors', id='Scene'),
    pytest.param(lambda: fd.Scene(resolutions=fd.Resolution(dx=0.1), **_SCENE_KW),
                 lambda: fd.Scene(**_SCENE_KW), 'resolution', id='Scene-resolution'),
]


@pytest.mark.parametrize('misspelt, meant, suggestion', CONSTRUCTORS)
def test_an_unknown_keyword_is_refused_with_the_nearest_name(misspelt, meant, suggestion):
    """Gate 9 door: every public constructor refuses a misspelt keyword and says which one it is nearest to."""
    with pytest.raises(TypeError) as e:
        misspelt()
    message = str(e.value)
    assert 'does not take' in message
    assert f"did you mean {suggestion!r}?" in message
    assert 'silently ignored' in message          # the reason, stated in the refusal


@pytest.mark.parametrize('misspelt, meant, suggestion', CONSTRUCTORS)
def test_the_keyword_it_meant_is_accepted(misspelt, meant, suggestion):
    """Gate 9 companion: the suggested spelling builds the object and carries the value."""
    built = meant()
    assert built == meant()
    fields = {f.name for f in dataclasses.fields(built)}
    assert suggestion in fields


@pytest.mark.parametrize('face, keyword', [
    pytest.param(fd.Periodic, 'phase', id='Periodic'),
    pytest.param(fd.PEC, 'parity', id='PEC'),
    pytest.param(fd.PMC, 'sigma', id='PMC'),
])
def test_a_bare_face_type_takes_no_keyword_and_says_so(face, keyword):
    """Gate 9 door: the faces without parameters refuse any keyword, naming the empty list they take."""
    with pytest.raises(TypeError, match='does not take') as e:
        face(**{keyword: 1})
    assert 'It takes []' in str(e.value)
    assert face() == face()                       # the accepted form: no arguments at all


def test_replace_goes_through_the_same_door():
    """Gate 9 door: ``scene.replace(monitor=...)`` is refused like the constructor, and ``replace`` re-validates."""
    scene = reference_scene()
    with pytest.raises(TypeError, match="did you mean 'monitors'"):
        scene.replace(monitor=())
    with pytest.raises(ValueError, match='outside'):
        scene.replace(monitors=(fd.Record('far', at=(2.0, 0.0, 0.0)),))
    moved = scene.replace(run=fd.Run(steps=40))
    assert moved.run.steps == 40 and scene.run.steps == 20 and moved.domain is scene.domain


# ---------------------------------------------------------------------------------------------
# The doors: Periodic and Bloch pair up, Symmetry does not
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('pair', [
    pytest.param((fd.Periodic(), fd.PEC()), id='periodic-low'),
    pytest.param((fd.PEC(), fd.Periodic()), id='periodic-high'),
    pytest.param((fd.Periodic(), fd.PMC()), id='periodic-pmc'),
    pytest.param((fd.Symmetry(parity=1), fd.Periodic()), id='periodic-symmetry'),
    pytest.param((fd.Bloch(phase=0.3), fd.PEC()), id='bloch-low'),
    pytest.param((fd.Periodic(), fd.Bloch(phase=0.3)), id='periodic-bloch'),
    pytest.param((fd.Bloch(phase=0.3), fd.Bloch(phase=0.4)), id='two-bloch-phases'),
    pytest.param((fd.Bloch(phase=0.3), fd.Bloch(k=0.3)), id='bloch-phase-and-k'),
])
def test_periodic_or_bloch_on_one_face_only_is_refused(pair):
    """Gate 9 door: a wrapping face wraps the axis; the refusal names the one object to give."""
    with pytest.raises(ValueError, match='both faces') as e:
        fd.Boundaries(x=pair, y=fd.PEC(), z=fd.PEC())
    assert 'Give one object for the axis' in str(e.value)


def test_periodic_and_bloch_on_both_faces_are_accepted():
    """Gate 9 companion: one object for the axis, an equal pair, and the preset all wrap the axis."""
    one = fd.Boundaries(x=fd.Periodic(), y=fd.Bloch(phase=0.3), z=fd.PEC())
    pair = fd.Boundaries(x=(fd.Periodic(), fd.Periodic()), y=(fd.Bloch(phase=0.3), fd.Bloch(phase=0.3)),
                         z=fd.PEC())
    assert one == pair
    preset = fd.Boundaries.periodic(z=fd.PEC())
    assert preset == fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.PEC())
    plan = fd.plan(reference_scene(boundaries=one))
    assert plan.faces['x-'].kind == plan.faces['x+'].kind == 'periodic'
    assert plan.faces['y-'].kind == plan.faces['y+'].kind == 'bloch'
    assert plan.faces['x-'].position is None      # a wrapping face has no wall


@pytest.mark.parametrize('value', [
    pytest.param(fd.Symmetry(parity=1), id='one-object-both-faces'),
    pytest.param((fd.Symmetry(parity=1), fd.Symmetry(parity=1)), id='pair-same-parity'),
    pytest.param((fd.Symmetry(parity=1), fd.Symmetry(parity=-1)), id='pair-opposite-parity'),
])
def test_symmetry_on_both_faces_is_refused(value):
    """Gate 9 door: two mirrors on one axis is not a half box; the refusal says what to pair it with."""
    with pytest.raises(ValueError, match='both faces') as e:
        fd.Boundaries(x=value, y=fd.PEC(), z=fd.PEC())
    assert 'PEC' in str(e.value) and 'PMC' in str(e.value)


@pytest.mark.parametrize('parity', [1, -1])
@pytest.mark.parametrize('far', [fd.PEC(), fd.PMC()], ids=['pec', 'pmc'])
def test_symmetry_on_one_face_is_accepted_and_the_plan_records_the_plane(parity, far):
    """Gate 9 companion: a mirror with PEC or PMC opposite, on either face; the plan says where the plane is."""
    low = fd.Boundaries(x=(fd.Symmetry(parity=parity), far), y=far, z=far)
    high = fd.Boundaries(x=(far, fd.Symmetry(parity=parity)), y=far, z=far)
    assert low.face('x-') == fd.Symmetry(parity=parity) and low.face('x+') == far
    assert high.face('x+') == fd.Symmetry(parity=parity) and high.face('x-') == far
    p_low, p_high = fd.plan(reference_scene(boundaries=low)), fd.plan(reference_scene(boundaries=high))
    assert (p_low.faces['x-'].kind, p_low.faces['x-'].plane, p_low.faces['x-'].parity) == \
        ('symmetry', 'node', parity)
    assert (p_high.faces['x+'].kind, p_high.faces['x+'].plane, p_high.faces['x+'].parity) == \
        ('symmetry', 'half', parity)
    assert p_low.faces['x-'].position == -0.5                       # the face itself
    assert p_high.faces['x+'].position == pytest.approx(0.5 - p_high.dx / 2)   # half a cell inside
    assert p_low.skeleton != p_high.skeleton


# ---------------------------------------------------------------------------------------------
# The doors: the Courant number
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('courant', [1.5, 1.0000001, 1.0, 0.0, -0.5, float('inf')])
def test_a_courant_number_outside_zero_to_one_is_refused_naming_the_cfl_condition(courant):
    """Gate 9 door: ``dt = courant dx / sqrt(3)`` is the 3-D CFL condition; nothing outside (0, 1) is feasible.

    ``1.0`` itself is refused: at the limit the corner Nyquist mode has ``cos(w dt) = -1``
    and the leapfrog recurrence admits ``(-1)^n (a + b n)`` -- measured on a periodic 8^3 box
    from random E of size 1e-3, the energy grows 1e5-fold in 2000 steps at 1.0 and stays
    bounded at 0.999 (2026-09-18). The refusal says so.
    """
    with pytest.raises(ValueError, match='CFL') as e:
        fd.Run(steps=10, courant=courant)
    assert '0 < courant < 1' in str(e.value)
    if courant == 1.0:
        assert 'Nyquist' in str(e.value)


@pytest.mark.parametrize('courant', [0.999, 0.5, 0.05])
def test_a_courant_number_below_one_is_accepted_and_sets_dt(courant):
    """Gate 9 companion: the plan's ``dt`` is ``courant dx / sqrt(3)`` and the run is finite."""
    run = fd.Run(steps=20, courant=courant)
    assert run.courant == courant
    scene = reference_scene(run=run)
    plan = fd.plan(scene)
    assert plan.dt == pytest.approx(courant * plan.dx / np.sqrt(3.0), rel=0, abs=1e-15)
    assert fd.plan(reference_scene()).dt == pytest.approx(0.5 * plan.dx / np.sqrt(3.0))   # the default
    # A stable run of a unit-amplitude mode stays bounded; an unstable one is astronomical
    # after 20 steps. The bound is loose on purpose: it is finiteness that is asserted.
    values = to_numpy(fd.run(scene)['probe'].values)
    assert np.all(np.isfinite(values)) and np.max(np.abs(values)) < 2.0


# ---------------------------------------------------------------------------------------------
# The doors: Bloch with no phase
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('kwargs', [
    pytest.param(dict(phase=0.0), id='phase-float'),
    pytest.param(dict(phase=0), id='phase-int'),
    pytest.param(dict(phase=-0.0), id='phase-negative-zero'),
    pytest.param(dict(phase=np.float64(0.0)), id='phase-np-float64'),
    pytest.param(dict(phase=np.array(0.0)), id='phase-np-array'),
    pytest.param(dict(k=0.0), id='k-float'),
    pytest.param(dict(k=0), id='k-int'),
])
def test_bloch_with_a_concrete_zero_is_refused_naming_periodic(kwargs):
    """Gate 9 door: the kernel's decision is a refusal, not a silent lowering -- a complex run for nothing."""
    with pytest.raises(ValueError, match='Periodic') as e:
        fd.Bloch(**kwargs)
    assert 'complex' in str(e.value)


def test_bloch_with_a_phase_is_accepted_and_makes_the_run_complex():
    """Gate 9 companion: any nonzero concrete phase, by ``phase`` or by ``k``, plans as a complex run."""
    assert fd.Bloch(phase=0.3).phase == 0.3 and fd.Bloch(k=0.3).k == 0.3
    assert fd.Bloch(phase=1e-9).phase == 1e-9                  # small is not zero
    assert fd.plan(bloch_scene(0.3)).dtype == 'complex128'
    assert fd.plan(reference_scene()).dtype == 'float64'
    res = fd.run(bloch_scene(0.3))
    assert to_numpy(res.final.E).dtype == np.complex128
    with pytest.raises(ValueError, match='exactly one'):
        fd.Bloch(phase=0.3, k=0.3)
    with pytest.raises(ValueError, match='exactly one'):
        fd.Bloch()


@pytest.mark.parametrize('value', [
    pytest.param('abc', id='str'),
    pytest.param([1, 2], id='list'),
    pytest.param(1j, id='complex'),
    pytest.param(np.array([0.3, 0.4]), id='two-phases'),
    pytest.param(True, id='bool'),
    pytest.param(object(), id='object'),
])
@pytest.mark.parametrize('what', ['phase', 'k'])
def test_bloch_with_anything_but_a_real_scalar_is_refused_naming_bloch(what, value):
    """Gate 9 door: the one value that passes unread is a tracer; a string, a list, a complex
    number or two phases used to construct and fail deep in the lowering with a NumPy
    message naming neither ``Bloch`` nor ``phase`` (2026-09-18)."""
    with pytest.raises(ValueError, match=rf'Bloch\({what}=') as e:
        fd.Bloch(**{what: value})
    assert 'real scalar' in str(e.value)


def test_bloch_accepts_a_traced_phase_unread():
    """Gate 9 companion: a JAX tracer and a Torch leaf pass the door; the leaf is read for the zero check."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    seen = []

    def build(phi):
        seen.append(fd.Bloch(phase=phi))
        return phi * 2.0
    jax.grad(build)(0.7)
    assert seen and seen[0].phase is not None
    assert fd.Bloch(phase=torch.tensor(0.7, requires_grad=True)).phase.requires_grad
    with pytest.raises(ValueError, match='Periodic'):
        fd.Bloch(phase=torch.tensor(0.0, requires_grad=True))


# ---------------------------------------------------------------------------------------------
# The doors: later stages, refused naming the stage
# ---------------------------------------------------------------------------------------------

#: Stage 1 turned ``Boundaries.open`` from a named refusal into the absorber preset, so the
#: gate-9 door that pinned the refusal now pins the preset. The refusal it replaces is kept
#: for the two spellings that are still not a preset: a bare number, and an unknown keyword.
def test_boundaries_open_is_the_absorber_preset():
    """Gate 9 door, stage 1: ``open`` puts one absorbing face on every axis, with overrides."""
    pml = fd.PML(cells=12, wavelength=0.55)
    everywhere = fd.Boundaries.open(pml)
    assert all(everywhere.face(f) is pml for f in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+'))
    mixed = fd.Boundaries.open(pml, x=fd.Periodic(), z=(fd.Symmetry(parity=-1), pml))
    assert mixed.face('x-') == fd.Periodic() and mixed.face('y+') is pml
    assert mixed.face('z-') == fd.Symmetry(parity=-1) and mixed.face('z+') is pml
    # the default face is a PML with no wavelength, which the scene -- not the preset --
    # refuses, so the preset never picks a CFS pole for anybody
    assert isinstance(fd.Boundaries.open().face('z-'), fd.PML)


@pytest.mark.parametrize('args, kwargs', [
    pytest.param((), dict(pml=0.5), id='unknown-keyword'),
    pytest.param((0.5,), {}, id='a-bare-number'),
    pytest.param((fd.PEC(),), {}, id='a-non-absorbing-face'),
])
def test_boundaries_open_refuses_what_is_not_an_absorbing_face(args, kwargs):
    with pytest.raises(TypeError) as e:
        fd.Boundaries.open(*args, **kwargs)
    assert 'Boundaries.open' in str(e.value)


def test_the_periodic_preset_is_the_one_that_works():
    """Gate 9 companion: ``Boundaries.periodic(z=)`` is the stage-0 preset and takes z by keyword only."""
    preset = fd.Boundaries.periodic(z=(fd.PEC(), fd.PMC()))
    assert preset.face('x-') == fd.Periodic() and preset.face('y+') == fd.Periodic()
    assert preset.face('z-') == fd.PEC() and preset.face('z+') == fd.PMC()
    with pytest.raises(TypeError):
        fd.Boundaries.periodic(fd.PEC())


LATER_STAGES = [
    pytest.param('pml', 'stage 1', id='pml'),
    pytest.param('materials', 'stage 1', id='materials'),
    pytest.param('source', 'stage 2', id='source'),       # the stage-2 spelling is sources=
    pytest.param('shapes', 'stage 4', id='shapes'),
    pytest.param('structure', 'stage 4', id='structure'),
    pytest.param('decay', r'Run\(decay=Decay', id='decay'),
]


@pytest.mark.parametrize('keyword, stage', LATER_STAGES)
def test_a_later_stage_keyword_on_scene_is_refused_naming_the_stage(keyword, stage):
    """A misplaced or pending Scene keyword names its supported spelling or stage."""
    with pytest.raises(NotImplementedError, match=stage) as e:
        fd.Scene(**{keyword: ()}, **_SCENE_KW)
    message = str(e.value)
    assert f'Scene({keyword}=...)' in message
    assert 'did you mean' not in message
    with pytest.raises(NotImplementedError, match=stage):
        reference_scene().replace(**{keyword: ()})


def test_run_periods_needs_a_source_and_is_one_length_or_the_other():
    """Gate 9 door, as stage 2 narrowed it: ``periods`` is legal and counts a source's periods.

    A scene with no source has nothing to count periods of, and ``steps`` with ``periods`` is
    two lengths for one run; tests/fdtd/test_fd_sources_doors.py holds the rest of the doors.
    """
    assert fd.Run(periods=10).periods == 10.0
    with pytest.raises(ValueError, match='one or the other'):
        fd.Run(steps=10, periods=10)
    with pytest.raises(ValueError, match='no source') as e:
        fd.plan(fd.Scene(**dict(_SCENE_KW, run=fd.Run(periods=10))))
    assert 'Run(steps=...)' in str(e.value)


def test_a_stage_0_scene_is_accepted_and_runs():
    """Gate 9 companion: the same four keywords without the early ones build, plan, check and run."""
    scene = fd.Scene(**_SCENE_KW)                    # no initial field, no monitors
    assert scene.initial is None and scene.monitors == ()
    plan = fd.plan(scene)
    assert (plan.nx, plan.ny, plan.nz) == (8, 8, 8) and plan.has_E0 is False and plan.has_H0 is False
    res = fd.run(scene)
    assert not np.any(to_numpy(res.final.E)) and not np.any(to_numpy(res.final.H))
    res = fd.run(reference_scene())
    assert res['probe'].values.shape == (20,) and np.any(to_numpy(res.final.E))


# ---------------------------------------------------------------------------------------------
# The doors: the parity is a value, not a Python type
# ---------------------------------------------------------------------------------------------

def _mirror_scene(parity):
    return reference_scene(boundaries=fd.Boundaries(x=(fd.Symmetry(parity=parity), fd.PEC()),
                                                    y=fd.PEC(), z=fd.PEC()))


@pytest.mark.parametrize('parity', [
    pytest.param(np.int64(1), id='np-int64'),
    pytest.param(np.int32(-1), id='np-int32'),
    pytest.param(1.0, id='float'),
    pytest.param(-1.0, id='negative-float'),
])
def test_a_parity_spelled_another_way_is_the_same_mirror(parity):
    """Gate 9 door: ``np.int64(1)`` and ``1.0`` are the mirror ``1`` is -- one plan, one skeleton.

    The skeleton encodes the Python type of a static value, so the door normalises the
    parity to ``int``; before it did, ``np.int64(1)`` passed the door and failed in
    ``fd.plan`` as "int64 has no place in a static plan", and ``1.0`` hashed apart from ``1``
    so ``fd.run(scene, plan=)`` refused a plan built from an equal scene (2026-09-18).
    """
    mirror = fd.Symmetry(parity=parity)
    assert mirror.parity == int(parity) and type(mirror.parity) is int
    scene, reference = _mirror_scene(parity), _mirror_scene(int(parity))
    plan = fd.plan(scene)
    assert plan.faces['x-'].parity == int(parity)
    assert plan.skeleton == fd.plan(reference).skeleton
    assert fd.run(scene, plan=fd.plan(reference)).plan.skeleton == plan.skeleton


@pytest.mark.parametrize('parity', [
    pytest.param(True, id='bool'),
    pytest.param(2, id='two'),
    pytest.param(0.5, id='half'),
    pytest.param('1', id='str'),
    pytest.param(np.array([1, -1]), id='array'),
])
def test_a_parity_that_is_not_plus_or_minus_one_is_refused(parity):
    """Gate 9 door: ``True`` is not a mirror eigenvalue anyone meant; nor is anything but +-1."""
    with pytest.raises(ValueError, match='mirror eigenvalue'):
        fd.Symmetry(parity=parity)
