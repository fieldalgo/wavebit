"""JobSpec is the serialisable form of a run, and the private historical executor executes it statelessly.

Two properties carry the whole design and are gated here:

* **Parity** — a spec must produce byte-identical results to the equivalent
  :func:`wavebit.rcwa` call. The spec layer is a new way to *say* a run, never a new way
  to *compute* one.
* **Round-trip** — a spec must survive JSON unchanged, because it exists to cross process
  and network boundaries. Equal specs must hash equal, and differing ones must not.

Validation is tested for rejection, not just acceptance: a bad spec caught at submission is
far cheaper than one caught on a worker.
"""

from wavebit._common.periodic import Incidence as _legacy_wavebit_Incidence

from wavebit._common.periodic import Layer as _legacy_wavebit_Layer

from wavebit._common._jobs_v1 import Numerics as _legacy_wavebit_Numerics

from wavebit._common.periodic import Stack as _legacy_wavebit_Stack

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

from wavebit._common._jobs_v1 import run as _legacy_wavebit_run
import dataclasses
import json
import re

import numpy as np
import pytest

import wavebit
from wavebit._common._jobs_v1 import JobSpec
from wavebit._common._jobs_v1 import Sweep
from wavebit._common._jobs_v1 import SpecError
from wavebit.rcwa._engine.enums import Factorization
from wavebit import Backend, Material


# `version` is written out, not left to a default, and that is the point of the key rather
# than boilerplate around it: a document that does not state its schema gets refused, because
# version 3 moved the material numbers from refractive index to permittivity and an absent
# version is the absence of a statement about which of the two these are. 12.25 is 3.5**2.
BINARY_GRATING = {
    'version': '10',
    'numerics': {'backend': 'numpy', 'truncation_order': [10, 0]},
    'stack': {
        'top': 1.0, 'bot': 2.25, 'period': [700.0],
        'layers': [{'thickness': 300.0, 'structure': [[1.0, 1.0, 12.25, 12.25]]}],
    },
    'incidence': {'wavelength': 900.0, 'theta': 0.0, 'phi': 0.0, 'tetm': 'te'},
}


# ---------------------------------------------------------------- parity

def test_run_matches_the_equivalent_rcwa_call():
    """The gate: a spec computes exactly what the legacy keyword call computes."""
    result = _legacy_wavebit_run(BINARY_GRATING)

    reference = _legacy_wavebit_rcwa(
        backend='numpy', wavelength=900., period=[700.], truncation_order=[10, 0],
        top=1., bot=2.25, theta=0., phi=0., tetm='te',
        structure=np.array([[[1.0, 1.0, 12.25, 12.25]]]), thickness=[300.],
    ).solve()

    assert np.array_equal(result.R, reference.R)
    assert np.array_equal(result.T, reference.T)
    assert result.R.sum() + result.T.sum() == pytest.approx(1.0, abs=1e-13)


def test_run_accepts_spec_object_mapping_and_json():
    """All three input forms are the same run — the caller picks whichever it holds."""
    spec = JobSpec.from_dict(BINARY_GRATING)
    from_obj = _legacy_wavebit_run(spec)
    from_map = _legacy_wavebit_run(BINARY_GRATING)
    from_str = _legacy_wavebit_run(spec.to_json())

    for other in (from_map, from_str):
        assert np.array_equal(from_obj.R, other.R)
        assert np.array_equal(from_obj.T, other.T)


# ---------------------------------------------------------------- round-trip and identity

def test_spec_survives_json_unchanged():
    spec = JobSpec.from_dict(BINARY_GRATING)
    revived = JobSpec.from_json(spec.to_json())

    assert revived.to_dict() == spec.to_dict()
    assert revived.key() == spec.key()
    json.dumps(spec.to_dict())                      # must be plain JSON, not just dict-shaped


def test_key_tracks_content_not_spelling():
    spec = JobSpec.from_dict(BINARY_GRATING)

    # same run, written differently: defaults stated vs omitted
    reworded = dict(BINARY_GRATING)
    reworded['numerics'] = {'backend': 'numpy', 'truncation_order': [10, 0], 'complex_dtype': 128}
    reworded['incidence'] = {'wavelength': 900.0, 'theta': 0.0, 'phi': 0.0, 'tetm': 'te'}
    assert JobSpec.from_dict(reworded).key() == spec.key()

    # a different run
    changed = dict(BINARY_GRATING)
    changed['incidence'] = {**BINARY_GRATING['incidence'], 'wavelength': 901.0}
    assert JobSpec.from_dict(changed).key() != spec.key()


def test_tetm_normalises_to_the_word_and_has_no_interior():
    """``tetm`` is a declaration with exactly two values — the canonical form is the word.

    The retired ``pol`` normalised the other way, to a TM *fraction*, because it accepted the
    interior of ``[0, 1]``. That interior is what made it two different kinds of thing under
    one key: at the ends a declaration, in between a linear angle in a bad unit. A spec that
    wants the angle writes ``psi``.
    """
    assert JobSpec.from_dict(BINARY_GRATING).to_dict()['incidence']['tetm'] == 'te'
    tm = JobSpec.from_dict({**BINARY_GRATING,
                            'incidence': {**BINARY_GRATING['incidence'], 'tetm': 'tm'}})
    assert tm.to_dict()['incidence']['tetm'] == 'tm'
    assert JobSpec.from_json(tm.to_json()).key() == tm.key()
    assert tm.key() != JobSpec.from_dict(BINARY_GRATING).key()

    for interior in (0.5, 0, 1):
        with pytest.raises(SpecError, match='tetm'):
            JobSpec.from_dict({**BINARY_GRATING,
                               'incidence': {**BINARY_GRATING['incidence'], 'tetm': interior}})


def test_enums_accept_name_or_value():
    spec = JobSpec.from_dict({
        **BINARY_GRATING,
        'numerics': {'backend': 'jax', 'factorization': 'NVF', 'truncation_order': 3},
    })
    assert spec.numerics.backend is Backend.JAX
    assert spec.numerics.factorization is Factorization.NVF
    assert spec.numerics.truncation_order == (3, 0)              # a bare int is the x order


# ---------------------------------------------------------------- sweeps

def test_sweep_expands_to_the_cartesian_product():
    spec = JobSpec.from_dict({
        **BINARY_GRATING,
        # radians, and the test said 10.0 until the units rule caught it
        'sweep': {'wavelength': {'linspace': [500.0, 900.0, 5]}, 'theta': [0.0, 0.2]},
    })
    assert len(spec) == 10                          # 5 wavelengths x 2 angles

    points = spec.expand()
    assert len(points) == 10
    assert [(p.incidence.wavelength, p.incidence.theta) for p in points[:3]] == [
        (500.0, 0.0), (500.0, 0.2), (600.0, 0.0)]

    # every point is independently executable, and distinct
    assert all(not p.sweep for p in points)
    assert len({p.key() for p in points}) == 10
    # the scene is untouched by the sweep
    assert all(p.stack.thickness == spec.stack.thickness for p in points)


def test_expand_is_identity_without_a_sweep():
    spec = JobSpec.from_dict(BINARY_GRATING)
    assert len(spec) == 1
    assert spec.expand() == [spec]


def test_linspace_endpoints_are_exact():
    axis = Sweep.from_dict({'wavelength': {'linspace': [500.0, 1000.0, 6]}}).axes['wavelength']
    assert axis[0] == 500.0
    assert axis[-1] == pytest.approx(1000.0)
    assert len(axis) == 6
    assert Sweep.from_dict({'theta': {'linspace': [0.7, 0.7, 1]}}).axes['theta'] == [0.7]


def test_running_a_sweep_spec_is_refused_with_the_way_out():
    spec = JobSpec.from_dict({**BINARY_GRATING, 'sweep': {'wavelength': [500.0, 600.0]}})
    with pytest.raises(SpecError, match=r'expand'):
        _legacy_wavebit_run(spec)


def test_expanded_sweep_points_match_direct_calls():
    """Expansion must not perturb the run — point i is exactly the run with that wavelength."""
    spec = JobSpec.from_dict({**BINARY_GRATING, 'sweep': {'wavelength': [800.0, 900.0]}})

    for point in spec.expand():
        direct = dict(BINARY_GRATING)
        direct['incidence'] = {**BINARY_GRATING['incidence'],
                               'wavelength': point.incidence.wavelength}
        assert np.array_equal(_legacy_wavebit_run(point).R, _legacy_wavebit_run(direct).R)


# ---------------------------------------------------------------- rejection

@pytest.mark.parametrize('spec, message', [
    ({**BINARY_GRATING, 'wavelength': 900.0}, r'unknown field'),          # misplaced key
    ({**BINARY_GRATING, 'numerics': {'truncation_order': [10, 0], 'backand': 0}}, r'unknown field'),
    ({**BINARY_GRATING, 'numerics': {'backend': 'cupy'}}, r'not a Backend'),
    ({**BINARY_GRATING, 'numerics': {'truncation_order': [1, 2, 3]}}, r'1 or 2 values'),
    ({**BINARY_GRATING, 'numerics': {'truncation_order': [-1, 0]}}, r'cannot be negative'),
    ({**BINARY_GRATING, 'numerics': {'complex_dtype': 32}}, r'128 or 64'),
    ({**BINARY_GRATING, 'version': '99'}, r'version'),
    # `wavebit.fdtd` exists; a spec still cannot carry one, because every `numerics` field
    # is harmonic vocabulary. The refusal has to say that rather than that it does not exist.
    ({**BINARY_GRATING, 'solver': 'fdtd'}, r'a spec describes an RCWA run'),
    # removed in version 4, and refused by name: it was never read, so a caller who wrote
    # it was promised something that never happened and needs to be told which half was true
    ({**BINARY_GRATING, 'outputs': []}, r'never read'),
    ({**BINARY_GRATING, 'outputs': ['R']}, r'never read'),
    ({**BINARY_GRATING, 'sweep': {'thickness': [1.0]}}, r'unknown field'),
    ({**BINARY_GRATING, 'sweep': {'wavelength': {}}}, r"one of 'values' or 'linspace'"),
    ({**BINARY_GRATING, 'sweep': {'wavelength': {'values': [], }}}, r'no values'),
    ({**BINARY_GRATING, 'sweep': {'wavelength': {'linspace': [1, 2]}}}, r'start, stop, num'),
    ({**BINARY_GRATING,
      'stack': {'layers': [{'thickness': -1.0, 'structure': [[1.0]]}]}}, r'non-negative'),
    ({**BINARY_GRATING, 'stack': {'layers': [{'structure': [[1.0]]}]}},
     r"'thickness' and 'structure'"),
    # The pre-rename spellings. They were refused *by name* until 2026-08-26, with a sentence
    # about the quantity having moved as well as the key; that machinery is gone, because
    # wavebit has never been released and the reader it addressed does not exist yet. They are
    # unknown fields now, and `SPEC_VERSION` is what a consumer tests.
    ({**BINARY_GRATING, 'stack': dict(BINARY_GRATING['stack'], n_top=1.0)}, r'unknown field'),
    ({**BINARY_GRATING,
      'stack': {'layers': [{'thickness': 1.0, 'eps': [[1.0]]}]}}, r'unknown field'),
    # a material object with a key nothing reads: `mu` misspelled must not read as absent,
    # which would be a silently non-magnetic medium
    ({**BINARY_GRATING, 'stack': dict(BINARY_GRATING['stack'],
                                      top={'epsilon': 1.0, 'my': 1.0})}, r'unknown field'),
    ({**BINARY_GRATING, 'stack': dict(BINARY_GRATING['stack'], top={'mu': 1.0})},
     r"needs 'epsilon'"),
    ({**BINARY_GRATING, 'stack': dict(BINARY_GRATING['stack'], top='Si')},
     r'no lookup from a material name'),
    # `true` is legible and not numeric; left alone it becomes a medium of permittivity 1
    ({**BINARY_GRATING, 'stack': dict(BINARY_GRATING['stack'], bot=True)},
     r'expected a number'),
])
def test_bad_specs_are_rejected_at_submission(spec, message):
    with pytest.raises(SpecError, match=message):
        JobSpec.from_dict(spec)


def test_raster_ref_is_refused_by_the_stateless_entry_point():
    """A reference is not self-contained; ``run`` takes specs that fully determine a run."""
    spec = {**BINARY_GRATING,
            'stack': {'layers': [{'thickness': 300.0,
                                  'structure': {'raster_ref': 's3://x'}}]}}
    with pytest.raises(SpecError, match=r'raster_ref'):
        JobSpec.from_dict(spec)


def test_run_rejects_a_type_it_cannot_interpret():
    with pytest.raises(SpecError, match=r'expected a JobSpec'):
        _legacy_wavebit_run(42)


# ---------------------------------------------------------------- the version gate

#: The same grating in the two readings of its own numbers. `INDEX_NUMBERS` is what a document
#: written the day before the material flip holds: 1.5 and 3.5 are refractive indices, and
#: `BINARY_GRATING` is their squares. Nothing distinguishes the two documents except the
#: quantity their numbers are in, which is what `version` states.
INDEX_NUMBERS = {
    'numerics': {'backend': 'numpy', 'truncation_order': [10, 0]},
    'stack': {
        'top': 1.0, 'bot': 1.5, 'period': [700.0],
        'layers': [{'thickness': 300.0, 'structure': [[1.0, 1.0, 3.5, 3.5]]}],
    },
    'incidence': {'wavelength': 900.0, 'theta': 0.0, 'phi': 0.0, 'tetm': 'te'},
}


def test_a_document_that_states_no_version_is_refused():
    """Absence is not a claim of currency, and reading it as one made the gate a no-op.

    A missing version used to default to :data:`SPEC_VERSION`, which meant the one check that
    exists to catch a stale document skipped precisely the documents that state nothing about
    their age. It is refused now, and the refusal has to be reachable from both directions:
    the well-formed current document without a version, and the stale one.
    """
    versionless = {k: v for k, v in BINARY_GRATING.items() if k != 'version'}
    with pytest.raises(SpecError, match=r'version'):
        JobSpec.from_dict(versionless)
    with pytest.raises(SpecError, match=r'version'):
        _legacy_wavebit_run(versionless)                     # the same gate on the executing path
    with pytest.raises(SpecError, match=r'version'):
        JobSpec.from_json(json.dumps(versionless))

    # stating it is the whole fix; nothing else about the document has to change
    assert JobSpec.from_dict({**versionless, 'version': '10'}).key() == \
        JobSpec.from_dict(BINARY_GRATING).key()


def test_the_stale_document_only_the_version_can_catch():
    """Renamed keys, kept numbers -- the failure the version bump exists for.

    Every other check in this module passes ``INDEX_NUMBERS``: the keys are the version 3
    spellings, the raster is a raster, the angles are radians. It solves, and it conserves
    energy to 1e-13. It is simply a different grating -- R = 0.175 against 0.205 -- and no
    downstream check can see the difference, because both are legal media. The version field
    is the only place in the document where that difference is written down.
    """
    with pytest.raises(SpecError, match=r'version'):
        JobSpec.from_dict(INDEX_NUMBERS)

    # and it is a *different scene*, not a near miss: state the version and it runs, happily,
    # conserving energy and answering a different question
    stale = _legacy_wavebit_run({**INDEX_NUMBERS, 'version': '10'})
    current = _legacy_wavebit_run(BINARY_GRATING)
    assert stale.R.sum() + stale.T.sum() == pytest.approx(1.0, abs=1e-13)
    assert current.R.sum() + current.T.sum() == pytest.approx(1.0, abs=1e-13)
    # far apart, so this is a statement about the scene and not about a tolerance -- and the
    # two numbers the docstring quotes, so that sentence cannot go stale unnoticed
    assert abs(float(stale.R.sum()) - float(current.R.sum())) > 0.02
    assert float(stale.R.sum()) == pytest.approx(0.17530, abs=1e-4)
    assert float(current.R.sum()) == pytest.approx(0.20533, abs=1e-4)


def test_the_version_free_path_is_the_python_one():
    """``JobSpec(...)`` needs no version because it cannot be stale.

    The mandatory field is a rule about *parsing*: ``from_dict`` cannot see whether its mapping
    was typed two lines up or decoded from a queue, so it holds both to the same contract. A
    spec assembled out of this build's own classes has no such ambiguity -- it is versioned by
    the build that assembled it -- and ``to_dict`` always emits the field, so the requirement
    can never break a round trip.
    """
    from wavebit._common._jobs_v1 import SPEC_VERSION
    from wavebit._common.periodic import Stack
    from wavebit._common.periodic import Incidence

    spec = JobSpec(stack=Stack.from_structure(np.array([[[1.0, 12.25]]]), [300.0],
                                              top=1.0, bot=2.25, period=[700.0]),
                   incidence=Incidence(wavelength=900.0, tetm='te'))
    assert spec.version == SPEC_VERSION
    assert 'version' in spec.to_dict()
    assert JobSpec.from_json(spec.to_json()).key() == spec.key()


def test_a_pre_rename_layer_key_is_refused_as_an_unknown_field():
    """``eps`` was refused by name, with the sentence about the quantity moving, until the
    retired machinery was deleted on 2026-08-26.

    It is an unknown field now. That is a real loss of help -- ``eps`` held a grid of
    *refractive indices* and ``structure`` holds permittivity, so a document re-keyed and left
    at its old numbers is a legal grating made of different glass -- and it is the loss the
    deletion accepted: wavebit has never been released, so the reader that sentence addressed
    does not exist yet, and the renames belong in the first release's notes instead.

    Pinned here rather than dropped, because the refusal still has to fire. Silently accepting
    ``eps`` beside ``structure`` is the failure; reporting it plainly is not.
    """
    document = {'version': '10',
                'numerics': BINARY_GRATING['numerics'],
                'incidence': BINARY_GRATING['incidence'],
                'stack': {'top': 1.0, 'bot': 1.5, 'period': [700.0],
                          'layers': [{'thickness': 300.0,
                                      'structure': [[1.0, 1.0, 3.5, 3.5]],
                                      'eps': [[1.0, 1.0, 3.5, 3.5]]}]}}
    with pytest.raises(SpecError) as excinfo:
        JobSpec.from_dict(document)
    message = str(excinfo.value)
    assert "'eps'" in message or "['eps']" in message, message
    assert 'unknown field' in message, message
    assert 'thickness' in message and 'structure' in message, message   # and what is allowed


# ---------------------------------------------------------------- materials

def test_a_material_survives_the_round_trip_at_a_half_space():
    """``Stack.top`` is documented to take a Material, so a spec has to be able to carry one.

    Before this the whole ``Stack`` was unserialisable the moment one appeared: ``_jsonable``
    refused even the non-magnetic ``Material(2.25)``. Loudly, but a schema that one of its own
    domain objects cannot be written in is not one schema shared by every caller.
    """
    from wavebit._common.periodic import Stack
    from wavebit._common.periodic import Incidence

    spec = JobSpec(stack=Stack(layers=[], top=Material(2.25), bot=Material(4.0, mu=1.0),
                               period=[700.0]),
                   incidence=Incidence(wavelength=900.0))
    d = spec.to_dict()
    assert d['stack']['top'] == {'epsilon': 2.25}
    assert d['stack']['bot'] == {'epsilon': 4.0, 'mu': 1.0}
    json.dumps(d)                                    # plain JSON, not merely dict-shaped

    revived = JobSpec.from_json(spec.to_json())
    assert revived.stack.top == Material(2.25)
    assert revived.stack.bot == Material(4.0, mu=1.0)
    assert revived.to_dict() == d and revived.key() == spec.key()

    # `mu` present is the difference between two *different* materials, and the key that
    # carries it has to be present exactly when one was named -- reading a literal 1.0 back as
    # absence would move the scene off the permeability path it asked for.
    assert revived.stack.top.is_magnetic is False
    assert revived.stack.bot.is_magnetic is True
    assert Material(4.0) != Material(4.0, mu=1.0)


def test_null_mu_reads_as_absent_so_there_is_one_parsed_form():
    """JSON's null and a missing key say the same thing, so they must hash the same."""
    named = {**BINARY_GRATING,
             'stack': dict(BINARY_GRATING['stack'], top={'epsilon': 2.25, 'mu': None})}
    absent = {**BINARY_GRATING,
              'stack': dict(BINARY_GRATING['stack'], top={'epsilon': 2.25})}
    assert JobSpec.from_dict(named).key() == JobSpec.from_dict(absent).key()
    assert JobSpec.from_dict(named).stack.top.is_magnetic is False


def test_the_object_form_and_the_scalar_shorthand_are_the_same_physics():
    """``Material(2.25)`` and ``2.25`` are one input, and stay two spellings.

    The physics has to be identical -- the shorthand is exactly what
    :func:`wavebit._common.periodic.as_material` turns into the object. The *documents* stay
    distinct, though: a bare number is written back as a bare number, or every spec written
    before materials had a JSON form would change its ``key()``.
    """
    as_object = {**BINARY_GRATING,
                 'stack': dict(BINARY_GRATING['stack'], bot={'epsilon': 2.25})}
    assert np.array_equal(_legacy_wavebit_run(as_object).R, _legacy_wavebit_run(BINARY_GRATING).R)
    assert JobSpec.from_dict(BINARY_GRATING).to_dict()['stack']['bot'] == 2.25
    assert JobSpec.from_dict(as_object).key() != JobSpec.from_dict(BINARY_GRATING).key()


def test_a_lossy_half_space_is_written_as_a_pair():
    """JSON has no complex type, so a material number may be ``[re, im]`` -- as polarization is.

    Only at a slot the reader knows is a material: ``top`` / ``bot``, and the ``epsilon`` /
    ``mu`` entries of the object form. Inside a recipe a two-list is indistinguishable from a
    recipe fragment, which is why a ``Material`` is the way to put a complex one there.
    """
    lossy = {**BINARY_GRATING,
             'stack': dict(BINARY_GRATING['stack'], bot=[2.25, 0.1])}
    spec = JobSpec.from_dict(lossy)
    assert spec.stack.bot == complex(2.25, 0.1)
    assert JobSpec.from_json(spec.to_json()).key() == spec.key()

    inside = JobSpec.from_dict({**BINARY_GRATING,
                                'stack': dict(BINARY_GRATING['stack'],
                                              top={'epsilon': [1.0, 0.2]})})
    assert inside.stack.top == Material(complex(1.0, 0.2))
    assert JobSpec.from_json(inside.to_json()).key() == inside.key()

    with pytest.raises(SpecError, match=r'\[re, im\]'):
        JobSpec.from_dict({**BINARY_GRATING,
                           'stack': dict(BINARY_GRATING['stack'], bot=[2.25, 0.1, 0.0])})


def test_a_material_inside_a_shape_recipe_survives_too():
    """A Material is self-identifying in JSON, so it travels at any depth the schema allows.

    ``_jsonable`` writes one wherever it meets it -- a block background, a parametric shape's
    ``material`` -- so the reader has to meet it at the same depths, or ``to_dict`` would emit
    a document its own ``from_dict`` reads as something else.
    """
    from wavebit._common.periodic import Stack
    from wavebit._common.periodic import Incidence

    layer = {'background': Material(12.96),
             'shapes': [{'type': 'rectangle', 'cx': 0.0, 'cy': 0.0, 'lx': 0.2, 'ly': 0.2,
                         'material': Material(2.25, mu=1.5)}]}
    spec = JobSpec(stack=Stack(layers=[_legacy_wavebit_Layer(thickness=0.2, structure=layer)],
                               period=[1.0, 1.0]),
                   incidence=Incidence(wavelength=0.9))

    d = spec.to_dict()
    assert d['stack']['layers'][0]['structure']['background'] == {'epsilon': 12.96}
    json.dumps(d)

    revived = JobSpec.from_json(spec.to_json()).stack[0].structure
    assert revived['background'] == Material(12.96)
    assert revived['shapes'][0]['material'] == Material(2.25, mu=1.5)
    assert revived['shapes'][0]['lx'] == 0.2         # the rest of the recipe is untouched
    assert JobSpec.from_json(spec.to_json()).key() == spec.key()


def test_a_raster_layer_is_still_a_raster_and_not_a_recipe():
    """The material revival must not reach into a grid of numbers."""
    spec = JobSpec.from_dict(BINARY_GRATING)
    structure = spec.stack[0].structure
    assert isinstance(structure, np.ndarray)
    assert np.array_equal(structure, np.array([[1.0, 1.0, 12.25, 12.25]]))


# ---------------------------------------------------------------- units

def test_degrees_where_radians_belong_are_refused():
    """The trap this rule exists for: nothing downstream can catch it.

    A polar angle of 10 makes the solver return finite numbers that conserve no energy, and
    they look like absorption, which a lossy stack does legitimately. So the only place it can
    be caught is here.
    """
    with pytest.raises(SpecError, match=r'RADIANS'):
        JobSpec.from_dict({**BINARY_GRATING,
                           'incidence': {**BINARY_GRATING['incidence'], 'theta': 10.0}})

    # and by the same route a sweep takes, which reaches the solver through expand()
    with pytest.raises(SpecError, match=r'RADIANS'):
        JobSpec.from_dict({**BINARY_GRATING, 'sweep': {'theta': [0.0, 5.0]}})


def test_real_polar_angles_pass():
    import math
    spec = JobSpec.from_dict({
        **BINARY_GRATING,
        'sweep': {'theta': [0.0, math.radians(10), math.radians(60), -math.radians(45)]}})
    assert len(spec) == 4
    # A lossless stack must still conserve energy at a real angle.
    for point in spec.expand():
        r = _legacy_wavebit_run(point)
        assert r.R.sum() + r.T.sum() == pytest.approx(1.0, abs=1e-9)


def _oblique_spec():
    """A hex cell with an off-centre feature — asymmetric, so a lost basis shows.

    ``period`` is stated *as well*, and equal to the two edge lengths ``hexagonal(400)``
    gives, so that the basis contributes the angle and nothing else. That is what makes
    the pair of tests below sharp: drop ``lattice`` anywhere along the path and the run
    degrades to exactly the square the second of them compares against, rather than to some
    other scale -- which would part the two numbers for a reason unrelated to the skew.
    """
    basis = np.asarray(wavebit.Lattice.hexagonal(400.0))
    eps = np.ones((1, 16, 16))
    eps[0, 3:8, 2:7] = 12.25
    return basis, JobSpec(
        stack=_legacy_wavebit_Stack.from_structure(eps, [300.0], top=1.0, bot=1.0,
                                          period=(400.0, 400.0),
                                          lattice=basis.tolist()),
        incidence=_legacy_wavebit_Incidence(wavelength=900.0, theta=0.1, phi=0.3, psi=0.2),
        numerics=_legacy_wavebit_Numerics(backend='numpy', truncation_order=[4, 4],
                                  factorization=Factorization.NVF))


def test_an_oblique_cell_survives_the_round_trip():
    """``JobSpec`` calls itself a *complete* description, and could not describe one.

    ``Stack`` carried ``period`` alone until 2026-08-23 (``SPEC_VERSION`` 5 -> 6), and on a
    skewed cell ``period`` is ``(|a1|, |a2|)`` -- the two edge lengths, with the angle already
    gone, so it cannot reconstruct the basis. Every oblique run was therefore unserialisable,
    quietly: the document came back describing a *rectangle* of the same edge lengths.
    """
    basis, spec = _oblique_spec()
    back = JobSpec.from_json(spec.to_json())
    assert np.allclose(np.asarray(back.stack.lattice), basis)

    a = _legacy_wavebit_run(spec)
    b = _legacy_wavebit_run(back)
    assert np.asarray(a.R).tobytes() == np.asarray(b.R).tobytes()

    # And the field, not only the efficiencies. `R` is a sum over orders, so it survives a
    # per-order wavevector that is wrong by a whole reciprocal step -- the field is where such a
    # thing is a different *shape* rather than a different total. The Bloch-condition test in
    # `tests/rcwa/test_oblique_lattice.py` is what pins that; this is the same array being compared
    # for a different reason, and it is simply the larger one: 5 axes against 1.
    fields = []
    for one in (spec, back):
        solver = _legacy_wavebit_rcwa(**one.stack.as_kwargs(), **one.incidence.as_kwargs(),
                              **one.numerics.as_kwargs())
        solver.answers = 'field'          # `numerics.as_kwargs()` already carries one
        solver.solve()
        fields.append(np.asarray(solver.reconstruct_field(sample_x=7, sample_y=5, sample_z=3)))
    assert fields[0].shape == fields[1].shape
    assert float(np.max(np.abs(fields[0]))) > 1e-6          # a zero field would pass anything
    assert fields[0].tobytes() == fields[1].tobytes()


def test_the_round_trip_is_not_merely_carrying_the_edge_lengths():
    """The check with teeth: the same edge lengths as a *rectangle* give a different answer.

    Without it, a spec that dropped `lattice` and kept `period=(400, 400)` would still
    round-trip, still solve, and still conserve energy — just a different structure.
    """
    basis, spec = _oblique_spec()
    oblique = float(np.sum(_legacy_wavebit_run(spec).R))

    square = JobSpec(stack=_legacy_wavebit_Stack.from_structure(
        spec.stack.structure, spec.stack.thickness, top=1.0, bot=1.0, period=(400.0, 400.0)),
        incidence=spec.incidence, numerics=spec.numerics)
    assert square.stack.lattice is None
    assert not np.isclose(oblique, float(np.sum(_legacy_wavebit_run(square).R)))


def test_a_spec_without_a_lattice_is_unchanged():
    """The axis-aligned document does not gain a key it has no use for."""
    spec = JobSpec.from_json(json.dumps(BINARY_GRATING))
    assert spec.stack.lattice is None
    assert 'lattice' not in json.loads(spec.to_json())['stack']
    assert 'lattice' not in spec.stack.as_kwargs()


# ---------------------------------------------------------------- enum-typed numerics fields

def _numerics_enum_fields():
    """``{name: (enum class, None allowed)}`` for every enum-annotated field of ``Numerics``.

    Read off the annotations here rather than from a list, so a field added with an enum type
    is parametrized below without anyone adding it -- which is the point: the defect these
    tests pin was one field at a time. `test_the_enum_field_list_is_not_vacuous` keeps the
    derivation honest.
    """
    import dataclasses
    import types
    import typing
    from enum import Enum

    hints = typing.get_type_hints(_legacy_wavebit_Numerics)
    out = {}
    for f in dataclasses.fields(_legacy_wavebit_Numerics):
        hint = hints[f.name]
        union = typing.get_origin(hint) in (typing.Union, types.UnionType)
        args = typing.get_args(hint) if union else (hint,)
        enums = [a for a in args if isinstance(a, type) and issubclass(a, Enum)]
        if enums:
            out[f.name] = (enums[0], type(None) in args)
    return out


ENUM_FIELDS = _numerics_enum_fields()
ENUM_MEMBERS = [(name, member) for name, (cls, _) in ENUM_FIELDS.items() for member in cls]


def _spellings(member):
    """Every way the JSON door accepts ``member``: itself and its name -- in either case, except
    ``backend``, whose name is taken exactly as written, as every door that takes one does."""
    out = [member, member.name.lower()]
    if not isinstance(member, Backend):
        out.append(member.name.upper())
    return out


def test_the_enum_field_list_is_not_vacuous():
    """A derivation that found nothing would pass every parametrized test below by running none."""
    assert {'backend', 'fourier_coefficients', 'factorization', 'raster_normal_method',
            'nvf_projector', 'li_nesting', 'answers'} <= set(ENUM_FIELDS)


@pytest.mark.parametrize('name, member', ENUM_MEMBERS,
                         ids=[f'{n}={m.name.lower()}' for n, m in ENUM_MEMBERS])
def test_numerics_built_directly_is_the_numerics_from_dict_builds(name, member):
    """``Numerics(li_nesting='symmetric')`` and the parsed document have to be one object.

    Until 2026-09-27 only ``from_dict`` coerced, one field at a time, so direct construction
    kept the string: ``as_kwargs`` then raised ``AttributeError: 'str' object has no attribute
    'value'`` on six of the seven fields (``int('jax')`` on ``backend``), and the two that did
    not raise wrote ``'FIELD'`` / ``1`` into the document where ``from_dict`` wrote ``'field'``
    / ``'jax'`` -- one run, two cache keys.
    """
    reference = _legacy_wavebit_Numerics.from_dict({name: member.name.lower()})
    for spelling in _spellings(member):
        built = _legacy_wavebit_Numerics(**{name: spelling})
        assert getattr(built, name) is member, spelling
        assert built == reference == _legacy_wavebit_Numerics.from_dict({name: spelling}), spelling

        # the solver keyword: the plain value, never the member, never dropped
        kwargs = built.as_kwargs()
        assert kwargs == reference.as_kwargs(), spelling
        assert kwargs[name] == member.value and type(kwargs[name]) is type(member.value), (
            spelling, kwargs.get(name))

        # the document: a plain string, the canonical one, and it reads back as itself
        written = built.to_dict()[name]
        assert type(written) is str and written == member.name.lower(), (spelling, written)
        assert _legacy_wavebit_Numerics.from_dict(json.loads(json.dumps(built.to_dict()))) == built

        # and the whole spec: one run, one key, whichever spelling built it
        spec = JobSpec(numerics=built)
        assert spec.key() == JobSpec(numerics=reference).key(), spelling
        assert JobSpec.from_json(spec.to_json()).key() == spec.key(), spelling


@pytest.mark.parametrize('name', sorted(ENUM_FIELDS))
def test_numerics_refuses_a_name_that_is_no_member_on_both_doors(name):
    """Refused at construction, with the message the JSON door gives -- not stored as typed."""
    with pytest.raises(SpecError) as direct:
        _legacy_wavebit_Numerics(**{name: 'bogus'})
    with pytest.raises(SpecError) as parsed:
        _legacy_wavebit_Numerics.from_dict({name: 'bogus'})
    assert str(direct.value) == str(parsed.value)
    assert str(direct.value).startswith(f'numerics.{name}: ')


@pytest.mark.parametrize('name', sorted(ENUM_FIELDS))
def test_numerics_reads_none_as_the_json_door_does(name):
    """``None`` is "unset" where the annotation is ``Optional`` and a refusal where it is not."""
    _, optional = ENUM_FIELDS[name]
    if optional:
        for numerics in (_legacy_wavebit_Numerics(**{name: None}), _legacy_wavebit_Numerics.from_dict({name: None})):
            assert getattr(numerics, name) is None
            assert name not in numerics.as_kwargs()
        return
    with pytest.raises(SpecError) as direct:
        _legacy_wavebit_Numerics(**{name: None})
    with pytest.raises(SpecError) as parsed:
        _legacy_wavebit_Numerics.from_dict({name: None})
    assert str(direct.value) == str(parsed.value)


def _other_member(name, member):
    """A member of ``name``'s enum that is not ``member``, so an assignment has to move it."""
    return next(m for m in ENUM_FIELDS[name][0] if m is not member)


@pytest.mark.parametrize('name, member', ENUM_MEMBERS,
                         ids=[f'{n}={m.name.lower()}' for n, m in ENUM_MEMBERS])
def test_numerics_assigned_after_construction_is_the_numerics_construction_builds(name, member):
    """``n = Numerics(); n.li_nesting = 'symmetric'`` is ``Numerics(li_nesting='symmetric')``.

    Until 2026-09-27 construction normalised in ``__post_init__`` and assignment did not, so the
    door left over stored the string: ``as_kwargs`` raised on it as it had on construction
    before, and the integer backend code ``1`` (refused since 2026-10-11) assigned to
    ``n.backend`` wrote ``1`` into the document where construction wrote ``'jax'`` -- one run,
    two cache keys. Each assignment starts from another member, so
    one that stored nothing fails here rather than matching a default it left behind.
    """
    built = _legacy_wavebit_Numerics(**{name: member.name.lower()})
    for spelling in _spellings(member):
        assigned = _legacy_wavebit_Numerics(**{name: _other_member(name, member)})
        setattr(assigned, name, spelling)
        assert getattr(assigned, name) is member, spelling
        assert assigned == built, spelling
        assert assigned.as_kwargs() == built.as_kwargs(), spelling
        assert assigned.to_dict() == built.to_dict(), spelling
        assert assigned.to_dict()[name] == member.name.lower(), spelling
        assert JobSpec(numerics=assigned).key() == JobSpec(numerics=built).key(), spelling


@pytest.mark.parametrize('name', sorted(ENUM_FIELDS))
def test_numerics_assignment_refuses_what_from_dict_refuses_and_keeps_the_field(name):
    """A name that is no member, a code that is none, and ``None`` where the field is not
    ``Optional``: refused on assignment with the message the JSON door gives, and refused
    before anything is stored, so the field keeps the member it had."""
    enum_cls, optional = ENUM_FIELDS[name]
    held = list(enum_cls)[-1]
    for value in ('bogus', 99) + (() if optional else (None,)):
        numerics = _legacy_wavebit_Numerics(**{name: held})
        with pytest.raises(SpecError) as assigned:
            setattr(numerics, name, value)
        with pytest.raises(SpecError) as parsed:
            _legacy_wavebit_Numerics.from_dict({name: value})
        assert str(assigned.value) == str(parsed.value), value
        assert getattr(numerics, name) is held, value
    if optional:                        # and "unset" is assignable where it is constructible
        numerics = _legacy_wavebit_Numerics(**{name: held})
        setattr(numerics, name, None)
        assert getattr(numerics, name) is None and name not in numerics.as_kwargs()
        assert numerics == _legacy_wavebit_Numerics.from_dict({name: None})


#: What each field needs beside it for a solver to accept it. A field missing here is built
#: with nothing beside it; if its value is then refused, add the row.
_SOLVER_COMPANIONS = {
    'raster_normal_method': {'factorization': 'nvf'},
    'nvf_projector': {'factorization': 'nvf'},
    'li_nesting': {'factorization': 'li'},
}
#: The pytree exists on the JAX backend only, so `backend` crosses it as 'jax' and nothing else.
_PYTREE_MEMBERS = [(n, m) for n, m in ENUM_MEMBERS if n != 'backend' or m is Backend.JAX]


@pytest.mark.parametrize('name, member', _PYTREE_MEMBERS,
                         ids=[f'{n}={m.name.lower()}' for n, m in _PYTREE_MEMBERS])
def test_a_directly_built_numerics_reaches_the_solver_and_crosses_its_pytree(name, member):
    """The string spelling, through ``run``'s own assembly, into a jitted solve's rebuild.

    `tests/rcwa/test_pytree_round_trip.py` holds the solver-side round trip for one value per field;
    this is the spec side of the same seam, for every member.
    """
    pytest.importorskip('jax')
    from jax import tree_util

    eps = np.ones((1, 8, 8))
    eps[0, 2:5, 1:6] = 6.0
    spec = JobSpec(
        stack=_legacy_wavebit_Stack.from_structure(eps, [120.0], top=1.0, bot=2.25,
                                           period=(400.0, 400.0)),
        incidence=_legacy_wavebit_Incidence(wavelength=900.0, theta=0.2, phi=0.1, psi=0.3),
        numerics=_legacy_wavebit_Numerics(truncation_order=(2, 2),
                                  **{'backend': 'jax', **_SOLVER_COMPANIONS.get(name, {}),
                                     name: member.name.lower()}))
    solver = _legacy_wavebit_rcwa(**spec.numerics.as_kwargs(), **spec.stack.as_kwargs(),
                          **spec.incidence.as_kwargs())
    assert getattr(solver, name) == member

    leaves, treedef = tree_util.tree_flatten(solver)
    rebuilt = tree_util.tree_unflatten(treedef, leaves)
    assert getattr(rebuilt, name) == member


# ---------------------------------------------------------------- the attribute door

def _spec_dataclasses():
    """Every dataclass the spec module builds a run from, read off its namespace, by name.

    Read rather than listed, so a dataclass the module gains is held to the door the day it
    lands; `test_the_spec_dataclass_list_is_not_vacuous` keeps the derivation honest.
    """
    from wavebit._common import _jobs_v1 as spec_module
    return {v.__name__: v for v in vars(spec_module).values()
            if isinstance(v, type) and dataclasses.is_dataclass(v)}


SPEC_DATACLASSES = _spec_dataclasses()
#: The mutable ones -- a frozen dataclass refuses every assignment already.
MUTABLE = sorted(name for name, cls in SPEC_DATACLASSES.items()
                 if not cls.__dataclass_params__.frozen)
MUTABLE_FIELDS = [(name, f.name) for name in MUTABLE
                  for f in dataclasses.fields(SPEC_DATACLASSES[name])]
_FIELD_IDS = [f'{c}.{f}' for c, f in MUTABLE_FIELDS]
#: What a mutable class needs to be built. A class missing here is built bare; if that raises,
#: add the row.
_BUILD = {'Layer': lambda: _legacy_wavebit_Layer(thickness=300.0, structure=np.ones((1, 4)))}


def _build(name):
    return _BUILD.get(name, SPEC_DATACLASSES[name])()


def _declared(name):
    return sorted(f.name for f in dataclasses.fields(SPEC_DATACLASSES[name]))


def _typo(field_name):
    """One letter off: dropped from a long name, doubled on a short one (``kx`` -> ``kxx``)."""
    return field_name[:-1] if len(field_name) > 4 else field_name + field_name[-1]


def test_the_spec_dataclass_list_is_not_vacuous():
    """And every dataclass in it is either frozen or behind the door -- none is merely mutable."""
    from wavebit._common.declarations import _DeclaredFieldsOnly
    assert {'Incidence', 'JobSpec', 'Layer', 'Numerics', 'Stack', 'Sweep'} <= set(MUTABLE)
    assert {'Material', 'TabulatedMaterial'} <= set(SPEC_DATACLASSES) - set(MUTABLE)
    for name in MUTABLE:
        assert issubclass(SPEC_DATACLASSES[name], _DeclaredFieldsOnly), name


def test_a_typo_on_the_incidence_is_refused_and_the_angle_is_not_left_behind():
    """The finding: ``inc.thet = 0.2`` was stored, and ``as_kwargs`` still wrote ``theta=0``.

    The solve then answered for normal incidence with nothing to say the angle had not moved --
    the defect the solver's own door (`CoreFactorizationMixin.__setattr__`) closed on
    2026-08-27, left open on the objects a spec is built from. The same typo through a spec's
    own ``incidence`` and through the document door is refused and pointed at ``theta`` alike.
    """
    inc = _legacy_wavebit_Incidence(wavelength=900.0, theta=0.0)
    with pytest.raises(AttributeError, match=r"Incidence has no attribute 'thet' to set\. "
                                             r"Did you mean 'theta'\?"):
        inc.thet = 0.2
    assert 'thet' not in vars(inc) and not hasattr(inc, 'thet')
    assert inc.as_kwargs()['theta'] == 0.0

    spec = JobSpec.from_dict(BINARY_GRATING)
    with pytest.raises(AttributeError, match=r"Did you mean 'theta'\?"):
        spec.incidence.thet = 0.2
    with pytest.raises(SpecError, match=r"\['thet'\]\. Did you mean 'theta'\?"):
        JobSpec.from_dict({**BINARY_GRATING,
                           'incidence': {**BINARY_GRATING['incidence'], 'thet': 0.2}})


@pytest.mark.parametrize('cls_name, field_name', MUTABLE_FIELDS, ids=_FIELD_IDS)
def test_a_typo_of_every_field_is_refused_with_the_field_it_meant(cls_name, field_name):
    """One letter off each declared field, on each class: refused, pointed home, not stored."""
    obj, typo = _build(cls_name), _typo(field_name)
    assert not hasattr(type(obj), typo), typo            # else it is a class member, not a typo
    before = {k: id(v) for k, v in vars(obj).items()}
    with pytest.raises(AttributeError) as refusal:
        setattr(obj, typo, 0.2)
    message = str(refusal.value)
    assert message.startswith(
        f'{cls_name} has no attribute {typo!r} to set. Did you mean {field_name!r}?'), message
    assert message.split('Settable fields: ')[1].split('.\n')[0].split(', ') == _declared(cls_name)
    assert {k: id(v) for k, v in vars(obj).items()} == before


@pytest.mark.parametrize('cls_name, field_name', MUTABLE_FIELDS, ids=_FIELD_IDS)
def test_every_declared_field_still_assigns(cls_name, field_name):
    """Closing the door must leave every field through it, onto the instance, as given.

    An enum-typed ``Numerics`` field converts what it is given (the assignment tests above), so
    it is given a member, which it keeps as is; any other field takes any object at all.
    """
    enum_field = ENUM_FIELDS.get(field_name) if cls_name == 'Numerics' else None
    obj = _build(cls_name)
    value = list(enum_field[0])[-1] if enum_field else object()
    setattr(obj, field_name, value)
    assert vars(obj)[field_name] is value and getattr(obj, field_name) is value


def test_the_retired_nesting_is_pointed_at_li_nesting_on_both_numerics_doors():
    """``nesting`` is no alias (`tests/rcwa/test_li_nesting.py`); on ``Numerics`` it was stored, and
    the spec then asked the solver for the fixed nesting the caller had meant to leave."""
    numerics = _legacy_wavebit_Numerics(li_nesting='fixed')
    with pytest.raises(AttributeError, match=r"Numerics has no attribute 'nesting' to set\. "
                                             r"Did you mean 'li_nesting'\?"):
        numerics.nesting = 'symmetric'
    assert 'nesting' not in vars(numerics)
    assert numerics.as_kwargs()['li_nesting'] == 'fixed'
    with pytest.raises(SpecError, match=r"\['nesting'\]\. Did you mean 'li_nesting'\?"):
        _legacy_wavebit_Numerics.from_dict({'nesting': 'symmetric'})


#: Where the document door parses each class's keys: the mapping that holds them.
_DOCUMENT_AT = {
    'JobSpec': lambda keys: {'version': '10', **keys},
    'Numerics': lambda keys: {'version': '10', 'numerics': keys},
    'Stack': lambda keys: {'version': '10', 'stack': keys},
    'Layer': lambda keys: {'version': '10', 'stack': {'layers': [keys]}},
    'Incidence': lambda keys: {'version': '10', 'incidence': keys},
}
_DOCUMENT_FIELDS = [(c, f) for c, f in MUTABLE_FIELDS if c in _DOCUMENT_AT]


def test_every_class_with_a_document_door_is_parametrized():
    """`Sweep` is the one without: its document keys are axis names, and its field is ``axes``."""
    assert set(MUTABLE) - set(_DOCUMENT_AT) == {'Sweep'}


@pytest.mark.parametrize('cls_name, field_name', _DOCUMENT_FIELDS,
                         ids=[f'{c}.{f}' for c, f in _DOCUMENT_FIELDS])
def test_the_document_door_refuses_the_same_typo_with_the_same_hint(cls_name, field_name):
    """A key misspelled in a document and the same name assigned on the object it parses into
    name the same field. The document door refused unknown keys already, listing what is
    allowed, but only a renamed NVF option carried a hint; the near match is the attribute
    door's now."""
    typo = _typo(field_name)
    with pytest.raises(SpecError) as parsed:
        JobSpec.from_dict(_DOCUMENT_AT[cls_name]({typo: 0.2}))
    with pytest.raises(AttributeError) as assigned:
        setattr(_build(cls_name), typo, 0.2)
    hint = f' Did you mean {field_name!r}?'
    assert f'unknown field(s) {[typo]}.{hint} Allowed: ' in str(parsed.value), parsed.value
    assert hint in str(assigned.value)
    # one vocabulary on both doors: what the document allows is what the class declares
    assert str(parsed.value).split('Allowed: ')[1] == str(_declared(cls_name))


def test_several_unknown_keys_are_each_pointed_home():
    with pytest.raises(SpecError, match=r"Did you mean 'phi' for 'ph', 'theta' for 'thet'\?"):
        JobSpec.from_dict({**BINARY_GRATING, 'incidence': {'thet': 0.2, 'ph': 0.1}})


def test_the_retired_nvf_spellings_are_renamed_only_where_the_new_name_lives():
    """``numerics.normal_method`` gets the rename; ``stack.projector`` is not sent to a slot on
    another class. The document door offered the rename under ``numerics`` only, by checking
    where it was; the shared hint asks the question that matters -- can the owner take it."""
    rename = "Renamed NVF option(s): 'normal_method' -> 'raster_normal_method'."
    for door in (lambda: setattr(_legacy_wavebit_Numerics(), 'normal_method', 'contour'),
                 lambda: _legacy_wavebit_Numerics.from_dict({'normal_method': 'contour'})):
        with pytest.raises((AttributeError, SpecError), match=re.escape(rename)):
            door()
    for door in (lambda: setattr(_legacy_wavebit_Stack(), 'projector', 'linear'),
                 lambda: JobSpec.from_dict({'version': '10', 'stack': {'projector': 'linear'}})):
        with pytest.raises((AttributeError, SpecError)) as refusal:
            door()
        assert 'Renamed' not in str(refusal.value), refusal.value
        assert 'nvf_projector' not in str(refusal.value), refusal.value


def test_what_the_door_lets_through_besides_the_fields():
    """The solver's rule, so the three doors predict alike: a private name is stored, a method
    can be stubbed on an instance, and a read-only property answers with Python's own message
    rather than "has no attribute ... did you mean" -- ``Stack.thickness`` exists; it is derived.
    """
    readonly = []
    for name in MUTABLE:
        obj = _build(name)
        obj._note = 'mine'
        assert vars(obj)['_note'] == 'mine'
        for attr, member in vars(type(obj)).items():
            if isinstance(member, property) and member.fset is None:
                readonly.append(f'{name}.{attr}')
                with pytest.raises(AttributeError, match='has no setter'):
                    setattr(obj, attr, None)
    assert {'Stack.thickness', 'Stack.structure', 'Stack.mu', 'Layer.is_raster'} <= set(readonly)

    spec = JobSpec(numerics=_legacy_wavebit_Numerics())
    spec.numerics.check = lambda: ('stubbed',)
    assert spec.check() == ('[numerics] stubbed',)
    assert dataclasses.replace(_legacy_wavebit_Incidence(), theta=0.3).theta == 0.3


def test_a_copied_spec_keeps_the_door_and_its_key():
    """Pickle and deepcopy restore ``__dict__`` without coming through the door; the copy is the
    same run, and it refuses a typo as the original does -- the door is the class's, not state."""
    import copy
    import pickle
    spec = JobSpec.from_dict({**BINARY_GRATING, 'sweep': {'wavelength': [800.0, 900.0]}})
    spec._note = 'mine'
    for twin in (pickle.loads(pickle.dumps(spec)), copy.deepcopy(spec)):
        assert twin.key() == spec.key() and twin._note == 'mine'
        assert [p.key() for p in twin.expand()] == [p.key() for p in spec.expand()]
        with pytest.raises(AttributeError, match=r"Did you mean 'truncation_order'\?"):
            twin.numerics.truncation_ordr = (3, 0)


# ---------------------------------------------------------------- never guessed across quantities

#: Spelled like a refractive index. ``n_top`` / ``n_bot`` are the half-spaces' spellings from
#: before the material slots took permittivity; the rest are the same quantity assembled other
#: ways -- camel case, glued, numbered, suffixed, abbreviated ``ri``, named ``nr`` / ``nd``,
#: beside another name for a half-space or a background, or a part of an index -- which is
#: what makes the refusal a rule.
INDEX_SPELLINGS = ('n_top', 'n_bot', 'nTop', 'ntop', 'N_bot', 'n1', 'n_1', 'n_structure',
                   'nk_top', 'top_index', 'refractive_index',
                   'RI_top', 'ri_top', 'top_ri', 'riTop', 'ritop', 'nr_top', 'nd_top', 'top_n',
                   'n_sub', 'nsub', 'n_substrate', 'n_bg', 'n_material', 'n_eff',
                   'n_struct', 'nstruct', 'struct_n')
#: An ``n`` that is not an index: a count (``n_orders``, ``nG``, ``nr_steps``), a dimension
#: (``nd_array``), a background glued short (``bgnd``). Each gets the guess plain difflib gives,
#: which is what a name spelled like no quantity gets. The last five are counts of something a
#: class does declare, so every class has a count whose plain guess is a real field.
COUNT_SPELLINGS = ('n_orders', 'n_threads', 'n_grid', 'n_harmonics', 'nG', 'n_theta',
                   'n_wavelength', 'nr_steps', 'nd_array', 'bgnd',
                   'n_layers', 'n_thickness', 'n_order', 'n_axes', 'n_sweep')
_INDEX_NOTE = 'relative permittivity (eps = n**2) or a Material'


def _assigned_refusal(cls_name, name):
    with pytest.raises(AttributeError) as refusal:
        setattr(_build(cls_name), name, 1.5)
    return str(refusal.value)


def _parsed_refusal(cls_name, name):
    with pytest.raises(SpecError) as refusal:
        JobSpec.from_dict(_DOCUMENT_AT[cls_name]({name: 1.5}))
    return str(refusal.value)


def _both_doors(cls_name, name):
    return [_assigned_refusal(cls_name, name)] + (
        [_parsed_refusal(cls_name, name)] if cls_name in _DOCUMENT_AT else [])


def test_n_top_and_n_bot_are_told_the_slots_are_permittivity_not_pointed_at_them():
    """The finding: ``Stack().n_top = 1.5`` said "Did you mean 'top'?", and so did the document.

    Following it hands a refractive index to a slot that reads relative permittivity, and 1.5
    is a fine permittivity: the solve conserves energy and says nothing. The plain nearest match
    does cross -- asserted first, so this test fails if the rule stops overruling it, rather
    than passing because difflib happened not to reach.
    """
    import difflib
    for name, slot in (('n_top', 'top'), ('n_bot', 'bot')):
        assert difflib.get_close_matches(name, _declared('Stack'), n=1, cutoff=0.6) == [slot]
        stack = _legacy_wavebit_Stack(top=1.0, bot=2.25)
        with pytest.raises(AttributeError) as assigned:
            setattr(stack, name, 1.5)
        with pytest.raises(SpecError) as parsed:
            JobSpec.from_dict({'version': '10', 'stack': {name: 1.5}})
        for message in (str(assigned.value), str(parsed.value)):
            assert 'Did you mean' not in message, message
            assert f"{name!r} is spelled like a refractive index" in message, message
            assert _INDEX_NOTE in message and 'eps_from_refractive_index' in message, message
        assert (stack.top, stack.bot) == (1.0, 2.25) and name not in vars(stack)


@pytest.mark.parametrize('cls_name', MUTABLE)
def test_no_index_spelling_is_guessed_on_any_spec_door(cls_name):
    """Every index spelling on every class, both doors: no guess, and the note in its place."""
    for name in INDEX_SPELLINGS:
        for message in _both_doors(cls_name, name):
            assert 'Did you mean' not in message, message
            assert f"{name!r} is spelled like a refractive index" in message, message
            assert _INDEX_NOTE in message, message


@pytest.mark.parametrize('cls_name', MUTABLE)
def test_a_count_spelled_with_n_is_guessed_as_plain_difflib_would(cls_name):
    """The other direction: an ``n`` that means "number of" is no index, so it neither gets
    the note nor loses its guess. Read as an index, ``n_theta`` lost "Did you mean 'theta'?"
    on ``Incidence`` and was told the material slots take permittivity, which is about nothing
    it asked; ``n_orders`` was told the same. The guess asserted is difflib's own, and each
    class has at least one count it reaches, so this cannot pass by every guess being none."""
    import difflib
    known = _declared(cls_name)
    reached = 0
    for name in COUNT_SPELLINGS:
        plain = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
        reached += bool(plain)
        for message in _both_doors(cls_name, name):
            assert 'spelled like' not in message, message
            if plain:
                assert f'Did you mean {plain[0]!r}?' in message, message
            else:
                assert 'Did you mean' not in message, message
    assert reached, f'no count in COUNT_SPELLINGS is near a field of {cls_name}'


def test_a_permeability_spelling_is_not_pointed_at_a_permittivity_slot():
    """``mu_top`` is as near ``top`` as ``n_top`` is, and crosses the other way."""
    import difflib
    assert difflib.get_close_matches('mu_top', _declared('Stack'), n=1, cutoff=0.6) == ['top']
    assert difflib.get_close_matches('mu_structure', _declared('Layer'), n=1,
                                     cutoff=0.6) == ['structure']
    for name in ('mu_top', 'mu_bot', 'mutop', 'mu_structure'):
        for message in _both_doors('Stack', name):       # no permeability here: told where
            assert 'Did you mean' not in message, message
            assert f"{name!r} is spelled like a permeability" in message, message
            assert 'Material(epsilon, mu)' in message, message
        for message in _both_doors('Layer', name):       # `mu` is here, and is not near enough
            assert 'Did you mean' not in message, message
            assert 'spelled like' not in message, message


@pytest.mark.parametrize('cls_name, name, meant', [
    ('Stack', 'eps_top', 'top'),          # a permittivity spelling, to a permittivity slot
    ('Stack', 'tpo', 'top'),              # spelled like nothing: matched against everything
    ('Stack', 'bott', 'bot'),
    ('Layer', 'mu_', 'mu'),               # a permeability spelling, to the permeability field
    ('Layer', 'structur', 'structure'),
    ('Incidence', 'thet', 'theta'),
])
def test_a_guess_within_one_quantity_is_still_offered(cls_name, name, meant):
    """The rule withholds a guess across quantities, not every guess near a material slot."""
    for message in _both_doors(cls_name, name):
        assert f'Did you mean {meant!r}?' in message, message
        assert 'spelled like' not in message, message


def test_where_an_index_is_taken_the_note_does_not_say_otherwise():
    """A dispersion table takes ``index``: an index spelling is matched against it, and no note
    claims that nothing here takes one. A material object takes ``epsilon`` and ``mu`` only."""
    def table(**keys):
        return {'version': '10',
                'stack': {'top': {'dispersion': {'wavelengths': [800.0, 900.0], **keys}}}}

    with pytest.raises(SpecError) as near:
        JobSpec.from_dict(table(n_index=[1.5, 1.5]))
    assert "Did you mean 'index'?" in str(near.value), near.value
    assert 'spelled like' not in str(near.value), near.value
    with pytest.raises(SpecError) as far:
        JobSpec.from_dict(table(n=[1.5, 1.5]))
    assert 'Did you mean' not in str(far.value) and 'spelled like' not in str(far.value)
    with pytest.raises(SpecError) as material:
        JobSpec.from_dict({'version': '10', 'stack': {'top': {'n': 1.5}}})
    assert "'n' is spelled like a refractive index" in str(material.value), material.value
    assert "Allowed: ['epsilon', 'mu']" in str(material.value), material.value


def test_several_index_spellings_share_one_note_beside_the_guesses():
    with pytest.raises(SpecError) as refusal:
        JobSpec.from_dict({'version': '10', 'stack': {'n_top': 1.5, 'n_bot': 1.0, 'perio': [1.0]}})
    message = str(refusal.value)
    assert "Did you mean 'period' for 'perio'?" in message, message
    assert "'n_bot', 'n_top' are spelled like a refractive index" in message, message
    assert message.count(_INDEX_NOTE) == 1 and "'top' for" not in message, message


@pytest.mark.parametrize('name, quantities', [
    ('n_top', {'index'}), ('nTop', {'index'}), ('ntop', {'index'}), ('NTOP', {'index'}),
    ('n1', {'index'}), ('top_index', {'index'}), ('refractiveIndex', {'index'}),
    ('mu_top', {'permeability'}), ('mubot', {'permeability'}), ('permeability', {'permeability'}),
    ('eps_top', {'permittivity'}), ('epsilon', {'permittivity'}), ('top', {'permittivity'}),
    ('background', {'permittivity'}), ('eps_from_refractive_index', {'permittivity', 'index'}),
    # an `n` that is not a word of its own, and not glued to a slot, is no index
    ('nesting', set()), ('normal_method', set()), ('nvf_projector', set()), ('thet', set()),
    # `ri` is an index wherever it stands; so is `n` / `ri` glued to any name a slot goes by
    ('RI_top', {'index'}), ('riTop', {'index'}), ('ritop', {'index'}), ('ri', {'index'}),
    ('nsub', {'index'}), ('bgn', {'index'}), ('ribg', {'index'}),
    # `n`, `nr`, `nd`: an index alone, numbered, or beside a slot word or a part of an index
    ('n', {'index'}), ('N', {'index'}), ('nr', {'index'}), ('nd', {'index'}), ('n_1', {'index'}),
    ('top_n', {'index'}), ('n_sub', {'index'}), ('N_bot', {'index'}), ('n_bg', {'index'}),
    ('n_material', {'index'}), ('n_cladding', {'index'}), ('n_air', {'index'}),
    ('n_top2', {'index'}), ('nr_top', {'index'}), ('nd_top', {'index'}), ('n_eff', {'index'}),
    ('n_real', {'index'}), ('n_im', {'index'}),
    # nor is one that counts or sizes something, nor `bg` + `nd`; and a bare `k` is never read
    ('n_orders', set()), ('n_threads', set()), ('n_grid', set()), ('n_harmonics', set()),
    ('nG', set()), ('n_theta', set()), ('n_wavelength', set()), ('nr_steps', set()),
    ('nd_array', set()), ('bgnd', set()), ('n_layer', set()), ('n_cores', set()),
    ('k', set()), ('k_top', set()),
])
def test_what_a_name_is_spelled_as(name, quantities):
    from wavebit._common.names import name_quantities
    assert name_quantities(name) == quantities


def test_no_settable_name_is_read_as_a_quantity_it_is_not():
    """The rule reads a quantity off the spelling, so a field spelled like one would have its
    typos matched against the wrong names. Every name a door can set, on every door, is read
    as what it holds: the material slots as permittivity, ``mu`` as permeability, the rest as
    nothing. A field added as ``n_layers`` would fail here before it lost its guesses."""
    from wavebit._common.names import name_quantities
    solver = _legacy_wavebit_rcwa(structure=np.ones((1, 1, 4)), thickness=[100.0], wavelength=900.0,
                          period=[1.0, 1.0])
    names = {f for _, f in MUTABLE_FIELDS} | set(solver._condition_names())
    expected = {'top': {'permittivity'}, 'bot': {'permittivity'},
                'structure': {'permittivity'}, 'mu': {'permeability'}}
    for name in sorted(names):
        assert name_quantities(name) == expected.get(name, set()), name


def test_the_material_slots_are_the_manifests():
    """``structure`` beside the slots the manifest names for a material, and nothing else."""
    from wavebit._common.names import _MATERIAL_SLOTS
    from wavebit.manifest import manifest
    named = manifest()['material_dispersion']['automatic_slots']
    assert _MATERIAL_SLOTS == {slot.split('.')[-1] for slot in named} | {'structure'}


# ---------------------------------------------------------------- a key that is not a string

_GRATING_STACK = BINARY_GRATING['stack']


@pytest.mark.parametrize('build', [
    lambda: _legacy_wavebit_Numerics.from_dict({1: 2}),
    lambda: _legacy_wavebit_Numerics.from_dict({1: 2, 'backend': 'numpy', 'x': 3}),
    lambda: Sweep.from_dict({0: [1.0]}),
    lambda: Sweep.from_dict({'wavelength': {0: [1.0]}}),
    lambda: JobSpec.from_dict({**BINARY_GRATING, 'incidence': {None: 1}}),
    lambda: JobSpec.from_dict({**BINARY_GRATING, (1, 2): 3}),
    lambda: JobSpec.from_dict({**BINARY_GRATING, 'stack': {**_GRATING_STACK, 1.5: 2}}),
    lambda: JobSpec.from_dict({**BINARY_GRATING,
                               'stack': {**_GRATING_STACK, 'layers': [{0: 1}]}}),
    lambda: JobSpec.from_dict({**BINARY_GRATING,
                               'stack': {**_GRATING_STACK, 'top': {'epsilon': 2.0, 1: 1}}}),
], ids=['numerics', 'numerics-mixed', 'sweep', 'sweep-axis', 'incidence', 'spec', 'stack',
        'layer', 'material'])
def test_a_key_that_is_not_a_string_is_refused_as_a_spec_error(build):
    """A parsed mapping with a key that is not a string is no document this schema reads, and
    ``from_dict``'s contract is ``SpecError``. It was a bare ``TypeError`` from inside the hint:
    difflib cannot match an int, and ``sorted`` cannot order one beside a string."""
    with pytest.raises(SpecError, match=r': field names are strings, got .* Allowed: \['):
        build()
