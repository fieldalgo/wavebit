"""JobSpec is the serialisable form of a run, and ``wavebit.run`` executes it statelessly.

Two properties carry the whole design and are gated here:

* **Parity** — a spec must produce byte-identical results to the equivalent
  :func:`wavebit.rcwa` call. The spec layer is a new way to *say* a run, never a new way
  to *compute* one.
* **Round-trip** — a spec must survive JSON unchanged, because it exists to cross process
  and network boundaries. Equal specs must hash equal, and differing ones must not.

Validation is tested for rejection, not just acceptance: a bad spec caught at submission is
far cheaper than one caught on a worker.
"""
import json

import numpy as np
import pytest

import wavebit
from wavebit import JobSpec, Sweep, SpecError, Backend, Factorization, Material


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
    result = wavebit.run(BINARY_GRATING)

    reference = wavebit.rcwa(
        backend=0, wavelength=900., period=[700.], truncation_order=[10, 0],
        top=1., bot=2.25, theta=0., phi=0., tetm='te',
        structure=np.array([[[1.0, 1.0, 12.25, 12.25]]]), thickness=[300.],
    ).solve()

    assert np.array_equal(result.R, reference.R)
    assert np.array_equal(result.T, reference.T)
    assert result.R.sum() + result.T.sum() == pytest.approx(1.0, abs=1e-13)


def test_run_accepts_spec_object_mapping_and_json():
    """All three input forms are the same run — the caller picks whichever it holds."""
    spec = JobSpec.from_dict(BINARY_GRATING)
    from_obj = wavebit.run(spec)
    from_map = wavebit.run(BINARY_GRATING)
    from_str = wavebit.run(spec.to_json())

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

    # same run, written differently: enum by name vs by int, defaults stated vs omitted
    reworded = dict(BINARY_GRATING)
    reworded['numerics'] = {'backend': 0, 'truncation_order': [10, 0], 'complex_dtype': 128}
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
        'numerics': {'backend': 'JAX', 'factorization': 'nvf', 'truncation_order': 3},
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
        wavebit.run(spec)


def test_expanded_sweep_points_match_direct_calls():
    """Expansion must not perturb the run — point i is exactly the run with that wavelength."""
    spec = JobSpec.from_dict({**BINARY_GRATING, 'sweep': {'wavelength': [800.0, 900.0]}})

    for point in spec.expand():
        direct = dict(BINARY_GRATING)
        direct['incidence'] = {**BINARY_GRATING['incidence'],
                               'wavelength': point.incidence.wavelength}
        assert np.array_equal(wavebit.run(point).R, wavebit.run(direct).R)


# ---------------------------------------------------------------- rejection

@pytest.mark.parametrize('spec, message', [
    ({**BINARY_GRATING, 'wavelength': 900.0}, r'unknown field'),          # misplaced key
    ({**BINARY_GRATING, 'numerics': {'truncation_order': [10, 0], 'backand': 0}}, r'unknown field'),
    ({**BINARY_GRATING, 'numerics': {'backend': 'cupy'}}, r'not a Backend'),
    ({**BINARY_GRATING, 'numerics': {'truncation_order': [1, 2, 3]}}, r'1 or 2 values'),
    ({**BINARY_GRATING, 'numerics': {'truncation_order': [-1, 0]}}, r'cannot be negative'),
    ({**BINARY_GRATING, 'numerics': {'complex_dtype': 32}}, r'128 or 64'),
    ({**BINARY_GRATING, 'version': '99'}, r'version'),
    ({**BINARY_GRATING, 'solver': 'fdtd'}, r'only .rcwa.'),
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
        wavebit.run(42)


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
        wavebit.run(versionless)                     # the same gate on the executing path
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
    stale = wavebit.run({**INDEX_NUMBERS, 'version': '10'})
    current = wavebit.run(BINARY_GRATING)
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
    from wavebit._core.spec import SPEC_VERSION
    from wavebit import Stack, Incidence

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
    from wavebit import Stack, Incidence

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
    :func:`wavebit._core.domain.as_material` turns into the object. The *documents* stay
    distinct, though: a bare number is written back as a bare number, or every spec written
    before materials had a JSON form would change its ``key()``.
    """
    as_object = {**BINARY_GRATING,
                 'stack': dict(BINARY_GRATING['stack'], bot={'epsilon': 2.25})}
    assert np.array_equal(wavebit.run(as_object).R, wavebit.run(BINARY_GRATING).R)
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
    from wavebit import Stack, Incidence

    layer = {'background': Material(12.96),
             'shapes': [{'type': 'rectangle', 'cx': 0.0, 'cy': 0.0, 'lx': 0.2, 'ly': 0.2,
                         'material': Material(2.25, mu=1.5)}]}
    spec = JobSpec(stack=Stack(layers=[wavebit.Layer(thickness=0.2, structure=layer)],
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
    they look like absorption — which the solver's physical check deliberately permits,
    since a lossy stack does exactly that. So the only place it can be caught is here.
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
        r = wavebit.run(point)
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
        stack=wavebit.Stack.from_structure(eps, [300.0], top=1.0, bot=1.0,
                                          period=(400.0, 400.0),
                                          lattice=basis.tolist()),
        incidence=wavebit.Incidence(wavelength=900.0, theta=0.1, phi=0.3, psi=0.2),
        numerics=wavebit.Numerics(backend=0, truncation_order=[4, 4],
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

    a = wavebit.run(spec)
    b = wavebit.run(back)
    assert np.asarray(a.R).tobytes() == np.asarray(b.R).tobytes()

    # And the field, not only the efficiencies. `R` is a sum over orders, so it survives a
    # per-order wavevector that is wrong by a whole reciprocal step -- the field is where such a
    # thing is a different *shape* rather than a different total. The Bloch-condition test in
    # `tests/test_oblique_lattice.py` is what pins that; this is the same array being compared
    # for a different reason, and it is simply the larger one: 5 axes against 1.
    fields = []
    for one in (spec, back):
        solver = wavebit.rcwa(**one.stack.as_kwargs(), **one.incidence.as_kwargs(),
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
    oblique = float(np.sum(wavebit.run(spec).R))

    square = JobSpec(stack=wavebit.Stack.from_structure(
        spec.stack.structure, spec.stack.thickness, top=1.0, bot=1.0, period=(400.0, 400.0)),
        incidence=spec.incidence, numerics=spec.numerics)
    assert square.stack.lattice is None
    assert not np.isclose(oblique, float(np.sum(wavebit.run(square).R)))


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

    hints = typing.get_type_hints(wavebit.Numerics)
    out = {}
    for f in dataclasses.fields(wavebit.Numerics):
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
    """Every way the JSON door accepts ``member``: itself, its name in either case, its code."""
    out = [member, member.name.lower(), member.name.upper()]
    if isinstance(member.value, int):
        out.append(int(member))
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
    reference = wavebit.Numerics.from_dict({name: member.name.lower()})
    for spelling in _spellings(member):
        built = wavebit.Numerics(**{name: spelling})
        assert getattr(built, name) is member, spelling
        assert built == reference == wavebit.Numerics.from_dict({name: spelling}), spelling

        # the solver keyword: the plain value, never the member, never dropped
        kwargs = built.as_kwargs()
        assert kwargs == reference.as_kwargs(), spelling
        assert kwargs[name] == member.value and type(kwargs[name]) is type(member.value), (
            spelling, kwargs.get(name))

        # the document: a plain string, the canonical one, and it reads back as itself
        written = built.to_dict()[name]
        assert type(written) is str and written == member.name.lower(), (spelling, written)
        assert wavebit.Numerics.from_dict(json.loads(json.dumps(built.to_dict()))) == built

        # and the whole spec: one run, one key, whichever spelling built it
        spec = JobSpec(numerics=built)
        assert spec.key() == JobSpec(numerics=reference).key(), spelling
        assert JobSpec.from_json(spec.to_json()).key() == spec.key(), spelling


@pytest.mark.parametrize('name', sorted(ENUM_FIELDS))
def test_numerics_refuses_a_name_that_is_no_member_on_both_doors(name):
    """Refused at construction, with the message the JSON door gives -- not stored as typed."""
    with pytest.raises(SpecError) as direct:
        wavebit.Numerics(**{name: 'bogus'})
    with pytest.raises(SpecError) as parsed:
        wavebit.Numerics.from_dict({name: 'bogus'})
    assert str(direct.value) == str(parsed.value)
    assert str(direct.value).startswith(f'numerics.{name}: ')


@pytest.mark.parametrize('name', sorted(ENUM_FIELDS))
def test_numerics_reads_none_as_the_json_door_does(name):
    """``None`` is "unset" where the annotation is ``Optional`` and a refusal where it is not."""
    _, optional = ENUM_FIELDS[name]
    if optional:
        for numerics in (wavebit.Numerics(**{name: None}), wavebit.Numerics.from_dict({name: None})):
            assert getattr(numerics, name) is None
            assert name not in numerics.as_kwargs()
        return
    with pytest.raises(SpecError) as direct:
        wavebit.Numerics(**{name: None})
    with pytest.raises(SpecError) as parsed:
        wavebit.Numerics.from_dict({name: None})
    assert str(direct.value) == str(parsed.value)


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

    `tests/test_pytree_round_trip.py` holds the solver-side round trip for one value per field;
    this is the spec side of the same seam, for every member.
    """
    pytest.importorskip('jax')
    from jax import tree_util

    eps = np.ones((1, 8, 8))
    eps[0, 2:5, 1:6] = 6.0
    spec = JobSpec(
        stack=wavebit.Stack.from_structure(eps, [120.0], top=1.0, bot=2.25,
                                           period=(400.0, 400.0)),
        incidence=wavebit.Incidence(wavelength=900.0, theta=0.2, phi=0.1, psi=0.3),
        numerics=wavebit.Numerics(truncation_order=(2, 2),
                                  **{'backend': 'jax', **_SOLVER_COMPANIONS.get(name, {}),
                                     name: member.name.lower()}))
    solver = wavebit.rcwa(**spec.numerics.as_kwargs(), **spec.stack.as_kwargs(),
                          **spec.incidence.as_kwargs())
    assert getattr(solver, name) == member

    leaves, treedef = tree_util.tree_flatten(solver)
    rebuilt = tree_util.tree_unflatten(treedef, leaves)
    assert getattr(rebuilt, name) == member
