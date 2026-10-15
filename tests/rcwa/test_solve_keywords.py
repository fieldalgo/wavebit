"""The `**kwargs` door on the solve entry points: what it takes, and what it must refuse.

`solve`, `solve_field`, `freeze_geometry`, `eps_conv_matrices` and `mu_conv_matrices` all take
keywords so an optimizer can move a design variable and re-solve in one call. Every one of them
used to end with a bare ``[setattr(self, k, v) for k, v in kwargs.items()]``, which accepts
anything: a misspelled condition became a brand-new attribute, the real one never moved, and
the **unchanged** answer came back with no exception and no warning.

Measured before the fix, on a 1D grating: ``solve(thicknes=[900.])`` returned T = 0.851185066096,
bit-identical to the base, while the real ``thickness=[900.]`` moves T by thirteen points. The
constructor had refused unknown keywords by name since long before (``_refuse_retired``, pinned
in ``test_domain.py`` with this exact typo); the decision simply never reached the entry points
a sweep actually calls.

Crossed rather than spot-checked. The defect lived in five entry points x three backends, and
the accepted set is derived per class, so a copy that drifted on one backend would be invisible
from another. Every theorem here runs over the product.
"""

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import itertools

import numpy as np
import pytest


P, WL, H = 1500.0, 900.0, 300.0

#: `None` is a legal value for several conditions, so absence needs its own marker.
_MISSING = object()
GRATING = np.array([[[1.0, 1.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0]]])

#: The five doors, as callables taking `**kwargs`. `solve_field` needs its sample counts and
#: `answers='field'`; the scene below declares it so one solver serves all five.
DOORS = {
    'solve': lambda s, **kw: s.solve(**kw),
    # JAX's `solve_field` is jit-compiled and refuses without a frozen geometry, so the door
    # is opened the way its docstring says to. That is the *same* `_solve_and_field` body the
    # other two backends reach directly, which is the point of testing it here at all.
    'solve_field': lambda s, **kw: s.solve_field(
        4, 4, 4, geometry=(s.freeze_geometry() if s.backend == 'jax' else None), **kw),
    'freeze_geometry': lambda s, **kw: s.freeze_geometry(**kw),
    'eps_conv_matrices': lambda s, **kw: s.eps_conv_matrices(**kw),
    'mu_conv_matrices': lambda s, **kw: s.mu_conv_matrices(**kw),
}

#: Names a caller plausibly writes and this door must refuse. Each is a *different* way to be
#: wrong: a near miss, a plural, a pre-rename spelling with no property, a private slot, and a
#: real attribute that is not a condition.
REFUSED = {
    'thicknes': 'a one-letter typo',
    'wavelenght': 'a transposition of the most swept condition there is',
    'truncation_orders': 'a plural',
    'max_order': 'a pre-rename spelling with no property to refuse it by name',
    '_tetm': 'the private slot behind a property, which skips the setter that would refuse it',
    'backend': 'a real attribute that is not a condition',
    'T1': 'internal state',
    # Retired spellings until 2026-08-26, when the machinery that named their
    # replacements was deleted -- wavebit has never been released, so the reader those
    # messages addressed does not exist yet. They are ordinary unknown keywords now, and
    # this is where that is pinned.
    'ucell': 'the pre-rename spelling of `structure`',
    'pol': 'the pre-rename spelling of `tetm`',
    'ucell_materials': 'a keyword that was deleted rather than renamed',
}

#: Names that must still work -- and that the suite already passes through these doors.
ACCEPTED = ('thickness', 'structure', 'mu', 'factorization', 'fourier_coefficients',
            # Plain attributes, not properties -- deriving the set from properties alone
            # refused all three, and a wavelength sweep is the most ordinary call here.
            'wavelength', 'layer_algorithm', 'shape_nvf_grid_size')


def _scene(backend='numpy', **over):
    kw = dict(backend=backend, structure=GRATING, period=[P, P], truncation_order=[2, 0],
              thickness=[H], wavelength=WL, top=1.0, bot=1.0, answers='field')
    kw.update(over)
    return _legacy_wavebit_rcwa(**kw)


def _backends():
    out = ['numpy']
    for backend in ('jax', 'torch'):
        if pytest.importorskip.__module__ and __import__('importlib').util.find_spec(backend):
            out.append(backend)
    return out


BACKENDS = _backends()


@pytest.mark.parametrize('door,name', list(itertools.product(sorted(DOORS), sorted(REFUSED))))
@pytest.mark.parametrize('backend', BACKENDS)
def test_a_keyword_that_is_not_a_condition_is_refused_by_name(backend, door, name):
    """Every door x every way of being wrong. The product is the point.

    A spot check on `solve` alone would have passed while four other doors kept the hole -- and
    they did keep it, in the same file, for the same reason: the line was copied.
    """
    pytest.importorskip(backend)
    solver = _scene(backend)
    with pytest.raises(TypeError, match=r'unexpected keyword argument'):
        DOORS[door](solver, **{name: [900.0]})
    # And the refusal did not half-apply. Three of these names -- `_tetm`, `backend`, `T1` --
    # are real attributes, so "was it stored" cannot be the question; the question is whether
    # the value moved. A refusal that mutates first is worse than the silence it replaced.
    before = getattr(solver, name, _MISSING)
    with pytest.raises(TypeError):
        DOORS[door](solver, **{name: [900.0]})
    after = getattr(solver, name, _MISSING)
    assert before is after or np.array_equal(np.asarray(before, dtype=object),
                                             np.asarray(after, dtype=object)), (
        f'{name!r} changed from {before!r} to {after!r} on the way to being refused')


@pytest.mark.parametrize('name', sorted(REFUSED))
def test_the_refusal_names_the_keyword_and_suggests_a_near_miss(name):
    """A refusal that does not say which name, or what was meant, sends the reader back to grep."""
    solver = _scene()
    with pytest.raises(TypeError) as caught:
        solver.solve(**{name: [900.0]})
    message = str(caught.value)
    assert repr(name) in message, message
    assert 'Settable conditions:' in message, message

    near = {'thicknes': 'thickness', 'wavelenght': 'wavelength',
            'truncation_orders': 'truncation_order', '_tetm': 'tetm'}
    if name in near:
        assert f"did you mean {near[name]!r}" in message, (
            f'{name!r} is one edit from {near[name]!r} and the message did not say so: {message}')


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_real_condition_still_reaches_the_solve_and_moves_the_answer(backend):
    """The refusal must not have closed the door it is guarding.

    Asserted as *the answer moves*, not as *the call succeeds*: an assignment that lands in the
    wrong place also succeeds, and that is the whole defect.
    """
    pytest.importorskip(backend)
    solver = _scene(backend)
    base = float(np.asarray(solver.solve().T).sum())
    moved = float(np.asarray(solver.solve(thickness=[3 * H]).T).sum())
    assert moved != pytest.approx(base, abs=1e-9), (base, moved)
    assert float(np.asarray(solver.thickness)[0]) == pytest.approx(3 * H)


#: A scene every door can be moved on: three rows, so it is 2D; a raster ``mu``, so
#: `mu_conv_matrices` has something to build; NVF, so a freeze records a normal field per
#: quantity; and `answers='field'`, so `solve_field` is open.
DOOR_EPS = np.array([[[1.0, 1.0, 4.0, 4.0], [4.0, 4.0, 1.0, 1.0], [1.0, 4.0, 4.0, 1.0]]])
DOOR_MU = np.array([[[1.0, 1.0, 1.6, 1.6], [1.6, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])

#: Per door, a condition its answer depends on and the value it is moved to. `thickness`
#: changes nothing a convolution matrix or a frozen record holds, so the builders are moved by
#: the field they build instead.
MOVES = {
    'solve_field': ('thickness', [3 * H]),
    'freeze_geometry': ('structure', DOOR_EPS[:, ::-1]),
    'eps_conv_matrices': ('structure', DOOR_EPS[:, ::-1]),
    'mu_conv_matrices': ('mu', DOOR_MU[:, :, ::-1]),
}


def _door_scene(backend, **over):
    kw = dict(backend=backend, structure=DOOR_EPS, mu=DOOR_MU, period=[P, P],
              truncation_order=[2, 2], thickness=[H], wavelength=WL, top=1.0, bot=1.0,
              answers='field', factorization='nvf')
    kw.update(over)
    return _legacy_wavebit_rcwa(**kw)


def _leaves(value):
    """Every array in a door's answer, in a fixed order, as NumPy."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _leaves(item)]
    return [np.asarray(value.detach() if hasattr(value, 'detach') else value)]


#: What each door answers with, as arrays. The frozen record is read by its content, since a
#: record holding arrays compares by identity and two freezes of one scene are never `==`.
READOUT = {
    'solve_field': lambda out: _leaves((out[0].incident.R, out[0].incident.T, out[1])),
    'freeze_geometry': lambda g: _leaves((g.normal_field, g.mu_normal_field)),
    'eps_conv_matrices': _leaves,
    'mu_conv_matrices': _leaves,
}


def _gap(a, b):
    assert len(a) == len(b), (len(a), len(b))
    return max(float(np.abs(x - y).max()) for x, y in zip(a, b))


@pytest.mark.parametrize('door', sorted(MOVES))
@pytest.mark.parametrize('backend', BACKENDS)
def test_a_real_condition_through_every_other_door_answers_as_the_scene_built_with_it(
        backend, door):
    """The accepting half, on the four doors the test above does not open.

    The refusal is crossed over all five doors; the acceptance was asserted on `solve` alone,
    so a door that took a real keyword and dropped it -- the original defect, one door over --
    would have passed. Here each door is handed a condition its answer depends on, and the
    answer must be the one a solver *built* with that condition gives, and must differ from the
    unmoved scene's, so a door that ignored the keyword cannot pass on a condition that happened
    to change nothing. And every door keeps the condition afterwards, as `solve` does.

    That last half held on every door but one until 2026-09-24. JAX's `solve_field` is
    jit-compiled over a copy of the solver and took its keywords inside the trace, so it
    answered for the new thickness and left the live solver at the old one, while NumPy and
    Torch kept it -- the same sweep loop's next `solve()` answered two different scenes.
    """
    pytest.importorskip(backend)
    if backend == 'jax':
        _legacy_wavebit_enable_x64()
    name, value = MOVES[door]
    solver = _door_scene(backend)
    moved = READOUT[door](DOORS[door](solver, **{name: value}))
    built = READOUT[door](DOORS[door](_door_scene(backend, **{name: value})))
    unmoved = READOUT[door](DOORS[door](_door_scene(backend)))

    assert _gap(moved, built) < 1e-12, door
    assert _gap(unmoved, built) > 1e-6, f'{name!r} moved nothing {door} answers with'
    assert np.array_equal(_leaves(getattr(solver, name))[0], np.asarray(value)), door


#: Conditions that are not arrays: a dispatch string and the integers that size the problem.
#: JAX's `solve_field` handed its keywords to `jax.jit` as arguments until 2026-09-24, so the
#: strings were refused as values it cannot trace and `truncation_order` reached its setter as
#: a tracer, where it needs integers. The device is the third such condition, and it is not
#: here: `'cpu'` is where every solver in this process already is, so the entry asserted
#: nothing. It is exercised on two forced host devices, in a subprocess, by
#: `tests/rcwa/test_jax_device_placement.py::test_the_jitted_field_door_moves_the_solver_it_is_given`.
NON_ARRAY = {'layer_algorithm': 'SMM', 'truncation_order': [1, 1]}


@pytest.mark.parametrize('name', sorted(NON_ARRAY))
@pytest.mark.parametrize('backend', BACKENDS)
def test_a_condition_that_is_not_an_array_passes_the_field_door_on_every_backend(backend, name):
    """``solve_field(layer_algorithm='SMM')`` answers, as the scene built with it, and keeps it.

    A sweep over the connecting algorithm or the truncation is as ordinary as one over a
    thickness, and on NumPy and Torch it always worked through this door. On JAX's jitted door
    it raised: ``TypeError`` for the strings, ``ValueError`` for ``truncation_order``. Such a
    condition is static data of the compiled call now, so a new value compiles once more, the
    way a new solver would, and is kept like any other.

    JAX's door is handed a freeze of the scene it is moved *to*. A freeze describes one scene,
    and the NVF record is sampled at a size the truncation sets (12 x 12 at order 2, 6 x 8 at
    order 1), so the unmoved scene's freeze is stale for ``truncation_order`` -- and the jitted
    door replays a record on trust, as its docstring says.
    """
    pytest.importorskip(backend)
    if backend == 'jax':
        _legacy_wavebit_enable_x64()
    value = NON_ARRAY[name]
    solver = _door_scene(backend)
    built = _door_scene(backend, **{name: value})
    geometry = built.freeze_geometry() if backend == 'jax' else None
    moved = solver.solve_field(4, 4, 4, geometry=geometry, **{name: value})
    want = built.solve_field(4, 4, 4, geometry=geometry)
    assert _gap(READOUT['solve_field'](moved), READOUT['solve_field'](want)) < 1e-12
    assert getattr(solver, name) == getattr(built, name)


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_field_door_refused_for_its_declaration_assigns_nothing(backend):
    """A solver that did not declare ``answers='field'`` is refused before any keyword lands.

    The refusal was checked first on every backend while the keywords were assigned inside the
    call. Moving JAX's assignment out of the trace put it ahead of the jitted body, where the
    declaration used to be read, so the order is pinned here: refused, and the thickness is the
    one the solver was built with.
    """
    pytest.importorskip(backend)
    solver = _door_scene(backend, answers='amplitude')
    with pytest.raises(RuntimeError, match='field'):
        DOORS['solve_field'](solver, thickness=[3 * H])
    assert float(np.asarray(_leaves(solver.thickness)[0])[0]) == H


@pytest.mark.parametrize('name', ACCEPTED)
def test_every_keyword_the_suite_passes_is_still_accepted(name):
    """The accepted set is derived, so it can shrink by accident. These five are load-bearing.

    Each is passed through one of these doors somewhere in this suite or in `studies/`; a
    derivation that stopped including one would break those callers rather than this test, and
    a green suite would not obviously say why.
    """
    assert name in _scene()._condition_names(), (
        f'{name!r} is passed through a solve entry point elsewhere in this suite and the '
        f'derived condition set no longer contains it')


@pytest.mark.parametrize('backend', BACKENDS)
def test_the_condition_set_is_derived_not_listed(backend):
    """It has to track the code, or it is a second list to keep in sync.

    Two sources, and the test recomputes both rather than restating an answer. Properties alone
    is what a first version used, and it refused `wavelength`, `layer_algorithm` and
    `shape_nvf_grid_size` -- plain attributes, and exactly what a sweep moves. Constructor parameters
    alone would drop the retired spellings, which have no parameter and must stay in so their
    own setters can say what replaced them.

    `backend` is subtracted: it is a constructor parameter that cannot take effect afterwards.
    """
    import inspect

    pytest.importorskip(backend)
    solver = _scene(backend)
    derived = solver._condition_names()

    declared = {name
                for klass in type(solver).__mro__
                for name, parameter in inspect.signature(klass.__init__).parameters.items()
                if name != 'self'
                and parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)}
    settable = {name
                for klass in type(solver).__mro__
                for name, member in vars(klass).items()
                if isinstance(member, property) and member.fset is not None
                and not name.startswith('_')}
    assert derived == (declared | settable) - {'backend'}, derived ^ ((declared | settable) - {'backend'})

    # Neither source alone is the set, which is the whole reason there are two -- and after
    # the retired spellings were deleted the property half came within one name of
    # contributing nothing. `formulation` is that name: a property with a setter and no
    # constructor parameter. If it ever gains one, this collapses to the signature and the
    # union should go with it.
    assert derived - settable, 'the constructor contributes nothing; properties alone would do'
    assert derived - declared == {'formulation'}, sorted(derived - declared)
    assert 'wavelength' in derived and 'backend' not in derived
    assert 'ucell' not in derived and 'pol' not in derived


# ------------------------------------------- the sixth door: plain attribute assignment

#: Every backend that is importable, by name. The guard is derived per class, so a
#: version that drifted on one backend would be invisible from another -- the same argument
#: the keyword tests above are crossed for.
_GUARD_BACKENDS = ['numpy']
for _bk in ('jax', 'torch'):
    try:
        __import__(_bk)
    except ImportError:                                    # pragma: no cover - env-dependent
        pass
    else:
        _GUARD_BACKENDS.append(_bk)


def _guard_solver(backend, cls=None):
    kw = dict(structure=GRATING, period=[P, P], truncation_order=(3, 0), thickness=[H],
              wavelength=WL, theta=0.0)
    return cls(**kw) if cls is not None else _legacy_wavebit_rcwa(backend=backend, **kw)


@pytest.mark.parametrize('backend', _GUARD_BACKENDS)
@pytest.mark.parametrize('typo, meant', [('thicknes', 'thickness'), ('wavelenght', 'wavelength'),
                                         ('truncation_orders', 'truncation_order'),
                                         ('perio', 'period'), ('max_order', None),
                                         ('ucell', None)])
def test_a_typoed_attribute_is_refused_rather_than_stored(backend, typo, meant):
    """The door the five above did not cover, and the one the tutorials teach.

    ``rcwa(thicknes=...)`` and ``solve(thicknes=...)`` have been refused since 2026-08-26.
    ``solver.thicknes = [900.]`` was stored as a new attribute until 2026-08-27, and the solve
    then answered for the thickness that was *not* changed -- bit-identically to the untouched
    run, so there was nothing to notice. Every one of these spellings moved the answer when
    written correctly, which is what makes the silence a wrong answer rather than a no-op.
    """
    solver = _guard_solver(backend)
    before = float(np.sum(np.asarray(solver.solve().T)).real)

    with pytest.raises(AttributeError, match=typo):
        setattr(solver, typo, [900.0])
    assert not hasattr(solver, typo)

    if meant is not None:
        # and the name it was a typo *of* moves the answer, so the silence was not harmless
        message = str(pytest.raises(AttributeError,
                                    lambda: setattr(solver, typo, [900.0])).value)
        assert meant in message, message
        setattr(solver, meant, [900.0] if meant == 'thickness' else getattr(solver, meant))
        if meant == 'thickness':
            assert float(np.sum(np.asarray(solver.solve().T)).real) != before


def test_a_name_spelled_like_another_quantity_is_not_pointed_at_a_material_condition():
    """``solve(n_top=1.5)`` said "did you mean 'top'?" -- 0.75 against this door's 0.7 cutoff.

    The descriptor branch below keeps ``solver.n_top = 1.5`` off the hint (the solver has a
    read-only ``n_top``), but the keyword door and every other index spelling reached it: a
    refractive index pointed at a permittivity slot, the quantity error ``docs/reference/common/conventions.md``
    exists to prevent. The guess is ``enums.near_match`` now, which never crosses quantities;
    the plain match is asserted first so this cannot pass by difflib happening not to reach.
    """
    import difflib
    from tests.api.test_spec import INDEX_SPELLINGS
    solver = _guard_solver('numpy')
    legal = sorted(solver._condition_names())
    assert difflib.get_close_matches('n_top', legal, n=1, cutoff=0.7) == ['top']
    assert difflib.get_close_matches('mu_top', legal, n=1, cutoff=0.6) == ['top']
    assert {'n_top', 'n_bot', 'ntop', 'n_structure', 'RI_top', 'n_sub'} <= set(INDEX_SPELLINGS)
    for name in INDEX_SPELLINGS:
        with pytest.raises(TypeError) as keyword:
            solver.solve(**{name: 1.5})
        messages = [str(keyword.value)]
        if not hasattr(type(solver), name):
            with pytest.raises(AttributeError) as attribute:
                setattr(solver, name, 1.5)
            messages.append(str(attribute.value))
        for message in messages:
            assert 'did you mean' not in message.lower(), message
            assert 'relative permittivity (eps = n**2) or a Material' in message, message
    with pytest.raises(AttributeError) as permeability:     # `mu` is a condition: no note
        solver.mu_top = 1.5
    assert 'did you mean' not in str(permeability.value).lower(), permeability.value


def test_a_count_spelled_with_n_keeps_the_guess_plain_difflib_gives():
    """The other direction on both solver doors: ``n_theta`` counts angles, it is no index. Read
    as one, ``solve(n_theta=...)`` lost "did you mean 'theta'?" and was told the material slots
    take permittivity. Each door's own cutoff (0.7 on the keywords, 0.6 on the attribute), and
    at least one count reaches a condition, so this is not every guess being none."""
    import difflib
    from tests.api.test_spec import COUNT_SPELLINGS
    solver = _guard_solver('numpy')
    legal = sorted(solver._condition_names())
    reached = 0
    for name in COUNT_SPELLINGS + ('n_period', 'n_phi'):
        assert not hasattr(type(solver), name), name
        with pytest.raises(TypeError) as keyword:
            solver.solve(**{name: 1.5})
        with pytest.raises(AttributeError) as attribute:
            setattr(solver, name, 1.5)
        for message, cutoff, spelled in ((str(keyword.value), 0.7, '(did you mean {!r}?)'),
                                         (str(attribute.value), 0.6, 'Did you mean {!r}?')):
            plain = difflib.get_close_matches(name, legal, n=1, cutoff=cutoff)
            reached += bool(plain)
            assert 'spelled like' not in message, message
            if plain:
                assert spelled.format(plain[0]) in message, message
            else:
                assert 'did you mean' not in message.lower(), message
    assert reached >= 4, reached


@pytest.mark.parametrize('backend', _GUARD_BACKENDS)
def test_the_guard_leaves_the_descriptor_protocol_in_charge(backend):
    """A read-only property is not a missing attribute, and must not be reported as one.

    This is the branch that costs the least to leave out and the most to get wrong. Without it
    the guard preempts the property protocol: ``solver.n_top = 1.5`` came back "has no attribute
    'n_top' to set. Did you mean 'top'?" -- pointing a refractive *index* at a *permittivity*
    slot, which is the one quantity error ``docs/reference/common/conventions.md`` exists to prevent. Twelve
    properties were affected, and the type is right in both versions, so a test that checked
    only ``pytest.raises(AttributeError)`` could not see it.
    """
    solver = _guard_solver(backend)
    for prop in ('n_top', 'n_bot', 'modeling_path', 'grating_type', 'scattering',
                 'formulation_assigned', 'order_window', 'tetm_index'):
        with pytest.raises(AttributeError, match='has no setter'):
            setattr(solver, prop, 1.5)


@pytest.mark.parametrize('backend', _GUARD_BACKENDS)
def test_a_method_can_still_be_stubbed_on_an_instance(backend):
    """The other half of that branch, and a capability a guard silently removes.

    ``monkeypatch.setattr(solver, 'solve', ...)`` is how a test isolates a caller from the
    solve. The first version of the guard refused it with a message saying ``solve`` did not
    exist -- of a method that does. Nothing in this suite patches an instance today, which is
    exactly why it needs pinning here rather than being left to be discovered downstream.
    """
    solver = _guard_solver(backend)
    solver.solve = lambda *a, **k: 'stubbed'
    assert solver.solve() == 'stubbed'


@pytest.mark.parametrize('backend', _GUARD_BACKENDS)
def test_a_subclass_opts_its_own_state_in_with_a_class_attribute(backend):
    """Refused by default, and there is a way in that is not a private name.

    A solver subclass cannot simply add ``self.label = ...``: the guard cannot tell that from a
    typo, and letting it through on subclasses alone would make whether a name is checked
    depend on which class wrote it. Declaring ``label = None`` on the class opts it in through
    the class-presence branch -- a decision spelled out once, where a reader can see it.
    """
    base = type(_guard_solver(backend))
    plain = base.__new__(base)                     # no __init__: the guard is what is at issue
    with pytest.raises(AttributeError, match='label'):
        plain.label = 'case-3'

    declared = type('Labelled', (base,), {'label': None})
    solver = _guard_solver(backend, cls=declared)
    solver.label = 'case-3'
    assert solver.label == 'case-3'
    assert float(np.sum(np.asarray(solver.solve().T)).real) > 0.0
