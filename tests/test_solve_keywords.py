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
import itertools

import numpy as np
import pytest

import wavebit

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
        4, 4, 4, geometry=(s.freeze_geometry() if s.backend == 1 else None), **kw),
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
    'max_order': 'the pre-rename spelling, which has no property to refuse it by name',
    '_tetm': 'the private slot behind a property, which skips the setter that would refuse it',
    'backend': 'a real attribute that is not a condition',
    'T1': 'internal state',
}

#: Names that must still work -- and that the suite already passes through these doors.
ACCEPTED = ('thickness', 'structure', 'mu', 'factorization', 'fourier_coefficients',
            # Plain attributes, not properties -- deriving the set from properties alone
            # refused all three, and a wavelength sweep is the most ordinary call here.
            'wavelength', 'layer_algorithm', 'normal_grid')


def _scene(backend=0, **over):
    kw = dict(backend=backend, structure=GRATING, period=[P, P], truncation_order=[2, 0],
              thickness=[H], wavelength=WL, top=1.0, bot=1.0, answers='field')
    kw.update(over)
    return wavebit.rcwa(**kw)


def _backends():
    out = [0]
    for backend, module in ((1, 'jax'), (2, 'torch')):
        if pytest.importorskip.__module__ and __import__('importlib').util.find_spec(module):
            out.append(backend)
    return out


BACKENDS = _backends()
BACKEND_IDS = {0: 'numpy', 1: 'jax', 2: 'torch'}


@pytest.mark.parametrize('door,name', list(itertools.product(sorted(DOORS), sorted(REFUSED))))
@pytest.mark.parametrize('backend', BACKENDS, ids=lambda b: BACKEND_IDS[b])
def test_a_keyword_that_is_not_a_condition_is_refused_by_name(backend, door, name):
    """Every door x every way of being wrong. The product is the point.

    A spot check on `solve` alone would have passed while four other doors kept the hole -- and
    they did keep it, in the same file, for the same reason: the line was copied.
    """
    if backend:
        pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
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


@pytest.mark.parametrize('backend', BACKENDS, ids=lambda b: BACKEND_IDS[b])
def test_a_real_condition_still_reaches_the_solve_and_moves_the_answer(backend):
    """The refusal must not have closed the door it is guarding.

    Asserted as *the answer moves*, not as *the call succeeds*: an assignment that lands in the
    wrong place also succeeds, and that is the whole defect.
    """
    if backend:
        pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
    solver = _scene(backend)
    base = float(np.asarray(solver.solve().T).sum())
    moved = float(np.asarray(solver.solve(thickness=[3 * H]).T).sum())
    assert moved != pytest.approx(base, abs=1e-9), (base, moved)
    assert float(np.asarray(solver.thickness)[0]) == pytest.approx(3 * H)


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


@pytest.mark.parametrize('retired,replacement', [('ucell', 'structure'), ('n_top', 'top'),
                                                 ('n_bot', 'bot'), ('pol', 'tetm')])
def test_a_retired_spelling_is_refused_by_what_replaced_it_not_as_unknown(retired, replacement):
    """The retired names keep their own message, which is the reason they are in the set at all.

    They are properties whose setters raise by name, so the derivation includes them and the
    unknown-keyword check passes them through to a refusal that says what replaced them. A
    derivation that excluded them would downgrade four careful messages into one generic
    'unexpected keyword', and the quantity changed with each of those renames -- a reader told
    only to rename the keyword would keep the number.
    """
    solver = _scene()
    with pytest.raises(AttributeError) as caught:
        solver.solve(**{retired: 1.0})
    assert replacement in str(caught.value), str(caught.value)


@pytest.mark.parametrize('backend', BACKENDS, ids=lambda b: BACKEND_IDS[b])
def test_the_condition_set_is_derived_not_listed(backend):
    """It has to track the code, or it is a second list to keep in sync.

    Two sources, and the test recomputes both rather than restating an answer. Properties alone
    is what a first version used, and it refused `wavelength`, `layer_algorithm` and
    `normal_grid` -- plain attributes, and exactly what a sweep moves. Constructor parameters
    alone would drop the retired spellings, which have no parameter and must stay in so their
    own setters can say what replaced them.

    `backend` is subtracted: it is a constructor parameter that cannot take effect afterwards.
    """
    import inspect

    if backend:
        pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
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

    # Neither source alone is the set, which is the whole reason there are two.
    assert derived - settable, 'the constructor contributes nothing; properties alone would do'
    assert derived - declared, 'the properties contribute nothing; the constructor alone would do'
    assert 'wavelength' in derived and 'ucell' in derived and 'backend' not in derived
