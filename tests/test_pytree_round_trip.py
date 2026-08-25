"""One exhaustive check that a flattened solver comes back as the solver it was.

`SolverJax` is a registered pytree, and `solve_field` is `jax.jit`-decorated at class level,
so `self` is flattened and rebuilt on **every call**. Anything `_tree_flatten` forgets is not
merely lost -- the solve then runs on the rebuilt solver, which is to say on a different scene
than the caller configured, and it answers.

The record of that is in `_tree_flatten`'s own comments, which read as a list of things found
missing one at a time: `polarization` rode in aux and made a differentiated pair return exactly
`0j`; `answers` was absent, so a solver that declared `'field'` came back reporting
`'amplitude'` *and* shared a jit cache entry with one that had not declared it; `lattice` and
`normal_grid` were both missing, and on a 60-degree cell the jitted call answered the
rectangular problem -- T = 0.6565 against 0.5460, energy conserving throughout.

Each of those got a test next to the feature it belonged to, in five different files. This is
the one that is about the round trip itself, and it is written so the *next* omission is caught
by the test that already exists rather than by the bug that follows it:

* every key `_tree_flatten` puts in `aux_data` must appear in `CARRIED` below, so adding a key
  without adding it here fails;
* every attribute in `CARRIED` must come back equal;
* and the scene is built so that no attribute sits at its default -- an omitted key that
  happens to default to the right value proves nothing.
"""
import numpy as np
import pytest

import wavebit

pytest.importorskip('jax')

P, WL, H = 400.0, 900.0, 150.0

#: What a round trip has to preserve, as attribute names on the solver. Aux keys and child
#: slots together -- the split between them is `_tree_flatten`'s business, not this test's.
CARRIED = (
    'backend', 'tetm', 'truncation_order', 'layer_algorithm', 'complex_dtype',
    'fourier_coefficients', 'factorization', 'normal_method', 'incidence_side', 'answers',
    'normal_grid', 'theta', 'phi', 'psi', 'chi', 'polarization', 'period', 'wavelength',
    'thickness', 'structure', 'mu', 'top', 'bot', 'lattice', 'excitation',
)

#: Aux keys that are not attributes under the same name. `psi_given` is the *fact* that `psi`
#: was named rather than derived, and `device` is a per-backend handle that compares by
#: identity rather than by value -- both are checked separately below.
AUX_NOT_IN_CARRIED = {'psi_given', 'device'}


def _scene(**over):
    """A solver with nothing at its default: magnetic, skewed, elliptical, field-declared."""
    basis = np.column_stack(([P, 0.0], [P / 2, P * np.sqrt(3) / 2]))
    rng = np.random.default_rng(0)
    eps = np.where(rng.random((1, 16, 16)) < 0.4, 5.76, 1.0)
    mu = np.where(rng.random((1, 16, 16)) < 0.3, 1.8, 1.0)
    kw = dict(backend=1, structure=eps, mu=mu, period=[P, P], lattice=basis.tolist(),
              truncation_order=[3, 2], thickness=[H, 0.5 * H], wavelength=WL,
              top=wavebit.Material(2.25, mu=1.4), bot=wavebit.Material(4.0),
              theta=0.31, phi=0.47, psi=0.29, chi=np.pi / 9,
              factorization='nvf', normal_method='contour', normal_grid=512,
              incidence_side='bottom', layer_algorithm='SMM', answers='field',
              complex_dtype=1, fourier_coefficients='integrated',
              excitation={(0, 0): 1.0, (1, 0): 0.5j})
    kw.update(over)
    return wavebit.rcwa(**kw)


def _one_row(**over):
    """A 1D scene, for the incidence spellings `_scene` cannot hold.

    `psi`, `polarization` and `tetm` are three mutually exclusive ways to state the same thing:
    naming two is refused. So no single scene can leave all three off their defaults, and
    "nothing at its default" is a claim about the scenes *together*.
    """
    kw = dict(backend=1, structure=np.ones((1, 1, 12)), period=[P, P],
              truncation_order=[3, 0], thickness=[H], wavelength=WL, top=1.0, bot=1.0)
    kw.update(over)
    return wavebit.rcwa(**kw)


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, wavebit.Material) or isinstance(b, wavebit.Material):
        return _same(a.epsilon, b.epsilon) and _same(a.mu, b.mu)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        return np.array_equal(np.asarray(a), np.asarray(b))
    except Exception:                                  # noqa: BLE001 - a handle, not a value
        return a == b


def _round_trip(solver):
    from jax import tree_util
    leaves, treedef = tree_util.tree_flatten(solver)
    return tree_util.tree_unflatten(treedef, leaves)


def test_the_carried_list_covers_every_aux_key():
    """The list above cannot drift from `_tree_flatten` without this failing.

    Without it, the natural way to add an aux key is to add it and stop -- which is exactly how
    `answers`, `lattice` and `normal_grid` came to be missing in the first place. This makes
    the omission a red test rather than a wrong number three months later.
    """
    _, aux = _scene()._tree_flatten()
    unchecked = set(aux) - set(CARRIED) - AUX_NOT_IN_CARRIED
    assert not unchecked, (
        f'these ride in aux_data and nothing here compares them after a round trip: '
        f'{sorted(unchecked)}. Add them to CARRIED, or to AUX_NOT_IN_CARRIED with the reason '
        f'they cannot be compared by name.')


@pytest.mark.parametrize('attribute', CARRIED)
def test_a_round_trip_preserves(attribute):
    """Every carried attribute, one test each so a failure names the one that was lost."""
    original = _scene()
    rebuilt = _round_trip(original)
    before, after = getattr(original, attribute), getattr(rebuilt, attribute)
    assert _same(before, after), f'{attribute}: {before!r} -> {after!r}'


def test_the_scene_holds_nothing_at_its_default():
    """Otherwise an omitted key that defaults to the right value would pass.

    Compared against a solver built with only what `rcwa` requires: every attribute this test
    carries has to differ, or the scene is not exercising it.
    """
    rich = _scene()
    plain = wavebit.rcwa(backend=1, structure=np.ones((1, 8, 8)), period=[P, P],
                         truncation_order=[1, 1], thickness=[H], wavelength=WL,
                         top=1.0, bot=1.0)
    others = (_one_row(polarization=(1.0, -1j)), _one_row(tetm='te', phi=None))
    same_as_default = [
        a for a in CARRIED
        if a not in ('backend', 'period', 'wavelength')
        and all(_same(getattr(s, a), getattr(plain, a)) for s in (rich, *others))]
    assert not same_as_default, (
        f'these sit at their default in both round-trip scenes, so a `_tree_flatten` that '
        f'dropped them would still pass: {same_as_default}')


def test_psi_given_survives_because_the_psi_child_would_overwrite_it():
    """`psi_given` is a fact about how the incidence was *stated*, not a value.

    The psi child is always present -- it is the traced leaf, and it is non-None even when it
    was merely derived from `tetm`. So a rebuild that did not carry `psi_given` would come back
    claiming an explicit `psi` that nobody wrote, and the solver would then refuse a
    `polarization` pair it had accepted a moment earlier.
    """
    for given, kw in ((True, {'psi': 0.29}), (False, {'tetm': 'te', 'phi': None})):
        solver = _scene(**kw) if given else _one_row(**kw)
        assert solver._psi_given is given, kw
        assert _round_trip(solver)._psi_given is given, kw


def test_the_rebuild_names_its_arguments():
    """`_tree_unflatten` may not hand the children to `__init__` by position.

    It used to: `*core, chi, polarization, mu = children` and then `cls(*core, ...)`, which made
    the order of `children` and the order of `__init__`'s leading parameters one undeclared
    contract between two files. Adding a positional parameter anywhere but the end would have
    shifted every child by one and rebuilt a *different solver*, in silence -- and since
    `solve_field` flattens `self` on every call, the wrong solver is the one that answers.

    `_tree_flatten`'s own comments warned about it twice ("It rides last so the leading tuple
    still lines up with `__init__`'s positional parameters"). A warning in a comment is a note
    that the next person has to read; this is the version that fails.

    Read from the source rather than from behaviour, because behaviour cannot see it: a
    positional rebuild is correct right up until the signature moves, and then it is silently
    wrong. Only the shape of the call says which one is in the file.
    """
    import ast
    import inspect

    from wavebit.on_jax import solver as jax_solver

    tree = ast.parse(inspect.getsource(jax_solver))
    body = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == '_tree_unflatten')
    starred = [n for n in ast.walk(body) if isinstance(n, ast.Starred)]
    assert not starred, (
        'a `*` unpack in `_tree_unflatten` means the children are matched to `__init__` by '
        'position again. Pass them by name -- `CHILD_NAMES` is the listing.')

    names = set(jax_solver.CHILD_NAMES)
    flattened, _ = _scene()._tree_flatten()
    assert len(flattened) == len(names), (
        f'`CHILD_NAMES` lists {len(names)} children and `_tree_flatten` produced '
        f'{len(flattened)}; they are the same tuple and have to stay the same length.')


def test_every_constructor_parameter_is_carried_by_the_pytree():
    """The direction the guard above cannot see, and the one `excitation` was lost in.

    `test_the_carried_list_covers_every_aux_key` runs aux -> CARRIED. It catches a key added to
    `_tree_flatten` and not compared here. It cannot catch the thing that actually keeps
    happening, which is a parameter added to the *constructor* and put in neither slot -- there
    is no aux key to notice, so the check has nothing to fire on.

    That is how `excitation` got here. It is this branch's feature, it went into the signature
    and into the solve, and the pytree never learned about it. `solve_field` is jit-decorated
    at class level, so `self` is flattened and rebuilt every call, and the rebuild handed the
    solve `excitation = None`: a multimode scene silently answered the single-wave problem.
    Eager against jitted on a 1D grating, R read 0.148814934 where eager gave 0.164651118 --
    and 0.148814934 is exactly what the same scene answers with no excitation at all.

    Read from the signature rather than from a list, so nothing has to be remembered. A
    parameter that genuinely must not ride (there are none today) fails here and has to say so
    in writing.
    """
    import inspect

    from wavebit.on_jax.emsolver.rcwa import RCWAJax
    from wavebit.on_jax.solver import CHILD_NAMES

    _, aux = _scene()._tree_flatten()
    carried = set(CHILD_NAMES) | set(aux)

    # `**retired` is `_refuse_retired`'s catch-all, not a condition -- it holds names that are
    # refused rather than stored, so there is nothing for a round trip to preserve.
    declared = [name for name, p in inspect.signature(RCWAJax.__init__).parameters.items()
                if name != 'self'
                and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]

    dropped = [name for name in declared if name not in carried]
    assert not dropped, (
        f'these are constructor parameters and the JAX pytree carries neither a child nor an '
        f'aux key for them: {dropped}. A jitted call rebuilds the solver through `__init__` '
        f'(`SolverJax._tree_unflatten`), so each of these comes back at its default and the '
        f'solve answers a scene the caller did not configure. Add it to `CHILD_NAMES` if it '
        f'is a value (it may then be traced and differentiated), or to `aux_data` if the '
        f'solves branch on it.')

#: A grating whose orders actually carry power, and a period long enough that order 1 is inside
#: the light line -- `_one_row`'s 400 nm at 900 nm makes it evanescent, which `excitation`
#: refuses by name. On a uniform slab the derivative below is 1e-16 and proves nothing.
_LIT = np.array([[[1., 1., 1., 4., 4., 4., 4., 1., 1., 1., 1., 1.]]])


def _lit(excitation):
    return wavebit.rcwa(backend=1, structure=_LIT, period=[3 * WL, 3 * WL],
                        truncation_order=[3, 0], thickness=[220.0], wavelength=WL,
                        top=1.0, bot=1.0, excitation=excitation)


def test_excitation_rides_as_a_child_so_an_amplitude_is_a_leaf_and_not_a_cache_key():
    """Which slot it rides in, measured three ways -- because the obvious way does not work.

    `test_a_round_trip_preserves[excitation]` passes with the dict in *either* slot: aux data
    round-trips too. The difference only shows up in the treedef.

    As a child, JAX flattens the dict with its keys in the treedef and its amplitudes as
    leaves, which is exactly the split the property documents: "they may be traced ... while
    the keys may not, because which entries of the incident column are non-zero decides array
    structure rather than a value."

    In aux the amplitudes become part of the treedef, and both halves of that are bad. Every
    amplitude is a new jit cache key, so a sweep over beam weights recompiles per step. And
    `jax.grad` of a function of the solver hands the aux back untouched -- measured on this
    scene, it returns ``{0: 1.0, 1: 0.5j}``, the *inputs*, where the child slot gives
    ``{0: -0.00129, 1: 0.00513 - 0.00258j}``. That is worse than the ``0j`` `polarization`
    used to return: a caller stepping downhill would step by the amplitude itself, and an
    assertion that the gradient is merely non-zero passes.
    """
    import jax
    import jax.numpy as jnp
    from jax import tree_util

    lit, other, unlit = _lit({0: 1.0, 1: 0.5j}), _lit({0: 1.0, 1: 0.9j}), _lit(None)

    # 1. The amplitudes are leaves. In aux the lit solver flattens to exactly as many leaves
    #    as the unlit one, which is the same thing as not carrying them.
    assert len(tree_util.tree_leaves(lit)) == len(tree_util.tree_leaves(unlit)) + 2, (
        'the two beam amplitudes are not leaves of the flattened solver')

    # 2. Changing an amplitude does not change the structure -- one compilation serves both.
    assert tree_util.tree_structure(lit) == tree_util.tree_structure(other), (
        'two solvers differing only in a beam amplitude have different treedefs, so every '
        'amplitude in a sweep is a fresh jit compilation')

    # 3. The gradient is a gradient, not the input handed back.
    grad = jax.grad(lambda s: jnp.sum(s.solve().T).real)(lit).excitation
    for order, amplitude in lit.excitation.items():
        assert complex(np.asarray(grad[order])) != complex(amplitude), (
            f'order {order}: grad came back as the amplitude itself ({amplitude!r}), which is '
            f'what aux_data does -- it is passed through, not differentiated')
    assert max(abs(complex(np.asarray(v))) for v in grad.values()) > 1e-6, (
        f'every beam amplitude derivative is at the noise floor: {grad}')
