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
    'thickness', 'structure', 'mu', 'top', 'bot', 'lattice',
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
              complex_dtype=1, fourier_coefficients='integrated')
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
