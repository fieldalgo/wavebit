"""A backend is named: ``'numpy'``, ``'jax'`` or ``'torch'``, at every door that takes one.

The names were the integers ``0`` / ``1`` / ``2`` until 2026-10-11. Nothing was released, so the
integers were deleted rather than aliased: they fall to the same refusal as any other spelling,
and the refusal lists the names. ``docs/reference/common/conventions.md`` ("Backend names") is the contract this
pins.

Every door is listed by hand rather than discovered, because the failure this guards against is
a door that kept its own parser -- ``meshgrid_xy`` compared ``backend == 1`` on its own until the
rename, and the spec door coerced through its own table -- and a door nobody listed is the one
that would keep it.
"""

from wavebit.compat import legacy as _legacy_wavebit
import json

import numpy as np
import pytest

import wavebit
from wavebit import Backend
from wavebit._common.enums import DIFFERENTIABLE_BACKENDS, backend_of

NAMES = ['numpy', 'jax', 'torch']

#: Every spelling that is not a backend. The integers are the retired spelling; ``True`` is an
#: ``int`` to Python and must not reach a table keyed on them; the case and the library's other
#: names are near misses a case-folding parser would accept.
REFUSED = [0, 1, 2, True, False, None, 1.0, 'JAX', 'Numpy', 'pytorch', 'np', 'cupy', '']

LISTED = r"expected one of \['numpy', 'jax', 'torch'\]"

SCENE = dict(structure=np.ones((1, 4, 4)), period=[400.0, 400.0], truncation_order=[1, 1],
             thickness=[150.0], wavelength=900.0, top=1.0, bot=1.0)


def _class_of(name):
    """The solver class a name builds, imported only when that backend is asked for."""
    module = __import__(f'wavebit.rcwa._backends.{name}_backend', fromlist=['x'])
    return getattr(module, {'numpy': 'SolverNumpy', 'jax': 'SolverJax', 'torch': 'SolverTorch'}[name])


# ------------------------------------------------------------------------------- the enum

def test_the_members_are_their_names():
    assert [m.value for m in Backend] == NAMES
    assert [m.name.lower() for m in Backend] == NAMES
    for member in Backend:
        assert member == member.value and isinstance(member, str)
        assert backend_of(member) is member and backend_of(member.value) is member


def test_the_differentiable_table_answers_to_a_name():
    assert DIFFERENTIABLE_BACKENDS == {Backend.JAX, Backend.TORCH}
    assert 'jax' in DIFFERENTIABLE_BACKENDS and 'torch' in DIFFERENTIABLE_BACKENDS
    assert 'numpy' not in DIFFERENTIABLE_BACKENDS


@pytest.mark.parametrize('value', REFUSED, ids=repr)
def test_the_parser_refuses_every_other_spelling_and_lists_the_names(value):
    with pytest.raises(ValueError, match=LISTED):
        backend_of(value)


# ------------------------------------------------------------------------------- rcwa

@pytest.mark.parametrize('name', NAMES)
def test_rcwa_takes_the_name_by_keyword_and_by_position_and_reports_the_member(name):
    pytest.importorskip(name)
    for solver in (_legacy_wavebit.rcwa(backend=name, **SCENE), _legacy_wavebit.rcwa(name, **SCENE),
                   _legacy_wavebit.rcwa(backend=Backend(name), **SCENE)):
        assert solver.backend is Backend(name)
        assert solver.backend == name
        assert type(solver) is _class_of(name)


def test_the_default_is_numpy():
    assert _legacy_wavebit.rcwa(**SCENE).backend is Backend.NUMPY


@pytest.mark.parametrize('value', REFUSED, ids=repr)
def test_rcwa_refuses_every_other_spelling_by_keyword_and_by_position(value):
    with pytest.raises(ValueError, match=LISTED):
        _legacy_wavebit.rcwa(backend=value, **SCENE)
    with pytest.raises(ValueError, match=LISTED):
        _legacy_wavebit.rcwa(value, **SCENE)


@pytest.mark.parametrize('name', NAMES)
def test_the_coherent_door_parses_the_backend_too(name):
    pytest.importorskip(name)
    facade = _legacy_wavebit.rcwa(backend=name, excitation={'top': _legacy_wavebit.PlaneWave()}, **SCENE)
    assert isinstance(facade, _legacy_wavebit.CoherentSolver)
    assert facade.backend is Backend(name)
    with pytest.raises(ValueError, match=LISTED):
        _legacy_wavebit.rcwa(backend=1, excitation={'top': _legacy_wavebit.PlaneWave()}, **SCENE)


# ------------------------------------------------------------------------------- the classes

@pytest.mark.parametrize('name', NAMES)
def test_a_solver_class_takes_its_own_name_and_refuses_another(name):
    """The keyword exists for the JAX pytree, which hands ``backend`` back on every jitted call.
    It is parsed like the other doors, and a name the class does not compute in is refused
    rather than reported by ``solver.backend`` for a solve that never ran there."""
    pytest.importorskip(name)
    cls = _class_of(name)
    assert cls._BACKEND_ID is Backend(name)
    assert cls(**SCENE).backend is Backend(name)
    assert cls(backend=name, **SCENE).backend is Backend(name)
    for other in set(NAMES) - {name}:
        with pytest.raises(ValueError, match='names another backend'):
            cls(backend=other, **SCENE)
    with pytest.raises(ValueError, match=LISTED):
        cls(backend=NAMES.index(name), **SCENE)


def test_assigning_the_attribute_is_the_same_door():
    """``solver.backend = ...`` on a built solver is parsed and held to the class like the
    keyword: it once stored ``1`` or ``'torch'`` as given, so a NumPy solver reported
    ``'torch'``, and on a JAX solver the integer failed later, inside ``tree_unflatten``."""
    solver = _legacy_wavebit.rcwa(**SCENE)
    with pytest.raises(ValueError, match=LISTED):
        solver.backend = 1
    with pytest.raises(ValueError, match='names another backend'):
        solver.backend = 'torch'
    assert solver.backend is Backend.NUMPY              # a refused assignment stores nothing
    solver.backend = 'numpy'
    assert solver.backend is Backend.NUMPY


def test_the_member_survives_the_jax_pytree():
    jax = pytest.importorskip('jax')
    solver = _legacy_wavebit.rcwa(backend='jax', **SCENE)
    leaves, treedef = jax.tree_util.tree_flatten(solver)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.backend is Backend.JAX
    # the treedef is the jit cache key: two flattenings of one solver must share it
    assert jax.tree_util.tree_flatten(rebuilt)[1] == treedef


# ------------------------------------------------------------------------------- meshgrid

@pytest.mark.parametrize('name', NAMES)
def test_the_meshgrid_helpers_take_the_name(name):
    pytest.importorskip(name)
    lattice = wavebit.Lattice.hexagonal(400.0)
    reference = [np.asarray(c) for c in _legacy_wavebit.meshgrid_xy((3, 5), lattice=lattice, backend=name)]
    for got in (lattice.meshgrid_xy((3, 5), backend=name),
                lattice.meshgrid_xy((3, 5), backend=Backend(name))):
        assert [np.asarray(c).tobytes() for c in got] == [c.tobytes() for c in reference]
    # computed in that backend's namespace: the same points as NumPy's, to round-off
    numpy = [np.asarray(c) for c in _legacy_wavebit.meshgrid_xy((3, 5), lattice=lattice)]
    assert all(np.allclose(a, b, rtol=0, atol=1e-12) for a, b in zip(reference, numpy))
    uv = lattice.meshgrid_uv((3, 5), backend=name)
    assert all(np.allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-15)
               for a, b in zip(uv, lattice.meshgrid_uv((3, 5))))


@pytest.mark.parametrize('value', [1, 'JAX', 'pytorch'], ids=repr)
def test_the_meshgrid_helpers_refuse_every_other_spelling(value):
    lattice = wavebit.Lattice.hexagonal(400.0)
    for call in (lambda: _legacy_wavebit.meshgrid_xy((3, 5), lattice=lattice, backend=value),
                 lambda: lattice.meshgrid_xy((3, 5), backend=value),
                 lambda: lattice.meshgrid_uv((3, 5), backend=value)):
        with pytest.raises(ValueError, match=LISTED):
            call()


# ------------------------------------------------------------------------------- the spec

@pytest.mark.parametrize('name', NAMES)
def test_the_spec_stores_the_name_and_hands_the_solver_the_name(name):
    for spelling in (name, Backend(name)):
        numerics = _legacy_wavebit.Numerics(backend=spelling)
        assert numerics.backend is Backend(name)
        kwargs = numerics.as_kwargs()
        assert kwargs['backend'] == name and type(kwargs['backend']) is str
        written = json.loads(json.dumps(numerics.to_dict()))['backend']
        assert written == name
        assert _legacy_wavebit.Numerics.from_dict({'backend': name}) == numerics


@pytest.mark.parametrize('value', [0, 1, 2, True, 'JAX', 'pytorch'], ids=repr)
def test_the_spec_refuses_every_other_spelling_on_every_door(value):
    with pytest.raises(_legacy_wavebit.SpecError, match=r"numerics\.backend: .* Expected one of "
                                                r"\['numpy', 'jax', 'torch'\]") as direct:
        _legacy_wavebit.Numerics(backend=value)
    with pytest.raises(_legacy_wavebit.SpecError) as parsed:
        _legacy_wavebit.Numerics.from_dict({'backend': value})
    assigned = _legacy_wavebit.Numerics(backend='jax')
    with pytest.raises(_legacy_wavebit.SpecError) as later:
        assigned.backend = value
    assert assigned.backend is Backend.JAX              # a refused assignment stores nothing
    assert str(direct.value) == str(parsed.value) == str(later.value)


# ------------------------------------------------------------------------------- the manifest

def test_the_manifest_publishes_the_names():
    from wavebit.manifest import manifest
    document = manifest()
    assert document['enums']['backend'] == NAMES
    assert list(document['backends']) == NAMES
