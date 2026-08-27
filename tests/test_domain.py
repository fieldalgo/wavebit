"""Typed domain objects (wavebit.Layer / Stack / Incidence) are a non-breaking, adapt-in layer.

A ``Stack`` round-trips losslessly to and from the ``(structure, thickness)`` pair, and
driving :func:`wavebit.rcwa` through ``stack=`` / ``incidence=`` must give byte-identical
results to passing the equivalent raw keyword arguments (the Phase 2 "API parity" gate).
"""

import numpy as np
import pytest

import wavebit
from wavebit import Stack, Incidence, Material, Tetm


# ---------------------------------------------------------------- round-trip

def test_stack_roundtrips_a_raster_structure():
    raster = np.arange(2 * 1 * 6, dtype=np.float64).reshape(2, 1, 6) + 1.0
    thickness = [0.15, 0.30]

    stack = Stack.from_structure(raster, thickness, top=4.0, bot=1.0, period=[0.77])

    assert len(stack) == 2
    assert stack.thickness == thickness
    assert isinstance(stack.structure, np.ndarray)
    assert np.array_equal(stack.structure, raster)         # exact rebuild
    assert stack.structure.dtype == raster.dtype
    # each layer is (thickness[i], structure[i]) -- the field is `structure`, not `eps`:
    # only the raster encoding is one quantity, and even there it is a sampled field
    # rather than a single value.
    for i, layer in enumerate(stack):
        assert layer.thickness == thickness[i]
        assert np.array_equal(layer.structure, raster[i])
        assert layer.is_raster


def test_stack_roundtrips_an_analytic_structure():
    layers = [{'background': 12.96, 'shapes': []}, {'background': 1.0, 'shapes': []}]
    stack = Stack.from_structure(layers, [0.1, 0.2])
    assert stack.structure == layers
    assert not stack[0].is_raster


def test_empty_stack_is_none():
    assert Stack([]).structure is None
    assert Stack.from_structure(None, []).structure is None


# ------------------------------------------------------------- solver parity

def _de(res):
    return float(np.array(res.R).sum()), float(np.array(res.T).sum())


def test_raster_1d_parity():
    structure = np.array([[[9.0, 9.0, 9.0, 1.0, 1.0, 1.0]]])
    thickness, period = [0.15], [0.77]

    legacy = wavebit.rcwa(backend=0, top=4.0, bot=1.0, theta=0.0, phi=None,
                             tetm=Tetm.TE, truncation_order=[6, 0], period=period, wavelength=0.777,
                             thickness=thickness, structure=structure).solve()

    stack = Stack.from_structure(structure, thickness, top=4.0, bot=1.0, period=period)
    incidence = Incidence(wavelength=0.777, theta=0.0, phi=None, tetm=Tetm.TE)
    typed = wavebit.rcwa(backend=0, truncation_order=[6, 0], stack=stack, incidence=incidence).solve()

    assert _de(typed) == _de(legacy)


def test_raster_2d_parity():
    rng = np.random.default_rng(0)
    structure = (rng.uniform(1.0, 4.0, size=(2, 8, 8))).astype(np.float64)
    thickness, period = [0.2, 0.3], [1.0, 1.0]

    legacy = wavebit.rcwa(backend=0, top=1.0, bot=2.25, theta=0.1, phi=0.2,
                             psi=np.pi / 4, truncation_order=[3, 3], period=period, wavelength=0.9,
                             thickness=thickness, structure=structure).solve()

    stack = Stack.from_structure(structure, thickness, top=1.0, bot=2.25, period=period)
    incidence = Incidence(wavelength=0.9, theta=0.1, phi=0.2, psi=np.pi / 4)
    typed = wavebit.rcwa(backend=0, truncation_order=[3, 3], stack=stack, incidence=incidence).solve()

    assert _de(typed) == _de(legacy)


def test_conical_psi_parity():
    structure = np.array([[[9.0, 9.0, 9.0, 1.0, 1.0, 1.0]]])
    thickness, period = [0.15], [0.77]

    legacy = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=0.3, phi=0.0,
                             psi=0.4, truncation_order=[6, 0], period=period, wavelength=0.777,
                             thickness=thickness, structure=structure).solve()

    stack = Stack.from_structure(structure, thickness, period=period)
    incidence = Incidence(wavelength=0.777, theta=0.3, phi=0.0, psi=0.4)
    typed = wavebit.rcwa(backend=0, truncation_order=[6, 0], stack=stack, incidence=incidence).solve()

    assert _de(typed) == _de(legacy)


# --------------------------------------------------------------- material half-spaces

def test_material_half_spaces_are_the_same_physics_as_the_shorthand():
    """``Stack.top`` takes a Material or a number, and the two must be one input.

    :func:`wavebit._core.domain.as_material` is the single coercion point, so the object and
    the bare number reach the solver as the same medium. The parity here is byte-level for the
    same reason the rest of this file's parity is: the typed layer is a way to *say* a scene,
    never a second way to compute one.
    """
    structure = np.array([[[9.0, 9.0, 9.0, 1.0, 1.0, 1.0]]])
    kw = dict(backend=0, truncation_order=[6, 0], period=[0.77], wavelength=0.777,
              thickness=[0.15], structure=structure, theta=0.0, phi=None, tetm=Tetm.TE)

    plain = wavebit.rcwa(top=4.0, bot=1.0, **kw).solve()
    typed = wavebit.rcwa(top=Material(4.0), bot=Material(1.0), **kw).solve()
    assert _de(typed) == _de(plain)

    # and through a Stack, which passes the value on unread rather than coercing it
    stack = Stack.from_structure(structure, [0.15], top=Material(4.0), bot=Material(1.0),
                                 period=[0.77])
    assert stack.as_kwargs()['top'] == Material(4.0)
    through = wavebit.rcwa(backend=0, truncation_order=[6, 0], stack=stack,
                           incidence=Incidence(wavelength=0.777, theta=0.0, phi=None,
                                               tetm=Tetm.TE)).solve()
    assert _de(through) == _de(plain)


@pytest.mark.parametrize('slot', ['top', 'bot'])
@pytest.mark.parametrize('name, value', [
    ('a raster', np.full((1, 8), 2.25)),
    ('a tensor', np.diag([2.25, 2.25, 3.0])),
    ('a mu raster', Material(2.25, mu=np.full((2, 2), 1.5))),
])
def test_a_half_space_that_is_not_one_material_is_refused(slot, name, value):
    """A half-space is semi-infinite and homogeneous, so its numbers are single values.

    Refused at the half-space slot rather than in :func:`as_material`, which is deliberately
    permissive -- it cannot tell a vmapped tracer from a raster grid
    (``test_material.py::test_arrays_and_tracers_are_not_refused``). *Half-space* is a
    stronger claim than *material*, so this slot can be precise where that one cannot.

    Until 2026-08-22 a shaped half-space travelled as far as ``_separable_kxy`` and raised
    ``operands could not be broadcast together with shapes (1, 8) and (5,)`` -- loud, but
    naming neither the argument at fault nor what a half-space is. The `(5,)` was the order
    count, which no caller has any reason to recognise.
    """
    kw = dict(backend=0, truncation_order=[2, 0], period=[0.77], wavelength=0.777, thickness=[0.15],
              theta=0.0, phi=None, tetm=Tetm.TE, **{slot: value})
    kw.setdefault('top' if slot == 'bot' else 'bot', 1.0)
    with pytest.raises(ValueError) as excinfo:
        wavebit.rcwa(**kw)
    message = str(excinfo.value)
    assert f'`{slot}`' in message and 'half-space' in message
    assert 'layer' in message and '`structure`' in message      # says where it does belong


@pytest.mark.parametrize('value', [2.25, 2.25 + 0.3j, np.asarray(2.25), np.asarray([2.25]),
                                   Material(2.25, mu=1.5)])
def test_one_value_in_any_spelling_is_still_one_value(value):
    """The refusal is a *size*, not a type: a 0-d or one-element array is a scalar.

    Which is also what keeps it safe under trace. The test never reads the value, only the
    shape -- and a shape is static under ``jit``, ``grad`` and ``vmap`` alike, with a vmapped
    scalar reading ``()`` inside the trace. ``test_backend_jax.py`` runs the traced solves
    that would fail if this had been a value comparison.
    """
    wavebit.rcwa(backend=0, truncation_order=[2, 0], period=[0.77], wavelength=0.777,
                 thickness=[0.15], theta=0.0, phi=None, tetm=Tetm.TE, top=value, bot=1.0)


# ------------------------------------------------- the spellings that used to be named

# `ucell`, `eps`, `n_top`, `n_bot`, `n_base`, `base`, `n_index`, `pol` and `ucell_materials`
# were refused **by name** until 2026-08-26, each with the sentence that its *quantity* had
# changed too -- `ucell` held refractive indices and `structure` holds permittivity, so a
# script re-keyed and left at its old numbers is a legal, energy-conserving, different scene.
#
# That machinery is deleted. wavebit has never been released, so the reader those messages
# addressed does not exist yet, and the renames belong in the first release's notes. What has
# to survive is the refusal itself: a name nothing reads must not be accepted.

@pytest.mark.parametrize('keyword', [
    dict(pol='te'),
    dict(ucell_materials=[]),
    dict(ucell=np.array([[[9.0, 1.0]]])),
    dict(n_top=1.5),
    dict(eps=np.array([[[9.0, 1.0]]])),
])
def test_a_pre_rename_constructor_keyword_is_refused_as_unknown(keyword):
    """Plainly now, and the plainness is the whole change -- but still refused.

    Silently accepting one is the failure this guards; naming its replacement was help that
    had no audience yet.
    """
    with pytest.raises(TypeError, match=r'unexpected keyword argument'):
        wavebit.rcwa(backend=0, truncation_order=[2, 0], thickness=[0.1],
                     structure=np.array([[[9.0, 1.0]]]), **keyword)


def test_a_typo_in_a_live_keyword_is_not_swallowed_by_the_catch_all():
    """``**unknown`` collects everything, so the refusal has to raise the ordinary case.

    This is the half of `_refuse_retired` that survived the deletion, under the name
    `_refuse_unknown_keywords`: without it the catch-all would swallow a misspelled live
    keyword, which is worse than the Python error it replaced.
    """
    with pytest.raises(TypeError, match=r"unexpected keyword argument\(s\) \['wavelenght'\]"):
        wavebit.rcwa(backend=0, truncation_order=[2, 0], thickness=[0.1],
                     structure=np.array([[[9.0, 1.0]]]), wavelenght=700.0)


def test_the_derived_index_accessors_survived_the_deletion():
    """`n_top` / `n_bot` are **read-only derived properties**, not retired spellings.

    Only their setters were the retired half. The getters return `sqrt(eps*mu)` and every solve
    reads them -- `n_inc` and the four solve dispatch sites -- so deleting all four defs, which
    is what "four retired properties" reads like, takes an ordinary solve down with
    `AttributeError: no attribute 'n_bot'`.

    The input is permittivity and this is derived from it; both are true at once.
    """
    solver = wavebit.rcwa(backend=0, truncation_order=[2, 0], thickness=[0.1],
                          structure=np.array([[[9.0, 1.0]]]), top=Material(4.0), bot=2.25)
    assert complex(solver.n_top) == pytest.approx(2.0)      # sqrt(4.0)
    assert complex(solver.n_bot) == pytest.approx(1.5)      # sqrt(2.25)
    with pytest.raises(AttributeError):
        solver.n_top = 1.5                                   # derived, so not settable
    assert np.isfinite(np.asarray(solver.solve().T).sum())


# --------------------------------------------------------------- conflicts

def test_double_specify_raises():
    structure = np.array([[[9.0, 1.0]]])
    stack = Stack.from_structure(structure, [0.1])
    with pytest.raises(ValueError, match='stack'):
        wavebit.rcwa(backend=0, stack=stack, structure=structure, truncation_order=[2, 0])
    with pytest.raises(ValueError, match='incidence'):
        wavebit.rcwa(backend=0, incidence=Incidence(theta=0.1), theta=0.2, truncation_order=[2, 0])
