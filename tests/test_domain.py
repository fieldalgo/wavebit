"""Typed domain objects (wavebit.Layer / Stack / Incidence) are a non-breaking, adapt-in layer.

A ``Stack`` round-trips losslessly to and from the ``(structure, thickness)`` pair, and
driving :func:`wavebit.rcwa` through ``stack=`` / ``incidence=`` must give byte-identical
results to passing the equivalent raw keyword arguments (the Phase 2 "API parity" gate).
"""
import dataclasses

import numpy as np
import pytest

import wavebit
from wavebit import Stack, Incidence, Material, Pol
from wavebit._core.domain import (Layer, RETIRED_MATERIAL_NAMES, _VALUE_CHANGED_TOO,
                                  retired_material_name)


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


def test_stack_roundtrips_a_block_structure():
    # block encoding: one [background, [objects...]] entry per layer -> stays a list
    blocks = [
        [4.0, [['rectangle', 50, 150, 31, 300, 12.96, 0, 0, 0]]],
        [12.96, []],
    ]
    stack = Stack.from_structure(blocks, [205, 100000])
    assert stack.structure == blocks                       # list preserved verbatim
    assert not stack[0].is_raster


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
                             tetm=Pol.TE, max_order=[6, 0], period=period, wavelength=0.777,
                             thickness=thickness, structure=structure).solve()

    stack = Stack.from_structure(structure, thickness, top=4.0, bot=1.0, period=period)
    incidence = Incidence(wavelength=0.777, theta=0.0, phi=None, tetm=Pol.TE)
    typed = wavebit.rcwa(backend=0, max_order=[6, 0], stack=stack, incidence=incidence).solve()

    assert _de(typed) == _de(legacy)


def test_raster_2d_parity():
    rng = np.random.default_rng(0)
    structure = (rng.uniform(1.0, 4.0, size=(2, 8, 8))).astype(np.float64)
    thickness, period = [0.2, 0.3], [1.0, 1.0]

    legacy = wavebit.rcwa(backend=0, top=1.0, bot=2.25, theta=0.1, phi=0.2,
                             psi=np.pi / 4, max_order=[3, 3], period=period, wavelength=0.9,
                             thickness=thickness, structure=structure).solve()

    stack = Stack.from_structure(structure, thickness, top=1.0, bot=2.25, period=period)
    incidence = Incidence(wavelength=0.9, theta=0.1, phi=0.2, psi=np.pi / 4)
    typed = wavebit.rcwa(backend=0, max_order=[3, 3], stack=stack, incidence=incidence).solve()

    assert _de(typed) == _de(legacy)


def test_conical_psi_parity():
    structure = np.array([[[9.0, 9.0, 9.0, 1.0, 1.0, 1.0]]])
    thickness, period = [0.15], [0.77]

    legacy = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=0.3, phi=0.0,
                             psi=0.4, max_order=[6, 0], period=period, wavelength=0.777,
                             thickness=thickness, structure=structure).solve()

    stack = Stack.from_structure(structure, thickness, period=period)
    incidence = Incidence(wavelength=0.777, theta=0.3, phi=0.0, psi=0.4)
    typed = wavebit.rcwa(backend=0, max_order=[6, 0], stack=stack, incidence=incidence).solve()

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
    kw = dict(backend=0, max_order=[6, 0], period=[0.77], wavelength=0.777,
              thickness=[0.15], structure=structure, theta=0.0, phi=None, tetm=Pol.TE)

    plain = wavebit.rcwa(top=4.0, bot=1.0, **kw).solve()
    typed = wavebit.rcwa(top=Material(4.0), bot=Material(1.0), **kw).solve()
    assert _de(typed) == _de(plain)

    # and through a Stack, which passes the value on unread rather than coercing it
    stack = Stack.from_structure(structure, [0.15], top=Material(4.0), bot=Material(1.0),
                                 period=[0.77])
    assert stack.as_kwargs()['top'] == Material(4.0)
    through = wavebit.rcwa(backend=0, max_order=[6, 0], stack=stack,
                           incidence=Incidence(wavelength=0.777, theta=0.0, phi=None,
                                               tetm=Pol.TE)).solve()
    assert _de(through) == _de(plain)


# --------------------------------------------------------------- retired spellings

def test_the_retired_layer_key_is_in_the_one_table():
    """``eps`` -> ``structure`` belongs in ``RETIRED_MATERIAL_NAMES`` like every other rename.

    The table exists so that the constructors, the analytic path's key funnel and
    ``_core/spec.py`` cannot drift into saying different things about one rename. ``eps`` was
    missing from it, which is not a decision about ``eps`` -- it made
    ``retired_material_name('eps')`` a ``KeyError``, so the spec layer's refusal could not use
    the shared text even if it asked for it.

    It names a *structure* rather than a material, and it is still here for the reason the
    table's own comment gives: the raster it held was a grid of refractive indices, so the
    rename is only half of the migration.
    """
    assert 'eps' in RETIRED_MATERIAL_NAMES
    new, _what = RETIRED_MATERIAL_NAMES['eps']
    assert new == 'structure'
    # the replacement has to be a name that exists, or the advice sends a reader nowhere
    assert new in {f.name for f in dataclasses.fields(Layer)}

    message = retired_material_name('eps', 'stack.layers[0]')
    assert "'eps' is retired" in message and "'structure'" in message
    assert 'PERMITTIVITY' in message          # the half a plain rename message would omit
    assert '12.25' in message                 # n = 3.5 spelled out, both directions named
    assert 'eps_from_index' in message and 'index_from_eps' in message


def test_the_solver_refuses_the_retired_layer_keyword_by_name():
    """Adding it to the table is what makes every reader of the table refuse it."""
    with pytest.raises(TypeError, match='PERMITTIVITY'):
        wavebit.rcwa(backend=0, max_order=[2, 0], eps=np.array([[[9.0, 1.0]]]), thickness=[0.1])


def test_the_two_figures_the_refusal_quotes_are_the_readme_scene_measured():
    """Every retired spelling ends in a measurement; solve it and check the message.

    ``_VALUE_CHANGED_TOO`` argues its case with two numbers -- the README quickstart migrated
    properly, and the same scene with each material literal left at the refractive index it
    used to be. For a while it quoted a pair (0.0635 / 0.9814) that no variant of that scene
    produces; they came from a different structure and were copied along. Nothing failed,
    because a docstring's arithmetic has no reader but a person.

    Which makes it worth a test rather than a re-check: the sentence's whole job is to convince
    someone that the wrong reading is *plausible*, and it argues that with a figure. A reader
    who reproduces the scene and gets a third answer learns the opposite of what the sentence
    is for. Pinning the two to four decimals also keeps them honest against the message text
    itself, so the prose and the solve cannot drift apart in either direction.

    The numbers belong to the quickstart, not to the convention: if that scene changes,
    re-measure and update both copies (here, and ``docs/conventions.md``).
    """
    def quickstart(eps_layer, eps_substrate):
        res = wavebit.rcwa(
            backend=0, wavelength=900., period=[700.], max_order=[10, 0],
            top=1., bot=eps_substrate, theta=0., phi=0., tetm='te',
            structure=np.array([[[1.0, 1.0, eps_layer, eps_layer]]]),
            thickness=[300.]).solve()
        return float(np.sum(res.R)), float(np.sum(res.T))

    migrated, migrated_t = quickstart(wavebit.eps_from_index(3.5), wavebit.eps_from_index(1.5))
    literals_kept, kept_t = quickstart(3.5, 1.5)

    assert migrated == pytest.approx(0.20533, abs=5e-6)
    assert literals_kept == pytest.approx(0.17530, abs=5e-6)
    assert f'{migrated:.4f}' in _VALUE_CHANGED_TOO
    assert f'{literals_kept:.4f}' in _VALUE_CHANGED_TOO

    # And the reason the pair has to be quoted at all: the wrong reading is a legal lossless
    # grating, so it conserves energy exactly as the right one does. Nothing downstream can
    # separate them, which is what leaves the refusal as the only place it can be said.
    assert migrated + migrated_t == pytest.approx(1.0, abs=1e-12)
    assert literals_kept + kept_t == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------- conflicts

def test_double_specify_raises():
    structure = np.array([[[9.0, 1.0]]])
    stack = Stack.from_structure(structure, [0.1])
    with pytest.raises(ValueError, match='stack'):
        wavebit.rcwa(backend=0, stack=stack, structure=structure, max_order=[2, 0])
    with pytest.raises(ValueError, match='incidence'):
        wavebit.rcwa(backend=0, incidence=Incidence(theta=0.1), theta=0.2, max_order=[2, 0])
