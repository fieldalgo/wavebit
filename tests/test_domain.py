"""Typed domain objects (fieldwave.Layer / Stack / Incidence) are a non-breaking, adapt-in layer.

A ``Stack`` round-trips losslessly to and from the legacy ``(ucell, thickness)`` pair, and
driving :func:`fieldwave.rcwa` through ``stack=`` / ``incidence=`` must give byte-identical
results to passing the equivalent raw keyword arguments (the Phase 2 "API parity" gate).
"""
import numpy as np
import pytest

import fieldwave
from fieldwave import Layer, Stack, Incidence, Pol


# ---------------------------------------------------------------- round-trip

def test_stack_roundtrips_raster_ucell():
    ucell = np.arange(2 * 1 * 6, dtype=np.float64).reshape(2, 1, 6) + 1.0
    thickness = [0.15, 0.30]

    stack = Stack.from_ucell(ucell, thickness, n_top=2.0, n_bot=1.0, period=[0.77])

    assert len(stack) == 2
    assert stack.thickness == thickness
    assert isinstance(stack.ucell, np.ndarray)
    assert np.array_equal(stack.ucell, ucell)          # exact rebuild
    assert stack.ucell.dtype == ucell.dtype
    # each layer is (thickness[i], ucell[i])
    for i, layer in enumerate(stack):
        assert layer.thickness == thickness[i]
        assert np.array_equal(layer.eps, ucell[i])
        assert layer.is_raster


def test_stack_roundtrips_vector_ucell():
    # vector encoding: one [bg_material, [objects...]] entry per layer -> stays a list
    ucell = [
        [2.0, [['rectangle', 50, 150, 31, 300, 3.6, 0, 0, 0]]],
        [3.6, []],
    ]
    stack = Stack.from_ucell(ucell, [205, 100000])
    assert stack.ucell == ucell                        # list preserved verbatim
    assert not stack[0].is_raster


def test_stack_roundtrips_analytic_ucell():
    ucell = [{'material': 3.6, 'shapes': []}, {'material': 1.0, 'shapes': []}]
    stack = Stack.from_ucell(ucell, [0.1, 0.2])
    assert stack.ucell == ucell
    assert not stack[0].is_raster


def test_empty_stack_is_none():
    assert Stack([]).ucell is None
    assert Stack.from_ucell(None, []).ucell is None


# ------------------------------------------------------------- solver parity

def _de(res):
    return float(np.array(res.R).sum()), float(np.array(res.T).sum())


def test_raster_1d_parity():
    ucell = np.array([[[3.0, 3.0, 3.0, 1.0, 1.0, 1.0]]])
    thickness, period = [0.15], [0.77]

    legacy = fieldwave.rcwa(backend=0, n_top=2.0, n_bot=1.0, theta=0.0, phi=None,
                             pol=Pol.TE, fto=[6, 0], period=period, wavelength=0.777,
                             thickness=thickness, ucell=ucell).conv_solve()

    stack = Stack.from_ucell(ucell, thickness, n_top=2.0, n_bot=1.0, period=period)
    incidence = Incidence(wavelength=0.777, theta=0.0, phi=None, pol=Pol.TE)
    typed = fieldwave.rcwa(backend=0, fto=[6, 0], stack=stack, incidence=incidence).conv_solve()

    assert _de(typed) == _de(legacy)


def test_raster_2d_parity():
    rng = np.random.default_rng(0)
    ucell = (rng.uniform(1.0, 4.0, size=(2, 8, 8))).astype(np.float64)
    thickness, period = [0.2, 0.3], [1.0, 1.0]

    legacy = fieldwave.rcwa(backend=0, n_top=1.0, n_bot=1.5, theta=0.1, phi=0.2,
                             pol=0.5, fto=[3, 3], period=period, wavelength=0.9,
                             thickness=thickness, ucell=ucell).conv_solve()

    stack = Stack.from_ucell(ucell, thickness, n_top=1.0, n_bot=1.5, period=period)
    incidence = Incidence(wavelength=0.9, theta=0.1, phi=0.2, pol=0.5)
    typed = fieldwave.rcwa(backend=0, fto=[3, 3], stack=stack, incidence=incidence).conv_solve()

    assert _de(typed) == _de(legacy)


def test_conical_psi_parity():
    ucell = np.array([[[3.0, 3.0, 3.0, 1.0, 1.0, 1.0]]])
    thickness, period = [0.15], [0.77]

    legacy = fieldwave.rcwa(backend=0, n_top=1.0, n_bot=1.0, theta=0.3, phi=0.0,
                             psi=0.4, fto=[6, 0], period=period, wavelength=0.777,
                             thickness=thickness, ucell=ucell).conv_solve()

    stack = Stack.from_ucell(ucell, thickness, period=period)
    incidence = Incidence(wavelength=0.777, theta=0.3, phi=0.0, psi=0.4)
    typed = fieldwave.rcwa(backend=0, fto=[6, 0], stack=stack, incidence=incidence).conv_solve()

    assert _de(typed) == _de(legacy)


# --------------------------------------------------------------- conflicts

def test_double_specify_raises():
    ucell = np.array([[[3.0, 1.0]]])
    stack = Stack.from_ucell(ucell, [0.1])
    with pytest.raises(ValueError, match='stack'):
        fieldwave.rcwa(backend=0, stack=stack, ucell=ucell, fto=[2, 0])
    with pytest.raises(ValueError, match='incidence'):
        fieldwave.rcwa(backend=0, incidence=Incidence(theta=0.1), theta=0.2, fto=[2, 0])
