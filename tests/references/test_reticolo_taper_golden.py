"""Tapered line gratings against RETICOLO, from ``golden_reticolo_taper.npz``.

``rcwa.ProfiledLayer`` turns a :class:`~wavebit.rcwa.Trapezoid` into a staircase of rectangles.
Two claims are checked against RETICOLO V10, whose numbers come from
``gen_goldens_reticolo_taper.py`` on a staircase built without the solver's slicer:

* **The same staircase is the same answer.** At the uniform slice counts in ``SAME_STAIRCASE``
  RETICOLO is handed each slice as an exact 1D texture, so the two codes build the same
  convolution matrices and agree to round-off -- 1e-10 is the pre-registered bound, and
  3.5e-14 the worst measured (2026-11-18). That covers the solve and the geometry at once:
  the slices' widths are asserted separately against the cases module, to 1e-15.
* **The automatic count reaches the profile.** RETICOLO's 64, 128 and 256 uniform slices are
  extrapolated to the limit (L. F. Richardson, Phil. Trans. R. Soc. A 210, 307 (1911)), and
  the solver's ``Slicing(count='auto')`` answer must lie within the slicing ``tolerance`` of it.
  Measured: 2.9e-5 to 7.3e-5 against a tolerance of 1e-4.

The solver runs a line on its 2D path with ``orders=(15, 0)`` and Li's factorization, which is
exact for the full-height rectangles a line's slices are; RETICOLO runs its 1D TE/TM path. A
y-invariant cell makes the two the same formulation. Efficiency is invariant under a lateral
shift, so RETICOLO's texture origin at the cell's edge needs no correction.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path

import math

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.rcwa._profile import expand, plan_fractions
from tests.references._reticolo_taper_cases import (EPS_LINE, EPS_SUBSTRATE, LADDER, LIGHT, ORDERS, PERIOD,
                                                    PROFILES, SAME_STAIRCASE, WAVELENGTH, edge, key, staircase)

GOLDEN = np.load(repository_path('tests/references/golden_reticolo_taper.npz'))


def _problem(name, theta=0., pol='te'):
    cd, swa, top, foot, height = PROFILES[name]
    line = rcwa.Trapezoid(cd=cd, swa=swa, top_rounding=top, footing=foot)
    layer = rcwa.ProfiledLayer(thickness=height, regions=(rcwa.Region(geometry=line, material=Material(epsilon=EPS_LINE)),))
    return rcwa.Problem(units=Units(length='um'),
                        stack=rcwa.Stack(period=(PERIOD, PERIOD), bottom=Material(epsilon=EPS_SUBSTRATE), layers=(layer,)),
                        excitation=rcwa.Excitation(wavelength=WAVELENGTH, sources=(
                            rcwa.PlaneWave(theta=theta, phi=0., polarization='s' if pol == 'te' else 'p'),)))


def _orders(result, which, orders):
    return np.asarray(getattr(result, which))[0, 0][np.asarray(orders) + ORDERS]


def _efficiencies(result, k):
    return np.concatenate([_orders(result, 'R', GOLDEN[k + '__R_orders']),
                           _orders(result, 'T', GOLDEN[k + '__T_orders'])])


def _golden(k):
    return np.concatenate([GOLDEN[k + '__R'], GOLDEN[k + '__T']])


@pytest.mark.parametrize('name', PROFILES)
@pytest.mark.parametrize('count', SAME_STAIRCASE)
def test_the_slices_are_the_independently_built_staircase(name, count):
    problem = _problem(name)
    fractions = (plan_fractions(problem, 0, count, 'uniform'),)
    sliced, _ = expand(problem, fractions, (False,))
    for layer, (thickness, left, right) in zip(sliced.stack.layers, staircase(name, count)):
        rectangle = layer.regions[0].geometry
        assert abs(layer.thickness - thickness) < 1e-15
        assert abs(rectangle.size[0] - (right - left)) < 1e-15
        assert abs(rectangle.center[0] - (right + left)/2) < 1e-15
        assert rectangle.size[1] == PERIOD


@pytest.mark.parametrize('name', PROFILES)
@pytest.mark.parametrize('theta, pol', LIGHT)
@pytest.mark.parametrize('count', SAME_STAIRCASE)
def test_the_same_staircase_agrees_with_reticolo(name, theta, pol, count):
    numerics = rcwa.Numerics(orders=(ORDERS, 0), factorization='li',
                             slicing=rcwa.Slicing(count=count, rule='uniform', estimate=None))
    k = key(name, count, theta, pol)
    result = rcwa.run(_problem(name, theta, pol), numerics=numerics)
    assert np.max(np.abs(_efficiencies(result, k) - _golden(k))) < 1e-10


@pytest.mark.slow
@pytest.mark.parametrize('name', PROFILES)
@pytest.mark.parametrize('theta, pol', LIGHT)
def test_the_automatic_count_lands_within_tolerance_of_the_converged_profile(name, theta, pol):
    ladder = [_golden(key(name, count, theta, pol)) for count in LADDER]
    observed = math.log2(np.max(np.abs(ladder[1] - ladder[0]))/np.max(np.abs(ladder[2] - ladder[1])))
    limit = ladder[2] + (ladder[2] - ladder[1])/(2**min(observed, 2.) - 1)
    slicing = rcwa.Slicing(max_count=256)
    result = rcwa.run(_problem(name, theta, pol),
                      numerics=rcwa.Numerics(orders=(ORDERS, 0), factorization='li', slicing=slicing))
    report = result.diagnostics['z_slicing'][0]
    assert report['converged'] and result.validity.checks['z_slicing_converged']
    assert np.max(np.abs(_efficiencies(result, key(name, LADDER[0], theta, pol)) - limit)) < slicing.tolerance


@pytest.mark.slow
def test_the_default_factorization_solves_the_same_staircase():
    """NVF's normal field of a full-height rectangle is the x axis, where NVF is Li's rule."""
    count, (theta, pol) = SAME_STAIRCASE[0], LIGHT[3]
    numerics = rcwa.Numerics(orders=(ORDERS, 0), slicing=rcwa.Slicing(count=count, rule='uniform', estimate=None))
    k = key('rounded', count, theta, pol)
    result = rcwa.run(_problem('rounded', theta, pol), numerics=numerics)
    assert np.max(np.abs(_efficiencies(result, k) - _golden(k))) < 1e-10


def test_the_cases_module_builds_its_outline_from_the_corners():
    """The independent geometry meets the straight wall where each arc ends: C1, as drawn."""
    for name in PROFILES:
        height = PROFILES[name][4]
        z = np.linspace(0., height, 4001)
        for side in (-1, 1):
            x = np.array([edge(name, v, side) for v in z])
            assert np.max(np.abs(np.diff(x))) < 0.01*height
