"""Flat multilayers with **lossless** double negatives, against PyMoosh 4.0.1.

The lossless double negative is the case fmmax cannot arbitrate (``test_fmmax_dng_golden.py``):
its modes are picked by ``Im(kz) > 0``, which a lossless medium cannot read. PyMoosh writes the
choice down -- the principal ``kz``, negated in a double-negative lit or last medium
(``_pymoosh_dng_cases.py`` quotes its ``coefficient_S``) -- and that is this solver's root for
every lossless medium: Maystre's cut for a positive one and its mirror for a double negative
(``conventions.boundary_kz``). The two codes share nothing else: PyMoosh composes 2x2 scattering
matrices of scalar TE / TM interfaces, this solver an RCWA stack with ``truncation_order = 0`` on
the TE / TM path, the coupled 2D path and the first-order path, ETM and SMM.

Each case puts a lossless double negative where a rule decides something: as the substrate (the
mirror root), as the lit superstrate (the incident wave's root and the signed index of ``t_p``),
as an inner layer (the layer's mode labels), and all three at once; each at four angles, the
last past some medium's critical angle; both polarizations; every ``|mu| != 1``.

The incidence is PyMoosh's in-plane wavevector, handed to this solver through its ``kx`` door:
PyMoosh's angle is the phase's, so from a double negative its ``kx`` has the sign opposite to the
one this solver's ``theta`` gives (the direction the light travels). A flat stack answers both
signs alike, so the ``theta`` door is checked at ``-theta`` only for agreement, not for the sign.

Measured when frozen (2026-10-04, PyMoosh 4.0.1 and numpy 2.5.3 in its environment; numpy 2.5.1
here): ``r``, ``t``, ``R`` and ``T`` agree to 8.8e-15 at worst over the 32 rows, three paths and
both algorithms; the bound is ``TOLERANCE``. Replacing each double negative by its
double-positive mirror ``(-eps, -mu)`` -- the same ``|n|`` and admittance magnitudes, the other
sign of index -- misses by 1.8 to 2.9 wherever a double negative lies beyond the lit side. A
lossless double negative *as the lit side* has exactly its mirror's admittances, so a flat stack
lit from it answers as lit from the mirror: there the golden pins the incident wave's root and
the sign of the index in ``t_p`` (either one wrong moves the rows by O(1)), not the medium's
sign.

**A PyMoosh limit found while building this.** Its inner layers take ``Im(kz) >= 0``, so a
lossless double-negative layer's propagating modes keep the principal root, whose power runs
against the label. Where that mislabelled admittance cancels a neighbour's, PyMoosh's interface
matrix divides by zero: ``(-6, -1.5)`` beside ``(4, 1)`` (impedance-matched: ``|eps/mu| = 4`` on
both) gave ``R + T = 3.10`` at normal incidence and ``1 + 2.6e-10`` at 1e-3 rad, where this
solver gives 1 to the bit (its labels follow the power, ``conventions.layer_label_rule``). The
cases avoid such pairs; the generator refuses any row with ``|R + T - 1| > 1e-12``.

Values are frozen by ``gen_goldens_pymoosh_dng.py`` in an environment of its own. Nothing here
imports PyMoosh. Never regenerate to make this pass.
"""
import pathlib
import sys
import warnings

import numpy as np
import pytest

import wavebit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _pymoosh_dng_cases import (CASES, PYMOOSH_VERSION, WL, media,   # noqa: E402
                                pymoosh_kx)

GOLDEN_PATH = HERE / 'golden_pymoosh_dng.npz'

#: Measured 8.8e-15 at worst (``r``, ``t``, ``R``, ``T``; every row, path and algorithm).
TOLERANCE = 1e-13

PATHS = ('tetm', '2d', 'first_order')


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with gen_goldens_pymoosh_dng.py')
    return np.load(GOLDEN_PATH)


def _signed_index(eps, mu):
    """``sqrt(eps)*sqrt(mu)``, each root principal -- written out here, not the solver's."""
    return np.sqrt(complex(eps)) * np.sqrt(complex(mu))


def _solve(name, index, path, algo, pol, door='k', stack=None):
    """``(r, t_H, R, T)`` of polarization ``pol`` (0 TE, 1 TM) at the case's ``index``-th angle.

    ``t_H`` is the transmitted ``H_y`` ratio for TM -- this solver's ``t_p`` is the ``E`` ratio,
    ``t_H * (n1/mu1)/(n2/mu2)`` -- so that both codes report the same quantities. ``stack``
    replaces the case's media (lit side first), for the controls."""
    case = CASES[name]
    stack = media(name) if stack is None else stack
    inner = stack[1:-1][::-1]                         # this solver lists layers bottom up
    thickness = [d for _, d in case['layers']][::-1]
    theta = case['thetas'][index]
    one_d = path == 'tetm'
    shape = (1, 2) if one_d else (2, 2)
    kw = dict(backend=0, top=wavebit.Material(*stack[0]), bot=wavebit.Material(*stack[-1]),
              wavelength=WL, thickness=thickness, layer_algorithm=algo, answers='amplitude',
              structure=np.stack([np.full(shape, e) for e, _ in inner]),
              mu=np.stack([np.full(shape, m) for _, m in inner]))
    if door == 'k':
        kw['kx'] = 2 * np.pi / WL * pymoosh_kx(name, theta)
        if not one_d:
            kw['ky'] = 0.0
    else:
        # This solver's angle names the light's direction: from a double negative it is the
        # negative of PyMoosh's for the same |kx| (``_pymoosh_dng_cases.py``).
        lit_negative = _signed_index(*stack[0]).real < 0
        kw.update(theta=-theta if lit_negative else theta, phi=0.0)
    if one_d:
        solver = wavebit.rcwa(**kw, period=[1000.0], truncation_order=0,
                              tetm='te' if pol == 0 else 'tm')
    else:
        solver = wavebit.rcwa(**kw, period=(1000.0, 1000.0), truncation_order=(0, 0),
                              polarization=(1.0, 0.0) if pol == 0 else (0.0, 1.0))
        if path == 'first_order':
            solver.formulation = 'first_order'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = solver.solve(auto_detune=False)
    sub = result.incident if one_d else (result.s_inc if pol == 0 else result.p_inc)
    r, t = (sub.r_s, sub.t_s) if pol == 0 else (sub.r_p, sub.t_p)
    r, t = (complex(np.sum(np.asarray(v))) for v in (r, t))
    if pol == 1:
        (e1, m1), (e2, m2) = stack[0], stack[-1]
        t /= (_signed_index(e1, m1) / m1) / (_signed_index(e2, m2) / m2)
    return np.asarray([r, t, float(np.sum(sub.R)), float(np.sum(sub.T))])


ROWS = [(name, i, pol) for name in CASES for i in range(len(CASES[name]['thetas']))
        for pol in (0, 1)]


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('name', list(CASES))
def test_every_row_is_pymooshs(golden, name, path, algo):
    """``r``, ``t``, ``R`` and ``T``, both polarizations, every angle, through the ``kx`` door."""
    for i in range(len(CASES[name]['thetas'])):
        for pol in (0, 1):
            want = golden[f'{name}__pol{pol}'][i]
            got = _solve(name, i, path, algo, pol)
            gap = np.abs(got - want).max()
            assert gap < TOLERANCE, (name, i, pol, path, algo, gap, got, want)


@pytest.mark.parametrize('name', [n for n in CASES if CASES[n]['top'][0] < 0])
def test_the_theta_door_from_a_double_negative_meets_the_same_rows(golden, name):
    """From a double negative this solver's ``theta`` is minus PyMoosh's for the same ``|kx|``;
    lit that way the rows are PyMoosh's again, on every path."""
    for path in PATHS:
        for i in range(len(CASES[name]['thetas'])):
            for pol in (0, 1):
                got = _solve(name, i, path, 'SMM', pol, door='theta')
                assert np.abs(got - golden[f'{name}__pol{pol}'][i]).max() < TOLERANCE


#: The cases with a double negative beyond the lit side. (A lossless lit double negative has its
#: mirror's admittances exactly, so a flat stack cannot tell the two apart -- see the module.)
BEYOND = [name for name in CASES
          if any(e.real < 0 and m.real < 0 for e, m in media(name)[1:])]


@pytest.mark.parametrize('name', BEYOND)
def test_the_golden_tells_a_double_negative_from_its_mirror(golden, name):
    """Agreement is a statement about the double negatives: each replaced by its
    double-positive mirror ``(-eps, -mu)`` -- the same ``|n|``, the same admittance magnitudes,
    the other sign of index -- misses by far more than the tolerance (1.8 at least)."""
    mirrored = [(-e, -m) if e.real < 0 and m.real < 0 else (e, m) for e, m in media(name)]
    worst = 0.0
    for i in range(len(CASES[name]['thetas'])):
        for pol in (0, 1):
            got = _solve(name, i, 'tetm', 'SMM', pol, stack=mirrored)
            worst = max(worst, np.abs(got - golden[f'{name}__pol{pol}'][i]).max())
    assert worst > 1e12 * TOLERANCE, (name, worst)


def test_the_golden_holds_the_scenes_the_cases_state(golden):
    assert str(golden['__pymoosh_version']) == PYMOOSH_VERSION
    for name, case in CASES.items():
        np.testing.assert_array_equal(golden[f'{name}__kx'],
                                      [pymoosh_kx(name, theta) for theta in case['thetas']])
        for pol in (0, 1):
            rows = golden[f'{name}__pol{pol}']
            assert rows.shape == (len(case['thetas']), 4)
            assert np.abs(rows[:, 2].real + rows[:, 3].real - 1).max() < 1e-12
        assert any(e.real < 0 and m.real < 0 for e, m in media(name))
