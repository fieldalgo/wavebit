"""A pole model's strength sits where its component's permittivity sits.

**The defect this file exists for.** `_time_loop` builds each electric component's
permittivity with ``eps_for`` and its Drude/Lorentz state with ``pole_state``, one line apart
and with the Yee position spelled twice. 1c521ea moved ``E_y``'s permittivity to the y
half-step when the grating became crossed and left its pole fill on the y nodes, so on a
crossed dispersive raster ``E_y``'s pole strength sat half a cell off the Yee position ``E_y``
is updated at. While the raster has no y axis (``ny == 1``) the two spellings are one
computation, which is why no 1D grating, no uniform slab and no test saw it; the measured
cost on the crossed damped metal was 0.03 of absorbed power (s: 0.402 against RCWA's 0.431
at a 1/30 grid).

Two tests, and the order is the argument:

1. **the structural one**, which needs no solve. It lets ``_time_loop`` set up a crossed
   Drude scene, records every permittivity and pole set it builds, pins each component's
   permittivity to its Yee position, and holds each component's pole fill to the fill
   *recomputed at that position*. Energy conservation, an RCWA gap or a Meep tolerance all
   see this only through its size; this sees it at any size.
2. **the physics one**: the legacy route on a crossed Drude metal against RCWA, with a
   tolerance the shipped ``E_y`` offset fails. It pins ``E_y`` only: measured, a y offset on
   ``E_z``'s pole fill stays inside it (and one on ``E_x`` fails only through p absorption),
   so for ``E_x`` and ``E_z`` the structural test is the guard.

The electric components are the only ones with poles -- ``pole_state`` is electric only and
there is no magnetic dispersion -- so ``E_x``, ``E_y`` and ``E_z`` are the whole set.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

import wavebit
from wavebit._core import fdtd as fdtd_module

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'references'))
from _meep_crossed_metal_case import CASE as METAL  # noqa: E402
from _meep_crossed_material_cases import epsilon  # noqa: E402
from test_meep_crossed_materials_golden import scene, extract, gap  # noqa: E402


class _Built(Exception):
    """Raised once the stepper has built every pole set, so no time step is taken."""


#: The Yee component each ``eps_for`` call belongs to, named by the axis it averages
#: harmonically along -- which is the axis the component points along (`component_property`).
_COMPONENT_OF_AXIS = {'x': 'ex', 'y': 'ey', 'z': 'ez'}


def _crossed_drude_grid(subpixel, resolution):
    """The damped-metal raster, on a grid where a y half-step changes the fill."""
    raster = np.ones(METAL['dispersion'].shape, dtype=complex)
    return fdtd_module._Grid(
        METAL['wavelength'], METAL['period'][0], raster, list(METAL['thickness']),
        METAL['n_top'] ** 2, METAL['n_bot'] ** 2, resolution=resolution, n_pml=6, courant=0.99,
        subpixel=subpixel, theta=np.deg2rad(METAL['theta_deg']),
        phi=np.deg2rad(METAL['phi_deg']), dispersion=METAL['dispersion'], poles=METAL['poles'],
        period_y=METAL['period'][1])


def _what_the_stepper_builds(grid, monkeypatch):
    """``(eps positions by component, pole-set fills in construction order)``, no time step.

    ``eps_for`` is wrapped on the instance and ``_PoleSet`` replaced in the module, so the
    recording is of the calls ``_time_loop`` itself makes -- the call sites are what carried
    the defect, not ``pole_state``.
    """
    positions, fills = {}, []
    eps_for = grid.eps_for

    def recording_eps_for(z_half, x_half, harmonic_axis, y_half=False):
        positions[_COMPONENT_OF_AXIS[harmonic_axis]] = (z_half, x_half, y_half)
        return eps_for(z_half, x_half, harmonic_axis, y_half=y_half)

    class RecordingPoleSet:
        def __init__(self, shape, dt, poles, pole_fills, dtype):
            fills.append([np.asarray(f) for f in pole_fills])
            if len(fills) == 3:
                raise _Built

        def advance(self, e):
            # Reached only if a component's pole set was never built: stop at the first step
            # so the count below names the problem instead of a missing attribute.
            raise _Built

    monkeypatch.setattr(grid, 'eps_for', recording_eps_for)
    monkeypatch.setattr(fdtd_module, '_PoleSet', RecordingPoleSet)
    with pytest.raises(_Built):
        # Long enough to clear the too-short-run refusal; nothing is stepped.
        fdtd_module._time_loop(grid, None, 10 ** 7, [METAL['wavelength']], leg='s')
    assert len(fills) == 3, f'{len(fills)} pole sets built for the three electric components'
    return positions, fills


#: Each electric component's Yee position as ``(z_half, x_half, y_half)``, from the engine's
#: own comments in `_time_loop`: Ey on the z and x nodes and the y half-step, Ex on the x
#: half-step, Ez on the z half-step. Pinned separately from the co-location check, because
#: moving a component's permittivity *and* its pole fill together keeps them co-located.
_YEE = {'ey': (False, False, True), 'ex': (False, True, False), 'ez': (True, False, False)}


# Point sampling can tell the y node from the y half-step only if no raster wall lands on a
# y node, i.e. `ny` is not a multiple of the raster's 3 rows. At resolutions 20/30/40
# (ny = 6/9/12) every wall is on a node and the two samplings are identical (0.0); at
# 25/35/45 (ny = 8/10/14) they differ by 1.0. Subpixel averaging differs by 0.5 at any of
# them. The resolving-power check at the end of the test is what holds this choice.
@pytest.mark.parametrize('subpixel, resolution', [(True, 30.0), (False, 25.0)])
def test_every_pole_fill_is_sampled_where_its_component_permittivity_is(subpixel, resolution,
                                                                        monkeypatch):
    grid = _crossed_drude_grid(subpixel, resolution)
    assert grid.ny > 1, 'the defect needs a y axis; a one-row raster cannot show it'
    positions, fills = _what_the_stepper_builds(grid, monkeypatch)
    assert positions == _YEE, 'a permittivity is sampled off its Yee position'

    # `_time_loop` builds the TE triple's pole set (Ey) and then the TM triple's (Ex, Ez).
    (pole_id,) = sorted(grid.poles)
    for component, built in zip(('ey', 'ex', 'ez'), fills):
        z_half, x_half, y_half = positions[component]
        want = grid.fill_for(pole_id, z_half, x_half, y_half=y_half)
        (got,) = built
        assert got.shape == want.shape, component
        np.testing.assert_array_equal(
            got, want, err_msg=f'{component}: pole fill is not sampled at its eps position '
                               f'(z_half, x_half, y_half) = {positions[component]}')

    # The test's own resolving power, for every component: on this grid a half step in y or
    # in x genuinely changes each fill, so each comparison above is one of positions and not
    # of values that happen to agree. Ey's y flip is the shipped defect.
    for component, built in zip(('ey', 'ex', 'ez'), fills):
        z_half, x_half, y_half = positions[component]
        for moved_x, moved_y in ((x_half, not y_half), (not x_half, y_half)):
            off = grid.fill_for(pole_id, z_half, moved_x, y_half=moved_y)
            assert np.max(np.abs(off - built[0])) > 0.1, (component, moved_x, moved_y)


#: Measured at this 1/30 grid against RCWA order 11 (absorption 0.4310 s / 0.4514 p; order 7
#: agrees to 6e-4). Half a cell off, the legacy route absorbed 0.4024 / 0.4460 -- s out by
#: 0.029 -- with a worst complex-amplitude gap of 0.063. Co-located: 0.4326 / 0.4601, gaps
#: 0.0016 / 0.0087, amplitude gap 0.031. p's absorption gap is the one that grows under the
#: fix (0.0054 -> 0.0087); it is the grid's own error and not the offset's, which is why the
#: bound is one both legs clear co-located and the offset s leg fails by a factor of two.
ABSORPTION_TOL = 0.015
AMPLITUDE_TOL = 0.045


def test_crossed_drude_metal_absorption_matches_rcwa():
    """The legacy route (``backend`` omitted) on the damped crossed metal, against RCWA.

    Absorption is the quantity a misplaced pole strength moves first; only order (0, 0)
    propagates, so it is ``1 - R - T`` on one order each side. The amplitude gap is the
    same comparison ``test_meep_crossed_metal_golden.py`` makes, at half its cost.
    """
    result, diag = wavebit.fdtd(
        **scene(METAL), dispersion=METAL['dispersion'], poles=METAL['poles'],
        resolution=30., n_periods=160, n_pml=12, pad_scale=1., courant=.99, subpixel=True,
        grading=None, return_diagnostics=True)
    assert 'grid' in diag, 'backend omitted must reach the legacy route'
    assert diag['grid'].ny > 1
    got = extract(result, METAL)
    kw = scene(METAL)
    kw['structure'] = epsilon(METAL, METAL['wavelength'])
    kw['truncation_order'] = dict(x=11, y=11)
    ref = extract(wavebit.rcwa(backend='numpy', **kw).solve(), METAL)
    amplitude = gap(got, ref, METAL)
    assert amplitude < AMPLITUDE_TOL, f'worst complex-amplitude gap {amplitude:.4f} from RCWA'
    for leg in ('s', 'p'):
        measured = 1 - got[f'{leg}.R'].sum() - got[f'{leg}.T'].sum()
        want = 1 - ref[f'{leg}.R'].sum() - ref[f'{leg}.T'].sum()
        assert abs(measured - want) < ABSORPTION_TOL, (
            f'{leg}: legacy FDTD absorbs {measured:.5f}, RCWA {want:.5f}')
