"""The route-parity study's material capture records what the legacy engine steps with.

``studies/fdtd_route_parity.py`` compares the two FDTD routes only where their material inputs
match, and it reads legacy's inputs by replacing ``_Grid.eps_for``, ``mu_for``, ``pole_state``,
``_PoleSet.__init__`` and common's ``fdtd_grid`` with recording wrappers (``MaterialCapture``).
Each wrapper restates the signature of the function it replaces. When the Ey pole-fill fix
gave ``pole_state`` a ``y_half`` keyword, the wrapper kept the old five arguments and every
run of the study stopped on its first scene with a ``TypeError``. Nothing in the suite entered
the capture, so that went unseen until the study was rerun by hand (2026-09-15).

Two checks, no solve, well under a second each:

1. every wrapper accepts every argument its target accepts, so the next signature change on
   the engine side fails here rather than an hour into a rerun;
2. on the crossed damped-metal Drude grid, the Ey pole fill the capture records is the fill at
   Ey's y half-step -- the one the engine steps with -- and differs from the y-node fill. So
   the study's material comparison can still see the defect it was built to catch.
"""
import inspect

import numpy as np
import pytest

from studies import fdtd_route_parity as parity
from wavebit._core import fdtd as fdtd_module

METAL = parity.METAL


def _refusal(wrapper, target, method):
    """Why ``wrapper`` would refuse a call shaped like one ``target`` accepts, or ``None``.

    The call passes every required argument positionally and every optional or keyword-only
    one by keyword, which is how the engine passes ``y_half``. ``method`` drops ``self``
    from the target and puts a stand-in for it first, as the wrappers receive it.
    """
    params = list(inspect.signature(target).parameters.values())[1 if method else 0:]
    args = [object()] if method else []
    kwargs = {}
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is p.empty and p.kind is not p.KEYWORD_ONLY:
            args.append(object())
        else:
            kwargs[p.name] = object()
    try:
        inspect.signature(wrapper).bind(*args, **kwargs)
    except TypeError as error:
        return str(error)
    return None


def test_every_capture_wrapper_accepts_what_its_target_accepts():
    with parity.MaterialCapture() as capture:
        installed = [(owner, name, original, getattr(owner, name))
                     for owner, name, original in capture._saved]
    assert len(installed) == 5
    for owner, name, original, wrapper in installed:
        assert wrapper is not original, f'{name} was not replaced'
        refusal = _refusal(wrapper, original, inspect.isclass(owner))
        assert refusal is None, f'the capture wrapper for {name} refuses a valid call: {refusal}'

    # The check's own resolving power: the wrapper the Ey fix broke is refused.
    def five_argument_pole_state(grid, shape, z_half, x_half, dtype):
        return None

    assert _refusal(five_argument_pole_state, fdtd_module._Grid.pole_state, True) is not None


class _Stepped(Exception):
    """Raised at the first pole step; every pole set has been built by then."""


def _crossed_drude_grid():
    """The damped-metal raster at 1/30, where a y half-step changes Ey's fill by half."""
    raster = np.ones(METAL['dispersion'].shape, dtype=complex)
    return fdtd_module._Grid(
        METAL['wavelength'], METAL['period'][0], raster, list(METAL['thickness']),
        METAL['n_top'] ** 2, METAL['n_bot'] ** 2, resolution=30., n_pml=6, courant=0.99,
        subpixel=True, theta=np.deg2rad(METAL['theta_deg']), phi=np.deg2rad(METAL['phi_deg']),
        dispersion=METAL['dispersion'], poles=METAL['poles'], period_y=METAL['period'][1])


def test_the_capture_records_the_ey_fill_the_engine_steps_with(monkeypatch):
    grid = _crossed_drude_grid()
    assert grid.ny > 1, 'the defect needs a y axis; a one-row raster cannot show it'

    def stop(self, *args, **kwargs):
        raise _Stepped

    monkeypatch.setattr(fdtd_module._PoleSet, 'advance', stop)
    with parity.MaterialCapture() as capture:
        with pytest.raises(_Stepped):
            # Long enough to clear the too-short-run refusal; nothing is stepped.
            fdtd_module._time_loop(grid, None, 10 ** 7, [METAL['wavelength']], leg='s')

    (pole_id,) = sorted(grid.poles)
    ((omega0, gamma, strength),) = grid.poles[pole_id]
    key = ('pole', 'ey', (float(omega0), float(gamma), 0))
    assert key in capture.arrays, sorted(k for k in capture.arrays if k[0] == 'pole')
    _, recorded = capture.arrays[key]
    at_half_step = strength*np.broadcast_to(
        grid.fill_for(pole_id, False, False, y_half=True), recorded.shape)
    on_node = strength*np.broadcast_to(
        grid.fill_for(pole_id, False, False, y_half=False), recorded.shape)
    np.testing.assert_array_equal(recorded, at_half_step)
    assert np.max(np.abs(on_node - recorded)) > .1*np.max(np.abs(recorded)), (
        'on this grid the y node and the y half-step must give different fills, or the check '
        'above could not tell them apart')
