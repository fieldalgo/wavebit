"""`reconstruct_field` must be reconstructing the scene that was solved.

Its own first line promises "the state left behind by the last solve", and it does walk that:
`T1` and `layer_info_list` are the solve's. But it re-reads `wavelength`, `thickness`, the cell,
the incidence angles, `incidence_side`, the half-space media and the retained order window from
the solver as it goes -- so a condition changed between the solve and the reconstruction
produced a field belonging to **neither** scene.

Measured on a 1D grating before the fix: solve at 900 nm, move `wavelength` to 600 without
re-solving, reconstruct.

    against the field before the change    max|d| = 1.105    so the new wavelength WAS read
    against the field after re-solving     max|d| = 0.582    so it is not the new scene either

Both non-zero is the whole finding. Either one alone is explainable -- the first as "it ignored
the change", the second as "it applied it" -- and it was doing half of each.

Crossed over every condition the reconstruction reads, because the mix is per-condition: a
check that watched only `wavelength` would leave eight other ways in. The last three joined on
2026-09-28, after a review measured the first of them: a 1D grating solved under `top=1.0`,
reconstructed under `top=1.3`, came back 0.19 from the field before and 0.11 from the re-solved
one, unrefused. The lit medium's index sets the incident wavevector, which `order_kxy` and the
centre phase read live. An `excitation` that moves the window's centre keeps its size, so no
shape mismatch caught that one either: `None -> {1: 1.0}` came back 1.87 and 1.76 away.
"""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

import wavebit

P, WL, H = 1500.0, 900.0, 150.0
GRATING = np.array([[[1.0, 1.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0]]])

#: Every live read `reconstruct_field` makes, as (name, a value that differs from the scene's).
#: `period` is here as well as `lattice` because the two are separate reads and a solver may
#: carry either spelling.
MOVES = {
    'wavelength': 600.0,
    'thickness': [2 * H],
    'theta': 0.3,
    'phi': 0.2,
    'period': [2 * P, 2 * P],
    'incidence_side': 'bottom',
    'top': 2.25,
    'bot': wavebit.Material(1.0, mu=2.25),           # permeability alone: both constants count
    'excitation': {1: 1.0},                          # the window's centre moves, not its size
}


def _scene(backend='numpy', **over):
    kw = dict(backend=backend, structure=GRATING, period=[P, P], truncation_order=[2, 0],
              thickness=[H], wavelength=WL, top=1.0, bot=1.0, answers='field',
              theta=0.1, phi=0.0)
    kw.update(over)
    return _legacy_wavebit.rcwa(**kw)


def _field(solver):
    return np.asarray(solver.reconstruct_field(4, 4, 4)[0])


@pytest.mark.parametrize('name', sorted(MOVES))
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_moving_any_condition_the_reconstruction_reads_is_refused(backend, name):
    """One case per condition, because the mix is per-condition and each is a separate read."""
    pytest.importorskip(backend)
    solver = _scene(backend)
    solver.solve()
    _field(solver)                                   # the honest one still works

    setattr(solver, name, MOVES[name])
    with pytest.raises(RuntimeError, match=r'scene has changed'):
        _field(solver)

    solver.solve()                                   # and re-solving clears it
    assert np.isfinite(_field(solver)).all()


@pytest.mark.parametrize('name', sorted(MOVES))
def test_reassigning_the_same_value_is_not_a_change(name):
    """The check compares values, not assignments.

    A cascade or a caller writing a condition back onto itself must not be told the scene moved
    -- `complex_dtype`'s re-cast does exactly that to several conditions.
    """
    solver = _scene()
    solver.solve()
    setattr(solver, name, getattr(solver, name))
    assert np.isfinite(_field(solver)).all()


def test_narrowing_the_width_after_a_solve_is_a_scene_change_too():
    """`complex_dtype` re-casts the *conditions* and leaves the solve state where it was.

    `T1` and `layer_info_list` hold what the solve computed, at the width it ran at; the cascade
    does not touch them. So a reconstruction after narrowing would read float32 conditions
    against complex128 modes -- the same mix, arrived at a different way.

    The width is in the key explicitly rather than left to fall out of the numbers, because
    `complex_dtype=1` moves `theta=0.1` and leaves `theta=0.0` alone: without it, whether this
    was caught would depend on whether the scene's values happen to survive float32.
    """
    for theta in (0.0, 0.1):                          # one survives the narrowing, one does not
        solver = _scene(theta=theta)
        solver.solve()
        solver.complex_dtype = 1
        with pytest.raises(RuntimeError, match=r'scene has changed'):
            _field(solver)
        solver.solve()
        assert np.isfinite(_field(solver)).all(), theta


def test_solve_field_never_meets_the_check():
    """It solves and reconstructs in one call, so its state is by construction the live scene.

    A check placed in the shared `_solve_and_field` body instead of in `reconstruct_field`
    would fire here, on a solver that has just been solved.
    """
    solver = _scene()
    _, cell = solver.solve_field(4, 4, 4)
    assert np.isfinite(np.asarray(cell[0])).all()

    # The discriminating case: a solver that HAS solved, whose scene then moves. A check placed
    # at the top of `_solve_and_field` would compare against the old solve and refuse -- while
    # the call is about to re-solve the new scene, so there is nothing stale about it. A fresh
    # solver cannot show this: its `_solved_scene` is None and the check skips.
    for name, value in MOVES.items():
        used = _scene()
        used.solve()
        setattr(used, name, value)
        _, cell = used.solve_field(4, 4, 4)          # solves the new scene, then reconstructs
        assert np.isfinite(np.asarray(cell[0])).all(), name

    # And the reconstruction it returns is the NEW scene's, not a mix: it must differ from what
    # the unchanged scene reconstructs.
    base = _scene()
    _, base_cell = base.solve_field(4, 4, 4)
    moved = _scene()
    moved.solve()
    moved.wavelength = MOVES['wavelength']
    _, moved_cell = moved.solve_field(4, 4, 4)
    assert not np.allclose(np.asarray(base_cell[0]), np.asarray(moved_cell[0]))


def test_a_jitted_reconstruction_is_not_refused_on_its_traced_conditions():
    """Under a trace the comparison is skipped, which is the rule every staleness check follows.

    `solve_field` on JAX jits over `self`, so `wavelength` and the rest arrive as tracers and
    `complex()` on one raises. Returning `None` from the key is what makes that a skip rather
    than an exception out of the middle of a trace.
    """
    pytest.importorskip('jax')
    _legacy_wavebit.enable_x64()
    solver = _scene('jax')
    geometry = solver.freeze_geometry()
    _, cell = solver.solve_field(4, 4, 4, geometry=geometry)
    assert np.isfinite(np.asarray(cell[0])).all()


def test_the_field_before_and_after_a_change_were_two_different_wrong_answers():
    """The measurement the refusal is built on, kept as a test.

    Re-solving after the change gives a field that differs from *both* the pre-change
    reconstruction and -- before this fix -- from the mixed one. Asserted here as the part that
    is still observable: the two legitimate fields differ, so the mixed one could not have
    equalled both.
    """
    solver = _scene()
    solver.solve()
    before = _field(solver)

    solver.wavelength = 600.0
    solver.solve()
    after = _field(solver)

    assert not np.allclose(before, after), (
        'the two scenes answer the same field, so this scene cannot show the mix')


@pytest.mark.parametrize('name', ['thickness', 'period', 'theta'])
def test_a_differentiated_condition_is_not_a_scene_change_either(name):
    """A value snapshot must neither change nor detach a live autograd condition."""
    torch = pytest.importorskip('torch')

    solver = _scene('torch')
    leaf = torch.tensor(np.atleast_1d(np.asarray(getattr(solver, name), dtype=float)),
                        dtype=torch.float64, requires_grad=True)
    setattr(solver, name, leaf if name != 'theta' else leaf[0])

    result = solver.solve()                          # `_finish` takes the key here
    torch.sum(result.T).real.backward()
    assert leaf.grad is not None, 'the gradient did not reach the leaf'

    # Each concrete autograd condition is comparable by value, and reconstructing
    # the unchanged scene must remain valid with its graph intact.
    solver.answers = 'field'
    solver.solve()
    assert np.isfinite(np.asarray(solver.reconstruct_field(4, 4, 4)[0].detach())).all()
