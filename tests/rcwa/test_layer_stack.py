"""A stack is one stack: as many thicknesses as the structure has layers.

Every solve sizes its layer loop off ``len(thickness)``, so the two ways of getting this wrong
failed in opposite directions and neither said what the caller had done:

* **more layers than thicknesses** -- the extra convolution slots were built, paid for, and
  never indexed. Measured on a 1D grating: a two-layer structure with one thickness answered
  T = 0.851185066096, *bit-identical* to the one-layer structure with one thickness. The second
  layer was silently not there.
* **more thicknesses than layers** -- ``IndexError: index 1 is out of bounds for axis 0 with
  size 1``, raised from inside the solve, which says where it died rather than what was wrong.

The whole 3x3 matrix is run, on every modeling path, because the defect is a *relation* between
two conditions and the diagonal is the only part of it that was ever correct. Spot-checking one
mismatch would have checked one of the two directions.
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest


P, WL, H = 1500.0, 900.0, 150.0

_GROOVE = np.array([[1.0, 1.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0]])

#: The same one-layer stack in the two spellings `structure` accepts, each of which reports
#: its layer count as `len()`: a raster is `(n_layers, Ny, Nx)`, and the shape spelling is a
#: list with one entry per layer.
def _layers(path, count):
    if path == 'raster':
        return np.stack([_GROOVE * (1.0 + 0.5 * i) for i in range(count)])
    return [{'background': 1.0,
             'shapes': [{'type': 'rectangle', 'cx': -0.1 * P, 'cy': 0.0,
                         'lx': 0.3 * P, 'ly': P, 'material': 4.0 + i}]}
            for i in range(count)]


PATHS = ('raster', 'smooth')
COUNTS = (1, 2, 3)


def _solver(path, layers, thicknesses, backend='numpy', **over):
    kw = dict(backend=backend, structure=_layers(path, layers),
              thickness=[H] * thicknesses, period=[P, P], truncation_order=[2, 0],
              wavelength=WL, top=1.0, bot=1.0)
    if path == 'smooth':
        kw['factorization'] = 'nvf'
    kw.update(over)
    return _legacy_wavebit_rcwa(**kw)


@pytest.mark.parametrize('thicknesses', COUNTS)
@pytest.mark.parametrize('layers', COUNTS)
@pytest.mark.parametrize('path', PATHS)
def test_a_stack_needs_one_thickness_per_layer(path, layers, thicknesses):
    """The full matrix. Only the diagonal solves; every off-diagonal cell is refused by name.

    Run over both modeling paths because the layer count is read through `len` and the two
    spell a stack differently -- a raster's first axis, and a list per layer for the shape
    spelling. A rule that held for one spelling and not the other would be invisible from
    either alone.
    """
    solver = _solver(path, layers, thicknesses)

    if layers == thicknesses:
        result = solver.solve()
        assert np.isfinite(np.asarray(result.T).sum())
        return

    with pytest.raises(ValueError, match=r'one thickness per layer'):
        solver.solve()


@pytest.mark.parametrize('path', PATHS)
def test_the_refusal_says_both_counts_and_which_way_it_is_wrong(path):
    """A refusal that says only "mismatch" sends the reader to count them by hand.

    The two directions failed differently and are worth telling apart: one dropped layers in
    silence, the other reached past the end of the structure. The message names the one that
    happened.
    """
    with pytest.raises(ValueError) as too_many_layers:
        _solver(path, 3, 1).solve()
    message = str(too_many_layers.value)
    assert '3 layer' in message and '1 entry' in message, message
    assert 'never indexed' in message, message

    with pytest.raises(ValueError) as too_many_thicknesses:
        _solver(path, 1, 3).solve()
    message = str(too_many_thicknesses.value)
    assert '1 layer' in message and '3 entr' in message, message
    assert 'IndexError' in message, message


@pytest.mark.parametrize('mismatch', [(2, 1), (1, 2)], ids=['extra-layer', 'extra-thickness'])
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_every_backend_refuses_the_same_mismatch(backend, mismatch):
    """The check lives in `_core`, so all three must reach it -- and reach it before the solve.

    Torch raised its own dtype error and JAX its own tracer error from further in, so a per-
    backend refusal would have read as three unrelated bugs.
    """
    pytest.importorskip(backend)
    layers, thicknesses = mismatch
    with pytest.raises(ValueError, match=r'one thickness per layer'):
        _solver('raster', layers, thicknesses, backend=backend).solve()


@pytest.mark.parametrize('entry', ['solve', 'freeze_geometry', 'eps_conv_matrices'])
def test_the_mismatch_is_refused_at_every_door_that_builds(entry):
    """`_check_before_building` is the one call every build goes through, which is why it is there.

    A check in `solve` alone would leave `freeze_geometry` recording a geometry for a stack that
    cannot be solved, and the failure would surface later on a replay.
    """
    solver = _solver('raster', 2, 1)
    with pytest.raises(ValueError, match=r'one thickness per layer'):
        getattr(solver, entry)()


def test_a_half_built_stack_is_not_refused_until_it_is_built():
    """The two conditions arrive in either order, so the mid-assignment state has to be legal.

    Assigning a two-layer structure to a solver that still holds one thickness is a legal
    half-step; refusing in the setter would make the order of two assignments matter. Same
    reason `excitation`'s legality is read at build time rather than in its setter.
    """
    solver = _solver('raster', 1, 1)
    solver.structure = _layers('raster', 2)          # legal, and not yet solvable
    assert len(solver.structure) == 2
    solver.thickness = [H, H]                        # now it is
    assert np.isfinite(np.asarray(solver.solve().T).sum())


def test_the_extra_layer_used_to_be_dropped_in_silence():
    """The regression this exists for, stated as the equality that must no longer be reachable.

    A two-layer structure with one thickness answered exactly what the one-layer structure
    answers -- same digits, no warning. Asserted here as: the shorter stack still solves, the
    mismatched one does not, and so the two can no longer return the same number.
    """
    one = float(np.asarray(_solver('raster', 1, 1).solve().T).sum())
    assert np.isfinite(one)
    with pytest.raises(ValueError, match=r'one thickness per layer'):
        _solver('raster', 2, 1).solve()
