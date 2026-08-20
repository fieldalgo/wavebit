"""A *patterned* anisotropic layer against fmmax, from a checked-in golden.

This is the case that had no oracle. ``test_fmmax_golden.py`` covers uniform tensors, where
Berreman and RETICOLO also reach; here the tensor varies across the cell, and of the four
reference solvers surveyed for this project only fmmax both accepts a patterned complex
off-diagonal and demonstrably uses it. So this file is what stands behind
``wavebit._core.factorization.pn_conv_matrices``.

**It asserts convergence, not identity, and the tolerance is measured rather than chosen.**
Three formulations are in play — the Popov-Neviere rearrangement under test, fmmax's ``FFT``
(the plain direct rule) and fmmax's ``JONES_DIRECT`` (a vector formulation built differently).
They agree only in the limit. The golden therefore holds *both* fmmax answers, and the tolerance
each case is allowed is the spread between them: no claim can be sharper than the distance
between two of the oracle's own results.

What that leaves is still a real check. The spreads are 4.8e-6 / 3.0e-5 / 1.7e-4 across the
three contrasts, and the solver lands on ``JONES_DIRECT`` to 3e-7, 1.4e-7 and 2.0e-5 — an order
or more inside the band, and on the vector side of it, which is where a vector formulation
belongs. A sign error, a transposed raster or a dropped off-diagonal all move the answer by far
more than the band: with the raster transpose missing, the s/p split came out *reversed* and the
gap was 30x the band.

Values are frozen by ``gen_goldens_fmmax_patterned.py``, which needs the ``fwref`` env. Never
regenerate to make this pass — the whole worth of the file is that the numbers did not come
from this codebase.
"""
import pathlib
import sys

import numpy as np
import pytest

import wavebit
from wavebit._core.factorization import normal_vector_field, pn_conv_matrices

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _fmmax_patterned_cases import (                                           # noqa: E402
    CASES, FTO, GRID, PERIOD, THETA, THICK, WL, tensor_rasters)

GOLDEN_PATH = HERE / 'golden_fmmax_patterned.npz'


def solve(n_in, g, fto=FTO):
    """``[R_s, T_s, R_p, T_p]`` for one patterned gyrotropic layer between vacuum."""
    eps, gyro = tensor_rasters(n_in, g)

    eps33 = np.zeros((GRID, GRID, 3, 3), dtype=complex)
    eps33[..., 0, 0] = eps33[..., 1, 1] = eps33[..., 2, 2] = eps
    eps33[..., 0, 1], eps33[..., 1, 0] = gyro, -gyro         # the physical tensor

    nx, ny = normal_vector_field(eps)
    epx, epy, epz_i, epxy, epyx = pn_conv_matrices(np, eps33, nx, ny, fto, fto)

    # `structure` is the scalar permittivity raster, handed over as it stands: the `np.sqrt`
    # that used to sit here existed only because the constructor took an index and squared it
    # back. It selects the modeling path and the grating type; the convolution matrices the
    # solve actually uses are the ones passed to `solve_conv` below.
    mee = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=np.deg2rad(THETA), phi=0.0,
                       psi=0.0, fto=(fto, fto), period=(PERIOD, PERIOD), wavelength=WL,
                       thickness=[THICK], structure=np.expand_dims(eps, axis=0))
    res = mee.solve_conv(np.stack([epx]), np.stack([epy]), np.stack([epz_i]),
                             np.stack([epxy]), np.stack([epyx]))
    te, tm = res.res_te_inc, res.res_tm_inc
    return np.array([float(te.R.sum()), float(te.T.sum()),
                     float(tm.R.sum()), float(tm.T.sum())])


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'{GOLDEN_PATH.name} missing; regenerate with '
                    'gen_goldens_fmmax_patterned.py')
    return np.load(GOLDEN_PATH)


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_lands_inside_the_spread_between_fmmax_own_formulations(golden, name):
    fft, jones = golden[f'{name}__fft'], golden[f'{name}__jones']
    spread = np.abs(fft - jones).max()
    got = solve(*CASES[name])

    # Inside the band the two fmmax formulations bracket. Stated as a distance to the nearer
    # of the two rather than to a midpoint: the three schemes do not converge at the same rate,
    # so a midpoint would be an arbitrary target dressed up as a reference.
    gap = min(np.abs(got - fft).max(), np.abs(got - jones).max())
    assert gap < spread, (f'{name}: {gap:.2e} from the nearer fmmax formulation, past the '
                          f'{spread:.2e} the two of them differ by\n'
                          f'  got   {np.array2string(got, precision=9)}\n'
                          f'  fft   {np.array2string(fft, precision=9)}\n'
                          f'  jones {np.array2string(jones, precision=9)}')


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_it_is_the_vector_formulation_that_it_agrees_with(golden, name):
    """Not decoration: it is what says the factorization is doing its job.

    A Popov-Neviere rearrangement and a Jones-vector formulation are both attempts to respect
    the field discontinuity at the boundary; the direct rule is the one that ignores it. If
    this solver sat nearer ``FFT`` than ``JONES_DIRECT``, the vector machinery would be
    present in the code and absent from the answer — which is exactly the failure mode that a
    pure R+T check cannot see.
    """
    fft, jones = golden[f'{name}__fft'], golden[f'{name}__jones']
    got = solve(*CASES[name])
    assert np.abs(got - jones).max() < np.abs(got - fft).max()


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_energy_closes_on_these_lossless_layers(name):
    """``eps`` is Hermitian at every pixel, so the layer is lossless however it is patterned.

    Independent of the golden — a theorem, so it holds whatever fmmax says, and it is the one
    check here that would survive the oracle being wrong.
    """
    got = solve(*CASES[name])
    for pol, (r, t) in (('s', got[:2]), ('p', got[2:])):
        assert abs(r + t - 1.0) < 1e-6, f'{name} {pol}: R+T = {r + t:.12f}'


def test_the_golden_is_not_two_copies_of_one_answer(golden):
    """Guards the oracle. The whole design rests on two *independent* fmmax formulations.

    If a future edit passed the same ``formulation`` twice, the spread would collapse to zero,
    the tolerance with it, and the tests above would start failing for a reason that has
    nothing to do with the solver. Better to fail here, where the message says so.
    """
    for name in CASES:
        spread = np.abs(golden[f'{name}__fft'] - golden[f'{name}__jones']).max()
        assert spread > 1e-9, (f'{name}: the two frozen formulations differ by {spread:.2e} — '
                               'they are not two formulations')
