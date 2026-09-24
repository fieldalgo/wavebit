"""Cartesian tensor Li against checked-in, independent fmmax optical values.

No reference software runs here. The tolerance is the existing spread between
fmmax's FFT and JONES_DIRECT formulations at the stored Fourier order.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import wavebit

sys.path.insert(0, str(Path(__file__).parent))
from _fmmax_patterned_cases import CASES, MAX_ORDER, PERIOD, THETA, THICK, WL


@pytest.mark.parametrize('name', list(CASES))
def test_tensor_li_inside_independent_reference_spread(name):
    index, gyro = CASES[name]
    epsilon = np.eye(3, dtype=complex) * index**2
    epsilon[0,1], epsilon[1,0] = gyro, -gyro
    solver = wavebit.rcwa(
        wavelength=WL, period=(PERIOD,PERIOD), thickness=[THICK],
        theta=np.deg2rad(THETA), phi=0.,
        truncation_order=(MAX_ORDER,MAX_ORDER), factorization='li',
        structure=[dict(background=1.,shapes=[dict(
            type='rectangle',cx=-PERIOD/24,lx=PERIOD*5/12,ly=PERIOD/3,
            material=wavebit.Material(epsilon))])],
    )
    result = solver.solve(auto_detune=False)
    got = np.array([result.s_inc.R.sum(),result.s_inc.T.sum(),
                    result.p_inc.R.sum(),result.p_inc.T.sum()])
    with np.load(Path(__file__).with_name('golden_fmmax_patterned.npz')) as reference:
        fft, jones = reference[name+'__fft'], reference[name+'__jones']
    spread = abs(fft-jones).max()
    error = min(abs(got-fft).max(), abs(got-jones).max())
    assert error < spread, (name, got, error, spread)
    np.testing.assert_allclose(got.reshape(2,2).sum(axis=1),1.,atol=2e-10,rtol=0)
