"""Integrated tensor shapes against the existing independent fmmax goldens.

The reference raster has grid-aligned straight boundaries, so its geometry
is exactly the rectangle below. No reference solver or new dependency runs.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import wavebit

sys.path.insert(0,str(Path(__file__).parent))
from _fmmax_patterned_cases import CASES, MAX_ORDER, PERIOD, THETA, THICK, WL


@pytest.mark.parametrize('name',list(CASES))
def test_integrated_tensor_pillar_inside_independent_formulation_spread(name):
    index,g = CASES[name]
    e = np.eye(3,dtype=complex)*index**2
    e[0,1],e[1,0] = g,-g
    solver = wavebit.rcwa(structure=[dict(background=1.,shapes=[dict(
        type='rectangle',cx=-PERIOD/24,lx=PERIOD*5/12,ly=PERIOD/3,
        material=wavebit.Material(e))])],period=(PERIOD,PERIOD),thickness=[THICK],
        wavelength=WL,theta=np.deg2rad(THETA),phi=0.,
        truncation_order=(MAX_ORDER,MAX_ORDER),shape_nvf_grid_size=256)
    r = solver.solve(auto_detune=False)
    got = np.array([r.s_inc.R.sum(),r.s_inc.T.sum(),r.p_inc.R.sum(),r.p_inc.T.sum()])
    with np.load(Path(__file__).with_name('golden_fmmax_patterned.npz')) as reference:
        fft,jones = reference[name+'__fft'],reference[name+'__jones']
    gap = min(abs(got-fft).max(),abs(got-jones).max())
    assert gap < abs(fft-jones).max()
