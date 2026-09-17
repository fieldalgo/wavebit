"""Three-axis complex Bloch update and gradient reference from live FDTDX."""
from pathlib import Path
import runpy

import numpy as np
import pytest


def test_fdtdx_three_axis_complex_bloch_fields_and_material_gradient():
    pytest.importorskip('jax')
    here=Path(__file__).resolve().parent
    actual=runpy.run_path(str(here/'interface/validate_grid_fdtdx_bloch.py'))['worker']('wavebit')
    with np.load(here/'golden_native_bloch_external.npz',allow_pickle=False) as expected:
        for key in ('E','H','E_imag','H_imag','value','gradient'):
            np.testing.assert_allclose(actual[key],expected[key],rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(actual['gradient'],actual['finite_difference'],rtol=1e-7)
        np.testing.assert_allclose(expected['gradient'],expected['finite_difference'],rtol=1e-7)
