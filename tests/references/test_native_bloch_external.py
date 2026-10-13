"""Three-axis complex Bloch update and gradient reference from live FDTDX."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import runpy

import numpy as np
import pytest


def test_fdtdx_three_axis_complex_bloch_fields_and_material_gradient():
    pytest.importorskip('jax')
    here=repository_path('tests/references')
    actual=runpy.run_path(str(here/'interface/validate_grid_fdtdx_bloch.py'))['worker']('wavebit')
    with np.load(here/'golden_native_bloch_external.npz',allow_pickle=False) as expected:
        for key in ('E','H','E_imag','H_imag','value','gradient'):
            np.testing.assert_allclose(actual[key],expected[key],rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(actual['gradient'],actual['finite_difference'],rtol=1e-7)
        np.testing.assert_allclose(expected['gradient'],expected['finite_difference'],rtol=1e-7)
