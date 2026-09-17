"""External FDTD goldens: Meep public API and FDTDX native update functions."""
from pathlib import Path
import runpy

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent


def test_meep_slab_fields_power_and_independent_airy_refinement():
    pytest.importorskip('jax')
    worker=runpy.run_path(str(HERE/'interface/validate_grid_meep.py'))['worker']
    errors=[]
    with np.load(HERE/'golden_native_grid_external.npz',allow_pickle=False) as ref:
        wavelengths=ref['wavelengths']
        n=1.5
        phase=2*np.pi*n*.5/wavelengths
        r=(1-n)/(1+n)
        airy=(1-r*r)*np.exp(1j*phase)/(1-r*r*np.exp(2j*phase))*np.exp(-2j*np.pi*.5/wavelengths)
        for resolution in (20,40,80):
            actual=worker('wavebit',resolution)
            fields=np.array(actual['field_real'])+1j*np.array(actual['field_imag'])
            expected=ref[f'meep-grid-{resolution}_fields']
            # Identical planar Yee interface averaging; finite pulse/window error
            # at the coarsest grid prevents a machine-precision claim throughout.
            np.testing.assert_allclose(fields,expected,rtol=0,atol=1e-6)
            np.testing.assert_allclose(actual['transmission'],ref[f'meep-grid-{resolution}_power'],rtol=0,atol=1e-6)
            errors.append(np.max(abs(expected[:,0]-airy)))
        assert errors[-1]<.008
        assert np.all(np.log2(np.array(errors[:-1])/errors[1:])>1.9)
        np.testing.assert_allclose(ref['meep-grid-runtime_fields'],ref['meep-grid-40_fields'],rtol=0,atol=2e-9)


def test_fdtdx_three_dimensional_material_fields_and_gradient():
    pytest.importorskip('jax')
    worker=runpy.run_path(str(HERE/'interface/validate_grid_fdtdx.py'))['worker']
    actual=worker('wavebit')
    with np.load(HERE/'golden_native_grid_external.npz',allow_pickle=False) as ref:
        for key in ('E','H','value','gradient'):
            np.testing.assert_allclose(actual[key],ref['fdtdx_'+key],rtol=1e-11,atol=1e-12)
        assert abs(actual['gradient'])>.1
        np.testing.assert_allclose(actual['gradient'],actual['finite_difference'],rtol=1e-7)
        np.testing.assert_allclose(ref['fdtdx_gradient'],ref['fdtdx_finite_difference'],rtol=1e-7)
