"""External CPML and electric dispersive material references; no Meep import."""
from pathlib import Path
import json
import runpy

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


@pytest.mark.parametrize('resolution', [40, 80])
def test_low_loss_mixed_metal_against_meep(resolution):
    pytest.importorskip('jax')
    worker = runpy.run_path(str(ROOT/'studies/validate_grid_dispersive.py'))['worker']
    reference = json.loads((HERE/'golden_low_loss_metal_meep.json').read_text())
    expected, = [v for v in reference['runs'] if v['resolution'] == resolution]
    actual = worker('wavebit', resolution, 'low-loss-mixed', duration=160., pole_sampling='point')
    for key in ('model', 'resolution', 'duration', 'wavelengths', 'pml_width', 'pole_sampling'):
        assert actual[key] == expected[key]
    def fields(data):
        return np.array(data['field_real'])+1j*np.array(data['field_imag'])
    np.testing.assert_allclose(fields(actual), fields(expected), atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual['transmission'], expected['transmission'], atol=1e-6, rtol=0)


@pytest.mark.parametrize('model',['dielectric','lorentz','drude','mixed'])
def test_meep_cpml_material_fields_and_resolution_convergence(model):
    pytest.importorskip('jax')
    worker=runpy.run_path(str(ROOT/'studies/validate_grid_dispersive.py'))['worker']
    errors=[]
    with np.load(HERE/'golden_native_material_external.npz',allow_pickle=False) as ref:
        for resolution in (40,80):
            # Match Meep's documented point sampling of the dispersive part.
            actual=worker('wavebit',resolution,model,pole_sampling='point')
            fields=np.array(actual['field_real'])+1j*np.array(actual['field_imag'])
            name=f'meep-{model}-{resolution}-band'
            np.testing.assert_allclose(fields,ref[name+'_fields'],atol=3e-6,rtol=0)
            np.testing.assert_allclose(actual['transmission'],ref[name+'_power'],atol=3e-6,rtol=0)
            # Independently average tangential pole weights and check Airy;
            # this is a different discretization and must not inherit Meep's
            # point-sampling golden as its exact answer.
            average=worker('wavebit',resolution,model)
            errors.append(average['airy_error'])
        assert errors[-1]<.008
        assert errors[0]/errors[1]>3.5


def test_fdtdx_two_pole_updates_and_oscillator_strength_derivative():
    pytest.importorskip('jax')
    actual=runpy.run_path(str(ROOT/'studies/validate_grid_fdtdx_ade.py'))['worker']('wavebit')
    with np.load(HERE/'golden_native_material_external.npz',allow_pickle=False) as ref:
        for key in ('E','H','value','gradient'):
            np.testing.assert_allclose(actual[key],ref['fdtdx_'+key],rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(actual['gradient'],actual['finite_difference'],rtol=1e-7)
        np.testing.assert_allclose(ref['fdtdx_gradient'],ref['fdtdx_finite_difference'],rtol=1e-7)


def test_cpml_width_and_runtime_controls():
    pytest.importorskip('jax')
    worker=runpy.run_path(str(ROOT/'studies/validate_grid_dispersive.py'))['worker']
    baseline=worker('wavebit',40,'dielectric')
    def fields(d):
        return np.array(d['field_real'])+1j*np.array(d['field_imag'])
    for kw in ({'duration':36.},{'pml_width':2.}):
        control=worker('wavebit',40,'dielectric',**kw)
        np.testing.assert_allclose(fields(control),fields(baseline),atol=3e-6,rtol=0)
