"""Static graded scene meshes preserve interfaces, records and backend graphs."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import json
import numpy as np
import pytest
import wavebit


def options(**changes):
    return dict(dict(thickness=[.3],top=1.,bot=1.,period=.1,wavelength=1.,resolution=40,
                     n_periods=30,truncation_order=0,tetm='te',grading=True),**changes)


def test_graded_slab_saves_cells_and_converges_to_airy():
    phase=2*np.pi*1.5*.3;r=-.2
    expected=(1-r*r)*np.exp(1j*phase)/(1-r*r*np.exp(2j*phase))
    errors=[]
    for resolution in (40,80):
        result,diag=_legacy_wavebit_fdtd(structure=np.full((1,1,1),2.25),backend='numpy',
                                 **options(resolution=resolution,return_diagnostics=True))
        errors.append(abs(result.incident.t_s.item()-expected))
        np.testing.assert_allclose(result.R+result.T,1,atol=.004)
        assert np.ptp(diag['z_spacing'])>0
        assert np.min(abs(diag['z_nodes']))<1e-14
        assert np.min(abs(diag['z_nodes']-.3))<1e-14
        assert diag['shape'][0]<.8*(4*resolution+int(np.ceil(.3*resolution))+28)
    assert errors[1]<errors[0]*.5
    assert errors[1]<.003


@pytest.mark.parametrize('backend',['jax','torch'])
def test_graded_scene_material_gradient_and_original_bottom_coordinates(backend):
    kw=options(resolution=30,incidence_side='bottom',return_diagnostics=True,
               record_field=17,record_components=['ey','hx'],field_monitors={'out':.7})
    def run(v,which):
        _,d=_legacy_wavebit_fdtd(structure=np.full((1,1,1),v) if which=='numpy' else v*ones,backend=which,**kw)
        return d['monitors']['out']['power_z'],d['fields']['ey']['values'],d['fields']['ey']['z']
    if backend=='jax':
        jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        ones=jnp.ones((1,1,1))
        power,history,z=jax.jit(lambda v:run(v,'jax'))(2.25)
        derivative=jax.jit(jax.grad(lambda v:run(v,'jax')[0].sum()))(2.25)
        host=np.asarray
    else:
        torch=pytest.importorskip('torch')
        ones=torch.ones((1,1,1),dtype=torch.float64)
        v=torch.tensor(2.25,dtype=torch.float64,requires_grad=True)
        power,history,z=run(v,'torch')
        derivative=torch.autograd.grad(power.sum(),v)[0].detach().numpy()
        host=lambda x:x.detach().numpy()
    expected,expected_history,expected_z=run(2.25,'numpy')
    np.testing.assert_allclose(host(power),expected,atol=1e-11)
    np.testing.assert_allclose(host(history),expected_history,atol=1e-11)
    np.testing.assert_allclose(host(z),expected_z,atol=1e-12)
    assert np.all(np.diff(host(z))>0)
    assert expected.item()>0
    delta=1e-4
    finite=(run(2.25+delta,'numpy')[0].sum()-run(2.25-delta,'numpy')[0].sum())/(2*delta)
    np.testing.assert_allclose(derivative,finite,rtol=1e-5,atol=1e-8)


@pytest.mark.slow
def test_graded_bottom_stack_against_independent_meep_reference():
    ref=json.loads((repository_path('tests/references/golden_scene_bottom_meep.json')).read_text())
    result=_legacy_wavebit_fdtd(structure=np.asarray(ref['epsilon'])[::-1].reshape(-1,1,1),backend='numpy',
                        **options(thickness=ref['thickness'][::-1],wavelength=ref['wavelengths'],
                                  resolution=80,incidence_side='bottom',n_periods=35))
    for key in ('r','t'):
        expected=np.array(ref[key+'_real'])+1j*np.array(ref[key+'_imag'])
        actual=np.array([getattr(v.incident,key+'_s').item() for v in result])
        np.testing.assert_allclose(actual,expected,atol=.003,rtol=0)


def test_graded_conical_magnetic_slab_against_rcwa():
    structure=np.array([[[2.1]],[[3.2]]]);mu=np.array([[[1.1]],[[1.3]]])
    kw=options(thickness=[.13,.21],top=wavebit.Material(1.2,1.1),bot=wavebit.Material(2.,1.3),
               theta=.27,phi=.41,tetm=None,polarization=(1.,.6j),resolution=50)
    result=_legacy_wavebit_fdtd(structure=structure,mu=mu,backend='numpy',**kw)
    reference=_legacy_wavebit_rcwa(structure=structure,mu=mu,backend='numpy',
                           **{k:v for k,v in kw.items() if k not in ('resolution','n_periods','grading')}).solve()
    for name in ('r_s','r_p','t_s','t_p'):
        np.testing.assert_allclose(getattr(result.incident,name),getattr(reference.incident,name),atol=.012)
