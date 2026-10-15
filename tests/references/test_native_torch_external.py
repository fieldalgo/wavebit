"""Torch executes the shared core against existing independent FDTDX goldens."""

from wavebit.fdtd._periodic.grid import fdtd_grid as _legacy_wavebit_fdtd_grid
from pathlib import Path

import numpy as np
import pytest



@pytest.mark.parametrize('case',['base','ade','bloch'])
def test_torch_fields_and_gradient_against_external_fdtdx(case):
    torch=pytest.importorskip('torch')
    shape=(9,7,5);rng=np.random.default_rng(2419)
    def field():
        values=rng.normal(size=(3,)+shape)
        if case=='bloch':
            values=values+1j*rng.normal(size=(3,)+shape)
        return torch.as_tensor(values*.1)
    electric,magnetic=field(),field()
    pattern=torch.as_tensor(rng.uniform(.1,.9,size=(3,)+shape))
    mu=torch.as_tensor(rng.uniform(1.,1.7,size=(3,)+shape))
    parameter=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    epsilon=2+(.2 if case=='ade' else parameter)*pattern
    poles=() if case!='ade' else ((0.,.3,.7*(1+parameter*pattern)),(1.2,.2,.8*(1+parameter*pattern)))
    out=_legacy_wavebit_fdtd_grid(epsilon,spacing=(1.,)*3,dt=.2,steps=53 if case=='ade' else 37,
                         mu=mu,electric=electric,magnetic=magnetic,poles=poles,backend='torch',
                         bloch_phase=(.3,-.2,.4) if case=='bloch' else (0.,0.,0.),checkpoint_steps=8)
    value=(pattern*out['E'].abs()**2).sum()+.3*(out['H'][:,2:5,1:4,1:3].abs()**2).sum()
    derivative=torch.autograd.grad(value,parameter)[0]
    filenames={'base':'golden_native_grid_external.npz','ade':'golden_native_material_external.npz',
               'bloch':'golden_native_bloch_external.npz'}
    with np.load(Path(__file__).with_name(filenames[case]),allow_pickle=False) as ref:
        prefix='' if case=='bloch' else 'fdtdx_'
        for key in ('E','H'):
            expected=ref[prefix+key]
            if case=='bloch':
                expected=expected+1j*ref[key+'_imag']
            np.testing.assert_allclose(out[key].detach().numpy(),expected,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(value.detach(),ref[prefix+'value'],rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(derivative.detach(),ref[prefix+'gradient'],rtol=1e-10,atol=1e-12)
