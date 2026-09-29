"""The retry policy must inspect current conditions, not a previous graph."""
import warnings
import numpy as np
import pytest
import wavebit

PILLAR=np.ones((1,20));PILLAR[:,6:14]=12.25
STRUCTURE=np.stack([np.full((1,20),2.25),PILLAR])
CFG=dict(backend=2,period=[1000.],truncation_order=(3,0),top=1.21,bot=1.69,
         theta=0.,phi=0.,psi=np.pi/2,structure=STRUCTURE,answers='field')


@pytest.mark.parametrize('algorithm',['ETM','SMM'])
def test_previous_torch_runtime_does_not_select_an_autodiff_retry_for_a_concrete_scene(algorithm):
    torch=pytest.importorskip('torch')
    s=wavebit.rcwa(**CFG,layer_algorithm=algorithm,wavelength=520.,
                   thickness=torch.tensor([300.,200.],dtype=torch.float64,requires_grad=True))
    s.solve()
    assert s.T1.requires_grad
    s.thickness=[300.,200.]
    s.wavelength=500.
    fresh=wavebit.rcwa(**CFG,layer_algorithm=algorithm,wavelength=500.,thickness=[300.,200.])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got=s.solve()
        expected=fresh.solve()
    print(algorithm,complex(s.wavelength),complex(fresh.wavelength),flush=True)
    assert complex(s.wavelength)==complex(fresh.wavelength)
    np.testing.assert_allclose(got.T,expected.T,rtol=0,atol=1e-12)
