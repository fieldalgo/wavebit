"""Maxwell length-scaling invariance constrains simultaneous geometry gradients."""
import numpy as np
import pytest
import wavebit


def conditions(scale,family):
    eps=wavebit.Material(np.diag([4.+.07j,4.3+.09j,4.6+.11j])) if family=='tensor-li' else 4.+.07j
    return dict(wavelength=900.*scale,period=[600.*scale,750.*scale],thickness=[170.*scale],
                structure=[dict(background=1.,shapes=[dict(type='rectangle',lx=130.*scale,
                    ly=180.*scale,cx=23.*scale,cy=-17.*scale,material=eps)])])


def make(backend,family,scale=1.):
    return wavebit.rcwa(backend=backend,truncation_order=(1,1),theta=.17,phi=.29,
                       polarization=(.8,.6j),answers='field',factorization='nvf' if family=='scalar-nvf' else 'li',
                       shape_nvf_grid_size=16 if family=='scalar-nvf' else None,**conditions(scale,family))


def objective(s,xp,scale,family,geometry=None):
    r=s.solve(**conditions(scale,family),geometry=geometry,auto_detune=False)
    f=r.reconstruct_field(xp.reshape(27.*scale,(1,)),xp.reshape(-35.*scale,(1,)),xp.reshape(63.*scale,(1,)))
    return r.T.sum().real+.07*xp.sum(xp.abs(f)**2)+.11*xp.real(xp.sum(f))


@pytest.mark.parametrize('backend',['jax','jit','torch'])
@pytest.mark.parametrize('family',['scalar-li','scalar-nvf','tensor-li'])
def test_all_lengths_scale_without_changing_the_optical_problem(backend,family):
    pytest.importorskip('torch' if backend=='torch' else 'jax')
    baseline=float(objective(make(0,family),np,np.asarray(1.),family))
    for scale in (.37,1.8):
        value=float(objective(make(0,family,scale),np,np.asarray(scale),family))
        assert value==pytest.approx(baseline,abs=3e-11,rel=3e-11)
    s=make(1 if backend!='torch' else 2,family)
    if backend in ('jax','jit'):
        import jax
        import jax.numpy as jnp
        wavebit.enable_x64()
        geometry=s.freeze_geometry() if backend=='jit' else None
        calculate=jax.value_and_grad(lambda scale:objective(s,jnp,scale,family,geometry))
        if backend=='jit':calculate=jax.jit(calculate)
        with jax.checking_leaks():value,gradient=map(float,calculate(jnp.asarray(1.)))
    else:
        import torch
        scale=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
        result=objective(s,torch,scale,family);result.backward()
        value,gradient=float(result.detach()),float(scale.grad)
    assert value==pytest.approx(baseline,abs=3e-11,rel=3e-11)
    assert abs(gradient)<3e-10
