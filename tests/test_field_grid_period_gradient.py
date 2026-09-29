"""A vacuum plane wave fixes the derivative of count-based sample positions."""
import numpy as np
import pytest
import wavebit

P=np.array([700.,850.]);THETA=.31;PHI=.43;WL=900.;D=170.;Z=63.
KXY=2*np.pi/WL*np.sin(THETA)*np.array([np.cos(PHI),np.sin(PHI)])


def samples(kind,count):
    return (count if kind!='y' else [-27.,11.],count if kind!='x' else [-17.,21.])


def analytic(kind,count,door):
    sx,sy=samples(kind,count)
    x=np.linspace(-.5,.5,count)*P[0] if isinstance(sx,int) else np.asarray(sx)
    y=np.linspace(-.5,.5,count)*P[1] if isinstance(sy,int) else np.asarray(sy)
    dx=np.linspace(-.5,.5,count) if isinstance(sx,int) else np.zeros(len(x))
    dy=np.linspace(-.5,.5,count) if isinstance(sy,int) else np.zeros(len(y))
    zphase=0. if door=='boundary' else 2*np.pi/WL*np.cos(THETA)*(D-Z)
    ey=np.cos(PHI)*np.exp(1j*(KXY[0]*x[None,:]+KXY[1]*y[:,None]+zphase))
    weights=np.arange(1,ey.size+1).reshape(ey.shape)/ey.size
    value=np.real(np.sum(weights*ey))
    grad=np.array([np.real(np.sum(weights*ey*1j*KXY[0]*dx[None,:])),
                   np.real(np.sum(weights*ey*1j*KXY[1]*dy[:,None]))])
    return value,grad


@pytest.mark.parametrize('backend',['jax','torch'])
@pytest.mark.parametrize('door',['legacy','result','boundary'])
@pytest.mark.parametrize('kind',['x','y','both'])
@pytest.mark.parametrize('count',[1,3])
def test_period_tangent_includes_motion_of_count_samples(backend,door,kind,count):
    s=wavebit.rcwa(backend=1 if backend=='jax' else 2,structure=np.ones((1,1,1)),
                  thickness=[D],period=P,wavelength=WL,theta=THETA,phi=PHI,
                  truncation_order=(0,0),answers='field')
    sx,sy=samples(kind,count)
    def loss(p,xp):
        # Keep solve and legacy reconstruction in their shared scope.
        if door=='legacy':
            _,f=s.solve_field_eager(sx,sy,[Z],period=p) if backend=='jax' else s.solve_field(sx,sy,[Z],period=p)
            ey=f[0,0,...,1]
        else:
            r=s.solve(period=p,auto_detune=False)
            f=r.boundary_fields('top',sx,sy,part='incoming') if door=='boundary' else r.reconstruct_field(sx,sy,[Z])[0]
            ey=f[...,1]
        weights=xp.asarray(np.arange(1,ey.size+1).reshape(ey.shape)/ey.size) if backend=='jax' else xp.arange(1,ey.numel()+1,dtype=xp.float64).reshape(ey.shape)/ey.numel()
        return xp.real(xp.sum(weights*ey))
    if backend=='jax':
        jax=pytest.importorskip('jax')
        import jax.numpy as jnp
        wavebit.enable_x64()
        if kind!='both' and door!='legacy':
            with pytest.raises(ValueError,match='skewed or traced lattice'):
                jax.value_and_grad(lambda p:loss(p,jnp))(jnp.asarray(P))
            return
        value,grad=jax.value_and_grad(lambda p:loss(p,jnp))(jnp.asarray(P))
        value,grad=float(value),np.asarray(grad)
    else:
        torch=pytest.importorskip('torch')
        p=torch.tensor(P,dtype=torch.float64,requires_grad=True)
        value=loss(p,torch);value.backward()
        value,grad=float(value.detach()),p.grad.numpy()
    expected,expected_grad=analytic(kind,count,door)
    assert value==pytest.approx(expected,abs=2e-12)
    np.testing.assert_allclose(grad,expected_grad,atol=2e-12,rtol=2e-10)
