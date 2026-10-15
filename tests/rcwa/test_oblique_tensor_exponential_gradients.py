"""Independent oblique Maxwell boundary problem and parameter gradients."""

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest
from scipy.linalg import expm

ROOT=np.array([0.,0.,.27,.41,170.,900.,27.,-35.,.43])
DE=np.array([[.2,.31+.07j,.27],[.31-.07j,-.1,.19+.03j],[.27,.19-.03j,.4]])+.04j*np.eye(3)
DM=np.array([[.07,-.03+.01j,.04],[-.03-.01j,.02,.06],[.04,.06,-.02]])+.003j*np.eye(3)
E=np.array([[2.8+.08j,.11+.06j,.13-.03j],[.11-.06j,3.4+.1j,-.08+.05j],[.13+.03j,-.08-.05j,3.1+.07j]])
M=np.array([[1.2+.015j,.03+.02j,-.02], [.03-.02j,1.4+.025j,.04+.01j],[-.02,.04-.01j,1.1+.02j]])
WEIGHT=np.array([.7+.2j,-.3+.1j,.4-.17j,.2+.5j,-.11+.23j,.31+.19j])
POL=np.array([.8,.6j])


def base(profile):
    return (4*np.eye(3,dtype=complex),np.eye(3,dtype=complex)) if profile=='isotropic' else (E,M)


def maxwell(e,m,kx,ky):
    electric=np.zeros((3,4),complex);magnetic=np.zeros((3,4),complex)
    electric[:2,:2]=np.eye(2);magnetic[:2,2:]=np.eye(2)
    electric[2]=(ky*magnetic[0]-kx*magnetic[1]-e[2,:2]@electric[:2])/e[2,2]
    magnetic[2]=(kx*electric[1]-ky*electric[0]-m[2,:2]@magnetic[:2])/m[2,2]
    d=e@electric;b=m@magnetic
    return np.array([kx*electric[2]+b[1],ky*electric[2]-b[0],kx*magnetic[2]-d[1],ky*magnetic[2]+d[0]])


def reference(v,profile,side):
    e,m=base(profile);e=e+v[0]*DE;m=m+v[1]*DM
    kx,ky=np.sin(v[2])*np.array([np.cos(v[3]),np.sin(v[3])]);kz=np.cos(v[2])
    shat=np.array([-np.sin(v[3]),np.cos(v[3]),0.])
    def modes(sign):
        k=np.array([kx,ky,sign*kz]);s=shat;p=np.cross(s,k)
        return np.array([np.r_[a[:2],np.cross(k,a)[:2]] for a in (s,p)]).T
    plus,minus=modes(1),modes(-1)
    k0=2*np.pi/v[5]
    lower=expm(1j*k0*maxwell(E,M,kx,ky)*90.)
    upper=expm(1j*k0*maxwell(e,m,kx,ky)*v[4])
    total=upper@lower
    if side=='top':
        coeff=np.linalg.solve(np.concatenate([total@minus,-plus],axis=1),minus@POL)
        bottom=minus@coeff[:2]
    else:
        # Public p has its in-plane electric part opposite k_parallel on
        # both sides, hence the sign relative to s cross k for bottom input.
        coeff=np.linalg.solve(np.concatenate([-total@minus,plus],axis=1),total@plus@(POL*np.array([1.,-1.])))
        bottom=plus@(POL*np.array([1.,-1.]))+minus@coeff[:2]
    state=expm(1j*k0*maxwell(e,m,kx,ky)*v[8]*v[4])@lower@bottom
    ez=(ky*state[2]-kx*state[3]-e[2,:2]@state[:2])/e[2,2]
    hz=(kx*state[1]-ky*state[0]-m[2,:2]@state[2:])/m[2,2]
    return np.r_[state[:2],ez,state[2:],hz]*np.exp(1j*k0*(kx*v[6]+ky*v[7]))


def objective(xp,f):
    return xp.real(xp.sum(f*xp.asarray(WEIGHT)))+.13*xp.sum(xp.abs(f)**2)


def make(backend,profile,side,algorithm):
    e,m=base(profile)
    s=_legacy_wavebit_rcwa(backend=backend,structure=np.stack([E,e])[:,None,None],mu=np.stack([M,m])[:,None,None],
        thickness=[90.,ROOT[4]],period=(600.,750.),wavelength=ROOT[5],truncation_order=(0,0),theta=ROOT[2],phi=ROOT[3],
        polarization=POL,incidence_side=side,answers='field',factorization='li',layer_algorithm=algorithm)
    s.formulation='first_order';return s


def actual(s,xp,v,profile,geometry=None):
    e,m=base(profile)
    eps=xp.stack([xp.asarray(E),xp.asarray(e)+v[0]*xp.asarray(DE)])[:,None,None]
    mu=xp.stack([xp.asarray(M),xp.asarray(m)+v[1]*xp.asarray(DM)])[:,None,None]
    d=xp.stack([xp.asarray(90.),v[4]])
    r=s.solve(structure=eps,mu=mu,thickness=d,theta=v[2],phi=v[3],wavelength=v[5],geometry=geometry,auto_detune=False)
    return r.reconstruct_field(v[6:7],v[7:8],(90.+v[8:9]*v[4]))[0,0,0]


@pytest.mark.parametrize('backend',['jax','jit','torch'])
@pytest.mark.parametrize('profile',['isotropic','full'])
@pytest.mark.parametrize('side',['top','bottom'])
@pytest.mark.parametrize('algorithm',['ETM','SMM'])
def test_oblique_field_and_nine_parameter_gradient_match_maxwell_boundary_problem(backend,profile,side,algorithm):
    pytest.importorskip('torch' if backend=='torch' else 'jax')
    want=reference(ROOT,profile,side)
    got=actual(make('numpy',profile,side,algorithm),np,ROOT,profile)
    np.testing.assert_allclose(got,want,rtol=3e-11,atol=3e-11)
    s=make('torch' if backend=='torch' else 'jax',profile,side,algorithm)
    if backend in ('jax','jit'):
        import jax
        import jax.numpy as jnp
        _legacy_wavebit_enable_x64()
        geometry=s.freeze_geometry() if backend=='jit' else None
        calculation=jax.value_and_grad(lambda v:objective(jnp,actual(s,jnp,v,profile,geometry)))
        if backend=='jit':calculation=jax.jit(calculation)
        with jax.checking_leaks():value,grad=calculation(jnp.asarray(ROOT))
        value,grad=float(value),np.asarray(grad)
    else:
        import torch
        v=torch.tensor(ROOT,dtype=torch.float64,requires_grad=True)
        result=objective(torch,actual(s,torch,v,profile));result.backward()
        value,grad=float(result.detach()),v.grad.numpy()
    assert value==pytest.approx(objective(np,want),abs=3e-11)
    expected=[]
    for j in range(len(ROOT)):
        delta=1e-5*max(1.,abs(ROOT[j]));plus=ROOT.copy();minus=ROOT.copy();plus[j]+=delta;minus[j]-=delta
        expected.append((objective(np,reference(plus,profile,side))-objective(np,reference(minus,profile,side)))/(2*delta))
    np.testing.assert_allclose(grad,expected,rtol=4e-5,atol=2e-8)
