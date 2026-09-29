"""Independent Maxwell boundary-value oracle for material and local-field tangents."""
import numpy as np
import pytest
from scipy.linalg import expm
import wavebit

DE = np.array([[.2,.31,.27],[.31,-.1,.19],[.27,.19,.4]])
DM = np.array([[.07,-.03,.04],[-.03,.02,.06],[.04,.06,-.02]])
FIXED_E = np.array([[2.8+.08j,.11,.13],[.11,3.4+.1j,-.08],[.13,-.08,3.1+.07j]])
FIXED_M = np.array([[1.2+.015j,.03,-.02],[.03,1.4+.025j,.04],[-.02,.04,1.1+.02j]])
WEIGHT = np.array([.7+.2j,-.3+.1j,.4-.17j,.2+.5j,-.11+.23j,.31+.19j])


def tensors(profile):
    if profile == 'isotropic':
        return np.eye(3, dtype=complex)*4., np.eye(3, dtype=complex)
    return FIXED_E.copy(), FIXED_M.copy()


def operator(eps, mu):
    e = eps[:2,:2]-eps[:2,2,None]*eps[None,2,:2]/eps[2,2]
    m = mu[:2,:2]-mu[:2,2,None]*mu[None,2,:2]/mu[2,2]
    return np.array([[0,0,m[1,0],m[1,1]], [0,0,-m[0,0],-m[0,1]],
                     [-e[1,0],-e[1,1],0,0], [e[0,0],e[0,1],0,0]], dtype=complex)


def independent_field(t, profile, multilayer):
    e, m = tensors(profile)
    layers = [(e+t*DE, m+t*DM, 170.)]
    if multilayer:
        layers.insert(0,(FIXED_E,FIXED_M,90.))
    k0=2*np.pi/900.
    plus=np.array([[1,0],[0,1],[0,-1],[1,0]],complex)
    minus=np.array([[1,0],[0,1],[0,1],[-1,0]],complex)
    total=np.eye(4,dtype=complex)
    for eps,mu,d in layers:
        total=expm(1j*k0*operator(eps,mu)*d)@total
    # No incidence from below; a unit y-polarized wave enters at the top.
    amplitudes=np.linalg.solve(np.concatenate([total@minus,-plus],axis=1),minus[:,1])
    state=minus@amplitudes[:2]
    for eps,mu,d in layers[:-1]:
        state=expm(1j*k0*operator(eps,mu)*d)@state
    eps,mu,d=layers[-1]
    state=expm(1j*k0*operator(eps,mu)*(.43*d))@state
    ez=-eps[2,:2]@state[:2]/eps[2,2]
    hz=-mu[2,:2]@state[2:]/mu[2,2]
    return np.array([state[0],state[1],ez,state[2],state[3],hz])


def objective(xp, field):
    return xp.real(xp.sum(field*xp.asarray(WEIGHT)))+.13*xp.sum(xp.abs(field)**2)


def solver(backend, profile, multilayer):
    e,m=tensors(profile)
    eps=np.stack([FIXED_E,e] if multilayer else [e])[:,None,None,...]
    mu=np.stack([FIXED_M,m] if multilayer else [m])[:,None,None,...]
    d=[90.,170.] if multilayer else [170.]
    s=wavebit.rcwa(backend=backend, structure=eps, mu=mu, thickness=d,
                   period=(600.,750.), wavelength=900., truncation_order=(0,0),
                   theta=0., phi=0., answers='field', factorization='li')
    s.formulation='4ff'
    return s


def actual(s,xp,t,profile,multilayer,geometry=None):
    e,m=tensors(profile)
    eps=xp.asarray(e)+t*xp.asarray(DE)
    mu=xp.asarray(m)+t*xp.asarray(DM)
    eps=xp.stack([xp.asarray(FIXED_E),eps] if multilayer else [eps])[:,None,None,...]
    mu=xp.stack([xp.asarray(FIXED_M),mu] if multilayer else [mu])[:,None,None,...]
    r=s.solve(structure=eps,mu=mu,geometry=geometry)
    return r.reconstruct_field([0.],[0.],[(90. if multilayer else 0.)+.43*170.])[0,0,0]


@pytest.mark.parametrize('backend',['jax','jit','torch'])
@pytest.mark.parametrize('profile',['isotropic','full'])
@pytest.mark.parametrize('multilayer',[False,True])
def test_field_and_joint_constitutive_derivative_match_independent_maxwell_exponential(backend,profile,multilayer):
    reference=independent_field(0.,profile,multilayer)
    np.testing.assert_allclose(actual(solver(0,profile,multilayer),np,0.,profile,multilayer),reference,atol=2e-12,rtol=2e-12)
    s=solver(2 if backend=='torch' else 1,profile,multilayer)
    if backend in ('jax','jit'):
        jax=pytest.importorskip('jax')
        import jax.numpy as jnp
        wavebit.enable_x64()
        geometry=s.freeze_geometry() if backend=='jit' else None
        calculation=jax.value_and_grad(lambda t: objective(jnp,actual(s,jnp,t,profile,multilayer,geometry)))
        if backend=='jit': calculation=jax.jit(calculation)
        with jax.checking_leaks():
            value,gradient=map(float,calculation(jnp.asarray(0.)))
    else:
        torch=pytest.importorskip('torch')
        t=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        result=objective(torch,actual(s,torch,t,profile,multilayer))
        result.backward()
        value,gradient=float(result.detach()),float(t.grad)
    h=1e-4
    expected=(objective(np,independent_field(h,profile,multilayer))-objective(np,independent_field(-h,profile,multilayer)))/(2*h)
    assert value==pytest.approx(float(objective(np,reference)),abs=2e-12)
    assert abs(expected)>1e-4
    assert gradient==pytest.approx(float(expected),rel=2e-5,abs=2e-8)
