"""Full epsilon/mu patterns: independent laminate and Maxwell transfer references.

References below use explicit interface integrals and a matrix exponential;
none calls production Fourier builders, mode solvers, or stack recursions.
"""
import numpy as np
import pytest
from scipy.linalg import expm

import wavebit


def host(a):
    return np.asarray(a.detach().cpu() if hasattr(a, 'detach') else a)


def tensors(loss=0.):
    e = np.array([[3.1,.23+.11j,.31-.07j], [.23-.11j,4.2,-.19+.09j],
                  [.31+.07j,-.19-.09j,2.8]], complex) + 1j*loss*np.eye(3)
    m = np.array([[1.3,.04j,.08],[-.04j,1.1,.03j],[.08,-.03j,1.2]],complex)
    return e,m


def solver(backend=0, factorization='li', **kw):
    formulation=kw.pop("formulation",None)
    e,m = tensors()
    opts = dict(backend=backend, wavelength=730., period=(600.,600.),
                thickness=[73.], truncation_order=(1,1), theta=.23, phi=.19,
                factorization=factorization, top=1., bot=1.,
                structure=[dict(background=1., shapes=[dict(type='rectangle',
                    lx=237.,ly=310.,material=wavebit.Material(e,m))])])
    opts.update(kw)
    if factorization == 'nvf': opts.setdefault('shape_nvf_grid_size',24)
    if 'lattice' in opts: opts.pop('period',None)
    s=wavebit.rcwa(**opts)
    if formulation is not None: s.formulation=formulation
    return s


def laminate_blocks(e, b, axis, h, width=217.3, center=23.1):
    """Exact Fourier integral of D_normal/E_tangent relations on two intervals."""
    delta = np.arange(-h,h+1)[:,None]-np.arange(-h,h+1)[None,:]
    f = width/600.
    indicator = f*np.sinc(delta*f)*np.exp(-2j*np.pi*delta*(center/600.))
    I = np.eye(2*h+1)
    def C(fn): return fn(b)*I+(fn(e)-fn(b))*indicator
    inverse_normal = np.linalg.inv(C(lambda v:1/v[axis,axis]))
    B = [C(lambda v,j=j: v[axis,j]/v[axis,axis]) for j in range(3)]
    Cn = [C(lambda v,i=i: v[i,axis]/v[axis,axis]) for i in range(3)]
    result = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i==j==axis: a=inverse_normal
            elif i==axis: a=inverse_normal@B[j]
            elif j==axis: a=Cn[i]@inverse_normal
            else:
                a=C(lambda v: v[i,j]-v[i,axis]*v[axis,j]/v[axis,axis])+Cn[i]@inverse_normal@B[j]
            result[i][j]=a
    return result


def slot_values(blocks):
    return [blocks[0][0],blocks[1][1],np.linalg.inv(blocks[2][2]),
            -blocks[0][1],-blocks[1][0],-blocks[0][2],-blocks[1][2],
            -blocks[2][0],-blocks[2][1]]


def maxwell_transfer(e, m, kx, ky, distance, wavelength):
    """Eliminate longitudinal fields by solving the two z curl equations.

    Build the differential operator by applying the constitutive and curl
    equations to each transverse basis vector, then exponentiate directly.
    """
    N=len(kx); Kx=np.diag(kx); Ky=np.diag(ky)
    E=np.zeros((3*N,4*N),complex); H=np.zeros_like(E)
    E[:2*N,:2*N]=np.eye(2*N); H[:2*N,2*N:]=np.eye(2*N)
    E[2*N:]=np.linalg.solve(e[2*N:,2*N:],Ky@H[:N]-Kx@H[N:2*N]-e[2*N:,:2*N]@E[:2*N])
    H[2*N:]=np.linalg.solve(m[2*N:,2*N:],Kx@E[N:2*N]-Ky@E[:N]-m[2*N:,:2*N]@H[:2*N])
    D=e@E; B=m@H
    derivative=np.concatenate([Kx@E[2*N:]+B[N:2*N], Ky@E[2*N:]-B[:N],
                               Kx@H[2*N:]-D[N:2*N], Ky@H[2*N:]+D[:N]])
    return expm(2j*np.pi/wavelength*distance*derivative)


def reference_efficiencies(blocks_e, blocks_m, s, side='top'):
    """Vacuum ports; solve all incoming/outgoing plane waves in one linear system."""
    kx,ky=[host(v) for v in s.order_kxy()]
    if kx.ndim==1: kx,ky=np.meshgrid(kx,ky)
    kx,ky=kx.ravel(),ky.ravel(); N=len(kx)
    kz=np.sqrt(1.-kx*kx-ky*ky+0j)
    def modes(sign):
        fields=[]; flux=[]
        for pol in range(2):
            for j in range(N):
                rho=np.sqrt(kx[j]**2+ky[j]**2)
                sx,sy=(kx[j]/rho,ky[j]/rho) if abs(rho)>1e-12 else (1.,0.)
                k=np.array([kx[j],ky[j],sign*kz[j]])
                E=np.array([-sy,sx,0.],complex) if pol==0 else np.array([sx*k[2],sy*k[2],-rho])
                H=np.cross(k,E)
                v=np.zeros(4*N,complex)
                for index,a in enumerate((E[0],E[1],H[0],H[1])): v[index*N+j]=a
                fields.append(v);flux.append(.5*np.real(E[0]*H[1].conjugate()-E[1]*H[0].conjugate()))
        return np.array(fields).T,np.array(flux)
    plus,fp=modes(1); minus,fm=modes(-1)
    P=maxwell_transfer(np.block(blocks_e),np.block(blocks_m),kx,ky,float(host(s.thickness)[0]),float(s.wavelength))
    out=[]
    for pol in range(2):
        j=pol*N+N//2
        if side=='top':
            rt=np.linalg.solve(np.concatenate([-plus,P@minus],axis=1),minus[:,j]); fr,ft,fi=fp,-fm,-fm[j]
        else:
            rt=np.linalg.solve(np.concatenate([-P@minus,plus],axis=1),P@plus[:,j]); fr,ft,fi=-fm,fp,fp[j]
        r,t=rt[:2*N],rt[2*N:]
        out.append((np.sum(fr*abs(r)**2)/fi,np.sum(ft*abs(t)**2)/fi))
    return out


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('factorization',['li','nvf'])
@pytest.mark.parametrize('loss',[0.,.13])
@pytest.mark.parametrize('side',['top','bottom'])
@pytest.mark.parametrize('algorithm',['ETM','SMM'])
def test_full_laminate_against_independent_integrals_and_transfer(axis,factorization,loss,side, algorithm):
    e,m=tensors(loss); bg=np.diag([1.4,1.7,1.6]).astype(complex); bm=np.diag([1.05,1.1,1.08]).astype(complex)
    shape=dict(type='rectangle',lx=217.3 if axis==0 else 600.,ly=217.3 if axis==1 else 600.,
               cx=23.1 if axis==0 else 0.,cy=23.1 if axis==1 else 0.,material=wavebit.Material(e,m))
    s=solver(layer_algorithm=algorithm,factorization=factorization,truncation_order=(1,0) if axis==0 else (0,1),
             structure=[dict(background=wavebit.Material(bg,bm),shapes=[shape])],incidence_side=side)
    eb=laminate_blocks(e,bg,axis,1); mb=laminate_blocks(m,bm,axis,1)
    for actual,expected in ((s.eps_conv_matrices(),eb),(s.mu_conv_matrices(),mb)):
        assert len(actual)==9
        for a,b in zip(actual,slot_values(expected)): np.testing.assert_allclose(a[0],b,atol=3e-12,rtol=0)
    r=s.solve(auto_detune=False)
    assert s.formulation_assigned=='first_order'
    for inc,(R,T) in zip((r.s_inc,r.p_inc),reference_efficiencies(eb,mb,s,side)):
        np.testing.assert_allclose([inc.R.sum(),inc.T.sum()],[R,T],atol=3e-11,rtol=0)
        if loss==0: assert R+T==pytest.approx(1.,abs=3e-12)
        else: assert R+T<1.


@pytest.mark.parametrize('backend',[0,1,2])
@pytest.mark.parametrize('factorization',['li','nvf'])
def test_full_pattern_solve_and_field_across_backends(backend,factorization):
    s=solver(backend,factorization,answers='field')
    reference=solver(0,factorization,answers='field')
    solve_field=getattr(s,'solve_field_eager',s.solve_field)
    r,f=solve_field(3,3,3,geometry=s.freeze_geometry())
    want,fw=reference.solve_field(3,3,3,geometry=reference.freeze_geometry())
    for inc in ('s_inc','p_inc'):
        for key in ('r_s','r_p','t_s','t_p'):
            np.testing.assert_allclose(host(getattr(getattr(r,inc),key)),getattr(getattr(want,inc),key),atol=3e-11,rtol=0)
    np.testing.assert_allclose(host(f),fw,atol=5e-10,rtol=0)


@pytest.mark.parametrize('factorization',['li','nvf'])
@pytest.mark.parametrize('quantity',['epsilon','mu'])
def test_uniform_full_tensor_raster_matches_exact_film(factorization,quantity):
    e,m=tensors(); e,m=(e,np.eye(3)) if quantity=='epsilon' else (np.eye(3)*2.,m)
    shape=(1,3,4,3,3)
    kw={} if factorization=='li' else dict(shape_nvf_grid_size=None)
    r=solver(factorization=factorization,structure=np.broadcast_to(e,shape),mu=np.broadcast_to(m,shape),**kw).solve(auto_detune=False)
    exact=solver(factorization=factorization,structure=[dict(background=wavebit.Material(e,m),shapes=[])]).solve(auto_detune=False)
    for inc in ('s_inc','p_inc'):
        for key in ('r_s','r_p','t_s','t_p'):
            np.testing.assert_allclose(getattr(getattr(r,inc),key),getattr(getattr(exact,inc),key),atol=2e-11,rtol=0)


@pytest.mark.parametrize('side',['top','bottom'])
def test_magnetic_halfspaces_forced_first_order_agrees_with_second_order(side):
    mat=wavebit.Material([2.3,3.1,2.9],mu=[1.3,1.2,1.1]).rotated(.23)
    options=dict(top=wavebit.Material(1.6,1.2),bot=wavebit.Material(2.2,1.4),
                 structure=[dict(background=mat,shapes=[])],incidence_side=side)
    r=solver(formulation='first_order',**options).solve(auto_detune=False)
    want=solver(formulation='second_order',**options).solve(auto_detune=False)
    for inc in ('s_inc','p_inc'):
        for key in ('r_s','r_p','t_s','t_p'):
            np.testing.assert_allclose(getattr(getattr(r,inc),key),getattr(getattr(want,inc),key),atol=2e-12,rtol=0)


@pytest.mark.parametrize('factorization',['li','nvf'])
@pytest.mark.parametrize('backend',[1,2])
@pytest.mark.parametrize('angle',[0.,.27])
@pytest.mark.parametrize('algorithm',['ETM','SMM'])
def test_full_tensor_tilt_gradients_include_zero_coupling(factorization,backend,angle, algorithm):
    # At angle=0 the tensor is z-decoupled, but its z-coupling tangent is not.
    frozen=solver(factorization=factorization).freeze_geometry()
    def loss(a,b=backend):
        mat=wavebit.Material([2.4,3.2,4.1],mu=[1.1,1.3,1.4]).rotated(a,'y').rotated(.21)
        s=solver(b,factorization,layer_algorithm=algorithm,truncation_order=(1,1),
                 structure=[dict(background=1.,shapes=[dict(type='rectangle',lx=237.,ly=310.,material=mat)])])
        return s.solve(geometry=frozen,auto_detune=False).p_inc.R.sum()
    delta=2e-5
    fd=(float(loss(angle+delta,0))-float(loss(angle-delta,0)))/(2*delta)
    if backend==1:
        import jax
        import jax.numpy as jnp
        value,gradient=jax.jit(jax.value_and_grad(loss))(jnp.asarray(angle))
    else:
        import torch
        x=torch.tensor(angle,dtype=torch.float64,requires_grad=True)
        value=loss(x);gradient,=torch.autograd.grad(value,x)
    assert np.isfinite(host(value))
    assert float(host(gradient))==pytest.approx(fd,abs=3e-7,rel=2e-4)


@pytest.mark.parametrize('factorization',['li','nvf'])
@pytest.mark.parametrize('side',['top','bottom'])
def test_full_tensor_electric_magnetic_duality(factorization,side):
    e,m=tensors(.07)
    def run(a,b):
        return solver(factorization=factorization,incidence_side=side,
                      structure=[dict(background=1.,shapes=[dict(type='rectangle',lx=237.,ly=310.,
                          material=wavebit.Material(a,b))])]).solve(auto_detune=False)
    r=run(e,m);dual=run(m,e)
    for inc,other in ((r.s_inc,dual.p_inc),(r.p_inc,dual.s_inc)):
        np.testing.assert_allclose(inc.R,other.R,atol=2e-11,rtol=0)
        np.testing.assert_allclose(inc.T,other.T,atol=2e-11,rtol=0)


def test_sampled_full_tensor_converges_to_integrated_shape_on_oblique_cell():
    e,m=tensors(.08)
    options=dict(lattice=wavebit.Lattice.oblique(600.,600.,np.pi/3),
                 structure=[dict(background=1.,shapes=[dict(type='circle',lx=280.,material=wavebit.Material(e,m))])])
    def powers(r): return np.array([r.s_inc.R.sum(),r.s_inc.T.sum(),r.p_inc.R.sum(),r.p_inc.T.sum()])
    # The two routes use different normal extensions. Refine Fourier order as
    # well as the sampling grid; grid refinement alone cannot make them equal.
    errors=[]
    for h in (1,5):
        exact=solver(factorization='nvf',shape_nvf_grid_size=128,
                     truncation_order=(h,h),**options).solve(auto_detune=False)
        sampled=solver(factorization='nvf',shape_nvf_grid_size=256,
                       fourier_coefficients='sampled',truncation_order=(h,h),**options).solve(auto_detune=False)
        errors.append(np.max(abs(powers(sampled)-powers(exact))))
    assert errors[-1]<errors[0]/3
    assert errors[-1]<8e-4


def test_z_coupling_in_only_mu_selects_first_order_and_survives_backside():
    _,m=tensors()
    shape=dict(type='rectangle',lx=237.,ly=310.,material=wavebit.Material(2.3,m))
    for side in ('top','bottom'):
        s=solver(structure=[dict(background=1.,shapes=[shape])],incidence_side=side)
        assert len(s.eps_conv_matrices())==5
        assert len(s.mu_conv_matrices())==9
        r=s.solve(auto_detune=False)
        assert s.formulation_assigned=='first_order'
        assert float(r.R.sum()+r.T.sum())==pytest.approx(1.,abs=2e-12)
        s.formulation='second_order'
        with pytest.raises(ValueError,match='mu'):s.solve(auto_detune=False)


@pytest.mark.parametrize('factorization',['li','nvf'])
@pytest.mark.parametrize('algorithm',['ETM','SMM'])
def test_jitted_full_tensor_field_reconstruction(factorization, algorithm):
    s=solver(1,factorization,answers='field',layer_algorithm=algorithm)
    frozen=s.freeze_geometry()
    eager=s.solve_field_eager(2,2,2,geometry=frozen)
    compiled=s.solve_field(2,2,2,geometry=frozen)
    np.testing.assert_allclose(host(eager[1]),host(compiled[1]),atol=2e-10,rtol=0)


def test_full_li_and_nvf_approach_same_two_dimensional_result():
    # Independent factorizations converge rather than agreeing at a fixed low order.
    results={}
    for rule in ('li','nvf'):
        rows=[]
        for h in (1,7):
            kw={} if rule=='li' else dict(shape_nvf_grid_size=64)
            r=solver(factorization=rule,truncation_order=(h,h),**kw).solve(auto_detune=False)
            rows.append(np.array([r.s_inc.R.sum(),r.s_inc.T.sum()]))
        results[rule]=rows
    coarse=np.max(abs(results['li'][0]-results['nvf'][0]))
    fine=np.max(abs(results['li'][1]-results['nvf'][1]))
    assert fine<coarse/5
    assert fine<2e-4


@pytest.mark.parametrize('side',['top','bottom'])
@pytest.mark.parametrize('algorithm',['ETM','SMM'])
def test_full_magnetic_field_satisfies_both_maxwell_curls(side, algorithm):
    e,m=tensors(.03)
    s=solver(truncation_order=(0,0),incidence_side=side,layer_algorithm=algorithm,
             structure=[dict(background=wavebit.Material(e,m),shapes=[])],answers='field')
    z=np.linspace(1.,72.,81)
    _,field=s.solve_field(1,1,z,incidences=('s','p'))
    kx,ky=(complex(host(k).ravel()[0]) for k in s.order_kxy()); k0=2*np.pi/float(s.wavelength)
    dz=z[1]-z[0]
    for Fz in field[:,:,0,0,:]:
        # Fourth-order central differences of independently sampled public-z fields.
        derivative=(Fz[:-4]-8*Fz[1:-3]+8*Fz[3:-1]-Fz[4:])/(12*dz)
        value=Fz[2:-2]; E,H=value[:,:3],value[:,3:]; dE,dH=derivative[:,:3],derivative[:,3:]
        def curl(v,d):
            return np.stack([1j*k0*ky*v[:,2]-d[:,1],d[:,0]-1j*k0*kx*v[:,2],
                             1j*k0*(kx*v[:,1]-ky*v[:,0])],axis=-1)
        residual=max(np.max(abs(curl(E,dE)-1j*k0*(H@m.T))),
                     np.max(abs(curl(H,dH)+1j*k0*(E@e.T))))
        assert residual/(k0*np.max(abs(value)))<2e-8


@pytest.mark.parametrize('factorization',['li','nvf'])
@pytest.mark.parametrize('backend',[1,2])
def test_full_tensor_rectangle_geometry_derivative(factorization,backend):
    e,m=tensors(.04)
    def make(width,b):
        return solver(b,factorization,structure=[dict(background=1.,shapes=[
            dict(type='rectangle',lx=width,ly=310.,material=wavebit.Material(e,m))])])
    frozen=make(237.,0).freeze_geometry()
    def loss(width,b=backend):
        return make(width,b).solve(geometry=frozen,auto_detune=False).s_inc.R.sum()
    step=2e-3
    fd=float((loss(237.+step,0)-loss(237.-step,0))/(2*step))
    if backend==1:
        import jax
        import jax.numpy as jnp
        derivative=jax.jit(jax.grad(loss))(jnp.asarray(237.))
    else:
        import torch
        width=torch.tensor(237.,dtype=torch.float64,requires_grad=True)
        derivative,=torch.autograd.grad(loss(width),width)
    assert float(host(derivative))==pytest.approx(fd,abs=2e-8,rel=2e-4)


def test_ambiguous_mode_partition_is_invalid_under_jit():
    import jax
    import jax.numpy as jnp
    from wavebit._core.solve_2d_first_order import split_forward_backward
    invalid=1j*jnp.eye(4,dtype=jnp.complex128)
    result=jax.jit(lambda a:split_forward_backward(jnp,a,1))(invalid)
    assert all(not np.isfinite(host(a)).any() for a in result)


def test_coarse_nvf_refuses_ambiguous_modes_and_refinement_recovers():
    # A deterministic tilted positive-definite material where a low-order PN
    # operator perturbs modal flux enough to make its direction ambiguous.
    rng=np.random.default_rng(451)
    for _ in range(11):
        q,_=np.linalg.qr(rng.normal(size=(3,3)));p,_=np.linalg.qr(rng.normal(size=(3,3)))
        e=q@np.diag(rng.uniform(1.2,6.,3))@q.T;m=p@np.diag(rng.uniform(.8,1.8,3))@p.T
        theta,phi=rng.uniform(.05,.8),rng.uniform(0,3.)
        lx,ly=rng.uniform(180,360),rng.uniform(180,360)
    s=solver(factorization='nvf',theta=theta,phi=phi,shape_nvf_grid_size=32,
             structure=[dict(background=1.,shapes=[dict(type='rectangle',lx=lx,ly=ly,
                 material=wavebit.Material(e,m))])])
    with pytest.raises(ValueError,match='Increase the Fourier order'):
        s.solve(auto_detune=False)
    s.truncation_order=(3,3)
    r=s.solve(auto_detune=False)
    assert np.isfinite(r.R).all() and np.isfinite(r.T).all()
    assert abs(float(r.R.sum()+r.T.sum())-1.)<5e-6


@pytest.mark.parametrize('backend',[0,1,2])
@pytest.mark.parametrize('factorization',['li','nvf'])
def test_full_tensor_single_precision(backend,factorization):
    s=solver(backend,factorization,complex_dtype=1,answers='field')
    solve_field=getattr(s,'solve_field_eager',s.solve_field)
    result,field=solve_field(2,2,2,geometry=s.freeze_geometry())
    ref,ref_field=solver(0,factorization,answers='field').solve_field(2,2,2)
    assert field.dtype==s.xp.complex64
    np.testing.assert_allclose(host(field),ref_field,atol=2e-5,rtol=2e-5)
    for inc in ('s_inc','p_inc'):
        for name in ('R','T'):
            np.testing.assert_allclose(host(getattr(getattr(result,inc),name)),
                                       getattr(getattr(ref,inc),name),atol=4e-6,rtol=2e-5)
