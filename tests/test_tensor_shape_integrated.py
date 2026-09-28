"""Continuous shape integration, independent series and geometry derivatives."""
import numpy as np
import pytest

import wavebit
from wavebit._core.backend import array_namespace
from wavebit._core.tensor_shapes import weighted_coefficients, _frequencies


def host(a):
    return np.asarray(a.detach().cpu() if hasattr(a,'detach') else a)


def solver(backend=0, width=217., shape='circle', n=24, **kw):
    options = dict(wavelength=730.,period=(600.,600.),thickness=[173.],
                   truncation_order=(1,1),shape_nvf_grid_size=n,theta=.23,phi=.19,
                   structure=[dict(background=1.,shapes=[dict(type=shape,lx=width,
                       **({'ly':600.} if shape=='rectangle' else {}),
                       material=wavebit.Material([2.25,4.,2.8]).rotated(.31))])])
    options.update(kw)
    return wavebit.rcwa(backend=backend,**options)


@pytest.mark.parametrize('n',[7,8,17,18])
def test_weighted_fourier_interpolant_against_gauss_integral(n):
    # Integrate an independently written trigonometric field over a box,
    # including the even-grid Nyquist sine, on half-pixel sample locations.
    xp = array_namespace(np.array(0.))
    x=(np.arange(n)+.5)/n
    X,Y=np.meshgrid(x,x)
    def field(x,y):
        return 1.2+np.cos(2*np.pi*(x+2*y))+.3j*np.sin(2*np.pi*y)+.2*np.sin(2*np.pi*(n//2)*x)
    q=_frequencies(xp,n,1,xp.float64)
    ky,kx=q[:,None],q[None,:]
    cx,cy,dx,dy=.43,.61,.37,.26
    F=dx*dy*np.sinc(kx*dx)*np.sinc(ky*dy)*np.exp(-2j*np.pi*(kx*cx+ky*cy))
    actual=weighted_coefficients(xp,field(X,Y),np.fft.fft2(np.fft.ifftshift(F)),1,1)
    nodes,weights=np.polynomial.legendre.leggauss(70)
    XX,YY=np.meshgrid(cx+dx/2*nodes,cy+dy/2*nodes)
    W=weights[:,None]*weights[None,:]*dx*dy/4
    expected=np.array([[np.sum(W*field(XX,YY)*np.exp(-2j*np.pi*(k*XX+l*YY)))
                       for k in range(-2,3)] for l in range(-2,3)])
    np.testing.assert_allclose(actual,expected,atol=1e-14)


@pytest.mark.parametrize('n',[15,24])
def test_stripe_matches_exact_tensor_pn_series(n):
    # A stripe's normal is x. Write PN's five scalar functions explicitly;
    # their Fourier series follow directly from the interval endpoints.
    e=np.array([[3.1,.23+.11j,0.],[.23-.11j,4.2,0.],[0.,0.,2.8]])
    bg=np.array([[1.7,.12,0.],[.12,2.1,0.],[0.,0.,1.9]])
    width=217.3;h=2
    s=solver(width=width,shape='rectangle',n=n,truncation_order=(h,0),
             structure=[dict(background=wavebit.Material(bg),shapes=[dict(
                 type='rectangle',lx=width,ly=600.,material=wavebit.Material(e))])])
    orders=np.arange(-h,h+1);delta=orders[:,None]-orders[None,:]
    F=width/600*np.sinc(delta*width/600)
    I=np.eye(2*h+1)
    def integrate(a,b): return b*I+(a-b)*F
    def ab(t):
        return ([[1.,0.],[t[1,0]/t[0,0],t[1,1]-t[1,0]*t[0,1]/t[0,0]]],
                [[1/t[0,0],-t[0,1]/t[0,0]],[0.,1.]])
    ae,be=ab(e);ag,bg_=ab(bg)
    def block(a,b): return np.block([[integrate(a[i][j],b[i][j]) for j in range(2)] for i in range(2)])
    effective=block(ae,ag)@np.linalg.inv(block(be,bg_));m=2*h+1
    expected=[effective[:m,:m],effective[m:,m:],np.linalg.inv(integrate(e[2,2],bg[2,2])),
              -effective[:m,m:],-effective[m:,:m]]
    assert s.fourier_coefficients=='integrated'
    for actual,want in zip(s.eps_conv_matrices(),expected):
        np.testing.assert_allclose(actual[0],want,atol=2e-12)


@pytest.mark.parametrize('backend',[0,1,2])
def test_default_integrated_and_explicit_sampled(backend):
    integrated=solver(backend=backend).solve(auto_detune=False)
    expected=solver().solve(auto_detune=False)
    for name in ['r_s','r_p','t_s','t_p']:
        np.testing.assert_allclose(host(getattr(integrated.s_inc,name)),getattr(expected.s_inc,name),atol=2e-11)
    sampled=solver(backend=backend,fourier_coefficients='sampled')
    assert sampled.fourier_coefficients=='sampled'
    assert np.isfinite(host(sampled.solve(auto_detune=False).R)).all()


@pytest.mark.parametrize('shape',['circle','ellipse','rectangle'])
def test_rigid_translation_is_covariant(shape):
    shapes=[dict(type=shape,lx=211.,**({} if shape=='circle' else dict(ly=157.,angle=.17)),
                 material=wavebit.Material([3.,4.,2.5]))]
    s=solver(structure=[dict(background=1.,shapes=shapes)])
    base=s.eps_conv_matrices()
    dx,dy=41.7,-33.2
    moved=solver(structure=[dict(background=1.,shapes=[dict(shapes[0],cx=dx,cy=dy)])]).eps_conv_matrices()
    u,v=np.meshgrid(np.arange(-1,2),np.arange(-1,2));u=u.ravel();v=v.ravel()
    phase=np.exp(-2j*np.pi*((u[:,None]-u[None,:])*dx+(v[:,None]-v[None,:])*dy)/600)
    for a,b in zip(moved,base):np.testing.assert_allclose(a,b*phase,atol=3e-12)


@pytest.mark.parametrize('backend',[1,2])
def test_shape_width_gradient_survives_a_subpixel_change(backend):
    frozen=solver(backend=backend).freeze_geometry()
    def f(w):
        return solver(backend=backend,width=w).solve(geometry=frozen,auto_detune=False).R.sum()
    if backend==1:
        import jax
        derivative=jax.jit(jax.grad(f))(217.)
    else:
        import torch
        w=torch.tensor(217.,dtype=torch.float64,requires_grad=True)
        derivative,=torch.autograd.grad(f(w),w)
    expected=(host(f(217.001))-host(f(216.999)))/.002
    assert abs(expected)>1e-5
    np.testing.assert_allclose(host(derivative),expected,rtol=3e-5,atol=1e-9)


def test_painted_rectangles_have_exact_area_and_holes():
    shapes=[dict(type='rectangle',lx=310.,ly=270.,material=wavebit.Material([2.,3.,4.])),
            dict(type='rectangle',lx=120.,ly=140.,cx=40.,material=wavebit.Material([5.,6.,7.]))]
    s=solver(structure=[dict(background=1.,shapes=shapes)],truncation_order=(0,0),n=17)
    epszz=1.+(4.-1.)*310*270/600**2+(7.-4.)*120*140/600**2
    assert s.eps_conv_matrices().epz_i[0,0,0]==pytest.approx(1/epszz,abs=2e-13)


def test_equal_trace_tensor_contact_is_not_erased():
    shapes=[dict(type='rectangle',cx=-100.,lx=200.,ly=600.,material=wavebit.Material([2.,4.,3.])),
            dict(type='rectangle',cx=100.,lx=200.,ly=600.,material=wavebit.Material([4.,2.,3.]))]
    s=solver(structure=[dict(background=1.,shapes=shapes)],truncation_order=(1,0))
    # At zero Fourier order a laminate has harmonic epsilon_x and arithmetic epsilon_y.
    zero=solver(structure=s.structure,truncation_order=(0,0)).eps_conv_matrices()
    assert zero.epx[0,0,0]==pytest.approx(1/(1/3+1/6+1/12),abs=2e-12)
    assert zero.epy[0,0,0]==pytest.approx(7/3,abs=2e-12)


def test_uniform_tensor_identity_with_artificial_shapes():
    e=wavebit.Material([2.,3.,4.]).rotated(.23)
    shapes=[dict(type='circle',lx=170.,cx=-100.,material=e),
            dict(type='rectangle',lx=130.,ly=100.,cx=140.,material=e)]
    actual=solver(structure=[dict(background=e,shapes=shapes)]).eps_conv_matrices()
    expected=solver(structure=[dict(background=e,shapes=[])]).eps_conv_matrices()
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=2e-12)


def test_frozen_topology_and_coefficient_mode_are_checked():
    from wavebit._core.geometry import StaleGeometryError
    s=solver();frozen=s.freeze_geometry()
    with pytest.raises(StaleGeometryError):
        solver(n=32).eps_conv_matrices(geometry=frozen)
    with pytest.raises(StaleGeometryError):
        solver(fourier_coefficients='sampled').eps_conv_matrices(geometry=frozen)
    sampled=solver(fourier_coefficients='sampled').freeze_geometry()
    with pytest.raises(StaleGeometryError):s.eps_conv_matrices(geometry=sampled)


def test_magnetic_tensor_uses_its_own_region_values():
    dielectric=wavebit.Material([2.25,4.,2.8]).rotated(.31)
    magnetic=wavebit.Material(1.,mu=dielectric.epsilon)
    a=solver(structure=[dict(background=1.,shapes=[dict(type='circle',lx=217.,material=dielectric)])])
    b=solver(structure=[dict(background=wavebit.Material(1.,1.),shapes=[dict(type='circle',lx=217.,material=magnetic)])])
    for e,m in zip(a.eps_conv_matrices(),b.mu_conv_matrices()):np.testing.assert_allclose(e,m,atol=2e-12)


@pytest.mark.parametrize('backend',[0,1,2])
def test_painted_curve_agrees_across_backends(backend):
    shapes=[dict(type='rectangle',lx=240.,ly=220.,material=wavebit.Material([2.,3.,4.])),
            dict(type='circle',lx=180.,cx=90.,cy=30.,material=wavebit.Material([3.,4.,2.]))]
    def build(b):
        return solver(backend=b,structure=[dict(background=1.,shapes=shapes)],n=12).eps_conv_matrices()
    expected=build(0)
    for a,b in zip(build(backend),expected):np.testing.assert_allclose(host(a),b,atol=3e-12)


@pytest.mark.parametrize('parameter,value',[('x',40.),('angle',.1),('period',600.),('epsilon',4.2)])
def test_painted_shape_and_cell_gradients(parameter,value):
    import jax
    import jax.numpy as jnp
    def make(v):
        p=dict(x=40.,angle=.1,period=600.,epsilon=4.2);p[parameter]=v
        shapes=[dict(type='rectangle',lx=270.,ly=210.,angle=p['angle'],
                     material=wavebit.Material([2.,3.,4.]).rotated(.2)),
                dict(type='rectangle',lx=130.,ly=140.,cx=p['x'],cy=50.,
                     material=wavebit.Material(jnp.stack([p['epsilon'],p['epsilon']*0+2.,p['epsilon']*0+3.])))]
        return solver(backend=1,structure=[dict(background=1.,shapes=shapes)],
                      period=(p['period'],600.),n=12)
    frozen=make(value).freeze_geometry()
    def f(v):return make(v).solve(geometry=frozen,auto_detune=False).R.sum()
    actual=jax.jit(jax.grad(f))(value)
    h=1e-5
    expected=(host(f(value+h))-host(f(value-h)))/(2*h)
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=1e-9)


def test_frozen_disjoint_shapes_refuse_new_overlap():
    from wavebit._core.geometry import StaleGeometryError
    def make(x):
        shapes=[dict(type='rectangle',cx=-120.,lx=120.,ly=180.,material=wavebit.Material([2.,3.,4.])),
                dict(type='rectangle',cx=x,lx=120.,ly=180.,material=wavebit.Material([3.,4.,5.]))]
        return solver(structure=[dict(background=1.,shapes=shapes)],n=12)
    frozen=make(120.).freeze_geometry()
    with pytest.raises(StaleGeometryError):make(-60.).eps_conv_matrices(geometry=frozen)


def test_integrated_geometry_does_not_call_the_material_rasterizer(monkeypatch):
    from wavebit._core import anisotropy
    def unexpected(*args,**kwargs):raise AssertionError('material rasterization is not an integrated path')
    monkeypatch.setattr(anisotropy,'_shape_snapshot',unexpected)
    assert np.isfinite(solver().solve(auto_detune=False).R).all()


def test_periodic_self_overlap_is_refused_by_integration_but_sampled_is_available():
    with pytest.raises(ValueError,match='periodic|image|period'):
        solver(width=900.).solve(auto_detune=False)
    assert np.isfinite(solver(width=900.,fourier_coefficients='sampled').solve(auto_detune=False).R).all()
