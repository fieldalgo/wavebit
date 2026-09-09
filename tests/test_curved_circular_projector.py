"""Curved complex NVF: exact normals, periodic extension and contact derivatives."""
import numpy as np
import pytest
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice
from test_shape_circular_projector import solver

CELL=as_lattice(period=(700.,700.))


def shape(kind='ellipse',**kwargs):
    s=dict(type=kind,lx=300.,material=4.)
    if kind=='ellipse':s['ly']=200.
    return dict(s,**kwargs)


def field(scene,points,cell=CELL,plan=None):
    if plan is None:plan=P.plan_gamma(scene,cell)
    xx,xy,yy,yx=P._piece_field(np,*points,scene,plan,None,cell,np.float64,dyadic=True)
    return np.array([[xx,xy],[yx,yy]])


@pytest.mark.parametrize('kind',['circle','ellipse'])
def test_exact_interface_normals_center_regularity_and_periodicity(kind):
    s=shape(kind,angle=.31)
    co,si=np.cos(.31),np.sin(.31)
    rot=np.array([[co,-si],[si,co]])
    t=np.linspace(0.,2*np.pi,103,endpoint=False)
    a,b=150.,100. if kind=='ellipse' else 150.
    pts=350.+rot@np.array([a*np.cos(t),b*np.sin(t)])
    normals=rot@np.array([np.cos(t)/a,np.sin(t)/b])
    normals/=np.linalg.norm(normals,axis=0)
    got=field([s],pts)
    np.testing.assert_allclose(got,np.einsum('in,jn->ijn',normals,normals),atol=3e-14,rtol=0)
    points=np.array([[350.,350.+1e-6,350.,23.,699.999999,700.000001],
                     [350.,350.,350.+1e-6,72.,199.,199.]])
    got=field([s],points)
    want=np.array([[.5,-.5j],[.5j,.5]])
    np.testing.assert_allclose(got[:,:,:3],np.broadcast_to(want[:,:,None],(2,2,3)),atol=1e-11,rtol=0)
    assert np.max(abs(got[:,:,-1]-got[:,:,-2]))<1e-7
    np.testing.assert_allclose(field([s],points+np.array([[700.],[-1400.]])),got,atol=2e-13,rtol=0)
    np.testing.assert_allclose(np.einsum('ijn,jkn->ikn',got,got),got,atol=3e-15,rtol=0)
    np.testing.assert_allclose(got,np.conj(got.swapaxes(0,1)),atol=0,rtol=0)


def test_extra_periodic_images_are_identically_zero():
    scene=[shape(lx=720.,ly=200.,angle=.17)]
    cell=as_lattice(lattice=((700.,130.),(0.,600.)))
    plan=P.plan_gamma(scene,cell)
    points=np.random.default_rng(17).uniform(0.,700.,(2,41))
    larger=plan._replace(ring=plan.ring+1)
    np.testing.assert_allclose(field(scene,points,cell,larger),field(scene,points,cell,plan),atol=2e-15,rtol=0)


def contact(kind,gap):
    if kind=='period':return [shape('circle',lx=700.-gap,material=12.25)]
    if kind=='ellipse':return [shape(cx=-150.-gap/2),shape(cx=150.+gap/2,material=9.)]
    return [shape('circle',cx=-150.-gap/2),
            dict(type='rectangle',lx=300.,ly=220.,cx=150.+gap/2,material=9.)]


@pytest.mark.parametrize('kind',['period','ellipse','mixed'])
@pytest.mark.parametrize('grid',[63,64,128])
def test_contact_value_and_one_sided_slope(kind,grid):
    def total(g):return solver(contact(kind,g),grid=grid,truncation_order=(2,2)).solve().T.sum()
    base=total(0.)
    large=(total(1e-4)-base)/1e-4
    small=(total(1e-6)-base)/1e-6
    assert np.isfinite(small) and abs(small)>1e-5
    assert small==pytest.approx(large,rel=.01,abs=2e-7)
    assert abs(total(1e-8)-base)<1e-9


@pytest.mark.parametrize('kind',['period','ellipse','mixed'])
def test_contact_frozen_jit_gradient_matches_eager_secants(kind):
    jax=pytest.importorskip('jax');import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    def build(g,b=0):return solver(contact(kind,g),backend=b,grid=32,truncation_order=(1,1))
    geometry=build(0.).freeze_geometry()
    def loss(g):return build(g,1).solve(geometry=geometry).T.sum().real
    value,grad=jax.jit(jax.value_and_grad(loss))(jnp.asarray(0.))
    base=build(0.).solve().T.sum()
    assert float(value)==pytest.approx(base,abs=2e-12)
    for h in (1e-4,1e-5):
        fd=(build(h).solve().T.sum()-base)/h
        assert float(grad)==pytest.approx(fd,rel=3e-4,abs=2e-7)


@pytest.mark.parametrize('backend',[1,2])
def test_rotated_skew_cell_and_precision_match_numpy(backend):
    pytest.importorskip('jax' if backend==1 else 'torch')
    options=dict(lattice=((700.,130.),(0.,600.)),grid=32,truncation_order=(1,1))
    scene=[shape(lx=720.,ly=200.,angle=.17,material=4.+.02j)]
    want=solver(scene,**options).solve()
    s=solver(scene,backend=backend,complex_dtype=1,**options)
    got=s.solve(geometry=s.freeze_geometry())
    np.testing.assert_allclose(np.asarray(got.T),want.T,rtol=0,atol=3e-5)
    np.testing.assert_allclose(np.asarray(got.R),want.R,rtol=0,atol=3e-5)


def test_torch_curve_width_and_material_derivatives():
    torch=pytest.importorskip('torch')
    def build(width,epsilon,backend=0):
        return solver([shape(lx=width,angle=.17,material=epsilon)],backend=backend,grid=32,truncation_order=(1,1))
    width=torch.tensor(310.,dtype=torch.float64,requires_grad=True)
    eps=torch.tensor(4.,dtype=torch.float64,requires_grad=True)
    s=build(width,eps,2)
    derivatives=torch.autograd.grad(s.solve(geometry=s.freeze_geometry()).T.sum().real,(width,eps))
    for i,d in enumerate(derivatives):
        for h in (1e-4,1e-5):
            a=[310.,4.];b=a.copy();a[i]+=h;b[i]-=h
            fd=(build(*a).solve().T.sum()-build(*b).solve().T.sum())/(2*h)
            assert d.item()==pytest.approx(fd,rel=3e-4,abs=2e-7)


def test_dyads_rotate_with_the_cell_and_shape():
    angle=.37
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    cell=as_lattice(lattice=rot@np.diag([700.,700.]))
    points=np.random.default_rng(12).uniform(100.,600.,(2,37))
    want=np.einsum('ij,jkn,lk->iln',rot,field([shape()],points),rot)
    np.testing.assert_allclose(field([shape(angle=angle)],rot@points,cell),want,atol=2e-13,rtol=0)


def test_tiled_curved_field_matches_the_full_grid(monkeypatch):
    import array_api_compat.numpy as xp
    scene=[shape()]
    monkeypatch.setattr(P,'_tiles_pay',lambda *args:False)
    full=P.dyadic_projector_fields(xp,(71,79),scene,xp.float64,CELL)
    monkeypatch.setattr(P,'_tiles_pay',lambda *args:True)
    tiles=P.dyadic_projector_fields(xp,(71,79),scene,xp.float64,CELL)
    for a,b in zip(full,tiles):np.testing.assert_array_equal(a,b)


def test_frozen_curve_can_cross_old_nearest_image_switch():
    initial=[shape(lx=720.,angle=.17)]
    moved=[shape(lx=690.,angle=.17)]
    options=dict(grid=32,truncation_order=(1,1))
    geometry=solver(initial,**options).freeze_geometry()
    got=solver(moved,**options).solve(geometry=geometry)
    want=solver(moved,**options).solve()
    np.testing.assert_allclose(got.T,want.T,atol=2e-13,rtol=0)


def test_curved_painting_freezes_and_solves():
    scene=[shape('circle'),shape('circle',cx=100.)]
    s=solver(scene,grid=32,truncation_order=(1,1))
    geometry=s.freeze_geometry()
    assert geometry.boundary_plans[0] is not None
    result=s.solve(geometry=geometry)
    assert np.isfinite(result.T).all()


def test_disk_alias_uses_the_circle_projector():
    options=dict(grid=32,truncation_order=(1,1))
    a=solver([shape('circle')],**options).solve()
    b=solver([shape('disk')],**options).solve()
    np.testing.assert_array_equal(a.T,b.T)


def test_curved_jitted_fields_match_eager_fields():
    jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
    s=solver([shape()],backend=1,grid=32,truncation_order=(1,1),answers='field')
    geometry=s.freeze_geometry()
    a,af=s.solve_field_eager(3,3,2,geometry=geometry)
    b,bf=jax.jit(lambda:s.solve_field(3,3,2,geometry=geometry))()
    np.testing.assert_allclose(a.T,b.T,atol=2e-12,rtol=0)
    np.testing.assert_allclose(af,bf,atol=2e-11,rtol=0)


@pytest.mark.parametrize('width',[1e-6,1e-9])
def test_small_circle_in_single_precision_has_a_finite_center(width):
    import array_api_compat.numpy as xp
    fields=P.dyadic_projector_fields(xp,(17,17),[shape('circle',lx=width)],xp.float32,CELL)
    for f in fields:assert np.isfinite(f).all()
    xx,xy,yy,yx=fields
    np.testing.assert_allclose(xx+yy,1.,atol=3e-7,rtol=0)
    np.testing.assert_allclose([xx[8,8],xy[8,8],yy[8,8],yx[8,8]],[.5,-.5j,.5,.5j],atol=2e-7,rtol=0)
