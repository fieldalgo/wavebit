"""Complex NVF follows the polygon boundaries used by curved paintings."""
import numpy as np
import pytest
import array_api_compat.numpy as xp
import wavebit
from wavebit._core import boundary as B, parametric as P, clipper
from test_shape_circular_projector import solver

CELL=((700.,0.),(0.,700.))


def shapes(width=300.,epsilon=9.,mu=1.,shift=(0.,0.)):
    x,y=shift
    return [dict(type='ellipse',lx=width,ly=200.,angle=.17,cx=x,cy=y,material=4.),
            dict(type='circle',lx=250.,cx=130.+x,cy=20.+y,
                 material=wavebit.Material(epsilon,mu=mu))]


def build(scene=None,backend=0,**kwargs):
    return solver(shapes() if scene is None else scene,backend=backend,grid=32,
                  truncation_order=(1,1),phi=.13,psi=.23,**kwargs)


def projector(scene,points,cell=CELL,ring_extra=0,chunk=None,corrected=True):
    plan=P.plan_gamma(scene,cell)
    verts=[B.shape_vertices(xp,s,cell) for s in scene]
    options={}
    if corrected:
        def reference(p):
            return xp.stack(P._piece_field(xp,p[:,0],p[:,1],scene,plan,None,cell,xp.float64,
                                           dyadic=True),axis=-1)
        options=dict(reference=reference,curved_edges=[
            s.get('type') in ('circle','disk','ellipse') for s,v in zip(scene,verts)
            for _ in range(len(v))])
    return B.polygon_dyadic_projector(xp,xp.asarray(points),verts,cell,xp.float64,
                                     plan.reduction,plan.ring+ring_extra,chunk=chunk,**options)


@pytest.mark.parametrize('corrected',[False,True])
def test_exact_normals_of_resolved_polygon_interfaces(corrected):
    scene=shapes()
    verts=[B.shape_vertices(xp,s,CELL) for s in scene]
    recs,loops,transform,interfaces=B.plan_layer(scene,CELL)
    vals=clipper.eval_table(xp,recs,verts)
    regs=[(s['material'],[clipper.loop_verts(xp,lp,vals) for lp in ll])
          for s,ll in zip(scene,loops)]
    A,E,_,live=B.replay_interfaces(xp,regs,1.,CELL,transform,interfaces)
    E=E[live];A=A[live]
    points=A+.413*E  # Avoid segment endpoints and the undefined corner normal.
    n=np.stack([E[:,1],-E[:,0]],axis=-1)/np.linalg.norm(E,axis=1)[:,None]
    got=projector(scene,points,corrected=corrected)
    want=np.stack([n[:,0]**2,n[:,0]*n[:,1],n[:,1]**2,n[:,0]*n[:,1]],axis=-1)
    np.testing.assert_allclose(got,want,atol=3e-12,rtol=0)
    # A facet normal differs from the smooth curve normal except at its midpoint.
    assert len(points)>256


def test_periodicity_ring_independence_and_chunking():
    pts=np.random.default_rng(8).uniform(-200.,900.,(31,2))
    base=projector(shapes(),pts)
    for got in (projector(shapes(),pts+[1400.,-700.]),
                projector(shapes(),pts,ring_extra=1),projector(shapes(),pts,chunk=3)):
        np.testing.assert_allclose(got,base,atol=3e-12,rtol=0)
    xx,xy,yy,yx=base.T
    np.testing.assert_allclose(xx+yy,1.,atol=3e-15,rtol=0)
    np.testing.assert_allclose(xy,np.conj(yx),atol=0,rtol=0)
    np.testing.assert_allclose(xx*yy-xy*yx,0.,atol=3e-15,rtol=0)


def test_geometry_projector_does_not_switch_at_material_equality():
    points=np.random.default_rng(12).uniform(0.,700.,(29,2))
    base=projector(shapes(epsilon=4.),points)
    for eps in (4.+1e-12,9.,9.+.2j):
        np.testing.assert_array_equal(projector(shapes(epsilon=eps),points),base)
    center=build(shapes(epsilon=4.)).solve().T.sum()
    slopes=[(build(shapes(epsilon=4.+h)).solve().T.sum()-center)/h for h in (1e-4,1e-6)]
    assert slopes[1]==pytest.approx(slopes[0],rel=.01,abs=1e-7)
    assert abs(slopes[1])>1e-5


def test_public_build_uses_the_polygon_projector_coefficients():
    scene=shapes()
    s=build(scene)
    g=s.freeze_geometry()
    expected=B.to_conv_mat_boundary_layer(xp,scene,1.,1,1,CELL,xp.complex128,
        normal_grid=32,plan=g.boundary_plans[0],projector='circular',
        projector_plan=g.gamma_plans[0])
    for a,b in zip(s._eps_conv_matrices(g),expected):
        np.testing.assert_allclose(a[0],b,atol=2e-12,rtol=0)


@pytest.mark.parametrize('backend',[1,2])
def test_lossy_magnetic_painting_in_single_precision_matches_numpy(backend):
    pytest.importorskip('jax' if backend==1 else 'torch')
    scene=shapes(epsilon=9.+.04j,mu=1.2+.01j)
    options=dict(lattice=((700.,140.),(0.,600.)))
    want=build(scene,**options).solve()
    s=build(scene,backend=backend,complex_dtype=1,**options)
    g=s.freeze_geometry();got=s.solve(geometry=g)
    np.testing.assert_allclose(np.asarray(got.T),want.T,atol=3e-5,rtol=0)
    np.testing.assert_allclose(np.asarray(got.R),want.R,atol=3e-5,rtol=0)
    assert all(m.dtype==s.xp.complex64 for m in s._eps_conv_matrices(g))


def test_frozen_jit_geometry_and_material_derivatives_match_eager_differences():
    jax=pytest.importorskip('jax');import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    geometry=build().freeze_geometry()
    def loss(p):return build(shapes(p[0],p[1]),backend=1).solve(geometry=geometry).T.sum().real
    point=jnp.array([300.,9.])
    value,gradient=jax.jit(jax.value_and_grad(loss))(point)
    assert float(value)==pytest.approx(build().solve().T.sum(),abs=3e-12)
    for i,got in enumerate(gradient):
        assert np.isfinite(got) and abs(got)>1e-7
        for h in (1e-4,1e-5):
            a=np.array(point);b=a.copy();a[i]+=h;b[i]-=h
            fd=(build(shapes(*a)).solve().T.sum()-build(shapes(*b)).solve().T.sum())/(2*h)
            assert float(got)==pytest.approx(fd,rel=3e-4,abs=3e-7)


def test_torch_geometry_and_permeability_derivatives():
    torch=pytest.importorskip('torch')
    width=torch.tensor(300.,dtype=torch.float64,requires_grad=True)
    mu=torch.tensor(1.2,dtype=torch.float64,requires_grad=True)
    s=build(shapes(width,mu=mu),backend=2)
    got=torch.autograd.grad(s.solve(geometry=s.freeze_geometry()).T.sum().real,(width,mu))
    for i,g in enumerate(got):
        assert np.isfinite(g.item()) and abs(g.item())>1e-7
        for h in (1e-4,1e-5):
            a=[300.,1.2];b=a.copy();a[i]+=h;b[i]-=h
            fd=(build(shapes(a[0],mu=a[1])).solve().T.sum()-
                build(shapes(b[0],mu=b[1])).solve().T.sum())/(2*h)
            assert g.item()==pytest.approx(fd,rel=3e-4,abs=3e-7)


def test_rigid_translation_keeps_public_diffraction_efficiencies():
    want=build().solve()
    for shift in ((.137,17.29),(1400.137,-682.71)):
        got=build(shapes(shift=shift)).solve()
        np.testing.assert_allclose(got.T,want.T,atol=2e-11,rtol=0)
        np.testing.assert_allclose(got.R,want.R,atol=2e-11,rtol=0)


def test_frozen_painting_rejects_changed_intersections():
    from wavebit._core.geometry import StaleGeometryError
    g=build().freeze_geometry()
    with pytest.raises(StaleGeometryError):
        build(shapes(width=50.)).solve(geometry=g)


def test_polygon_refinement_approaches_the_analytic_contact_solve():
    # This isolates polygon approximation at fixed Fourier order. The original
    # polygon-only extension jumped by ~.016 from the analytic extension here.
    scene=[dict(type='ellipse',lx=300.,ly=200.,cx=-150.,material=4.),
           dict(type='ellipse',lx=300.,ly=200.,cx=150.,material=9.)]
    s=solver(scene,grid=64,truncation_order=(2,2))
    target=s.solve().T.sum()
    errors=[]
    for count in (64,128,256,512):
        mats=B.to_conv_mat_boundary_layer(xp,scene,1.,2,2,CELL,xp.complex128,
            normal_grid=64,n_poly=count,projector='circular')
        result=s.solve_conv(*[xp.stack([m]) for m in mats])
        errors.append(abs(result.T.sum()-target))
    # Geometry error and sampled-projector error can cancel at an intermediate
    # polygon count. Require convergence over the ladder, not monotonic totals.
    # Both routes sample with the first ellipse at the centre of the cell since
    # 2026-09-11 (they anchored on the scene's centre before, which on this
    # symmetric scene was the cell centre); the ladder reads 4.4e-5, 7.2e-6,
    # 1.35e-6, 1.0e-6 there, a floor twice the old one and still 1e4 below the
    # 1.6e-2 the polygon-only extension jumped by.
    assert max(errors[-2:])<2e-6, errors
    assert errors[-1]<errors[0]/32, errors


def test_corrected_projector_rotates_covariantly():
    angle=.37
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    cell=rotation@np.asarray(CELL)
    scene=shapes()
    moved=[]
    for s in scene:
        cx,cy=rotation@np.array([s.get('cx',0.),s.get('cy',0.)])
        # A circle's angle rotates its polygon chords, which is relevant here.
        moved.append(dict(s,cx=cx,cy=cy,angle=s.get('angle',0.)+angle))
    pts=np.random.default_rng(7).uniform(200.,550.,(19,2))
    def tensor(values):
        xx,xy,yy,yx=values.T
        return np.array([[xx,xy],[yx,yy]])
    want=np.einsum('ij,jkn,lk->iln',rotation,tensor(projector(scene,pts)),rotation)
    got=tensor(projector(moved,pts@rotation.T,cell))
    np.testing.assert_allclose(got,want,atol=3e-12,rtol=0)


def test_jitted_painted_fields_match_eager():
    jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
    s=build(backend=1,answers='field');geometry=s.freeze_geometry()
    a,af=s.solve_field_eager(3,3,2,geometry=geometry)
    b,bf=jax.jit(lambda:s.solve_field(3,3,2,geometry=geometry))()
    np.testing.assert_allclose(a.T,b.T,atol=3e-12,rtol=0)
    np.testing.assert_allclose(af,bf,atol=3e-11,rtol=0)
