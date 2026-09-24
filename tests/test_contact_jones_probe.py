"""Gates for the experimental NVF projector, including unresolved regressions."""
import numpy as np
import pytest
import wavebit
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice
from contact_jones_probe import field_builder, matrices

CELL=as_lattice(period=(700.,700.))


def pair(gap=0.,epsilon=9.):
    return [dict(type='rectangle',lx=300.,ly=300.,cx=-150.-gap/2,material=4.),
            dict(type='rectangle',lx=300.,ly=300.,cx=150.+gap/2,material=epsilon)]


def solver(shapes,grid=128,backend=0,order=3):
    return wavebit.rcwa(backend=backend,period=(700.,700.),wavelength=900.,top=1.,bot=2.25,
        theta=.2,phi=0.,psi=0.,thickness=(300.,),truncation_order=(order,order),
        shape_nvf_grid_size=grid,structure=[dict(background=1.,shapes=shapes)])


def test_projector_is_rank_one_hermitian_and_normal_on_real_interfaces():
    X=np.array([350.,350.,350.,50.,650.,200.,500.,200.])
    Y=np.array([250.,400.,350.,350.,350.,200.,500.,350.])
    scene=pair()
    xx,xy,yx,yy=field_builder()(np,X,Y,scene,P.plan_gamma(scene,CELL),None,CELL,np.float64)
    np.testing.assert_allclose(xx+yy,1.,rtol=0,atol=3e-15)
    np.testing.assert_allclose(xy,np.conj(yx),rtol=0,atol=3e-15)
    np.testing.assert_allclose(xx*yy-xy*yx,0.,rtol=0,atol=3e-15)
    np.testing.assert_allclose(xx[:7],[1.,1.,1.,1.,1.,0.,0.],rtol=0,atol=3e-15)
    assert np.isfinite(xx).all() and np.isfinite(xy).all()


def test_director_lift_rotates_covariantly():
    angle=.37
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    lattice=as_lattice(lattice=rotation@np.diag([700.,700.]))
    scene=[dict(type='rectangle',lx=200.,ly=130.,material=4.)]
    moved=[dict(scene[0],angle=angle)]
    points=np.array([[311.,217.,460.],[250.,310.,440.]])
    transformed=rotation@points
    def tensor(shapes,cell,points):
        xx,xy,yx,yy=field_builder()(np,*points,shapes,P.plan_gamma(shapes,cell),None,cell,np.float64)
        return np.stack([np.stack([xx,xy]),np.stack([yx,yy])])
    original=tensor(scene,CELL,points)
    want=np.einsum('ij,jkn,lk->iln',rotation,original,rotation)
    np.testing.assert_allclose(tensor(moved,lattice,transformed),want,rtol=0,atol=2e-12)


@pytest.mark.parametrize('grid',[127,128,256])
def test_contact_and_material_limits_have_no_sampled_value_step(monkeypatch,grid):
    monkeypatch.setattr(P,'to_conv_mat_parametric_nvf',matrices)
    for kind in ('period','pair','material'):
        def total(value):
            shapes=([dict(type='rectangle',lx=350.,ly=700.-value,material=12.25)]
                    if kind=='period' else pair(epsilon=4.+value) if kind=='material' else pair(value))
            return solver(shapes,grid).solve().T.sum()
        base=total(0.)
        large=(total(1e-4)-base)/1e-4
        small=(total(1e-6)-base)/1e-6
        assert np.isfinite(small) and abs(small)>1e-5
        assert small==pytest.approx(large,rel=.01,abs=1e-7)
        assert abs(total(1e-8)-base)<1e-9


@pytest.mark.parametrize('kind',['period','pair','material'])
def test_frozen_jit_gradient_matches_two_secants(monkeypatch,kind):
    jax=pytest.importorskip('jax');import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    monkeypatch.setattr(P,'to_conv_mat_parametric_nvf',matrices)
    def scene(v):
        return ([dict(type='rectangle',lx=350.,ly=700.-v,material=12.25)]
                if kind=='period' else pair(epsilon=4.+v) if kind=='material' else pair(v))
    center=1e-4
    geometry=solver(scene(center)).freeze_geometry()
    def loss(v):
        return jnp.sum(solver(scene(v),backend=1).solve(geometry=geometry).T).real
    got=float(jax.jit(jax.grad(loss))(jnp.asarray(center)))
    assert np.isfinite(got) and abs(got)>1e-5
    for h in (1e-5,1e-6):
        fd=(solver(scene(center+h)).solve().T.sum()-solver(scene(center-h)).solve().T.sum())/(2*h)
        assert got==pytest.approx(fd,rel=3e-4,abs=2e-7)


@pytest.mark.xfail(strict=True,raises=AssertionError,
    reason='Retained artificial seams still change the finite-order extension')
def test_equal_material_split_is_equivalent_to_one_rectangle(monkeypatch):
    monkeypatch.setattr(P,'to_conv_mat_parametric_nvf',matrices)
    split=solver(pair(epsilon=4.)).solve().T.sum()
    merged=solver([dict(type='rectangle',lx=600.,ly=300.,material=4.)]).solve().T.sum()
    assert split==pytest.approx(merged,rel=0,abs=1e-10)


def test_split_and_merged_material_transforms_are_identical():
    import array_api_compat.numpy as xp
    merged=[dict(type='rectangle',lx=600.,ly=300.,material=4.)]
    a=P.assemble_eps_coeffs(xp,pair(epsilon=4.),1.,3,3,CELL,xp.complex128)
    b=P.assemble_eps_coeffs(xp,merged,1.,3,3,CELL,xp.complex128)
    for got,want in zip(a,b):
        np.testing.assert_allclose(got,want,rtol=0,atol=3e-15)
