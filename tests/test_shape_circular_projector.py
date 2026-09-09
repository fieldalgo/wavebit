"""Production complex NVF for rectangles: contact continuity and live gradients."""
import numpy as np
import pytest
import wavebit
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice
from contact_jones_probe import matrices as independent_matrices


def pair(gap=0.,epsilon=9.):
    return [dict(type='rectangle',lx=300.,ly=300.,cx=-150.-gap/2,material=4.),
            dict(type='rectangle',lx=300.,ly=300.,cx=150.+gap/2,material=epsilon)]


def solver(shapes,backend=0,grid=128,**kwargs):
    options=dict(backend=backend,period=(700.,700.),wavelength=900.,top=1.,bot=2.25,
        theta=.2,phi=0.,psi=0.,thickness=(300.,),truncation_order=(3,3),
        normal_grid=grid,projector='circular',factorization='nvf',
        structure=[dict(background=1.,shapes=shapes)])
    options.update(kwargs)
    if 'lattice' in kwargs:
        options.pop('period',None)
    return wavebit.rcwa(**options)


@pytest.mark.parametrize('grid',[127,128,256])
def test_production_contact_and_material_limits(grid):
    for kind in ('period','pair','material'):
        def total(value):
            shapes=([dict(type='rectangle',lx=350.,ly=700.-value,material=12.25)]
                    if kind=='period' else pair(epsilon=4.+value) if kind=='material' else pair(value))
            s=solver(shapes,grid=grid)
            assert s.factorization=='nvf' and s.projector=='circular'
            return s.solve().T.sum()
        base=total(0.)
        large=(total(1e-4)-base)/1e-4
        small=(total(1e-6)-base)/1e-6
        assert np.isfinite(small) and abs(small)>1e-5
        assert small==pytest.approx(large,rel=.01,abs=1e-7)
        assert abs(total(1e-8)-base)<1e-9


@pytest.mark.parametrize('kind',['period','pair','material'])
def test_frozen_jit_derivative_matches_two_finite_differences(kind):
    jax=pytest.importorskip('jax');import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    def shapes(v):
        return ([dict(type='rectangle',lx=350.,ly=700.-v,material=12.25)]
                if kind=='period' else pair(epsilon=4.+v) if kind=='material' else pair(v))
    center=1e-4;geometry=solver(shapes(center)).freeze_geometry()
    def loss(v):
        return jnp.sum(solver(shapes(v),backend=1).solve(geometry=geometry).T).real
    got=float(jax.jit(jax.grad(loss))(jnp.asarray(center)))
    assert np.isfinite(got) and abs(got)>1e-5
    for h in (1e-5,1e-6):
        fd=(solver(shapes(center+h)).solve().T.sum()-solver(shapes(center-h)).solve().T.sum())/(2*h)
        assert got==pytest.approx(fd,rel=3e-4,abs=2e-7)


def test_same_coefficients_as_independent_jones_instrument():
    shapes=pair(1.)
    s=solver(shapes)
    want=independent_matrices(s.xp,shapes,1.,3,3,s.lattice,s.complex_dtype,normal_grid=128)
    got=s._eps_conv_matrices(s.freeze_geometry())
    for a,b in zip(got,want):
        np.testing.assert_allclose(a[0],b,rtol=0,atol=2e-12)


def test_both_projectors_have_separate_cache_entries_and_default_is_unchanged():
    s=solver(pair(1.));P._PROJECTOR_CACHE.clear()
    linear=s.solve(projector='linear').T.copy()
    circular=s.solve(projector='circular').T.copy()
    np.testing.assert_array_equal(s.solve(projector='linear').T,linear)
    np.testing.assert_array_equal(s.solve(projector='circular').T,circular)
    plain=solver(pair(1.),projector=None)
    np.testing.assert_array_equal(plain.solve().T,linear)
    assert np.max(abs(linear-circular))>1e-6


def test_tiled_dyads_equal_full_grid(monkeypatch):
    import array_api_compat.numpy as xp
    shapes=pair();cell=as_lattice(period=(700.,700.));plan=P.plan_gamma(shapes,cell)
    monkeypatch.setattr(P,'_tiles_pay',lambda *args:False)
    want=P.dyadic_projector_fields(xp,(73,81),shapes,xp.float64,cell,plan)
    monkeypatch.setattr(P,'_tiles_pay',lambda *args:True)
    got=P.dyadic_projector_fields(xp,(73,81),shapes,xp.float64,cell,plan)
    for a,b in zip(got,want):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('backend',[0,1,2])
def test_painted_magnetic_rectangles_and_single_precision(backend):
    if backend:
        pytest.importorskip('jax' if backend==1 else 'torch')
    shapes=[dict(type='rectangle',lx=300.,ly=250.,angle=.1,material=wavebit.Material(4.+.02j,mu=1.1)),
            dict(type='rectangle',lx=180.,ly=150.,cx=30.,angle=.2,material=wavebit.Material(7.+.04j,mu=1.2))]
    want=solver(shapes,truncation_order=(1,1),grid=32).solve()
    s=solver(shapes,backend=backend,truncation_order=(1,1),grid=32,complex_dtype=1)
    g=s.freeze_geometry();got=s.solve(geometry=g)
    for a,b in ((got.R,want.R),(got.T,want.T)):
        np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=0,atol=2e-5)
    assert all(m.dtype==s.xp.complex64 for m in s._eps_conv_matrices(g))


def test_square_alias_uses_the_same_rectangle_projector():
    a=solver([dict(type='square',lx=200.,material=4.)]).solve()
    b=solver([dict(type='rectangle',lx=200.,ly=200.,material=4.)]).solve()
    np.testing.assert_array_equal(a.T,b.T)


def test_jitted_fields_through_public_api():
    jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
    s=solver(pair(1.),backend=1,grid=32,truncation_order=(1,1),answers='field')
    g=s.freeze_geometry()
    a,af=s.solve_field_eager(3,3,2,geometry=g)
    b,bf=s.solve_field(3,3,2,geometry=g)
    np.testing.assert_allclose(b.T,a.T,rtol=0,atol=2e-11)
    np.testing.assert_allclose(bf,af,rtol=0,atol=2e-10)


@pytest.mark.parametrize('kind',['period','pair'])
def test_frozen_contact_can_open_and_retains_one_sided_jit_derivative(kind):
    jax=pytest.importorskip('jax');import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    def shapes(gap):
        return ([dict(type='rectangle',lx=350.,ly=700.-gap,material=12.25)]
                if kind=='period' else pair(gap))
    geometry=solver(shapes(0.)).freeze_geometry()
    for gap in (0.,1e-6,1e-4,1.):
        s=solver(shapes(gap))
        np.testing.assert_allclose(s.solve(geometry=geometry).T,s.solve().T,rtol=0,atol=2e-12)
    def loss(gap):
        return jnp.sum(solver(shapes(gap),backend=1).solve(geometry=geometry).T).real
    derivative=float(jax.jit(jax.grad(loss))(jnp.asarray(0.)))
    assert np.isfinite(derivative) and abs(derivative)>1e-5
    zero=solver(shapes(0.)).solve().T.sum()
    for h in (1e-4,1e-5):
        fd=(solver(shapes(h)).solve().T.sum()-zero)/h
        assert derivative==pytest.approx(fd,rel=3e-4,abs=2e-7)


def test_circular_record_still_rejects_a_new_overlap():
    from wavebit._core.factorization import StaleGeometryError
    geometry=solver(pair(1.)).freeze_geometry()
    with pytest.raises(StaleGeometryError,match='overlap'):
        solver(pair(-1.)).solve(geometry=geometry)


def test_circular_record_rejects_changed_shape_count():
    from wavebit._core.factorization import StaleGeometryError
    geometry=solver(pair()).freeze_geometry()
    with pytest.raises(StaleGeometryError,match='image plan'):
        solver(pair()[:1]).solve(geometry=geometry)


@pytest.mark.parametrize('backend',[1,2])
def test_skew_cell_and_rotated_rectangle_agree_across_backends(backend):
    pytest.importorskip('jax' if backend==1 else 'torch')
    if backend==1:
        import jax;jax.config.update('jax_enable_x64',True)
    shapes=[dict(type='rectangle',lx=210.,ly=160.,angle=.31,material=4.+.03j)]
    options=dict(lattice=((700.,130.),(0.,600.)),grid=32,truncation_order=(1,1))
    want=solver(shapes,**options).solve()
    s=solver(shapes,backend=backend,**options)
    got=s.solve(geometry=s.freeze_geometry())
    np.testing.assert_allclose(np.asarray(got.T),want.T,rtol=0,atol=2e-11)
    np.testing.assert_allclose(np.asarray(got.R),want.R,rtol=0,atol=2e-11)


def test_torch_width_and_permeability_gradients_stay_live():
    torch=pytest.importorskip('torch')
    def scene(width,mu,backend):
        return solver([dict(type='rectangle',lx=width,ly=230.,angle=.17,
                           material=wavebit.Material(4.+.03j,mu=mu))],backend=backend,
                      grid=32,truncation_order=(1,1),phi=.13,psi=.23)
    width=torch.tensor(310.,dtype=torch.float64,requires_grad=True)
    mu=torch.tensor(1.2,dtype=torch.float64,requires_grad=True)
    s=scene(width,mu,2)
    derivatives=torch.autograd.grad(s.solve(geometry=s.freeze_geometry()).T.sum().real,(width,mu))
    for i,got in enumerate(derivatives):
        assert np.isfinite(got.item()) and abs(got.item())>1e-7
        for h in (1e-4,1e-5):
            a=[310.,1.2];b=a.copy();a[i]+=h;b[i]-=h
            fd=(scene(*a,0).solve().T.sum()-scene(*b,0).solve().T.sum())/(2*h)
            assert got.item()==pytest.approx(fd,rel=3e-4,abs=2e-7)


@pytest.mark.parametrize('shapes,background',[([],4.),
    ([dict(type='rectangle',lx=700.,ly=700.,material=4.)],1.)])
def test_uniform_layer_agrees_with_uniform_raster(shapes,background):
    s=solver(shapes,grid=32,truncation_order=(1,1),
             structure=[dict(background=background,shapes=shapes)])
    want=wavebit.rcwa(backend=0,period=(700.,700.),wavelength=900.,top=1.,bot=2.25,
        theta=.2,phi=0.,psi=0.,thickness=(300.,),truncation_order=(1,1),
        structure=np.full((1,2,2),4.)).solve()
    got=s.solve(geometry=s.freeze_geometry())
    np.testing.assert_allclose(got.T,want.T,rtol=0,atol=2e-12)
    np.testing.assert_allclose(got.R,want.R,rtol=0,atol=2e-12)
