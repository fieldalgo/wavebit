"""Production Cartesian shape Li agrees with independently painted raster integrals."""
import numpy as np
import pytest
import wavebit
from wavebit._core import factorization as F
from wavebit._core.geometry import StaticGeometry


def box(lx=350., ly=300., cx=0., cy=0., material=4.):
    return dict(type='rectangle', lx=lx, ly=ly, cx=cx, cy=cy, material=material)


def solver(shapes, backend=0, background=1., **kwargs):
    options = dict(backend=backend, period=(700.,700.), wavelength=900., top=1., bot=2.25,
                   theta=.2, phi=0., psi=0., thickness=(300.,), truncation_order=(2,2),
                   factorization='li', structure=[dict(background=background, shapes=shapes)])
    options.update(kwargs)
    return wavebit.rcwa(**options)


@pytest.mark.parametrize('shapes', [[], [box(350,700)],
    [box(300,300,-150,material=4),box(300,300,150,material=9)],
    [box(400,400,material=4+.2j),box(200,200,50,50,9+.1j)],
    [box(400,400,-100,material=4),box(400,300,100,50,9)]])
def test_matrices_match_independent_raster_painting(shapes):
    x,y=np.meshgrid((np.arange(28)+.5)*25-350,(np.arange(28)+.5)*25-350)
    cell=np.full((28,28),1.,dtype=complex)
    for s in shapes:
        cell[(abs(x-s['cx'])<s['lx']/2)&(abs(y-s['cy'])<s['ly']/2)]=s['material']
    want=F.to_conv_mat_raster_integrated(cell[None],2,2)
    s=solver(shapes); got=s._eps_conv_matrices(s.freeze_geometry())
    assert got[3:] == (None,None)
    for a,b in zip(got[:3],want[:3]):
        np.testing.assert_allclose(a,b,rtol=0,atol=2e-12)


def test_contact_plan_opens_a_gap_but_refuses_edge_crossing():
    def shapes(gap):
        return [box(300,300,-150-gap/2),box(300,300,150+gap/2,material=9)]
    g=solver(shapes(0)).freeze_geometry()
    for gap in (0,1e-6,1.,20.):
        live=solver(shapes(gap))
        np.testing.assert_allclose(live.solve(geometry=g).T, live.solve().T,rtol=0,atol=2e-12)
    with pytest.raises(F.StaleGeometryError,match='edge order'):
        solver(shapes(-1)).solve(geometry=g)
    with pytest.raises(F.StaleGeometryError,match='shape count'):
        solver([box()]).solve(geometry=g)


@pytest.mark.parametrize('geometry', [StaticGeometry(), 'nvf'])
def test_wrong_geometry_route_is_refused(geometry):
    if geometry == 'nvf':
        geometry=solver([box()],factorization='nvf',shape_nvf_grid_size=16).freeze_geometry()
    with pytest.raises(F.StaleGeometryError,match='factorization'):
        solver([box()]).solve(geometry=geometry)


def test_li_record_cannot_be_replayed_as_nvf():
    g=solver([box()]).freeze_geometry()
    with pytest.raises(F.StaleGeometryError,match='NVF'):
        solver([box()],factorization='nvf',shape_nvf_grid_size=16).solve(geometry=g)


@pytest.mark.parametrize('option', [dict(shape_nvf_grid_size=32),dict(raster_normal_method='distance'),dict(nvf_projector='linear')])
def test_unused_normal_options_are_refused(option):
    with pytest.raises((NotImplementedError,ValueError)):
        solver([box()],**option).solve()


@pytest.mark.parametrize('shape', [dict(type='circle',lx=100,material=4),dict(box(),angle=.1),box(lx=701),box(cx=350)])
def test_unsupported_geometry_is_refused_on_freeze_and_solve(shape):
    for method in ('freeze_geometry','solve'):
        with pytest.raises((NotImplementedError,ValueError)):
            getattr(solver([shape]),method)()


def test_jitted_gap_and_material_derivatives_match_two_finite_differences():
    jax=pytest.importorskip('jax');import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    def scene(gap,eps,backend):
        return solver([box(300,300,-150-gap/2),box(300,300,150+gap/2,material=eps)],backend)
    geometry=scene(0.,4.,0).freeze_geometry()
    def loss(v):
        return jnp.sum(scene(v[0],v[1],1).solve(geometry=geometry).T).real
    center=np.array([1e-4,4.])
    value,grad=jax.jit(jax.value_and_grad(loss))(jnp.asarray(center))
    assert np.isfinite(grad).all() and np.all(abs(grad)>1e-7)
    for i in (0,1):
        for h in (1e-5,1e-6):
            a=center.copy();b=center.copy();a[i]+=h;b[i]-=h
            fd=(scene(*a,0).solve().T.sum()-scene(*b,0).solve().T.sum())/(2*h)
            assert float(grad[i]) == pytest.approx(fd,rel=3e-4,abs=2e-8)
    baseline=scene(0.,4.,0).solve().T.sum()
    assert abs(scene(1e-8,4.,0).solve().T.sum()-baseline)<1e-9
    assert abs(scene(0.,4.+1e-8,0).solve().T.sum()-baseline)<1e-9


@pytest.mark.parametrize('backend',[1,2])
def test_magnetic_solve_matches_numpy(backend):
    pytest.importorskip('jax' if backend==1 else 'torch')
    if backend==1:
        import jax;jax.config.update('jax_enable_x64',True)
    shapes=[box(material=wavebit.Material(4.,mu=1.3))]
    bg=wavebit.Material(1.,mu=1.1)
    want=solver(shapes,background=bg).solve()
    s=solver(shapes,backend,background=bg);got=s.solve(geometry=s.freeze_geometry())
    for a,b in ((got.R,want.R),(got.T,want.T)):
        np.testing.assert_allclose(np.asarray(a),b,rtol=0,atol=2e-11)


def test_jitted_field_replays_cartesian_plan():
    jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
    s=solver([box()],1,answers='field',truncation_order=(1,1))
    geometry=s.freeze_geometry()
    want,wf=s.solve_field_eager(3,3,2,geometry=geometry)
    got,gf=s.solve_field(3,3,2,geometry=geometry)
    np.testing.assert_allclose(got.T,want.T,rtol=0,atol=2e-11)
    np.testing.assert_allclose(gf,wf,rtol=0,atol=2e-10)


def test_single_period_spanning_rectangle_matches_independent_contact_reference():
    from contact_instruments import rectangular_li
    for gap in (0.,1e-8,1e-4,175.):
        s=solver([box(350,700-gap,material=12.25)],truncation_order=(3,3))
        got=s._eps_conv_matrices()
        want=rectangular_li(s.xp,s.structure[0]['shapes'],1.,3,3,
                            s._geometry_lattice_for(None),s.complex_dtype)
        for a,b in zip(got[:3],want[:3]):
            np.testing.assert_allclose(a[0],b,rtol=0,atol=2e-12)


def test_permeability_has_its_own_live_material_values():
    shapes=[box(350,350,material=wavebit.Material(4.,mu=1.7))]
    s=solver(shapes,background=wavebit.Material(2.,mu=1.2))
    g=s.freeze_geometry()
    x,y=np.meshgrid((np.arange(16)+.5)*700/16-350,(np.arange(16)+.5)*700/16-350)
    inside=(abs(x)<175)&(abs(y)<175)
    for got,bg,hi in ((s._eps_conv_matrices(g),2.,4.),(s._mu_conv_matrices(g),1.2,1.7)):
        want=F.to_conv_mat_raster_integrated(np.where(inside,hi,bg)[None],2,2)
        for a,b in zip(got[:3],want[:3]):
            np.testing.assert_allclose(a,b,rtol=0,atol=2e-12)


def test_torch_retains_geometry_and_material_gradients():
    torch=pytest.importorskip('torch')
    width=torch.tensor(350.,dtype=torch.float64,requires_grad=True)
    mu=torch.tensor(1.3,dtype=torch.float64,requires_grad=True)
    def scene(w,m,backend):
        return solver([box(lx=w,material=wavebit.Material(4.,mu=m))],backend)
    s=scene(width,mu,2);g=s.freeze_geometry()
    loss=s.solve(geometry=g).T.sum().real
    derivatives=torch.autograd.grad(loss,(width,mu))
    for i,derivative in enumerate(derivatives):
        assert np.isfinite(derivative.item()) and abs(derivative.item())>1e-7
        for h in (1e-4,1e-5):
            a=[350.,1.3];b=a.copy();a[i]+=h;b[i]-=h
            fd=(scene(*a,0).solve().T.sum()-scene(*b,0).solve().T.sum())/(2*h)
            assert derivative.item()==pytest.approx(fd,rel=2e-4,abs=2e-8)


def test_skew_cell_and_invalid_sizes_are_refused():
    for options in (dict(period=None,lattice=((700.,100.),(0.,700.))),):
        with pytest.raises((NotImplementedError,ValueError)):
            solver([box()],**options).freeze_geometry()
    for size in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='positive finite'):
            solver([box(lx=size)]).freeze_geometry()


def test_geometry_hash_distinguishes_cartesian_edge_order():
    a=solver([box(300,300,-150),box(300,300,150)]).freeze_geometry()
    b=solver([box(300,300,-149),box(300,300,149)]).freeze_geometry()
    c=solver([box(300,300,-151),box(300,300,151)]).freeze_geometry()
    assert a != b and a == c and hash(a) == hash(c)


@pytest.mark.parametrize('backend',[0,1,2])
def test_single_precision_is_preserved(backend):
    if backend:
        pytest.importorskip('jax' if backend==1 else 'torch')
    s=solver([box()],backend,complex_dtype=1)
    matrices=s._eps_conv_matrices(s.freeze_geometry())
    assert all(m.dtype==s.xp.complex64 for m in matrices[:3])
    np.testing.assert_allclose(np.asarray(s.solve().T),solver([box()]).solve().T,rtol=0,atol=2e-5)


def test_period_spanning_gap_derivative_matches_independent_reference():
    jax=pytest.importorskip('jax');import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    def scene(gap,backend):
        return solver([box(350,700-gap,material=12.25)],backend,truncation_order=(3,3))
    geometry=scene(0.,0).freeze_geometry()
    def loss(gap):
        return jnp.sum(scene(gap,1).solve(geometry=geometry).T).real
    derivative=float(jax.jit(jax.grad(loss))(jnp.asarray(1e-4)))
    assert derivative==pytest.approx(.00147644,rel=1e-4)
    for h in (1e-5,1e-6):
        fd=(scene(1e-4+h,0).solve().T.sum()-scene(1e-4-h,0).solve().T.sum())/(2*h)
        assert derivative==pytest.approx(fd,rel=2e-4,abs=1e-7)


@pytest.mark.parametrize('backend',[1,2])
def test_live_cell_period_derivative(backend):
    def scene(period):
        return solver([box()],backend,period=(period,700.))
    geometry=scene(700.).freeze_geometry()
    if backend==1:
        jax=pytest.importorskip('jax');import jax.numpy as jnp
        jax.config.update('jax_enable_x64',True)
        def loss(period):
            return jnp.sum(scene(period).solve(geometry=geometry).T).real
        derivative=float(jax.jit(jax.grad(loss))(jnp.asarray(700.)))
    else:
        torch=pytest.importorskip('torch')
        period=torch.tensor(700.,dtype=torch.float64,requires_grad=True)
        loss=scene(period).solve(geometry=geometry).T.sum().real
        derivative=torch.autograd.grad(loss,period)[0].item()
    assert np.isfinite(derivative) and abs(derivative)>1e-7
    for h in (1e-3,1e-4):
        fd=(solver([box()],period=(700.+h,700.)).solve().T.sum()
            -solver([box()],period=(700.-h,700.)).solve().T.sum())/(2*h)
        assert derivative==pytest.approx(fd,rel=2e-4,abs=2e-8)
