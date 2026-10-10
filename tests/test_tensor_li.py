"""Tensor Li: scalar reduction, independent laminate relations and live gradients."""
import numpy as np
import pytest

import wavebit
from wavebit._core.geometry import StaleGeometryError


def host(a):
    return np.asarray(a.detach().cpu() if hasattr(a, 'detach') else a)


def material(loss=0., magnetic=False):
    e = np.array([[3.1 + 1j*loss, .23+.11j, 0.],
                  [.23-.11j, 4.2 + 1j*loss, 0.], [0., 0., 2.8 + 1j*loss]])
    mu = np.array([[1.3,.08j,0.],[-.08j,1.1,0.],[0.,0.,1.2]]) if magnetic else None
    return wavebit.Material(e, mu)


def scene(backend='numpy', width=240., mat=None, **kw):
    options = dict(backend=backend, wavelength=730., period=(600.,600.),
                   thickness=[173.], truncation_order=(2,1), theta=.23, phi=.19,
                   factorization='li', top=1., bot=2.25,
                   structure=[dict(background=1., shapes=[dict(type='rectangle',
                       lx=width, ly=300., material=material() if mat is None else mat)])])
    options.update(kw)
    if options.get('lattice') is not None:
        options.pop('period', None)
    return wavebit.rcwa(**options)


@pytest.mark.parametrize('coefficients', ['sampled', 'integrated'])
@pytest.mark.parametrize('orders', [(0,0), (2,1), (1,3)])
def test_scalar_tensor_is_exactly_the_scalar_li_operator(coefficients, orders):
    e = np.random.default_rng(4).uniform(1.,4.,(1,3,4)).astype(complex) + .07j
    got = scene(structure=e[...,None,None]*np.eye(3), truncation_order=orders,
                fourier_coefficients=coefficients).eps_conv_matrices()
    want = scene(structure=e, truncation_order=orders,
                 fourier_coefficients=coefficients).eps_conv_matrices()
    for a,b in zip(got[:3],want[:3]):
        np.testing.assert_allclose(a,b,atol=3e-12,rtol=0)
    for a in got[3:]:
        np.testing.assert_allclose(a,0.,atol=1e-14)


@pytest.mark.parametrize('axis', [0,1])
@pytest.mark.parametrize('loss', [0., .15])
def test_laminate_matches_independent_continuous_field_relations(axis, loss):
    # On a stripe, D_normal and E_tangent are continuous. Integrate their
    # constitutive relations directly over the two intervals, without using
    # the production mixed-variable transform or its Fourier helpers.
    normal, tangent = axis, 1-axis
    e = material(loss).epsilon
    b = np.array([[1.7,.12,0.],[.12,2.1,0.],[0.,0.,1.9]],complex)
    h, width, center = 2, 217.3, 23.1
    delta = np.arange(-h,h+1)[:,None]-np.arange(-h,h+1)[None,:]
    fill = width/600
    indicator = fill*np.sinc(delta*fill)*np.exp(-2j*np.pi*delta*(center/600))
    I = np.eye(2*h+1)
    def C(f): return f(b)*I+(f(e)-f(b))*indicator
    A = np.linalg.inv(C(lambda t: 1/t[normal,normal]))
    B = C(lambda t: t[normal,tangent]/t[normal,normal])
    Cn = C(lambda t: t[tangent,normal]/t[normal,normal])
    D = C(lambda t: t[tangent,tangent]-t[tangent,normal]*t[normal,tangent]/t[normal,normal])
    blocks = [[None,None],[None,None]]
    blocks[normal][normal], blocks[normal][tangent] = A, A@B
    blocks[tangent][normal], blocks[tangent][tangent] = Cn@A, D+Cn@A@B
    shape = dict(type='rectangle',lx=width if axis==0 else 600.,
                 ly=width if axis==1 else 600.,cx=center if axis==0 else 0.,
                 cy=center if axis==1 else 0.,material=wavebit.Material(e))
    s = scene(structure=[dict(background=wavebit.Material(b),shapes=[shape])],
              truncation_order=(h,0) if axis==0 else (0,h))
    got = s.eps_conv_matrices()
    want = [blocks[0][0],blocks[1][1],np.linalg.inv(C(lambda t: t[2,2])),
            -blocks[0][1],-blocks[1][0]]
    for a,b in zip(got,want):
        np.testing.assert_allclose(a[0],b,atol=3e-12,rtol=0)
    # PN provides an independent factorization on the same exact stripe.
    reference = scene(structure=s.structure, truncation_order=s.truncation_order,
                      factorization='nvf',shape_nvf_grid_size=16).solve(auto_detune=False)
    result = s.solve(auto_detune=False)
    for inc in ('s_inc','p_inc'):
        for name in ('r_s','r_p','t_s','t_p'):
            np.testing.assert_allclose(getattr(getattr(result,inc),name),
                                       getattr(getattr(reference,inc),name),atol=3e-11,rtol=0)


@pytest.mark.parametrize('backend',['numpy','jax','torch'])
@pytest.mark.parametrize('loss',[0.,.1])
def test_magnetic_painting_matches_raster_and_conserves_passivity(backend,loss):
    a = material(loss,magnetic=True)
    b = wavebit.Material([5.+1j*loss,2.5+1j*loss,3.+1j*loss],mu=[1.1,1.2,1.3]).rotated(.2)
    shapes = [dict(type='rectangle',lx=300.,ly=300.,material=a),
              dict(type='rectangle',lx=150.,ly=150.,cx=75.,cy=75.,material=b)]
    s = scene(backend,structure=[dict(background=1.,shapes=shapes)])
    X,Y = wavebit.meshgrid_xy((8,8),period=(600.,600.))
    eps = np.broadcast_to(np.eye(3),(8,8,3,3)).copy().astype(complex)
    mu = eps.copy()
    for shape in shapes:
        mask = (abs(X-shape.get('cx',0.))<shape['lx']/2)&(abs(Y-shape.get('cy',0.))<shape['ly']/2)
        m = shape['material']
        eps[mask],mu[mask] = np.asarray(m.epsilon),np.asarray(m.mu)
    raster = scene(backend,structure=eps[None],mu=mu[None])
    for get in ('eps_conv_matrices','mu_conv_matrices'):
        expected = getattr(raster,get)()
        for a,b in zip(getattr(s,get)(geometry=s.freeze_geometry()),expected):
            np.testing.assert_allclose(host(a),host(b),atol=3e-12,rtol=0)
    result = s.solve(auto_detune=False)
    for inc in (result.s_inc,result.p_inc):
        energy = host(inc.R.sum()+inc.T.sum())
        assert energy <= 1.+3e-12
        if loss==0.: assert energy==pytest.approx(1.,abs=3e-12)


@pytest.mark.parametrize('backend',['numpy','jax','torch'])
def test_uniform_rotated_tensor_uses_exact_operator(backend):
    m = wavebit.Material([2.25,3.4,4.]).rotated(.4,'y').rotated(.2)
    options = dict(structure=[dict(background=m,shapes=[])],truncation_order=(0,0))
    result = scene(backend,**options).solve(auto_detune=False)
    reference = scene(**options,factorization='nvf').solve(auto_detune=False)
    for name in ('r_s','r_p','t_s','t_p'):
        np.testing.assert_allclose(host(getattr(result.s_inc,name)),getattr(reference.s_inc,name),atol=2e-12)


@pytest.mark.parametrize('backend',['jax','torch'])
def test_shape_material_and_period_gradients(backend):
    frozen = scene().freeze_geometry()
    def loss(v):
        m = wavebit.Material([v[1],4.2,2.8]).rotated(v[2])
        return scene(backend,width=v[0],mat=m,period=(v[3],600.)).solve(
            geometry=frozen,auto_detune=False).s_inc.T.sum().real
    center = [237.,3.1,.31,600.]
    if backend=='jax':
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64',True)
        gradient = jax.jit(jax.grad(loss))(jnp.array(center))
    else:
        import torch
        v = torch.tensor(center,dtype=torch.float64,requires_grad=True)
        gradient, = torch.autograd.grad(loss(v),v)
    for i,step in enumerate([.001,1e-5,1e-5,.001]):
        a,b = np.array(center),np.array(center)
        a[i]+=step;b[i]-=step
        finite = (host(loss(a))-host(loss(b)))/(2*step)
        assert abs(finite)>1e-7
        np.testing.assert_allclose(host(gradient)[i],finite,rtol=5e-5,atol=2e-9)


@pytest.mark.parametrize('coefficients',['sampled','integrated'])
def test_jitted_raster_gradients_need_no_frozen_normal_field(coefficients):
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    mask = jnp.array([[0.,1.,1.],[0.,0.,1.]])
    def loss(v):
        e = wavebit.Material(jnp.array([v,4.,2.8])).rotated(.31).epsilon
        t = jnp.eye(3)+(e-jnp.eye(3))*mask[...,None,None]
        return scene('jax',structure=t[None],fourier_coefficients=coefficients).solve(auto_detune=False).T.sum()
    derivative = jax.jit(jax.grad(loss))(3.1)
    fd = (loss(3.10001)-loss(3.09999))/.00002
    np.testing.assert_allclose(derivative,fd,rtol=4e-5,atol=1e-9)


@pytest.mark.parametrize('option,match',[
    (dict(lattice=wavebit.Lattice.oblique(600.,600.,np.pi/3),period=None),'rectangular'),
    (dict(nvf_projector='linear'),'no reader'),
    (dict(raster_normal_method='distance'),'no reader'),
    (dict(shape_nvf_grid_size=32),'no reader'),
    (dict(fourier_coefficients='sampled'),'integrated'),
])
def test_unavailable_options_are_not_ignored(option,match):
    for method in ('freeze_geometry','solve'):
        with pytest.raises(NotImplementedError,match=match):
            getattr(scene(**option),method)()


def test_z_coupling_and_unsupported_shapes():
    tilted=scene(mat=wavebit.Material([2.,3.,4.]).rotated(.2,'y'))
    tilted.solve()
    assert tilted.formulation_assigned=='first_order'
    with pytest.raises(NotImplementedError,match='rectangles'):
        scene(structure=[dict(background=1.,shapes=[dict(type='circle',lx=240.,material=material())])]).solve()
    frozen = scene().freeze_geometry()
    with pytest.raises(StaleGeometryError):
        scene(factorization='nvf',shape_nvf_grid_size=16).solve(geometry=frozen)
    frozen = scene(factorization='nvf',shape_nvf_grid_size=16).freeze_geometry()
    with pytest.raises(StaleGeometryError,match='factorization'):
        scene().solve(geometry=frozen)


def test_raster_uniform_values_preserve_all_pixel_derivatives():
    import torch
    x = torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    mask = torch.tensor([[0.,0.],[0.,1.]],dtype=torch.float64)
    t = (2.+(x-2.)*mask)[None,...,None,None]*torch.eye(3,dtype=torch.float64)
    loss = scene('torch',structure=t,truncation_order=(0,0)).eps_conv_matrices().epx.real.sum()
    derivative, = torch.autograd.grad(loss,x)
    assert derivative.item()==pytest.approx(.25,abs=1e-12)


def test_traced_z_coupling_is_not_silently_discarded():
    import jax
    import jax.numpy as jnp
    def operator(z):
        e = jnp.diag(jnp.array([2.,3.,4.])).at[0,2].set(z)
        t = jnp.stack([jnp.eye(3),e])[None,None]
        return scene('jax',structure=t).eps_conv_matrices().epxz
    assert np.isfinite(jax.jit(operator)(0.)).all()
    coupled = jax.jit(operator)(.1)
    assert np.isfinite(coupled).all()
    assert np.max(np.abs(coupled))>1e-3


@pytest.mark.parametrize('backend',['numpy','jax','torch'])
def test_single_precision_and_field_reconstruction(backend):
    s = scene(backend,complex_dtype=1,answers='field',truncation_order=(1,1))
    assert all(a.dtype==s.xp.complex64 for a in s.eps_conv_matrices())
    solve_field = getattr(s, 'solve_field_eager', s.solve_field)
    result, field = solve_field(3,3,2,geometry=s.freeze_geometry())
    assert np.isfinite(host(field)).all()
    np.testing.assert_allclose(host(result.T),scene(truncation_order=(1,1)).solve().T,atol=3e-5,rtol=0)


def test_fixed_rotation_keeps_nested_torch_material_components_live():
    import torch
    v = torch.tensor(3.1,dtype=torch.float64,requires_grad=True)
    m = wavebit.Material([[v,.2,0.],[.2,4.,0.],[0.,0.,2.8]]).rotated(.3)
    derivative, = torch.autograd.grad(m.epsilon.real.trace(),v)
    assert derivative.item()==pytest.approx(1.,abs=1e-12)


def test_li_tensor_spec_round_trip_keeps_the_selected_rule():
    s = scene()
    spec = wavebit.JobSpec(
        stack=wavebit.Stack([wavebit.Layer(173.,s.structure[0])],period=(600.,600.),top=1.,bot=2.25),
        incidence=wavebit.Incidence(wavelength=730.,theta=.23,phi=.19),
        numerics=wavebit.Numerics(factorization='li',truncation_order=(2,1)),
    )
    restored = wavebit.JobSpec.from_json(spec.to_json())
    np.testing.assert_allclose(wavebit.run(restored).T,s.solve().T,atol=2e-12,rtol=0)


def test_jitted_tensor_li_field_reuses_the_geometry_plan():
    s = scene('jax',answers='field',truncation_order=(1,1))
    geometry = s.freeze_geometry()
    want, expected = s.solve_field_eager(3,3,2,geometry=geometry)
    got, actual = s.solve_field(3,3,2,geometry=geometry)
    np.testing.assert_allclose(got.T,want.T,atol=2e-11,rtol=0)
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=0)
