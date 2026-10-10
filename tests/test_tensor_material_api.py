"""Public tensor assignments: optical oracles, differentiation and serialization."""
import json

import numpy as np
from _conv_frame import slots_about_centre
import pytest

import wavebit
from wavebit import Material, TabulatedMaterial
from wavebit._core.geometry import StaleGeometryError
from test_uniform_anisotropic import slab_s, slab_p


def host(a):
    if hasattr(a, 'detach'):
        a = a.detach().cpu()
    return np.asarray(a)


def film(material, backend='numpy', **kwargs):
    options = dict(backend=backend, wavelength=600., period=(300., 300.),
                   thickness=[220.], structure=[dict(background=material, shapes=[])],
                   truncation_order=(0, 0), phi=0.)
    options.update(kwargs)
    return wavebit.rcwa(**options)


def pattern(material=None, backend='numpy', **kwargs):
    material = Material([2.25, 4., 2.8]) if material is None else material
    options = dict(wavelength=700., period=(600., 600.), thickness=[170.],
                   structure=[dict(background=1., shapes=[dict(
                       type='circle', cx=20., cy=-30., lx=270., material=material)])],
                   truncation_order=(1, 1), shape_nvf_grid_size=24, phi=.1,
                   fourier_coefficients='sampled')
    options.update(kwargs)
    if options.get('lattice') is not None:
        options.pop('period', None)
    return wavebit.rcwa(backend=backend, **options)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('theta', [0., .47])
def test_diagonal_assignment_matches_independent_airy(backend, theta):
    res = film(Material([2.25, 4., 2.8]), backend, theta=theta).solve(auto_detune=False)
    for actual, expected in [(res.s_inc, slab_s(4., np.sin(theta))),
                             (res.p_inc, slab_p(2.25, 2.8, np.sin(theta)))]:
        np.testing.assert_allclose(host(actual.R.sum()), abs(expected[0])**2, atol=2e-12)
        np.testing.assert_allclose(host(actual.T.sum()), abs(expected[1])**2, atol=2e-12)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_rotated_material_matches_waveplate_amplitude(backend):
    angle = .38
    m = Material([4., 2.25, 2.25]).rotated(angle)
    res = film(m, backend).solve(auto_detune=False)
    re, te = slab_s(4.)
    ro, to = slab_s(2.25)
    c, s = np.cos(angle), np.sin(angle)
    np.testing.assert_allclose(host(res.s_inc.t_s), [[te*s*s+to*c*c]], atol=2e-12)
    # This complex amplitude catches an xy sign error which R/T cannot see.
    # Top incidence travels -z: public p=s cross k is -x at phi=0.
    np.testing.assert_allclose(host(res.s_inc.t_p), [[-(te-to)*s*c]], atol=2e-12)
    assert host(res.s_inc.R.sum()+res.s_inc.T.sum()) == pytest.approx(1., abs=2e-12)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_full_tensor_and_stack_order_against_berreman(side):
    from test_anisotropic_berreman import berreman_efficiencies, berreman_efficiencies_from_below
    e = Material([2.25, 3.4, 4.]).rotated(.4, 'y').rotated(.2).epsilon
    f = np.diag([3., 2., 2.8])
    # Independent oracle uses downward z and a top-to-bottom layer list.
    Q = np.diag([1., 1., -1.])
    layers = [(Q@f@Q, 90.), (Q@e@Q, 120.)]
    oracle = berreman_efficiencies if side == 'top' else berreman_efficiencies_from_below
    expected = oracle(layers, .31, wl=600.)
    res = film(Material(e), structure=[dict(background=Material(e), shapes=[]),
                                      dict(background=Material(f), shapes=[])],
               thickness=[120., 90.], theta=.31, incidence_side=side).solve(auto_detune=False)
    for name, inc in [('s', res.s_inc), ('p', res.p_inc)]:
        for key in ('R_s','R_p','T_s','T_p'):
            np.testing.assert_allclose(getattr(inc,key).sum(), expected[name][key], atol=2e-11)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_tensor_shape_equals_raster_and_frozen_replay(backend):
    solver = pattern(backend=backend)
    X, Y = wavebit.meshgrid_xy((24, 24), period=(600.,600.))
    mask = (X-20.)**2+(Y+30.)**2 <= 135.**2
    raster = np.where(mask[...,None,None], np.diag([2.25,4.,2.8]), np.eye(3))[None]
    other = wavebit.rcwa(backend=backend, wavelength=700., period=(600.,600.), thickness=[170.],
                         structure=raster, truncation_order=(1,1), phi=.1)
    frozen = solver.freeze_geometry()
    for a, b, c in zip(solver.eps_conv_matrices(), other.eps_conv_matrices(),
                       solver.eps_conv_matrices(geometry=frozen)):
        np.testing.assert_allclose(host(a), host(b), atol=2e-12)
        np.testing.assert_allclose(host(a), host(c), atol=2e-12)
    result = solver.solve(auto_detune=False)
    np.testing.assert_allclose(host(result.R), host(other.solve(auto_detune=False).R), atol=2e-12)


def test_all_components_define_boundaries_and_small_grid_is_repeated():
    from wavebit._core.factorization import pn_eps_conv_matrices, normal_vector_field
    a, b = np.diag([2.,4.,3.]), np.diag([4.,2.,3.])  # equal trace, different phases
    tensor = np.stack([a,b])[None, None]
    solver = wavebit.rcwa(structure=tensor, wavelength=700., period=(600.,600.),
                         thickness=[170.], truncation_order=(1,1))
    repeated = np.repeat(np.repeat(tensor[0], 6, axis=0), 3, axis=1)
    labels = np.tile([0.,0.,0.,1.,1.,1.], (6,1))
    nx,ny = normal_vector_field(labels, lattice=np.diag([600.,600.]))
    expected = slots_about_centre(pn_eps_conv_matrices(np, repeated, nx, ny, 1, 1), (1, 1))
    for got, want in zip(solver.eps_conv_matrices(), expected):
        np.testing.assert_allclose(got[0], want, atol=2e-12)


def test_scalar_epsilon_with_tensor_mu_and_reverse():
    for e,m in [(np.ones((1,1,1))*2.25, np.diag([1.2,1.3,1.4])[None,None,None]),
                (np.diag([2.25,4.,2.8])[None,None,None], np.ones((1,1,1))*1.2)]:
        solver = wavebit.rcwa(structure=e, mu=m, wavelength=600., period=(300.,300.),
                             thickness=[220.], truncation_order=(0,0))
        res = solver.solve(auto_detune=False)
        np.testing.assert_allclose(res.s_inc.R.sum()+res.s_inc.T.sum(), 1., atol=2e-12)


def test_tensor_json_round_trip_constant_table_and_raster():
    values = [Material([2.25+.02j,4.+.03j,2.8]).rotated(.3),
              TabulatedMaterial([500.,700.], [[2.,3.,2.5],[3.,4.,2.9]],
                                mu=[[1.1,1.2,1.3],[1.2,1.3,1.4]]).rotated(.2)]
    structures = [dict(background=v, shapes=[]) for v in values]
    structures.append(np.diag([2.25+.02j,4.+.03j,2.8])[None,None])
    for structure in structures:
        spec = wavebit.JobSpec(stack=wavebit.Stack([wavebit.Layer(220., structure)], period=(300.,300.)),
                              incidence=wavebit.Incidence(wavelength=600., phi=0.))
        restored = wavebit.JobSpec.from_json(spec.to_json())
        assert restored.key() == spec.key()
        np.testing.assert_allclose(wavebit.run(restored).R, wavebit.run(spec).R, atol=2e-12)


@pytest.mark.parametrize('backend', ['numpy','jax','torch'])
def test_tensor_dispersion_resolves_each_wavelength(backend):
    m = TabulatedMaterial([500.,700.], [[2.,3.,2.5],[3.,4.,2.9]])
    solver = film(m, backend)
    for w in (550.,650.):
        expected = film(m.at(w), backend, wavelength=w).solve(auto_detune=False)
        actual = solver.solve(wavelength=w, auto_detune=False)
        np.testing.assert_allclose(host(actual.R), host(expected.R), atol=2e-12)
    spectrum = m.at(np.array([500.,600.,700.]))
    with pytest.raises(ValueError, match='wavelength'):
        film(spectrum).solve()
    with pytest.raises(ValueError, match='wavelength'):
        spectrum.rotated(.2)


def test_unsupported_requests_and_stale_geometry_fail_loudly():
    tilted = Material([2.25,3.,4.]).rotated(.3, 'y')
    assert len(pattern(tilted).eps_conv_matrices())==9
    film(Material(tilted.epsilon, mu=1.)).solve()
    with pytest.raises(ValueError, match='half-space'):
        film(Material([2.,3.,4.]), top=Material([2.,3.,4.]))
    assert pattern(fourier_coefficients='integrated').fourier_coefficients == 'integrated'
    with pytest.raises(NotImplementedError, match='linear'):
        pattern(nvf_projector='circular').solve()
    with pytest.raises(ValueError, match='tetm'):
        film(Material([2.,3.,4.]), tetm='te').solve()
    solver = pattern()
    frozen = solver.freeze_geometry()
    solver.period = [650.,600.]
    with pytest.raises(StaleGeometryError):
        solver.eps_conv_matrices(geometry=frozen)


@pytest.mark.parametrize('backend', ['jax','torch'])
def test_material_rotation_gradient_against_finite_difference(backend):
    def objective(a):
        m = Material([4.,2.25,2.8]).rotated(a)
        return film(m, 'jax' if backend=='jax' else 'torch').solve(auto_detune=False).s_inc.t_p.real.sum()
    if backend == 'jax':
        import jax
        derivative = jax.jit(jax.grad(objective))(.3)
    else:
        import torch
        a = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        y = objective(a)
        derivative, = torch.autograd.grad(y, a)
    expected = (host(objective(.30001))-host(objective(.29999)))/.00002
    np.testing.assert_allclose(host(derivative), expected, rtol=2e-6, atol=1e-8)


def test_jax_pattern_material_gradient_with_frozen_geometry():
    import jax
    import jax.numpy as jnp
    frozen = pattern(backend='jax').freeze_geometry()
    def f(e):
        m = Material(jnp.stack([e, e*0+4., e*0+2.8]))
        return pattern(m, backend='jax').solve(geometry=frozen, auto_detune=False).R.sum()
    derivative = jax.jit(jax.grad(f))(2.25)
    expected = (host(f(2.25001))-host(f(2.24999)))/.00002
    np.testing.assert_allclose(host(derivative), expected, rtol=2e-5, atol=1e-8)


def test_tensor_picture_selects_component_and_paints_in_order():
    a = Material([2.,3.,4.]).rotated(.25)
    b = Material([5.,6.,7.])
    shapes = [dict(type='rectangle', cx=0.,cy=0.,lx=300.,ly=260.,material=a),
              dict(type='circle', cx=80.,cy=20.,lx=220.,material=b)]
    solver = pattern(a, structure=[dict(background=1., shapes=shapes)])
    xx = solver.slice_structure(n=24, component='xx')
    xy = solver.slice_structure(n=24, component='xy')
    X,Y = wavebit.meshgrid_xy((24,24), period=(600.,600.))
    rectangle = (abs(X)<=150.) & (abs(Y)<=130.)
    circle = (X-80.)**2+(Y-20.)**2<=110.**2
    np.testing.assert_allclose(xx, np.where(circle,5.,np.where(rectangle,a.epsilon[0,0],1.)))
    np.testing.assert_allclose(xy, np.where(circle,0.,np.where(rectangle,a.epsilon[0,1],0.)))
    ax = solver.plot_structure(n=24, component='xy')
    assert 'eps_xy' in ax.get_title()
    import matplotlib.pyplot as plt
    plt.close(ax.figure)
    with pytest.raises(ValueError, match='component'):
        solver.slice_structure()
    with pytest.raises(NotImplementedError, match='components'):
        solver.slice_structure(show='n', component='xx')


@pytest.mark.parametrize('backend', ['numpy','jax','torch'])
def test_full_tensor_and_magnetic_tensor_across_backends(backend):
    for m in (Material([2.,3.,4.]).rotated(.3,'x'),
              Material([2.,3.,4.],mu=[1.1,1.3,1.2]).rotated(.3)):
        got = film(m,backend,theta=.27).solve(auto_detune=False)
        expected = film(m,theta=.27).solve(auto_detune=False)
        for inc in ('s_inc','p_inc'):
            for key in ('r_s','r_p','t_s','t_p'):
                np.testing.assert_allclose(host(getattr(getattr(got,inc),key)),
                                           getattr(getattr(expected,inc),key), atol=2e-11)


def test_uniform_pixel_values_do_not_discard_pixel_gradients():
    import torch
    # Changing a pixel away from [0,0] must change the zero-order mean.
    x = torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    mask = torch.tensor([[0.,0.],[0.,1.]],dtype=torch.float64)
    eps = (2.+(x-2.)*mask)[None,...,None,None]*torch.eye(3,dtype=torch.float64)
    solver = wavebit.rcwa(backend='torch', structure=eps, period=(600.,600.), thickness=[170.],
                         wavelength=700., truncation_order=(0,0))
    yy = solver.eps_conv_matrices().epy.real.sum()
    derivative, = torch.autograd.grad(yy,x)
    assert derivative.item() == pytest.approx(.25,abs=1e-12)


def test_tensor_serialization_rejects_malformed_or_nonfinite_values():
    spec = wavebit.JobSpec(stack=wavebit.Stack([wavebit.Layer(100.,dict(
        background=Material([2.,3.,4.]),shapes=[]))],period=(600.,600.)))
    for content in ({'shape':[3,3],'tensor':[[1.,0.]]},
                    {'shape':[3],'tensor':[[1.,0.],[2.,0.],[float('nan'),0.]]},
                    {'shape':[3],'tensor':[True,2.,3.]},
                    {'shape':[3],'tensor':[1.,2.,3.],'ignored':1}):
        obj = json.loads(spec.to_json())
        obj['stack']['layers'][0]['structure']['background']['epsilon'] = content
        with pytest.raises(wavebit.SpecError):
            wavebit.JobSpec.from_dict(obj)
    for value in ([2.,3.,float('inf')],np.full((3,3),float('nan'))):
        with pytest.raises(ValueError,match='finite'):
            film(Material(value)).solve()


def test_frozen_pattern_rejects_moved_or_merged_material_interfaces():
    solver = pattern()
    frozen = solver.freeze_geometry()
    for replacement in (Material([1.,1.,1.]),):
        solver.structure = [dict(background=1.,shapes=[dict(
            type='circle',cx=20.,cy=-30.,lx=270.,material=replacement)])]
        with pytest.raises(StaleGeometryError,match='boundaries'):
            solver.eps_conv_matrices(geometry=frozen)
    import torch
    radius = torch.tensor(270.,requires_grad=True)
    with pytest.raises(ValueError,match='geometry derivatives'):
        pattern(backend='torch',structure=[dict(background=1.,shapes=[dict(
            type='circle',lx=radius,material=Material([2.,3.,4.]))])]).solve()


def test_jax_traced_z_coupling_is_not_discarded():
    import jax
    import jax.numpy as jnp
    solver = pattern(backend='jax')
    frozen = solver.freeze_geometry()
    def f(xz):
        e = jnp.diag(jnp.array([2.25,4.,2.8])).at[0,2].set(xz)
        return pattern(Material(e),backend='jax').eps_conv_matrices(geometry=frozen).epxz
    assert np.isfinite(host(jax.jit(f)(0.))).all()
    coupled=host(jax.jit(f)(.2))
    assert np.isfinite(coupled).all()
    assert np.max(np.abs(coupled))>1e-3


def test_sampled_tensor_lattice_derivative_is_refused():
    import jax
    import torch
    frozen = pattern(backend='jax').freeze_geometry()
    with pytest.raises(ValueError, match='lattice derivatives'):
        jax.jit(jax.grad(lambda p: pattern(backend='jax', period=(p,600.)).solve(
            geometry=frozen,auto_detune=False).R.sum()))(600.)
    with pytest.raises(ValueError, match='lattice derivatives'):
        pattern(backend='torch', period=torch.tensor([600.,600.], requires_grad=True)).solve()


def test_periodic_tensor_shapes_on_an_unreduced_cell_match_brute_force_images():
    basis = np.array([[600.,1860.],[0.,420.]])
    crystal = Material([2.,3.,4.])
    shapes = [dict(type='circle',cx=260.,cy=0.,lx=300.,material=crystal),
              dict(type='rectangle',cx=-280.,cy=50.,lx=160.,ly=190.,material=Material([5.,6.,7.]))]
    solver = pattern(structure=[dict(background=1.,shapes=shapes)], lattice=basis, period=None)
    X,Y = wavebit.meshgrid_xy((24,24),lattice=basis)
    expected = np.ones_like(X)
    for shape, value in zip(shapes,[2.,5.]):
        mask = np.zeros(X.shape,bool)
        for m in range(-10,11):
            for k in range(-10,11):
                ox,oy = basis @ [m,k]
                dx,dy = X-shape['cx']-ox,Y-shape['cy']-oy
                mask |= (dx**2+dy**2<=(shape['lx']/2)**2 if shape['type']=='circle'
                         else (abs(dx)<=shape['lx']/2)&(abs(dy)<=shape['ly']/2))
        expected = np.where(mask,value,expected)
    np.testing.assert_allclose(solver.slice_structure(n=24,component='xx'),expected)
    shift = basis @ [17,-8]
    shifted = [dict(s,cx=s['cx']+shift[0],cy=s['cy']+shift[1]) for s in shapes]
    moved = pattern(structure=[dict(background=1.,shapes=shifted)],lattice=basis,period=None)
    for got,want in zip(moved.eps_conv_matrices(),solver.eps_conv_matrices()):
        np.testing.assert_allclose(got,want,atol=2e-12)
