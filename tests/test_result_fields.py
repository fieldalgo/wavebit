"""Common result fields: boundary physics, snapshots and differentiable sampling."""
import numpy as np
import pytest
import wavebit


def host(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def scene(**updates):
    return dict(wavelength=1.3, period=(2.1, 2.3), top=1.7, bot=2.25,
                structure=np.array([[[4., 4., 1., 1.]], [[2., 3., 3., 2.]]]),
                thickness=[.17, .23], truncation_order=(2, 0), answers='field') | updates


@pytest.mark.parametrize('theta', [.23, -.23])
@pytest.mark.parametrize('kind', ['te', 'tm', 'conical', '2ff', '4ff'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_all_paths_match_legacy_and_tangential_boundary_continuity(kind, side, algorithm, theta):
    """The result's field, the solver's cell and the boundary spectra are one field.

    ``theta = -0.23`` puts the incident order at ``kx < 0``, where the pure 1D path's incident
    wave is ``-y`` (TE) or the turned-over ``p_hat`` (TM): the boundary spectra that
    ``field_result`` builds from the amplitudes and the interior the kernels walk have to
    agree on that sign, which no ``kx > 0`` case exercises.
    """
    options = scene(theta=theta, incidence_side=side, layer_algorithm=algorithm)
    if kind in ('te', 'tm'):
        options['tetm'] = kind
    else:
        options.update(phi=.4, polarization=(.6, .8j))
    if kind in ('2ff', '4ff'):
        options.update(structure=np.repeat(options['structure'], 2, axis=1),
                       truncation_order=(1, 1))
    solver = wavebit.rcwa(**options)
    if kind == '4ff':
        solver.formulation = '4ff'
    result = solver.solve(auto_detune=False)
    points = dict(sample_x=[-.31, .09], sample_y=[.07], sample_z=[0., .12, .31, .4])
    field = result.reconstruct_field(**points)
    legacy = solver.reconstruct_field(**points)
    if kind in ('te', 'tm'):
        expected = np.zeros_like(field)
        expected[..., [1, 3, 5] if kind == 'te' else [4, 0, 2]] = legacy
    else:
        expected = legacy[0]
    np.testing.assert_allclose(field, expected, atol=3e-12)
    for index, boundary in [(0, 'bottom'), (-1, 'top')]:
        total = result.boundary_fields(boundary, points['sample_x'], points['sample_y'])
        np.testing.assert_allclose(field[index, ..., [0, 1, 3, 4]],
                                   total[..., [0, 1, 3, 4]].transpose(2, 0, 1), atol=3e-12)
        incoming = result.boundary_fields(boundary, points['sample_x'], points['sample_y'], 'incoming')
        outgoing = result.boundary_fields(boundary, points['sample_x'], points['sample_y'], 'outgoing')
        np.testing.assert_allclose(total, incoming+outgoing, atol=1e-13)


@pytest.mark.parametrize('backend', [0, 1, 2])
@pytest.mark.parametrize('sources', [False, True])
def test_count_coordinate_layout_and_result_lifetime(backend, sources):
    options = scene(backend=backend, truncation_order=0)
    options.update({'excitation': wavebit.PlaneWave(theta=.2)} if sources else {'tetm':'te', 'theta':.2})
    solver = wavebit.rcwa(**options)
    result = solver.solve()
    field = host(result.reconstruct_field(3, 2, 4))
    assert field.shape == (8, 2, 3, 6)
    # Each layer includes its own two faces; avoid the repeated interface here.
    z = [.0, .17/3, .17*2/3, .17+.23/3, .17+.23*2/3, .4]
    explicit = host(result.reconstruct_field([-1.05, 0., 1.05], [-1.15, 1.15], z))
    np.testing.assert_allclose(explicit, field[[0, 1, 2, 5, 6, 7]], atol=3e-12)
    boundary = host(result.boundary_fields('top', 3, 2))
    solver.solve(wavelength=1.6, thickness=[.21, .19])
    np.testing.assert_array_equal(host(result.reconstruct_field(3, 2, 4)), field)
    np.testing.assert_array_equal(host(result.boundary_fields('top', 3, 2)), boundary)


@pytest.mark.parametrize('source', [False, True])
def test_interface_normal_jump_and_layer_count_limits(source):
    options = scene(truncation_order=0, structure=np.array([[[2.]], [[5.]]]))
    options.update({'excitation': wavebit.PlaneWave(theta=.3, polarization=(0.,1.))}
                   if source else {'theta':.3, 'phi':0., 'polarization':(0.,1.)})
    result = wavebit.rcwa(**options).solve()
    count = host(result.reconstruct_field([0.], [0.], 2))[:,0,0]
    point = host(result.reconstruct_field([0.], [0.], [.17]))[0,0,0]
    np.testing.assert_allclose(count[2], point, atol=1e-13)
    np.testing.assert_allclose(2*count[1,2], 5*count[2,2], atol=2e-12)
    assert abs(count[1,2]-count[2,2]) > .01


@pytest.mark.parametrize('sampling', [4, [.05, .12]])
def test_jax_jit_result_and_thickness_derivative(sampling):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    solver = wavebit.rcwa(**scene(backend=1, truncation_order=0, tetm='te'))
    geometry = solver.freeze_geometry()
    def solve(d):
        return solver.solve(thickness=jnp.array([d, .23]), geometry=geometry, auto_detune=False)
    def loss(d):
        f = solve(d).reconstruct_field([0.], [0.], sampling)
        return jnp.sum(jnp.abs(f[...,1])**2)
    derivative = jax.jit(jax.grad(loss))(.17)
    h = 1e-5
    fd = (loss(.17+h)-loss(.17-h))/(2*h)
    np.testing.assert_allclose(derivative, fd, rtol=3e-7, atol=1e-7)
    result = jax.jit(solve)(.17)
    np.testing.assert_allclose(result.reconstruct_field([0.],[0.],sampling),
                               solve(.17).reconstruct_field([0.],[0.],sampling), atol=1e-12)


@pytest.mark.parametrize('sampling', [4, [.05, .12]])
def test_torch_thickness_derivative(sampling):
    torch = pytest.importorskip('torch')
    solver = wavebit.rcwa(**scene(backend=2, truncation_order=0, tetm='te'))
    def loss(d):
        r = solver.solve(thickness=torch.stack([d, d.new_tensor(.23)]), auto_detune=False)
        return torch.sum(torch.abs(r.reconstruct_field([0.], [0.], sampling)[...,1])**2)
    d = torch.tensor(.17, dtype=torch.float64, requires_grad=True)
    value = loss(d)
    grad, = torch.autograd.grad(value, d)
    h = 1e-5
    fd = (loss(d.detach()+h)-loss(d.detach()-h))/(2*h)
    np.testing.assert_allclose(host(grad), host(fd), rtol=3e-7, atol=1e-7)


def test_answer_levels_and_invalid_coordinates():
    solver = wavebit.rcwa(**scene(answers='amplitude', tetm='te'))
    result = solver.solve()
    assert result.boundary_fields('top', [0.], [0.]).shape == (1,1,6)
    with pytest.raises(RuntimeError, match="answers='field'"):
        result.reconstruct_field()
    result = solver.solve(answers='field')
    for spec in (0, -1, True, 2.5, [], [np.nan], [1j], [[0.]]):
        with pytest.raises((TypeError, ValueError)):
            result.reconstruct_field(sample_x=spec)
    for spec in ([-.1], [.5]):
        with pytest.raises(ValueError, match='inside'):
            result.reconstruct_field(sample_z=spec)
    with pytest.raises(ValueError, match='unavailable'):
        result.reconstruct_field(incidence='p')


#: Two lit sets: one on the ``kx > 0`` side only, and one straddling ``kx = 0``. The second is
#: the one where a lit order's own s/p frame differs from the incident order's, so a field
#: built from a column signed once too often (or not at all) moves the incident/outgoing
#: interference in ``total`` -- the one flux here that is not blind to a per-order sign.
LIT = {'forward': {0: 1., 1: .3j}, 'straddling': {-1: .4+.2j, 0: 1., 1: .3j}}


@pytest.mark.parametrize('lit', list(LIT))
@pytest.mark.parametrize('polarization', ['te', 'tm', 'coupled'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_boundary_flux_matches_efficiencies_with_absorbing_magnetic_ports(polarization, side,
                                                                         lit):
    options = scene(top=wavebit.Material(1.7+.15j, 1.2),
                    bot=wavebit.Material(2.25+.1j, 1.1), incidence_side=side,
                    theta=.2, excitation=LIT[lit])
    options.update({'tetm':polarization} if polarization != 'coupled'
                   else {'phi':.3, 'polarization':(.6,.8j)})
    solver = wavebit.rcwa(**options)
    result = solver.solve(auto_detune=False)
    kx = np.real(host(solver.order_kxy()[0])).ravel()
    (centre, half), _ = solver.order_window
    signs = {m: np.sign(kx[m - centre + half]) for m in LIT[lit]}
    assert (signs[-1] < 0) if lit == 'straddling' else all(v > 0 for v in signs.values()), signs
    # Cell integration cancels all different-order cross terms.
    x = np.linspace(-1.05, 1.05, 81, endpoint=False)
    def flux(boundary, part):
        f = result.boundary_fields(boundary, x, [0.], part)
        return .5*np.mean(np.real(f[...,0]*np.conj(f[...,4])-f[...,1]*np.conj(f[...,3])))
    other = 'bottom' if side == 'top' else 'top'
    incoming = flux(side, 'incoming')
    np.testing.assert_allclose(-flux(side, 'outgoing')/incoming, result.R.sum(), atol=2e-12)
    np.testing.assert_allclose(flux(other, 'outgoing')/incoming, result.T.sum(), atol=2e-12)
    np.testing.assert_allclose(flux(side, 'total')/incoming, result.power_entering, atol=2e-12)


def test_result_owns_mutable_solve_arrays_and_pure_variants():
    solver = wavebit.rcwa(**scene(theta=.2, phi=.3, polarization=(.6,.8j)))
    result = solver.solve()
    points = dict(sample_x=[-.1,.2], sample_y=[0.], sample_z=[.04,.21])
    for name in ('incident','s','p'):
        np.testing.assert_allclose(result.reconstruct_field(**points, incidence=name),
                                   solver.reconstruct_field(**points, incidences=(name,))[0], atol=2e-12)
    before = result.reconstruct_field(**points)
    solver.T1[...] = 0
    solver.thickness[...] = [.05, .07]
    np.testing.assert_array_equal(result.reconstruct_field(**points), before)
