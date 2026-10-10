"""Two-sided electric inputs: independent slab matching, flux and field superposition."""
import numpy as np
import pytest
import wavebit


def host(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def scene(**kwargs):
    options = dict(wavelength=1.3, period=(2., 2.2), top=1., bot=2.25,
                   structure=np.array([[[4., 4., 1., 1.]]]), thickness=[.12],
                   truncation_order=(2, 0), answers='field')
    options.update(kwargs)
    return options


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('pol', [(1., 0.), (0., 1.), (.6, .8j)])
def test_one_sided_limit_matches_existing_pattern_solver(side, pol):
    options = scene()
    old = wavebit.rcwa(**options, theta=.25, phi=.6, polarization=pol,
                       incidence_side=side, layer_algorithm='SMM')
    new = wavebit.rcwa(**options, excitation={side: wavebit.PlaneWave(theta=.25, phi=.6, polarization=pol)})
    a, b = old.solve(auto_detune=False), new.solve()
    np.testing.assert_allclose(b.output_top if side == 'top' else b.output_bottom, a.R.sum(), atol=2e-12)
    np.testing.assert_allclose(b.output_bottom if side == 'top' else b.output_top, a.T.sum(), atol=2e-12)
    points = dict(sample_x=[-.17, .12], sample_y=[-.11, .06], sample_z=[.02, .1])
    # Both are unit at x = y = 0 on the lit face (the ordinary solve since 2026-09-27), so the
    # fields are one field, with no factor between them.
    reference = old.reconstruct_field(**points)[0]
    np.testing.assert_allclose(new.reconstruct_field(**points), reference, atol=2e-11)


@pytest.mark.parametrize('theta', [.3, -.3])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('tetm, pol', [('te', (1., 0.)), ('tm', (0., 1.))])
def test_the_pure_1d_solve_is_the_coherent_plane_wave(tetm, pol, side, theta):
    """``tetm`` lights the stack with ``PlaneWave(polarization=(1, 0) | (0, 1))``, exactly.

    An independent route: the coherent door runs the first-order cascade and builds its incident
    wave from ``_plane_basis``, sharing nothing with the pure 1D kernels. The two agree since
    2026-09-27, when the pure path's incidence became the incident order's own s/p wave at
    unit E; before, it was ``+y`` at any angle (a sign at ``theta < 0``) and TM was a unit
    ``H_y`` (1.5 lit from the glass below).

    With no factor at all: both plane waves are unit at ``x = y = 0``, the cell centre, since
    the ordinary solve moved its lateral phase reference there on 2026-09-27. Until then it was
    unit at the cell corner, and this comparison carried ``exp(-1j k_inc . P/2)``.
    """
    options = scene()
    old = wavebit.rcwa(**options, theta=theta, tetm=tetm, incidence_side=side)
    new = wavebit.rcwa(**options, excitation={side: wavebit.PlaneWave(theta=theta,
                                                                      polarization=pol)})
    a, b = old.solve(auto_detune=False), new.solve()
    np.testing.assert_allclose(b.output_top if side == 'top' else b.output_bottom, a.R.sum(), atol=2e-12)
    np.testing.assert_allclose(b.output_bottom if side == 'top' else b.output_top, a.T.sum(), atol=2e-12)
    points = dict(sample_x=[-.17, .12], sample_y=[0.], sample_z=[.02, .1])
    np.testing.assert_allclose(new.reconstruct_field(**points), a.reconstruct_field(**points),
                               atol=2e-11)


def slab_reference(wavelength, d, eps, mus, kx, polarization, phase):
    """Four tangential continuity equations, independent of RCWA/S matrices."""
    k0 = 2*np.pi/wavelength
    qb, ql, qt = np.sqrt(np.asarray(eps)*mus - (kx/k0)**2 + 0j)
    q = np.array([qb, ql, qt])
    Y = q/mus if polarization == 's' else np.asarray(eps)/q
    yb, yl, yt = Y
    p = np.exp(1j*k0*ql*d)
    ib, it = np.exp(1j*phase), 1.+0j
    if polarization == 'p':
        index = np.sqrt(np.asarray(eps)*mus)
        ib *= -qb/index[0]
        it *= -qt/index[2]
    A = np.array([[-1, 0, 1, 1], [yb, 0, yl, -yl],
                  [0, -1, p, 1/p], [0, -yt, yl*p, -yl/p]], complex)
    ob, ot, f, b = np.linalg.solve(A, [ib, yb*ib, it, -yt*it])
    pin = .5*(yb.real*abs(ib)**2 + yt.real*abs(it)**2)
    outputs = .5*np.array([yt.real*abs(ot)**2, yb.real*abs(ob)**2])/pin
    net = .5*np.real((ib+ob)*np.conj(yb*(ib-ob)) - (it+ot)*np.conj(yt*(ot-it)))
    return outputs, net/pin, (ib, it, ob, ot, f, b, ql, yl)


@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('phase', [0., .5*np.pi, np.pi])
@pytest.mark.parametrize('lossy_ports', [False, True])
def test_two_sided_matches_independent_magnetic_slab(pol, phase, lossy_ports):
    eps = np.array([2.25, 4.+.7j, 1.], complex)
    if lossy_ports:
        eps += np.array([.08j, 0., .12j])
    mus = np.array([1.2, 1.1, 1.4])
    kx, d, wavelength = .7, .19, 1.3
    options = scene(wavelength=wavelength, thickness=[d], truncation_order=0,
                    structure=[dict(background=wavebit.Material(eps[1], mus[1]), shapes=[])],
                    bot=wavebit.Material(eps[0], mus[0]), top=wavebit.Material(eps[2], mus[2]))
    pair = (1., 0.) if pol == 's' else (0., 1.)
    s = wavebit.rcwa(**options, excitation={
        'top': wavebit.PlaneWave(kx=kx, polarization=pair),
        'bottom': wavebit.PlaneWave(kx=kx, polarization=pair, amplitude=np.exp(1j*phase))})
    r = s.solve()
    out, absorb, values = slab_reference(wavelength, d, eps, mus, kx, pol, phase)
    np.testing.assert_allclose([r.output_top, r.output_bottom], out, atol=2e-12)
    np.testing.assert_allclose(r.absorption, absorb, atol=2e-12)
    if lossy_ports:
        assert abs(r.absorption - (1-r.output_top-r.output_bottom)) > 1e-4
    else:
        np.testing.assert_allclose(r.absorption, 1-r.output_top-r.output_bottom, atol=2e-12)
    _, _, _, _, f, b, ql, yl = values
    z = np.array([0., .07, d])
    fz, bz = f*np.exp(2j*np.pi/wavelength*ql*z), b*np.exp(-2j*np.pi/wavelength*ql*z)
    expected = np.zeros((3, 6), complex)
    if pol == 's':
        expected[:, 1] = fz+bz
        expected[:, 3] = -yl*(fz-bz)
        expected[:, 5] = kx/(2*np.pi/wavelength)/mus[1]*(fz+bz)
    else:
        expected[:, 0] = fz+bz
        expected[:, 2] = -kx/(2*np.pi/wavelength)/ql*(fz-bz)
        expected[:, 4] = yl*(fz-bz)
    np.testing.assert_allclose(s.reconstruct_field([0.], [0.], z)[:, 0, 0], expected, atol=3e-12)
    # All four tangential components are continuous on both outside interfaces.
    inside = s.reconstruct_field([.13], [.07], [0., d])[:, 0, 0]
    for j, side in enumerate(('bottom', 'top')):
        np.testing.assert_allclose(inside[j, [0, 1, 3, 4]],
                                   r.boundary_fields(side, [.13], [.07])[0, 0, [0, 1, 3, 4]], atol=3e-12)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_patterned_oblique_tensor_superposition(backend):
    material = wavebit.Material([2.5, 3.1, 4.2], mu=[1.1, 1.3, 1.2]).rotated(.3, 'y')
    options = scene(backend=backend, period=None, lattice=[[2., .4], [0., 1.7]],
                    thickness=[.14, .08], truncation_order=(1, 1), factorization='nvf', shape_nvf_grid_size=32,
                    structure=[dict(background=1., shapes=[dict(type='rectangle', lx=.6, ly=.7, material=material)]),
                               dict(background=material, shapes=[])])
    # period=None still names the argument; use only lattice.
    options.pop('period')
    top = wavebit.PlaneWave(kx=.2, ky=.3, polarization=(.6, .8j))
    bottom = wavebit.PlaneWave(kx=.2, ky=.3, polarization=(.8, .6), amplitude=.5j)
    separate = [wavebit.rcwa(**options, excitation={side: beam}).solve()
                for side, beam in [('top', top), ('bottom', bottom)]]
    joint = wavebit.rcwa(**options, excitation={'top': top, 'bottom': bottom}).solve()
    assert joint.group_count == 1
    for side in ('top', 'bottom'):
        for part in ('incoming', 'outgoing', 'total'):
            expected = sum(host(r.boundary_fields(side, [-.1, .2], [.1], part)) for r in separate)
            np.testing.assert_allclose(host(joint.boundary_fields(side, [-.1, .2], [.1], part)), expected, atol=3e-11)
    args = ([-.1, .2], [.1], [.04, .16])
    np.testing.assert_allclose(host(joint.reconstruct_field(*args)),
                               sum(host(r.reconstruct_field(*args)) for r in separate), atol=3e-11)
    assert np.isfinite(host(joint.absorption))


def test_different_sectors_preserve_local_interference_and_integrate_rectangle():
    beams = {'top': wavebit.PlaneWave(kx=.2), 'bottom': wavebit.PlaneWave(kx=.8, amplitude=.7j)}
    options = scene(top=1., bot=1., structure=np.ones((1, 1, 1)), truncation_order=0, thickness=[.3])
    with pytest.raises(ValueError, match='explicit power_region'):
        wavebit.rcwa(**options, excitation=beams).solve()
    r = wavebit.rcwa(**options, excitation=beams, power_region=((-2., 3.), (-1., 1.))).solve()
    assert r.group_count == 2
    assert r.absorption is None  # A finite box can have lateral energy flow.
    x = np.linspace(-2., 3., 20001)
    field = r.boundary_fields('top', x, [0.])[0]
    flux = .5*np.real(field[:, 0]*field[:, 4].conj()-field[:, 1]*field[:, 3].conj())
    other = r.boundary_fields('bottom', x, [0.])[0]
    other_flux = .5*np.real(other[:, 0]*other[:, 4].conj()-other[:, 1]*other[:, 3].conj())
    integrate = np.trapezoid
    np.testing.assert_allclose(r.power.net_into_stack, integrate(other_flux-flux, x)/5, atol=1e-10)
    # Spatial averaging over the whole plane cancels cross-sector terms, while
    # the coherent total field still has position-dependent interference.
    full = r.boundary_power('infinite')
    np.testing.assert_allclose(full.output_top+full.output_bottom, 1., atol=2e-12)
    np.testing.assert_allclose(full.absorption, 0., atol=2e-12)
    assert np.ptp(abs(field[:, 1])**2) > 1.


def test_reciprocal_shift_shares_one_group_and_order_phase_reference():
    """Two coherent spellings of one input, and the ordinary ``excitation`` door with the SAME weights.

    ``OrderExcitation`` weights and ``excitation=`` weights are both amplitudes at ``x = y = 0``
    since 2026-09-27, so the one dict lights both doors with one field. Until then the ordinary
    weights were referenced at the cell corner, and this test lit it with the corner-converted
    ones -- ``exp(-1j k_m . P/2)`` per order, a relative ``-1`` on order 1.
    """
    options = scene(top=1., bot=1., thickness=[.23], truncation_order=(1, 0))
    q, g = .2, 2*np.pi/options['period'][0]
    # The same physical inputs expressed as separate Beam directions or orders.
    a = wavebit.rcwa(**options, excitation={'top': [wavebit.PlaneWave(kx=q), wavebit.PlaneWave(kx=q+g, amplitude=.5j)]}).solve()
    b = wavebit.rcwa(**options, excitation={'top': wavebit.OrderExcitation(kx=q, orders={0: 1., 1: .5j})}).solve()
    assert a.group_count == b.group_count == 1
    for side in ('top', 'bottom'):
        np.testing.assert_allclose(a.boundary_fields(side, [-.3, 0., .2]),
                                   b.boundary_fields(side, [-.3, 0., .2]), atol=2e-12)
    old = wavebit.rcwa(**options, kx=q, polarization=(1., 0.), layer_algorithm='SMM',
                       excitation={0: 1., 1: .5j})
    old.solve(auto_detune=False)
    args = dict(sample_x=[-.3, 0., .2], sample_y=[0.], sample_z=[.1])
    np.testing.assert_allclose(a.reconstruct_field(**args), old.reconstruct_field(**args)[0], atol=3e-11)


def test_normal_incidence_azimuth_rotates_jones_axes():
    options = scene(top=1., bot=1., structure=np.ones((1, 1, 1)), truncation_order=0)
    a = wavebit.rcwa(**options, excitation={'top': wavebit.PlaneWave(theta=0., phi=np.pi/2)}).solve()
    b = wavebit.rcwa(**options, excitation={'top': wavebit.PlaneWave(polarization=(0., 1.))}).solve()
    np.testing.assert_allclose(a.boundary_fields('top', [0.]), b.boundary_fields('top', [0.]), atol=1e-12)


@pytest.mark.parametrize('top', ['lossy', 'nim'])
def test_torch_gradients_at_normal_incidence_are_the_one_sided_limit(top):
    """At ``theta = 0`` the incident order sits at ``k_par = 0``, where the port basis used to
    take ``sqrt(kx**2 + ky**2)``: its infinite derivative met a zero ``dk`` as ``NaN``, so every
    Torch gradient through a normal-incidence ``PlaneWave`` was ``NaN`` -- with respect to the
    lit medium's permittivity too, which does not move ``k_par`` at all. And with the frame there
    fixed at azimuth ``0`` while ``k_par`` leaves along ``phi``, a finite square root would still
    have given ``p_hat`` no tilt, so ``dE_z/dtheta`` came out ``0``.

    Pinned against finite differences on the NumPy backend: the derivative with respect to the
    lit medium's ``Re(eps)`` (central) and with respect to ``theta`` (one-sided, ``theta`` cannot
    be negative here), of the output powers and of every field component at both faces, lit
    from a lossy dielectric and from a lossy double negative, at ``phi = 0.6``."""
    torch = pytest.importorskip('torch')
    grating = np.array([[[4., 4., 1., 1., 2.25, 2.25]]], dtype=complex)

    def lit(x):
        return wavebit.Material(x + 0.01j) if top == 'lossy' else wavebit.Material(
            x - 5.25 + 0.1j, mu=-0.5 + 0.05j)

    def outputs(re_eps, theta, backend):
        r = wavebit.rcwa(backend=backend, wavelength=500., period=[700.], structure=grating,
                         thickness=[180.], truncation_order=(3, 0), answers='field', top=lit(re_eps),
                         bot=2.25, excitation={'top': wavebit.PlaneWave(
                             theta=theta, phi=0.6, polarization=(0.6, 0.8j))}).solve()
        fields = [r.boundary_fields(side, [37.], [0.], part=part).reshape(-1)
                  for side, part in (('top', 'incoming'), ('top', 'outgoing'), ('bottom', 'total'))]
        return [r.output_top, r.output_bottom] + fields

    def flat(values):
        return np.concatenate([np.ravel(host(v)) for v in values])

    def autodiff(which):
        x = torch.tensor(2.25 if which == 'eps' else 0.0, dtype=torch.float64, requires_grad=True)
        values = outputs(x, 0.0, 'torch') if which == 'eps' else outputs(2.25, x, 'torch')
        out = []
        for v in values:
            v = torch.reshape(torch.as_tensor(v), (-1,)).to(torch.complex128)
            for k in range(v.shape[0]):
                pair = []
                for part in (torch.real(v[k]), torch.imag(v[k])):
                    grad, = torch.autograd.grad(part, x, retain_graph=True, allow_unused=True)
                    pair.append(0.0 if grad is None else float(grad))
                out.append(pair[0] + 1j * pair[1])
        return np.asarray(out)

    h = 1e-6
    fd_eps = (flat(outputs(2.25 + h, 0.0, 'numpy')) - flat(outputs(2.25 - h, 0.0, 'numpy'))) / (2 * h)
    ad_eps = autodiff('eps')
    assert np.all(np.isfinite(ad_eps))
    np.testing.assert_allclose(ad_eps, fd_eps, rtol=0, atol=1e-7 * np.abs(fd_eps).max())
    h = 1e-8
    fd_theta = (flat(outputs(2.25, h, 'numpy')) - flat(outputs(2.25, 0.0, 'numpy'))) / h
    ad_theta = autodiff('theta')
    assert np.all(np.isfinite(ad_theta))
    np.testing.assert_allclose(ad_theta, fd_theta, rtol=0, atol=1e-6 * np.abs(fd_theta).max())


def test_cancellation_and_conflicting_or_evanescent_inputs_are_rejected():
    with pytest.raises(ValueError, match='not both'):
        wavebit.PlaneWave(theta=.2, kx=.3)
    with pytest.raises(ValueError, match='not both'):
        wavebit.PlaneWave(polarization=(1., 0.), psi=.4)
    with pytest.raises(ValueError, match='incidence_side'):
        wavebit.rcwa(**scene(), incidence_side='bottom', excitation={'top': wavebit.PlaneWave()})
    with pytest.raises(ValueError, match='positive normal power'):
        wavebit.rcwa(**scene(), excitation={'top': [wavebit.PlaneWave(), wavebit.PlaneWave(amplitude=-1.)]}).solve()
    with pytest.raises(ValueError, match='evanescent'):
        wavebit.rcwa(**scene(), excitation={'top': wavebit.PlaneWave(kx=100.)}).solve()


def test_thick_absorber_and_solve_updates():
    options = scene(top=1., bot=1., structure=np.array([[[(-15.96+1.6j)]]]), thickness=[1000.], truncation_order=0)
    s = wavebit.rcwa(**options, excitation={'top': wavebit.PlaneWave(), 'bottom': wavebit.PlaneWave(amplitude=1j)})
    r = s.solve()
    np.testing.assert_allclose(r.output_top+r.output_bottom, abs((1-(.2+4j))/(1+(.2+4j)))**2, atol=2e-12)
    f = s.reconstruct_field([0.], [0.], [0., 500., 1000.])
    assert np.isfinite(f).all()
    np.testing.assert_allclose(f[1], 0., atol=1e-100)
    changed = s.solve(excitation={'top': wavebit.PlaneWave()})
    np.testing.assert_allclose(changed.output_top, r.output_top+r.output_bottom, atol=2e-12)
    np.testing.assert_allclose(changed.output_bottom, 0., atol=1e-100)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_tilted_tensor_one_sided_matches_etm(side):
    mat = wavebit.Material([2.4, 3.1, 4.2], mu=[1.1, 1.3, 1.2]).rotated(.3, 'y')
    options = scene(truncation_order=0, structure=[dict(background=mat, shapes=[])])
    old = wavebit.rcwa(**options, kx=.2, ky=.3, polarization=(.6, .8j), incidence_side=side, layer_algorithm='ETM')
    a = old.solve(auto_detune=False)
    b = wavebit.rcwa(**options, excitation={side: wavebit.PlaneWave(kx=.2, ky=.3, polarization=(.6, .8j))}).solve()
    np.testing.assert_allclose(b.output_top if side == 'top' else b.output_bottom, a.R.sum(), atol=3e-12)
    points = dict(sample_x=[-.1, .2], sample_y=[.08], sample_z=[.01, .11])
    expected = old.reconstruct_field(**points)[0]       # both unit at x = y = 0: no factor
    np.testing.assert_allclose(b.reconstruct_field(**points), expected, atol=3e-12)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_backend_outputs_match_numpy(backend):
    options = scene(excitation={'top': wavebit.PlaneWave(kx=.2, ky=.3, polarization=(.6, .8j)),
                                'bottom': wavebit.PlaneWave(kx=.2, ky=.3, amplitude=.5j)})
    reference = wavebit.rcwa(**options).solve()
    actual = wavebit.rcwa(**options, backend=backend).solve()
    np.testing.assert_allclose([host(actual.output_top), host(actual.output_bottom)],
                               [reference.output_top, reference.output_bottom], atol=3e-12)
    np.testing.assert_allclose(host(actual.reconstruct_field([.1], [.2], [.05])),
                               reference.reconstruct_field([.1], [.2], [.05]), atol=3e-12)


@pytest.mark.parametrize('amplitude', [[1., .5], complex('nan')])
def test_amplitude_is_a_finite_scalar(amplitude):
    with pytest.raises(ValueError, match='finite complex number'):
        wavebit.rcwa(**scene(), excitation={'top': wavebit.PlaneWave(amplitude=amplitude)}).solve()


def test_common_polarization_angles_and_beam_local_override():
    options = scene(psi=.4, chi=.2)
    common = wavebit.rcwa(**options, excitation={'top': wavebit.PlaneWave()}).solve()
    local = wavebit.rcwa(**scene(), excitation={'top': wavebit.PlaneWave(psi=.4, chi=.2)}).solve()
    np.testing.assert_allclose(common.boundary_fields('top', [0.]), local.boundary_fields('top', [0.]), atol=2e-12)
    solver = wavebit.rcwa(**options, excitation={'top': wavebit.PlaneWave(polarization=(1., 0.))})
    assert solver.formulation == 'first_order'
    solver.solve()
    with pytest.raises(ValueError, match='source'):
        solver.theta = .2
    with pytest.raises(ValueError, match='SMM'):
        solver.solve(layer_algorithm='ETM')


# ------------------------------------------------------ the facade's attribute door

#: The scene conditions each source states for itself, and the facade's own state.
_SOURCE_INPUTS = {'theta', 'phi', 'incidence_side', 'tetm', 'kx', 'ky', 'polarization', 'psi', 'chi'}
_FACADE_OWN = {'excitation', 'power_region', 'length_unit'}


def _facade(backend):
    return wavebit.rcwa(**scene(), backend=backend, excitation={'top': wavebit.PlaneWave()})


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('typo, hint', [('thicknes', "Did you mean 'thickness'?"),
                                        ('perio', "Did you mean 'period'?"),
                                        ('nesting', "Did you mean 'li_nesting'?"),
                                        ('normal_method', "'normal_method' -> 'raster_normal_method'"),
                                        ('label', None)])
def test_an_unknown_attribute_on_the_facade_is_refused_like_on_the_solver(backend, typo, hint):
    """``s.thicknes = ...`` used to be stored on the facade, and the solve answered for the old scene.

    The ordinary solver has refused it since 2026-08-27 (``tests/test_solve_keywords.py``) and
    the retired ``nesting`` since ``li_nesting`` was named (``tests/test_li_nesting.py``), but
    the facade's last ``__setattr__`` branch stored any public name that was not a scene
    property on itself, where nothing reads it. The refusal is now the solver's own: the same
    ``AttributeError`` and the same hint, drawn from what the facade can actually set -- which
    leaves out the incidence each source states for itself.
    """
    facade, plain = _facade(backend), wavebit.rcwa(**scene(), backend=backend)
    refusal = str(pytest.raises(AttributeError, setattr, facade, typo, [.3]).value)
    assert f'CoherentSolver has no attribute {typo!r} to set.' in refusal, refusal
    assert typo not in vars(facade) and not hasattr(facade, typo)
    solver_refusal = str(pytest.raises(AttributeError, setattr, plain, typo, [.3]).value)
    for message in (refusal, solver_refusal):
        assert (hint in message) if hint else ('Did you mean' not in message), message
    settable = set(refusal.split('Settable conditions: ')[1].split('.\n')[0].split(', '))
    assert _FACADE_OWN | {'thickness', 'li_nesting', 'layer_algorithm'} <= settable, settable
    assert not settable & _SOURCE_INPUTS, settable & _SOURCE_INPUTS

    # the two keyword doors of the same facade refuse the same name
    with pytest.raises(TypeError, match=typo):
        facade.solve(**{typo: [.3]})
    with pytest.raises(TypeError, match=typo):
        wavebit.rcwa(**scene(), backend=backend, excitation={'top': wavebit.PlaneWave()},
                     **{typo: [.3]})


def test_the_facade_door_reads_a_quantity_off_the_name_as_the_solver_does():
    """``facade.n_top`` reaches the scene's read-only property; every other index spelling
    reaches the refusal, where the plain nearest match pointed ``ntop`` / ``RI_top`` at ``top``
    -- a refractive index sent to a permittivity slot. Withheld, with the note, as on the
    solver. And the other direction: a count (``n_wavelength``, ``n_thickness``) is no index,
    and keeps the guess plain difflib draws from what the facade can set."""
    import difflib
    from test_spec import COUNT_SPELLINGS, INDEX_SPELLINGS
    facade = _facade('numpy')
    crossed = []
    for name in INDEX_SPELLINGS:
        if hasattr(type(facade._scene), name):             # `n_top`, `n_bot`: "has no setter"
            continue
        message = str(pytest.raises(AttributeError, setattr, facade, name, 1.5).value)
        settable = message.split('Settable conditions: ')[1].split('.\n')[0].split(', ')
        crossed += difflib.get_close_matches(name, sorted(settable), n=1, cutoff=0.6)
        assert 'Did you mean' not in message, message
        assert f"{name!r} is spelled like a refractive index" in message, message
        assert 'relative permittivity (eps = n**2) or a Material' in message, message
        assert name not in vars(facade)
    assert 'top' in crossed, crossed                       # the plain match would have crossed
    reached = 0
    for name in COUNT_SPELLINGS:
        assert not hasattr(type(facade), name) and not hasattr(type(facade._scene), name), name
        message = str(pytest.raises(AttributeError, setattr, facade, name, 1.5).value)
        settable = message.split('Settable conditions: ')[1].split('.\n')[0].split(', ')
        plain = difflib.get_close_matches(name, sorted(settable), n=1, cutoff=0.6)
        reached += bool(plain)
        assert 'spelled like' not in message, message
        assert (f'Did you mean {plain[0]!r}?' in message) if plain \
            else ('Did you mean' not in message), message
    assert reached >= 2, reached


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_the_facade_door_forwards_every_scene_condition_and_keeps_its_own(backend):
    """Closing the door must leave every legitimate name through it working.

    The conditions are enumerated from the scene class rather than listed, so one the solver
    gains is covered the day it lands: each is written through the facade and has to land on
    the scene, where the solve reads it, never on the facade, where nothing does.
    """
    facade = _facade(backend)
    conditions = type(facade._scene)._condition_names() - _SOURCE_INPUTS - _FACADE_OWN
    assert {'thickness', 'li_nesting', 'layer_algorithm', 'wavelength'} <= conditions
    for name in sorted(conditions):
        setattr(facade, name, getattr(facade, name))
        assert name not in vars(facade), name
    facade.li_nesting = 'symmetric'
    assert facade._scene.li_nesting == wavebit.Nesting.SYMMETRIC == facade.li_nesting
    assert facade.layer_algorithm == 'SMM'

    # the facade's own state stays on the facade, and the scene never sees it
    sources = {'bottom': wavebit.PlaneWave(amplitude=.5)}
    facade.excitation, facade.power_region, facade.length_unit = sources, 'cell', 1e-6
    assert [vars(facade)[name] for name in ('excitation', 'power_region', 'length_unit')] \
        == [sources, 'cell', 1e-6]
    assert facade._scene.excitation is None
    assert not {'power_region', 'length_unit'} & set(vars(facade._scene))
    # a private name is the facade's; a class member keeps the descriptor protocol in charge
    facade._note = 'mine'
    assert vars(facade)['_note'] == 'mine'
    facade.solve = lambda *a, **k: 'stubbed'
    assert facade.solve() == 'stubbed'
    for name in ('backend', 'formulation_assigned', 'grating_type', 'n_top'):
        with pytest.raises(AttributeError, match='has no setter'):
            setattr(facade, name, 1.5)
    for name in sorted(_SOURCE_INPUTS):
        with pytest.raises(ValueError, match='excitation source inputs'):
            setattr(facade, name, getattr(facade, name))

    # and a forwarded value reaches the solve: the spelling the typo above meant moves the answer
    fresh = _facade(backend)
    before = host(fresh.solve().output_bottom)
    fresh.thickness = [.3]
    assert not np.allclose(host(fresh.solve().output_bottom), before)
