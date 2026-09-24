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
    # The legacy field exp uses cell-corner coordinates. Beam amplitudes use x=y=0.
    q = np.array(old._incident_kxy(old.wavelength))*2*np.pi/old.wavelength
    reference = old.reconstruct_field(**points)[0]*np.exp(-1j*q@np.array(options['period'])/2)
    np.testing.assert_allclose(new.reconstruct_field(**points), reference, atol=2e-11)


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


@pytest.mark.parametrize('backend', [0, 1, 2])
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
    options = scene(top=1., bot=1., thickness=[.23], truncation_order=(1, 0))
    q, g = .2, 2*np.pi/options['period'][0]
    # The same physical inputs expressed as separate Beam directions or orders.
    a = wavebit.rcwa(**options, excitation={'top': [wavebit.PlaneWave(kx=q), wavebit.PlaneWave(kx=q+g, amplitude=.5j)]}).solve()
    b = wavebit.rcwa(**options, excitation={'top': wavebit.OrderExcitation(kx=q, orders={0: 1., 1: .5j})}).solve()
    assert a.group_count == b.group_count == 1
    for side in ('top', 'bottom'):
        np.testing.assert_allclose(a.boundary_fields(side, [-.3, 0., .2]),
                                   b.boundary_fields(side, [-.3, 0., .2]), atol=2e-12)
    # Legacy input phase is defined at the Fourier origin (corner).
    old = wavebit.rcwa(**options, kx=q, polarization=(1., 0.), layer_algorithm='SMM',
                       excitation={0: np.exp(-1j*q), 1: .5j*np.exp(-1j*(q+g))})
    old.solve(auto_detune=False)
    args = dict(sample_x=[-.3, 0., .2], sample_y=[0.], sample_z=[.1])
    np.testing.assert_allclose(a.reconstruct_field(**args), old.reconstruct_field(**args)[0], atol=3e-11)


def test_normal_incidence_azimuth_rotates_jones_axes():
    options = scene(top=1., bot=1., structure=np.ones((1, 1, 1)), truncation_order=0)
    a = wavebit.rcwa(**options, excitation={'top': wavebit.PlaneWave(theta=0., phi=np.pi/2)}).solve()
    b = wavebit.rcwa(**options, excitation={'top': wavebit.PlaneWave(polarization=(0., 1.))}).solve()
    np.testing.assert_allclose(a.boundary_fields('top', [0.]), b.boundary_fields('top', [0.]), atol=1e-12)


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
    expected = old.reconstruct_field(**points)[0]*np.exp(-1j*np.array([.2, .3])@np.array(options['period'])/2)
    np.testing.assert_allclose(b.reconstruct_field(**points), expected, atol=3e-12)


@pytest.mark.parametrize('backend', [0, 1, 2])
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
    assert solver.formulation == '4ff'
    solver.solve()
    with pytest.raises(ValueError, match='source'):
        solver.theta = .2
    with pytest.raises(ValueError, match='SMM'):
        solver.solve(layer_algorithm='ETM')
