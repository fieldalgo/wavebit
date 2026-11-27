"""Angular-spectrum propagation against analytic waves, beams and derivatives."""
import numpy as np
import pytest

from wavebit import Units, Material
from wavebit.errors import ValidationError
from wavebit.observations import FieldSamples, RawFieldSpectrum
from wavebit.postprocess.propagation import ASM, PhasorPlane, Plane, from_phasor, plan_asm, propagate


def plane(nx=12, ny=9, *, dx=.27, dy=.31, carrier=(0., 0.)):
    x = 1.3+np.arange(nx)*dx
    y = -.7+np.arange(ny)*dy
    values = np.exp(1j*(carrier[0]*x[None, :]+carrier[1]*y[:, None]))
    return PhasorPlane(values=values.astype(np.complex128), x=x, y=y, z=.2,
        wavelength=.83, units=Units(length='um'), medium_n=1.2, component='Ey',
        direction='+z', boundary='periodic', carrier_kxy=carrier)


def field(result):
    return result.field.values[0, :, :, 0]


@pytest.mark.parametrize('shape', [(12, 9), (11, 10), (12, 1)])
@pytest.mark.parametrize('direction', ['+z', '-z'])
@pytest.mark.parametrize('carrier', [(0., 0.), (.21, -.17)])
def test_fft_bin_plane_wave_and_zero_distance(shape, direction, carrier):
    p = plane(*shape, carrier=carrier).replace(direction=direction)
    kx = 2*np.pi/(len(p.x)*.27)+carrier[0]
    ky = (0. if len(p.y) == 1 else -2*np.pi/(len(p.y)*.31))+carrier[1]
    u = np.exp(1j*(kx*p.x[None, :]+ky*p.y[:, None]))
    p = p.replace(values=u)
    prepared = plan_asm(p)
    kz = np.sqrt((2*np.pi*p.medium_n/p.wavelength)**2-kx*kx-ky*ky)
    np.testing.assert_allclose(field(prepared(0.)), u, atol=1e-10, rtol=1e-10)
    result = prepared(.43)
    np.testing.assert_allclose(field(result), u*np.exp(1j*kz*.43), atol=1e-10, rtol=1e-10)
    assert result.field.coordinates['z'] == pytest.approx(.2+(.43 if direction == '+z' else -.43))
    assert result.field.units['Ey'] == 'V/m'
    assert not result.diagnostics['power_flux_available']


def test_evanescent_decay_and_defined_forward_at_cutoff():
    p = plane(16, 1, dx=.1)
    kx = 2*np.pi*5/(16*.1)
    p = p.replace(values=np.exp(1j*kx*p.x)[None, :])
    alpha = np.sqrt(kx*kx-(2*np.pi*p.medium_n/p.wavelength)**2)
    np.testing.assert_allclose(field(plan_asm(p)(.2)), p.values*np.exp(-alpha*.2),
                               atol=1e-10, rtol=1e-10)
    cutoff = plane(8, 1, dx=.125).replace(wavelength=1., medium_n=1.)
    cutoff = cutoff.replace(values=np.exp(2j*np.pi*cutoff.x)[None, :])
    result = plan_asm(cutoff)(.7)
    np.testing.assert_allclose(field(result), cutoff.values, atol=1e-10, rtol=1e-10)
    assert not result.diagnostics['parameter_gradient_defined']


@pytest.mark.parametrize('unit', ['m', 'um', 'nm'])
def test_units_same_grid_and_public_dispatch(unit):
    p = plane()
    factor = 1e-6/Units(length=unit).length_unit_meters
    scaled = p.replace(x=p.x*factor, y=p.y*factor, z=p.z*factor,
                       wavelength=p.wavelength*factor, units=Units(length=unit))
    target = Plane(x=p.x, y=p.y, distance=.17, units=p.units)
    result = propagate(scaled, target=target, method='asm')
    np.testing.assert_allclose(field(result), field(plan_asm(p)(.17)), atol=1e-10, rtol=1e-10)
    assert result.field.units['coordinates'] == 'um'
    np.testing.assert_allclose(result.field.coordinates['wavelength'], [p.wavelength], atol=1e-10)
    np.testing.assert_allclose(result.field.coordinates['z'], .37, atol=1e-10)


def test_isolated_padding_zero_distance_and_beam_convergence():
    intensities = []
    for size, padding in ((256, 128), (512, 256), (512, 512)):
        x = np.arange(size)*320/size-160
        p = PhasorPlane(values=np.exp(-(x/20)**2)[None, :].astype(complex),
            x=x, y=np.array([0.]), z=0., wavelength=1., units=Units(length='um'),
            medium_n=1., component='Ey', direction='+z', boundary='isolated')
        prepared = plan_asm(p, numerics=ASM(padding=(0, padding)))
        np.testing.assert_allclose(field(prepared(0.)), p.values, atol=1e-10, rtol=1e-10)
        intensity = abs(field(prepared(100.))[0])**2
        # One-dimensional paraxial Gaussian, with z_R = pi*w0^2/lambda.
        spread = np.sqrt(1+(100/(np.pi*20**2))**2)
        exact = np.exp(-2*(x/(20*spread))**2)/spread
        assert np.max(abs(intensity-exact)) < 1e-3
        intensities.append(intensity)
    np.testing.assert_allclose(intensities[0], intensities[1][::2], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(intensities[1], intensities[2], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('change', ['nonuniform', 'descending', 'loss', 'negative_n',
                                  'direction', 'real_values', 'carrier', 'periodic_padding',
                                  'singleton_padding', 'negative_distance', 'target_grid', 'target_complex'])
def test_invalid_contracts_are_refused(change):
    p, numerics = plane(), ASM()
    if change == 'nonuniform': p = p.replace(x=p.x+np.arange(len(p.x))**2*.001)
    if change == 'descending': p = p.replace(x=p.x[::-1])
    if change == 'loss': p = p.replace(medium_n=1.+.1j)
    if change == 'negative_n': p = p.replace(medium_n=-1.)
    if change == 'direction': p = p.replace(direction='auto')
    if change == 'real_values': p = p.replace(values=p.values.real)
    if change == 'carrier': p = p.replace(carrier_kxy=(1j, 0.))
    if change == 'periodic_padding': numerics = ASM(padding=(1, 1))
    if change == 'singleton_padding':
        p, numerics = plane(12, 1).replace(boundary='isolated'), ASM(padding=(1, 1))
    with pytest.raises(ValidationError):
        if change == 'negative_distance':
            plan_asm(p)(-.1)
        elif change in ('target_grid', 'target_complex'):
            x = p.x+(.01 if change == 'target_grid' else 1e-18j)
            propagate(p, target=Plane(x=x, y=p.y, distance=.1, units=p.units), method='asm')
        else:
            plan_asm(p, numerics=numerics)


def samples():
    p = plane()
    return FieldSamples(values=p.values[None, ..., None], components=('Ey',),
        coordinates={'x': p.x, 'y': p.y, 'z': p.z, 'wavelength': np.array([p.wavelength])},
        units={'Ey': 'V/m', 'coordinates': 'um'}, sampling='collocated phasor',
        metadata={'normalization': 'soft-source spectrum'})


def adapt(s, **kw):
    return from_phasor(s, component='Ey', medium_n=1.2, direction='-z',
                       boundary='periodic', assume_outgoing=True, **kw)


def test_phasor_adapter_refuses_raw_uncalibrated_and_unasserted_inputs():
    s = samples()
    assert adapt(s).direction == '-z'
    for invalid in (s.replace(metadata={}), s.replace(units={'Ey': 'V s/m', 'coordinates': 'um'}),
                    RawFieldSpectrum(values=s.values, components=s.components, coordinates=s.coordinates,
                        units=s.units, sampling=s.sampling, metadata=s.metadata)):
        with pytest.raises(ValidationError):
            adapt(invalid)
    with pytest.raises(ValidationError):
        from_phasor(s, component='Ey', medium_n=1., direction='+z', boundary='periodic')


def test_actual_fdtd_calibrated_exterior_plane_connects():
    from wavebit.fdtd import periodic as fdtd
    from wavebit.units import C0
    q = fdtd.Problem(units=Units(length='um'),
        stack=fdtd.Stack(period=(.2,), top=Material(1.), bottom=Material(1.),
            layers=(fdtd.Layer(thickness=.1, epsilon=np.full((1, 2), 2.25)),)),
        excitation=fdtd.Excitation(wavelength=(.9,), sources=(fdtd.PlaneWave(side='top'),)),
        outputs=fdtd.Outputs(planes={'outside': -.2}))
    n = fdtd.Numerics(spacing=1/30, duration_seconds=55*.9e-6/C0, orders=(0, 0))
    result = fdtd.run(q, numerics=n)
    s = result.planes['outside']
    p = from_phasor(s, component='Ey', medium_n=1., direction='-z',
                    boundary='periodic', assume_outgoing=True)
    assert np.max(abs(p.values)) > .1
    # Normal incidence and a uniform slab give a spatially constant transmitted wave.
    np.testing.assert_allclose(p.values, np.full_like(p.values, p.values[0, 0]), atol=1e-10, rtol=1e-10)
    out = plan_asm(p)(.15)
    np.testing.assert_allclose(field(out), p.values*np.exp(2j*np.pi*.15/.9), atol=1e-10, rtol=1e-10)
    assert out.diagnostics['continuum_approximation']
    assert p.metadata['source_metadata']['normalization'] == s.metadata['normalization']


def backend_plane(name):
    p = plane(8, 5, dx=.37, dy=.41)
    p = p.replace(values=p.values*(1+.2*np.cos(2*np.pi*np.arange(8)/8))[None, :])
    if name == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as xp
    else:
        xp = pytest.importorskip('torch')
    return p, p.replace(values=xp.asarray(p.values)), xp


@pytest.mark.parametrize('name', ['jax', 'torch'])
@pytest.mark.parametrize('boundary', ['periodic', 'isolated'])
def test_backend_forward_and_parameter_gradients(name, boundary):
    original, p, xp = backend_plane(name)
    original, p = original.replace(boundary=boundary), p.replace(boundary=boundary)
    prepared = plan_asm(p)
    expected = field(plan_asm(original)(.19))
    got = field(prepared(.19))
    actual = np.asarray(got) if name == 'jax' else got.detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    # Independent central differences of a spatially weighted coherent observable.
    def objective(parameters):
        re, im, d, wl, n = (parameters[i] for i in range(5))
        out = field(prepared(d, values=p.values*(re+1j*im), wavelength=wl, medium_n=n))
        return xp.real(out[1, 2])+.3*xp.imag(out[2, 4])
    center = np.array([1.1, .13, .19, .83, 1.2])
    if name == 'jax':
        import jax
        automatic = np.asarray(jax.jit(jax.grad(objective))(xp.asarray(center)))
    else:
        theta = xp.tensor(center, dtype=xp.float64, requires_grad=True)
        automatic = xp.autograd.grad(objective(theta), theta)[0].detach().cpu().numpy()
    for step in (3e-5, 1e-5):
        finite = []
        for i in range(5):
            perturb = np.eye(5)[i]*step
            finite.append(float((objective(xp.asarray(center+perturb))-objective(xp.asarray(center-perturb)))/(2*step)))
        error = np.linalg.norm(automatic-np.array(finite))/max(1., np.linalg.norm(finite))
        assert error < 1e-6


def test_jit_invalid_dynamic_parameters_report_nan_and_false():
    jax = pytest.importorskip('jax')
    _, p, _ = backend_plane('jax')
    prepared = plan_asm(p)
    evaluate = jax.jit(lambda d, w, n, u: prepared(d, wavelength=w, medium_n=n, values=u))
    valid = evaluate(.1, .83, 1.2, p.values)
    assert valid.diagnostics['input_valid']
    np.testing.assert_allclose(field(valid), field(prepared(.1)), atol=1e-10, rtol=1e-10)
    for d, w, n in ((-.1, .83, 1.2), (.1, 0., 1.2), (.1, .83, -1.)):
        out = evaluate(d, w, n, p.values)
        assert not out.diagnostics['input_valid']
        assert np.isnan(field(out)).all()
    out = evaluate(.1, .83, 1.2, p.values.at[0, 0].set(np.nan))
    assert not out.diagnostics['input_valid']
    assert np.isnan(field(out)).all()


def test_cuda_forward_and_gradient_when_available():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA hardware is required for this verification')
    original = plane(8, 5, dx=.37, dy=.41)
    p = original.replace(values=torch.tensor(original.values, device='cuda', requires_grad=True))
    distance = torch.tensor(.19, dtype=torch.float64, device='cuda', requires_grad=True)
    out = field(plan_asm(p)(distance))
    assert out.device.type == 'cuda'
    np.testing.assert_allclose(out.detach().cpu().numpy(), field(plan_asm(original)(.19)),
                               atol=1e-10, rtol=1e-10)
    grad = torch.autograd.grad(torch.real(out[0, 0]), distance)[0]
    exact = np.real(1j*2*np.pi*1.2/.83*np.exp(2j*np.pi*1.2/.83*.19))
    np.testing.assert_allclose(grad.detach().cpu(), exact, atol=1e-10, rtol=1e-10)
