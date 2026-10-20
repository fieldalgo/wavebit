"""Public general-scene scattering against independent discrete Airy equations.

The oracle and budgets are the established scalar Yee slab gate, not another
Wavebit solve. Both devices exercise public PlaneWave, FieldMonitor and FluxMonitor outputs.
The 24/32-period comparison qualifies the finite Fourier window separately from
the .05/.025 spatial refinement. No CPU/GPU speed comparison is implied.
"""
import numpy as np
import pytest

from wavebit import Material, Units, fdtd

from tests.fdtd.test_fd_airy import (
    BAND, DISCRETE_REL, DISCRETE_TOL, absolute, airy_continuum, airy_discrete,
    discrete_wavenumber, relative,
)


pytestmark = pytest.mark.slow
FLUX_TOL = 2e-5  # Existing plane-flux normalization: test_fd_flux.py.
FRONT, BACK = -.1, .1
OMEGA = 2 * np.pi / np.asarray(BAND)


def slab_problem(epsilon=2.25):
    """A fixed physical film; None removes it for the incident reference."""
    units = Units(length='um')
    structures = () if epsilon is None else (fdtd.Structure(
        name='film', shape=fdtd.SnappedSlab(z=(FRONT, BACK)),
        material=Material(epsilon=epsilon)),)
    monitors = tuple(monitor for name, z in (('front', -.6), ('back', .6))
                     for monitor in (
                         fdtd.FieldMonitor(name, plane=fdtd.Plane(z=z), wavelengths=BAND,
                                    components=('Ey',)),
                         fdtd.FluxMonitor(name + '_flux', plane=fdtd.Plane(z=z), wavelengths=BAND)))
    return fdtd.Problem(units=units, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 3.)), background=Material(epsilon=1.),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()), structures=structures),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=-.9), direction='+z',
            polarization='s', pulse=fdtd.GaussianPulse(
                wavelength=1., frequency_sigma_hz=.15 / units.time_unit_seconds)),),
        outputs=fdtd.Outputs(monitors=monitors))


@pytest.fixture(scope='module', params=['cpu', 'cuda:0'])
def measurements(request):
    """Cache only small host observations, never fields or an autograd graph."""
    torch = pytest.importorskip('torch')
    device = request.param
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and CUDA device')
    cache = {}

    def measure(dx, periods, epsilon):
        key = (dx, periods, epsilon)
        if key in cache:
            return cache[key]
        # Captured pytest output retains the attempted configuration on failure.
        print(f'Airy attempt: device={device}, dx={dx}, periods={periods}, '
              f'epsilon={epsilon}', flush=True)
        problem = slab_problem(epsilon)
        numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=dx),
            duration=fdtd.Duration(periods=periods, source='pw'), courant=.5,
            absorber=fdtd.PML(cells=24, wavelength=1.))
        prepared = fdtd.prepare(problem, numerics=numerics, backend='torch', device=device)
        assert prepared.report.ok
        with torch.no_grad():
            result = prepared.run(problem)
        assert bool(result.validity.checks['runtime_stability'])
        sample = dict(dx=dx, periods=periods,
                      dt=prepared.contract['resolved_dt_seconds'] / problem.units.time_unit_seconds,
                      steps=prepared.contract['resolved_steps'])
        for name in ('front', 'back'):
            value = result[name].values['Ey']
            flux = result[name + '_flux'].spectral_energy
            assert value.device == torch.device(device) and value.dtype == torch.complex128
            assert flux.device == torch.device(device) and flux.dtype == torch.float64
            field = value.detach().cpu().numpy().reshape(len(BAND), -1)
            mean = field.mean(axis=1)
            assert np.isfinite(field).all() and np.max(np.abs(mean)) > 0
            # This normal plane wave must remain uniform over the periodic cell.
            assert np.max(np.abs(field - mean[:, None])) / np.max(np.abs(mean)) < 5e-13
            z = np.asarray(result[name].coordinates['Ey'][2])
            assert z.size == 1
            sample[name] = mean
            sample[name + '_z'] = float(z[0])
            sample[name + '_flux'] = flux.detach().cpu().numpy()
            assert np.isfinite(sample[name + '_flux']).all()
        cache[key] = sample
        return sample

    return measure


def scattering(measure, dx, periods, epsilon=2.25):
    slab = measure(dx, periods, epsilon)
    reference = measure(dx, periods, None)
    assert slab['dt'] == reference['dt'] and slab['steps'] == reference['steps']
    for name in ('front', 'back'):
        assert slab[name + '_z'] == reference[name + '_z']
        # Do not divide by weak band tails: the existing source-normalization
        # gate likewise requires measured amplitudes above 0.1 of their peak.
        amplitude = np.abs(reference[name])
        assert np.min(amplitude) / np.max(amplitude) > .1
        assert np.all(reference[name + '_flux'] > 0)
    np.testing.assert_allclose(reference['front_flux'] / reference['back_flux'],
                               1., rtol=0., atol=FLUX_TOL)
    k = discrete_wavenumber(OMEGA, 1., dx, slab['dt'])
    # Subtract complex fields before squaring. Same-probe references cancel
    # the actual source spectrum and propagation to each measurement plane.
    r = ((slab['front'] - reference['front']) / reference['front']
         * np.exp(2j * k * (slab['front_z'] - FRONT)))
    t = slab['back'] / reference['back'] * np.exp(1j * k * (BACK - FRONT))
    # Flux already collocates H. In this lossless 1-D exterior, its real
    # Poynting cross terms cancel, so net front flux is incident minus reflected.
    # Raw Field Ey/Hx products would not provide that collocation contract.
    R = 1 - slab['front_flux'] / reference['front_flux']
    T = slab['back_flux'] / reference['back_flux']
    np.testing.assert_allclose(R, np.abs(r) ** 2, rtol=0., atol=FLUX_TOL)
    np.testing.assert_allclose(T, np.abs(t) ** 2, rtol=0., atol=FLUX_TOL)
    assert np.isfinite(r).all() and np.isfinite(t).all()
    return r, t, slab


def assert_discrete(r, t, sample):
    dx, dt = sample['dx'], sample['dt']
    expected = airy_discrete(OMEGA, 1.5, round((BACK - FRONT) / dx), dx, dt)
    for actual, want in zip((r, t), expected):
        assert absolute(actual, want) < DISCRETE_TOL, (sample, actual, want)
        assert relative(actual, want) < DISCRETE_REL, (sample, actual, want)


def test_public_slab_matches_independent_discrete_airy(measurements):
    """Smallest independent gate: one coarse slab/reference pair per device."""
    assert_discrete(*scattering(measurements, .05, 24))


def test_public_vacuum_slab_scattering_control(measurements):
    r, t, sample = scattering(measurements, .05, 24, epsilon=1.)
    assert np.max(np.abs(r)) == 0.
    k = discrete_wavenumber(OMEGA, 1., sample['dx'], sample['dt'])
    assert relative(t, np.exp(1j * k * (BACK - FRONT))) < 1e-12


def test_public_slab_duration_and_grid_refinement(measurements):
    """Qualify the finite window and reuse the established half-cell correction."""
    continuum_errors = []
    for dx in (.05, .025):
        early = scattering(measurements, dx, 24)
        late = scattering(measurements, dx, 32)
        assert_discrete(*early)
        assert_discrete(*late)
        for a, b in zip(early[:2], late[:2]):
            assert absolute(a, b) < DISCRETE_TOL, (dx, a, b)
            assert relative(a, b) < DISCRETE_REL, (dx, a, b)
        r, _, sample = late
        k = discrete_wavenumber(OMEGA, 1., dx, sample['dt'])
        beta = discrete_wavenumber(OMEGA, 1.5, dx, sample['dt']) * (BACK - FRONT)
        # A cell-snapped epsilon slab assigns nodes 0..N-1. Its continuum
        # interfaces are half a cell before the requested faces. This is the
        # exact same corrected-continuum convergence gate as test_fd_airy.
        corrected = airy_continuum(1.5, 1., beta)[0] * np.exp(-1j * k * dx)
        continuum_errors.append(relative(r, corrected))
    assert continuum_errors[0] > 3e-2, continuum_errors
    assert 3.2 < continuum_errors[0] / continuum_errors[1] < 5., continuum_errors
