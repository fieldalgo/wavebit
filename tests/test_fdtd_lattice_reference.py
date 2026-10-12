"""Validate the independent harmonic Yee reference before interpreting its errors."""
import numpy as np
import pytest

import wavebit
from wavebit._core.conventions import yee_longitudinal_phase
from studies.fdtd_lattice_reference import capture_grid, solve_harmonic
from studies.fdtd_rayleigh import BAND, PERIOD, THETA, amplitudes


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_outgoing_phase_satisfies_the_discrete_recurrence_without_growth(backend):
    if backend == 'jax':
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')
    from wavebit.backends.time_domain import time_loop_backend
    runtime = time_loop_backend(backend)
    q = runtime.array(np.array([.7+0j, 1.4j, 2.8+0j]))
    phase = yee_longitudinal_phase(runtime.xp, q, 1., 1.)
    phase = phase.detach().numpy() if backend == 'torch' else np.asarray(phase)
    multiplier = np.exp(1j*phase)
    np.testing.assert_allclose(multiplier+1/multiplier, 2-np.array([.7+0j, 1.4j, 2.8+0j])**2, atol=1e-13)
    assert np.all(abs(multiplier) <= 1+1e-14)
    assert phase[0].real > 0 and phase[1].imag > 0 and phase[2].imag > 0
    gap = yee_longitudinal_phase(runtime.xp, runtime.array(np.array([2.8])), 1., 1.)
    gap = gap.detach().numpy() if backend == 'torch' else np.asarray(gap)
    assert np.isfinite(gap).all() and gap.imag.item() > 0


@pytest.mark.parametrize('pol', ['te', 'tm'])
def test_vacuum_reference_has_exact_discrete_phase_and_no_diffraction(pol):
    grid = capture_grid(40, np.ones((1, 1, 1)))
    dz, _, dx = grid['spacing']; dt = grid['dt']
    kx = 2*np.pi/.96*np.sin(THETA)
    for wavelength in BAND:
        omega = 2*np.sin(np.pi*dt/wavelength)/dt
        kz = 2*np.arcsin(dz*np.sqrt(omega**2-(2*np.sin(kx*dx/2)/dx)**2)/2)/dz
        result = solve_harmonic(grid, wavelength, pol)
        expected = np.array([[0j, 0j], [0j, np.exp(1j*kz*.2)]])
        np.testing.assert_allclose(result['amplitudes'], expected, atol=1e-11)
        assert result['relative_residual'] < 1e-11


@pytest.mark.parametrize('epsilon', [2.25, 4.])
@pytest.mark.parametrize('pol', ['te', 'tm'])
def test_uniform_slab_matches_closed_discrete_interface_recursion(epsilon, pol):
    # Half-cell electric averaging gives TE Y=sin(kz*dz). For TM, matching
    # the staggered Hy and Ex interface equations gives Y=tan(kz*dz/2)/eps.
    # The two interface cos factors cancel through a slab in equal half spaces.
    grid = capture_grid(40, np.full((1, 1, 1), epsilon))
    dz, _, dx = grid['spacing']; dt = grid['dt']
    kx = 2*np.pi/.96*np.sin(THETA)
    for wavelength in BAND:
        omega = 2*np.sin(np.pi*dt/wavelength)/dt
        eps = np.array([1., epsilon])
        k = 2*np.arcsin(dz*np.sqrt(eps*omega**2-(2*np.sin(kx*dx/2)/dx)**2)/2)/dz
        admittance = np.sin(k*dz) if pol == 'te' else np.tan(k*dz/2)/eps
        r = (admittance[0]-admittance[1])/(admittance[0]+admittance[1])
        phase = np.exp(1j*k[1]*.2)
        expected = np.array([r*(1-phase**2)/(1-r*r*phase**2), (1-r*r)*phase/(1-r*r*phase**2)])
        result = solve_harmonic(grid, wavelength, pol)
        np.testing.assert_allclose(result['amplitudes'][:, 1], expected, atol=1e-11)
        np.testing.assert_allclose(result['amplitudes'][:, 0], 0, atol=1e-11)


@pytest.mark.parametrize('pol', ['te', 'tm'])
def test_exact_exterior_ghosts_make_boundary_plane_positions_irrelevant(pol):
    grid = capture_grid(40)
    near = dict(grid, monitor_z=[grid['monitor_z'][0]+2, grid['monitor_z'][1]-2])
    far = dict(grid, monitor_z=[grid['monitor_z'][0]-3, grid['monitor_z'][1]+3])
    for wavelength in BAND:
        expected = solve_harmonic(grid, wavelength, pol)['amplitudes']
        for changed in (near, far):
            result = solve_harmonic(changed, wavelength, pol)
            np.testing.assert_allclose(result['amplitudes'], expected, atol=1e-11)


@pytest.mark.parametrize('pol', ['te', 'tm'])
@pytest.mark.parametrize('wavelength', BAND)
def test_lossless_grating_conserves_discrete_poynting_flux(pol, wavelength):
    # This is the staggered Maxwell flux, not a continuum q weight applied to
    # lattice amplitudes. It must balance across an arbitrary lossless layer.
    result = solve_harmonic(capture_grid(40), wavelength, pol)
    electric, magnetic = result['electric'], result['magnetic']
    flux = []
    for j in (1, electric.shape[1]-2):
        h = .5*(magnetic[:, j]+magnetic[:, j-1])
        flux.append(.5*np.real(electric[0, j]*h[1].conj()
                              -electric[1, j]*h[0].conj()).mean())
    assert flux[0] > 0
    np.testing.assert_allclose(flux[0], flux[1], rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('pol', ['te', 'tm'])
@pytest.mark.parametrize('wavelength', BAND)
def test_reconstructed_fields_obey_native_maxwell_updates_in_the_interior(backend, pol, wavelength):
    if backend == 'jax':
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')
    result = solve_harmonic(capture_grid(40), wavelength, pol)
    dt = result['dt']
    electric = result['electric']
    magnetic = result['magnetic']*np.exp(1j*np.pi*dt/wavelength)
    out = wavebit.fdtd_grid(result['epsilon'], spacing=result['spacing'], dt=dt, steps=1,
                            electric=electric, magnetic=magnetic, bloch_phase=result['bloch_phase'],
                            backend=backend)
    phase = np.exp(-2j*np.pi*dt/wavelength)
    # Native periodic z ghosts differ from the oracle's radiation boundary.
    # One update cannot reach this interior; x wraps retain the same Bloch phase.
    for key, initial in (('E', electric), ('H', magnetic)):
        values = out[key].detach().numpy() if backend == 'torch' else np.asarray(out[key])
        np.testing.assert_allclose(values[:, 2:-2], initial[:, 2:-2]*phase, atol=1e-11)


@pytest.mark.parametrize('pol', ['te', 'tm'])
def test_discrete_slab_refines_to_continuous_airy(pol):
    epsilon = 2.25
    q0 = np.sqrt(1-(BAND/.96*np.sin(THETA))**2)
    q1 = np.sqrt(epsilon-(BAND/.96*np.sin(THETA))**2)
    r = (q0-q1)/(q0+q1) if pol == 'te' else (epsilon*q0-q1)/(epsilon*q0+q1)
    phase = np.exp(2j*np.pi*q1*.2/BAND)
    expected = np.array([r*(1-phase**2)/(1-r*r*phase**2), (1-r*r)*phase/(1-r*r*phase**2)]).T
    errors = []
    for resolution in (40, 80):
        grid = capture_grid(resolution, np.full((1, 1, 1), epsilon))
        actual = np.array([solve_harmonic(grid, wavelength, pol)['amplitudes'][:, 1] for wavelength in BAND])
        errors.append(np.max(abs(actual-expected)))
    assert errors[1] < .002 and errors[1] < .4*errors[0]


def test_settled_uniform_time_domain_amplitudes_match_the_same_grid_reference():
    structure = np.full((1, 1, 1), 2.25)
    grid = capture_grid(40, structure)
    results, diagnostics = wavebit.fdtd(
        structure=structure, thickness=[.2], top=1., bot=1., period=PERIOD,
        wavelength=BAND, theta=THETA, tetm='te', truncation_order=1,
        backend='numpy', resolution=40, n_pml=40, n_periods=80, pulse_width=3.,
        align_layers=True, diffraction_distance=.2, diffraction_method='directional',
        source_distance=1., pml_reflection=1e-12, convergence_window=200, return_diagnostics=True)
    assert diagnostics['convergence']['scattering_settled']
    for pol in ('te', 'tm'):
        expected = np.array([solve_harmonic(grid, wavelength, pol)['amplitudes'] for wavelength in BAND])
        np.testing.assert_allclose(amplitudes(results, 1, pol), expected, atol=1e-6)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('pol', ['te', 'tm'])
def test_cpml_harmonic_fields_obey_the_entire_native_update(monkeypatch, backend, pol):
    from importlib import import_module
    grid_module = import_module('wavebit._core.grid_fdtd')
    from studies.fdtd_lattice_reference import solve_cpml_harmonic
    from wavebit._core.fdtd import cpml_profiles
    if backend == 'jax':
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')
    data = capture_grid(40)
    wavelength, dt = float(BAND[0]), data['dt']
    result = solve_cpml_harmonic(data, wavelength, pol)
    e, h = result['electric'], result['magnetic']
    widths = data['z_spacing']; nodes = np.r_[0., np.cumsum(widths)]
    cells = data['pml_cells']
    pe, ph = cpml_profiles(nodes, nodes[:-1]+widths/2, nodes[cells],
                           nodes[-1]-nodes[-cells-1], dt, data['pml_wavelength'], 1., 1.,
                           target_reflection=data['pml_reflection'])
    old_phase, half_phase = np.exp(2j*np.pi*dt/wavelength), np.exp(1j*np.pi*dt/wavelength)
    # Manufacture the two CPML memory states at their actual previous times.
    # This test-only runtime hook does not add a public checkpoint/resume API.
    e_derivative = (np.roll(e, -1, axis=1)-e)/widths[None, :, None, None]
    dual = .5*(widths+np.roll(widths, 1))
    h_derivative = (h-np.roll(h, 1, axis=1))/dual[None, :, None, None]
    auxiliary_e = (pe['a']/(1-pe['b']*old_phase))[:-1, None, None]*h_derivative[[1, 0]]*half_phase
    auxiliary_h = (ph['a']/(1-ph['b']*old_phase))[:, None, None]*e_derivative[[1, 0]]*old_phase
    factory = grid_module.time_loop_backend
    def with_memory(*args, **kwargs):
        runtime = factory(*args, **kwargs)
        class Runtime:
            def __getattr__(self, name):
                return getattr(runtime, name)
            def advance(self, step, state, steps, checkpoint_steps):
                state = list(state)
                state[4], state[5] = runtime.array(auxiliary_e), runtime.array(auxiliary_h)
                return runtime.advance(step, tuple(state), steps, checkpoint_steps)
        return Runtime()
    monkeypatch.setattr(grid_module, 'time_loop_backend', with_memory)
    out = wavebit.fdtd_grid(data['epsilon'], spacing=data['spacing'], z_spacing=widths,
                            dt=dt, steps=1, backend=backend, electric=e, magnetic=h*half_phase,
                            source=result['source'], pulse=np.ones(1), bloch_phase=data['bloch_phase'],
                            pml_cells=cells, pml_wavelength=data['pml_wavelength'],
                            pml_reflection=data['pml_reflection'])
    for key, initial in (('E', e), ('H', h*half_phase)):
        values = out[key].detach().numpy() if backend == 'torch' else np.asarray(out[key])
        np.testing.assert_allclose(values, initial/old_phase, atol=1e-10)


def test_finite_cpml_reference_matches_settled_time_domain_scattering():
    from studies.fdtd_lattice_reference import solve_cpml_harmonic
    structure = np.full((1, 1, 1), 2.25)
    data = capture_grid(40, structure, pml_width=1., pml_reflection=1e-12)
    results = wavebit.fdtd(
        structure=structure, thickness=[.2], top=1., bot=1., period=PERIOD,
        wavelength=BAND, theta=THETA, tetm='te', truncation_order=1, backend='numpy',
        resolution=40, n_pml=40, n_periods=80, pulse_width=3., align_layers=True,
        diffraction_distance=.2, diffraction_method='directional', source_distance=1., pml_reflection=1e-12)
    for pol in ('te', 'tm'):
        expected = np.array([solve_cpml_harmonic(data, wavelength, pol)['amplitudes'] for wavelength in BAND])
        np.testing.assert_allclose(amplitudes(results, 1, pol), expected, atol=1e-8)
